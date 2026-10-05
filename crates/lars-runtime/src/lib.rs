//! The SDK runtime: open a pack, answer typed questions, fit calibrators, from any thread.
//!
//! Every language binding sits on this crate (the C ABI in `lars-ffi`, Node through napi-rs).
//! A loaded model belongs to one worker thread, which owns the llama.cpp context and runs one
//! request at a time from a queue; nothing else touches it. Handles (`Lars`) are cheap, `Send`
//! and `Sync`: they parse requests on the caller's thread and queue them. Handles opened on the
//! same pack with the same options share one worker, so a process loads each model once.
//!
//! Each handle keeps its own named calibrators (one per predicate, fitted with `calibrate` or
//! loaded with `set_calibrator`), sent along with every request; the pack's pooled calibrator is
//! the default. Every request submitted gets exactly one reply: its answer, an error, or
//! `cancelled` if it was cancelled (or its handle closed) before it ran.

pub mod error;
pub mod pack;

use std::collections::{HashMap, HashSet};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender};
use std::sync::{Arc, LazyLock, Mutex, Weak};
use std::time::Duration;

use indexmap::IndexMap;
use lars_core::backend::{Backend, MockBackend};
use lars_core::calibration::{Calibrator, certified_threshold};
use lars_core::engine::{Engine, EngineOptions};
use lars_core::eval::{self, Example};
use lars_core::head;
use lars_core::primitives::argmax;
use lars_core::schema::Request;
use lars_llama::{FlashAttention, LlamaBackend, LlamaOptions};
use serde_json::{Value, json};

pub use error::{Error, Result};
pub use pack::Pack;

pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// How to run a pack. Options that change logits (template, layout, flash attention) belong to
/// the pack, not here.
#[derive(Debug, Clone, PartialEq)]
pub struct OpenOptions {
    /// Layers on the GPU (Metal): -1 for all, 0 for CPU only.
    pub gpu_layers: i32,
    /// CPU threads; 0 for the number of cores.
    pub threads: i32,
    pub batch_suffixes: bool,
    pub reuse_prefix: bool,
    pub max_rows: Option<usize>,
    pub max_input_tokens: Option<usize>,
    /// Free the model after this many idle seconds and reload it on the next request; 0 keeps it.
    pub idle_unload_s: f64,
    /// Check every pack file's sha256 on open, not only its size.
    pub verify: bool,
}

impl Default for OpenOptions {
    fn default() -> Self {
        let llama = LlamaOptions::default();
        let engine = EngineOptions::default();
        OpenOptions {
            gpu_layers: llama.gpu_layers,
            threads: llama.n_threads,
            batch_suffixes: llama.batch_suffixes,
            reuse_prefix: llama.reuse_prefix,
            max_rows: engine.max_rows,
            max_input_tokens: engine.max_input_tokens,
            idle_unload_s: 0.0,
            verify: false,
        }
    }
}

impl OpenOptions {
    /// From the JSON a binding passes; absent fields keep their defaults, unknown ones are refused.
    pub fn from_json(value: &Value) -> Result<OpenOptions> {
        let mut options = OpenOptions::default();
        let Some(map) = value.as_object() else {
            return if value.is_null() { Ok(options) } else { Err(Error::invalid("options must be an object")) };
        };
        let bad = |key: &str| Error::invalid(format!("options.{key} has the wrong type"));
        for (key, v) in map {
            match key.as_str() {
                "gpu_layers" => options.gpu_layers = v.as_i64().ok_or_else(|| bad(key))? as i32,
                "threads" => options.threads = v.as_i64().ok_or_else(|| bad(key))? as i32,
                "batch_suffixes" => options.batch_suffixes = v.as_bool().ok_or_else(|| bad(key))?,
                "reuse_prefix" => options.reuse_prefix = v.as_bool().ok_or_else(|| bad(key))?,
                "max_rows" => options.max_rows = v.as_u64().map(|n| n as usize).filter(|n| *n > 0),
                "max_input_tokens" => options.max_input_tokens = v.as_u64().map(|n| n as usize).filter(|n| *n > 0),
                "idle_unload_s" => options.idle_unload_s = v.as_f64().ok_or_else(|| bad(key))?,
                "verify" => options.verify = v.as_bool().ok_or_else(|| bad(key))?,
                other => return Err(Error::invalid(format!("unknown option {other:?}"))),
            }
        }
        Ok(options)
    }

    fn key(&self) -> String {
        format!("{self:?}")
    }
}

// --------------------------------------------------------------------------- worker

/// Receives a request's id and its result.
pub type Reply = Box<dyn FnOnce(u64, Result<String>) + Send>;

enum Work {
    Load,
    Evaluate(Box<Request>),
    Calibrate { examples: Vec<Example>, source: Option<String>, max_error: Option<f64> },
}

struct Job {
    id: u64,
    work: Work,
    calibrators: Arc<IndexMap<String, Calibrator>>,
    reply: Reply,
}

#[derive(Default)]
struct WorkerState {
    queued: AtomicUsize,
    loaded: AtomicBool,
    loads: AtomicUsize,
    cancelled: Mutex<HashSet<u64>>,
    model_id: Mutex<Option<String>>,
    stopping: AtomicBool,
}

/// One loaded pack: the worker thread's queue and what it reports.
struct Host {
    sender: Mutex<Option<Sender<Job>>>,
    worker: Mutex<Option<std::thread::JoinHandle<()>>>,
    state: Arc<WorkerState>,
    pack: Arc<Pack>,
}

impl Host {
    /// Stop the worker: queued requests are replied to with `cancelled`, the model is freed, and
    /// the thread is joined (unless this is the worker thread itself, closing from a callback).
    fn stop(&self) {
        self.state.stopping.store(true, Ordering::SeqCst);
        self.sender.lock().unwrap_or_else(|e| e.into_inner()).take();
        let worker = self.worker.lock().unwrap_or_else(|e| e.into_inner()).take();
        if let Some(worker) = worker {
            if worker.thread().id() != std::thread::current().id() {
                let _ = worker.join();
            }
        }
    }
}

impl Drop for Host {
    /// The last handle on a pack is gone: free its model before returning, so `close` is
    /// deterministic and nothing of llama.cpp outlives it.
    fn drop(&mut self) {
        self.stop();
    }
}

/// At process exit, stop every worker before llama.cpp's static destructors run: the Metal
/// backend asserts if a model is still loaded when its device is torn down. Registered after the
/// first model loads, so it runs before the destructors registered during that load.
extern "C" fn stop_all_at_exit() {
    let hosts: Vec<Arc<Host>> = match HOSTS.try_lock() {
        Ok(hosts) => hosts.values().filter_map(Weak::upgrade).collect(),
        Err(_) => return,
    };
    for host in &hosts {
        host.stop();
    }
}

unsafe extern "C" {
    fn atexit(callback: extern "C" fn()) -> std::ffi::c_int;
}

static EXIT_HOOK: std::sync::Once = std::sync::Once::new();

static NEXT_ID: AtomicU64 = AtomicU64::new(1);
static HOSTS: LazyLock<Mutex<HashMap<String, Weak<Host>>>> = LazyLock::new(|| Mutex::new(HashMap::new()));

fn build_engine(pack: &Pack, options: &OpenOptions) -> Result<Engine> {
    let backend: Box<dyn Backend> = match pack.backend {
        pack::BackendKind::Mock => Box::new(MockBackend::default()),
        pack::BackendKind::LlamaCpp => {
            let llama = LlamaOptions {
                gpu_layers: options.gpu_layers,
                n_threads: options.threads,
                batch_suffixes: options.batch_suffixes,
                reuse_prefix: options.reuse_prefix,
                flash_attention: if pack.flash_attention { FlashAttention::On } else { FlashAttention::Off },
                ..LlamaOptions::default()
            };
            let model = pack.model_file.as_ref().expect("a llama.cpp pack has a model file");
            let name = format!("{}@{}", pack.name, pack.version);
            Box::new(LlamaBackend::load(model, pack.template.clone(), &name, llama)?)
        }
    };
    let engine_options = EngineOptions {
        version: VERSION.to_string(),
        max_rows: options.max_rows,
        max_input_tokens: options.max_input_tokens,
        layout: pack.layout,
    };
    Ok(Engine::new(backend, pack.calibrator.clone(), engine_options)?)
}

fn calibrate(
    engine: &mut Engine,
    pack: &Pack,
    examples: &[Example],
    source: Option<&str>,
    max_error: Option<f64>,
) -> Result<String> {
    if examples.is_empty() {
        return Err(Error::invalid("calibrate needs at least one labeled example"));
    }
    // A head is tried only from MIN_HEAD_LABELS examples that all ask the same question (same
    // type and options); only then are hidden states read, so smaller fits score exactly as before.
    let shapes: Vec<(Option<&str>, Option<Vec<String>>)> =
        examples.iter().map(|e| (e.question["type"].as_str(), eval::option_keys(&e.question).ok())).collect();
    let one_shape = shapes.iter().all(|s| *s == shapes[0]) && shapes[0].0 != Some("multi");
    let collected = eval::collect_with(engine, examples, one_shape && examples.len() >= head::MIN_HEAD_LABELS)?;
    let fitted = eval::calibrate_collected(examples, &collected, source);
    // A fit on a handful of labels can be worse than the pack's own calibrator: a Platt fit on 30
    // rows can move a yes/no boundary the wrong way. Keep it only if, cross-validated on the same
    // labels, it predicts held-out labels significantly better than the pack's: the 95% upper
    // bound on the paired per-example log-loss difference must be below zero.
    let folds = examples.len().min(5);
    let mut differences = Vec::with_capacity(examples.len());
    let (mut fitted_loss, mut pack_loss) = (0.0, 0.0);
    for fold in 0..folds {
        let held: Vec<usize> = (0..examples.len()).filter(|i| i % folds == fold).collect();
        let train: Vec<usize> = (0..examples.len()).filter(|i| i % folds != fold).collect();
        let (train_examples, train_rows) = eval::subset(examples, &collected, &train);
        let (held_examples, held_rows) = eval::subset(examples, &collected, &held);
        let candidate = eval::calibrate_collected(&train_examples, &train_rows, source);
        let with_fit = eval::log_losses(&candidate, &held_examples, &held_rows)?;
        let with_pack = eval::log_losses(&pack.calibrator, &held_examples, &held_rows)?;
        fitted_loss += with_fit.iter().sum::<f64>();
        pack_loss += with_pack.iter().sum::<f64>();
        differences.extend(with_fit.iter().zip(&with_pack).map(|(f, p)| f - p));
    }
    let n = differences.len().max(1) as f64;
    let (fitted_loss, pack_loss) = (fitted_loss / n, pack_loss / n);
    let mean = differences.iter().sum::<f64>() / n;
    let variance = differences.iter().map(|d| (d - mean) * (d - mean)).sum::<f64>() / (n - 1.0).max(1.0);
    let use_fitted = folds >= 2 && mean + 1.96 * (variance / n).sqrt() < 0.0;
    let mut calibrator = if use_fitted { fitted } else { pack.calibrator.clone() };
    let mut kept = if use_fitted { "fitted" } else { "pack" };
    let mut cv_log_loss = json!({"fitted": fitted_loss, "pack": pack_loss});
    let mut probs: Vec<Vec<f64>> =
        eval::score_examples(&calibrator, examples, &collected)?.into_iter().map(|(p, _, _)| p).collect();
    // Then a recalibration or a head on top, chosen the same way, for one repeated question.
    let keys = collected.rows[0].1.clone();
    let mut out_of_fold = false;
    if one_shape {
        let targets: Vec<Vec<f64>> = examples.iter().map(|e| eval::target_vector(e, &keys)).collect();
        let choice = head::choose_head(&probs, collected.hidden.as_deref(), &targets, &keys, folds);
        for (name, loss) in &choice.cv_log_loss[1..] {
            cv_log_loss[name] = json!(loss);
        }
        if let Some(chosen) = choice.head {
            kept = if chosen.uses_hidden() { "head" } else { "recal" };
            calibrator.head = Some(chosen);
            // Scored in-sample, a head looks better than it is: report and certify on the
            // cross-validated answers instead.
            probs = choice.out_of_fold;
            out_of_fold = true;
        }
    }
    let confidences: Vec<f64> = probs.iter().map(|p| p.iter().copied().fold(0.0, f64::max)).collect();
    let correct: Vec<f64> = probs
        .iter()
        .zip(examples)
        .map(|(p, e)| if Some(argmax(p)) == eval::hard_index(&keys, &e.label).ok() { 1.0 } else { 0.0 })
        .collect();
    let accuracy = correct.iter().sum::<f64>() / correct.len() as f64;
    let certified = max_error.map(|budget| match certified_threshold(&confidences, &correct, budget) {
        Some(c) => json!({"max_error": budget, "threshold": c.threshold, "accepted": c.accepted, "errors": c.errors,
                          "coverage": c.coverage, "error_upper_95": c.error_upper}),
        None => json!({"max_error": budget, "threshold": null,
                       "reason": format!("{} labeled examples cannot certify an error rate of {budget} at 95% \
                                          confidence; certifying 5% with no errors takes 73 accepted answers",
                                         examples.len())}),
    });
    Ok(json!({"calibrator": pack.bind(&calibrator), "examples": examples.len(), "accuracy": accuracy,
              "certified": certified, "kept": kept, "base": if use_fitted { "fitted" } else { "pack" },
              "out_of_fold": out_of_fold,
              "cross_validated_log_loss": cv_log_loss})
    .to_string())
}

fn run_worker(receiver: Receiver<Job>, state: Arc<WorkerState>, pack: Arc<Pack>, options: OpenOptions) {
    let mut engine: Option<Engine> = None;
    let idle = (options.idle_unload_s > 0.0).then(|| Duration::from_secs_f64(options.idle_unload_s));
    loop {
        if state.stopping.load(Ordering::SeqCst) {
            break;
        }
        let job = match (idle, engine.is_some()) {
            (Some(limit), true) => match receiver.recv_timeout(limit) {
                Ok(job) => job,
                Err(RecvTimeoutError::Timeout) => {
                    engine = None;
                    state.loaded.store(false, Ordering::SeqCst);
                    continue;
                }
                Err(RecvTimeoutError::Disconnected) => break,
            },
            _ => match receiver.recv() {
                Ok(job) => job,
                Err(_) => break,
            },
        };
        state.queued.fetch_sub(1, Ordering::SeqCst);
        if state.cancelled.lock().unwrap().remove(&job.id) {
            deliver(job.reply, job.id, Err(Error::cancelled()));
            continue;
        }
        let outcome = catch_unwind(AssertUnwindSafe(|| -> Result<String> {
            if engine.is_none() {
                let built = build_engine(&pack, &options)?;
                EXIT_HOOK.call_once(|| unsafe {
                    atexit(stop_all_at_exit);
                });
                *state.model_id.lock().unwrap() = Some(built.model_id());
                state.loads.fetch_add(1, Ordering::SeqCst);
                state.loaded.store(true, Ordering::SeqCst);
                engine = Some(built);
            }
            let engine = engine.as_mut().unwrap();
            engine.named_calibrators = (*job.calibrators).clone();
            match job.work {
                Work::Load => Ok(String::new()),
                Work::Evaluate(request) => Ok(engine.evaluate(&request)?.to_json().to_string()),
                Work::Calibrate { examples, source, max_error } => {
                    calibrate(engine, &pack, &examples, source.as_deref(), max_error)
                }
            }
        }));
        let result = outcome.unwrap_or_else(|_| {
            // A panic may have left the model mid-call; start from a fresh load next time.
            engine = None;
            state.loaded.store(false, Ordering::SeqCst);
            Err(Error::internal("the engine panicked; it will be reloaded"))
        });
        // A cancel that arrived while the job ran was too late; forget it.
        state.cancelled.lock().unwrap().remove(&job.id);
        deliver(job.reply, job.id, result);
    }
    // Stopping: every request still queued gets its one reply, then the model is freed.
    while let Ok(job) = receiver.try_recv() {
        state.queued.fetch_sub(1, Ordering::SeqCst);
        deliver(job.reply, job.id, Err(Error::cancelled()));
    }
    drop(engine);
    state.loaded.store(false, Ordering::SeqCst);
}

/// Run a reply callback; a panicking callback must not take the worker down with it.
fn deliver(reply: Reply, id: u64, result: Result<String>) {
    let _ = catch_unwind(AssertUnwindSafe(|| reply(id, result)));
}

fn host_for(dir: PathBuf, options: &OpenOptions) -> Result<Arc<Host>> {
    let dir = dir.canonicalize().map_err(|e| Error::pack(format!("{}: {e}", dir.display())))?;
    let key = format!("{}|{}", dir.display(), options.key());
    let mut hosts = HOSTS.lock().unwrap();
    hosts.retain(|_, host| host.strong_count() > 0);
    if let Some(host) = hosts.get(&key).and_then(Weak::upgrade) {
        return Ok(host);
    }
    let pack = Arc::new(Pack::load(&dir, options.verify)?);
    let (sender, receiver) = mpsc::channel();
    let state = Arc::new(WorkerState::default());
    let (worker_state, worker_pack, worker_options) = (state.clone(), pack.clone(), options.clone());
    let worker = std::thread::Builder::new()
        .name(format!("lars-{}", pack.name))
        .spawn(move || run_worker(receiver, worker_state, worker_pack, worker_options))
        .map_err(|e| Error::internal(format!("could not start the worker thread: {e}")))?;
    let host = Arc::new(Host { sender: Mutex::new(Some(sender)), worker: Mutex::new(Some(worker)), state, pack });
    hosts.insert(key, Arc::downgrade(&host));
    Ok(host)
}

// --------------------------------------------------------------------------- handle

/// An open pack. Clone-free and thread-safe: share it behind an `Arc`, or open the pack again,
/// which shares the loaded model but keeps its own calibrators.
pub struct Lars {
    host: Arc<Host>,
    /// Swapped in whole, so requests already queued keep the set they were sent with. Shared
    /// with calibrate replies, which run on the worker thread.
    calibrators: Arc<Mutex<Arc<IndexMap<String, Calibrator>>>>,
    pending: Arc<Mutex<HashSet<u64>>>,
}

impl Lars {
    /// Open a pack (a directory, or `name[@version]` in the pack cache) and load its model.
    pub fn open(reference: &str, options: OpenOptions) -> Result<Lars> {
        let dir = pack::resolve(reference)?;
        let lars = Lars {
            host: host_for(dir, &options)?,
            calibrators: Arc::new(Mutex::new(Arc::new(IndexMap::new()))),
            pending: Arc::new(Mutex::new(HashSet::new())),
        };
        lars.wait(|reply| lars.submit(Work::Load, reply))?;
        Ok(lars)
    }

    pub fn pack(&self) -> &Pack {
        &self.host.pack
    }

    fn submit(&self, work: Work, reply: Reply) -> u64 {
        let id = NEXT_ID.fetch_add(1, Ordering::SeqCst);
        self.pending.lock().unwrap().insert(id);
        let pending = self.pending.clone();
        let reply: Reply = Box::new(move |id, result| {
            pending.lock().unwrap().remove(&id);
            reply(id, result)
        });
        let calibrators = self.calibrators.lock().unwrap().clone();
        let job = Job { id, work, calibrators, reply };
        self.host.state.queued.fetch_add(1, Ordering::SeqCst);
        let sent = match self.host.sender.lock().unwrap().as_ref() {
            Some(sender) => sender.send(job).map_err(|mpsc::SendError(job)| job),
            None => Err(job),
        };
        if let Err(job) = sent {
            self.host.state.queued.fetch_sub(1, Ordering::SeqCst);
            deliver(job.reply, id, Err(Error::cancelled()));
        }
        id
    }

    fn wait(&self, submit: impl FnOnce(Reply) -> u64) -> Result<String> {
        let (sender, receiver) = mpsc::channel();
        submit(Box::new(move |_, result| {
            let _ = sender.send(result);
        }));
        receiver.recv().unwrap_or_else(|_| Err(Error::internal("the worker thread stopped without replying")))
    }

    /// Answer a System One request (JSON); blocks until it has run.
    pub fn evaluate(&self, request_json: &str) -> Result<String> {
        let request = Request::from_json(request_json)?;
        self.wait(|reply| self.submit(Work::Evaluate(Box::new(request)), reply))
    }

    /// Queue a request; `reply` runs on the worker thread with its answer. Returns its id, for
    /// `cancel`. A request that does not parse is replied to at once, on this thread.
    pub fn evaluate_async(&self, request_json: &str, reply: Reply) -> u64 {
        match Request::from_json(request_json) {
            Ok(request) => self.submit(Work::Evaluate(Box::new(request)), reply),
            Err(error) => {
                deliver(reply, 0, Err(error.into()));
                0
            }
        }
    }

    /// Fit a calibrator named `name` on labeled examples (the eval JSONL format) and use it for
    /// questions that name it. With `max_error`, also certify an acceptance threshold for that
    /// error budget. Returns the calibrator (bound to this pack) and what was certified.
    pub fn calibrate(&self, name: &str, examples_jsonl: &str, max_error: Option<f64>) -> Result<String> {
        let (sender, receiver) = mpsc::channel();
        self.calibrate_async(
            name,
            examples_jsonl,
            max_error,
            Box::new(move |_, result| {
                let _ = sender.send(result);
            }),
        )?;
        receiver.recv().unwrap_or_else(|_| Err(Error::internal("the worker thread stopped without replying")))
    }

    pub fn calibrate_async(
        &self,
        name: &str,
        examples_jsonl: &str,
        max_error: Option<f64>,
        reply: Reply,
    ) -> Result<u64> {
        if name.is_empty() {
            return Err(Error::invalid("a calibrator needs a name"));
        }
        if let Some(budget) = max_error.filter(|b| !(0.0..1.0).contains(b) || *b == 0.0) {
            return Err(Error::invalid(format!("max_error {budget} is not between 0 and 1")));
        }
        let examples = eval::read_examples(examples_jsonl)?;
        let (name, calibrators) = (name.to_string(), self.calibrators.clone());
        let pack = self.host.pack.clone();
        let store: Reply = Box::new(move |id, result| {
            if let Ok(text) = &result {
                let fitted: Value = serde_json::from_str(text).expect("calibrate returns JSON");
                if let Ok(c) = pack.bound_calibrator(&fitted["calibrator"]) {
                    let mut guard = calibrators.lock().unwrap();
                    let mut map = (**guard).clone();
                    map.insert(name, c);
                    *guard = Arc::new(map);
                }
            }
            reply(id, result)
        });
        Ok(self.submit(Work::Calibrate { examples, source: None, max_error }, store))
    }

    /// Load a calibrator (JSON from `calibrate`, possibly saved earlier) under `name`.
    pub fn set_calibrator(&self, name: &str, calibrator_json: &str) -> Result<()> {
        let value: Value =
            serde_json::from_str(calibrator_json).map_err(|e| Error::invalid(format!("calibrator: {e}")))?;
        let calibrator = self.host.pack.bound_calibrator(&value)?;
        let mut guard = self.calibrators.lock().unwrap();
        let mut map = (**guard).clone();
        map.insert(name.to_string(), calibrator);
        *guard = Arc::new(map);
        Ok(())
    }

    /// Every named calibrator of this handle, bound to the pack, as one JSON object.
    pub fn calibrators_json(&self) -> String {
        let map = self.calibrators.lock().unwrap().clone();
        let out: serde_json::Map<String, Value> =
            map.iter().map(|(name, c)| (name.clone(), self.host.pack.bind(c))).collect();
        Value::Object(out).to_string()
    }

    /// Cancel a queued request: it is replied to with `cancelled` instead of running. True if it
    /// had not finished yet; one that had already started still finishes and replies normally.
    pub fn cancel(&self, id: u64) -> bool {
        if !self.pending.lock().unwrap().contains(&id) {
            return false;
        }
        self.host.state.cancelled.lock().unwrap().insert(id);
        true
    }

    pub fn status(&self) -> String {
        let state = &self.host.state;
        let pack = &self.host.pack;
        json!({
            "pack": {"name": pack.name, "version": pack.version, "dir": pack.dir, "backend": format!("{:?}", pack.backend),
                     "layout": pack.layout.as_str(), "llama_cpp": pack.llama_cpp},
            "model": *state.model_id.lock().unwrap(),
            "loaded": state.loaded.load(Ordering::SeqCst),
            "loads": state.loads.load(Ordering::SeqCst),
            "queued": state.queued.load(Ordering::SeqCst),
            "calibrators": self.calibrators.lock().unwrap().keys().collect::<Vec<_>>(),
            "sdk": VERSION,
            "llama_cpp": lars_llama::LLAMA_CPP_VERSION,
            "runtime_matches_pack": pack.llama_cpp.as_deref().is_none_or(|v| v == lars_llama::LLAMA_CPP_VERSION),
        })
        .to_string()
    }
}

impl Drop for Lars {
    /// Closing a handle cancels its queued requests (each still gets its `cancelled` reply).
    /// The model stays loaded while another handle uses it.
    fn drop(&mut self) {
        let pending: Vec<u64> = self.pending.lock().unwrap().iter().copied().collect();
        self.host.state.cancelled.lock().unwrap().extend(pending);
    }
}
