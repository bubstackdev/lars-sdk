//! The llama.cpp backend: GGUF models in-process, on CPU or Metal.
//!
//! A row's logits come from one prefill of the shared prefix into sequence 0, then the row's
//! suffix decoded after it, reading the next-token logits at the suffix's last token. Two ways
//! to run the suffixes:
//!
//! - **batched** (the default, `batch_suffixes`): copy the prefix's cells to one sequence per
//!   suffix (the cache is unified, so nothing is copied) and decode all suffixes in one pass.
//!   Twenty questions over one state take about a third of the time. A row's logits can move by
//!   up to 0.009 in probability with what it is batched with; a pooled calibrator absorbs far
//!   larger differences than that (CPU against Metal, RESULTS.md 2026-10-05).
//! - **one at a time**: decode the suffix into sequence 0, read, then truncate back to the
//!   prefix, as the reference's llama.cpp backend does with save/restore. A prefix with a single
//!   suffix always runs this way, so a one-question call (and so every eval and calibration
//!   capture) is bit-identical to the reference whatever the setting.
//!
//! With `reuse_prefix` (the default), the tokens left in sequence 0 by the previous call are kept
//! as far as they match the next prefix, so a constant head (system prompt, template), or a whole
//! state asked about again, is prefilled once. Measured bit-identical to a fresh prefill
//! (RESULTS.md, 2026-10-02).
//!
//! Flash attention is off by default: on, it moves probabilities by up to 0.035 on Metal and
//! 0.54 on CPU, and off matches the reference engine bit for bit. Whatever a pack's calibrator
//! was fitted with is what it must be served with.
//!
//! The boundary check matches the reference: if the prefix's tokens are not a prefix of the
//! full prompt's tokens, that row is prefilled in full instead.

use std::collections::HashMap;
use std::ffi::{CString, c_char, c_void};
use std::path::Path;
use std::ptr::NonNull;
use std::sync::{Arc, Mutex, Once};

use lars_core::backend::Backend;
use lars_core::error::{Error, Result};
use lars_core::render::Template;
use lars_llama_sys as sys;

pub use sys::LLAMA_CPP_VERSION;

#[derive(Debug, Clone)]
pub struct LlamaOptions {
    /// Cache cells for one call: the prefix plus every suffix decoded with it.
    pub n_ctx: u32,
    pub n_batch: u32,
    pub n_ubatch: u32,
    /// Sequences per pass when suffixes are batched (one holds the prefix).
    pub n_seq_max: u32,
    /// CPU threads; 0 picks the number of available cores.
    pub n_threads: i32,
    /// Layers offloaded to the GPU (Metal); -1 for all, 0 for CPU only.
    pub gpu_layers: i32,
    pub flash_attention: FlashAttention,
    pub batch_suffixes: bool,
    pub reuse_prefix: bool,
    /// Let llama.cpp repack weights for its CPU kernels (faster on CPU, but a second copy in memory).
    pub repack_cpu_weights: bool,
    /// For experiments: when set, the final hidden state at each row's answer position is appended
    /// here in decode order (row order for one-row calls). Rows then run one at a time.
    pub hidden_states: Option<Arc<Mutex<Vec<Vec<f32>>>>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FlashAttention {
    Auto,
    On,
    Off,
}

impl Default for LlamaOptions {
    fn default() -> Self {
        LlamaOptions {
            n_ctx: 8192,
            n_batch: 2048,
            n_ubatch: 512,
            n_seq_max: 16,
            n_threads: 0,
            gpu_layers: -1,
            flash_attention: FlashAttention::Off,
            batch_suffixes: true,
            reuse_prefix: true,
            repack_cpu_weights: true,
            hidden_states: None,
        }
    }
}

static INIT: Once = Once::new();

unsafe extern "C" fn quiet_progress(_: f32, _: *mut c_void) -> bool {
    true
}

unsafe extern "C" fn log_callback(level: sys::ggml_log_level, text: *const c_char, _: *mut c_void) {
    // llama.cpp is chatty; its log goes to stderr only when LARS_LLAMA_LOG is set, or for errors.
    let verbose = std::env::var_os("LARS_LLAMA_LOG").is_some();
    if (verbose || level >= sys::ggml_log_level_GGML_LOG_LEVEL_ERROR) && !text.is_null() {
        let message = unsafe { std::ffi::CStr::from_ptr(text) }.to_string_lossy();
        eprint!("{message}");
    }
}

fn init() {
    INIT.call_once(|| unsafe {
        sys::llama_log_set(Some(log_callback), std::ptr::null_mut());
        sys::llama_backend_init();
    });
}

fn backend_error(message: impl Into<String>) -> Error {
    Error::Backend(message.into())
}

/// An owned `llama_batch`.
struct Batch {
    raw: sys::llama_batch,
    capacity: usize,
}

impl Batch {
    fn new(capacity: usize, n_seq_max: usize) -> Batch {
        let raw = unsafe { sys::llama_batch_init(capacity as i32, 0, n_seq_max as i32) };
        Batch { raw, capacity }
    }

    fn clear(&mut self) {
        self.raw.n_tokens = 0;
    }

    fn push(&mut self, token: sys::llama_token, pos: sys::llama_pos, seq: sys::llama_seq_id, output: bool) {
        let i = self.raw.n_tokens as usize;
        assert!(i < self.capacity, "batch overflow");
        unsafe {
            *self.raw.token.add(i) = token;
            *self.raw.pos.add(i) = pos;
            *self.raw.n_seq_id.add(i) = 1;
            *(*self.raw.seq_id.add(i)).add(0) = seq;
            *self.raw.logits.add(i) = output as i8;
        }
        self.raw.n_tokens += 1;
    }

    fn len(&self) -> usize {
        self.raw.n_tokens as usize
    }
}

impl Drop for Batch {
    fn drop(&mut self) {
        unsafe { sys::llama_batch_free(self.raw) }
    }
}

pub struct LlamaBackend {
    model: NonNull<sys::llama_model>,
    ctx: NonNull<sys::llama_context>,
    vocab: *const sys::llama_vocab,
    n_vocab: usize,
    template: Template,
    model_name: String,
    options: LlamaOptions,
    labels: HashMap<String, Option<sys::llama_token>>,
    /// Tokens currently held in sequence 0.
    cached: Vec<sys::llama_token>,
    batch: Batch,
}

// The model and context are only touched through `&mut self` or immutable queries.
unsafe impl Send for LlamaBackend {}

impl Drop for LlamaBackend {
    fn drop(&mut self) {
        unsafe {
            sys::llama_free(self.ctx.as_ptr());
            sys::llama_model_free(self.model.as_ptr());
        }
    }
}

impl LlamaBackend {
    pub fn load(path: &Path, template: Template, model_name: &str, options: LlamaOptions) -> Result<LlamaBackend> {
        init();
        let c_path =
            CString::new(path.to_string_lossy().as_bytes()).map_err(|_| backend_error("model path has a NUL"))?;
        let mut model_params = unsafe { sys::llama_model_default_params() };
        model_params.n_gpu_layers = options.gpu_layers;
        model_params.progress_callback = Some(quiet_progress);
        model_params.use_extra_bufts = options.repack_cpu_weights;
        let model = NonNull::new(unsafe { sys::llama_model_load_from_file(c_path.as_ptr(), model_params) })
            .ok_or_else(|| backend_error(format!("could not load {}", path.display())))?;

        let mut ctx_params = unsafe { sys::llama_context_default_params() };
        let threads = if options.n_threads > 0 {
            options.n_threads
        } else {
            std::thread::available_parallelism().map_or(4, |n| n.get() as i32)
        };
        ctx_params.n_ctx = options.n_ctx;
        ctx_params.n_batch = options.n_batch;
        ctx_params.n_ubatch = options.n_ubatch;
        ctx_params.n_seq_max = options.n_seq_max;
        ctx_params.n_threads = threads;
        ctx_params.n_threads_batch = threads;
        ctx_params.kv_unified = true;
        ctx_params.no_perf = true;
        if options.hidden_states.is_some() {
            ctx_params.pooling_type = sys::llama_pooling_type_LLAMA_POOLING_TYPE_NONE;
        }
        ctx_params.flash_attn_type = match options.flash_attention {
            FlashAttention::Auto => sys::llama_flash_attn_type_LLAMA_FLASH_ATTN_TYPE_AUTO,
            FlashAttention::On => sys::llama_flash_attn_type_LLAMA_FLASH_ATTN_TYPE_ENABLED,
            FlashAttention::Off => sys::llama_flash_attn_type_LLAMA_FLASH_ATTN_TYPE_DISABLED,
        };
        let Some(ctx) = NonNull::new(unsafe { sys::llama_init_from_model(model.as_ptr(), ctx_params) }) else {
            unsafe { sys::llama_model_free(model.as_ptr()) };
            return Err(backend_error("could not create a llama.cpp context"));
        };
        let vocab = unsafe { sys::llama_model_get_vocab(model.as_ptr()) };
        let n_vocab = unsafe { sys::llama_vocab_n_tokens(vocab) } as usize;
        let batch = Batch::new(options.n_batch.max(options.n_ctx) as usize, options.n_seq_max as usize);
        Ok(LlamaBackend {
            model,
            ctx,
            vocab,
            n_vocab,
            template,
            model_name: model_name.to_string(),
            options,
            labels: HashMap::new(),
            cached: Vec::new(),
            batch,
        })
    }

    pub fn options(&self) -> &LlamaOptions {
        &self.options
    }

    /// The model's own chat template from its GGUF metadata, if it has one.
    pub fn gguf_chat_template(&self) -> Option<String> {
        let raw = unsafe { sys::llama_model_chat_template(self.model.as_ptr(), std::ptr::null()) };
        (!raw.is_null()).then(|| unsafe { std::ffi::CStr::from_ptr(raw) }.to_string_lossy().into_owned())
    }

    pub fn tokenize(&self, text: &str, add_special: bool, parse_special: bool) -> Vec<sys::llama_token> {
        let mut tokens = vec![0; text.len() + 8];
        loop {
            let n = unsafe {
                sys::llama_tokenize(
                    self.vocab,
                    text.as_ptr() as *const c_char,
                    text.len() as i32,
                    tokens.as_mut_ptr(),
                    tokens.len() as i32,
                    add_special,
                    parse_special,
                )
            };
            if n >= 0 {
                tokens.truncate(n as usize);
                return tokens;
            }
            tokens.resize((-n) as usize, 0);
        }
    }

    fn label_id(&mut self, label: &str) -> Option<sys::llama_token> {
        if let Some(id) = self.labels.get(label) {
            return *id;
        }
        let ids = self.tokenize(&format!(" {label}"), false, false);
        let id = (ids.len() == 1).then(|| ids[0]);
        self.labels.insert(label.to_string(), id);
        id
    }

    fn memory(&self) -> sys::llama_memory_t {
        unsafe { sys::llama_get_memory(self.ctx.as_ptr()) }
    }

    /// Decode the batch; outputs are read with `logits_at`.
    fn decode(&mut self) -> Result<()> {
        match unsafe { sys::llama_decode(self.ctx.as_ptr(), self.batch.raw) } {
            0 => Ok(()),
            1 => Err(backend_error("llama.cpp found no room in the cache; raise n_ctx")),
            code => Err(backend_error(format!("llama_decode failed ({code})"))),
        }
    }

    fn logits_at(&self, batch_index: usize, label_ids: &[sys::llama_token]) -> Result<Vec<f64>> {
        let row = unsafe { sys::llama_get_logits_ith(self.ctx.as_ptr(), batch_index as i32) };
        if row.is_null() {
            return Err(backend_error("llama.cpp returned no logits"));
        }
        let row = unsafe { std::slice::from_raw_parts(row, self.n_vocab) };
        Ok(label_ids.iter().map(|&id| row[id as usize] as f64).collect())
    }

    /// Make sequence 0 hold exactly `tokens`, keeping what matches from the previous call when
    /// `reuse_prefix` is on, and clearing every other sequence.
    fn set_prefix(&mut self, tokens: &[sys::llama_token]) -> Result<()> {
        let memory = self.memory();
        let keep = if self.options.reuse_prefix {
            self.cached.iter().zip(tokens).take_while(|(a, b)| a == b).count()
        } else {
            0
        };
        let truncated = keep > 0 && unsafe { sys::llama_memory_seq_rm(memory, 0, keep as i32, -1) };
        let keep = if truncated { keep } else { 0 };
        if keep == 0 {
            unsafe { sys::llama_memory_clear(memory, false) };
        }
        for seq in 1..self.options.n_seq_max as i32 {
            unsafe { sys::llama_memory_seq_rm(memory, seq, -1, -1) };
        }
        self.cached = tokens[..keep].to_vec();
        for chunk in tokens[keep..].chunks(self.options.n_batch as usize) {
            self.batch.clear();
            let start = self.cached.len();
            for (i, &token) in chunk.iter().enumerate() {
                self.batch.push(token, (start + i) as i32, 0, false);
            }
            self.decode()?;
            self.cached.extend_from_slice(chunk);
        }
        Ok(())
    }

    fn check_fits(&self, tokens: usize) -> Result<()> {
        if tokens > self.options.n_ctx as usize {
            return Err(backend_error(format!(
                "a row needs {tokens} tokens; the context holds {} (raise n_ctx)",
                self.options.n_ctx
            )));
        }
        Ok(())
    }

    /// Logits after `prefix + suffix` for each suffix decoded alone after the prefix.
    fn one_at_a_time(&mut self, prefix: &[i32], suffixes: &[Vec<i32>], labels: &[Vec<i32>]) -> Result<Vec<Vec<f64>>> {
        self.set_prefix(prefix)?;
        let memory = self.memory();
        let mut results = Vec::with_capacity(suffixes.len());
        for (row, (suffix, label_ids)) in suffixes.iter().zip(labels).enumerate() {
            if row > 0 && self.cached.len() != prefix.len() {
                // The previous row could not be truncated away (a recurrent model): prefill again.
                self.set_prefix(prefix)?;
            }
            self.check_fits(prefix.len() + suffix.len())?;
            // Embeddings make every token in a decode an output, so they are on only for the suffix.
            let hidden = self.options.hidden_states.is_some();
            if hidden {
                unsafe { sys::llama_set_embeddings(self.ctx.as_ptr(), true) };
            }
            for chunk_start in (0..suffix.len()).step_by(self.options.n_batch as usize) {
                let chunk = &suffix[chunk_start..suffix.len().min(chunk_start + self.options.n_batch as usize)];
                self.batch.clear();
                for (i, &token) in chunk.iter().enumerate() {
                    let at = chunk_start + i;
                    self.batch.push(token, (prefix.len() + at) as i32, 0, at + 1 == suffix.len());
                }
                self.decode()?;
            }
            results.push(self.logits_at(self.batch.len() - 1, label_ids)?);
            if hidden {
                let at = unsafe { sys::llama_get_embeddings_ith(self.ctx.as_ptr(), self.batch.len() as i32 - 1) };
                unsafe { sys::llama_set_embeddings(self.ctx.as_ptr(), false) };
                if at.is_null() {
                    return Err(backend_error("llama.cpp returned no hidden state"));
                }
                let n = unsafe { sys::llama_model_n_embd_out(self.model.as_ptr()) } as usize;
                let row = unsafe { std::slice::from_raw_parts(at, n) }.to_vec();
                self.options.hidden_states.as_ref().unwrap().lock().unwrap().push(row);
            }
            if !unsafe { sys::llama_memory_seq_rm(memory, 0, prefix.len() as i32, -1) } {
                // The cache cannot drop a tail for this model (recurrent state): start over, and
                // prefill the prefix again only if another row follows.
                unsafe { sys::llama_memory_clear(memory, false) };
                self.cached.clear();
            }
        }
        Ok(results)
    }

    /// Logits for every suffix, decoded together in passes of up to n_seq_max - 1 sequences.
    fn batched(&mut self, prefix: &[i32], suffixes: &[Vec<i32>], labels: &[Vec<i32>]) -> Result<Vec<Vec<f64>>> {
        self.set_prefix(prefix)?;
        let memory = self.memory();
        let mut results = vec![Vec::new(); suffixes.len()];
        let lanes = (self.options.n_seq_max as usize - 1).max(1);
        let mut start = 0;
        while start < suffixes.len() {
            // As many suffixes as fit in the lanes, the context and one batch.
            let (mut end, mut cells) = (start, prefix.len());
            while end < suffixes.len()
                && end - start < lanes
                && cells + suffixes[end].len() <= self.options.n_ctx as usize
                && cells - prefix.len() + suffixes[end].len() <= self.options.n_batch as usize
            {
                cells += suffixes[end].len();
                end += 1;
            }
            if end == start {
                return self.one_at_a_time(prefix, &suffixes[start..], &labels[start..]).map(|rest| {
                    results.truncate(start);
                    results.extend(rest);
                    results
                });
            }
            self.batch.clear();
            let mut outputs = Vec::with_capacity(end - start);
            for (lane, suffix) in suffixes[start..end].iter().enumerate() {
                let seq = lane as i32 + 1;
                unsafe { sys::llama_memory_seq_cp(memory, 0, seq, -1, -1) };
                for (i, &token) in suffix.iter().enumerate() {
                    self.batch.push(token, (prefix.len() + i) as i32, seq, i + 1 == suffix.len());
                }
                outputs.push(self.batch.len() - 1);
            }
            self.decode()?;
            for (offset, &index) in outputs.iter().enumerate() {
                results[start + offset] = self.logits_at(index, &labels[start + offset])?;
            }
            for lane in 0..end - start {
                unsafe { sys::llama_memory_seq_rm(memory, lane as i32 + 1, -1, -1) };
            }
            start = end;
        }
        Ok(results)
    }
}

impl Backend for LlamaBackend {
    fn name(&self) -> &str {
        "llamacpp"
    }

    fn model_name(&self) -> &str {
        &self.model_name
    }

    fn template(&self) -> &Template {
        &self.template
    }

    fn is_single_token(&self, label: &str) -> bool {
        self.tokenize(&format!(" {label}"), false, false).len() == 1
    }

    fn count_tokens(&self, text: &str) -> usize {
        self.tokenize(text, false, true).len()
    }

    fn label_logits(&mut self, prefix: &str, suffixes: &[String], labels: &[Vec<String>]) -> Result<Vec<Vec<f64>>> {
        let label_ids: Vec<Vec<i32>> = labels
            .iter()
            .map(|row| {
                row.iter()
                    .map(|l| self.label_id(l).ok_or_else(|| backend_error(format!("label {l:?} is not one token"))))
                    .collect::<Result<Vec<_>>>()
            })
            .collect::<Result<_>>()?;
        let prefix_ids = self.tokenize(prefix, true, true);
        self.check_fits(prefix_ids.len())?;
        let mut clean: Vec<(usize, Vec<i32>)> = Vec::new();
        let mut split: Vec<(usize, Vec<i32>)> = Vec::new();
        for (i, suffix) in suffixes.iter().enumerate() {
            let full = self.tokenize(&format!("{prefix}{suffix}"), true, true);
            if full.starts_with(&prefix_ids) {
                clean.push((i, full[prefix_ids.len()..].to_vec()));
            } else {
                split.push((i, full));
            }
        }
        let mut results = vec![Vec::new(); suffixes.len()];
        if !clean.is_empty() {
            let rows: Vec<Vec<i32>> = clean.iter().map(|(_, s)| s.clone()).collect();
            let row_labels: Vec<Vec<i32>> = clean.iter().map(|(i, _)| label_ids[*i].clone()).collect();
            let logits = if self.options.batch_suffixes && rows.len() > 1 && self.options.hidden_states.is_none() {
                self.batched(&prefix_ids, &rows, &row_labels)?
            } else {
                self.one_at_a_time(&prefix_ids, &rows, &row_labels)?
            };
            for ((i, _), z) in clean.iter().zip(logits) {
                results[*i] = z;
            }
        }
        // Rows whose tokens merge across the prefix boundary: the full prompt, prefilled alone.
        for (i, full) in split {
            let (head, last) = full.split_at(full.len() - 1);
            let z = self.one_at_a_time(head, &[last.to_vec()], std::slice::from_ref(&label_ids[i]))?;
            results[i] = z.into_iter().next().unwrap();
        }
        Ok(results)
    }
}
