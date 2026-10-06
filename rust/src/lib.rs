//! Typed decisions as functions in your code, answered by a small language model running inside
//! your process.
//!
//! ```no_run
//! use moedex_lars::{Lars, Options};
//!
//! #[derive(Clone, Copy, Debug, PartialEq, Options)]
//! enum Team {
//!     /// Payments, charges, refunds
//!     Billing,
//!     /// Bugs, crashes, outages
//!     Technical,
//! }
//!
//! # fn main() -> moedex_lars::Result<()> {
//! let lars = Lars::open("lars-4b").wait()?;
//! let is_pissed = lars.noul("isPissed", "The customer is angry");
//! let team = lars.choice::<Team>("team", "Which team should handle this?");
//!
//! let message = "I was charged twice and nobody answers!";
//! if is_pissed.ask(message).wait()? {
//!     // escalate
//! }
//! let team: Team = team.ask(message).wait()?;
//! # Ok(()) }
//! ```
//!
//! A predicate asks one typed question about its input: a [`Noul`] (yes or no), a [`Choice`]
//! (one of your options), a [`Score`] (a level on an ordered scale) or a [`Multi`] (every option
//! that applies). The model never generates text: it reads the input once and gives a
//! probability for each option, so the answer is always one of them.
//!
//! Every call returns a [`Pending`]: `.await` it in async code (on any executor), or `.wait()`
//! for it in blocking code. The request is queued on the model's worker thread when the call is
//! made, and dropping the `Pending` before it runs cancels it. [`Lars`] and predicates are cheap
//! to clone, `Send` and `Sync`; the model stays loaded while any of them is alive.

extern crate self as moedex_lars;

mod ask;
mod pending;
mod predicate;

use std::collections::{HashMap, HashSet};
use std::ffi::OsStr;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::Path;
use std::sync::{Arc, Mutex};

pub use ask::Questions;
pub use lars_runtime::{Error, OpenOptions, Result};
pub use moedex_lars_derive::Options;
pub use pending::Pending;
pub use predicate::{
    Choice, ChoiceDecision, Key, Kind, Labeled, Multi, MultiDecision, Noul, NoulDecision, Options, Predicate, Score,
    ScoreDecision, kind,
};
/// The JSON this crate speaks: build states and option descriptions with `serde_json::json!`.
pub use serde_json;
use serde_json::{Map, Value, json};

/// Text or structured content: an input's state, instructions, option descriptions.
pub type Content = Value;

/// A span of the input and how much it moved the answer (with `explain`).
#[derive(Debug, Clone, PartialEq)]
pub struct Evidence {
    pub span: String,
    pub effect: f64,
}

/// An acceptance threshold certified on labeled examples: the Wilson 95% bound on the error rate.
#[derive(Debug, Clone, PartialEq)]
pub struct Certified {
    pub max_error: f64,
    /// Accept answers whose confidence is at least this. `None` when the labels can't certify
    /// `max_error` (`reason` says why); then no answer is accepted until a calibration that
    /// certifies replaces it.
    pub threshold: Option<f64>,
    pub accepted: Option<usize>,
    pub errors: Option<usize>,
    /// The share of the labeled examples that would be accepted.
    pub coverage: Option<f64>,
    pub error_upper_95: Option<f64>,
    pub reason: Option<String>,
}

/// What a calibrated predicate now uses. Each step is taken only when it is significantly better
/// than the last on held-out labels (5-fold cross-validated log loss).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Kept {
    /// The pack's calibrator.
    Pack,
    /// A refit of it on these labels.
    Fitted,
    /// That, rescaled with a bias per option.
    Recal,
    /// That, plus a readout of the model's hidden state (tried from 50 labels of one question).
    Head,
}

/// A calibration's outcome.
#[derive(Debug, Clone, PartialEq)]
pub struct CalibrationResult {
    pub examples: usize,
    pub kept: Kept,
    /// Accuracy on the labeled examples: cross-validated for `Recal` and `Head`, else in-sample.
    pub accuracy: f64,
    pub certified: Option<Certified>,
}

/// A structural relation between noul predicates asked together (`Lars::ask_with`).
#[derive(Debug, Clone, PartialEq)]
pub struct Constraint {
    kind: &'static str,
    questions: Vec<String>,
}

impl Constraint {
    /// Two nouls that say opposite things: their probabilities sum to 1.
    pub fn complement<I: ?Sized>(a: &Noul<I>, b: &Noul<I>) -> Constraint {
        Constraint { kind: "complement", questions: vec![a.name().into(), b.name().into()] }
    }

    /// Nouls of which at most one holds: their probabilities sum to at most 1.
    pub fn exclusive<'a, I: ?Sized + 'a>(nouls: impl IntoIterator<Item = &'a Noul<I>>) -> Constraint {
        Constraint { kind: "exclusive", questions: nouls.into_iter().map(|n| n.name().into()).collect() }
    }
}

/// An open pack. Cheap to clone, `Send` and `Sync`.
#[derive(Clone)]
pub struct Lars {
    pub(crate) inner: Arc<Inner>,
}

pub(crate) struct Inner {
    pub(crate) runtime: lars_runtime::Lars,
    /// Kept apart from the runtime handle: replies hold it on the worker thread, and the last
    /// runtime handle must never be dropped there.
    pub(crate) registry: Arc<Registry>,
}

#[derive(Default)]
pub(crate) struct Registry {
    state: Mutex<RegistryState>,
}

#[derive(Default)]
struct RegistryState {
    questions: HashMap<String, String>,
    calibrated: HashSet<String>,
    certified: HashMap<String, Certified>,
}

impl Registry {
    /// Without a certificate every answer is accepted. With one, answers at or above its
    /// threshold are; a certificate that failed (no threshold) accepts none, so a gate whose
    /// error budget could not be established stays shut.
    pub(crate) fn accepts(&self, name: &str, confidence: f64) -> bool {
        let state = self.state.lock().unwrap();
        state.certified.get(name).is_none_or(|c| c.threshold.is_some_and(|t| confidence >= t))
    }

    fn calibrated(&self, name: &str, certified: Option<Certified>) {
        let mut state = self.state.lock().unwrap();
        state.calibrated.insert(name.to_string());
        match certified {
            Some(c) => state.certified.insert(name.to_string(), c),
            None => state.certified.remove(name),
        };
    }
}

impl std::fmt::Debug for Lars {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let pack = self.pack();
        f.debug_struct("Lars").field("pack", &pack["name"]).field("version", &pack["version"]).finish()
    }
}

fn parse(text: &str) -> Result<Value> {
    serde_json::from_str(text).map_err(|e| Error::internal(format!("the engine's reply is not JSON: {e}")))
}

/// Run a reply's decoding, turning a panic into an error rather than losing the reply.
fn guarded<T>(decode: impl FnOnce() -> Result<T>) -> Result<T> {
    catch_unwind(AssertUnwindSafe(decode)).unwrap_or_else(|_| Err(Error::internal("reading the answer panicked")))
}

impl Lars {
    /// Open a pack (a directory, or `name[@version]` from the pack cache, downloaded on first
    /// use) and load its model, on a thread of its own.
    pub fn open(pack: impl AsRef<OsStr>) -> Pending<Lars> {
        Lars::open_with(pack, OpenOptions::default())
    }

    pub fn open_with(pack: impl AsRef<OsStr>, options: OpenOptions) -> Pending<Lars> {
        let Some(reference) = pack.as_ref().to_str().map(str::to_string) else {
            return Pending::ready(Err(Error::invalid("the pack reference is not UTF-8")));
        };
        let (pending, filler) = Pending::new();
        let open = move || {
            let result = lars_runtime::Lars::open(&reference, options)
                .map(|runtime| Lars { inner: Arc::new(Inner { runtime, registry: Arc::default() }) });
            filler.fill(result);
        };
        match std::thread::Builder::new().name("lars-open".into()).spawn(open) {
            Ok(_) => pending,
            Err(e) => Pending::ready(Err(Error::internal(format!("open: {e}")))),
        }
    }

    /// The SDK version.
    pub fn version() -> &'static str {
        lars_runtime::VERSION
    }

    /// The pack's manifest (`pack.json`): name, version, model, evaluation.
    pub fn pack(&self) -> &Value {
        &self.inner.runtime.pack().manifest
    }

    /// Pack, model, loaded or not, queue depth, calibrators, versions.
    pub fn status(&self) -> Value {
        parse(&self.inner.runtime.status()).unwrap_or(Value::Null)
    }

    // ------------------------------------------------------------------ predicates

    /// A yes/no predicate: `ask` resolves to `true` when the claim holds of the input.
    pub fn noul(&self, name: &str, instructions: impl Into<Content>) -> Noul {
        let question = json!({"type": "noul", "instructions": instructions.into()});
        Predicate::new(self, name, question, kind::Noul { threshold: 0.5 })
    }

    /// Pick one variant of an enum that derives [`Options`].
    pub fn choice<T: Options>(&self, name: &str, instructions: impl Into<Content>) -> Choice<T> {
        let question = json!({"type": "choice", "instructions": instructions.into(), "criteria": criteria::<T>()});
        Predicate::new(self, name, question, kind::Choice { fallback: None, permutations: 0 })
    }

    /// Pick one of options given at runtime, as `(key, description)` pairs.
    pub fn choice_keys<K: Into<String>, D: Into<Content>>(
        &self,
        name: &str,
        instructions: impl Into<Content>,
        options: impl IntoIterator<Item = (K, D)>,
    ) -> Choice<String> {
        let question = json!({"type": "choice", "instructions": instructions.into(), "criteria": keyed(options)});
        Predicate::new(self, name, question, kind::Choice { fallback: None, permutations: 0 })
    }

    /// Place the input on ordered levels, lowest first: `ask` resolves to the expected level.
    pub fn score<L: Into<Content>>(
        &self,
        name: &str,
        instructions: impl Into<Content>,
        levels: impl IntoIterator<Item = L>,
    ) -> Score {
        let levels: Vec<Content> = levels.into_iter().map(Into::into).collect();
        let question = json!({"type": "score", "instructions": instructions.into(), "criteria": levels});
        Predicate::new(self, name, question, kind::Score { levels })
    }

    /// Every variant of an enum that applies.
    pub fn multi<T: Options>(&self, name: &str, instructions: impl Into<Content>) -> Multi<T> {
        let question = json!({"type": "multi", "instructions": instructions.into(), "criteria": criteria::<T>()});
        Predicate::new(self, name, question, kind::Multi(Default::default()))
    }

    /// Every option that applies, of options given at runtime.
    pub fn multi_keys<K: Into<String>, D: Into<Content>>(
        &self,
        name: &str,
        instructions: impl Into<Content>,
        options: impl IntoIterator<Item = (K, D)>,
    ) -> Multi<String> {
        let question = json!({"type": "multi", "instructions": instructions.into(), "criteria": keyed(options)});
        Predicate::new(self, name, question, kind::Multi(Default::default()))
    }

    // ------------------------------------------------------------------ requests

    /// Ask several predicates about one input in one request; the model reads the input once.
    /// The input's state comes from the first predicate.
    ///
    /// ```ignore
    /// let (angry, team) = lars.ask(message, (&is_angry, &team)).await?;
    /// ```
    pub fn ask<I: ?Sized, Q: Questions<I>>(&self, input: &I, questions: Q) -> Pending<Q::Decisions> {
        self.ask_with(input, questions, &[])
    }

    /// `ask`, with structural relations between its nouls.
    pub fn ask_with<I: ?Sized, Q: Questions<I>>(
        &self,
        input: &I,
        questions: Q,
        constraints: &[Constraint],
    ) -> Pending<Q::Decisions> {
        let parts = questions.parts();
        let Some(first) = parts.first() else {
            return Pending::ready(Err(Error::invalid("ask needs at least one predicate")));
        };
        let (mut asked, mut calibrators) = (Map::new(), Map::new());
        for part in &parts {
            if let Err(error) = self.register(part.name, part.question) {
                return Pending::ready(Err(error));
            }
            asked.insert(part.name.into(), part.question.clone());
            if self.is_calibrated(part.name) {
                calibrators.insert(part.name.into(), part.name.into());
            }
        }
        let mut extensions = json!({ "calibrators": calibrators });
        if !constraints.is_empty() {
            extensions["constraints"] =
                constraints.iter().map(|c| json!({"kind": c.kind, "questions": c.questions})).collect();
        }
        let request = json!({"state": (first.state)(input), "questions": asked, "lars": extensions});
        let (decoder, registry) = (questions.decoder(), self.inner.registry.clone());
        self.submit(request, move |response| decoder(&response["answers"], &|name, c| registry.accepts(name, c)))
    }

    /// Send a raw System One request.
    pub fn evaluate(&self, request: &Value) -> Pending<Value> {
        self.submit(request.clone(), Ok)
    }

    pub(crate) fn submit<T: Send + 'static>(
        &self,
        request: Value,
        decode: impl FnOnce(Value) -> Result<T> + Send + 'static,
    ) -> Pending<T> {
        let (pending, filler) = Pending::new();
        let reply = Box::new(move |_, result: Result<String>| {
            filler.fill(result.and_then(|text| guarded(|| decode(parse(&text)?))));
        });
        let id = self.inner.runtime.evaluate_async(&request.to_string(), reply);
        pending.cancels(self, id)
    }

    pub(crate) fn calibrate(&self, name: &str, jsonl: &str, max_error: Option<f64>) -> Pending<CalibrationResult> {
        let (pending, filler) = Pending::new();
        let (registry, calibrated) = (self.inner.registry.clone(), name.to_string());
        let reply = Box::new(move |_, result: Result<String>| {
            filler.fill(result.and_then(|text| {
                guarded(|| {
                    let fit = calibration_result(&parse(&text)?)?;
                    registry.calibrated(&calibrated, fit.certified.clone());
                    Ok(fit)
                })
            }));
        });
        match self.inner.runtime.calibrate_async(name, jsonl, max_error, reply) {
            Ok(id) => pending.cancels(self, id),
            Err(error) => Pending::ready(Err(error)),
        }
    }

    /// A predicate's name stands for one question: calibrators are kept by name.
    pub(crate) fn register(&self, name: &str, question: &Value) -> Result<()> {
        let key = question.to_string();
        let mut state = self.inner.registry.state.lock().unwrap();
        match state.questions.get(name) {
            Some(existing) if *existing != key => {
                Err(Error::invalid(format!("a predicate named {name:?} already asks a different question")))
            }
            Some(_) => Ok(()),
            None => {
                state.questions.insert(name.to_string(), key);
                Ok(())
            }
        }
    }

    pub(crate) fn is_calibrated(&self, name: &str) -> bool {
        self.inner.registry.state.lock().unwrap().calibrated.contains(name)
    }

    // ------------------------------------------------------------------ calibrations

    /// Every calibrator fitted or loaded on this handle, with the question each was fitted for
    /// and its certified threshold: the file format all three SDKs save and load.
    pub fn calibrations(&self) -> Value {
        let state = self.inner.registry.state.lock().unwrap();
        let calibrators = parse(&self.inner.runtime.calibrators_json()).unwrap_or_default();
        let mut questions = Map::new();
        for name in calibrators.as_object().into_iter().flat_map(|m| m.keys()) {
            if let Some(question) = state.questions.get(name) {
                questions.insert(name.clone(), parse(question).unwrap_or_default());
            }
        }
        let certified: Map<String, Value> = state
            .certified
            .iter()
            .map(|(name, c)| {
                let mut saved = json!({"maxError": c.max_error, "threshold": c.threshold});
                if c.accepted.is_some() {
                    saved["accepted"] = json!(c.accepted);
                    saved["errors"] = json!(c.errors);
                    saved["coverage"] = json!(c.coverage);
                    saved["errorUpper95"] = json!(c.error_upper_95);
                }
                if let Some(reason) = &c.reason {
                    saved["reason"] = json!(reason);
                }
                (name.clone(), saved)
            })
            .collect();
        let pack = self.pack();
        json!({
            "format": 2,
            "pack": {"name": pack["name"], "version": pack["version"]},
            "calibrators": calibrators,
            "questions": questions,
            "certified": certified,
        })
    }

    pub fn save_calibrations(&self, path: impl AsRef<Path>) -> Result<()> {
        let text = serde_json::to_string_pretty(&self.calibrations()).expect("JSON serializes") + "\n";
        std::fs::write(path.as_ref(), text).map_err(|e| Error::invalid(format!("{}: {e}", path.as_ref().display())))
    }

    /// Load calibrations saved from this pack (by any of the SDKs). Each replaces the predicate's
    /// calibrator and its certificate (clearing it if the file has none); other predicates keep
    /// theirs. A calibration is bound to the question it was fitted for: a predicate of that name
    /// that asks anything else is refused, here or when it is defined later. The whole file is
    /// checked first, so one that is refused changes nothing.
    pub fn load_calibrations(&self, saved: &Value) -> Result<()> {
        if saved["format"] != 2 {
            return Err(Error::invalid("these calibrations are not format 2; calibrate again with this SDK"));
        }
        let pack = self.pack();
        if saved["pack"]["name"] != pack["name"] || saved["pack"]["version"] != pack["version"] {
            return Err(Error::invalid(format!(
                "these calibrations are for pack {}@{}, not {}@{}",
                saved["pack"]["name"], saved["pack"]["version"], pack["name"], pack["version"]
            )));
        }
        let calibrators = saved["calibrators"].as_object().ok_or_else(|| Error::invalid("no calibrators to load"))?;
        let mut incoming = Vec::with_capacity(calibrators.len());
        {
            let state = self.inner.registry.state.lock().unwrap();
            for name in calibrators.keys() {
                let question = saved["questions"]
                    .get(name)
                    .filter(|q| q.is_object())
                    .ok_or_else(|| Error::invalid(format!("the saved calibration for {name:?} records no question")))?
                    .to_string();
                if state.questions.get(name).is_some_and(|existing| *existing != question) {
                    return Err(Error::invalid(format!(
                        "the saved calibration for {name:?} was fitted for a different question than the predicate of \
                         that name here"
                    )));
                }
                let certified = match saved["certified"].get(name) {
                    None | Some(Value::Null) => None,
                    Some(c) => Some(saved_certified(name, c)?),
                };
                incoming.push((name.clone(), question, certified));
            }
        }
        self.inner.runtime.set_calibrators(&saved["calibrators"].to_string())?;
        let mut state = self.inner.registry.state.lock().unwrap();
        for (name, question, certified) in incoming {
            state.questions.insert(name.clone(), question);
            state.calibrated.insert(name.clone());
            match certified {
                Some(c) => state.certified.insert(name, c),
                None => state.certified.remove(&name),
            };
        }
        Ok(())
    }

    pub fn load_calibrations_from(&self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path).map_err(|e| Error::invalid(format!("{}: {e}", path.display())))?;
        self.load_calibrations(&parse(&text).map_err(|e| Error::invalid(format!("{}: {}", path.display(), e.message)))?)
    }
}

/// A certificate as the calibrations file records it.
fn saved_certified(name: &str, c: &Value) -> Result<Certified> {
    let bad = |field: &str| Error::invalid(format!("the certificate for {name:?} has a bad {field}"));
    let number = |field: &str| match &c[field] {
        Value::Null => Ok(None),
        v => v.as_f64().map(Some).ok_or_else(|| bad(field)),
    };
    let count = |field: &str| match &c[field] {
        Value::Null => Ok(None),
        v => v.as_u64().map(|n| Some(n as usize)).ok_or_else(|| bad(field)),
    };
    Ok(Certified {
        max_error: number("maxError")?.ok_or_else(|| bad("maxError"))?,
        threshold: number("threshold")?,
        accepted: count("accepted")?,
        errors: count("errors")?,
        coverage: number("coverage")?,
        error_upper_95: number("errorUpper95")?,
        reason: match &c["reason"] {
            Value::Null => None,
            v => Some(v.as_str().ok_or_else(|| bad("reason"))?.to_string()),
        },
    })
}

fn criteria<T: Options>() -> Value {
    T::options().into_iter().map(|(key, description)| (key.to_string(), json!(description))).collect()
}

fn keyed<K: Into<String>, D: Into<Content>>(options: impl IntoIterator<Item = (K, D)>) -> Value {
    options.into_iter().map(|(key, description)| (key.into(), description.into())).collect()
}

fn calibration_result(raw: &Value) -> Result<CalibrationResult> {
    let kept = match raw["kept"].as_str() {
        Some("pack") => Kept::Pack,
        Some("fitted") => Kept::Fitted,
        Some("recal") => Kept::Recal,
        Some("head") => Kept::Head,
        other => return Err(Error::internal(format!("calibrate kept {other:?}"))),
    };
    let certified = raw["certified"].as_object().map(|c| Certified {
        max_error: c["max_error"].as_f64().unwrap_or_default(),
        threshold: c["threshold"].as_f64(),
        accepted: c.get("accepted").and_then(Value::as_u64).map(|n| n as usize),
        errors: c.get("errors").and_then(Value::as_u64).map(|n| n as usize),
        coverage: c.get("coverage").and_then(Value::as_f64),
        error_upper_95: c.get("error_upper_95").and_then(Value::as_f64),
        reason: c.get("reason").and_then(Value::as_str).map(str::to_string),
    });
    Ok(CalibrationResult {
        examples: raw["examples"].as_u64().unwrap_or_default() as usize,
        kept,
        accuracy: raw["accuracy"].as_f64().unwrap_or_default(),
        certified,
    })
}
