//! Predicates: one typed question each, asked of any input through a state selector.

use std::borrow::Borrow;
use std::marker::PhantomData;
use std::sync::Arc;

use serde_json::{Map, Value, json};

use crate::{CalibrationResult, Content, Error, Evidence, Lars, Pending, Result};

/// The options of a choice or multi: an enum with `#[derive(Options)]`, or `String` keys given
/// at runtime (`Lars::choice_keys`).
pub trait Key: Clone + Send + Sync + 'static {
    /// The key the model's answer names.
    fn key(&self) -> String;
    fn from_key(key: &str) -> Option<Self>;
}

impl Key for String {
    fn key(&self) -> String {
        self.clone()
    }

    fn from_key(key: &str) -> Option<String> {
        Some(key.to_string())
    }
}

/// An enum whose variants are the options of a choice or multi. Derive it:
///
/// ```
/// #[derive(Clone, Copy, Debug, PartialEq, moedex_lars::Options)]
/// enum Team {
///     /// Payments, charges, refunds
///     Billing,
///     /// Bugs, crashes, outages
///     Technical,
///     #[option("Anything else", key = "other")]
///     Unsorted,
/// }
/// ```
///
/// A variant's doc comment (or `#[option("...")]`) is the description the model reads. Its key
/// is its name in snake_case (or `#[option(key = "...")]`), as in the C# SDK.
pub trait Options: Key {
    /// Every option's key and description, in declaration order.
    fn options() -> Vec<(&'static str, Option<&'static str>)>;
}

mod sealed {
    pub trait Sealed {}
}

/// What a kind of predicate asks and how its answer reads: [`kind::Noul`], [`kind::Choice`],
/// [`kind::Score`] or [`kind::Multi`].
pub trait Kind: sealed::Sealed + Send + Sync + 'static {
    /// What `ask` resolves to.
    type Value: Send + 'static;
    /// What `decide` resolves to: the value and the probabilities behind it.
    type Decision: Send + 'static;

    #[doc(hidden)]
    fn decision(&self, answer: &Value, accepts: &dyn Fn(f64) -> bool) -> Result<Self::Decision>;
    #[doc(hidden)]
    fn value(&self, decision: Self::Decision) -> Self::Value;
    #[doc(hidden)]
    fn extensions(&self, _extensions: &mut Map<String, Value>) {}
}

/// A kind whose predicates can be calibrated on labels.
pub trait Labeled: Kind {
    /// A noul's label is a `bool`, a choice's one of its options, a score's a level index (0 for
    /// the first level).
    type Label;

    #[doc(hidden)]
    fn label(&self, label: &Self::Label) -> String;
}

/// The kinds of predicate.
pub mod kind {
    use super::*;

    #[derive(Debug, Clone)]
    pub struct Noul {
        pub(crate) threshold: f64,
    }

    #[derive(Debug, Clone)]
    pub struct Choice<T> {
        pub(crate) fallback: Option<T>,
        pub(crate) permutations: usize,
    }

    #[derive(Debug, Clone)]
    pub struct Score {
        pub(crate) levels: Vec<Content>,
    }

    #[derive(Debug, Clone)]
    pub struct Multi<T>(pub(crate) PhantomData<fn() -> T>);
}

/// A yes/no predicate: `ask` resolves to `true` when the claim holds and the answer is accepted.
pub type Noul<I = str> = Predicate<kind::Noul, I>;
/// Pick one option: `ask` resolves to one of `T`.
pub type Choice<T, I = str> = Predicate<kind::Choice<T>, I>;
/// Place the input on ordered levels, lowest first: `ask` resolves to the expected level.
pub type Score<I = str> = Predicate<kind::Score, I>;
/// Every option that applies: `ask` resolves to the options whose probability is at least 0.5.
pub type Multi<T, I = str> = Predicate<kind::Multi<T>, I>;

#[derive(Debug, Clone, PartialEq)]
pub struct NoulDecision {
    pub value: bool,
    /// P(yes).
    pub p: f64,
    /// The top probability: how likely the answer is right, once calibrated.
    pub confidence: f64,
    /// False when a certified threshold exists for this predicate and the confidence is below it.
    pub accepted: bool,
    pub abstain: Option<bool>,
    pub evidence: Option<Vec<Evidence>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ChoiceDecision<T> {
    pub value: T,
    /// Each option's probability, in the order the options were declared.
    pub probabilities: Vec<(T, f64)>,
    pub confidence: f64,
    pub accepted: bool,
    /// How much the answer moved across option orderings (with `permutations`).
    pub order_sensitivity: Option<f64>,
    pub abstain: Option<bool>,
    pub evidence: Option<Vec<Evidence>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ScoreDecision {
    /// The expected level, sum of i * P(level i); may fall between levels.
    pub value: f64,
    /// The most probable level, and its description.
    pub level: usize,
    pub label: Content,
    pub probabilities: Vec<f64>,
    pub confidence: f64,
    pub accepted: bool,
    pub abstain: Option<bool>,
    pub evidence: Option<Vec<Evidence>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MultiDecision<T> {
    pub value: Vec<T>,
    pub probabilities: Vec<(T, f64)>,
    pub evidence: Option<Vec<Evidence>>,
}

impl sealed::Sealed for kind::Noul {}
impl<T: Key> sealed::Sealed for kind::Choice<T> {}
impl sealed::Sealed for kind::Score {}
impl<T: Key> sealed::Sealed for kind::Multi<T> {}

fn number(answer: &Value, field: &str) -> Result<f64> {
    answer[field].as_f64().ok_or_else(|| Error::internal(format!("the answer has no {field}")))
}

fn evidence(answer: &Value) -> Option<Vec<Evidence>> {
    answer["evidence"].as_array().map(|items| {
        items
            .iter()
            .map(|e| Evidence {
                span: e["span"].as_str().unwrap_or_default().to_string(),
                effect: e["effect"].as_f64().unwrap_or_default(),
            })
            .collect()
    })
}

fn option<T: Key>(key: &str) -> Result<T> {
    T::from_key(key).ok_or_else(|| Error::internal(format!("the answer names an unknown option {key:?}")))
}

fn probabilities<T: Key>(answer: &Value) -> Result<Vec<(T, f64)>> {
    let map = answer["probabilities"].as_object().ok_or_else(|| Error::internal("the answer has no probabilities"))?;
    map.iter().map(|(key, p)| Ok((option(key)?, p.as_f64().unwrap_or_default()))).collect()
}

impl Kind for kind::Noul {
    type Value = bool;
    type Decision = NoulDecision;

    fn decision(&self, answer: &Value, accepts: &dyn Fn(f64) -> bool) -> Result<NoulDecision> {
        let p = number(answer, "noul")?;
        let confidence = p.max(1.0 - p);
        Ok(NoulDecision {
            value: p >= self.threshold,
            p,
            confidence,
            accepted: accepts(confidence),
            abstain: answer["abstain"].as_bool(),
            evidence: evidence(answer),
        })
    }

    fn value(&self, d: NoulDecision) -> bool {
        d.value && d.accepted
    }
}

impl Labeled for kind::Noul {
    type Label = bool;

    fn label(&self, label: &bool) -> String {
        if *label { "yes" } else { "no" }.to_string()
    }
}

impl<T: Key> Kind for kind::Choice<T> {
    type Value = T;
    type Decision = ChoiceDecision<T>;

    fn decision(&self, answer: &Value, accepts: &dyn Fn(f64) -> bool) -> Result<ChoiceDecision<T>> {
        let probabilities = probabilities(answer)?;
        let confidence = probabilities.iter().map(|(_, p)| *p).fold(0.0, f64::max);
        let choice = answer["choice"].as_str().ok_or_else(|| Error::internal("the answer has no choice"))?;
        Ok(ChoiceDecision {
            value: option(choice)?,
            probabilities,
            confidence,
            accepted: accepts(confidence),
            order_sensitivity: answer["order_sensitivity"].as_f64(),
            abstain: answer["abstain"].as_bool(),
            evidence: evidence(answer),
        })
    }

    fn value(&self, d: ChoiceDecision<T>) -> T {
        match &self.fallback {
            Some(fallback) if !d.accepted => fallback.clone(),
            _ => d.value,
        }
    }

    fn extensions(&self, extensions: &mut Map<String, Value>) {
        if self.permutations > 0 {
            extensions.insert("permutations".into(), self.permutations.into());
        }
    }
}

impl<T: Key> Labeled for kind::Choice<T> {
    type Label = T;

    fn label(&self, label: &T) -> String {
        label.key()
    }
}

impl Kind for kind::Score {
    type Value = f64;
    type Decision = ScoreDecision;

    fn decision(&self, answer: &Value, accepts: &dyn Fn(f64) -> bool) -> Result<ScoreDecision> {
        let probabilities: Vec<f64> =
            (0..self.levels.len()).map(|i| answer["probabilities"][i.to_string()].as_f64().unwrap_or(0.0)).collect();
        let (level, confidence) = probabilities
            .iter()
            .copied()
            .enumerate()
            .fold((0, f64::MIN), |best, (i, p)| if p > best.1 { (i, p) } else { best });
        Ok(ScoreDecision {
            value: number(answer, "score")?,
            level,
            label: self.levels.get(level).cloned().unwrap_or(Value::Null),
            probabilities,
            confidence,
            accepted: accepts(confidence),
            abstain: answer["abstain"].as_bool(),
            evidence: evidence(answer),
        })
    }

    fn value(&self, d: ScoreDecision) -> f64 {
        d.value
    }
}

impl Labeled for kind::Score {
    type Label = usize;

    fn label(&self, label: &usize) -> String {
        label.to_string()
    }
}

impl<T: Key> Kind for kind::Multi<T> {
    type Value = Vec<T>;
    type Decision = MultiDecision<T>;

    fn decision(&self, answer: &Value, _accepts: &dyn Fn(f64) -> bool) -> Result<MultiDecision<T>> {
        let selected = answer["selected"].as_array().ok_or_else(|| Error::internal("the answer has no selection"))?;
        Ok(MultiDecision {
            value: selected.iter().map(|k| option(k.as_str().unwrap_or_default())).collect::<Result<_>>()?,
            probabilities: probabilities(answer)?,
            evidence: evidence(answer),
        })
    }

    fn value(&self, d: MultiDecision<T>) -> Vec<T> {
        d.value
    }
}

/// What the model reads for an input.
pub(crate) type StateFn<I> = Arc<dyn Fn(&I) -> Content + Send + Sync>;

/// One typed question, asked of inputs of type `I` (`str` unless a state selector says
/// otherwise). Cheap to clone; it keeps its pack loaded while it lives.
pub struct Predicate<K: Kind, I: ?Sized = str> {
    pub(crate) lars: Lars,
    pub(crate) name: Arc<str>,
    pub(crate) question: Value,
    pub(crate) kind: Arc<K>,
    pub(crate) state: StateFn<I>,
    abstain_margin: Option<f64>,
    explain: bool,
}

impl<K: Kind, I: ?Sized> Clone for Predicate<K, I> {
    fn clone(&self) -> Self {
        Predicate {
            lars: self.lars.clone(),
            name: self.name.clone(),
            question: self.question.clone(),
            kind: self.kind.clone(),
            state: self.state.clone(),
            abstain_margin: self.abstain_margin,
            explain: self.explain,
        }
    }
}

impl<K: Kind, I: ?Sized> std::fmt::Debug for Predicate<K, I> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Predicate").field("name", &self.name).field("question", &self.question).finish()
    }
}

impl<K: Kind> Predicate<K, str> {
    pub(crate) fn new(lars: &Lars, name: &str, question: Value, kind: K) -> Self {
        Predicate {
            lars: lars.clone(),
            name: name.into(),
            question,
            kind: Arc::new(kind),
            state: Arc::new(|text: &str| Value::String(text.to_string())),
            abstain_margin: None,
            explain: false,
        }
    }
}

impl<K: Kind, I: ?Sized> Predicate<K, I> {
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The System One question this predicate asks.
    pub fn question(&self) -> &Value {
        &self.question
    }

    /// Ask it of another input type: `state` picks what the model reads, so whole records (and
    /// the personal data in them) stay out of prompts.
    ///
    /// ```ignore
    /// let urgent = lars.noul("urgent", "The ticket needs a reply today")
    ///     .state(|t: &Ticket| json!({"subject": t.subject, "body": t.body}));
    /// ```
    pub fn state<J: ?Sized>(self, state: impl Fn(&J) -> Content + Send + Sync + 'static) -> Predicate<K, J> {
        Predicate {
            lars: self.lars,
            name: self.name,
            question: self.question,
            kind: self.kind,
            state: Arc::new(state),
            abstain_margin: self.abstain_margin,
            explain: self.explain,
        }
    }

    /// Mark an answer abstained when its top two probabilities are within this margin.
    pub fn abstain_margin(mut self, margin: f64) -> Self {
        self.abstain_margin = Some(margin);
        self
    }

    /// Return the spans of the input that moved the answer most (one extra pass per sentence).
    pub fn explain(mut self) -> Self {
        self.explain = true;
        self
    }

    /// The answer as a plain value.
    pub fn ask(&self, input: &I) -> Pending<K::Value> {
        let kind = self.kind.clone();
        self.answer(input, move |d| kind.value(d))
    }

    /// The answer with the probabilities behind it.
    pub fn decide(&self, input: &I) -> Pending<K::Decision> {
        self.answer(input, |d| d)
    }

    fn answer<T: Send + 'static>(&self, input: &I, then: impl FnOnce(K::Decision) -> T + Send + 'static) -> Pending<T> {
        if let Err(error) = self.lars.register(&self.name, &self.question) {
            return Pending::ready(Err(error));
        }
        let mut extensions = Map::new();
        self.kind.extensions(&mut extensions);
        if let Some(margin) = self.abstain_margin {
            extensions.insert("abstain_margin".into(), margin.into());
        }
        if self.explain {
            extensions.insert("explain".into(), true.into());
        }
        if self.lars.is_calibrated(&self.name) {
            extensions.insert("calibrators".into(), json!({ &*self.name: &*self.name }));
        }
        let request = json!({
            "state": (self.state)(input),
            "questions": { &*self.name: self.question },
            "lars": extensions,
        });
        let (name, kind, registry) = (self.name.clone(), self.kind.clone(), self.lars.inner.registry.clone());
        self.lars.submit(request, move |response| {
            let answer =
                response["answers"].get(&*name).ok_or_else(|| Error::internal(format!("no answer for {name}")))?;
            Ok(then(kind.decision(answer, &|confidence| registry.accepts(&name, confidence))?))
        })
    }
}

impl<K: Labeled, I: ?Sized> Predicate<K, I> {
    /// Fit this predicate's own calibrator on labeled examples. With `max_error` (e.g. `0.05`),
    /// also certify an acceptance threshold for that error rate, at 95% confidence; answers below
    /// it are then not `accepted`.
    pub fn calibrate<X: Borrow<I>>(
        &self,
        examples: impl IntoIterator<Item = (X, K::Label)>,
        max_error: impl Into<Option<f64>>,
    ) -> Pending<CalibrationResult> {
        if let Err(error) = self.lars.register(&self.name, &self.question) {
            return Pending::ready(Err(error));
        }
        let mut jsonl = String::new();
        for (input, label) in examples {
            let row = json!({
                "state": (self.state)(input.borrow()),
                "question": self.question,
                "label": self.kind.label(&label),
            });
            jsonl.push_str(&row.to_string());
            jsonl.push('\n');
        }
        self.lars.calibrate(&self.name, &jsonl, max_error.into())
    }
}

impl<I: ?Sized> Predicate<kind::Noul, I> {
    /// P(yes) at or above which the answer is yes; default 0.5.
    pub fn threshold(mut self, threshold: f64) -> Self {
        Arc::make_mut(&mut self.kind).threshold = threshold;
        self
    }

    /// What a yes means, if the claim alone is not enough.
    pub fn yes(mut self, meaning: impl Into<Content>) -> Self {
        self.question["criteria"]["true"] = meaning.into();
        self
    }

    /// What a no means.
    pub fn no(mut self, meaning: impl Into<Content>) -> Self {
        self.question["criteria"]["false"] = meaning.into();
        self
    }
}

impl<T: Key, I: ?Sized> Predicate<kind::Choice<T>, I> {
    /// What `ask` returns (not `decide`) when the answer is not accepted under a certified
    /// threshold.
    pub fn fallback(mut self, fallback: T) -> Self {
        Arc::make_mut(&mut self.kind).fallback = Some(fallback);
        self
    }

    /// Extra option orderings to average over, against order bias (each costs a pass).
    pub fn permutations(mut self, permutations: usize) -> Self {
        Arc::make_mut(&mut self.kind).permutations = permutations;
        self
    }
}
