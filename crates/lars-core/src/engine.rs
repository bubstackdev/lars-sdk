//! The engine: request in, typed answers out, as the reference's `engine.py`.
//!
//! 1. Plan rows (base, permutation, ablation) from the request.
//! 2. Group rows by prefix and ask the backend for label logits.
//! 3. Reduce each question's rows to one answer: calibrated probabilities, confidence, order
//!    sensitivity, abstention, evidence.
//! 4. Apply declared constraints across noul answers.

use indexmap::{IndexMap, IndexSet};
use serde::Serialize;
use serde_json::Value;

use crate::backend::Backend;
use crate::calibration::{Calibrator, fused_probability};
use crate::error::{Error, Result};
use crate::labels::assign_labels;
use crate::primitives::{
    argmax, choice_confidence, expected_score, mean_rows, order_sensitivity, pairwise_sum, score_confidence, sigmoid,
    softmax, top_margin, total_variation,
};
use crate::py;
use crate::render::{Layout, Row, plan_rows, render_content};
use crate::schema::{
    Constraint, ConstraintKind, DEFAULT_MODEL, Kind, MAX_CHOICE_OPTIONS, Options, Question, Request, py_repr,
};

/// The official SDKs send `jev-latest` unless told otherwise, so it names the loaded model too.
pub const DEFAULT_ALIASES: [&str; 2] = [DEFAULT_MODEL, "jev-latest"];
pub const MAX_EVIDENCE: usize = 3;
/// A schema-valid request can plan thousands of rows; these bound what one call can ask.
pub const DEFAULT_MAX_ROWS: usize = 512;
pub const DEFAULT_MAX_INPUT_TOKENS: usize = 32768;
const CONSTRAINT_ROUNDS: usize = 2000;
const CONSTRAINT_TOLERANCE: f64 = 1e-10;

// --------------------------------------------------------------------------- answers

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Evidence {
    pub span: String,
    /// Total variation distance the answer moved when this span was removed.
    pub effect: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct NoulAnswer {
    pub noul: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub abstain: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub evidence: Option<Vec<Evidence>>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ChoiceAnswer {
    pub choice: String,
    pub probabilities: IndexMap<String, f64>,
    pub confidence: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub order_sensitivity: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub abstain: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub evidence: Option<Vec<Evidence>>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ScoreAnswer {
    pub score: f64,
    pub legend: IndexMap<String, Value>,
    pub probabilities: IndexMap<String, f64>,
    pub confidence: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub abstain: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub evidence: Option<Vec<Evidence>>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct MultiAnswer {
    pub probabilities: IndexMap<String, f64>,
    pub selected: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub evidence: Option<Vec<Evidence>>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum Answer {
    Noul(NoulAnswer),
    Choice(ChoiceAnswer),
    Score(ScoreAnswer),
    Multi(MultiAnswer),
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Usage {
    pub input_tokens: usize,
    pub output_tokens: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Response {
    pub model: String,
    pub answers: IndexMap<String, Answer>,
    pub usage: Usage,
}

impl Response {
    pub fn to_json(&self) -> Value {
        serde_json::to_value(self).expect("responses serialize")
    }
}

// --------------------------------------------------------------------------- engine

pub struct EngineOptions {
    pub version: String,
    /// Rows one request may plan; None for no limit.
    pub max_rows: Option<usize>,
    /// Input tokens one request may need; None for no limit.
    pub max_input_tokens: Option<usize>,
    /// The prompt layout; a pack's calibrator is valid only for the layout it was fitted on.
    pub layout: Layout,
}

impl Default for EngineOptions {
    fn default() -> Self {
        EngineOptions {
            version: env!("CARGO_PKG_VERSION").to_string(),
            max_rows: Some(DEFAULT_MAX_ROWS),
            max_input_tokens: Some(DEFAULT_MAX_INPUT_TOKENS),
            layout: Layout::V1,
        }
    }
}

struct Scored {
    probs: Vec<f64>,
    logits: Vec<f64>,
}

pub struct Engine {
    backend: Box<dyn Backend>,
    /// Used by every question whose request names no calibrator.
    pub calibrator: Calibrator,
    /// Calibrators fitted per decision, chosen by `lars.calibrator` or `lars.calibrators`.
    pub named_calibrators: IndexMap<String, Calibrator>,
    options: EngineOptions,
    labels: Vec<String>,
}

impl Engine {
    pub fn new(backend: Box<dyn Backend>, calibrator: Calibrator, options: EngineOptions) -> Result<Engine> {
        let labels = assign_labels(MAX_CHOICE_OPTIONS, |label| backend.is_single_token(label))?;
        Ok(Engine { backend, calibrator, named_calibrators: IndexMap::new(), options, labels })
    }

    pub fn with_named_calibrators(mut self, named: IndexMap<String, Calibrator>) -> Engine {
        self.named_calibrators = named;
        self
    }

    pub fn model_id(&self) -> String {
        format!("lars-{}+{}:{}", self.options.version, self.backend.name(), self.backend.model_name())
    }

    pub fn labels(&self) -> &[String] {
        &self.labels
    }

    pub fn backend(&self) -> &dyn Backend {
        self.backend.as_ref()
    }

    /// Answer a request.
    pub fn evaluate(&mut self, request: &Request) -> Result<Response> {
        let model_id = self.model_id();
        let (rows, usage) = self.plan(request, &model_id)?;
        let scored = self.score(request, &rows)?;
        let mut answers = self.reduce(request, &rows, &scored)?;
        apply_constraints(&mut answers, &request.lars.constraints);
        Ok(Response { model: model_id, answers, usage })
    }

    /// Rows and usage for a request, after the model-name and budget checks.
    pub fn plan(&self, request: &Request, model_id: &str) -> Result<(Vec<Row>, Usage)> {
        if !DEFAULT_ALIASES.contains(&request.model.as_str()) && request.model != model_id {
            return Err(Error::UnknownModel(format!(
                "model {} is not served here; use {} or {}",
                py_repr(&request.model),
                py_repr(DEFAULT_MODEL),
                py_repr(model_id)
            )));
        }
        let rows = plan_rows(request, &self.labels);
        if let Some(max) = self.options.max_rows.filter(|max| rows.len() > *max) {
            return Err(Error::Budget(format!(
                "request plans {} model rows, over the budget of {max}; use fewer questions, options, \
                 permutations, or turn off explain",
                rows.len()
            )));
        }
        let usage = self.usage(request, &rows);
        if let Some(max) = self.options.max_input_tokens.filter(|max| usage.input_tokens > *max) {
            return Err(Error::Budget(format!(
                "request needs {} input tokens, over the budget of {max}; shorten the state or turn off explain",
                usage.input_tokens
            )));
        }
        Ok((rows, usage))
    }

    /// The calibrator each question uses: its named one if the request names one, else the default.
    pub fn calibrators_for<'a>(&'a self, request: &Request) -> Result<IndexMap<String, &'a Calibrator>> {
        let options = &request.lars;
        let mut unknown: Vec<&String> = options
            .calibrator
            .iter()
            .chain(options.calibrators.values())
            .filter(|name| !self.named_calibrators.contains_key(*name))
            .collect::<IndexSet<_>>()
            .into_iter()
            .collect();
        unknown.sort();
        if !unknown.is_empty() {
            let mut known: Vec<&String> = self.named_calibrators.keys().collect();
            known.sort();
            let known =
                if known.is_empty() { "none are loaded".to_string() } else { join(known.iter().map(|s| s.as_str())) };
            return Err(Error::Invalid(format!(
                "unknown calibrator {}; known: {known}",
                join(unknown.iter().map(|s| py_repr(s)))
            )));
        }
        let mut extra: Vec<&String> =
            options.calibrators.keys().filter(|q| !request.questions.contains_key(*q)).collect();
        extra.sort();
        if !extra.is_empty() {
            return Err(Error::Invalid(format!(
                "lars.calibrators names questions not in the request: {}",
                join(extra.iter().map(|s| s.as_str()))
            )));
        }
        Ok(request
            .questions
            .keys()
            .map(|qid| {
                let name = options.calibrators.get(qid).or(options.calibrator.as_ref());
                (qid.clone(), name.map_or(&self.calibrator, |n| &self.named_calibrators[n]))
            })
            .collect())
    }

    /// Uncalibrated label logits for one question's base row, for evaluation and calibration.
    pub fn raw_logits(&mut self, state: &Value, question: &Question) -> Result<(Vec<String>, Vec<f64>)> {
        let request = Request {
            state: state.clone(),
            model: DEFAULT_MODEL.into(),
            questions: IndexMap::from([("q".to_string(), question.clone())]),
            lars: Options::default(),
        };
        let rows: Vec<Row> = plan_rows(&request, &self.labels).into_iter().filter(|r| r.variant == "base").collect();
        let row = rows.first().ok_or_else(|| Error::Invalid("this question has no single base row".into()))?;
        let scored = self.score(&request, std::slice::from_ref(row))?;
        Ok((row.keys.clone(), scored.into_iter().next().unwrap().logits))
    }

    /// Every planned row of a request with its raw label logits, for diagnostics.
    pub fn raw_rows(&mut self, request: &Request) -> Result<Vec<(Row, Vec<f64>)>> {
        let model_id = self.model_id();
        let (rows, _) = self.plan(request, &model_id)?;
        let scored = self.score(request, &rows)?;
        Ok(rows.into_iter().zip(scored).map(|(row, s)| (row, s.logits)).collect())
    }

    // ----------------------------------------------------------------- scoring

    fn score(&mut self, request: &Request, rows: &[Row]) -> Result<Vec<Scored>> {
        let temperatures: Vec<f64> = {
            let calibrators = self.calibrators_for(request)?;
            rows.iter().map(|row| calibrators[&row.question_id].temperature_for(row.kind)).collect()
        };
        let base_state = render_content(&request.state);
        let template = self.backend.template().clone();
        let mut groups: IndexMap<String, Vec<(usize, String)>> = IndexMap::new();
        for (index, row) in rows.iter().enumerate() {
            let state_text = row.state_text.as_deref().unwrap_or(&base_state);
            let (prefix, suffix) = self.options.layout.compose(&template, state_text, &row.suffix_body);
            groups.entry(prefix).or_default().push((index, suffix));
        }
        let mut scored: Vec<Option<Scored>> = (0..rows.len()).map(|_| None).collect();
        for (prefix, members) in &groups {
            let suffixes: Vec<String> = members.iter().map(|(_, s)| s.clone()).collect();
            let labels: Vec<Vec<String>> = members.iter().map(|(i, _)| rows[*i].labels.clone()).collect();
            let logits = self.backend.label_logits(prefix, &suffixes, &labels)?;
            if logits.len() != members.len() {
                return Err(Error::Backend(format!("backend returned {} rows for {}", logits.len(), members.len())));
            }
            for ((index, _), row_logits) in members.iter().zip(logits) {
                if row_logits.len() != rows[*index].labels.len() {
                    return Err(Error::Backend("backend returned the wrong number of label logits".into()));
                }
                let probs = softmax(&row_logits, temperatures[*index]);
                scored[*index] = Some(Scored { probs, logits: row_logits });
            }
        }
        Ok(scored.into_iter().map(|s| s.expect("every row scored")).collect())
    }

    // ----------------------------------------------------------------- reduction

    fn reduce(&self, request: &Request, rows: &[Row], scored: &[Scored]) -> Result<IndexMap<String, Answer>> {
        let options = &request.lars;
        let calibrators = self.calibrators_for(request)?;
        let mut by_question: IndexMap<&str, Vec<usize>> = IndexMap::new();
        for (i, row) in rows.iter().enumerate() {
            by_question.entry(row.question_id.as_str()).or_default().push(i);
        }
        let mut answers = IndexMap::new();
        for (qid, question) in &request.questions {
            let cal = calibrators[qid];
            let items = by_question.get(qid.as_str()).cloned().unwrap_or_default();
            let variant = |i: &usize| rows[*i].variant.as_str();
            let base: Vec<usize> =
                items.iter().filter(|i| variant(i) == "base" || variant(i).starts_with("base:")).copied().collect();
            let perms: Vec<usize> = items.iter().filter(|i| variant(i).starts_with("perm:")).copied().collect();
            let ablations: Vec<usize> = items.iter().filter(|i| variant(i).starts_with("ablate:")).copied().collect();
            let span = |i: usize| rows[i].ablated_span.clone();

            let answer = match question {
                Question::Noul { .. } => {
                    let p_yes = noul_prob(cal, yes_logit(&scored[base[0]]), Kind::Noul, options.features.get(qid));
                    let effects: Vec<(Option<String>, f64)> = ablations
                        .iter()
                        .map(|&i| (span(i), (noul_prob(cal, yes_logit(&scored[i]), Kind::Noul, None) - p_yes).abs()))
                        .collect();
                    Answer::Noul(NoulAnswer {
                        noul: py::round(p_yes, 4),
                        abstain: options.abstain_margin.map(|m| (p_yes - 0.5).abs() * 2.0 < m),
                        evidence: evidence(effects, options.explain),
                    })
                }
                Question::Choice { criteria, .. } => {
                    let keys: Vec<&String> = criteria.keys().collect();
                    let dists: Vec<Vec<f64>> =
                        base.iter().chain(&perms).map(|&i| realign(&rows[i], &scored[i], &keys)).collect();
                    let mean = mean_rows(&dists);
                    let effects: Vec<(Option<String>, f64)> = ablations
                        .iter()
                        .map(|&i| (span(i), total_variation(&realign(&rows[i], &scored[i], &keys), &mean)))
                        .collect();
                    Answer::Choice(ChoiceAnswer {
                        choice: keys[argmax(&mean)].clone(),
                        probabilities: round_probs(keys.iter().map(|k| k.as_str()), &mean),
                        confidence: py::round(choice_confidence(&mean), 4),
                        order_sensitivity: (!perms.is_empty()).then(|| py::round(order_sensitivity(&dists), 4)),
                        abstain: options.abstain_margin.map(|m| top_margin(&mean) < m),
                        evidence: evidence(effects, options.explain),
                    })
                }
                Question::Score { criteria, .. } => {
                    let probs = &scored[base[0]].probs;
                    let legend = criteria
                        .iter()
                        .enumerate()
                        .map(|(i, c)| {
                            (i.to_string(), if c.is_null() { Value::from(format!("level {i}")) } else { c.clone() })
                        })
                        .collect();
                    let effects: Vec<(Option<String>, f64)> =
                        ablations.iter().map(|&i| (span(i), total_variation(&scored[i].probs, probs))).collect();
                    Answer::Score(ScoreAnswer {
                        score: py::round(expected_score(probs), 4),
                        legend,
                        probabilities: round_probs(rows[base[0]].keys.iter().map(String::as_str), probs),
                        confidence: py::round(score_confidence(probs), 4),
                        abstain: options.abstain_margin.map(|m| top_margin(probs) < m),
                        evidence: evidence(effects, options.explain),
                    })
                }
                Question::Multi { .. } => {
                    // One yes/no row per option: variants "base:<key>" and "ablate:<i>:<key>".
                    let mut per_option: IndexMap<String, f64> = IndexMap::new();
                    for &i in &base {
                        let key = rows[i].variant.split_once(':').unwrap().1;
                        per_option.insert(
                            key.to_string(),
                            py::round(noul_prob(cal, yes_logit(&scored[i]), Kind::Multi, None), 4),
                        );
                    }
                    let mut by_unit: IndexMap<String, Vec<f64>> = IndexMap::new();
                    let mut spans: IndexMap<String, Option<String>> = IndexMap::new();
                    for &i in &ablations {
                        let mut parts = rows[i].variant.splitn(3, ':');
                        let (_, unit, key) = (parts.next(), parts.next().unwrap(), parts.next().unwrap());
                        spans.insert(unit.to_string(), span(i));
                        let ablated = noul_prob(cal, yes_logit(&scored[i]), Kind::Multi, None);
                        by_unit
                            .entry(unit.to_string())
                            .or_default()
                            .push((ablated - per_option.get(key).copied().unwrap_or(0.0)).abs());
                    }
                    let effects = by_unit
                        .iter()
                        .map(|(unit, values)| (spans[unit].clone(), pairwise_sum(values) / values.len() as f64))
                        .collect();
                    Answer::Multi(MultiAnswer {
                        selected: per_option.iter().filter(|(_, p)| **p >= 0.5).map(|(k, _)| k.clone()).collect(),
                        probabilities: per_option,
                        evidence: evidence(effects, options.explain),
                    })
                }
            };
            answers.insert(qid.clone(), answer);
        }
        Ok(answers)
    }

    // ----------------------------------------------------------------- usage

    fn usage(&self, request: &Request, rows: &[Row]) -> Usage {
        let template = self.backend.template();
        let base_state = render_content(&request.state);
        let mut seen: IndexSet<String> = IndexSet::new();
        let mut tokens = 0;
        for row in rows {
            let state_text = row.state_text.as_deref().unwrap_or(&base_state);
            let (prefix, suffix) = self.options.layout.compose(template, state_text, &row.suffix_body);
            tokens += self.backend.count_tokens(&suffix);
            if seen.insert(prefix.clone()) {
                tokens += self.backend.count_tokens(&prefix);
            }
        }
        Usage { input_tokens: tokens, output_tokens: rows.len() }
    }
}

fn join<S: AsRef<str>>(items: impl Iterator<Item = S>) -> String {
    items.map(|s| s.as_ref().to_string()).collect::<Vec<_>>().join(", ")
}

/// Raw yes-minus-no logit for a two-label row.
fn yes_logit(item: &Scored) -> f64 {
    item.logits[0] - item.logits[1]
}

/// P(yes) from the raw yes-minus-no logit.
///
/// With caller evidence and a fusion fitted on the same feature names, P(yes) is the fused
/// logistic. Otherwise a fitted Platt pair gives sigmoid(a * z + b), or the kind's temperature
/// applies: sigmoid(z / T), the special case a = 1/T, b = 0.
pub fn noul_prob(cal: &Calibrator, yes_logit: f64, kind: Kind, features: Option<&IndexMap<String, f64>>) -> f64 {
    if let Some(fusion) = cal.fusion_for(kind, features) {
        return fused_probability(fusion, yes_logit, features.unwrap());
    }
    match cal.platt_for(kind) {
        None => sigmoid(yes_logit / cal.temperature_for(kind)),
        Some((a, b)) => sigmoid(a * yes_logit + b),
    }
}

/// A row's distribution (in presented order) mapped back to canonical key order.
fn realign(row: &Row, item: &Scored, canonical: &[&String]) -> Vec<f64> {
    let by_key: IndexMap<&str, f64> = row.keys.iter().map(String::as_str).zip(item.probs.iter().copied()).collect();
    canonical.iter().map(|k| by_key[k.as_str()]).collect()
}

fn round_probs<'a>(keys: impl Iterator<Item = &'a str>, probs: &[f64]) -> IndexMap<String, f64> {
    keys.zip(probs).map(|(k, p)| (k.to_string(), py::round(*p, 4))).collect()
}

/// The top spans by how far the answer moved when each was removed from the state.
fn evidence(effects: Vec<(Option<String>, f64)>, enabled: bool) -> Option<Vec<Evidence>> {
    if !enabled || effects.is_empty() {
        return None;
    }
    let mut ranked: Vec<(String, f64)> = effects.into_iter().map(|(s, v)| (s.unwrap_or_default(), v)).collect();
    ranked.sort_by(|a, b| b.1.total_cmp(&a.1)); // stable, as Python's sorted
    Some(
        ranked
            .into_iter()
            .take(MAX_EVIDENCE)
            .filter(|(_, v)| *v > 0.0)
            .map(|(span, v)| Evidence { span, effect: py::round(v, 4) })
            .collect(),
    )
}

// --------------------------------------------------------------------------- constraints

/// Enforce the declared constraints on noul answers, jointly.
///
/// Constraints that share no question are applied as declared: a complement averages the
/// pair, an exclusive group over 1 is rescaled. When constraints share a question, one pass in
/// order can undo an earlier one, so the answers become the nearest probabilities, in least
/// squares, that satisfy all of them (Dykstra's alternating projections). Rounding comes last:
/// exclusive members round down so their sum cannot exceed 1, and a complement's second answer
/// is 1 minus its rounded first.
pub fn apply_constraints(answers: &mut IndexMap<String, Answer>, constraints: &[Constraint]) {
    let applicable: Vec<&Constraint> = constraints
        .iter()
        .filter(|c| c.questions.iter().all(|q| matches!(answers.get(q), Some(Answer::Noul(_)))))
        .collect();
    if applicable.is_empty() {
        return;
    }
    let noul = |q: &str| match &answers[q] {
        Answer::Noul(a) => a.noul,
        _ => unreachable!(),
    };
    let mut p: IndexMap<String, f64> = IndexMap::new();
    for c in &applicable {
        for q in &c.questions {
            p.entry(q.clone()).or_insert_with(|| noul(q));
        }
    }
    let members: Vec<&String> = applicable.iter().flat_map(|c| &c.questions).collect();
    let distinct: IndexSet<&String> = members.iter().copied().collect();
    if members.len() == distinct.len() {
        for c in &applicable {
            let qs = &c.questions;
            match c.kind {
                ConstraintKind::Complement => {
                    let x = (p[&qs[0]] + 1.0 - p[&qs[1]]) / 2.0;
                    p[&qs[0]] = x;
                    p[&qs[1]] = 1.0 - x;
                }
                ConstraintKind::Exclusive => {
                    let total = qs.iter().fold(0.0, |acc, q| acc + p[q]);
                    if total > 1.0 {
                        for q in qs {
                            p[q] /= total;
                        }
                    }
                }
            }
        }
    } else {
        p = project_constraints(&p, &applicable);
    }
    let exclusive: IndexSet<&String> =
        applicable.iter().filter(|c| c.kind == ConstraintKind::Exclusive).flat_map(|c| &c.questions).collect();
    let mut rounded: IndexMap<String, f64> = p
        .iter()
        .map(|(q, v)| (q.clone(), if exclusive.contains(q) { (v * 1e4).floor() / 1e4 } else { py::round(*v, 4) }))
        .collect();
    for c in &applicable {
        if c.kind == ConstraintKind::Complement {
            let value = py::round(1.0 - rounded[&c.questions[0]], 4);
            rounded[&c.questions[1]] = value;
        }
    }
    for (q, value) in rounded {
        if let Some(Answer::Noul(a)) = answers.get_mut(&q) {
            a.noul = value;
        }
    }
}

/// The nearest point to `p`, in least squares, satisfying every constraint and 0 <= p <= 1, by
/// Dykstra's algorithm over each complement's line, each exclusive group's half-space and the box.
fn project_constraints(p: &IndexMap<String, f64>, constraints: &[&Constraint]) -> IndexMap<String, f64> {
    enum Set {
        Complement(usize, usize),
        Exclusive(Vec<usize>),
        Box,
    }
    let keys: Vec<&String> = p.keys().collect();
    let index = |q: &String| keys.iter().position(|k| *k == q).unwrap();
    let mut x: Vec<f64> = p.values().copied().collect();
    let mut sets: Vec<Set> = constraints
        .iter()
        .map(|c| match c.kind {
            ConstraintKind::Complement => Set::Complement(index(&c.questions[0]), index(&c.questions[1])),
            ConstraintKind::Exclusive => Set::Exclusive(c.questions.iter().map(index).collect()),
        })
        .collect();
    sets.push(Set::Box);
    let mut increments = vec![vec![0.0; x.len()]; sets.len()];
    for _ in 0..CONSTRAINT_ROUNDS {
        let previous = x.clone();
        for (k, set) in sets.iter().enumerate() {
            let y: Vec<f64> = x.iter().zip(&increments[k]).map(|(a, b)| a + b).collect();
            let mut z = y.clone();
            match set {
                Set::Complement(a, b) => {
                    let shift = (y[*a] + y[*b] - 1.0) / 2.0;
                    z[*a] -= shift;
                    z[*b] -= shift;
                }
                Set::Exclusive(idx) => {
                    let total = pairwise_sum(&idx.iter().map(|&i| y[i]).collect::<Vec<_>>());
                    if total > 1.0 {
                        for &i in idx {
                            z[i] -= (total - 1.0) / idx.len() as f64;
                        }
                    }
                }
                Set::Box => z.iter_mut().for_each(|v| *v = v.clamp(0.0, 1.0)),
            }
            increments[k] = y.iter().zip(&z).map(|(a, b)| a - b).collect();
            x = z;
        }
        let change = x.iter().zip(&previous).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
        if change < CONSTRAINT_TOLERANCE {
            break;
        }
    }
    keys.into_iter().cloned().zip(x).collect()
}
