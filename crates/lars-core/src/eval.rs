//! Labeled evaluation sets, as the reference's `evalset.py`.
//!
//! One JSON object per line with `state`, `question` and `label`, optionally `soft_label` and
//! `features`. `state` and `question` may be JSON values or JSON-encoded strings (the jev-bench
//! convention). `label` is the option key for choice, the level index as a string for score,
//! and "0" or "1" (or "yes", "true") for noul.

use std::time::Instant;

use indexmap::IndexMap;
use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};

use crate::calibration::{Calibrator, brier, coverage_at_error, ece, fit_fusion, fit_platt, fit_temperature};
use crate::engine::{Engine, noul_prob};
use crate::error::{Error, Result};
use crate::primitives::{argmax, max, pairwise_sum, softmax};
use crate::py;
use crate::schema::{Kind, Request};

#[derive(Debug, Clone, PartialEq)]
pub struct Example {
    pub state: Value,
    pub question: Value,
    pub label: String,
    pub soft_label: Option<Map<String, Value>>,
    /// Named numeric evidence for a noul, fused with the model when a calibration is fitted with it.
    pub features: Option<IndexMap<String, f64>>,
}

impl Example {
    /// A stable content hash, so per-row dumps from two runs can be checked for alignment.
    pub fn id(&self) -> String {
        let blob = py::dumps(&json!([self.state, self.question, self.label]), None, true);
        Sha256::digest(blob.as_bytes()).iter().take(8).map(|b| format!("{b:02x}")).collect()
    }
}

/// Decode a JSON-encoded string (object, array, or quoted string) when that is what it is.
pub fn maybe_json(value: &Value) -> Value {
    if let Value::String(s) = value {
        let stripped = py::strip(s);
        if stripped.is_empty() || stripped.starts_with(['{', '[', '"']) {
            if let Ok(decoded) = serde_json::from_str::<Value>(stripped) {
                return decoded;
            }
        }
    }
    value.clone()
}

fn as_float(value: &Value) -> Option<f64> {
    match value {
        Value::Number(n) => n.as_f64(),
        Value::String(s) => py::strip(s).parse().ok(),
        Value::Bool(b) => Some(if *b { 1.0 } else { 0.0 }),
        _ => None,
    }
}

/// Every example in JSONL text.
pub fn read_examples(text: &str) -> Result<Vec<Example>> {
    let mut examples = Vec::new();
    for (number, line) in text.lines().enumerate() {
        if py::strip(line).is_empty() {
            continue;
        }
        let bad = |what: &str| Error::Invalid(format!("line {}: {what}", number + 1));
        let raw: Value = serde_json::from_str(line).map_err(|e| bad(&e.to_string()))?;
        let question = maybe_json(raw.get("question").ok_or_else(|| bad("no question"))?);
        if !question.is_object() {
            return Err(bad("question must be a JSON object"));
        }
        let soft = raw.get("soft_label").map(maybe_json);
        let features = match raw.get("features") {
            Some(Value::Object(map)) => Some(
                map.iter()
                    .map(|(k, v)| as_float(v).map(|x| (k.clone(), x)).ok_or_else(|| bad("feature is not a number")))
                    .collect::<Result<IndexMap<_, _>>>()?,
            ),
            _ => None,
        };
        examples.push(Example {
            state: maybe_json(raw.get("state").ok_or_else(|| bad("no state"))?),
            question,
            label: py::str_of(raw.get("label").ok_or_else(|| bad("no label"))?),
            soft_label: match soft {
                Some(Value::Object(map)) => Some(map),
                _ => None,
            },
            features,
        });
    }
    Ok(examples)
}

/// Raw logits for each example's question, and how long the model took per example.
pub struct Collected {
    pub rows: Vec<(Kind, Vec<String>, Vec<f64>)>,
    pub ms_per_example: f64,
}

pub fn collect(engine: &mut Engine, examples: &[Example]) -> Result<Collected> {
    let started = Instant::now();
    let mut rows = Vec::with_capacity(examples.len());
    for example in examples {
        let request = Request::from_value(&json!({"state": example.state, "questions": {"q": example.question}}))?;
        let question = &request.questions["q"];
        if question.kind() == Kind::Multi {
            return Err(Error::Invalid(
                "eval and calibrate do not support multi questions yet (one label cannot score per-option answers); \
                 split them into one noul per option"
                    .into(),
            ));
        }
        let (keys, logits) = engine.raw_logits(&example.state, question)?;
        rows.push((question.kind(), keys, logits));
    }
    let ms = started.elapsed().as_secs_f64() * 1000.0 / examples.len().max(1) as f64;
    Ok(Collected { rows, ms_per_example: ms })
}

fn noul_label(label: &str) -> &'static str {
    if matches!(label, "1" | "true" | "yes") { "yes" } else { "no" }
}

fn target_vector(example: &Example, keys: &[String]) -> Vec<f64> {
    if let Some(soft) = example.soft_label.as_ref().filter(|s| !s.is_empty()) {
        let vec: Vec<f64> = keys.iter().map(|k| soft.get(k).and_then(as_float).unwrap_or(0.0)).collect();
        let total = pairwise_sum(&vec);
        return if total > 0.0 { vec.iter().map(|v| v / total).collect() } else { vec };
    }
    keys.iter().map(|k| if *k == example.label { 1.0 } else { 0.0 }).collect()
}

/// [P(yes), P(no)]: the normalized soft label when there is one, else the hard label.
fn noul_target(example: &Example) -> Vec<f64> {
    if let Some(soft) = example.soft_label.as_ref().filter(|s| !s.is_empty()) {
        let (mut yes, mut total) = (0.0, 0.0);
        for (k, v) in soft {
            let v = as_float(v).unwrap_or(0.0);
            if noul_label(&k.to_lowercase()) == "yes" {
                yes += v;
            }
            total += v;
        }
        if total > 0.0 {
            return vec![yes / total, 1.0 - yes / total];
        }
    }
    if noul_label(&example.label) == "yes" { vec![1.0, 0.0] } else { vec![0.0, 1.0] }
}

fn is_binary(kind: Kind) -> bool {
    matches!(kind, Kind::Noul | Kind::Multi)
}

#[derive(Debug, Clone, PartialEq)]
pub struct EvalResult {
    pub count: usize,
    pub accuracy: f64,
    pub ece: f64,
    pub brier: f64,
    pub coverage_at_5pct: f64,
    pub threshold_at_5pct: f64,
    pub per_kind: IndexMap<String, (usize, f64)>,
    pub ms_per_example: f64,
    pub temperatures: IndexMap<String, f64>,
    pub platt: IndexMap<String, Option<(f64, f64)>>,
}

impl EvalResult {
    pub fn to_json(&self) -> Value {
        let per_kind: Map<String, Value> =
            self.per_kind.iter().map(|(k, (n, acc))| (k.clone(), json!({"count": n, "accuracy": acc}))).collect();
        let platt: Map<String, Value> =
            self.platt.iter().map(|(k, p)| (k.clone(), p.map_or(Value::Null, |(a, b)| json!([a, b])))).collect();
        json!({
            "count": self.count, "accuracy": self.accuracy, "ece": self.ece, "brier": self.brier,
            "coverage_at_5pct": self.coverage_at_5pct, "threshold_at_5pct": self.threshold_at_5pct,
            "per_kind": per_kind, "ms_per_example": self.ms_per_example,
            "temperatures": self.temperatures, "platt": platt,
        })
    }
}

/// Score examples with the engine's default calibrator. With `dump`, each example's
/// probabilities, target and hit are appended in input order.
pub fn evaluate(engine: &mut Engine, examples: &[Example], dump: Option<&mut Vec<Value>>) -> Result<EvalResult> {
    let collected = collect(engine, examples)?;
    evaluate_collected(engine, examples, &collected, dump)
}

/// `evaluate` on logits already collected for the same examples, without another model pass.
pub fn evaluate_collected(
    engine: &Engine,
    examples: &[Example],
    collected: &Collected,
    mut dump: Option<&mut Vec<Value>>,
) -> Result<EvalResult> {
    let cal = &engine.calibrator;
    let (mut confidences, mut correct, mut probs, mut targets) = (vec![], vec![], vec![], vec![]);
    let mut per_kind: IndexMap<String, Vec<bool>> = IndexMap::new();
    for (example, (kind, keys, logits)) in examples.iter().zip(&collected.rows) {
        let (p, label, target) = if is_binary(*kind) {
            let p_yes = noul_prob(cal, logits[0] - logits[1], *kind, example.features.as_ref());
            (vec![p_yes, 1.0 - p_yes], noul_label(&example.label).to_string(), noul_target(example))
        } else {
            (softmax(logits, cal.temperature_for(*kind)), example.label.clone(), target_vector(example, keys))
        };
        let hit = keys[argmax(&p)] == label;
        confidences.push(max(&p));
        correct.push(if hit { 1.0 } else { 0.0 });
        per_kind.entry(kind.as_str().to_string()).or_default().push(hit);
        if let Some(rows) = dump.as_deref_mut() {
            rows.push(json!({"id": example.id(), "keys": keys, "p": p, "target": target, "hit": hit}));
        }
        probs.push(p);
        targets.push(target);
    }
    let (coverage, threshold) = coverage_at_error(&confidences, &correct, 0.05);
    let mut kinds: Vec<Kind> = collected.rows.iter().map(|(k, _, _)| *k).collect();
    kinds.sort_by_key(|k| k.as_str());
    kinds.dedup();
    Ok(EvalResult {
        count: collected.rows.len(),
        accuracy: if correct.is_empty() { 0.0 } else { pairwise_sum(&correct) / correct.len() as f64 },
        ece: ece(&confidences, &correct, 15),
        brier: brier(&probs, &targets),
        coverage_at_5pct: coverage,
        threshold_at_5pct: threshold,
        per_kind: per_kind
            .into_iter()
            .map(|(k, hits)| {
                let n = hits.len();
                (k, (n, hits.iter().filter(|h| **h).count() as f64 / n as f64))
            })
            .collect(),
        ms_per_example: collected.ms_per_example,
        temperatures: kinds.iter().map(|k| (k.as_str().to_string(), cal.temperature_for(*k))).collect(),
        platt: kinds.iter().map(|k| (k.as_str().to_string(), cal.platt_for(*k))).collect(),
    })
}

/// Fit per-kind temperatures and Platt scalings (and a fusion where every example of a kind
/// carries features).
pub fn calibrate(engine: &mut Engine, examples: &[Example], source: Option<&str>) -> Result<Calibrator> {
    let collected = collect(engine, examples)?;
    Ok(calibrate_collected(examples, &collected, source))
}

/// `calibrate` on logits already collected for the same examples.
pub fn calibrate_collected(examples: &[Example], collected: &Collected, source: Option<&str>) -> Calibrator {
    let mut calibrator = Calibrator { fitted_on: source.map(String::from), ..Calibrator::default() };
    let mut by_kind: IndexMap<Kind, Vec<(usize, Vec<f64>)>> = IndexMap::new();
    for (i, (example, (kind, keys, _))) in examples.iter().zip(&collected.rows).enumerate() {
        let target = if is_binary(*kind) { noul_target(example) } else { target_vector(example, keys) };
        by_kind.entry(*kind).or_default().push((i, target));
    }
    for (kind, members) in &by_kind {
        let logits: Vec<&Vec<f64>> = members.iter().map(|(i, _)| &collected.rows[*i].2).collect();
        if is_binary(*kind) {
            // Platt on the raw yes-minus-no logit subsumes temperature (a = 1/T) and adds a bias.
            let z: Vec<f64> = logits.iter().map(|row| row[0] - row[1]).collect();
            let y: Vec<f64> = members.iter().map(|(_, t)| t[0]).collect();
            calibrator.platt.insert(kind.as_str().into(), fit_platt(&z, &y, 1e-2, 100));
            calibrator.temperatures.insert(kind.as_str().into(), 1.0);
            let evidence: Vec<&IndexMap<String, f64>> =
                members.iter().filter_map(|(i, _)| examples[*i].features.as_ref().filter(|f| !f.is_empty())).collect();
            if !evidence.is_empty() && evidence.len() == members.len() {
                let owned: Vec<IndexMap<String, f64>> = evidence.into_iter().cloned().collect();
                calibrator.fusion.insert(kind.as_str().into(), fit_fusion(&z, &owned, &y, 1e-2));
            }
        } else {
            let rows: Vec<Vec<f64>> = logits.into_iter().cloned().collect();
            let targets: Vec<Vec<f64>> = members.iter().map(|(_, t)| t.clone()).collect();
            calibrator.temperatures.insert(kind.as_str().into(), fit_temperature(&rows, &targets));
        }
    }
    calibrator
}
