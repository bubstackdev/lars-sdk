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
    /// The human target as a normalized distribution in option order, when the row has one.
    pub soft_label: Option<Vec<f64>>,
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
        let keys = option_keys(&question).map_err(|e| bad(&e))?;
        let label = py::str_of(raw.get("label").ok_or_else(|| bad("no label"))?);
        let soft = soft_target(&keys, raw.get("soft_label").unwrap_or(&Value::Null)).map_err(|e| bad(&e))?;
        hard_index(&keys, &label).map_err(|e| bad(&e))?;
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
            label,
            soft_label: soft,
            features,
        });
    }
    Ok(examples)
}

/// Raw logits for each example's question, and how long the model took per example.
pub struct Collected {
    pub rows: Vec<(Kind, Vec<String>, Vec<f64>)>,
    /// The final hidden state per example, when asked for (or needed by the engine's calibrator)
    /// and the backend can read it.
    pub hidden: Option<Vec<Vec<f32>>>,
    pub ms_per_example: f64,
}

pub fn collect(engine: &mut Engine, examples: &[Example]) -> Result<Collected> {
    collect_with(engine, examples, false)
}

/// `collect`, also reading hidden states when `want_hidden` and the backend can.
pub fn collect_with(engine: &mut Engine, examples: &[Example], want_hidden: bool) -> Result<Collected> {
    let started = Instant::now();
    let mut rows = Vec::with_capacity(examples.len());
    let mut hidden = Vec::with_capacity(examples.len());
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
        let (keys, logits, h) = engine.raw_row(&example.state, question, want_hidden)?;
        rows.push((question.kind(), keys, logits));
        hidden.push(h);
    }
    let ms = started.elapsed().as_secs_f64() * 1000.0 / examples.len().max(1) as f64;
    let hidden = hidden.into_iter().collect::<Option<Vec<_>>>().filter(|h| !h.is_empty());
    Ok(Collected { rows, hidden, ms_per_example: ms })
}

fn noul_label(label: &str) -> &'static str {
    if matches!(label, "1" | "true" | "yes") { "yes" } else { "no" }
}

/// A question's option keys, as targets are ordered: yes/no for binary questions, level
/// indices for score, criteria keys for choice.
pub fn option_keys(question: &Value) -> std::result::Result<Vec<String>, String> {
    let criteria = &question["criteria"];
    match question["type"].as_str() {
        Some("noul" | "multi") => Ok(vec!["yes".into(), "no".into()]),
        Some("score") => Ok((0..criteria.as_array().map_or(0, Vec::len)).map(|i| i.to_string()).collect()),
        Some("choice") => Ok(criteria.as_object().map(|m| m.keys().cloned().collect()).unwrap_or_default()),
        _ => Err("question has no known type".into()),
    }
}

fn is_binary_keys(keys: &[String]) -> bool {
    keys == ["yes", "no"]
}

/// yes/1/true or no/0/false, case-insensitive; anything else is an error.
fn binary_key(value: &str) -> std::result::Result<&'static str, String> {
    match value.to_lowercase().as_str() {
        "yes" | "1" | "true" => Ok("yes"),
        "no" | "0" | "false" => Ok("no"),
        _ => Err(format!("unknown binary label {}", crate::schema::py_repr(value))),
    }
}

pub fn hard_index(keys: &[String], label: &str) -> std::result::Result<usize, String> {
    let label = if is_binary_keys(keys) { binary_key(label)?.to_string() } else { label.to_string() };
    keys.iter().position(|k| *k == label).ok_or_else(|| format!("hard label {label:?} is not a question option"))
}

/// Finite, nonnegative weights with positive mass, normalized.
fn normalize(weights: Vec<f64>) -> std::result::Result<Vec<f64>, String> {
    if weights.is_empty() || weights.iter().any(|w| !w.is_finite() || *w < 0.0) {
        return Err("target weights must be finite, nonnegative numbers".into());
    }
    let total: f64 = weights.iter().sum();
    if !total.is_finite() || total <= 0.0 {
        return Err("target must have finite positive mass".into());
    }
    Ok(weights.iter().map(|w| w / total).collect())
}

/// A soft target, decoded and validated as the corrected lars loader does (`lars.targets`):
/// a JSON-encoded string is decoded first; a scalar is P(yes) for a binary question; a list is
/// in option order (binary order is [yes, no]); a map is by option key (binary keys accept
/// yes/1/true and no/0/false) and may omit zero-mass options. Malformed targets are errors,
/// never a silent fall back to the hard label.
fn soft_target(keys: &[String], value: &Value) -> std::result::Result<Option<Vec<f64>>, String> {
    let decoded;
    let value = match value {
        Value::String(text) => {
            decoded = serde_json::from_str::<Value>(text).map_err(|_| "soft_label must be valid JSON".to_string())?;
            &decoded
        }
        other => other,
    };
    let binary = is_binary_keys(keys);
    let number = |v: &Value| v.as_f64().filter(|_| v.is_number()).ok_or("soft_label weights must be numbers");
    let weights = match value {
        Value::Null => return Ok(None),
        Value::Number(_) => {
            let p = number(value)?;
            if !binary || !(0.0..=1.0).contains(&p) {
                return Err("scalar soft_label must be a binary probability in [0, 1]".into());
            }
            vec![p, 1.0 - p]
        }
        Value::Object(map) => {
            let mut weights = vec![0.0; keys.len()];
            let mut seen = vec![false; keys.len()];
            for (key, weight) in map {
                let key = if binary { binary_key(key)?.to_string() } else { key.clone() };
                let i = keys.iter().position(|k| *k == key).ok_or(format!("soft_label has unknown option {key:?}"))?;
                if seen[i] {
                    return Err(format!("soft_label repeats option {key:?} through aliases"));
                }
                seen[i] = true;
                weights[i] = number(weight)?;
            }
            weights
        }
        Value::Array(items) => {
            if items.len() != keys.len() {
                return Err(format!("soft_label has {} values, expected {}", items.len(), keys.len()));
            }
            items.iter().map(number).collect::<std::result::Result<_, _>>()?
        }
        _ => return Err("soft_label must be a probability, vector, or option map".into()),
    };
    normalize(weights).map(Some)
}

/// The target distribution in `keys` order: the soft label when there is one, else one-hot.
pub fn target_vector(example: &Example, keys: &[String]) -> Vec<f64> {
    if let Some(soft) = &example.soft_label {
        return soft.clone();
    }
    let hot = hard_index(keys, &example.label).unwrap_or(usize::MAX);
    (0..keys.len()).map(|i| if i == hot { 1.0 } else { 0.0 }).collect()
}

/// [P(yes), P(no)].
fn noul_target(example: &Example) -> Vec<f64> {
    target_vector(example, &["yes".to_string(), "no".to_string()])
}

fn is_binary(kind: Kind) -> bool {
    matches!(kind, Kind::Noul | Kind::Multi)
}

/// One example's calibrated distribution in `keys` order, through the calibrator's head if it
/// has one.
fn distribution(
    cal: &Calibrator,
    example: &Example,
    (kind, keys, logits): &(Kind, Vec<String>, Vec<f64>),
    hidden: Option<&[f32]>,
) -> Result<Vec<f64>> {
    let p = if is_binary(*kind) {
        let p_yes = noul_prob(cal, logits[0] - logits[1], *kind, example.features.as_ref());
        vec![p_yes, 1.0 - p_yes]
    } else {
        softmax(logits, cal.temperature_for(*kind))
    };
    match &cal.head {
        None => Ok(p),
        Some(head) if head.keys == *keys => head.apply(&p, hidden),
        Some(head) => Err(Error::Invalid(format!(
            "the calibrator's head was fitted for options [{}], not [{}]",
            head.keys.join(", "),
            keys.join(", ")
        ))),
    }
}

/// Each example's calibrated distribution, its confidence (the top probability) and whether the
/// top option is the hard label, under `cal`, from logits already collected.
pub fn score_examples(
    cal: &Calibrator,
    examples: &[Example],
    collected: &Collected,
) -> Result<Vec<(Vec<f64>, f64, bool)>> {
    examples
        .iter()
        .zip(&collected.rows)
        .enumerate()
        .map(|(i, (example, row))| {
            let hidden = collected.hidden.as_ref().map(|h| h[i].as_slice());
            let p = distribution(cal, example, row, hidden)?;
            let hit = Some(argmax(&p)) == hard_index(&row.1, &example.label).ok();
            let confidence = max(&p);
            Ok((p, confidence, hit))
        })
        .collect()
}

/// Mean log loss of `cal` against each example's target (its soft label, else the hard label).
pub fn log_loss(cal: &Calibrator, examples: &[Example], collected: &Collected) -> Result<f64> {
    let losses = log_losses(cal, examples, collected)?;
    Ok(pairwise_sum(&losses) / losses.len().max(1) as f64)
}

/// Each example's log loss under `cal`.
pub fn log_losses(cal: &Calibrator, examples: &[Example], collected: &Collected) -> Result<Vec<f64>> {
    let scored = score_examples(cal, examples, collected)?;
    Ok(examples
        .iter()
        .zip(&collected.rows)
        .zip(&scored)
        .map(|((example, (_, keys, _)), (p, _, _))| {
            let target = target_vector(example, keys);
            -target.iter().zip(p).map(|(t, p)| t * p.clamp(1e-12, 1.0).ln()).sum::<f64>()
        })
        .collect())
}

/// The examples and collected rows at `indices`, for fitting or scoring a subset.
pub fn subset(examples: &[Example], collected: &Collected, indices: &[usize]) -> (Vec<Example>, Collected) {
    let rows = indices.iter().map(|&i| collected.rows[i].clone()).collect();
    let hidden = collected.hidden.as_ref().map(|h| indices.iter().map(|&i| h[i].clone()).collect());
    (
        indices.iter().map(|&i| examples[i].clone()).collect(),
        Collected { rows, hidden, ms_per_example: collected.ms_per_example },
    )
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
    for (i, (example, row)) in examples.iter().zip(&collected.rows).enumerate() {
        let (kind, keys, _) = row;
        let p = distribution(cal, example, row, collected.hidden.as_ref().map(|h| h[i].as_slice()))?;
        let (label, target) = if is_binary(*kind) {
            (noul_label(&example.label).to_string(), noul_target(example))
        } else {
            (example.label.clone(), target_vector(example, keys))
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

#[cfg(test)]
mod tests {
    use super::*;

    fn keys(names: &[&str]) -> Vec<String> {
        names.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn soft_targets_decode_as_the_corrected_lars_loader() {
        let binary = keys(&["yes", "no"]);
        let three = keys(&["a", "b", "c"]);
        let decode = |k: &[String], v: Value| soft_target(k, &v);
        assert_eq!(decode(&binary, json!("0.25")), Ok(Some(vec![0.25, 0.75])));
        assert_eq!(decode(&binary, json!(1)), Ok(Some(vec![1.0, 0.0])));
        assert_eq!(decode(&binary, json!({"1": 3, "0": 1})), Ok(Some(vec![0.75, 0.25])));
        assert_eq!(decode(&three, json!("[0.0, 0.0, 2.0]")), Ok(Some(vec![0.0, 0.0, 1.0])));
        assert_eq!(decode(&three, json!({"b": 1})), Ok(Some(vec![0.0, 1.0, 0.0])));
        assert_eq!(decode(&three, Value::Null), Ok(None));
        for bad in [
            json!("not json"),
            json!(1.5),
            json!([1, 2]),
            json!({"z": 1}),
            json!([0, 0, 0]),
            json!([-1, 1, 1]),
            json!({"a": "x"}),
        ] {
            assert!(decode(&three, bad.clone()).is_err() || decode(&binary, bad.clone()).is_err(), "{bad}");
        }
        assert!(decode(&three, json!(0.5)).is_err(), "a scalar is only for binary questions");
        assert!(decode(&binary, json!({"yes": 1, "true": 1})).is_err(), "aliases repeat an option");
        assert!(hard_index(&binary, "maybe").is_err());
        assert_eq!(hard_index(&binary, "True"), Ok(0));
    }
}
