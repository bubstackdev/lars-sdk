//! The System One request, validated as the reference's pydantic models validate it.
//!
//! Requests are parsed from JSON values by hand rather than derived, so the rules are exact:
//! the same limits, unknown fields refused inside questions and `lars` options and ignored at
//! the top level, and content fields limited to string, object, array or null. The reference's
//! lax coercions (numeric strings for integers, and the like) are not reproduced; integers may
//! be written as integral floats.

use indexmap::IndexMap;
use serde_json::{Map, Value};

use crate::error::{Error, Result};

pub const MAX_CHOICE_OPTIONS: usize = 255;
pub const MIN_CHOICE_OPTIONS: usize = 2;
pub const MIN_SCORE_LEVELS: usize = 2;
pub const MAX_SCORE_LEVELS: usize = 10;
pub const MAX_MULTI_OPTIONS: usize = 255;
pub const MAX_PERMUTATIONS: usize = 16;
pub const DEFAULT_MODEL: &str = "lars-latest";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Kind {
    Noul,
    Choice,
    Score,
    Multi,
}

impl Kind {
    pub const ALL: [Kind; 4] = [Kind::Noul, Kind::Choice, Kind::Score, Kind::Multi];

    pub fn as_str(self) -> &'static str {
        match self {
            Kind::Noul => "noul",
            Kind::Choice => "choice",
            Kind::Score => "score",
            Kind::Multi => "multi",
        }
    }

    pub fn parse(name: &str) -> Option<Kind> {
        Kind::ALL.into_iter().find(|k| k.as_str() == name)
    }
}

/// Text or structured content: a string, object, array or null (`JSONContent`).
pub type Content = Value;

#[derive(Debug, Clone, PartialEq)]
pub enum Question {
    Noul { instructions: Content, criteria: Option<NoulCriteria> },
    Choice { instructions: Content, criteria: IndexMap<String, Content> },
    Score { instructions: Content, criteria: Vec<Content> },
    Multi { instructions: Content, criteria: IndexMap<String, Content> },
}

#[derive(Debug, Clone, PartialEq)]
pub struct NoulCriteria {
    pub yes: Content,
    pub no: Content,
}

impl Question {
    pub fn kind(&self) -> Kind {
        match self {
            Question::Noul { .. } => Kind::Noul,
            Question::Choice { .. } => Kind::Choice,
            Question::Score { .. } => Kind::Score,
            Question::Multi { .. } => Kind::Multi,
        }
    }

    pub fn instructions(&self) -> &Content {
        match self {
            Question::Noul { instructions, .. }
            | Question::Choice { instructions, .. }
            | Question::Score { instructions, .. }
            | Question::Multi { instructions, .. } => instructions,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConstraintKind {
    Complement,
    Exclusive,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Constraint {
    pub kind: ConstraintKind,
    pub questions: Vec<String>,
}

/// The `lars` request extensions. Everything defaults off.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Options {
    pub permutations: usize,
    pub explain: bool,
    pub abstain_margin: Option<f64>,
    pub constraints: Vec<Constraint>,
    pub features: IndexMap<String, IndexMap<String, f64>>,
    pub calibrator: Option<String>,
    pub calibrators: IndexMap<String, String>,
    /// Accepted for wire compatibility; the SDK never escalates.
    pub escalate: Option<bool>,
    pub escalate_below: Option<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Request {
    pub state: Content,
    pub model: String,
    pub questions: IndexMap<String, Question>,
    pub lars: Options,
}

impl Request {
    /// Parse and validate a request from JSON text.
    pub fn from_json(text: &str) -> Result<Request> {
        let value: Value = serde_json::from_str(text).map_err(|e| Error::Schema(format!("invalid JSON: {e}")))?;
        Request::from_value(&value)
    }

    /// Validate a decoded JSON request.
    pub fn from_value(value: &Value) -> Result<Request> {
        let body = object(value, "request")?;
        let state = content(body.get("state").ok_or_else(|| missing("state"))?, "state")?;
        let model = match body.get("model") {
            None => DEFAULT_MODEL.to_string(),
            Some(Value::String(s)) => s.clone(),
            Some(_) => return Err(schema("model: Input should be a valid string")),
        };
        let questions_value = object(body.get("questions").ok_or_else(|| missing("questions"))?, "questions")?;
        if questions_value.is_empty() {
            return Err(schema("questions: Dictionary should have at least 1 item after validation, not 0"));
        }
        let mut questions = IndexMap::new();
        for (qid, q) in questions_value {
            questions.insert(qid.clone(), question(q, qid)?);
        }
        let lars = match body.get("lars") {
            None => Options::default(),
            Some(v) => options(v)?,
        };
        let request = Request { state, model, questions, lars };
        request.check_references()?;
        Ok(request)
    }

    fn check_references(&self) -> Result<()> {
        for qid in self.lars.features.keys() {
            match self.questions.get(qid) {
                None => return Err(schema(format!("features reference unknown question {}", py_repr(qid)))),
                Some(q) if q.kind() != Kind::Noul => {
                    return Err(schema(format!("features question {} must be a noul", py_repr(qid))));
                }
                _ => {}
            }
        }
        for constraint in &self.lars.constraints {
            for qid in &constraint.questions {
                match self.questions.get(qid) {
                    None => return Err(schema(format!("constraint references unknown question {}", py_repr(qid)))),
                    Some(q) if q.kind() != Kind::Noul => {
                        return Err(schema(format!("constraint question {} must be a noul", py_repr(qid))));
                    }
                    _ => {}
                }
            }
            if constraint.kind == ConstraintKind::Complement && constraint.questions.len() != 2 {
                return Err(schema("complement constraints take exactly two questions"));
            }
        }
        Ok(())
    }
}

/// Python's `repr()` of a string, as the reference quotes names in messages.
pub(crate) fn py_repr(s: &str) -> String {
    if s.contains('\'') && !s.contains('"') { format!("\"{s}\"") } else { format!("'{}'", s.replace('\'', "\\'")) }
}

fn schema(message: impl Into<String>) -> Error {
    Error::Schema(message.into())
}

fn missing(field: &str) -> Error {
    schema(format!("{field}: Field required"))
}

fn object<'a>(value: &'a Value, location: &str) -> Result<&'a Map<String, Value>> {
    value.as_object().ok_or_else(|| schema(format!("{location}: Input should be a valid dictionary")))
}

fn content(value: &Value, location: &str) -> Result<Content> {
    match value {
        Value::String(_) | Value::Object(_) | Value::Array(_) | Value::Null => Ok(value.clone()),
        _ => Err(schema(format!(
            "{location}: Input should be a valid string; Input should be a valid dictionary; \
             Input should be a valid list"
        ))),
    }
}

fn forbid_extra(map: &Map<String, Value>, allowed: &[&str], location: &str) -> Result<()> {
    match map.keys().find(|k| !allowed.contains(&k.as_str())) {
        Some(extra) => Err(schema(format!("{location}.{extra}: Extra inputs are not permitted"))),
        None => Ok(()),
    }
}

fn question(value: &Value, qid: &str) -> Result<Question> {
    let location = format!("questions.{qid}");
    let map = object(value, &location)?;
    let tag = match map.get("type") {
        Some(Value::String(t)) => t.as_str(),
        Some(_) | None => {
            return Err(schema(format!("{location}: Unable to extract tag using discriminator 'type'")));
        }
    };
    let instructions = match map.get("instructions") {
        Some(v) => content(v, &format!("{location}.instructions"))?,
        None => Value::Null,
    };
    let criteria = map.get("criteria");
    match tag {
        "noul" => {
            forbid_extra(map, &["type", "instructions", "criteria"], &location)?;
            let criteria = match criteria {
                None | Some(Value::Null) => None,
                Some(v) => {
                    let c = object(v, &format!("{location}.criteria"))?;
                    forbid_extra(c, &["true", "false"], &format!("{location}.criteria"))?;
                    let get = |key: &str| -> Result<Content> {
                        c.get(key).map_or(Ok(Value::Null), |v| content(v, &format!("{location}.criteria.{key}")))
                    };
                    Some(NoulCriteria { yes: get("true")?, no: get("false")? })
                }
            };
            Ok(Question::Noul { instructions, criteria })
        }
        "choice" | "multi" => {
            forbid_extra(map, &["type", "instructions", "criteria"], &location)?;
            let c = object(criteria.ok_or_else(|| missing(&format!("{location}.criteria")))?, &location)?;
            let mut options = IndexMap::new();
            for (key, v) in c {
                options.insert(key.clone(), content(v, &format!("{location}.criteria.{key}"))?);
            }
            if tag == "choice" {
                if !(MIN_CHOICE_OPTIONS..=MAX_CHOICE_OPTIONS).contains(&options.len()) {
                    return Err(schema(format!(
                        "{location}.criteria: Value error, choice criteria must have \
                         {MIN_CHOICE_OPTIONS}-{MAX_CHOICE_OPTIONS} options"
                    )));
                }
                Ok(Question::Choice { instructions, criteria: options })
            } else {
                if !(1..=MAX_MULTI_OPTIONS).contains(&options.len()) {
                    return Err(schema(format!(
                        "{location}.criteria: Value error, multi criteria must have 1-{MAX_MULTI_OPTIONS} options"
                    )));
                }
                Ok(Question::Multi { instructions, criteria: options })
            }
        }
        "score" => {
            forbid_extra(map, &["type", "instructions", "criteria"], &location)?;
            let levels = match criteria {
                Some(Value::Array(items)) => items,
                Some(_) => return Err(schema(format!("{location}.criteria: Input should be a valid list"))),
                None => return Err(missing(&format!("{location}.criteria"))),
            };
            let levels = levels
                .iter()
                .enumerate()
                .map(|(i, v)| content(v, &format!("{location}.criteria.{i}")))
                .collect::<Result<Vec<_>>>()?;
            if !(MIN_SCORE_LEVELS..=MAX_SCORE_LEVELS).contains(&levels.len()) {
                return Err(schema(format!(
                    "{location}.criteria: Value error, score criteria must have \
                     {MIN_SCORE_LEVELS}-{MAX_SCORE_LEVELS} levels"
                )));
            }
            Ok(Question::Score { instructions, criteria: levels })
        }
        other => Err(schema(format!(
            "{location}: Input tag {} found using 'type' does not match any of the expected tags: \
             'noul', 'choice', 'score', 'multi'",
            py_repr(other)
        ))),
    }
}

fn integer(value: &Value, location: &str) -> Result<i64> {
    if let Some(i) = value.as_i64() {
        return Ok(i);
    }
    match value.as_f64() {
        Some(f) if value.is_number() && f.fract() == 0.0 && f.abs() < 9.0e15 => Ok(f as i64),
        _ => Err(schema(format!("{location}: Input should be a valid integer"))),
    }
}

fn number(value: &Value, location: &str) -> Result<f64> {
    match value {
        Value::Number(n) => n.as_f64().ok_or_else(|| schema(format!("{location}: Input should be a valid number"))),
        _ => Err(schema(format!("{location}: Input should be a valid number"))),
    }
}

fn unit_interval(value: &Value, location: &str) -> Result<Option<f64>> {
    if value.is_null() {
        return Ok(None);
    }
    let x = number(value, location)?;
    if x < 0.0 {
        return Err(schema(format!("{location}: Input should be greater than or equal to 0")));
    }
    if x > 1.0 {
        return Err(schema(format!("{location}: Input should be less than or equal to 1")));
    }
    Ok(Some(x))
}

fn optional_string(value: &Value, location: &str) -> Result<Option<String>> {
    match value {
        Value::Null => Ok(None),
        Value::String(s) => Ok(Some(s.clone())),
        _ => Err(schema(format!("{location}: Input should be a valid string"))),
    }
}

fn boolean(value: &Value, location: &str) -> Result<bool> {
    value.as_bool().ok_or_else(|| schema(format!("{location}: Input should be a valid boolean")))
}

fn options(value: &Value) -> Result<Options> {
    let map = object(value, "lars")?;
    forbid_extra(
        map,
        &[
            "permutations",
            "explain",
            "abstain_margin",
            "constraints",
            "features",
            "calibrator",
            "calibrators",
            "escalate",
            "escalate_below",
        ],
        "lars",
    )?;
    let mut options = Options::default();
    if let Some(v) = map.get("permutations") {
        let p = integer(v, "lars.permutations")?;
        if p < 0 {
            return Err(schema("lars.permutations: Input should be greater than or equal to 0"));
        }
        if p > MAX_PERMUTATIONS as i64 {
            return Err(schema(format!("lars.permutations: Input should be less than or equal to {MAX_PERMUTATIONS}")));
        }
        options.permutations = p as usize;
    }
    if let Some(v) = map.get("explain") {
        options.explain = boolean(v, "lars.explain")?;
    }
    if let Some(v) = map.get("abstain_margin") {
        options.abstain_margin = unit_interval(v, "lars.abstain_margin")?;
    }
    if let Some(v) = map.get("constraints") {
        let items = v.as_array().ok_or_else(|| schema("lars.constraints: Input should be a valid list"))?;
        for (i, item) in items.iter().enumerate() {
            options.constraints.push(constraint(item, &format!("lars.constraints.{i}"))?);
        }
    }
    if let Some(v) = map.get("features") {
        for (qid, named) in object(v, "lars.features")? {
            let mut values = IndexMap::new();
            for (name, x) in object(named, &format!("lars.features.{qid}"))? {
                values.insert(name.clone(), number(x, &format!("lars.features.{qid}.{name}"))?);
            }
            options.features.insert(qid.clone(), values);
        }
    }
    if let Some(v) = map.get("calibrator") {
        options.calibrator = optional_string(v, "lars.calibrator")?;
    }
    if let Some(v) = map.get("calibrators") {
        for (qid, name) in object(v, "lars.calibrators")? {
            let name = name
                .as_str()
                .ok_or_else(|| schema(format!("lars.calibrators.{qid}: Input should be a valid string")))?;
            options.calibrators.insert(qid.clone(), name.to_string());
        }
    }
    if let Some(v) = map.get("escalate") {
        options.escalate = if v.is_null() { None } else { Some(boolean(v, "lars.escalate")?) };
    }
    if let Some(v) = map.get("escalate_below") {
        options.escalate_below = unit_interval(v, "lars.escalate_below")?;
    }
    Ok(options)
}

fn constraint(value: &Value, location: &str) -> Result<Constraint> {
    let map = object(value, location)?;
    forbid_extra(map, &["kind", "questions"], location)?;
    let kind = match map.get("kind") {
        Some(Value::String(k)) if k == "complement" => ConstraintKind::Complement,
        Some(Value::String(k)) if k == "exclusive" => ConstraintKind::Exclusive,
        Some(_) => return Err(schema(format!("{location}.kind: Input should be 'complement' or 'exclusive'"))),
        None => return Err(missing(&format!("{location}.kind"))),
    };
    let questions = match map.get("questions") {
        Some(Value::Array(items)) => items
            .iter()
            .map(|q| {
                q.as_str()
                    .map(str::to_string)
                    .ok_or_else(|| schema(format!("{location}.questions: Input should be a valid string")))
            })
            .collect::<Result<Vec<_>>>()?,
        Some(_) => return Err(schema(format!("{location}.questions: Input should be a valid list"))),
        None => return Err(missing(&format!("{location}.questions"))),
    };
    if questions.len() < 2 {
        return Err(schema(format!(
            "{location}.questions: List should have at least 2 items after validation, not {}",
            questions.len()
        )));
    }
    Ok(Constraint { kind, questions })
}
