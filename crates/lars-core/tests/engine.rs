//! Engine parity with the reference: the same logits must give the same answers.

mod common;

use std::collections::HashMap;

use common::{compare_json, fixture, labels};
use indexmap::IndexMap;
use lars_core::backend::{Backend, MockBackend};
use lars_core::calibration::Calibrator;
use lars_core::engine::{Engine, EngineOptions};
use lars_core::error::{Error, Result};
use lars_core::py;
use lars_core::render::{Template, compose_prompt, plan_rows, render_content};
use lars_core::schema::Request;
use serde_json::Value;

/// Returns the reference's logits for each prompt it saw.
struct Replay {
    logits: HashMap<(String, String), Vec<f64>>,
    template: Template,
}

impl Backend for Replay {
    fn name(&self) -> &str {
        "replay"
    }
    fn model_name(&self) -> &str {
        "fixture"
    }
    fn template(&self) -> &Template {
        &self.template
    }
    fn is_single_token(&self, label: &str) -> bool {
        label.chars().count() <= 2
    }
    fn count_tokens(&self, text: &str) -> usize {
        py::split_whitespace(text).count().max(1)
    }
    fn label_logits(&mut self, prefix: &str, suffixes: &[String], _: &[Vec<String>]) -> Result<Vec<Vec<f64>>> {
        suffixes
            .iter()
            .map(|s| {
                self.logits
                    .get(&(prefix.to_string(), s.clone()))
                    .cloned()
                    .ok_or_else(|| Error::Backend("unseen".into()))
            })
            .collect()
    }
}

fn replay_for(request: Option<&Request>, row_logits: &Value) -> Replay {
    let mut logits = HashMap::new();
    if let (Some(request), Value::Array(per_row)) = (request, row_logits) {
        let base = render_content(&request.state);
        for (row, z) in plan_rows(request, &labels()).iter().zip(per_row) {
            let (prefix, suffix) =
                compose_prompt(&Template::ChatMl, row.state_text.as_deref().unwrap_or(&base), &row.suffix_body);
            logits.insert((prefix, suffix), common::floats(z));
        }
    }
    Replay { logits, template: Template::ChatMl }
}

fn calibrators(manifest: &Value, group: &str) -> IndexMap<String, Calibrator> {
    manifest["calibrators"][group]
        .as_object()
        .unwrap()
        .iter()
        .map(|(name, c)| (name.clone(), Calibrator::from_json(c).unwrap()))
        .collect()
}

fn options(budgets: &Value) -> EngineOptions {
    let get = |k: &str| budgets.get(k).and_then(Value::as_u64).map(|v| v as usize);
    EngineOptions {
        version: "0.0.1".into(),
        max_rows: get("max_rows").or(Some(512)),
        max_input_tokens: get("max_input_tokens").or(Some(32768)),
        ..EngineOptions::default()
    }
}

#[test]
fn reduction_matches_the_reference() {
    let manifest = fixture("manifest.json");
    let defaults = calibrators(&manifest, "default");
    let named = calibrators(&manifest, "named");
    let data = fixture("reduce.json");
    let (mut responses, mut errors) = (0, 0);
    let mut inexact = Vec::new();
    for case in data["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let parsed = Request::from_value(&case["request"]);
        if let Some(expected) = case.get("error") {
            let python_message = expected["message"].as_str().unwrap();
            let error = match parsed {
                Err(e) => e,
                Ok(request) => {
                    let backend = replay_for(Some(&request), &case["row_logits"]);
                    let mut engine =
                        Engine::new(Box::new(backend), defaults["identity"].clone(), options(&case["budgets"]))
                            .unwrap()
                            .with_named_calibrators(named.clone());
                    engine.evaluate(&request).expect_err(name)
                }
            };
            match (expected["stage"].as_str().unwrap(), &error) {
                ("schema", Error::Schema(message)) => {
                    let wanted = python_message.trim_start_matches("Value error, ");
                    assert!(message.contains(wanted), "{name}: {message:?} lacks {wanted:?}");
                }
                ("engine", e) => {
                    let kind = expected["kind"].as_str().unwrap();
                    let variant_ok = match e {
                        Error::Budget(_) => kind == "BudgetError",
                        Error::UnknownModel(_) => kind == "UnknownModelError",
                        Error::Invalid(_) => kind == "ValueError",
                        _ => false,
                    };
                    assert!(variant_ok, "{name}: {e:?} is not a {kind}");
                    assert_eq!(e.message(), python_message, "{name}");
                }
                (stage, e) => panic!("{name}: expected a {stage} error, got {e:?}"),
            }
            assert_eq!(error.error_type(), "invalid_request");
            errors += 1;
            continue;
        }
        let request = parsed.unwrap_or_else(|e| panic!("{name}: {e}"));
        let backend = replay_for(Some(&request), &case["row_logits"]);
        let default = case["default_calibrator"].as_str().unwrap();
        let mut engine = Engine::new(Box::new(backend), defaults[default].clone(), options(&Value::Null))
            .unwrap()
            .with_named_calibrators(named.clone());
        let response = engine.evaluate(&request).unwrap_or_else(|e| panic!("{name}: {e}"));
        compare_json(&response.to_json(), &case["response"], 1e-12, name, &mut inexact);
        responses += 1;
    }
    assert!(responses >= 100 && errors >= 25, "{responses} responses, {errors} errors");
    assert!(inexact.is_empty(), "inexact numbers:\n{}", inexact.join("\n"));
}

#[test]
fn mock_backend_matches_the_reference() {
    let data = fixture("mock.json");
    let mut inexact = Vec::new();
    for case in data["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let request = Request::from_value(&case["request"]).unwrap();

        let mut mock = MockBackend::default();
        let base = render_content(&request.state);
        for (i, (row, expected)) in
            plan_rows(&request, &labels()).iter().zip(case["row_logits"].as_array().unwrap()).enumerate()
        {
            let (prefix, suffix) =
                compose_prompt(&Template::Plain, row.state_text.as_deref().unwrap_or(&base), &row.suffix_body);
            let logits = mock.label_logits(&prefix, &[suffix], std::slice::from_ref(&row.labels)).unwrap().remove(0);
            for (a, b) in logits.iter().zip(common::floats(expected)) {
                assert_eq!(a.to_bits(), b.to_bits(), "{name} row {i}: {a} != {b}");
            }
        }

        let options = EngineOptions { version: "0.0.1".into(), ..EngineOptions::default() };
        let mut engine = Engine::new(Box::new(MockBackend::default()), Calibrator::default(), options).unwrap();
        assert_eq!(engine.model_id(), data["model_id"].as_str().unwrap());
        let response = engine.evaluate(&request).unwrap();
        compare_json(&response.to_json(), &case["response"], 1e-12, name, &mut inexact);
    }
    assert!(inexact.is_empty(), "inexact numbers:\n{}", inexact.join("\n"));
}
