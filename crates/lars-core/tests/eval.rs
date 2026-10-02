//! Evaluation and calibration parity on Molar Triage with the mock backend.

mod common;

use common::{compare_json, fixture};
use lars_core::MockBackend;
use lars_core::calibration::Calibrator;
use lars_core::engine::{Engine, EngineOptions};
use lars_core::eval::{self, maybe_json, read_examples};
use serde_json::Value;

fn engine(calibrator: Calibrator) -> Engine {
    let options = EngineOptions { version: "0.0.1".into(), ..EngineOptions::default() };
    Engine::new(Box::new(MockBackend::default()), calibrator, options).unwrap()
}

fn jsonl(value: &Value) -> String {
    value.as_array().unwrap().iter().map(|l| l.as_str().unwrap()).collect::<Vec<_>>().join("\n")
}

fn without_timing(mut value: Value) -> Value {
    value["ms_per_example"] = Value::from(0.0);
    value
}

#[test]
fn eval_and_calibrate_match_the_reference() {
    let data = fixture("eval.json");
    let calibration_set = read_examples(&jsonl(&data["calibration_examples"])).unwrap();
    let test_set = read_examples(&jsonl(&data["test_examples"])).unwrap();
    let ids: Vec<String> = test_set.iter().map(|e| e.id()).collect();
    assert_eq!(ids, common::strings(&data["example_ids"]));

    let mut inexact = Vec::new();
    let fitted =
        eval::calibrate(&mut engine(Calibrator::default()), &calibration_set, Some("molar_triage.calibration.jsonl"))
            .unwrap();
    compare_json(&fitted.to_json(), &data["calibrated"], 1e-9, "calibrated", &mut inexact);

    let mut rows = Vec::new();
    let raw = eval::evaluate(&mut engine(Calibrator::default()), &test_set, Some(&mut rows)).unwrap();
    compare_json(&without_timing(raw.to_json()), &data["eval_raw"], 1e-12, "eval_raw", &mut inexact);
    compare_json(&Value::from(rows), &data["eval_raw_rows"], 1e-12, "eval_raw_rows", &mut inexact);

    let reference = Calibrator::from_json(&data["calibrated"]).unwrap();
    let calibrated = eval::evaluate(&mut engine(reference), &test_set, None).unwrap();
    compare_json(
        &without_timing(calibrated.to_json()),
        &data["eval_calibrated"],
        1e-12,
        "eval_calibrated",
        &mut inexact,
    );
    eprintln!("{} numbers differ in the last digits:\n{}", inexact.len(), inexact.join("\n"));
}

#[test]
fn json_encoded_fields_decode_as_the_reference() {
    let data = fixture("eval.json");
    for case in data["maybe_json"].as_array().unwrap() {
        assert_eq!(maybe_json(&case["value"]), case["out"], "{}", case["value"]);
    }
}
