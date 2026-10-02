//! Calibration and primitive parity with the reference.

mod common;

use common::{assert_close, fixture, floats};
use indexmap::IndexMap;
use lars_core::calibration::{self, Fusion};
use lars_core::primitives;
use serde_json::Value;

fn rows(value: &Value) -> Vec<Vec<f64>> {
    value.as_array().unwrap().iter().map(floats).collect()
}

fn features(value: &Value) -> IndexMap<String, f64> {
    value.as_object().unwrap().iter().map(|(k, v)| (k.clone(), v.as_f64().unwrap())).collect()
}

#[test]
fn primitives_match_numpy() {
    let data = fixture("primitives.json");
    let tol = 1e-15;
    for case in data["softmax"].as_array().unwrap() {
        let out = primitives::softmax(&floats(&case["logits"]), case["temperature"].as_f64().unwrap());
        for (a, b) in out.iter().zip(floats(&case["out"])) {
            assert_close(*a, b, tol, "softmax");
        }
    }
    for case in data["sigmoid"].as_array().unwrap() {
        assert_close(primitives::sigmoid(case["x"].as_f64().unwrap()), case["out"].as_f64().unwrap(), tol, "sigmoid");
    }
    type Unary = fn(&[f64]) -> f64;
    let unary: [(&str, Unary); 4] = [
        ("choice_confidence", primitives::choice_confidence),
        ("expected_score", primitives::expected_score),
        ("score_confidence", primitives::score_confidence),
        ("top_margin", primitives::top_margin),
    ];
    for (name, f) in unary {
        for case in data[name].as_array().unwrap() {
            assert_close(f(&floats(&case["p"])), case["out"].as_f64().unwrap(), tol, name);
        }
    }
    for case in data["total_variation"].as_array().unwrap() {
        let out = primitives::total_variation(&floats(&case["p"]), &floats(&case["q"]));
        assert_close(out, case["out"].as_f64().unwrap(), tol, "total_variation");
    }
    for case in data["order_sensitivity"].as_array().unwrap() {
        assert_close(primitives::order_sensitivity(&rows(&case["ds"])), case["out"].as_f64().unwrap(), tol, "order");
    }
}

#[test]
fn grid_matches_numpy_geomspace() {
    let data = fixture("calibration.json");
    let expected = floats(&data["geomspace_0.05_20_400"]);
    let ours = calibration::geomspace(0.05, 20.0, 400);
    let exact = ours.iter().zip(&expected).filter(|(a, b)| a.to_bits() == b.to_bits()).count();
    for (a, b) in ours.iter().zip(&expected) {
        assert_close(*a, *b, 1e-15, "geomspace");
    }
    // Bit-identical where the platform libm matches numpy's (it does on macOS); close everywhere.
    eprintln!("{exact} of 400 grid points bit-identical");
}

#[test]
fn fits_match_the_reference() {
    let data = fixture("calibration.json");
    for case in data["fit_temperature"].as_array().unwrap() {
        let t = calibration::fit_temperature(&rows(&case["logits"]), &rows(&case["targets"]));
        assert_close(t, case["out"].as_f64().unwrap(), 1e-14, "fit_temperature");
    }
    for (i, case) in data["fit_platt"].as_array().unwrap().iter().enumerate() {
        let (a, b) = calibration::fit_platt(&floats(&case["scores"]), &floats(&case["labels"]), 1e-2, 100);
        let expected = floats(&case["out"]);
        assert_close(a, expected[0], 1e-9, &format!("platt {i} a"));
        assert_close(b, expected[1], 1e-9, &format!("platt {i} b"));
    }
    for (i, case) in data["fit_fusion"].as_array().unwrap().iter().enumerate() {
        let feats: Vec<IndexMap<String, f64>> = case["features"].as_array().unwrap().iter().map(features).collect();
        let fusion = calibration::fit_fusion(&floats(&case["logits"]), &feats, &floats(&case["labels"]), 1e-2);
        let expected = &case["out"];
        assert_eq!(fusion.features, common::strings(&expected["features"]), "fusion {i} names");
        for (a, b) in fusion.weights.iter().zip(floats(&expected["weights"])) {
            assert_close(*a, b, 1e-8, &format!("fusion {i} weight"));
        }
        assert_close(fusion.bias, expected["bias"].as_f64().unwrap(), 1e-8, &format!("fusion {i} bias"));
        let reference = Fusion {
            features: common::strings(&expected["features"]),
            weights: floats(&expected["weights"]),
            bias: expected["bias"].as_f64().unwrap(),
        };
        for probe in case["probes"].as_array().unwrap() {
            let p = calibration::fused_probability(
                &reference,
                probe["logit"].as_f64().unwrap(),
                &features(&probe["features"]),
            );
            assert_close(p, probe["out"].as_f64().unwrap(), 1e-15, "fused_probability");
        }
    }
}

#[test]
fn metrics_match_the_reference() {
    let data = fixture("calibration.json");
    for case in data["metrics"].as_array().unwrap() {
        let (conf, hit) = (floats(&case["confidences"]), floats(&case["correct"]));
        if let Some(expected) = case.get("ece") {
            assert_close(calibration::ece(&conf, &hit, 15), expected.as_f64().unwrap(), 1e-15, "ece");
        }
        for (key, budget) in [("coverage_at_error", 0.05), ("coverage_at_error_10", 0.10)] {
            if let Some(expected) = case.get(key) {
                let (coverage, threshold) = calibration::coverage_at_error(&conf, &hit, budget);
                assert_eq!(vec![coverage, threshold], floats(expected), "{key}");
            }
        }
    }
    for case in data["brier"].as_array().unwrap() {
        let out = calibration::brier(&rows(&case["probs"]), &rows(&case["targets"]));
        assert_close(out, case["out"].as_f64().unwrap(), 1e-15, "brier");
    }
}
