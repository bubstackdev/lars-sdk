//! Shared helpers for the reference-parity tests: fixture loading and tolerant JSON comparison.

#![allow(dead_code)]

use std::path::PathBuf;

use serde_json::Value;

pub fn fixture(name: &str) -> Value {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/data").join(name);
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    serde_json::from_str(&text).unwrap()
}

pub fn labels() -> Vec<String> {
    strings(&fixture("manifest.json")["labels"])
}

pub fn strings(value: &Value) -> Vec<String> {
    value.as_array().unwrap().iter().map(|v| v.as_str().unwrap().to_string()).collect()
}

pub fn floats(value: &Value) -> Vec<f64> {
    value.as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect()
}

pub fn close(a: f64, b: f64, tolerance: f64) -> bool {
    (a - b).abs() <= tolerance * (1.0 + b.abs())
}

pub fn assert_close(actual: f64, expected: f64, tolerance: f64, context: &str) {
    assert!(close(actual, expected, tolerance), "{context}: {actual} != {expected} (tolerance {tolerance})");
}

/// Walk two JSON values; structure, strings and booleans must be equal, numbers within
/// `tolerance`. Returns the paths whose numbers differ at all, for reporting.
pub fn compare_json(actual: &Value, expected: &Value, tolerance: f64, path: &str, inexact: &mut Vec<String>) {
    match (actual, expected) {
        (Value::Number(a), Value::Number(b)) => {
            let (a, b) = (a.as_f64().unwrap(), b.as_f64().unwrap());
            assert!(close(a, b, tolerance), "{path}: {a} != {b}");
            if a != b {
                inexact.push(format!("{path}: {a} vs {b}"));
            }
        }
        (Value::Object(a), Value::Object(b)) => {
            let mut a_keys: Vec<_> = a.keys().collect();
            let mut b_keys: Vec<_> = b.keys().collect();
            a_keys.sort();
            b_keys.sort();
            assert_eq!(a_keys, b_keys, "{path}: keys differ");
            for (key, value) in b {
                compare_json(&a[key], value, tolerance, &format!("{path}.{key}"), inexact);
            }
        }
        (Value::Array(a), Value::Array(b)) => {
            assert_eq!(a.len(), b.len(), "{path}: lengths differ");
            for (i, (x, y)) in a.iter().zip(b).enumerate() {
                compare_json(x, y, tolerance, &format!("{path}[{i}]"), inexact);
            }
        }
        _ => assert_eq!(actual, expected, "{path}"),
    }
}
