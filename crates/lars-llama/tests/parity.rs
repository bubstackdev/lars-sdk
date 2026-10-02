//! Parity with the reference engine's llama.cpp backend on a real model. Needs the GGUF the
//! fixture was recorded with, named by LARS_TEST_GGUF; skips without it.
//!
//!     LARS_TEST_GGUF=~/Code/personal/lars/models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf \
//!         cargo test --release -p lars-llama -- --nocapture

use std::path::PathBuf;

use lars_core::calibration::Calibrator;
use lars_core::engine::{Engine, EngineOptions};
use lars_core::render::Template;
use lars_core::schema::Request;
use lars_llama::{FlashAttention, LlamaBackend, LlamaOptions};
use serde_json::{Value, json};

#[test]
fn logits_match_the_reference_bit_for_bit_with_flash_attention_off() {
    let Some(model) = std::env::var_os("LARS_TEST_GGUF").map(PathBuf::from) else {
        eprintln!("skipped: set LARS_TEST_GGUF to the fixture's model");
        return;
    };
    let stem = model.file_stem().unwrap().to_string_lossy().to_lowercase();
    let fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!("../../fixtures/data/llama-{stem}.json"));
    let fixture: Value = serde_json::from_str(&std::fs::read_to_string(&fixture).unwrap()).unwrap();
    // llama-cpp-python, which recorded the fixture, runs with flash attention off.
    let options = LlamaOptions { flash_attention: FlashAttention::Off, ..LlamaOptions::default() };
    let backend = LlamaBackend::load(&model, Template::ChatMl, &stem, options).unwrap();
    let mut engine = Engine::new(Box::new(backend), Calibrator::default(), EngineOptions::default()).unwrap();
    let rows = fixture["rows"].as_array().unwrap();
    for case in rows {
        let request =
            Request::from_value(&json!({"state": case["state"], "questions": {"q": case["question"]}})).unwrap();
        let (keys, logits) = engine.raw_logits(&case["state"], &request.questions["q"]).unwrap();
        let expected: Vec<f64> = case["logits"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
        assert_eq!(keys, lars_core_strings(&case["keys"]), "{}", case["id"]);
        assert_eq!(logits, expected, "{}", case["id"]);
    }
    eprintln!("{} rows bit-identical", rows.len());
}

fn lars_core_strings(value: &Value) -> Vec<String> {
    value.as_array().unwrap().iter().map(|v| v.as_str().unwrap().to_string()).collect()
}
