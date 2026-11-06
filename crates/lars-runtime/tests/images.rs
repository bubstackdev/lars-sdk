//! Real image inference runs when LARS_TEST_VISION_PACK names a compatible format 2 pack.
use lars_runtime::{Lars, OpenOptions, Pack};
use serde_json::{Value, json};

fn manifest() -> Value {
    json!({"format":2, "name":"vision-test", "version":"0", "backend":"llamacpp", "layout":"v1",
        "template":"chatml", "model":{"file":"model.gguf", "bytes":5, "sha256":"00".repeat(32)},
        "projector":{"file":"projector.gguf", "bytes":5, "sha256":"11".repeat(32)},
        "vision":{"preprocessing":"stb-rgb-white-v1", "max_tokens":256},
        "runtime":{"llama_cpp":lars_llama::LLAMA_CPP_VERSION, "flash_attention":"off"}})
}

#[test]
fn projector_preprocessing_and_runtime_are_bound_to_vision_calibration() {
    let dir = std::env::temp_dir().join(format!("lars-vision-pack-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("model.gguf"), b"model").unwrap();
    std::fs::write(dir.join("projector.gguf"), b"model").unwrap();
    let load = |manifest: &Value| {
        std::fs::write(dir.join("pack.json"), manifest.to_string()).unwrap();
        Pack::load(&dir, false)
    };
    let mut m = manifest();
    let a = load(&m).unwrap();
    let calibration = a.bind(&lars_core::Calibrator::default());
    m["projector"]["sha256"] = json!("22".repeat(32));
    assert!(load(&m).unwrap().bound_calibrator(&calibration).is_err());
    m["vision"]["max_tokens"] = json!(512);
    assert_ne!(a.identity(), load(&m).unwrap().identity());
    m["vision"]["preprocessing"] = json!("unknown");
    assert!(load(&m).is_err());
    m = manifest();
    m["runtime"]["llama_cpp"] = json!("wrong");
    assert!(load(&m).is_err());
    m = manifest();
    m["format"] = json!(1);
    assert!(load(&m).is_err());
    m = manifest();
    std::fs::remove_file(dir.join("projector.gguf")).unwrap();
    assert!(load(&m).is_err());
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn real_images_change_answers_and_do_not_leak_between_requests() {
    let Ok(pack) = std::env::var("LARS_TEST_VISION_PACK") else { return };
    let cpu = std::env::var_os("LARS_TEST_VISION_CPU").is_some();
    let lars =
        Lars::open(&pack, OpenOptions { gpu_layers: if cpu { 0 } else { -1 }, threads: 4, ..Default::default() })
            .unwrap();
    let request = json!({"state":null,"questions":{
        "red":{"type":"noul","instructions":"The image is predominantly red"},
        "color":{"type":"choice","instructions":"What is the predominant color in the image?",
            "criteria":{"red":"Red","blue":"Blue","other":"Another color"}}}})
    .to_string();
    let red = include_bytes!("../../../fixtures/images/red.png");
    let blue = include_bytes!("../../../fixtures/images/blue.png");
    let ask = |bytes: &[u8]| -> Value { serde_json::from_str(&lars.evaluate_image(&request, bytes).unwrap()).unwrap() };
    let a = ask(red);
    let b = ask(blue);
    let again = ask(red);
    assert_eq!(a["answers"], again["answers"], "image A/B/A must be deterministic");
    assert_ne!(a["answers"]["red"]["noul"], b["answers"]["red"]["noul"], "model must receive the pixels");
    assert!(a["answers"]["red"]["noul"].as_f64().unwrap() > 0.5);
    assert!(b["answers"]["red"]["noul"].as_f64().unwrap() < 0.5);
    assert_eq!(a["answers"]["color"]["choice"], "red");
    assert_eq!(b["answers"]["color"]["choice"], "blue");
    assert_eq!(a["answers"]["red"]["calibrated"], false);
    assert!(a["usage"]["input_tokens"].as_u64().unwrap() > 64);
    assert_eq!(lars.evaluate_image(&request, b"corrupt image").unwrap_err().error_type, "invalid_request");
    let mut oversized = red.to_vec();
    oversized[16..20].copy_from_slice(&100_000_u32.to_be_bytes());
    oversized[20..24].copy_from_slice(&100_000_u32.to_be_bytes());
    assert!(lars.evaluate_image(&request, &oversized).is_err());
    let mut animated = red.to_vec();
    let mut chunk = vec![0, 0, 0, 8, b'a', b'c', b'T', b'L'];
    chunk.extend([0; 12]);
    animated.splice(33..33, chunk);
    assert!(lars.evaluate_image(&request, &animated).unwrap_err().message.contains("animated PNG"));
    let mut unsupported: Value = serde_json::from_str(&request).unwrap();
    unsupported["lars"] = json!({"explain":true});
    assert!(lars.evaluate_image(&unsupported.to_string(), red).is_err());
    assert_eq!(ask(red)["answers"], a["answers"], "errors must release request image state");
}
