//! The runtime on a mock pack: answers, concurrency, cancellation, calibrators, shared models,
//! idle unload. No model needed.

use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Barrier};
use std::time::Duration;

use lars_runtime::{Lars, OpenOptions};
use serde_json::{Value, json};

fn mock_pack(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("lars-runtime-test-{name}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let manifest = json!({"format": 1, "name": name, "version": "0.0.0", "backend": "mock", "layout": "v1",
                          "template": "plain"});
    std::fs::write(dir.join("pack.json"), manifest.to_string()).unwrap();
    dir
}

fn open(name: &str, options: OpenOptions) -> Lars {
    Lars::open(mock_pack(name).to_str().unwrap(), options).unwrap()
}

const REQUEST: &str = r#"{"state": "I was charged twice. Refund me!", "questions": {
    "refund": {"type": "noul", "instructions": "The customer asks for money back"},
    "team": {"type": "choice", "instructions": "Which team?", "criteria": {"billing": "Charges", "technical": "Bugs"}}}}"#;

fn molar_calibration() -> String {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/data/eval.json");
    let data: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    data["calibration_examples"].as_array().unwrap().iter().map(|l| l.as_str().unwrap()).collect::<Vec<_>>().join("\n")
}

#[test]
fn answers_match_the_core_engine() {
    let lars = open("answers", OpenOptions::default());
    let answer: Value = serde_json::from_str(&lars.evaluate(REQUEST).unwrap()).unwrap();
    let mut engine = lars_core::Engine::new(
        Box::new(lars_core::MockBackend::default()),
        lars_core::Calibrator::default(),
        lars_core::EngineOptions { version: lars_runtime::VERSION.into(), ..Default::default() },
    )
    .unwrap();
    let direct = engine.evaluate(&lars_core::Request::from_json(REQUEST).unwrap()).unwrap().to_json();
    assert_eq!(answer, direct);
    let error = lars.evaluate(r#"{"state": "x", "questions": {}}"#).unwrap_err();
    assert_eq!(error.error_type, "invalid_request");
}

#[test]
fn every_request_from_many_threads_gets_exactly_one_reply() {
    let lars = Arc::new(open("concurrency", OpenOptions::default()));
    let (threads, per_thread) = (8, 60);
    let replies = Arc::new(AtomicUsize::new(0));
    let barrier = Arc::new(Barrier::new(threads));
    let (done_tx, done_rx) = mpsc::channel();
    let handles: Vec<_> = (0..threads)
        .map(|t| {
            let (lars, replies, barrier, done_tx) = (lars.clone(), replies.clone(), barrier.clone(), done_tx.clone());
            std::thread::spawn(move || {
                barrier.wait();
                for i in 0..per_thread {
                    if (t + i) % 2 == 0 {
                        lars.evaluate(REQUEST).unwrap();
                        replies.fetch_add(1, Ordering::SeqCst);
                    } else {
                        let (replies, done_tx) = (replies.clone(), done_tx.clone());
                        lars.evaluate_async(
                            REQUEST,
                            Box::new(move |_, result| {
                                result.unwrap();
                                replies.fetch_add(1, Ordering::SeqCst);
                                done_tx.send(()).unwrap();
                            }),
                        );
                    }
                }
            })
        })
        .collect();
    for handle in handles {
        handle.join().unwrap();
    }
    for _ in 0..threads * per_thread / 2 {
        done_rx.recv_timeout(Duration::from_secs(30)).unwrap();
    }
    assert_eq!(replies.load(Ordering::SeqCst), threads * per_thread);
    let status: Value = serde_json::from_str(&lars.status()).unwrap();
    assert_eq!(status["queued"], 0);
}

#[test]
fn cancelled_and_closed_requests_still_reply_once() {
    let lars = open("cancel", OpenOptions::default());
    let (tx, rx) = mpsc::channel();
    let ids: Vec<u64> = (0..400)
        .map(|_| {
            let tx = tx.clone();
            lars.evaluate_async(REQUEST, Box::new(move |_, result| tx.send(result.map_err(|e| e.error_type)).unwrap()))
        })
        .collect();
    let cancelled_now = ids[200..].iter().filter(|id| lars.cancel(**id)).count();
    drop(lars); // closing cancels whatever is still queued
    let results: Vec<_> = (0..400).map(|_| rx.recv_timeout(Duration::from_secs(30)).unwrap()).collect();
    assert!(rx.recv_timeout(Duration::from_millis(200)).is_err(), "no request replies twice");
    let cancelled = results.iter().filter(|r| *r == &Err("cancelled")).count();
    assert!(results.iter().all(|r| r.is_ok() || r == &Err("cancelled")));
    eprintln!("{cancelled} of 400 cancelled ({cancelled_now} cancel calls found the request pending)");
}

#[test]
fn calibrators_are_fitted_bound_to_the_pack_and_kept_per_handle() {
    let lars = open("calibrate", OpenOptions::default());
    let fitted: Value =
        serde_json::from_str(&lars.calibrate("triage", &molar_calibration(), Some(0.05)).unwrap()).unwrap();
    assert_eq!(fitted["examples"], 90);
    assert_eq!(fitted["calibrator"]["pack"]["name"], "calibrate");
    assert!(fitted["certified"]["threshold"].is_null(), "90 mock answers cannot certify 5%");
    let status: Value = serde_json::from_str(&lars.status()).unwrap();
    assert_eq!(status["calibrators"], json!(["triage"]));

    let named = r#"{"state": "My tooth hurts", "questions": {"pain": {"type": "noul", "instructions": "In pain"}},
                    "lars": {"calibrator": "triage"}}"#;
    lars.evaluate(named).unwrap();
    // Another handle on the same pack shares the model but not the calibrators.
    let other = Lars::open(lars.pack().dir.to_str().unwrap(), OpenOptions::default()).unwrap();
    assert_eq!(other.evaluate(named).unwrap_err().error_type, "invalid_request");
    let saved = fitted["calibrator"].to_string();
    other.set_calibrator("triage", &saved).unwrap();
    other.evaluate(named).unwrap();
    let status: Value = serde_json::from_str(&other.status()).unwrap();
    assert_eq!(status["loads"], 1, "both handles use one loaded model");

    let mut foreign = fitted["calibrator"].clone();
    foreign["pack"]["name"] = json!("some-other-pack");
    assert_eq!(other.set_calibrator("x", &foreign.to_string()).unwrap_err().error_type, "invalid_request");
    let mut unbound = fitted["calibrator"].clone();
    unbound.as_object_mut().unwrap().remove("pack");
    assert!(other.set_calibrator("x", &unbound.to_string()).unwrap_err().message.contains("records no pack"));
}

#[test]
fn an_idle_model_unloads_and_reloads_on_the_next_request() {
    let options = OpenOptions { idle_unload_s: 0.2, ..OpenOptions::default() };
    let lars = open("idle", options);
    lars.evaluate(REQUEST).unwrap();
    std::thread::sleep(Duration::from_millis(600));
    let status: Value = serde_json::from_str(&lars.status()).unwrap();
    assert_eq!((status["loaded"].clone(), status["loads"].clone()), (json!(false), json!(1)));
    lars.evaluate(REQUEST).unwrap();
    let status: Value = serde_json::from_str(&lars.status()).unwrap();
    assert_eq!((status["loaded"].clone(), status["loads"].clone()), (json!(true), json!(2)));
}

#[test]
fn bad_packs_and_options_are_clear_errors() {
    assert_eq!(Lars::open("./no/such/pack", OpenOptions::default()).err().unwrap().error_type, "pack_error");
    assert!(lars_runtime::OpenOptions::from_json(&json!({"gpu_layer": 1})).is_err());
    let dir = mock_pack("bad");
    std::fs::write(dir.join("pack.json"), r#"{"format": 9}"#).unwrap();
    assert!(Lars::open(dir.to_str().unwrap(), OpenOptions::default()).err().unwrap().message.contains("format"));
}

#[test]
fn a_fitted_calibrator_replaces_the_packs_only_when_it_is_significantly_better() {
    let lars = open("keep", OpenOptions::default());
    let question = r#"{"type": "noul", "instructions": "The message asks for a refund"}"#;
    let state = |i: usize| format!("message {i} about an order");
    let mock_says_yes = |i: usize| -> bool {
        let request = format!(r#"{{"state": "{}", "questions": {{"q": {question}}}}}"#, state(i));
        let answer: Value = serde_json::from_str(&lars.evaluate(&request).unwrap()).unwrap();
        answer["answers"]["q"]["noul"].as_f64().unwrap() >= 0.5
    };
    let examples = |n: usize, label: &dyn Fn(usize) -> bool| -> String {
        (0..n)
            .map(|i| {
                format!(
                    r#"{{"state": "{}", "question": {question}, "label": "{}"}}"#,
                    state(i),
                    if label(i) { "yes" } else { "no" }
                )
            })
            .collect::<Vec<_>>()
            .join("\n")
    };
    // 200 labels that always contradict the mock: a fit that flips its answers is clearly better
    // (and a recalibration on top of it can be better still).
    let fitted: Value =
        serde_json::from_str(&lars.calibrate("flip", &examples(200, &|i| !mock_says_yes(i)), None).unwrap()).unwrap();
    assert_ne!(fitted["kept"], "pack", "{}", fitted["cross_validated_log_loss"]);
    let flip = r#"{"state": "message 7 about an order", "questions": {"q": {"type": "noul",
                   "instructions": "The message asks for a refund"}}, "lars": {"calibrator": "flip"}}"#;
    let flipped: Value = serde_json::from_str(&lars.evaluate(flip).unwrap()).unwrap();
    assert_eq!(flipped["answers"]["q"]["noul"].as_f64().unwrap() >= 0.5, !mock_says_yes(7));
    // 20 labels agreeing with the mock: a refit gains nothing significant, so the pack's calibrator
    // stays the base (a recalibration may still sharpen it, since the labels are its own answers).
    let kept: Value =
        serde_json::from_str(&lars.calibrate("agree", &examples(20, &mock_says_yes), None).unwrap()).unwrap();
    assert_eq!(kept["base"], "pack", "{}", kept["cross_validated_log_loss"]);
    assert_ne!(kept["kept"], "head");
}

#[test]
fn a_head_learns_what_only_the_hidden_state_shows() {
    // The mock's logits ignore the animal; its hidden state (hashed state words) does not. The
    // label is "yes" for zebras, so only a head on the hidden state can learn it.
    let lars = open("head", OpenOptions::default());
    let question = r#"{"type": "noul", "instructions": "Is this one striped?"}"#;
    let animal = |i: usize| if (i * 7 + 3) % 5 < 2 { "zebra" } else { "horse" };
    let state = |i: usize| format!("sighting {i} of a {}", animal(i));
    let examples: String = (0..150)
        .map(|i| format!(r#"{{"state": "{}", "question": {question}, "label": "{}"}}"#, state(i), animal(i) == "zebra"))
        .collect::<Vec<_>>()
        .join("\n");
    let fitted: Value = serde_json::from_str(&lars.calibrate("striped", &examples, Some(0.05)).unwrap()).unwrap();
    assert_eq!(fitted["kept"], "head", "{}", fitted["cross_validated_log_loss"]);
    assert_eq!(fitted["out_of_fold"], true);
    assert!(fitted["accuracy"].as_f64().unwrap() > 0.9, "cross-validated accuracy {}", fitted["accuracy"]);
    let head = &fitted["calibrator"]["head"];
    assert_eq!(head["dim"], lars_core::backend::MOCK_HIDDEN);
    assert_eq!(head["keys"], json!(["yes", "no"]));

    // New sightings, through the head; and through a saved copy on another handle.
    let other = Lars::open(lars.pack().dir.to_str().unwrap(), OpenOptions::default()).unwrap();
    other.set_calibrator("striped", &fitted["calibrator"].to_string()).unwrap();
    let mut hits = 0;
    for i in 1000..1040 {
        let request = format!(
            r#"{{"state": "{}", "questions": {{"q": {question}}}, "lars": {{"calibrator": "striped"}}}}"#,
            state(i)
        );
        let answer: Value = serde_json::from_str(&lars.evaluate(&request).unwrap()).unwrap();
        let p = answer["answers"]["q"]["noul"].as_f64().unwrap();
        hits += usize::from((p >= 0.5) == (animal(i) == "zebra"));
        assert_eq!(other.evaluate(&request).unwrap(), lars.evaluate(&request).unwrap());
    }
    assert!(hits >= 36, "{hits}/40");

    // A head answers only the options it was fitted for.
    let choice = r#"{"state": "a zebra", "questions": {"q": {"type": "choice", "instructions": "Which?",
                     "criteria": {"a": "A", "b": "B"}}}, "lars": {"calibrator": "striped"}}"#;
    assert!(lars.evaluate(choice).unwrap_err().message.contains("fitted for [yes, no]"));

    // Under MIN_HEAD_LABELS, no head is tried.
    let few: String = examples.lines().take(30).collect::<Vec<_>>().join("\n");
    let small: Value = serde_json::from_str(&lars.calibrate("few", &few, None).unwrap()).unwrap();
    assert_ne!(small["kept"], "head");
    assert!(small["cross_validated_log_loss"].get("head:3").is_none());
}
