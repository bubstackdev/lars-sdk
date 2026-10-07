//! moedex-lars on a mock pack (no model: deterministic, keyword-driven answers that test the
//! plumbing, not judgment), and on a real pack when LARS_TEST_PACK names one.

use std::future::Future;
use std::path::PathBuf;
use std::pin::pin;
use std::sync::Arc;
use std::task::{Context, Poll, Wake, Waker};
use std::thread::Thread;

use moedex_lars::serde_json::{Value, json};
use moedex_lars::{Constraint, Kept, Lars, OpenOptions, Options};

fn mock_pack(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("lars-rust-{name}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let manifest = json!({"format": 1, "name": name, "version": "0.0.0", "backend": "mock", "template": "plain"});
    std::fs::write(dir.join("pack.json"), manifest.to_string()).unwrap();
    dir
}

fn open(name: &str) -> Lars {
    Lars::open_with(mock_pack(name), OpenOptions { threads: 2, ..Default::default() }).wait().unwrap()
}

/// A minimal executor, so the tests show `.await` needs no particular async runtime.
fn block_on<F: Future>(future: F) -> F::Output {
    struct Unpark(Thread);
    impl Wake for Unpark {
        fn wake(self: Arc<Self>) {
            self.0.unpark();
        }
    }
    let waker = Waker::from(Arc::new(Unpark(std::thread::current())));
    let mut cx = Context::from_waker(&waker);
    let mut future = pin!(future);
    loop {
        if let Poll::Ready(output) = future.as_mut().poll(&mut cx) {
            return output;
        }
        std::thread::park();
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Options)]
enum Team {
    /// Payments, charges,
    /// refunds
    Billing,
    /// Bugs, outages, errors
    Technical,
    #[option("Anything else", key = "other")]
    FrontDesk,
}

#[derive(Clone, Copy, Debug, PartialEq, Options)]
enum Tag {
    /// Wants money back
    Refund,
    /// Reports a defect
    Bug,
}

#[test]
fn derive_reads_doc_comments_and_keys() {
    assert_eq!(
        Team::options(),
        vec![
            ("billing", Some("Payments, charges, refunds")),
            ("technical", Some("Bugs, outages, errors")),
            ("other", Some("Anything else")),
        ]
    );
    assert_eq!(<Team as moedex_lars::Key>::from_key("other"), Some(Team::FrontDesk));
    assert_eq!(moedex_lars::Key::key(&Team::Billing), "billing");
}

#[test]
fn opens_a_pack_and_reports_it() {
    let lars = open("open");
    assert_eq!(lars.pack()["name"], "open");
    assert!(Lars::version().starts_with("0."));
    let status = lars.status();
    assert_eq!(status["loaded"], true);
    assert_eq!(status["queued"], 0);
}

#[test]
fn nouls_resolve_to_booleans() {
    let lars = open("noul");
    let is_refund = lars.noul("isRefund", "The customer asks for a refund");
    let value: bool = is_refund.ask("Please refund my order").wait().unwrap();
    let d = is_refund.decide("Please refund my order").wait().unwrap();
    assert_eq!(value, d.value);
    assert_eq!(d.value, d.p >= 0.5);
    assert!((d.confidence - d.p.max(1.0 - d.p)).abs() < 1e-12);
    assert!(d.accepted);
    assert_eq!(is_refund.name(), "isRefund");

    let strict = lars.noul("strictRefund", "The customer asks for a refund").threshold(0.999);
    let d = strict.decide("Please refund my order").wait().unwrap();
    assert_eq!(d.value, d.p >= 0.999);

    let explained =
        lars.noul("yesNo", "The customer asks for a refund").yes("They want money back").no("Anything else");
    assert_eq!(explained.question()["criteria"], json!({"true": "They want money back", "false": "Anything else"}));
    explained.ask("refund").wait().unwrap();
}

#[test]
fn choices_resolve_to_enum_variants_or_keys() {
    let lars = open("choice");
    let team = lars.choice::<Team>("team", "Which team should handle this?");
    let value: Team = team.ask("I was charged twice for my payment").wait().unwrap();
    let d = team.decide("I was charged twice for my payment").wait().unwrap();
    assert_eq!(value, d.value);
    let options: Vec<Team> = d.probabilities.iter().map(|(t, _)| *t).collect();
    assert_eq!(options, [Team::Billing, Team::Technical, Team::FrontDesk]);
    assert_eq!(d.confidence, d.probabilities.iter().map(|(_, p)| *p).fold(0.0, f64::max));

    let keyed = lars.choice_keys("keyedTeam", "Which team?", [("billing", "Charges"), ("technical", "Bugs")]);
    let value: String = keyed.ask("I was charged twice").wait().unwrap();
    assert!(value == "billing" || value == "technical");
}

#[test]
fn scores_and_multis() {
    let lars = open("score");
    let levels = ["routine", "soon", "today", "emergency"];
    let urgency = lars.score("urgency", "How urgent?", levels);
    let level = urgency.ask("The server is down").wait().unwrap();
    assert!((0.0..=3.0).contains(&level));
    let d = urgency.decide("The server is down").wait().unwrap();
    assert_eq!(d.probabilities.len(), 4);
    assert_eq!(d.label, levels[d.level]);

    let tags = lars.multi::<Tag>("tags", "Which apply?");
    let selected: Vec<Tag> = tags.ask("refund my money, the app has a bug").wait().unwrap();
    assert!(selected.len() <= 2);
    let keyed = lars.multi_keys("keyedTags", "Which apply?", [("refund", "Wants money back"), ("bug", "A defect")]);
    keyed.ask("refund please").wait().unwrap();
}

#[test]
fn a_state_selector_picks_what_the_model_reads() {
    struct Ticket {
        subject: &'static str,
        body: &'static str,
        #[allow(dead_code)]
        email: &'static str,
    }
    let lars = open("state");
    let seen = Arc::new(std::sync::Mutex::new(Value::Null));
    let record = seen.clone();
    let urgent = lars.noul("ticketUrgent", "The ticket is urgent").state(move |t: &Ticket| {
        let state = json!({"subject": t.subject, "body": t.body});
        *record.lock().unwrap() = state.clone();
        state
    });
    let ticket = Ticket { subject: "Down", body: "Everything is down", email: "someone@example.com" };
    urgent.ask(&ticket).wait().unwrap();
    assert_eq!(*seen.lock().unwrap(), json!({"subject": "Down", "body": "Everything is down"}));
}

#[test]
fn ask_answers_several_predicates_in_one_request() {
    let lars = open("ask");
    let angry = lars.noul("angry", "The customer is angry");
    let team = lars.choice::<Team>("askTeam", "Which team?");
    let urgency = lars.score("askUrgency", "How urgent?", ["later", "now"]);
    let (a, t, u) = lars.ask("I was charged twice and I am furious", (&angry, &team, &urgency)).wait().unwrap();
    assert_eq!(a.value, a.p >= 0.5);
    assert!(t.probabilities.len() == 3);
    assert!(u.probabilities.len() == 2);

    let refund = lars.noul("refund", "The customer asks for money back");
    let nouls = lars.ask("refund me", [&angry, &refund]).wait().unwrap();
    assert_eq!(nouls.len(), 2);
    let many: Vec<_> = (0..20).map(|i| lars.noul(&format!("q{i}"), format!("Claim {i}"))).collect();
    assert_eq!(lars.ask("one message", &many).wait().unwrap().len(), 20);

    let constrained =
        lars.ask_with("refund me", (&angry, &refund), &[Constraint::exclusive([&angry, &refund])]).wait().unwrap();
    assert!(constrained.0.p + constrained.1.p <= 1.0 + 1e-9);

    let error = lars.ask("x", &Vec::<moedex_lars::Noul>::new()).wait().unwrap_err();
    assert_eq!(error.error_type, "invalid_request");
}

#[test]
fn await_works_on_any_executor_and_many_calls_resolve() {
    let lars = open("async");
    let is_refund = lars.noul("isRefund", "The customer asks for a refund");
    let value = block_on(is_refund.ask("refund please")).unwrap();
    assert_eq!(value, is_refund.decide("refund please").wait().unwrap().value);

    let pending: Vec<_> = (0..300).map(|i| is_refund.ask(&format!("message {i}: refund please"))).collect();
    let values: Vec<bool> = pending.into_iter().map(|p| block_on(p).unwrap()).collect();
    assert_eq!(values.len(), 300);

    let threads: Vec<_> = (0..8)
        .map(|t| {
            let p = is_refund.clone();
            std::thread::spawn(move || (0..25).map(|i| p.ask(&format!("{t}/{i} refund")).wait().unwrap()).count())
        })
        .collect();
    assert_eq!(threads.into_iter().map(|t| t.join().unwrap()).sum::<usize>(), 200);
}

#[test]
fn dropping_a_pending_cancels_it_and_nothing_hangs() {
    let lars = open("cancel");
    let p = lars.noul("p", "Anything");
    for i in 0..200 {
        drop(p.ask(&format!("dropped {i}")));
    }
    assert!(p.ask("still answers").wait().is_ok());
}

#[test]
fn calibration_fits_a_calibrator_and_refuses_to_certify_on_too_few_labels() {
    let lars = open("calibrate");
    let is_refund = lars.noul("calibrated", "The customer asks for a refund");
    let examples: Vec<(String, bool)> = (0..40)
        .map(|i| (if i % 2 == 1 { format!("refund order {i}") } else { format!("hello {i}") }, i % 2 == 1))
        .collect();
    let fit = is_refund.calibrate(examples, 0.05).wait().unwrap();
    assert_eq!(fit.examples, 40);
    let certified = fit.certified.unwrap();
    assert_eq!(certified.threshold, None);
    assert!(certified.reason.unwrap().contains("cannot certify"));
    assert!(lars.status()["calibrators"].as_array().unwrap().contains(&json!("calibrated")));
    // Certification was asked for and failed: the gate stays shut.
    let d = is_refund.decide("Please refund my order").wait().unwrap();
    assert!(!d.accepted);
    assert!(!is_refund.ask("Please refund my order").wait().unwrap());

    let saved = lars.calibrations();
    assert!(saved["calibrators"].get("calibrated").is_some());
    assert_eq!(saved["certified"]["calibrated"]["maxError"], 0.05);
    let path = std::env::temp_dir().join(format!("lars-rust-calibrations-{}.json", std::process::id()));
    lars.save_calibrations(&path).unwrap();
    let other = Lars::open(lars.status()["pack"]["dir"].as_str().unwrap()).wait().unwrap();
    other.load_calibrations_from(&path).unwrap();
    assert!(other.status()["calibrators"].as_array().unwrap().contains(&json!("calibrated")));
    assert_eq!(other.calibrations()["certified"], saved["certified"]);
}

#[test]
fn a_certified_threshold_gates_the_value_and_an_unaccepted_choice_falls_back() {
    let lars = open("certify");
    let is_refund = lars.noul("certified", "The customer asks for a refund");
    let inputs: Vec<String> = (0..200).map(|i| format!("refund the order number {i}")).collect();
    let labels: Vec<bool> = inputs.iter().map(|x| is_refund.decide(x).wait().unwrap().value).collect();
    let fit = block_on(is_refund.calibrate(inputs.iter().map(String::as_str).zip(labels), 0.05)).unwrap();
    assert!(fit.accuracy > 0.9);
    assert!(matches!(fit.kept, Kept::Pack | Kept::Fitted | Kept::Recal | Kept::Head));
    if let Some(threshold) = fit.certified.and_then(|c| c.threshold) {
        let d = is_refund.decide(&inputs[0]).wait().unwrap();
        assert_eq!(d.accepted, d.confidence >= threshold);
        assert_eq!(is_refund.ask(&inputs[0]).wait().unwrap(), d.value && d.accepted);
    }

    let team = lars.choice::<Team>("fallbackTeam", "Which team?").fallback(Team::FrontDesk);
    let inputs: Vec<String> = (0..100).map(|i| format!("message {i} about something")).collect();
    let labels: Vec<Team> = inputs.iter().map(|x| team.decide(x).wait().unwrap().value).collect();
    let fit = team.calibrate(inputs.iter().map(String::as_str).zip(labels), 0.0001).wait().unwrap();
    assert_eq!(fit.certified.unwrap().threshold, None);
    assert!(!team.decide("anything").wait().unwrap().accepted);
    assert_eq!(team.ask("anything").wait().unwrap(), Team::FrontDesk);
}

#[test]
fn saved_calibrations_carry_their_question_and_certificate() {
    let lars = open("saved");
    let refund = lars.noul("refund", "The customer asks for a refund");
    let examples: Vec<(String, bool)> = (0..40).map(|i| (format!("refund order {i}"), i % 3 != 0)).collect();
    refund.calibrate(examples.clone(), 0.05).wait().unwrap();
    let saved = lars.calibrations();
    assert_eq!(saved["format"], 2);
    assert_eq!(saved["questions"]["refund"], *refund.question());

    // Loaded elsewhere, the calibration is bound to its question: the same name asking the
    // opposite is refused, and the failed certificate still shuts the gate.
    let dir = lars.status()["pack"]["dir"].as_str().unwrap().to_string();
    let other = Lars::open(&dir).wait().unwrap();
    other.load_calibrations(&saved).unwrap();
    let flipped = other.noul("refund", "The customer is NOT asking for a refund");
    assert_eq!(flipped.ask("refund me").wait().unwrap_err().error_type, "invalid_request");
    let same = other.noul("refund", "The customer asks for a refund");
    assert!(!same.decide("refund me").wait().unwrap().accepted);

    // Defined first, the other question makes the load fail, and nothing changes.
    let third = Lars::open(&dir).wait().unwrap();
    third.noul("refund", "The customer is NOT asking for a refund").ask("x").wait().unwrap();
    assert_eq!(third.load_calibrations(&saved).unwrap_err().error_type, "invalid_request");
    assert!(third.status()["calibrators"].as_array().unwrap().is_empty());

    // Options count as the question: same keys, different descriptions, refused.
    let team = lars.choice_keys("desk", "Which desk?", [("a", "Charges"), ("b", "Bugs")]);
    let labels: Vec<(String, String)> = (0..10).map(|i| (format!("m{i}"), "a".to_string())).collect();
    team.calibrate(labels, None).wait().unwrap();
    let fourth = Lars::open(&dir).wait().unwrap();
    fourth.load_calibrations(&lars.calibrations()).unwrap();
    let renamed = fourth.choice_keys("desk", "Which desk?", [("a", "Refunds"), ("b", "Bugs")]);
    assert_eq!(renamed.ask("x").wait().unwrap_err().error_type, "invalid_request");

    // A replacement without a certificate clears the old one.
    let mut certified = saved.clone();
    certified["certified"]["refund"] = json!({"maxError": 0.05, "threshold": 0.5});
    other.load_calibrations(&certified).unwrap();
    assert_eq!(other.calibrations()["certified"]["refund"]["threshold"], 0.5);
    let mut uncertified = saved.clone();
    uncertified["certified"] = json!({});
    other.load_calibrations(&uncertified).unwrap();
    assert!(other.calibrations()["certified"].get("refund").is_none());
    assert!(same.decide("refund me").wait().unwrap().accepted);

    // A bad entry anywhere in the file changes nothing.
    let mut bad = certified.clone();
    bad["calibrators"]["broken"] = json!({"kind": "nonsense"});
    bad["questions"]["broken"] = json!({"type": "noul", "instructions": "x"});
    assert!(other.load_calibrations(&bad).is_err());
    assert!(other.calibrations()["certified"].get("refund").is_none());
    assert!(other.calibrations()["calibrators"].get("broken").is_none());
    let mut bad_certificate = certified.clone();
    bad_certificate["certified"]["refund"]["threshold"] = json!("high");
    assert!(other.load_calibrations(&bad_certificate).is_err());
    assert!(other.calibrations()["certified"].get("refund").is_none());

    // Older files and other packs are refused.
    let mut old = saved.clone();
    old["format"] = json!(1);
    assert!(other.load_calibrations(&old).is_err());
    let mut elsewhere = saved;
    elsewhere["pack"]["name"] = json!("another");
    assert!(other.load_calibrations(&elsewhere).is_err());
}

#[test]
fn calibrations_are_bound_to_the_state_version() {
    struct Ticket {
        subject: &'static str,
        body: &'static str,
    }
    let lars = open("versions");
    let urgent = lars
        .noul("urgent", "The ticket needs a reply today")
        .state(|t: &Ticket| json!({"subject": t.subject}))
        .state_version("subject-only");
    let tickets: Vec<Ticket> = (0..20).map(|_| Ticket { subject: "Down", body: "Everything is down" }).collect();
    urgent.calibrate(tickets.iter().map(|t| (t, true)), None).wait().unwrap();
    let saved = lars.calibrations();
    assert_eq!(saved["stateVersions"]["urgent"], "subject-only");

    // The same version loads; another version, or none, is refused.
    let dir = lars.status()["pack"]["dir"].as_str().unwrap().to_string();
    let same = Lars::open(&dir).wait().unwrap();
    same.load_calibrations(&saved).unwrap();
    let reread = same
        .noul("urgent", "The ticket needs a reply today")
        .state(|t: &Ticket| json!({"subject": t.subject, "body": t.body}))
        .state_version("subject-and-body");
    let ticket = Ticket { subject: "Down", body: "Everything is down" };
    assert_eq!(reread.ask(&ticket).wait().unwrap_err().error_type, "invalid_request");
    let unversioned = same.noul("urgent", "The ticket needs a reply today");
    assert_eq!(unversioned.ask("x").wait().unwrap_err().error_type, "invalid_request");
    let matching = same
        .noul("urgent", "The ticket needs a reply today")
        .state(|t: &Ticket| json!({"subject": t.subject}))
        .state_version("subject-only");
    matching.ask(&ticket).wait().unwrap();

    let other = Lars::open(&dir).wait().unwrap();
    other
        .noul("urgent", "The ticket needs a reply today")
        .state(|t: &Ticket| json!({"body": t.body}))
        .state_version("body-only")
        .ask(&ticket)
        .wait()
        .unwrap();
    assert_eq!(other.load_calibrations(&saved).unwrap_err().error_type, "invalid_request");
}

#[test]
fn pull_returns_a_pack_directory() {
    let dir = mock_pack("pull");
    assert_eq!(Lars::pull(dir.to_str().unwrap()).wait().unwrap(), dir);
    assert_eq!(Lars::pull("./no/such/pack").wait().unwrap_err().error_type, "pack_error");
}

#[test]
fn names_are_unique_per_question_and_errors_have_a_type() {
    let lars = open("errors");
    lars.noul("unique", "One question").ask("x").wait().unwrap();
    let error = lars.noul("unique", "Another question").ask("x").wait().unwrap_err();
    assert_eq!(error.error_type, "invalid_request");

    let error = lars.evaluate(&json!({"state": "x", "questions": {}})).wait().unwrap_err();
    assert_eq!(error.error_type, "invalid_request");
    let error = Lars::open("./no/such/pack").wait().unwrap_err();
    assert_eq!(error.error_type, "pack_error");
}

#[test]
fn a_real_pack_judges_plain_cases() {
    let Ok(pack) = std::env::var("LARS_TEST_PACK") else { return };
    let lars = Lars::open(pack).wait().unwrap();
    let is_pissed = lars.noul("isPissed", "The customer is angry");
    assert!(is_pissed.ask("This is the third time your app ate my order. I am done with you people.").wait().unwrap());
    assert!(!is_pissed.ask("Thanks so much, the new update works great!").wait().unwrap());

    #[derive(Clone, Copy, Debug, PartialEq, Options)]
    enum Desk {
        /// Payments, charges, invoices, refunds
        Billing,
        /// Bugs, crashes, outages, error messages
        Technical,
        /// Pricing questions, upgrades, new contracts
        Sales,
    }
    let desk = lars.choice::<Desk>("desk", "Which team should handle this message?");
    assert_eq!(desk.ask("I was charged twice on my credit card this month").wait().unwrap(), Desk::Billing);
    assert_eq!(desk.ask("The app crashes every time I open settings").wait().unwrap(), Desk::Technical);
    assert_eq!(
        desk.ask("What would it cost to move our 40 seats to the enterprise plan?").wait().unwrap(),
        Desk::Sales
    );

    let angry = lars.noul("angry", "The customer is angry");
    let refund = lars.noul("refund", "The customer asks for money back");
    let urgency =
        lars.score("urgency", "How urgently does this need a reply?", ["whenever", "this week", "today", "right now"]);
    let message = "You charged me twice and I want my money back NOW or I'm calling my bank.";
    let (a, r, u) = block_on(lars.ask(message, (&angry, &refund, &urgency))).unwrap();
    assert!(a.value && r.value && u.value > 1.5);
}
