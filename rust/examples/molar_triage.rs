//! Molar Triage in Rust: sort a dental practice's inbox with three predicates, then calibrate
//! them on 30 labeled messages and measure on 30 others. The messages are fictional
//! (examples/molar_triage).
//!
//!     cargo run --release -p moedex-lars --example molar_triage -- <pack>

use moedex_lars::serde_json::{self, Value};
use moedex_lars::{Key, Lars, Options, Result};

#[derive(Clone, Copy, Debug, PartialEq, Options)]
enum Department {
    /// Cleanings, checkups, whitening, sensitivity advice
    Hygiene,
    /// Fillings, crowns, chipped or broken teeth, lost fillings
    Restorative,
    /// Braces, aligners, retainers, wires
    Orthodontics,
    /// Extractions, wisdom teeth, implants, post-operative care
    Surgery,
    /// Insurance, invoices, payment plans, estimates
    Billing,
    /// Scheduling, forms, hours, directions, records requests
    FrontDesk,
}

const PAIN: &str = "The sender says they are in pain right now.";
const DEPARTMENT: &str = "Which department should take this message?";
const URGENCY: &str = "How urgently does this message need attention?";

/// One labeled message for one question.
struct Row {
    state: String,
    instructions: String,
    label: String,
}

fn rows(file: &str) -> Vec<Row> {
    let path = format!("{}/../examples/molar_triage/{file}", env!("CARGO_MANIFEST_DIR"));
    std::fs::read_to_string(path)
        .expect("the Molar Triage data")
        .lines()
        .map(|line| {
            let row: Value = serde_json::from_str(line).unwrap();
            Row {
                state: row["state"].as_str().unwrap().into(),
                instructions: row["question"]["instructions"].as_str().unwrap().into(),
                label: row["label"].as_str().unwrap().into(),
            }
        })
        .collect()
}

fn main() -> Result<()> {
    let pack = std::env::args().nth(1).unwrap_or_else(|| "lars-2b".into());
    let lars = Lars::open(pack).wait()?;

    let in_pain = lars.noul("inPain", PAIN);
    let department = lars.choice::<Department>("department", DEPARTMENT);
    let urgency = lars.score(
        "urgency",
        URGENCY,
        [
            "Routine: can wait for the next regular visit",
            "Soon: book within a few weeks",
            "This week: schedule in the next few days",
            "Today: needs same-day attention",
            "Emergency: go to the practice or an emergency room now",
        ],
    );

    // The predicates read like the policy they implement.
    let triage = |message: &str| -> Result<String> {
        let (pain, dept, urgent) = lars.ask(message, (&in_pain, &department, &urgency)).wait()?;
        Ok(if urgent.level >= 4 {
            "call the patient now".into()
        } else if pain.value && urgent.level >= 3 {
            format!("same-day slot with {:?}", dept.value)
        } else {
            format!("queue for {:?}", dept.value)
        })
    };
    println!("{}", triage("I knocked out my front tooth playing hockey twenty minutes ago and it's bleeding a lot.")?);
    println!("{}", triage("Can I get a copy of my last invoice for my insurance?")?);

    let (calibration, test) = (rows("molar_triage.calibration.jsonl"), rows("molar_triage.test.jsonl"));
    let of = |all: &[Row], instructions: &str| -> Vec<(String, String)> {
        all.iter().filter(|r| r.instructions == instructions).map(|r| (r.state.clone(), r.label.clone())).collect()
    };
    let accuracy = || -> Result<String> {
        let share = |hits: Vec<bool>| format!("{}/{}", hits.iter().filter(|h| **h).count(), hits.len());
        let mut pain = vec![];
        for (state, label) in of(&test, PAIN) {
            pain.push(in_pain.decide(&state).wait()?.value == (label == "yes"));
        }
        let mut dept = vec![];
        for (state, label) in of(&test, DEPARTMENT) {
            dept.push(Some(department.ask(&state).wait()?) == Department::from_key(&label));
        }
        let mut level = vec![];
        for (state, label) in of(&test, URGENCY) {
            level.push(urgency.decide(&state).wait()?.level.to_string() == label);
        }
        Ok(format!("in pain {}, department {}, urgency {}", share(pain), share(dept), share(level)))
    };

    println!("\ntest accuracy, pack calibrator:       {}", accuracy()?);
    in_pain.calibrate(of(&calibration, PAIN).into_iter().map(|(s, l)| (s, l == "yes")), None).wait()?;
    department
        .calibrate(of(&calibration, DEPARTMENT).into_iter().map(|(s, l)| (s, Department::from_key(&l).unwrap())), None)
        .wait()?;
    urgency.calibrate(of(&calibration, URGENCY).into_iter().map(|(s, l)| (s, l.parse().unwrap())), None).wait()?;
    println!("test accuracy, calibrated on 30 rows: {}", accuracy()?);
    Ok(())
}
