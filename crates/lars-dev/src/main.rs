//! Developer CLI for lars-core and the llama.cpp backend: answer requests, evaluate and
//! calibrate on labeled JSONL, capture raw logits, time requests, and check parity with the
//! reference engine's llama.cpp logits.
//!
//!     lars-dev ask       --model M.gguf request.json
//!     lars-dev eval      --model M.gguf --data set.jsonl [--calibration c.json] [--dump rows.jsonl]
//!     lars-dev calibrate --model M.gguf --data set.jsonl --out c.json
//!     lars-dev capture   --model M.gguf --data set.jsonl --out logits.jsonl
//!     lars-dev bench     --model M.gguf request.json [--repeat 20] [--vary-state]
//!     lars-dev parity    --model M.gguf --fixture fixtures/data/llama-<model>.json
//!     lars-dev compare   a.jsonl b.jsonl          (two captures of the same examples)
//!     lars-dev capture-requests --model M.gguf --requests requests.json --out rows.jsonl
//!
//! Backend flags: --template chatml|plain|gemma|llama3|<captured.json>, --cpu, --threads N,
//! --layout v1|v2, --no-batch-suffixes, --no-reuse-prefix, --no-repack, --flash off|on|auto (default off), --n-ctx N. `--model mock` uses the
//! mock backend.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use lars_core::backend::{Backend, MockBackend};
use lars_core::calibration::Calibrator;
use lars_core::engine::{Engine, EngineOptions};
use lars_core::eval::{self, read_examples};
use lars_core::primitives::argmax;
use lars_core::render::{Layout, Template};
use lars_core::schema::Request;
use lars_llama::{FlashAttention, LlamaBackend, LlamaOptions};
use serde_json::{Value, json};

type Failure = Box<dyn std::error::Error>;

struct Args {
    positional: Vec<String>,
    flags: HashMap<String, String>,
}

impl Args {
    fn parse(raw: impl Iterator<Item = String>) -> Args {
        let switches = ["cpu", "no-batch-suffixes", "no-reuse-prefix", "no-repack", "vary-state"];
        let (mut positional, mut flags) = (Vec::new(), HashMap::new());
        let mut raw = raw.peekable();
        while let Some(arg) = raw.next() {
            if let Some(name) = arg.strip_prefix("--") {
                let value = if switches.contains(&name) { "1".to_string() } else { raw.next().unwrap_or_default() };
                flags.insert(name.to_string(), value);
            } else {
                positional.push(arg);
            }
        }
        Args { positional, flags }
    }

    fn get(&self, name: &str) -> Option<&str> {
        self.flags.get(name).map(String::as_str)
    }

    fn need(&self, name: &str) -> Result<&str, Failure> {
        self.get(name).ok_or_else(|| format!("--{name} is required").into())
    }

    fn number<T: std::str::FromStr>(&self, name: &str, default: T) -> Result<T, Failure> {
        match self.get(name) {
            None => Ok(default),
            Some(v) => v.parse().map_err(|_| format!("--{name} {v:?} is not a number").into()),
        }
    }
}

fn template(spec: Option<&str>) -> Result<Template, Failure> {
    let spec = spec.unwrap_or("chatml");
    if let Some(t) = Template::named(spec) {
        return Ok(t);
    }
    let captured: Value = serde_json::from_str(&std::fs::read_to_string(spec)?)?;
    let text = |k: &str| captured[k].as_str().map(String::from).ok_or(format!("{spec}: no {k}"));
    Ok(Template::Captured {
        before_system: text("before_system")?,
        between: text("between")?,
        after_user: text("after_user")?,
        trims_content: captured["trims_content"].as_bool().unwrap_or(false),
    })
}

fn llama_options(args: &Args) -> Result<LlamaOptions, Failure> {
    let defaults = LlamaOptions::default();
    Ok(LlamaOptions {
        n_ctx: args.number("n-ctx", defaults.n_ctx)?,
        n_threads: args.number("threads", 0)?,
        gpu_layers: if args.get("cpu").is_some() { 0 } else { -1 },
        flash_attention: match args.get("flash").unwrap_or("off") {
            "on" => FlashAttention::On,
            "auto" => FlashAttention::Auto,
            _ => FlashAttention::Off,
        },
        batch_suffixes: args.get("no-batch-suffixes").is_none(),
        reuse_prefix: args.get("no-reuse-prefix").is_none(),
        repack_cpu_weights: args.get("no-repack").is_none(),
        ..defaults
    })
}

fn backend(args: &Args) -> Result<Box<dyn Backend>, Failure> {
    let model = args.need("model")?;
    if model == "mock" {
        return Ok(Box::new(MockBackend::default()));
    }
    let path = PathBuf::from(model);
    let name = path.file_stem().map_or("model".into(), |s| s.to_string_lossy().into_owned());
    Ok(Box::new(LlamaBackend::load(&path, template(args.get("template"))?, &name, llama_options(args)?)?))
}

fn engine(args: &Args) -> Result<Engine, Failure> {
    let calibrator = match args.get("calibration") {
        Some(path) => Calibrator::from_json(&serde_json::from_str(&std::fs::read_to_string(path)?)?)?,
        None => Calibrator::default(),
    };
    let layout = match args.get("layout") {
        None => Layout::V1,
        Some(name) => Layout::parse(name).ok_or(format!("--layout {name:?}: use v1 or v2"))?,
    };
    let options = EngineOptions { version: "0.0.1".into(), layout, ..EngineOptions::default() };
    Ok(Engine::new(backend(args)?, calibrator, options)?)
}

fn examples(args: &Args) -> Result<Vec<eval::Example>, Failure> {
    let mut examples = read_examples(&std::fs::read_to_string(args.need("data")?)?)?;
    if let Some(limit) = args.get("limit") {
        examples.truncate(limit.parse()?);
    }
    Ok(examples)
}

fn read_request(path: &str) -> Result<Request, Failure> {
    Ok(Request::from_json(&std::fs::read_to_string(path)?)?)
}

fn print(value: &Value) {
    println!("{}", serde_json::to_string_pretty(value).unwrap());
}

fn percentile(sorted: &[f64], q: f64) -> f64 {
    sorted[((sorted.len() - 1) as f64 * q).round() as usize]
}

fn main() -> Result<(), Failure> {
    let mut raw = std::env::args().skip(1);
    let command = raw.next().unwrap_or_default();
    let args = Args::parse(raw);
    match command.as_str() {
        "ask" => {
            let request = read_request(args.positional.first().ok_or("give a request file")?)?;
            print(&engine(&args)?.evaluate(&request)?.to_json());
        }
        "eval" => {
            let examples = examples(&args)?;
            let mut engine = engine(&args)?;
            let mut rows = Vec::new();
            let result = eval::evaluate(&mut engine, &examples, Some(&mut rows))?;
            if let Some(dump) = args.get("dump") {
                let lines: Vec<String> = rows.iter().map(|r| r.to_string()).collect();
                std::fs::write(dump, lines.join("\n") + "\n")?;
            }
            print(&result.to_json());
        }
        "calibrate" => {
            let examples = examples(&args)?;
            let data = args.need("data")?;
            let fitted = eval::calibrate(&mut engine(&args)?, &examples, Some(data))?;
            std::fs::write(args.need("out")?, serde_json::to_string_pretty(&fitted.to_json())? + "\n")?;
            print(&fitted.to_json());
        }
        "capture" => {
            // Raw logits per example, so calibrators and metrics can be recomputed on CPU later.
            let examples = examples(&args)?;
            let mut engine = engine(&args)?;
            let collected = eval::collect(&mut engine, &examples)?;
            let mut lines = Vec::with_capacity(examples.len());
            for (example, (kind, keys, logits)) in examples.iter().zip(&collected.rows) {
                lines.push(
                    json!({"id": example.id(), "kind": kind.as_str(), "keys": keys, "logits": logits,
                           "label": example.label, "soft_label": example.soft_label, "features": example.features})
                    .to_string(),
                );
            }
            std::fs::write(args.need("out")?, lines.join("\n") + "\n")?;
            eprintln!("{} rows, {:.1} ms per example", lines.len(), collected.ms_per_example);
        }
        "bench" => {
            let request = read_request(args.positional.first().ok_or("give a request file")?)?;
            let repeat: usize = args.number("repeat", 20)?;
            let started = Instant::now();
            let mut engine = engine(&args)?;
            let load_ms = started.elapsed().as_secs_f64() * 1000.0;
            // --vary-state: a different state on every call (a new user each time), so only the
            // constant head can be reused; without it the same request repeats.
            let vary = args.get("vary-state").is_some();
            let mut times = Vec::with_capacity(repeat + 1);
            for i in 0..=repeat {
                let mut call = request.clone();
                if let (true, Value::String(state)) = (vary, &request.state) {
                    call.state = Value::String(format!("Ticket {}. {state}", 1000 + i));
                }
                let t = Instant::now();
                engine.evaluate(&call)?;
                times.push(t.elapsed().as_secs_f64() * 1000.0);
            }
            let first = times.remove(0);
            times.sort_by(f64::total_cmp);
            print(&json!({
                "load_ms": load_ms, "first_ms": first, "p50_ms": percentile(&times, 0.5),
                "p90_ms": percentile(&times, 0.9), "min_ms": times[0], "repeat": repeat, "vary_state": vary,
                "llama_cpp": lars_llama::LLAMA_CPP_VERSION,
            }));
        }
        "parity" => parity(&args)?,
        "compare" => compare(&args)?,
        "capture-requests" => {
            // Every row of every request in a JSON array, so multi-row batching can be compared.
            let requests: Vec<Value> = serde_json::from_str(&std::fs::read_to_string(args.need("requests")?)?)?;
            let mut engine = engine(&args)?;
            let started = Instant::now();
            let mut lines = Vec::new();
            for (r, body) in requests.iter().enumerate() {
                for (i, (row, logits)) in engine.raw_rows(&Request::from_value(body)?)?.into_iter().enumerate() {
                    lines.push(
                        json!({"id": format!("{r}:{i}:{}:{}", row.question_id, row.variant), "logits": logits})
                            .to_string(),
                    );
                }
            }
            std::fs::write(args.need("out")?, lines.join("\n") + "\n")?;
            eprintln!("{} rows in {:.1} s", lines.len(), started.elapsed().as_secs_f64());
        }
        _ => {
            eprintln!(
                "usage: lars-dev ask|eval|calibrate|capture|bench|parity --model M.gguf ... (see the source header)"
            );
            std::process::exit(2);
        }
    }
    Ok(())
}

/// Compare raw logits with the reference engine's llama.cpp backend on the same rows.
fn parity(args: &Args) -> Result<(), Failure> {
    let fixture: Value = serde_json::from_str(&std::fs::read_to_string(Path::new(args.need("fixture")?))?)?;
    let mut engine = engine(args)?;
    let (mut rows, mut same_top, mut max_diff, mut sum_diff, mut max_p_diff) = (0, 0, 0f64, 0f64, 0f64);
    let mut disagreements = Vec::new();
    for case in fixture["rows"].as_array().ok_or("fixture has no rows")? {
        let request = Request::from_value(&json!({"state": case["state"], "questions": {"q": case["question"]}}))?;
        let (_, logits) = engine.raw_logits(&case["state"], &request.questions["q"])?;
        let reference: Vec<f64> = case["logits"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
        let diff = logits.iter().zip(&reference).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
        let (p, q) = (lars_core::primitives::softmax(&logits, 1.0), lars_core::primitives::softmax(&reference, 1.0));
        max_p_diff = max_p_diff.max(p.iter().zip(&q).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max));
        max_diff = max_diff.max(diff);
        sum_diff += diff;
        rows += 1;
        if argmax(&logits) == argmax(&reference) {
            same_top += 1;
        } else {
            disagreements.push(json!({"id": case["id"], "ours": logits, "reference": reference}));
        }
    }
    print(&json!({
        "rows": rows, "same_top_label": same_top, "max_abs_logit_diff": max_diff,
        "mean_max_abs_logit_diff": sum_diff / rows.max(1) as f64, "max_abs_prob_diff": max_p_diff,
        "reference": fixture["meta"], "ours": {"llama_cpp": lars_llama::LLAMA_CPP_VERSION},
        "disagreements": disagreements,
    }));
    Ok(())
}

fn read_capture(path: &str) -> Result<Vec<Value>, Failure> {
    std::fs::read_to_string(path)?
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l).map_err(Failure::from))
        .collect()
}

/// How far two captures of the same examples differ: identical rows, top-label agreement,
/// and the largest logit and probability differences.
fn compare(args: &Args) -> Result<(), Failure> {
    let [a, b] = args.positional.as_slice() else { return Err("give two capture files".into()) };
    let (a, b) = (read_capture(a)?, read_capture(b)?);
    if a.len() != b.len() {
        return Err(format!("{} rows against {}", a.len(), b.len()).into());
    }
    let (mut identical, mut same_top, mut max_logit, mut max_prob) = (0, 0, 0f64, 0f64);
    for (x, y) in a.iter().zip(&b) {
        if x["id"] != y["id"] {
            return Err(format!("row {} is not row {}", x["id"], y["id"]).into());
        }
        let floats =
            |v: &Value| -> Vec<f64> { v["logits"].as_array().unwrap().iter().map(|f| f.as_f64().unwrap()).collect() };
        let (p, q) = (floats(x), floats(y));
        identical += (p == q) as usize;
        same_top += (argmax(&p) == argmax(&q)) as usize;
        max_logit = max_logit.max(p.iter().zip(&q).map(|(s, t)| (s - t).abs()).fold(0.0, f64::max));
        let (sp, sq) = (lars_core::primitives::softmax(&p, 1.0), lars_core::primitives::softmax(&q, 1.0));
        max_prob = max_prob.max(sp.iter().zip(&sq).map(|(s, t)| (s - t).abs()).fold(0.0, f64::max));
    }
    print(&json!({"rows": a.len(), "identical": identical, "same_top_label": same_top,
                  "max_abs_logit_diff": max_logit, "max_abs_prob_diff": max_prob}));
    Ok(())
}
