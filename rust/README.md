# moedex-lars

Typed decisions as functions in your code, answered by a small language model running in your
Rust process. No server, no Ollama, no network after the model pack is downloaded.

```rust
use moedex_lars::{Lars, Options};

#[derive(Clone, Copy, Debug, PartialEq, Options)]
enum Team {
    /// Payments, charges, refunds
    Billing,
    /// Bugs, crashes, outages
    Technical,
}

let lars = Lars::open("lars-2b").wait()?;
let is_pissed = lars.noul("isPissed", "The customer is angry");
let team = lars.choice::<Team>("team", "Which team should handle this?");

if is_pissed.ask(message).await? { escalate(message) }     // async, on any executor
let team: Team = team.ask(message).wait()?;                  // blocking
```

Every call returns a `Pending`: `.await` it in async code, or `.wait()` for it in blocking code.
The request is queued on the model's worker thread when the call is made. Dropping a `Pending`
before its request runs cancels it.

- **Predicates:**
  - `noul`: `bool`
  - `choice::<T>`: a variant of an enum that derives `Options`
  - `score`: the expected level, as an `f64`
  - `multi::<T>`: a `Vec<T>`
  - `choice_keys` and `multi_keys` take options known only at runtime, and answer with `String` keys.
- **Decisions:** `decide` gives the probabilities behind an answer, its confidence, and whether a
  certified threshold accepts it.
- **Several questions at once:** `lars.ask(input, (&a, &b, &c))` asks several predicates in one
  request, so the model reads the input once, and returns a tuple of decisions. Arrays, slices
  and `Vec`s of one kind of predicate work too.
- **Other input types:** `.state(|t: &Ticket| json!({...}))` picks what the model reads, so whole
  records stay out of prompts.
- **Calibration:** `calibrate(labels, max_error)` fits the predicate to your labels and can certify
  a threshold. `save_calibrations` and `load_calibrations_from` use the same file format as the
  TypeScript and C# SDKs.

`#[derive(Options)]` turns a variant into an option:
- its description, the text the model reads, is its doc comment, or `#[option("...")]`
- its key is its name in snake_case, or `#[option(key = "...")]`

Building compiles llama.cpp, so it needs CMake and a C/C++ toolchain. On macOS it runs on Metal;
elsewhere it runs on the CPU.

Example: `cargo run --release -p moedex-lars --example molar_triage -- <pack>`.
