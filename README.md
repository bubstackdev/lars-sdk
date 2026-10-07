# LARS SDK

Typed decisions as functions in your code, answered by a small language model running inside
your process.

```ts
const lars = await Lars.open({ pack: "lars-2b" });
const isPissed = lars.noul("isPissed", "The customer is angry");

if (await isPissed(message)) escalate(message);
```

```csharp
using var lars = await Lars.OpenAsync("lars-2b");
var isPissed = lars.Noul("isPissed", "The customer is angry");

if (await isPissed.AskAsync(message)) Escalate(message);
```

```rust
let lars = Lars::open("lars-2b").await?;
let is_pissed = lars.noul("isPissed", "The customer is angry");

if is_pissed.ask(message).await? { escalate(message) }
```

A predicate asks one typed question about its input:
- **noul:** yes or no
- **choice:** one of your options
- **score:** a level on an ordered scale
- **multi:** every option that applies

The model never generates text. It reads your input once and gives a probability for each
option you declared, so the answer is always one of them. Probabilities are calibrated, so
"0.9" means right about 90% of the time on data like the calibration set. You can calibrate a
predicate on your own labels, and get a threshold certified for an error budget.

There is no server, no Ollama, and no network after a model pack is downloaded. The model is
llama.cpp, linked into the library, on Metal on Macs and on the CPU everywhere else.

This repository is the SDK: TypeScript (`@moedex/lars`), C# (`Moedex.Lars`) and Rust
(`moedex-lars`), over one Rust core. The engine, the benchmark work and the training live in the
[lars](https://github.com/moedex/lars) repository; this SDK reproduces its prompts byte for
byte and its answers to the last digit.

> **Status: pre-release.** Nothing is published yet. Packages and packs are built by CI as
> artifacts. Build from source to try it (below).
>
> Using it from another project: [USING.md](USING.md).

## What a predicate gives you

| predicate | TypeScript | C# | Rust | resolves to |
|---|---|---|---|---|
| noul | `lars.noul(name, claim)` | `lars.Noul(name, claim)` | `lars.noul(name, claim)` | `boolean` |
| choice | `lars.choice(name, question, { key: "description", ... })` | `lars.Choice<TEnum>(name, question)` | `lars.choice::<Enum>(name, question)` | one of your keys / enum members |
| score | `lars.score(name, question, levels)` | `lars.Score(name, question, levels...)` | `lars.score(name, question, levels)` | the expected level (fractional) |
| multi | `lars.multi(name, question, { key: "description", ... })` | `lars.Multi<TEnum>(name, question)` | `lars.multi::<Enum>(name, question)` | the keys that apply |

Each predicate also has:
- **`decide` (TS, Rust) / `AskAsync` (C#):** the decision with its probabilities and confidence, and
  whether a certified threshold accepts it
- **`calibrate` / `CalibrateAsync` (C#):** fits its own calibrator on your labels
- **`lars.ask(input, { a, b, c })` (TS) / `AskAsync(input, a, b, c)` (C#) /
  `lars.ask(input, (&a, &b, &c))` (Rust):** puts several predicates to the model in one request,
  which reads the input once

Inputs are strings, or objects sent as JSON. Use a state selector to pick the fields that matter,
so whole records, and the personal data in them, stay out of prompts:

```ts
const urgent = lars.noul<Ticket>("urgent", "The ticket needs a reply today",
  { state: (t) => ({ subject: t.subject, body: t.body }) });
```

C# choices are enums; `[Option]` gives each member the description the model reads:

```csharp
public enum Team
{
    [Option("Payments, charges, refunds")] Billing,
    [Option("Bugs, crashes, outages")] Technical,
}
Team team = await lars.Choice<Team>("team", "Which team should handle this?").AskAsync(message);
```

Rust choices are enums too, and `#[derive(Options)]` makes each variant's doc comment its
description. Every Rust call returns a `Pending`: `.await` it on any executor, or `.wait()` for it
in blocking code.

```rust
#[derive(Clone, Copy, Debug, PartialEq, Options)]
enum Team {
    /// Payments, charges, refunds
    Billing,
    /// Bugs, crashes, outages
    Technical,
}
let team: Team = lars.choice::<Team>("team", "Which team should handle this?").ask(message).await?;
```

Full examples: `ts/examples/molar-triage.ts`, `dotnet/examples/MolarTriage` and
`rust/examples/molar_triage.rs`, which sort a dental practice's inbox.

## Packs

A pack is a model plus everything its probabilities depend on:
- the quantized weights
- the chat template and prompt layout
- the llama.cpp build it was validated on
- flash attention on or off
- a pooled calibrator fitted on jev-bench validation

`Lars.open({ pack: "lars-2b" })` downloads a pack once from the Hub, checks its sha256 and
caches it. A path to a pack directory works too. `npx lars pull lars-2b` (or `Lars.pull`)
downloads one ahead of time. See [docs/packs.md](docs/packs.md).

Current development packs, each a base model with the LARS LoRA, evaluated through this SDK on
jev-bench test (22 configs):

| pack | base model | quantization | size | macro accuracy | Brier |
|---|---|---|---|---|---|
| `lars-2b-dev` (default) | Qwen3.5-2B | Q4_K_M | 1.3 GB | 0.716 | 0.333 |
| `lars-8b-dev` 0.0.2 (top tier) | Qwen3-8B | Q3_K_M | 4.1 GB | 0.748 | 0.287 |
| `lars-0.6b-dev` (smallest) | Qwen3-0.6B | Q4_K_M | 0.4 GB | 0.646 | 0.421 |

The 8B certifies the most answers at a strict error budget; the 2B is the balance of accuracy,
memory and speed. `lars-4b-dev` (0.736) and `lars-1.7b-dev` (0.672) are retired: the 8B ties the
4B on accuracy and certifies more, and the 2B beats the 1.7B by 4 points (`RESULTS.md`,
2026-10-07).

For comparison, the hosted Jev model's published number on the same benchmark is 0.733 /
0.349. The small packs close most of the gap on routing and triage, and lose most on world
knowledge (`RESULTS.md`, 2026-10-05).

## Calibration

The pack's calibrator is a good default. Calibrating a predicate on your own labels adapts it
to your data. With an error budget, it also certifies a threshold: answers the threshold doesn't
accept make the boolean `false`, and `decide().accepted` says why. If the labels can't certify
the budget, nothing is accepted until they can.

```ts
const fit = await isPissed.calibrate(labeled, { maxError: 0.05 });
// fit.kept: "pack", "fitted", "recal" or "head"; fit.certified: threshold, coverage, error bound, or why not
```

How many labels it takes, and what the numbers mean: [docs/calibration.md](docs/calibration.md).

## Speed

Indicative only: measured on an M5 Max while other jobs ran. Quiet-machine numbers are pending.

| | one yes/no question | three questions (Molar Triage) | twenty questions, one message |
|---|---|---|---|
| 2B, Metal | 18–40 ms | 77–290 ms | 190–215 ms |
| 8B Q3_K_M, Metal | 60–120 ms | 180–730 ms | 410–510 ms |
| 0.6B, Metal | 12–24 ms | 45 ms | 65 ms |
| 2B, CPU (12 threads) | 240–260 ms | 0.6–2.5 s | 1.5–1.9 s |


The lower figure in a range is the same input asked again; the higher is a new input. Memory is
about the model file plus 1 GB, more on CPU, where llama.cpp repacks weights. Loading takes 0.3–0.5 s
on Metal and about 1.4 s on CPU.

## Limits

- **Not a knowledge base.** These are small models: they judge what the text says, and are weak
  on facts it doesn't contain (mmlu 0.42–0.70 across the packs).
- **Fine-grained scales are the weakest primitive.** On Molar Triage, the exact urgency level
  (of five) was right on 11–14 of 30 messages. Use a score for ordering, or a noul for the
  threshold you care about.
- **A threshold only holds for data like its labels.** Certified thresholds and calibrated
  probabilities describe the labels they were fitted on. Recalibrate when your inputs change.
- **The 8B on CPU is too slow for a hot path** (about 0.5 s for a new input). On machines
  without Metal, use a small pack or move the call off the request path.
- **Platforms:**
  - macOS (arm64, x64) uses Metal; Linux (x64, arm64, glibc 2.28+) and Windows x64 run on the
    CPU
  - no CUDA or Vulkan builds yet
  - no browsers or mobile before 1.0
- **English.** The packs were trained and evaluated in English.

## How it works

```
TypeScript (@moedex/lars)          C# (Moedex.Lars)                     Rust (moedex-lars)
        │ napi-rs (crates/lars-node)        │ P/Invoke (crates/lars-ffi, lars.h)        │ directly
        └──────────────┬────────────────────┴───────────────────────────────────────────┘
                lars-runtime    packs, downloads, one worker thread per loaded model, calibrators
                lars-core       request schema, prompt rendering, answers, constraints, calibration
                lars-llama      llama.cpp, vendored at a pinned commit and linked statically
```

Each question becomes a prompt that ends in `Answer:`. The model reads it once, and the engine
takes the next-token logits of the option letters. Those become a distribution over your options,
then a calibrated answer.

The shared prefix (system prompt and input) is computed once per input, and a request's
questions are decoded in one batch. One worker thread owns each loaded model. Calls from any
thread queue on it, and async calls never block the event loop or the thread pool.

## Building from source

Needs Rust (stable), CMake, and a C/C++ toolchain; Node 22+ and pnpm for TypeScript; .NET 10 for C#.

```sh
git clone --recursive https://github.com/moedex/lars-sdk && cd lars-sdk
cargo test -p lars-core                       # the core, against the reference fixtures
cargo test -p moedex-lars                     # the Rust SDK

cd ts && pnpm install && pnpm build:native && pnpm test
cd ../dotnet && cargo build --release -p lars-ffi \
  && LARS_NATIVE=$PWD/../target/release/liblars.dylib dotnet test Moedex.Lars.Tests
```

All three test suites run on a mock pack with no model. Set `LARS_TEST_PACK` to a pack directory to
run the real-model tests too.

| | |
|---|---|
| `crates/` | Rust: `lars-core`, `lars-llama-sys`, `lars-llama`, `lars-runtime`, `lars-ffi`, `lars-node`, `lars-dev` (developer CLI) |
| `ts/` | `@moedex/lars` |
| `dotnet/` | `Moedex.Lars`, its tests and example |
| `rust/` | `moedex-lars` and its `#[derive(Options)]` crate, tests and example |
| `fixtures/` | reference outputs from the Python engine, and the scripts that produce them |
| `sweep/` | the model-size sweep and training pipeline behind the packs |
| `PLAN.md`, `RESULTS.md` | the plan to 1.0, and every measurement so far |

## License

MIT. Packs carry their base model's license (Qwen3: Apache 2.0) and the training data's
attributions.
