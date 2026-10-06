# Using the LARS SDK from another project

A short handoff for projects that want to try LARS. Last updated 2026-10-06.

## What it is

Typed yes/no, pick-one, rate-on-a-scale and pick-all-that-apply decisions, written as functions
in your code. A small language model (Qwen3 with a LARS LoRA, through llama.cpp) answers them
inside your process: no server, no network, no API keys.

- **Answers are always one of your options:** the model reads the input once and gives a
  probability for each option you declared. It never generates text.
- **Probabilities are calibrated,** and you can recalibrate a predicate on your own labels and
  certify a threshold for an error budget.
- **Speed:** on Metal, about 12–24 ms per question on the 0.6B and 40–90 ms on the 4B.

Use it for triage, routing and gating: is this message urgent, which team owns it, is this
diff risky, does this log show a failure. It is not a knowledge base, so don't ask it facts the
input doesn't contain.

## Where it lives

- **Code:** `~/Code/personal/lars-sdk`, which is github.com/moedex/lars-sdk (GitLab has a fork).
  The engine, benchmarks and training are in `~/Code/personal/lars`.
- **Status: pre-release, nothing published.** No npm, NuGet, crates.io or model Hub yet, so use
  it from the local checkout as below. The API may still change before 1.0.
- **Packs (model + calibrator):** `~/Code/personal/lars-sdk/models/packs/<name>/0.0.1`. Open
  them by path; names like `"lars-4b"` only work once packs are published.
- **Platforms:** macOS on Metal. Linux and Windows run on the CPU, but are built only in CI so
  far. No browser or mobile before 1.0.

## Which pack

| pack | size | jev-bench accuracy | use it for |
|---|---|---|---|
| `lars-0.6b-dev` | 0.4 GB | 0.646 | fast iteration, and production when you have labels to calibrate on |
| `lars-1.7b-dev` | 1.1 GB | 0.672 | in between; may be dropped |
| `lars-4b-dev` | 2.5 GB | 0.736 | the "no labels" tier |
| `lars-8b-dev` | 5.0 GB | 0.749 | experimental top tier |

For reference, the hosted Jev model scores 0.733 on the same benchmark.

**The top tier is being decided today:** the 4B is retraining on the 8B's data, and only one of
them stays. Keep the pack path in one config value.

## TypeScript (Node 22+)

```sh
pnpm add link:$HOME/Code/personal/lars-sdk/ts      # or: npm install ~/Code/personal/lars-sdk/ts
```

```ts
import { Lars } from "@moedex/lars";

const lars = await Lars.open(`${process.env.HOME}/Code/personal/lars-sdk/models/packs/lars-0.6b-dev/0.0.1`);
const urgent = lars.noul("urgent", "The message needs a reply today");
const team = lars.choice("team", "Which team should handle this?", {
  billing: "Payments, charges, refunds",
  technical: "Bugs, crashes, outages",
});

if (await urgent(message)) page();
const d = await team.decide(message);            // value, probabilities, confidence, accepted
const r = await lars.ask(message, { urgent, team }); // several questions, one pass over the input
```

## C# (.NET 10)

In the `.csproj`:
```xml
<ProjectReference Include="$(HOME)/Code/personal/lars-sdk/dotnet/Moedex.Lars/Moedex.Lars.csproj" />
```

Then build the native library once (`cargo build --release -p lars-ffi` in lars-sdk), and run
with `LARS_NATIVE=$HOME/Code/personal/lars-sdk/target/release/liblars.dylib`.

```csharp
using Moedex.Lars;

public enum Team
{
    [Option("Payments, charges, refunds")] Billing,
    [Option("Bugs, crashes, outages")] Technical,
}

using var lars = await Lars.OpenAsync(packPath);
var urgent = lars.Noul("urgent", "The message needs a reply today");
if (await urgent.AskAsync(message)) Page();     // or urgent.Check(message), blocking
Team team = await lars.Choice<Team>("team", "Which team should handle this?").AskAsync(message);
```

## Rust

In `Cargo.toml` (the first build compiles llama.cpp, so it needs CMake and a C++ compiler):
```toml
moedex-lars = { path = "/Users/ZKeown/Code/personal/lars-sdk/rust" }
```

```rust
use moedex_lars::{Lars, Options};

#[derive(Clone, Copy, Debug, PartialEq, Options)]
enum Team {
    /// Payments, charges, refunds
    Billing,
    /// Bugs, crashes, outages
    Technical,
}

let lars = Lars::open(pack_path).wait()?;           // or .await, on any executor
let urgent = lars.noul("urgent", "The message needs a reply today");
let team = lars.choice::<Team>("team", "Which team should handle this?");
if urgent.ask(message).wait()? { page() }
let (u, t) = lars.ask(message, (&urgent, &team)).wait()?;   // one pass, a tuple of decisions
```

## Each language's working example

Each one sorts a dental inbox, then calibrates on 30 labels:
- `ts/examples/molar-triage.ts`
- `dotnet/examples/MolarTriage`
- `rust/examples/molar_triage.rs`

## Getting good answers

- **Write the question as a plain claim or question about the input,** and describe options by
  what belongs in them: "Payments, charges, refunds", not "billing".
- **Send only what matters.** A state selector (`state:` in TypeScript, `State` in C#,
  `.state(...)` in Rust) picks the fields, so whole records and personal data stay out of
  prompts.
- **Calibrate on your own labels when you can.** `calibrate(labels)` on a predicate refits only
  when that is significantly better on held-out labels, and says what it kept. Accuracy on
  business decisions none of the packs trained on (typed-decisions):

  | labels per predicate | 4B | 0.6B |
  |---|---|---|
  | none | 0.61 | 0.44 |
  | 25 | 0.67 | 0.56 |
  | 50 | 0.70 | 0.61 |
  | 100 | 0.72 | 0.66 |
  | 300 | 0.76 | 0.72 |

  With about 100 labels, the 0.6B gets close to the uncalibrated 4B, at a sixth of the size
  and 3–4x the speed.
- **Certify a threshold when mistakes are costly.** `calibrate(labels, maxError: 0.05)`
  certifies a threshold, and answers below it aren't `accepted`. That needs at least 73
  accepted labels with no mistakes among them for a 5% budget. At 200 labels it typically
  accepts 7% of answers on the 0.6B and 22–27% on the 4B.
- **Save calibrations** with `saveCalibrations` / `SaveCalibrations` / `save_calibrations`. The
  file format is the same in all three languages.

## Known limits

- **Weak on world knowledge:** it judges what the text says.
- **Exact levels on fine scales are the weakest primitive:** 11–14 of 30 on a 5-level urgency
  scale. Use a score for ordering, or a yes/no question for the threshold you care about.
- **English only.**
- **The 4B on the CPU is about 0.5 s per new input,** too slow for a hot path off Metal.
- **The training Mac is shared:** while a training run is going, calls on the big packs are
  slower and slow the training down.

## More

- `README.md`: the API.
- `docs/calibration.md`: labels, thresholds and what they mean.
- `docs/packs.md`: packs, caching and environment variables.
- `RESULTS.md`: every measurement.
- `HANDOFF.md`: what is running and what is decided.
