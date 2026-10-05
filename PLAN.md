# LARS SDK plan, 2026-10-02

LARS typed decisions as functions in application code, with the model running inside the
calling process: no server, no Ollama, nothing hosted. `if (await isPissed(message))` in
TypeScript, `if (isPissed.Check(message))` in C#.

1.0 means TypeScript and C# packages on npm and NuGet, prebuilt for macOS, Linux and
Windows, serving one published model pack whose accuracy and calibration were measured
through the SDK's own runtime. Check boxes as work lands.

## Decisions

| question | decision (2026-10-02) |
|---|---|
| languages for 1.0 | TypeScript (Node 22+, Bun) and C# (.NET 10; .NET 8 leaves support in November 2026) |
| platforms | local machines: macOS arm64 and x64, Linux x64 and arm64, Windows x64. Browsers and mobile after 1.0 |
| hosted models | none. Local only; the prototype's Haiku escalation is out of scope |
| inference | llama.cpp, vendored at a pinned commit and linked into the SDK's native library |
| engine logic | written once, in a Rust core. Languages wrap it and never reimplement it |
| language boundary | a C ABI that takes and returns System One JSON |
| reference | the Python engine in `~/Code/personal/lars` at a pinned commit. The core reproduces its prompts byte for byte |
| training | stays in the lars repo (Python, MLX). The SDK runs inference, calibration fitting and evaluation |

### Why one core

A calibrator is valid for one combination of weights, quantization, runtime, prompt bytes
and label tokens. The prototype has measured what breaks it. Re-quantizing the fused 4B at
4 bits took it from 0.740 to 0.703. Changing only the shape of MLX's output projection moved
probabilities by up to 0.037 without changing a top label. Two hand-written renderers
would drift apart, and every calibrator would quietly stop meaning anything. So there is one
renderer and one runtime, and calibrators are fitted through the same code that serves.

## The model question

Codex's answers (2026-10-02) say nothing has been measured yet:
- nothing below 4B on jev-bench, zero-shot or with a LoRA
- no fused GGUF
- nothing on CPU, and no parity across runtimes

So the SDK measures model size itself, in the runtime it ships (Phase 2b), rather than adding
experiments to the training work.

- **Candidates:** Qwen3-0.6B and Qwen3-1.7B, both Apache 2.0, with Qwen3-4B-Instruct-2507 as
  the reference. Gemma and LFM2 need a redistribution review before they're worth measuring.
- **Accuracy is the wrong headline for a small model.** In a private routing probe (139 test
  rows), the zero-shot 0.6B matched a trained 4B's accuracy (88.5% against 84.2%). At a
  threshold chosen on validation, though, it could accept only 7.9% of rows, against the 4B's
  67.6%. The sweep's headline is therefore coverage at an error budget, next to macro accuracy
  and Brier. The threshold is chosen on validation and measured on test, as
  `evals/risk_coverage.py` does.
- **Templates:** the small Qwen3 models have hybrid-thinking chat templates. A pack's template
  must come from the model, not from hand-written ChatML (see "Prompt parity details").

## Architecture

```
TypeScript (@moedex/lars)              C# (Moedex.Lars)
  predicates, inferred types             predicates, enums, Task / bool
        │ napi-rs                              │ P/Invoke (LibraryImport)
        ▼                                      ▼
  lars-node (.node per platform)        lars-ffi (liblars.dylib / .so / .dll), C ABI in lars.h
        └──────────────────┬───────────────────┘
                           ▼
                     lars-core (Rust)
     schema · render · labels · reduce · constraints · calibration · eval · packs · worker
                           │
                     lars-llama-sys ──▶ llama.cpp @ pinned commit (CPU everywhere, Metal on macOS)
```

`lars-node` links the core directly rather than going through the C ABI. Both front ends
are thin.

### What the core ports from the prototype

| Python (`lars`) | core module | notes |
|---|---|---|
| `schema.py` | `schema` | serde, same limits; `extra="forbid"` becomes `deny_unknown_fields` |
| `render.py` | `render` | byte-identical output, including Python-compatible JSON rendering and the model's own chat template (below) |
| `labels.py` | `labels` | 255 verified single-token labels per pack, or the pack fails to load |
| `primitives.py` | `primitives` | |
| `engine.py` | `engine` | rows, reduction, constraints (Dykstra), evidence, budgets (512 rows, 32,768 tokens) |
| `calibration.py` | `calibration` | temperature, Platt, fusion, ECE, Brier, coverage; same calibrator JSON |
| `evalset.py` | `eval` | the JSONL format, eval and calibrate. Follows Codex's stricter soft-target decoding once it is committed |
| `backends/mock.py` | `backend::mock` | so SDK tests run without a model |
| `backends/llamacpp.py` | `backend::llama` | rewritten on llama.cpp's C API |

Not ported: MLX, the server, MCP, the launchd service, escalation, pointer heads,
multi-adapter ensembles and training.

### Inference on llama.cpp

- GGUF weights are memory-mapped, so several processes share one copy through the page cache.
- At load, tokenize `" A"` to `" ZZ"` and keep the first 255 single-token labels, as `assign_labels` does.
- Rows are grouped by prefix as in the prototype. The boundary check is the same: if the
  prefix's tokens aren't a prefix of the full prompt's, that row gets a full prefill.
- Prefill the prefix into sequence 0. Then copy its cache cells to one sequence per suffix
  and decode every suffix in one batch, reading logits at each suffix's last token. That
  answers all of a request's questions in one forward pass, instead of the prototype's
  restore-then-decode loop for each suffix.
- Cache the constant head of every prompt (system prompt and template up to the state fence) across calls.
- **Measure batch invariance before batching is on by default.** llama.cpp doesn't
  guarantee the same logits for a row decoded alone and the same row inside a batch. A
  shape change of exactly this kind moved probabilities on MLX. If the difference matters,
  rows decode one at a time by default and batching is opt-in.
- llama.cpp stays pinned. A pack records the commit it was validated on, and any bump
  re-runs the parity fixtures and the pack's evaluation.
- `n_ctx` and `n_seq_max` are sized from the request budgets, and suffix batches are chunked to fit.

### Render versions

- **v1** is the prototype's layout: system prompt, state, question. Every existing adapter
  and calibrator was trained or fitted on it.
- **v2** puts the question first: system prompt, question and options, state, `Answer:`. A
  predicate is defined once and called on many states, so with v2 its prefix is prefilled
  once and each call prefills only its state. It ships only if Codex's measurement
  (question 4 below) says accuracy holds, and adapters must be trained on v2 to use it.
- A pack names its render version. The core supports both.

### Prompt parity details

- **Chat templates come from the model.** Training renders prompts with the tokenizer's chat
  template, with thinking switched off (`MLXBackend._hf_template`).
  - Checked 2026-10-02: for Qwen3-4B-Instruct-2507 this is exactly the hand-written ChatML.
    For Qwen3.5-4B it adds `<think>\n\n</think>\n\n` after the assistant header. Codex reports
    the same for the small hybrid Qwen3 models.
  - The fixture generator captures each model's template once, as a literal string with
    system and user placeholders, and the pack carries it. The core substitutes into it, with
    no Jinja at runtime.
  - The prototype's llama.cpp backend uses hand-written ChatML, so for hybrid models it isn't a
    reference; MLX is.
- JSON state renders like Python's `json.dumps(indent=2, ensure_ascii=False)`:
  - insertion order is kept
  - floats use Python's repr (`1e-05`, `1e+16`, `1.0`), not Rust's
  - integers keep their digits
- Splitting the state into units and `strip()` follow Python's Unicode whitespace rules,
  which differ from Rust's on a few code points. Python's lookbehind split is reimplemented
  by hand, since the `regex` crate has no lookbehind.
- Permutation rows: Python shuffles with `random.Random(f"{qid}:{n}")`. The core reproduces
  CPython's string seeding, MT19937 and `shuffle` (about 80 lines, checked against Python),
  so `perm:k` rows match exactly.
- JavaScript objects list integer-like keys first, in ascending order, whatever order they
  were written in. The core can't undo that; the TS docs say so.

### Language boundary

```c
lars_engine *lars_open(const char *pack_dir, const char *options_json, lars_error **err);
char *lars_evaluate(lars_engine *, const char *request_json, lars_error **err);  /* blocks */
void  lars_evaluate_async(lars_engine *, const char *request_json, lars_callback cb, void *user);
char *lars_calibrate(lars_engine *, const char *examples_jsonl, lars_error **err);
int   lars_set_calibrator(lars_engine *, const char *name, const char *calibrator_json, lars_error **err);
char *lars_status(lars_engine *);
void  lars_cancel(lars_engine *, uint64_t request_id);
void  lars_string_free(char *);  void lars_error_free(lars_error *);  void lars_close(lars_engine *);
```

- JSON in and out, in the System One request and response shapes plus the `lars`
  extensions. Serializing costs microseconds next to a forward pass, and the ABI stays
  fixed while the language APIs grow.
- Errors are `{message, error_type}`, as in the prototype's server. No panic crosses the
  boundary (`catch_unwind`).
- Each engine owns one worker thread and a queue, and only that thread touches the llama
  context. A blocking call waits on the worker; an async call gets a callback from it.
  Neither ties up Node's libuv pool or .NET's thread pool.
- Engines opened on the same pack in one process share one loaded model, reference-counted.
- Idle unload, as in `lars.lazy`, is an option and off by default.

### Model packs

```
lars-small@1/                 (the name is a placeholder)
  pack.json         name, version, base model and its license, render version,
                    chat template (captured, thinking off), llama.cpp commit it was
                    validated on, file hashes, evaluation summary
  model.gguf
  calibration.json  pooled calibrator, in the prototype's format
  LICENSE, NOTICE   base model and training-data attributions (lars DESIGN.md §8.1)
```

- `Lars.open({ pack: "lars-small" })` downloads the pack once from the Hub
  (`moedex/lars`, under `sdk/`), checks sha256 and caches it:
  - macOS: `~/Library/Caches/lars`
  - Linux: `$XDG_CACHE_HOME/lars`
  - Windows: `%LOCALAPPDATA%\lars`

  The download lives in the core (ureq with rustls, honoring `HTTPS_PROXY`), so both
  languages share it.
- `LARS_PACK_DIR` or an explicit path works offline, and `LARS_OFFLINE=1` never downloads.
- A calibrator records the pack it was fitted on. Loading it against another pack is an error.
- Weights never go inside the npm or NuGet packages.
- Dev pack `qwen3-4b-zs@0`: built from `lars/models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf`,
  zero-shot and uncalibrated. Local only, never published.

## Developer API

TypeScript:

```ts
import { Lars } from "@moedex/lars";

const lars = await Lars.open({ pack: "lars-small" });

const isPissed = lars.noul("isPissed", "The customer is angry", { threshold: 0.7 });
if (await isPissed(message)) escalate(message);           // resolves to a boolean
const d = await isPissed.decide(message);                 // { p: 0.83, abstain: false }

const team = lars.choice("team", "Which team should handle this?",
  { billing: "Payments, payouts", technical: "Bugs, outages" });
const t = await team(message);                            // "billing" | "technical"

const urgency = lars.score("urgency", "How urgent is this?", ["routine", "soon", "today", "emergency"]);
const tags = lars.multi("tags", "Which apply?", { refund: "Wants money back", bug: "Reports a defect" });

// Several questions over one state in one pass; constraints apply across them.
const r = await lars.ask(message, { isPissed, team, urgency });

const fit = await isPissed.calibrate(labeled, { maxError: 0.05 });  // [{ state, label: true }, ...]
fit.threshold; fit.coverage;                              // with its 95% interval
await lars.saveCalibrations("./lars-calibrations");
```

- With `maxError`, `calibrate` also picks the acceptance threshold on the labels and reports
  the expected coverage with its interval.
- `decide()` then says whether an answer is accepted, and the boolean form treats an
  unaccepted answer as false.
- For a small model, this matters more than raw accuracy (see "The model question"). With a
  few dozen labels the interval is wide; the Phase 2b few-label curves say how wide.

- Calls are always async and resolve to plain values. An object is always truthy in
  JavaScript, so returning a `Decision` would make every `if` pass.
- Predicates take `string | object`, and objects go in as JSON. A `state: (x) => ...`
  option picks the fields, so whole records, and the personal data in them, stay out of prompts.
- The predicate's name (`"isPissed"`) is its question id and the key for its calibrator,
  so names are unique within an engine.

C#:

```csharp
using var lars = await Lars.OpenAsync(new LarsOptions { Pack = "lars-small" });

var isPissed = lars.Noul("isPissed", "The customer is angry", threshold: 0.7);
if (isPissed.Check(message)) Escalate(message);          // blocks: console apps, scripts
if (await isPissed.AskAsync(message)) Escalate(message); // Decision converts to bool
Decision d = await isPissed.AskAsync(message);            // d.P, d.Abstain

public enum Team
{
    [Option("Payments, payouts")] Billing,
    [Option("Bugs, outages")] Technical,
}
var team = lars.Choice<Team>("team", "Which team should handle this?");
Team t = await team.AskAsync(message);
```

- `Decision` has `implicit operator bool`. A noul predicate also converts implicitly to
  `Func<object, bool>`, so `Func<object, bool> isPissed = lars.Noul(...); if (isPissed(user))`
  compiles. That call is synchronous.
- `AskAsync` completes from the native callback and takes a `CancellationToken`, which
  cancels a queued request.
- In 1.0, enum options are read by reflection. A source generator comes only if trimming
  or AOT needs it.

## Parity with the Python reference

`fixtures/generate.py` runs the Python engine from a `git archive` of a pinned lars commit
(`ab59a0b` to start), using the lars venv's interpreter. Codex's working tree and venv are
never touched. The fixtures are small JSON files and are committed.

| fixture | contents | must match |
|---|---|---|
| templates | each candidate model's chat template, captured from its tokenizer with thinking off | literal string |
| render | rows for about 60 requests: every question type, JSON and Unicode state, every template, the sentinel inside state, explain over 24 units, multi, permutations | prefix and suffix byte-identical |
| reduce | the same requests with seeded synthetic logits per row, with and without calibrators, constraints, abstain, evidence and features | choice, selected and abstain equal; numbers within 1e-4 |
| calibration | inputs and fitted values for `fit_temperature`, `fit_platt` and `fit_fusion`; ECE, Brier and coverage | within 1e-6 |
| mock | mock-backend logits on the render requests | within 1e-12 |
| llama | raw label logits from the Python llama.cpp backend on the dev pack, over Molar Triage and 200 sampled jev-bench validation rows | the same top label on every row; report the largest logit difference |

llama-cpp-python bundles its own llama.cpp, so the llama fixture runs in a separate venv
under `fixtures/`, and exact logit equality isn't expected. After that check, the core's
own logits become the regression fixture, re-checked on every llama.cpp bump. That Python
backend is a valid reference only for models whose template is plain ChatML, such as the dev
pack's Qwen3-4B-Instruct-2507. For hybrid-thinking models, compare against MLX instead: top
answers must agree, and logit differences are reported.

## Phases

### Phase 0: repository and reference (no model)

- [x] `git init`, Rust workspace under `crates/`, and `fixtures/`. `ts/` and `dotnet/` come in
  their own phases.
- [x] CI (`.github/workflows/ci.yml`) runs fmt, clippy and tests on macOS, Linux and Windows.
  Written, not run yet: there's no remote.
- [x] `fixtures/generate.py` against `ab59a0b`. Fixtures not committed yet.

**Done 2026-10-02:** two runs of `generate.py` give byte-identical fixtures. They cover:

| fixture | cases |
|---|---|
| render | 52 requests, 162 rows, 648 prompts across four templates |
| reduce | 104 responses, 26 refused requests |
| mock | 46 responses |
| calibration | fits and metrics |
| eval | Molar Triage |
| templates | 4 captured models |

The captured templates confirm Codex's point. Qwen3-0.6B and 1.7B add
`<think>\n\n</think>\n\n`. Qwen3.5-4B adds it and also trims message content. Only
Qwen3-4B-Instruct-2507 is plain ChatML.

### Phase 1: core logic (no model)

- [x] `schema`: validated by hand over parsed JSON, with the same limits. Error messages
  contain pydantic's, and `error_type` is identical. Lax coercions such as numeric strings
  for integers are not reproduced.
- [x] `render` v1: Python-compatible JSON, unit splitting, nonce (sha256, first 8 hex),
  `compose_prompt`, and templates. Tests use the prototype's four; packs use captured ones.
- [x] `labels`, `primitives` (with numpy's pairwise summation), `reduce`, constraints,
  evidence, budgets.
- [x] CPython-compatible shuffle for permutation rows.
- [x] `calibration`: fitting and metrics; calibrator JSON identical to the prototype's.
- [x] `backend::mock`.
- [x] `eval`: JSONL reader, evaluate, calibrate.

**Done 2026-10-02 (`cargo test`, 18 tests, clippy clean):**
- Rendering: every row field and every prompt byte matches.
- Responses: every number matches exactly across all 104 replay and 46 mock responses, after
  the API's 4-decimal rounding.
- Mock backend: logits are bit-identical.
- Calibration: the 400-point temperature grid and the fitted temperatures are bit-identical
  on macOS. Platt is within 1e-9, fusion within 1e-8.
- Eval and calibrate on Molar Triage match exactly.
- Tests that depend on the platform's libm compare with a tolerance, so CI on Linux and
  Windows checks closeness rather than bits.

### Phase 2: llama.cpp backend

Needs `cmake` (`brew install cmake`); llama.cpp builds only with CMake.

- [x] `lars-llama-sys`: llama.cpp b11349 as a shallow git submodule, built by CMake from
  `build.rs`. The build is static, with Metal embedded on macOS and no OpenMP. Bindings are
  generated by `regenerate-bindings.sh` and committed. A clean build takes about 20 s.
- [x] `lars-llama` (`LlamaBackend`):
  - [x] load the model and verify labels
  - [x] tokenize with special tokens
  - [x] boundary check with full-prefill fallback
  - [x] batched suffix decode over copied sequences (`batch_suffixes`, opt-in)
  - [x] reuse of the previous call's prefix (`reuse_prefix`, opt-in)
  - [x] context sizing; a row over `n_ctx` is a clear error
- [ ] Dev pack from the local GGUF: moved to Phase 3 with the pack format. Until then
  `lars-dev` takes `--model` and `--template`.
- [x] `lars-dev` CLI: `ask`, `eval`, `calibrate`, `capture`, `capture-requests`, `compare`,
  `bench`, `parity`.
- [x] Measure, on a quiet machine (Codex shares this GPU, so take latency only when nothing
  else is running). Results are in `RESULTS.md`:
  - [x] parity against the Python backend: bit-identical with flash attention off, with or
    without prefix reuse; `cargo test -p lars-llama` checks it with `LARS_TEST_GGUF`
  - [x] run-to-run determinism: bit-identical
  - [x] batch invariance: not invariant (432 of 498 rows move, by up to 0.009)
  - [x] invariance to CPU thread count: bit-identical at 6, 12 and 18 threads
  - [x] CPU against Metal: not identical; 90th / 99th percentile probability difference
    0.029 / 0.168, one top label of 164 differs. A native build differs from the portable one
    by as much.
  - [x] p50 latency on CPU and on Metal for one noul, the Molar Triage request and twenty
    questions

**Done 2026-10-02.**
- **Defaults:** flash attention off, prefix reuse on, batching off.
- **Calibrators and compute paths:**
  - Each pack fixes its flash-attention setting.
  - CPU, Metal and each build's SIMD path give slightly different logits.
  - So a calibrator either has to be fitted per compute path, or shown to transfer between
    them.
  - Phase 2b adds that check, since fitting per OS and architecture would mean calibrating
    every build.
- **Batching:** a request with many questions is 3.6× faster batched. Its drift (0.009) is
  20 times smaller than the CPU/Metal difference, so if calibrators transfer across paths,
  batching should become the default.

### Phase 2b: model sweep in the shipping runtime

No GPU training and no new work for Codex. The SDK's own harness answers the size questions.

- [x] GGUFs for Qwen3-0.6B, Qwen3-1.7B and Qwen3-4B-Instruct-2507, at BF16, Q8_0 and
  Q4_K_M. All are built from the original weights with the pinned llama.cpp's converter and
  quantizer (`sweep/prepare.sh`), and each gets its captured template.
- [x] `lars-dev capture`: raw label logits for every jev-bench validation and test row, in
  render v1 and v2, on Metal and on CPU (`sweep/run.sh`, `sweep/queue.sh`).
- [x] From the captures (`sweep/analyze.py`): pooled and per-config calibration, coverage at
  5% (empirical and Wilson thresholds), few-label curves, v2 against v1.
- [ ] Quiet-machine latency for the candidate packs (`bench/run.sh`): still to do.
- [x] The private TurnCommerce decisions: dropped from Phase 2b. The jev-bench routing and
  triage configs answered the size question.
- [x] One table in `RESULTS.md` (2026-10-05).
- [x] Calibrator transfer across compute paths: it transfers. Accuracy and Brier change by
  ≤0.0004, so one calibrator per pack, and batching can be the default.
- [ ] CPU memory with weight repacking off: not measured yet; part of the quiet latency run.
- [x] Small-model LoRAs (`sweep/train.sh`): 0.6B 0.651 and 1.7B 0.683 at Q8_0, against the
  4B LoRA's 0.743.

**Done 2026-10-05 except latency.**
- **v1 layout:** v2 loses at every size.
- **Candidate packs:**
  - 4B LoRA Q4_K_M: 2.5 GB, 0.736
  - 1.7B LoRA Q4_K_M: 1.3 GB, 0.672
  - 0.6B LoRA Q4_K_M: 0.5 GB, 0.646
- **Still to decide:** which small tier ships for CPU, by quiet latency; and imatrix
  quantization, which should close the remaining Q4_K_M gap.

### Phase 3: native library and lifecycle

- [x] `lars-runtime`, which every binding sits on (Node links it directly):
  - [x] a worker thread per loaded model with a queue, async replies, cancellation
  - [x] models shared across handles on the same pack; the idle-unload option
  - [x] per-handle named calibrators, and `calibrate` with a Wilson-certified threshold
- [x] `lars-ffi`: the C ABI over it, `include/lars.h` generated by cbindgen and checked in CI
- [x] Packs:
  - [x] manifest, sizes and sha256, cache directories, offline mode
  - [x] downloads from the Hub, verified before install
  - [x] each calibrator bound to its pack: unbound or foreign calibrators are refused
  - [x] `lars-dev pack` builds one: the pooled calibrator fitted and the evaluation run through the core
- [x] Tests:
  - [x] a C smoke test (`crates/lars-ffi/tests/c/smoke.c`)
  - [x] concurrency: 8 threads in Rust, 4 in C
  - [x] downloads against a local server
  - [x] leak check with `leaks` on macOS; ASan on Linux not yet

**Done 2026-10-05.**
- **A C program on the dev pack** (`models/packs/lars-4b-dev/0.0.1`, the 4B LoRA Q4_K_M) opens
  it, answers sync and async requests, answers from four threads at once, calibrates, cancels
  and closes, with exactly one callback per request.
- **No leaks:** the same program on the mock pack runs under `leaks` with none found.
- **The pack's evaluation matches the Python analysis exactly.** It ran through the core: 0.7361
  macro accuracy and 0.3042 Brier, per config identical to `sweep/analyze.py`.

### Phase 4: TypeScript

- [x] `lars-node` on napi-rs over `lars-runtime`, with promises resolved from the worker thread
  (never libuv's pool).
- [x] `@moedex/lars`:
  - [x] `Lars.open`, plus `noul`, `choice`, `score` and `multi` with inferred return types
  - [x] `decide`, and `ask` for several questions with constraints
  - [x] options: threshold, abstain margin, explain, permutations, state selector, fallback
  - [x] `calibrate` with a certified threshold that gates the boolean; save and load calibrations
  - [ ] cancellation (`AbortSignal`)
  - [ ] `npx lars pull <pack>`
- [x] Tests:
  - [x] vitest on the mock pack, and a real-pack test with `LARS_TEST_PACK`
  - [x] in CI on Node 22 and 24, and Bun, on three OSes
- [x] Molar Triage example in TypeScript.

**Done 2026-10-05,** apart from cancellation and `lars pull`.

### Phase 5: C#

- [x] `Moedex.Lars`, targeting `net10.0`:
  - [x] `LibraryImport` over `lars-ffi`, with a `SafeHandle`
  - [x] callbacks through `UnmanagedCallersOnly`; continuations off the worker thread
- [x] The TypeScript surface in C# idiom:
  - [x] `Check` and `AskAsync`
  - [x] `Decision` converting to `bool`
  - [x] `Choice<TEnum>` with `[Option]`
  - [x] the `Func<TInput, bool>` conversion
  - [x] `CancellationToken` cancelling queued calls
- [x] xUnit tests on the mock pack, plus a real-model test behind `LARS_TEST_PACK`; CI on three OSes.
- [x] Molar Triage example in C#.

**Done 2026-10-05.** Enum options are read by reflection; no source generator yet.

### Phase 6: the first real pack (needs Phase 2b and a LoRA from Codex)

- [ ] Choose the base model, quantization and render version from Phase 2b.
- [ ] Codex trains a LoRA for that size on MLX, in the chosen render version with the
  captured template.
- [ ] Fuse and convert to GGUF two ways, then check each against the unfused adapter for
  top-answer agreement on captured rows, as the 300-row MLX check did:
  - [ ] plain Q4_K_M
  - [ ] the adapter's layers kept at Q8_0, the GGUF counterpart of `fuse_mixed.py`
- [ ] Fit the pooled calibrator through the SDK core on jev-bench validation, and evaluate
  on test through the core. MLX numbers don't count for a pack.
- [ ] The few-label calibration curve through the core, turned into guidance on how many
  examples to label.
- [ ] LICENSE and NOTICE; publish to `moedex/lars` under `sdk/` once you sign off.

Done when the pack's manifest carries its own as-shipped accuracy, calibration and latency.

### Phase 7: packaging and 1.0

- [x] CI builds the native library and the Node addon per platform (`package.yml`):
  - `osx-arm64`, and `osx-x64` cross-compiled
  - `linux-x64` and `linux-arm64`, in manylinux_2_28 containers (glibc 2.28)
  - `win-x64`
- [x] npm: `@moedex/lars`, with one optional `@moedex/lars-<platform>-<arch>` per platform
  (os, cpu and libc fields). NuGet: `Moedex.Lars` with `runtimes/<rid>/native`.
- [x] Install tests from the built packages, into fresh projects, on five runners (macOS arm64
  and Intel, Linux x64 and arm64, Windows). Verified locally on macOS arm64 for npm (Node and
  Bun) and NuGet; the x86_64 build was checked under Rosetta through the C ABI on a real pack.
- [x] Docs:
  - [x] README
  - [x] package READMEs
  - [x] `docs/calibration.md`
  - [x] `docs/packs.md`
  - [ ] an API reference generated from the doc comments
- [ ] Publishing to npm, NuGet and the Hub: held until you sign off (2026-10-05).
- [ ] Version 1.0.0, tag, release notes: with publishing.

## Codex's answers, and what's left for Codex

**2026-10-02: Codex is shut down; every experiment below is the SDK's.**
- Codex's uncommitted lars work was reviewed and committed as found on lars branch
  `codex-wip-2026-10-02` (`b4a1180`); the lars test suite passes.
- `sweep/train.sh` trains small-model LoRAs from that branch on the repaired corpus E, then
  fuses, converts, quantizes and captures them through the SDK, unattended under launchd.

Answered 2026-10-02:
- Questions 1–6 have no measurements yet.
- Questions 7 and 9 changed the plan: packs use captured templates, and the private decisions
  become an outside evaluation set.
- Question 8: Qwen3-0.6B and 1.7B are Apache 2.0.

| item | who |
|---|---|
| zero-shot size sweep, GGUF accuracy, CPU latency, v1 against v2 zero-shot, few-label curves, llama.cpp determinism | the SDK, Phase 2b |
| MLX against llama.cpp logit parity | the SDK, on the chosen model |
| a LoRA for the chosen size, in the chosen layout | Codex |
| the fused-GGUF agreement check | the SDK, with Codex's adapter |
| committing the stricter soft-target decoding in `evalset.py` | Codex; the fixtures then move to that commit |
| FYI: the prototype's llama.cpp backend uses hand-written ChatML, which isn't the training prompt for hybrid-thinking models | Codex |
| Gemma and LFM2 redistribution review | only if a non-Qwen candidate looks worth it |

## Risks

| risk | mitigation |
|---|---|
| runtime changes shift logits (llama.cpp bumps, batching, CPU against Metal) and void calibrators | pinned llama.cpp; packs record their validated commit; calibrators fitted through the core; parity fixtures on every bump |
| the smallest models are too weak | the numbers decide; per-predicate calibration; fall back to a larger tier, as the prototype has 4b and 30b |
| a small model answers well but its confidence says little (the 0.6B probe: similar accuracy, 7.9% coverage) | choose on coverage at an error budget measured on test, not accuracy; LoRA; per-predicate thresholds |
| a hand-written template that isn't the training prompt | captured templates in every pack; a fixture per candidate model |
| CPU latency is too high for an inline `if` | measured in Phase 2; render v2 and batching are the levers |
| native packaging (glibc versions, MSVC, Intel Macs) | old-glibc build images; install tests from the published packages |
| prompt parity edge cases | the core renders all JSON itself; fixtures cover floats, Unicode and the sentinel |
| memory, one model per process | mmap shares weights across processes; one loaded model per pack per process |

## Not in 1.0

- Browsers (WASM) and mobile.
- CUDA, Vulkan and ROCm builds.
- Python, Go, JVM and Swift bindings.
- Hosted escalation, pointer heads, multi-adapter ensembles.
- Training, which stays in the lars repo.

## Settled 2026-10-02

1. Package names: npm `@moedex/lars` and NuGet `Moedex.Lars`.
2. Repository: `moedex/lars-sdk` on GitHub. The TC GitLab copy is a fork of it.
3. `brew install cmake` when Phase 2 starts.
