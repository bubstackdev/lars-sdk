# Flash teacher bridge

A persistent JSONL worker and Python `FlashBackend` for LARS offline teacher capture.
The worker links the existing Flash engine without modifying or rebuilding that
checkout. This is the offline teacher path; a public Rust/C FFI is a later step.

## Build and CPU checks

From the LARS repo:

```sh
cmake -S sweep/flash_teacher -B target/flash-teacher \
  -DFLASH_REPO=/Users/ZKeown/Code/lab/flash -DCMAKE_BUILD_TYPE=Release
cmake --build target/flash-teacher -j 2

FLASH_TOKENIZER_MODEL=/Users/ZKeown/models/flash/qwen38-flash-next-flash-v1.gguf \
FLASH_CHAT_TEMPLATE=/Users/ZKeown/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/chat_template.jinja \
.venv-train/bin/python -m unittest training.tests.test_flash_backend training.tests.test_flash_teacher_plan -v
```

`flash-teacher-mock` is a standalone CPU executable with a state-sensitive synthetic
engine. It exercises the real JSON protocol and prefix/suffix orchestration; its
scores are not language-model predictions. It deliberately includes a token-boundary
merge to verify full-prompt fallback. For a portable mock-only build, set
`-DFLASH_BUILD_RUNTIME=OFF`; no Flash/Metal library is linked into that executable.

`flash-teacher` defaults to **vocabulary-only CPU mode**. It sets `vocab_only`, zero
GPU layers and no devices, does not register inference backends, and never constructs
Flash Metal/model objects. CPU tests use the real Flash GGUF vocabulary and compare
sample token IDs to the official Qwen3.8 tokenizer used with Flash's validated chat
template. The subsequently authorized GPU checks are recorded below; CPU tests alone
do not establish numerical parity or teacher quality.

## Prepared pilot

The existing `sweep/out/flash-teacher-pilot/` contains 250 examples, ten per source,
deterministically selected from the existing dev split without consulting predictions.
CPU preparation verified every option label and rendered prompt. The longest prompt
has1422 tokens; average424.04. Only the vocabulary was loaded (about94MiB peak worker
RSS); the model weights were not loaded. These dev examples have been used in earlier
experiments and are not pristine final confirmation.

Reproduce CPU preparation:

```sh
.venv-train/bin/python -m sweep.flash_teacher_capture prepare \
  --worker target/flash-teacher/flash-teacher \
  --model /Users/ZKeown/models/flash/qwen38-flash-next-flash-v1.gguf \
  --template /Users/ZKeown/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/chat_template.jinja \
  --records sweep/out/accuracy-full/data/dev.records.jsonl \
  --wire sweep/out/accuracy-full/data/dev.wire.jsonl \
  --out sweep/out/flash-teacher-pilot --per-source 10
```

Original gold labels/soft targets remain attached to records. `plan.json` freezes
input, template, worker and selected-data hashes. Weight path/size/mtime identify the
local candidate, not a cryptographic weight checksum; authoritative comparisons
should also pin the model's independently verified checkpoint hash. Changes require
new preparation/output rather than silently reusing predictions.

## GPU capture

Only run after the user authorizes GPU use and competing workloads have room. This
worker loads its own large model instance; it does not connect to an existing Flash
process. It must not run alongside the user's resident Flash instance by assumption.
The explicit flag is intentional:

```sh
.venv-train/bin/python -m sweep.flash_teacher_capture capture \
  --plan sweep/out/flash-teacher-pilot --allow-gpu
.venv-train/bin/python -m sweep.flash_teacher_capture report \
  --plan sweep/out/flash-teacher-pilot \
  --student sweep/out/accuracy-full/baseline-s0-dev.jsonl
```

Capture returns raw option-token logits; no sampling or reasoning generation. Prefix
snapshots have one retained slot. Every suffix is tokenized jointly with the prefix;
if BPE changes that boundary, the worker evaluates the full prompt. Requested label
order is preserved, duplicate/multitoken labels and overlong/empty prompts rejected.
Errors invalidate cached state. The worker's stdout is JSONL only; diagnostics use
stderr. A memory-pressure watchdog is mandatory in GPU mode and exits only its own
process if pressure stops being normal or cannot be read. It never touches other jobs.

The Python adapter serializes requests, checks protocol IDs/shapes/finiteness, bounds
read time, and closes only its own subprocess. CPU mock outputs are marked synthetic;
the report refuses to treat them as teacher evidence. Capture preserves hard gold
separately from soft labels, detects changed prepared examples, writes a partial
artifact until completion, and reports load/capture times separately.

First measure teacher accuracy by source/kind versus the matched2B decisions. Fit a
Flash-specific calibrator on separate gold calibration labels before judging its
probabilities or using soft targets. Keep certification gold independent of teacher
outputs. A larger parameter count and deterministic decoding do not establish teacher
quality; this pilot gates additional label generation/distillation work.

## Protocol v1

The first stdout line advertises `protocol`, `mode`, `synthetic`, `gpu_loaded`, context
and (real vocabulary) model/architecture/vocab size. Each request is one JSON object
with `id` and `op`. Responses echo the ID and contain `ok` plus `result` or `error`.
Operations: `info`, `tokenize` (`text`, `special`), `reset`, `quit`, and `label_logits`
(`prefix`, `suffixes`, per-row `labels`, optional `use_cache`). Label results include
logits, label token IDs, prompt token counts and prefix/fallback/forwarded-token stats.
EOF releases the owned worker. No public network service is opened.


## Measured GPU pilot, 2026-10-22

User-authorized real-model parity passed with exactly matching observed cached/full,
replayed and reordered-label logits on six decisions (including the1422-token longest
pilot prompt) and a two-question shared-prefix batch. See
`sweep/out/flash-teacher-pilot/gpu-parity.json` and `run-parity.py` for scope.

The250-example capture completed in189.56s after6.54s model load. Flash raw accuracy
was74.8% versus the trained2B's75.6%; paired accuracy delta -0.8pp,95%CI
[-6.51,+4.44]pp. Raw Brier worsened0.29519 ->0.32951. This small previously used dev
subset does not establish a useful overall teacher advantage; no distillation or
training was started. `comparison.json` and `paired-analysis.json` retain the matched
results and per-source/kind breakdown. The worker exited and released its allocation.


## Response-format investigation

The follow-up `sweep/out/flash-prompt-investigation/` reproduced all250 original control
probabilities exactly. Removing assistant `Answer:` yielded75.6% accuracy, tying the
trained2B; Brier0.32235 remained worse than2B0.29519. Bare letters achieved74.78% on
230 eligible examples, versus2B74.35% on those same230. The format changes did not
establish a reliable overall advantage. Full paired intervals and per-kind/source
metrics are in `comparison.json`; INVESTIGATION.md explains the bounded conclusions.

`score(..., label_prefix="")` explicitly reads bare tokens; default `" "` is unchanged.
The worker advertises `label_prefixes` in its handshake. Python rejects unsupported
prefix requests to prevent older workers silently scoring the wrong token IDs.
Bare multi-token labels must never be approximated by a first-token logit.

The original long worker exited its own process on memory pressure after243 complete
comparisons. Validated those rows and finished only7 in a fresh worker; all results
are retained. The final diagnostic tool rotates workers after100 rows with no overlap;
this change is CPU tested and retains the mandatory pressure guard. For future runs,
build a fresh directory so earlier frozen worker hashes remain reproducible:

```sh
cmake -S sweep/flash_teacher -B target/flash-teacher-current -DCMAKE_BUILD_TYPE=Release
cmake --build target/flash-teacher-current -j2
.venv-train/bin/python -m sweep.flash_prompt_investigation \
  --out sweep/out/flash-prompt-investigation-new --allow-gpu
```

The frozen source/binary and one-off validated recovery script remain with the original
experiment. The diagnostic refuses an existing plan rather than overwrite predictions.
It checks GPU opt-in, source identities and token continuation boundaries before load.
No automatic interrupted-run resume is implemented. All examples retain original gold;
no training, distillation or model promotion is performed by this command.


## Bounded generated-answer diagnostic

The native worker now advertises `generation: greedy-v1`. `generate` accepts a full
rendered prompt and max_new_tokens1..2048, validates prompt+budget against context,
clears any scoring snapshot, and autoregressively selects the full vocabulary's
largest next-token logit until EOS or the budget. This uses actual trunk decoding,
without sampling or speculative MTP. It clears continuation state afterward. Raw
piece bytes are transported as an integer array so truncation inside a UTF-8 code
point cannot corrupt JSON; Python decodes with replacement. Responses retain token
IDs, token counts and EOS/length finish reason. CPU vocabulary mode refuses generation.

Python `render_chat(system, user, thinking=True, reasoning_effort="low")` uses the
official thinking-enabled chat template. Existing `template()` and `label_logits`
defaults are unchanged. Record reasoning/template settings at the call site; generic
backend provenance's thinking flag describes its default scoring template.

`sweep.flash_reasoning_pilot` freezes32 state-disjoint existing-dev cases:24 source-
balanced student/Flash prediction disagreements and8 agreement controls. Selection
never consults gold correctness. Original hard gold/soft targets are retained outside
prompts. Generated-direct (thinking off,128 tokens) and low-effort thinking (1024)
share the same task instructions; final output must be exactly FINAL: <option label>.
Thinking outputs require a nonempty completed thinking block. Missing, malformed,
unknown or truncated answers count incorrect. Generated multi-letter labels are parsed
as complete strings, avoiding any first-token approximation.

```sh
cmake -S sweep/flash_teacher -B target/flash-teacher-reasoning -DCMAKE_BUILD_TYPE=Release
cmake --build target/flash-teacher-reasoning -j2
.venv-train/bin/python -m sweep.flash_reasoning_pilot prepare
# Requires user-authorized GPU availability:
.venv-train/bin/python -u -m sweep.flash_reasoning_pilot capture --allow-gpu
.venv-train/bin/python -m sweep.flash_reasoning_pilot report
```

The plan pins inputs/code/worker/template and model file size/mtime, not an independent
full weight checksum. Four smoke cases gate only format completion and projected
runtime (20min maximum), never accuracy. Workers rotate every four cases without
overlap; memory watchdog stays enabled. Every completed arm persists immediately.
A failed/interrupted capture can reuse only matching-provenance saved arms; do not
bypass a recorded smoke/runtime gate without deliberately approving a new experiment.
No labels are exported for training or certification. These selected reused dev cases
cannot estimate overall accuracy or certify any deployment.


Measured reasoning pilot: all32 cases completed,277.93s total. Trained2B, Flash direct
logits and reasoning each got19/32 correct. Reasoning corrected1 Flash error and lost
1 Flash hit. All32 reasoning finals were valid; the generated-direct control got16/32
correct with26 valid finals (5 length failures,1 format failure). Reasoning mean5.84s
per decision. This deliberately disagreement-enriched development cohort cannot
estimate deployment accuracy and establishes no net stronger-teacher advantage.
Results and provenance are in sweep/out/flash-reasoning-pilot/. No training or
calibration followed.21 CPU tests pass; workers exited and GPU allocation released.
