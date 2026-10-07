# Results

Measurements of the SDK's own runtime, newest section last. Unless a section says otherwise:
Apple M5 Max (6 performance and 12 efficiency cores, 128 GB), macOS 27, llama.cpp b11349
built by `lars-llama-sys` (static, Metal on, `GGML_NATIVE=OFF`), and the dev model
Qwen3-4B-Instruct-2507 Q4_K_M (2,497,281,120 bytes) with ChatML. That template is exactly its
chat template with thinking off (`fixtures/data/templates.json`).

## 2026-10-02: llama.cpp backend parity, determinism and invariance

**Parity with the reference.** `lars-dev parity` against
`fixtures/data/llama-qwen3-4b-instruct-2507-q4_k_m.json`: 240 rows recorded by the reference
engine's own llama.cpp backend (lars `ab59a0b`, llama-cpp-python 0.3.36, Metal). The rows are
Molar Triage plus every single-row question of the render fixtures.

| SDK setting | same top label | largest logit difference | mean of per-row largest | largest probability difference |
|---|---|---|---|---|
| flash attention off | 240 / 240 | **0.0** | 0.0 | 0.0 |
| flash attention auto (on, on Metal) | 240 / 240 | 0.223 | 0.026 | 0.010 |

With flash attention off, the Rust backend is bit-identical to the reference. llama-cpp-python
runs with flash attention off by default, and its bundled llama.cpp computes this model the
same way as b11349. `cargo test -p lars-llama` checks this whenever `LARS_TEST_GGUF` names the
model.

**What moves logits.** `lars-dev capture` and `capture-requests` over 444 rows: Molar Triage,
plus 12 random validation rows from each of the 22 jev-bench configs. The jev-bench rows are
kept out of the repository. Each change is compared against the default: one row at a time,
no prefix reuse, flash attention auto.

| change | rows identical | same top label | largest logit difference | largest probability difference |
|---|---|---|---|---|
| the same run again | 444 / 444 | 444 | 0 | 0 |
| flash attention on, explicitly (auto picks on) | 444 / 444 | 444 | 0 | 0 |
| flash attention off | 0 / 444 | 444 | 0.198 | **0.035** |
| reuse the previous call's prefix | 442 / 444 | 444 | 0.020 | 0.00002 |
| batch suffixes (498 rows from 60 multi-question requests) | 66 / 498 | 498 | 0.095 | **0.009** |

- **Flash attention is part of the model's identity.** Switching it moves probabilities by
  up to 0.035, more than any other setting. A pack has to fix it, and its calibrator has to
  be fitted with the same setting.
- **Batching suffixes is not invariant.** A row's logits depend on what it was decoded with.
  It cut the 60 Molar Triage requests (permutations and evidence on) from 29.6 s to 20.5 s.
  It stays opt-in, and one row at a time stays the default.
- **Prefix reuse is close to free.** It changed 2 rows out of 444, by at most 0.00002 in
  probability. It's still off by default until the latency numbers show it's worth having.

**CPU.** 164 rows: Molar Triage's first 120 rows, plus every sixth jev-bench row of the sample.

| comparison | rows identical | same top label | largest probability difference | 90th / 99th percentile |
|---|---|---|---|---|
| CPU at 6 threads against 12 and against 18 | 164 / 164 | 164 | 0 | 0 / 0 |
| CPU against Metal, flash attention off on both | 0 / 164 | 163 | 0.246 | 0.029 / 0.168 |
| CPU with flash attention auto against off | 0 / 164 | 162 | 0.541 | 0.029 / 0.289 |

- **Thread count doesn't change CPU logits.**
- **CPU and Metal are different models, as far as a calibrator is concerned.** Half the rows
  agree to within 0.00005 in probability: they are peaked enough that small logit noise
  doesn't show. The large differences are rows where two options' logits are close:
  - a 5-level score row: logits 0.6 apart, probability moved 0.25
  - a 28-option choice row: logits 1.6 apart
  - one top label flipped

  This is quantization noise on two kernel paths. On CPU, ggml quantizes activations for its
  4-bit dot products; Metal does not.
- **A pack calibrator must be fitted on the compute path it serves,** or the transfer between
  paths must be measured on accuracy and Brier and shown not to matter. Flash attention is
  part of that path on both CPU and Metal. Phase 2b settles which: it evaluates both paths on
  jev-bench, including a calibrator fitted on one and served on the other.
- Time per row (single-question rows) on the portable build:

| configuration | ms per row |
|---|---|
| Metal | 111 |
| CPU, 6 threads | 1,719 |
| CPU, 12 threads | 1,061 |
| CPU, 18 threads | 1,072 |

  The efficiency cores help up to 12 threads.

**Build flags.** A `GGML_NATIVE=ON` build changes CPU logits against the portable build: 0 of
164 rows identical, 163 with the same top label, probabilities up to 0.231 apart. It is no
faster: 508 ms against 474 ms for one noul at 12 threads. Packages ship the portable build,
and build flags count as part of the compute path.

## 2026-10-02: latency of the dev model

`bench/run.sh`, p50 over 20 repeats on Metal and 5 on CPU, after one untimed call.
VM at about 20% of one core in the background. Requests are in `bench/`:
- one noul about a 15-word message
- Molar Triage: noul, 6-option choice and 5-level score, with explain and abstain on
- twenty nouls about a 70-word message

| configuration | one noul | Molar Triage | twenty questions | peak RSS | load |
|---|---|---|---|---|---|
| Metal, flash off | 91 ms | 179 ms | 843 ms | 3.7 GB | 0.26 s |
| Metal, flash on | 82 ms | 177 ms | 834 ms | 3.7 GB | 0.25 s |
| Metal, flash on, batched suffixes | 81 ms | 128 ms | **230 ms** | 3.7 GB | 0.25 s |
| Metal, flash off, prefix reuse | **38 ms** | 135 ms | 783 ms | 3.7 GB | 0.26 s |
| CPU 6 threads, flash off | 826 ms | 2,458 ms | 5,470 ms | 6.1 GB | 1.4 s |
| CPU 12 threads, flash off | 474 ms | 1,414 ms | 3,158 ms | 6.1 GB | 1.4 s |
| CPU 12 threads, flash off, prefix reuse | 127 ms | 1,011 ms | 2,596 ms | 6.1 GB | 1.4 s |

- The first timed call is within 5% of p50 everywhere, so there is no warm-up beyond load.
- **Prefix reuse here repeats the same request,** so the whole state is reused. That is the
  `isPissed(user)` case asked of one user more than once, or several predicates over one
  state. A new state reuses only the constant head.
- **Batching is what makes many questions cheap:** twenty questions take 230 ms instead of
  834 ms. Its cost is that a row's probability can move by up to 0.009 with what it was asked
  alongside. Identical requests still give identical answers.
- **The 4B on CPU is too slow for an inline `if`:** about 0.5 s for one noul on a novel
  state. CPU latency is what the Phase 2b size sweep has to buy back.
- **CPU memory is 6.1 GB for a 2.5 GB file.** llama.cpp repacks Q4_K weights for its CPU
  kernels on top of the memory-mapped original, and loading takes 1.4 s against 0.26 s on
  Metal. Worth checking with repacking off (`use_extra_bufts`) in Phase 2b.

New defaults in `LlamaOptions`, from these numbers:
- **flash attention off:** parity with the reference, and CPU and Metal agree more closely
- **prefix reuse on:** parity unchanged, bit for bit
- **batching off,** pending the Phase 2b transfer check

`bench/run.sh` reproduces the table with the new flag names.

## 2026-10-02: Phase 2b, zero-shot model sizes in the shipping runtime (first results)

**Setup:**
- **Models:** GGUFs built from the original weights with the pinned llama.cpp's
  `convert_hf_to_gguf.py` and `llama-quantize` (`sweep/prepare.sh`).
- **Data:** jev-bench, 22 configs, 4,120 validation and 4,400 test rows. Raw logits are
  captured by `sweep/run.sh` and analyzed on CPU by `sweep/analyze.py`, which uses the
  reference engine's own calibration functions and the corrected soft-target decoding.
- **Machine:** busy with the sweep's CPU queue throughout, so latency here is indicative only.

| model (Metal, layout) | raw acc | per-config calibrated | pooled calibrated | pooled Brier | coverage at 5%, empirical (test error) | coverage at 5%, Wilson (test error) |
|---|---|---|---|---|---|---|
| Qwen3-0.6B Q4_K_M, v1 | 0.338 | 0.403 | 0.399 | 0.629 | 8.2% (9.9%) | 0% |
| Qwen3-0.6B Q4_K_M, v2 | 0.357 | 0.366 | 0.358 | 0.647 | 3.8% (10.1%) | 0% |
| Qwen3-0.6B Q8_0, v1 | 0.385 | 0.418 | 0.409 | 0.617 | 8.5% (8.9%) | 6.6% (4.1%) |
| Qwen3-1.7B Q4_K_M, v1 | 0.408 | 0.459 | 0.423 | 0.629 | 8.9% (11.0%) | 2.9% (1.6%) |
| Qwen3-1.7B Q4_K_M, v2 | 0.351 | 0.408 | 0.367 | 0.660 | 6.8% (6.7%) | 0% |
| Qwen3-4B-Instruct-2507 Q4_K_M, v1 | 0.655 | 0.673 | 0.655 | 0.434 | 26.0% (7.3%) | 7.7% (4.7%) |

- **The 4B through the SDK reproduces the prototype:** 0.673 per-config calibrated, against
  0.662 for the same model zero-shot on MLX (lars RESULTS, 2026-09-22).
- **The small models are far behind zero-shot, and it's the models, not the pipeline.**
  Three checks:
  - llama.cpp's tokenization of the prompts is identical to the original tokenizer's.
  - The reference engine's MLX backend on the original BF16 weights gives the SDK's numbers
    on validation:

    | config | reference MLX, BF16 | SDK, Q4_K_M |
    |---|---|---|
    | boolq | 0.655 | 0.655 |
    | mmlu | 0.415 | 0.380 |
    | arc_challenge | 0.485 | 0.465 |
    | paws | 0.435 | 0.435 |

  - The empty think block in the hybrid template doesn't matter. On BF16, with and without
    it: boolq 0.655 against 0.660, mmlu 0.430 against 0.430, paws 0.435 against 0.460.

  The small models put 99–100% of their probability on the label tokens and answer with
  strong position biases: the 0.6B says "yes" to every boolq and paws row. Part of the gap
  to the 4B is post-training: the 4B is the newer 2507 Instruct release, the small models
  are the original hybrid Qwen3.
- **Q4_K_M costs the 0.6B** 1.0 to 4.7 points against Q8_0, depending on the measure.
- **Question-first (v2) is worse zero-shot:** −3.7 points per-config calibrated on the
  0.6B and −5.1 on the 1.7B. The 4B is pending; v1 stays unless it disagrees.
- **A threshold chosen on validation doesn't hold its error budget on test:** the empirical
  5% threshold gives 7–11% error. A threshold that requires the Wilson 95% upper bound to
  be at most 5% holds the budget (1.6–4.7% on test), at a fraction of the coverage.

**Few labels per decision** (per config, n validation rows, 10 draws, against the pooled
calibrator):

| model | n | Brier change | accuracy change | coverage, empirical (test error) | coverage, Wilson (test error) |
|---|---|---|---|---|---|
| 1.7B Q4_K_M | 25 | −0.044 | +3.2 pts | 19.3% (19.2%) | 0% |
| 1.7B Q4_K_M | 100 | −0.052 | +3.7 pts | 13.9% (17.1%) | 1.0% (2.1%) |
| 4B Q4_K_M | 25 | −0.038 | +1.6 pts | 35.4% (13.1%) | 0% |
| 4B Q4_K_M | 100 | −0.046 | +1.9 pts | 30.5% (10.5%) | 2.7% (4.0%) |
| 4B Q4_K_M | 200 | −0.039 | +1.6 pts | 29.7% (11.8%) | 9.2% (5.4%) |

- **Calibrating on a few dozen of your own labels helps every model** more than the pooled
  calibrator does: Brier improves by 0.04–0.05, and Platt moves binary decisions.
- **Those labels cannot certify an error budget.** A threshold whose error on the labels is
  at most 5% runs at 10–20% on test. Certifying 5% at 95% confidence takes 73
  accepted rows with no errors, so below about 100 labels the conservative policy accepts
  nothing.
- **What this means for the SDK:** `calibrate(examples, { maxError })` uses the
  conservative bound, and reports "not enough labels to certify" rather than an optimistic
  threshold.

**Latency of the 0.6B Q4_K_M on Metal,** with the CPU queue running alongside:

| request | p50 |
|---|---|
| one noul, prefix reuse | 12 ms |
| one noul, no prefix reuse | 24 ms |
| Molar Triage | 45 ms |
| twenty questions | 297 ms |
| twenty questions, batched | 65 ms |

Peak RSS is 1.5 GB. Its CPU numbers that day are unusable: two 12-thread jobs on 18 cores.

## 2026-10-02: the prototype's 4B LoRA survives conversion to GGUF

`sweep/train.sh` fused the prototype's corpus-E 4B adapter (`lars/checkpoints/lora-4b-e-s0`,
trained on the 4-bit MLX base) into the dequantized base. It then converted the result to
GGUF and quantized it with the pinned llama.cpp, and captured it through the SDK on Metal,
layout v1. The fuse-and-convert path is lossless: an identity adapter fused and converted the
same way reproduces the base model's logits bit for bit on 60 rows.

| 4B corpus-E LoRA | pooled acc | pooled Brier | pooled ECE | coverage at 5%, empirical (test error) | coverage at 5%, Wilson (test error) |
|---|---|---|---|---|---|
| prototype on MLX, as served (lars RESULTS, 2026-09-30) | 0.741–0.742 | 0.305 | 0.074–0.076 | | |
| SDK, GGUF Q8_0 | **0.743** | **0.294** | 0.079 | 53.1% (6.4%) | 23.0% (2.4%) |
| SDK, GGUF Q4_K_M (2.5 GB) | **0.736** | 0.304 | 0.081 | 48.8% (5.5%) | 17.5% (2.5%) |
| SDK, zero-shot Q4_K_M | 0.655 | 0.434 | 0.147 | 26.0% (7.3%) | 7.7% (4.7%) |

- **Q8_0 reproduces the published 4B tier,** and Q4_K_M keeps all but 0.7 points of it. On
  MLX, 4-bit re-quantization dropped the same adapter from 0.740 to 0.703. Q4_K_M keeps
  some tensors at higher precision, and that is enough.
- **The adapter is worth 8.8 points over zero-shot,** and more than doubles conservative
  coverage, from 7.7% to 17.5–23% at about 2.5% error.
- **Where it gains:** clinc150 0.66 to 0.92, measuring_hate_speech 0.49 to 0.81,
  civil_comments 0.67 to 0.94.
- **Where it loses:** stsb 0.41 to 0.30, sst5 0.50 to 0.39, helpsteer2_helpfulness 0.37 to
  0.29, the prototype's known weakness on score tasks.

## 2026-10-05: Phase 2b complete, small-model LoRAs and CPU/Metal transfer

Everything in `HANDOFF.md` (2026-10-02) finished unattended, with no failures.
`sweep/out/summary.md` has every row; the headline numbers are below. All are pooled
calibrated, jev-bench test, layout v1, Metal unless a row says CPU.

**LoRAs.** `sweep/train.sh` trained the 1.7B in 2 h 35 min and the 0.6B in 1 h 35 min on
Metal: BF16 base, repaired corpus E, the 4B recipe. Both beat their untrained model at
selection. Each was then fused, converted and quantized with the pinned llama.cpp.

| model | zero-shot Q8_0 | LoRA Q8_0 | LoRA Q4_K_M | LoRA Brier (Q8_0) | LoRA coverage at 5%, Wilson (test error) | file (Q4_K_M) |
|---|---|---|---|---|---|---|
| Qwen3-0.6B | 0.409 | **0.651** | 0.646 | 0.411 | 6.6% (3.8%) | 0.5 GB |
| Qwen3-1.7B | 0.469 | **0.683** | 0.672 | 0.356 | 10.5% (3.7%) | 1.3 GB |
| Qwen3-4B-Instruct-2507 | 0.657 | **0.743** | 0.736 | 0.294 | 23.0% (2.4%) | 2.5 GB |

- **The LoRA lifts the small models more than the 4B:** +24 points for the 0.6B, +21 for
  the 1.7B, +9 for the 4B. The adapted 0.6B and 1.7B match or beat the zero-shot 4B on
  accuracy, with better Brier.
- **The gap to the 4B LoRA is knowledge and fine-grained scores; routing and triage nearly
  close.** Per config, 0.6B / 1.7B / 4B LoRA:
  - nearly closed: banking77 0.74 / 0.74 / 0.78, clinc150 0.80 / 0.89 / 0.91, massive
    0.80 / 0.83 / 0.84, civil_comments 0.92 / 0.90 / 0.90, sms_spam 0.96 / 0.97 / 0.96
  - still open: mmlu 0.42 / 0.53 / 0.70, arc_challenge 0.66 / 0.80 / 0.90, sst5
    0.36 / 0.31 / 0.46
- **Q4_K_M costs 0.5–1.1 points with a LoRA.**
- **Few labels on an adapted model:** 100–200 labels a decision barely move Brier (−0.004 to
  −0.013). The adapter already fixed what a per-decision Platt fit fixed on zero-shot
  models. They still set a usable Wilson threshold: at 200 labels, 7–27% coverage at 2.4–4%
  test error.

**One calibrator per pack works across CPU and Metal.** A pooled calibrator fitted on one
path and served on the other changes macro accuracy and Brier by at most 0.0004 at every
size and quantization. Example: 4B Q4_K_M, fitted on Metal and served on CPU, Brier 0.4472
against 0.4475. Coverage at the empirical threshold moves by up to 2 points, which is within
that metric's noise. The row-level differences seen in 2026-10-02 (up to 0.25 in probability
on near-tied rows) average out. So a pack ships one calibrator, and batching, whose drift
was 0.009, can be on by default.

**Question-first (v2) is dropped.** It loses at every size, the 4B included: 0.662 → 0.617
per-config calibrated at Q4_K_M, and 0.673 → 0.625 at Q8_0.

**Importance-matrix quantization recovers Q4_K_M.** The dev file in lars/models is
Unsloth's Q4_K_M with an imatrix. On the zero-shot 4B (pooled accuracy / Brier):

| quantization | pooled acc | Brier |
|---|---|---|
| Unsloth Q4_K_M, imatrix | 0.655 | 0.434 |
| our plain Q4_K_M | 0.641 | 0.446 |
| our Q8_0 | 0.657 | 0.429 |

An imatrix computed on training-split prompts should make LoRA Q4_K_M packs as good as
Q8_0 at about 60% of the size.

**CPU time, indicative only:** the CPU Q8_0 captures ran Saturday 04:02–11:05, after every
other job had finished. Median per jev-bench file at 12 threads, per row:

| model | ms per row |
|---|---|
| 0.6B | 139 |
| 1.7B | 345 |
| 4B | 860 |

jev-bench rows carry long passages and up to 151 options, heavier than a one-line predicate.
The benches taken during the sweep were contended and are not reported. Quiet-machine
latency is still to do.

## 2026-10-05: the first pack, evaluated through the SDK

`lars-dev pack` built `lars-4b-dev@0.0.1` from the 4B corpus-E LoRA at Q4_K_M:
- **Calibrator:** fitted pooled over all 4,120 jev-bench validation rows, through the Rust core
  with the strict soft-target decoding. 12 minutes on a busy machine.
- **Evaluation:** run on the 4,400 test rows with that calibrator.

| | macro accuracy | macro Brier |
|---|---|---|
| the pack, evaluated by the core | 0.7361 | 0.3042 |
| `sweep/analyze.py` on the same model's captures (Python, reference calibration functions) | 0.7361 | 0.3042 |

Per-config accuracy is identical on all 22 configs. Two independent implementations of
calibration fitting and soft-target decoding agree end to end.

## 2026-10-05: the three dev packs, and Molar Triage through both SDKs

`lars-dev pack` built `lars-1.7b-dev` and `lars-0.6b-dev` (LoRA, Q4_K_M) like the 4B: pooled
calibrator fitted and test evaluated through the core. Both match `sweep/analyze.py` exactly.

| pack | file | jev-bench macro accuracy | Brier |
|---|---|---|---|
| lars-4b-dev | 2.5 GB | 0.7361 | 0.3042 |
| lars-1.7b-dev | 1.3 GB | 0.6720 | 0.3701 |
| lars-0.6b-dev | 0.5 GB | 0.6461 | 0.4213 |

Molar Triage (`examples/molar_triage`, fictional, which no pack trained on), run through the C#
example. The TypeScript example gives the same answers on the 4B. Out of 30 test messages:

| pack | in pain | department | urgency (exact level) | knocked-out tooth |
|---|---|---|---|---|
| 4B | 28 | 28 | 14 | "call the patient now" |
| 1.7B | 27 | 28 | 14 | queued for restorative |
| 0.6B | 29 | 24 | 11 | "call the patient now" |

- **Exact urgency level is weak at every size:** 11–14 of 30. Score tasks are the known gap.
- **Calibrating on 30 labels is not always an improvement.** On the 1.7B it took "in pain"
  from 27 to 24 of 30: a Platt fit on 30 rows moved the yes/no boundary the wrong way. On
  the other packs and predicates it changed nothing.
- **What this means for `calibrate`:** a per-predicate calibrator fitted on a handful of
  labels should not silently replace the pack's.

**Fixed the same day.** `calibrate` now cross-validates the fit against the pack's calibrator
on the same labels (5-fold, by log loss). It keeps the fit only when the 95% upper bound on the
paired per-example log-loss difference is below zero, and the result says which one it kept.

Choosing on the cross-validated mean alone was not enough. On the 1.7B it kept a fit that was
worse on test: 27 → 24 of 30, log loss 0.342 → 0.366. With the significance rule, Molar
Triage's "in pain" on 30 labels:

| pack | calibrator kept | test before | test after |
|---|---|---|---|
| 4B | pack | 28/30, log loss 0.166 | 28/30, 0.166 |
| 1.7B | pack | 27/30, 0.342 | 27/30, 0.342 |
| 0.6B | pack | 29/30, 0.169 | 29/30, 0.169 |

A runtime test checks the other direction: 200 labels that contradict the model replace the
pack's calibrator.

## 2026-10-05: packages, installed from scratch on five platforms

`package.yml` (run 37328585896) built the native library and the Node addon:
- `osx-arm64`, and `osx-x64` cross-compiled
- `linux-x64` and `linux-arm64`, in manylinux_2_28 containers, needing GLIBC_2.28 at most
- `win-x64`

It packed `@moedex/lars` with five optional platform packages, and `Moedex.Lars` with five
`runtimes/<rid>/native` libraries. 44 MB in all.

Each runner then installed the packages into an empty project, with nothing from the repository
and no `LARS_NATIVE`, and answered a noul and a choice on a mock pack:

| runner | npm (Node 22) | NuGet (.NET 10) |
|---|---|---|
| macOS arm64 | ok | ok |
| macOS Intel (`macos-15-intel`) | ok | ok |
| Linux x64 | ok | ok |
| Linux arm64 | ok | ok |
| Windows x64 | ok | ok |

Locally, the `osx-x64` C library also ran the C smoke test under Rosetta on the real 0.6B pack.

## 2026-10-05: better accuracy at the same memory, or a bigger model in it?

jev-bench test, pooled calibrator, Metal (sweep/out/summary.md). Coverage is Wilson's at a 5%
error budget, with the error on accepted test answers in brackets.

| model | file | pooled acc | Brier | Wilson coverage |
|---|---|---|---|---|
| 4B pack (LoRA, Q4_K_M) | 2.5 GB | 0.736 | 0.304 | 17.5% (2.5%) |
| 4B pack, imatrix Q4_K_M | 2.5 GB | 0.735 | 0.301 | 19.7% (2.2%) |
| 1.7B pack, imatrix | 1.1 GB | 0.677 (from 0.672) | 0.362 | 11.6% |
| 0.6B pack, imatrix | 0.4 GB | 0.650 (from 0.646) | 0.419 | 6.4% |
| Qwen3-30B-A3B + LoRA, all 128 experts, Q4_K_M | 18.6 GB | 0.761 | 0.277 | 21.1% (1.5%) |
| same, 64 most-used experts per layer | 9.8 GB | 0.712 | 0.340 | 14.6% |
| Qwen3.5-9B zero-shot, Q3_K_M | 4.7 GB | 0.685 | 0.376 | 3.3% |
| Qwen3.5-9B zero-shot, Q4_K_M | 5.8 GB | 0.693 | 0.366 | 6.9% |
| Qwen3.5-2B zero-shot, Q4_K_M | 1.3 GB | 0.493 | 0.548 | 4.0% |
| (Qwen3-4B zero-shot, Q4_K_M) | 2.5 GB | 0.641 | 0.446 | 7.7% |

**An importance matrix is a small, free gain.** Computed on 1,500 training-split prompts and
rendered with each model's template, it gives +0.4 to +0.5 points at 0.6B and 1.7B. At 4B,
accuracy is flat, Brier improves slightly, and coverage gains 2 points. The file size is
unchanged. Packs should be rebuilt from these files.

**The 30B is a larger tier, not a 4B replacement.**
- The prototype's 30B LoRA, fused and converted, matches its reference (0.761 against 0.765)
  and is the best model measured: Brier 0.277, and 21% of answers certified at 1.5% error.
- Its gains over the 4B are knowledge: strategyqa_closed +11.5, mmlu +10, arc +5.5.
  Classification configs barely move.
- It needs about 20 GB resident. With mmap, resident size equals the file size, so there is
  no saving there.

**Pruning experts by routing frequency loses exactly that knowledge.** We kept the 64
experts per layer that carry 87.5% of the routing weight on LARS prompts. The model lost 4.9
points: arc −21.5, mmlu −13, hate speech −7. Frequency is the wrong criterion. REAP-style
saliency, which weighs gate value by output norm, would be the next try, but the target
(9.8 GB) is already four 4B packs.

**Qwen3.5 zero-shot:**
- The 9B is 4.4 points better than the 4B base at about twice the file size. It is still 5
  points under the LoRA'd 4B, and its probabilities certify poorly.
- The 2B beats Qwen3-1.7B zero-shot by 7 points.
- Both are candidates for LoRA training, not replacements as they stand. At Q4_K_M the 9B
  gains another 0.8 points.
- Training them here is slow:
  - mlx_lm's Metal kernel for the gated delta rule has no gradient. In training mode it falls
    back to a per-token loop that keeps a state per token per layer, and the process was
    OOM-killed at one 4,096-token batch.
  - lars `f42781a` adds a chunked, differentiable version: forward substitution within 64-token
    chunks, one state per chunk. It matches the loop to 6e-7 on CPU, gradients included.
  - A power-series inverse was tried first and is numerically unstable: NaN on correlated keys.
  - With it, the 2B trains at 6.75 s per step (61 GB peak), against 2.1 s for Qwen3-1.7B:
    about 10 hours per epoch. The 9B takes 23 s per step at 2,048-token batches (89 GB peak):
    about 73 hours. So the overnight bigger-model run is Qwen3-8B (dense, 6.5 s per step,
    about 9 hours).
- Qwen3.5 uses hybrid recurrent layers, which llama.cpp cannot truncate, so the backend
  re-prefills the prefix instead (`lars-llama`). Prefix reuse on vs off gives identical logits
  on 126 rows. Batched vs sequential differs by at most 0.0009 in probability
  (sweep/out/qwen35-checks.txt).

## 2026-10-05: typed-decisions transfer, and per-predicate heads on hidden states

[LocalLLaMA/typed-decisions](https://huggingface.co/datasets/LocalLLaMA/typed-decisions)
(Apache-2.0):
- 4 business workflows (agent traces, customer service, invoices, security alerts), 5 typed
  questions each, 400 test and 1,200 train cases.
- Gold is the mean of three temperature-0.7 samples from an unnamed ~4B-class teacher. The
  teacher is independent of TypeSafe and not Jev, so the data is usable under the lars policy.
- `sweep/typed_decisions.py --convert` writes one eval row per question, with the gold
  distribution as `soft_label`. The same script scores the captures.
- Our KL (gold ‖ model) and Brier (summed over options) reproduce the published Uniform row
  exactly (0.444 / 0.238).

**Zero-shot, with each pack's own calibrator, one question per call:**

| | acc | KL | Brier | noul / choice / score acc |
|---|---|---|---|---|
| Prior (ignores input) | 0.481 | 0.327 | 0.181 | |
| lars-4b-dev | 0.607 | 0.321 | 0.167 | 0.738 / 0.617 / 0.501 |
| lars-1.7b-dev | 0.532 | 0.502 | 0.235 | |
| lars-0.6b-dev | 0.440 | 0.566 | 0.300 | |
| Jev 1.13.0 (published) | 0.727 | 1.442 | 0.148 | |
| meraGPT Decider 1 (published) | 0.768 | 0.096 | 0.052 | |

Transfer is weak:
- The packs were LoRA'd on classification benchmarks. These workflows want judgement about
  agent runs and invoices.
- Score questions (urgency, severity) are the weakest at 0.50.
- The pooled calibrator matters: the 4B's KL is 0.618 raw and 0.321 calibrated.

**A per-predicate head on the final hidden state fixes most of it.**
- `lars-dev capture --hidden` reads the final hidden state at the answer position. It is
  llama.cpp's embeddings output, switched on only for the suffix. With it on, probabilities
  move by at most 0.002, because every suffix token becomes an output.
- sweep/heads.py simulates a user who has labeled n examples of one predicate, with n drawn
  from train, 5 draws, scored on test.
- The head is `z = a·log p_pack + b + W h`:
  - h is centered and projected onto the labeled rows' span.
  - W is ridge-shrunk with λ from 5-fold CV. a and b get a unit prior.
  - λ = ∞ ("recal") is a Platt/temperature refit with a per-option bias, on top of the pack.
- Selection follows `lars calibrate`'s rule: step from pack to recal to heads with weaker
  shrinkage, only when each step beats the current choice by mean + 1.96 SE < 0 on paired
  CV log losses.
- Macro over the 20 predicates (= micro: 100 test cases each). Fits take seconds on CPU.

Hard labels (one answer per example, the usual case). Probabilities are scored against those
labels, as accuracy / log loss / Brier / ECE:

| labels per predicate | 4B recal | 4B head | 0.6B recal | 0.6B head |
|---|---|---|---|---|
| 0 (pack) | 0.607 / 0.901 / 0.512 / 0.129 | | 0.440 / 1.264 / 0.701 / 0.218 | |
| 25 | 0.665 / 0.759 / 0.438 / 0.124 | 0.664 / 0.802 / 0.457 / 0.132 | 0.555 / 0.965 / 0.554 / 0.108 | 0.567 / 1.001 / 0.562 / 0.126 |
| 50 | 0.669 / 0.734 / 0.425 / 0.112 | 0.701 / 0.697 / 0.402 / 0.111 | 0.559 / 0.935 / 0.541 / 0.098 | 0.606 / 0.891 / 0.506 / 0.104 |
| 100 | 0.672 / 0.721 / 0.421 / 0.116 | 0.724 / 0.618 / 0.366 / 0.098 | 0.565 / 0.918 / 0.535 / 0.094 | 0.659 / 0.779 / 0.451 / 0.096 |
| 300 | 0.681 / 0.711 / 0.416 / 0.100 | 0.755 / 0.537 / 0.319 / 0.084 | 0.576 / 0.907 / 0.528 / 0.090 | 0.716 / 0.657 / 0.383 / 0.078 |

Soft labels (the gold distributions; the leaderboard's metrics), head only, as acc / KL /
Brier:

| labels per predicate | 4B head | 0.6B head |
|---|---|---|
| 25 | 0.692 / 0.161 / 0.086 | 0.597 / 0.253 / 0.134 |
| 50 | 0.726 / 0.127 / 0.069 | 0.651 / 0.200 / 0.109 |
| 100 | 0.747 / 0.106 / 0.057 | 0.692 / 0.163 / 0.089 |
| 300 | 0.782 / 0.084 / 0.045 | 0.739 / 0.126 / 0.068 |
| *OpenDecider-small-td, 4B + LoRA on all of train (published)* | *0.792 / 0.080 / 0.043* | |

What this says:
- **Recal is +6 points (4B) and +12 (0.6B) from 25 labels.** It adds a per-option bias to
  the pack's output, which `calibrate` does not have today. That is why calibrate's refit
  barely moves accuracy here (0.607 → 0.62).
- **A head needs about 50 hard labels to beat recal.** At 25 the selection rule still admits
  heads that do slightly worse. From 50 it gains steadily: the 4B goes 0.607 → 0.755 at 300.
- **The head closes most of the gap between model sizes.** The 0.6B goes 0.440 → 0.716 at 300
  hard labels, and the 4B reaches 0.755. The size gap falls from 17 points to 4, at a sixth of
  the memory.
- **With soft labels, a frozen 4B plus linear heads is within a point of a 4B LoRA
  fine-tuned on the same train split.**
- **Hard-label heads are sharper than the teacher's spread.** Against the soft gold, their KL
  barely improves (0.32 → 0.30). That is a property of the gold being a sampling
  distribution, not miscalibration: against the labels they were fitted on, ECE falls from
  0.129 to 0.084.

Caveats:
- Gold is one teacher's judgement, and scores above its 0.735 self-agreement mean the head is
  learning that teacher's habits. A user's own labels are exactly that, which is the point.
- Below 50 labels, ship recal and not a head.

**Built into the SDK the same day.** `calibrate` now steps pack/fitted → recal → head:
- Implementation is in `lars-core/src/head.rs`: a Jacobi eigendecomposition for the span basis
  and L-BFGS for the fit, with no new dependencies. Heads are tried from 50 labels of one
  question shape.
- It matches sweep/heads.py to 4 decimals of test log loss on three real predicates
  (`lars-dev fit-head`).
- Hidden states are read only for predicates whose head needs them. Each row's last token is
  decoded alone with embeddings on, so no extra logits are materialized. All other rows stay
  bit-identical to the reference (240/240).

End to end through `@moedex/lars` on the 0.6B pack, calibrating on the first 100 train cases
of a predicate and answering its 100 test cases:

| predicate | labels | test accuracy | kept | calibrate | per call |
|---|---|---|---|---|---|
| customer_service/urgency (score) | 100 | 0.49 → 0.66 | head | 13 s | 121 → 117 ms |
| invoice_processing/disposition (choice) | 100 | 0.38 → 0.71 | head | 11 s | 110 → 112 ms |
| security_incidents/true_positive (noul) | 100 | 0.69 → 0.74 | fitted | 12 s | 95 → 108 ms |
| customer_service/urgency (score) | 30 | 0.49 → 0.62 | recal | 4 s | 132 → 126 ms |

Calibrate time is the model pass over the labels. Four saved calibrations take 45 KB.

**In-domain, heads add little and do no net harm.** jev-bench, with labels drawn from
validation and scored on test, hard labels. Macro over the 20 configs with fixed options, as
accuracy / log loss:

| labels | 4B pack | 0.6B pack |
|---|---|---|
| 0 | 0.734 / 0.740 | 0.653 / 1.039 |
| 25 | 0.738 / 0.745 | 0.656 / 1.032 |
| 50 | 0.737 / 0.737 | 0.664 / 1.011 |
| 100 | 0.739 / 0.725 | 0.661 / 0.996 |

The packs were trained on these tasks, so there is little left to learn. The gains come where
the pack's answer format misfits: stsb goes 0.37 → 0.43 and yelp5 0.57 → 0.65 (4B, 200
labels). One case got slightly worse: sms_spam's log loss rose from 0.118 to 0.15 on the 4B at
25–100 labels, with accuracy unchanged.

## 2026-10-06: certification, after an external review

A review from the Moe project (`~/Code/moe/docs/integrations/lars-sdk-findings.md`) found three
behavior bugs through the Rust SDK's mock backend, and questioned the statistics. All four are
fixed, in all three SDKs, with regression tests.

1. **A failed certification accepted everything.** `maxError` with too few labels gave a null
   threshold, which every binding treated like no certification at all. Now a null threshold
   accepts nothing: nouls answer `false` and choices their fallback.
2. **Saved calibrations didn't record their question.** A calibration loaded into a new handle
   could be used by a predicate of the same name asking the opposite. The file is now format 2:
   each calibrator carries its question, and a different question under that name is refused,
   whether defined before the load or after. Wording, option descriptions and option order all
   count.
3. **Loading kept stale certificates.** A replacement calibrator without a certificate kept the
   old one's threshold. Loading now replaces each predicate's calibrator and certificate
   together. The whole file is validated first, and the runtime binds every calibrator before
   any is used (`set_calibrators`, also in the C ABI and the Node addon), so a refused file
   changes nothing.
4. **The threshold scan overstated its 95%.** It tried every threshold with a Wilson bound and
   kept the best, which multiplies the chances of a lucky pass. Simulated with the true error
   exactly at a 5% budget, the share of trials that certified anyway:

   | labels | best of all thresholds | fixed sequence, as shipped |
   |---|---|---|
   | 100 | 2.5–2.6% | 3.2% |
   | 200 | 5.5–5.9% | 2.1% |
   | 500 | 9.6–10.7% | 1.4% |
   | 1,000 | 13.1–13.9% | 1.9% |

   - **Ranges:** over two confidence distributions, uniform and sharp. The fixed-sequence
     column is the sharp one.
   - **Trials:** 1,500–3,000 per cell, a standard error of about 0.4 points.
   - **Nominal rate:** 2.5% (one-sided). The certificate now comes from a fixed sequence:
   - most confident first
   - starting where the confidences themselves predict the budget is met, with a fifth to spare
   - stopping at the first threshold that fails

   A fitted calibrator is also certified on out-of-fold answers now, as recalibrations and
   heads already were.

   `lars-core` has the simulation as tests:
   - an error rate just over the budget is certified at most 3.5% of the time, at 200 and 1,000
     labels
   - a calibrated model is still certified

**Cost:** coverage at a 5% budget, jev-bench test, pooled calibrator, threshold per config on
its own validation rows (`sweep/analyze.py`, policy `fixed`):

| model | best of all (accepted error) | fixed sequence (accepted error, 95% upper) |
|---|---|---|
| 0.6B pack | 4.5% (4.5%) | 4.5% (4.5%, 8.4%) |
| 1.7B pack | 11.9% (3.6%) | 8.1% (3.1%, 5.4%) |
| 4B pack | 17.5% (2.5%) | 13.6% (2.5%, 4.1%) |
| 4B pack, imatrix | 19.7% (2.2%) | 12.1% (2.1%, 3.7%) |
| 8B + typed-decisions, Q4_K_M | 27.6% (2.4%) | 22.0% (1.9%, 2.9%) |
| 8B + typed-decisions, Q3_K_M | 25.8% (2.4%) | 19.0% (2.4%, 3.7%) |

About a fifth to a third less coverage on the larger packs. The old scan's accepted errors
were also under 5% on this data, because its worst case needs errors flat at the budget, which
jev-bench doesn't have. The 0.6B's accepted error is 4.5%, with a 95% upper bound of 8.4%: on
too few accepted answers to show it holds.

**Still open:**
- **No input-selector binding:** a changed state selector isn't detected. Closures can't be
  saved, so this needs a caller-supplied version if it matters.
- **The same labels both choose the calibrator and certify it.** For the strictest use,
  certify on a held-out set (docs/calibration.md).

## 2026-10-07: the 4B and Qwen3.5-2B trained like the 8B

Both trained on corpus E (soft) plus typed-decisions train, with the 8B's recipe. jev-bench test,
pooled calibrator, Metal. Certified coverage is the fixed-sequence certificate at a 5% budget
(error on accepted test answers in brackets).

| model | size | accuracy | Brier | certified coverage |
|---|---|---|---|---|
| 8B + td, Q4_K_M | 5.0 GB | 0.749 | 0.285 | 22.0% (1.9%) |
| 8B + td, Q3_K_M | 4.1 GB | 0.748 | 0.287 | 19.0% (2.4%) |
| 4B + td, Q4_K_M | 2.5 GB | 0.740 | 0.295 | 12.0% (1.5%) |
| 4B pack (corpus E only) | 2.5 GB | 0.736 | 0.304 | 13.6% (2.5%) |
| Qwen3.5-2B + td, Q4_K_M | 1.3 GB | 0.716 | 0.333 | 15.3% (3.1%) |
| 1.7B pack | 1.1 GB | 0.677 | 0.362 | 8.1% (3.1%) |
| Qwen3.5-2B zero-shot | 1.3 GB | 0.493 | 0.548 | |

typed-decisions test (held out), pack calibrator, and with per-predicate heads:
- **4B + td:** 0.770. With heads, 0.80 at 200 labels and 0.81 at 300.
- **8B + td:** 0.773. With heads, 0.806 at 300 labels.

- **The typed-decisions data is worth little on jev-bench for the 4B:** +0.4 points. Size explains
  most of the 4B-to-8B gap that remains, 0.9 points.
- **Top tier: the 8B.** It ties the 4B on accuracy and certifies 1.6-1.8x as many answers. By the
  rule set on 2026-10-06, a tie on accuracy with a coverage gap over 3 points means the 8B at
  Q3_K_M replaces the 4B.
- **Qwen3.5-2B is the best small model by far:** 4 points over the 1.7B at about the same size,
  2.4 under the 4B at half its size. Its typed-decisions scores and latency are pending.

**Two pipeline bugs, both fixed:**
- **The 4B's fuse failed offline.** mlx_lm fetches the base's README for a model card; fuse now
  reads the local snapshot.
- **The first 2B build scored 0.316.** Its norms were doubled: mlx_lm adds 1.0 to Qwen3.5's
  zero-centered norms and saves them shifted, and the converter adds 1.0 again
  (`sweep/unshift_norms.py`).

Neither showed in the smoke tests: those checked that each step ran, not that the converted model
answered like the trained one. Every build now runs `check` (`sweep/lora_lib.sh`) before
capturing: 80 validation rows through MLX (base + adapter) and through the GGUF, whose
accuracies must agree within 0.08. It takes about 15 s per model:
- the fixed 2B passes: MLX 0.850, GGUF 0.863
- the broken build fails: GGUF 0.300

**Qwen3.5-2B, pack `lars-2b-etd-dev`:**
- **typed-decisions test:** 0.759 with the pack calibrator. The 4B scores 0.770 and the 8B 0.773.
- **With heads:** 0.78 at 100 labels and 0.784-0.788 at 300. The 4B reaches 0.797 and 0.80-0.81,
  so labels lift the 2B a little less than the 4B.

**Latency, Metal, p50, a new input each call** (`sweep/out/latency-tiers.md`; shared machine).
The 2B ran at a load average of 5-6, the others at 9-18, so its numbers are somewhat
flattered.

| request | Qwen3.5-2B | 1.7B | 4B | 8B Q3_K_M |
|---|---|---|---|---|
| one noul | 40 ms | 39 ms | 80 ms | 118 ms |
| Molar Triage (3 questions) | 292 ms | 174 ms | 502 ms | 725 ms |
| twenty questions | 223 ms | 195 ms | 282 ms | 511 ms |
| one noul, CPU 12 threads | 261 ms | 126 ms | 341 ms | 463 ms |
| peak memory (Metal) | 1.8 GB | 2.0 GB | 3.6 GB | 5.0 GB |

- **The same input asked again doesn't get faster on the 2B** (39 ms against 40 ms for a new
  one). On the Qwen3 models it halves. Prefix reuse doesn't help Qwen3.5's recurrent layers here.
- **Lineup:**
  - **The 2B replaces the 1.7B and becomes the default.** It is 4 points more accurate, about
    as fast on single questions, and up to 1.7x slower on several at once.
  - **The 8B at Q3_K_M is the top tier.** The 4B is dropped.
  - **The 0.6B stays** as the smallest.

