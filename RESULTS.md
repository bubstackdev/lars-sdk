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
  at most 5% runs at 10–20% on test. Certifying 5% at 95% confidence takes about 60
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
