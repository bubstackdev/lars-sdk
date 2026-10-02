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
