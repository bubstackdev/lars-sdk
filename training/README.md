# Training

The LoRA trainer behind the packs, and what it needs, all inside lars-sdk.

- `src/lars/`: the lars reference engine and trainer. The original lars checkout was deleted
  on 2026-10-07. This is its source at `ab59a0b`, recovered from `fixtures/.reference/`, with
  the SDK's later changes replayed from the session that wrote them:
  - `train/gated_delta_chunked.py`: a differentiable, chunked gated delta rule, so Qwen3.5 trains.
    Exact against mlx-lm's reference loop: 1.6e-6 on CPU.
  - `train/lora.py`: uses it for training, and gains `--max-steps` and progress lines.
  - `train/data.py`: a noul record keeps its yes/no descriptions as criteria.
- `requirements.txt`, `setup.sh`: `sh training/setup.sh` creates `.venv-train`. mlx-lm is
  pinned at 0.31.3.
- `score_mlx.py`: scores `lars-dev prompts` output with MLX, for the build check.

## The corpus

The original corpus and its scripts were lost on 2026-10-07. Rebuilt on 2026-10-08 (gitignored,
offline from the Hugging Face cache):

    .venv-train/bin/python -m lars.train.build --out data/train-soft
    .venv-train/bin/python -m lars.train.corpus_e

- `data/train-soft`: `build.py` at its defaults. `train/data.py` now decodes every jev-bench soft
  label the way the evaluation does (`_soft_target`). Before, only dict labels were read, so
  civil_comments (a bare P(yes)) and measuring_hate_speech (a list) trained on hard labels.
- `data/train-e-soft`, corpus E: its train splits without the license-dropped sources (jev-bench
  sst5, yelp5 and stsb; tasksource ANLI): 20,000 Open-Jev, 17,285 jev-bench and 700 tasksource rows.
- Not byte-verified against the lost files. `build.py` does reproduce lars's 2026-09-22 corpus byte
  for byte at that corpus's sizes, and corpus E's step counts match the logged runs (`corpus_e.py`).
- `sh sweep/retrain06.sh` retrains the 0.6B on it, the check that it reproduces the packs.

## Base models

The run scripts (`sweep/rebuild.sh`, `sweep/retrain06.sh`) read base models from lars-sdk's own
cache, `.hf/hub` (gitignored), offline. The shared `~/.cache/huggingface` was pruned on 2026-10-09
and took every Qwen base with it. Fetch a base before its run:

    HF_HUB_CACHE=$PWD/.hf/hub .venv-train/bin/python -c "from huggingface_hub import snapshot_download as s; s('Qwen/Qwen3-8B')"

## Accuracy experiments

`sweep/accuracy_ladder.py` compares a matched LoRA baseline with an ordinal-loss LoRA,
targeted teacher supervision, and a shared residual option scorer. Run GPU stages sequentially.
It controls only its own subprocesses and never suspends or changes other sessions' jobs.

Restore the data first with `tools/fetch_jevbench.py`, `lars.train.build`,
`lars.train.corpus_e`, then `sweep/typed_decisions.py --convert` and `--records` as above.
Model arguments can be local, read-only Hugging Face snapshot directories; the runner uses
offline loading and writes its environment and data fingerprints into the output directory.

```sh
.venv-train/bin/python -m unittest discover -s training/tests -v
.venv-train/bin/python training/check_features.py \
    --model /path/to/Qwen3.5-2B/snapshot --records data/train-e-soft/jev-bench.train.jsonl \
    --out sweep/out/feature-readout-parity.json
.venv-train/bin/python sweep/accuracy_ladder.py \
    --model /path/to/Qwen3.5-2B/snapshot \
    --teacher /path/to/Qwen3.5-9B/snapshot \
    --out sweep/out/accuracy-pilot --limit 4000 --steps 250 \
    --eval-rows 30 --seeds 0,1,2 --final
```

This is a short pilot, not a reproduction of the full-corpus pack scores. The 4,000-record
sample balances source and primitive kind; tasksource has the historical 700-record budget.
All recipes use the same data, training steps, source holdout, selection split, and seed.
Choice options shuffle identically across matched runs. The ordinal recipe adds weight 1
on the ranked probability score across the K-1 ordered boundaries; choice and noul losses
are unchanged. The default trainer keeps the original objective (`--ordinal-weight 0`).

The teacher must beat the untrained student on validation before distillation runs. It supplies
targets only on sources where it also beats the trained baseline on validation, and only on
training examples where its argmax agrees with the gold label. Within each source, student
errors and uncertain answers get priority, capped at 500 rows. Targets mix 50% gold and 50%
teacher probabilities; other training targets stay intact. No teacher labels are invented for
held-out states. Inspect `data/distilled-s*.manifest.json` for how much supervision changed.

The shared scorer trains on features from the adapted baseline, selects its checkpoint on
whole held-out sources, then serves through the existing numpy pointer scorer. Feature extraction
uses `--holdout-sources` to match the LoRA source split and skips prompts over 2,048 tokens.
It is an experiment in the Python engine; this does not add shared-head support to SDK packs.
Feature logits use the model's output projection and dtype, matching ordinary inference.
The former numpy float32 reconstruction bypassed BF16 rounding and changed near-tied answers.
`training/check_features.py` checks real prompts before running a new backbone through this path.

Validation states are divided between checkpoint selection and recipe comparison. Test states
are excluded from training and validation. `--final` freezes the recipe with the highest mean
validation macro accuracy across seeds, then compares it with matching baselines on jev-bench
test, typed-decisions test, and an independent 1,000-record Open-Jev OOD sample. OOD states are
also excluded from training and validation. The reported probabilities are **uncalibrated MLX
outputs**, not shipping GGUF pack metrics or certified coverage. A winning recipe still needs
full-corpus training, independent calibration, quantization, and SDK evaluation before promotion.

`status.json` names the current stage; individual stage logs retain commands and failures.
`audit-results.json` gives per-source/per-kind metrics, matched-seed accuracy changes, and
paired bootstrap intervals (state clusters within each fixed source). These intervals measure
example sampling uncertainty, not generalization to unseen source families. `recipe_summary`
reports variation across training seeds. Score rows include exact accuracy, absolute error,
and accuracy across every binary operational cutoff. Long rows are excluded from all compared
models together, with the exclusion count reported.

Re-running the same command verifies completed stage inputs, code, and output hashes before
skipping them. Failed stages are retried. Changing the configuration requires a fresh output
directory; an incomplete training stage restarts rather than resuming optimizer state. The runner
does not modify existing packs, publish artifacts, or install a scheduler.

## Full-corpus confirmation

`python -m sweep.full_accuracy` reproduces the trained 2B baseline, then tests targeted
distillation against that baseline with calibration, quantization, and SDK serving included.
The completed pilot stays untouched. The full run lives in `sweep/out/accuracy-full/`.

Prerequisites are the restored reference pack in `models/reference-2b/packs/lars-2b-dev/0.0.1`,
the pinned raw jev-bench snapshot in `.hf/hub`, the training environment, `target/release/lars-dev`,
`target/llama-tools/bin/{llama-imatrix,llama-quantize}`, and `sweep/.venv-convert` installed from
the vendored `requirements/requirements-convert_hf_to_gguf.txt`. The existing TypeScript build
and native addon are used to verify each experimental pack through `Lars.open(..., verify: true)`.

```sh
.venv-train/bin/python -m sweep.full_accuracy \
    --model /path/to/Qwen3.5-2B/snapshot --teacher /path/to/Qwen3.5-9B/snapshot \
    --out sweep/out/accuracy-full
```

Add `--smoke` and use a different output directory for the complete tiny training-to-SDK check.
A smoke run cannot qualify a recipe for promotion. `--prepare-only` writes and verifies the
data/configuration without starting GPU work. Output directories reject configuration or
protocol changes. A per-directory file lock prevents two copies of the run from overlapping.

The full training population is 39,846 decisions: corpus E with its historical 700-record
tasksource budget, plus typed-decisions train, after state exclusions. All three seeds train
one full epoch with 4,096-token batches, CE+Brier, and the final checkpoint, matching the
historical rebuilding rule. The distillation comparison keeps the same ordered population,
seed, optimizer schedule, and training budget. Only qualifying targets change. The teacher
is calibrated separately before comparison with the calibrated full baseline; its source
eligibility comes exclusively from development rows. Other sources retain their gold targets.

Checkpoint/parity checks, teacher development checks, calibration, certificate fitting, and
confirmation have separate input states. The primary confirmation set contains 4,397 unused
jev-bench test decisions beyond the pilot's first 200 rows per config; all prior pilot states
are excluded. Separate fresh samples contain 1,000 Open-Jev OOD and 1,000 Open-Jev test
decisions. The already examined typed-decisions test is explicitly a regression check, not
a fresh confirmation claim. Hard labels remain separate from soft targets: accuracy uses
the hard label while calibration and Brier use the target distribution.

Before training, the backed-up pack must reproduce its own stored SDK metrics. Each full
baseline must then land within 1 accuracy point and 0.015 Brier of that reference on the
historical benchmark, using historical calibration only for this reproduction check.
Failure stops the experiment for diagnosis before spending GPU time on distillation.

Each adapter is fused, its Qwen3.5 norm shift repaired, converted with `--no-mtp`, and
quantized to Q4_K_M with an importance matrix from this run's training prompts only.
MLX/GGUF agreement is measured on SDK-rendered prompts. A new per-kind calibrator is fitted
through the Rust SDK core on independent calibration rows. Certification uses another set
of labels and the existing fixed-sequence Wilson policy at a 5% error budget. Certificate
rows are deduplicated by source and input state. Sources with no certificate accept nothing.

The predeclared rule is in `protocol.json`. A candidate must gain at least 1 accuracy point,
or 3 exact-score accuracy points with overall accuracy no more than 0.2 points lower.
The selected gain must have a positive paired bootstrap lower bound and improve on at
least two seeds. Every comparison suite must lose no more than 0.5 accuracy points or
0.005 Brier. Certified coverage may drop at most 1 point, must be measurable, and each
candidate seed's accepted-error Wilson upper bound must be at most 5%. Intervals resample
training seeds and paired state clusters within the fixed source suite; three seeds still
give limited information about training variability.

`results.json` and `decision.json` update after each matched seed; unfinished and smoke runs
are never eligible. `status.json` records active stages, failures, or `stopped_at_gate`.
All commands, input/output hashes, versions, and process IDs are retained for audit and
resume. Any qualifying result remains an experimental pack for review; the workflow does
not publish or replace an existing pack.


### Source and teacher-mixture ablations

After the fixed-checkpoint development comparison, prepare the five target-only arms
with `.venv-train/bin/python -m sweep.distillation_ablation`; add `--run` to train and
evaluate them sequentially with Metal access. The default output is
`sweep/out/distillation-ablation-s0/`. Its protocol fixes four single-source mix50 arms
and one all-source mix25 arm, seed 0, full epoch and final checkpoint, reusing the
original population and eligible target identities. Saved baseline/all50 selection
captures are controls. Evaluation uses only the frozen select split and raw MLX
probabilities, with equal weight for seen/unseen source macro accuracy. This workflow
stops after its development report and never promotes a pack or restarts the full sweep.
Tests: `.venv-train/bin/python -m unittest training.tests.test_distillation_ablation
training.tests.test_checkpoint_selection`.


### Development-guided training-data refresh

`.venv-train/bin/python -m sweep.data_correction` prepares a select-only error audit
and a source-count-preserving refresh from unused pinned jev-bench training rows.
Add `--run` for the authorized two-arm, 10732-step maximum comparison. Default output:
`sweep/out/data-correction-s0/`. Candidates exclude normalized state overlaps, conflicting
labels and ambiguous/hard-soft-disagreeing targets, while retaining original source
labels. No teacher is used. Raw MLX select/dev evaluation follows; existing dev has
been used before and is not fresh final confirmation. The run stops after its report.
Tests: `.venv-train/bin/python -m unittest training.tests.test_data_correction`.


### Recovering the trained 8B tier

`sweep.capacity_8b` uses a pinned local Qwen3-8B base, whose revision/files and resolved
path are recorded in `sweep/out/capacity-8b/base-model.json` and `base-path.txt`. The
historical trained8B artifact is unavailable; the user explicitly authorized a rebuild.
Command: `.venv-train/bin/python -m sweep.capacity_8b`. A50-step throughput probe gates
a single full epoch at <=12 projected training hours. The gold corpus and optimizer
match today's2B. It builds Q3_K_M with a training-only imatrix, verifies conversion and
actual SDK behavior, calibrates independently, and compares paired dev accuracy/latency
before stopping. No test evaluation or promotion. Re-execution skips content-verified
completed stages; interrupted training restarts. Test with
`.venv-train/bin/python -m unittest training.tests.test_capacity_8b`.


### Flash persistent teacher worker (CPU tested)

See [Flash teacher bridge](../sweep/flash_teacher/README.md) for build, tests, protocol
and explicit later GPU capture commands. Code lives in `sweep/flash_teacher/`, the
Python backend in `src/lars/backends/flash.py`, and pilot tooling in
`sweep/flash_teacher_capture.py`. The Flash checkout is linked read-only; nothing
there is edited or rebuilt. A250-row pilot is CPU-prepared with the actual Flash
vocabulary/template. Fifteen tests pass, including token-ID reference parity,
boundary fallback, snapshot orchestration, protocol recovery, owned-process cleanup,
and gold-label preservation. CPU tests do not establish actual GPU logits or quality.
No GPU work, 8B resume, training or publication was started.
