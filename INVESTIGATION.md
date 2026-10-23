# Seed-0 investigation, 2026-10-21

The user agreed to stop the full sweep and investigate after the first matched pair.
The verified runner and its own seed-1 training child were terminated; both exited.
No other session was touched. `status.json` is `stopped_for_investigation`.
The user subsequently authorized the checkpoint comparison. An inference-only run is
now active in `sweep/out/checkpoint-selection-s0/`; no new training was launched.

## Evidence

The completed seed-0 independent comparison fails the frozen promotion criteria:
primary accuracy 0.715331 -> 0.708293, primary Brier 0.335167 -> 0.341007;
typed regression accuracy 0.7435 -> 0.7355. OOD accuracy improves 0.809 -> 0.816,
and Open-Jev test accuracy improves 0.715 -> 0.740. Primary accuracy's paired 95% CI
is [-0.015673, 0.001619]. This is mixed, single-seed evidence, not proof of a general loss.

Both models have their best 300-row heldout Brier at step 2500. Baseline accuracy there
is 0.7433 vs final 0.6900; distilled is 0.7000 vs final 0.6667. Heldout includes only
three excluded sources. Historical trainer documentation explicitly warns that this
small heldout Brier is a poor global checkpoint proxy. Therefore these curves justify
an investigation, not a claim that step 2500 wins on a broader suite. The full run
intentionally exported the final checkpoint to reproduce the historical baseline;
this was the frozen protocol, not an accidental best-checkpoint export failure.

Record-by-record inspection confirms the same 39,846 rows in the same order. Only
`target` fields changed: ARC 500, MMLU 224, MNLI 500, HelpSteer verbosity 500.
This is 1724/39846 (4.33%) of records. No other fields changed. Held-out source names
may appear in teacher eligibility but have no training records and received no changes.

Primary confirmation losses include MASSIVE -7.0 points, StrategyQA closed -5.0,
Yelp5 -4.0, FEVER -3.5 and SST5 -3.0. None of those sources received changed targets.
MMLU gains 3.0 points but Brier worsens 0.02193. These descriptive patterns are
consistent with cross-task interference or checkpoint/calibration effects; they do not
identify a causal mechanism. We have not isolated raw-vs-calibrated differences,
checkpoint choice or quantization effects across full suites.

The apparent earlier SDK label flip was a separate checker rounding bug, fixed and
verified with 80 actual prompts and three regression tests. Both final models pass
actual SDK parity. Do not conflate that harness bug with measured model quality.

## Next controlled investigation

1. Use the saved baseline/distilled seed-0 step-2500 and final adapters. No retraining
   is needed. Freeze the candidate checkpoints and selection metric before evaluating.
2. Evaluate on the existing state-disjoint `select` split (2642 rows). Use equal weight
   for seen-source and unseen-source macro accuracy, consistent with the trainer's
   existing selection design. Record per-kind/source metrics and matched predictions.
   Do not select checkpoints using confirmation/test outcomes or the 300-row monitor.
3. Compare each recipe's development-selected checkpoint with its final checkpoint.
   If checkpoint choice helps both, investigate training duration/learning-rate schedule
   before further teacher-target changes. If distilled still loses, ablate changed target
   sources or reduce mixture strength using development evidence.
4. Treat the already examined seed-0 confirmation suites as regression evidence for
   subsequent decisions. They are no longer a fresh untouched confirmation for a new
   checkpoint-selection procedure. Freeze new evaluation/seed plans before further runs.
5. Keep the current model/pack in place. Preserve original protocol/results; write any
   new investigation to a separate output directory. Do not resume remaining seed runs
   without a deliberate decision after investigation.

Machine-readable source breakdown, target audit and checkpoint histories are saved in
`sweep/out/accuracy-full/investigation-seed0.json`. Original matched-pair results and
promotion checks remain in `results.json` and `decision.json`; eligibility is false.
The reason field says remaining seeds pending, but completed-seed checks also fail.

Seed-1 partial artifacts are preserved; interrupted training has no optimizer resume
and would restart if the original sweep were explicitly resumed. Verify historical
PID identities before any future process action. No automatic restart was installed.

## Authorized checkpoint comparison running

Command: `.venv-train/bin/python -m sweep.checkpoint_selection`. Output directory:
`sweep/out/checkpoint-selection-s0/`. Its `protocol.json` freezes baseline/distilled
seed 0 at steps 2500 and 5366, using the existing 2642-row `select` split, raw MLX
probabilities and the equal mean of seen/unseen source macro hard-label accuracy.
Earlier checkpoint wins exact ties. All candidates must share the same population
and targets; no long-row omissions are allowed. Per-source, per-kind, accuracy and
Brier metrics are retained. The separate inference runner has its own log, PID, lock
and content-verified completed-stage markers. `partial-results.json` updates after
each checkpoint; `results.json` and state `complete` mark all four finished. It stops
after comparison and cannot restart the full training sweep or publish a pack.

Two regression tests verify equal seen/unseen weighting despite unequal populations,
hard-label handling, and rejection of a missing group. Run them with
`.venv-train/bin/python -m unittest training.tests.test_checkpoint_selection`.
The original sweep remains stopped.

## Checkpoint comparison completed, 2026-10-21

All four checkpoints completed inference on the same 2642 selection decisions. The
frozen equal seen/unseen macro accuracy rule selects final step 5366 for both recipes:

| Recipe | Step 2500 selection score | Final selection score | Final seen / unseen |
| --- | ---: | ---: | ---: |
| Baseline | 0.671689 | 0.676593 | 0.784852 / 0.568333 |
| Distilled | 0.650518 | 0.664957 | 0.786580 / 0.543333 |

The distilled final checkpoint loses 0.011636 selection score vs baseline (1.16 points).
Final unweighted all-source macro accuracy is 0.732888 baseline vs 0.728201 distilled;
raw macro Brier is 0.298820 vs 0.295199. Distillation improves seen-source macro accuracy
by 0.17 points but loses 2.50 points on unseen sources. These are raw MLX selection
metrics, not calibrated quantized confirmation metrics. No fresh significance or
generalization claim is made from this development comparison.

The hypothesized benefit of switching to step 2500 is unsupported on this broader
selection split: final checkpoints win for both recipes. This does not exhaust other
checkpoint choices, but the small three-source heldout peak was a misleading proxy for
the frozen broader selection metric. Do not claim checkpoint choice explains the
distillation regression. The evidence points toward investigating target-mixture/source
tradeoffs next. A bounded source-by-source distillation ablation (and/or lower mixture
strength), evaluated on development only, can separate hypotheses; no such training is
authorized by the checkpoint-comparison request or launched automatically.

The inference runner completed and exited. Original full sweep remains stopped.
Authoritative output: `sweep/out/checkpoint-selection-s0/results.json`; all four raw
matched captures and protocol are preserved there.

## Source/mixture ablation authorized and launched

The user requested setup and then explicitly instructed starting now. Runner:
`.venv-train/bin/python -m sweep.distillation_ablation --run`. Output:
`sweep/out/distillation-ablation-s0/` with frozen protocol, input hashes, target manifests,
runner log/PID, and per-stage status. All five datasets were prepared and checked;
four new target-mixing regression tests and the two selection-metric tests pass.

Arms, sequential, seed 0, one full epoch/final step, same optimizer/order/population:
ARC-only mix50 (500 changed), MMLU-only mix50 (224), MNLI-only mix50 (500),
HelpSteer verbosity-only mix50 (500), then all four mix25 (1724 changed). Each has
39,846 records. Lower mixing scales the displacement from original soft gold to the
saved mix50 target; it does not replace soft gold with hard labels. Teacher selection,
candidate eligibility and capped changed-record identities are inherited unchanged.

Controls reuse the saved baseline/final and all-source mix50/final raw MLX captures.
All five runs evaluate the same 2642-row select split with the same balanced seen/unseen
macro accuracy metric. No confirmation/test/calibration/certificate capture, quantization,
pack publication or extra seed is performed. The runner stops after results/ranking;
all findings are exploratory single-seed development evidence, never promotion evidence.
Expect roughly 16–20 hours total at the latest throughput, subject to resource contention.
`partial-results.json` updates after each completed arm; `results.json` marks completion.
Rerunning the same command resumes verified stages, but interrupted training restarts.
The original full-corpus three-seed sweep remains stopped. Other sessions remain untouched.


## 2026-10-22: five-arm ablation complete

All five full-epoch seed-0 training/evaluation arms completed overnight and the
runner stopped automatically. No additional GPU job or training sweep was launched.
Frozen development selection results (raw MLX, same 2642 rows):

| Recipe | Selection score | Change vs baseline |
| --- | ---: | ---: |
| verbosity-only-mix50 | 67.924% | +0.265 points |
| baseline | 67.659% | +0.000 points |
| all-mix25 | 67.180% | -0.479 points |
| mmlu-only-mix50 | 66.933% | -0.727 points |
| arc-only-mix50 | 66.689% | -0.971 points |
| all-mix50 | 66.496% | -1.164 points |
| mnli-only-mix50 | 66.355% | -1.305 points |

Only verbosity-only mix50 beats baseline, by 0.265 points. Its seen-source accuracy
is 78.5152% vs baseline 78.4852%; unseen 57.3333% vs 56.8333%. Macro Brier improves
0.298820 -> 0.293878. Lowering the combined mix to 25% reduces the combined recipe's
loss but does not beat baseline. All three other isolated sources lose selection score;
MNLI is worst. These are descriptive, adaptively examined, single-seed development
results. They do not establish reproducible gain or causal effects beyond this seed;
source interactions and differences in changed-record counts remain relevant.

Next candidate, if authorized: freeze verbosity-only as the sole candidate, replicate
on matched seeds and evaluate on independent development/confirmation evidence with
serving/regression gates. The +0.265-point development gain alone is insufficient for
promotion. Do not automatically launch more work. Full metrics and captures are in
`sweep/out/distillation-ablation-s0/results.json` and that directory.


## 2026-10-22: targeted training-data refresh launched

The user authorized an error audit and matched data experiment, then confirmed
continuation. Code: `sweep/data_correction.py`. Run: `sweep/out/data-correction-s0/`.
The select-only audit picks GoEmotions, HelpSteer verbosity, banking77 and LEDGAR
by low accuracy and class imbalance. Corpus-E source exclusions and heldout sources
remain unchanged. The audit reports duplicates/annotation disagreement; it does not
claim ambiguous soft labels are wrong or automatically overwrite labels.

Prepared 577 unused human-labeled training examples: GoEmotions 195, verbosity 270,
banking77 58, LEDGAR 54. These replace majority-class examples with error-prone
underrepresented classes, retaining exactly 39846 rows and unchanged source counts.
All new examples come from the pinned original train split, retain original soft
labels, have >=0.8 target confidence and hard/soft agreement. Selection requires
class support >=3 in select; no dev predictions enter selection. Normalized whole
state matching excludes training/evaluation/heldout/pilot overlaps; conflicting
candidate decision groups are rejected. Prompt preflight leaves token headroom.
All original kept examples retain their labels; source/kind proportions stay fixed.
This is a sampling intervention, not verified correction of mislabeled examples.

Two new arms, baseline and refreshed, run sequentially at exactly 5366 optimizer
steps each, same seed/optimizer and batch limits, fixed final checkpoint. Epoch
ceiling is two only to guarantee the fixed step count despite prompt-length changes;
this is a matched compute budget, not necessarily one complete pass per arm.
Maximum new training budget: 10732 steps. Both are evaluated in raw MLX on select
and dev. Dev was previously used for teacher gating, so it is a separate diagnostic
check but not a pristine fresh final test. No confirmation/test predictions guide
data, no relabeling/teacher, no quantization, publication, extra seeds or automatic
continuation after the report. Expected duration roughly 7–9 active hours.

Five new data-selection tests plus existing selection/ablation tests pass (11 total).
Protocol, audit, input hashes, exact replacements and data hashes are frozen in the
run directory. Log/PID/status track the detached runner. Original sweeps stay stopped.


## HF substitute reconnaissance, 2026-10-22

Requested during the data-refresh setup. No new source was downloaded or mixed into
the frozen comparison. Current cards inspected:

- https://huggingface.co/datasets/unswnlporg/BESSTIE/blob/main/README.md : creator
  card declares CC-BY-4.0 and English sentiment/sarcasm across Australian, Indian,
  British English. Potential sentiment augmentation; not equivalent to SST5's
  five-way scale. Underlying-source terms still require provenance verification.
- https://huggingface.co/datasets/google-research-datasets/paws : Wiki human-labeled
  paraphrases, 49,401 train pairs; card explicitly permits use for any purpose.
  Useful semantic-equivalence training, binary labels rather than STS-B's graded
  similarity. PAWS is already represented in LARS; expansion is a separate ablation.
- https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023 : closest five-star
  review task fit, with 1–5 rating and text fields. Inspected card does not establish
  an unambiguous permissive dataset license; do not call it a cleared substitute.
- https://huggingface.co/datasets/sealuzh/app_reviews : native 1–5 stars, 288,065 rows,
  but licensing section is explicitly More Information Needed. Same limitation.
- TweetEval and STS22-crosslingual cards list unknown license; SICK and Financial
  PhraseBank list non-commercial licenses. They do not solve the recorded exclusion
  concern merely by being available on HF. No clean, task-identical replacement for
  all three exclusions was established in this initial search.


## 2026-10-22: sampling comparison stopped; stronger interventions reviewed

User explicitly stopped data-correction-s0 before baseline training finished. Verified
runner and its own training child exited. State: stopped_by_user. No refreshed-data
training or new GPU job was started. The 577-row swap was a weak causal hypothesis;
baseline retraining was redundant with the saved control. Do not resume automatically.

CPU-only analysis of existing calibrated development captures (2471 rows) is saved in
sweep/out/stronger-intervention/headroom.json:

- Trained 2B macro accuracy 73.3750%, zero-shot 9B teacher 69.0498%.
- Fixed confidence fallback loses for every tested threshold from .3 to .9.
- Two state-hash folds, teacher selected per source on the OTHER fold using at least
  20 tuning rows and a 5-point teacher advantage: 74.9750% (+1.60 points), Brier
  0.282150 vs 0.295073, escalating 20.28% of rows. This is exploratory development
  routing using known source identities. It is not a deployable arbitrary-question
  router or fresh test result. Small source samples and different serving paths
  (2B GGUF vs 9B MLX) limit interpretation. Calibration labels were separate.
- Full-dev teacher task improvements: ARC +11 points, FEVER +6, helpfulness +17,
  MMLU +7, STS-B +12. Overall score accuracy loses 53.45% vs 57.80%; routing all
  score questions is unsupported.
- Gold-assisted oracle accuracy 82.8764% is a headroom bound, not achievable routing.

Existing stronger evidence: trained 8B historically achieved 74.8-74.9% vs 2B 71.6%
on the old shared benchmark, with higher memory/latency. Predicate-specific heads
historically lifted the trained 2B from75.9% to78.4-78.8% on typed-decisions with300
labels/predicate; trained 4B from77.0% to80-81%. Heads require local labels and stable
option semantics. These old populations are different from the current dev analysis.
Do not present older 4B pre-domain-adaptation gains of15 points as expected on today's2B.

Recommendation: gate further investment with an inference-only matched benchmark of
an existing trained larger pack and conservative task-aware fallback. Freeze routing
by supported predicate/schema; measure accuracy, Brier, accepted coverage, escalation
rate, memory and latency. A small development comparison comes before any large run.
Existing 8B artifacts may require restoration. No download, GPU benchmark, deployment
or further training was launched in this review. Another path is per-predicate heads
if users can supply labels. Keep the current pack until independent evidence passes.


## 2026-10-22: trained 8B artifact unavailable; rebuild explicitly authorized

The inference-only comparison could not start as proposed: no trained 8B pack or
adapter exists in project storage, searched local model caches/other lab checkouts,
or the existing HF backup. HF aether-models/lars-packs at revision
4af4d532cd004bc3a6c1683d6c202e1497c0d2ec contains only 0.6B and 2B packs. Historical
handoff explicitly says the 8B was lost, never rebuilt, and partial checkpoints were
local; those checkpoints are also absent now. Do not imply the historical74.8% tier
is available or that a zero-shot 8B is the same comparison.

The user replied: "If it isnt on HF... we have to rebuild it :/". This explicitly
authorizes the rebuild before the inference comparison. The official Qwen3-8B base,
revision b968826d9c46dd6066d109eabc6255188de91218 (16.38GB weights), is downloading into
project .hf/hub. base-model.json/base-path.txt in sweep/out/capacity-8b record identity
and readiness. An authorized detached runner awaits the download, validates file
sizes, then runs sweep.capacity_8b. Its log/PID/status are in that output directory.

Pipeline: fifty real training steps on the full frozen clean corpus first, reporting
seconds/step, total batches and peak training memory. Stop at an evidence gate if
projected training exceeds12 hours. Otherwise perform one full gold-only epoch,
seed0/rank8/scale20/lr2e-5/batch4096/maxprompt2048, same39846 gold training decisions
and300 heldout rows as today's2B. Export final checkpoint, fuse, convert, training-only
imatrix and Q3_K_M. Check80 identical prompts against MLX, independently calibrate
on the same1836 calibration rows as2B, emit an experimental local8B pack and verify
actual Node SDK predictions (zero label flips, probability tolerance0.001).

Then capture both models on the same2471 dev examples in current core/runtime; report
accuracy, balanced seen/unseen accuracy, Brier, paired state-cluster interval. Measure
same three single-question requests (noul/choice/score),20 repeated calls with varied
state per model, sequential. Latency is shared-machine indication, not isolated SLA.
Useful development gain is frozen at2 accuracy points without Brier worsening >.005.
Neither dev evaluation nor this rebuild qualifies for promotion. Historical8B data
was reconstructed; do not promise exact reproduction of its old weights/numbers.

The runner stops at failed throughput/conversion/SDK gates or after the comparison.
No distillation, other sampling sweeps, extra seeds or automatic publication. Current
probe budget is a projected training-hours gate, not a wall-clock kill timer. Two new
CPU tests verify timing projection and8B model/template/quantization/identity metadata.
Tests and diff checks pass. No other sessions/processes are controlled.


## 8B paused at explicit user request

The user needs the GPU and will request restart later. The verified runner and its
own training child exited. Status: paused_by_user; no automatic restart. Training
had reached about290/5021 steps, before the first500-step checkpoint. No full-run
checkpoint exists. Base download and50-step probe remain saved. The trainer has no
optimizer-state resume, so a later explicit restart repeats the full training while
skipping verified completed preparation/probe stages. Other sessions were untouched.


## Flash teacher integration prepared and CPU tested

The user authorized setup and CPU tests only while another session's Flash workload
uses the GPU. No other sessions/processes were changed. The8B remains paused.

Implemented persistent JSONL C++/Objective-C++ teacher worker and Python FlashBackend,
plus CPU preparation, explicit GPU capture and matched-student reporting commands.
The worker is built in LARS and links Flash's existing static library; no Flash repo
edits or rebuild. Raw label logits come from last-position trunk output, no sampling
or speculative generation. One retained prefix snapshot, joint tokenization boundary
checks and full-prompt fallback; suffixes are independent. Model loading requires
explicit --allow-gpu in both the worker and capture command. A GPU-mode memory guard
exits only its own worker on elevated/unreadable pressure. No public service or C FFI
was added; this is the offline teacher bridge first.

Compiled native runtime and standalone mock worker. Mock executable links only
libc++/libSystem (no Metal). Fifteen CPU tests passed without skips, including the
actual qwen4exp GGUF tokenizer, official HF token-ID comparisons, real template,
label order, empty suffix, deliberately merging token boundary,30 randomized cache
parity cases, malformed JSON/error recovery, timeout/ID mismatch, subprocess cleanup,
gold soft/hard label semantics, changed-plan rejection and synthetic-evidence refusal.
Mock logits are synthetic and explicitly cannot count as teacher-quality evidence.

CPU-prepared pilot: sweep/out/flash-teacher-pilot/,250 decisions (10 per source),
max1422 prompt tokens, mean424.04. Every prompt/option label passes actual vocabulary
validation. Zero actual BPE boundary mismatches;11 adjacent shared-prefix opportunities.
Vocabulary-only worker peak RSS98385920 bytes (~94MiB); no weights or Metal model
loaded. cpu-validation.json records the check. Original gold labels remain attached;
plans pin inputs/template/worker/selected-data hashes and weight path/size/mtime (not
a full cryptographic weight checksum). No real GPU parity, teacher accuracy, timing,
calibration fitting or distillation has been run. Existing dev is development only.

Later, only when authorized and GPU/memory capacity permits a separate large Flash
instance: run the capture command in sweep/flash_teacher/README.md, check real
snapshot/full-pass GPU parity, and compare raw predictions to the saved matched2B
capture before considering more labels. Fit probabilities on separate gold calibration
and keep certification labels independent. Current user's Flash job must remain alone.

Resuming the paused8B requires explicit direction. The new backend file changes its
all-training-code fingerprint; revalidate provenance/use an intentional fresh output
configuration rather than silently editing a frozen code manifest to claim continuity.


## Authorized Flash GPU pilot completed, 2026-10-22

After the user released the GPU, normal memory pressure and no matching resident
Flash/LARS workers were observed. Only this session's two sequential worker instances
were used, with the mandatory memory-pressure watchdog; no other job was controlled.
The parity worker and capture worker both exited.8B training remains paused.

Real-model checks covered noul/choice/score, the1422-token longest pilot prompt, repeated
prefix restore, requested label reversal, and a shared-prefix two-question batch.
All cached/full/replay/reordered comparisons had exactly zero observed logit/probability
difference and zero label flips. This is bounded evidence, not exhaustive long-context
or SDK/Rust/C-FFI parity. Reproducer and gpu-parity.json are in the pilot output.

The250-row pilot completed without skipped decisions; original gold labels and soft
targets were retained. Comparison to the saved raw trained2B predictions:

| Metric | Trained2B | Flash |
|---|---:|---:|
| Accuracy (25 equally sized sources) |75.6% |74.8% |
| Raw Brier, lower better |0.29519 |0.32951 |
| Binary accuracy (82 rows) |91.46% |86.59% |
| Choice accuracy (93 rows) |76.34% |77.42% |
| Score exact accuracy (75 rows) |57.33% |58.67% |
| Score MAE, lower better |0.5067 |0.4800 |

Paired accuracy delta -0.8 percentage points; within-source state-cluster bootstrap
95%CI [-6.51,+4.44]pp. Both correct158, only student correct31, only Flash correct29,
both wrong32. Strong-looking per-source changes (MMLU+5/10, SST5+4/10, hate-speech-4/10)
are descriptive samples of ten and do not establish reliable teacher eligibility.

Model load6.54s; capture189.56s for250 decisions (~0.758s/decision, includes Python
render/tokenization/protocol/I/O), total196.10s. One observed worker RSS sample was
~76.1GiB, not a measured peak. Memory pressure was normal at both monitoring samples.
No teacher calibrator was fitted; raw Brier is not a calibrated deployment comparison.
This existing development subset is not pristine confirmation. There is no evidence
yet for blanket distillation or shipping a Flash tier; do not infer overall inferiority
from this inconclusive small pilot either. A next bounded experiment could test prompt/
label-format sensitivity and broader matched development performance before training.
Artifacts: sweep/out/flash-teacher-pilot/comparison.json, paired-analysis.json,
capture.jsonl, capture-metadata.json, gpu-parity.json, run-parity.py, run-analysis.py.


## Flash response-format investigation, 2026-10-22

The user requested investigation after the initial pilot. Froze the same250 decisions,
original gold/soft targets, label pool, model file size/mtime, worker/template/input
hashes, and two prompts before scoring. Model weights still lack an independent full
checksum. No option order changes, no omitted alternatives, no teacher targets used
as gold. The first prompt reproduces the canonical assistant `Answer:` prefill and
reads ` A`-style tokens. The second removes only `Answer:` after the official empty
thinking block, reading both bare and space-prefixed labels in one forward pass.
Every evaluated continuation was checked against joint full-prompt tokenization.

CJ, CZ and EJ are multi-token without a leading space. Rather than score only their
first token or silently omit options, the bare comparison excludes all20 LEDGAR/CLINC
rows requiring them. The full space-prefixed comparison retains all250 decisions.

| Full250 decisions | Accuracy | Raw Brier (lower better) |
|---|---:|---:|
| Trained2B |75.6% (189) |0.29519 |
| Flash canonical |74.8% (187) |0.32951 |
| Flash native response, spaced labels |75.6% (189) |0.32235 |

Native vs canonical Flash gain+0.8pp; paired state-cluster95%CI [-1.56,+3.07]pp.
Native vs2B delta0pp;95%CI [-5.49,+5.54]pp. Native Flash binary accuracy84.15%
vs2B91.46%; choice78.49% vs76.34%; score62.67% vs57.33% (75 score decisions).
Three of the score gains against canonical Flash are HelpSteer helpfulness examples;
its per-source sample is only ten. These are exploratory development results.

On the identical230 bare-eligible decisions, trained2B and canonical Flash each score
74.35%, native spaced75.22%, native bare74.78%. Bare vs canonical gain+0.43pp;
95%CI [-1.70,+2.61]pp. Bare Brier0.33347 vs2B0.30533; no response-format rescue is
established. All250 canonical control probability vectors matched the original run
exactly, with zero label flips. This argues against a cache/prompt-reconstruction
regression; it does not independently prove Flash's underlying model math fidelity.

Qualitative gold-disagreement inspection found non-spam personal/motivational SMS
called spam, blunt political criticism called toxic, and gold-negative hate-speech
examples assigned borderline levels. This is consistent with different benchmark
classification boundaries, not proof of a single causal failure. The trained corpus
includes969 SMS and1000 measuring-hate-speech records; SST5/Yelp5/STS-B have none,
and civil_comments is held out as a source. Thus simple training coverage alone does
not explain all observed source differences. The system prompt also asks for one
option letter while multi-letter labels exist; that inconsistency was noted but was
not changed in this experiment. Shipped prompt/label defaults remain unchanged.

A resource event interrupted the first worker after243 complete comparisons: its
mandatory memory-pressure watchdog exited only its own process. One earlier RSS
sample was~88GiB and pressure was normal then; the exact pressure trigger's cause and
peak allocation were not measured. After pressure returned to normal, validated all
saved keys/order/gold/probabilities and pinned inputs; completed only7 remaining rows
in a fresh guarded worker. The control still matched exactly across the restart.
Both workers exited and pressure was normal afterward. No other job was signaled.
First-stage timing was not retained on guard exit; plan mtime approximates start.
The recovery took~6.87s load plus7.68s capture/cleanup. Failure/status and recovery
provenance are retained; no examples were silently discarded.

Added explicit empty/space label-prefix scoring for diagnostics; handshake capability
negotiation prevents an old protocol-1 worker silently treating bare labels as spaced.
New diagnostic workers rotate after at most100 rows, closing the prior model before
loading the next; this bounds worker lifetime without disabling the memory guard.
The real-model results used the earlier frozen binary; the final rotation/capability
changes are CPU verified separately, not a new GPU quality run. Original pilot and
investigation binaries are preserved; latest build is target/flash-teacher-current/.

Conclusion: neither format variant establishes a reliable overall teacher advantage;
raw probability quality remains worse, and no distillation should follow from this
pilot. If Flash is tested further as a teacher, a bounded reasoning/generation trial
with gold-checked disagreements is a different method worth evaluating before larger
capture or training. Actual reasoning/generation is not implemented by this bridge.
No calibration, training, shipping or new certification was performed;8B remains paused.

Artifacts: sweep/out/flash-prompt-investigation/{plan,status,comparison}.json,
canonical_spaced.jsonl, native_spaced.jsonl, native_bare.jsonl,
first-attempt-failure.json, capture.partial.jsonl (all250 recovered decisions),
run-resume.py, and frozen-source/. The frozen source/binary predates the final CPU
hardening and is retained for reproduction; do not overwrite the original experiment.

Final CPU hardening verification:17 tests passed, including actual GGUF vocabulary/
template checks, old-worker prefix rejection, and a101-row mock capture that opens
exactly two non-overlapping worker instances and preserves original control/gold across
rotation. No GPU rerun was needed for these orchestration changes. Evidence hashes
are in cpu-hardening-validation.json in the investigation output.


## Bounded reasoning teacher pilot completed, 2026-10-22

The user authorized the proposed reasoning-enabled teacher test, then asked whether
LARS's useful deployment path should involve domain/company calibration rather than
continued universal accuracy chasing. Answered yes: representative local labels and
per-predicate calibration/adaptation, followed by independent accepted-error/coverage
measurement, should be the default deployment expectation. Probability calibration
alone does not necessarily change answers or supply missing knowledge; the SDK's
per-option recalibration and fitted heads can change decisions. Historical calibration
learning curves in docs/calibration.md/USING.md support testing that path, but are not
current-pack or new-company guarantees. A local learning-curve run was recommended,
not launched; the bounded teacher experiment below was finished as authorized.

Implemented actual deterministic greedy full-vocabulary generation in the native
worker/Python adapter. It uses successive trunk forward passes, official thinking-
enabled template, low reasoning effort, explicit EOS/length limits, no sampling or
speculative MTP. CPU vocabulary workers refuse generation; old workers are rejected
unless they advertise greedy-v1. Piece bytes avoid malformed UTF-8 protocol output.
Generation clears scoring/cache state before and after.21 CPU tests passed, covering
EOS/budgets/context/recovery, native/Python CPU gates, capability checks, strict final
parsing, multi-letter labels, gold-independent selection and failure accounting.

Before GPU load, froze32 distinct states from the existing250-row development pilot:
24 student-vs-Flash prediction disagreements,8 agreement controls. Source-balanced
hash selection uses predictions/source/state only, never gold correctness. Original
hard gold and soft targets are stored outside the rendered prompts. Task system text
requires supplied evidence/criteria and a final line FINAL: <label>, including multi-
letter labels. Compared generated-direct thinking-off128 tokens with low-effort
thinking-on1024 tokens; these budgets and the changed task prompt are part of the
methods, not a clean isolation of thinking alone. Initial4-case gate checked only
completion/runtime (4/4 reasoning valid, estimated276.74s<1200s), never accuracy.

| Method | All32 correct | Disagreements24 | Agreement controls8 | Valid generated answers |
|---|---:|---:|---:|---:|
| Saved trained2B direct logits |19 |11 |8 |n/a |
| Saved Flash direct logits |19 |11 |8 |n/a |
| Generated-direct control |16 |9 |7 |26/32 |
| Reasoning-enabled generation |19 |12 |7 |32/32 |

All64 generated arms completed and were retained. Invalid/truncated output counts
incorrect: generated-direct had5 token-budget failures and1 invalid final format;
none occurred in reasoning. Against Flash direct logits, reasoning corrected1 error,
introduced1 error and changed5 predictions (3 changed wrong answers remained wrong).
Against the trained2B it corrected11 errors and introduced11. This small enriched
cohort cannot estimate population/deployment accuracy; no confidence certificate or
probability calibration is inferred from one-hot generated choices. The controls
happened to be8/8 gold-correct for both baselines; that was not a selection criterion.

Total runtime277.93s, including8 sequential model loads (one worker per4 cases).
Generation itself220.44s and11738 output tokens across both arms. Reasoning mean5.84s,
median4.29s/decision,10950 output tokens total. Generated-direct mean1.04s,median0.62s.
These measurements include tokenization/protocol/prefill/decode per arm and are specific
to this local bounded diagnostic, not quiet-machine serving benchmarks. Sampled
memory pressure was normal; no guard exit occurred. All workers exited; pressure
normal afterward. Other sessions/processes were untouched.8B remains paused.

No net teacher accuracy advantage was established even with completed reasoning.
Do not expand this into blanket target replacement or more broad training on this
basis. Prioritize a representative local calibration/adaptation learning curve,
with disjoint gold evaluation and accepted-error/coverage versus label cost. If using
teacher suggestions to assist annotation, retain independent human/domain gold for
validation/certification; generated labels cannot certify themselves. This result
does not prove an intrinsic ceiling for local models or rule out other teacher methods.

Artifacts: sweep/out/flash-reasoning-pilot/{plan,status,comparison,analysis}.json,
cases.jsonl,capture.jsonl,smoke.json,cpu-validation.json,frozen-source/.
No model weights changed, no training data exported, no calibration fitted, and no
shipping/certification/publishing performed.


## Local calibration learning curve setup and Codegraph case seed, 2026-10-22

User authorized setup, selected saved workflows first, and permitted Codegraph mining
for case creation. Implemented CPU-only sweep/local_calibration.py and documented it
in sweep/LOCAL_CALIBRATION.md. Reuses current baseline2B Q4_K_M raw dev/typed-regression
captures and frozen pooled calibration. Those four synthetic workflow types were
included in backbone training and used in past evaluations; this is development
local-adaptation evidence, not new-company or untouched final confirmation.

Whole normalized states and case families remain disjoint across fitting, certificate
and evaluation roles, including across multiple questions about one input. Base-
training normalized states are excluded.491 distinct states yield2455 decisions:
100 fitting states,300 certificate states,91 evaluation states (20 predicates,455
held-out decisions). Each predicate has25 fitting,75 certificate,21–26 test labels.
The same fixed roles serve every budget/draw;0/10/25 fitting labels are supported,
50/100/200 are explicitly unsupported, with no padding, duplicate examples or pooling
across predicates to manufacture sample support. Three draws are not independent tests.

Hard labels fit the adapters; original soft targets are retained separately for
metrics. Five-fold fitting-only CV uses the existing paired log-loss guard to keep
the pack or choose a refit/option-bias recalibration. Optional aligned hidden vectors
allow heads from50 fitting labels; current cached rehearsal has no hidden vectors
and evaluates no heads. This offline Python study follows SDK principles without
claiming byte-identical calibration parity or exporting a deployable calibrator.

| Fitting labels per predicate | Macro accuracy | Hard Brier | Soft Brier | Accepted coverage at5% |
|---|---:|---:|---:|---:|
|0 |0.74759 |0.41956 |0.06672 |0 |
|10 |0.74967 |0.39556 |0.12198 |0 |
|25 |0.75408 |0.34067 |0.10884 |0 |

The+0.65pp accuracy change is descriptive. Hard-label probability metrics improve at
25 labels, while agreement with the soft benchmark targets worsens. No threshold
qualified at5% on any predicate/draw's independent75-label certificate set. The rule
fails closed: accepted error is undefined, never reported as zero.75 labels barely
exceed the73-clean Wilson minimum and do not support a well-powered coverage study.
Evaluation thresholds are frozen; fixed-threshold test diagnostics are exploratory,
not an oracle for deployment selection. Per-predicate bounds do not establish a
company-wide/multiplicity-adjusted guarantee. Label costs include fit+75 certificate
labels, with21–26 evaluation labels additional per predicate.

Five targeted CPU tests pass: normalized state/family separation, one-model/fixed-
question alignment, unreviewed-gold rejection, hard/soft-gold separation, evaluation
label isolation from method/threshold choice, nested budgets, unsupported budgets,
changed-frozen-input rejection, Wilson minimum/first-failure stop, hidden-head gating
and a positive signal fixture, and source-rule boundary/precedence checks. SciPy1.18.1
was installed in the project training environment; calibration-study optional
requirements were added to training/pyproject.toml. No GPU inference or training ran.

Codegraph rag_search/rag_context read seven immutable source sections from six systems:
FABAL refunds, Billing invoices/retries, DomainOwner returns, Billing.Payout result
handling, FinancialReporting story HTTP retries, and DomainData escalation matrix.
Source links, indexed commit revisions, complete section contents and retrieval
snapshots are saved locally. No runbook commands, operational APIs, messages or
external mutations were executed. This is an initial source harvest, not a claim to
have exhausted every accessible Codegraph repository.

sweep/codegraph_calibration_cases.py constructs212 deterministic contract probes in43
families over7 decision types:112 billing retries,40 HTTP retries,20 payout results,
16 refund prerequisites,12 invoice closures,9 escalation contacts,3 domain returns.
They include exhausted-retry boundaries, error precedence, and the strict>5min
escalation condition. No model predictions label these cases. Expected answers are
rule-derived from constructed inputs and retrieved documentation; source interpretation
still needs domain review. All cases remain marked rule_derived_unreviewed, with
source evidence, revisions and per-case family IDs. They are not production traffic
or independently reviewed company gold, and were not mixed into the benchmark curve.
The calibration runner rejects this draft status. An annotation CSV exposes draft
answers with empty reviewed-label/reviewer columns. Variant counts do not create
independent source evidence; related families must remain together in future splits.

Artifacts: sweep/out/local-calibration/{plan,status,results,summary,environment,
validation}.json,captures.jsonl,pack-calibration.json,frozen-source/;
sweep/out/codegraph-calibration/{cases.records,cases.wire,evidence}.jsonl,
cases-manifest.json,sources/,review.csv,review-manifest.json. Outputs/source material
remain gitignored and local. No weights, pack defaults, training, certification,
publication or other sessions' processes were changed;8B remains paused.

## 2026-10-22: Codegraph audit and current 2B baseline completed

The authorized source review and baseline run completed on all212 original draft
policy probes in about24 seconds. Frozen audit tiers separate136 implementation-
corroborated cases (95 correct,69.9%),29 documentation-only cases (16 correct), and
47 quarantined cases (32 correct). Corroboration inspected revision-pinned code and
test assertions; remote tests were not executed and labels remain unreviewed.
Quarantine:40 FinancialReporting HTTP retry documentation/code conflicts,4 FABAL
withdrawal documentation/code conflicts,3 DomainOwner cases missing runtime guards.
Prompts explicitly contain the documentation policy, so these conflicts cannot be
resolved by changing gold alone. Original case files and labels were preserved.

Billing retries cause38 of41 corroborated errors:74/112 correct, with3 delayed and
35 fault cases predicted as immediate retry.112 variants represent only7 families.
Invoice closure12/12; non-withdrawal refund prerequisites9/12. Observed source-majority
reference on corroborated cases is70.6%, a descriptive class-imbalance reference,
not an independently fitted deployment baseline. All four short MySQL outages
incorrectly trigger escalation;7/10 error-bearing payout cases miss error precedence.
Exact retry and time rules should stay in code. Independent reviewed case families
are needed before a credible adaptation comparison on remaining judgment tasks.

Hidden control capture failed the predeclared probability gate: maxdelta0.00395974
against0.001, no answer flips. Logits-only control passed exactly; main capture used
that fallback and has no hidden vectors. Model/runtime/template identities matched
the previous trained2B baseline. All212 native row hashes match original inputs.
Three tests pass for tier exclusions, serving parity and own-child-only memory guard.
Artifacts: sweep/out/codegraph-baseline/{plan,status,results,error-analysis}.json,
audit.jsonl,audit-sources/,scored.jsonl,controls*.parity.json,frozen-source/.
Our workers exited; no fitting, certification, training or deployment occurred.

## 2026-10-22: historical judgment-task preparation

User authorized the proposed pivot to Codegraph judgment cases. Retrieved67 historical
Shortcut stories from payment/transfer/cart/WHOIS/DNS discovery plus4 pinned runbooks;
selected44 diagnostic checkpoints rather than generating rule-grid variants. This is
purposive retrospective sampling, not exhaustive access or production prevalence.
Agent-curated labels remain unreviewed. Exact quoted evidence anchors checked against
saved source fields; no independent semantic adjudication is claimed. Current ticket
snapshots include edited descriptions, automated reports and corrected diagnoses, so
the cases represent reconstructed post-observation decisions, not prospective intake.

Family grouping joins explicit mechanism groups and formal story links, even through
unretrieved shared stubs.34 proposed dependence groups; deterministic hash split:
fit13cases/10groups,certificate19/14groups,test12/10groups. Class coverage remains sparse:
no infrastructure in fit/test and no configuration in certificate. No resplitting to
manufacture coverage or gains.44 labels are draft; no inference/adaptation/training ran.
The next comparison is predeclared0/10-label current2B versus fit-only guarded adaptation
after independent review and expansion. Larger budgets and5% certification unsupported.

Active artifacts: sweep/out/codegraph-judgment/bundle-v3/{inputs,cases.draft,evidence}.jsonl,
review-blind.csv,readiness.json,manifest.json,frozen-driver.py; parent study-plan.json,
validation.json,curation.json,sources/,discovery/.v1/v2 superseded before any predictions.
Model inputs exclude label/provenance/outcome fields, identities and actual domains.
Credential-like XML/JSON values redacted in local snapshots; one sensitive source body
omitted. Source snapshots otherwise remain local/gitignored. No instructions retrieved
from documents executed, external writes/messages, GPU use, or control of other sessions.
Seven relevant tests pass; frozen outputs and67 source hashes verified.8B remains paused.

## 2026-10-22: confidence audit, local correction UI and provisional CPU comparison

User authorized autonomous completion with human review below0.8 confidence. Harvested
36 additional story snapshots, curated31 more cases, audited all75.103 snapshots total,
64 proposed dependence groups; all44 original inputs and roles unchanged.56 cases meet
the threshold;19 are queued. Subjective confidence measures the proposed track given
the reconstructed observations; it is not empirical reliability or human gold. Pinned
code/change excerpts supplement5 cases, including two refund repositories. Privacy
remains uncertain because the relevant policy conditions are not in the case facts.

Local UI runs at127.0.0.1:8765 via sweep/judgment_review.py; user-facing server retained.
Source excerpts are text-rendered; no CDN, remote storage or publication. SQLite keeps
versioned corrections, conflict protection, fact edits, exclusions, deferrals, reopening,
family concerns and exports. Original evidence/cohort files stay frozen. The DB is bound
to the queue identity. Do not overwrite live reviews.sqlite3; user may be annotating.
review-20261022/{queue,needs-review,validation}.json,needs-review.csv,SQLite DB and QA
artifacts are local/gitignored.18 targeted Python tests and11 real-browser checks pass.
Browser changes used fixture-only data in a separate Chrome profile; QA processes exited.

Cached75 native2B outputs in capture-v4:8 control rows exact, native row IDs/options
verified,~8seconds, GPU worker exited. No hidden extraction, training or certification.
Predictions remain outside the review UI. A separate explicitly agent-assessed CPU
diagnostic follows the authorized confidence rule; the standard reviewed-gold gate is
unchanged. One case per group leaves23fit/14certificate/9test,0/10 supported labels.
Fit-only CV keeps recalibration in all three10-label draws. Baseline4/9; adapted6/9,
4/9,5/9, averaging5/9 on the same held-out groups. Corrections/regressions2/0,2/2,3/2.
HardBrier pack.70984 ->.58208mean (raw.68014). This is not an established gain or company
validation.25+ budgets unsupported, no certificate or deployable parameters exported.
Active probe-v4-final includes frozen plan, results, selected rows and driver snapshot.
Future exported user corrections can enter a fresh probe with provenance retained;
changed facts require recapture, and excluded/deferred/family-concern cases are omitted.
No remote changes or control of other sessions occurred.8B remains paused.

## 2026-10-23: retire the judgment-category study; correct evidence separation

User correctly questioned the review burden for already fixed bugs. The prior task
assigned curator-created investigation tracks, and several observation lists included
later findings. This invalidates using the provisional4/9 -> mean5/9 comparison as
adaptation evidence. The old artifacts/corrections are preserved; study-retirement.json
marks the question/results retired and the probe CLI refuses that question signature.

New builder sweep/resolution_replay.py admits only exact dated earlier comment excerpts,
with separate direct excerpts of a later recorded finding/resolution. Latest edit time
must precede an explicit cutoff, including earlier findings in related tickets. It never
uses current descriptions or curator titles/summaries as features. Source offsets,
timestamps, hashes and links retained. Actual original-description history could not
be fetched: Codegraph tools absent this turn and plugin discovery found no match.

Active resolution-replay-v2 has5 fixtures/5groups: cart catalog references, transfer-
security scope, payment-group refund propagation, push transport resilience, partial-
transfer refund reconciliation.4 pre-finding,1 pre-resolution. The last does not assert
a causal diagnosis of the original refund delay. Sparse symptoms need not uniquely
identify the later outcome. This is a replay fixture set, not a credible learning curve.
No new model scoring, fitting, certification or GPU work. Prior roles retained only
as provenance,2fit/2certificate/1test. Direct labels documented_outcome, not human gold.

70excluded:36unverifiable earlier versions,20unconfirmed outcomes,5ambiguous policy/
resolution,4already-diagnostic or vague-fix records,3insufficient earlier facts,1duplicate,
1different incident. The last pairs earlier auth outage observations with a later OOM
finding for 'another' outage; it was excluded rather than treating ticket identity as
incident identity. No closed-bug re-labeling tasks are requested.

Local page8765 now displays the audit and evidence timelines. Prior reviews remain
read-only at/review; writes return410. Consistent SQLite backup retained alongside the
new bundle; original DB untouched (0events at backup). Replaced only the recorded own
review-server PID after checking its exact module/data/db/port command. New own PID44826
stored in server.pid; detached app retained.25 targeted Python checks and7 read-only
Chrome checks pass, no saved reviews mutated.8B remains paused; other sessions untouched.


2026-10-23 — Codegraph history access restored. Pinned 73 ticket histories and
re-fetched seven candidate full records. Recovered creation events enable six
additional exact-source replay fixtures, expanding active resolution-replay-v3 to
11 cases / 11 groups; 64 cases remain excluded, zero human review tasks. Seven
candidate comment sets match their prior snapshots. The transfer-lock race case
141835 stays separately staged because it is outside the frozen cohort. 32 prior
version-history exclusions have recovered histories but remain unadmitted.
Builder now verifies creation-event type, exact quote, chronology and pinned
history hashes. Original/current descriptions are not interchangeable. No GPU,
model fitting or scoring; prior judgment-category metrics remain retired. Local
read-only audit updated to v3, prior review DB preserved. More independent cases
per outcome are still needed before measuring calibration gains.


2026-10-23 — Completed all32 remaining history admission decisions:1 admitted
(142001),31 specific exclusions,0 pending. Fourteen were already diagnosed at
creation. Active replay-v4 is12 cases/12groups/63exclusions. Added description-
revision outcome/cutoff support without allowing revised descriptions as inputs.
Deeper nameserver workflow search audited16 new stories and pinned10 histories;
three new fixtures plus existing140396 yield4groups with4different actions.
No repeated source-backed action currently has independent fit/test examples.
No split/comparison/GPU work started. Case decisions, source hashes and prospective
study gates are saved under workflow-audit-20261023; searches were first-page,
not exhaustive.26 relevant Python tests+7read-only Chrome checks pass; original
review DB preserved and local audit updated to v4. Sparse symptom intake does not
prove the eventual repair was inferable; diagnostic checkpoint data and repeated
independent outcomes are the missing ingredients.


2026-10-23 — Continued mining to361 full ticket/history records. Built a NEW
source-documented final queue-disposition task, respecting the user's explicit
pre-action evidence requirement.42 admitted cases/41 dependence groups;29
have diagnostic observations and13 only earlier alerts/symptoms. Source version
anchors, edit/creation times, conservative known action-time bounds, and explicit
completed action labels are validated. Two vague 'cleared' outcomes were excluded.
No old invented triage category or subjective confidence became gold.
Frozen25-fit/16-test groups before capture, one representative per family.16test
contains11diagnostic/5sparse cases.8native controls exact;41case captures took8.18s,
worker exited. CPU0/10/25 fitting-only CV uses proper log loss and a class-prior
baseline. Existing pack10/16;10-label draws12,12,10;25labels12/16, same as always
reprocess. Diagnostic9/11 unchanged; calibrated discard0/4. Prior-only Brier/log
loss .378/.567 beats fitted .400/.587. Paired accuracy delta95% [-.125,.375].
No meaningful judgment gain established; calibration mainly learns the majority
preference. No production accuracy/certification/deployment claim.37tests+8Chrome
checks pass; source and split hashes verified, including identical input/logit
parity across real independent incidents. UI now serves queue-pilot audit-v1,
read-only correction archive preserved.8B remains paused; no other session touched.
Frozen artifacts/scripts and limits in LOCAL_CALIBRATION.md. Do not tune against
this examined heldout set and describe a later run as independent validation.


2026-10-23 — structured pre-action exploration COMPLETE (fitting-only).
User authorized brief current-2B inference, not 8B training. Fresh artifacts:
sweep/out/codegraph-judgment/queue-structured-20261023/{study-plan,results,manifest}.json.
Exactly 25 original fit groups; old 16 examined test groups excluded before extraction.
Original quotes retained; literal structured matches carry exact quote evidence;
unrecorded fields explicitly unknown. No outcome/current-description fields used.
Original versus structured pack: 16/25 versus13/25 accuracy; AUC .64 versus .78;
balanced accuracy .625 versus .70. Fixed-penalty fivefold stratified group OOF
calibration: both19/25 (76%), discard1/5; structured Brier .310/logloss .484 versus
original .353/.530. Jeffreys class-prior baseline20/25 (80%), Brier .320/logloss .501.
Simple exploratory rules decide6/25, all6 match source actions; prior fallback22/25.
Rule design saw fitting cases/labels: neither six-of-six nor22/25 validates rules.
Seven cases are sparse alerts/symptoms. Ownership mismatch coverage0/25; no facts
were inferred to fill gaps. All figures exploratory; no independent accuracy gain.
Ranking is mildly encouraging, accuracy is not. Next useful experiment is a fresh,
independent, timestamp-audited holdout with both outcomes; freeze these features,
prompt, rules, and calibration procedure first. No retuning on old16 or claiming
this cross-validation validates human feature selection. No further GPU run started.
25 native captures+8 controls took4.469s; parity probability delta0, flips0; worker
exited. 15 targeted Python tests PASS. Frozen input/output hashes verified.
No model weights, shipped pack, certificate, UI or external systems changed.
Other sessions untouched; 8B stays paused. Existing audit UI shows prior pilot,
not these new fitting-only results. See results.json for detailed class metrics.

2026-10-23 — fresh validation admission audit COMPLETE; evidence gate FAILED.
Frozen structured extractor/prompt/rules and affine(a=.9827921721,b=1.2155592281)
from original25 fit groups BEFORE fresh searches. Pack choice temperature1.3810558844;
Jeffreys reprocess prior20.5/26. Old361 full stories excluded, old16 test remains spent.
Artifacts: sweep/out/codegraph-judgment/queue-fresh-validation-20261023/.
16 prespecified/extended searches fully paginated (43 pages; no pagination truncation),
377 unseen full stories saved sanitized,14 selected histories audited. This is bounded
search coverage, NOT all Codegraph. Four source-backed independent groups admitted:
113637 discard(diagnostic);119161 reprocess(diagnostic);116075 reprocess(queue-count
symptom);132446 reprocess(customer symptom). All exact input quotes precede outcome
record; no same-note diagnostic enrichment. Actual execution clocks often unavailable.
Full formal-link/parent/shared-PagerDuty/shared-MR/commit graph over old+new snapshots
excludes24 new stories dependent on prior work. Four admitted remain independent.
No_verified_completed_queue_disposition is a screening exclusion, not an exhaustive
historical proof of no action. Explicit exclusions distinguish proposals, action before
first record, domain portfolio purges and unclear final mechanisms.
Gate requires20 groups,>=5 per mechanism,>=10 diagnostic. Actual4 groups (3reprocess,
1discard),2diagnostic. Insufficient: NO scoring, comparison, fitting or GPU work started.
Fresh cases quarantined; do not tune on them. Never relax timing to fill the cohort.
prospective-case-template.json captures immutable pre-action checkpoint, source links,
related incidents/fixes and separate post-action outcome; collect all eligible consecutive
incidents. Need at least16 additional independent groups, including4 discard and8
additional diagnostic groups, before this small pilot can run. More needed for precision.
New sweep/queue_validation.py records admission gate, evidence and dependence audit;
explicit past return to consumer queue is accepted, generic 'cleared' alone is not.
19 targeted tests PASS; all artifact/source hashes and four timelines verified.
8B paused; other sessions untouched; existing UI still displays prior25-label pilot.

2026-10-23 — prospective collection and target-utility review SET UP.
Local UI http://127.0.0.1:8766/ is RUNNING. Own PID34912, saved under
sweep/out/codegraph-judgment/queue-prospective-20261023/server.pid and server.log.
New sweep/queue_collection.py + judgment_review_ui/collection.html. Fresh append-only
collection.sqlite3 starts EMPTY. No synthetic QA or historical cases in live journal.
User saves literal observations and source links BEFORE queue action. Current local
UTC checkpoint immutable; new decision checkpoint IDs for changed facts, same family.
Separate assessment supports reprocess_only/discard_only/either/insufficient, confidence,
rationale and checkpoint-only attestation. Other reviewers' answers and actual action
absent from assessment API. Reviewer identity/blinding self-reported, not authenticated.
Outcome requires explicit source quote and reported action start strictly after saved
checkpoint and no later than now. Success separate. SQLite immutable triggers + journal
hash chain check; not independent trusted execution-time proof. Low confidence<.8,
either/insufficient cases get review_needed in export. No automatic fitting promotion.
Target review begins at20 completed cases with2 reviewers; candidate threshold80%
single-action agreement is a discussion trigger, not validation. Inspect both-acceptable,
insufficient evidence, reviewer disagreement and observed-action disagreement. Family,
source timing and class support audits still mandatory before any scoring.
No automatic ingestion is connected. This is a local capture/review tool, not a claim
that future cases are accumulating unattended. No messages sent or systems changed.
Available Codegraph tools have no Slack/PagerDuty reader. Limited read-only probes:
resolved Grafana incidents(limit10) returnednone; DomainInventoryApi Republish logs
2026-10-22T00Z..2026-10-23T00Z returned0. Not exhaustive; source-access-probes.json saved.
Existing four historical candidates remain quarantined; old audit UI8765 unchanged.
Frozen20-group/5-per-label/10-diagnostic validation gate remains; model unchanged.
25 targeted Python tests +10 isolated browser workflow/security/mobile checks PASS.
Live HTTP checks PASS, empty journal. protocol.json pins code/UI/test hashes; validation
and desktop/mobile screenshots saved. 8B paused, GPU untouched; other sessions untouched.
Export: .venv-train/bin/python -m sweep.queue_collection export --db
sweep/out/codegraph-judgment/queue-prospective-20261023/collection.sqlite3
