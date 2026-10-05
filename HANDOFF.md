# Handoff, 2026-10-05 evening

## Running overnight (unattended)

`dev.lars-sdk.overnight` (`sweep/overnight.sh`, under launchd and caffeinate) started at 15:08.
Log: `sweep/out/overnight.log`; results summary: `sweep/out/overnight-results.txt`. It unloads
itself when done. Stop it with `launchctl bootout gui/$(id -u)/dev.lars-sdk.overnight`.

| step | what | expected |
|---|---|---|
| 1 | LoRA on **Qwen3-8B**, corpus E + 1,000 typed-decisions train cases, the corpus-E recipe, 4,948 steps (`sweep/out/train-Qwen3-8B-lora-etd-s0.log`) | done about 00:15 |
| 2 | fuse, convert, imatrix Q4_K_M (5.0 GB) and Q3_K_M; jev-bench capture and report for each | about 02:00 |
| 3 | pack `models/packs/lars-8b-dev/0.0.1`; typed-decisions test and train through it, with hidden states; zero-shot score and per-predicate heads into `overnight-results.txt` | about 03:30 |
| 4 | the untrained Qwen3-8B at Q4_K_M, captured as the zero-shot reference | about 04:30 |
| 5 | LoRA on **Qwen3.5-2B**, same data (about 10 h), then fuse, convert, capture | into the afternoon; stopping it loses nothing earlier |

Training cannot resume. If the machine reboots, launchd restarts the script, finished steps
are skipped, and an unfinished training starts over.

**What to compare in the morning:**
- `Qwen3-8B-lora-etd-s0-Q4_K_M` against the 4B pack: jev-bench 0.736 pooled, Wilson coverage
  17.5%; and against the 30B anchor: 0.761, 21.1%.
- On typed-decisions: the 4B pack scored 0.607 zero-shot. The 8B trained on typed-decisions
  train, so its number belongs in the leaderboard's "fitted" table (OpenDecider-small-td 0.792).
- The 8B's own zero-shot row, to separate the size gain from the LoRA gain.

## Decided today

- **Per-predicate heads are in the SDK** (`calibrate`: pack/fitted → recal → head, each step
  significant on 5-fold CV). typed-decisions with 100 labels: 0.6B 0.44 → 0.66, 4B 0.61 → 0.72.
  In-domain (jev-bench), they add 0.5–1 point and do no net harm. RESULTS.md has the tables.
- **Training on typed-decisions train is approved.** The test split stays held out.
- **The Qwen3.5-9B LoRA is infeasible on this machine for now.** mlx_lm's gated-delta kernel
  has no gradient. lars now has a chunked, differentiable version (`f42781a`, exact against the
  reference loop), but the 9B trains at about 23 s per step at 2,048-token batches, 89 GB peak:
  about 73 hours for one epoch. The 2B is about 6.75 s per step, about 10 hours.
- **Default tier:** still open. Suggestion in the conversation: lead with labels, so the 0.6B or
  1.7B plus `calibrate` is the default, and the 4B is the "no labels" tier.

## Open

1. Morning: read `overnight-results.txt` and `summary.md`; decide whether the 8B becomes a tier
   (rule: at least 2 points over the tier it would replace, Wilson coverage no worse).
2. Rebuild the three packs from the imatrix Q4_K_M files (a small, free gain).
3. If the 8B with typed-decisions helps, retrain the 4B the same way to separate data from size.
4. Phase 6: the published packs need your sign-off before anything goes to the Hub.
5. Quiet-machine latency (never measured), `lars pull`, an API reference, TS cancellation,
   ASan on Linux, a C# enum source generator.

## Things to know

- **lars repo:** local branch `codex-wip-2026-10-02`, not pushed. Today's commits there:
  `3485841` (training records keep noul yes/no criteria) and `f42781a` (chunked gated delta rule).
  Adapters in `checkpoints/`: `lora-{1.7b,0.6b}-sdk-e-s0`, tonight `lora-8b-sdk-etd-s0` and
  `lora-qwen3.5-2b-sdk-etd-s0`.
- **typed-decisions:** `sweep/typed_decisions.py --convert` (eval JSONL), `--records` (training
  records: 1,000 train cases + 200 for checkpoint selection), and scoring. Data in
  `sweep/out/typed-decisions/`.
- **Heads offline:** `sweep/heads_capture.sh` (hidden states through a pack) and `sweep/heads.py`.
  Hidden states for the 4B and 0.6B packs are in `sweep/out/heads/`.
- **`models/`:** large and gitignored. The 30B GGUFs (k128 18.6 GB, k64 9.8 GB, k64-IQ3 7.7 GB)
  and the Qwen3.5 BF16 files can be deleted if space is needed.
- Disabled launchd plists are in `~/Library/LaunchAgents/lars-sdk-disabled/`.
