# Handoff, 2026-10-06 morning

## Running (unattended), 2026-10-06

Two launchd jobs, one after the other; each unloads itself. Stop one with
`launchctl bootout gui/$(id -u)/dev.lars-sdk.<name>`.

1. `dev.lars-sdk.train4b` (`sweep/train4b.sh`, log `sweep/out/train4b.log`), started training
   07:45. The 4B trained exactly like the 8B (BF16, corpus E soft + typed-decisions train), then
   imatrix Q4_K_M, a jev-bench capture, pack `lars-4b-etd-dev` and typed-decisions
   (`sweep/out/train4b-results.txt`). It decides the top tier: keep the 4B if it lands within
   about 1 point of accuracy and 3 points of Wilson coverage of the 8B; else the 8B at Q3_K_M
   replaces it. Never both.
2. `dev.lars-sdk.train35` (`sweep/train35.sh`, log `sweep/out/train35.log`) waits for it, then
   trains Qwen3.5-2B the same way (from scratch: the overnight run was stopped at step 2,300 to
   let the 4B go first), converts with `--no-mtp` (mlx_lm drops the MTP block when it fuses), and
   captures. With lars `585ffae` (a blocked inverse in the chunked gated delta rule, 2.9x faster,
   23 GB instead of 64 GB peak) training should take about 2.5 hours instead of 7.

Overlapping GPU jobs costs a lot: the old rule ran at 16.2 s per step next to the 4B training,
against 4.7 s alone. Run one GPU job at a time.

The overnight run (`sweep/archive/overnight.sh`) finished everything for the 8B by 04:37:
- `lars-8b-dev`: jev-bench 0.749 (Q4_K_M, 5.0 GB) and 0.748 (Q3_K_M, 4.1 GB), Brier 0.285,
  Wilson coverage 27.6%.
- The untrained 8B scored 0.627.
- On typed-decisions test (it trained on train): 0.773, KL 0.095, Brier 0.050.
- Per-predicate heads on top: 0.806 at 300 labels.

## Decided today

- **Tonight, 2026-10-07:** `dev.lars-sdk.train35-4b` (`sweep/train35_4b.sh`, log
  `sweep/out/train35_4b.log`) waits until 18:00, then trains Qwen3.5-4B like the 2B.
  - **Expected:** about 8-9 h of training (48.6 GB peak in the smoke test), then the build check,
    jev-bench, pack `lars-4b35-dev` and typed-decisions, all done around 04:00.
  - **The question:** does it match the 8B (0.748) at about 2.5 GB? The smoke test passed end to
    end on the real base (check: MLX 0.775, GGUF 0.788).
- **2026-10-07 lineup (agreed):**
  - `lars-2b-dev` 0.0.1: Qwen3.5-2B, the default.
  - `lars-8b-dev` 0.0.2: Q3_K_M, the top tier.
  - `lars-0.6b-dev`: the smallest.
  - `lars-4b-dev`, `lars-1.7b-dev` and `lars-8b-dev` 0.0.1 are retired but left on disk.
  - Built by `sweep/promote.sh`.
  - Next: prefix reuse for Qwen3.5 (snapshot the recurrent state instead of re-reading the input).
- **2026-10-07 results (RESULTS.md):**
  - 4B + td 0.740, 8B + td 0.749, Qwen3.5-2B + td 0.716.
  - Top tier: the 8B (Q3_K_M), by the coverage rule.
  - The 2B is the best small model. Its pack, typed-decisions and latency come from
    `sweep/finish2b.sh` (running from 09:00; results in `sweep/out/finish2b-results.txt` and
    `sweep/out/latency-tiers.md`).
  - Builds now gate on `check` (MLX vs GGUF on 80 rows).
- **2026-10-07, two pipeline bugs fixed:**
  - **4B fuse failed offline.** mlx_lm fetches the base's README for a model card; `fuse` now
    reads the local snapshot. The 4B resumed from its finished adapter at 07:26.
  - **The fused Qwen3.5-2B was broken (0.316, against 0.493 untrained).** mlx_lm adds 1.0 to five
    zero-centered norm tensors on load, the fused save keeps the shift, and the converter adds
    1.0 again. `sweep/unshift_norms.py` now runs after every fuse. The 2B rebuilds from its
    adapter after the 4B finishes. Training was fine: selection 0.43 -> 0.778. Broken outputs
    are in `models/broken` and `sweep/out/broken`.
- **Certification fixes (review from Moe, `~/Code/moe/docs/integrations/lars-sdk-findings.md`):**
  - a failed certification accepts nothing
  - calibration files (format 2) carry their questions
  - loading replaces certificates and is atomic
  - thresholds come from a fixed sequence, valid at about 2.5%
  Coverage drops by about a fifth to a third; RESULTS.md 2026-10-06.
- **Rust SDK:** `moedex-lars` (`rust/`, with `#[derive(Options)]` in `rust/derive`), on the same
  runtime as TypeScript and C#. Molar Triage gives the same answers as TypeScript. `lars` is
  taken on crates.io. `USING.md` is the handoff for other projects.
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
