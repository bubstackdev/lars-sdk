# Handoff, 2026-10-02 afternoon

Phase 2b runs unattended under two launchd jobs. Neither needs this session or the network,
both survive a reboot (they restart at login and resume), and each unloads itself when done.
Keep the Mac on power with the lid open: `caffeinate` blocks idle and system sleep, but not
closing the lid on battery.

## What is running

| launchd job | script | doing | expected done |
|---|---|---|---|
| `dev.lars-sdk.sweep` | `sweep/weekend.sh` | Metal queue (`sweep/out/queue-metal.log`): zero-shot 4B v1, v2 and Q8_0, latency benchmarks, then Q8_0 v2 for every size | Friday evening |
| | | CPU queue (`queue-cpu.log`): Q4_K_M v1 for every size, then Q8_0 v1. This feeds the CPU/Metal calibrator-transfer check. | Sunday night or Monday |
| `dev.lars-sdk.train` | `sweep/train.sh` (`sweep/out/train.log`) | the 4B LoRA as GGUF (done); training the 1.7B LoRA (started 14:22, about 2 h); then fuse, convert and capture; then the same for the 0.6B | 1.7B about 17:30, 0.6B about 20:00 |

Results land without anyone watching:
- `sweep/out/summary.md` and `summary.json` are rewritten after every finished run. They
  cover every complete run, the few-label curves, and CPU↔Metal transfer once a size has
  both paths.
- Each run's analysis is cached in its own directory; the jev-bench rows stay out of git.
- New adapters go to `lars/checkpoints/lora-1.7b-sdk-e-s0` and `lora-0.6b-sdk-e-s0`, with
  training logs in `sweep/out/train-*.log`.

| | command |
|---|---|
| progress | `tail sweep/out/*.log`, or open `sweep/out/summary.md` |
| pause both sweep queues | `pkill -STOP -f sweep/queue.sh`; `-CONT` resumes |
| stop the sweep | `launchctl bootout gui/$(id -u)/dev.lars-sdk.sweep` |
| stop training | `launchctl bootout gui/$(id -u)/dev.lars-sdk.train` |

## Where things stand

Details are in `RESULTS.md`; the plan with ticked boxes is in `PLAN.md`.
- **Phases 0–2 are done.** The Rust core matches the Python reference exactly, and the
  llama.cpp backend is bit-identical to the reference's with flash attention off.
- **Zero-shot sizes:** the 4B-2507 reaches 0.673 per-config calibrated. The 0.6B (0.40)
  and 1.7B (0.46) are weak. That's the models, checked against the reference engine on the
  original weights; it isn't the pipeline.
- **Question-first layout (v2)** is worse on both small models. It's dropped unless the 4B
  v2 run disagrees.
- **The prototype's 4B LoRA as GGUF** matches its MLX numbers: Q8_0 0.743 and Q4_K_M 0.736,
  pooled. **A 4B Q4_K_M pack (2.5 GB) is shippable on Metal.**
- **Error budgets need the conservative threshold** (Wilson bound). The empirical one runs
  at 7–20% error on test, so `calibrate({ maxError })` must use the bound and report when
  there aren't enough labels to certify a budget.

## Decisions waiting on the running jobs

1. **The CPU tier:** whether the 1.7B or 0.6B LoRA comes close enough to the 4B. The 4B
   gained 8.8 points from its LoRA. On CPU the 4B is about 0.5 s per noul, too slow for an
   inline `if`.
2. **One calibrator or one per compute path,** and whether batching becomes the default.
   The CPU/Metal transfer rows in `summary.md` decide both.
3. **Quiet-machine latency** for the chosen sizes. Pause the queues, run
   `bench/run.sh models/<model>.gguf`, then resume them.

## Next work that doesn't wait on the sweep

Phase 3 (`PLAN.md`): `lars-ffi` (C ABI, worker thread, async callbacks), the pack format and
the downloader. It needs no model decision; the 4B LoRA Q4_K_M can be the first real pack
for testing.

## Things to know

- **lars repo:** checked out on local branch `codex-wip-2026-10-02` (`b4a1180`, Codex's
  reviewed work, not pushed). Training runs from it. Don't switch branches while
  `dev.lars-sdk.train` runs.
- **`models/`** holds about 40 GB of GGUFs and fused models, gitignored.
- **CPU latency taken during the sweep is meaningless:** two 12-thread jobs on 18 cores.
