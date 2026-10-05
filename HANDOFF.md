# Handoff, 2026-10-05

Nothing is running. The Phase 2b sweep and small-model training finished over the weekend
(sweep on Saturday at 11:12, training on Friday at 19:24) with no failures, and both launchd
jobs unloaded themselves. The launchd plists stay in `~/Library/LaunchAgents` for a re-run.

Results are in `RESULTS.md` (2026-10-05) and every row is in `sweep/out/summary.md`.

| finding | detail |
|---|---|
| LoRAs rescue the small models | pooled at Q8_0: 0.6B 0.651, 1.7B 0.683, 4B 0.743 |
| one calibrator per pack | it transfers between CPU and Metal; batching can be the default |
| layout | v2 is dropped; v1 stays |
| quantization | imatrix Q4_K_M should match Q8_0 |

## Tonight (unattended)

`dev.lars-sdk.night` (`sweep/night.sh`) is loaded and waiting under caffeinate. At 01:30 it:
1. measures quiet-machine latency for the five candidate packs, on CPU and Metal, for a new
   state and for the same state, at 6 and 12 threads, with CPU repacking on and off
   (`sweep/out/latency.md`, load average per row)
2. builds importance-matrix Q4_K_M versions of the three LoRA models from training-split
   prompts, captures them, and adds them to `sweep/out/summary.md`

Log: `sweep/out/night.log`. Stop it with `launchctl bootout gui/$(id -u)/dev.lars-sdk.night`.

## Done today

- Batching is on by default. Single-question calls stay bit-identical to the reference.
- Phase 3: `lars-runtime`, the C ABI (`lars.h`), packs with verified downloads, and the first
  pack `models/packs/lars-4b-dev/0.0.1`. Its evaluation through the core, 0.7361 / 0.3042,
  equals the Python analysis exactly. CI is green on all three OSes.

## Open

1. Choose the CPU tier (0.6B or 1.7B) from tonight's latency, then build its pack.
2. Phase 4: TypeScript (napi-rs over `lars-runtime`).
3. Phase 5: C# (P/Invoke over `lars.h`).
4. ASan on Linux in CI.

## Things to know

- **lars repo:** still on local branch `codex-wip-2026-10-02` (`b4a1180`, not pushed). The
  small-model adapters are `checkpoints/lora-1.7b-sdk-e-s0` and `lora-0.6b-sdk-e-s0`.
- **`models/`:** about 50 GB of GGUFs and fused models, gitignored. The candidate packs are
  `models/Qwen3-{0.6B,1.7B,4B-Instruct-2507}-lora-e-s0-{Q8_0,Q4_K_M}.gguf`.
