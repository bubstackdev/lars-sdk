# Handoff, 2026-10-05 morning

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

## Open

1. Quiet-machine latency for the candidate packs. The CPU tier choice, 0.6B or 1.7B, depends on it.
2. imatrix quantization for the LoRA packs.
3. Batching on by default in `LlamaOptions`.
4. Phase 3: `lars-ffi`, the pack format, the downloader.

## Things to know

- **lars repo:** still on local branch `codex-wip-2026-10-02` (`b4a1180`, not pushed). The
  small-model adapters are `checkpoints/lora-1.7b-sdk-e-s0` and `lora-0.6b-sdk-e-s0`.
- **`models/`:** about 50 GB of GGUFs and fused models, gitignored. The candidate packs are
  `models/Qwen3-{0.6B,1.7B,4B-Instruct-2507}-lora-e-s0-{Q8_0,Q4_K_M}.gguf`.
