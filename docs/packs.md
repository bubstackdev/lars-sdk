# Packs

A pack is a directory holding everything that determines a model's probabilities. A calibrator
is valid for exactly one pack. Changing the weights, the quantization, the chat template, the
prompt layout, the llama.cpp build or flash attention all move logits (`RESULTS.md`, 2026-10-02),
so a pack fixes them all.

```
lars-2b/0.1.0/
  pack.json          the manifest
  model.gguf         the weights
  calibration.json   the pooled calibrator, bound to this pack
```

## Getting packs

| you give | the SDK |
|---|---|
| a path to a directory with `pack.json` | uses it as is |
| `name@version` | uses the cached copy, or downloads that version |
| `name` | uses the newest cached version, or downloads the newest published one |

Downloads come from `<LARS_PACK_URL>/<name>/<version>/`, by default the `sdk/` folder of the
`moedex/lars` Hub repository. `<base>/<name>/index.json` (`{"latest": "0.1.0"}`) names the newest
version. Every file is checked against its sha256 as it downloads. The pack is assembled in a
temporary directory and moved into the cache only once complete, so a failed download installs
nothing.

| variable | |
|---|---|
| `LARS_PACK_DIR` | the pack cache; default `~/Library/Caches/lars/packs` (macOS), `$XDG_CACHE_HOME/lars/packs` or `~/.cache/lars/packs` (Linux), `%LOCALAPPDATA%\lars\packs` (Windows) |
| `LARS_PACK_URL` | where packs are downloaded from |
| `LARS_OFFLINE=1` | never download; only cached packs or paths open |

For containers and air-gapped machines, copy a pack directory into the image and open it by path,
or set `LARS_PACK_DIR`.

On open, every file's size is checked. Its sha256 is checked too with `verify: true`; that takes
a few seconds for a 2.5 GB model.

## `pack.json`

```json
{
  "format": 1,
  "name": "lars-2b",
  "version": "0.1.0",
  "backend": "llamacpp",
  "model": {"file": "model.gguf", "bytes": 2497280960, "sha256": "...",
            "base": "Qwen/Qwen3-4B-Instruct-2507", "license": "Apache-2.0", "quantization": "Q4_K_M"},
  "calibration": {"file": "calibration.json", "sha256": "...", "fitted_on": "4120 jev-bench validation rows, pooled"},
  "template": "chatml",
  "layout": "v1",
  "runtime": {"llama_cpp": "b11349", "flash_attention": "off"},
  "evaluation": {"benchmark": "jev-bench test, through this pack", "macro_accuracy": 0.736, "macro_brier": 0.304,
                 "per_config": {}}
}
```

| field | |
|---|---|
| `format` | 1 |
| `backend` | `llamacpp`, or `mock` (no model: deterministic keyword answers, for tests) |
| `template` | a named template (`chatml`, `plain`, `gemma`, `llama3`), or one captured from the model's own chat template with thinking off: `{before_system, between, after_user, trims_content}` |
| `layout` | `v1`: system prompt, then the input, then the question. Question-first (`v2`) measured worse at every size and is not used |
| `runtime.llama_cpp` | the llama.cpp build the pack was validated on. `status()` reports `runtime_matches_pack: false` if the SDK's differs |
| `evaluation` | how the pack scored, measured through this SDK |

## Calibrators are bound to their pack

`calibration.json`, and every calibrator `calibrate` returns, records the pack it was fitted on:

```json
"pack": {"name": "lars-2b", "version": "0.1.0", "model_sha256": "..."}
```

`set_calibrator` / `loadCalibrations` refuse a calibrator that records another pack, or no pack.

## Building a pack

```sh
cargo run --release -p lars-dev -- pack --model model.gguf --template chatml \
  --name lars-2b --version 0.1.0 --data-dir <jev-bench eval data> --out packs/lars-2b/0.1.0 \
  --base Qwen/Qwen3-4B-Instruct-2507 --license Apache-2.0 --quantization Q4_K_M
```

The command:
1. hard-links the model into the pack and hashes it
2. fits the pooled calibrator through the SDK on every `*.validation.jsonl` in the data directory
3. evaluates on every `*.test.jsonl` with that calibrator
4. writes `pack.json` with the results

The packs' models come from `sweep/train.sh`: a LoRA trained with lars, fused, converted to GGUF
and quantized with this repository's pinned llama.cpp.
