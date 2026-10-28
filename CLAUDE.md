# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

LARS SDK: typed decisions (`noul` yes/no, `choice`, `score`, `multi`) answered in-process by a small
llama.cpp model that never generates text — it reads next-token logits over option letters after a
prompt ending in `Answer:`, turns them into a distribution over the declared options, then
calibrates. Bindings for TypeScript (`@moedex/lars`, `ts/`), C# (`Moedex.Lars`, `dotnet/`) and Rust
(`moedex-lars`, `rust/`) all sit on one Rust core. Status: 0.1.0-rc.1, nothing published; packs
are opened by path (`models/packs/<name>/<version>`, gitignored).

`README.md` is the API, `USING.md` the consumer handoff, `docs/packs.md` and `docs/calibration.md`
the pack format and calibration semantics, `PLAN.md` the road to 1.0, `RESULTS.md` every
measurement. `HANDOFF.md` is the running research log (newest entry first) — read its top entries
before touching `sweep/` or `training/`: research is currently paused, and no GPU runs, pack
promotion or artifact deletion should happen unless the user asks.

## Architecture

```
ts (napi-rs: crates/lars-node)   dotnet (P/Invoke: crates/lars-ffi, include/lars.h)   rust/ (direct)
                         └──────────────── lars-runtime ────────────────┘
lars-runtime   packs (resolve/download/sha256/cache), one worker thread per loaded model, per-handle calibrators
lars-core      request schema, prompt rendering, reduction to answers, calibration, eval — no llama.cpp
lars-llama     the llama.cpp Backend; lars-llama-sys = bindgen bindings + statically linked submodule
lars-dev       developer CLI (ask/eval/calibrate/capture/bench/parity/pack/prompts); `--model mock` uses MockBackend
```

- **Engine logic lives only in Rust.** Language bindings parse/forward and never reimplement
  rendering, reduction or calibration. The language boundary is JSON (the C ABI takes and returns
  request/response JSON).
- **Byte-exact parity with the Python reference engine is the core invariant.** A calibrator is only
  valid for one combination of weights, quantization, runtime, prompt bytes and label tokens, so
  `lars-core` reproduces the reference prompts byte for byte and its numbers to the last digit
  (including `py.rs`/`pyrandom.rs`, which mimic Python formatting and `random` shuffles). Any change
  to rendering, templates, layout or reduction invalidates calibrators and pack measurements.
  `fixtures/data/*.json` are reference outputs from the Python engine (`fixtures/generate.py`, pinned
  commit `ab59a0b`); `cargo test -p lars-core` checks against them. Don't hand-edit fixtures.
- **Runtime threading:** one worker thread owns each llama.cpp context and runs one request at a time
  from a queue; `Lars` handles are cheap, `Send + Sync`, and handles on the same pack+options share a
  worker. Every submitted request gets exactly one reply (answer, error, or `cancelled`). Shared
  prefix (system prompt + input) is computed once and a request's questions decode in one batch.
- **Calibration files** have the same format in all three languages and are bound to the predicate's
  question and its `stateVersion`; keep the three bindings in sync when changing predicate behaviour.
- `training/src/lars/` is the Python reference engine and MLX LoRA trainer (recovered copy; the
  original lars checkout is gone). `sweep/` holds the model-size sweep, training/evaluation scripts
  and research experiments; outputs go to gitignored `sweep/out/` (very large).

## Commands

Requires Rust stable (edition 2024, MSRV 1.85), CMake, a C/C++ toolchain, the llama.cpp submodule
(`git submodule update --init`), Node 22+ with pnpm, and .NET 10.

```sh
# Rust (CI runs exactly these)
cargo fmt --all --check                       # rustfmt.toml: max_width 120
cargo clippy -p lars-core --all-targets -- -D warnings
cargo clippy -p lars-llama -p lars-dev -p lars-runtime -p lars-ffi -p moedex-lars -p moedex-lars-derive --all-targets -- -D warnings
cargo test -p lars-core                       # no llama.cpp build needed; fast
cargo test --release -p lars-llama -p lars-runtime -p lars-ffi -p moedex-lars
cargo test -p lars-core --test render <name>  # a single test (tests/: calibration, engine, eval, render)
cargo build --release -p lars-dev

# TypeScript (from ts/)
pnpm install && pnpm build:native             # builds crates/lars-node, copies to ts/native/
pnpm typecheck && pnpm test                   # vitest; single test: pnpm vitest run -t "<name>"

# C#
cargo build --release -p lars-ffi
LARS_NATIVE=$PWD/target/release/liblars.dylib dotnet test dotnet/Moedex.Lars.Tests   # --filter <name> for one

# Python training/research tests (venv from `sh training/setup.sh`)
.venv-train/bin/python -m unittest discover -s training/tests -v
```

- All SDK suites run on a mock pack with no model. `LARS_TEST_PACK=<pack dir>` adds real-model tests;
  `LARS_TEST_GGUF=<gguf>` enables `crates/lars-llama/tests/parity.rs` against
  `fixtures/data/llama-*.json`.
- Changing the C ABI in `crates/lars-ffi` requires regenerating `include/lars.h`; CI diffs it:
  `cbindgen --config crates/lars-ffi/cbindgen.toml --crate lars-ffi --output crates/lars-ffi/include/lars.h` (cbindgen 0.29.4).
- After bumping the llama.cpp submodule, run `crates/lars-llama-sys/regenerate-bindings.sh`.
- Pack env vars: `LARS_PACK_DIR` (cache), `LARS_PACK_URL` (download base), `LARS_OFFLINE=1`.
- Latency: `bench/run.sh <model.gguf>` — only meaningful on a quiet machine (the M5 Max is shared
  with training jobs).
- Python tests touching Flash (`training/tests/test_flash_*`) need a Flash teacher build; see
  `sweep/flash_teacher/README.md` and the 2026-10-23 entry in `HANDOFF.md` for the mock-only CMake
  build (`-DFLASH_BUILD_RUNTIME=OFF`) and forcing MLX to CPU.
