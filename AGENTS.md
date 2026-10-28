# Repository Guidelines

## Project Structure & Module Organization

`crates/` contains the Rust core, llama.cpp bindings, runtime, C ABI, Node addon, and developer CLI. `rust/` contains the public Rust SDK and derive crate; `ts/src/` and `dotnet/Moedex.Lars/` contain the TypeScript and C# SDKs. Tests live beside their packages in `tests/`, `ts/test/`, and `dotnet/Moedex.Lars.Tests/`. Reference data lives in `fixtures/data/`; usage examples are under `examples/` and SDK directories. `docs/` documents calibration and packs; `sweep/` and `training/` contain research tooling.

## Build, Test, and Development Commands

Use stable Rust, CMake, and a C/C++ toolchain; TypeScript needs Node 22+ and pnpm, and C# needs .NET 10. Initialize native dependencies with `git submodule update --init --recursive`.

- `cargo test -p lars-core`: validate the core against reference fixtures.
- `cargo test -p moedex-lars`: test the Rust SDK.
- `cargo build --release -p lars-dev`: build the developer CLI.
- In `ts/`, run `pnpm install --frozen-lockfile`, then `pnpm build:native`, `pnpm build`, `pnpm typecheck`, and `pnpm test` to install, build, and validate the SDK.
- From the root, run `cargo build --release -p lars-ffi`, then `LARS_NATIVE="$PWD/target/release/liblars.dylib" dotnet test dotnet/Moedex.Lars.Tests -c Release` on macOS; use the platform's native library filename elsewhere.

## Coding Style & Naming Conventions

Use four-space indentation in Rust/C# and two spaces in TypeScript. Run `cargo fmt --all --check`; `rustfmt.toml` sets a 120-column width. CI runs Clippy with warnings denied. TypeScript uses strict NodeNext modules: retain `.js` import extensions, camelCase members, and PascalCase types. Follow Rust snake_case and C# PascalCase conventions.

## Testing Guidelines

Use Rust's built-in tests, Vitest (`ts/test/**/*.test.ts`), and xUnit (`LarsTests.cs`). Add behavioral regression tests in the affected package. Mock packs permit testing without models; `LARS_TEST_PACK` enables real-pack tests and `LARS_TEST_GGUF` enables llama.cpp model tests. No numeric coverage threshold is enforced in CI.

## Commit & Pull Request Guidelines

History uses short, imperative subjects such as “Fix the docs job”; no mandatory prefix is apparent. Keep commits focused. PRs should explain the behavior changed, link relevant issues, and list validation results and model-dependent tests skipped. Include screenshots for UI changes.

## Agent-Specific Instructions

Never suspend, terminate, or alter another terminal session or its processes. Control only jobs started by the current session. If resources are contested, wait, reduce your own workload, or report contention; never create suspension timers.
