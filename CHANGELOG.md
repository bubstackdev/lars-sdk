# Changelog

## 0.1.0-rc.1 (2026-10-07, release candidate, not published)

The first release candidate of the LARS SDK. Typed decisions as functions in your code,
answered by a small model running in your process, from TypeScript, C# or Rust.

### Packages

| registry | package | |
|---|---|---|
| npm | `@moedex/lars`, plus `@moedex/lars-<platform>-<arch>` | Node 22+, Bun; `npx lars pull` |
| NuGet | `Moedex.Lars` | .NET 10; native AOT works |
| crates.io | `moedex-lars` (with `moedex-lars-derive`, `lars-runtime`, `lars-llama`, `lars-llama-sys`, `lars-core`) | the first build compiles llama.cpp (CMake, a C++ compiler) |

### Packs

Served from the `sdk/` folder of the `moedex/lars` Hub repository. Each has a model card and a
NOTICE with its training data's licenses.

| pack | base | size | jev-bench | certified at 5% | use |
|---|---|---|---|---|---|
| `lars-2b` 0.1.0 | Qwen3.5-2B, Q4_K_M | 1.3 GB | 0.716 | 15.3% | the default |
| `lars-8b` 0.1.0 | Qwen3-8B, Q3_K_M | 4.1 GB | 0.748 | 19.0% | the most automation at a strict error budget |
| `lars-0.6b` 0.1.0 | Qwen3-0.6B, Q4_K_M | 0.4 GB | 0.650 | 6.4% | the smallest, with labels to calibrate on |

For reference, the hosted Jev model's published jev-bench number is 0.733.

### What's in it

- **Predicates:** `noul`, `choice`, `score` and `multi`, each with `decide` for the probabilities
  behind an answer.
  - **Several at once:** `ask` puts several predicates to one input in one pass, with optional
    constraints between yes/no predicates.
  - **State selectors** pick the fields the model reads, with a `stateVersion`.
- **Calibration:** calibrate a predicate on your own labels.
  - **What it tries:** a refit, a recalibration and a hidden-state head. Each step is kept only
    when significantly better under cross-validation.
  - **Certified thresholds:** a threshold certified for an error budget, chosen in a fixed sequence
    so the 95% claim holds. When certification fails, nothing is accepted.
  - **Saving:** calibrations save to one file format all three SDKs read. Each calibration is bound
    to its pack, its question and its state version.
- **Cancellation:** `AbortSignal` (TypeScript), `CancellationToken` (C#), or dropping the `Pending`
  (Rust).
- **Packs:** downloaded once, verified by sha256 and cached. `pull` fetches the newest ahead of
  time, and `LARS_OFFLINE=1` never downloads.
- **Prefix reuse:** the input is read once per request and reused across calls. On Qwen3.5 this
  works through a snapshot, bit-identical to re-reading.

### Verified for this candidate

On macOS arm64 (M5 Max), from the built artifacts:
- **Tests:** the Rust, TypeScript and C# suites pass on mock packs, and the real-pack tests pass on
  all three packs in all three languages. Each pack was downloaded with `pull` from a local server
  laid out like the Hub, then opened offline.
- **Install tests:** the npm tarballs and the NuGet package install into fresh projects.
- **Crates:** the six crates package and build from their own tarballs. `lars-llama-sys` is
  6.1 MB compressed, under crates.io's 10 MB limit.
- **AddressSanitizer:** the core, runtime and Rust SDK suites pass under it.

### Not verified, and known limits

- **Linux and Windows aren't built or tested for this candidate.** GitHub Actions hasn't started a
  job since 2026-10-06 (an account billing limit). The package workflow builds and install-tests
  them on five runners once Actions runs again. This candidate's artifacts are macOS only, arm64
  and x64.
- **The top tier may change.** A Qwen3.5-4B pack trains overnight on 2026-10-07. If it matches the
  8B at about 2.5 GB, it replaces `lars-8b` before 0.1.0.
- **typed-decisions scores of the 2B and 8B are in-domain:** they trained on its train split. They
  are also above the dataset's teacher self-agreement, 0.735. They are not accuracy on unfamiliar
  decisions.
- **Certified thresholds describe data like their labels.** The labels that choose a calibrator
  also certify its threshold; for strict use, check on a held-out set (docs/calibration.md).
- **Licensing caveats:**
  - **Open-Jev:** its question wording was adapted from TypeSafe's docs, whose license wasn't
    verified.
  - **typed-decisions:** its labels come from an unnamed teacher model.
  - Both are recorded in the packs' cards.
- **The models:** weak on world knowledge, weakest on exact levels of fine-grained scales, and
  English only.
