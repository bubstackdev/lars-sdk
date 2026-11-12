# Installing the GitHub prerelease

The `v0.1.0-rc.1` GitHub release provides TypeScript, C# and Rust distributions.
Nothing is published to npm, NuGet or crates.io yet. Model packs are downloaded
separately; this release does not change or retrain a model pack. Image APIs are
experimental and do not imply certified image accuracy.

Download assets from <https://github.com/bubstackdev/lars-sdk/releases/tag/v0.1.0-rc.1>.
`build.json` records the exact SDK and llama.cpp commits. Verify downloaded files
against `SHA256SUMS` (for example, `shasum -a 256 -c SHA256SUMS` after downloading all
assets). `THIRD_PARTY_LICENSES.tar.gz` contains dependency notices.

## TypeScript / Node

```sh
npm install https://github.com/bubstackdev/lars-sdk/releases/download/v0.1.0-rc.1/moedex-lars-0.1.0-rc.1.tgz
```

This standalone tarball includes native addons for macOS ARM64/x64, Linux glibc
ARM64/x64 and Windows x64. No compiler, npm account, install script or separate
native download is needed. It can also be downloaded first and installed from a
local path offline. Node 22, 24 and 26 are tested. Linux requires glibc 2.28+;
musl/Alpine is not supported. On macOS the runtime uses Metal; Linux and Windows
use CPU inference. Open a pinned pack directory for reproducible inference.

The `npm-*.tgz` assets are the separate registry-ready packages, retained for future
npm publication. They are not the standalone GitHub installer: the main registry
package expects its platform package to be available at the matching npm version.

## C# / .NET 10

Download `Moedex.Lars.0.1.0-rc.1.nupkg` into a local feed directory, then:

```sh
dotnet add package Moedex.Lars --version 0.1.0-rc.1 --source /absolute/path/to/feed
```

The package includes all five native runtimes. Use a local feed in `NuGet.Config`
for subsequent restores. No NuGet registry publication is required.

## Rust

Extract `lars-sdk-0.1.0-rc.1-source.tar.gz` and use the `rust/` directory as a path
dependency:

```toml
[dependencies]
moedex-lars = { path = "../lars-sdk-0.1.0-rc.1/rust" }
```

The archive includes the pinned llama.cpp source, workspace manifests, lockfile,
and all internal path dependencies. It needs no Git checkout or submodule fetch.
Building requires Rust, CMake and a C/C++ toolchain. External Rust dependencies
still download from crates.io unless already cached. Preserve the whole extracted
workspace, not only `rust/`. It also contains the C ABI, header and source SDK tests.

## Building and verifying a release

The `package` GitHub Actions workflow builds native libraries, prepares both npm
distributions and NuGet, archives the full pinned source, and install-tests the
artifacts on all five platforms. Rust tests run from the extracted source archive,
without Git metadata. Download its `packages` artifact: `release/` contains the
files to attach to GitHub, including checksums. Publish only after all required jobs
succeed. The workflow does not publish to any package registry.

Future npm publication should publish the five `npm-moedex-lars-<platform>-<arch>`
tarballs first, then `npm-moedex-lars-0.1.0-rc.1.tgz`, with `--access public --tag rc`.
Use those already tested tarballs; do not rebuild a different artifact for publication.
