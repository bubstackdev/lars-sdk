#!/usr/bin/env python3
"""Verify an installed alpha and rebuild its source in isolated directories. Publishes nothing."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tarfile
import tempfile
import time


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""): h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--workdir", type=Path)
    args = parser.parse_args()
    bundle, pack = args.bundle.resolve(), args.pack.resolve()
    if platform.system() != "Darwin" or platform.machine() != "arm64": parser.error("alpha verification is macOS arm64 only")
    for line in (bundle / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split("  ", 1); path = (bundle / name).resolve()
        if not path.is_relative_to(bundle) or not path.is_file() or sha(path) != digest: parser.error(f"checksum mismatch: {name}")
    build = json.loads((bundle / "build.json").read_text())
    work = args.workdir.resolve() if args.workdir else Path(tempfile.mkdtemp(prefix="lars-alpha-clean-"))
    if args.workdir:
        if work.exists(): parser.error("workdir must not exist")
        work.mkdir(parents=True)
    env = os.environ.copy()
    for key in ["LARS_NATIVE", "LARS_PACK_DIR", "CARGO_TARGET_DIR", "RUSTFLAGS", "CARGO_ENCODED_RUSTFLAGS"]: env.pop(key, None)
    log = work / "verification.log"
    def run(argv, cwd, extra=None):
        print("Running: " + " ".join(map(str, argv)), flush=True)
        with log.open("a") as out:
            out.write("\n$ " + " ".join(map(str, argv)) + "\n"); out.flush()
            subprocess.run(list(map(str, argv)), cwd=cwd, env={**env, **(extra or {})}, stdout=out, stderr=subprocess.STDOUT, check=True)
    started = time.monotonic()
    consumer = work / "consumer"; consumer.mkdir()
    (consumer / "package.json").write_text('{"name":"lars-alpha-clean-consumer","private":true,"type":"module"}\n')
    run(["npm", "install", "--offline", "--ignore-scripts", "--no-audit", "--no-fund", "--package-lock=false", bundle / build["package"]], consumer)
    shutil.copyfile(bundle / "evaluate.mjs", consumer / "evaluate.mjs")
    shutil.copytree(bundle / "fixtures", consumer / "fixtures")
    run(["node", "evaluate.mjs", "--pack", pack, "--out", "installed-result.json"], consumer)
    installed = json.loads((consumer / "installed-result.json").read_text())
    if installed["summary"]["errors"] or installed["summary"]["completed"] != 50: raise RuntimeError("installed package did not complete the expected suite")
    extracted = work / "source"; extracted.mkdir()
    with tarfile.open(bundle / build["source_archive"], "r:gz") as archive:
        for member in archive.getmembers():
            if not member.isfile() or not (extracted / member.name).resolve().is_relative_to(extracted): raise RuntimeError("unsafe source archive member")
        archive.extractall(extracted, filter="data")
    source = extracted / "lars-image-alpha-source"
    if (source / ".git").exists() or (source / "target").exists() or (source / "ts/node_modules").exists(): raise RuntimeError("source extraction is not clean")
    if json.loads((source / "ALPHA-SOURCE.json").read_text())["source_tree_sha256"] != build["source"]["source_tree_sha256"]: raise RuntimeError("source identity mismatch")
    run(["npx", "--yes", "pnpm@11.0.0", "install", "--frozen-lockfile"], source / "ts")
    run(["node", "ts/scripts/build-native.mjs"], source, {"CARGO_TARGET_DIR": str(source / "target")})
    run(["npm", "run", "build"], source / "ts")
    run(["node", "packaging/vision-alpha/evaluate.mjs", "--pack", pack, "--suite", "fixtures/vision-alpha/suite.json",
         "--sdk", "ts/dist/index.js", "--controls", "none", "--out", "source-result.json"], source)
    rebuilt = json.loads((source / "source-result.json").read_text())
    if rebuilt["summary"]["errors"] or rebuilt["summary"]["completed"] != 50: raise RuntimeError("rebuilt source did not complete the expected suite")
    original = {r["id"]: r["prediction"] for r in installed["records"] if r["mode"] == "intact"}
    disagreements = [r["id"] for r in rebuilt["records"] if r["prediction"] != original[r["id"]]]
    report = {
        "format": 1, "status": "passed", "host": installed["host"], "seconds": time.monotonic() - started,
        "package_sha256": sha(bundle / build["package"]), "source_archive_sha256": sha(bundle / build["source_archive"]),
        "source_tree_sha256": build["source"]["source_tree_sha256"], "model_lock_sha256": sha(bundle / "model.lock.json"),
        "installed": installed["summary"], "rebuilt_from_clean_source": rebuilt["summary"],
        "prediction_disagreements": disagreements,
        "scope": "Source extraction had no Git metadata, native artifacts, target directory, or node_modules. Cargo/npm registries used normal caches/network. Installed consumer required no Rust/compiler and installed the standalone tarball offline.",
    }
    if disagreements: raise RuntimeError(f"installed and rebuilt predictions differ: {disagreements}")
    (bundle / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    for name, path in [("installed-validation.json", consumer / "installed-result.json"), ("source-validation.json", source / "source-result.json")]: shutil.copyfile(path, bundle / name)
    entries = [f"{sha(p)}  {p.relative_to(bundle).as_posix()}" for p in sorted(bundle.rglob("*")) if p.is_file() and p.name != "SHA256SUMS"]
    (bundle / "SHA256SUMS").write_text("\n".join(entries) + "\n")
    print(f"Verified installed package and clean source. Details: {bundle / 'verification.json'}")
    print(f"Retained build/test log: {log}")


if __name__ == "__main__": main()
