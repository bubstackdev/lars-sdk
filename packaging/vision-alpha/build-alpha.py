#!/usr/bin/env python3
"""Build an unpublished macOS arm64 npm alpha and checksummed source/tester bundle."""
import argparse
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[2]
VERSION = "0.1.0-image-alpha.1"


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def checksums(directory):
    entries = [f"{sha(p)}  {p.relative_to(directory).as_posix()}" for p in sorted(directory.rglob("*"))
               if p.is_file() and p.name != "SHA256SUMS"]
    (directory / "SHA256SUMS").write_text("\n".join(entries) + "\n")


def license_notices(destination):
    destination.mkdir()
    meta = json.loads(subprocess.check_output(["cargo", "metadata", "--locked", "--format-version", "1", "--filter-platform", "aarch64-apple-darwin"], cwd=ROOT))
    packages = []
    for package in meta["packages"]:
        source = Path(package["manifest_path"]).parent
        if source.is_relative_to(ROOT): continue
        candidates = [p for p in source.iterdir() if p.is_file() and p.name.lower().startswith(("license", "copying", "notice"))]
        if package.get("license_file"): candidates.append(source / package["license_file"])
        directory = destination / f"{package['name']}-{package['version']}"; directory.mkdir()
        names = []
        for path in sorted(set(candidates)):
            if path.is_file(): shutil.copyfile(path, directory / path.name); names.append(path.name)
        if not names and package.get("repository") == "https://github.com/napi-rs/napi-rs":
            vcs = json.loads((source / ".cargo_vcs_info.json").read_text())
            if vcs["git"]["sha1"] != "0db7ceb47d0401faa84f9a80163437ff313ecc7c":
                raise RuntimeError("review napi-rs notice after changing its revision")
            shutil.copyfile(ROOT / "packaging/vision-alpha/licenses/napi-MIT.txt", directory / "LICENSE")
            names.append("LICENSE")
        if not names: raise RuntimeError(f"review missing dependency license text: {package['name']}")
        packages.append({"name": package["name"], "version": package["version"], "license": package["license"], "files": names})
    llama = ROOT / "crates/lars-llama-sys/llama.cpp"
    native = destination / "native"; native.mkdir()
    for filename, path in [("llama.cpp-LICENSE", llama / "LICENSE"), ("ggml-LICENSE", llama / "ggml/LICENSE")]:
        if path.is_file(): shutil.copyfile(path, native / filename)
    for path in (llama / "vendor").rglob("*LICENSE*"):
        if path.is_file(): shutil.copyfile(path, native / path.relative_to(llama / "vendor").as_posix().replace("/", "-"))
    for name in ["stb/stb_image.h", "miniaudio/miniaudio.h", "sheredom/subprocess.h"]:
        text = (llama / "vendor" / name).read_text()
        notice = text[text.rfind("/*"):]
        if not any(marker in notice.lower() for marker in ["permission", "public domain", "unlicense"]):
            # subprocess.h keeps its license at the beginning, unlike stb/miniaudio.
            notice = text
        (native / (name.replace("/", "-") + "-notice.txt")).write_text(notice)
    (destination / "dependencies.json").write_text(json.dumps(packages, indent=2) + "\n")


def source_snapshot(target):
    def listed(directory):
        raw = subprocess.check_output(["git", "-C", str(directory), "ls-files", "--cached", "--others", "--exclude-standard", "-z"])
        return [os.fsdecode(p) for p in raw.split(b"\0") if p]
    paths = [Path(p) for p in listed(ROOT) if (ROOT / p).is_file()]
    submodule = Path("crates/lars-llama-sys/llama.cpp")
    paths += [submodule / p for p in listed(ROOT / submodule) if (ROOT / submodule / p).is_file()]
    paths = sorted(set(paths))
    fingerprint = hashlib.sha256()
    entries = []
    for path in paths:
        full = ROOT / path
        if not full.resolve().is_relative_to(ROOT):
            raise RuntimeError(f"source path resolves outside checkout: {path}")
        mode = 0o755 if full.stat().st_mode & 0o111 else 0o644
        digest = sha(full)
        fingerprint.update(f"{path.as_posix()}\0{mode:o}\0{digest}\n".encode())
        entries.append((path, mode))
    identity = {
        "format": 1, "base_commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
        "source_tree_sha256": fingerprint.hexdigest(), "files": len(entries),
        "includes_working_tree_changes": bool(subprocess.check_output(["git", "-C", str(ROOT), "status", "--porcelain"])),
        "llama_cpp": (ROOT / "crates/lars-llama-sys/llama-cpp-revision.txt").read_text().strip(),
        "note": "The source tree hash identifies this snapshot, including uncommitted implementation files; the base commit alone does not.",
    }
    with target.open("wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as zipped, tarfile.open(fileobj=zipped, mode="w") as archive:
        for path, mode in entries:
            data = (ROOT / path).read_bytes()
            info = tarfile.TarInfo("lars-image-alpha-source/" + path.as_posix())
            info.size = len(data); info.mode = mode; info.mtime = 0
            archive.addfile(info, io.BytesIO(data))
        data = (json.dumps(identity, indent=2) + "\n").encode()
        info = tarfile.TarInfo("lars-image-alpha-source/ALPHA-SOURCE.json"); info.size = len(data); info.mode = 0o644
        archive.addfile(info, io.BytesIO(data))
    return identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "dist" / f"vision-alpha-{VERSION}")
    parser.add_argument("--skip-build", action="store_true", help="Use an already built release addon and TypeScript output")
    args = parser.parse_args()
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        parser.error("this alpha artifact is restricted to macOS arm64")
    if args.out.exists(): parser.error("output already exists; choose a new directory")
    if not args.skip_build:
        subprocess.run(["node", "ts/scripts/build-native.mjs"], cwd=ROOT, check=True)
        subprocess.run(["npm", "run", "build", "--prefix", "ts"], cwd=ROOT, check=True)
    args.out = args.out.resolve(); args.out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".lars-alpha-", dir=args.out.parent) as temporary:
        temporary = Path(temporary); bundle = temporary / "bundle"; bundle.mkdir()
        package = temporary / "package"; package.mkdir()
        original = json.loads((ROOT / "ts/package.json").read_text())
        manifest = {k: original[k] for k in ["name", "license", "repository", "type", "main", "types", "exports", "engines"]}
        manifest.update({"version": VERSION, "description": "Experimental macOS arm64 image decisions developer alpha",
                         "os": ["darwin"], "cpu": ["arm64"], "files": ["dist", "native", "README.md", "LICENSE", "THIRD_PARTY_LICENSES"]})
        (package / "package.json").write_text(json.dumps(manifest, indent=2) + "\n")
        shutil.copytree(ROOT / "ts/dist", package / "dist")
        (package / "native").mkdir()
        shutil.copyfile(ROOT / "ts/native/lars.darwin-arm64.node", package / "native/lars.darwin-arm64.node")
        shutil.copyfile(ROOT / "packaging/vision-alpha/START-HERE.md", package / "README.md")
        shutil.copyfile(ROOT / "LICENSE", package / "LICENSE")
        license_notices(package / "THIRD_PARTY_LICENSES")
        subprocess.run(["npm", "pack", "--ignore-scripts", "--pack-destination", str(bundle)], cwd=package, check=True)
        for filename in ["START-HERE.md", "FEEDBACK.md", "model.lock.json", "prepare-model.py", "evaluate.mjs", "verify-alpha.py", "reference-results.json"]:
            shutil.copyfile(ROOT / "packaging/vision-alpha" / filename, bundle / filename)
        shutil.copytree(ROOT / "fixtures/vision-alpha", bundle / "fixtures")
        source = source_snapshot(bundle / f"lars-image-alpha-source-{VERSION}.tar.gz")
        build = {
            "format": 1, "alpha_version": VERSION, "scope": "macOS arm64 developer alpha; unpublished",
            "package": f"moedex-lars-{VERSION}.tgz", "source_archive": f"lars-image-alpha-source-{VERSION}.tar.gz",
            "source": source, "node": subprocess.check_output(["node", "--version"], text=True).strip(),
            "rust": subprocess.check_output(["rustc", "--version"], text=True).strip(),
            "host": {"system": platform.system(), "architecture": platform.machine(), "release": platform.release()},
            "model_lock_sha256": sha(bundle / "model.lock.json"),
            "native_sha256": sha(package / "native/lars.darwin-arm64.node"),
        }
        (bundle / "build.json").write_text(json.dumps(build, indent=2) + "\n")
        checksums(bundle)
        bundle.rename(args.out)
    print(f"Built unpublished tester bundle: {args.out}")


if __name__ == "__main__": main()
