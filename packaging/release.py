#!/usr/bin/env python3
"""Prepare and checksum GitHub distributions; never publish to a registry."""
import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import runpy
import shutil
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[1]
PLATFORMS = ["darwin-arm64", "darwin-x64", "linux-arm64", "linux-x64", "win32-x64"]


def git(*args, cwd=ROOT):
    return subprocess.check_output(["git", *args], cwd=cwd)


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare(out):
    version = json.loads((ROOT / "ts/package.json").read_text())["version"]
    if git("diff", "HEAD", "--exit-code"):
        raise RuntimeError("release source has tracked changes")
    submodule = ROOT / "crates/lars-llama-sys/llama.cpp"
    pinned = (submodule.parent / "llama-cpp-revision.txt").read_text().strip()
    if git("rev-parse", "HEAD", cwd=submodule).decode().strip() != pinned:
        raise RuntimeError("llama.cpp checkout does not match its pinned revision")
    identity = {"format": 1, "version": version, "commit": git("rev-parse", "HEAD").decode().strip(),
                "llama_cpp": pinned, "platforms": PLATFORMS, "registries_published": False}
    out.mkdir(parents=True, exist_ok=True)
    source = out / f"lars-sdk-{version}-source.tar.gz"
    prefix = f"lars-sdk-{version}/"
    with source.open("wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as zipped, tarfile.open(fileobj=zipped, mode="w") as archive:
        for directory, archive_prefix in [(ROOT, prefix), (submodule, prefix + "crates/lars-llama-sys/llama.cpp/")]:
            data = git("archive", "--format=tar", "--prefix=" + archive_prefix, "HEAD", cwd=directory)
            with tarfile.open(fileobj=io.BytesIO(data), mode="r:") as original:
                for member in original:
                    if member.issym() or member.islnk():
                        raise RuntimeError(f"source archive contains a link: {member.name}")
                    archive.addfile(member, original.extractfile(member) if member.isfile() else None)
        payload = (json.dumps(identity, indent=2) + "\n").encode()
        info = tarfile.TarInfo(prefix + "RELEASE-SOURCE.json")
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))
    (out / "build.json").write_text(json.dumps(identity, indent=2) + "\n")
    shutil.copyfile(ROOT / "RELEASE.md", out / "RELEASE.md")
    notices = out.parent / "notices"
    runpy.run_path(str(ROOT / "packaging/vision-alpha/build-alpha.py"))["license_notices"](notices, target=None)
    shutil.make_archive(str(out / "THIRD_PARTY_LICENSES"), "gztar", notices.parent, notices.name)


def finalize(out):
    version = json.loads((ROOT / "ts/package.json").read_text())["version"]
    for package in (out.parent / "npm").glob("*.tgz"):
        shutil.copyfile(package, out / ("npm-" + package.name))
    shutil.copyfile(out.parent / "github" / f"moedex-lars-{version}.tgz", out / f"moedex-lars-{version}.tgz")
    for package in (out.parent / "nuget").glob("*.nupkg"):
        shutil.copyfile(package, out / package.name)
    required = [f"moedex-lars-{version}.tgz", f"npm-moedex-lars-{version}.tgz", f"Moedex.Lars.{version}.nupkg",
                f"lars-sdk-{version}-source.tar.gz", "build.json", "RELEASE.md", "THIRD_PARTY_LICENSES.tar.gz"]
    required += [f"npm-moedex-lars-{p}-{version}.tgz" for p in PLATFORMS]
    for name in required:
        if not (out / name).is_file():
            raise RuntimeError(f"missing release artifact: {name}")
    (out / "SHA256SUMS").write_text("".join(f"{sha(out / name)}  {name}\n" for name in sorted(required)))


def verify(out):
    for line in (out / "SHA256SUMS").read_text().splitlines():
        expected, name = line.split("  ", 1)
        if Path(name).name != name or sha(out / name) != expected:
            raise RuntimeError(f"release checksum mismatch: {name}")
    print("Release checksums verified")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "finalize", "verify"])
    parser.add_argument("--out", type=Path, default=ROOT / "dist/release")
    args = parser.parse_args()
    {"prepare": prepare, "finalize": finalize, "verify": verify}[args.action](args.out.resolve())
