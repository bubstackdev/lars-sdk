#!/usr/bin/env python3
"""Download and verify the pinned alpha model, or reuse verified local artifacts."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import urllib.request


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def verify(path, item):
    return path.is_file() and path.stat().st_size == item["bytes"] and digest(path) == item["sha256"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lock", type=Path, default=Path(__file__).with_name("model.lock.json"))
    parser.add_argument("--artifacts-dir", type=Path, help="Reuse local model.gguf/projector.gguf or upstream filenames")
    args = parser.parse_args()
    lock = json.loads(args.lock.read_text())
    if lock.get("format") != 1 or set(lock["artifacts"]) != {"model", "projector"}:
        parser.error("unsupported model lock")
    manifest = {
        "format": 2, "name": lock["pack_name"], "version": lock["pack_version"], "backend": "llamacpp",
        "template": lock["template"], "layout": "v1",
        "runtime": {"llama_cpp": lock["llama_cpp"], "flash_attention": "off"},
        "vision": {"preprocessing": "stb-rgb-white-v1", "max_tokens": lock["visual_max_tokens"]},
        "upstream": {k: lock[k] for k in ["repository", "revision", "license", "model_card"]},
    }
    for role, item in lock["artifacts"].items():
        manifest[role] = {"file": f"{role}.gguf", "bytes": item["bytes"], "sha256": item["sha256"]}
    if args.output.exists():
        path = args.output / "pack.json"
        if not path.is_file() or json.loads(path.read_text()) != manifest:
            parser.error("existing output is not this pinned alpha pack; choose a new directory")
        for role, item in lock["artifacts"].items():
            if not verify(args.output / f"{role}.gguf", item):
                parser.error(f"existing {role} fails verification; choose a new output directory")
        print(f"Verified existing pack: {args.output.resolve()}")
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".lars-model-", dir=args.output.parent) as temporary:
        staging = Path(temporary) / "pack"
        staging.mkdir()
        for role, item in lock["artifacts"].items():
            target = staging / f"{role}.gguf"
            if args.artifacts_dir:
                candidates = [args.artifacts_dir / item["filename"], args.artifacts_dir / f"{role}.gguf"]
                source = next((p for p in candidates if p.is_file()), None)
                if source is None:
                    parser.error(f"missing local {role}")
                print(f"Copying and verifying {role}", flush=True)
                shutil.copyfile(source, target)
            else:
                url = f"https://huggingface.co/{lock['repository']}/resolve/{lock['revision']}/{item['filename']}"
                print(f"Downloading {role}: {item['bytes']} bytes", flush=True)
                received = 0
                with urllib.request.urlopen(url, timeout=60) as response, target.open("wb") as out:
                    while True:
                        block = response.read(1024 * 1024)
                        if not block:
                            break
                        received += len(block)
                        if received > item["bytes"]:
                            parser.error(f"{role} download exceeds the pinned size")
                        out.write(block)
                        if received % (64 * 1024 * 1024) == 0:
                            print(f"  {received}/{item['bytes']} bytes", flush=True)
            if not verify(target, item):
                parser.error(f"{role} size or SHA-256 differs from the lock; no usable pack was written")
        (staging / "pack.json").write_text(json.dumps(manifest, indent=2) + "\n")
        staging.rename(args.output)
    print(f"Ready: {args.output.resolve()}")


if __name__ == "__main__":
    main()
