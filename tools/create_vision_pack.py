#!/usr/bin/env python3
"""Package compatible local language/projector GGUFs for experimental image answers."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--projector", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--name", default="lars-vision-dev")
    parser.add_argument("--version", default="0.0.0")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--template", choices=["chatml", "gemma", "llama3", "plain"], default="chatml")
    parser.add_argument("--runtime-version", help="Override the linked llama.cpp version (default: checkout's full commit hash)")
    args = parser.parse_args()
    if not 1 <= args.max_tokens <= 4096:
        parser.error("--max-tokens must be between 1 and 4096")
    if args.output.exists():
        parser.error("--output must not exist; existing packs are never overwritten")
    for path in [args.model, args.projector]:
        if not path.is_file() or path.stat().st_size == 0:
            parser.error(f"not a nonempty GGUF file: {path}")
        with path.open("rb") as f:
            if f.read(4) != b"GGUF":
                parser.error(f"not a GGUF file: {path}")
    repo = Path(__file__).resolve().parents[1]
    runtime = args.runtime_version or (repo / "crates/lars-llama-sys/llama-cpp-revision.txt").read_text().strip()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".lars-vision-", dir=args.output.parent) as temporary:
        staging = Path(temporary) / "pack"
        staging.mkdir()

        def copy(source, filename):
            dest = staging / filename
            shutil.copyfile(source, dest)
            digest = hashlib.sha256()
            with dest.open("rb") as f:
                for block in iter(lambda: f.read(1024 * 1024), b""):
                    digest.update(block)
            return {"file": filename, "bytes": dest.stat().st_size, "sha256": digest.hexdigest()}

        manifest = {
            "format": 2, "name": args.name, "version": args.version, "backend": "llamacpp",
            "model": copy(args.model, "model.gguf"),
            "projector": copy(args.projector, "projector.gguf"),
            "template": args.template, "layout": "v1",
            "runtime": {"llama_cpp": runtime, "flash_attention": "off"},
            "vision": {"preprocessing": "stb-rgb-white-v1", "max_tokens": args.max_tokens},
        }
        (staging / "pack.json").write_text(json.dumps(manifest, indent=2) + "\n")
        staging.rename(args.output)
    print(args.output.resolve())
    print("Experimental, uncalibrated pack. Weight compatibility and task accuracy require evaluation.")


if __name__ == "__main__":
    main()
