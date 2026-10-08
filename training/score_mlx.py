"""Score rendered prompts with MLX: a base model, optionally with a LoRA adapter, as trained.

The prompts come from `lars-dev prompts`, so they are exactly what the SDK serves. Each row's
answer is the label whose token (" <label>") has the highest next-token logit. Prints the
accuracy as JSON, for the build check (sweep/lora_lib.sh), which compares it with the GGUF's.

    .venv-train/bin/python training/score_mlx.py --model Qwen/Qwen3.5-2B --adapter checkpoints/x --prompts p.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlx.core as mx
from mlx_lm import load


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="a Hugging Face id or local path")
    parser.add_argument("--adapter", help="a LoRA adapter directory (adapter_config.json, adapters.safetensors)")
    parser.add_argument("--prompts", type=Path, required=True)
    args = parser.parse_args()

    model, tokenizer = load(args.model, adapter_path=args.adapter)
    rows = [json.loads(line) for line in args.prompts.read_text().splitlines() if line.strip()]
    right = 0
    for row in rows:
        ids = []
        for label in row["labels"]:
            token = tokenizer.encode(f" {label}", add_special_tokens=False)
            if len(token) != 1:
                raise SystemExit(f"label {label!r} is not one token")
            ids.append(token[0])
        tokens = mx.array([tokenizer.encode(row["prompt"], add_special_tokens=True)])
        logits = model(tokens)[0, -1]
        scores = [logits[i].item() for i in ids]
        right += int(max(range(len(scores)), key=scores.__getitem__) == row["target"])
    print(json.dumps({"count": len(rows), "accuracy": right / max(len(rows), 1)}))


if __name__ == "__main__":
    main()
