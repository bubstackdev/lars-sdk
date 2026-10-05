"""Fuse the attention LoRA into Qwen3-30B-A3B's BF16 weights and keep only the experts LARS uses.

For every MoE layer, the experts are ranked by the gate weight they carried on LARS prompts
(`sweep/moe_profile.py`). The top `--keep` stay, renumbered in their original order, and the
router's rows for the dropped experts go with them. The router still picks `num_experts_per_tok`
(8) of the remaining experts and renormalizes their weights, as the model always does
(`norm_topk_prob`). With `--keep 128`, it only fuses. Output is a Hugging Face checkpoint for
llama.cpp's `convert_hf_to_gguf.py`. Shards are streamed, so memory stays near one shard.

    sweep/.venv-convert/bin/python sweep/moe_shrink.py --src <BF16 snapshot> --adapter <lars lora dir> \
        --usage sweep/out/moe-usage.json --keep 64 --out models/fused/qwen3-30b-lars-k64
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

EXPERT = re.compile(r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(.+)$")
ROUTER = re.compile(r"^model\.layers\.(\d+)\.mlp\.gate\.weight$")
ATTENTION = re.compile(r"^(model\.layers\.\d+\.self_attn\.[qkvo]_proj)\.weight$")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--usage", type=Path, required=True)
    parser.add_argument("--keep", type=int, required=True, help="experts kept per layer")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    config = json.loads((args.src / "config.json").read_text())
    n_experts, top_k = config["num_experts"], config["num_experts_per_tok"]
    if not top_k <= args.keep <= n_experts:
        raise SystemExit(f"--keep must be between {top_k} and {n_experts}")

    usage = json.loads(args.usage.read_text())["layers"]
    kept = []  # per layer: original expert indices, ascending
    for layer in usage:
        ranked = sorted(range(n_experts), key=lambda e: -layer["weights"][e])
        kept.append(sorted(ranked[: args.keep]))
    renumber = [{old: new for new, old in enumerate(k)} for k in kept]

    adapter_config = json.loads((args.adapter / "adapter_config.json").read_text())
    scale = float(adapter_config["lora_parameters"]["scale"])
    lora = {}
    with safe_open(args.adapter / "adapters.safetensors", "pt") as f:
        for key in f.keys():
            lora[key] = f.get_tensor(key).to(torch.float32)

    args.out.mkdir(parents=True, exist_ok=True)
    index = json.loads((args.src / "model.safetensors.index.json").read_text())
    shards = sorted(set(index["weight_map"].values()))
    weight_map, fused, total_bytes = {}, 0, 0
    for shard in shards:
        tensors = {}
        with safe_open(args.src / shard, "pt") as f:
            for key in f.keys():
                expert, router, attention = EXPERT.match(key), ROUTER.match(key), ATTENTION.match(key)
                if expert:
                    layer, e, rest = int(expert[1]), int(expert[2]), expert[3]
                    if e not in renumber[layer]:
                        continue
                    name = f"model.layers.{layer}.mlp.experts.{renumber[layer][e]}.{rest}"
                    tensors[name] = f.get_tensor(key)
                elif router:
                    layer = int(router[1])
                    tensors[key] = f.get_tensor(key)[kept[layer]].contiguous()
                elif attention and f"{attention[1]}.lora_a" in lora:
                    weight = f.get_tensor(key)
                    a, b = lora[f"{attention[1]}.lora_a"], lora[f"{attention[1]}.lora_b"]
                    delta = (scale * b.T) @ a.T
                    tensors[key] = (weight.to(torch.float32) + delta).to(weight.dtype)
                    fused += 1
                else:
                    tensors[key] = f.get_tensor(key)
        if not tensors:  # every tensor in it was a dropped expert
            continue
        for name, tensor in tensors.items():
            weight_map[name] = shard
            total_bytes += tensor.numel() * tensor.element_size()
        save_file(tensors, args.out / shard, metadata={"format": "pt"})
        print(f"{shard}: {len(tensors)} tensors", flush=True)

    expected = sum(1 for k in lora if k.endswith(".lora_a"))
    if fused != expected:
        raise SystemExit(f"fused {fused} attention projections, the adapter has {expected}")
    config["num_experts"] = args.keep
    (args.out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (args.out / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total_bytes}, "weight_map": weight_map}, indent=2) + "\n")
    for extra in args.src.iterdir():
        if extra.suffix in {".json", ".txt", ".jinja"} and extra.name not in {"config.json", "model.safetensors.index.json"}:
            shutil.copy(extra, args.out / extra.name)
    (args.out / "lars-shrink.json").write_text(json.dumps({"keep": args.keep, "kept": kept, "adapter": str(args.adapter),
                                                           "usage": str(args.usage)}) + "\n")
    print(f"fused {fused} projections; kept {args.keep} of {n_experts} experts per layer; {total_bytes / 1e9:.1f} GB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
