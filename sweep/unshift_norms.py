"""Undo mlx_lm's Qwen3.5 norm shift in a fused model, so llama.cpp's converter reads it right.

Qwen3.5 stores these RMSNorm weights zero-centered (the layer multiplies by 1 + weight). mlx_lm's
sanitize adds 1.0 to them on load, and `mlx_lm fuse` saves them shifted. convert_hf_to_gguf then
adds 1.0 again, and the model is broken: a fused 2B LoRA scored 0.316 on jev-bench, against 0.493
for the untrained model and 0.78 on the training run's own selection set. A no-op for other models.

    python unshift_norms.py <fused hf dir>
"""

import json
import sys
from pathlib import Path

import mlx.core as mx

NORMS = (".input_layernorm.weight", ".post_attention_layernorm.weight", "model.norm.weight",
         ".q_norm.weight", ".k_norm.weight")


def main(directory: Path) -> None:
    config = json.loads((directory / "config.json").read_text())
    if not str(config.get("model_type", "")).startswith("qwen3_5"):
        print(f"unshift_norms: {config.get('model_type')} is not Qwen3.5; nothing to do")
        return
    changed = 0
    for path in sorted(directory.glob("*.safetensors")):
        weights, metadata = mx.load(str(path), return_metadata=True)
        for key, value in weights.items():
            if key.endswith(NORMS):
                weights[key] = (value.astype(mx.float32) - 1.0).astype(value.dtype)
                changed += 1
        # mx.load maps the file lazily: write a new one beside it, then replace it.
        mx.eval(weights)
        partial = path.with_name(path.name + ".part.safetensors")
        mx.save_safetensors(str(partial), weights, metadata=metadata)
        partial.replace(path)
    if changed == 0:
        raise SystemExit("unshift_norms: a Qwen3.5 model with no norm weights to unshift")
    print(f"unshift_norms: subtracted 1.0 from {changed} norm weights")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
