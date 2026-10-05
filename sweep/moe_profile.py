"""Which experts Qwen3-30B-A3B routes to on LARS prompts.

Loads the 4-bit MLX model with the attention LoRA (routing depends on the adapted attention),
renders training-split prompts exactly as the engine serves them, and records for every MoE layer
how often each expert is chosen and the total gate weight it carries (top-k weights renormalized,
as the model uses them). No validation or test row is used.

    ~/Code/personal/lars/.venv/bin/python sweep/moe_profile.py --out sweep/out/moe-usage.json [--records 1500]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

LARS = Path.home() / "Code" / "personal" / "lars"
sys.path.insert(0, str(LARS / "src"))

import mlx.core as mx  # noqa: E402
import mlx_lm  # noqa: E402
import mlx_lm.models.qwen3_moe as qwen3_moe  # noqa: E402

from lars.labels import label_candidates  # noqa: E402
from lars.render import TEMPLATES, compose_prompt, render_content  # noqa: E402
from lars.train.data import read_records  # noqa: E402
from lars.train.features import _render  # noqa: E402

MODEL = "mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit"
ADAPTER = LARS / "checkpoints" / "lora-30b-e-s0"
TRAIN = [LARS / "data" / "train-e-soft" / f"{name}.train.jsonl" for name in ("open-jev", "jev-bench", "tasksource-jev")]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    model, tokenizer = mlx_lm.load(MODEL, adapter_path=str(ADAPTER))
    blocks = [layer.mlp for layer in model.layers if isinstance(layer.mlp, qwen3_moe.Qwen3MoeSparseMoeBlock)]
    n_experts = blocks[0].num_experts
    counts = [mx.zeros(n_experts) for _ in blocks]
    weights = [mx.zeros(n_experts) for _ in blocks]

    def hooked(index: int, block):
        def call(x):
            gates = mx.softmax(block.gate(x), axis=-1, precise=True)
            k = block.top_k
            inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
            scores = mx.take_along_axis(gates, inds, axis=-1)
            if block.norm_topk_prob:
                scores /= mx.sum(scores, axis=-1, keepdims=True)
            flat = inds.reshape(-1)
            counts[index] = counts[index].at[flat].add(mx.ones(flat.shape))
            weights[index] = weights[index].at[flat].add(scores.reshape(-1).astype(mx.float32))
            y = block.switch_mlp(x, inds)
            return (y * scores[..., None]).sum(axis=-2)

        return call

    for i, block in enumerate(blocks):
        block.__call__ = hooked(i, block)  # instance attribute; nn.Module dispatches through __call__ on the class
    # nn.Module.__call__ is looked up on the type, so patch the class to defer to an instance hook.
    original = qwen3_moe.Qwen3MoeSparseMoeBlock.__call__
    qwen3_moe.Qwen3MoeSparseMoeBlock.__call__ = lambda self, x: self.__dict__["__call__"](x) if "__call__" in self.__dict__ else original(self, x)

    records = [r for path in TRAIN for r in read_records(path)]
    rng = random.Random(args.seed)
    rng.shuffle(records)
    labels = list(label_candidates())[:255]
    started, tokens = time.time(), 0
    for n, record in enumerate(records[: args.records]):
        order = list(range(len(record.options)))
        if record.kind == "choice":
            rng.shuffle(order)
        body, _, _ = _render(record, labels, order)
        prefix, suffix = compose_prompt(TEMPLATES["chatml"], render_content(record.state), body)
        ids = mx.array([tokenizer.encode(prefix + suffix, add_special_tokens=False)])
        model(ids)
        tokens += ids.shape[1]
        if n % 50 == 0:
            mx.eval(counts, weights)
            print(f"{n} prompts, {tokens} tokens, {tokens / (time.time() - started):.0f} tok/s", file=sys.stderr, flush=True)
    mx.eval(counts, weights)
    out = {"model": MODEL, "adapter": str(ADAPTER), "records": args.records, "tokens": tokens, "top_k": blocks[0].top_k,
           "layers": [{"counts": c.tolist(), "weights": w.tolist()} for c, w in zip(counts, weights, strict=True)]}
    args.out.write_text(json.dumps(out) + "\n")
    print(f"wrote {args.out}: {len(blocks)} layers x {n_experts} experts over {tokens} tokens", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
