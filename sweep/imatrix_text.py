"""Text for llama-imatrix: training-split prompts rendered exactly as the engine serves them.

An importance matrix weights quantization error by how much each weight's inputs matter on
representative text. The representative text here is the adapters' own training split
(corpus E, repaired targets), rendered with the model's chat template through `Answer:`, so the
activations it measures are the ones served. No validation or test row is used.

    ~/Code/personal/lars/.venv/bin/python sweep/imatrix_text.py --template sweep/templates/Qwen3-1.7B.json \
        --out models/imatrix-Qwen3-1.7B.txt [--records 1500]

Run it with llama-imatrix's `--parse-special`, so the template's special tokens are tokens.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

LARS = Path.home() / "Code" / "personal" / "lars"
sys.path.insert(0, str(LARS / "src"))

from lars.labels import label_candidates  # noqa: E402
from lars.render import TEMPLATES, compose_prompt, render_content  # noqa: E402
from lars.train.data import read_records  # noqa: E402
from lars.train.features import _render  # noqa: E402

TRAIN = [LARS / "data" / "train-e-soft" / f"{name}.train.jsonl" for name in ("open-jev", "jev-bench", "tasksource-jev")]


def template_fn(spec: str):
    if spec in TEMPLATES:
        return TEMPLATES[spec]
    captured = json.loads(Path(spec).read_text())

    def apply(system: str, user: str) -> str:
        if captured.get("trims_content"):
            system, user = system.strip(), user.strip()
        return captured["before_system"] + system + captured["between"] + user + captured["after_user"]

    return apply


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", required=True, help="chatml or a captured template JSON")
    parser.add_argument("--records", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    records = [r for path in TRAIN for r in read_records(path)]
    rng = random.Random(args.seed)
    rng.shuffle(records)
    labels = list(label_candidates())[:255]
    template = template_fn(args.template)
    prompts = []
    for record in records[: args.records]:
        order = list(range(len(record.options)))
        if record.kind == "choice":
            rng.shuffle(order)
        body, _, _ = _render(record, labels, order)
        prefix, suffix = compose_prompt(template, render_content(record.state), body)
        prompts.append(prefix + suffix)
    args.out.write_text("\n".join(prompts) + "\n", encoding="utf-8")
    sources = sorted({r.source.split("/")[0] for r in records[: args.records]})
    print(f"{args.out.name}: {len(prompts)} prompts from {', '.join(sources)}, {args.out.stat().st_size / 1e6:.1f} MB",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
