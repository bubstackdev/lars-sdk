"""Check that feature extraction keeps the normal label readout on real model prompts."""
import argparse
import json
from pathlib import Path

import numpy as np

from lars.backends.mlx import MLXBackend
from lars.labels import assign_labels
from lars.render import compose_prompt, render_content
from lars.train.data import read_records
from lars.train.features import _render


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--records", required=True)
    parser.add_argument("--limit", type=int, default=80)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    backend = MLXBackend(args.model)
    records = list(read_records(args.records))[:args.limit]
    labels = assign_labels(max(len(r.options) for r in records), backend.is_single_token)
    differences, flips = [], 0
    for r in records:
        body, _, ends = _render(r, labels, list(range(len(r.options))))
        prefix, suffix = compose_prompt(backend.template(), render_content(r.state), body)
        tokens = [tuple(labels[:len(r.options)])]
        a = backend.label_logits(prefix, [suffix], tokens)[0]
        b, _, _ = backend.label_logits_with_features(prefix, [suffix], tokens, [ends])[0]
        def probs(z):
            p = np.exp(z - np.max(z))
            return p / p.sum()
        differences.append(float(np.max(np.abs(probs(a) - probs(b)))))
        flips += int(a.argmax() != b.argmax())
    result = {"rows": len(records), "label_flips": flips, "max_probability_difference": max(differences),
              "passed": flips == 0 and max(differences) <= 1e-5}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
