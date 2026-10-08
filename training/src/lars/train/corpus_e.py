"""Corpus E: the train splits of the soft corpus, without the sources whose licenses forbid
published weights. Every LARS pack's LoRA trained on it (attributions: sweep/release_packs.py).

    python -m lars.train.build --out data/train-soft
    python -m lars.train.corpus_e --soft data/train-soft --out data/train-e-soft

Reconstructed on 2026-10-08 after the original script was lost. The dropped sources are the ones
the trainer's `--select-skip` names and the release NOTICE leaves out; filtering after the build's
caps (not before) matches the original's step counts: 5,370 against 5,279 for Qwen3.5-2B with
typed-decisions train, and 4,449 steps, 2 h 35 min at 2.1 s per step, for Qwen3-1.7B.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

FILES = ["open-jev.train.jsonl", "jev-bench.train.jsonl", "tasksource-jev.train.jsonl"]
DROPPED = (
    "jev-bench/sst5",  # SST: no license for redistribution
    "jev-bench/yelp5",  # Yelp dataset terms: non-commercial
    "jev-bench/stsb",  # STS benchmark: mixed, partly unlicensed sources
    "tasksource-jev/anli/",  # ANLI: CC BY-NC 4.0
)


def dropped(source: str) -> bool:
    return any(source == d or (d.endswith("/") and source.startswith(d)) for d in DROPPED)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--soft", default="data/train-soft")
    parser.add_argument("--out", default="data/train-e-soft")
    args = parser.parse_args()
    soft, out = Path(args.soft), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, dict] = {}
    for name in FILES:
        kept, removed = Counter(), Counter()
        with (soft / name).open() as src, (out / name).open("w") as dst:
            for line in src:
                if not line.strip():
                    continue
                source = json.loads(line)["source"]
                if dropped(source):
                    removed[source] += 1
                else:
                    kept[source] += 1
                    dst.write(line)
        manifest[name] = {"rows": sum(kept.values()), "sources": dict(sorted(kept.items())),
                          "dropped": dict(sorted(removed.items()))}
        print(f"{name}: kept {sum(kept.values())}, dropped {sum(removed.values())}", flush=True)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
