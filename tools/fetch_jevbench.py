"""jev-bench eval files for the sweep, pack building and the build check: data/evals/<config>.<split>.jsonl.

Each config's first 200 test and validation rows of Praveenrajus/jev-bench, unchanged. That is
the subset lars's evals/data held, as the row ids of every capture in sweep/out show, so numbers
measured on these files compare with every earlier measurement.

    python3 tools/fetch_jevbench.py [--out data/evals] [--rows 200]
"""

from __future__ import annotations

import argparse
from pathlib import Path

REPO = "Praveenrajus/jev-bench"
REVISION = "18f88da81c28c2bec55edc31f63f2afdfba109ea"
CONFIGS = ["arc_challenge", "banking77", "boolq", "chaosnli", "civil_comments", "clinc150", "fever_evidence",
           "go_emotions", "helpsteer2_helpfulness", "helpsteer2_verbosity", "ledgar", "massive",
           "measuring_hate_speech", "mmlu", "mnli", "paws", "sms_spam", "sst5", "stsb", "strategyqa_closed",
           "strategyqa_grounded", "yelp5"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent.parent / "data" / "evals")
    parser.add_argument("--rows", type=int, default=200)
    args = parser.parse_args()
    from huggingface_hub import snapshot_download

    root = Path(snapshot_download(REPO, repo_type="dataset", revision=REVISION, allow_patterns=["data/*/*.jsonl"]))
    args.out.mkdir(parents=True, exist_ok=True)
    for config in CONFIGS:
        for split in ("validation", "test"):
            source = root / "data" / config / f"{split}.jsonl"
            lines = [l for l in source.read_text(encoding="utf-8").split("\n") if l][: args.rows] if source.exists() else []
            (args.out / f"{config}.{split}.jsonl").write_text("".join(line + "\n" for line in lines))
            print(f"{config}.{split}: {len(lines)}")


if __name__ == "__main__":
    main()
