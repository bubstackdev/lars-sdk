"""Latency tables from `bench/run.sh` output: one table per request, a row per model and setting.

    python3 sweep/latency.py sweep/out/bench-night-*.jsonl > sweep/out/latency.md
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict


def main() -> int:
    rows = [json.loads(line) for path in sys.argv[1:] for line in open(path) if line.strip()]
    by_request: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_request[row["request"]].append(row)
    for request, entries in by_request.items():
        print(f"\n### {request}\n")
        print("| model | setting | p50 | p90 | first call | peak RSS | load | load avg |")
        print("|---|---|---|---|---|---|---|---|")
        for r in entries:
            print(f"| {r['model']} | {r['config']} | {r['p50_ms']:.0f} ms | {r['p90_ms']:.0f} ms | {r['first_ms']:.0f} ms | "
                  f"{r['peak_rss_mb'] / 1024:.1f} GB | {r['load_ms'] / 1000:.2f} s | {r['load_avg']} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
