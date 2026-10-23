"""Compare saved checkpoints on the frozen selection split; no training or test selection."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import sys

import numpy as np

from lars.train.accuracy import load_capture, metrics
from lars.train.data import read_records
from sweep.confirmation import summary
from sweep.experiment import Runner, atomic_json

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = {
    "recipes": ["baseline", "distilled"], "steps": [2500, 5366], "seed": 0,
    "split": "select", "selection": "equal mean of seen-source and unseen-source macro hard-label accuracy",
    "tie_break": "earlier checkpoint", "probabilities": "raw MLX, no calibration or quantization",
    "interpretation": "development selection only; not fresh confirmation or a promotion decision",
}


def selection_metrics(rows, trained):
    result = summary(rows)
    per = result["per_source"]
    groups = {name: [m["accuracy"] for source, m in per.items() if (source in trained) == seen]
              for name, seen in (("seen", True), ("unseen", False))}
    if not all(groups.values()):
        raise ValueError("selection requires both seen and unseen sources")
    result["seen_macro_accuracy"] = float(np.mean(groups["seen"]))
    result["unseen_macro_accuracy"] = float(np.mean(groups["unseen"]))
    result["selection_score"] = (result["seen_macro_accuracy"] + result["unseen_macro_accuracy"]) / 2
    kinds = defaultdict(list)
    for row in rows.values():
        kinds[row["kind"]].append(row)
    result["per_kind"] = {kind: metrics(group) for kind, group in kinds.items()}
    return result


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "sweep/out/checkpoint-selection-s0")
    args = parser.parse_args()
    out = args.out.resolve()
    import fcntl
    out.mkdir(parents=True, exist_ok=True)
    with (out / "run.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = out / "protocol.json"
        if path.exists() and json.loads(path.read_text()) != PROTOCOL:
            raise ValueError("protocol differs; use a new output directory")
        atomic_json(path, PROTOCOL)
        full = ROOT / "sweep/out/accuracy-full"
        config = json.loads((full / "config.json").read_text())
        data = full / "data"
        trained = {r.source for r in read_records(data / "train.jsonl")}
        env = dict(os.environ, HF_HOME=str(ROOT / ".hf"), HF_HUB_OFFLINE="1",
                   LARS_MLX_CACHE_GB="8", TOKENIZERS_PARALLELISM="false")
        runner = Runner(out, env)
        captures, results = {}, {}
        code = [Path(__file__), ROOT / "sweep/full_accuracy.py", ROOT / "sweep/confirmation.py",
                ROOT / "training/src/lars/train/accuracy.py", ROOT / "training/src/lars/backends/mlx.py"]
        for recipe in PROTOCOL["recipes"]:
            for step in PROTOCOL["steps"]:
                name = f"{recipe}-step-{step}"
                adapter = full / "training" / f"{recipe}-s0.steps" / f"step-{step}"
                output = out / f"{name}.jsonl"
                task = out / f"{name}.task.json"
                atomic_json(task, {"operation": "mlx-capture", "model": config["model"],
                    "adapter": str(adapter), "records": str(data / "select.records.jsonl"),
                    "wire": str(data / "select.wire.jsonl"), "output": str(output)})
                runner.stage(name, command=[sys.executable, "-m", "sweep.full_accuracy", "--task", task],
                    inputs=[*code, task, adapter, data / "select.records.jsonl", data / "select.wire.jsonl",
                            data / "train.jsonl", out / "protocol.json"], outputs=[output])
                rows = load_capture(output)
                if any("p" not in row for row in rows.values()):
                    raise ValueError("selection rows may not be silently skipped")
                if captures and rows.keys() != next(iter(captures.values())).keys():
                    raise ValueError("checkpoint selection populations differ")
                if captures:
                    ref = next(iter(captures.values()))
                    if any(row["hard_index"] != ref[key]["hard_index"] or row["target"] != ref[key]["target"]
                           for key, row in rows.items()):
                        raise ValueError("checkpoint selection targets differ")
                captures[name] = rows
                results[name] = selection_metrics(rows, trained)
                atomic_json(out / "partial-results.json", results)
                print(name, json.dumps({k: results[name][k] for k in
                    ("n", "selection_score", "seen_macro_accuracy", "unseen_macro_accuracy", "macro_brier")} ), flush=True)
        selected = {recipe: max(PROTOCOL["steps"], key=lambda step:
                    (results[f"{recipe}-step-{step}"]["selection_score"], -step)) for recipe in PROTOCOL["recipes"]}
        atomic_json(out / "results.json", {"protocol": PROTOCOL, "metrics": results, "selected_steps": selected,
            "selected_distilled_minus_baseline": results[f"distilled-step-{selected['distilled']}"]["selection_score"] -
                results[f"baseline-step-{selected['baseline']}"]["selection_score"]})
        atomic_json(out / "status.json", {"state": "complete", "selected_steps": selected})
        print("Development comparison complete; no training, publication or sweep restart", flush=True)


if __name__ == "__main__":
    main()
