"""Source and mixture ablations using frozen teacher targets; development evidence only."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
from importlib.metadata import distributions
import json
import math
import os
from pathlib import Path
import sys

from lars.train.accuracy import load_capture, record_key
from lars.train.data import read_records, write_records
from sweep.accuracy_ladder import digest
from sweep.checkpoint_selection import selection_metrics
from sweep.confirmation import HOLD
from sweep.experiment import Runner, atomic_json

ROOT = Path(__file__).resolve().parents[1]
FULL = ROOT / "sweep/out/accuracy-full"
CONTROL = ROOT / "sweep/out/checkpoint-selection-s0"
ARMS = [
    {"name": "arc-only-mix50", "sources": ["jev-bench/arc_challenge"], "mix": .5},
    {"name": "mmlu-only-mix50", "sources": ["jev-bench/mmlu"], "mix": .5},
    {"name": "mnli-only-mix50", "sources": ["jev-bench/mnli"], "mix": .5},
    {"name": "verbosity-only-mix50", "sources": ["jev-bench/helpsteer2_verbosity"], "mix": .5},
    {"name": "all-mix25", "sources": None, "mix": .25},
]
PROTOCOL = {
    "arms": ARMS, "seed": 0, "epochs": 1, "expected_steps": 5366,
    "batch_tokens": 4096, "max_tokens": 2048, "checkpoint": "fixed final step",
    "optimizer": {"lr": 2e-5, "rank": 8, "scale": 20., "dropout": 0., "keys": "all", "brier_weight": 1., "ordinal_weight": 0.},
    "teacher": "reuse exactly the 1724 existing gold-agreeing target changes; no teacher rerun or eligibility reselection",
    "selection": "equal mean seen/unseen source macro hard-label accuracy, raw MLX",
    "evaluation": "2642-row select split only; no confirmation, certificate, calibration or test evaluation",
    "controls": ["saved baseline step 5366", "saved all-source mix50 step 5366"],
    "tie_break": "higher selection score, then lower mixture, then arm name",
    "limits": "five new single-seed runs, sequential; stop after development report; no automatic promotion or further seeds",
    "interpretation": "exploratory development ablation; single-source effects may interact and do not establish causality across seeds",
}


def retarget(gold, distilled, sources, mix):
    if not math.isfinite(mix) or not 0 < mix <= .5:
        raise ValueError("mixture must be finite and in (0, 0.5]")
    if len(gold) != len(distilled):
        raise ValueError("training populations differ")
    output, changed = [], Counter()
    for original, soft in zip(gold, distilled, strict=True):
        if replace(original, target=[]) != replace(soft, target=[]):
            raise ValueError("teacher target population/order/metadata differ")
        if sources is None or original.source in sources:
            # Existing soft = (1-.5)*gold + .5*teacher. Scale its displacement to
            # obtain mix25 without introducing new teacher predictions or candidates.
            target = [g + (mix / .5) * (s - g) for g, s in zip(original.target, soft.target, strict=True)]
            if mix == .5:
                target = list(soft.target)
            if any(not math.isfinite(v) or v < 0 for v in target) or abs(sum(target) - 1) > 1e-8:
                raise ValueError("invalid mixed target")
            result = replace(original, target=target)
        else:
            result = original
        if result.target != original.target:
            changed[original.source] += 1
        output.append(result)
    return output, dict(sorted(changed.items()))


def freeze(path, value):
    if path.exists() and json.loads(path.read_text()) != value:
        raise ValueError(f"frozen configuration differs: {path}; use a new output directory")
    atomic_json(path, value)


def prepare(out):
    data = FULL / "data"
    gold, soft = list(read_records(data / "train.jsonl")), list(read_records(data / "distilled-s0.jsonl"))
    config = json.loads((FULL / "config.json").read_text())
    inputs = [data / "train.jsonl", data / "distilled-s0.jsonl", data / "heldout.jsonl",
              data / "select.records.jsonl", data / "select.wire.jsonl",
              CONTROL / "baseline-step-5366.jsonl", CONTROL / "distilled-step-5366.jsonl",
              Path(config["model"]) / "config.json"]
    freeze(out / "protocol.json", PROTOCOL)
    freeze(out / "inputs.json", {"model": config["model"], "files": {str(p): digest(p) for p in inputs}})
    out_data = out / "data"
    out_data.mkdir(exist_ok=True)
    manifest = {"records": len(gold), "arms": {}}
    union = Counter()
    for a, b in zip(gold, soft, strict=True):
        if a.target != b.target:
            union[a.source] += 1
    if set(union) != {s for arm in ARMS if arm["sources"] for s in arm["sources"]}:
        raise ValueError("unexpected teacher-target source set")
    if sum(union.values()) != 1724:
        raise ValueError("expected the original 1724 target changes")
    for arm in ARMS:
        records, counts = retarget(gold, soft, arm["sources"], arm["mix"])
        path = out_data / (arm["name"] + ".jsonl")
        if path.exists() and list(read_records(path)) != records:
            raise ValueError(f"prepared targets changed: {path}")
        write_records(records, path)
        manifest["arms"][arm["name"]] = {"changed_per_source": counts, "changed": sum(counts.values()),
                                                  "records": len(records), "sha256": digest(path)}
    freeze(out / "data-manifest.json", manifest)
    return config, gold


def run(out, config, gold):
    data = FULL / "data"
    trained = {r.source for r in gold}
    env = dict(os.environ, HF_HOME=str(ROOT / ".hf"), HF_HUB_OFFLINE="1",
               LARS_MLX_CACHE_GB="8", TOKENIZERS_PARALLELISM="false")
    runner = Runner(out, env)
    training_code = sorted((ROOT / "training/src/lars").rglob("*.py"))
    worker_code = [ROOT / "sweep/full_accuracy.py", ROOT / "sweep/confirmation.py", ROOT / "sweep/experiment.py",
                   ROOT / "sweep/checkpoint_selection.py", Path(__file__), *training_code]
    atomic_json(out / "environment.json", {"python": sys.version,
        "packages": {d.metadata["Name"]: d.version for d in distributions()},
        "code": {str(p): digest(p) for p in worker_code}})
    captures = {"baseline": load_capture(CONTROL / "baseline-step-5366.jsonl"),
                "all-mix50": load_capture(CONTROL / "distilled-step-5366.jsonl")}
    results = {name: selection_metrics(rows, trained) for name, rows in captures.items()}
    expected = {json.loads(line)["record_key"]: json.loads(line) for line in
                (data / "select.wire.jsonl").read_text().split("\n") if line.strip()}

    def check(rows):
        if rows.keys() != expected.keys():
            raise ValueError("selection population differs")
        for key, row in rows.items():
            if "p" not in row or row["hard_index"] != expected[key]["hard_index"]:
                raise ValueError("missing predictions or changed hard labels")
            if row["target"] != captures["baseline"][key]["target"]:
                raise ValueError("selection soft targets differ")
    for rows in captures.values():
        check(rows)
    for arm in ARMS:
        name = arm["name"]
        records = out / "data" / f"{name}.jsonl"
        adapter = out / "training" / name
        options = [sys.executable, "-m", "lars.train.lora", "--model", config["model"],
            "--records", records, data / "heldout.jsonl", "--holdout-sources", ",".join(HOLD),
            "--limit", "100000", "--epochs", "1", "--seed", "0", "--lr", "2e-5", "--rank", "8",
            "--scale", "20", "--dropout", "0", "--keys", "all", "--brier-weight", "1", "--ordinal-weight", "0",
            "--batch-tokens", "4096", "--max-tokens", "2048", "--eval-every", "500", "--out", adapter]
        history = adapter.with_name(adapter.name + ".history.json")
        steps = adapter.with_name(adapter.name + ".steps")
        runner.stage(name + "-train", command=options,
            inputs=[*training_code, Path(__file__), records, data / "heldout.jsonl", out / "protocol.json", out / "inputs.json"],
            outputs=[history, steps])
        h = json.loads(history.read_text())
        if h[-1]["step"] != PROTOCOL["expected_steps"]:
            raise ValueError("training step count differs from matched controls")
        final = steps / f"step-{h[-1]['step']}"
        capture = out / f"{name}-select.jsonl"
        task = out / f"{name}-select.task.json"
        atomic_json(task, {"operation": "mlx-capture", "model": config["model"], "adapter": str(final),
            "records": str(data / "select.records.jsonl"), "wire": str(data / "select.wire.jsonl"), "output": str(capture)})
        runner.stage(name + "-select", command=[sys.executable, "-m", "sweep.full_accuracy", "--task", task],
            inputs=[*worker_code, task, final, data / "select.records.jsonl", data / "select.wire.jsonl"], outputs=[capture])
        rows = load_capture(capture)
        check(rows)
        results[name] = selection_metrics(rows, trained)
        results[name]["delta_vs_baseline"] = results[name]["selection_score"] - results["baseline"]["selection_score"]
        results[name]["delta_vs_all_mix50"] = results[name]["selection_score"] - results["all-mix50"]["selection_score"]
        atomic_json(out / "partial-results.json", results)
        print(name, "selection_score", results[name]["selection_score"], flush=True)
    mixes = {"baseline": 0., "all-mix50": .5, **{a["name"]: a["mix"] for a in ARMS}}
    ranked = sorted(results, key=lambda n: (-results[n]["selection_score"], mixes[n], n))
    atomic_json(out / "results.json", {"protocol": PROTOCOL, "metrics": results, "ranking": ranked,
        "promotion_eligible": False, "reason": "single-seed development exploration only; no fresh confirmation"})
    atomic_json(out / "status.json", {"state": "complete", "ranking": ranked, "promotion_eligible": False})
    print("Ablation complete; no sweep restart, publication or additional seeds", flush=True)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "sweep/out/distillation-ablation-s0")
    parser.add_argument("--run", action="store_true", help="run the five sequential training/evaluation arms after setup")
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    import fcntl
    with (out / "run.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        config, gold = prepare(out)
        if args.run:
            run(out, config, gold)
        else:
            print((out / "data-manifest.json").read_text())


if __name__ == "__main__":
    main()
