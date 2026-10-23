"""Audit development errors and compare a fixed-budget targeted training refresh.

Only select errors guide data selection. Dev is evaluated after data/protocol freeze.
No external labels, teacher relabeling, confirmation or test predictions are used.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time

import numpy as np

from lars.train.accuracy import load_capture, record_key, state_key
from lars.train.data import read_records, write_records
from sweep.accuracy_ladder import digest
from sweep.checkpoint_selection import selection_metrics
from sweep.confirmation import HOLD, REVISION, pairs_from_wire
from sweep.distillation_ablation import freeze
from sweep.experiment import Runner, atomic_json

ROOT = Path(__file__).resolve().parents[1]
FULL = ROOT / "sweep/out/accuracy-full"
CONTROL = ROOT / "sweep/out/checkpoint-selection-s0"
POLICY = {
    "seed": 0, "train_steps_per_arm": 5366, "arms": ["baseline", "refreshed"],
    "max_new_training_steps": 10732, "batch_tokens": 4096, "max_tokens": 2048,
    "epochs_ceiling": 2, "checkpoint": "fixed final step; never best heldout",
    "candidate_sources": "up to four seen sources, select accuracy < .8, >=200 train rows, >=3 classes, max/min count >=4; lowest select accuracy first, then source name",
    "replacement_fraction_per_source": .30,
    "candidate_labels": "unchanged human dataset labels and soft targets; hard/soft argmax agree; max target >= .8",
    "selection": "source-class error rate with add-one smoothing, inverse sqrt current class frequency; replace currently most common class only while counts differ by >1",
    "pool": "pinned jev-bench raw training split only, unused states; no license-excluded or heldout sources",
    "source_exclusions": [*HOLD, "jev-bench/sst5", "jev-bench/yelp5", "jev-bench/stsb"],
    "novelty": "exclude normalized whole-state overlap with all prior role partitions, original train, heldout and pilot; remove candidate decision groups with conflicting hard labels",
    "selection_metric": "equal mean seen/unseen source macro hard-label accuracy, raw MLX",
    "audit_split": "select", "evaluation_splits": ["select", "dev"],
    "interpretation": "single-seed development experiment; select is adaptive, dev previously used for teacher gating; neither is a fresh final test",
    "optimizer": {"lr": 2e-5, "rank": 8, "scale": 20., "dropout": 0., "keys": "all", "brier_weight": 1., "ordinal_weight": 0.},
    "stop": "after two training arms and four captures; no further seeds, teacher work, quantization, publication or confirmation",
}


def normalized(value):
    if isinstance(value, str):
        return re.sub(r"\s+", " ", value).strip().casefold()
    if isinstance(value, list):
        return [normalized(v) for v in value]
    if isinstance(value, dict):
        return {k: normalized(v) for k, v in value.items()}
    return value


def hash_value(value):
    return hashlib.sha256(json.dumps(normalized(value), sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def content_key(record):
    return hash_value(record.state)


def decision_key(record):
    return hash_value([record.state, record.to_question()])


def label_name(record, index):
    return str(index) if record.kind == "score" else record.options[index]


def hard_class(record):
    return label_name(record, int(np.argmax(record.target)))


def clean_candidates(pairs, blocked, token_ok):
    groups = defaultdict(list)
    rejected = Counter()
    for record, hard in pairs:
        record.validate()
        if content_key(record) in blocked:
            rejected["state_overlap"] += 1
            continue
        groups[decision_key(record)].append((record, hard))
    output, states = [], set()
    for key, group in sorted(groups.items()):
        if len({label_name(r, h) for r, h in group}) != 1:
            rejected["conflicting_hard_labels"] += len(group)
            continue
        # Stable best-confidence representative for repeated input decisions.
        record, hard = max(group, key=lambda p: (max(p[0].target), record_key(p[0])))
        if int(np.argmax(record.target)) != hard or max(record.target) < .8:
            rejected["ambiguous_or_hard_soft_disagreement"] += len(group)
            continue
        if content_key(record) in states:
            rejected["repeated_candidate_state"] += len(group)
            continue
        if len(record.options) > 160 or not token_ok(record):
            rejected["length_or_options"] += len(group)
            continue
        output.append(record)
        states.add(content_key(record))
        rejected["duplicate_decision"] += len(group) - 1
    return output, dict(rejected)


def refresh_source(original, candidates, errors, totals, limit):
    """Replace majority examples with unused error-prone minority-class training rows."""
    result = list(original)
    counts = Counter(hard_class(r) for r in result)
    pools = defaultdict(list)
    for r in candidates:
        pools[hard_class(r)].append(r)
    for values in pools.values():
        values.sort(key=record_key, reverse=True)
    swaps = []
    replaced = set()
    for _ in range(limit):
        eligible = [c for c, pool in pools.items() if pool and errors.get(c, 0) > 0
                    and totals.get(c, 0) >= 3 and counts.get(c, 0) + 1 < max(counts.values())]
        if not eligible:
            break
        desired = min(eligible, key=lambda c: (-((errors[c] + 1) / (totals[c] + 2)) /
                                               math.sqrt(counts.get(c, 0) + 1), c))
        # Only original examples may be removed, never immediately replace an insertion.
        donors = [i for i, r in enumerate(result) if i not in replaced and hard_class(r) != desired]
        if not donors:
            break
        index = min(donors, key=lambda i: (-counts[hard_class(result[i])], record_key(result[i])))
        old = result[index]
        if counts[hard_class(old)] <= counts.get(desired, 0) + 1:
            break
        new = pools[desired].pop()
        if old.source != new.source or old.kind != new.kind:
            raise ValueError("replacement changes source or kind")
        counts[hard_class(old)] -= 1
        counts[desired] += 1
        result[index] = new
        replaced.add(index)
        swaps.append({"removed": record_key(old), "inserted": record_key(new),
                      "old_class": hard_class(old), "new_class": desired})
    return result, swaps


def token_filter(model):
    from transformers import AutoTokenizer
    from lars.labels import assign_labels
    from lars.render import compose_prompt, render_content
    from lars.train.features import _render
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    labels = assign_labels(160, lambda x: len(tokenizer.encode(" " + x, add_special_tokens=False)) == 1)
    def template(system, user):
        return tokenizer.apply_chat_template([{"role": "system", "content": system}, {"role": "user", "content": user}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
    def check(record):
        body, _, _ = _render(record, labels, list(range(len(record.options))))
        prefix, suffix = compose_prompt(template, render_content(record.state), body)
        # Reserve room for tokenization differences when the trainer permutes options.
        return len(tokenizer.encode(prefix + suffix, add_special_tokens=False)) <= 1984
    return check


def prepare(out):
    data = FULL / "data"
    model = json.loads((FULL / "config.json").read_text())["model"]
    train = list(read_records(data / "train.jsonl"))
    select = list(read_records(data / "select.records.jsonl"))
    capture_path = CONTROL / "baseline-step-5366.jsonl"
    capture = load_capture(capture_path)
    if {record_key(r) for r in select} != capture.keys():
        raise ValueError("audit records and baseline predictions differ")
    by_source, audit = defaultdict(list), {}
    for r in train:
        by_source[r.source].append(r)
    source_rows = defaultdict(list)
    for r in select:
        source_rows[r.source].append((r, capture[record_key(r)]))
    for source, rows in source_rows.items():
        counts = Counter(hard_class(r) for r in by_source[source])
        errors, totals, confusion = Counter(), Counter(), Counter()
        for r, row in rows:
            gold, pred = label_name(r, row["hard_index"]), label_name(r, int(np.argmax(row["p"])))
            totals[gold] += 1
            errors[gold] += int(gold != pred)
            if gold != pred:
                confusion[(gold, pred)] += 1
        train_rows = by_source[source]
        identities = defaultdict(list)
        for r in train_rows:
            identities[decision_key(r)].append(r)
        audit[source] = {"select_n": len(rows), "select_accuracy": 1 - sum(errors.values()) / len(rows),
            "train_n": len(train_rows), "train_class_counts": dict(sorted(counts.items())),
            "class_imbalance": max(counts.values()) / min(counts.values()) if counts else None,
            "soft_target_max_below_0.6": sum(max(r.target) < .6 for r in train_rows),
            "duplicate_decision_excess": sum(len(v) - 1 for v in identities.values()),
            "duplicate_hard_label_conflict_groups": sum(len({hard_class(r) for r in v}) > 1 for v in identities.values()),
            "errors_by_class": dict(errors), "select_class_support": dict(totals),
            "top_confusions": [{"gold": a, "predicted": b, "count": n} for (a, b), n in confusion.most_common(10)]}
    chosen = sorted([s for s, a in audit.items() if s.startswith("jev-bench/") and s not in POLICY["source_exclusions"]
        and a["select_accuracy"] < .8 and a["train_n"] >= 200 and len(a["train_class_counts"]) >= 3
        and a["class_imbalance"] >= 4], key=lambda s: (audit[s]["select_accuracy"], s))[:4]
    if not chosen:
        raise ValueError("audit does not support the predeclared intervention")
    freeze(out / "protocol.json", POLICY)
    freeze(out / "audit.json", {"split": "select only", "sources": audit, "selected_sources": chosen,
        "note": "soft-label ambiguity and conflicting annotations are reported, not automatically relabeled as errors"})
    # Read only states from other partitions to reject leakage, never their predictions.
    blocked_paths = sorted(data.glob("*.records.jsonl")) + [data / "heldout.jsonl"]
    blocked_paths += sorted((ROOT / "sweep/out/accuracy-pilot/data").glob("*.jsonl"))
    blocked = {content_key(r) for p in blocked_paths for r in read_records(p)}
    blocked.update(content_key(r) for r in train)
    raw = ROOT / ".hf/hub/datasets--Praveenrajus--jev-bench/snapshots" / REVISION / "data"
    raw_paths = [raw / s.split("/", 1)[1] / "train.jsonl" for s in chosen]
    input_paths = [data / "train.jsonl", data / "select.records.jsonl", capture_path, *blocked_paths, *raw_paths]
    freeze(out / "inputs.json", {"model": model, "files": {str(p): digest(p) for p in input_paths}})
    token_ok = token_filter(model)
    refreshed = list(train)
    manifest = {"original_rows": len(train), "sources": {}}
    insertions = []
    for source, path in zip(chosen, raw_paths, strict=True):
        candidates, exclusions = clean_candidates(pairs_from_wire(path, source.split("/", 1)[1]), blocked, token_ok)
        indices = [i for i, r in enumerate(train) if r.source == source]
        records, swaps = refresh_source([train[i] for i in indices], candidates,
            audit[source]["errors_by_class"], audit[source]["select_class_support"],
            int(POLICY["replacement_fraction_per_source"] * len(indices)))
        for i, record in zip(indices, records, strict=True):
            refreshed[i] = record
        new_keys = {s["inserted"] for s in swaps}
        new_records = [r for r in records if record_key(r) in new_keys]
        insertions.extend(new_records)
        # Subsequent sources cannot reuse the same input state.
        blocked.update(content_key(r) for r in new_records)
        manifest["sources"][source] = {"eligible_pool": len(candidates), "excluded": exclusions,
            "replaced": len(swaps), "before_classes": audit[source]["train_class_counts"],
            "after_classes": dict(sorted(Counter(hard_class(r) for r in records).items())), "swaps": swaps}
    if len(train) != len(refreshed) or Counter(r.source for r in train) != Counter(r.source for r in refreshed):
        raise ValueError("source populations changed")
    if not insertions:
        raise ValueError("no supported unused training examples found")
    manifest["replaced"] = len(insertions)
    manifest["refreshed_rows"] = len(refreshed)
    manifest["original_soft_targets_unchanged"] = True
    out_data = out / "data"
    out_data.mkdir(exist_ok=True)
    for name, records in (("baseline", train), ("refreshed", refreshed), ("correction-set", insertions)):
        path = out_data / f"{name}.jsonl"
        if path.exists() and list(read_records(path)) != records:
            raise ValueError("prepared training data changed")
        write_records(records, path)
        manifest[name + "_sha256"] = digest(path)
    freeze(out / "data-manifest.json", manifest)
    return model, train


def run(out, model, train):
    data = FULL / "data"
    trained = {r.source for r in train}
    env = dict(os.environ, HF_HOME=str(ROOT / ".hf"), HF_HUB_OFFLINE="1", LARS_MLX_CACHE_GB="8", TOKENIZERS_PARALLELISM="false")
    runner = Runner(out, env)
    code = [Path(__file__), ROOT / "sweep/checkpoint_selection.py", ROOT / "sweep/experiment.py",
            *sorted((ROOT / "training/src/lars").rglob("*.py"))]
    freeze(out / "code.json", {str(p): digest(p) for p in code})
    captures, results = {}, {}
    for arm in POLICY["arms"]:
        adapter = out / "training" / arm
        history = adapter.with_name(adapter.name + ".history.json")
        steps = adapter.with_name(adapter.name + ".steps")
        records = out / "data" / f"{arm}.jsonl"
        runner.stage(arm + "-train", command=[sys.executable, "-m", "lars.train.lora", "--model", model,
            "--records", records, data / "heldout.jsonl", "--holdout-sources", ",".join(HOLD),
            "--limit", "100000", "--epochs", "2", "--max-steps", "5366", "--seed", "0",
            "--lr", "2e-5", "--rank", "8", "--scale", "20", "--dropout", "0", "--keys", "all",
            "--brier-weight", "1", "--ordinal-weight", "0", "--batch-tokens", "4096", "--max-tokens", "2048",
            "--eval-every", "500", "--out", adapter], inputs=[*code, records, data / "heldout.jsonl", out / "protocol.json"],
            outputs=[history, steps])
        h = json.loads(history.read_text())
        if h[-1]["step"] != POLICY["train_steps_per_arm"]:
            raise ValueError("fixed compute budget mismatch")
        captures[arm], results[arm] = {}, {}
        for split in POLICY["evaluation_splits"]:
            output = out / f"{arm}-{split}.jsonl"
            task = out / f"{arm}-{split}.task.json"
            atomic_json(task, {"operation": "mlx-capture", "model": model, "adapter": str(steps / "step-5366"),
                "records": str(data / f"{split}.records.jsonl"), "wire": str(data / f"{split}.wire.jsonl"), "output": str(output)})
            runner.stage(arm + "-" + split, command=[sys.executable, "-m", "sweep.full_accuracy", "--task", task],
                inputs=[*code, ROOT / "sweep/full_accuracy.py", task, steps / "step-5366",
                        data / f"{split}.records.jsonl", data / f"{split}.wire.jsonl"], outputs=[output])
            rows = load_capture(output)
            expected = {json.loads(l)["record_key"]: json.loads(l) for l in
                        (data / f"{split}.wire.jsonl").read_text().split("\n") if l.strip()}
            if rows.keys() != expected.keys() or any("p" not in r or r["hard_index"] != expected[k]["hard_index"] for k, r in rows.items()):
                raise ValueError("evaluation population or labels changed")
            if arm == "refreshed" and any(r["target"] != captures["baseline"][split][k]["target"] for k, r in rows.items()):
                raise ValueError("evaluation soft targets changed")
            captures[arm][split] = rows
            results[arm][split] = selection_metrics(rows, trained)
            atomic_json(out / "partial-results.json", results)
    deltas = {split: {key: results["refreshed"][split][key] - results["baseline"][split][key] for key in
        ("selection_score", "seen_macro_accuracy", "unseen_macro_accuracy", "macro_accuracy", "macro_brier")} for split in POLICY["evaluation_splits"]}
    atomic_json(out / "results.json", {"metrics": results, "deltas": deltas, "promotion_eligible": False,
        "reason": "single-seed development investigation; confirmation not evaluated"})
    atomic_json(out / "status.json", {"state": "complete", "deltas": deltas, "finished": time.time()})
    print("Fixed-budget data comparison complete; stopped without promotion or additional work", flush=True)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "sweep/out/data-correction-s0")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    import fcntl
    with (out / "run.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        model, train = prepare(out)
        if args.run:
            run(out, model, train)
        else:
            manifest = json.loads((out / "data-manifest.json").read_text())
            print(json.dumps({"rows": manifest["refreshed_rows"], "replaced": manifest["replaced"],
                "sources": {s: {k: v for k, v in m.items() if k != "swaps"} for s, m in manifest["sources"].items()}}, indent=2))


if __name__ == "__main__":
    main()
