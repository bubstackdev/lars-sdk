"""Frozen data splits and CPU statistics for full-corpus confirmation.

Promotion is an evidence gate only. This module never publishes or replaces a pack.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import statistics

import numpy as np

from lars.train.accuracy import load_capture, metrics, record_key, state_key
from lars.train.data import from_jev_bench, read_records, write_records
from sweep import analyze as scoring
from sweep.accuracy_ladder import digest

ROOT = Path(__file__).resolve().parents[1]
REVISION = "18f88da81c28c2bec55edc31f63f2afdfba109ea"
HOLD = ["jev-bench/civil_comments", "jev-bench/fever_evidence", "jev-bench/helpsteer2_helpfulness"]
POLICY = {
    "seeds": [0, 1, 2], "epochs": 1, "batch_tokens": 4096, "max_tokens": 2048,
    "checkpoint": "final step, fixed before evaluation", "quantization": "Q4_K_M with training-only imatrix",
    "distillation": {"mix": .5, "max_changes_per_source": 500,
                     "teacher_gate": "strictly better development accuracy, gold-agreeing training answers"},
    "reference_parity": {"max_accuracy_difference": .0005, "max_brier_difference": .001},
    "reproduction": {"max_accuracy_drop": .01, "max_brier_increase": .015},
    "conversion": {"max_accuracy_difference": .08, "min_label_agreement": .85},
    "sdk_serving": {"max_probability_difference": .001, "max_label_flips": 0},
    "promotion": {"min_accuracy_gain": .01, "min_score_accuracy_gain": .03,
                  "max_overall_accuracy_drop_for_score_gain": .002,
                  "max_suite_accuracy_drop": .005, "max_brier_increase": .005,
                  "max_certified_coverage_drop": .01, "max_accepted_error": .05,
                  "max_accepted_error_upper95": .05,
                  "min_improving_seeds": 2,
                  "requires_positive_paired_ci": True},
    "calibration": "separate validation rows, per-kind pooled Platt/temperature, fitted through SDK",
    "certificate": "separate rows; per-source fixed-sequence Wilson upper bound at 5% error",
    "confirmation": "no prior pilot states; selected by fixed content hashes before predictions",
}


def ranked(rows, salt):
    return sorted(rows, key=lambda r: hashlib.sha256((salt + record_key(r)).encode()).hexdigest())


def pairs_from_wire(path: Path, config: str):
    raw = [json.loads(l) for l in path.read_text().split("\n") if l.strip()]
    records = list(from_jev_bench(raw, config))
    result = []
    for r, row in zip(records, raw, strict=True):
        q = row["question"]
        q = json.loads(q) if isinstance(q, str) else q
        if r.kind == "noul" and q.get("criteria"):
            r.descriptions = [q["criteria"].get("true"), q["criteria"].get("false")]
        if "workflow" in row:
            r = replace(r, id=f"{row['case']}:{row['qid']}", source=f"typed-decisions/{row['workflow']}")
        label = str(row["label"])
        hard = (0 if label.lower() in {"yes", "true", "1"} else 1) if r.kind == "noul" else \
               (r.options.index(label) if r.kind == "choice" else int(label))
        result.append((r, hard))
    return result


def save_dataset(directory: Path, name: str, pairs):
    records = [r for r, _ in pairs]
    write_records(records, directory / f"{name}.records.jsonl")
    with (directory / f"{name}.wire.jsonl").open("w") as handle:
        for r, hard in pairs:
            label = r.options[hard] if r.kind != "score" else str(hard)
            handle.write(json.dumps({"state": r.state, "question": r.to_question(), "label": label,
                "soft_label": r.target, "record_key": record_key(r), "source": r.source,
                "record_id": r.id, "hard_index": hard}) + "\n")


def prepare(out: Path, smoke=False):
    data = out / "data"
    data.mkdir(parents=True, exist_ok=True)
    pilot = ROOT / "sweep/out/accuracy-pilot/data"
    prior = {state_key(r) for p in pilot.glob("*.jsonl") for r in read_records(p)}
    raw = ROOT / ".hf/hub/datasets--Praveenrajus--jev-bench/snapshots" / REVISION / "data"
    datasets = defaultdict(list)
    for folder in sorted(raw.iterdir()):
        if not folder.is_dir():
            continue
        name = folder.name
        if (folder / "validation.jsonl").exists():
            rows = pairs_from_wire(folder / "validation.jsonl", name)
            datasets["select"] += rows[:100]
            datasets["dev"] += rows[100:200]
            datasets["calibration"] += rows[200:300]
            datasets["certificate"] += rows[300:]
        if not (folder / "test.jsonl").exists():
            continue
        rows = pairs_from_wire(folder / "test.jsonl", name)
        datasets["legacy"] += rows[:200]
        candidates = [(r, h) for r, h in rows[200:] if state_key(r) not in prior]
        candidates.sort(key=lambda pair: hashlib.sha256(("confirmation-v1" + record_key(pair[0])).encode()).hexdigest())
        datasets["confirmation"] += candidates[:200]
    # Whole states choose the partition across every dataset/configuration. Earlier roles
    # take priority, preventing repeated passages or multi-question cases from leaking.
    seen = set()
    for name in ("legacy", "select", "dev", "calibration", "certificate", "confirmation"):
        datasets[name] = [(r, h) for r, h in datasets[name] if state_key(r) not in seen]
        seen.update(state_key(r) for r, _ in datasets[name])
    unique_certificates = {}
    for r, hard in datasets["certificate"]:
        unique_certificates.setdefault((r.source, state_key(r)), (r, hard))
    datasets["certificate"] = list(unique_certificates.values())
    td = ROOT / "sweep/out/typed-decisions"
    for r in read_records(td / "records/typed-decisions.select.jsonl"):
        if state_key(r) in seen:
            continue
        role = "select" if int(state_key(r)[:8], 16) % 2 else "dev"
        datasets[role].append((r, int(np.argmax(r.target))))
    # typed-decisions test has already been examined; retain it as a regression check,
    # explicitly separate from the fresh confirmation claims.
    datasets["typed-regression"] = pairs_from_wire(td / "typed-decisions.test.jsonl", "typed-decisions")
    blocked = {state_key(r) for pairs in datasets.values() for r, _ in pairs}
    for split, name in (("ood", "ood"), ("test", "open-confirmation")):
        records = ranked(list(read_records(ROOT / f"data/train-soft/open-jev.{split}.jsonl")), name + "-v1")
        chosen, states = [], set()
        for r in records:
            key = state_key(r)
            if key in prior | blocked or key in states:
                continue
            chosen.append((r, int(np.argmax(r.target))))
            states.add(key)
            if len(chosen) == 1000:
                break
        datasets[name] = chosen
        blocked.update(states)
    held, train = [], []
    inputs = [ROOT / "data/train-e-soft" / f"{s}.train.jsonl" for s in ("open-jev", "jev-bench", "tasksource-jev")]
    inputs += [td / "records/typed-decisions.train.jsonl"]
    for path in inputs:
        records = list(read_records(path))
        if path.name == "tasksource-jev.train.jsonl":
            records = records[:700]
        for r in records:
            if state_key(r) in blocked or len(r.options) > 160:
                continue
            (held if r.source in HOLD else train).append(r)
    train = list({record_key(r): r for r in train}.values())
    held = [r for s in HOLD for r in ranked([r for r in held if r.source == s], "held")[:100]]
    held_states = {state_key(r) for r in held}
    train = [r for r in train if state_key(r) not in held_states]
    if smoke:
        train = ranked(train, "smoke")[:40]
        held = [next(r for r in held if r.source == s) for s in HOLD]
        for name, pairs in datasets.items():
            groups = defaultdict(list)
            for r, h in pairs:
                groups[r.source].append((r, h))
            datasets[name] = [p for group in groups.values() for p in group[:1]]
    for name, pairs in datasets.items():
        save_dataset(data, name, pairs)
    write_records(train, data / "train.jsonl")
    write_records(held, data / "heldout.jsonl")
    # Assert every role's states are disjoint, except legacy evaluation versus the
    # already-seen typed regression set. Neither can influence confirmation selection.
    partitions = {n: {state_key(r) for r, _ in pairs} for n, pairs in datasets.items()}
    partitions["train"] = {state_key(r) for r in train}
    partitions["heldout"] = {state_key(r) for r in held}
    for a in partitions:
        for b in partitions:
            if a < b and partitions[a] & partitions[b]:
                raise ValueError(f"states overlap between {a} and {b}")
    manifest = {"rows": {n: len(pairs) for n, pairs in datasets.items()}, "train": len(train), "heldout": len(held),
                "prior_pilot_states": len(prior), "inputs": {str(p): digest(p) for p in inputs},
                "outputs": {p.name: digest(p) for p in sorted(data.glob("*.jsonl"))},
                "benchmark_revision": REVISION, "smoke": smoke}
    (out / "data-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (out / "protocol.json").write_text(json.dumps(POLICY | {"smoke": smoke}, indent=2) + "\n")
    selected = []
    for source in ("boolq", "banking77", "mnli", "sms_spam", "arc_challenge", "massive", "paws", "helpsteer2_helpfulness"):
        selected += [(r,h) for r,h in datasets["select"] if r.source == "jev-bench/"+source][:10]
    save_dataset(data, "parity", selected)
    return manifest


def row_keys(record):
    return [str(i) for i in range(len(record.options))] if record.kind == "score" else record.options


def canonical_sdk(raw_path: Path, data: Path, name: str, out: Path):
    raw = [json.loads(l) for l in raw_path.read_text().split("\n") if l.strip()]
    wire = [json.loads(l) for l in (data / f"{name}.wire.jsonl").read_text().split("\n") if l.strip()]
    records = list(read_records(data / f"{name}.records.jsonl"))
    if len(raw) != len(records):
        raise ValueError("SDK capture population differs")
    with out.open("w") as handle:
        for row, meta, r in zip(raw, wire, records, strict=True):
            keys = row_keys(r)
            if row["kind"] != r.kind or row["keys"] != keys or str(row["label"]) != str(meta["label"]):
                raise ValueError(f"SDK capture order differs for {r.id}")
            z = np.asarray(row["logits"], dtype=float)
            p = np.exp(z - z.max()); p /= p.sum()
            handle.write(json.dumps({"key": record_key(r), "state_key": state_key(r), "id": r.id,
                "source": r.source, "kind": r.kind, "target": r.target, "hard_index": meta["hard_index"],
                "keys": keys, "logits": z.tolist(), "p": p.tolist()}) + "\n")


def apply_calibration(capture: Path, cal: dict, allow_skipped=False):
    output = {}
    cal = {"platt": cal.get("platt", {}), "temperature": cal.get("temperatures", cal.get("temperature", {}))}
    for key, row in load_capture(capture).items():
        if "logits" not in row:
            if allow_skipped:
                continue
            raise ValueError("confirmation cannot silently exclude long rows")
        r = scoring.Row(row["source"], row["kind"], row.get("keys", []), np.asarray(row["logits"]),
                        row.get("hard_index", int(np.argmax(row["target"]))), np.asarray(row["target"]))
        output[key] = row | {"p": scoring.predict(cal, r).tolist()}
    return output


def summary(capture):
    groups = defaultdict(list)
    for row in capture.values():
        groups[row["source"]].append(row)
    per = {s: metrics(rows) for s, rows in sorted(groups.items())}
    scores = [r for r in capture.values() if r["kind"] == "score"]
    return {"n": len(capture), "macro_accuracy": statistics.mean(m["accuracy"] for m in per.values()),
            "macro_brier": statistics.mean(m["brier"] for m in per.values()), "per_source": per,
            "scores": metrics(scores)}


def certificates(calibrated, test):
    groups, tests = defaultdict(list), defaultdict(list)
    for r in calibrated.values():
        groups[r["source"]].append(r)
    for r in test.values():
        tests[r["source"]].append(r)
    per, accepted, errors = {}, 0, 0
    for source, rows in tests.items():
        cert = groups.get(source, [])
        members = [scoring.Row(source, r["kind"], r.get("keys", []), np.log(np.maximum(r["p"], 1e-12)),
                    r["hard_index"], np.asarray(r["target"])) for r in cert]
        # threshold() treats noul logits as a log odds difference and score/choice as log p;
        # identity therefore recovers these calibrated distributions without refitting.
        threshold = scoring.threshold(scoring.IDENTITY, members, "fixed") if members else None
        kept = [r for r in rows if threshold is not None and max(r["p"]) >= threshold]
        wrong = sum(int(np.argmax(r["p"])) != r["hard_index"] for r in kept)
        per[source] = {"labels": len(cert), "threshold": threshold, "coverage": len(kept)/len(rows),
                       "accepted": len(kept), "errors": wrong}
        accepted += len(kept); errors += wrong
    return {"macro_coverage": statistics.mean(p["coverage"] for p in per.values()), "accepted": accepted,
            "accepted_error": errors / accepted if accepted else None,
            "accepted_error_95ci": list(scoring.wilson(errors, accepted)), "per_source": per}


def pilot_analysis(out: Path):
    root = ROOT / "sweep/out/accuracy-pilot"
    result = {}
    for split in ("final", "typed-final", "ood"):
        groups = defaultdict(list)
        for seed in POLICY["seeds"]:
            a, b = (load_capture(root / f"{recipe}-s{seed}-{split}.jsonl") for recipe in ("baseline", "distilled"))
            def compare(keys, group):
                x, y = metrics([a[k] for k in keys]), metrics([b[k] for k in keys])
                groups[group].append({"seed": seed, "n": len(keys), "baseline": x, "distilled": y,
                                      "accuracy_delta": y["accuracy"]-x["accuracy"], "brier_delta": y["brier"]-x["brier"]})
            for kind in ("noul", "choice", "score"):
                keys = [k for k in a if "p" in a[k] and "p" in b[k] and a[k]["kind"] == kind]
                if keys:
                    compare(keys, "kind/"+kind)
            for source in {r["source"] for r in a.values()}:
                compare([k for k in a if "p" in a[k] and "p" in b[k] and a[k]["source"] == source], source)
            levels = {int(np.argmax(r["target"])) for r in a.values() if r["kind"] == "score"}
            for level in sorted(levels):
                compare([k for k in a if "p" in a[k] and "p" in b[k] and a[k]["kind"] == "score"
                         and int(np.argmax(a[k]["target"])) == level], f"score_gold_level/{level}")
        result[split] = {name: {"mean_accuracy_delta": statistics.mean(r["accuracy_delta"] for r in rows),
            "mean_brier_delta": statistics.mean(r["brier_delta"] for r in rows), "seeds": rows}
            for name, rows in groups.items()}
    out.write_text(json.dumps(result, indent=2) + "\n")


def paired_interval(baselines, candidates, score_only=False, draws=2000):
    """Resample training seeds and paired state clusters within the fixed source suite."""
    rng = np.random.default_rng(20261020)
    keys = sorted(baselines[0])
    for a, b in zip(baselines, candidates, strict=True):
        if set(a) != set(keys) or set(b) != set(keys):
            raise ValueError("confirmation populations differ")
        if any(a[k]["target"] != b[k]["target"] or a[k]["hard_index"] != b[k]["hard_index"] for k in keys):
            raise ValueError("confirmation gold differs")
    if score_only:
        keys = [k for k in keys if baselines[0][k]["kind"] == "score"]
    if not keys:
        return None
    groups = defaultdict(lambda: defaultdict(list))
    for key in keys:
        r = baselines[0][key]
        groups[r["source"]][r["state_key"]].append(key)
    seed_indices = rng.integers(0, len(baselines), (draws, len(baselines)))
    all_draws = []
    for states in groups.values():
        sums, counts = [], []
        for members in states.values():
            sums.append([sum(int(np.argmax(b[k]["p"]) == b[k]["hard_index"]) -
                             int(np.argmax(a[k]["p"]) == a[k]["hard_index"]) for k in members)
                         for a,b in zip(baselines,candidates,strict=True)])
            counts.append(len(members))
        sums, counts = np.asarray(sums, float), np.asarray(counts)
        state_indices = rng.integers(0, len(sums), (draws, len(sums)))
        per_seed = sums[state_indices].sum(axis=1) / counts[state_indices].sum(axis=1)[:,None]
        all_draws.append(np.take_along_axis(per_seed, seed_indices, axis=1).mean(axis=1))
    low, high = np.quantile(np.mean(all_draws, axis=0), [.025,.975])
    return {"ci95": [float(low),float(high)], "method": "paired state clusters within source; training seeds resampled",
            "seeds": len(baselines), "draws": draws}


def promotion(suites, intervals, coverage, policy=None):
    policy = policy or POLICY["promotion"]
    checks = {}
    primary = suites["confirmation"]
    a, b = primary["baseline"], primary["distilled"]
    delta = b["accuracy"]-a["accuracy"]
    score_delta = b["score_accuracy"]-a["score_accuracy"]
    acc_ci = intervals["accuracy"]["ci95"][0]
    score_ci = intervals["score_accuracy"]["ci95"][0] if intervals["score_accuracy"] else -1
    accuracy_path = delta >= policy["min_accuracy_gain"] and acc_ci > 0 and \
        sum(v > 0 for v in primary["accuracy_deltas"]) >= policy["min_improving_seeds"]
    score_path = score_delta >= policy["min_score_accuracy_gain"] and score_ci > 0 and \
        delta >= -policy["max_overall_accuracy_drop_for_score_gain"] and \
        sum(v > 0 for v in primary["score_accuracy_deltas"]) >= policy["min_improving_seeds"]
    checks["useful_primary_gain"] = accuracy_path or score_path
    for name, suite in suites.items():
        checks[name+"_accuracy"] = suite["distilled"]["accuracy"] >= suite["baseline"]["accuracy"] - policy["max_suite_accuracy_drop"]
        checks[name+"_brier"] = suite["distilled"]["brier"] <= suite["baseline"]["brier"] + policy["max_brier_increase"]
    checks["certificate_available"] = coverage["baseline"]["macro_coverage"] > 0 and coverage["distilled"]["accepted"] > 0
    checks["certified_coverage"] = coverage["distilled"]["macro_coverage"] >= coverage["baseline"]["macro_coverage"] - policy["max_certified_coverage_drop"]
    error = coverage["distilled"]["accepted_error"]
    checks["accepted_error"] = error is not None and error <= policy["max_accepted_error"]
    checks["accepted_error_upper95"] = coverage["distilled"]["worst_seed_error_upper95"] <= policy["max_accepted_error_upper95"]
    return {"eligible": all(checks.values()), "checks": checks, "accuracy_delta": delta, "score_accuracy_delta": score_delta,
            "policy": policy, "action": "evidence for review only; no pack is automatically promoted"}
