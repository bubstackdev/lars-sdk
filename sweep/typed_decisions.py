"""Score captured logits on LocalLLaMA/typed-decisions (test) the way its leaderboard does.

Input: sweep/out/typed-decisions/typed-decisions.{train,test}.jsonl (one row per question, written by
the converter) and `lars-dev capture` outputs on the test rows. Each capture is scored with a
calibrator (the pack's calibration.json, or none), like a user gets it. CPU only.

    python sweep/typed_decisions.py --convert      (download and write the JSONL; needs pyarrow, huggingface_hub)
    python sweep/typed_decisions.py --records      (lars training records from the converted train split)
    python sweep/typed_decisions.py NAME=capture.jsonl[:calibration.json] ... [--detail]

Metrics: accuracy against the gold argmax, KL(gold || model) in nats, Brier and top-label ECE (10 bins),
overall, per question type and per workflow/question. The Uniform and Prior reference rows are
recomputed first, so the definitions can be checked against the published table.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

D = Path(__file__).resolve().parent / "out" / "typed-decisions"
EPS = 1e-12


def keys_of(q: dict) -> list[str]:
    if q["type"] == "noul":
        return ["yes", "no"]
    crit = q["criteria"]
    return [str(i) for i in range(len(crit))] if q["type"] == "score" else list(crit)


def gold_vector(row: dict) -> np.ndarray:
    q, soft = row["question"], row["soft_label"]
    if q["type"] == "noul":
        v = [soft.get("true", 0.0), soft.get("false", 0.0)]
    else:
        v = [soft.get(k, 0.0) for k in keys_of(q)]
    v = np.asarray(v, dtype=np.float64)
    return v / v.sum()


def load(split: str) -> list[dict]:
    rows = [json.loads(l) for l in (D / f"typed-decisions.{split}.jsonl").read_text().splitlines() if l.strip()]
    for r in rows:
        r["gold"] = gold_vector(r)
        r["type"] = r["question"]["type"]
    return rows


def calibrated(z: np.ndarray, kind: str, cal: dict | None) -> np.ndarray:
    if kind == "noul":
        a, b = (cal or {}).get("platt", {}).get("noul", [1.0, 0.0])
        p = 1.0 / (1.0 + np.exp(-(a * (z[0] - z[1]) + b)))
        return np.asarray([p, 1.0 - p])
    t = (cal or {}).get("temperatures", {}).get(kind, 1.0)
    e = np.exp((z - z.max()) / t)
    return e / e.sum()


def ece(conf: np.ndarray, hit: np.ndarray, bins: int = 10) -> float:
    idx = np.minimum((conf * bins).astype(int), bins - 1)
    return float(sum(abs(conf[idx == b].mean() - hit[idx == b].mean()) * (idx == b).sum()
                     for b in range(bins) if (idx == b).any()) / len(conf))


def metrics(rows: list[dict], preds: list[np.ndarray]) -> dict:
    hit = np.asarray([float(p.argmax() == r["gold"].argmax()) for r, p in zip(rows, preds)])
    conf = np.asarray([p.max() for p in preds])
    kl = [float((r["gold"] * (np.log(r["gold"] + EPS) - np.log(np.clip(p, EPS, 1)))).sum()) for r, p in zip(rows, preds)]
    brier_sum = [float(((p - r["gold"]) ** 2).sum()) for r, p in zip(rows, preds)]
    brier_mean = [b / len(p) for b, p in zip(brier_sum, preds)]
    return {"n": len(rows), "accuracy": float(hit.mean()), "kl": float(np.mean(kl)), "brier": float(np.mean(brier_sum)),
            "brier_per_option": float(np.mean(brier_mean)), "ece": ece(conf, hit)}


def report(name: str, rows: list[dict], preds: list[np.ndarray], detail: bool = False) -> dict:
    out = {"all": metrics(rows, preds)}
    groups: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        groups[r["type"]].append(i)
        groups[f"{r['workflow']}/{r['qid']}"].append(i)
    for g, idx in sorted(groups.items()):
        out[g] = metrics([rows[i] for i in idx], [preds[i] for i in idx])
    a = out["all"]
    types = "  ".join(f"{t} {out[t]['accuracy']:.3f}" for t in ("noul", "choice", "score") if t in out)
    print(f"{name:34s} acc {a['accuracy']:.3f}  KL {a['kl']:.3f}  Brier {a['brier']:.3f} ({a['brier_per_option']:.3f}/opt)"
          f"  ECE {a['ece']:.3f}   {types}")
    if detail:
        for g, m in out.items():
            if "/" in g:
                print(f"    {g:45s} acc {m['accuracy']:.3f}  KL {m['kl']:.3f}")
    return out


def convert() -> None:
    """One eval row per question: state, question, gold argmax as label, gold distribution as soft_label."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    D.mkdir(parents=True, exist_ok=True)
    for split in ("test", "train"):
        path = hf_hub_download("LocalLLaMA/typed-decisions", f"all/{split}-00000-of-00001.parquet", repo_type="dataset")
        lines = []
        for r in pq.read_table(path).to_pylist():
            state, questions, gold = (json.loads(r[k]) if isinstance(r[k], str) else r[k] for k in ("state", "questions", "gold"))
            for qid, q in questions.items():
                g = gold[qid]
                lines.append(json.dumps({"state": state, "question": q, "label": str(g["label"]), "soft_label": g["probabilities"],
                                         "case": r["id"], "workflow": r["workflow"], "qid": qid}))
        (D / f"typed-decisions.{split}.jsonl").write_text("\n".join(lines) + "\n")
        print(split, len(lines), "rows")


def records() -> None:
    """The train split as lars.train records: 1,000 cases to train on and 200 (50 per workflow) to
    select checkpoints with. Sources are typed-decisions/<workflow>. The test split is never used."""
    rows = [json.loads(l) for l in (D / "typed-decisions.train.jsonl").read_text().splitlines() if l.strip()]
    cases: dict[str, list[str]] = defaultdict(list)
    for r in rows:
        if r["case"] not in cases[r["workflow"]]:
            cases[r["workflow"]].append(r["case"])
    select = {c for wf in cases.values() for c in sorted(wf, key=lambda c: hashlib.sha256(c.encode()).hexdigest())[:50]}
    out = {"train": [], "select": []}
    for r in rows:
        q, gold = r["question"], gold_vector(r)
        record = {"id": f"{r['case']}:{r['qid']}", "source": f"typed-decisions/{r['workflow']}", "kind": q["type"],
                  "state": r["state"], "question": q["instructions"], "target": [float(x) for x in gold]}
        if q["type"] == "noul":
            crit = q.get("criteria") or {}
            record |= {"options": ["yes", "no"], "descriptions": [crit.get("true"), crit.get("false")] if crit else []}
        elif q["type"] == "choice":
            record |= {"options": list(q["criteria"]), "descriptions": list(q["criteria"].values())}
        else:
            record |= {"options": list(q["criteria"]), "descriptions": []}
        out["select" if r["case"] in select else "train"].append(json.dumps(record))
    (D / "records").mkdir(exist_ok=True)
    for name, lines in out.items():
        (D / "records" / f"typed-decisions.{name}.jsonl").write_text("\n".join(lines) + "\n")
        print(name, len(lines), "records")


def main() -> int:
    if "--convert" in sys.argv:
        convert()
        return 0
    if "--records" in sys.argv:
        records()
        return 0
    test, train = load("test"), load("train")
    results = {}
    results["uniform"] = report("Uniform (README 0.308/0.444/0.238)", test, [np.full(len(r["gold"]), 1 / len(r["gold"])) for r in test])
    for how in ("soft", "hard"):
        sums: dict[str, np.ndarray] = {}
        for r in train:
            key = f"{r['workflow']}/{r['qid']}"
            add = r["gold"] if how == "soft" else np.eye(len(r["gold"]))[r["gold"].argmax()]
            sums[key] = sums.get(key, 0) + add
        prior = [sums[f"{r['workflow']}/{r['qid']}"] / sums[f"{r['workflow']}/{r['qid']}"].sum() for r in test]
        results[f"prior-{how}"] = report(f"Prior, {how} (README 0.470/0.347/0.189)", test, prior)
    for spec in [a for a in sys.argv[1:] if not a.startswith("--")]:
        name, _, rest = spec.partition("=")
        capture, _, calpath = rest.partition(":")
        cap = [json.loads(l) for l in Path(capture).read_text().splitlines() if l.strip()]
        if len(cap) != len(test):
            print(f"{name}: {len(cap)} rows, expected {len(test)}; skipped")
            continue
        cal = json.loads(Path(calpath).read_text()) if calpath else None
        preds = [calibrated(np.asarray(c["logits"], dtype=np.float64), r["type"], cal) for c, r in zip(cap, test)]
        results[name] = report(name, test, preds, detail="--detail" in sys.argv)
        if cal:
            raw = [calibrated(np.asarray(c["logits"], dtype=np.float64), r["type"], None) for c, r in zip(cap, test)]
            results[name + " (raw)"] = report(name + " (raw)", test, raw)
    (D / "scores.json").write_text(json.dumps(results, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
