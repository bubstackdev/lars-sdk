"""Phase 2b analysis: accuracy, calibration, coverage, few-label curves and calibrator transfer,
from raw logits captured by `sweep/run.sh`. CPU only; no model is loaded.

    fixtures/.venv/bin/python sweep/analyze.py sweep/out/<run> [...] [--few-label] \
        [--transfer sweep/out/<fit-run> sweep/out/<serve-run>] [--json summary.json]

Calibration, metrics and coverage use the reference engine's own functions (the pinned
`git archive` that fixtures/generate.py extracts), so numbers mean what they mean in lars:
noul questions get a Platt fit on the yes-minus-no logit, choice and score a temperature.

- raw: no calibration (temperature 1).
- per-config: each config's calibrator fitted on its own validation split (chaosnli, which has
  none, borrows mnli's), as the prototype's headline tables do.
- pooled: one calibrator per kind fitted on every validation row, the way a pack ships.
- coverage at 5%: per config, the threshold that keeps validation error at or under 5% with the
  pooled calibrator, frozen and applied to test (a config with no such threshold accepts
  nothing). Reported as macro coverage and the pooled accepted error with a Wilson interval.
- few-label: per config, n validation rows drawn at random (10 draws), a calibrator and a
  threshold fitted on just those, measured on test.

Soft targets are decoded as the corrected lars loader does: a map by option key, a vector in
option order (binary order [yes, no]) or a scalar P(yes), plain or JSON-encoded. Accuracy
always uses the hard label.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = sorted((ROOT / "fixtures" / ".reference").glob("*/src"))
if not REFERENCE:
    sys.exit("run fixtures/generate.py first: it extracts the reference engine")
sys.path.insert(0, str(REFERENCE[0]))

from lars import evalset  # noqa: E402
from lars.calibration import coverage_at_error, ece, fit_platt, fit_temperature  # noqa: E402
from lars.primitives import sigmoid, softmax  # noqa: E402

DATA = Path(os.environ.get("LARS_EVAL_DATA", Path.home() / "Code" / "personal" / "lars" / "evals" / "data"))
FEW = [10, 25, 50, 100, 200]
DRAWS = 10


def noul_yes(label: str) -> bool:
    return label in {"1", "true", "yes"}


def decode_soft(raw, kind: str, keys: list[str]):
    value = raw
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return None
    if value is None:
        return None
    binary = kind in {"noul", "multi"}
    if isinstance(value, dict):
        if binary:
            yes = sum(float(v) for k, v in value.items() if noul_yes(str(k).lower()))
            vec = [yes, sum(float(v) for v in value.values()) - yes]
        else:
            vec = [float(value.get(k, 0.0)) for k in keys]
    elif isinstance(value, (int, float)) and binary:
        vec = [float(value), 1.0 - float(value)]
    elif isinstance(value, list) and len(value) == len(keys):
        vec = [float(v) for v in value]
    else:
        return None
    vec = np.asarray(vec, dtype=np.float64)
    if not np.all(np.isfinite(vec)) or (vec < 0).any() or vec.sum() <= 0:
        return None
    return vec / vec.sum()


class Row:
    __slots__ = ("config", "kind", "keys", "z", "hard", "target")

    def __init__(self, config, kind, keys, z, hard, target):
        self.config, self.kind, self.keys, self.z, self.hard, self.target = config, kind, keys, z, hard, target


def load_split(run: Path, config: str, split: str) -> list[Row] | None:
    capture, source = run / f"{config}.{split}.jsonl", DATA / f"{config}.{split}.jsonl"
    if not capture.exists():
        return None
    captured = [json.loads(line) for line in capture.read_text().splitlines() if line.strip()]
    raw_lines = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    examples = list(evalset.read_examples(source))
    if [e.id for e in examples] != [c["id"] for c in captured]:
        raise SystemExit(f"{capture}: rows do not align with {source}")
    rows = []
    for c, raw in zip(captured, raw_lines, strict=True):
        kind, keys, z = c["kind"], c["keys"], np.asarray(c["logits"], dtype=np.float64)
        if kind in {"noul", "multi"}:
            hard = 0 if noul_yes(c["label"]) else 1
        else:
            hard = keys.index(c["label"])
        target = decode_soft(raw.get("soft_label"), kind, keys)
        if target is None:
            target = np.eye(len(keys))[hard]
        rows.append(Row(config, kind, keys, z, hard, target))
    return rows


def load_run(run: Path) -> dict[str, dict[str, list[Row]]]:
    configs = sorted({p.name.split(".")[0] for p in DATA.glob("*.jsonl")})
    out = {}
    for config in configs:
        splits = {s: load_split(run, config, s) for s in ("validation", "test")}
        if splits["test"] is None:
            raise SystemExit(f"{run}: no capture for {config}.test")
        out[config] = splits
    return out


# --------------------------------------------------------------------------- calibration


def fit(rows: list[Row]) -> dict:
    cal: dict = {"platt": {}, "temperature": {}}
    for kind in sorted({r.kind for r in rows}):
        members = [r for r in rows if r.kind == kind]
        if kind in {"noul", "multi"}:
            cal["platt"][kind] = fit_platt(np.asarray([r.z[0] - r.z[1] for r in members]),
                                           np.asarray([r.target[0] for r in members]))
        else:
            cal["temperature"][kind] = fit_temperature([r.z for r in members], [r.target for r in members])
    return cal


IDENTITY = {"platt": {}, "temperature": {}}


def predict(cal: dict, row: Row) -> np.ndarray:
    if row.kind in {"noul", "multi"}:
        if row.kind in cal["platt"]:
            a, b = cal["platt"][row.kind]
            p = sigmoid(a * (row.z[0] - row.z[1]) + b)
        else:
            p = sigmoid(row.z[0] - row.z[1])
        return np.asarray([p, 1.0 - p])
    return softmax(row.z, cal["temperature"].get(row.kind, 1.0))


def scores(cal: dict, rows: list[Row]) -> dict:
    probs = [predict(cal, r) for r in rows]
    conf = np.asarray([p.max() for p in probs])
    hit = np.asarray([float(int(p.argmax()) == r.hard) for p, r in zip(probs, rows, strict=True)])
    brier = float(np.mean([((p - r.target) ** 2).sum() for p, r in zip(probs, rows, strict=True)]))
    return {"accuracy": float(hit.mean()), "brier": brier, "ece": ece(conf, hit), "conf": conf, "hit": hit}


def threshold(cal: dict, rows: list[Row], policy: str = "empirical") -> float | None:
    """The lowest confidence to accept at, chosen on labeled `rows` for a 5% error budget.

    `empirical`: the largest coverage whose error on these rows is at most 5% (the reference's
    coverage_at_error). `wilson`: the largest coverage whose Wilson 95% upper bound on error is
    at most 5%, which accounts for how few rows the threshold was chosen on. None accepts nothing.
    """
    s = scores(cal, rows)
    if policy == "empirical":
        coverage, thr = coverage_at_error(s["conf"], s["hit"], 0.05)
        return thr if coverage > 0 else None
    order = np.argsort(-s["conf"], kind="stable")
    ranked, errors = s["conf"][order], np.cumsum(1.0 - s["hit"][order])
    best = None
    for k in range(1, len(ranked) + 1):
        if k < len(ranked) and ranked[k] == ranked[k - 1]:
            continue
        if wilson(int(errors[k - 1]), k)[1] <= 0.05:
            best = float(ranked[k - 1])
    return best


def accepted(cal: dict, thr: float | None, rows: list[Row]) -> tuple[int, int]:
    if thr is None:
        return 0, 0
    s = scores(cal, rows)
    mask = s["conf"] >= thr
    return int(mask.sum()), int((1.0 - s["hit"][mask]).sum())


def wilson(errors: int, n: int) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    z, p = 1.96, errors / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def validation_for(data: dict, config: str) -> list[Row]:
    rows = data[config]["validation"]
    return rows if rows else (data["mnli"]["validation"] if config == "chaosnli" else [])


def summarize(data: dict, pooled: dict | None = None, thresholds_from: dict | None = None) -> dict:
    """Macro metrics over configs. `pooled` (default: fitted on this run's validation) and the
    validation rows thresholds are chosen on (default: this run's) can come from another run."""
    pooled_rows = [r for c in data for r in (data[c]["validation"] or [])]
    pooled = pooled or fit(pooled_rows)
    thresholds_from = thresholds_from or data
    out = {"raw": [], "per_config": [], "pooled": []}
    policies = {"empirical": [0, 0, []], "wilson": [0, 0, []]}
    for config in data:
        test = data[config]["test"]
        out["raw"].append(scores(IDENTITY, test))
        out["per_config"].append(scores(fit(validation_for(data, config)), test))
        out["pooled"].append(scores(pooled, test))
        for policy, tally in policies.items():
            n, errors = accepted(pooled, threshold(pooled, validation_for(thresholds_from, config), policy), test)
            tally[0] += n
            tally[1] += errors
            tally[2].append(n / len(test))
    macro: dict = {name: {m: float(np.mean([s[m] for s in values])) for m in ("accuracy", "brier", "ece")}
                   for name, values in out.items()}
    for policy, (n, errors, coverages) in policies.items():
        key = "coverage_at_5pct" if policy == "empirical" else "coverage_at_5pct_wilson"
        low, high = wilson(errors, n)
        macro[key] = {"macro_coverage": float(np.mean(coverages)), "accepted": n,
                      "accepted_error": errors / n if n else 0.0, "accepted_error_95ci": [low, high]}
    macro["per_config_accuracy"] = {c: s["accuracy"] for c, s in zip(data, out["pooled"], strict=True)}
    return macro


def few_label(data: dict) -> dict:
    pooled = fit([r for c in data for r in (data[c]["validation"] or [])])
    curve = {}
    for n in FEW:
        briers, accs, covs, errs, wcovs, werrs, used = [], [], [], [], [], [], 0
        for config in data:
            validation, test = data[config]["validation"] or [], data[config]["test"]
            if len(validation) < n:
                continue
            used += 1
            rng = random.Random(f"{config}:{n}")
            for _ in range(DRAWS):
                sample = rng.sample(validation, n)
                cal = fit(sample)
                s = scores(cal, test)
                briers.append(s["brier"] - scores(pooled, test)["brier"])
                accs.append(s["accuracy"] - scores(pooled, test)["accuracy"])
                k, e = accepted(cal, threshold(cal, sample), test)
                covs.append(k / len(test))
                errs.append(e / k if k else 0.0)
                k, e = accepted(cal, threshold(cal, sample, "wilson"), test)
                wcovs.append(k / len(test))
                werrs.append((e, k))
        accepted_w = sum(k for _, k in werrs)
        curve[n] = {"configs": used, "brier_minus_pooled": float(np.mean(briers)),
                    "accuracy_minus_pooled": float(np.mean(accs)), "coverage": float(np.mean(covs)),
                    "accepted_error_mean": float(np.mean(errs)), "wilson_coverage": float(np.mean(wcovs)),
                    "wilson_accepted_error": sum(e for e, _ in werrs) / accepted_w if accepted_w else 0.0}
    return curve


def cached(run: Path, with_few_label: bool) -> dict:
    """A run's summary, recomputed only when its captures are newer than the cached one."""
    cache = run / "analysis.json"
    newest = max(p.stat().st_mtime for p in run.glob("*.jsonl"))
    if cache.exists() and cache.stat().st_mtime >= newest:
        summary = json.loads(cache.read_text())
        if "coverage_at_5pct_wilson" in summary and (not with_few_label or "few_label" in summary):
            return summary
    data = load_run(run)
    summary = summarize(data)
    if with_few_label:
        summary["few_label"] = few_label(data)
    cache.write_text(json.dumps(summary, indent=1) + "\n")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="*", type=Path)
    parser.add_argument("--few-label", action="store_true")
    parser.add_argument("--transfer", nargs=2, type=Path, metavar=("FIT_RUN", "SERVE_RUN"), action="append",
                        default=[])
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    report: dict = {"runs": {}, "transfer": []}
    print("Coverage at 5%: pooled calibrator, threshold per config chosen on validation, measured on test; "
          "empirical = error on validation at most 5%, Wilson = its 95% upper bound at most 5%.\n")
    print("| run | raw acc | per-config acc | pooled acc | pooled Brier | pooled ECE | coverage (empirical) | "
          "accepted error | coverage (Wilson) | accepted error |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for run in args.runs:
        summary = cached(run, args.few_label)
        report["runs"][run.name] = summary
        c, w = summary["coverage_at_5pct"], summary["coverage_at_5pct_wilson"]
        print(f"| {run.name} | {summary['raw']['accuracy']:.3f} | {summary['per_config']['accuracy']:.3f} | "
              f"{summary['pooled']['accuracy']:.3f} | {summary['pooled']['brier']:.3f} | "
              f"{summary['pooled']['ece']:.3f} | {c['macro_coverage']:.3f} | {c['accepted_error']:.3f} | "
              f"{w['macro_coverage']:.3f} | {w['accepted_error']:.3f} |")
    if args.few_label:
        print("\nFew-label calibration: per config, n validation rows (10 draws), against the pooled calibrator.\n")
        print("| run | n | configs | Brier minus pooled | accuracy minus pooled | coverage (empirical) | "
              "accepted error | coverage (Wilson) | accepted error |")
        print("|---|---|---|---|---|---|---|---|---|")
        for run in args.runs:
            for n, point in report["runs"][run.name]["few_label"].items():
                print(f"| {run.name} | {n} | {point['configs']} | {point['brier_minus_pooled']:+.4f} | "
                      f"{point['accuracy_minus_pooled']:+.4f} | {point['coverage']:.3f} | "
                      f"{point['accepted_error_mean']:.3f} | {point['wilson_coverage']:.3f} | "
                      f"{point['wilson_accepted_error']:.3f} |")
    for fit_run, serve_run in args.transfer:
        fitted, served = load_run(fit_run), load_run(serve_run)
        pooled = fit([r for c in fitted for r in (fitted[c]["validation"] or [])])
        own, moved = summarize(served), summarize(served, pooled=pooled, thresholds_from=fitted)
        entry = {"fit": fit_run.name, "serve": serve_run.name,
                 "accuracy": [own["pooled"]["accuracy"], moved["pooled"]["accuracy"]],
                 "brier": [own["pooled"]["brier"], moved["pooled"]["brier"]],
                 "coverage": [own["coverage_at_5pct"]["macro_coverage"], moved["coverage_at_5pct"]["macro_coverage"]],
                 "accepted_error": [own["coverage_at_5pct"]["accepted_error"],
                                    moved["coverage_at_5pct"]["accepted_error"]]}
        report["transfer"].append(entry)
        print(f"\nTransfer {fit_run.name} -> {serve_run.name} (own calibrator -> transferred): "
              f"acc {entry['accuracy'][0]:.4f} -> {entry['accuracy'][1]:.4f}, "
              f"Brier {entry['brier'][0]:.4f} -> {entry['brier'][1]:.4f}, "
              f"coverage {entry['coverage'][0]:.3f} -> {entry['coverage'][1]:.3f}, "
              f"accepted error {entry['accepted_error'][0]:.3f} -> {entry['accepted_error'][1]:.3f}")
    if args.json:
        args.json.write_text(json.dumps(report, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
