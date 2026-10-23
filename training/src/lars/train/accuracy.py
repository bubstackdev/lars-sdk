"""Canonical probability captures, paired evaluation, and conservative training distillation.

All captures retain gold targets separately from predictions. Distillation consumes only
explicit training records and excludes states present in selection or held-out records.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np

from lars.train.data import Record, read_records, write_records


def state_key(record: Record) -> str:
    return hashlib.sha256(json.dumps(record.state, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def record_key(record: Record) -> str:
    """Identity includes question and options; several decisions can share one state."""
    return hashlib.sha256(json.dumps([record.source, record.id, record.state, record.to_question()],
                                    sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def capture(backend, records: list[Record], out: Path, max_tokens: int = 2048, head=None) -> None:
    from lars.labels import assign_labels
    from lars.render import compose_prompt, render_content
    from lars.train.features import _render

    labels = assign_labels(max(len(r.options) for r in records), backend.is_single_token)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Resume only matching model/adapter/data identities at the orchestrator level.
    with out.open("w") as handle:
        for i, record in enumerate(records):
            body, _, ends = _render(record, labels, list(range(len(record.options))))
            prefix, suffix = compose_prompt(backend.template(), render_content(record.state), body)
            row = {"key": record_key(record), "state_key": state_key(record), "id": record.id, "source": record.source,
                   "kind": record.kind, "target": record.target}
            if backend.count_tokens(prefix + suffix) > max_tokens:
                row["skipped"] = "max_tokens"
            else:
                tokens = [tuple(labels[:len(record.options)])]
                if head is None:
                    z = np.asarray(backend.label_logits(prefix, [suffix], tokens)[0], dtype=np.float64)
                else:
                    z, h_ans, h_opt = backend.label_logits_with_features(prefix, [suffix], tokens, [ends])[0]
                    z = head.adjust(z, head.project(h_ans), head.project(h_opt), record.kind)
                if not np.isfinite(z).all():
                    raise ValueError(f"nonfinite logits for {record.id}")
                p = np.exp(z - z.max())
                row.update(logits=z.tolist(), p=(p / p.sum()).tolist())
            handle.write(json.dumps(row) + "\n")
            if (i + 1) % 100 == 0:
                handle.flush()
                print(f"captured {i + 1}/{len(records)}", flush=True)


def load_capture(path: Path) -> dict[str, dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    out = {r["key"]: r for r in rows}
    if len(out) != len(rows):
        raise ValueError(f"duplicate decisions in {path}")
    return out


def metrics(rows: list[dict]) -> dict:
    if not rows:
        return {"n": 0}
    hit, brier, logloss, absolute, thresholds = [], [], [], [], []
    for row in rows:
        p, t = np.asarray(row["p"]), np.asarray(row["target"])
        hard = row.get("hard_index", int(t.argmax()))
        hit.append(float(p.argmax() == hard))
        brier.append(float(((p - t)**2).sum()))
        logloss.append(float(-(t * np.log(np.maximum(p, 1e-12))).sum()))
        if row["kind"] == "score":
            absolute.append(float(abs(p.argmax() - hard)))
            # Compare every operational cutoff, including non-uniform gold targets.
            thresholds.extend(((np.cumsum(p)[:-1] >= .5) == (np.cumsum(t)[:-1] >= .5)).tolist())
    result = {"n": len(rows), "accuracy": float(np.mean(hit)), "brier": float(np.mean(brier)),
              "log_loss": float(np.mean(logloss))}
    if absolute:
        result.update(score_mae=float(np.mean(absolute)), threshold_accuracy=float(np.mean(thresholds)))
    return result


def report(captures: dict[str, dict[str, dict]], seed: int = 0) -> dict:
    """Require matching populations; bootstrap paired changes by state clusters within source."""
    names = list(captures)
    if not names:
        raise ValueError("no captures")
    keys = set(captures[names[0]])
    if any(set(c) != keys for c in captures.values()):
        raise ValueError("capture populations differ")
    usable = sorted(k for k in keys if all("p" in c[k] for c in captures.values()))
    if not usable:
        raise ValueError("no shared scored decisions")
    result = {"shared_rows": len(usable), "excluded_rows": len(keys) - len(usable), "models": {}}
    groups: dict[str, list[str]] = defaultdict(list)
    for k in usable:
        reference = captures[names[0]][k]
        if any(c[k]["target"] != reference["target"] for c in captures.values()):
            raise ValueError(f"gold targets differ for {k}")
        if any(c[k].get("hard_index") != reference.get("hard_index") for c in captures.values()):
            raise ValueError(f"hard labels differ for {k}")
        groups[reference["source"]].append(k)
    for name, cap in captures.items():
        by_source = {s: metrics([cap[k] for k in ks]) for s, ks in sorted(groups.items())}
        by_kind = {kind: metrics([cap[k] for k in usable if cap[k]["kind"] == kind])
                   for kind in ("noul", "choice", "score")}
        result["models"][name] = {"all": metrics([cap[k] for k in usable]), "per_source": by_source,
                                  "per_kind": by_kind,
                                  "macro_accuracy": float(np.mean([m["accuracy"] for m in by_source.values()])),
                                  "macro_brier": float(np.mean([m["brier"] for m in by_source.values()]))}
    # Resample paired decisions, clustered by state when captures supply a state digest.
    rng = np.random.default_rng(seed)
    for name in names[1:]:
        suffix = name.rsplit("-s", 1)[-1] if "-s" in name else None
        reference = f"baseline-s{suffix}" if suffix is not None and f"baseline-s{suffix}" in captures else names[0]
        ref = captures[reference]
        source_draws = []
        for ks in groups.values():
            clusters: dict[str, list[float]] = defaultdict(list)
            for k in ks:
                a, b = captures[name][k], ref[k]
                delta = float(np.argmax(a["p"]) == a.get("hard_index", int(np.argmax(a["target"])))) - \
                        float(np.argmax(b["p"]) == b.get("hard_index", int(np.argmax(b["target"]))))
                clusters[b.get("state_key", k)].append(delta)
            sums = np.asarray([sum(v) for v in clusters.values()])
            counts = np.asarray([len(v) for v in clusters.values()])
            idx = rng.integers(0, len(sums), size=(2000, len(sums)))
            source_draws.append(sums[idx].sum(-1) / counts[idx].sum(-1))
        lo, hi = np.quantile(np.mean(source_draws, axis=0), [.025, .975])
        result["models"][name]["paired_accuracy"] = {"reference": reference,
            "delta": result["models"][name]["macro_accuracy"] - result["models"][reference]["macro_accuracy"],
            "ci95": [float(lo), float(hi)], "bootstrap": "within-source state clusters; fixed source suite"}
    return result


def distill(records: list[Record], teacher: dict[str, dict], student: dict[str, dict],
            excluded: list[Record], mix: float = .5, per_source: int = 500,
            allowed_sources: set[str] | None = None) -> tuple[list[Record], dict]:
    """Blend only teacher answers agreeing with gold, prioritizing student mistakes/uncertainty.

    Retain the full training population, changing targets only for the selected rows. Thus
    the ablation compares supervision rather than a different sampling distribution.
    """
    if not 0 <= mix <= 1 or per_source < 1:
        raise ValueError("mix must be in [0, 1]; per_source must be positive")
    exclude = {state_key(r) for r in excluded}
    by_source: dict[str, list[tuple[float, str]]] = defaultdict(list)
    eligible = [r for r in records if state_key(r) not in exclude]
    if len({record_key(r) for r in eligible}) != len(eligible):
        raise ValueError("duplicate training records")
    for r in eligible:
        if allowed_sources is not None and r.source not in allowed_sources:
            continue
        key = record_key(r)
        a, b = teacher.get(key, {}), student.get(key, {})
        if "p" not in a or "p" not in b:
            continue
        p, q = np.asarray(a["p"]), np.asarray(b["p"])
        t = np.asarray(r.target)
        if p.shape != t.shape or q.shape != t.shape or not np.isfinite(p).all() or (p < 0).any() \
                or not np.isclose(p.sum(), 1):
            raise ValueError(f"invalid teacher distribution for {r.id}")
        if a["target"] != r.target or b["target"] != r.target:
            raise ValueError(f"capture gold mismatch for {r.id}")
        if p.argmax() != t.argmax():
            continue
        priority = float(q.argmax() != t.argmax()) + 1 - float(q.max())
        by_source[r.source].append((priority, key))
    selected = {key for group in by_source.values() for _, key in sorted(group, key=lambda x: (-x[0], x[1]))[:per_source]}
    output = [replace(r, target=((1 - mix) * np.asarray(r.target) + mix * np.asarray(teacher[record_key(r)]["p"])).tolist())
              if record_key(r) in selected else r for r in eligible]
    return output, {"input": len(records), "output": len(output), "excluded_by_state": len(records) - len(eligible),
                    "distilled": len(selected), "mix": mix, "cap_per_source": per_source,
                    "allowed_sources": sorted(allowed_sources) if allowed_sources is not None else None,
                    "eligible_per_source": {s: len(g) for s, g in sorted(by_source.items())}}


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    cap = sub.add_parser("capture")
    cap.add_argument("--model", required=True)
    cap.add_argument("--adapter")
    cap.add_argument("--head", type=Path)
    cap.add_argument("--projection", type=Path)
    cap.add_argument("--records", nargs="+", required=True)
    cap.add_argument("--out", type=Path, required=True)
    cap.add_argument("--max-tokens", type=int, default=2048)
    rep = sub.add_parser("report")
    rep.add_argument("--captures", nargs="+", required=True, help="name=path, baseline first")
    rep.add_argument("--out", type=Path, required=True)
    dis = sub.add_parser("distill")
    dis.add_argument("--records", nargs="+", required=True)
    dis.add_argument("--exclude", nargs="+", required=True)
    dis.add_argument("--teacher", type=Path, required=True)
    dis.add_argument("--student", type=Path, required=True)
    dis.add_argument("--teacher-audit", type=Path)
    dis.add_argument("--student-audit", type=Path)
    dis.add_argument("--mix", type=float, default=.5)
    dis.add_argument("--per-source", type=int, default=500)
    dis.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "report":
        captures = {name: load_capture(Path(path)) for name, path in (v.split("=", 1) for v in args.captures)}
        result = report(captures)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
    elif args.command == "capture":
        from lars.backends.mlx import MLXBackend
        from lars.heads import PointerHeadScorer
        if bool(args.head) != bool(args.projection):
            parser.error("--head and --projection must be supplied together")
        head = PointerHeadScorer.load(args.head, args.projection) if args.head else None
        capture(MLXBackend(args.model, adapter=args.adapter),
                [r for path in args.records for r in read_records(path)], args.out, args.max_tokens, head)
    else:
        records = [r for path in args.records for r in read_records(path)]
        excluded = [r for path in args.exclude for r in read_records(path)]
        if bool(args.teacher_audit) != bool(args.student_audit):
            parser.error("--teacher-audit and --student-audit must be supplied together")
        allowed = None
        if args.teacher_audit:
            quality = report({"student": load_capture(args.student_audit), "teacher": load_capture(args.teacher_audit)})
            a, b = quality["models"]["teacher"]["per_source"], quality["models"]["student"]["per_source"]
            allowed = {s for s in a if a[s]["accuracy"] > b[s]["accuracy"]}
        output, manifest = distill(records, load_capture(args.teacher), load_capture(args.student),
                                   excluded, args.mix, args.per_source, allowed)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        write_records(output, args.out)
        args.out.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(json.dumps(manifest))


if __name__ == "__main__":
    main()
