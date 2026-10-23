"""Sequential, resumable accuracy ablations. Never changes any other process or session.

Run from the repository root using .venv-train/bin/python. Model paths may point to
read-only cached snapshots. Each stage logs its exact command, inputs, and completion.
The pilot uses validation only; --final evaluates the selected candidates on test.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import replace
import hashlib
from importlib.metadata import distributions
import json
import os
from pathlib import Path
import random
import subprocess
import statistics
import sys
import time

from lars.train.accuracy import load_capture, record_key, report, state_key
from lars.train.data import from_jev_bench, read_records, write_records

ROOT = Path(__file__).resolve().parents[1]
HOLD = ["jev-bench/civil_comments", "jev-bench/fever_evidence", "jev-bench/helpsteer2_helpfulness"]


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def balanced(records, limit: int, seed: int):
    rng = random.Random(seed)
    groups = defaultdict(list)
    for r in records:
        groups[r.source, r.kind].append(r)
    for group in groups.values():
        rng.shuffle(group)
    output = []
    while groups and len(output) < limit:
        for key in sorted(list(groups)):
            output.append(groups[key].pop())
            if not groups[key]:
                del groups[key]
            if len(output) == limit:
                break
    return output


def prepare(out: Path, limit: int, eval_rows: int, seed: int):
    """Fix train/selection/audit/final partitions before looking at predictions."""
    data = out / "data"
    data.mkdir(parents=True, exist_ok=True)
    evals = {}
    for split in ("validation", "test"):
        rows = []
        for path in sorted((ROOT / "data/evals").glob(f"*.{split}.jsonl")):
            wire = [json.loads(line) for line in path.read_text().split("\n") if line.strip()]
            rows.extend(from_jev_bench(wire, path.name.split(".")[0]))
        evals[split] = rows
    td = ROOT / "sweep/out/typed-decisions"
    typed_final = []
    for line in (td / "typed-decisions.test.jsonl").read_text().split("\n"):
        if not line.strip():
            continue
        row = json.loads(line)
        record = next(from_jev_bench([row], "typed-decisions"))
        record = replace(record, id=f"{row['case']}:{row['qid']}", source=f"typed-decisions/{row['workflow']}")
        if record.kind == "noul" and row["question"].get("criteria"):
            criteria = row["question"]["criteria"]
            record.descriptions = [criteria.get("true"), criteria.get("false")]
        typed_final.append(record)
    selection = list(read_records(td / "records/typed-decisions.select.jsonl"))
    # Validation is partitioned by state, keeping related decisions together.
    def selection_state(r):
        return int(state_key(r)[:8], 16) % 2 == 0
    final_states = {state_key(r) for r in [*evals["test"], *typed_final]}
    td_selection_states = {state_key(r) for r in selection}
    selection += [r for r in evals["validation"] if selection_state(r) or state_key(r) in td_selection_states]
    selection = [r for r in selection if state_key(r) not in final_states]
    selection_states = {state_key(r) for r in selection}
    audit = [r for r in evals["validation"] if state_key(r) not in selection_states | final_states]
    selection = balanced(selection, eval_rows * len({r.source for r in selection}), seed)
    audit = balanced(audit, eval_rows * len({r.source for r in audit}), seed)
    blocked = {state_key(r) for r in [*evals["validation"], *evals["test"], *typed_final,
                                    *read_records(td / "records/typed-decisions.select.jsonl")]}
    train = []
    files = [ROOT / "data/train-e-soft" / name for name in
             ("open-jev.train.jsonl", "jev-bench.train.jsonl", "tasksource-jev.train.jsonl")]
    files.append(td / "records/typed-decisions.train.jsonl")
    for path in files:
        rows = list(read_records(path))
        if path.name == "tasksource-jev.train.jsonl":
            rows = rows[:700]  # keep the historical corpus-E tasksource budget
        train.extend(r for r in rows if len(r.options) <= 160 and state_key(r) not in blocked)
    unique = {record_key(r): r for r in train}
    held = balanced([r for r in unique.values() if r.source in HOLD], eval_rows * len(HOLD), seed)
    train = balanced([r for r in unique.values() if r.source not in HOLD], limit, seed)
    for name, records in (("train", train), ("heldout", held), ("select", selection), ("audit", audit),
                          ("final", evals["test"]), ("typed-final", typed_final)):
        write_records(records, data / f"{name}.jsonl")
    # Freeze a separate Open-Jev OOD sample; none of its rows enter training or selection.
    ood = balanced(list(read_records(ROOT / "data/train-soft/open-jev.ood.jsonl")), 1000, seed)
    ood_states = {state_key(r) for r in ood}
    train = [r for r in train if state_key(r) not in ood_states]
    write_records(train, data / "train.jsonl")
    for name, records in (("select", selection), ("audit", audit), ("heldout", held)):
        write_records([r for r in records if state_key(r) not in ood_states], data / f"{name}.jsonl")
    write_records(ood, data / "ood.jsonl")
    manifest = {"limit": limit, "eval_rows": eval_rows, "seed": seed, "heldout_sources": HOLD,
                "train_kinds": dict(Counter(r.kind for r in train)),
                "train_sources": dict(Counter(r.source for r in train)),
                "inputs": {str(p.relative_to(ROOT)): digest(p) for p in files},
                "outputs": {p.name: digest(p) for p in data.glob("*.jsonl")}}
    (out / "data-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return data


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--teacher", required=True)
    p.add_argument("--out", type=Path, default=Path("sweep/out/accuracy-pilot"))
    p.add_argument("--limit", type=int, default=4000)
    p.add_argument("--steps", type=int, default=250, help="0: full epoch")
    p.add_argument("--eval-rows", type=int, default=30)
    p.add_argument("--seeds", default="0")
    p.add_argument("--final", action="store_true", help="evaluate frozen winner and baseline on test/OOD")
    args = p.parse_args()
    if args.limit < 1 or args.steps < 0 or args.eval_rows < 1:
        p.error("limit and eval-rows must be positive; steps must be nonnegative")
    os.chdir(ROOT)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    config = vars(args) | {"out": str(out)}
    config_path = out / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("output directory belongs to a different configuration; use a fresh directory")
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    lock = out / "run.lock"
    import fcntl
    with lock.open("w") as lockfile:
        fcntl.flock(lockfile, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args, out)


def run(args, out):
    data = prepare(out, args.limit, args.eval_rows, 1729)
    env = dict(os.environ, HF_HOME=str(ROOT / ".hf"), HF_HUB_OFFLINE="1", LARS_MLX_CACHE_GB="8",
               TOKENIZERS_PARALLELISM="false")
    code = [*sorted((ROOT / "training/src/lars").rglob("*.py")), Path(__file__)]
    code_hash = hashlib.sha256("".join(digest(p) for p in code).encode()).hexdigest()
    (out / "environment.json").write_text(json.dumps({"python": sys.version, "code_sha256": code_hash,
        "packages": {d.metadata["Name"]: d.version for d in distributions()},
        "model": args.model, "teacher": args.teacher}, indent=2) + "\n")
    def stage(name, module, options, inputs=(), outputs=()):
        cmd = [sys.executable, "-m", module, *map(str, options)]
        identity = {"command": cmd, "code": code_hash, "inputs": {str(p): digest(Path(p)) for p in inputs}}
        done = out / f"{name}.done.json"
        previous = json.loads(done.read_text()) if done.exists() else {}
        intact = all(Path(p).is_file() and previous.get("outputs", {}).get(str(p)) == digest(Path(p)) for p in outputs)
        if previous.get("identity") == identity and intact:
            print(f"skip {name}: verified completed stage", flush=True)
            return
        status = {"stage": name, "state": "running", "command": cmd, "started": time.time()}
        (out / "status.json").write_text(json.dumps(status, indent=2))
        print(f"start {name}", flush=True)
        with (out / f"{name}.log").open("w") as log:
            result = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
        status.update(state="complete" if result.returncode == 0 else "failed", finished=time.time())
        (out / "status.json").write_text(json.dumps(status, indent=2))
        if result.returncode:
            raise RuntimeError(f"{name} failed: see {out / (name + '.log')}")
        done.write_text(json.dumps({"identity": identity, "finished": time.time(),
                                    "outputs": {str(p): digest(Path(p)) for p in outputs if Path(p).is_file()}}, indent=2))
        print(f"complete {name}", flush=True)

    def cap(name, model, records, adapter=None, head=None, projection=None):
        target = out / f"{name}.jsonl"
        options = ["capture", "--model", model, "--records", records, "--out", target]
        inputs = [records]
        if adapter:
            options += ["--adapter", adapter]
            inputs += [adapter / "adapters.safetensors", adapter / "adapter_config.json"]
        if head:
            options += ["--head", head, "--projection", projection]
            inputs += [head, projection]
        stage(name, "lars.train.accuracy", options, inputs, [target])
        return target

    def train(name, records, seed, weight=0):
        adapter = out / "adapters" / name
        options = ["--model", args.model, "--records", records, data / "heldout.jsonl", "--limit", args.limit + 10000,
                   "--holdout-sources", ",".join(HOLD), "--select-records", data / "select.jsonl",
                   "--select-per-source", args.eval_rows, "--select-unseen-per-source", args.eval_rows,
                   "--max-steps", args.steps, "--eval-every", max(1, args.steps or 500),
                   "--batch-tokens", 2048, "--ordinal-weight", weight, "--seed", seed, "--out", adapter]
        stage(name, "lars.train.lora", options, [records, data / "heldout.jsonl", data / "select.jsonl"],
              [adapter / "adapters.safetensors", adapter / "adapter_config.json"])
        return adapter

    # Teacher quality is measured before it supplies any targets.
    base_audit = cap("untrained-audit", args.model, data / "audit.jsonl")
    teacher_audit = cap("teacher-audit", args.teacher, data / "audit.jsonl")
    quality = report({"untrained": load_capture(base_audit), "teacher": load_capture(teacher_audit)})
    (out / "teacher-quality.json").write_text(json.dumps(quality, indent=2))
    teacher_ok = quality["models"]["teacher"]["macro_accuracy"] > quality["models"]["untrained"]["macro_accuracy"]
    caps = {}
    for seed in map(int, args.seeds.split(",")):
        baseline = train(f"baseline-s{seed}", data / "train.jsonl", seed)
        ordinal = train(f"ordinal-s{seed}", data / "train.jsonl", seed, 1.)
        caps[f"baseline-s{seed}"] = cap(f"baseline-s{seed}-audit", args.model, data / "audit.jsonl", baseline)
        caps[f"ordinal-s{seed}"] = cap(f"ordinal-s{seed}-audit", args.model, data / "audit.jsonl", ordinal)
        # Use the trained baseline to target difficult examples, not the untrained student.
        if teacher_ok:
            teacher_train = cap("teacher-train", args.teacher, data / "train.jsonl")
            student_train = cap(f"baseline-s{seed}-train", args.model, data / "train.jsonl", baseline)
            distilled = data / f"distilled-s{seed}.jsonl"
            stage(f"distill-s{seed}", "lars.train.accuracy", ["distill", "--records", data / "train.jsonl",
                  "--exclude", data / "select.jsonl", data / "heldout.jsonl", data / "audit.jsonl",
                  "--teacher", teacher_train, "--student", student_train, "--teacher-audit", teacher_audit,
                  "--student-audit", caps[f"baseline-s{seed}"], "--out", distilled],
                  [data / "train.jsonl", data / "select.jsonl", data / "heldout.jsonl", data / "audit.jsonl",
                   teacher_train, student_train, teacher_audit, caps[f"baseline-s{seed}"]], [distilled])
            adapter = train(f"distilled-s{seed}", distilled, seed)
            caps[f"distilled-s{seed}"] = cap(f"distilled-s{seed}-audit", args.model, data / "audit.jsonl", adapter)
        features = out / f"features-s{seed}"
        stage(f"features-s{seed}", "lars.train.extract", ["--model", args.model, "--adapter", baseline,
              "--records", data / "train.jsonl", data / "heldout.jsonl", "--limit", args.limit + 10000,
              "--holdout-sources", ",".join(HOLD), "--max-options", 160, "--max-tokens", 2048,
              "--proj-dim", 256, "--shuffles", 1, "--seed", seed, "--out", features],
              [data / "train.jsonl", data / "heldout.jsonl", baseline / "adapters.safetensors"],
              [features / "train.npz", features / "heldout.npz", features / "projection.npy"])
        head = out / f"shared-head-s{seed}.npz"
        stage(f"head-s{seed}", "lars.train.residual", ["--train", features / "train.npz", "--heldout",
              features / "heldout.npz", "--epochs", 10, "--out", head],
              [features / "train.npz", features / "heldout.npz"], [head])
        caps[f"shared-head-s{seed}"] = cap(f"shared-head-s{seed}-audit", args.model, data / "audit.jsonl",
                                          baseline, head, features / "projection.npy")
        results = report({name: load_capture(path) for name, path in caps.items()})
        per_recipe = defaultdict(list)
        for name, m in results["models"].items():
            per_recipe[name.rsplit("-s", 1)[0]].append(m["macro_accuracy"])
        results["recipe_summary"] = {name: {"seeds": len(values), "mean_macro_accuracy": statistics.mean(values),
            "std_macro_accuracy": statistics.stdev(values) if len(values) > 1 else None}
            for name, values in per_recipe.items()}
        (out / "audit-results.json").write_text(json.dumps(results, indent=2) + "\n")
    if args.final:
        # Choose the recipe by mean validation macro accuracy across seeds, then compare all
        # its seeds with their matching baselines on the untouched final suite and OOD set.
        recipes = defaultdict(list)
        for name, model in results["models"].items():
            recipes[name.rsplit("-s", 1)[0]].append(model["macro_accuracy"])
        winner = max(recipes, key=lambda name: sum(recipes[name]) / len(recipes[name]))
        (out / "selected-recipe.json").write_text(json.dumps({"recipe": winner, "validation": recipes}, indent=2))
        for split in ("final", "typed-final", "ood"):
            final_caps = {}
            for seed in map(int, args.seeds.split(",")):
                for recipe in dict.fromkeys(["baseline", winner]):
                    name = f"{recipe}-s{seed}"
                    is_head = recipe == "shared-head"
                    adapter = out / "adapters" / (f"baseline-s{seed}" if is_head else name)
                    final_caps[name] = cap(f"{name}-{split}", args.model, data / f"{split}.jsonl", adapter,
                        out / f"shared-head-s{seed}.npz" if is_head else None,
                        out / f"features-s{seed}/projection.npy" if is_head else None)
            (out / f"{split}-results.json").write_text(json.dumps(report(
                {name: load_capture(path) for name, path in final_caps.items()}), indent=2) + "\n")
    (out / "status.json").write_text(json.dumps({"state": "complete", "finished": time.time()}, indent=2))
    print("accuracy experiments complete", flush=True)


if __name__ == "__main__":
    main()
