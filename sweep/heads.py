"""Per-predicate heads on hidden states against per-predicate recalibration, offline.

Reads captures written by sweep/heads_capture.sh (logits plus the final hidden state at the answer
position, per row) and simulates a user who has labeled n examples of one predicate:

- pack:  the pack's pooled calibrator, no labels (what a user gets out of the box)
- refit: temperature (choice, score) or Platt (noul) fitted on the n labels, on raw logits
- head:  z = log p_pack + W h + b, with W and b L2-shrunk; lambda chosen by 5-fold cross-validation
         on the n labels from a grid that includes infinity (the pack itself), so with no signal
         it falls back to the pack. Fixed-lambda heads are reported too.

Predicates: each jev-bench config (labels drawn from validation, scored on test) and each
typed-decisions workflow/question (labels drawn from train, scored on test against the soft gold).
Configs whose options change from row to row (mmlu, arc) have no fixed option set, so the head has
nothing per-option to learn; they are reported with the pack and refit only.

    python sweep/heads.py sweep/out/heads/lars-4b-dev [--n 25,50,100,200] [--draws 5]
"""

from __future__ import annotations

import argparse
import json
import math
import zlib
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

TD = Path(__file__).resolve().parent / "out" / "typed-decisions"
LAMBDAS = [0.003, 0.03, 0.3, 3.0, 30.0]
RANK = 128


def noul_yes(label: str) -> bool:
    return str(label).lower() in {"1", "true", "yes"}


def soft_vector(raw, kind: str, keys: list[str]) -> np.ndarray | None:
    value = json.loads(raw) if isinstance(raw, str) else raw
    if value is None:
        return None
    binary = kind in {"noul", "multi"}
    if isinstance(value, dict):
        if binary:
            yes = sum(float(v) for k, v in value.items() if noul_yes(k))
            vec = [yes, sum(float(v) for v in value.values()) - yes]
        else:
            vec = [float(value.get(k, 0.0)) for k in keys]
    elif isinstance(value, (int, float)) and binary:
        vec = [float(value), 1.0 - float(value)]
    elif isinstance(value, list) and len(value) == len(keys):
        vec = [float(v) for v in value]
    else:
        return None
    vec = np.asarray(vec)
    return vec / vec.sum() if vec.sum() > 0 else None


class Rows:
    """One predicate's rows: raw logits, targets, hidden states."""

    def __init__(self, kind, keys, z, target, hidden):
        self.kind, self.keys, self.z, self.target, self.h = kind, keys, z, target, hidden

    def take(self, idx):
        return Rows(self.kind, self.keys, self.z[idx], self.target[idx], self.h[idx])

    def __len__(self):
        return len(self.z)


def load_capture(dir: Path, name: str) -> tuple[list[dict], np.ndarray] | None:
    cap, hid = dir / f"{name}.jsonl", dir / f"{name}.f32"
    if not hid.exists():
        return None
    rows = [json.loads(l) for l in cap.read_text().splitlines() if l.strip()]
    h = np.fromfile(hid, dtype="<f4").reshape(len(rows), -1).astype(np.float64)
    return rows, h


def predicate(rows: list[dict], h: np.ndarray, idx: list[int]) -> Rows | None:
    sub = [rows[i] for i in idx]
    keys = {tuple(r["keys"]) for r in sub}
    kind = sub[0]["kind"]
    if len(keys) != 1:
        return None
    keys = list(keys.pop())
    targets = []
    for r in sub:
        t = soft_vector(r.get("soft_label"), kind, keys)
        if t is None:
            hard = (0 if noul_yes(r["label"]) else 1) if kind in {"noul", "multi"} else keys.index(r["label"])
            t = np.eye(len(keys))[hard]
        targets.append(t)
    return Rows(kind, keys, np.asarray([r["logits"] for r in sub]), np.asarray(targets), h[idx])


# --------------------------------------------------------------------------- probabilities


def log_softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=-1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=-1, keepdims=True))


def log_sigmoid_pair(s: np.ndarray) -> np.ndarray:
    return np.stack([-np.logaddexp(0, -s), -np.logaddexp(0, s)], axis=-1)


def calibrated(cal: dict, rows: Rows) -> np.ndarray:
    """Log-probabilities under a calibration.json-style calibrator."""
    if rows.kind in {"noul", "multi"}:
        a, b = cal.get("platt", {}).get(rows.kind, [1.0, 0.0])
        return log_sigmoid_pair(a * (rows.z[:, 0] - rows.z[:, 1]) + b)
    return log_softmax(rows.z / cal.get("temperatures", {}).get(rows.kind, 1.0))


def refit(rows: Rows) -> dict:
    if rows.kind in {"noul", "multi"}:
        s = rows.z[:, 0] - rows.z[:, 1]
        y = rows.target[:, 0]

        def f(p):
            q = log_sigmoid_pair(p[0] * s + p[1])
            return -(y * q[:, 0] + (1 - y) * q[:, 1]).mean() + 1e-4 * (p[0] - 1) ** 2

        a, b = minimize(f, [1.0, 0.0], method="L-BFGS-B").x
        return {"platt": {rows.kind: [float(a), float(b)]}}

    def g(lt):
        return -(rows.target * log_softmax(rows.z / math.exp(lt[0]))).sum(axis=1).mean()

    t = math.exp(minimize(g, [0.0], method="L-BFGS-B", bounds=[(-3, 3)]).x[0])
    return {"temperatures": {rows.kind: t}}


class Head:
    """z = a * base + b + W h, with h centered and projected onto the labeled rows' span.

    lam = inf drops W: a recalibration of the pack's output (Platt or temperature on top of it).
    W is shrunk by lam; a and b by a unit prior (towards 1 and 0)."""

    def __init__(self, lam: float):
        self.lam = lam

    def features(self, h: np.ndarray) -> np.ndarray:
        if self.basis is None:
            return np.zeros((len(h), 0))
        return (h - self.mean) @ self.basis / self.scale

    def fit(self, base: np.ndarray, h: np.ndarray, target: np.ndarray) -> Head:
        self.basis = None
        if not math.isinf(self.lam):
            self.mean = h.mean(axis=0)
            # Ridge solutions lie in the span of the training rows, so a rank-limited basis loses nothing.
            _, sv, vt = np.linalg.svd(h - self.mean, full_matrices=False)
            keep = min(RANK, int((sv > 1e-6 * sv[0]).sum()))
            self.basis, self.scale = vt[:keep].T, 1.0
            self.scale = math.sqrt((self.features(h) ** 2).sum(axis=1).mean()) or 1.0
        f = self.features(h)
        n, k = target.shape
        d = f.shape[1]
        light = 1.0

        def loss(theta):
            a, w, b = theta[0], theta[1 : 1 + d * k].reshape(d, k), theta[1 + d * k :]
            q = log_softmax(a * base + f @ w + b)
            g = (np.exp(q) - target) / n
            reg = 0.0 if d == 0 else 0.5 * self.lam * (w**2).sum() / n
            ll = -(target * q).sum() / n + reg + 0.5 * light * ((b**2).sum() + (a - 1) ** 2) / n
            ga = (g * base).sum() + light * (a - 1) / n
            gw = f.T @ g + (0 if d == 0 else self.lam * w / n)
            gb = g.sum(axis=0) + light * b / n
            return ll, np.concatenate([[ga], gw.ravel(), gb])

        theta0 = np.concatenate([[1.0], np.zeros(d * k + k)])
        theta = minimize(loss, theta0, jac=True, method="L-BFGS-B", options={"maxiter": 500}).x
        self.a, self.w, self.b = theta[0], theta[1 : 1 + d * k].reshape(d, k), theta[1 + d * k :]
        return self

    def predict(self, base: np.ndarray, h: np.ndarray) -> np.ndarray:
        return log_softmax(self.a * base + self.features(h) @ self.w + self.b)


CANDIDATES = [math.inf, *sorted(LAMBDAS, reverse=True)]


def cv_choice(base: np.ndarray, h: np.ndarray, target: np.ndarray, rng) -> tuple[float | None, float | None]:
    """(argmin, significant) over the pack (None) and heads by lambda, from 5-fold CV log losses.

    `significant` steps from the pack to the recalibration (inf) to heads with weaker shrinkage,
    each step taken only if it beats the current choice by mean + 1.96 SE < 0 on paired
    per-example losses: the rule `lars calibrate` uses to replace a pack's calibrator."""
    n = len(target)
    folds = np.array_split(rng.permutation(n), min(5, n))
    losses = {None: -(target * base).sum(axis=1)}
    for lam in CANDIDATES:
        per = np.empty(n)
        for fold in folds:
            train = np.setdiff1d(np.arange(n), fold)
            q = Head(lam).fit(base[train], h[train], target[train]).predict(base[fold], h[fold])
            per[fold] = -(target[fold] * q).sum(axis=1)
        losses[lam] = per
    best = min(losses, key=lambda lam: losses[lam].mean())
    current = None
    for lam in CANDIDATES:
        diff = losses[lam] - losses[current]
        if diff.mean() + 1.96 * diff.std(ddof=1) / math.sqrt(n) < 0:
            current = lam
    return best, current


def metrics(logq: np.ndarray, target: np.ndarray) -> dict:
    p = np.exp(logq)
    return {
        "accuracy": float((p.argmax(axis=1) == target.argmax(axis=1)).mean()),
        "log_loss": float(-(target * logq).sum(axis=1).mean()),
        "kl": float((target * (np.log(np.clip(target, 1e-12, 1)) - logq)).sum(axis=1).mean()),
        "brier": float(((p - target) ** 2).sum(axis=1).mean()),
        # Against the argmax label, the view of a user whose labels are single answers.
        "log_loss_hard": float(-logq[np.arange(len(p)), target.argmax(axis=1)].mean()),
        "brier_hard": float(((p - np.eye(p.shape[1])[target.argmax(axis=1)]) ** 2).sum(axis=1).mean()),
        "ece_hard": ece_top(p, target),
    }


def ece_top(p: np.ndarray, target: np.ndarray, bins: int = 10) -> float:
    conf, hit = p.max(axis=1), (p.argmax(axis=1) == target.argmax(axis=1)).astype(float)
    idx = np.minimum((conf * bins).astype(int), bins - 1)
    return float(sum(abs(conf[idx == b].mean() - hit[idx == b].mean()) * (idx == b).sum() for b in range(bins) if (idx == b).any()) / len(p))


def simulate(pool: Rows, test: Rows, pack_cal: dict, sizes: list[int], draws: int, seed: int, hard: bool = False) -> dict:
    out: dict = {"pack": metrics(calibrated(pack_cal, test), test.target), "n_pool": len(pool), "n_test": len(test)}
    base_pool, base_test = calibrated(pack_cal, pool), calibrated(pack_cal, test)
    for n in sizes:
        if n > len(pool):
            continue
        acc: dict[str, list[dict]] = {}
        chosen = []
        for draw in range(draws):
            rng = np.random.default_rng(seed + 1000 * n + draw)
            idx = rng.choice(len(pool), n, replace=False)
            lab = pool.take(idx)
            if hard:  # a user labels each example with one answer, not a distribution
                lab.target = np.eye(lab.target.shape[1])[lab.target.argmax(axis=1)]
            acc.setdefault("refit", []).append(metrics(calibrated(refit(lab), test), test.target))
            head = {}
            for lam in (math.inf, 3.0, 30.0):
                head[lam] = Head(lam).fit(base_pool[idx], lab.h, lab.target)
                name = "recal" if math.isinf(lam) else f"head@{lam:g}"
                acc.setdefault(name, []).append(metrics(head[lam].predict(base_test, test.h), test.target))
            best, significant = cv_choice(base_pool[idx], lab.h, lab.target, rng)
            chosen.append((str(best), str(significant)))
            for name, lam in (("head-cv", best), ("head-sig", significant)):
                if lam is None:
                    q = base_test
                else:
                    model = head.get(lam) or Head(lam).fit(base_pool[idx], lab.h, lab.target)
                    q = model.predict(base_test, test.h)
                acc.setdefault(name, []).append(metrics(q, test.target))
        out[n] = {m: {k: float(np.mean([r[k] for r in rs])) for k in rs[0]} for m, rs in acc.items()}
        out[n]["chosen"] = chosen
    return out


def predicates(dir: Path) -> list[tuple[str, Rows, Rows]]:
    found = []
    td_test, td_train = load_capture(dir, "typed-decisions.test"), load_capture(dir, "typed-decisions.train")
    if td_test and td_train:
        meta = {s: [json.loads(l) for l in (TD / f"typed-decisions.{s}.jsonl").read_text().splitlines() if l.strip()]
                for s in ("test", "train")}
        groups = sorted({f"{m['workflow']}/{m['qid']}" for m in meta["test"]})
        for g in groups:
            ix = {s: [i for i, m in enumerate(meta[s]) if f"{m['workflow']}/{m['qid']}" == g] for s in meta}
            pool, test = predicate(*td_train, ix["train"]), predicate(*td_test, ix["test"])
            if pool and test:
                found.append((f"td:{g}", pool, test))
    for val in sorted(dir.glob("*.validation.f32")):
        config = val.name.split(".")[0]
        pv, pt = load_capture(dir, f"{config}.validation"), load_capture(dir, f"{config}.test")
        if not (pv and pt):
            continue
        pool, test = predicate(*pv, list(range(len(pv[0])))), predicate(*pt, list(range(len(pt[0]))))
        if pool and test and pool.keys == test.keys:
            found.append((f"jev:{config}", pool, test))
        else:
            print(f"jev:{config}: options vary by row; head skipped")
    return found


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", type=Path)
    ap.add_argument("--pack", type=Path, help="calibration.json (default: the pack named by dir)")
    ap.add_argument("--n", default="25,50,100,200")
    ap.add_argument("--draws", type=int, default=5)
    ap.add_argument("--hard", action="store_true", help="fit on argmax labels (scored on soft targets)")
    ap.add_argument("--only", default="", help="predicate name prefix, e.g. td: or jev:")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    cal_path = args.pack or Path("models/packs") / args.dir.name / "0.0.1" / "calibration.json"
    pack_cal = json.loads(cal_path.read_text())
    sizes = [int(s) for s in args.n.split(",")]
    results = {}
    for name, pool, test in predicates(args.dir):
        if not name.startswith(args.only):
            continue
        r = simulate(pool, test, pack_cal, sizes, args.draws, seed=zlib.crc32(name.encode()) % 10_000, hard=args.hard)
        results[name] = r
        line = f"{name:44s} pack {r['pack']['accuracy']:.3f}/{r['pack']['log_loss']:.3f}"
        for n in sizes:
            if n in r:
                line += f" | n={n} refit {r[n]['refit']['accuracy']:.3f}/{r[n]['refit']['log_loss']:.3f}"
                line += f" sig {r[n]['head-sig']['accuracy']:.3f}/{r[n]['head-sig']['log_loss']:.3f}"
        print(line, flush=True)
    (args.dir / f"heads{args.tag}.json").write_text(json.dumps(results, indent=1) + "\n")
    for family in ("td:", "jev:"):
        names = [k for k in results if k.startswith(family)]
        if not names:
            continue
        print(f"\n{family} macro over {len(names)} predicates (accuracy / log loss / Brier)")
        print(f"  {'pack':16s} " + " ".join(f"{np.mean([results[k]['pack'][m] for k in names]):.3f}" for m in ("accuracy", "log_loss", "brier")))
        for n in sizes:
            have = [k for k in names if n in results[k]]
            if len(have) < len(names):
                continue
            for method in ("refit", "recal", "head@3", "head@30", "head-cv", "head-sig"):
                vals = [np.mean([results[k][n][method][m] for k in have]) for m in ("accuracy", "log_loss", "brier")]
                print(f"  n={n:<4d}{method:11s} " + " ".join(f"{v:.3f}" for v in vals))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
