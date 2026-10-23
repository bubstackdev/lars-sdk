"""Recover a trained 8B tier, gated by a short throughput probe, then compare SDK inference."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import re
import sys
import time

import numpy as np

from lars.train.accuracy import load_capture
from lars.train.data import read_records
from sweep.accuracy_ladder import digest
from sweep.checkpoint_selection import selection_metrics
from sweep.confirmation import HOLD, canonical_sdk, apply_calibration, paired_interval
from sweep.distillation_ablation import freeze
from sweep.experiment import Runner, atomic_json
from sweep.full_accuracy import worker as shared_worker

ROOT = Path(__file__).resolve().parents[1]
FULL = ROOT / "sweep/out/accuracy-full"
DEV = ROOT / "target/release/lars-dev"
TOOLS = ROOT / "target/llama-tools/bin"
LLAMA = ROOT / "crates/lars-llama-sys/llama.cpp"
TEMPLATE = ROOT / "sweep/templates/Qwen3-8B.json"
POLICY = {
    "base": "Qwen/Qwen3-8B", "seed": 0, "rank": 8, "scale": 20, "lr": 2e-5,
    "epochs": 1, "batch_tokens": 4096, "max_tokens": 2048,
    "training": "same frozen clean gold corpus/heldout as current 2B; final checkpoint, no distillation",
    "historical_reproduction": "hyperparameter reconstruction, not byte-identical lost 8B weights/corpus",
    "probe_steps": 50, "max_projected_training_hours": 12,
    "quantization": "training-only imatrix Q3_K_M", "calibration": "same independent 1836-row calibration split as 2B",
    "comparison": "same 2471 dev decisions, current core CLI and actual Node SDK verification",
    "latency": "same three single-question requests, 20 calls each, varied state, sequential model runs; shared-machine indication only",
    "useful_gain": {"macro_accuracy": .02, "max_brier_increase": .005},
    "conversion": {"minimum_label_agreement": .85, "maximum_accuracy_difference": .08},
    "sdk": {"max_probability_difference": .001, "max_label_flips": 0},
    "stop": "stop at failed probe/conversion/SDK gate, otherwise stop after dev comparison; no other training, routing deployment or promotion",
    "interpretation": "development evidence only; historical 8B numbers are not assumed to reproduce",
}


def probe_projection(log):
    steps = re.search(r"(\d+) steps per epoch", log)
    progress = []
    for line in log.split("\n"):
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if "progress_step" in row:
            progress.append(row)
    if not steps or len(progress) < 3:
        raise ValueError("incomplete throughput probe")
    # Discard initial ten steps for compilation, but include the full population's batch shapes.
    first, last = progress[0], progress[-1]
    seconds_per_step = 60 * (last["minutes"] - first["minutes"]) / (last["progress_step"] - first["progress_step"])
    if seconds_per_step <= 0:
        raise ValueError("invalid probe timing")
    n = int(steps.group(1))
    return {"steps_per_epoch": n, "seconds_per_step": seconds_per_step,
            "projected_training_hours": seconds_per_step * n / 3600,
            "note": "estimate excludes periodic heldout evaluation, build/calibration/comparison and resource changes"}


def worker(task):
    if task["operation"] == "imatrix-text-8b":
        from transformers import AutoTokenizer
        from lars.labels import assign_labels
        from lars.render import compose_prompt, render_content
        from lars.train.features import _render
        from sweep.imatrix_text import template_fn
        tok = AutoTokenizer.from_pretrained(task["model"], local_files_only=True)
        labels = assign_labels(160, lambda x: len(tok.encode(" " + x, add_special_tokens=False)) == 1)
        records = list(read_records(task["records"]))
        random.Random(1729).shuffle(records)
        fn = template_fn(str(TEMPLATE))
        prompts = []
        for r in records[:1500]:
            body, _, _ = _render(r, labels, list(range(len(r.options))))
            prefix, suffix = compose_prompt(fn, render_content(r.state), body)
            prompts.append(prefix + suffix)
        Path(task["output"]).write_text("\n".join(prompts) + "\n")
    elif task["operation"] == "emit-8b":
        model, output = Path(task["model"]), Path(task["output"])
        output.mkdir(parents=True, exist_ok=True)
        if not (output / "model.gguf").exists():
            os.link(model, output / "model.gguf")
        identity = {"name": "lars-8b-rebuilt-dev", "version": "0.0.1", "model_sha256": digest(model)}
        cal = json.loads(Path(task["calibration"]).read_text()) | {"pack": identity}
        atomic_json(output / "calibration.json", cal)
        atomic_json(output / "pack.json", {"format": 1, "name": identity["name"], "version": identity["version"],
            "backend": "llamacpp", "model": {"file": "model.gguf", "bytes": model.stat().st_size,
                "sha256": identity["model_sha256"], "base": POLICY["base"], "license": "Apache-2.0", "quantization": "Q3_K_M"},
            "template": json.loads(TEMPLATE.read_text()), "layout": "v1",
            "runtime": {"flash_attention": "off", "llama_cpp": "fb4b273"},
            "calibration": {"file": "calibration.json", "sha256": digest(output / "calibration.json"),
                "fitted_on": "independent full-confirmation calibration split"},
            "evaluation": {"status": "experimental rebuilt model, pending development comparison"}})
    else:
        shared_worker(task)


def run(out, model):
    data = FULL / "data"
    freeze(out / "protocol.json", POLICY)
    inputs = [data / (n + ".jsonl") for n in ["train", "heldout", "calibration.wire", "dev.wire", "dev.records", "parity.wire", "parity.records"]]
    freeze(out / "inputs.json", {"model": str(model), "files": {str(p): digest(p) for p in inputs},
                                "base": json.loads((out / "base-model.json").read_text())})
    env = dict(os.environ, HF_HOME=str(ROOT / ".hf"), HF_HUB_OFFLINE="1", LARS_MLX_CACHE_GB="8", TOKENIZERS_PARALLELISM="false")
    runner = Runner(out, env)
    training_code = sorted((ROOT / "training/src/lars").rglob("*.py"))
    code = [Path(__file__), ROOT / "sweep/experiment.py", ROOT / "sweep/full_accuracy.py", ROOT / "sweep/confirmation.py", *training_code]
    freeze(out / "code.json", {str(p): digest(p) for p in code})

    def job(name, op, inputs, outputs, **params):
        task = out / (name + ".task.json")
        atomic_json(task, {"operation": op, **{k: str(v) if isinstance(v, Path) else v for k, v in params.items()}})
        runner.stage(name, command=[sys.executable, "-m", "sweep.capacity_8b", "--task", task],
                     inputs=[task, *code, *inputs], outputs=outputs)

    def train(name, probe=False):
        adapter = out / "training" / name
        history, steps = adapter.with_name(adapter.name + ".history.json"), adapter.with_name(adapter.name + ".steps")
        cmd = [sys.executable, "-m", "lars.train.lora", "--model", model, "--records", data / "train.jsonl", data / "heldout.jsonl",
            "--holdout-sources", ",".join(HOLD), "--limit", "100000", "--epochs", "1", "--seed", "0",
            "--batch-tokens", "4096", "--max-tokens", "2048", "--eval-every", "50" if probe else "500", "--out", adapter]
        if probe:
            cmd += ["--max-steps", "50"]
        runner.stage(name, command=cmd, inputs=[*training_code, data / "train.jsonl", data / "heldout.jsonl", out / "protocol.json"], outputs=[history, steps])
        h = json.loads(history.read_text())
        return steps / f"step-{h[-1]['step']}"

    train("probe", True)
    projection = probe_projection((out / "probe.log").read_text())
    probe_history = json.loads((out / "training/probe.history.json").read_text())
    projection["peak_memory_gb"] = max(row["peak_memory_gb"] for row in probe_history)
    projection["passed"] = projection["projected_training_hours"] <= POLICY["max_projected_training_hours"]
    atomic_json(out / "throughput.json", projection)
    if not projection["passed"]:
        atomic_json(out / "status.json", {"state": "stopped_at_gate", "reason": "projected training exceeds 12-hour budget", "probe": projection})
        print("Stopped at throughput gate", json.dumps(projection), flush=True)
        return
    print("Throughput gate passed", json.dumps(projection), flush=True)
    adapter = train("8b")
    fused = out / "fused-8b"
    job("fuse", "fuse", [adapter], [fused], model=model, adapter=adapter, output=fused)
    bf16 = out / "8b-BF16.gguf"
    runner.stage("convert", command=[ROOT / "sweep/.venv-convert/bin/python", LLAMA / "convert_hf_to_gguf.py", fused,
        "--outtype", "bf16", "--outfile", bf16], inputs=[fused, LLAMA / "convert_hf_to_gguf.py"], outputs=[bf16])
    text = out / "imatrix.txt"
    job("imatrix-text", "imatrix-text-8b", [data / "train.jsonl", TEMPLATE], [text], model=model, records=data / "train.jsonl", output=text)
    matrix = out / "8b.imatrix"
    runner.stage("imatrix", command=[TOOLS / "llama-imatrix", "-m", bf16, "-f", text, "--parse-special", "--chunks", "400", "-ngl", "99", "-o", matrix], inputs=[bf16, text, TOOLS / "llama-imatrix"], outputs=[matrix])
    quant = out / "8b-Q3_K_M.gguf"
    runner.stage("quantize", command=[TOOLS / "llama-quantize", "--imatrix", matrix, bf16, quant, "Q3_K_M"], inputs=[matrix, bf16, TOOLS / "llama-quantize"], outputs=[quant])

    def capture(name, weights, template, split):
        raw, canonical = out / (name + ".raw.jsonl"), out / (name + ".jsonl")
        runner.stage(name, command=[DEV, "capture", "--model", weights, "--template", template, "--data", data / (split + ".wire.jsonl"), "--out", raw],
            inputs=[DEV, weights, template, data / (split + ".wire.jsonl")], outputs=[raw])
        job(name + "-canonical", "canonical", [raw, data / (split + ".wire.jsonl"), data / (split + ".records.jsonl")], [canonical], raw=raw, data=data, dataset=split, output=canonical)
        return canonical

    parity = capture("8b-parity", quant, TEMPLATE, "parity")
    prompts = out / "8b-prompts.jsonl"
    runner.stage("8b-prompts", command=[DEV, "prompts", "--model", quant, "--template", TEMPLATE, "--data", data / "parity.wire.jsonl", "--out", prompts], inputs=[DEV, quant, TEMPLATE, data / "parity.wire.jsonl"], outputs=[prompts])
    mlx = out / "8b-mlx-parity.json"
    job("8b-mlx-parity", "mlx-prompts", [adapter, prompts], [mlx], model=model, adapter=adapter, prompts=prompts, output=mlx)
    a, b = list(load_capture(parity).values()), json.loads(mlx.read_text())
    agreement = float(np.mean([np.argmax(x["p"]) == np.argmax(y["p"]) for x, y in zip(a, b, strict=True)]))
    diff = abs(float(np.mean([np.argmax(x["p"]) == x["hard_index"] for x in a])) - float(np.mean([np.argmax(y["p"]) == x["hard_index"] for x, y in zip(a, b, strict=True)])))
    check = {"rows": len(a), "label_agreement": agreement, "accuracy_difference": diff}
    atomic_json(out / "conversion-check.json", check)
    if agreement < .85 or diff > .08:
        atomic_json(out / "status.json", {"state": "stopped_at_gate", "reason": "8B conversion parity failed", "check": check})
        return
    cal = out / "8b-calibration.json"
    runner.stage("8b-calibrate", command=[DEV, "calibrate", "--model", quant, "--template", TEMPLATE, "--data", data / "calibration.wire.jsonl", "--out", cal], inputs=[DEV, quant, TEMPLATE, data / "calibration.wire.jsonl"], outputs=[cal])
    pack = out / "pack-8b"
    job("8b-pack", "emit-8b", [quant, cal, TEMPLATE], [pack], model=quant, calibration=cal, output=pack)
    expected = out / "8b-sdk-expected.jsonl"
    runner.stage("8b-sdk-expected", command=[DEV, "eval", "--model", quant, "--template", TEMPLATE, "--calibration", cal, "--data", data / "parity.wire.jsonl", "--dump", expected], inputs=[DEV, quant, TEMPLATE, cal, data / "parity.wire.jsonl"], outputs=[expected])
    runner.stage("8b-sdk-pack", command=["node", ROOT / "sweep/check_pack.mjs", pack, data / "parity.wire.jsonl", expected, out / "8b-sdk-pack.json"], inputs=[pack, data / "parity.wire.jsonl", expected, ROOT / "sweep/check_pack.mjs", ROOT / "ts/native/lars.darwin-arm64.node"], outputs=[out / "8b-sdk-pack.json"])
    specs = {"2b": (FULL / "baseline-s0-Q4_K_M.gguf", ROOT / "sweep/templates/Qwen3.5-2B.json", FULL / "baseline-s0-calibration.json"), "8b": (quant, TEMPLATE, cal)}
    captures, metrics, latency = {}, {}, {}
    trained = {r.source for r in read_records(data / "train.jsonl")}
    records = list(read_records(data / "dev.records.jsonl"))
    for name, (weights, template, calibration) in specs.items():
        c = capture(name + "-dev", weights, template, "dev")
        captures[name] = apply_calibration(c, json.loads(calibration.read_text()))
        metrics[name] = selection_metrics(captures[name], trained)
        latency[name] = {}
        for kind in ("noul", "choice", "score"):
            record = next(r for r in records if r.kind == kind)
            request, output = out / (kind + ".request.json"), out / f"{name}-{kind}-latency.json"
            atomic_json(request, {"state": "Case: " + json.dumps(record.state, ensure_ascii=False), "questions": {"q": record.to_question()}})
            runner.stage(f"{name}-{kind}-latency", command=[DEV, "bench", request, "--model", weights, "--template", template, "--calibration", calibration, "--repeat", "20", "--vary-state"],
                inputs=[DEV, weights, template, calibration, request], outputs=[out / f"{name}-{kind}-latency.log"])
            value = json.loads((out / f"{name}-{kind}-latency.log").read_text())
            atomic_json(output, value)
            latency[name][kind] = value
    if captures["2b"].keys() != captures["8b"].keys():
        raise ValueError("comparison populations differ")
    for k, row in captures["2b"].items():
        if row["hard_index"] != captures["8b"][k]["hard_index"] or row["target"] != captures["8b"][k]["target"]:
            raise ValueError("comparison targets differ")
    delta = metrics["8b"]["macro_accuracy"] - metrics["2b"]["macro_accuracy"]
    atomic_json(out / "results.json", {"metrics": metrics, "latency": latency, "accuracy_delta": delta,
        "paired_interval": paired_interval([captures["2b"]], [captures["8b"]]),
        "useful_development_gain": delta >= .02 and metrics["8b"]["macro_brier"] <= metrics["2b"]["macro_brier"] + .005,
        "promotion_eligible": False, "interpretation": POLICY["interpretation"]})
    atomic_json(out / "status.json", {"state": "complete", "accuracy_delta": delta, "finished": time.time()})
    print("Rebuild and bounded development comparison complete; no further work launched", flush=True)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "sweep/out/capacity-8b")
    parser.add_argument("--task", type=Path)
    args = parser.parse_args()
    if args.task:
        worker(json.loads(args.task.read_text()))
        return
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    model = Path((out / "base-path.txt").read_text().strip())
    import fcntl
    with (out / "run.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            run(out, model)
        except Exception as exc:
            path = out / "status.json"
            previous = json.loads(path.read_text()) if path.exists() else {}
            atomic_json(path, previous | {"state": "failed", "error": str(exc), "finished": time.time()})
            raise


if __name__ == "__main__":
    main()
