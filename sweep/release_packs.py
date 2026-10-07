"""Stage the release packs in dist/packs/, laid out as the Hub serves them (<name>/<version>/ and
<name>/index.json), each with a model card (README.md) and NOTICE.

A release pack is a dev pack under its release name and version. The calibrator is fitted
deterministically from the model and jev-bench validation, so instead of refitting it (an hour of
GPU for the same numbers) this restamps the identity the calibrator is bound to and the
manifest's hashes. The model file is hard-linked, not copied.

    python3 sweep/release_packs.py            then: LARS_TEST_PACK=dist/packs/lars-2b/0.1.0 ...
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

SDK = Path(__file__).resolve().parent.parent
OUT = SDK / "dist" / "packs"
VERSION = "0.1.0"

# Training data, attributed per the lars data audit (lars DESIGN.md §8.1). Every pack's LoRA
# trained on corpus E; the 2B and 8B also on typed-decisions train.
CORPUS_E = [
    ("Open-Jev (open-jev/train)", "CC0 1.0"),
    ("SNLI, via tasksource-jev", "CC BY-SA 4.0"),
    ("bAbI NLI, via tasksource-jev", "BSD"),
    ("MultiNLI (GLUE MNLI; jev-bench mnli)", "OANC, with parts under CC BY-SA 3.0 / CC BY 3.0"),
    ("ARC-Challenge", "CC BY-SA 4.0"),
    ("BoolQ", "CC BY-SA 3.0"),
    ("FEVER", "CC BY-SA 3.0"),
    ("BANKING77, MASSIVE, LEDGAR, HelpSteer2, Measuring Hate Speech, SMS Spam Collection", "CC BY 4.0"),
    ("CLINC150", "CC BY 3.0"),
    ("GoEmotions", "Apache 2.0"),
    ("PAWS", "free for any purpose (Google)"),
    ("Civil Comments", "CC0 1.0"),
    ("StrategyQA, MMLU", "MIT"),
    ("jev-bench (Praveenrajus/jev-bench), the format and splits of the above", "per source, as listed"),
]
TYPED_DECISIONS = ("typed-decisions (LocalLLaMA/typed-decisions), train split", "Apache 2.0")

PACKS = [
    {
        "name": "lars-2b", "source": "lars-2b-dev/0.0.1", "role": "the default",
        "base": "Qwen/Qwen3.5-2B", "base_license": "Apache 2.0", "typed_decisions": True,
        "quantization": "Q4_K_M", "coverage": "15.3% (3.1% error on accepted test answers)",
        "td": "0.759", "latency": "40 ms for a new input, 18 ms asked again; about 0.26 s on CPU (12 threads)",
        "memory": "1.8 GB",
    },
    {
        "name": "lars-8b", "source": "lars-8b-dev/0.0.2", "role": "the top tier: the most answers certified at a strict error budget",
        "base": "Qwen/Qwen3-8B", "base_license": "Apache 2.0", "typed_decisions": True,
        "quantization": "Q3_K_M", "coverage": "19.0% (2.4% error on accepted test answers)",
        "td": "0.773 (measured on the Q4_K_M build)", "latency": "120 ms for a new input, 62 ms asked again; about 0.46 s on CPU",
        "memory": "5.0 GB",
    },
    {
        "name": "lars-0.6b", "source": "lars-0.6b-dev/0.0.2", "role": "the smallest, best with labels to calibrate on",
        "base": "Qwen/Qwen3-0.6B", "base_license": "Apache 2.0", "typed_decisions": False,
        "quantization": "Q4_K_M (importance matrix)", "coverage": "6.4% (3.5% error on accepted test answers)",
        "td": "0.44 with no labels, 0.66 calibrated on 100 labels per predicate", "latency": "12-24 ms on Metal",
        "memory": "about 1 GB",
    },
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def card(spec: dict, manifest: dict) -> str:
    ev = manifest["evaluation"]
    gb = manifest["model"]["bytes"] / 1e9
    data = CORPUS_E + ([TYPED_DECISIONS] if spec["typed_decisions"] else [])
    rows = "\n".join(f"| {s} | {lic} |" for s, lic in data)
    td_note = (
        "It trained on typed-decisions' train split, so that score is in-domain. It is also above the "
        "dataset's 0.735 teacher self-agreement, which the dataset card reads as learning the teacher; "
        "it is not a measure of accuracy on decisions unlike those."
        if spec["typed_decisions"] else
        "It never saw typed-decisions, so that score is accuracy on decisions it had never seen."
    )
    return f"""# {spec["name"]} {VERSION}

A LARS SDK pack, {spec["role"]}: {spec["base"]} with the LARS LoRA, {spec["quantization"]},
{gb:.2f} GB. Open it with `Lars.open("{spec["name"]}")` (TypeScript), `Lars.OpenAsync("{spec["name"]}")`
(C#) or `Lars::open("{spec["name"]}")` (Rust), or download it ahead of time with
`npx lars pull {spec["name"]}`. It answers typed questions (yes/no, choice, score, multi) with
calibrated probabilities, in-process, and never generates text.

## Measured through the SDK, with this pack

| | |
|---|---|
| jev-bench test (22 configs) | {ev["macro_accuracy"]:.3f} macro accuracy, Brier {ev["macro_brier"]:.3f} |
| certified coverage at a 5% error budget | {spec["coverage"]} |
| typed-decisions test | {spec["td"]} |
| latency, one question, Apple M5 Max, Metal | {spec["latency"]} |
| memory | {spec["memory"]} |

- **Calibration:** the pooled calibrator was fitted on 4,120 jev-bench validation rows through this
  pack's own model, quantization and runtime (llama.cpp {manifest["runtime"]["llama_cpp"]}, flash
  attention {manifest["runtime"]["flash_attention"]}).
- **Certified coverage:** the share of answers a threshold certified on each config's validation
  rows accepts on test.
- **typed-decisions:** {td_note}
- **Latency** was measured on a shared machine; treat it as indicative.

## Limits

- It judges what the input says, and is weak on facts the input doesn't contain.
- Exact levels on fine-grained scales are the weakest primitive.
- English only.
- Calibrated probabilities and certified thresholds describe data like what they were fitted on.

## License

The weights are released under Apache 2.0. The base model, {spec["base"]}, is
{spec["base_license"]}. The LoRA trained on the sources below; see NOTICE.

| training data | license |
|---|---|
{rows}

No training row carries outputs of Jev or of TypeSafe's models. Sources whose terms don't allow
this use were excluded: Yelp, SST-5, STS-B and ANLI. Whether share-alike terms reach trained weights
is untested.
"""


def notice(spec: dict) -> str:
    data = CORPUS_E + ([TYPED_DECISIONS] if spec["typed_decisions"] else [])
    lines = "\n".join(f"- {s}: {lic}" for s, lic in data)
    return f"""{spec["name"]} {VERSION}
Copyright 2026 Moedex. Licensed under the Apache License, Version 2.0.

Base model: {spec["base"]} ({spec["base_license"]}), Copyright the Qwen team, Alibaba Cloud.

The LoRA merged into these weights was trained on data from the following sources, used under
their licenses:
{lines}
"""


def main() -> None:
    for spec in PACKS:
        src = SDK / "models" / "packs" / spec["source"]
        dest = OUT / spec["name"] / VERSION
        if dest.exists():
            shutil.rmtree(dest)
        dest.mkdir(parents=True)
        manifest = json.loads((src / "pack.json").read_text())
        calibration = json.loads((src / "calibration.json").read_text())
        manifest.update(name=spec["name"], version=VERSION)
        calibration["pack"].update(name=spec["name"], version=VERSION)
        os.link(src / manifest["model"]["file"], dest / manifest["model"]["file"])
        (dest / "calibration.json").write_text(json.dumps(calibration, indent=2) + "\n")
        manifest["calibration"]["sha256"] = sha256(dest / "calibration.json")
        manifest["calibration"]["bytes"] = (dest / "calibration.json").stat().st_size
        manifest["model"]["license"] = "Apache-2.0"
        manifest["model"]["base_license"] = spec["base_license"]
        manifest["attribution"] = "NOTICE"
        (dest / "pack.json").write_text(json.dumps(manifest, indent=2) + "\n")
        (dest / "README.md").write_text(card(spec, manifest))
        (dest / "NOTICE").write_text(notice(spec))
        (OUT / spec["name"] / "index.json").write_text(json.dumps({"latest": VERSION}) + "\n")
        print(f"{spec['name']}@{VERSION}: from {spec['source']}, {manifest['model']['bytes'] / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
