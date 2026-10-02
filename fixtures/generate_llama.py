"""Reference raw logits from the Python engine's llama.cpp backend, for `lars-dev parity`.

Runs the reference engine (the same pinned `git archive` as generate.py) with its
`LlamaCppBackend` (llama-cpp-python) on a GGUF model, over Molar Triage and every single-row
question of the render fixtures, and records each row's raw label logits:

    uv venv --python 3.13 fixtures/.venv && uv pip install --python fixtures/.venv numpy pydantic llama-cpp-python
    fixtures/.venv/bin/python fixtures/generate_llama.py --model ~/Code/personal/lars/models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf

llama-cpp-python bundles its own llama.cpp, so exact equality with the SDK's pinned build is
not expected: parity is the same top label on every row, with the logit differences reported.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from generate import DEFAULT_COMMIT, extract_reference, valid_requests  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--template", default="chatml")
    parser.add_argument("--lars", type=Path, default=Path.home() / "Code" / "personal" / "lars")
    parser.add_argument("--commit", default=DEFAULT_COMMIT)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    reference, commit = extract_reference(args.lars.expanduser(), args.commit)
    sys.path.insert(0, str(reference / "src"))
    import llama_cpp

    from lars import evalset
    from lars.backends.llamacpp import LlamaCppBackend
    from lars.engine import Engine
    from lars.schema import SystemOneRequest

    backend = LlamaCppBackend(str(args.model.expanduser()), template=args.template)
    engine = Engine(backend)

    rows = []

    def record(source: str, state, question: dict) -> None:
        request = SystemOneRequest(state=state, questions={"q": question})
        keys, logits = engine.raw_logits(state, "q", request.questions["q"])
        rows.append({"id": f"{source}:{len(rows)}", "state": state, "question": question, "keys": list(keys),
                     "logits": [float(v) for v in logits]})

    molar = reference / "examples" / "molar_triage"
    for name in ["molar_triage.calibration.jsonl", "molar_triage.test.jsonl"]:
        for example in evalset.read_examples(molar / name):
            record("molar", example.state, example.question)
    for name, body in valid_requests():
        for qid, question in body["questions"].items():
            if question["type"] != "multi" and "ZERO_LOGITS" not in json.dumps(body["state"]):
                record(f"render-{name}-{qid}", body["state"], question)

    meta = {"lars_commit": commit, "backend": "lars.backends.llamacpp (llama-cpp-python)",
            "llama_cpp_python": llama_cpp.__version__, "model": args.model.name,
            "model_bytes": args.model.expanduser().stat().st_size, "template": args.template,
            "n_ctx": 8192, "n_gpu_layers": -1}
    out = args.out or HERE / "data" / f"llama-{args.model.stem.lower()}.json"
    out.write_text(json.dumps({"meta": meta, "rows": rows}, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {out.name}: {len(rows)} rows", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
