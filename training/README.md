# Training

The LoRA trainer behind the packs, and what it needs, all inside lars-sdk.

- `src/lars/`: the lars reference engine and trainer. The original lars checkout was deleted
  on 2026-10-07. This is its source at `ab59a0b`, recovered from `fixtures/.reference/`, with
  the SDK's later changes replayed from the session that wrote them:
  - `train/gated_delta_chunked.py`: a differentiable, chunked gated delta rule, so Qwen3.5 trains.
    Exact against mlx-lm's reference loop: 1.6e-6 on CPU.
  - `train/lora.py`: uses it for training, and gains `--max-steps` and progress lines.
  - `train/data.py`: a noul record keeps its yes/no descriptions as criteria.
- `requirements.txt`, `setup.sh`: `sh training/setup.sh` creates `.venv-train`. mlx-lm is
  pinned at 0.31.3.
- `score_mlx.py`: scores `lars-dev prompts` output with MLX, for the build check.

## The corpus

The original corpus and its scripts were lost on 2026-10-07. Rebuilt on 2026-10-08 (gitignored,
offline from the Hugging Face cache):

    .venv-train/bin/python -m lars.train.build --out data/train-soft
    .venv-train/bin/python -m lars.train.corpus_e

- `data/train-soft`: `build.py` at its defaults. `train/data.py` now decodes every jev-bench soft
  label the way the evaluation does (`_soft_target`). Before, only dict labels were read, so
  civil_comments (a bare P(yes)) and measuring_hate_speech (a list) trained on hard labels.
- `data/train-e-soft`, corpus E: its train splits without the license-dropped sources (jev-bench
  sst5, yelp5 and stsb; tasksource ANLI): 20,000 Open-Jev, 17,285 jev-bench and 700 tasksource rows.
- Not byte-verified against the lost files. `build.py` does reproduce lars's 2026-09-22 corpus byte
  for byte at that corpus's sizes, and corpus E's step counts match the logged runs (`corpus_e.py`).
- `sh sweep/retrain06.sh` retrains the 0.6B on it, the check that it reproduces the packs.
