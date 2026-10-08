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

Not recovered yet: the training corpus (`data/train-e-soft`) and the scripts that built it.
