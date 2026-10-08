#!/bin/sh
# Create the training environment, .venv-train at the repository root (gitignored), from
# training/requirements.txt. Needs uv.
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd)
uv venv --allow-existing --python 3.13 "$ROOT/.venv-train"
(cd "$ROOT" && uv pip install --python "$ROOT/.venv-train/bin/python" -r training/requirements.txt)
"$ROOT/.venv-train/bin/python" -c "import mlx.core, mlx_lm; print('mlx', mlx.core.__version__, 'mlx-lm', mlx_lm.__version__)"
