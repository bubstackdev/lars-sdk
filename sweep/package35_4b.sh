#!/bin/sh
# Package the Qwen3.5-4B LoRA from a surviving checkpoint (no training): fuse, convert, imatrix
# Q4_K_M, the build check, a jev-bench capture, pack lars-4b35-dev and typed-decisions through it
# (sweep/out/package35_4b-results.txt).
#
# Its training (2026-10-07) finished, but the checkpoints it would have selected (steps 500 and
# 1,000) were lost with the old lars checkout, so selection fell back to the untrained adapter.
# Step 4,500 is the best surviving one: selection 0.803, unseen sources 0.709.
#
#     sh sweep/package35_4b.sh [step]     log: sweep/out/package35_4b.log
set -u
SDK=$(cd "$(dirname "$0")/.." && pwd)
PY=$SDK/.venv-train/bin/python
CONVERT=$SDK/sweep/.venv-convert/bin/python
FIXPY=$SDK/fixtures/.venv/bin/python
LLAMA=$SDK/crates/lars-llama-sys/llama.cpp
TOOLS=$SDK/target/llama-tools/bin
DEV=$SDK/target/release/lars-dev
OUT=$SDK/sweep/out
TD=$OUT/typed-decisions
DATA=$SDK/data/evals
RESULTS=$OUT/package35_4b-results.txt
export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:$HOME/.cargo/bin"
export HF_HUB_OFFLINE=1 TMPDIR=${TMPDIR:-/tmp}
SMOKE=0 LIMIT_ROWS=""
STEP=${1:-4500}
BASE=Qwen/Qwen3.5-4B TEMPLATE=$SDK/sweep/templates/Qwen3.5-4B.json
CONVERT_ARGS=--no-mtp
NAME=Qwen3.5-4B-lora-etd-s0 PACK=lars-4b35-dev
CKPT=$SDK/checkpoints/lora-qwen3.5-4b-sdk-etd-s0.steps/step-$STEP
. "$SDK/sweep/lora_lib.sh"

step "package35_4b.sh started from step $STEP"
fuse "$BASE" "$CKPT" "$SDK/models/fused/$NAME" && build "$NAME" "$SDK/models/fused/$NAME" "$TEMPLATE" Q4_K_M \
    || { step "FAILED fuse/build $NAME"; exit 1; }
rm -rf "$SDK/models/fused/$NAME"
check "$BASE" "$CKPT" "$NAME-Q4_K_M" "$TEMPLATE" || exit 1
step "$NAME-Q4_K_M: capture"
capture "$NAME-Q4_K_M"
step "$NAME: pack $PACK"
pack_dir=$SDK/models/packs/$PACK/0.0.1
if [ -f "$pack_dir/pack.json" ] || "$DEV" pack --model "$SDK/models/$NAME-Q4_K_M.gguf" --name "$PACK" --version 0.0.1 \
        --data-dir "$DATA" --out "$pack_dir" --template "$TEMPLATE" --base "$BASE" --license Apache-2.0 \
        --quantization Q4_K_M > "$OUT/pack-$PACK.log" 2>&1; then
    jq -c '.evaluation | {macro_accuracy, macro_brier}' "$pack_dir/pack.json"
    step "$PACK: typed-decisions"
    typed_decisions "$PACK" || step "FAILED typed-decisions $PACK"
else
    step "FAILED pack $PACK"
fi
step "package35_4b.sh finished"
