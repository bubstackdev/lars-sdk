#!/bin/sh
# The reproducibility check after the 2026-10-08 loss: retrain the Qwen3-0.6B LoRA on the rebuilt
# corpus E (training/src/lars/train/corpus_e.py) exactly as sweep/train.sh trained it, then build
# the imatrix Q4_K_M, check it against MLX, capture jev-bench and pack lars-0.6b-dev 0.0.2 as
# sweep/promote.sh did. The lost pack scored 0.650 macro accuracy (0.646 plain Q4_K_M); if this one
# lands within about a point, the rebuilt corpus and trainer reproduce the packs.
#
# Result, 2026-10-08: 0.625 from the step selection chose (2,500), 0.637 (Brier 0.418, the lost
# pack's 0.419) from the final step, 4,449. Selection had fixes in the lost trainer, so the packs
# now build from the final step, as this script and sweep/rebuild.sh do.
#
#     sh sweep/retrain06.sh        log to the terminal; about 1 h 40 min of training, alone on the GPU
#     SMOKE=1 sh sweep/retrain06.sh    5 steps on 300 records, fuse, build, check, then cleans up
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
RESULTS=$OUT/retrain06-results.txt
export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:$HOME/.cargo/bin"
# Base models live in lars-sdk's own cache (.hf/, gitignored), not the shared one, which gets pruned.
export HF_HUB_OFFLINE=1 HF_HUB_CACHE=$SDK/.hf/hub LARS_MLX_CACHE_GB=16 TMPDIR=${TMPDIR:-/tmp}
HOLDOUT=jev-bench/civil_comments,jev-bench/fever_evidence,jev-bench/helpsteer2_helpfulness,tasksource-jev/babi_nli/three-arg-relations
SKIP=jev-bench/sst5,jev-bench/yelp5,jev-bench/stsb
RECORDS="data/train-e-soft/open-jev.train.jsonl data/train-e-soft/jev-bench.train.jsonl data/train-e-soft/tasksource-jev.train.jsonl"
SELECT="data/train-soft/jev-bench.validation.jsonl"
SMOKE=${SMOKE:-0} LIMIT_ROWS=""
BASE=Qwen/Qwen3-0.6B TEMPLATE=$SDK/sweep/templates/Qwen3-0.6B.json
. "$SDK/sweep/lora_lib.sh"

mkdir -p "$OUT" "$SDK/checkpoints" "$SDK/models/fused"
for f in $RECORDS $SELECT; do
    [ -s "$SDK/$f" ] || { step "missing $f: build the corpus first (training/README.md)"; exit 1; }
done

if [ "$SMOKE" = 1 ]; then
    TRAIN_ARGS="--limit 300 --max-steps 5 --eval-every 5 --select-per-source 3 --select-unseen-per-source 3 --holdout-fraction 0.1"
    SKIP=$SKIP NAME=Qwen3-0.6B-lora-e-smoke CKPT=$OUT/smoke-ckpt
else
    TRAIN_ARGS="--limit 50000 --holdout-sources $HOLDOUT"
    NAME=Qwen3-0.6B-lora-e-s0 CKPT=$SDK/checkpoints/lora-0.6b-sdk-e-s0
fi

step "retrain06.sh started (SMOKE=$SMOKE)"
step "$NAME: train"
train "$BASE" "$CKPT" "$OUT/train-$NAME.log" || { step "FAILED train $NAME (log: $OUT/train-$NAME.log)"; exit 1; }
grep -q '"improved": false' "$CKPT/adapter_config.json" && step "NOTE $NAME: no checkpoint beat the untrained model"
ADAPTER=$(ls -d "$CKPT.steps"/step-* | sort -t- -k2 -n | tail -1)
step "$NAME: fuse, imatrix Q4_K_M, from $(basename "$ADAPTER")"
fuse "$BASE" "$ADAPTER" "$SDK/models/fused/$NAME" && build "$NAME" "$SDK/models/fused/$NAME" "$TEMPLATE" Q4_K_M \
    || { step "FAILED fuse/build $NAME"; exit 1; }
rm -rf "$SDK/models/fused/$NAME"
check "$BASE" "$ADAPTER" "$NAME-Q4_K_M" "$TEMPLATE" || exit 1

if [ "$SMOKE" = 1 ]; then
    capture "$NAME-Q4_K_M" || { step "FAILED smoke capture"; exit 1; }
    rm -rf "$CKPT" "$CKPT".* "$SDK/models/$NAME"* "$OUT"/*"$NAME"* "$OUT"/smoke-*
    step "smoke done"
    exit 0
fi

step "$NAME-Q4_K_M: capture"
capture "$NAME-Q4_K_M"
PACK_DIR=$SDK/models/packs/lars-0.6b-dev/0.0.2
if [ ! -f "$PACK_DIR/pack.json" ]; then
    step "lars-0.6b-dev@0.0.2: pack"
    "$DEV" pack --model "$SDK/models/$NAME-Q4_K_M.gguf" --name lars-0.6b-dev --version 0.0.2 --data-dir "$DATA" \
        --out "$PACK_DIR" --template "$TEMPLATE" --base "$BASE" --license Apache-2.0 --quantization Q4_K_M \
        > "$OUT/pack-lars-0.6b-dev-0.0.2.log" 2>&1 || { step "FAILED pack"; rm -rf "$PACK_DIR"; exit 1; }
fi
{
    echo "## lars-0.6b-dev 0.0.2, rebuilt $(date '+%F')"
    echo "lost pack: 0.650 macro accuracy (imatrix Q4_K_M), 0.646 plain Q4_K_M; built from $(basename "$ADAPTER")"
    jq -c '.evaluation // .eval // {} | {macro_accuracy, macro_brier}' "$PACK_DIR/pack.json"
} | tee -a "$RESULTS"
step "retrain06.sh finished"
