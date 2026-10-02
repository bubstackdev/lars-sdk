#!/bin/sh
# Capture raw label logits for every jev-bench validation and test file, for one model on one
# compute path and layout. Resumable: finished files are skipped.
#
#     sweep/run.sh <name> <model.gguf> <metal|cpu> <v1|v2> [extra lars-dev flags]
#
# Output: sweep/out/<name>.<path>.<layout>/<config>.<split>.jsonl. jev-bench rows stay out of
# the repository (sweep/out is ignored); they come from the lars checkout, read only.
set -eu
NAME=$1 MODEL=$2 DEVICE=$3 LAYOUT=$4
shift 4
ROOT=$(cd "$(dirname "$0")/.." && pwd)
DATA=${LARS_EVAL_DATA:-$HOME/Code/personal/lars/evals/data}
BIN=$ROOT/target/release/lars-dev
BASE=$(echo "$NAME" | sed -E 's/-dev$//; s/-(BF16|Q8_0|Q4_K_M)$//; s/-lora-.*$//')
TEMPLATE=chatml
[ -f "$ROOT/sweep/templates/$BASE.json" ] && TEMPLATE=$ROOT/sweep/templates/$BASE.json
[ "$BASE" = "Qwen3-4B-Instruct-2507" ] && TEMPLATE=chatml
DEVICE_FLAGS=""
[ "$DEVICE" = cpu ] && DEVICE_FLAGS="--cpu --threads 12"
OUT=$ROOT/sweep/out/$NAME.$DEVICE.$LAYOUT
mkdir -p "$OUT"
started=$(date +%s)
for file in "$DATA"/*.jsonl; do
    target=$OUT/$(basename "$file")
    [ -s "$target" ] && continue
    "$BIN" capture --model "$MODEL" --template "$TEMPLATE" --layout "$LAYOUT" $DEVICE_FLAGS "$@" \
        --data "$file" --out "$target.part" 2>> "$OUT/capture.log"
    mv "$target.part" "$target"
done
echo "$NAME $DEVICE $LAYOUT: $(cat "$OUT"/*.jsonl | wc -l | tr -d ' ') rows in $(( $(date +%s) - started )) s"
