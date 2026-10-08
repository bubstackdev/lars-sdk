#!/bin/sh
# Rebuild the 2026-10-07 lineup's two larger packs after the 2026-10-08 loss, one GPU job at a
# time: the Qwen3.5-2B LoRA, then the Qwen3-8B LoRA, both trained as before (corpus E plus
# typed-decisions train, the corpus-E recipe, one epoch). Each then builds from its final
# checkpoint, not the selected one: the rebuilt 0.6B scored 0.637 from its final step against
# 0.625 from the step selection chose (the lost pack: 0.650). Then the build check, a jev-bench
# capture, the pack and typed-decisions through it (sweep/out/rebuild-results.txt).
#
#   lars-2b-dev 0.0.1   Qwen3.5-2B, imatrix Q4_K_M   was 0.716 (about 7 h of training)
#   lars-8b-dev 0.0.2   Qwen3-8B, imatrix Q3_K_M     was 0.748 (about 9 h)
#
#     nohup sh sweep/rebuild.sh > sweep/out/rebuild.log 2>&1 &
#     SMOKE=1 sh sweep/rebuild.sh      5 steps per model on 300 records, tiny captures, then cleans up
#
# Resumable: each finished step is skipped (an interrupted training starts over).
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
RESULTS=$OUT/rebuild-results.txt
export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:$HOME/.cargo/bin"
export HF_HUB_OFFLINE=1 LARS_MLX_CACHE_GB=16 TMPDIR=${TMPDIR:-/tmp}
HOLDOUT=jev-bench/civil_comments,jev-bench/fever_evidence,jev-bench/helpsteer2_helpfulness,tasksource-jev/babi_nli/three-arg-relations
SKIP=jev-bench/sst5,jev-bench/yelp5,jev-bench/stsb
RECORDS="data/train-e-soft/open-jev.train.jsonl data/train-e-soft/jev-bench.train.jsonl data/train-e-soft/tasksource-jev.train.jsonl $TD/records/typed-decisions.train.jsonl"
SELECT="data/train-soft/jev-bench.validation.jsonl $TD/records/typed-decisions.select.jsonl"
SMOKE=${SMOKE:-0}
. "$SDK/sweep/lora_lib.sh"

mkdir -p "$OUT" "$SDK/checkpoints" "$SDK/models/fused"
for f in $RECORDS $SELECT; do
    case $f in /*) p=$f ;; *) p=$SDK/$f ;; esac
    [ -s "$p" ] || { step "missing $f: build the corpus first (training/README.md)"; exit 1; }
done

if [ "$SMOKE" = 1 ]; then
    TRAIN_ARGS="--limit 300 --max-steps 5 --eval-every 5 --select-per-source 3 --select-unseen-per-source 3 --holdout-fraction 0.1"
    LIMIT_ROWS=40
    # A pack is calibrated on every validation file of its data directory: a tiny one for the smoke.
    mkdir -p "$OUT/smoke-data"
    for f in boolq banking77 sms_spam; do
        for split in validation test; do head -15 "$DATA/$f.$split.jsonl" > "$OUT/smoke-data/$f.$split.jsonl"; done
    done
    PACK_DATA=$OUT/smoke-data SUFFIX=smoke
else
    TRAIN_ARGS="--limit 50000 --holdout-sources $HOLDOUT"
    LIMIT_ROWS="" PACK_DATA=$DATA SUFFIX=s0
fi

final_step() {  # checkpoint -> its last saved step's adapter directory
    ls -d "$1.steps"/step-* 2>/dev/null | sort -t- -k2 -n | tail -1
}

one() {  # hub-model template-name quant pack version [CONVERT_ARGS]
    BASE=$1 TEMPLATE=$SDK/sweep/templates/$2.json
    CONVERT_ARGS=${6:-}
    NAME=$2-lora-etd-$SUFFIX
    CKPT=$SDK/checkpoints/lora-$(echo "$2" | tr '[:upper:]' '[:lower:]')-sdk-etd-$SUFFIX
    [ "$SMOKE" = 1 ] && PACK=lars-smoke-dev || PACK=$4
    PACK_DIR=$SDK/models/packs/$PACK/$5
    step "$NAME: train"
    train "$BASE" "$CKPT" "$OUT/train-$NAME.log" || { step "FAILED train $NAME (log: $OUT/train-$NAME.log)"; return 1; }
    ADAPTER=$(final_step "$CKPT")
    [ -n "$ADAPTER" ] || { step "FAILED $NAME: no saved steps in $CKPT.steps"; return 1; }
    step "$NAME: fuse, imatrix $3, from $(basename "$ADAPTER")"
    fuse "$BASE" "$ADAPTER" "$SDK/models/fused/$NAME" && build "$NAME" "$SDK/models/fused/$NAME" "$TEMPLATE" "$3" \
        || { step "FAILED fuse/build $NAME"; return 1; }
    rm -rf "$SDK/models/fused/$NAME"
    check "$BASE" "$ADAPTER" "$NAME-$3" "$TEMPLATE" || return 1
    step "$NAME-$3: capture"
    capture "$NAME-$3"
    if [ ! -f "$PACK_DIR/pack.json" ]; then
        step "$PACK@$5: pack"
        "$DEV" pack --model "$SDK/models/$NAME-$3.gguf" --name "$PACK" --version "$5" --data-dir "$PACK_DATA" \
            --out "$PACK_DIR" --template "$TEMPLATE" --base "$BASE" --license Apache-2.0 --quantization "$3" \
            > "$OUT/pack-$PACK-$5.log" 2>&1 || { step "FAILED pack $PACK@$5"; rm -rf "$PACK_DIR"; return 1; }
    fi
    { echo "## $PACK $5 ($NAME-$3, $(basename "$ADAPTER")), rebuilt $(date '+%F')"
      jq -c '.evaluation // .eval // {} | {macro_accuracy, macro_brier}' "$PACK_DIR/pack.json"; } | tee -a "$RESULTS"
    step "$PACK: typed-decisions"
    typed_decisions "$PACK" "$5"
    if [ "$SMOKE" = 1 ]; then
        rm -rf "$CKPT" "$CKPT".* "$SDK/models/$NAME"* "$SDK/models/packs/$PACK" "$OUT/heads/$PACK" "$OUT"/*"$NAME"* \
            "$OUT"/*"$PACK"*
    fi
}

step "rebuild.sh started (SMOKE=$SMOKE)"
# mlx_lm drops Qwen3.5's multi-token-prediction block when it fuses, but the config still declares it.
one Qwen/Qwen3.5-2B Qwen3.5-2B Q4_K_M lars-2b-dev 0.0.1 --no-mtp || step "lars-2b-dev not rebuilt"
one Qwen/Qwen3-8B Qwen3-8B Q3_K_M lars-8b-dev 0.0.2 || step "lars-8b-dev not rebuilt"
[ "$SMOKE" = 1 ] && rm -rf "$OUT"/smoke-* "$RESULTS"
step "rebuild.sh finished"
