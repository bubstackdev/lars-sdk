#!/bin/sh
# LoRA on Qwen3.5-4B, trained like the 2B (corpus E soft plus typed-decisions train), then imatrix
# Q4_K_M, the build check, a jev-bench capture, pack lars-4b35-dev and typed-decisions through it
# (sweep/out/train35_4b-results.txt). The question: does Qwen3.5-4B match the 8B (0.748) at about
# 2.5 GB?
#
# Unattended under launchd (~/Library/LaunchAgents/dev.lars-sdk.train35-4b.plist). It waits until
# START_HOUR (18:00) and for any other GPU job, is resumable (an interrupted training starts
# over), and unloads itself when done or failed. Log: sweep/out/train35_4b.log. SMOKE=1: 5 steps,
# tiny data, without waiting, then cleans up.
set -u
SDK=$(cd "$(dirname "$0")/.." && pwd)
LARS=$HOME/Code/personal/lars
PY=$LARS/.venv/bin/python
CONVERT=$SDK/sweep/.venv-convert/bin/python
FIXPY=$SDK/fixtures/.venv/bin/python
LLAMA=$SDK/crates/lars-llama-sys/llama.cpp
TOOLS=$SDK/target/llama-tools/bin
DEV=$SDK/target/release/lars-dev
OUT=$SDK/sweep/out
TD=$OUT/typed-decisions
DATA=$LARS/evals/data
RESULTS=$OUT/train35_4b-results.txt
export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:$HOME/.cargo/bin"
export HF_HUB_OFFLINE=1 LARS_MLX_CACHE_GB=16 TMPDIR=${TMPDIR:-/tmp}
HOLDOUT=jev-bench/civil_comments,jev-bench/fever_evidence,jev-bench/helpsteer2_helpfulness,tasksource-jev/babi_nli/three-arg-relations
SKIP=jev-bench/sst5,jev-bench/yelp5,jev-bench/stsb
RECORDS="data/train-e-soft/open-jev.train.jsonl data/train-e-soft/jev-bench.train.jsonl data/train-e-soft/tasksource-jev.train.jsonl $TD/records/typed-decisions.train.jsonl"
SELECT="data/train-soft/jev-bench.validation.jsonl $TD/records/typed-decisions.select.jsonl"
SMOKE=${SMOKE:-0}
START_HOUR=${START_HOUR:-18}
BASE=Qwen/Qwen3.5-4B TEMPLATE=$SDK/sweep/templates/Qwen3.5-4B.json
# mlx_lm drops the multi-token-prediction block when it fuses, but the config still declares it.
CONVERT_ARGS=--no-mtp
. "$SDK/sweep/lora_lib.sh"

finish() {  # status
    [ "$SMOKE" = 1 ] || launchctl bootout "gui/$(id -u)/dev.lars-sdk.train35-4b" 2>/dev/null
    exit "$1"
}

[ "$(git -C "$LARS" rev-parse --abbrev-ref HEAD)" = codex-wip-2026-10-02 ] || { step "lars is not on codex-wip-2026-10-02"; finish 1; }

if [ "$SMOKE" = 1 ]; then
    TRAIN_ARGS="--limit 300 --max-steps 5 --eval-every 5 --select-per-source 3 --select-unseen-per-source 3 --holdout-fraction 0.1"
    LIMIT_ROWS=40
    mkdir -p "$OUT/smoke-data"
    for f in boolq banking77 sms_spam; do
        for split in validation test; do head -15 "$DATA/$f.$split.jsonl" > "$OUT/smoke-data/$f.$split.jsonl"; done
    done
    PACK_DATA=$OUT/smoke-data
    NAME=Qwen3.5-4B-lora-etd-smoke PACK=lars-smoke-dev CKPT=$OUT/smoke-ckpt
else
    TRAIN_ARGS="--limit 50000 --holdout-sources $HOLDOUT"
    LIMIT_ROWS="" PACK_DATA=$DATA
    NAME=Qwen3.5-4B-lora-etd-s0 PACK=lars-4b35-dev CKPT=$LARS/checkpoints/lora-qwen3.5-4b-sdk-etd-s0
fi

step "train35_4b.sh started (SMOKE=$SMOKE)"
if [ "$SMOKE" != 1 ]; then
    while [ "$(date +%H)" -lt "$START_HOUR" ]; do sleep 60; done
    while pgrep -f 'sweep/(train4b|train35|overnight|finish2b|promote)\.sh' > /dev/null; do sleep 60; done
    step "GPU free"
fi
step "$NAME: train"
train "$BASE" "$CKPT" "$OUT/train-$NAME.log" || { step "FAILED train $NAME (see $OUT/train-$NAME.log)"; finish 1; }
grep -q '"improved": false' "$CKPT/adapter_config.json" && step "NOTE $NAME: no checkpoint beat the untrained model"
step "$NAME: fuse, convert, quantize"
fuse "$BASE" "$CKPT" "$SDK/models/fused/$NAME" && build "$NAME" "$SDK/models/fused/$NAME" "$TEMPLATE" Q4_K_M \
    || { step "FAILED fuse/build $NAME"; finish 1; }
rm -rf "$SDK/models/fused/$NAME"
check "$BASE" "$CKPT" "$NAME-Q4_K_M" "$TEMPLATE" || finish 1
step "$NAME-Q4_K_M: capture"
capture "$NAME-Q4_K_M"
step "$NAME: pack $PACK"
pack_dir=$SDK/models/packs/$PACK/0.0.1
if [ -f "$pack_dir/pack.json" ] || "$DEV" pack --model "$SDK/models/$NAME-Q4_K_M.gguf" --name "$PACK" --version 0.0.1 \
        --data-dir "$PACK_DATA" --out "$pack_dir" --template "$TEMPLATE" --base "$BASE" --license Apache-2.0 \
        --quantization Q4_K_M > "$OUT/pack-$PACK.log" 2>&1; then
    step "$PACK: typed-decisions"
    typed_decisions "$PACK" || step "FAILED typed-decisions $PACK"
else
    step "FAILED pack $PACK"
fi
if [ "$SMOKE" = 1 ]; then
    rm -rf "$SDK/models/$NAME"-* "$SDK/models/$NAME.imatrix.gguf" "$SDK/models/packs/$PACK" "$OUT/heads/$PACK" \
        "$CKPT"* "$OUT/smoke-data" "$OUT"/smoke-*.jsonl "$OUT/$NAME-Q4_K_M.metal.v1"
    step "smoke done"
    exit 0
fi
step "train35_4b.sh finished"
finish 0
