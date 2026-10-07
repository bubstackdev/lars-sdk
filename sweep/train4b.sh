#!/bin/sh
# The 4B trained exactly like last night's 8B: BF16 Qwen3-4B-Instruct-2507, corpus E (soft) plus
# 1,000 typed-decisions train cases, the corpus-E recipe, checkpoints selected on jev-bench
# validation plus 200 typed-decisions cases. Against the 8B it isolates size. Against the
# current 4B pack it shows what the data is worth.
#
# Unattended under launchd (~/Library/LaunchAgents/dev.lars-sdk.train4b.plist). It waits for
# sweep/overnight.sh to finish, is resumable (an interrupted training starts over), and unloads
# itself. Log: sweep/out/train4b.log; scores in sweep/out/train4b-results.txt.
#
# Train, then imatrix Q4_K_M, jev-bench capture and report, pack lars-4b-etd-dev, and
# typed-decisions through it (scores and the head simulation). SMOKE=1: the 0.6B, 5 steps,
# tiny data, without waiting.
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
RESULTS=$OUT/train4b-results.txt
export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:$HOME/.cargo/bin"
export HF_HUB_OFFLINE=1 LARS_MLX_CACHE_GB=16 TMPDIR=${TMPDIR:-/tmp}
HOLDOUT=jev-bench/civil_comments,jev-bench/fever_evidence,jev-bench/helpsteer2_helpfulness,tasksource-jev/babi_nli/three-arg-relations
SKIP=jev-bench/sst5,jev-bench/yelp5,jev-bench/stsb
RECORDS="data/train-e-soft/open-jev.train.jsonl data/train-e-soft/jev-bench.train.jsonl data/train-e-soft/tasksource-jev.train.jsonl $TD/records/typed-decisions.train.jsonl"
SELECT="data/train-soft/jev-bench.validation.jsonl $TD/records/typed-decisions.select.jsonl"
SMOKE=${SMOKE:-0}
. "$SDK/sweep/lora_lib.sh"

[ "$(git -C "$LARS" rev-parse --abbrev-ref HEAD)" = codex-wip-2026-10-02 ] || { step "lars is not on codex-wip-2026-10-02"; exit 1; }

if [ "$SMOKE" = 1 ]; then
    TRAIN_ARGS="--limit 300 --max-steps 5 --eval-every 5 --select-per-source 3 --select-unseen-per-source 3 --holdout-fraction 0.1"
    LIMIT_ROWS=40
    mkdir -p "$OUT/smoke-data"
    for f in boolq banking77 sms_spam; do
        for split in validation test; do head -15 "$DATA/$f.$split.jsonl" > "$OUT/smoke-data/$f.$split.jsonl"; done
    done
    DATA=$OUT/smoke-data
    BASE=Qwen/Qwen3-0.6B NAME=Qwen3-0.6B-lora-etd-smoke PACK=lars-smoke-dev TEMPLATE=$SDK/sweep/templates/Qwen3-0.6B.json
    CKPT=$OUT/smoke-ckpt
else
    # Without gradient checkpointing: the same gradients, 23% faster, 49 GB instead of 43 GB peak.
    TRAIN_ARGS="--limit 50000 --holdout-sources $HOLDOUT --no-grad-checkpoint"
    LIMIT_ROWS=""
    BASE=Qwen/Qwen3-4B-Instruct-2507 NAME=Qwen3-4B-Instruct-2507-lora-etd-s0 PACK=lars-4b-etd-dev
    TEMPLATE=$SDK/sweep/templates/Qwen3-4B-Instruct-2507.json CKPT=$LARS/checkpoints/lora-4b-sdk-etd-s0
fi

step "train4b.sh started (SMOKE=$SMOKE)"
if [ "$SMOKE" != 1 ]; then
    while pgrep -f 'sweep/overnight.sh' > /dev/null; do sleep 60; done
    step "overnight.sh is done; GPU free"
fi

step "$NAME: train"
if ! train "$BASE" "$CKPT" "$OUT/train-$NAME.log"; then
    step "FAILED train $NAME (see $OUT/train-$NAME.log)"
    exit 1
fi
grep -q '"improved": false' "$CKPT/adapter_config.json" && step "NOTE $NAME: no checkpoint beat the untrained model"
step "$NAME: fuse, convert, quantize"
fuse "$BASE" "$CKPT" "$SDK/models/fused/$NAME" && build "$NAME" "$SDK/models/fused/$NAME" "$TEMPLATE" Q4_K_M \
    || { step "FAILED fuse/build $NAME"; exit 1; }
check "$BASE" "$CKPT" "$NAME-Q4_K_M" "$TEMPLATE" || exit 1
step "$NAME-Q4_K_M: capture"
capture "$NAME-Q4_K_M"
step "$NAME: pack $PACK"
pack_dir=$SDK/models/packs/$PACK/0.0.1
if [ -f "$pack_dir/pack.json" ] || "$DEV" pack --model "$SDK/models/$NAME-Q4_K_M.gguf" --name "$PACK" --version 0.0.1 \
        --data-dir "$DATA" --out "$pack_dir" --template "$TEMPLATE" --base "$BASE" --license Apache-2.0 \
        --quantization Q4_K_M > "$OUT/pack-$PACK.log" 2>&1; then
    step "$PACK: typed-decisions"
    typed_decisions "$PACK"
else
    step "FAILED pack $PACK"
fi
rm -rf "$SDK/models/fused/$NAME"
if [ "$SMOKE" = 1 ]; then
    rm -rf "$SDK/models/$NAME"-* "$SDK/models/$NAME.imatrix.gguf" "$SDK/models/packs/$PACK" "$OUT/heads/$PACK" "$CKPT"* "$OUT/smoke-data" "$OUT"/smoke-*.jsonl
    step "smoke done"
    exit 0
fi
step "train4b.sh finished"
launchctl bootout "gui/$(id -u)/dev.lars-sdk.train4b" 2>/dev/null
