#!/bin/sh
# LoRA on Qwen3.5-2B, trained like the 8B and 4B (corpus E soft plus typed-decisions train), then
# imatrix Q4_K_M and a jev-bench capture. Training uses lars f42781a's chunked gated delta rule:
# about 4.6 s per step, 5,279 steps, about 7 hours.
#
# Unattended under launchd (~/Library/LaunchAgents/dev.lars-sdk.train35.plist). It waits for
# sweep/train4b.sh to finish, is resumable (an interrupted training starts over), and unloads
# itself. Log: sweep/out/train35.log. SMOKE=1: 5 steps and a 15-row capture, without waiting.
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
RESULTS=$OUT/train35-results.txt
export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:$HOME/.cargo/bin"
export HF_HUB_OFFLINE=1 LARS_MLX_CACHE_GB=16 TMPDIR=${TMPDIR:-/tmp}
HOLDOUT=jev-bench/civil_comments,jev-bench/fever_evidence,jev-bench/helpsteer2_helpfulness,tasksource-jev/babi_nli/three-arg-relations
SKIP=jev-bench/sst5,jev-bench/yelp5,jev-bench/stsb
RECORDS="data/train-e-soft/open-jev.train.jsonl data/train-e-soft/jev-bench.train.jsonl data/train-e-soft/tasksource-jev.train.jsonl $TD/records/typed-decisions.train.jsonl"
SELECT="data/train-soft/jev-bench.validation.jsonl $TD/records/typed-decisions.select.jsonl"
SMOKE=${SMOKE:-0}
LIMIT_ROWS=""
BASE=Qwen/Qwen3.5-2B TEMPLATE=$SDK/sweep/templates/Qwen3.5-2B.json
# mlx_lm drops the multi-token-prediction block when it fuses, but the config still declares it.
CONVERT_ARGS=--no-mtp
. "$SDK/sweep/lora_lib.sh"

[ "$(git -C "$LARS" rev-parse --abbrev-ref HEAD)" = codex-wip-2026-10-02 ] || { step "lars is not on codex-wip-2026-10-02"; exit 1; }

if [ "$SMOKE" = 1 ]; then
    TRAIN_ARGS="--limit 300 --max-steps 5 --eval-every 5 --select-per-source 3 --select-unseen-per-source 3 --holdout-fraction 0.1"
    NAME=Qwen3.5-2B-lora-etd-smoke CKPT=$OUT/smoke-ckpt
else
    TRAIN_ARGS="--limit 50000 --holdout-sources $HOLDOUT"
    NAME=Qwen3.5-2B-lora-etd-s0 CKPT=$LARS/checkpoints/lora-qwen3.5-2b-sdk-etd-s0
fi

step "train35.sh started (SMOKE=$SMOKE)"
if [ "$SMOKE" != 1 ]; then
    while pgrep -f 'sweep/train4b.sh|sweep/overnight.sh' > /dev/null; do sleep 60; done
    step "GPU free"
fi
step "$NAME: train"
train "$BASE" "$CKPT" "$OUT/train-$NAME.log" || { step "FAILED train $NAME (see $OUT/train-$NAME.log)"; exit 1; }
step "$NAME: fuse, convert, quantize"
fuse "$BASE" "$CKPT" "$SDK/models/fused/$NAME" && build "$NAME" "$SDK/models/fused/$NAME" "$TEMPLATE" Q4_K_M \
    || { step "FAILED fuse/build $NAME"; exit 1; }
rm -rf "$SDK/models/fused/$NAME"
step "$NAME-Q4_K_M: capture"
capture "$NAME-Q4_K_M"
if [ "$SMOKE" = 1 ]; then
    rm -rf "$SDK/models/$NAME"-* "$SDK/models/$NAME.imatrix.gguf" "$CKPT"* "$OUT"/smoke-*.jsonl
    step "smoke done"
    exit 0
fi
step "train35.sh finished"
launchctl bootout "gui/$(id -u)/dev.lars-sdk.train35" 2>/dev/null
