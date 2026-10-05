#!/bin/sh
# Overnight 2026-10-05: a bigger model, trained like the packs plus typed-decisions train.
# Unattended under launchd (~/Library/LaunchAgents/dev.lars-sdk.overnight.plist), resumable, and
# unloads itself when done. Log: sweep/out/overnight.log; each step's own log in sweep/out/.
#
# 1. Wait (up to 90 minutes) for the afternoon's GPU jobs: qwen35.sh and heads_capture.sh.
# 2. LoRA on Qwen3-8B (dense; the corpus-E recipe: rank 8, scale 20, lr 2e-5, every linear layer,
#    one epoch), on corpus E plus 1,000 typed-decisions train cases. Checkpoints are selected on
#    jev-bench validation plus the other 200 typed-decisions train cases. The test split is
#    never used. About 9 hours. Training cannot resume: an interrupted run starts over.
# 3. Fuse, convert, importance matrix, Q4_K_M and Q3_K_M. jev-bench capture and report for each.
# 4. Pack lars-8b-dev@0.0.1 from Q4_K_M (pooled calibrator). typed-decisions test and train
#    through it with hidden states. Score zero-shot (now "fitted": it trained on train) and
#    simulate per-predicate heads: sweep/out/overnight-results.txt.
# 5. The untrained Qwen3-8B at Q4_K_M, captured as the zero-shot reference.
# 6. LoRA on Qwen3.5-2B with the same data (about 10 hours; needs lars f42781a's chunked gated
#    delta rule), then fuse, convert and capture. It may run into the morning; stopping it
#    (launchctl bootout gui/$(id -u)/dev.lars-sdk.overnight) loses nothing that came before.
#
# SMOKE=1 runs the whole chain on Qwen3-0.6B with 5 training steps and 40-row captures, into
# names ending in -smoke, so the plumbing is checked before the night.
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
export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:$HOME/.cargo/bin"
export HF_HUB_OFFLINE=1 LARS_MLX_CACHE_GB=16 TMPDIR=${TMPDIR:-/tmp}
HOLDOUT=jev-bench/civil_comments,jev-bench/fever_evidence,jev-bench/helpsteer2_helpfulness,tasksource-jev/babi_nli/three-arg-relations
SKIP=jev-bench/sst5,jev-bench/yelp5,jev-bench/stsb
RECORDS="data/train-e-soft/open-jev.train.jsonl data/train-e-soft/jev-bench.train.jsonl data/train-e-soft/tasksource-jev.train.jsonl $TD/records/typed-decisions.train.jsonl"
SELECT="data/train-soft/jev-bench.validation.jsonl $TD/records/typed-decisions.select.jsonl"
SMOKE=${SMOKE:-0}
step() { echo "=== $(date '+%F %T') $*"; }

[ "$(git -C "$LARS" rev-parse --abbrev-ref HEAD)" = codex-wip-2026-10-02 ] || { step "lars is not on codex-wip-2026-10-02"; exit 1; }

if [ "$SMOKE" = 1 ]; then
    TRAIN_ARGS="--limit 300 --max-steps 5 --eval-every 5 --select-per-source 3 --select-unseen-per-source 3 --holdout-fraction 0.1"
    LIMIT_ROWS=40
    # A pack is calibrated on every validation file of its data directory: a tiny one for the smoke.
    mkdir -p "$OUT/smoke-data"
    for f in boolq banking77 sms_spam; do
        for split in validation test; do head -15 "$DATA/$f.$split.jsonl" > "$OUT/smoke-data/$f.$split.jsonl"; done
    done
    DATA=$OUT/smoke-data
else
    TRAIN_ARGS="--limit 50000 --holdout-sources $HOLDOUT"
    LIMIT_ROWS=""
fi

# Train once; an interrupted run leaves no .done and starts over.
train() {  # hub-model checkpoint log
    [ -f "$2/.done" ] && return 0
    rm -rf "$2" "$2.steps" "$2.last" "$2.history.json" "$2.baseline.json"
    (cd "$LARS" && "$PY" -m lars.train.lora --model "$1" --records $RECORDS --seed 0 \
        --select-records $SELECT --select-skip $SKIP --out "$2" $TRAIN_ARGS) > "$3" 2>&1 && touch "$2/.done"
}

snapshot() {
    "$PY" -c "from huggingface_hub import snapshot_download as s; print(s('$1', local_files_only=True))"
}

fuse() {  # hub-model adapter out
    [ -f "$3/config.json" ] && return 0
    rm -rf "$3.part"
    (cd "$LARS" && "$PY" -m mlx_lm fuse --model "$1" --adapter-path "$2" --save-path "$3.part") \
        > "$OUT/fuse-$(basename "$3").log" 2>&1 || return 1
    # The base's own tokenizer files: mlx_lm rewrites tokenizer_config with a newer transformers.
    base=$(snapshot "$1") || return 1
    for f in tokenizer.json tokenizer_config.json vocab.json merges.txt special_tokens_map.json; do
        [ -f "$base/$f" ] && cp "$base/$f" "$3.part/$f"
    done
    mv "$3.part" "$3"
}

# A Hugging Face directory -> models/<name>-<quant>.gguf for each quant, with an importance matrix
# computed on training-split prompts rendered with the base's template.
build() {  # name hf-dir template quants...
    b_name=$1 b_dir=$2 b_template=$3
    shift 3
    b_bf16=$SDK/models/$b_name-BF16.gguf
    b_text=$SDK/models/imatrix-$(basename "$b_template" .json).txt
    b_matrix=$SDK/models/$b_name.imatrix.gguf
    [ -s "$b_bf16" ] || { "$CONVERT" "$LLAMA/convert_hf_to_gguf.py" "$b_dir" --outtype bf16 --outfile "$b_bf16.part" \
        > "$OUT/convert-$b_name.log" 2>&1 && mv "$b_bf16.part" "$b_bf16"; } || return 1
    [ -s "$b_text" ] || "$PY" "$SDK/sweep/imatrix_text.py" --template "$b_template" --records 1500 --out "$b_text" \
        > "$OUT/imatrix-text-$b_name.log" 2>&1 || return 1
    [ -s "$b_matrix" ] || "$TOOLS/llama-imatrix" -m "$b_bf16" -f "$b_text" --parse-special --chunks 400 -ngl 99 -o "$b_matrix" \
        > "$OUT/imatrix-$b_name.log" 2>&1 || return 1
    for b_q in "$@"; do
        b_out=$SDK/models/$b_name-$b_q.gguf
        [ -s "$b_out" ] || { "$TOOLS/llama-quantize" --imatrix "$b_matrix" "$b_bf16" "$b_out.part" "$b_q" \
            > "$OUT/quantize-$b_name-$b_q.log" 2>&1 && mv "$b_out.part" "$b_out"; } || return 1
        ls -la "$b_out" | awk '{printf "%s: %.2f GB\n", $9, $5 / 1e9}'
    done
    rm -f "$b_bf16"
}

capture() {  # name (models/<name>.gguf)
    if [ "$SMOKE" = 1 ]; then
        cat "$DATA/boolq.test.jsonl" > "$OUT/smoke-rows.jsonl"
        "$DEV" capture --model "$SDK/models/$1.gguf" --template "$TEMPLATE" --data "$OUT/smoke-rows.jsonl" \
            --out "$OUT/smoke-capture.jsonl" || return 1
        return 0
    fi
    "$SDK/sweep/run.sh" "$1" "$SDK/models/$1.gguf" metal v1 || step "FAILED capture $1"
    "$SDK/sweep/report.sh"
}

# typed-decisions through a pack, with hidden states, then scores and the head simulation.
typed_decisions() {  # pack-name
    t_pack=$SDK/models/packs/$1/0.0.1
    t_dir=$OUT/heads/$1
    mkdir -p "$t_dir"
    for t_split in test train; do
        t_data=$TD/typed-decisions.$t_split.jsonl
        if [ -n "$LIMIT_ROWS" ]; then
            head -"$LIMIT_ROWS" "$t_data" > "$OUT/smoke-td-$t_split.jsonl"
            t_data=$OUT/smoke-td-$t_split.jsonl
        fi
        [ -s "$t_dir/typed-decisions.$t_split.f32" ] && continue
        "$DEV" capture --model "$t_pack/model.gguf" --template "$TEMPLATE" --data "$t_data" \
            --out "$t_dir/typed-decisions.$t_split.jsonl" --hidden "$t_dir/typed-decisions.$t_split.f32.part" \
            >> "$OUT/typed-decisions-$1.log" 2>&1 && mv "$t_dir/typed-decisions.$t_split.f32.part" "$t_dir/typed-decisions.$t_split.f32" \
            || { step "FAILED typed-decisions capture $1 $t_split"; return 1; }
    done
    [ "$SMOKE" = 1 ] && return 0
    {
        echo "## $1 on typed-decisions test (pack calibrator)"
        "$FIXPY" "$SDK/sweep/typed_decisions.py" "$1=$t_dir/typed-decisions.test.jsonl:$t_pack/calibration.json" | grep -v '^    '
        echo "## $1 per-predicate heads (hard labels)"
        "$FIXPY" "$SDK/sweep/heads.py" "$t_dir" --pack "$t_pack/calibration.json" --n 25,50,100,200,300 --only td: --hard --tag .td-hard | tail -32
    } >> "$OUT/overnight-results.txt" 2>&1
}

step "overnight.sh started (SMOKE=$SMOKE)"
if [ "$SMOKE" != 1 ]; then
    waited=0
    while pgrep -f 'sweep/qwen35.sh|sweep/heads_capture.sh' > /dev/null && [ $waited -lt 5400 ]; do
        sleep 60
        waited=$((waited + 60))
    done
    step "GPU free (waited $((waited / 60)) min)"
fi

# --- the 8B (the 0.6B under SMOKE) -------------------------------------------------------------
if [ "$SMOKE" = 1 ]; then
    BASE=Qwen/Qwen3-0.6B NAME=Qwen3-0.6B-lora-etd-smoke PACK=lars-smoke-dev TEMPLATE=$SDK/sweep/templates/Qwen3-0.6B.json
    CKPT=$OUT/smoke-ckpt QUANTS=Q4_K_M
else
    BASE=Qwen/Qwen3-8B NAME=Qwen3-8B-lora-etd-s0 PACK=lars-8b-dev TEMPLATE=$SDK/sweep/templates/Qwen3-8B.json
    CKPT=$LARS/checkpoints/lora-8b-sdk-etd-s0 QUANTS="Q4_K_M Q3_K_M"
fi
step "$NAME: train"
if train "$BASE" "$CKPT" "$OUT/train-$NAME.log"; then
    grep -q '"improved": false' "$CKPT/adapter_config.json" && step "NOTE $NAME: no checkpoint beat the untrained model"
    step "$NAME: fuse, convert, quantize"
    if fuse "$BASE" "$CKPT" "$SDK/models/fused/$NAME" && build "$NAME" "$SDK/models/fused/$NAME" "$TEMPLATE" $QUANTS; then
        for q in $QUANTS; do
            step "$NAME-$q: capture"
            capture "$NAME-$q"
        done
        step "$NAME: pack $PACK"
        pack_dir=$SDK/models/packs/$PACK/0.0.1
        if [ -f "$pack_dir/pack.json" ] || { "$DEV" pack --model "$SDK/models/$NAME-Q4_K_M.gguf" --name "$PACK" --version 0.0.1 \
                --data-dir "$DATA" --out "$pack_dir" --template "$TEMPLATE" --base "$BASE" --license Apache-2.0 \
                --quantization Q4_K_M > "$OUT/pack-$PACK.log" 2>&1; }; then
            step "$PACK: typed-decisions"
            typed_decisions "$PACK"
        else
            step "FAILED pack $PACK"
        fi
        [ "$SMOKE" = 1 ] || rm -rf "$SDK/models/fused/$NAME"
    else
        step "FAILED fuse/build $NAME"
    fi
else
    step "FAILED train $NAME (see $OUT/train-$NAME.log)"
fi
if [ "$SMOKE" = 1 ]; then
    step "smoke done"
    exit 0
fi

# --- the untrained 8B, for reference -------------------------------------------------------------
step "Qwen3-8B zero-shot: convert, quantize, capture"
if build Qwen3-8B "$(snapshot Qwen/Qwen3-8B)" "$TEMPLATE" Q4_K_M; then
    capture Qwen3-8B-Q4_K_M
else
    step "FAILED build Qwen3-8B"
fi

# --- Qwen3.5-2B ----------------------------------------------------------------------------------
BASE=Qwen/Qwen3.5-2B NAME=Qwen3.5-2B-lora-etd-s0 TEMPLATE=$SDK/sweep/templates/Qwen3.5-2B.json
CKPT=$LARS/checkpoints/lora-qwen3.5-2b-sdk-etd-s0
step "$NAME: train"
if train "$BASE" "$CKPT" "$OUT/train-$NAME.log"; then
    step "$NAME: fuse, convert, quantize, capture"
    if fuse "$BASE" "$CKPT" "$SDK/models/fused/$NAME" && build "$NAME" "$SDK/models/fused/$NAME" "$TEMPLATE" Q4_K_M; then
        capture "$NAME-Q4_K_M"
        rm -rf "$SDK/models/fused/$NAME"
    else
        step "FAILED fuse/build $NAME"
    fi
else
    step "FAILED train $NAME (see $OUT/train-$NAME.log)"
fi
step "overnight.sh finished"
launchctl bootout "gui/$(id -u)/dev.lars-sdk.overnight" 2>/dev/null
