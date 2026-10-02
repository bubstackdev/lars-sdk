#!/bin/sh
# LoRA adapters for the SDK's model sizes, measured through the SDK's own runtime. Unattended:
# started by launchd (~/Library/LaunchAgents/dev.lars-sdk.train.plist), resumable, and unloads
# itself when done.
#
# 1. The prototype's 4B corpus-E adapter (trained on the 4-bit MLX base): fused into a
#    dequantized base, converted to GGUF, quantized, captured. The fused-GGUF agreement check.
# 2. New LoRAs on Qwen3-1.7B, then Qwen3-0.6B: BF16 base, the repaired corpus E
#    (data/train-e-soft), the 4B corpus-E recipe (rank 8, scale 20, lr 2e-5, every linear
#    layer, one epoch, selection on seen and unseen validation sources equally). Trained with
#    lars on branch codex-wip-2026-10-02, then fused, converted, quantized, captured.
#
# Captures are Metal, layout v1, Q8_0 and Q4_K_M, into sweep/out like every other run, with a
# report after each. SMOKE=1 runs a tiny version end to end (a few steps on the 0.6B) instead.
set -u
SDK=$(cd "$(dirname "$0")/.." && pwd)
LARS=$HOME/Code/personal/lars
PY=$LARS/.venv/bin/python
CONVERT=$SDK/sweep/.venv-convert/bin/python
LLAMA=$SDK/crates/lars-llama-sys/llama.cpp
QUANTIZE=$(find "$SDK/target/llama-tools" -name llama-quantize -type f -perm -u+x | head -1)
export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:$HOME/.cargo/bin"
export HF_HUB_OFFLINE=1 LARS_MLX_CACHE_GB=16
HOLDOUT=jev-bench/civil_comments,jev-bench/fever_evidence,jev-bench/helpsteer2_helpfulness,tasksource-jev/babi_nli/three-arg-relations
SKIP=jev-bench/sst5,jev-bench/yelp5,jev-bench/stsb
RECORDS="data/train-e-soft/open-jev.train.jsonl data/train-e-soft/jev-bench.train.jsonl data/train-e-soft/tasksource-jev.train.jsonl"
SMOKE=${SMOKE:-0}
step() { echo "=== $(date '+%F %T') $*"; }

[ "$(git -C "$LARS" rev-parse --abbrev-ref HEAD)" = codex-wip-2026-10-02 ] || { step "lars is not on codex-wip-2026-10-02"; exit 1; }

# A fused Hugging Face directory -> models/<name>-{BF16,Q8_0,Q4_K_M}.gguf
gguf() {
    name=$1 dir=$2 out=$SDK/models/$1
    if [ ! -s "$out-BF16.gguf" ]; then
        "$CONVERT" "$LLAMA/convert_hf_to_gguf.py" "$dir" --outtype bf16 --outfile "$out-BF16.gguf.part" \
            > "$SDK/sweep/out/convert-$name.log" 2>&1 && mv "$out-BF16.gguf.part" "$out-BF16.gguf" || return 1
    fi
    for q in Q8_0 Q4_K_M; do
        [ -s "$out-$q.gguf" ] && continue
        "$QUANTIZE" "$out-BF16.gguf" "$out-$q.gguf.part" $q > /dev/null 2>&1 && mv "$out-$q.gguf.part" "$out-$q.gguf" || return 1
    done
}

capture() {
    for q in Q8_0 Q4_K_M; do
        "$SDK/sweep/run.sh" "$1-$q" "$SDK/models/$1-$q.gguf" metal v1 || step "FAILED capture $1-$q"
        "$SDK/sweep/report.sh"
    done
}

# A Hub model's cached snapshot directory (complete for offline use).
snapshot() {
    "$PY" -c "from huggingface_hub import snapshot_download as s; print(s('$1', local_files_only=True))"
}

fuse() {  # base, adapter, out [--dequantize]
    [ -f "$3/config.json" ] && return 0
    (cd "$LARS" && "$PY" -m mlx_lm fuse --model "$1" --adapter-path "$2" --save-path "$3.part" $4) \
        > "$SDK/sweep/out/fuse-$(basename "$3").log" 2>&1 || return 1
    # A LoRA leaves the tokenizer alone. Use the base's own files: mlx_lm rewrites tokenizer_config
    # with a newer transformers than llama.cpp's converter pins.
    base=$(snapshot "$1") || return 1
    for f in tokenizer.json tokenizer_config.json vocab.json merges.txt special_tokens_map.json; do
        [ -f "$base/$f" ] && cp "$base/$f" "$3.part/$f"
    done
    mv "$3.part" "$3"
}

mkdir -p "$SDK/models/fused"
if [ "$SMOKE" = 1 ]; then
    step "smoke: 5 steps on the 0.6B, fuse, convert, answer one request"
    ckpt=$SDK/sweep/out/smoke-lora
    (cd "$LARS" && "$PY" -m lars.train.lora --model Qwen/Qwen3-0.6B --records $RECORDS --limit 300 --seed 0 \
        --holdout-fraction 0.1 --max-steps 5 --eval-every 5 --select-records data/train-soft/jev-bench.validation.jsonl \
        --select-skip $SKIP --select-per-source 5 --select-unseen-per-source 5 --out "$ckpt") || exit 1
    fused=$SDK/sweep/out/smoke-fused
    rm -rf "$fused"
    fuse Qwen/Qwen3-0.6B "$ckpt" "$fused" "" || { step "FAILED fuse"; exit 1; }
    "$CONVERT" "$LLAMA/convert_hf_to_gguf.py" "$fused" --outtype bf16 --outfile "$SDK/sweep/out/smoke.gguf" \
        > "$SDK/sweep/out/convert-smoke.log" 2>&1 || { step "FAILED convert"; exit 1; }
    "$SDK/target/release/lars-dev" ask --model "$SDK/sweep/out/smoke.gguf" --template "$SDK/sweep/templates/Qwen3-0.6B.json" \
        "$SDK/bench/molar-triage.json" | head -12
    step "smoke done"
    exit 0
fi

step "train.sh started"
name=Qwen3-4B-Instruct-2507-lora-e-s0
step "$name: fuse the prototype's adapter into the dequantized 4-bit base, convert, capture"
fuse mlx-community/Qwen3-4B-Instruct-2507-4bit "$LARS/checkpoints/lora-4b-e-s0" "$SDK/models/fused/$name" --dequantize \
    && gguf "$name" "$SDK/models/fused/$name" && capture "$name" || step "FAILED $name"

for model in Qwen3-1.7B Qwen3-0.6B; do
    name=$model-lora-e-s0
    ckpt=$LARS/checkpoints/lora-$(echo "$model" | tr '[:upper:]' '[:lower:]')-sdk-e-s0
    if [ ! -f "$ckpt/adapters.safetensors" ]; then
        step "$name: train"
        (cd "$LARS" && "$PY" -m lars.train.lora --model "Qwen/$model" --records $RECORDS --limit 50000 --seed 0 \
            --holdout-sources $HOLDOUT --select-records data/train-soft/jev-bench.validation.jsonl --select-skip $SKIP \
            --out "$ckpt") > "$SDK/sweep/out/train-$name.log" 2>&1 || { step "FAILED train $name"; continue; }
    fi
    grep -q '"improved": false' "$ckpt/adapter_config.json" && step "NOTE $name: no checkpoint beat the untrained model"
    step "$name: fuse, convert, capture"
    fuse "Qwen/$model" "$ckpt" "$SDK/models/fused/$name" "" && gguf "$name" "$SDK/models/fused/$name" && capture "$name" \
        || step "FAILED $name"
done
step "train.sh finished"
launchctl bootout "gui/$(id -u)/dev.lars-sdk.train" 2>/dev/null
