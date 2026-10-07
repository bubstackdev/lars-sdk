#!/bin/sh
# The 2026-10-07 lineup: lars-2b-dev (Qwen3.5-2B + LoRA, the default), lars-8b-dev 0.0.2
# (Qwen3-8B + LoRA at Q3_K_M, the top tier) and lars-0.6b-dev 0.0.2 (from the imatrix Q4_K_M). Log: sweep/out/promote.log.
set -u
SDK=$(cd "$(dirname "$0")/.." && pwd)
DEV=$SDK/target/release/lars-dev
DATA=$HOME/Code/personal/lars/evals/data
OUT=$SDK/sweep/out
step() { echo "=== $(date '+%F %T') $*"; }
pack() {  # name version gguf template base quantization
    dir=$SDK/models/packs/$1/$2
    [ -f "$dir/pack.json" ] && { step "$1@$2 exists"; return 0; }
    step "$1@$2: pack"
    "$DEV" pack --model "$SDK/models/$3.gguf" --name "$1" --version "$2" --data-dir "$DATA" --out "$dir" \
        --template "$SDK/sweep/templates/$4.json" --base "$5" --license Apache-2.0 --quantization "$6" \
        > "$OUT/pack-$1-$2.log" 2>&1 || { step "FAILED pack $1@$2"; rm -rf "$dir"; return 1; }
    jq -c '.evaluation // .eval // {} | {macro_accuracy, macro_brier}' "$dir/pack.json" 2>/dev/null
}
pack lars-2b-dev 0.0.1 Qwen3.5-2B-lora-etd-s0-Q4_K_M Qwen3.5-2B Qwen/Qwen3.5-2B Q4_K_M
pack lars-8b-dev 0.0.2 Qwen3-8B-lora-etd-s0-Q3_K_M Qwen3-8B Qwen/Qwen3-8B Q3_K_M
pack lars-0.6b-dev 0.0.2 Qwen3-0.6B-lora-e-s0-imat-Q4_K_M Qwen3-0.6B Qwen/Qwen3-0.6B Q4_K_M
step "promote.sh finished"
