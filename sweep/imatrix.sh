#!/bin/sh
# Importance-matrix Q4_K_M versions of the LoRA models (the second half of night.sh, on demand):
# matrices from training-split prompts, quantize, capture, report. Resumable.
set -u
SDK=$(cd "$(dirname "$0")/.." && pwd)
cd "$SDK"
export HF_HUB_OFFLINE=1
TOOLS=$SDK/target/llama-tools/bin
OUT=$SDK/sweep/out
step() { echo "=== $(date '+%F %T') $*"; }
template() {
    case "$1" in
        Qwen3-0.6B*) echo "$SDK/sweep/templates/Qwen3-0.6B.json" ;;
        Qwen3-1.7B*) echo "$SDK/sweep/templates/Qwen3-1.7B.json" ;;
        *) echo chatml ;;
    esac
}
for base in ${MODELS:-Qwen3-4B-Instruct-2507 Qwen3-1.7B Qwen3-0.6B}; do
    name=$base-lora-e-s0
    text=$SDK/models/imatrix-$base.txt
    matrix=$SDK/models/$name.imatrix.gguf
    quant=$SDK/models/$name-imat-Q4_K_M.gguf
    if [ ! -s "$quant" ]; then
        step "imatrix $name"
        [ -s "$text" ] || "$HOME/Code/personal/lars/.venv/bin/python" sweep/imatrix_text.py \
            --template "$(template "$base")" --records 1500 --out "$text" || { step "FAILED text $name"; continue; }
        [ -s "$matrix" ] || "$TOOLS/llama-imatrix" -m "$SDK/models/$name-BF16.gguf" -f "$text" --parse-special \
            --chunks 400 -ngl 99 -o "$matrix" > "$OUT/imatrix-$name.log" 2>&1 || { step "FAILED imatrix $name"; continue; }
        "$TOOLS/llama-quantize" --imatrix "$matrix" "$SDK/models/$name-BF16.gguf" "$quant.part" Q4_K_M \
            > "$OUT/quantize-$name.log" 2>&1 && mv "$quant.part" "$quant" || { step "FAILED quantize $name"; continue; }
    fi
    sweep/run.sh "$name-imat-Q4_K_M" "$quant" metal v1 || step "FAILED capture $name-imat-Q4_K_M"
    sweep/report.sh
done
step "imatrix.sh finished"
