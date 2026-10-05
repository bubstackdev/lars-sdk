#!/bin/sh
# The quiet-machine run: latency of the candidate packs, then importance-matrix (imatrix)
# Q4_K_M versions of the LoRA models, captured and analyzed like every other run.
#
# Started by launchd (~/Library/LaunchAgents/dev.lars-sdk.night.plist) under caffeinate, which
# keeps the Mac awake until then. It waits for 01:30 (or starts at once between 01:00 and
# 05:00, after a reboot), is resumable, and unloads itself when done. Results:
#   sweep/out/latency.md        latency tables (load average recorded per row)
#   sweep/out/summary.md        the imatrix rows, next to every other run
#   sweep/out/night.log         progress
set -u
SDK=$(cd "$(dirname "$0")/.." && pwd)
cd "$SDK"
export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:$HOME/.cargo/bin"
export TMPDIR=${TMPDIR:-/tmp} HF_HUB_OFFLINE=1
TOOLS=$SDK/target/llama-tools/bin
OUT=$SDK/sweep/out
step() { echo "=== $(date '+%F %T') $*"; }

hour=$(date +%H)
if [ "$hour" -lt 1 ] || [ "$hour" -ge 5 ]; then
    target=$(date -j -v1H -v30M -v0S +%s)
    now=$(date +%s)
    [ "$target" -le "$now" ] && target=$((target + 86400))
    step "waiting until $(date -r "$target" '+%a %H:%M')"
    sleep $((target - now))
fi
step "night.sh started; load average $(sysctl -n vm.loadavg)"

template() {
    case "$1" in
        Qwen3-0.6B*) echo "$SDK/sweep/templates/Qwen3-0.6B.json" ;;
        Qwen3-1.7B*) echo "$SDK/sweep/templates/Qwen3-1.7B.json" ;;
        *) echo chatml ;;
    esac
}

# 1. Latency, first, while nothing else runs.
for name in Qwen3-0.6B-lora-e-s0-Q4_K_M Qwen3-0.6B-lora-e-s0-Q8_0 Qwen3-1.7B-lora-e-s0-Q4_K_M \
            Qwen3-1.7B-lora-e-s0-Q8_0 Qwen3-4B-Instruct-2507-lora-e-s0-Q4_K_M; do
    out=$OUT/bench-night-$name.jsonl
    [ -s "$out" ] && continue
    step "latency $name"
    bench/run.sh "$SDK/models/$name.gguf" target/release/lars-dev "$(template "$name")" > "$out.part" \
        && mv "$out.part" "$out" || step "FAILED latency $name"
done
python3 sweep/latency.py "$OUT"/bench-night-*.jsonl > "$OUT/latency.md"

# 2. Importance matrices from training-split prompts, imatrix Q4_K_M, capture, report.
for base in Qwen3-0.6B Qwen3-1.7B Qwen3-4B-Instruct-2507; do
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
step "night.sh finished"
launchctl bootout "gui/$(id -u)/dev.lars-sdk.night" 2>/dev/null
