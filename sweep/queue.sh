#!/bin/sh
# The Phase 2b capture queue for one compute path, in priority order. Resumable: finished runs
# and files are skipped. Waits for each model file (sweep/prepare.sh may still be writing them).
#
#     sweep/queue.sh metal      # every model and quantization, layouts v1 and v2, then latency
#     sweep/queue.sh cpu        # Q4_K_M then Q8_0, layout v1, for the calibrator-transfer check
set -u
DEVICE=$1
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"

model_file() { echo "$ROOT/models/$1.gguf"; }
wait_for() { until [ -s "$(model_file "$1")" ] && ! pgrep -f "llama-quantize.*$1" > /dev/null; do sleep 30; done; }

capture() {
    wait_for "$1"
    sweep/run.sh "$1" "$(model_file "$1")" "$DEVICE" "$2" || echo "FAILED $1 $DEVICE $2"
    sweep/report.sh
}

bench() {
    out=sweep/out/bench-$1.jsonl
    [ -s "$out" ] && return
    wait_for "$1"
    TMPDIR=${TMPDIR:-/tmp} bench/run.sh "$(model_file "$1")" target/release/lars-dev > "$out.part" && mv "$out.part" "$out"
    echo "bench $1 done"
}

MODELS="Qwen3-0.6B Qwen3-1.7B Qwen3-4B-Instruct-2507"
if [ "$DEVICE" = metal ]; then
    # The dev model (lars/models), as the baseline against the prototype's numbers.
    sweep/run.sh Qwen3-4B-Instruct-2507-Q4_K_M-dev "$HOME/Code/personal/lars/models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf" \
        metal v1 || echo "FAILED dev"
    sweep/report.sh
    for m in $MODELS; do
        capture "$m-Q4_K_M" v1
        capture "$m-Q4_K_M" v2
        capture "$m-Q8_0" v1
        bench "$m-Q4_K_M"
    done
    for m in $MODELS; do capture "$m-Q8_0" v2; done
else
    for m in $MODELS; do capture "$m-Q4_K_M" v1; done
    for m in $MODELS; do capture "$m-Q8_0" v1; done
fi
echo "queue $DEVICE done"
