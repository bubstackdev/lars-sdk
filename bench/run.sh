#!/bin/sh
# Latency of the bench requests on Metal and CPU, one JSON line per configuration.
#
#     bench/run.sh <model.gguf> [lars-dev binary] > results.jsonl
#
# Run on a quiet machine: latency taken while anything else uses the GPU or CPU is not a number.
set -eu
MODEL=$1
BIN=${2:-target/release/lars-dev}
HERE=$(cd "$(dirname "$0")" && pwd)

run() {
    label=$1 request=$2 repeat=$3
    shift 3
    # /usr/bin/time -l (macOS) reports peak resident memory on stderr.
    peak=$( { /usr/bin/time -l "$BIN" bench --model "$MODEL" "$HERE/$request.json" --repeat "$repeat" "$@" \
               > "$TMPDIR/lars-bench.json"; } 2>&1 | awk '/maximum resident set size/ {print $1}')
    python3 -c "import json,sys; r=json.load(open('$TMPDIR/lars-bench.json')); r.update(config='$label', request='$request', peak_rss_mb=round(int('${peak:-0}')/2**20)); print(json.dumps(r))"
}

for request in one-noul molar-triage twenty-questions; do
    run "metal" $request 20
    run "metal, no prefix reuse" $request 20 --no-reuse-prefix
    run "metal, flash on" $request 20 --flash on --no-reuse-prefix
    run "metal, batched" $request 20 --batch-suffixes
    run "cpu 12 threads" $request 5 --cpu --threads 12
    run "cpu 12 threads, no prefix reuse" $request 5 --cpu --threads 12 --no-reuse-prefix
done
