#!/bin/sh
# Latency of the bench requests on Metal and CPU, one JSON line per configuration.
#
#     bench/run.sh <model.gguf> [lars-dev binary] [template] > results.jsonl
#
# "new state" rows give each call a different state (a new user every time: only the constant
# head is reused); the others repeat one request (the same user asked again). Defaults are the
# backend's: suffixes batched, prefix reuse on, flash attention off, CPU weights repacked.
# Run on a quiet machine: latency taken while anything else uses the GPU or CPU is not a number.
set -eu
MODEL=$1
BIN=${2:-target/release/lars-dev}
TEMPLATE=${3:-chatml}
HERE=$(cd "$(dirname "$0")" && pwd)
TMPDIR=${TMPDIR:-/tmp}

run() {
    label=$1 request=$2 repeat=$3
    shift 3
    # /usr/bin/time -l (macOS) reports peak resident memory on stderr.
    peak=$( { /usr/bin/time -l "$BIN" bench --model "$MODEL" --template "$TEMPLATE" "$HERE/$request.json" \
               --repeat "$repeat" "$@" > "$TMPDIR/lars-bench.json"; } 2>&1 | awk '/maximum resident set size/ {print $1}')
    python3 -c "import json,sys; r=json.load(open('$TMPDIR/lars-bench.json')); r.update(config='$label', request='$request', model='$(basename "$MODEL" .gguf)', peak_rss_mb=round(int('${peak:-0}')/2**20), load_avg=open('/dev/stdin').read().split()[0]); print(json.dumps(r))" <<LOAD
$(sysctl -n vm.loadavg | tr -d '{}')
LOAD
}

for request in one-noul molar-triage twenty-questions; do
    run "metal, new state" $request 20 --vary-state
    run "metal, same state" $request 20
    run "metal, new state, sequential" $request 20 --vary-state --no-batch-suffixes
    run "cpu 12 threads, new state" $request 5 --cpu --threads 12 --vary-state
    run "cpu 12 threads, same state" $request 5 --cpu --threads 12
    run "cpu 6 threads, new state" $request 5 --cpu --threads 6 --vary-state
    run "cpu 12 threads, new state, no repack" $request 5 --cpu --threads 12 --vary-state --no-repack
done
