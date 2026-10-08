#!/bin/sh
# Analyze every complete sweep run (all 43 capture files) into sweep/out/summary.md and
# summary.json, with few-label curves and, where a model has both, CPU <-> Metal transfer.
# Safe to call often: each run's analysis is cached, and a concurrent call is skipped.
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
OUT=$ROOT/sweep/out
mkdir "$OUT/.report.lock" 2>/dev/null || exit 0
trap 'rmdir "$OUT/.report.lock"' EXIT
FILES=$(ls "${LARS_EVAL_DATA:-$(cd "$(dirname "$0")/.." && pwd)/data/evals}"/*.jsonl | wc -l | tr -d ' ')
runs="" transfers=""
for dir in "$OUT"/*.*.v[12]; do
    [ "$(ls "$dir"/*.jsonl 2>/dev/null | wc -l | tr -d ' ')" = "$FILES" ] || continue
    runs="$runs $dir"
    case "$dir" in
        *.cpu.*) metal=$(echo "$dir" | sed 's/\.cpu\./.metal./')
                 if [ "$(ls "$metal"/*.jsonl 2>/dev/null | wc -l | tr -d ' ')" = "$FILES" ]; then
                     transfers="$transfers --transfer $metal $dir --transfer $dir $metal"
                 fi ;;
    esac
done
[ -n "$runs" ] || exit 0
{
    echo "# Phase 2b sweep summary ($(date '+%Y-%m-%d %H:%M'))"
    echo
    # shellcheck disable=SC2086
    "$ROOT/fixtures/.venv/bin/python" "$ROOT/sweep/analyze.py" $runs --few-label $transfers --json "$OUT/summary.json.part"
} > "$OUT/summary.md.part" 2> "$OUT/report.err" && mv "$OUT/summary.md.part" "$OUT/summary.md" && mv "$OUT/summary.json.part" "$OUT/summary.json"
