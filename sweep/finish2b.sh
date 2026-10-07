#!/bin/sh
# The rebuilt Qwen3.5-2B LoRA: pack lars-2b-etd-dev, typed-decisions through it (scores and the
# head simulation, sweep/out/finish2b-results.txt), then latency for it and the packs it could
# replace, back to back (sweep/out/bench-tiers.jsonl, sweep/out/latency-tiers.md).
set -u
SDK=$(cd "$(dirname "$0")/.." && pwd)
LARS=$HOME/Code/personal/lars
PY=$LARS/.venv/bin/python
FIXPY=$SDK/fixtures/.venv/bin/python
DEV=$SDK/target/release/lars-dev
OUT=$SDK/sweep/out
TD=$OUT/typed-decisions
DATA=$LARS/evals/data
RESULTS=$OUT/finish2b-results.txt
SMOKE=0 LIMIT_ROWS=""
export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:$HOME/.cargo/bin" HF_HUB_OFFLINE=1
. "$SDK/sweep/lora_lib.sh"
NAME=Qwen3.5-2B-lora-etd-s0 PACK=lars-2b-etd-dev BASE=Qwen/Qwen3.5-2B TEMPLATE=$SDK/sweep/templates/Qwen3.5-2B.json
pack_dir=$SDK/models/packs/$PACK/0.0.1
step "$PACK: pack"
[ -f "$pack_dir/pack.json" ] || "$DEV" pack --model "$SDK/models/$NAME-Q4_K_M.gguf" --name "$PACK" --version 0.0.1 \
    --data-dir "$DATA" --out "$pack_dir" --template "$TEMPLATE" --base "$BASE" --license Apache-2.0 \
    --quantization Q4_K_M > "$OUT/pack-$PACK.log" 2>&1 || { step "FAILED pack $PACK"; exit 1; }
step "$PACK: typed-decisions"
typed_decisions "$PACK" || exit 1
step "latency"
: > "$OUT/bench-tiers.jsonl"
for m in "$NAME-Q4_K_M:Qwen3.5-2B" "Qwen3-1.7B-lora-e-s0-imat-Q4_K_M:Qwen3-1.7B" \
         "Qwen3-4B-Instruct-2507-lora-etd-s0-Q4_K_M:Qwen3-4B-Instruct-2507" "Qwen3-8B-lora-etd-s0-Q3_K_M:Qwen3-8B"; do
    "$SDK/bench/run.sh" "$SDK/models/${m%%:*}.gguf" "$DEV" "$SDK/sweep/templates/${m##*:}.json" >> "$OUT/bench-tiers.jsonl" \
        || step "FAILED bench ${m%%:*}"
done
python3 "$SDK/sweep/latency.py" "$OUT/bench-tiers.jsonl" > "$OUT/latency-tiers.md"
step "finish2b.sh finished"
