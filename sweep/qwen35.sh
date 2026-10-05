#!/bin/sh
# Zero-shot screen of Qwen3.5-9B (the "bigger model at 4B-class memory" candidate, about 4.7 GB at
# Q3_K_M) and Qwen3.5-2B (a better small base) through the SDK. Unattended under launchd
# (~/Library/LaunchAgents/dev.lars-sdk.qwen35.plist), resumable, unloads itself. Log: sweep/out/qwen35.log.
#
# 1. Download the original weights; capture each model's chat template with thinking off.
# 2. BF16 GGUF with the pinned converter, an importance matrix on training-split prompts rendered with
#    that template, then quantize: 9B at Q3_K_M and Q4_K_M, 2B at Q4_K_M.
# 3. Wait for the 30B job's GPU work. Then check that the backend handles Qwen3.5's recurrent state:
#    logits with prefix reuse and batching on must match them off (sweep/out/qwen35-checks.txt).
# 4. Capture jev-bench through the SDK and report, like every other run.
set -u
SDK=$(cd "$(dirname "$0")/.." && pwd)
cd "$SDK"
export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:$HOME/.cargo/bin"
export TMPDIR=${TMPDIR:-/tmp}
CONVERT_PY=$SDK/sweep/.venv-convert/bin/python
LARS_PY=$HOME/Code/personal/lars/.venv/bin/python
TOOLS=$SDK/target/llama-tools/bin
OUT=$SDK/sweep/out
DEV=$SDK/target/release/lars-dev
step() { echo "=== $(date '+%F %T') $*"; }

step "qwen35.sh started"
for base in Qwen3.5-9B Qwen3.5-2B; do
    src=$("$CONVERT_PY" -c "from huggingface_hub import snapshot_download as s; print(s('Qwen/$base', allow_patterns=['*.json','*.safetensors','*.txt','*.jinja','merges.txt','vocab.json','LICENSE','README.md']))" 2>> "$OUT/download-$base.log" | tail -1)
    [ -d "$src" ] || { step "FAILED download $base"; continue; }
    step "$base at $src"
    template=$SDK/sweep/templates/$base.json
    [ -s "$template" ] || HF_HUB_OFFLINE=1 "$LARS_PY" -c "
import json, sys
sys.path.insert(0, 'fixtures')
from generate import capture_template
t = capture_template('Qwen/$base')
open('$template', 'w').write(json.dumps(t, indent=1, ensure_ascii=False) + '\n')
print('template', t['trims_content'], repr(t['after_user']))" >> "$OUT/qwen35.log" 2>&1 || { step "FAILED template $base"; continue; }
    bf16=$SDK/models/$base-BF16.gguf
    [ -s "$bf16" ] || { "$CONVERT_PY" crates/lars-llama-sys/llama.cpp/convert_hf_to_gguf.py "$src" --outtype bf16 \
        --outfile "$bf16.part" > "$OUT/convert-$base.log" 2>&1 && mv "$bf16.part" "$bf16"; } || { step "FAILED convert $base"; continue; }
    text=$SDK/models/imatrix-$base.txt
    [ -s "$text" ] || HF_HUB_OFFLINE=1 "$LARS_PY" sweep/imatrix_text.py --template "$template" --records 1500 --out "$text" \
        >> "$OUT/qwen35.log" 2>&1 || { step "FAILED imatrix text $base"; continue; }
    matrix=$SDK/models/$base.imatrix.gguf
    [ -s "$matrix" ] || "$TOOLS/llama-imatrix" -m "$bf16" -f "$text" --parse-special --chunks 400 -ngl 99 -o "$matrix" \
        > "$OUT/imatrix-$base.log" 2>&1 || { step "FAILED imatrix $base"; continue; }
    case $base in *9B) quants="Q3_K_M Q4_K_M" ;; *) quants="Q4_K_M" ;; esac
    for q in $quants; do
        out=$SDK/models/$base-$q.gguf
        [ -s "$out" ] || { "$TOOLS/llama-quantize" --imatrix "$matrix" "$bf16" "$out.part" $q > "$OUT/quantize-$base-$q.log" 2>&1 \
            && mv "$out.part" "$out"; } || step "FAILED quantize $base-$q"
        [ -s "$out" ] && ls -la "$out" | awk '{printf "%s: %.2f GB\n", $9, $5 / 1e9}'
    done
done

while pgrep -f 'sweep/moe.sh' > /dev/null; do sleep 30; done
step "30B job done; GPU free"

# Recurrent-state checks on the 2B (fast): the same rows with the backend's reuse and batching on
# and off must give the same logits (batching may differ slightly, as on every model; reuse must not).
check=$OUT/qwen35-checks.txt
if [ ! -s "$check" ] && [ -s "$SDK/models/Qwen3.5-2B-Q4_K_M.gguf" ]; then
    sample=$OUT/qwen35-sample.jsonl
    for f in "$HOME"/Code/personal/lars/evals/data/*.validation.jsonl; do head -6 "$f"; done > "$sample"
    m=$SDK/models/Qwen3.5-2B-Q4_K_M.gguf t=$SDK/sweep/templates/Qwen3.5-2B.json
    "$DEV" capture --model "$m" --template "$t" --data "$sample" --out "$OUT/qwen35-reuse.jsonl" 2>> "$check"
    "$DEV" capture --model "$m" --template "$t" --data "$sample" --out "$OUT/qwen35-noreuse.jsonl" --no-reuse-prefix 2>> "$check"
    python3 -c "
import json
req = json.load(open('$SDK/examples/molar_triage/request.json'))
rows = [json.loads(l) for l in open('$SDK/examples/molar_triage/molar_triage.test.jsonl')]
states = list(dict.fromkeys(r['state'] for r in rows))[:12]
json.dump([{'state': s, 'questions': req['questions']} for s in states], open('$OUT/qwen35-requests.json', 'w'))"
    "$DEV" capture-requests --model "$m" --template "$t" --requests "$OUT/qwen35-requests.json" --out "$OUT/qwen35-batched.jsonl" 2>> "$check"
    "$DEV" capture-requests --model "$m" --template "$t" --requests "$OUT/qwen35-requests.json" --out "$OUT/qwen35-sequential.jsonl" \
        --no-batch-suffixes --no-reuse-prefix 2>> "$check"
    { echo "prefix reuse on vs off (single-question rows):"; "$DEV" compare "$OUT/qwen35-reuse.jsonl" "$OUT/qwen35-noreuse.jsonl"
      echo "batched + reuse vs sequential, no reuse (Molar Triage requests):"; "$DEV" compare "$OUT/qwen35-batched.jsonl" "$OUT/qwen35-sequential.jsonl"; } >> "$check" 2>&1
    step "recurrent-state checks written to $check"
fi

for name in Qwen3.5-9B-Q3_K_M Qwen3.5-2B-Q4_K_M Qwen3.5-9B-Q4_K_M; do
    [ -s "$SDK/models/$name.gguf" ] || continue
    sweep/run.sh "$name" "$SDK/models/$name.gguf" metal v1 || step "FAILED capture $name"
    sweep/report.sh
done
step "qwen35.sh finished"
launchctl bootout "gui/$(id -u)/dev.lars-sdk.qwen35" 2>/dev/null
