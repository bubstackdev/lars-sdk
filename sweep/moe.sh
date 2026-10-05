#!/bin/sh
# Can Qwen3-30B-A3B fit where the 4B does? Unattended, resumable, under launchd
# (~/Library/LaunchAgents/dev.lars-sdk.moe.plist); unloads itself when done. Log: sweep/out/moe.log.
#
# 1. Resume the BF16 download; profile expert routing on training-split prompts (moe_profile.py).
# 2. In parallel: importance-matrix Q4_K_M packs of the small LoRA models (imatrix.sh).
# 3. For each variant: keep the K most-used experts per layer and fuse the attention LoRA
#    (moe_shrink.py), convert to BF16 GGUF, compute an importance matrix on training-split prompts,
#    quantize, then capture jev-bench through the SDK and report. Intermediates are deleted once
#    a K's quantizations exist.
#
# Variants, most decisive first (the profile: 64 experts carry 87.5% of the routing weight on LARS
# prompts, 32 carry 63%, 96 carry 98%). IQ3 = experts at IQ3_S, attention at Q8_0, the rest Q4_K_M.
#   k64 Q4_K_M ~10 GB   k64 IQ3 ~8 GB   k32 Q4_K_M ~5 GB   k32 IQ3 ~4.5 GB
#   k128 Q4_K_M ~18 GB (unpruned: the curve's anchor)   k96 IQ3 ~11 GB   k48 Q4_K_M ~7.5 GB
set -u
SDK=$(cd "$(dirname "$0")/.." && pwd)
cd "$SDK"
export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:$HOME/.cargo/bin"
export TMPDIR=${TMPDIR:-/tmp}
CONVERT_PY=$SDK/sweep/.venv-convert/bin/python
LARS_PY=$HOME/Code/personal/lars/.venv/bin/python
TOOLS=$SDK/target/llama-tools/bin
OUT=$SDK/sweep/out
REPO=Qwen/Qwen3-30B-A3B-Instruct-2507
step() { echo "=== $(date '+%F %T') $*"; }

step "moe.sh started"
"$CONVERT_PY" -c "from huggingface_hub import snapshot_download as s; print(s('$REPO', allow_patterns=['*.json','*.safetensors','*.txt','*.jinja','LICENSE','README.md']))" \
    > "$OUT/download-30b.path" 2> "$OUT/download-30b.log" &
DOWNLOAD=$!

if [ ! -s "$OUT/moe-usage.json" ]; then
    step "profiling expert routing"
    HF_HUB_OFFLINE=1 "$LARS_PY" sweep/moe_profile.py --records 1500 --out "$OUT/moe-usage.json" > "$OUT/moe-profile.log" 2>&1 \
        || { step "FAILED profile"; exit 1; }
fi
step "profile done"

HF_HUB_OFFLINE=1 sweep/imatrix.sh >> "$OUT/imatrix.log" 2>&1 &
IMATRIX=$!

wait $DOWNLOAD || { step "FAILED download"; exit 1; }
SRC=$(tail -1 "$OUT/download-30b.path")
step "BF16 weights at $SRC"
if [ -e "$OUT/moe.hold" ]; then
    # Held by request: everything before the variants runs, then the job stops here until
    # sweep/out/moe.hold is removed and the job is started again.
    step "HOLD: download complete; holding before the variants (remove sweep/out/moe.hold to continue)"
    wait $IMATRIX
    step "held; small-model imatrix work finished"
    exit 0
fi
export HF_HUB_OFFLINE=1

quantize() {  # bf16 imatrix out spec
    case "$4" in
        Q4_K_M) "$TOOLS/llama-quantize" --imatrix "$2" "$1" "$3.part" Q4_K_M ;;
        IQ3) "$TOOLS/llama-quantize" --imatrix "$2" \
                 --tensor-type ffn_gate_exps=iq3_s --tensor-type ffn_up_exps=iq3_s --tensor-type ffn_down_exps=iq3_s \
                 --tensor-type attn_q=q8_0 --tensor-type attn_k=q8_0 --tensor-type attn_v=q8_0 --tensor-type attn_output=q8_0 \
                 "$1" "$3.part" Q4_K_M ;;
    esac > "$OUT/quantize-$(basename "$3" .gguf).log" 2>&1 && mv "$3.part" "$3"
}

specs_for() {
    case $1 in 64|32) echo "Q4_K_M IQ3" ;; 96) echo "IQ3" ;; *) echo "Q4_K_M" ;; esac
}

# Build every quantization of a K at once (the BF16 intermediates are deleted afterwards).
# (sh functions have no locals: build's variables are prefixed so the caller's k, spec, name survive)
build() {
    b_k=$1 b_name=Qwen3-30B-A3B-lars-k$1
    missing=""
    for b_spec in $(specs_for $b_k); do [ -s "$SDK/models/$b_name-$b_spec.gguf" ] || missing="$missing $b_spec"; done
    [ -n "$missing" ] || return 0
    fused=$SDK/models/fused/$b_name
    bf16=$SDK/models/$b_name-BF16.gguf
    matrix=$SDK/models/$b_name.imatrix.gguf
    step "$b_name: shrink and fuse"
    [ -f "$fused/config.json" ] || "$CONVERT_PY" sweep/moe_shrink.py --src "$SRC" \
        --adapter "$HOME/Code/personal/lars/checkpoints/lora-30b-e-s0" --usage "$OUT/moe-usage.json" --keep $b_k \
        --out "$fused" > "$OUT/shrink-$b_name.log" 2>&1 || { step "FAILED shrink $b_name"; return 1; }
    step "$b_name: convert"
    [ -s "$bf16" ] || { "$CONVERT_PY" crates/lars-llama-sys/llama.cpp/convert_hf_to_gguf.py "$fused" --outtype bf16 \
        --outfile "$bf16.part" > "$OUT/convert-$b_name.log" 2>&1 && mv "$bf16.part" "$bf16"; } || { step "FAILED convert $b_name"; return 1; }
    step "$b_name: importance matrix"
    [ -s "$matrix" ] || "$TOOLS/llama-imatrix" -m "$bf16" -f "$SDK/models/imatrix-Qwen3-4B-Instruct-2507.txt" \
        --parse-special --chunks 400 -ngl 99 -o "$matrix" > "$OUT/imatrix-$b_name.log" 2>&1 || { step "FAILED imatrix $b_name"; return 1; }
    for b_spec in $missing; do
        step "$b_name: quantize $b_spec"
        quantize "$bf16" "$matrix" "$SDK/models/$b_name-$b_spec.gguf" "$b_spec" || step "FAILED quantize $b_name-$b_spec"
    done
    rm -rf "$fused" "$bf16"
}

# Captures in order of what they decide: the 64-expert curve point, the unpruned anchor, then
# the rest.
for variant in 64:Q4_K_M 128:Q4_K_M; do
    k=${variant%%:*} spec=${variant#*:}
    name=Qwen3-30B-A3B-lars-k$k
    build $k || continue
    [ -s "$SDK/models/$name-$spec.gguf" ] || continue
    ls -la "$SDK/models/$name-$spec.gguf" | awk '{printf "%s: %.2f GB\n", $9, $5 / 1e9}'
    sweep/run.sh "$name-$spec" "$SDK/models/$name-$spec.gguf" metal v1 || step "FAILED capture $name-$spec"
    sweep/report.sh
done
wait $IMATRIX
sweep/report.sh
step "moe.sh finished"
launchctl bootout "gui/$(id -u)/dev.lars-sdk.moe" 2>/dev/null
