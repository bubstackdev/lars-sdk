# Shared steps for LoRA runs (sourced by sweep/train4b.sh): train, fuse, convert and quantize
# with an importance matrix, capture jev-bench, and typed-decisions through a pack. The caller
# sets SDK, LARS, PY, CONVERT, FIXPY, LLAMA, TOOLS, DEV, OUT, TD, DATA, RECORDS, SELECT, SKIP,
# TRAIN_ARGS, TEMPLATE, SMOKE, LIMIT_ROWS and RESULTS, and optionally CONVERT_ARGS. (sweep/overnight.sh has its own copy, written
# before this file; fold it in once that run is over.)

step() { echo "=== $(date '+%F %T') $*"; }

# Train once; an interrupted run leaves no .done and starts over.
train() {  # hub-model checkpoint log
    [ -f "$2/.done" ] && return 0
    rm -rf "$2" "$2.steps" "$2.last" "$2.history.json" "$2.baseline.json"
    (cd "$LARS" && "$PY" -m lars.train.lora --model "$1" --records $RECORDS --seed 0 \
        --select-records $SELECT --select-skip $SKIP --out "$2" $TRAIN_ARGS) > "$3" 2>&1 && touch "$2/.done"
}

snapshot() {
    "$PY" -c "from huggingface_hub import snapshot_download as s; print(s('$1', local_files_only=True))"
}

fuse() {  # hub-model adapter out
    [ -f "$3/config.json" ] && return 0
    rm -rf "$3.part"
    (cd "$LARS" && "$PY" -m mlx_lm fuse --model "$1" --adapter-path "$2" --save-path "$3.part") \
        > "$OUT/fuse-$(basename "$3").log" 2>&1 || return 1
    # The base's own tokenizer files: mlx_lm rewrites tokenizer_config with a newer transformers.
    base=$(snapshot "$1") || return 1
    for f in tokenizer.json tokenizer_config.json vocab.json merges.txt special_tokens_map.json; do
        [ -f "$base/$f" ] && cp "$base/$f" "$3.part/$f"
    done
    mv "$3.part" "$3"
}

# A Hugging Face directory -> models/<name>-<quant>.gguf for each quant, with an importance matrix
# computed on training-split prompts rendered with the base's template.
build() {  # name hf-dir template quants...
    b_name=$1 b_dir=$2 b_template=$3
    shift 3
    b_bf16=$SDK/models/$b_name-BF16.gguf
    b_text=$SDK/models/imatrix-$(basename "$b_template" .json).txt
    b_matrix=$SDK/models/$b_name.imatrix.gguf
    [ -s "$b_bf16" ] || { "$CONVERT" "$LLAMA/convert_hf_to_gguf.py" "$b_dir" --outtype bf16 ${CONVERT_ARGS:-} --outfile "$b_bf16.part" \
        > "$OUT/convert-$b_name.log" 2>&1 && mv "$b_bf16.part" "$b_bf16"; } || return 1
    [ -s "$b_text" ] || "$PY" "$SDK/sweep/imatrix_text.py" --template "$b_template" --records 1500 --out "$b_text" \
        > "$OUT/imatrix-text-$b_name.log" 2>&1 || return 1
    [ -s "$b_matrix" ] || "$TOOLS/llama-imatrix" -m "$b_bf16" -f "$b_text" --parse-special --chunks 400 -ngl 99 -o "$b_matrix" \
        > "$OUT/imatrix-$b_name.log" 2>&1 || return 1
    for b_q in "$@"; do
        b_out=$SDK/models/$b_name-$b_q.gguf
        [ -s "$b_out" ] || { "$TOOLS/llama-quantize" --imatrix "$b_matrix" "$b_bf16" "$b_out.part" "$b_q" \
            > "$OUT/quantize-$b_name-$b_q.log" 2>&1 && mv "$b_out.part" "$b_out"; } || return 1
        ls -la "$b_out" | awk '{printf "%s: %.2f GB\n", $9, $5 / 1e9}'
    done
    rm -f "$b_bf16"
}

capture() {  # name (models/<name>.gguf)
    if [ "$SMOKE" = 1 ]; then
        cat "$DATA/boolq.test.jsonl" > "$OUT/smoke-rows.jsonl"
        "$DEV" capture --model "$SDK/models/$1.gguf" --template "$TEMPLATE" --data "$OUT/smoke-rows.jsonl" \
            --out "$OUT/smoke-capture.jsonl" || return 1
        return 0
    fi
    "$SDK/sweep/run.sh" "$1" "$SDK/models/$1.gguf" metal v1 || step "FAILED capture $1"
    "$SDK/sweep/report.sh"
}

# typed-decisions through a pack, with hidden states, then scores and the head simulation.
typed_decisions() {  # pack-name
    t_pack=$SDK/models/packs/$1/0.0.1
    t_dir=$OUT/heads/$1
    mkdir -p "$t_dir"
    for t_split in test train; do
        t_data=$TD/typed-decisions.$t_split.jsonl
        if [ -n "$LIMIT_ROWS" ]; then
            head -"$LIMIT_ROWS" "$t_data" > "$OUT/smoke-td-$t_split.jsonl"
            t_data=$OUT/smoke-td-$t_split.jsonl
        fi
        [ -s "$t_dir/typed-decisions.$t_split.f32" ] && continue
        "$DEV" capture --model "$t_pack/model.gguf" --template "$TEMPLATE" --data "$t_data" \
            --out "$t_dir/typed-decisions.$t_split.jsonl" --hidden "$t_dir/typed-decisions.$t_split.f32.part" \
            >> "$OUT/typed-decisions-$1.log" 2>&1 && mv "$t_dir/typed-decisions.$t_split.f32.part" "$t_dir/typed-decisions.$t_split.f32" \
            || { step "FAILED typed-decisions capture $1 $t_split"; return 1; }
    done
    [ "$SMOKE" = 1 ] && return 0
    {
        echo "## $1 on typed-decisions test (pack calibrator)"
        "$FIXPY" "$SDK/sweep/typed_decisions.py" "$1=$t_dir/typed-decisions.test.jsonl:$t_pack/calibration.json" | grep -v '^    '
        echo "## $1 per-predicate heads (hard labels)"
        "$FIXPY" "$SDK/sweep/heads.py" "$t_dir" --pack "$t_pack/calibration.json" --n 25,50,100,200,300 --only td: --hard --tag .td-hard | tail -32
    } >> "$RESULTS" 2>&1
}

