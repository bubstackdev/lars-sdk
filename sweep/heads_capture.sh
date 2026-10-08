#!/bin/sh
# Logits plus final hidden states (answer position) through the 4B and 0.6B packs, for the
# per-predicate head experiment (sweep/heads.py): jev-bench validation and test, typed-decisions
# train and test. Resumable; output in sweep/out/heads/<pack>/. Log: sweep/out/heads/capture.log.
cd "$(dirname "$0")/.."
OUT=sweep/out/heads
DATA=$(cd "$(dirname "$0")/.." && pwd)/data/evals
TD=sweep/out/typed-decisions
mkdir -p $OUT
while pgrep -f 'typed-decisions/capture.sh' > /dev/null; do sleep 20; done
for spec in 4b:Qwen3-4B-Instruct-2507 0.6b:Qwen3-0.6B; do
    size=${spec%%:*} base=${spec#*:} dir=$OUT/lars-$size-dev
    mkdir -p $dir
    for src in $TD/typed-decisions.test.jsonl $TD/typed-decisions.train.jsonl $DATA/*.validation.jsonl $DATA/*.test.jsonl; do
        name=$(basename "$src" .jsonl)
        [ -s "$dir/$name.f32" ] && continue
        ./target/release/lars-dev capture --model models/packs/lars-$size-dev/0.0.1/model.gguf --template sweep/templates/$base.json \
            --data "$src" --out "$dir/$name.jsonl" --hidden "$dir/$name.f32.part" 2>> $OUT/capture.log \
            && mv "$dir/$name.f32.part" "$dir/$name.f32" || echo "$(date '+%T') FAILED $size $name" >> $OUT/capture.log
    done
    echo "$(date '+%T') lars-$size-dev done" >> $OUT/capture.log
done
