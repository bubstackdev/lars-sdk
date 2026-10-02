#!/bin/sh
# Build the Phase 2b model files: original Qwen weights -> BF16 GGUF -> Q8_0 and Q4_K_M, all with
# the pinned llama.cpp's own converter and quantizer, so every file is quantized the same way.
#
#     sweep/prepare.sh            (writes models/<name>-{BF16,Q8_0,Q4_K_M}.gguf)
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd)
LLAMA=$ROOT/crates/lars-llama-sys/llama.cpp
VENV=$ROOT/sweep/.venv-convert
TOOLS=$ROOT/target/llama-tools

if [ ! -x "$VENV/bin/python" ]; then
    uv venv -q --python 3.12 "$VENV"
    uv pip install -q --python "$VENV" -r "$LLAMA/requirements/requirements-convert_hf_to_gguf.txt" huggingface_hub
fi
if [ ! -x "$TOOLS/bin/llama-quantize" ]; then
    cmake -S "$LLAMA" -B "$TOOLS" -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_TESTS=OFF \
        -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_OPENSSL=OFF -DGGML_OPENMP=OFF > /dev/null
    cmake --build "$TOOLS" --target llama-quantize -j > /dev/null
fi
QUANTIZE=$(find "$TOOLS" -name llama-quantize -type f -perm -u+x | head -1)

for repo in Qwen/Qwen3-0.6B Qwen/Qwen3-1.7B Qwen/Qwen3-4B-Instruct-2507; do
    name=$(basename "$repo")
    src=$("$VENV/bin/python" -c "from huggingface_hub import snapshot_download as s; print(s('$repo', allow_patterns=['*.json','*.safetensors','*.txt','*.jinja','merges.txt','vocab.json']))")
    bf16=$ROOT/models/$name-BF16.gguf
    [ -f "$bf16" ] || "$VENV/bin/python" "$LLAMA/convert_hf_to_gguf.py" "$src" --outtype bf16 --outfile "$bf16" > /dev/null 2>&1
    for q in Q8_0 Q4_K_M; do
        out=$ROOT/models/$name-$q.gguf
        [ -f "$out" ] || "$QUANTIZE" "$bf16" "$out" $q > /dev/null 2>&1
    done
    ls -la "$ROOT/models/$name"-*.gguf | awk '{print $5, $9}'
done
