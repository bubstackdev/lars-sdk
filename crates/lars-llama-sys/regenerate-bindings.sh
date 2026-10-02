#!/bin/sh
# Regenerate src/bindings.rs from the vendored llama.h. Run after bumping the llama.cpp submodule;
# needs bindgen (`cargo install bindgen-cli`) and libclang. Builds never run this.
set -eu
cd "$(dirname "$0")"
export LIBCLANG_PATH="${LIBCLANG_PATH:-/Library/Developer/CommandLineTools/usr/lib}"
bindgen llama.cpp/include/llama.h -o src/bindings.rs \
  --allowlist-function 'llama_.*' --allowlist-type 'llama_.*' --allowlist-var 'LLAMA_.*' \
  --no-layout-tests --use-core \
  --raw-line "// Generated from llama.cpp $(git -C llama.cpp describe --tags --always) by regenerate-bindings.sh." \
  -- -I llama.cpp/ggml/include
