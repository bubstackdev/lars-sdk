//! Raw bindings to llama.cpp, vendored at a pinned commit (`llama.cpp/`, a git submodule) and
//! linked statically: CPU everywhere, plus Metal on Apple platforms.
//!
//! The commit is part of what a model pack is validated on: kernels change between llama.cpp
//! versions, and with them the logits calibrators were fitted on. `src/bindings.rs` is generated
//! by `regenerate-bindings.sh` and committed, so builds need no libclang.

#![allow(non_upper_case_globals, non_camel_case_types, non_snake_case, clippy::all)]

include!("bindings.rs");
pub mod vision;

/// The llama.cpp release these bindings and this build come from.
pub const LLAMA_CPP_VERSION: &str = env!("LARS_LLAMA_CPP_VERSION");
