//! Narrow bindings to vision.cpp; ownership remains with the model worker.
use crate::{llama_context, llama_model};
use std::ffi::{c_char, c_void};
unsafe extern "C" {
    pub fn lars_vision_open(
        path: *const c_char,
        model: *const llama_model,
        gpu: bool,
        threads: i32,
        max_tokens: i32,
    ) -> *mut c_void;
    pub fn lars_vision_close(ctx: *mut c_void);
    pub fn lars_vision_error(ctx: *const c_void) -> *const c_char;
    pub fn lars_vision_clear(ctx: *mut c_void);
    pub fn lars_vision_set_image(ctx: *mut c_void, bytes: *const u8, len: usize) -> i32;
    pub fn lars_vision_count(ctx: *mut c_void, text: *const c_char, len: usize, count: *mut usize) -> i32;
    pub fn lars_vision_score(
        ctx: *mut c_void,
        llama: *mut llama_context,
        text: *const c_char,
        len: usize,
        batch: i32,
    ) -> i32;
}
