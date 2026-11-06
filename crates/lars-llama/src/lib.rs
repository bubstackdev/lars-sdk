//! The llama.cpp backend: GGUF models in-process, on CPU or Metal.
//!
//! A row's logits come from one prefill of the shared prefix into sequence 0, then the row's
//! suffix decoded after it, reading the next-token logits at the suffix's last token. Two ways
//! to run the suffixes:
//!
//! - **batched** (the default, `batch_suffixes`): copy the prefix's cells to one sequence per
//!   suffix (the cache is unified, so nothing is copied) and decode all suffixes in one pass.
//!   Twenty questions over one state take about a third of the time. A row's logits can move by
//!   up to 0.009 in probability with what it is batched with; a pooled calibrator absorbs far
//!   larger differences than that (CPU against Metal, RESULTS.md 2026-10-05).
//! - **one at a time**: decode the suffix into sequence 0, read, then truncate back to the
//!   prefix, as the reference's llama.cpp backend does with save/restore. A prefix with a single
//!   suffix always runs this way, so a one-question call (and so every eval and calibration
//!   capture) is bit-identical to the reference whatever the setting.
//!
//! With `reuse_prefix` (the default), the tokens left in sequence 0 by the previous call are kept
//! as far as they match the next prefix, so a constant head (system prompt, template), or a whole
//! state asked about again, is prefilled once. Measured bit-identical to a fresh prefill
//! (RESULTS.md, 2026-10-02).
//!
//! Recurrent and hybrid models (Qwen3.5) keep a running state that can't be cut back to an
//! earlier token, so the cache can't be truncated to a shared prefix. For them, a spare sequence
//! holds a snapshot of the last prefix instead: a copy within the cache (`seq_cp`, the cells
//! shared until written), restored the same way. A state asked about again, or questions asked
//! one at a time, are not prefilled again; Qwen3.5-2B answers the same input again in 19 ms
//! instead of 39, and twenty questions one at a time in 0.42 s instead of 1.03. Bit-identical to
//! a fresh prefill. A new state is prefilled in full: snapshotting the shared head too saved up
//! to 15% but moved probabilities by up to 0.0013, since the recurrent state is computed in
//! different pieces (RESULTS.md, 2026-10-07). The snapshot's sequence is not a suffix lane.
//!
//! Flash attention is off by default: on, it moves probabilities by up to 0.035 on Metal and
//! 0.54 on CPU, and off matches the reference engine bit for bit. Whatever a pack's calibrator
//! was fitted with is what it must be served with.
//!
//! The boundary check matches the reference: if the prefix's tokens are not a prefix of the
//! full prompt's tokens, that row is prefilled in full instead.

use std::collections::HashMap;
use std::ffi::{CString, c_char, c_void};
use std::path::Path;
use std::ptr::NonNull;
use std::sync::Once;

use lars_core::backend::{Backend, LogitsAndHidden};
use lars_core::error::{Error, Result};
use lars_core::render::Template;
use lars_llama_sys as sys;

pub use sys::LLAMA_CPP_VERSION;

#[derive(Debug, Clone)]
pub struct LlamaOptions {
    /// Cache cells for one call: the prefix plus every suffix decoded with it.
    pub n_ctx: u32,
    pub n_batch: u32,
    pub n_ubatch: u32,
    /// Sequences per pass when suffixes are batched (one holds the prefix).
    pub n_seq_max: u32,
    /// CPU threads; 0 picks the number of available cores.
    pub n_threads: i32,
    /// Layers offloaded to the GPU (Metal); -1 for all, 0 for CPU only.
    pub gpu_layers: i32,
    pub flash_attention: FlashAttention,
    pub batch_suffixes: bool,
    pub reuse_prefix: bool,
    /// Let llama.cpp repack weights for its CPU kernels (faster on CPU, but a second copy in memory).
    pub repack_cpu_weights: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FlashAttention {
    Auto,
    On,
    Off,
}

impl Default for LlamaOptions {
    fn default() -> Self {
        LlamaOptions {
            n_ctx: 8192,
            n_batch: 2048,
            n_ubatch: 512,
            n_seq_max: 16,
            n_threads: 0,
            gpu_layers: -1,
            flash_attention: FlashAttention::Off,
            batch_suffixes: true,
            reuse_prefix: true,
            repack_cpu_weights: true,
        }
    }
}

static INIT: Once = Once::new();

unsafe extern "C" fn quiet_progress(_: f32, _: *mut c_void) -> bool {
    true
}

unsafe extern "C" fn log_callback(level: sys::ggml_log_level, text: *const c_char, _: *mut c_void) {
    // llama.cpp is chatty; its log goes to stderr only when LARS_LLAMA_LOG is set, or for errors.
    let verbose = std::env::var_os("LARS_LLAMA_LOG").is_some();
    if (verbose || level >= sys::ggml_log_level_GGML_LOG_LEVEL_ERROR) && !text.is_null() {
        let message = unsafe { std::ffi::CStr::from_ptr(text) }.to_string_lossy();
        eprint!("{message}");
    }
}

fn init() {
    INIT.call_once(|| unsafe {
        sys::llama_log_set(Some(log_callback), std::ptr::null_mut());
        sys::llama_backend_init();
    });
}

fn backend_error(message: impl Into<String>) -> Error {
    Error::Backend(message.into())
}

/// An owned `llama_batch`.
struct Batch {
    raw: sys::llama_batch,
    capacity: usize,
}

impl Batch {
    fn new(capacity: usize, n_seq_max: usize) -> Batch {
        let raw = unsafe { sys::llama_batch_init(capacity as i32, 0, n_seq_max as i32) };
        Batch { raw, capacity }
    }

    fn clear(&mut self) {
        self.raw.n_tokens = 0;
    }

    fn push(&mut self, token: sys::llama_token, pos: sys::llama_pos, seq: sys::llama_seq_id, output: bool) {
        let i = self.raw.n_tokens as usize;
        assert!(i < self.capacity, "batch overflow");
        unsafe {
            *self.raw.token.add(i) = token;
            *self.raw.pos.add(i) = pos;
            *self.raw.n_seq_id.add(i) = 1;
            *(*self.raw.seq_id.add(i)).add(0) = seq;
            *self.raw.logits.add(i) = output as i8;
        }
        self.raw.n_tokens += 1;
    }

    fn len(&self) -> usize {
        self.raw.n_tokens as usize
    }
}

impl Drop for Batch {
    fn drop(&mut self) {
        unsafe { sys::llama_batch_free(self.raw) }
    }
}

pub struct LlamaBackend {
    vision: Option<NonNull<std::ffi::c_void>>,
    image_active: bool,
    model: NonNull<sys::llama_model>,
    ctx: NonNull<sys::llama_context>,
    vocab: *const sys::llama_vocab,
    n_vocab: usize,
    template: Template,
    model_name: String,
    options: LlamaOptions,
    labels: HashMap<String, Option<sys::llama_token>>,
    /// Tokens currently held in sequence 0.
    cached: Vec<sys::llama_token>,
    batch: Batch,
    /// Read the final hidden state with each row's logits (`label_logits_hidden`).
    want_hidden: bool,
    /// A recurrent or hybrid model with `reuse_prefix`: the last prefix is reused from a snapshot.
    snapshot_prefixes: bool,
    /// The tokens the snapshot (sequence n_seq_max - 1) holds; empty when it holds nothing.
    snapshot: Vec<sys::llama_token>,
}

/// A row's label logits, and its hidden state when asked for.
type RowOut = (Vec<f64>, Option<Vec<f32>>);

// The model and context are only touched through `&mut self` or immutable queries.
unsafe impl Send for LlamaBackend {}

impl Drop for LlamaBackend {
    fn drop(&mut self) {
        unsafe {
            if let Some(vision) = self.vision {
                sys::vision::lars_vision_close(vision.as_ptr());
            }
            sys::llama_free(self.ctx.as_ptr());
            sys::llama_model_free(self.model.as_ptr());
        }
    }
}

impl LlamaBackend {
    pub fn load(path: &Path, template: Template, model_name: &str, options: LlamaOptions) -> Result<LlamaBackend> {
        init();
        let c_path =
            CString::new(path.to_string_lossy().as_bytes()).map_err(|_| backend_error("model path has a NUL"))?;
        let mut model_params = unsafe { sys::llama_model_default_params() };
        model_params.n_gpu_layers = options.gpu_layers;
        model_params.progress_callback = Some(quiet_progress);
        model_params.use_extra_bufts = options.repack_cpu_weights;
        let model = NonNull::new(unsafe { sys::llama_model_load_from_file(c_path.as_ptr(), model_params) })
            .ok_or_else(|| backend_error(format!("could not load {}", path.display())))?;

        let mut ctx_params = unsafe { sys::llama_context_default_params() };
        let threads = if options.n_threads > 0 {
            options.n_threads
        } else {
            std::thread::available_parallelism().map_or(4, |n| n.get() as i32)
        };
        ctx_params.n_ctx = options.n_ctx;
        ctx_params.n_batch = options.n_batch;
        ctx_params.n_ubatch = options.n_ubatch;
        ctx_params.n_seq_max = options.n_seq_max;
        ctx_params.n_threads = threads;
        ctx_params.n_threads_batch = threads;
        ctx_params.kv_unified = true;
        ctx_params.no_perf = true;
        // Per-token hidden states when a head asks for them (what an unspecified pooling resolves
        // to for these models anyway).
        ctx_params.pooling_type = sys::llama_pooling_type_LLAMA_POOLING_TYPE_NONE;
        ctx_params.flash_attn_type = match options.flash_attention {
            FlashAttention::Auto => sys::llama_flash_attn_type_LLAMA_FLASH_ATTN_TYPE_AUTO,
            FlashAttention::On => sys::llama_flash_attn_type_LLAMA_FLASH_ATTN_TYPE_ENABLED,
            FlashAttention::Off => sys::llama_flash_attn_type_LLAMA_FLASH_ATTN_TYPE_DISABLED,
        };
        let Some(ctx) = NonNull::new(unsafe { sys::llama_init_from_model(model.as_ptr(), ctx_params) }) else {
            unsafe { sys::llama_model_free(model.as_ptr()) };
            return Err(backend_error("could not create a llama.cpp context"));
        };
        let vocab = unsafe { sys::llama_model_get_vocab(model.as_ptr()) };
        let n_vocab = unsafe { sys::llama_vocab_n_tokens(vocab) } as usize;
        let batch = Batch::new(options.n_batch.max(options.n_ctx) as usize, options.n_seq_max as usize);
        let recurrent =
            unsafe { sys::llama_model_is_recurrent(model.as_ptr()) || sys::llama_model_is_hybrid(model.as_ptr()) };
        let snapshot_prefixes = recurrent && options.reuse_prefix && options.n_seq_max >= 3;
        Ok(LlamaBackend {
            vision: None,
            image_active: false,
            model,
            ctx,
            vocab,
            n_vocab,
            template,
            model_name: model_name.to_string(),
            options,
            labels: HashMap::new(),
            cached: Vec::new(),
            batch,
            want_hidden: false,
            snapshot_prefixes,
            snapshot: Vec::new(),
        })
    }

    pub fn options(&self) -> &LlamaOptions {
        &self.options
    }

    /// Attach the matching projector. Images are deliberately evaluated serially without
    /// cross-request prefix reuse until multimodal branching has independent validation.
    pub fn load_projector(&mut self, path: &Path, max_tokens: i32) -> Result<()> {
        let path =
            CString::new(path.to_string_lossy().as_bytes()).map_err(|_| backend_error("projector path has a NUL"))?;
        let threads = if self.options.n_threads > 0 {
            self.options.n_threads
        } else {
            std::thread::available_parallelism().map_or(4, |n| n.get() as i32)
        };
        let ptr = unsafe {
            sys::vision::lars_vision_open(
                path.as_ptr(),
                self.model.as_ptr(),
                self.options.gpu_layers != 0,
                threads,
                max_tokens,
            )
        };
        let vision = NonNull::new(ptr).ok_or_else(|| backend_error("could not load matching vision projector"))?;
        if let Some(old) = self.vision.replace(vision) {
            unsafe {
                sys::vision::lars_vision_close(old.as_ptr());
            }
        }
        Ok(())
    }

    fn vision_error(&self) -> lars_core::error::Error {
        let ptr = self.vision.expect("vision loaded");
        let message =
            unsafe { std::ffi::CStr::from_ptr(sys::vision::lars_vision_error(ptr.as_ptr())) }.to_string_lossy();
        backend_error(message)
    }

    /// The model's own chat template from its GGUF metadata, if it has one.
    pub fn gguf_chat_template(&self) -> Option<String> {
        let raw = unsafe { sys::llama_model_chat_template(self.model.as_ptr(), std::ptr::null()) };
        (!raw.is_null()).then(|| unsafe { std::ffi::CStr::from_ptr(raw) }.to_string_lossy().into_owned())
    }

    pub fn tokenize(&self, text: &str, add_special: bool, parse_special: bool) -> Vec<sys::llama_token> {
        let mut tokens = vec![0; text.len() + 8];
        loop {
            let n = unsafe {
                sys::llama_tokenize(
                    self.vocab,
                    text.as_ptr() as *const c_char,
                    text.len() as i32,
                    tokens.as_mut_ptr(),
                    tokens.len() as i32,
                    add_special,
                    parse_special,
                )
            };
            if n >= 0 {
                tokens.truncate(n as usize);
                return tokens;
            }
            tokens.resize((-n) as usize, 0);
        }
    }

    fn label_id(&mut self, label: &str) -> Option<sys::llama_token> {
        if let Some(id) = self.labels.get(label) {
            return *id;
        }
        let ids = self.tokenize(&format!(" {label}"), false, false);
        let id = (ids.len() == 1).then(|| ids[0]);
        self.labels.insert(label.to_string(), id);
        id
    }

    fn memory(&self) -> sys::llama_memory_t {
        unsafe { sys::llama_get_memory(self.ctx.as_ptr()) }
    }

    /// Decode the batch; outputs are read with `read_row`.
    fn decode(&mut self) -> Result<()> {
        match unsafe { sys::llama_decode(self.ctx.as_ptr(), self.batch.raw) } {
            0 => Ok(()),
            1 => Err(backend_error("llama.cpp found no room in the cache; raise n_ctx")),
            code => Err(backend_error(format!("llama_decode failed ({code})"))),
        }
    }

    /// The label logits at a batch index, and the hidden state there when `want_hidden`.
    fn read_row(&self, batch_index: usize, label_ids: &[sys::llama_token]) -> Result<RowOut> {
        let row = unsafe { sys::llama_get_logits_ith(self.ctx.as_ptr(), batch_index as i32) };
        if row.is_null() {
            return Err(backend_error("llama.cpp returned no logits"));
        }
        let row = unsafe { std::slice::from_raw_parts(row, self.n_vocab) };
        let logits = label_ids.iter().map(|&id| row[id as usize] as f64).collect();
        if !self.want_hidden {
            return Ok((logits, None));
        }
        let at = unsafe { sys::llama_get_embeddings_ith(self.ctx.as_ptr(), batch_index as i32) };
        if at.is_null() {
            return Err(backend_error("llama.cpp returned no hidden state"));
        }
        let n = unsafe { sys::llama_model_n_embd_out(self.model.as_ptr()) } as usize;
        Ok((logits, Some(unsafe { std::slice::from_raw_parts(at, n) }.to_vec())))
    }

    /// Decode the batch with embeddings on. They make every token of a decode an output (logits
    /// included), so callers decode only the tokens they read this way.
    fn decode_with_hidden(&mut self) -> Result<()> {
        unsafe { sys::llama_set_embeddings(self.ctx.as_ptr(), true) };
        let decoded = self.decode();
        unsafe { sys::llama_set_embeddings(self.ctx.as_ptr(), false) };
        decoded
    }

    fn snapshot_seq(&self) -> sys::llama_seq_id {
        self.options.n_seq_max as i32 - 1
    }

    /// Sequences a batched pass may decode suffixes into, after sequence 0.
    fn lanes(&self) -> usize {
        let reserved = if self.snapshot_prefixes { 2 } else { 1 };
        (self.options.n_seq_max as usize).saturating_sub(reserved).max(1)
    }

    /// Replace sequence `to` with a copy of sequence `from`.
    fn copy_seq(&self, from: sys::llama_seq_id, to: sys::llama_seq_id) {
        let memory = self.memory();
        unsafe {
            sys::llama_memory_seq_rm(memory, to, -1, -1);
            sys::llama_memory_seq_cp(memory, from, to, -1, -1);
        }
    }

    /// Decode `tokens` into sequence 0 after the `self.cached` tokens it already holds.
    fn extend_prefix(&mut self, tokens: &[sys::llama_token]) -> Result<()> {
        for chunk in tokens.chunks(self.options.n_batch as usize) {
            self.batch.clear();
            let start = self.cached.len();
            for (i, &token) in chunk.iter().enumerate() {
                self.batch.push(token, (start + i) as i32, 0, false);
            }
            self.decode()?;
            self.cached.extend_from_slice(chunk);
        }
        Ok(())
    }

    /// `set_prefix` for a recurrent model: restore the last prefix's snapshot instead of
    /// truncating, or prefill in full.
    fn set_prefix_from_snapshot(&mut self, tokens: &[sys::llama_token]) -> Result<()> {
        let memory = self.memory();
        for seq in 1..self.snapshot_seq() {
            unsafe { sys::llama_memory_seq_rm(memory, seq, -1, -1) };
        }
        if self.cached == tokens {
            return Ok(());
        }
        if self.snapshot == tokens {
            self.copy_seq(self.snapshot_seq(), 0);
            self.cached = tokens.to_vec();
            return Ok(());
        }
        unsafe {
            sys::llama_memory_seq_rm(memory, 0, -1, -1);
            sys::llama_memory_seq_rm(memory, self.snapshot_seq(), -1, -1);
        }
        self.cached.clear();
        self.snapshot.clear();
        self.extend_prefix(tokens)?;
        self.copy_seq(0, self.snapshot_seq());
        self.snapshot = tokens.to_vec();
        Ok(())
    }

    /// Sequence 0 held a prefix and a row's suffix: drop the suffix, or everything if the cache
    /// can't (a recurrent model), for the next row's `set_prefix` to restore.
    fn drop_suffix(&mut self, prefix_len: usize) {
        let memory = self.memory();
        if self.snapshot_prefixes {
            unsafe { sys::llama_memory_seq_rm(memory, 0, -1, -1) };
            self.cached.clear();
        } else if !unsafe { sys::llama_memory_seq_rm(memory, 0, prefix_len as i32, -1) } {
            // The cache cannot drop a tail for this model (recurrent state): start over, and
            // prefill the prefix again only if another row follows.
            unsafe { sys::llama_memory_clear(memory, false) };
            self.cached.clear();
        }
    }

    /// Make sequence 0 hold exactly `tokens`, keeping what matches from the previous call when
    /// `reuse_prefix` is on, and clearing every other sequence.
    fn set_prefix(&mut self, tokens: &[sys::llama_token]) -> Result<()> {
        if self.snapshot_prefixes {
            return self.set_prefix_from_snapshot(tokens);
        }
        let memory = self.memory();
        let keep = if self.options.reuse_prefix {
            self.cached.iter().zip(tokens).take_while(|(a, b)| a == b).count()
        } else {
            0
        };
        let truncated = keep > 0 && unsafe { sys::llama_memory_seq_rm(memory, 0, keep as i32, -1) };
        let keep = if truncated { keep } else { 0 };
        if keep == 0 {
            unsafe { sys::llama_memory_clear(memory, false) };
        }
        for seq in 1..self.options.n_seq_max as i32 {
            unsafe { sys::llama_memory_seq_rm(memory, seq, -1, -1) };
        }
        self.cached = tokens[..keep].to_vec();
        self.extend_prefix(&tokens[keep..])
    }

    fn check_fits(&self, tokens: usize) -> Result<()> {
        if tokens > self.options.n_ctx as usize {
            return Err(backend_error(format!(
                "a row needs {tokens} tokens; the context holds {} (raise n_ctx)",
                self.options.n_ctx
            )));
        }
        Ok(())
    }

    /// Logits after `prefix + suffix` for each suffix decoded alone after the prefix.
    fn one_at_a_time(&mut self, prefix: &[i32], suffixes: &[Vec<i32>], labels: &[Vec<i32>]) -> Result<Vec<RowOut>> {
        self.set_prefix(prefix)?;
        let mut results = Vec::with_capacity(suffixes.len());
        for (row, (suffix, label_ids)) in suffixes.iter().zip(labels).enumerate() {
            if suffix.is_empty() {
                return Err(backend_error("a row has no tokens after its prefix"));
            }
            if row > 0 && self.cached.len() != prefix.len() {
                // The previous row could not be truncated away (a recurrent model): prefill again,
                // or restore the prefix's snapshot.
                self.set_prefix(prefix)?;
            }
            self.check_fits(prefix.len() + suffix.len())?;
            // With hidden states, the last token is decoded on its own (see decode_with_hidden).
            let body = if self.want_hidden { suffix.len() - 1 } else { suffix.len() };
            for chunk_start in (0..body).step_by(self.options.n_batch as usize) {
                let chunk = &suffix[chunk_start..body.min(chunk_start + self.options.n_batch as usize)];
                self.batch.clear();
                for (i, &token) in chunk.iter().enumerate() {
                    let at = chunk_start + i;
                    self.batch.push(token, (prefix.len() + at) as i32, 0, at + 1 == suffix.len());
                }
                self.decode()?;
            }
            if self.want_hidden {
                self.batch.clear();
                self.batch.push(suffix[body], (prefix.len() + body) as i32, 0, true);
                self.decode_with_hidden()?;
            }
            results.push(self.read_row(self.batch.len() - 1, label_ids)?);
            self.drop_suffix(prefix.len());
        }
        Ok(results)
    }

    /// Logits for every suffix, decoded together in passes of up to n_seq_max - 1 sequences.
    fn batched(&mut self, prefix: &[i32], suffixes: &[Vec<i32>], labels: &[Vec<i32>]) -> Result<Vec<RowOut>> {
        self.set_prefix(prefix)?;
        let memory = self.memory();
        let mut results = vec![(Vec::new(), None); suffixes.len()];
        let lanes = self.lanes();
        let mut start = 0;
        while start < suffixes.len() {
            // As many suffixes as fit in the lanes, the context and one batch.
            let (mut end, mut cells) = (start, prefix.len());
            while end < suffixes.len()
                && end - start < lanes
                && cells + suffixes[end].len() <= self.options.n_ctx as usize
                && cells - prefix.len() + suffixes[end].len() <= self.options.n_batch as usize
            {
                cells += suffixes[end].len();
                end += 1;
            }
            if end == start || suffixes[start..end].iter().any(Vec::is_empty) {
                return self.one_at_a_time(prefix, &suffixes[start..], &labels[start..]).map(|rest| {
                    results.truncate(start);
                    results.extend(rest);
                    results
                });
            }
            // With hidden states, each lane's last token goes in a second decode of its own.
            let held_back = usize::from(self.want_hidden);
            self.batch.clear();
            let mut outputs = Vec::with_capacity(end - start);
            for (lane, suffix) in suffixes[start..end].iter().enumerate() {
                let seq = lane as i32 + 1;
                unsafe { sys::llama_memory_seq_cp(memory, 0, seq, -1, -1) };
                for (i, &token) in suffix[..suffix.len() - held_back].iter().enumerate() {
                    self.batch.push(token, (prefix.len() + i) as i32, seq, i + 1 == suffix.len());
                }
                outputs.push(self.batch.len().wrapping_sub(1));
            }
            if self.batch.len() > 0 {
                self.decode()?;
            }
            if self.want_hidden {
                self.batch.clear();
                outputs.clear();
                for (lane, suffix) in suffixes[start..end].iter().enumerate() {
                    let last = suffix.len() - 1;
                    self.batch.push(suffix[last], (prefix.len() + last) as i32, lane as i32 + 1, true);
                    outputs.push(self.batch.len() - 1);
                }
                self.decode_with_hidden()?;
            }
            for (offset, &index) in outputs.iter().enumerate() {
                results[start + offset] = self.read_row(index, &labels[start + offset])?;
            }
            for lane in 0..end - start {
                unsafe { sys::llama_memory_seq_rm(memory, lane as i32 + 1, -1, -1) };
            }
            start = end;
        }
        Ok(results)
    }

    /// Every row's logits (and hidden state when `want_hidden`), in row order.
    fn rows(&mut self, prefix: &str, suffixes: &[String], labels: &[Vec<String>]) -> Result<Vec<RowOut>> {
        let label_ids: Vec<Vec<i32>> = labels
            .iter()
            .map(|row| {
                row.iter()
                    .map(|l| self.label_id(l).ok_or_else(|| backend_error(format!("label {l:?} is not one token"))))
                    .collect::<Result<Vec<_>>>()
            })
            .collect::<Result<_>>()?;
        if self.image_active {
            if self.want_hidden {
                return Err(backend_error("image hidden-state heads are not supported"));
            }
            let vision = self.vision.expect("image requires projector");
            let mut out = Vec::with_capacity(suffixes.len());
            for (suffix, ids) in suffixes.iter().zip(&label_ids) {
                let text = format!("{prefix}{suffix}");
                let code = unsafe {
                    sys::vision::lars_vision_score(
                        vision.as_ptr(),
                        self.ctx.as_ptr(),
                        text.as_ptr().cast(),
                        text.len(),
                        self.options.n_batch as i32,
                    )
                };
                if code != 0 {
                    return Err(self.vision_error());
                }
                let logits = unsafe { sys::llama_get_logits_ith(self.ctx.as_ptr(), -1) };
                if logits.is_null() {
                    return Err(backend_error("vision evaluation returned no logits"));
                }
                let logits = unsafe { std::slice::from_raw_parts(logits, self.n_vocab) };
                out.push((ids.iter().map(|&id| logits[id as usize] as f64).collect(), None));
            }
            return Ok(out);
        }
        let prefix_ids = self.tokenize(prefix, true, true);
        self.check_fits(prefix_ids.len())?;
        let mut clean: Vec<(usize, Vec<i32>)> = Vec::new();
        let mut split: Vec<(usize, Vec<i32>)> = Vec::new();
        for (i, suffix) in suffixes.iter().enumerate() {
            let full = self.tokenize(&format!("{prefix}{suffix}"), true, true);
            if full.starts_with(&prefix_ids) {
                clean.push((i, full[prefix_ids.len()..].to_vec()));
            } else {
                split.push((i, full));
            }
        }
        let mut results = vec![(Vec::new(), None); suffixes.len()];
        if !clean.is_empty() {
            let rows: Vec<Vec<i32>> = clean.iter().map(|(_, s)| s.clone()).collect();
            let row_labels: Vec<Vec<i32>> = clean.iter().map(|(i, _)| label_ids[*i].clone()).collect();
            let out = if self.options.batch_suffixes && rows.len() > 1 {
                self.batched(&prefix_ids, &rows, &row_labels)?
            } else {
                self.one_at_a_time(&prefix_ids, &rows, &row_labels)?
            };
            for ((i, _), row) in clean.iter().zip(out) {
                results[*i] = row;
            }
        }
        // Rows whose tokens merge across the prefix boundary: the full prompt, prefilled alone.
        for (i, full) in split {
            let (head, last) = full.split_at(full.len() - 1);
            let row = self.one_at_a_time(head, &[last.to_vec()], std::slice::from_ref(&label_ids[i]))?;
            results[i] = row.into_iter().next().unwrap();
        }
        Ok(results)
    }
}

impl Backend for LlamaBackend {
    fn set_image(&mut self, bytes: &[u8]) -> Result<()> {
        let vision = self.vision.ok_or_else(|| backend_error("this pack has no vision projector"))?;
        self.cached.clear();
        self.snapshot.clear();
        if unsafe { sys::vision::lars_vision_set_image(vision.as_ptr(), bytes.as_ptr(), bytes.len()) } != 0 {
            return Err(Error::Invalid(self.vision_error().to_string()));
        }
        self.image_active = true;
        Ok(())
    }

    fn clear_image(&mut self) {
        self.image_active = false;
        self.cached.clear();
        self.snapshot.clear();
        unsafe {
            sys::llama_memory_clear(self.memory(), true);
            if let Some(v) = self.vision {
                sys::vision::lars_vision_clear(v.as_ptr());
            }
        }
    }

    fn count_input_tokens(&self, text: &str) -> Result<usize> {
        if !self.image_active || !text.contains("\u{1e}LARS_IMAGE\u{1e}") {
            return Ok(self.count_tokens(text));
        }
        let v = self.vision.expect("vision loaded");
        let mut count = 0;
        if unsafe { sys::vision::lars_vision_count(v.as_ptr(), text.as_ptr().cast(), text.len(), &mut count) } != 0 {
            return Err(self.vision_error());
        }
        Ok(count)
    }
    fn name(&self) -> &str {
        "llamacpp"
    }

    fn model_name(&self) -> &str {
        &self.model_name
    }

    fn template(&self) -> &Template {
        &self.template
    }

    fn is_single_token(&self, label: &str) -> bool {
        self.tokenize(&format!(" {label}"), false, false).len() == 1
    }

    fn count_tokens(&self, text: &str) -> usize {
        self.tokenize(text, false, true).len()
    }

    fn label_logits(&mut self, prefix: &str, suffixes: &[String], labels: &[Vec<String>]) -> Result<Vec<Vec<f64>>> {
        Ok(self.rows(prefix, suffixes, labels)?.into_iter().map(|(logits, _)| logits).collect())
    }

    fn label_logits_hidden(
        &mut self,
        prefix: &str,
        suffixes: &[String],
        labels: &[Vec<String>],
    ) -> Result<Option<LogitsAndHidden>> {
        self.want_hidden = true;
        let rows = self.rows(prefix, suffixes, labels);
        self.want_hidden = false;
        let (logits, hidden): (Vec<_>, Vec<_>) = rows?.into_iter().unzip();
        Ok(Some((logits, hidden.into_iter().map(|h| h.expect("hidden state read")).collect())))
    }
}
