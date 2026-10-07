//! The C ABI over `lars-runtime`: what the C#, Go and other bindings call. Header: `include/lars.h`.
//!
//! Everything crosses as UTF-8 JSON text: requests and responses in the System One shape, options,
//! calibrators, status, and errors as `{"message": ..., "error_type": ...}`.
//!
//! - Strings this library returns are owned by the caller; free them with `lars_string_free`.
//! - On failure a function returns NULL (or -1, or 0 for an id) and, when `error_json` is not
//!   NULL, sets `*error_json` to an error the caller frees.
//! - Callbacks run on the engine's worker thread, or on the calling thread before the call
//!   returns when a request does not parse (its id is then 0). The strings they receive are
//!   valid only during the call. Every accepted async call gets exactly one callback, even if it
//!   is cancelled or its engine is closed first; keep `user_data` alive until then.
//! - Engines are thread-safe: call any function from any thread, concurrently.
//! - No panic crosses this boundary; one becomes an `internal_error`.

use std::ffi::{CStr, CString, c_char, c_void};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::ptr;

use lars_runtime::{Error, Lars, OpenOptions, Result};

/// An open pack. Create with `lars_open`, release with `lars_close`.
pub struct LarsEngine(Lars);

/// Called with a request's id and either its result or its error (the other is NULL).
pub type LarsCallback = Option<
    unsafe extern "C" fn(
        user_data: *mut c_void,
        request_id: u64,
        result_json: *const c_char,
        error_json: *const c_char,
    ),
>;

struct UserData(*mut c_void);
// The caller promises user_data may be used from the worker thread (documented above).
unsafe impl Send for UserData {}

fn owned(text: String) -> *mut c_char {
    CString::new(text)
        .unwrap_or_else(|e| CString::new(e.into_vec().into_iter().filter(|b| *b != 0).collect::<Vec<_>>()).unwrap())
        .into_raw()
}

unsafe fn set_error(error_json: *mut *mut c_char, error: &Error) {
    if !error_json.is_null() {
        unsafe { *error_json = owned(error.to_json().to_string()) };
    }
}

unsafe fn text<'a>(pointer: *const c_char, what: &str) -> Result<&'a str> {
    if pointer.is_null() {
        return Err(Error::invalid(format!("{what} is NULL")));
    }
    unsafe { CStr::from_ptr(pointer) }.to_str().map_err(|_| Error::invalid(format!("{what} is not UTF-8")))
}

unsafe fn handle<'a>(pointer: *const LarsEngine) -> Result<&'a Lars> {
    if pointer.is_null() { Err(Error::invalid("engine is NULL")) } else { Ok(unsafe { &(*pointer).0 }) }
}

/// Run `f`, turning a panic into an internal error.
fn guarded<T>(f: impl FnOnce() -> Result<T>) -> Result<T> {
    catch_unwind(AssertUnwindSafe(f)).unwrap_or_else(|_| Err(Error::internal("panic in lars-ffi")))
}

/// Return `f`'s text, or NULL with `*error_json` set.
unsafe fn string_result(error_json: *mut *mut c_char, f: impl FnOnce() -> Result<String>) -> *mut c_char {
    match guarded(f) {
        Ok(text) => owned(text),
        Err(error) => {
            unsafe { set_error(error_json, &error) };
            ptr::null_mut()
        }
    }
}

fn reply_to(callback: LarsCallback, user_data: *mut c_void) -> lars_runtime::Reply {
    let user = UserData(user_data);
    Box::new(move |id, result: Result<String>| {
        let user = user;
        let Some(callback) = callback else { return };
        let (ok, err) = match result {
            Ok(text) => (Some(CString::new(text).unwrap_or_default()), None),
            Err(error) => (None, Some(CString::new(error.to_json().to_string()).unwrap_or_default())),
        };
        let as_ptr = |s: &Option<CString>| s.as_ref().map_or(ptr::null(), |s| s.as_ptr());
        unsafe { callback(user.0, id, as_ptr(&ok), as_ptr(&err)) };
    })
}

/// The library version, e.g. "0.1.0" (static; do not free).
#[unsafe(no_mangle)]
pub extern "C" fn lars_version() -> *const c_char {
    concat!(env!("CARGO_PKG_VERSION"), "\0").as_ptr() as *const c_char
}

/// Open a pack (a directory, or `name[@version]` in the pack cache) and load its model.
/// `options_json` may be NULL. Blocks while the model loads (well under a second to a few).
///
/// # Safety
/// `pack` must be a NUL-terminated string; `options_json` NULL or one; `error_json` NULL or writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lars_open(
    pack: *const c_char,
    options_json: *const c_char,
    error_json: *mut *mut c_char,
) -> *mut LarsEngine {
    let opened = guarded(|| {
        let reference = unsafe { text(pack, "pack") }?;
        let options = if options_json.is_null() {
            OpenOptions::default()
        } else {
            let raw = unsafe { text(options_json, "options_json") }?;
            let value = serde_json::from_str(raw).map_err(|e| Error::invalid(format!("options: {e}")))?;
            OpenOptions::from_json(&value)?
        };
        Lars::open(reference, options)
    });
    match opened {
        Ok(lars) => Box::into_raw(Box::new(LarsEngine(lars))),
        Err(error) => {
            unsafe { set_error(error_json, &error) };
            ptr::null_mut()
        }
    }
}

/// Download a pack ahead of time and return its directory (free it with `lars_string_free`): a
/// bare name gets the newest published version if it isn't installed, `name@version` that
/// version. Blocks while it downloads. NULL on failure, with `*error_json` set.
///
/// # Safety
/// `pack` must be a NUL-terminated string; `error_json` NULL or writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lars_pull(pack: *const c_char, error_json: *mut *mut c_char) -> *mut c_char {
    unsafe { string_result(error_json, || Ok(lars_runtime::pull(text(pack, "pack")?)?.to_string_lossy().into_owned())) }
}

/// Answer a System One request. Blocks until it has run.
///
/// # Safety
/// `engine` from `lars_open`; strings NUL-terminated; `error_json` NULL or writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lars_evaluate(
    engine: *const LarsEngine,
    request_json: *const c_char,
    error_json: *mut *mut c_char,
) -> *mut c_char {
    unsafe { string_result(error_json, || handle(engine)?.evaluate(text(request_json, "request_json")?)) }
}

/// Queue a request; `callback` gets its result. Returns its id (for `lars_cancel`), or 0 when
/// the request did not parse, in which case the callback has already run with the error.
///
/// # Safety
/// As `lars_evaluate`; `user_data` must stay valid until the callback has run.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lars_evaluate_async(
    engine: *const LarsEngine,
    request_json: *const c_char,
    callback: LarsCallback,
    user_data: *mut c_void,
) -> u64 {
    let submitted = guarded(|| {
        let lars = unsafe { handle(engine) }?;
        let request = unsafe { text(request_json, "request_json") }?;
        Ok(lars.evaluate_async(request, reply_to(callback, user_data)))
    });
    submitted.unwrap_or_else(|error| {
        reply_to(callback, user_data)(0, Err(error));
        0
    })
}

/// Fit a calibrator `name` on labeled examples (JSONL: state, question, label) and use it for
/// questions whose request names it. With `max_error` > 0, also certify an acceptance threshold
/// for that error budget. Returns `{calibrator, examples, accuracy, certified}`.
///
/// # Safety
/// As `lars_evaluate`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lars_calibrate(
    engine: *const LarsEngine,
    name: *const c_char,
    examples_jsonl: *const c_char,
    max_error: f64,
    error_json: *mut *mut c_char,
) -> *mut c_char {
    unsafe {
        string_result(error_json, || {
            let budget = (max_error > 0.0).then_some(max_error);
            handle(engine)?.calibrate(text(name, "name")?, text(examples_jsonl, "examples_jsonl")?, budget)
        })
    }
}

/// `lars_calibrate`, queued; the callback gets its result. Returns its id, or 0 with the
/// callback already run when the arguments are invalid.
///
/// # Safety
/// As `lars_evaluate_async`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lars_calibrate_async(
    engine: *const LarsEngine,
    name: *const c_char,
    examples_jsonl: *const c_char,
    max_error: f64,
    callback: LarsCallback,
    user_data: *mut c_void,
) -> u64 {
    let submitted = guarded(|| {
        let lars = unsafe { handle(engine) }?;
        let (name, examples) = unsafe { (text(name, "name")?, text(examples_jsonl, "examples_jsonl")?) };
        lars.calibrate_async(name, examples, (max_error > 0.0).then_some(max_error), reply_to(callback, user_data))
    });
    submitted.unwrap_or_else(|error| {
        reply_to(callback, user_data)(0, Err(error));
        0
    })
}

/// Load a calibrator saved from `lars_calibrate` (or `lars_calibrators`) under `name`. It must
/// have been fitted on this pack. Returns 0, or -1 with `*error_json` set.
///
/// # Safety
/// As `lars_evaluate`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lars_set_calibrator(
    engine: *const LarsEngine,
    name: *const c_char,
    calibrator_json: *const c_char,
    error_json: *mut *mut c_char,
) -> i32 {
    let result = guarded(|| unsafe {
        handle(engine)?.set_calibrator(text(name, "name")?, text(calibrator_json, "calibrator_json")?)
    });
    match result {
        Ok(()) => 0,
        Err(error) => {
            unsafe { set_error(error_json, &error) };
            -1
        }
    }
}

/// Load several calibrators at once, `{name: calibrator}` as `lars_calibrators` returns them. Each
/// must have been fitted on this pack; if any is not, none is loaded. Returns 0, or -1 with
/// `*error_json` set.
///
/// # Safety
/// As `lars_evaluate`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lars_set_calibrators(
    engine: *const LarsEngine,
    calibrators_json: *const c_char,
    error_json: *mut *mut c_char,
) -> i32 {
    let result = guarded(|| unsafe { handle(engine)?.set_calibrators(text(calibrators_json, "calibrators_json")?) });
    match result {
        Ok(()) => 0,
        Err(error) => {
            unsafe { set_error(error_json, &error) };
            -1
        }
    }
}

/// This engine's named calibrators as one JSON object, `{name: calibrator}`, for saving.
///
/// # Safety
/// `engine` from `lars_open`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lars_calibrators(engine: *const LarsEngine) -> *mut c_char {
    unsafe { string_result(ptr::null_mut(), || Ok(handle(engine)?.calibrators_json())) }
}

/// Status JSON: pack, model, loaded or not, queue depth, calibrators, versions.
///
/// # Safety
/// `engine` from `lars_open`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lars_status(engine: *const LarsEngine) -> *mut c_char {
    unsafe { string_result(ptr::null_mut(), || Ok(handle(engine)?.status())) }
}

/// Cancel a queued request: its callback gets a `cancelled` error instead of a result. Returns
/// true if it had not finished; one that had already started still finishes normally.
///
/// # Safety
/// `engine` from `lars_open`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lars_cancel(engine: *const LarsEngine, request_id: u64) -> bool {
    guarded(|| Ok(unsafe { handle(engine) }?.cancel(request_id))).unwrap_or(false)
}

/// Free a string this library returned. NULL is ignored.
///
/// # Safety
/// `text` must come from this library and not be freed twice.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lars_string_free(text: *mut c_char) {
    if !text.is_null() {
        drop(unsafe { CString::from_raw(text) });
    }
}

/// Close an engine. Its queued requests are cancelled (their callbacks still run); the model
/// stays loaded while another engine on the same pack uses it. NULL is ignored.
///
/// # Safety
/// `engine` from `lars_open`, not used again afterwards.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lars_close(engine: *mut LarsEngine) {
    if !engine.is_null() {
        let _ = catch_unwind(AssertUnwindSafe(|| drop(unsafe { Box::from_raw(engine) })));
    }
}
