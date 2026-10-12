//! The Node addon: `@moedex/lars` loads it and wraps it in the typed API.
//!
//! A thin layer over `lars-runtime`. Requests are queued on the calling (JS) thread, so a `cancel`
//! right after a call finds it; promises resolve from the runtime's worker thread through a
//! oneshot channel, so inference never occupies libuv's thread pool or blocks the event loop.
//! Everything crosses as JSON text; a rejected promise's message is the runtime error as JSON,
//! `{"message": ..., "error_type": ...}`, which the TypeScript layer turns into a `LarsError`.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use lars_runtime::{Error, Lars, OpenOptions};
use napi::bindgen_prelude::PromiseRaw;
use napi::{Env, Status};
use napi_derive::napi;
use tokio::sync::oneshot;

fn js_error(error: Error) -> napi::Error {
    napi::Error::new(Status::GenericFailure, error.to_json().to_string())
}

/// An open pack; see `lars_runtime::Lars`.
#[napi]
pub struct NativeLars {
    inner: Mutex<Option<Arc<Lars>>>,
    /// Cancellation tokens the TypeScript layer chose, to the runtime's ids of their requests,
    /// while those are queued. `None` marks a request that replied before its id was recorded.
    tokens: Arc<Mutex<HashMap<u32, Option<u64>>>>,
}

impl NativeLars {
    fn lars(&self) -> napi::Result<Arc<Lars>> {
        self.inner.lock().unwrap().clone().ok_or_else(|| js_error(Error::invalid("this pack has been closed")))
    }
}

async fn wait(receiver: oneshot::Receiver<lars_runtime::Result<String>>) -> napi::Result<String> {
    receiver
        .await
        .unwrap_or_else(|_| Err(Error::internal("the worker thread stopped without replying")))
        .map_err(js_error)
}

/// Open a pack (a directory, or `name[@version]` in the pack cache, downloaded on first use) with
/// options as JSON. Loading runs off the event loop.
#[napi]
pub async fn open(pack: String, options_json: Option<String>) -> napi::Result<NativeLars> {
    let options = match options_json {
        None => OpenOptions::default(),
        Some(text) => {
            let value = serde_json_value(&text)?;
            OpenOptions::from_json(&value).map_err(js_error)?
        }
    };
    let lars = tokio::task::spawn_blocking(move || Lars::open(&pack, options))
        .await
        .map_err(|e| js_error(Error::internal(format!("open: {e}"))))?
        .map_err(js_error)?;
    Ok(NativeLars { inner: Mutex::new(Some(Arc::new(lars))), tokens: Arc::default() })
}

fn serde_json_value(text: &str) -> napi::Result<lars_runtime::pack::Json> {
    lars_runtime::pack::parse_json(text).map_err(js_error)
}

#[napi]
impl NativeLars {
    /// The reply for a request submitted under `token`: forgets the token once it has replied.
    fn reply(&self, token: Option<u32>, sender: oneshot::Sender<lars_runtime::Result<String>>) -> lars_runtime::Reply {
        let tokens = self.tokens.clone();
        Box::new(move |_, result| {
            if let Some(token) = token {
                let mut tokens = tokens.lock().unwrap();
                if tokens.remove(&token).is_none() {
                    tokens.insert(token, None);
                }
            }
            let _ = sender.send(result);
        })
    }

    /// Record a submitted request's id under its token, unless it has already replied.
    fn remember(&self, token: Option<u32>, id: u64) {
        let Some(token) = token else { return };
        let mut tokens = self.tokens.lock().unwrap();
        if tokens.remove(&token).is_none() {
            tokens.insert(token, Some(id));
        }
    }

    /// Answer a System One request (JSON); resolves with the response JSON. `cancel_token`, any
    /// number the caller picks, lets `cancel` cancel it while it is queued. The request is queued
    /// before this returns (an `async fn` would queue it later, on the tokio runtime, and miss a
    /// `cancel` made in between).
    #[napi(ts_return_type = "Promise<string>")]
    pub fn evaluate<'env>(
        &self,
        env: &'env Env,
        request_json: String,
        cancel_token: Option<u32>,
    ) -> napi::Result<PromiseRaw<'env, String>> {
        let lars = self.lars()?;
        let (sender, receiver) = oneshot::channel();
        let id = lars.evaluate_async(&request_json, self.reply(cancel_token, sender));
        self.remember(cancel_token, id);
        env.spawn_future(wait(receiver))
    }

    /// Cancel the request submitted under `token` if it hasn't started: it rejects as
    /// `cancelled`. True if there was one to cancel.
    #[napi]
    pub fn cancel(&self, token: u32) -> bool {
        let Some(Some(id)) = self.tokens.lock().unwrap().get(&token).copied() else { return false };
        self.lars().map(|lars| lars.cancel(id)).unwrap_or(false)
    }

    /// Fit calibrator `name` on labeled examples (JSONL); resolves with
    /// `{calibrator, examples, accuracy, certified}`. Queued before this returns, like `evaluate`.
    #[napi(ts_return_type = "Promise<string>")]
    pub fn calibrate<'env>(
        &self,
        env: &'env Env,
        name: String,
        examples_jsonl: String,
        max_error: Option<f64>,
        cancel_token: Option<u32>,
    ) -> napi::Result<PromiseRaw<'env, String>> {
        let lars = self.lars()?;
        let (sender, receiver) = oneshot::channel();
        let id = lars
            .calibrate_async(&name, &examples_jsonl, max_error, self.reply(cancel_token, sender))
            .map_err(js_error)?;
        self.remember(cancel_token, id);
        env.spawn_future(wait(receiver))
    }

    /// Load a saved calibrator (JSON, fitted on this pack) under `name`.
    #[napi]
    pub fn set_calibrator(&self, name: String, calibrator_json: String) -> napi::Result<()> {
        self.lars()?.set_calibrator(&name, &calibrator_json).map_err(js_error)
    }

    /// Load several calibrators (`{name: calibrator}`, JSON) at once; if any is not for this
    /// pack, none is loaded.
    #[napi]
    pub fn set_calibrators(&self, calibrators_json: String) -> napi::Result<()> {
        self.lars()?.set_calibrators(&calibrators_json).map_err(js_error)
    }

    /// Every named calibrator, `{name: calibrator}`, as JSON.
    #[napi]
    pub fn calibrators(&self) -> napi::Result<String> {
        Ok(self.lars()?.calibrators_json())
    }

    #[napi]
    pub fn status(&self) -> napi::Result<String> {
        Ok(self.lars()?.status())
    }

    /// The pack's manifest (`pack.json`) as JSON.
    #[napi]
    pub fn manifest(&self) -> napi::Result<String> {
        Ok(self.lars()?.pack().manifest.to_string())
    }

    /// Release this handle. Queued requests are cancelled; the model stays loaded while another
    /// handle uses it. Further calls reject.
    #[napi]
    pub fn close(&self) {
        self.inner.lock().unwrap().take();
    }
}

/// Download a pack ahead of time (see `lars_runtime::pull`); resolves with its directory. Runs
/// off the event loop.
#[napi]
pub async fn pull(pack: String) -> napi::Result<String> {
    tokio::task::spawn_blocking(move || lars_runtime::pull(&pack))
        .await
        .map_err(|e| js_error(Error::internal(format!("pull: {e}"))))?
        .map(|dir| dir.to_string_lossy().into_owned())
        .map_err(js_error)
}

/// The SDK version.
#[napi]
pub fn version() -> String {
    lars_runtime::VERSION.to_string()
}
