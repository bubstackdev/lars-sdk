//! The Node addon: `@moedex/lars` loads it and wraps it in the typed API.
//!
//! A thin layer over `lars-runtime`. Promises resolve from the runtime's worker thread through a
//! oneshot channel, so inference never occupies libuv's thread pool or blocks the event loop.
//! Everything crosses as JSON text; a rejected promise's message is the runtime error as JSON,
//! `{"message": ..., "error_type": ...}`, which the TypeScript layer turns into a `LarsError`.

use std::sync::{Arc, Mutex};

use lars_runtime::{Error, Lars, OpenOptions};
use napi::Status;
use napi_derive::napi;
use tokio::sync::oneshot;

fn js_error(error: Error) -> napi::Error {
    napi::Error::new(Status::GenericFailure, error.to_json().to_string())
}

/// An open pack; see `lars_runtime::Lars`.
#[napi]
pub struct NativeLars {
    inner: Mutex<Option<Arc<Lars>>>,
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
    Ok(NativeLars { inner: Mutex::new(Some(Arc::new(lars))) })
}

fn serde_json_value(text: &str) -> napi::Result<lars_runtime::pack::Json> {
    lars_runtime::pack::parse_json(text).map_err(js_error)
}

#[napi]
impl NativeLars {
    /// Answer a System One request (JSON); resolves with the response JSON.
    #[napi]
    pub async fn evaluate(&self, request_json: String) -> napi::Result<String> {
        let lars = self.lars()?;
        let (sender, receiver) = oneshot::channel();
        lars.evaluate_async(
            &request_json,
            Box::new(move |_, result| {
                let _ = sender.send(result);
            }),
        );
        wait(receiver).await
    }

    /// Fit calibrator `name` on labeled examples (JSONL); resolves with
    /// `{calibrator, examples, accuracy, certified}`.
    #[napi]
    pub async fn calibrate(
        &self,
        name: String,
        examples_jsonl: String,
        max_error: Option<f64>,
    ) -> napi::Result<String> {
        let lars = self.lars()?;
        let (sender, receiver) = oneshot::channel();
        lars.calibrate_async(
            &name,
            &examples_jsonl,
            max_error,
            Box::new(move |_, result| {
                let _ = sender.send(result);
            }),
        )
        .map_err(js_error)?;
        wait(receiver).await
    }

    /// Load a saved calibrator (JSON, fitted on this pack) under `name`.
    #[napi]
    pub fn set_calibrator(&self, name: String, calibrator_json: String) -> napi::Result<()> {
        self.lars()?.set_calibrator(&name, &calibrator_json).map_err(js_error)
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

/// The SDK version.
#[napi]
pub fn version() -> String {
    lars_runtime::VERSION.to_string()
}
