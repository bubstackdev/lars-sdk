//! `Pending`: a queued call, awaited in async code or waited for in blocking code.
//!
//! The runtime replies on the model's worker thread, exactly once per request. The reply fills a
//! slot that both a waker (for `.await`, on any executor) and a condition variable (for `wait`)
//! watch, so nothing here depends on an async runtime.

use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Condvar, Mutex};
use std::task::{Context, Poll, Waker};

use crate::{Error, Lars, Result};

struct Slot<T> {
    state: Mutex<(Option<T>, Option<Waker>)>,
    filled: Condvar,
}

/// A call queued on the model's worker thread: `.await` it, or `wait()` for it.
///
/// The request is queued when the call is made, not when it is first polled. Dropping a
/// `Pending` before its request has started cancels it.
#[must_use = "a Pending does nothing useful unless awaited or waited for"]
pub struct Pending<T> {
    slot: Arc<Slot<Result<T>>>,
    cancel: Option<(Lars, u64)>,
    done: bool,
}

/// The sending half: fills the slot once. Dropped without filling it, it fills it with an error,
/// so a lost reply can't leave a caller waiting forever.
pub(crate) struct Filler<T> {
    slot: Option<Arc<Slot<Result<T>>>>,
}

impl<T> Pending<T> {
    pub(crate) fn new() -> (Pending<T>, Filler<T>) {
        let slot = Arc::new(Slot { state: Mutex::new((None, None)), filled: Condvar::new() });
        (Pending { slot: slot.clone(), cancel: None, done: false }, Filler { slot: Some(slot) })
    }

    /// Already answered (or failed) without queueing anything.
    pub(crate) fn ready(result: Result<T>) -> Pending<T> {
        let (pending, filler) = Pending::new();
        filler.fill(result);
        pending
    }

    /// What dropping it early cancels.
    pub(crate) fn cancels(mut self, lars: &Lars, id: u64) -> Pending<T> {
        self.cancel = Some((lars.clone(), id));
        self
    }

    /// Block this thread until the answer arrives. In async code, `.await` it instead.
    pub fn wait(mut self) -> Result<T> {
        let mut state = self.slot.state.lock().unwrap();
        loop {
            if let Some(result) = state.0.take() {
                self.done = true;
                return result;
            }
            state = self.slot.filled.wait(state).unwrap();
        }
    }
}

impl<T> Future for Pending<T> {
    type Output = Result<T>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Result<T>> {
        let this = self.get_mut();
        assert!(!this.done, "a Pending was polled after it completed");
        let mut state = this.slot.state.lock().unwrap();
        match state.0.take() {
            Some(result) => {
                this.done = true;
                Poll::Ready(result)
            }
            None => {
                state.1 = Some(cx.waker().clone());
                Poll::Pending
            }
        }
    }
}

impl<T> Drop for Pending<T> {
    fn drop(&mut self) {
        let Some((lars, id)) = &self.cancel else { return };
        if !self.done && self.slot.state.lock().unwrap().0.is_none() {
            lars.inner.runtime.cancel(*id);
        }
    }
}

impl<T> Filler<T> {
    pub(crate) fn fill(mut self, result: Result<T>) {
        if let Some(slot) = self.slot.take() {
            let mut state = slot.state.lock().unwrap();
            state.0 = Some(result);
            if let Some(waker) = state.1.take() {
                waker.wake();
            }
            slot.filled.notify_all();
        }
    }
}

impl<T> Drop for Filler<T> {
    fn drop(&mut self) {
        if let Some(slot) = self.slot.take() {
            (Filler { slot: Some(slot) }).fill(Err(Error::internal("the worker thread stopped without replying")));
        }
    }
}
