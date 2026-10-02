//! LARS typed decisions, shared by every language binding.
//!
//! A request (a state plus typed questions) is rendered into prompt rows, scored by a backend
//! that reads next-token logits over option labels, and reduced to calibrated answers. The
//! rendering, reduction and calibration reproduce the Python reference engine exactly, so
//! calibrators fitted by either serve correctly in both.

pub mod backend;
pub mod calibration;
pub mod engine;
pub mod error;
pub mod eval;
pub mod labels;
pub mod primitives;
pub mod py;
pub mod pyrandom;
pub mod render;
pub mod schema;

pub use backend::{Backend, MockBackend};
pub use calibration::Calibrator;
pub use engine::{Engine, EngineOptions, Response};
pub use error::{Error, Result};
pub use schema::Request;
