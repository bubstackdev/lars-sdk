//! Runtime errors, in the `{message, error_type}` shape every binding receives.

use std::fmt;

use serde_json::{Value, json};

#[derive(Debug, Clone, PartialEq)]
pub struct Error {
    /// `invalid_request`, `pack_error`, `cancelled`, `api_error` or `internal_error`.
    pub error_type: &'static str,
    pub message: String,
}

impl Error {
    pub fn invalid(message: impl Into<String>) -> Error {
        Error { error_type: "invalid_request", message: message.into() }
    }

    pub fn pack(message: impl Into<String>) -> Error {
        Error { error_type: "pack_error", message: message.into() }
    }

    pub fn cancelled() -> Error {
        Error { error_type: "cancelled", message: "the request was cancelled before it ran".into() }
    }

    pub fn internal(message: impl Into<String>) -> Error {
        Error { error_type: "internal_error", message: message.into() }
    }

    pub fn to_json(&self) -> Value {
        json!({"message": self.message, "error_type": self.error_type})
    }
}

impl From<lars_core::Error> for Error {
    fn from(error: lars_core::Error) -> Error {
        Error { error_type: error.error_type(), message: error.message().to_string() }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.error_type, self.message)
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;
