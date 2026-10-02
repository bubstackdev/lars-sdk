//! Errors, in the `{message, error_type}` shape of the System One API.

use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub enum Error {
    /// The request is not a valid System One request.
    Schema(String),
    /// The request would exceed a row or input-token budget.
    Budget(String),
    /// The request names a model this engine does not serve.
    UnknownModel(String),
    /// Any other request the engine refuses, such as an unknown calibrator.
    Invalid(String),
    /// The model failed.
    Backend(String),
}

impl Error {
    pub fn error_type(&self) -> &'static str {
        match self {
            Error::Backend(_) => "api_error",
            _ => "invalid_request",
        }
    }

    pub fn message(&self) -> &str {
        match self {
            Error::Schema(m) | Error::Budget(m) | Error::UnknownModel(m) | Error::Invalid(m) | Error::Backend(m) => m,
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.message())
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;
