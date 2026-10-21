//! Owned experimental image inputs. Decode and inference happen on the model worker.
use crate::{Content, Error, Result};
use std::io::Read;
use std::path::Path;
use std::sync::Arc;

/// A snapshot of one JPEG/PNG and optional text or JSON context.
#[derive(Clone)]
pub struct ImageInput {
    pub(crate) bytes: Arc<[u8]>,
    pub(crate) context: Content,
}

impl std::fmt::Debug for ImageInput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ImageInput").field("encoded_bytes", &self.bytes.len()).finish_non_exhaustive()
    }
}

impl ImageInput {
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        if bytes.is_empty() || bytes.len() > 20 * 1024 * 1024 {
            return Err(Error::invalid("image must contain 1 byte to 20 MiB"));
        }
        Ok(Self { bytes: bytes.into(), context: Content::Null })
    }

    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let file = std::fs::File::open(path).map_err(|e| Error::invalid(e.to_string()))?;
        let mut bytes = Vec::new();
        file.take(20 * 1024 * 1024 + 1).read_to_end(&mut bytes).map_err(|e| Error::invalid(e.to_string()))?;
        Self::from_bytes(&bytes)
    }

    pub fn with_context(mut self, context: impl Into<Content>) -> Self {
        self.context = context.into();
        self
    }
}
