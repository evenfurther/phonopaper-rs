//! Error types for the `phonopaper-rs` library.

use thiserror::Error;

/// The unified error type for all operations in this library.
#[derive(Debug, Error)]
pub enum PhonoPaperError {
    /// The `PhonoPaper` marker bands could not be found in the image.
    ///
    /// The `&'static str` carries a short human-readable reason explaining
    /// why detection failed (e.g. `"fewer than 3 dark runs found"`).
    #[error("PhonoPaper marker bands not found in image: {0}")]
    MarkerNotFound(&'static str),

    /// The image or audio data does not conform to the expected format.
    #[error("Invalid format: {0}")]
    InvalidFormat(String),

    /// An error occurred while loading or saving an image.
    #[error("Image error: {0}")]
    ImageError(#[from] image::ImageError),

    /// An error occurred while reading or writing a WAV file.
    #[error("Audio error: {0}")]
    AudioError(#[from] hound::Error),

    /// An I/O error occurred.
    #[error("I/O error: {0}")]
    IoError(#[from] std::io::Error),
}

/// Convenience type alias for `Result<T, PhonoPaperError>`.
pub type Result<T> = std::result::Result<T, PhonoPaperError>;
