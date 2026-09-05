//! The errors the crate's fallible operations return.

use std::fmt;

/// Something a caller passed in that the pipeline cannot work with.
///
/// Every variant describes a problem with the *input*, not a failure of the algorithm:
/// correspondence mapping itself does not fail, it just maps less of the image (see
/// [`Correspondence::coverage`](crate::Correspondence::coverage)).
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum Error {
    /// The pixel buffer's length does not match `width * height * bytes_per_pixel`.
    BufferLength {
        /// Length implied by the dimensions and pixel format.
        expected: usize,
        /// Length of the buffer that was handed over.
        actual: usize,
    },

    /// A photo has zero width or height.
    EmptyPhoto,

    /// The two photos have different dimensions.
    ///
    /// The mapping is defined over a shared pixel grid, so the inputs have to agree on
    /// one. Rescale or crop before calling.
    SizeMismatch {
        /// Dimensions of the first photo, as `(width, height)`.
        first: (usize, usize),
        /// Dimensions of the second photo.
        second: (usize, usize),
    },

    /// A photo is too small for the feature detector's sampling disc.
    PhotoTooSmall {
        /// The offending photo's dimensions.
        dimensions: (usize, usize),
        /// The smallest width and height the pipeline can work with.
        minimum: usize,
    },

    /// Serialized mapping data could not be decoded.
    Decode(DecodeError),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::BufferLength { expected, actual } => write!(
                f,
                "pixel buffer is {actual} bytes, but the given dimensions need {expected}"
            ),
            Error::EmptyPhoto => f.write_str("photo has zero width or height"),
            Error::SizeMismatch { first, second } => write!(
                f,
                "photos must have the same dimensions, got {}x{} and {}x{}",
                first.0, first.1, second.0, second.1
            ),
            Error::PhotoTooSmall {
                dimensions,
                minimum,
            } => write!(
                f,
                "photo is {}x{}, but correspondence mapping needs at least {minimum}x{minimum}",
                dimensions.0, dimensions.1
            ),
            Error::Decode(inner) => write!(f, "could not decode mapping data: {inner}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Decode(inner) => Some(inner),
            _ => None,
        }
    }
}

impl From<DecodeError> for Error {
    fn from(inner: DecodeError) -> Self {
        Error::Decode(inner)
    }
}

/// Why a serialized mapping could not be read back.
///
/// The encoding is a plain byte format that may well arrive from a file or over a
/// network, so decoding validates rather than trusting: no input, however malformed,
/// should be able to panic the decoder.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum DecodeError {
    /// The data does not start with the format's magic number.
    NotAMapping,

    /// The data is in a format version this build does not understand.
    UnsupportedVersion {
        /// The version found in the header.
        found: u16,
        /// The version this build writes and reads.
        supported: u16,
    },

    /// The data ends before the structure it declares is complete.
    Truncated {
        /// Bytes the header implies.
        expected: usize,
        /// Bytes actually present.
        actual: usize,
    },

    /// The header declares dimensions that are not self-consistent.
    InvalidHeader {
        /// What is wrong with it.
        reason: &'static str,
    },
}

impl fmt::Display for DecodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DecodeError::NotAMapping => f.write_str("missing magic number"),
            DecodeError::UnsupportedVersion { found, supported } => {
                write!(
                    f,
                    "format version {found}, but this build reads version {supported}"
                )
            }
            DecodeError::Truncated { expected, actual } => {
                write!(f, "expected {expected} bytes, found {actual}")
            }
            DecodeError::InvalidHeader { reason } => write!(f, "invalid header: {reason}"),
        }
    }
}

impl std::error::Error for DecodeError {}
