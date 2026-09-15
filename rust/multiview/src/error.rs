//! Everything that can stop a reconstruction.
//!
//! Each variant says which [`Stage`] failed and why, in terms a user can act on: a failed
//! run should tell someone how to take better photos, not just that something went wrong.

use std::fmt;

use crate::progress::Stage;
use crate::types::{PairId, ViewId};

/// Why a reconstruction could not continue.
#[derive(Debug)]
#[non_exhaustive]
pub enum Error {
    /// Fewer photos than a reconstruction needs.
    TooFewPhotos {
        /// How many were passed in.
        found: usize,
        /// How many are needed.
        minimum: usize,
    },
    /// A photo's dimensions differ from the first photo's. pixelmap only maps photos of
    /// identical size.
    SizeMismatch {
        /// The offending photo.
        view: ViewId,
        /// The first photo's dimensions.
        expected: (usize, usize),
        /// This photo's dimensions.
        found: (usize, usize),
    },
    /// A photo is smaller than [`pixelmap::MIN_DIMENSION`] on at least one side.
    PhotoTooSmall {
        /// The offending photo.
        view: ViewId,
        /// Its dimensions.
        dimensions: (usize, usize),
        /// The smallest usable width and height.
        minimum: usize,
    },
    /// The intrinsics have a non-finite parameter or a focal length that is not positive.
    InvalidIntrinsics,
    /// The progress callback asked the run to stop.
    Cancelled {
        /// The stage that was running.
        stage: Stage,
    },
    /// pixelmap rejected a pair of photos.
    Correspondence {
        /// The pair.
        pair: PairId,
        /// pixelmap's reason.
        source: pixelmap::Error,
    },
    /// Too few views are linked by well-mapped pairs to reconstruct from.
    DisconnectedViews {
        /// The largest set of views that are linked.
        connected: Vec<ViewId>,
        /// How many views are needed.
        minimum: usize,
        /// The coverage a pair needed to count as a link.
        min_coverage: f32,
    },
}

impl Error {
    /// The stage that failed.
    pub fn stage(&self) -> Stage {
        match self {
            Error::TooFewPhotos { .. }
            | Error::SizeMismatch { .. }
            | Error::PhotoTooSmall { .. }
            | Error::InvalidIntrinsics => Stage::Input,
            Error::Cancelled { stage } => *stage,
            Error::Correspondence { .. } | Error::DisconnectedViews { .. } => Stage::Pairs,
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::TooFewPhotos { found, minimum } => write!(
                f,
                "{found} photo(s) given; a reconstruction needs at least {minimum} views of the same scene"
            ),
            Error::SizeMismatch {
                view,
                expected: (ew, eh),
                found: (fw, fh),
            } => write!(
                f,
                "{view} is {fw}×{fh} but the first photo is {ew}×{eh}; all photos must come from one camera at one resolution"
            ),
            Error::PhotoTooSmall {
                view,
                dimensions: (w, h),
                minimum,
            } => write!(
                f,
                "{view} is {w}×{h}; both sides must be at least {minimum} pixels"
            ),
            Error::InvalidIntrinsics => f.write_str(
                "camera intrinsics must be finite, with a positive focal length",
            ),
            Error::Cancelled { stage } => write!(f, "cancelled during {stage}"),
            Error::Correspondence { pair, source } => {
                write!(f, "could not map {pair}: {source}")
            }
            Error::DisconnectedViews {
                connected,
                minimum,
                min_coverage,
            } => write!(
                f,
                "only {} view(s) are linked by pairs with at least {:.0}% coverage (need {minimum}); \
                 the photos share too little of the scene, so take them closer together",
                connected.len(),
                min_coverage * 100.0
            ),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Correspondence { source, .. } => Some(source),
            _ => None,
        }
    }
}
