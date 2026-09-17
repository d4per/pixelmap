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
    /// Too few points could be followed across three or more views.
    TooFewTracks {
        /// How many tracks reach three or more views.
        found: usize,
        /// How many are needed.
        required: usize,
    },
    /// No pair of views is usable as the starting point of a registration.
    NoUsablePair {
        /// Each pair that was not usable, and why.
        reasons: Vec<(PairId, String)>,
    },
    /// Too few views could be placed in one frame.
    RegistrationFailed {
        /// The views that were.
        registered: Vec<ViewId>,
        /// How many are needed.
        minimum: usize,
        /// Each view that could not be placed, and why.
        left_out: Vec<(ViewId, String)>,
    },
    /// Bundle adjustment did not reach an acceptable fit.
    BundleAdjustment {
        /// The median reprojection error before, in photo pixels.
        initial_median_px: f64,
        /// The median reprojection error after, in photo pixels.
        median_px: f64,
        /// The largest acceptable median, in photo pixels.
        required_px: f64,
    },
    /// Too few depth-map samples survived to fuse a surface from.
    InsufficientDepth {
        /// The fraction of samples that have a depth.
        valid_fraction: f64,
        /// The fraction needed.
        required: f64,
    },
    /// Fusion produced no surface.
    EmptyMesh,
    /// The fused surface falls apart into pieces, none of which holds much of it.
    FragmentedMesh {
        /// The share of triangles in the largest connected piece.
        largest_component: f64,
        /// The share needed.
        required: f64,
    },
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
            Error::TooFewTracks { .. } => Stage::Tracks,
            Error::NoUsablePair { .. } | Error::RegistrationFailed { .. } => Stage::Registration,
            Error::BundleAdjustment { .. } => Stage::BundleAdjustment,
            Error::InsufficientDepth { .. } => Stage::Depth,
            Error::EmptyMesh | Error::FragmentedMesh { .. } => Stage::Fusion,
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
            Error::TooFewTracks { found, required } => write!(
                f,
                "only {found} points could be followed across three or more photos (need {required}); \
                 the photos share too little of the scene"
            ),
            Error::NoUsablePair { reasons } => {
                f.write_str("no pair of photos is fit to start the reconstruction from")?;
                for (pair, reason) in reasons {
                    write!(f, "; {pair}: {reason}")?;
                }
                Ok(())
            }
            Error::RegistrationFailed {
                registered,
                minimum,
                left_out,
            } => {
                write!(
                    f,
                    "only {} view(s) could be placed in a common frame (need {minimum})",
                    registered.len()
                )?;
                for (view, reason) in left_out {
                    write!(f, "; {view} was left out: {reason}")?;
                }
                Ok(())
            }
            Error::BundleAdjustment {
                initial_median_px,
                median_px,
                required_px,
            } => {
                if median_px > initial_median_px {
                    write!(
                        f,
                        "bundle adjustment made the fit worse ({initial_median_px:.2} px → {median_px:.2} px median reprojection error); \
                         an earlier stage probably placed a camera wrongly"
                    )
                } else {
                    write!(
                        f,
                        "the cameras and points do not fit the photos: {median_px:.2} px median reprojection error after bundle adjustment \
                         (at most {required_px:.2} px is acceptable); an earlier stage probably placed a camera wrongly"
                    )
                }
            }
            Error::InsufficientDepth {
                valid_fraction,
                required,
            } => write!(
                f,
                "only {:.0}% of the photos could be given a depth that the views agree on (need {:.0}%); \
                 the scene may be too textureless, shiny or distant for its baseline",
                valid_fraction * 100.0,
                required * 100.0
            ),
            Error::EmptyMesh => f.write_str("fusing the depth maps produced no surface"),
            Error::FragmentedMesh {
                largest_component,
                required,
            } => write!(
                f,
                "the surface falls apart into fragments; the largest holds only {:.0}% of it (need {:.0}%)",
                largest_component * 100.0,
                required * 100.0
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
