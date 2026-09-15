//! Multi-view 3D reconstruction on top of [`pixelmap`].
//!
//! Given three or more photos of a static scene taken with one camera, this crate computes
//! dense correspondence between every pair of them with pixelmap, recovers where each
//! photo was taken from, and fuses the result into a single mesh. `pixelmap_model_3d`
//! lifts one mapping between two photos into a surface; this crate registers every view
//! into one frame so that their surfaces agree.
//!
//! # Status
//!
//! Under construction. Stages so far:
//!
//! 1. [`pairs`]: pixelmap over every pair of photos, and the check that they link up.
//! 2. [`twoview`]: the relative pose of each pair, and triage of weak pairs.
//!
//! Tracks, registration, bundle adjustment, dense depth, fusion and texturing follow.
//! [`synthetic`] provides scenes with known geometry to test each stage against.
//!
//! # Design rules
//!
//! - **No I/O, no platform.** Photos come in as RGBA buffers and a mesh goes out.
//!   Decoding, EXIF and resizing are the caller's job, which keeps the crate buildable for
//!   `wasm32-unknown-unknown`.
//! - **One type per coordinate space.** See [`types`].
//! - **Deterministic.** Every random choice draws from a generator derived from a single
//!   seed, as pixelmap's own solver does, so any run can be replayed exactly.
//! - **Fail loudly.** Each stage has exit criteria. A run that cannot produce a correct
//!   model returns an [`Error`] naming the stage and the reason, rather than a
//!   plausible-looking wrong model.
//!
//! # Feature flags
//!
//! - **`parallel`** *(default)* — passed through to pixelmap. Turn it off for wasm.

#![warn(missing_docs)]

pub mod calib;
pub mod error;
pub mod input;
pub mod lookup;
pub mod pairs;
pub mod pose;
pub mod progress;
pub mod rng;
pub mod synthetic;
pub mod triangulate;
pub mod twoview;
pub mod types;

pub use calib::{FocalSource, Intrinsics};
pub use error::Error;
pub use input::MIN_VIEWS;
pub use lookup::PairLookup;
pub use pairs::PairGraph;
pub use pose::Pose;
pub use progress::{Event, Flow, Stage};
pub use types::{Norm, PairId, PhotoPx, ViewId, World};
