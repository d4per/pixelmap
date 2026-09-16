//! Multi-view 3D reconstruction on top of [`pixelmap`].
//!
//! Given three or more photos of a static scene taken with one camera, this crate computes
//! dense correspondence between every pair of them with pixelmap, recovers where each
//! photo was taken from, and fuses the result into a single mesh. `pixelmap_model_3d`
//! lifts one mapping between two photos into a surface; this crate registers every view
//! into one frame so that their surfaces agree.
//!
//! # Stages
//!
//! In order:
//!
//! 1. [`pairs`]: pixelmap over every pair of photos, and the check that they link up.
//! 2. [`twoview`]: the relative pose of each pair, and triage of weak pairs.
//! 3. [`tracks`]: points followed across views, kept only where every view agrees.
//! 4. [`sfm`]: every camera in one frame by incremental PnP, with a sparse point cloud.
//! 5. [`ba`]: bundle adjustment of every camera and point together.
//! 6. [`depth`]: a dense depth map per view, cross-checked between views.
//! 7. [`fusion`]: the depth maps fused into one [`mesh::Mesh`].
//! 8. [`texture`]: colour from the photos, per vertex and as a texture atlas.
//!
//! [`pipeline::run`] runs them all in one call, reporting each as it finishes.
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

pub mod align;
pub mod ba;
pub mod calib;
pub mod depth;
pub mod error;
pub mod export;
pub mod fusion;
pub mod input;
pub mod lookup;
pub mod mesh;
pub mod pairs;
pub mod pipeline;
pub mod pnp;
pub mod pose;
pub mod progress;
pub mod rng;
pub mod sfm;
pub mod synthetic;
pub mod texture;
pub mod tracks;
pub mod triangulate;
pub mod twoview;
pub mod types;

pub use calib::{FocalSource, Intrinsics};
pub use error::Error;
pub use input::MIN_VIEWS;
pub use lookup::PairLookup;
pub use mesh::Mesh;
pub use pairs::PairGraph;
pub use pipeline::Reconstruction;
pub use pose::Pose;
pub use progress::{Event, Flow, PairMap, Stage};
pub use types::{Norm, PairId, PhotoPx, ViewId, World};
