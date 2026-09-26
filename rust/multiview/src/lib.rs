//! Multi-view 3D reconstruction on top of [`pixelmap`].
//!
//! Given three or more photos of a static scene taken with one camera, this crate computes
//! dense correspondence between every pair of them with pixelmap, recovers where each
//! photo was taken from, and fuses the result into a single mesh. `pixelmap_model_3d`
//! lifts one mapping between two photos into a surface; this crate registers every view
//! into one frame so that their surfaces agree.
//!
//! # Quick start
//!
//! Hand over the photos, say how much effort to spend, and follow along:
//!
//! ```no_run
//! use std::sync::Arc;
//! use pixelmap::{Photo, Quality};
//! use pixelmap_multiview::{Event, Flow, Focal, Options};
//!
//! # fn photos() -> Vec<Arc<Photo>> { Vec::new() }
//! let photos = photos();
//! let options = Options::new()
//!     .quality(Quality::Medium)
//!     .focal(Focal::Equivalent35mm(28.0));
//!
//! let model = pixelmap_multiview::run(&photos, &options, &mut |event: Event| {
//!     println!("{}: {}", event.stage(), event.message());
//!     Flow::Continue(())
//! })?;
//!
//! println!("{} triangles", model.mesh.triangles.len());
//! # Ok::<(), pixelmap_multiview::Error>(())
//! ```
//!
//! With the `threads` feature, `Job` runs that on a worker thread and hands back a
//! channel of events, a [`Status`] to poll, and a way to stop it.
//!
//! # Stages
//!
//! In order: pairwise correspondence with pixelmap, the relative pose of each pair, points
//! followed across views, every camera placed in one frame, bundle adjustment, a dense
//! depth map per view, fusion into one [`Mesh`], and colour from the photos. [`Stage`]
//! names them; every [`Event`] says which one it came from.
//!
//! # Feedback while it runs
//!
//! One pixelmap solve per pair of photos is the bulk of the work, so a run is minutes long
//! for a handful of photos and the caller hears about it throughout. [`Event`] carries
//! data rather than prose — which pair is being mapped, how far through it, why one was
//! rejected — so that showing progress never means parsing a sentence. The event that
//! finishes a pair carries that pair's [`PairMap`], the raw correspondence grid, for a
//! caller that wants to show the match as it is found.
//!
//! Returning [`Flow::Break`](std::ops::ControlFlow::Break) stops the run, which then
//! returns [`Error::Cancelled`]. A stop takes effect at the next checkpoint: within
//! pairwise correspondence that is one pixelmap schedule step, elsewhere it is prompt.
//!
//! # Design rules
//!
//! - **No I/O, no platform.** Photos come in as RGBA buffers and a mesh goes out.
//!   Decoding, EXIF and resizing are the caller's job, which keeps the crate buildable for
//!   `wasm32-unknown-unknown`.
//! - **One type per coordinate space.** See [`PhotoPx`], [`Norm`], [`World`].
//! - **Deterministic.** Every random choice draws from a generator derived from a single
//!   seed, as pixelmap's own solver does, so any run can be replayed exactly.
//! - **Fail loudly.** Each stage has exit criteria. A run that cannot produce a correct
//!   model returns an [`Error`] naming the stage and the reason, rather than a
//!   plausible-looking wrong model.
//!
//! # Dependencies in the API
//!
//! Positions, normals and poses are [`nalgebra`] types, and photos, quality and
//! correspondences are [`pixelmap`] types. Both crates are re-exported, so
//! `pixelmap_multiview::nalgebra` and `pixelmap_multiview::pixelmap` are always the versions
//! this crate was built against. A new major version of either is a breaking change here.
//!
//! # Feature flags
//!
//! - **`parallel`** *(default)* — passed through to pixelmap. Turn it off for wasm.
//! - **`threads`** *(default)* — `Job`, which runs a reconstruction on a worker thread.
//!   Turn it off for wasm, which has no threads to hand out; [`run`] works there and is
//!   what `Job` drives underneath.

#![warn(missing_docs)]
#![cfg_attr(docsrs, feature(doc_cfg))]

pub use nalgebra;
pub use pixelmap;

pub mod export;
pub mod pipeline;

#[cfg(feature = "threads")]
pub mod job;

// The stages themselves. Hidden rather than private, and hidden rather than published:
// the stage tests check one stage at a time against synthetic ground truth and, being
// integration tests, can only reach public paths. Keeping them out of the documentation
// says what `pub` alone does not — that these are the parts most likely to change as the
// geometry is tuned, and that nothing outside this repository should build on them. The
// types a caller actually names are re-exported below. `pixelmap` marks its benchmark
// harness the same way, for the same reason.
#[doc(hidden)]
pub mod align;
#[doc(hidden)]
pub mod ba;
#[doc(hidden)]
pub mod calib;
#[doc(hidden)]
pub mod depth;
#[doc(hidden)]
pub mod fusion;
#[doc(hidden)]
pub mod input;
#[doc(hidden)]
pub mod pairs;
#[doc(hidden)]
pub mod pnp;
#[doc(hidden)]
pub mod rng;
#[doc(hidden)]
pub mod sfm;
// Scenes with known geometry, rendered and matched exactly. For this crate's tests, not
// for building on.
#[doc(hidden)]
pub mod synthetic;
#[doc(hidden)]
pub mod texture;
#[doc(hidden)]
pub mod tracks;
#[doc(hidden)]
pub mod triangulate;
#[doc(hidden)]
pub mod twoview;

// Plumbing that nothing outside the crate needs to name. What a caller uses from these is
// re-exported below.
mod error;
mod event;
mod lookup;
mod mesh;
mod options;
mod pose;
mod types;

pub use calib::{FocalSource, Intrinsics};
pub use error::Error;
pub use event::{Event, Flow, Level, PairMap, Stage, Status};
pub use input::MIN_VIEWS;
pub use lookup::PairLookup;
pub use mesh::Mesh;
pub use options::{Focal, Options, DEFAULT_MAX_TEXTURE_SIZE};
pub use pairs::{PairGraph, MIN_PAIR_COVERAGE};
pub use pipeline::{reconstruct, run, Diagnostics, Model, PairReport};
pub use pose::Pose;
pub use sfm::DropReason;
pub use texture::Texture;
pub use twoview::Degeneracy;
pub use types::{Norm, PairId, PhotoPx, ViewId, World};

#[cfg(feature = "threads")]
pub use job::{Cancel, Job};

// Reached through `Model::diagnostics`, so they have to be nameable.
pub use ba::Report as AdjustmentReport;
pub use depth::{Counts as DepthCounts, DepthMap, DepthStats, Fate as SampleFate};
pub use sfm::{Registration, SparseModel, SparsePoint, Warning as RegistrationWarning};
pub use tracks::Track;
