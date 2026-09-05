//! Dense image correspondence: given two photographs of the same scene, work out where
//! each pixel of the first one went in the second.
//!
//! This is the reference implementation of the **PIXELMAP** framework ([white
//! paper](https://doi.org/10.36227/techrxiv.173749998.89779329/v1)). Every cell of an
//! *affine correspondence grid* acts as an autonomous agent holding its own local affine
//! transform; agents refine their transform against the image data and propagate what
//! they find to their neighbours, and a forward/backward consistency check culls the ones
//! that disagree. Repeating that coarse-to-fine yields a dense, geometrically consistent
//! mapping.
//!
//! Useful for optical flow, image registration and stitching, stereo matching, morphing,
//! and as the front half of a 3D reconstruction — the `pixelmap_model_3d` crate in the
//! repository lifts a finished mapping into a textured 3D mesh.
//!
//! # What it produces
//!
#![doc = include_str!("../doc/results.md")]
//!
//! # Quick start
//!
//! ```no_run
//! use pixelmap::{Correspondence, Photo, Quality};
//!
//! # fn decode(_: &str) -> (usize, usize, Vec<u8>) { (0, 0, Vec::new()) }
//! let (w, h, rgba) = decode("a.jpg");
//! let a = Photo::from_rgba(w, h, rgba)?;
//! let (w, h, rgba) = decode("b.jpg");
//! let b = Photo::from_rgba(w, h, rgba)?;
//!
//! let mapping = Correspondence::builder().quality(Quality::Low).run(a, b)?;
//!
//! // Where did the pixel at (120, 84) end up?
//! match mapping.lookup(120.0, 84.0) {
//!     Some((x, y)) => println!("({x:.1}, {y:.1})"),
//!     None => println!("not mapped here"),
//! }
//! # Ok::<(), pixelmap::Error>(())
//! ```
//!
//! # The contract
//!
//! **Input.** Both photos must have the same dimensions and be at least
//! [`MIN_DIMENSION`] on each side. Violations come back as an [`Error`], never a panic.
//!
//! **Output.** [`Correspondence::lookup`] answers in the coordinates of the photos you
//! passed in. Underneath, the solver works at a reduced resolution and the two
//! [`DensePhotoMap`]s reached through [`Correspondence::forward`] and
//! [`Correspondence::backward`] are in *that* space; [`Correspondence::working_scale`]
//! relates the two. Regions the algorithm could not map — occlusions, featureless sky,
//! anything the consistency check rejected — are reported as `None` rather than as a
//! sentinel value.
//!
//! **Determinism.** The order in which the solver drains its queue decides which local
//! optimum the relaxation settles into, so it is seeded, and the seed defaults to
//! [`DEFAULT_SEED`]. The same photos, schedule and seed give the same mapping — run to
//! run, thread to thread, and machine to machine. Use [`Builder::seed`] to vary it.
//! Enabling or disabling the `parallel` feature does not change the result.
//!
//! **Threading.** Everything the caller holds is `Send + Sync`, so a mapping can be
//! computed on a worker thread and the result shared afterwards.
//!
//! **Cost.** Roughly linear in pixels at the working resolution, times the number of
//! schedule steps. See [`Quality`].
//!
//! # Feature flags
//!
//! - **`parallel`** *(default)* — multi-threaded feature matching via rayon. Turn it off
//!   for `wasm32-unknown-unknown`, which has no threads to hand out; the matcher falls
//!   back to a serial search with the same result.
//! - **`image`** — `From<image::RgbaImage>` and `From<image::DynamicImage>` for [`Photo`].
//! - **`bench`** — compiles the matcher benchmark harness. Not part of the pipeline.

#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]
#![cfg_attr(docsrs, feature(doc_auto_cfg))]

pub mod correspondence;
pub mod dense_photo_map;
pub mod error;
pub mod photo;
pub mod pixelmap_processor;
pub mod processing_mode;

// Internal machinery. These are the parts most likely to change as the solver is tuned,
// so they stay private: publishing them would freeze implementation details into the
// crate's semver contract for no one's benefit.
mod ac_grid;
mod affine_transform;
mod affine_transform_cell;
mod circular_feature_descriptor;
mod circular_feature_descriptor_matcher;
mod circular_feature_grid;
mod correspondence_mapping_algorithm;
mod correspondence_scoring;
mod kdtree;
mod rng;

/// Head-to-head benchmark of the matcher's nearest-neighbour backends.
/// Enabled by the `bench` feature; not part of the pipeline, and not part of the crate's
/// public contract — hidden from the docs so it does not read as something to build on.
#[cfg(feature = "bench")]
#[doc(hidden)]
pub mod matcher_bench;

pub use correspondence::{correspond, Builder, Correspondence, Progress};
pub use dense_photo_map::DensePhotoMap;
pub use error::{DecodeError, Error};
pub use photo::{Photo, MIN_DIMENSION};
pub use pixelmap_processor::{PixelMapProcessor, DEFAULT_SEED};
pub use processing_mode::{IterationParams, ProcessingMode, Quality};
