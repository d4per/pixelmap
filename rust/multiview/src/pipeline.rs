//! The whole reconstruction, from photos to a textured mesh, in one call.
//!
//! [`run`] takes photos and [`Options`]; [`reconstruct`] starts from mappings that already
//! exist, which is how the synthetic tests feed it exact correspondences. Both report every
//! stage through the event callback and stop when it returns `Break`.
//!
//! [`Model`] carries what a caller came for — the mesh, its texture, the cameras, and which
//! photos made it. Everything else computed along the way is behind [`Model::diagnostics`]:
//! a reconstruction that went wrong is diagnosed from its intermediates, but most callers
//! never look, and [`Model::take_diagnostics`] lets them free it.

use std::sync::Arc;

use pixelmap::Photo;

use crate::ba;
use crate::calib::{FocalSource, Intrinsics};
use crate::depth::{self, DepthMap, DepthStats};
use crate::error::Error;
use crate::event::{emit, report, Event, Flow, Level, Stage};
use crate::fusion::{self, Fused};
use crate::input::{self, MIN_VIEWS};
use crate::lookup::PairLookup;
use crate::mesh::Mesh;
use crate::options::Options;
use crate::pairs::{self, PairGraph, MIN_PAIR_COVERAGE};
use crate::pose::Pose;
use crate::rng::Rng;
use crate::sfm::{self, SparseModel};
use crate::texture::{self, Texture};
use crate::tracks::{self, Track};
use crate::twoview::{self, Degeneracy, RelativePose, Verdict};
use crate::types::{PairId, ViewId};

/// Settings for every stage.
///
/// Derived from [`Options`]: these are tuning decisions, not choices a caller should have
/// to make, so they are not part of the crate's contract. There are about fifty of them and
/// they are meaningful only together. Reachable, and hidden, so that this crate's own tests
/// can drive a single stage to a chosen outcome; nothing outside should build on it.
#[doc(hidden)]
#[derive(Clone, Debug)]
pub struct Params {
    pub seed: u64,
    pub twoview: twoview::Params,
    pub tracks: tracks::Params,
    pub sfm: sfm::Params,
    pub ba: ba::Params,
    pub depth: depth::Params,
    pub fusion: fusion::Params,
    pub texture: texture::Params,
}

impl Default for Params {
    fn default() -> Self {
        Params::from_options(&Options::default())
    }
}

impl Params {
    /// The thresholds a run with these options judges by.
    pub(crate) fn from_options(options: &Options) -> Self {
        Params {
            seed: options.resolved_seed(),
            twoview: twoview::Params::default(),
            tracks: tracks::Params::default(),
            sfm: sfm::Params::default(),
            ba: ba::Params {
                refine_focal: options.refine_focal,
                ..ba::Params::default()
            },
            depth: depth::Params::default(),
            fusion: fusion::Params::default(),
            texture: texture::Params {
                max_atlas_size: options.max_texture_size,
                ..texture::Params::default()
            },
        }
    }
}

/// What two-view geometry made of one pair.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct PairReport {
    /// The pair.
    pub pair: PairId,
    /// The fraction of the first photo its mapping covers.
    pub coverage: f32,
    /// The second camera in the first camera's frame, scaled so that the distance between
    /// them is 1. `None` when the pair was rejected: a degenerate estimate is not a pose
    /// worth showing.
    pub pose: Option<Pose>,
    /// The share of sampled matches that agree on one camera motion. `None` when too few
    /// matches were found to estimate one.
    pub inlier_ratio: Option<f64>,
    /// The median angle between the two cameras' rays through the agreeing matches, in
    /// degrees. `None` when no camera motion could be estimated.
    pub median_angle_deg: Option<f64>,
    /// Why this pair could not anchor a registration, or `None` if it could.
    ///
    /// A pair is unusable either because no geometry could be estimated at all, or because
    /// the geometry that was estimated is degenerate; this does not distinguish the two.
    pub rejection: Option<Degeneracy>,
}

impl PairReport {
    fn new(pair: PairId, coverage: f32, estimate: &Result<RelativePose, Degeneracy>) -> PairReport {
        let (pose, inlier_ratio, median_angle_deg, rejection) = match estimate {
            Ok(e) => match &e.verdict {
                Verdict::Usable => (
                    Some(e.pose),
                    Some(e.inlier_ratio),
                    Some(e.median_angle_deg),
                    None,
                ),
                Verdict::Degenerate(reason) => (
                    None,
                    Some(e.inlier_ratio),
                    Some(e.median_angle_deg),
                    Some(reason.clone()),
                ),
            },
            Err(reason) => (None, None, None, Some(reason.clone())),
        };
        PairReport {
            pair,
            coverage,
            pose,
            inlier_ratio,
            median_angle_deg,
            rejection,
        }
    }
}

/// A finished reconstruction.
///
/// Everything spatial is in one frame, which the seed pair defines: its first camera sits
/// at the origin looking along +z with +y down the image, and the distance between the
/// seed pair's two cameras is the unit of length. Nothing is metric. The exporters in
/// [`crate::export`] turn the result half a revolution about x so viewers show it upright.
#[derive(Clone, Debug)]
pub struct Model {
    /// The surface, with normals.
    pub mesh: Mesh,
    /// Its colour, per vertex and as a texture atlas.
    pub texture: Texture,
    /// The camera the later stages used, with the focal length refined if it was.
    pub intrinsics: Intrinsics,
    /// Where each photo was taken from, indexed by [`ViewId`]. `None` for a photo that was
    /// left out.
    pub cameras: Vec<Option<Pose>>,
    /// How many photos went in.
    pub views: usize,
    /// The photos that made it into the model. Shorter than [`Self::views`] when one was
    /// left out; every such photo was reported as an [`Event::ViewDropped`] as it happened.
    pub connected: Vec<ViewId>,
    /// Every pair's two-view geometry, including the pairs that were not usable.
    pub pairs: Vec<PairReport>,
    diagnostics: Option<Box<Diagnostics>>,
}

impl Model {
    /// Everything computed on the way to the model, or `None` once
    /// [`Self::take_diagnostics`] has taken it.
    ///
    /// For working out why a reconstruction came out the way it did. Not needed to use the
    /// result.
    pub fn diagnostics(&self) -> Option<&Diagnostics> {
        self.diagnostics.as_deref()
    }

    /// Takes the diagnostics out of the model.
    ///
    /// They hold every depth map, the fate of every depth sample, and every track, which is
    /// usually more memory than the mesh itself. A caller that keeps the model around and
    /// never looks at them can drop them with this.
    pub fn take_diagnostics(&mut self) -> Option<Diagnostics> {
        self.diagnostics.take().map(|d| *d)
    }
}

/// The intermediates a reconstruction passed through.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct Diagnostics {
    /// The photos' dimensions.
    pub size: (usize, usize),
    /// The tracks.
    pub tracks: Vec<Track>,
    /// The cameras and points as registered, before bundle adjustment.
    pub registered: SparseModel,
    /// The cameras and points after bundle adjustment.
    pub adjusted: SparseModel,
    /// What bundle adjustment did.
    pub adjustment: ba::Report,
    /// The depth maps.
    pub depth: Vec<DepthMap>,
    /// Where each view's depth samples were lost.
    pub depth_stats: Vec<DepthStats>,
    /// The voxel edge length fusion used, in reconstruction units.
    pub voxel_size: f64,
    /// Allocated blocks of 8³ voxels.
    pub blocks: usize,
    /// The share of triangles in the largest connected piece, before small pieces were
    /// dropped.
    pub largest_component: f64,
    /// Triangles dropped with the small pieces.
    pub dropped_triangles: usize,
}

/// Reconstructs a textured mesh from `photos`.
///
/// The photos must all come from one camera at one fixed zoom, and there must be at least
/// [`MIN_VIEWS`] of them. `on_event` hears about every stage and can stop the run by
/// returning [`Flow::Break`](std::ops::ControlFlow::Break).
///
/// # Errors
/// Any [`Error`]; each one names the stage that failed and why.
pub fn run(
    photos: &[Arc<Photo>],
    options: &Options,
    on_event: &mut dyn FnMut(Event) -> Flow,
) -> Result<Model, Error> {
    // The camera depends on the photos' dimensions and validation depends on the camera,
    // so take the dimensions from the first photo and let validation judge the rest.
    let first = photos.first().ok_or(Error::TooFewPhotos {
        found: 0,
        minimum: MIN_VIEWS,
    })?;
    let intrinsics = options.focal.intrinsics(first.width(), first.height());

    let (width, height) = input::validate(photos, &intrinsics)?;
    emit(
        on_event,
        Event::Started {
            views: photos.len(),
            pairs: PairId::count(photos.len()),
        },
    )?;
    report(
        on_event,
        Stage::Input,
        1.0,
        format!("{} photos at {width}×{height}", photos.len()),
    )?;

    let graph = pairs::compute(photos, options.quality, options.resolved_seed(), on_event)?;
    reconstruct_with_params(
        &graph,
        photos,
        &intrinsics,
        &Params::from_options(options),
        on_event,
    )
}

/// Reconstructs a textured mesh from mappings already computed between `photos`.
///
/// For a caller that brings its own correspondences: build a [`PairGraph`] with
/// [`PairGraph::from_fn`] over anything that implements [`PairLookup`]. [`run`] is this
/// with pixelmap's correspondences, and apart from computing those it does the same.
///
/// The camera comes from `options.focal` as it does for [`run`]; `options.quality` is
/// not used, since there is no matching left to do.
///
/// # Errors
/// [`Error::GraphMismatch`] if `graph` does not span exactly one view per photo, and any
/// [`Error`] from the stages after pairwise correspondence.
pub fn reconstruct<L: PairLookup>(
    graph: &PairGraph<L>,
    photos: &[Arc<Photo>],
    options: &Options,
    on_event: &mut dyn FnMut(Event) -> Flow,
) -> Result<Model, Error> {
    let first = photos.first().ok_or(Error::TooFewPhotos {
        found: 0,
        minimum: MIN_VIEWS,
    })?;
    let intrinsics = options.focal.intrinsics(first.width(), first.height());
    reconstruct_with_params(
        graph,
        photos,
        &intrinsics,
        &Params::from_options(options),
        on_event,
    )
}

/// [`reconstruct`], with every stage's thresholds given outright.
///
/// For this crate's own tests, which drive one stage to a chosen outcome. Not part of the
/// contract; use [`reconstruct`].
///
/// # Errors
/// As [`reconstruct`].
#[doc(hidden)]
pub fn reconstruct_with_params<L: PairLookup>(
    graph: &PairGraph<L>,
    photos: &[Arc<Photo>],
    intrinsics: &Intrinsics,
    params: &Params,
    on_event: &mut dyn FnMut(Event) -> Flow,
) -> Result<Model, Error> {
    if graph.views() != photos.len() {
        return Err(Error::GraphMismatch {
            views: graph.views(),
            photos: photos.len(),
        });
    }
    let size = input::validate(photos, intrinsics)?;
    let precision = graph.precision_px() as f64;

    let connected = pairs::require_connected(graph, MIN_PAIR_COVERAGE)?;
    for view in (0..graph.views() as u32).map(ViewId) {
        if !connected.contains(&view) {
            emit(
                on_event,
                Event::ViewDropped {
                    stage: Stage::Pairs,
                    view,
                    reason: format!(
                        "no pair with at least {:.0}% coverage links it to the others",
                        MIN_PAIR_COVERAGE * 100.0
                    ),
                },
            )?;
        }
    }

    let root = Rng::new(params.seed);
    let total = PairId::count(graph.views());
    let mut pair_reports = Vec::with_capacity(total);
    let mut estimates = Vec::with_capacity(total);
    for (index, (pair, lookup)) in graph.pairs().enumerate() {
        let estimate = twoview::estimate(
            pair,
            lookup,
            intrinsics,
            size,
            &params.twoview,
            &mut root.derive(index as u64),
        );
        let outcome = PairReport::new(pair, lookup.coverage(), &estimate);
        match &outcome.rejection {
            Some(reason) => emit(
                on_event,
                Event::PairRejected {
                    pair,
                    index,
                    of: total,
                    reason: reason.clone(),
                },
            )?,
            None => {
                let estimate = estimate
                    .as_ref()
                    .expect("a pair with nothing against it has an estimate");
                report(
                    on_event,
                    Stage::TwoView,
                    (index + 1) as f32 / total as f32,
                    format!(
                        "{pair}: usable; {:.0}% mapped, {:.0}% of matches agree, {:.1}° between rays",
                        estimate.coverage * 100.0,
                        estimate.inlier_ratio * 100.0,
                        estimate.median_angle_deg
                    ),
                )?;
            }
        }
        pair_reports.push(outcome);
        estimates.push(estimate);
    }
    let relative: Vec<RelativePose> = estimates
        .into_iter()
        .filter_map(Result::ok)
        .filter(|r| connected.contains(&r.pair.a()) && connected.contains(&r.pair.b()))
        .collect();

    let tracks = tracks::build(graph, size, &params.tracks);
    let multi = tracks::require_enough(&tracks)?;
    report(
        on_event,
        Stage::Tracks,
        1.0,
        format!(
            "{} tracks, {multi} of them seen in three or more views",
            tracks.len()
        ),
    )?;

    let refine_focal = params.ba.refine_focal || intrinsics.source == FocalSource::Estimated;
    let registered = sfm::reconstruct_with_progress(
        &relative,
        &tracks,
        intrinsics,
        graph.views(),
        precision,
        &sfm::Params {
            adjust: ba::Params {
                refine_focal,
                ..params.sfm.adjust.clone()
            },
            ..params.sfm.clone()
        },
        &mut root.derive(u64::MAX),
        on_event,
    )
    .map_err(|error| explain_no_usable_pair(error, &pair_reports))?;
    for warning in &registered.warnings {
        match warning {
            // A view that was left out has already been reported, as it happened.
            sfm::Warning::Unregistered { .. } => {}
            other => emit(
                on_event,
                Event::Log {
                    stage: Stage::Registration,
                    level: Level::Warning,
                    message: other.to_string(),
                },
            )?,
        }
    }
    report(
        on_event,
        Stage::Registration,
        1.0,
        format!(
            "{} of {} views placed, starting from {}; {} points",
            registered.registered().len(),
            graph.views(),
            registered.seed,
            registered.points.len()
        ),
    )?;

    let ba::Adjusted {
        model: adjusted,
        intrinsics: refined,
        report: adjustment,
    } = ba::adjust_with_progress(
        &registered,
        &tracks,
        &registered.intrinsics,
        precision,
        &ba::Params {
            refine_focal,
            ..params.ba.clone()
        },
        on_event,
    )?;
    let mut summary = format!(
        "median reprojection error {:.2} px → {:.2} px",
        adjustment.initial_median_px, adjustment.final_median_px
    );
    if refine_focal {
        summary += &format!(
            "; focal length {:.1} px → {:.1} px",
            intrinsics.fx, refined.fx
        );
    }
    report(on_event, Stage::BundleAdjustment, 1.0, summary)?;

    let (maps, depth_stats) = depth::estimate_with_progress(
        graph,
        &adjusted.cameras,
        &refined,
        size,
        &params.depth,
        on_event,
    )?;
    let valid = depth::require_coverage(&maps)?;
    report(
        on_event,
        Stage::Depth,
        1.0,
        format!(
            "{:.0}% of samples have a depth, one sample every {} px",
            valid * 100.0,
            maps.first().map_or(0, |m| m.stride)
        ),
    )?;

    // Taken apart rather than kept whole: the mesh belongs to the model, and copying it
    // into the diagnostics as well would duplicate megabytes for nothing.
    let Fused {
        mesh,
        voxel_size,
        blocks,
        largest_component,
        dropped_triangles,
    } = fusion::fuse_with_progress(&maps, &adjusted.cameras, &refined, &params.fusion, on_event)?;
    report(
        on_event,
        Stage::Fusion,
        1.0,
        format!(
            "{} vertices, {} triangles, voxel size {voxel_size:.4}",
            mesh.positions.len(),
            mesh.triangles.len()
        ),
    )?;

    let texture = texture::build_with_progress(
        &mesh,
        &adjusted.cameras,
        &refined,
        photos,
        &maps,
        &params.texture,
        on_event,
    )?;
    report(
        on_event,
        Stage::Texture,
        1.0,
        format!(
            "{} charts in a {}×{} atlas",
            texture.charts,
            texture.atlas.width(),
            texture.atlas.height()
        ),
    )?;

    Ok(Model {
        mesh,
        texture,
        intrinsics: refined,
        cameras: adjusted.cameras.clone(),
        views: graph.views(),
        connected,
        pairs: pair_reports,
        diagnostics: Some(Box::new(Diagnostics {
            size,
            tracks,
            registered,
            adjusted,
            adjustment,
            depth: maps,
            depth_stats,
            voxel_size,
            blocks,
            largest_component,
            dropped_triangles,
        })),
    })
}

/// Registration only sees the pairs two-view geometry could estimate. When none was
/// usable, say why for every pair, including those it could not estimate at all.
fn explain_no_usable_pair(error: Error, reports: &[PairReport]) -> Error {
    match error {
        Error::NoUsablePair { .. } => Error::NoUsablePair {
            reasons: reports
                .iter()
                .filter_map(|r| Some((r.pair, r.rejection.clone()?)))
                .collect(),
        },
        other => other,
    }
}
