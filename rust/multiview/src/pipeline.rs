//! The whole reconstruction, from photos to a textured mesh, in one call.
//!
//! [`run`] takes photos and intrinsics; [`reconstruct`] starts from mappings that already
//! exist, which is how the synthetic tests feed it exact correspondences. Both report every
//! stage through the progress callback, with a one-line summary as each stage finishes,
//! and stop at the next stage boundary when the callback returns `Break`.
//!
//! Everything computed along the way is returned, not only the mesh. A reconstruction
//! that went wrong is diagnosed from its intermediates.

use std::sync::Arc;

use pixelmap::{Photo, Quality, DEFAULT_SEED};

use crate::ba;
use crate::calib::{FocalSource, Intrinsics};
use crate::depth::{self, DepthMap};
use crate::error::Error;
use crate::fusion::{self, Fused};
use crate::input;
use crate::lookup::PairLookup;
use crate::pairs::{self, PairGraph, MIN_PAIR_COVERAGE};
use crate::progress::{Event, Flow, Stage};
use crate::rng::Rng;
use crate::sfm::{self, SparseModel};
use crate::texture::{self, Texture};
use crate::tracks::{self, Track};
use crate::twoview::{self, Degeneracy, RelativePose, Verdict};
use crate::types::{PairId, ViewId};

/// Settings for every stage.
#[derive(Clone, Debug)]
pub struct Params {
    /// How much work pixelmap puts into each pair.
    pub quality: Quality,
    /// The seed for pixelmap and for every random choice after it.
    pub seed: u64,
    /// Two-view geometry.
    pub twoview: twoview::Params,
    /// Tracks.
    pub tracks: tracks::Params,
    /// Registration.
    pub sfm: sfm::Params,
    /// Bundle adjustment. The focal length is refined when this asks for it, and always
    /// when the intrinsics' focal length was only estimated.
    pub ba: ba::Params,
    /// Dense depth.
    pub depth: depth::Params,
    /// Fusion.
    pub fusion: fusion::Params,
    /// Texturing.
    pub texture: texture::Params,
}

impl Default for Params {
    fn default() -> Self {
        Params {
            quality: Quality::Low,
            seed: DEFAULT_SEED,
            twoview: twoview::Params::default(),
            tracks: tracks::Params::default(),
            sfm: sfm::Params::default(),
            ba: ba::Params::default(),
            depth: depth::Params::default(),
            fusion: fusion::Params::default(),
            texture: texture::Params::default(),
        }
    }
}

/// What two-view geometry made of one pair.
#[derive(Clone, Debug)]
pub struct PairReport {
    /// The pair.
    pub pair: PairId,
    /// The fraction of the first photo its mapping covers.
    pub coverage: f32,
    /// The estimate, or why there is none.
    pub estimate: Result<RelativePose, Degeneracy>,
}

/// A finished reconstruction and everything computed on the way to it.
#[derive(Clone, Debug)]
pub struct Reconstruction {
    /// The photos' dimensions.
    pub size: (usize, usize),
    /// The intrinsics the later stages used, with the focal length refined if it was.
    pub intrinsics: Intrinsics,
    /// The views linked by well-mapped pairs.
    pub connected: Vec<ViewId>,
    /// Every pair's two-view geometry.
    pub pairs: Vec<PairReport>,
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
    /// The fused surface.
    pub fused: Fused,
    /// The surface's colour.
    pub texture: Texture,
}

/// Reconstructs a textured mesh from `photos`.
///
/// # Errors
/// Any [`Error`]; each one names the stage that failed and why.
pub fn run(
    photos: &[Arc<Photo>],
    intrinsics: &Intrinsics,
    params: &Params,
    on_event: &mut dyn FnMut(Event) -> Flow,
) -> Result<Reconstruction, Error> {
    let (width, height) = input::validate(photos, intrinsics)?;
    report(
        on_event,
        Stage::Input,
        format!("{} photos at {width}×{height}", photos.len()),
    )?;
    let graph = pairs::compute(photos, params.quality, params.seed, on_event)?;
    reconstruct(&graph, photos, intrinsics, params, on_event)
}

/// Reconstructs a textured mesh from mappings already computed between `photos`.
///
/// # Errors
/// Any [`Error`] from the stages after pairwise correspondence.
///
/// # Panics
/// If `graph` does not span exactly the views in `photos`.
pub fn reconstruct<L: PairLookup>(
    graph: &PairGraph<L>,
    photos: &[Arc<Photo>],
    intrinsics: &Intrinsics,
    params: &Params,
    on_event: &mut dyn FnMut(Event) -> Flow,
) -> Result<Reconstruction, Error> {
    assert_eq!(
        graph.views(),
        photos.len(),
        "one mapping graph view per photo"
    );
    let size = input::validate(photos, intrinsics)?;
    let precision = graph.precision_px() as f64;

    let connected = pairs::require_connected(graph, MIN_PAIR_COVERAGE)?;
    if connected.len() < graph.views() {
        let dropped: Vec<String> = (0..graph.views() as u32)
            .map(ViewId)
            .filter(|v| !connected.contains(v))
            .map(|v| v.to_string())
            .collect();
        report(
            on_event,
            Stage::Pairs,
            format!(
                "leaving out {}: no pair with at least {:.0}% coverage links it to the others",
                dropped.join(", "),
                MIN_PAIR_COVERAGE * 100.0
            ),
        )?;
    }

    let root = Rng::new(params.seed);
    let total = PairId::count(graph.views());
    let mut pair_reports = Vec::with_capacity(total);
    for (index, (pair, lookup)) in graph.pairs().enumerate() {
        let estimate = twoview::estimate(
            pair,
            lookup,
            intrinsics,
            size,
            &params.twoview,
            &mut root.derive(index as u64),
        );
        let summary = match &estimate {
            Ok(r) => match &r.verdict {
                Verdict::Usable => format!(
                    "{pair}: usable; {:.0}% mapped, {:.0}% of matches agree, {:.1}° between rays",
                    r.coverage * 100.0,
                    r.inlier_ratio * 100.0,
                    r.median_angle_deg
                ),
                Verdict::Degenerate(reason) => format!("{pair}: not usable, {reason}"),
            },
            Err(reason) => format!("{pair}: not usable, {reason}"),
        };
        progress(
            on_event,
            Stage::TwoView,
            (index + 1) as f32 / total as f32,
            summary,
        )?;
        pair_reports.push(PairReport {
            pair,
            coverage: lookup.coverage(),
            estimate,
        });
    }
    let relative: Vec<RelativePose> = pair_reports
        .iter()
        .filter_map(|r| r.estimate.as_ref().ok())
        .filter(|r| connected.contains(&r.pair.a()) && connected.contains(&r.pair.b()))
        .cloned()
        .collect();

    let tracks = tracks::build(graph, size, &params.tracks);
    let multi = tracks::require_enough(&tracks)?;
    report(
        on_event,
        Stage::Tracks,
        format!(
            "{} tracks, {multi} of them seen in three or more views",
            tracks.len()
        ),
    )?;

    let registered = sfm::reconstruct(
        &relative,
        &tracks,
        intrinsics,
        graph.views(),
        precision,
        &params.sfm,
        &mut root.derive(u64::MAX),
    )
    .map_err(|error| explain_no_usable_pair(error, &pair_reports))?;
    for warning in &registered.warnings {
        progress(
            on_event,
            Stage::Registration,
            1.0,
            format!("warning: {warning}"),
        )?;
    }
    report(
        on_event,
        Stage::Registration,
        format!(
            "{} of {} views placed, starting from {}; {} points",
            registered.registered().len(),
            graph.views(),
            registered.seed,
            registered.points.len()
        ),
    )?;

    let refine_focal = params.ba.refine_focal || intrinsics.source == FocalSource::Estimated;
    let ba::Adjusted {
        model: adjusted,
        intrinsics: refined,
        report: adjustment,
    } = ba::adjust(
        &registered,
        &tracks,
        intrinsics,
        precision,
        &ba::Params {
            refine_focal,
            ..params.ba.clone()
        },
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
    report(on_event, Stage::BundleAdjustment, summary)?;

    let maps = depth::estimate(graph, &adjusted.cameras, &refined, size, &params.depth);
    let valid = depth::require_coverage(&maps)?;
    report(
        on_event,
        Stage::Depth,
        format!(
            "{:.0}% of samples have a depth, one sample every {} px",
            valid * 100.0,
            maps.first().map_or(0, |m| m.stride)
        ),
    )?;

    let fused = fusion::fuse(&maps, &adjusted.cameras, &refined, &params.fusion)?;
    report(
        on_event,
        Stage::Fusion,
        format!(
            "{} vertices, {} triangles, voxel size {:.4}",
            fused.mesh.positions.len(),
            fused.mesh.triangles.len(),
            fused.voxel_size
        ),
    )?;

    let texture = texture::build(
        &fused.mesh,
        &adjusted.cameras,
        &refined,
        photos,
        &maps,
        &params.texture,
    );
    report(
        on_event,
        Stage::Texture,
        format!(
            "{} charts in a {}×{} atlas",
            texture.charts,
            texture.atlas.width(),
            texture.atlas.height()
        ),
    )?;

    Ok(Reconstruction {
        size,
        intrinsics: refined,
        connected,
        pairs: pair_reports,
        tracks,
        registered,
        adjusted,
        adjustment,
        depth: maps,
        fused,
        texture,
    })
}

/// Reports a stage as finished.
fn report(
    on_event: &mut dyn FnMut(Event) -> Flow,
    stage: Stage,
    message: String,
) -> Result<(), Error> {
    progress(on_event, stage, 1.0, message)
}

fn progress(
    on_event: &mut dyn FnMut(Event) -> Flow,
    stage: Stage,
    fraction: f32,
    message: String,
) -> Result<(), Error> {
    if on_event(Event::new(stage, fraction, message)).is_break() {
        return Err(Error::Cancelled { stage });
    }
    Ok(())
}

/// Registration only sees the pairs two-view geometry could estimate. When none was
/// usable, say why for every pair, including those it could not estimate at all.
fn explain_no_usable_pair(error: Error, reports: &[PairReport]) -> Error {
    match error {
        Error::NoUsablePair { .. } => Error::NoUsablePair {
            reasons: reports
                .iter()
                .filter_map(|r| {
                    let reason = match &r.estimate {
                        Ok(estimate) => match &estimate.verdict {
                            Verdict::Usable => return None,
                            Verdict::Degenerate(reason) => reason.to_string(),
                        },
                        Err(reason) => reason.to_string(),
                    };
                    Some((r.pair, reason))
                })
                .collect(),
        },
        other => other,
    }
}
