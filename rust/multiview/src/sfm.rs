//! Stage 4: every camera placed in one frame, with a sparse point cloud.
//!
//! This does not build a reconstruction per pair and then try to reconcile their scales.
//! That approach drifts, and it is miserable to debug. Instead:
//!
//! 1. The best usable pair *defines* the frame. Its first camera is the origin and the
//!    distance between its two cameras is the unit of length.
//! 2. The tracks both of its views see are triangulated.
//! 3. The remaining views are registered one at a time by PnP against the points that
//!    exist so far, each followed by triangulating whatever it newly makes visible.
//!    A PnP solution is in the seed's frame and scale by construction, so there is no
//!    scale-fusion step anywhere.
//! 4. Finally every track is triangulated over every registered view that sees it, and
//!    points with a large reprojection error or poorly conditioned rays are dropped.

use std::fmt;

use nalgebra::Point3;

use crate::ba;
use crate::calib::Intrinsics;
use crate::error::Error;
use crate::event::{emit, report, silent, Event, Flow, Stage};
use crate::input::MIN_VIEWS;
use crate::pnp;
use crate::pose::Pose;
use crate::rng::Rng;
use crate::tracks::Track;
use crate::triangulate;
use crate::twoview::{RelativePose, Verdict};
use crate::types::{Norm, PairId, ViewId, World};

/// Tuning for [`reconstruct`].
#[derive(Clone, Debug)]
pub struct Params {
    /// The largest reprojection error a point or PnP inlier may have in any view, as a
    /// multiple of the mappings' precision.
    pub max_reprojection: f64,
    /// The smallest largest-angle between a point's rays, in degrees.
    pub min_point_angle_deg: f64,
    /// Below this PnP inlier ratio a view is left unregistered, unless it has
    /// [`Params::enough_pnp_inliers`].
    pub min_pnp_inlier_ratio: f64,
    /// A view with at least this many PnP inliers only needs
    /// [`Params::min_pnp_inlier_ratio_when_enough`]. That many points agreeing on a pose is
    /// no accident; the rest disagree because the points placed so far have drifted.
    pub enough_pnp_inliers: usize,
    /// The PnP inlier ratio a view with [`Params::enough_pnp_inliers`] needs.
    pub min_pnp_inlier_ratio_when_enough: f64,
    /// A view needs this many triangulated points it observes to be registered.
    pub min_registration_points: usize,
    /// The most PnP RANSAC iterations per view.
    pub max_iterations: usize,
    /// The probability PnP RANSAC should have of drawing one all-inlier sample.
    pub confidence: f64,
    /// Below this, as a fraction of the distance between the two farthest cameras, the
    /// camera centres are reported as collinear.
    pub min_camera_spread: f64,
    /// Bundle adjustment of the views placed so far, run while registering whenever their
    /// number has grown by half and before views that could not be placed are retried.
    /// Its focal length is refined only if this asks for it.
    pub adjust: ba::Params,
}

impl Default for Params {
    fn default() -> Self {
        Params {
            max_reprojection: 3.0,
            min_point_angle_deg: 1.5,
            min_pnp_inlier_ratio: 0.6,
            enough_pnp_inliers: 150,
            min_pnp_inlier_ratio_when_enough: 0.35,
            min_registration_points: 30,
            max_iterations: 1000,
            confidence: 0.999,
            min_camera_spread: 0.02,
            adjust: ba::Params {
                max_iterations: 15,
                rounds: 1,
                // The final adjustment judges the fit; this one only has to improve it.
                max_median_error: f64::INFINITY,
                ..ba::Params::default()
            },
        }
    }
}

/// A triangulated track.
#[derive(Clone, Debug, PartialEq)]
pub struct SparsePoint {
    /// Where it is.
    pub position: World,
    /// The index of the track it came from.
    pub track: usize,
    /// How many registered views it was triangulated from.
    pub observations: usize,
    /// Mean reprojection error over those views, in photo pixels.
    pub error_px: f64,
    /// The widest angle between any two of its rays, in degrees.
    pub angle_deg: f64,
}

/// How a view was added by PnP.
#[derive(Clone, Debug, PartialEq)]
pub struct Registration {
    /// The view.
    pub view: ViewId,
    /// How many triangulated points it observed when it was registered.
    pub candidates: usize,
    /// How many of them agreed with the pose found.
    pub inliers: usize,
    /// `inliers / candidates`.
    pub inlier_ratio: f64,
}

/// Something worth telling the user that did not stop the reconstruction.
#[derive(Clone, Debug, PartialEq)]
pub enum Warning {
    /// A view could not be placed and was left out.
    Unregistered {
        /// The view.
        view: ViewId,
        /// Why.
        reason: String,
    },
    /// The camera centres lie nearly on one line, which constrains depth along it weakly.
    CollinearCameras {
        /// The largest distance of a centre from the line through the two farthest
        /// centres, as a fraction of their distance.
        deviation: f64,
    },
}

impl fmt::Display for Warning {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Warning::Unregistered { view, reason } => {
                write!(f, "{view} was left out: {reason}")
            }
            Warning::CollinearCameras { deviation } => write!(
                f,
                "the camera positions lie almost on one line ({:.1}% off it); vary the height or distance between shots",
                deviation * 100.0
            ),
        }
    }
}

/// Cameras in one frame, and the points triangulated from them.
#[derive(Clone, Debug)]
pub struct SparseModel {
    /// Each view's pose, or `None` for a view that could not be registered.
    pub cameras: Vec<Option<Pose>>,
    /// The triangulated tracks.
    pub points: Vec<SparsePoint>,
    /// The pair that defines the frame.
    pub seed: PairId,
    /// The intrinsics the cameras were placed with, with the focal length refined if
    /// adjusting during registration refined it.
    pub intrinsics: Intrinsics,
    /// How each view other than the seed pair was registered, in the order they were.
    pub registrations: Vec<Registration>,
    /// Anything that went less than well.
    pub warnings: Vec<Warning>,
}

impl SparseModel {
    /// The views that have a pose.
    pub fn registered(&self) -> Vec<ViewId> {
        self.cameras
            .iter()
            .enumerate()
            .filter(|(_, c)| c.is_some())
            .map(|(v, _)| ViewId(v as u32))
            .collect()
    }
}

/// Registers the views of `tracks` into one frame, starting from the best usable pair in
/// `relative`.
///
/// `precision_px` is the mappings' typical localization error, which thresholds scale
/// with.
///
/// # Errors
/// [`Error::NoUsablePair`] if no pair can seed the reconstruction;
/// [`Error::RegistrationFailed`] if fewer than [`MIN_VIEWS`] views could be registered.
pub fn reconstruct(
    relative: &[RelativePose],
    tracks: &[Track],
    intrinsics: &Intrinsics,
    views: usize,
    precision_px: f64,
    params: &Params,
    rng: &mut Rng,
) -> Result<SparseModel, Error> {
    reconstruct_with_progress(
        relative,
        tracks,
        intrinsics,
        views,
        precision_px,
        params,
        rng,
        &mut silent,
    )
}

/// [`reconstruct`], reporting each view as it is placed through `on_event`.
///
/// # Errors
/// As [`reconstruct`], plus [`Error::Cancelled`] if `on_event` asks the run to stop.
#[allow(clippy::too_many_arguments)]
pub fn reconstruct_with_progress(
    relative: &[RelativePose],
    tracks: &[Track],
    intrinsics: &Intrinsics,
    views: usize,
    precision_px: f64,
    params: &Params,
    rng: &mut Rng,
    on_event: &mut dyn FnMut(Event) -> Flow,
) -> Result<SparseModel, Error> {
    let mut intrinsics = *intrinsics;
    let mut threshold = params.max_reprojection * precision_px / mean_focal(&intrinsics);
    let min_angle = params.min_point_angle_deg.to_radians();

    let seed = relative
        .iter()
        .filter(|r| r.is_usable() && r.pair.b().index() < views)
        .max_by(|a, b| seed_score(a).total_cmp(&seed_score(b)))
        .ok_or_else(|| Error::NoUsablePair {
            reasons: relative
                .iter()
                .filter_map(|r| match &r.verdict {
                    Verdict::Degenerate(reason) => Some((r.pair, reason.to_string())),
                    Verdict::Usable => None,
                })
                .collect(),
        })?;

    let mut cameras: Vec<Option<Pose>> = vec![None; views];
    cameras[seed.pair.a().index()] = Some(Pose::identity());
    cameras[seed.pair.b().index()] = Some(seed.pose);
    report(
        on_event,
        Stage::Registration,
        2.0 / views as f32,
        format!("{} defines the frame; 2 of {views} views placed", seed.pair),
    )?;

    let mut normalized = normalize(tracks, &intrinsics);
    let mut points: Vec<Option<World>> = vec![None; tracks.len()];
    triangulate_missing(&normalized, &cameras, &mut points, threshold, min_angle);

    let mut registrations = Vec::new();
    let mut warnings = Vec::new();
    // Every registration and every adjustment changes the model. A view that could not be
    // placed is tried again once it has, but never twice against the same model, so the
    // loop ends: both kinds of change are bounded by the number of views.
    let mut version = 0usize;
    let mut tried_at: Vec<Option<usize>> = vec![None; views];
    let mut reasons: Vec<Option<String>> = vec![None; views];
    let mut adjusted_at = version;
    let mut placed_at_adjustment = 2usize;
    loop {
        // The view not yet tried against this model that sees the most points.
        let candidate = (0..views)
            .filter(|&v| cameras[v].is_none() && tried_at[v] != Some(version))
            .map(|v| (v, observed(&normalized, &points, ViewId(v as u32))))
            .max_by_key(|&(v, count)| (count, std::cmp::Reverse(v)));
        let Some((v, count)) = candidate else {
            // Every view left is stuck. Adjusting what is placed may remove the drift that
            // kept them out; if it changes the model, try them all again.
            let stuck = cameras.iter().any(Option::is_none);
            if stuck
                && adjusted_at != version
                && adjust_placed(
                    tracks,
                    seed.pair,
                    precision_px,
                    params,
                    &mut cameras,
                    &mut points,
                    &mut intrinsics,
                    on_event,
                )?
            {
                normalized = normalize(tracks, &intrinsics);
                threshold = params.max_reprojection * precision_px / mean_focal(&intrinsics);
                triangulate_missing(&normalized, &cameras, &mut points, threshold, min_angle);
                version += 1;
                adjusted_at = version;
                continue;
            }
            break;
        };
        let view = ViewId(v as u32);
        tried_at[v] = Some(version);
        let placed = cameras.iter().flatten().count();

        let attempt = if count < params.min_registration_points {
            Err(format!(
                "it sees only {count} of the points placed so far (needs {})",
                params.min_registration_points
            ))
        } else {
            let (world, image): (Vec<Point3<f64>>, Vec<Norm>) = normalized
                .iter()
                .zip(&points)
                .filter_map(|(obs, point)| {
                    let point = (*point)?;
                    let &(_, n) = obs.iter().find(|(o, _)| *o == view)?;
                    Some((point.0, n))
                })
                .unzip();
            match pnp::ransac(
                &world,
                &image,
                threshold,
                params.max_iterations,
                params.confidence,
                rng,
            ) {
                None => Err("no camera position fits the points it sees".to_string()),
                Some(solution) => {
                    let inliers = solution.inliers.len();
                    let ratio = inliers as f64 / count as f64;
                    let needed = if inliers >= params.enough_pnp_inliers {
                        params.min_pnp_inlier_ratio_when_enough
                    } else {
                        params.min_pnp_inlier_ratio
                    };
                    if ratio >= needed {
                        Ok((solution, ratio))
                    } else if inliers >= params.enough_pnp_inliers {
                        Err(format!(
                            "only {:.0}% of the {count} points it sees agree on where it was (needs {:.0}%)",
                            ratio * 100.0,
                            needed * 100.0
                        ))
                    } else {
                        Err(format!(
                            "only {inliers} ({:.0}%) of the {count} points it sees agree on where it was (needs {:.0}%, or {} points and {:.0}%)",
                            ratio * 100.0,
                            needed * 100.0,
                            params.enough_pnp_inliers,
                            params.min_pnp_inlier_ratio_when_enough * 100.0
                        ))
                    }
                }
            }
        };
        let (solution, inlier_ratio) = match attempt {
            Ok(found) => found,
            Err(reason) => {
                report(
                    on_event,
                    Stage::Registration,
                    placed as f32 / views as f32,
                    format!("{view} not placed yet: {reason}"),
                )?;
                reasons[v] = Some(reason);
                continue;
            }
        };

        cameras[v] = Some(solution.pose);
        reasons[v] = None;
        registrations.push(Registration {
            view,
            candidates: count,
            inliers: solution.inliers.len(),
            inlier_ratio,
        });
        let placed = cameras.iter().flatten().count();
        report(
            on_event,
            Stage::Registration,
            placed as f32 / views as f32,
            format!(
                "placed {view} on {} of the {count} points it sees; {placed} of {views} views placed",
                solution.inliers.len()
            ),
        )?;
        version += 1;
        triangulate_missing(&normalized, &cameras, &mut points, threshold, min_angle);

        // Adjust whenever the number of placed views has grown by half, so the drift a
        // registration inherits stays small without adjusting after every view.
        if 2 * placed >= 3 * placed_at_adjustment {
            placed_at_adjustment = placed;
            adjusted_at = version;
            if adjust_placed(
                tracks,
                seed.pair,
                precision_px,
                params,
                &mut cameras,
                &mut points,
                &mut intrinsics,
                on_event,
            )? {
                normalized = normalize(tracks, &intrinsics);
                threshold = params.max_reprojection * precision_px / mean_focal(&intrinsics);
                triangulate_missing(&normalized, &cameras, &mut points, threshold, min_angle);
                version += 1;
                adjusted_at = version;
            }
        }
    }

    for (v, reason) in reasons.into_iter().enumerate() {
        let (None, Some(reason)) = (cameras[v], reason) else {
            continue;
        };
        let view = ViewId(v as u32);
        emit(
            on_event,
            Event::ViewDropped {
                stage: Stage::Registration,
                view,
                reason: reason.clone(),
            },
        )?;
        warnings.push(Warning::Unregistered { view, reason });
    }

    let registered: Vec<ViewId> = (0..views)
        .filter(|&v| cameras[v].is_some())
        .map(|v| ViewId(v as u32))
        .collect();
    if registered.len() < MIN_VIEWS {
        return Err(Error::RegistrationFailed {
            registered,
            minimum: MIN_VIEWS,
            left_out: warnings
                .into_iter()
                .filter_map(|warning| match warning {
                    Warning::Unregistered { view, reason } => Some((view, reason)),
                    Warning::CollinearCameras { .. } => None,
                })
                .collect(),
        });
    }

    let focal = mean_focal(&intrinsics);

    let points = normalized
        .iter()
        .enumerate()
        .filter_map(|(track, obs)| {
            let t = triangulate_track(obs, &cameras, threshold, min_angle)?;
            Some(SparsePoint {
                position: t.point,
                track,
                observations: t.observations,
                error_px: t.mean_error * focal,
                angle_deg: t.angle.to_degrees(),
            })
        })
        .collect();

    let centres: Vec<Point3<f64>> = cameras.iter().flatten().map(Pose::centre).collect();
    if let Some(deviation) = collinearity(&centres) {
        if deviation < params.min_camera_spread {
            warnings.push(Warning::CollinearCameras { deviation });
        }
    }

    Ok(SparseModel {
        cameras,
        points,
        seed: seed.pair,
        intrinsics,
        registrations,
        warnings,
    })
}

fn mean_focal(intrinsics: &Intrinsics) -> f64 {
    (intrinsics.fx + intrinsics.fy) / 2.0
}

/// Every track's observations in normalized camera coordinates.
fn normalize(tracks: &[Track], intrinsics: &Intrinsics) -> Vec<Vec<(ViewId, Norm)>> {
    tracks
        .iter()
        .map(|t| {
            t.observations
                .iter()
                .map(|&(v, p)| (v, intrinsics.normalize(p)))
                .collect()
        })
        .collect()
}

/// How many of the points triangulated so far `view` observes.
fn observed(normalized: &[Vec<(ViewId, Norm)>], points: &[Option<World>], view: ViewId) -> usize {
    normalized
        .iter()
        .zip(points)
        .filter(|(obs, point)| point.is_some() && obs.iter().any(|(o, _)| *o == view))
        .count()
}

/// Bundle adjustment of the views placed so far, written back into `cameras`, `points`
/// and `intrinsics`. Points it drops are cleared, for the caller to triangulate again.
/// `false`, with nothing changed, if the adjustment did not improve the fit.
#[allow(clippy::too_many_arguments)]
fn adjust_placed(
    tracks: &[Track],
    seed: PairId,
    precision_px: f64,
    params: &Params,
    cameras: &mut [Option<Pose>],
    points: &mut [Option<World>],
    intrinsics: &mut Intrinsics,
    on_event: &mut dyn FnMut(Event) -> Flow,
) -> Result<bool, Error> {
    let model = SparseModel {
        cameras: cameras.to_vec(),
        points: points
            .iter()
            .enumerate()
            .filter_map(|(track, point)| {
                Some(SparsePoint {
                    position: (*point)?,
                    track,
                    observations: 0,
                    error_px: 0.0,
                    angle_deg: 0.0,
                })
            })
            .collect(),
        seed,
        intrinsics: *intrinsics,
        registrations: Vec::new(),
        warnings: Vec::new(),
    };
    let Ok(adjusted) = ba::adjust(&model, tracks, intrinsics, precision_px, &params.adjust) else {
        return Ok(false);
    };
    let placed = cameras.iter().flatten().count();
    let mut message = format!(
        "adjusted the {placed} views placed so far: median reprojection error {:.2} px → {:.2} px",
        adjusted.report.initial_median_px, adjusted.report.final_median_px
    );
    if params.adjust.refine_focal {
        message += &format!(
            "; focal length {:.1} px → {:.1} px",
            intrinsics.fx, adjusted.intrinsics.fx
        );
    }
    report(
        on_event,
        Stage::Registration,
        placed as f32 / cameras.len() as f32,
        message,
    )?;

    cameras.copy_from_slice(&adjusted.model.cameras);
    points.fill(None);
    for point in &adjusted.model.points {
        points[point.track] = Some(point.position);
    }
    // The focal length's source stays as it was: the final adjustment decides from it
    // whether to refine the focal length further.
    *intrinsics = Intrinsics {
        source: intrinsics.source,
        ..adjusted.intrinsics
    };
    Ok(true)
}

/// Prefer pairs that both overlap well and see the scene from well-separated positions.
fn seed_score(relative: &RelativePose) -> f64 {
    relative.coverage as f64 * relative.median_angle_deg
}

struct Triangulated {
    point: World,
    observations: usize,
    mean_error: f64,
    angle: f64,
}

/// A track's point over every registered view that observes it, if it passes the
/// reprojection and ray-angle checks.
fn triangulate_track(
    observations: &[(ViewId, Norm)],
    cameras: &[Option<Pose>],
    threshold: f64,
    min_angle: f64,
) -> Option<Triangulated> {
    let observations: Vec<(Pose, Norm)> = observations
        .iter()
        .filter_map(|&(v, n)| Some((cameras.get(v.index()).copied().flatten()?, n)))
        .collect();
    if observations.len() < 2 {
        return None;
    }
    let initial = triangulate::dlt(&observations)?;
    let point = triangulate::refine(&observations, initial, 10);
    if !triangulate::in_front(&observations, &point) {
        return None;
    }
    let errors: Vec<f64> = triangulate::reprojection_errors(&observations, &point).collect();
    if errors.iter().any(|&e| e.is_nan() || e > threshold) {
        return None;
    }

    let centres: Vec<Point3<f64>> = observations.iter().map(|(pose, _)| pose.centre()).collect();
    let mut angle = 0.0f64;
    for (i, a) in centres.iter().enumerate() {
        for b in &centres[i + 1..] {
            angle = angle.max(triangulate::triangulation_angle(a, b, &point));
        }
    }
    if angle < min_angle {
        return None;
    }

    Some(Triangulated {
        point,
        observations: observations.len(),
        mean_error: errors.iter().sum::<f64>() / errors.len() as f64,
        angle,
    })
}

fn triangulate_missing(
    normalized: &[Vec<(ViewId, Norm)>],
    cameras: &[Option<Pose>],
    points: &mut [Option<World>],
    threshold: f64,
    min_angle: f64,
) {
    for (obs, slot) in normalized.iter().zip(points.iter_mut()) {
        if slot.is_none() {
            *slot = triangulate_track(obs, cameras, threshold, min_angle).map(|t| t.point);
        }
    }
}

/// How far the centres stray from the line through the two farthest of them, as a
/// fraction of the distance between those two. `None` for fewer than three centres.
fn collinearity(centres: &[Point3<f64>]) -> Option<f64> {
    if centres.len() < 3 {
        return None;
    }
    let mut farthest = (0, 1, 0.0);
    for (i, a) in centres.iter().enumerate() {
        for (j, b) in centres.iter().enumerate().skip(i + 1) {
            let d = (a - b).norm();
            if d > farthest.2 {
                farthest = (i, j, d);
            }
        }
    }
    let (i, j, length) = farthest;
    if length <= 0.0 {
        return None;
    }
    let direction = (centres[j] - centres[i]) / length;
    let deviation = centres
        .iter()
        .map(|c| {
            let offset = c - centres[i];
            (offset - direction * offset.dot(&direction)).norm()
        })
        .fold(0.0, f64::max);
    Some(deviation / length)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn measures_how_far_cameras_stray_from_a_line() {
        let line = [
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(2.0, 0.01, 0.0),
        ];
        assert!(collinearity(&line).unwrap() < 0.01);
        let arc = [
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.3, 0.0),
            Point3::new(2.0, 0.0, 0.0),
        ];
        assert!((collinearity(&arc).unwrap() - 0.15).abs() < 1e-9);
        assert!(collinearity(&arc[..2]).is_none());
    }
}
