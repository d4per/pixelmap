//! Stage 5: bundle adjustment. Every camera and every point is refined together to
//! minimize reprojection error.
//!
//! Registration places each camera against the points that existed when it was added, so
//! errors compound. A seed pair whose baseline direction is a degree off distorts every
//! point triangulated from it, and every camera registered against those points inherits
//! the distortion. Adjusting everything jointly removes that.
//!
//! The problem is small by structure-from-motion standards: a handful of cameras and some
//! thousands of points. So it is solved directly:
//!
//! - **Levenberg–Marquardt** on reprojection error in photo pixels.
//! - **Schur complement.** Each point only couples to the cameras that see it, so the
//!   normal equations have a 3 × 3 block per point. Those are eliminated, the small dense
//!   camera system is solved by Cholesky, and the points are back-substituted.
//! - **Gauge.** A reconstruction can be moved, turned and scaled without changing a single
//!   reprojection. The seed pair's first camera is held fixed, which pins position and
//!   orientation. The largest translation component of its second camera is held fixed
//!   too, which pins scale.
//! - **Robust loss.** Huber, so that the observations that survived the earlier filters but
//!   are still wrong pull on the solution linearly rather than quadratically. Each track is
//!   also weighted by how well its views agreed with each other.
//! - **Two rounds.** Optimize, drop observations still far off, optimize again.
//! - **Focal length** is optionally a shared parameter, a scale on `fx` and `fy`. Worth
//!   enabling when the focal length was estimated rather than read from EXIF: a wrong
//!   focal length does not add noise to a model, it skews it.

use nalgebra::{
    DMatrix, DVector, Matrix2x3, Matrix3, Matrix3x6, Point3, Rotation3, SMatrix, Vector2, Vector3,
};

use crate::calib::{FocalSource, Intrinsics};
use crate::error::Error;
use crate::event::{report, silent, Event, Flow, Stage};
use crate::pose::Pose;
use crate::sfm::{SparseModel, SparsePoint};
use crate::tracks::Track;
use crate::triangulate;
use crate::twoview::median;
use crate::types::World;

/// Tuning for [`adjust`].
#[derive(Clone, Debug)]
pub struct Params {
    /// Levenberg–Marquardt iterations per round.
    pub max_iterations: usize,
    /// A round ends when an accepted step lowers the cost by less than this fraction.
    pub tolerance: f64,
    /// Where the Huber loss turns from quadratic to linear, as a multiple of the
    /// mappings' precision.
    pub huber: f64,
    /// Between rounds, observations further off than this, as a multiple of the
    /// precision, are dropped.
    pub outlier: f64,
    /// Rounds of optimization, with outlier removal in between.
    pub rounds: usize,
    /// Whether to refine the focal length.
    pub refine_focal: bool,
    /// The largest acceptable median reprojection error afterwards, as a multiple of the
    /// precision.
    pub max_median_error: f64,
}

impl Default for Params {
    fn default() -> Self {
        Params {
            max_iterations: 50,
            tolerance: 1e-7,
            huber: 1.0,
            outlier: 2.0,
            rounds: 2,
            refine_focal: false,
            max_median_error: 1.0,
        }
    }
}

/// What bundle adjustment did.
#[derive(Clone, Debug, PartialEq)]
pub struct Report {
    /// Levenberg–Marquardt iterations run, over all rounds.
    pub iterations: usize,
    /// Whether the last round converged rather than running out of iterations.
    pub converged: bool,
    /// Median reprojection error over all observations before, in photo pixels.
    pub initial_median_px: f64,
    /// And after.
    pub final_median_px: f64,
    /// Observations dropped as outliers between rounds.
    pub removed_observations: usize,
    /// Points dropped because fewer than two observations were left.
    pub removed_points: usize,
}

/// A model after bundle adjustment.
#[derive(Clone, Debug)]
pub struct Adjusted {
    /// The refined cameras and points.
    pub model: SparseModel,
    /// The intrinsics, with the focal length refined if that was asked for.
    pub intrinsics: Intrinsics,
    /// What happened.
    pub report: Report,
}

/// Refines every camera and point of `model` together.
///
/// `precision_px` is the mappings' typical localization error, which the thresholds in
/// `params` scale with.
///
/// # Errors
/// [`Error::BundleAdjustment`] if the fit afterwards is worse than before or worse than
/// `params.max_median_error` allows. Either one means an earlier stage placed a camera
/// wrongly.
pub fn adjust(
    model: &SparseModel,
    tracks: &[Track],
    intrinsics: &Intrinsics,
    precision_px: f64,
    params: &Params,
) -> Result<Adjusted, Error> {
    adjust_with_progress(model, tracks, intrinsics, precision_px, params, &mut silent)
}

/// [`adjust`], reporting each round through `on_event`.
///
/// # Errors
/// As [`adjust`], plus [`Error::Cancelled`] if `on_event` asks the run to stop.
pub fn adjust_with_progress(
    model: &SparseModel,
    tracks: &[Track],
    intrinsics: &Intrinsics,
    precision_px: f64,
    params: &Params,
    on_event: &mut dyn FnMut(Event) -> Flow,
) -> Result<Adjusted, Error> {
    let mut slot_of = vec![None; model.cameras.len()];
    let mut poses = Vec::new();
    for (view, camera) in model.cameras.iter().enumerate() {
        if let Some(pose) = camera {
            slot_of[view] = Some(poses.len());
            poses.push(*pose);
        }
    }
    let seed_a = slot_of[model.seed.a().index()].expect("the seed pair is registered");
    let seed_b = slot_of[model.seed.b().index()].expect("the seed pair is registered");

    let mut points = Vec::with_capacity(model.points.len());
    let mut observations = Vec::with_capacity(model.points.len());
    let mut weights = Vec::with_capacity(model.points.len());
    let mut track_of = Vec::with_capacity(model.points.len());
    for point in &model.points {
        let track = &tracks[point.track];
        let seen: Vec<Observation> = track
            .observations
            .iter()
            .filter_map(|&(view, p)| {
                Some(Observation {
                    slot: slot_of.get(view.index()).copied().flatten()?,
                    u: p.x() as f64,
                    v: p.y() as f64,
                })
            })
            .collect();
        if seen.len() < 2 {
            continue;
        }
        points.push(point.position.0);
        observations.push(seen);
        weights.push(1.0 / (1.0 + track.residual as f64 / precision_px));
        track_of.push(point.track);
    }

    let mut parameters = 0;
    let fixed_component = poses[seed_b].translation.iamax();
    let camera_index: Vec<[Option<usize>; 6]> = (0..poses.len())
        .map(|slot| {
            std::array::from_fn(|k| {
                let fixed = slot == seed_a || (slot == seed_b && k == 3 + fixed_component);
                (!fixed).then(|| {
                    parameters += 1;
                    parameters - 1
                })
            })
        })
        .collect();
    let focal_index = params.refine_focal.then(|| {
        parameters += 1;
        parameters - 1
    });

    let mut problem = Problem {
        poses,
        points,
        observations,
        weights,
        camera: Camera {
            fx: intrinsics.fx,
            fy: intrinsics.fy,
            cx: intrinsics.cx,
            cy: intrinsics.cy,
        },
        scale: 1.0,
        camera_index,
        focal_index,
        parameters,
        huber: params.huber * precision_px,
    };

    let initial_median_px = median(problem.errors());
    let mut iterations = 0;
    let mut converged = false;
    let mut removed_observations = 0;
    let rounds = params.rounds.max(1);
    for round in 0..rounds {
        let (ran, done) = problem.optimize(params.max_iterations, params.tolerance);
        iterations += ran;
        converged = done;
        report(
            on_event,
            Stage::BundleAdjustment,
            (round + 1) as f32 / rounds as f32,
            format!(
                "round {} of {rounds}: {ran} iterations, median error {:.2} px",
                round + 1,
                median(problem.errors())
            ),
        )?;
        if round + 1 == rounds {
            break;
        }
        let removed = problem.remove_outliers(params.outlier * precision_px);
        removed_observations += removed;
        if removed == 0 {
            break;
        }
    }
    let final_median_px = median(problem.errors());

    let required_px = params.max_median_error * precision_px;
    let worse = final_median_px > initial_median_px + 0.01 * precision_px;
    if final_median_px.is_nan() || final_median_px > required_px || worse {
        return Err(Error::BundleAdjustment {
            initial_median_px,
            median_px: final_median_px,
            required_px,
        });
    }

    let mut cameras = model.cameras.clone();
    for (view, slot) in slot_of.iter().enumerate() {
        if let Some(slot) = slot {
            cameras[view] = Some(problem.poses[*slot]);
        }
    }

    let camera = problem.camera;
    let mut removed_points = 0;
    let mut adjusted_points = Vec::with_capacity(problem.points.len());
    for (i, position) in problem.points.iter().enumerate() {
        let seen = &problem.observations[i];
        if seen.len() < 2 {
            removed_points += 1;
            continue;
        }
        let errors: Vec<f64> = seen
            .iter()
            .map(|o| camera.error(&problem.poses[o.slot], position, problem.scale, o))
            .collect();
        let centres: Vec<Point3<f64>> = seen
            .iter()
            .map(|o| problem.poses[o.slot].centre())
            .collect();
        let world = World(*position);
        let mut angle = 0.0f64;
        for (k, a) in centres.iter().enumerate() {
            for b in &centres[k + 1..] {
                angle = angle.max(triangulate::triangulation_angle(a, b, &world));
            }
        }
        adjusted_points.push(SparsePoint {
            position: world,
            track: track_of[i],
            observations: seen.len(),
            error_px: errors.iter().sum::<f64>() / errors.len() as f64,
            angle_deg: angle.to_degrees(),
        });
    }

    let refined = Intrinsics {
        fx: intrinsics.fx * problem.scale,
        fy: intrinsics.fy * problem.scale,
        source: if params.refine_focal {
            FocalSource::Refined
        } else {
            intrinsics.source
        },
        ..*intrinsics
    };

    Ok(Adjusted {
        model: SparseModel {
            cameras,
            points: adjusted_points,
            seed: model.seed,
            intrinsics: refined,
            registrations: model.registrations.clone(),
            warnings: model.warnings.clone(),
        },
        intrinsics: refined,
        report: Report {
            iterations,
            converged,
            initial_median_px,
            final_median_px,
            removed_observations,
            removed_points,
        },
    })
}

/// The reprojection error assigned to a point behind a camera, in pixels.
const BEHIND_PX: f64 = 1e4;

/// One point seen by one camera.
#[derive(Copy, Clone, Debug)]
struct Observation {
    /// The camera's position in [`Problem::poses`].
    slot: usize,
    u: f64,
    v: f64,
}

#[derive(Copy, Clone, Debug)]
struct Camera {
    fx: f64,
    fy: f64,
    cx: f64,
    cy: f64,
}

impl Camera {
    /// Where `point` lands in a camera at `pose`, in pixels, if it is in front of it.
    fn project(&self, pose: &Pose, point: &Point3<f64>, scale: f64) -> Option<(f64, f64)> {
        let c = pose.to_camera(point);
        (c.z > 1e-9).then(|| {
            (
                scale * self.fx * c.x / c.z + self.cx,
                scale * self.fy * c.y / c.z + self.cy,
            )
        })
    }

    fn error(&self, pose: &Pose, point: &Point3<f64>, scale: f64, o: &Observation) -> f64 {
        self.project(pose, point, scale)
            .map_or(BEHIND_PX, |(x, y)| (x - o.u).hypot(y - o.v))
    }
}

/// The normal equations of one point: its own block, its gradient, and its coupling to
/// each camera parameter it depends on.
struct PointBlock {
    v: Matrix3<f64>,
    g: Vector3<f64>,
    w: Vec<(usize, Vector3<f64>)>,
}

struct Normal {
    u: DMatrix<f64>,
    g: DVector<f64>,
    blocks: Vec<PointBlock>,
}

struct Problem {
    poses: Vec<Pose>,
    points: Vec<Point3<f64>>,
    observations: Vec<Vec<Observation>>,
    weights: Vec<f64>,
    camera: Camera,
    /// The multiplier on the focal length being refined.
    scale: f64,
    /// Each camera's parameter indices, `None` where held fixed: rotation update, then
    /// translation.
    camera_index: Vec<[Option<usize>; 6]>,
    focal_index: Option<usize>,
    parameters: usize,
    huber: f64,
}

impl Problem {
    fn errors(&self) -> Vec<f64> {
        self.points
            .iter()
            .zip(&self.observations)
            .filter(|(_, seen)| seen.len() >= 2)
            .flat_map(|(point, seen)| {
                seen.iter()
                    .map(|o| self.camera.error(&self.poses[o.slot], point, self.scale, o))
            })
            .collect()
    }

    fn rho(&self, error: f64) -> f64 {
        if error <= self.huber {
            error * error
        } else {
            2.0 * self.huber * error - self.huber * self.huber
        }
    }

    fn cost(&self, poses: &[Pose], points: &[Point3<f64>], scale: f64) -> f64 {
        points
            .iter()
            .zip(&self.observations)
            .zip(&self.weights)
            .map(|((point, seen), weight)| {
                weight
                    * seen
                        .iter()
                        .map(|o| self.rho(self.camera.error(&poses[o.slot], point, scale, o)))
                        .sum::<f64>()
            })
            .sum()
    }

    fn normal_equations(&self) -> Normal {
        let n = self.parameters;
        let mut u = DMatrix::zeros(n, n);
        let mut g = DVector::zeros(n);
        let mut blocks = Vec::with_capacity(self.points.len());

        for ((point, seen), weight) in self
            .points
            .iter()
            .zip(&self.observations)
            .zip(&self.weights)
        {
            let mut block = PointBlock {
                v: Matrix3::zeros(),
                g: Vector3::zeros(),
                w: Vec::new(),
            };
            if seen.len() < 2 {
                blocks.push(block);
                continue;
            }
            for o in seen {
                let pose = &self.poses[o.slot];
                let rotated = pose.rotation * point.coords;
                let c = rotated + pose.translation;
                if c.z <= 1e-9 {
                    continue;
                }
                let (sx, sy) = (self.scale * self.camera.fx, self.scale * self.camera.fy);
                let iz = 1.0 / c.z;
                let r = Vector2::new(
                    sx * c.x * iz + self.camera.cx - o.u,
                    sy * c.y * iz + self.camera.cy - o.v,
                );
                let projection = Matrix2x3::new(
                    sx * iz,
                    0.0,
                    -sx * c.x * iz * iz,
                    0.0,
                    sy * iz,
                    -sy * c.y * iz * iz,
                );
                // c = exp(ω)·R·X + t, so ∂c/∂ω = −[R·X]× and ∂c/∂t = I.
                let mut motion = Matrix3x6::<f64>::zeros();
                motion
                    .fixed_view_mut::<3, 3>(0, 0)
                    .copy_from(&(-rotated.cross_matrix()));
                motion
                    .fixed_view_mut::<3, 3>(0, 3)
                    .copy_from(&Matrix3::identity());
                let mut jc = SMatrix::<f64, 2, 7>::zeros();
                jc.fixed_view_mut::<2, 6>(0, 0)
                    .copy_from(&(projection * motion));
                jc[(0, 6)] = self.camera.fx * c.x * iz;
                jc[(1, 6)] = self.camera.fy * c.y * iz;
                let jp = projection * pose.rotation.matrix();

                let e = r.norm();
                let w = weight * if e <= self.huber { 1.0 } else { self.huber / e };

                let mut index = [None; 7];
                index[..6].copy_from_slice(&self.camera_index[o.slot]);
                index[6] = self.focal_index;
                for a in 0..7 {
                    let Some(ia) = index[a] else { continue };
                    let ca = jc.column(a);
                    g[ia] += w * ca.dot(&r);
                    for (b, ib) in index.iter().enumerate() {
                        if let Some(ib) = ib {
                            u[(ia, *ib)] += w * ca.dot(&jc.column(b));
                        }
                    }
                    let coupling = (ca.transpose() * jp).transpose() * w;
                    match block.w.iter_mut().find(|(i, _)| *i == ia) {
                        Some((_, existing)) => *existing += coupling,
                        None => block.w.push((ia, coupling)),
                    }
                }
                block.v += jp.transpose() * jp * w;
                block.g += jp.transpose() * r * w;
            }
            blocks.push(block);
        }
        Normal { u, g, blocks }
    }

    /// The damped step: camera parameters from the reduced system, then each point by
    /// back-substitution.
    fn solve(&self, normal: &Normal, lambda: f64) -> Option<(DVector<f64>, Vec<Vector3<f64>>)> {
        let n = self.parameters;
        let mut s = normal.u.clone();
        for i in 0..n {
            s[(i, i)] += lambda * normal.u[(i, i)] + 1e-9;
        }
        let mut rhs = -normal.g.clone();
        let mut inverses = Vec::with_capacity(normal.blocks.len());
        for block in &normal.blocks {
            let mut v = block.v;
            for k in 0..3 {
                v[(k, k)] += lambda * block.v[(k, k)] + 1e-9;
            }
            let Some(inverse) = v.try_inverse() else {
                inverses.push(None);
                continue;
            };
            let inverse_g = inverse * block.g;
            for (a, wa) in &block.w {
                let wa_inverse = inverse * wa;
                rhs[*a] += wa.dot(&inverse_g);
                for (b, wb) in &block.w {
                    s[(*a, *b)] -= wa_inverse.dot(wb);
                }
            }
            inverses.push(Some(inverse));
        }

        let dc = if n == 0 {
            DVector::zeros(0)
        } else {
            s.cholesky()?.solve(&rhs)
        };
        let dp = normal
            .blocks
            .iter()
            .zip(&inverses)
            .map(|(block, inverse)| match inverse {
                Some(inverse) => {
                    let mut rhs = -block.g;
                    for (a, wa) in &block.w {
                        rhs -= wa * dc[*a];
                    }
                    inverse * rhs
                }
                None => Vector3::zeros(),
            })
            .collect();
        Some((dc, dp))
    }

    fn stepped(
        &self,
        dc: &DVector<f64>,
        dp: &[Vector3<f64>],
    ) -> (Vec<Pose>, Vec<Point3<f64>>, f64) {
        let delta = |index: Option<usize>| index.map_or(0.0, |i| dc[i]);
        let poses = self
            .poses
            .iter()
            .zip(&self.camera_index)
            .map(|(pose, index)| {
                let d: [f64; 6] = std::array::from_fn(|k| delta(index[k]));
                let mut rotation = Rotation3::new(Vector3::new(d[0], d[1], d[2])) * pose.rotation;
                rotation.renormalize();
                Pose {
                    rotation,
                    translation: pose.translation + Vector3::new(d[3], d[4], d[5]),
                }
            })
            .collect();
        let points = self.points.iter().zip(dp).map(|(p, d)| p + d).collect();
        (poses, points, self.scale + delta(self.focal_index))
    }

    /// Levenberg–Marquardt until the cost stops falling. Returns the iterations run and
    /// whether it converged.
    fn optimize(&mut self, max_iterations: usize, tolerance: f64) -> (usize, bool) {
        let mut lambda = 1e-4;
        let mut current = self.cost(&self.poses, &self.points, self.scale);
        for iteration in 1..=max_iterations {
            let normal = self.normal_equations();
            loop {
                if lambda > 1e10 {
                    // No step lowers the cost any more: at a minimum.
                    return (iteration, true);
                }
                let Some((dc, dp)) = self.solve(&normal, lambda) else {
                    lambda *= 10.0;
                    continue;
                };
                let (poses, points, scale) = self.stepped(&dc, &dp);
                let cost = self.cost(&poses, &points, scale);
                if cost < current {
                    let decrease = (current - cost) / current;
                    self.poses = poses;
                    self.points = points;
                    self.scale = scale;
                    current = cost;
                    lambda = (lambda / 10.0).max(1e-12);
                    if decrease < tolerance {
                        return (iteration, true);
                    }
                    break;
                }
                lambda *= 10.0;
            }
        }
        (max_iterations, false)
    }

    /// Drops observations whose reprojection error exceeds `threshold` pixels. Returns
    /// how many.
    fn remove_outliers(&mut self, threshold: f64) -> usize {
        let (camera, poses, scale) = (self.camera, &self.poses, self.scale);
        let mut removed = 0;
        for (point, seen) in self.points.iter().zip(self.observations.iter_mut()) {
            let before = seen.len();
            seen.retain(|o| camera.error(&poses[o.slot], point, scale, o) <= threshold);
            removed += before - seen.len();
        }
        removed
    }
}
