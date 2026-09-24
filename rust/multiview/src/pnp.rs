//! Camera pose from points whose 3D positions are already known (Perspective-n-Point).
//!
//! A normalized DLT on six points inside RANSAC, then Levenberg–Marquardt on the
//! reprojection error of the inliers. The DLT is degenerate when every point lies on one
//! plane. EPnP would handle that case, and can replace [`dlt`] without changing anything
//! else here.

use nalgebra::{
    Matrix2x3, Matrix3, Matrix3x4, Matrix3x6, Matrix4, Matrix6, Point3, Rotation3, SMatrix,
    SVector, Vector2, Vector3, Vector6,
};

use crate::pose::Pose;
use crate::rng::Rng;
use crate::twoview::ransac_iterations;
use crate::types::Norm;

/// A camera pose and the correspondences that support it.
#[derive(Clone, Debug)]
pub struct Solution {
    /// The camera.
    pub pose: Pose,
    /// Indices of the correspondences within the threshold.
    pub inliers: Vec<usize>,
    /// RANSAC iterations run.
    pub iterations: usize,
}

/// Estimates the pose of a camera that sees `world[i]` at `image[i]`.
///
/// `threshold` is the largest reprojection error an inlier may have, in normalized
/// camera units. `None` with fewer than six correspondences, or when no sample gives a
/// pose.
pub fn ransac(
    world: &[Point3<f64>],
    image: &[Norm],
    threshold: f64,
    max_iterations: usize,
    confidence: f64,
    rng: &mut Rng,
) -> Option<Solution> {
    let n = world.len();
    if n < 6 || image.len() != n {
        return None;
    }
    let threshold_sq = threshold * threshold;

    let mut best: Option<(Pose, usize)> = None;
    let mut sample = [0usize; 6];
    let mut needed = max_iterations;
    let mut iterations = 0;
    while iterations < needed {
        iterations += 1;
        rng.sample_distinct(n, &mut sample);
        let Some(mut pose) = dlt(world, image, &sample) else {
            continue;
        };
        let mut count = count_inliers(&pose, world, image, threshold_sq);
        if best.is_some_and(|(_, c)| count <= c) {
            continue;
        }
        // A pose from six noisy points is rough; polish each new best on its own inliers
        // before judging how many iterations are still needed.
        let inliers = inlier_indices(&pose, world, image, threshold_sq);
        if inliers.len() >= 6 {
            let polished = refine(pose, world, image, &inliers, 5);
            let polished_count = count_inliers(&polished, world, image, threshold_sq);
            if polished_count > count {
                pose = polished;
                count = polished_count;
            }
        }
        best = Some((pose, count));
        needed = ransac_iterations(count as f64 / n as f64, 6, confidence).min(max_iterations);
    }

    let (mut pose, _) = best?;
    let mut inliers = inlier_indices(&pose, world, image, threshold_sq);
    for _ in 0..3 {
        if inliers.len() < 6 {
            break;
        }
        let refined = refine(pose, world, image, &inliers, 10);
        let refined_inliers = inlier_indices(&refined, world, image, threshold_sq);
        if refined_inliers.len() < inliers.len() {
            break;
        }
        pose = refined;
        inliers = refined_inliers;
    }
    Some(Solution {
        pose,
        inliers,
        iterations,
    })
}

/// The direct linear transform over the correspondences at `indices`, with both point
/// sets normalized first. Least squares when given more than six.
pub fn dlt(world: &[Point3<f64>], image: &[Norm], indices: &[usize]) -> Option<Pose> {
    if indices.len() < 6 {
        return None;
    }
    let count = indices.len() as f64;
    let centroid = indices
        .iter()
        .map(|&i| world[i].coords)
        .sum::<Vector3<f64>>()
        / count;
    let spread = indices
        .iter()
        .map(|&i| (world[i].coords - centroid).norm())
        .sum::<f64>()
        / count;
    let (cu, cv) = indices.iter().fold((0.0, 0.0), |(u, v), &i| {
        (u + image[i].x() / count, v + image[i].y() / count)
    });
    let image_spread = indices
        .iter()
        .map(|&i| ((image[i].x() - cu).powi(2) + (image[i].y() - cv).powi(2)).sqrt())
        .sum::<f64>()
        / count;
    if spread.is_nan() || spread <= 1e-12 || image_spread.is_nan() || image_spread <= 1e-12 {
        return None;
    }
    let s3 = 3f64.sqrt() / spread;
    let s2 = std::f64::consts::SQRT_2 / image_spread;

    let mut ata = SMatrix::<f64, 12, 12>::zeros();
    for &i in indices {
        let x = (world[i].coords - centroid) * s3;
        let u = (image[i].x() - cu) * s2;
        let v = (image[i].y() - cv) * s2;
        let r1 = SVector::<f64, 12>::from_column_slice(&[
            x.x,
            x.y,
            x.z,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            -u * x.x,
            -u * x.y,
            -u * x.z,
            -u,
        ]);
        let r2 = SVector::<f64, 12>::from_column_slice(&[
            0.0,
            0.0,
            0.0,
            0.0,
            x.x,
            x.y,
            x.z,
            1.0,
            -v * x.x,
            -v * x.y,
            -v * x.z,
            -v,
        ]);
        ata += r1 * r1.transpose() + r2 * r2.transpose();
    }

    let eigen = ata.symmetric_eigen();
    let column = eigen.eigenvectors.column(eigen.eigenvalues.imin());
    let entries: [f64; 12] = std::array::from_fn(|k| column[k]);
    let normalized = Matrix3x4::from_row_slice(&entries);

    // Undo both normalizations: P = T₂⁻¹ · P̂ · T₃.
    let image_denormalize = Matrix3::new(1.0 / s2, 0.0, cu, 0.0, 1.0 / s2, cv, 0.0, 0.0, 1.0);
    #[rustfmt::skip]
    let world_normalize = Matrix4::new(
        s3, 0.0, 0.0, -s3 * centroid.x,
        0.0, s3, 0.0, -s3 * centroid.y,
        0.0, 0.0, s3, -s3 * centroid.z,
        0.0, 0.0, 0.0, 1.0,
    );
    let mut p = image_denormalize * normalized * world_normalize;

    // P = λ[R | t] up to sign; a negative λ shows up as a left-handed M.
    if p.fixed_view::<3, 3>(0, 0).determinant() < 0.0 {
        p = -p;
    }
    let m: Matrix3<f64> = p.fixed_view::<3, 3>(0, 0).into_owned();
    let svd = m.svd(true, true);
    let (u, v_t) = (svd.u?, svd.v_t?);
    let scale = svd.singular_values.mean();
    if scale.is_nan() || scale <= 0.0 {
        return None;
    }
    let mut rotation = Rotation3::from_matrix_unchecked(u * v_t);
    rotation.renormalize();
    Some(Pose {
        rotation,
        translation: p.column(3).into_owned() / scale,
    })
}

/// Levenberg–Marquardt on the squared reprojection error of the correspondences at
/// `indices`, starting from `pose`.
pub fn refine(
    pose: Pose,
    world: &[Point3<f64>],
    image: &[Norm],
    indices: &[usize],
    iterations: usize,
) -> Pose {
    let cost = |pose: &Pose| {
        indices
            .iter()
            .map(|&i| error_sq(pose, &world[i], image[i]))
            .sum::<f64>()
    };
    let mut pose = pose;
    let mut current = cost(&pose);
    let mut lambda = 1e-3;

    for _ in 0..iterations {
        let mut h = Matrix6::<f64>::zeros();
        let mut g = Vector6::<f64>::zeros();
        for &i in indices {
            let rotated = pose.rotation * world[i].coords;
            let c = rotated + pose.translation;
            if c.z <= 1e-9 {
                continue;
            }
            let residual = Vector2::new(c.x / c.z - image[i].x(), c.y / c.z - image[i].y());
            let projection = Matrix2x3::new(
                1.0 / c.z,
                0.0,
                -c.x / (c.z * c.z),
                0.0,
                1.0 / c.z,
                -c.y / (c.z * c.z),
            );
            // c = exp(ω)·R·X + t, so ∂c/∂ω = −[R·X]× and ∂c/∂t = I.
            let mut motion = Matrix3x6::<f64>::zeros();
            motion
                .fixed_view_mut::<3, 3>(0, 0)
                .copy_from(&(-rotated.cross_matrix()));
            motion
                .fixed_view_mut::<3, 3>(0, 3)
                .copy_from(&Matrix3::identity());
            let j = projection * motion;
            h += j.transpose() * j;
            g += j.transpose() * residual;
        }

        let mut damped = h;
        for k in 0..6 {
            damped[(k, k)] += lambda * h[(k, k)] + 1e-12;
        }
        let Some(cholesky) = damped.cholesky() else {
            lambda *= 10.0;
            continue;
        };
        let delta = cholesky.solve(&-g);
        let candidate = Pose {
            rotation: Rotation3::new(delta.fixed_rows::<3>(0).into_owned()) * pose.rotation,
            translation: pose.translation + delta.fixed_rows::<3>(3),
        };
        let candidate_cost = cost(&candidate);
        if candidate_cost < current {
            pose = candidate;
            current = candidate_cost;
            lambda = (lambda / 10.0).max(1e-12);
            if delta.norm() < 1e-12 {
                break;
            }
        } else {
            lambda *= 10.0;
            if lambda > 1e8 {
                break;
            }
        }
    }
    pose
}

/// Squared reprojection error in normalized units. A point behind the camera counts as a
/// large, finite error, so that costs stay comparable.
fn error_sq(pose: &Pose, point: &Point3<f64>, observed: Norm) -> f64 {
    let c = pose.to_camera(point);
    if c.z <= 1e-9 {
        return 1.0;
    }
    (c.x / c.z - observed.x()).powi(2) + (c.y / c.z - observed.y()).powi(2)
}

fn count_inliers(pose: &Pose, world: &[Point3<f64>], image: &[Norm], threshold_sq: f64) -> usize {
    world
        .iter()
        .zip(image)
        .filter(|(p, n)| error_sq(pose, p, **n) < threshold_sq)
        .count()
}

fn inlier_indices(
    pose: &Pose,
    world: &[Point3<f64>],
    image: &[Norm],
    threshold_sq: f64,
) -> Vec<usize> {
    (0..world.len())
        .filter(|&i| error_sq(pose, &world[i], image[i]) < threshold_sq)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scene(rng: &mut Rng, count: usize) -> Vec<Point3<f64>> {
        (0..count)
            .map(|_| {
                Point3::new(
                    rng.next_f64() * 4.0 - 2.0,
                    rng.next_f64() * 3.0 - 1.5,
                    rng.next_f64() * 2.0 - 1.0,
                )
            })
            .collect()
    }

    #[test]
    fn recovers_a_camera_despite_outliers() {
        let mut rng = Rng::new(5);
        let truth = Pose::look_at(
            &Point3::new(1.5, 0.8, -6.0),
            &Point3::origin(),
            &Vector3::y(),
        );
        let world = scene(&mut rng, 300);
        let mut image: Vec<Norm> = world
            .iter()
            .map(|p| truth.project(&crate::World(*p)).unwrap())
            .collect();
        for n in image.iter_mut().take(90) {
            *n = Norm::new(rng.next_f64() - 0.5, rng.next_f64() - 0.5);
        }

        let solution = ransac(&world, &image, 1e-3, 1000, 0.999, &mut rng).expect("a pose");
        let rotation_error = (solution.pose.rotation * truth.rotation.inverse())
            .scaled_axis()
            .norm();
        assert!(
            rotation_error < 1e-6,
            "rotation off by {rotation_error} rad"
        );
        assert!((solution.pose.centre() - truth.centre()).norm() < 1e-5);
        assert_eq!(solution.inliers.len(), 210);
    }
}
