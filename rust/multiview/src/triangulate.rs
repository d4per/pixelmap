//! Points from rays: where the lines of sight of several cameras meet.

use nalgebra::{Matrix2x3, Matrix3, Matrix4, Point3, Vector2, Vector3};

use crate::pose::Pose;
use crate::types::{Norm, World};

/// Linear (DLT) triangulation of one point seen by two or more cameras.
///
/// Minimizes an algebraic error rather than reprojection error, so it is a starting point
/// for refinement rather than a final answer. `None` with fewer than two observations, or
/// when the rays are so close to parallel that the point is at infinity.
///
/// Does not check that the point is in front of the cameras; see [`in_front`].
pub fn dlt(observations: &[(Pose, Norm)]) -> Option<World> {
    if observations.len() < 2 {
        return None;
    }
    let mut ata = Matrix4::<f64>::zeros();
    for (pose, n) in observations {
        let p = pose.matrix();
        let (r0, r1, r2) = (
            p.row(0).into_owned(),
            p.row(1).into_owned(),
            p.row(2).into_owned(),
        );
        let a = r2 * n.x() - r0;
        let b = r2 * n.y() - r1;
        ata += a.transpose() * a + b.transpose() * b;
    }

    let eigen = ata.symmetric_eigen();
    let v = eigen.eigenvectors.column(eigen.eigenvalues.imin());
    let w = v[3];
    if w.abs() <= 1e-12 * v.norm() {
        return None;
    }
    let point = Point3::new(v[0] / w, v[1] / w, v[2] / w);
    (point.coords.iter().all(|c| c.is_finite())).then_some(World(point))
}

/// Gauss–Newton on the point's reprojection error, in normalized camera units, starting
/// from `initial`. A step is kept only if it lowers the error.
pub fn refine(observations: &[(Pose, Norm)], initial: World, iterations: usize) -> World {
    let cost = |x: &Point3<f64>| -> f64 {
        observations
            .iter()
            .map(|(pose, n)| {
                let c = pose.to_camera(x);
                if c.z <= 1e-12 {
                    1.0
                } else {
                    (c.x / c.z - n.x()).powi(2) + (c.y / c.z - n.y()).powi(2)
                }
            })
            .sum()
    };
    let mut x = initial.0;
    let mut current = cost(&x);
    for _ in 0..iterations {
        let mut h = Matrix3::<f64>::zeros();
        let mut g = Vector3::<f64>::zeros();
        for (pose, n) in observations {
            let c = pose.to_camera(&x);
            if c.z <= 1e-12 {
                continue;
            }
            let projection = Matrix2x3::new(
                1.0 / c.z,
                0.0,
                -c.x / (c.z * c.z),
                0.0,
                1.0 / c.z,
                -c.y / (c.z * c.z),
            );
            let j = projection * pose.rotation.matrix();
            let r = Vector2::new(c.x / c.z - n.x(), c.y / c.z - n.y());
            h += j.transpose() * j;
            g += j.transpose() * r;
        }
        let Some(delta) = h.cholesky().map(|c| c.solve(&-g)) else {
            break;
        };
        let candidate = x + delta;
        let candidate_cost = cost(&candidate);
        if candidate_cost.is_nan() || candidate_cost >= current {
            break;
        }
        x = candidate;
        current = candidate_cost;
        if delta.norm() <= 1e-12 * (1.0 + x.coords.norm()) {
            break;
        }
    }
    World(x)
}

/// The reprojection error of `point` in each observation, in normalized camera units.
/// Infinite where the point is behind the camera.
pub fn reprojection_errors<'a>(
    observations: &'a [(Pose, Norm)],
    point: &'a World,
) -> impl Iterator<Item = f64> + 'a {
    observations.iter().map(move |(pose, n)| {
        pose.project(point).map_or(f64::INFINITY, |p| {
            ((p.x() - n.x()).powi(2) + (p.y() - n.y()).powi(2)).sqrt()
        })
    })
}

/// Whether `point` lies in front of every camera in `observations`.
pub fn in_front(observations: &[(Pose, Norm)], point: &World) -> bool {
    observations
        .iter()
        .all(|(pose, _)| pose.to_camera(&point.0).z > 0.0)
}

/// The angle, in radians, between the rays from two camera centres to `point`.
///
/// Depth precision scales with roughly `1 / sin` of this: below a couple of degrees a
/// point's depth is mostly noise.
pub fn triangulation_angle(centre_a: &Point3<f64>, centre_b: &Point3<f64>, point: &World) -> f64 {
    (point.0 - centre_a).angle(&(point.0 - centre_b))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recovers_a_point_seen_by_three_cameras() {
        let target = Point3::new(0.2, -0.1, 0.5);
        let poses = [
            Pose::look_at(
                &Point3::new(-2.0, 0.0, -4.0),
                &Point3::origin(),
                &Vector3::y(),
            ),
            Pose::look_at(
                &Point3::new(0.0, 1.0, -5.0),
                &Point3::origin(),
                &Vector3::y(),
            ),
            Pose::look_at(
                &Point3::new(2.5, 0.0, -3.0),
                &Point3::origin(),
                &Vector3::y(),
            ),
        ];
        let observations: Vec<_> = poses
            .iter()
            .map(|pose| (*pose, pose.project(&World(target)).unwrap()))
            .collect();

        let point = dlt(&observations).expect("well-conditioned rays");
        assert!((point.0 - target).norm() < 1e-9);
        assert!(in_front(&observations, &point));
        assert!(dlt(&observations[..1]).is_none());

        let angle = triangulation_angle(&poses[0].centre(), &poses[2].centre(), &point);
        assert!(angle.to_degrees() > 30.0);
    }
}
