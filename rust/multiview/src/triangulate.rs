//! Points from rays: where the lines of sight of several cameras meet.

use nalgebra::{Matrix4, Point3};

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
    use nalgebra::Vector3;

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
