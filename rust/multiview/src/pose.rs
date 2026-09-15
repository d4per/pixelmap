//! Camera poses.

use nalgebra::{Matrix3, Matrix3x4, Point3, Rotation3, Vector3};

use crate::types::{Norm, World};

/// Where a camera is and which way it faces, stored as the transform from world
/// coordinates into the camera's frame: `x_cam = R · x_world + t`.
///
/// The camera frame is the usual computer-vision one: x to the right of the image, y
/// down, z forward along the optical axis.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Pose {
    /// `R`: world axes into camera axes.
    pub rotation: Rotation3<f64>,
    /// `t`: the world origin, in the camera's frame.
    pub translation: Vector3<f64>,
}

impl Pose {
    /// A camera at the world origin, looking along the world z axis.
    pub fn identity() -> Self {
        Pose {
            rotation: Rotation3::identity(),
            translation: Vector3::zeros(),
        }
    }

    /// A camera at `eye` looking at `target`, oriented so that `up` points up in the image.
    ///
    /// `up` need not be exactly perpendicular to the viewing direction, but must not be
    /// parallel to it.
    pub fn look_at(eye: &Point3<f64>, target: &Point3<f64>, up: &Vector3<f64>) -> Self {
        let forward = (target - eye).normalize();
        let right = forward.cross(up).normalize();
        let down = forward.cross(&right);
        let rotation = Rotation3::from_matrix_unchecked(Matrix3::from_rows(&[
            right.transpose(),
            down.transpose(),
            forward.transpose(),
        ]));
        let translation = -(rotation * eye.coords);
        Pose {
            rotation,
            translation,
        }
    }

    /// The camera centre, in world coordinates.
    pub fn centre(&self) -> Point3<f64> {
        Point3::from(-(self.rotation.inverse() * self.translation))
    }

    /// A world point in this camera's frame.
    pub fn to_camera(&self, point: &Point3<f64>) -> Point3<f64> {
        self.rotation * point + self.translation
    }

    /// Where a world point lands in this camera, or `None` if it is not in front of it.
    pub fn project(&self, point: &World) -> Option<Norm> {
        let p = self.to_camera(&point.0);
        (p.z > 0.0).then(|| Norm::new(p.x / p.z, p.y / p.z))
    }

    /// This pose expressed in `reference`'s camera frame: the transform from `reference`'s
    /// camera coordinates into this camera's.
    pub fn relative_to(&self, reference: &Pose) -> Pose {
        let rotation = self.rotation * reference.rotation.inverse();
        Pose {
            rotation,
            translation: self.translation - rotation * reference.translation,
        }
    }

    /// The transform from this camera's frame back into world coordinates.
    pub fn inverse(&self) -> Pose {
        let rotation = self.rotation.inverse();
        Pose {
            rotation,
            translation: -(rotation * self.translation),
        }
    }

    /// The projection matrix `[R | t]`.
    pub fn matrix(&self) -> Matrix3x4<f64> {
        let mut m = Matrix3x4::zeros();
        m.fixed_view_mut::<3, 3>(0, 0)
            .copy_from(self.rotation.matrix());
        m.set_column(3, &self.translation);
        m
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn look_at_centres_the_target_with_up_at_the_top() {
        let eye = Point3::new(1.0, 2.0, -5.0);
        let target = Point3::new(0.0, 0.5, 1.0);
        let pose = Pose::look_at(&eye, &target, &Vector3::y());

        assert!((pose.centre() - eye).norm() < 1e-12);
        let centre = pose.project(&World(target)).expect("target is in front");
        assert!(centre.x().abs() < 1e-12 && centre.y().abs() < 1e-12);

        let above = pose.project(&World(target + Vector3::y())).unwrap();
        assert!(above.y() < 0.0, "image y points down");
        assert!(pose.project(&World(eye - (target - eye))).is_none());
    }

    #[test]
    fn relative_pose_chains_camera_frames() {
        let a = Pose::look_at(
            &Point3::new(-1.0, 0.0, -4.0),
            &Point3::origin(),
            &Vector3::y(),
        );
        let b = Pose::look_at(
            &Point3::new(2.0, 1.0, -3.0),
            &Point3::origin(),
            &Vector3::y(),
        );
        let rel = b.relative_to(&a);
        let x = Point3::new(0.3, -0.2, 0.7);
        let via_a = rel.to_camera(&a.to_camera(&x));
        assert!((via_a - b.to_camera(&x)).norm() < 1e-12);

        let back = a.inverse().to_camera(&a.to_camera(&x));
        assert!((back - x).norm() < 1e-12);
    }
}
