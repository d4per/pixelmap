//! Similarity alignment between point sets.
//!
//! A reconstruction without a known distance in it is only defined up to position,
//! orientation and scale. To compare one with ground truth, or with an earlier run, align
//! the two first.

use nalgebra::{Matrix3, Point3, Rotation3, Vector3};

/// `x ↦ scale · R · x + t`.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Similarity {
    /// Uniform scale.
    pub scale: f64,
    /// Rotation.
    pub rotation: Rotation3<f64>,
    /// Translation, applied after scaling and rotating.
    pub translation: Vector3<f64>,
}

impl Similarity {
    /// Applies the transform to a point.
    pub fn apply(&self, p: &Point3<f64>) -> Point3<f64> {
        Point3::from(self.rotation * p.coords * self.scale + self.translation)
    }
}

/// The similarity that best maps `from` onto `to` in the least-squares sense (Umeyama,
/// 1991). `None` if the sets differ in length, have fewer than three points, or all of
/// `from` coincide.
pub fn umeyama(from: &[Point3<f64>], to: &[Point3<f64>]) -> Option<Similarity> {
    if from.len() != to.len() || from.len() < 3 {
        return None;
    }
    let n = from.len() as f64;
    let mean_from = from.iter().map(|p| p.coords).sum::<Vector3<f64>>() / n;
    let mean_to = to.iter().map(|p| p.coords).sum::<Vector3<f64>>() / n;

    let mut covariance = Matrix3::zeros();
    let mut variance = 0.0;
    for (f, t) in from.iter().zip(to) {
        let df = f.coords - mean_from;
        covariance += (t.coords - mean_to) * df.transpose();
        variance += df.norm_squared();
    }
    covariance /= n;
    variance /= n;
    if variance <= 0.0 {
        return None;
    }

    let svd = covariance.svd(true, true);
    let (u, v_t) = (svd.u?, svd.v_t?);
    let mut signs = Vector3::repeat(1.0);
    if u.determinant() * v_t.determinant() < 0.0 {
        signs[svd.singular_values.imin()] = -1.0;
    }
    let mut rotation = Rotation3::from_matrix_unchecked(u * Matrix3::from_diagonal(&signs) * v_t);
    rotation.renormalize();
    let scale = svd.singular_values.component_mul(&signs).sum() / variance;
    Some(Similarity {
        scale,
        rotation,
        translation: mean_to - rotation * mean_from * scale,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recovers_a_known_similarity() {
        let truth = Similarity {
            scale: 2.5,
            rotation: Rotation3::from_euler_angles(0.3, -1.1, 2.0),
            translation: Vector3::new(1.0, -4.0, 0.5),
        };
        let from = [
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.2, -0.3),
            Point3::new(-0.5, 1.4, 0.8),
            Point3::new(0.3, -0.7, 2.0),
        ];
        let to: Vec<_> = from.iter().map(|p| truth.apply(p)).collect();
        let found = umeyama(&from, &to).unwrap();
        assert!((found.scale - truth.scale).abs() < 1e-9);
        for (f, t) in from.iter().zip(&to) {
            assert!((found.apply(f) - t).norm() < 1e-9);
        }
        assert!(umeyama(&from[..2], &to[..2]).is_none());
    }
}
