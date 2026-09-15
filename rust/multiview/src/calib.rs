//! Camera intrinsics: the pinhole matrix K shared by every view.
//!
//! A wrong focal length does not make a *noisy* model, it makes a *skewed* one — a
//! projective distortion that looks plausible until someone measures it. So the
//! intrinsics carry a record of where the focal length came from, for the caller to show.

use nalgebra::{Matrix3, Point2};

use crate::types::{Norm, PhotoPx};

/// The focal length used when nothing better is known, as a multiple of the photo's long
/// edge. That is roughly a 43 mm-equivalent lens, the middle of the range phones and kit
/// zooms shoot at.
pub const ESTIMATED_FOCAL_FACTOR: f64 = 1.2;

/// Where a focal length came from.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum FocalSource {
    /// Supplied by the caller, for example from a calibration.
    Provided,
    /// Derived from the EXIF 35 mm-equivalent focal length.
    Exif35mm,
    /// A guess: [`ESTIMATED_FOCAL_FACTOR`] times the long edge. Worth refining during
    /// bundle adjustment, and worth telling the user about.
    Estimated,
}

/// A pinhole camera with square pixels and no skew.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Intrinsics {
    /// Horizontal focal length, in pixels.
    pub fx: f64,
    /// Vertical focal length, in pixels.
    pub fy: f64,
    /// Horizontal position of the principal point, in pixels.
    pub cx: f64,
    /// Vertical position of the principal point, in pixels.
    pub cy: f64,
    /// Where the focal length came from.
    pub source: FocalSource,
}

impl Intrinsics {
    /// A camera with focal length `focal_px` and its principal point at the centre of a
    /// `width` × `height` photo.
    pub fn from_focal(focal_px: f64, width: usize, height: usize, source: FocalSource) -> Self {
        Intrinsics {
            fx: focal_px,
            fy: focal_px,
            cx: width as f64 / 2.0,
            cy: height as f64 / 2.0,
            source,
        }
    }

    /// From an EXIF `FocalLengthIn35mmFilm` value, for a photo of `width` × `height` pixels.
    ///
    /// Scales by the frame diagonal, which is how cameras generally define the
    /// equivalence. For a 3:2 photo this is the same as `f × long edge / 36`; for 4:3
    /// the two differ by about 4%.
    ///
    /// The 35 mm equivalent does not depend on pixel count, so it is fine to apply this to
    /// photos that have already been resized.
    pub fn from_35mm(focal_35mm: f64, width: usize, height: usize) -> Self {
        let full_frame_diagonal_mm = (36.0f64 * 36.0 + 24.0 * 24.0).sqrt();
        let diagonal_px = ((width * width + height * height) as f64).sqrt();
        let focal_px = focal_35mm * diagonal_px / full_frame_diagonal_mm;
        Self::from_focal(focal_px, width, height, FocalSource::Exif35mm)
    }

    /// The fallback for photos without EXIF: see [`ESTIMATED_FOCAL_FACTOR`].
    pub fn estimated(width: usize, height: usize) -> Self {
        let focal_px = ESTIMATED_FOCAL_FACTOR * width.max(height) as f64;
        Self::from_focal(focal_px, width, height, FocalSource::Estimated)
    }

    /// The same camera after its photos have been resized by `factor`.
    pub fn scaled(self, factor: f64) -> Self {
        Intrinsics {
            fx: self.fx * factor,
            fy: self.fy * factor,
            cx: self.cx * factor,
            cy: self.cy * factor,
            source: self.source,
        }
    }

    /// Whether every parameter is finite and both focal lengths are positive.
    pub fn is_valid(&self) -> bool {
        [self.fx, self.fy, self.cx, self.cy]
            .iter()
            .all(|v| v.is_finite())
            && self.fx > 0.0
            && self.fy > 0.0
    }

    /// The matrix K.
    pub fn matrix(&self) -> Matrix3<f64> {
        Matrix3::new(
            self.fx, 0.0, self.cx, //
            0.0, self.fy, self.cy, //
            0.0, 0.0, 1.0,
        )
    }

    /// Applies K⁻¹ to a pixel.
    pub fn normalize(&self, p: PhotoPx) -> Norm {
        Norm(Point2::new(
            (p.x() as f64 - self.cx) / self.fx,
            (p.y() as f64 - self.cy) / self.fy,
        ))
    }

    /// Applies K to normalized camera coordinates.
    pub fn denormalize(&self, n: Norm) -> PhotoPx {
        PhotoPx::new(
            (n.x() * self.fx + self.cx) as f32,
            (n.y() * self.fy + self.cy) as f32,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn equivalent_focal_length_matches_the_width_formula_on_3_2_frames() {
        let k = Intrinsics::from_35mm(28.0, 1800, 1200);
        assert!((k.fx - 28.0 * 1800.0 / 36.0).abs() < 1e-9, "fx = {}", k.fx);
        assert_eq!(k.fx, k.fy);
        assert_eq!((k.cx, k.cy), (900.0, 600.0));
        assert_eq!(k.source, FocalSource::Exif35mm);
    }

    #[test]
    fn equivalent_focal_length_is_resolution_independent() {
        let full = Intrinsics::from_35mm(26.0, 4032, 3024);
        let small = Intrinsics::from_35mm(26.0, 1008, 756);
        let scaled = full.scaled(0.25);
        assert!((small.fx - scaled.fx).abs() < 1e-9);
        assert!((small.cx - scaled.cx).abs() < 1e-9);
    }

    #[test]
    fn normalizing_round_trips() {
        let k = Intrinsics::estimated(1200, 900);
        assert!(k.is_valid());
        let p = PhotoPx::new(123.5, 876.25);
        let back = k.denormalize(k.normalize(p));
        assert!((back.x() - p.x()).abs() < 1e-3 && (back.y() - p.y()).abs() < 1e-3);

        let centre = k.normalize(PhotoPx::new(600.0, 450.0));
        assert_eq!((centre.x(), centre.y()), (0.0, 0.0));

        let n = k.normalize(p);
        let projected = k.matrix() * nalgebra::Vector3::new(n.x(), n.y(), 1.0);
        assert!((projected.x - p.x() as f64).abs() < 1e-3);
    }

    #[test]
    fn rejects_unusable_parameters() {
        let mut k = Intrinsics::estimated(100, 100);
        k.fx = 0.0;
        assert!(!k.is_valid());
        k.fx = f64::NAN;
        assert!(!k.is_valid());
    }
}
