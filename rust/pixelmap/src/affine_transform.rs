/// Represents a 2D affine transformation. It includes:
/// - A reference origin in integer coordinates (`origin_x`, `origin_y`).
/// - A 2×2 linear transform matrix (`a11`, `a12`, `a21`, `a22`).
/// - A translation offset (`translate_x`, `translate_y`).
///
/// This allows a point `(x, y)` to be transformed into `(X, Y)` by:
///
/// ```text
///   let dx = x - origin_x;
///   let dy = y - origin_y;
///   X = (dx * a11) + (dy * a12) + translate_x;
///   Y = (dx * a21) + (dy * a22) + translate_y;
/// ```
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct AffineTransform {
    /// Integer reference x-coordinate (the "origin" in the input space).
    pub origin_x: u16,

    /// Integer reference y-coordinate (the "origin" in the input space).
    pub origin_y: u16,

    /// Translation offset in the transformed space (x-direction).
    pub translate_x: f32,

    /// Translation offset in the transformed space (y-direction).
    pub translate_y: f32,

    /// Matrix entry: row 1, col 1 (often scale in X or combined with rotation/shear).
    pub a11: f32,

    /// Matrix entry: row 1, col 2 (often shear or rotation).
    pub a12: f32,

    /// Matrix entry: row 2, col 1 (often shear or rotation).
    pub a21: f32,

    /// Matrix entry: row 2, col 2 (often scale in Y or combined with rotation/shear).
    pub a22: f32,
}

impl AffineTransform {
    /// Checks if the scale factors (the magnitudes of each row in the 2×2 portion)
    /// are within the range `[1/scale_bound, scale_bound]`.
    ///
    /// This ensures that the transform matrix isn't scaling too large or too small.
    ///
    /// # Parameters
    /// - `scale_bound`: A positive number representing the maximum allowed scale.
    ///
    /// # Returns
    /// - `true` if both scale factors lie in the valid range.
    /// - `false` otherwise, or if `scale_bound` is non-positive.
    ///
    /// # Examples
    ///
    /// ```text
    /// let t = AffineTransform {
    ///     origin_x: 0,
    ///     origin_y: 0,
    ///     translate_x: 0.0,
    ///     translate_y: 0.0,
    ///     a11: 1.0,
    ///     a12: 0.0,
    ///     a21: 0.0,
    ///     a22: 1.0,
    /// };
    /// assert!(t.is_scale_valid(2.0)); // Both row magnitudes are 1.0, so it's valid.
    /// assert!(!t.is_scale_valid(0.0)); // Invalid scale_bound.
    /// ```
    pub fn is_scale_valid(&self, scale_bound: f32) -> bool {
        // If scale_bound is not positive, the range check doesn't make sense.
        if scale_bound <= 0.0 {
            return false;
        }

        let min_scale = 1.0 / scale_bound;

        // Calculate the length of the first row (a11, a12) — effectively a "scale" in X.
        let scale_x = (self.a11.powi(2) + self.a12.powi(2)).sqrt();

        // Calculate the length of the second row (a21, a22) — effectively a "scale" in Y.
        let scale_y = (self.a21.powi(2) + self.a22.powi(2)).sqrt();

        // Check each scale factor is within [min_scale, scale_bound].
        scale_x > min_scale && scale_x < scale_bound && scale_y > min_scale && scale_y < scale_bound
    }

    /// Creates a **new** `AffineTransform` in which the origin is shifted to
    /// a new integer coordinate `(x, y)`. The existing transform is applied
    /// to `(x, y)` to calculate the new translation offsets.
    ///
    /// # Parameters
    /// - `x`: The new origin x-coordinate (integer).
    /// - `y`: The new origin y-coordinate (integer).
    ///
    /// # Returns
    /// A new `AffineTransform` with:
    /// - `origin_x = x`, `origin_y = y`
    /// - `translate_x` and `translate_y` set to the transformed position of `(x, y)`
    ///   under the old transform (maintaining consistent relative positioning).
    /// - The same `a11`, `a12`, `a21`, `a22` values as the original.
    ///
    /// # Examples
    /// ```text
    /// let t_old = AffineTransform {
    ///     origin_x: 0,
    ///     origin_y: 0,
    ///     translate_x: 100.0,
    ///     translate_y: 200.0,
    ///     a11: 1.0,
    ///     a12: 0.0,
    ///     a21: 0.0,
    ///     a22: 1.0,
    /// };
    ///
    /// // Move the origin to (50, 50). In the old transform, (50, 50)
    /// // transforms to (50 + 100, 50 + 200) => (150, 250).
    /// let t_new = t_old.extrapolate_mapping(50, 50);
    /// assert_eq!(t_new.origin_x, 50);
    /// assert_eq!(t_new.origin_y, 50);
    /// assert_eq!(t_new.translate_x, 150.0);
    /// assert_eq!(t_new.translate_y, 250.0);
    /// ```
    pub fn extrapolate_mapping(&self, x: u16, y: u16) -> AffineTransform {
        let dx = (x as isize - self.origin_x as isize) as f32;
        let dy = (y as isize - self.origin_y as isize) as f32;
        let x2 = dx * self.a11 + dy * self.a12 + self.translate_x;
        let y2 = dx * self.a21 + dy * self.a22 + self.translate_y;
        AffineTransform {
            origin_x: x,
            origin_y: y,
            translate_x: x2,
            translate_y: y2,
            ..*self
        }
    }
}
