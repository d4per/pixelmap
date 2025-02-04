use std::cell::Cell;
use crate::affine_transform::AffineTransform;

/// A container for storing an [`AffineTransform`] along with its
/// associated “best score”. Internally uses [`Cell`], allowing for
/// interior mutability.
///
/// Typical usage includes:
/// - Storing an optional [`AffineTransform`] (or `None` if not set).
/// - Recording a floating-point “best score” value, which can
///   be updated based on some criteria (e.g., the transform quality).
#[derive(Debug, Clone)]
pub struct AffineTransformCell {
    /// Holds the current [`AffineTransform`] (or `None` if none is set).
    /// Wrapped in a [`Cell`] for interior mutability, allowing it to be
    /// updated without requiring a mutable reference to `AffineTransformCell`.
    cm: Cell<Option<AffineTransform>>,

    /// A “best score” associated with the transform, typically used
    /// to compare how optimal or preferable one transform is over another.
    best_score: Cell<f32>,
}

impl AffineTransformCell {
    /// Creates a new `AffineTransformCell` with no transform set
    /// (`None`) and the `best_score` initialized to [`f32::MAX`].
    ///
    /// # Examples
    /// ```
    /// # use pixelmap::affine_transform_cell::AffineTransformCell;
    /// let cell = AffineTransformCell::new_empty();
    /// assert_eq!(cell.get_affine_transform(), None);
    /// assert_eq!(cell.get_score(), f32::MAX);
    /// ```
    pub fn new_empty() -> Self {
        AffineTransformCell {
            cm: Cell::new(None),
            best_score: Cell::new(f32::MAX),
        }
    }

    /// Updates the cell to store a given [`AffineTransform`] along with
    /// a new “best score”. This overrides any previous transform and score.
    ///
    /// # Parameters
    /// - `cm`: The new [`AffineTransform`] to store.
    /// - `new_score`: A floating-point value that describes how good
    ///   or optimal this transform is (lower or higher depending on usage).
    ///
    /// # Examples
    /// ```
    /// # use pixelmap::affine_transform::AffineTransform;
    /// # use pixelmap::affine_transform_cell::AffineTransformCell;
    /// let cell = AffineTransformCell::new_empty();
    /// let transform = AffineTransform {
    ///     origin_x: 0,
    ///     origin_y: 0,
    ///     translate_x: 10.0,
    ///     translate_y: 20.0,
    ///     a11: 1.0, a12: 0.0,
    ///     a21: 0.0, a22: 1.0,
    /// };
    ///
    /// cell.set(transform, 42.0);
    /// assert_eq!(cell.get_score(), 42.0);
    /// assert!(cell.get_affine_transform().is_some());
    /// ```
    pub fn set(&self, cm: AffineTransform, new_score: f32) {
        self.cm.set(Some(cm));
        self.best_score.set(new_score);
    }

    /// Retrieves the current “best score.”
    ///
    /// # Returns
    /// The floating-point “best score” stored in this cell.
    ///
    /// # Examples
    /// ```
    /// use pixelmap::affine_transform_cell::AffineTransformCell;
    /// use pixelmap::affine_transform::AffineTransform;
    /// let cell = AffineTransformCell::new_empty();
    /// assert_eq!(cell.get_score(), f32::MAX);
    /// ```
    pub fn get_score(&self) -> f32 {
        self.best_score.get()
    }

    /// Retrieves the current [`AffineTransform`], if any.
    ///
    /// # Returns
    /// An [`Option<AffineTransform>`], which is `Some(transform)` if a transform
    /// has been set, or `None` if this cell is still empty.
    ///
    /// # Examples
    /// ```
    /// use pixelmap::affine_transform::AffineTransform;
    /// use pixelmap::affine_transform_cell::AffineTransformCell;
    /// let cell = AffineTransformCell::new_empty();
    /// assert_eq!(cell.get_affine_transform(), None);
    ///
    /// let transform = AffineTransform {
    ///     origin_x: 0,
    ///     origin_y: 0,
    ///     translate_x: 10.0,
    ///     translate_y: 20.0,
    ///     a11: 1.0, a12: 0.0,
    ///     a21: 0.0, a22: 1.0,
    /// };
    /// cell.set(transform, 42.0);
    /// let maybe_transform = cell.get_affine_transform();
    /// assert!(maybe_transform.is_some());
    /// ```
    pub fn get_affine_transform(&self) -> Option<AffineTransform> {
        self.cm.get()
    }
}
