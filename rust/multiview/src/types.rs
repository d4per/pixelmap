//! Identifiers, and one newtype per coordinate space.
//!
//! Multi-view pipelines have one characteristic bug: a point in one coordinate space
//! handed to code that expects another. It type-checks as a bare `(f32, f32)`, runs
//! cleanly, and produces a skewed model. So every space the pipeline uses gets its own
//! type, and conversions between them are explicit calls on whatever defines the
//! relationship — [`crate::Intrinsics`] for pixels and normalized camera coordinates.

use std::cmp::Ordering;
use std::fmt;

use nalgebra::{Point2, Point3};

/// The index of a photo in the input set.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ViewId(pub u32);

impl ViewId {
    /// The id as an index into per-view collections.
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

impl fmt::Display for ViewId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "view {}", self.0)
    }
}

/// An unordered pair of distinct views, stored with the lower id first.
///
/// Correspondence is computed once per pair, from [`Self::a`] to [`Self::b`]; the other
/// direction is the same mapping read backwards.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PairId {
    a: ViewId,
    b: ViewId,
}

impl PairId {
    /// The pair made of `i` and `j`, in either order. `None` if they are the same view.
    pub fn new(i: ViewId, j: ViewId) -> Option<Self> {
        match i.cmp(&j) {
            Ordering::Less => Some(PairId { a: i, b: j }),
            Ordering::Greater => Some(PairId { a: j, b: i }),
            Ordering::Equal => None,
        }
    }

    /// The view with the lower id: the first photo of the correspondence run.
    pub fn a(self) -> ViewId {
        self.a
    }

    /// The view with the higher id: the second photo of the correspondence run.
    pub fn b(self) -> ViewId {
        self.b
    }

    /// Whether `view` is one of the two.
    pub fn contains(self, view: ViewId) -> bool {
        self.a == view || self.b == view
    }

    /// The other view of the pair, or `None` if `view` is not part of it.
    pub fn other(self, view: ViewId) -> Option<ViewId> {
        if view == self.a {
            Some(self.b)
        } else if view == self.b {
            Some(self.a)
        } else {
            None
        }
    }

    /// Every pair over `views` views, in lexicographic order: `(0, 1), (0, 2), …, (1, 2), …`.
    pub fn all(views: u32) -> impl Iterator<Item = PairId> {
        (0..views).flat_map(move |a| {
            (a + 1..views).map(move |b| PairId {
                a: ViewId(a),
                b: ViewId(b),
            })
        })
    }

    /// How many pairs `views` views form: `n(n − 1) / 2`.
    pub fn count(views: usize) -> usize {
        views * views.saturating_sub(1) / 2
    }
}

impl fmt::Display for PairId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "views {}–{}", self.a.0, self.b.0)
    }
}

/// A position in one of the photos handed to the pipeline, in pixels.
///
/// These are the coordinates [`pixelmap::Correspondence::lookup`] speaks. Not the
/// original files' pixels, if the caller resized them first, and not pixelmap's internal
/// working resolution.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct PhotoPx(pub Point2<f32>);

impl PhotoPx {
    /// The point `(x, y)`.
    pub fn new(x: f32, y: f32) -> Self {
        PhotoPx(Point2::new(x, y))
    }

    /// The horizontal coordinate.
    pub fn x(self) -> f32 {
        self.0.x
    }

    /// The vertical coordinate.
    pub fn y(self) -> f32 {
        self.0.y
    }
}

/// Normalized camera coordinates: a pixel with K⁻¹ applied, so that `(x, y, 1)` is the
/// direction of its ray in the camera's own frame.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Norm(pub Point2<f64>);

impl Norm {
    /// The point `(x, y)`.
    pub fn new(x: f64, y: f64) -> Self {
        Norm(Point2::new(x, y))
    }

    /// The horizontal coordinate.
    pub fn x(self) -> f64 {
        self.0.x
    }

    /// The vertical coordinate.
    pub fn y(self) -> f64 {
        self.0.y
    }
}

/// A point in the reconstruction's frame.
///
/// The seed pair defines that frame: its first camera sits at the origin, and the distance
/// between its two cameras is the unit of length. Nothing here is metric.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct World(pub Point3<f64>);

impl World {
    /// The point `(x, y, z)`.
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        World(Point3::new(x, y, z))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pairs_are_unordered_and_exclude_self_pairs() {
        let (i, j) = (ViewId(3), ViewId(1));
        let pair = PairId::new(i, j).expect("distinct views");
        assert_eq!(pair, PairId::new(j, i).unwrap());
        assert_eq!((pair.a(), pair.b()), (ViewId(1), ViewId(3)));
        assert_eq!(pair.other(ViewId(1)), Some(ViewId(3)));
        assert_eq!(pair.other(ViewId(2)), None);
        assert!(pair.contains(ViewId(3)));
        assert!(PairId::new(i, i).is_none());
    }

    #[test]
    fn enumerates_every_pair_once() {
        for views in 0..7u32 {
            let pairs: Vec<_> = PairId::all(views).collect();
            assert_eq!(pairs.len(), PairId::count(views as usize));
            assert!(pairs.windows(2).all(|w| w[0] < w[1]), "sorted and distinct");
        }
        let four: Vec<_> = PairId::all(4).map(|p| (p.a().0, p.b().0)).collect();
        assert_eq!(four, [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]);
    }
}
