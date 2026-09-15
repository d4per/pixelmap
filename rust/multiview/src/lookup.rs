//! The one interface every geometry stage reads correspondences through.
//!
//! In the pipeline the correspondences come from pixelmap. In tests they come from
//! [`crate::synthetic`], which knows the exact answer. Keeping the stages behind this
//! trait is what lets them be tested against ground truth, with the matcher's own
//! limitations taken out of the picture.

use crate::types::PhotoPx;

/// A dense mapping between the two photos of a pair, in [`PhotoPx`] coordinates.
pub trait PairLookup {
    /// Where the pixel `p` of the pair's first photo ended up in the second, or `None`
    /// if it is not mapped.
    fn a_to_b(&self, p: PhotoPx) -> Option<PhotoPx>;

    /// Where the pixel `p` of the pair's second photo came from in the first, or `None`.
    fn b_to_a(&self, p: PhotoPx) -> Option<PhotoPx>;

    /// The fraction of the first photo that is mapped, from `0.0` to `1.0`.
    fn coverage(&self) -> f32;

    /// The spacing, in photo pixels, at which neighbouring lookups stop being the same
    /// measurement. Sampling more densely than this adds matches without adding
    /// information.
    fn native_stride(&self) -> f32;

    /// The typical localization error of a lookup, in photo pixels. Inlier thresholds are
    /// expressed as multiples of this.
    fn precision_px(&self) -> f32;
}

/// A pair's mapping read in a chosen direction.
///
/// Mappings are stored once per pair, from the lower view id to the higher. This turns
/// one around when a stage needs the other direction.
pub struct Directed<'a, L: ?Sized> {
    lookup: &'a L,
    reversed: bool,
}

impl<'a, L: PairLookup + ?Sized> Directed<'a, L> {
    /// `lookup` read from its first photo to its second, or the other way round if
    /// `reversed` is set.
    pub fn new(lookup: &'a L, reversed: bool) -> Self {
        Directed { lookup, reversed }
    }

    /// Maps a point from the source view to the target view.
    pub fn map(&self, p: PhotoPx) -> Option<PhotoPx> {
        if self.reversed {
            self.lookup.b_to_a(p)
        } else {
            self.lookup.a_to_b(p)
        }
    }

    /// Maps a point from the target view back to the source view.
    pub fn map_back(&self, p: PhotoPx) -> Option<PhotoPx> {
        if self.reversed {
            self.lookup.a_to_b(p)
        } else {
            self.lookup.b_to_a(p)
        }
    }
}

impl<L: PairLookup + ?Sized> PairLookup for Directed<'_, L> {
    fn a_to_b(&self, p: PhotoPx) -> Option<PhotoPx> {
        self.map(p)
    }

    fn b_to_a(&self, p: PhotoPx) -> Option<PhotoPx> {
        self.map_back(p)
    }

    fn coverage(&self) -> f32 {
        // Coverage is only measured from the first photo, so a reversed pair reports
        // an estimate. Both directions went through the same consistency check.
        self.lookup.coverage()
    }

    fn native_stride(&self) -> f32 {
        self.lookup.native_stride()
    }

    fn precision_px(&self) -> f32 {
        self.lookup.precision_px()
    }
}
