//! What the caller chooses.
//!
//! One dial that matters — [`Quality`] — and a short list of things only the caller can
//! know: what the camera's focal length was, how large a texture the result may carry, and
//! which seed to reproduce a run from. Everything else the pipeline works out for itself.
//!
//! The thresholds each stage judges by are deliberately not here. There are about fifty of
//! them, they are meaningful only together, and the right values follow from [`Quality`]
//! and from the precision of the mappings actually found. Exposing them would turn tuning
//! decisions into this crate's contract.

use pixelmap::{Quality, DEFAULT_SEED};

use crate::calib::{FocalSource, Intrinsics};

/// The largest texture atlas a reconstruction produces by default, in pixels on a side.
pub const DEFAULT_MAX_TEXTURE_SIZE: usize = 8192;

/// What is known about the focal length the photos were taken at.
///
/// All the photos must come from one camera at one fixed zoom, so this describes all of
/// them at once.
#[derive(Copy, Clone, Debug, Default, PartialEq)]
#[non_exhaustive]
pub enum Focal {
    /// Nothing is known. It is guessed from the photos' dimensions and then refined during
    /// bundle adjustment, which is why [`Options::refine_focal`] is forced on in this case.
    #[default]
    Unknown,
    /// The 35 mm equivalent focal length, as EXIF reports it.
    Equivalent35mm(f64),
    /// The focal length in pixels, of the photos as handed in. Pass this when the camera
    /// is calibrated, or when the photos were resized after EXIF was read.
    Pixels(f64),
}

impl Focal {
    /// The camera this focal length describes, for photos of `width` × `height`.
    ///
    /// A run works this out for itself; this is for a caller that wants to show what the
    /// camera will be taken to be before starting one, without having to reimplement the
    /// mapping and drift out of step with it.
    pub fn intrinsics(self, width: usize, height: usize) -> Intrinsics {
        match self {
            Focal::Unknown => Intrinsics::estimated(width, height),
            Focal::Equivalent35mm(mm) => Intrinsics::from_35mm(mm, width, height),
            Focal::Pixels(px) => Intrinsics::from_focal(px, width, height, FocalSource::Provided),
        }
    }
}

/// Settings for a reconstruction.
///
/// Built by chaining, as pixelmap's own [`Builder`](pixelmap::Builder) is:
///
/// ```
/// use pixelmap::Quality;
/// use pixelmap_multiview::{Focal, Options};
///
/// let options = Options::new()
///     .quality(Quality::Medium)
///     .focal(Focal::Equivalent35mm(28.0));
/// ```
///
/// The fields are readable so that a caller can show what a run was given; they are set
/// through the methods so that adding a setting later does not break anyone.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
#[must_use]
pub struct Options {
    /// How much work pixelmap puts into each pair. The dominant cost of a run: a pair
    /// takes about 0.6 s at [`Quality::Low`] and about 2.3 s at [`Quality::Medium`], and
    /// there are N(N−1)/2 pairs.
    pub quality: Quality,
    /// What is known about the camera.
    pub focal: Focal,
    /// The seed for pixelmap and for every random choice after it. `None` uses
    /// [`DEFAULT_SEED`]. Two runs agreeing on photos, quality and seed give the same model.
    pub seed: Option<u64>,
    /// Refine the focal length during bundle adjustment. Always done when [`Self::focal`]
    /// is [`Focal::Unknown`], whatever this says.
    pub refine_focal: bool,
    /// The largest texture atlas to produce, in pixels on a side. The default suits a file
    /// on disc; a caller embedding the atlas in a page wants less.
    pub max_texture_size: usize,
}

impl Default for Options {
    fn default() -> Self {
        Options {
            quality: Quality::Low,
            focal: Focal::Unknown,
            seed: None,
            refine_focal: false,
            max_texture_size: DEFAULT_MAX_TEXTURE_SIZE,
        }
    }
}

impl Options {
    /// Settings with every default: [`Quality::Low`], an unknown focal length, and
    /// [`DEFAULT_SEED`].
    pub fn new() -> Self {
        Options::default()
    }

    /// How much work pixelmap puts into each pair.
    pub fn quality(mut self, quality: Quality) -> Self {
        self.quality = quality;
        self
    }

    /// What is known about the camera.
    pub fn focal(mut self, focal: Focal) -> Self {
        self.focal = focal;
        self
    }

    /// Pins the seed every random choice derives from. `None` goes back to
    /// [`DEFAULT_SEED`].
    pub fn seed(mut self, seed: impl Into<Option<u64>>) -> Self {
        self.seed = seed.into();
        self
    }

    /// Refine the focal length during bundle adjustment.
    pub fn refine_focal(mut self, refine: bool) -> Self {
        self.refine_focal = refine;
        self
    }

    /// The largest texture atlas to produce, in pixels on a side.
    pub fn max_texture_size(mut self, pixels: usize) -> Self {
        self.max_texture_size = pixels;
        self
    }

    /// The seed to use, resolving `None` to [`DEFAULT_SEED`].
    pub(crate) fn resolved_seed(&self) -> u64 {
        self.seed.unwrap_or(DEFAULT_SEED)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_unknown_focal_length_is_marked_as_estimated() {
        let intrinsics = Focal::Unknown.intrinsics(800, 600);
        assert_eq!(intrinsics.source, FocalSource::Estimated);
    }

    #[test]
    fn a_focal_length_in_pixels_is_used_as_given() {
        let intrinsics = Focal::Pixels(900.0).intrinsics(800, 600);
        assert_eq!(intrinsics.source, FocalSource::Provided);
        assert_eq!(intrinsics.fx, 900.0);
    }

    #[test]
    fn chaining_leaves_the_untouched_settings_at_their_defaults() {
        let options = Options::new().quality(Quality::High);
        assert_eq!(options.quality, Quality::High);
        assert_eq!(options.focal, Focal::Unknown);
        assert_eq!(options.max_texture_size, DEFAULT_MAX_TEXTURE_SIZE);
        assert_eq!(Options::new().seed(7).resolved_seed(), 7);
        assert_eq!(Options::new().resolved_seed(), DEFAULT_SEED);
        assert_eq!(Options::new().seed(7).seed(None), Options::new());
    }
}
