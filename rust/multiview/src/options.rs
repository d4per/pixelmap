//! What the caller chooses.
//!
//! One dial that matters — [`Quality`] — and a short list of things only the caller can
//! know: what the camera's focal length was, how large a texture the result may carry, and
//! which seed to reproduce a run from. Everything else the pipeline works out for itself.
//!
//! The one exception is pixelmap's neighborhood radius, passed through as is for a caller
//! tuning the pairwise correspondence. It is pixelmap's setting, not one of this crate's
//! thresholds, and it means here what it means there.
//!
//! The thresholds each stage judges by are deliberately not here. There are about fifty of
//! them, they are meaningful only together, and the right values follow from [`Quality`]
//! and from the precision of the mappings actually found. Exposing them would turn tuning
//! decisions into this crate's contract.

use pixelmap::{Quality, DEFAULT_SEED, MAX_NEIGHBORHOOD_RADIUS};

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
/// let exif_focal_35mm = 28.0; // The photos' FocalLengthIn35mmFilm, from EXIF.
/// let options = Options::new()
///     .quality(Quality::Medium)
///     .focal(Focal::Equivalent35mm(exif_focal_35mm))
///     .refine_focal(true);
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
    /// is [`Focal::Unknown`], whatever this says. Worth turning on for a focal length from
    /// EXIF too, which is rounded to whole millimetres and does not account for any crop.
    pub refine_focal: bool,
    /// The largest texture atlas to produce, in pixels on a side. The default suits a file
    /// on disc; a caller embedding the atlas in a page wants less.
    pub max_texture_size: usize,
    /// The radius, in working pixels, of the disc pixelmap compares two photos over, in
    /// every step of its schedule. `None` leaves pixelmap's default, 5. See
    /// [`Builder::neighborhood_radius`](pixelmap::Builder::neighborhood_radius).
    pub neighborhood_radius: Option<usize>,
    /// The disc radius of pixelmap's initial solve alone, the one that grows each pair's
    /// mapping out from its feature matches. `None` falls back to
    /// [`Self::neighborhood_radius`], and then to pixelmap's default. See
    /// [`Builder::init_neighborhood_radius`](pixelmap::Builder::init_neighborhood_radius).
    pub init_neighborhood_radius: Option<usize>,
}

impl Default for Options {
    fn default() -> Self {
        Options {
            quality: Quality::Low,
            focal: Focal::Unknown,
            seed: None,
            refine_focal: false,
            max_texture_size: DEFAULT_MAX_TEXTURE_SIZE,
            neighborhood_radius: None,
            init_neighborhood_radius: None,
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

    /// The radius of the disc pixelmap scores matches over, in every solve of every pair.
    /// `None` goes back to pixelmap's default.
    ///
    /// A larger disc matches weakly textured or strongly warped regions more reliably, but
    /// blurs the mapping across depth edges, and the time per pair grows with its area: a
    /// radius of 9 roughly doubles it. Since every mapped cell becomes a 3D point, test a
    /// change on the reconstruction itself rather than on pixelmap's benchmarks alone.
    ///
    /// # Panics
    /// If `radius` is 0 or above [`MAX_NEIGHBORHOOD_RADIUS`].
    pub fn neighborhood_radius(mut self, radius: impl Into<Option<usize>>) -> Self {
        self.neighborhood_radius = radius.into().map(checked_radius);
        self
    }

    /// The radius of the disc pixelmap's initial solve scores matches over, the solve that
    /// grows each pair's mapping out from its feature matches. `None` goes back to
    /// [`Self::neighborhood_radius`]. On pixelmap's benchmarks a radius of 9 here was the
    /// cheap gain, for 20–30% more time per pair.
    ///
    /// # Panics
    /// If `radius` is 0 or above [`MAX_NEIGHBORHOOD_RADIUS`].
    pub fn init_neighborhood_radius(mut self, radius: impl Into<Option<usize>>) -> Self {
        self.init_neighborhood_radius = radius.into().map(checked_radius);
        self
    }

    /// The seed to use, resolving `None` to [`DEFAULT_SEED`].
    pub(crate) fn resolved_seed(&self) -> u64 {
        self.seed.unwrap_or(DEFAULT_SEED)
    }
}

/// `radius`, if pixelmap accepts it as a disc radius. Checked here rather than left to
/// pixelmap, so that a bad value panics where it is set and not minutes into a run.
fn checked_radius(radius: usize) -> usize {
    assert!(
        (1..=MAX_NEIGHBORHOOD_RADIUS).contains(&radius),
        "neighborhood radius must be in 1..={MAX_NEIGHBORHOOD_RADIUS}, not {radius}"
    );
    radius
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
        assert_eq!(options.neighborhood_radius, None);
        assert_eq!(options.init_neighborhood_radius, None);
    }

    #[test]
    fn a_radius_can_be_set_and_reset() {
        let options = Options::new()
            .neighborhood_radius(7)
            .init_neighborhood_radius(9);
        assert_eq!(options.neighborhood_radius, Some(7));
        assert_eq!(options.init_neighborhood_radius, Some(9));
        let reset = options
            .neighborhood_radius(None)
            .init_neighborhood_radius(None);
        assert_eq!(reset, Options::new());
    }

    #[test]
    #[should_panic(expected = "neighborhood radius")]
    fn a_radius_of_zero_is_refused() {
        let _ = Options::new().neighborhood_radius(0);
    }

    #[test]
    #[should_panic(expected = "neighborhood radius")]
    fn a_radius_pixelmap_cannot_score_is_refused() {
        let _ = Options::new().init_neighborhood_radius(MAX_NEIGHBORHOOD_RADIUS + 1);
    }
}
