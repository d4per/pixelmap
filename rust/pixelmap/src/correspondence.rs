//! The crate's entry point: run the pipeline over a pair of photos and hold the result.

use std::sync::Arc;

use crate::dense_photo_map::DensePhotoMap;
use crate::error::Error;
use crate::photo::Photo;
use crate::pixelmap_processor::{PixelMapProcessor, DEFAULT_SEED};
use crate::processing_mode::{IterationParams, Quality};

/// How far along a run is, reported to [`Builder::run_with_progress`].
///
/// The library never prints: a caller who wants to show progress asks for it, and one
/// who does not is not made to pay for output they did not ask for.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct Progress {
    /// Schedule steps finished so far, counting the initial matching pass.
    pub step: usize,
    /// Total number of steps this run will perform.
    pub total: usize,
}

impl Progress {
    /// How far along the run is, from `0.0` to `1.0`.
    pub fn fraction(&self) -> f32 {
        if self.total == 0 {
            1.0
        } else {
            self.step as f32 / self.total as f32
        }
    }
}

/// A finished mapping between two photos, in both directions.
///
/// Build one with [`Correspondence::builder`], or with [`crate::correspond`] to accept
/// every default.
#[derive(Clone, Debug)]
pub struct Correspondence {
    forward: DensePhotoMap,
    backward: DensePhotoMap,
    comparisons: usize,
    source_size: (usize, usize),
}

impl Correspondence {
    /// Starts configuring a run.
    pub fn builder() -> Builder {
        Builder::new()
    }

    /// Where the pixel at `(x, y)` of the first photo ended up in the second, or `None`
    /// if the algorithm could not map that point.
    ///
    /// Both the argument and the result are in the coordinates of the photos you passed
    /// in. The solver works at a reduced resolution internally (see
    /// [`Self::working_scale`]); this converts in and out of it for you, which is almost
    /// always what a caller wants. [`Self::forward`] exposes the underlying grid for
    /// those who need the working-resolution numbers.
    pub fn lookup(&self, x: f32, y: f32) -> Option<(f32, f32)> {
        let scale = self.working_scale();
        let (mx, my) = self.forward.lookup(x * scale, y * scale)?;
        Some((mx / scale, my / scale))
    }

    /// The reverse of [`Self::lookup`]: where a pixel of the second photo came from in
    /// the first.
    pub fn lookup_back(&self, x: f32, y: f32) -> Option<(f32, f32)> {
        let scale = self.working_scale();
        let (mx, my) = self.backward.lookup(x * scale, y * scale)?;
        Some((mx / scale, my / scale))
    }

    /// The dimensions of the photos this mapping was computed from.
    pub fn source_dimensions(&self) -> (usize, usize) {
        self.source_size
    }

    /// The factor between source pixels and the working resolution the solver finished
    /// at: multiply a source coordinate by this to index [`Self::forward`] directly.
    ///
    /// The pipeline scales the photos down to keep the search tractable, so the raw grids
    /// are not in input coordinates. Below `1.0` when the inputs were larger than the
    /// schedule's final working width.
    pub fn working_scale(&self) -> f32 {
        self.forward.dimensions().0 as f32 / self.source_size.0 as f32
    }

    /// The mapping from the first photo into the second, in working-resolution
    /// coordinates.
    pub fn forward(&self) -> &DensePhotoMap {
        &self.forward
    }

    /// The mapping from the second photo back into the first, in working-resolution
    /// coordinates.
    pub fn backward(&self) -> &DensePhotoMap {
        &self.backward
    }

    /// Splits the pair apart, in `(forward, backward)` order.
    pub fn into_parts(self) -> (DensePhotoMap, DensePhotoMap) {
        (self.forward, self.backward)
    }

    /// The fraction of the first photo that ended up with a usable mapping, from `0.0`
    /// to `1.0`.
    ///
    /// Well below 1.0 is normal and not a failure: occlusions, featureless regions and
    /// anything the forward/backward consistency check rejected are all excluded. A very
    /// low value means the two photos had little in common — or that they are related by
    /// something the affine grid cannot express.
    pub fn coverage(&self) -> f32 {
        self.forward.calculate_used_area()
    }

    /// How many region comparisons the solver performed. A rough measure of the work done.
    pub fn comparisons(&self) -> usize {
        self.comparisons
    }
}

/// Configures a correspondence-mapping run.
///
/// ```no_run
/// use pixelmap::{Correspondence, Photo, Quality};
///
/// # fn photos() -> (Photo, Photo) { unimplemented!() }
/// # let (a, b) = photos();
/// let mapping = Correspondence::builder()
///     .quality(Quality::High)
///     .seed(42)
///     .run(a, b)?;
/// # Ok::<(), pixelmap::Error>(())
/// ```
#[derive(Clone, Debug)]
pub struct Builder {
    quality: Quality,
    schedule: Option<Vec<IterationParams>>,
    seed: u64,
    final_max_dist: f32,
}

impl Default for Builder {
    fn default() -> Self {
        Builder::new()
    }
}

impl Builder {
    /// A builder with every default: [`Quality::Low`] and [`DEFAULT_SEED`].
    pub fn new() -> Self {
        Builder {
            quality: Quality::Low,
            schedule: None,
            seed: DEFAULT_SEED,
            final_max_dist: 2.0,
        }
    }

    /// How much work to put in. Defaults to [`Quality::Low`].
    pub fn quality(mut self, quality: Quality) -> Self {
        self.quality = quality;
        self
    }

    /// Replaces the quality preset's iteration schedule with an explicit one.
    ///
    /// For callers tuning the algorithm itself; [`Self::quality`] covers ordinary use.
    /// The working width the run starts from still comes from [`Self::quality`].
    pub fn schedule(mut self, steps: impl Into<Vec<IterationParams>>) -> Self {
        self.schedule = Some(steps.into());
        self
    }

    /// Pins the seed for the solver's queue shuffling. Defaults to [`DEFAULT_SEED`].
    ///
    /// Two runs that agree on inputs, schedule and seed produce the same mapping.
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// How far a round trip through both mappings may land from where it started, in
    /// working-resolution pixels, before a cell is discarded. Defaults to `2.0`.
    ///
    /// Lower keeps only the mappings both directions agree on closely, at the cost of
    /// coverage; higher keeps more, including some that are wrong.
    pub fn max_round_trip_error(mut self, max_dist: f32) -> Self {
        self.final_max_dist = max_dist;
        self
    }

    /// Runs the pipeline.
    ///
    /// # Errors
    /// [`Error::SizeMismatch`] if the photos differ in size, [`Error::EmptyPhoto`] or
    /// [`Error::PhotoTooSmall`] if either is unusable.
    pub fn run(
        &self,
        photo1: impl Into<Arc<Photo>>,
        photo2: impl Into<Arc<Photo>>,
    ) -> Result<Correspondence, Error> {
        self.run_with_progress(photo1, photo2, |_| {})
    }

    /// Runs the pipeline, calling `on_progress` after the initial matching pass and after
    /// every schedule step.
    ///
    /// # Errors
    /// As [`Self::run`].
    pub fn run_with_progress(
        &self,
        photo1: impl Into<Arc<Photo>>,
        photo2: impl Into<Arc<Photo>>,
        mut on_progress: impl FnMut(Progress),
    ) -> Result<Correspondence, Error> {
        let photo1 = photo1.into();
        let photo2 = photo2.into();
        let source_size = (photo1.width(), photo1.height());

        photo1.validate()?;
        photo2.validate()?;
        if (photo1.width(), photo1.height()) != (photo2.width(), photo2.height()) {
            return Err(Error::SizeMismatch {
                first: (photo1.width(), photo1.height()),
                second: (photo2.width(), photo2.height()),
            });
        }

        let steps: &[IterationParams] =
            self.schedule.as_deref().unwrap_or_else(|| self.quality.steps());
        // The initial matching pass counts as a step, so the reported total matches the
        // number of callbacks.
        let total = steps.len() + 1;

        let mut processor =
            PixelMapProcessor::with_seed(photo1, photo2, self.quality.photo_width(), self.seed);
        processor.init();
        on_progress(Progress { step: 1, total });

        for (index, params) in steps.iter().enumerate() {
            params.apply(&mut processor);
            on_progress(Progress { step: index + 2, total });
        }

        let comparisons = processor.get_total_comparisons();
        let (forward, backward) = processor.get_result(self.final_max_dist);
        Ok(Correspondence { forward, backward, comparisons, source_size })
    }
}

/// Maps two photos onto each other with every default.
///
/// Shorthand for `Correspondence::builder().run(photo1, photo2)`.
///
/// # Errors
/// As [`Builder::run`].
pub fn correspond(
    photo1: impl Into<Arc<Photo>>,
    photo2: impl Into<Arc<Photo>>,
) -> Result<Correspondence, Error> {
    Correspondence::builder().run(photo1, photo2)
}
