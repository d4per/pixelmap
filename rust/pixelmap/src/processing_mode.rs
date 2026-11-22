//! Ready-made iteration schedules for [`PixelMapProcessor`].
//!
//! Running the correspondence mapping is a sequence of [`PixelMapProcessor::iterate`]
//! calls with gradually changing parameters. The four presets defined here trade
//! processing time for accuracy:
//!
//! - [`ProcessingMode::Low`]: fast, but may be less accurate.
//! - [`ProcessingMode::Medium`]: slower, but more accurate.
//! - [`ProcessingMode::High`]: slower still, and more accurate again.
//! - [`ProcessingMode::Ultra`]: as High, plus extra steps at up to 3200 px for photos
//!   wider than 1600 px. The same as High on smaller photos.
//!
#![doc = include_str!("../doc/quality-scales.md")]
//!
//! ```no_run
//! # use std::sync::Arc;
//! # use pixelmap::photo::Photo;
//! # use pixelmap::pixelmap_processor::PixelMapProcessor;
//! # use pixelmap::processing_mode::ProcessingMode;
//! # let (photo1, photo2) = (Arc::new(Photo::default()), Arc::new(Photo::default()));
//! let mode = ProcessingMode::Low;
//! let mut processor = PixelMapProcessor::new(photo1, photo2, mode.photo_width());
//! processor.set_fill_mode(mode.fill_mode());
//! processor.init();
//! mode.run(&mut processor);
//! let (map1, map2) = processor.finish(2.0);
//! ```

use std::fmt;
use std::str::FromStr;

use crate::pixelmap_processor::{FillMode, PixelMapProcessor};

/// How much work the correspondence mapping should put into a photo pair.
///
/// The name this is exported under at the crate root; `ProcessingMode` is the historical
/// spelling and remains an alias for it.
pub type Quality = ProcessingMode;

/// The largest disc radius the solver can score over.
///
/// The colour differences over a disc are summed in 32-bit integers, and a radius of
/// 51 is the largest whose disc cannot overflow them.
pub const MAX_NEIGHBORHOOD_RADIUS: usize = 51;

/// The parameters of a single [`PixelMapProcessor::iterate`] call.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct IterationParams {
    /// Internal working width the photos are scaled to for this iteration.
    pub photo_width: usize,
    /// Size (in pixels) of one cell of the affine correspondence grid.
    pub grid_cell_size: usize,
    /// Radius, in working pixels, of the disc each cell's candidate matches are scored
    /// over. At most [`MAX_NEIGHBORHOOD_RADIUS`].
    pub neighborhood_radius: usize,
    /// How many smoothing passes are applied to the grid before re-running.
    pub smooth_iterations: usize,
    /// Maximum forward/backward distance before a mapping is treated as an outlier.
    pub clean_max_dist: f32,
}

impl IterationParams {
    /// Runs one iteration with these parameters.
    pub fn apply(&self, processor: &mut PixelMapProcessor) {
        processor.iterate(
            self.photo_width,
            self.grid_cell_size,
            self.neighborhood_radius,
            self.smooth_iterations,
            self.clean_max_dist,
        );
    }
}

/// All presets use the same grid geometry, only the working width and the
/// outlier threshold vary between steps.
const fn step(photo_width: usize, clean_max_dist: f32) -> IterationParams {
    IterationParams {
        photo_width,
        grid_cell_size: 5,
        neighborhood_radius: 5,
        smooth_iterations: 2,
        clean_max_dist,
    }
}

const LOW_STEPS: [IterationParams; 4] = [
    step(400, 3.0),
    step(400, 2.0),
    step(400, 1.0),
    step(400, 1.0),
];

const MEDIUM_STEPS: [IterationParams; 10] = [
    step(400, 3.0),
    step(400, 2.0),
    step(400, 1.0),
    step(400, 0.5),
    step(400, 1.0),
    step(400, 2.0),
    step(400, 2.0),
    step(800, 3.0),
    step(800, 1.0),
    step(800, 2.0),
];

const HIGH_STEPS: [IterationParams; 13] = [
    step(400, 3.0),
    step(400, 2.0),
    step(400, 1.0),
    step(400, 0.5),
    step(400, 1.0),
    step(400, 2.0),
    step(400, 2.0),
    step(800, 3.0),
    step(800, 1.0),
    step(800, 2.0),
    step(1600, 1.0),
    step(1600, 2.0),
    step(1600, 2.0),
];

/// [`HIGH_STEPS`], then two steps at 3200 px that only run for photos wider than 1600 px.
const ULTRA_STEPS: [IterationParams; 15] = [
    step(400, 3.0),
    step(400, 2.0),
    step(400, 1.0),
    step(400, 0.5),
    step(400, 1.0),
    step(400, 2.0),
    step(400, 2.0),
    step(800, 3.0),
    step(800, 1.0),
    step(800, 2.0),
    step(1600, 1.0),
    step(1600, 2.0),
    step(1600, 2.0),
    step(3200, 1.0),
    step(3200, 2.0),
];

/// Ultra's extra steps run only above this source width, where High's schedule ends.
const ULTRA_EXTRA_ABOVE: usize = 1600;

/// How much work the correspondence mapping should put into a photo pair.
///
/// Each preset is a coarse-to-fine schedule of [`IterationParams`]. Higher settings both
/// run more iterations and finish at a higher working resolution, so cost grows faster
/// than the step count suggests. Above [`ProcessingMode::Low`], no step works wider than
/// the photos themselves (see [`ProcessingMode::steps_for`]), so on small photos the
/// higher presets cost less than their nominal widths suggest.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ProcessingMode {
    /// Fast, but may be less accurate.
    Low,
    /// Slower, but more accurate.
    Medium,
    /// Slower still, and more accurate again.
    High,
    /// The best result, and the slowest for photos wider than 1600 px: High, then two
    /// more steps at the photo's own width, up to 3200 px. For photos up to 1600 px wide
    /// it is the same as [`ProcessingMode::High`].
    Ultra,
}

impl ProcessingMode {
    /// All modes, in increasing order of processing time.
    pub const ALL: [ProcessingMode; 4] = [
        ProcessingMode::Low,
        ProcessingMode::Medium,
        ProcessingMode::High,
        ProcessingMode::Ultra,
    ];

    /// The lowercase name of this mode, as accepted on the command line.
    pub fn name(&self) -> &'static str {
        match self {
            ProcessingMode::Low => "low",
            ProcessingMode::Medium => "medium",
            ProcessingMode::High => "high",
            ProcessingMode::Ultra => "ultra",
        }
    }

    /// The working width to pass to [`PixelMapProcessor::new`] for this mode: the
    /// width the initial feature matching runs at (or the photos' own width, if they
    /// are narrower).
    pub fn photo_width(&self) -> usize {
        match self {
            ProcessingMode::Low => 400,
            ProcessingMode::Medium => 800,
            ProcessingMode::High | ProcessingMode::Ultra => 1600,
        }
    }

    /// The nominal iteration schedule of this mode, for photos wider than every step.
    ///
    /// What actually runs depends on the photos' width; see [`Self::steps_for`].
    pub fn steps(&self) -> &'static [IterationParams] {
        match self {
            ProcessingMode::Low => &LOW_STEPS,
            ProcessingMode::Medium => &MEDIUM_STEPS,
            ProcessingMode::High => &HIGH_STEPS,
            ProcessingMode::Ultra => &ULTRA_STEPS,
        }
    }

    /// The iteration schedule this mode runs for photos `source_width` pixels wide.
    ///
    /// Above [`Self::Low`], every step works at no more than the photos' own width:
    /// scaling a photo up adds no detail to match, only interpolation the solver then
    /// fits to, which measurably lowered accuracy and doubled the time. [`Self::Ultra`]
    /// also drops its 3200 px steps unless the photos are wider than 1600 px, where
    /// [`Self::High`] stops. [`Self::Low`] runs [`Self::steps`] as it is.
    pub fn steps_for(&self, source_width: usize) -> Vec<IterationParams> {
        let steps = self.steps();
        match self {
            ProcessingMode::Low => steps.to_vec(),
            ProcessingMode::Ultra if source_width <= ULTRA_EXTRA_ABOVE => {
                capped(&steps[..HIGH_STEPS.len()], source_width)
            }
            _ => capped(steps, source_width),
        }
    }

    /// How the mode fills the cells the final map left empty; see [`FillMode`].
    ///
    /// The presets above [`Self::Low`] fill them from earlier, coarser iterations, which
    /// raises coverage and the share of pixels mapped within a few pixels of the truth,
    /// at the cost of a larger average error over the mapped pixels. Override it with
    /// [`crate::Builder::fill_mode`].
    pub fn fill_mode(&self) -> FillMode {
        match self {
            ProcessingMode::Low => FillMode::Off,
            _ => FillMode::Plain,
        }
    }

    /// Runs the whole schedule, [`Self::steps_for`] the processor's photos, on an
    /// already initialized processor.
    ///
    /// The mode's [`Self::fill_mode`] is the processor's to set, before this runs.
    pub fn run(&self, processor: &mut PixelMapProcessor) {
        self.run_with_progress(processor, |_, _| {});
    }

    /// Runs the whole schedule, calling `on_step(completed, total)` after every
    /// iteration. Useful for driving a progress indicator.
    pub fn run_with_progress(
        &self,
        processor: &mut PixelMapProcessor,
        mut on_step: impl FnMut(usize, usize),
    ) {
        let steps = self.steps_for(processor.source_width());
        for (index, params) in steps.iter().enumerate() {
            params.apply(processor);
            on_step(index + 1, steps.len());
        }
    }
}

/// `steps` with every working width capped at `source_width`.
fn capped(steps: &[IterationParams], source_width: usize) -> Vec<IterationParams> {
    steps
        .iter()
        .map(|s| IterationParams {
            photo_width: s.photo_width.min(source_width),
            ..*s
        })
        .collect()
}

/// The string handed to [`ProcessingMode::from_str`] did not name a mode.
///
/// A distinct type rather than a `String`, so that a caller can match on the failure and
/// so that the impl satisfies [`std::error::Error`] like the rest of the crate's errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseProcessingModeError {
    /// The input that could not be parsed.
    input: String,
}

impl ParseProcessingModeError {
    /// The string that was not a mode name.
    pub fn input(&self) -> &str {
        &self.input
    }
}

impl fmt::Display for ParseProcessingModeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "invalid processing mode: {}. Use 'low', 'medium', 'high', or 'ultra'.",
            self.input
        )
    }
}

impl std::error::Error for ParseProcessingModeError {}

impl FromStr for ProcessingMode {
    type Err = ParseProcessingModeError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let name = s.trim().to_lowercase();
        ProcessingMode::ALL
            .into_iter()
            .find(|mode| mode.name() == name)
            .ok_or_else(|| ParseProcessingModeError {
                input: s.to_owned(),
            })
    }
}

impl fmt::Display for ProcessingMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}
