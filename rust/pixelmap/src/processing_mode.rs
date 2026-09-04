//! Ready-made iteration schedules for [`PixelMapProcessor`].
//!
//! Running the correspondence mapping is a sequence of [`PixelMapProcessor::iterate`]
//! calls with gradually changing parameters. The three presets defined here trade
//! processing time for accuracy:
//!
//! - [`ProcessingMode::Low`]: fast, but may be less accurate.
//! - [`ProcessingMode::Medium`]: slower, but more accurate.
//! - [`ProcessingMode::High`]: slowest, but likely the best result.
//!
//! ```no_run
//! # use std::sync::Arc;
//! # use pixelmap::photo::Photo;
//! # use pixelmap::pixelmap_processor::PixelMapProcessor;
//! # use pixelmap::processing_mode::ProcessingMode;
//! # let (photo1, photo2) = (Arc::new(Photo::default()), Arc::new(Photo::default()));
//! let mode = ProcessingMode::Low;
//! let mut processor = PixelMapProcessor::new(photo1, photo2, mode.photo_width());
//! processor.init();
//! mode.run(&mut processor);
//! let (map1, map2) = processor.get_result(2.0);
//! ```

use std::fmt;
use std::str::FromStr;

use crate::pixelmap_processor::PixelMapProcessor;

/// How much work the correspondence mapping should put into a photo pair.
///
/// The name this is exported under at the crate root; `ProcessingMode` is the historical
/// spelling and remains an alias for it.
pub type Quality = ProcessingMode;

/// The parameters of a single [`PixelMapProcessor::iterate`] call.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct IterationParams {
    /// Internal working width the photos are scaled to for this iteration.
    pub photo_width: usize,
    /// Size (in pixels) of one cell of the affine correspondence grid.
    pub grid_cell_size: usize,
    /// How far each cell looks around itself for a better match.
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

/// How much work the correspondence mapping should put into a photo pair.
///
/// Each preset is a coarse-to-fine schedule of [`IterationParams`]. Higher settings both
/// run more iterations and finish at a higher working resolution, so cost grows faster
/// than the step count suggests.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ProcessingMode {
    /// Fast, but may be less accurate.
    Low,
    /// Slower, but more accurate.
    Medium,
    /// Slowest, but likely the best result.
    High,
}

impl ProcessingMode {
    /// All modes, in increasing order of processing time.
    pub const ALL: [ProcessingMode; 3] =
        [ProcessingMode::Low, ProcessingMode::Medium, ProcessingMode::High];

    /// The lowercase name of this mode, as accepted on the command line.
    pub fn name(&self) -> &'static str {
        match self {
            ProcessingMode::Low => "low",
            ProcessingMode::Medium => "medium",
            ProcessingMode::High => "high",
        }
    }

    /// The working width to pass to [`PixelMapProcessor::new`] for this mode.
    pub fn photo_width(&self) -> usize {
        match self {
            ProcessingMode::Low => 400,
            ProcessingMode::Medium => 800,
            ProcessingMode::High => 1600,
        }
    }

    /// The iteration schedule of this mode.
    pub fn steps(&self) -> &'static [IterationParams] {
        match self {
            ProcessingMode::Low => &LOW_STEPS,
            ProcessingMode::Medium => &MEDIUM_STEPS,
            ProcessingMode::High => &HIGH_STEPS,
        }
    }

    /// Runs the whole schedule on an already initialized processor.
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
        let steps = self.steps();
        for (index, params) in steps.iter().enumerate() {
            params.apply(processor);
            on_step(index + 1, steps.len());
        }
    }
}

impl FromStr for ProcessingMode {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let name = s.trim().to_lowercase();
        ProcessingMode::ALL
            .into_iter()
            .find(|mode| mode.name() == name)
            .ok_or_else(|| format!("invalid processing mode: {s}. Use 'low', 'medium', or 'high'."))
    }
}

impl fmt::Display for ProcessingMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}
