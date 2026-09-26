//! What a running reconstruction tells the caller, and how to stop it.
//!
//! A reconstruction runs one pixelmap solve per pair of photos, which is minutes of work
//! for a handful of photos. The caller gets an [`Event`] at every checkpoint and answers
//! with a [`Flow`]; `Break` stops the run, and the run then returns
//! [`Error::Cancelled`](crate::Error::Cancelled).
//!
//! Events carry data, not prose. An earlier version reported everything as a formatted
//! sentence, and the web frontend ended up parsing those sentences back into facts with
//! regular expressions — which made the exact wording, down to the en dash in a
//! [`PairId`]'s `Display`, part of this crate's contract by accident. Every fact a caller
//! is likely to want is now a field: which pair, how far through it, why one was rejected.
//! [`Event::message`] renders an event as a line for a log, for callers that want one.

use std::borrow::Cow;
use std::fmt;
use std::ops::ControlFlow;

use crate::sfm::DropReason;
use crate::twoview::Degeneracy;
use crate::types::{PairId, PhotoPx, ViewId};

/// What a progress callback returns. `ControlFlow::Break(())` asks the pipeline to stop,
/// and the run then returns [`crate::Error::Cancelled`].
pub type Flow = ControlFlow<()>;

/// How much a message matters.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Level {
    /// Something that happened, worth showing but not worth worrying about.
    Info,
    /// Something that will make the result worse, but not so wrong as to stop the run.
    Warning,
}

impl fmt::Display for Level {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Level::Info => "info",
            Level::Warning => "warning",
        })
    }
}

/// The dense correspondence found between one pair of photos.
///
/// Handed over on the event that finishes a pair, so that a caller can show what the
/// matcher made of it. This is the raw grid, not a picture: how to draw it is the caller's
/// decision.
///
/// Deliberately plain data — no borrows, no generics, no `Arc` — so it survives a trip
/// across a language boundary unchanged.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct PairMap {
    /// The pair this maps.
    pub pair: PairId,
    /// Grid columns.
    pub columns: usize,
    /// Grid rows.
    pub rows: usize,
    /// Photo pixels between neighbouring grid cells.
    pub cell_size: f32,
    /// Where each cell of the first photo lands in the second, as `x, y` pairs, row by
    /// row. `NaN` where the cell is unmapped.
    ///
    /// In the coordinates of the photos handed to the pipeline, not the lower resolution
    /// the solver works at.
    pub points: Vec<f32>,
    /// The fraction of the first photo that is mapped, from `0.0` to `1.0`.
    pub coverage: f32,
}

impl PairMap {
    /// The position in the first photo of grid cell `(column, row)`.
    pub fn pixel(&self, column: usize, row: usize) -> PhotoPx {
        PhotoPx::new(column as f32 * self.cell_size, row as f32 * self.cell_size)
    }

    /// Where grid cell `(column, row)` lands in the second photo, if it is mapped.
    pub fn point(&self, column: usize, row: usize) -> Option<PhotoPx> {
        let i = (row * self.columns + column) * 2;
        let (x, y) = (*self.points.get(i)?, *self.points.get(i + 1)?);
        (x.is_finite() && y.is_finite()).then(|| PhotoPx::new(x, y))
    }

    /// How many cells are mapped.
    pub fn mapped(&self) -> usize {
        self.points
            .iter()
            .step_by(2)
            .filter(|x| x.is_finite())
            .count()
    }
}

/// The pipeline's stages, in the order they run.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[non_exhaustive]
pub enum Stage {
    /// Checking the photos and intrinsics.
    Input,
    /// Dense correspondence between every pair of photos.
    Pairs,
    /// Relative pose of each pair, and triage of pairs too weak to use.
    TwoView,
    /// Points followed across views, filtered for cycle consistency.
    Tracks,
    /// Every camera placed in one frame.
    Registration,
    /// Joint refinement of cameras and points.
    BundleAdjustment,
    /// A dense depth map per view.
    Depth,
    /// Depth maps merged into one mesh.
    Fusion,
    /// Colour from the photos onto the mesh.
    Texture,
}

impl Stage {
    /// Every stage, in order.
    pub const ALL: [Stage; 9] = [
        Stage::Input,
        Stage::Pairs,
        Stage::TwoView,
        Stage::Tracks,
        Stage::Registration,
        Stage::BundleAdjustment,
        Stage::Depth,
        Stage::Fusion,
        Stage::Texture,
    ];

    /// A short human-readable name.
    pub fn name(self) -> &'static str {
        match self {
            Stage::Input => "input",
            Stage::Pairs => "pairwise correspondence",
            Stage::TwoView => "two-view geometry",
            Stage::Tracks => "tracks",
            Stage::Registration => "registration",
            Stage::BundleAdjustment => "bundle adjustment",
            Stage::Depth => "dense depth",
            Stage::Fusion => "fusion",
            Stage::Texture => "texturing",
        }
    }

    /// The share of a typical run's time spent in this stage. Used to turn per-stage
    /// progress into one overall fraction.
    ///
    /// Rough shares, not a measurement of any one run: the real split moves with the
    /// quality and the number of photos. Pairwise correspondence dominates, since it is
    /// N(N−1)/2 solver runs; dense depth and fusion are the only other stages that take
    /// noticeable time.
    fn weight(self) -> f32 {
        match self {
            Stage::Input => 0.01,
            Stage::Pairs => 0.70,
            Stage::TwoView => 0.03,
            Stage::Tracks => 0.02,
            Stage::Registration => 0.02,
            Stage::BundleAdjustment => 0.03,
            Stage::Depth => 0.08,
            Stage::Fusion => 0.08,
            Stage::Texture => 0.03,
        }
    }

    /// How far through the whole run a stage that is `fraction` done is, with each stage
    /// weighted by its expected cost.
    fn overall(self, fraction: f32) -> f32 {
        let finished: f32 = Stage::ALL
            .iter()
            .take_while(|&&s| s < self)
            .map(|s| s.weight())
            .sum();
        (finished + self.weight() * fraction.clamp(0.0, 1.0)).min(1.0)
    }
}

impl fmt::Display for Stage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// Something that happened while reconstructing.
///
/// Each variant carries what it is about, so that a caller never has to parse
/// [`Event::message`] to find out. The pair variants carry their position in the run as
/// well (`index` of `of`), because a caller showing progress needs it and deriving it from
/// a fraction means reimplementing this crate's pair ordering.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub enum Event {
    /// The run has begun, and this is how much work it is.
    Started {
        /// How many photos.
        views: usize,
        /// How many pairs of them, which is how many solver runs.
        pairs: usize,
    },

    /// A stage reached a checkpoint. `fraction` runs from `0.0` to `1.0` within the stage.
    Stage {
        /// The stage.
        stage: Stage,
        /// How far through that stage the run is.
        fraction: f32,
        /// What happened, for display.
        message: String,
    },

    /// The solver advanced within one pair's mapping.
    PairProgress {
        /// The pair.
        pair: PairId,
        /// Its position in the run, from zero.
        index: usize,
        /// How many pairs there are.
        of: usize,
        /// Schedule steps finished so far, counting the initial matching pass.
        step: u32,
        /// How many steps this pair takes.
        steps: u32,
    },

    /// One pair's mapping is finished, and here it is.
    PairMapped {
        /// The pair.
        pair: PairId,
        /// Its position in the run, from zero.
        index: usize,
        /// How many pairs there are.
        of: usize,
        /// The correspondence found. Boxed because only one event per pair carries a map
        /// and the rest should not pay for the space.
        map: Box<PairMap>,
    },

    /// Two-view geometry cannot use a pair. Its correspondences can still contribute to
    /// dense depth, where more than two views condition the result.
    PairRejected {
        /// The pair.
        pair: PairId,
        /// Its position in the run, from zero.
        index: usize,
        /// How many pairs there are.
        of: usize,
        /// Why it cannot be used.
        reason: Degeneracy,
    },

    /// A photo was left out of the reconstruction.
    ViewDropped {
        /// The stage that left it out.
        stage: Stage,
        /// The photo.
        view: ViewId,
        /// Why.
        reason: DropReason,
    },

    /// Something worth saying that is not progress.
    Log {
        /// The stage that said it.
        stage: Stage,
        /// How much it matters.
        level: Level,
        /// What it is.
        message: String,
    },
}

impl Event {
    /// The stage this event came from.
    pub fn stage(&self) -> Stage {
        match self {
            Event::Started { .. } => Stage::Input,
            Event::Stage { stage, .. } => *stage,
            Event::PairProgress { .. } | Event::PairMapped { .. } => Stage::Pairs,
            Event::PairRejected { .. } => Stage::TwoView,
            Event::ViewDropped { stage, .. } | Event::Log { stage, .. } => *stage,
        }
    }

    /// How far through the whole run this event is, from `0.0` to `1.0`, with each stage
    /// weighted by its expected cost.
    ///
    /// `None` when the event says nothing about progress — a log line or a dropped view
    /// can happen at any point within a stage, and letting those report the stage's start
    /// would drive a progress bar backwards. A caller showing a bar leaves it where it is.
    pub fn progress(&self) -> Option<f32> {
        let within =
            |index: usize, of: usize, extra: f32| (index as f32 + extra) / of.max(1) as f32;
        let fraction = match self {
            Event::Started { .. } => 0.0,
            Event::Stage { fraction, .. } => *fraction,
            Event::PairProgress {
                index,
                of,
                step,
                steps,
                ..
            } => within(*index, *of, *step as f32 / (*steps).max(1) as f32),
            Event::PairMapped { index, of, .. } | Event::PairRejected { index, of, .. } => {
                within(*index + 1, *of, 0.0)
            }
            Event::ViewDropped { .. } | Event::Log { .. } => return None,
        };
        Some(self.stage().overall(fraction))
    }

    /// The event as one line for a log.
    ///
    /// Borrowed where the event already holds the words, allocated where it has to put
    /// numbers into them.
    pub fn message(&self) -> Cow<'_, str> {
        match self {
            Event::Started { views, pairs } => {
                Cow::Owned(format!("{views} photos, {pairs} pairs to map"))
            }
            Event::Stage { message, .. } => Cow::Borrowed(message),
            Event::PairProgress {
                pair, step, steps, ..
            } => Cow::Owned(format!("mapping {pair}, step {step}/{steps}")),
            Event::PairMapped { pair, map, .. } => Cow::Owned(format!(
                "mapped {pair}: {:.0}% coverage",
                map.coverage * 100.0
            )),
            Event::PairRejected { pair, reason, .. } => {
                Cow::Owned(format!("{pair}: not usable, {reason}"))
            }
            Event::ViewDropped { view, reason, .. } => {
                Cow::Owned(format!("{view} was left out: {reason}"))
            }
            Event::Log { message, .. } => Cow::Borrowed(message),
        }
    }
}

/// Where a run has got to, for a caller that would rather ask than follow every event.
///
/// Fold events into one of these with [`Status::update`]; that is all
/// `Job::status` does.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct Status {
    /// The stage now running.
    pub stage: Stage,
    /// How far through the whole run it is, from `0.0` to `1.0`.
    pub progress: f32,
    /// How many pairs have been mapped.
    pub pairs_mapped: usize,
    /// How many there are to map. Zero until the run reports how much work it is.
    pub pairs_total: usize,
    /// The most recent line worth showing.
    pub message: String,
}

impl Default for Status {
    fn default() -> Self {
        Status {
            stage: Stage::Input,
            progress: 0.0,
            pairs_mapped: 0,
            pairs_total: 0,
            message: String::new(),
        }
    }
}

impl Status {
    /// A status at the very start of a run.
    pub fn new() -> Self {
        Status::default()
    }

    /// Folds one event in.
    pub fn update(&mut self, event: &Event) {
        self.stage = event.stage();
        if let Some(progress) = event.progress() {
            // Only ever forwards: a warning arriving late in a stage should not appear to
            // undo work already reported.
            self.progress = self.progress.max(progress);
        }
        match event {
            Event::Started { pairs, .. } => self.pairs_total = *pairs,
            Event::PairMapped { .. } => self.pairs_mapped += 1,
            _ => {}
        }
        self.message = event.message().into_owned();
    }
}

/// Reports `event`, and turns a `Break` from the callback into
/// [`crate::Error::Cancelled`].
///
/// Every stage goes through this, so that cancellation means the same thing everywhere.
pub(crate) fn emit(
    on_event: &mut dyn FnMut(Event) -> Flow,
    event: Event,
) -> Result<(), crate::Error> {
    let stage = event.stage();
    if on_event(event).is_break() {
        return Err(crate::Error::Cancelled { stage });
    }
    Ok(())
}

/// Reports that `stage` is `fraction` done.
pub(crate) fn report(
    on_event: &mut dyn FnMut(Event) -> Flow,
    stage: Stage,
    fraction: f32,
    message: String,
) -> Result<(), crate::Error> {
    emit(
        on_event,
        Event::Stage {
            stage,
            fraction: fraction.clamp(0.0, 1.0),
            message,
        },
    )
}

/// A callback that ignores every event and never cancels.
///
/// What the plain, progress-free form of each stage passes to the reporting form. Since it
/// never breaks, those cannot return [`crate::Error::Cancelled`].
pub(crate) fn silent(_: Event) -> Flow {
    Flow::Continue(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weights_cover_the_whole_run() {
        let total: f32 = Stage::ALL.iter().map(|s| s.weight()).sum();
        assert!((total - 1.0).abs() < 1e-4, "weights sum to {total}");
        assert!(Stage::ALL.windows(2).all(|w| w[0] < w[1]));
    }

    #[test]
    fn overall_progress_only_moves_forwards() {
        let mut last = 0.0;
        for stage in Stage::ALL {
            for step in 0..=4 {
                let event = Event::Stage {
                    stage,
                    fraction: step as f32 / 4.0,
                    message: String::new(),
                };
                let progress = event.progress().expect("a stage event reports progress");
                assert!(progress >= last, "{stage} at {step}/4 went backwards");
                last = progress;
            }
        }
        assert!((last - 1.0).abs() < 1e-4);
    }

    #[test]
    fn a_log_line_does_not_move_the_bar() {
        let log = Event::Log {
            stage: Stage::Depth,
            level: Level::Warning,
            message: "something".into(),
        };
        assert!(log.progress().is_none());

        let mut status = Status::new();
        status.update(&Event::Stage {
            stage: Stage::Depth,
            fraction: 0.5,
            message: String::new(),
        });
        let halfway = status.progress;
        status.update(&log);
        assert_eq!(status.progress, halfway, "a log line left the bar alone");
        assert_eq!(status.message, "something");
    }

    #[test]
    fn a_pair_reports_where_it_is_without_the_caller_counting() {
        let started = Event::PairProgress {
            pair: PairId::new(ViewId(0), ViewId(2)).expect("distinct views"),
            index: 1,
            of: 3,
            step: 2,
            steps: 5,
        };
        // Pairs is the whole run's first 70%, after Input's 1%.
        let expected = Stage::Pairs.overall((1.0 + 2.0 / 5.0) / 3.0);
        assert_eq!(started.progress(), Some(expected));
        assert_eq!(started.message(), "mapping views 0–2, step 2/5");
    }
}
