//! Progress reporting and cancellation.
//!
//! A reconstruction runs one pixelmap solve per pair of photos, which is minutes of work
//! for a handful of photos. The caller gets an [`Event`] at every checkpoint and answers
//! with a [`Flow`]; `Break` stops the run at that checkpoint.

use std::fmt;
use std::ops::ControlFlow;

/// What a progress callback returns. `ControlFlow::Break(())` asks the pipeline to stop,
/// and the run then returns [`crate::Error::Cancelled`].
pub type Flow = ControlFlow<()>;

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
    /// These are estimates until the stages exist and can be timed. Pairwise
    /// correspondence dominates, since it is N(N−1)/2 solver runs, and fusion is the only
    /// other stage expected to take noticeable time.
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
}

impl fmt::Display for Stage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// A progress report from a running reconstruction.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct Event {
    /// The stage that is running.
    pub stage: Stage,
    /// How far through that stage the run is, from `0.0` to `1.0`.
    pub stage_fraction: f32,
    /// What is happening, for display.
    pub message: String,
}

impl Event {
    /// A report that `stage` is `stage_fraction` done. The fraction is clamped to `[0, 1]`.
    pub fn new(stage: Stage, stage_fraction: f32, message: impl Into<String>) -> Self {
        Event {
            stage,
            stage_fraction: stage_fraction.clamp(0.0, 1.0),
            message: message.into(),
        }
    }

    /// How far through the whole run this is, from `0.0` to `1.0`, with each stage
    /// weighted by its expected cost.
    pub fn fraction(&self) -> f32 {
        let finished: f32 = Stage::ALL
            .iter()
            .take_while(|&&s| s < self.stage)
            .map(|s| s.weight())
            .sum();
        (finished + self.stage.weight() * self.stage_fraction).min(1.0)
    }
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
    fn overall_fraction_only_moves_forwards() {
        let mut last = 0.0;
        for stage in Stage::ALL {
            for step in 0..=4 {
                let fraction = Event::new(stage, step as f32 / 4.0, "").fraction();
                assert!(fraction >= last, "{stage} at {step}/4 went backwards");
                last = fraction;
            }
        }
        assert!((last - 1.0).abs() < 1e-4);
        assert_eq!(Event::new(Stage::Pairs, 7.0, "").stage_fraction, 1.0);
    }
}
