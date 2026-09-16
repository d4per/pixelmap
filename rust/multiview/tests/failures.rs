//! The captures a reconstruction cannot be made from must fail with a reason a
//! photographer can act on, not produce a plausible-looking wrong model.

use std::sync::Arc;

use pixelmap::Photo;
use pixelmap_multiview::pipeline::{self, Reconstruction};
use pixelmap_multiview::synthetic::{Scene, SyntheticSet};
use pixelmap_multiview::{Error, Flow, PairGraph, PairLookup, PhotoPx, Stage, ViewId};

const WIDTH: usize = 320;
const HEIGHT: usize = 240;

fn photos(set: &SyntheticSet) -> Vec<Arc<Photo>> {
    (0..set.views())
        .map(|v| Arc::new(set.render(ViewId(v as u32))))
        .collect()
}

fn attempt(set: &SyntheticSet) -> Result<Reconstruction, Error> {
    let graph = PairGraph::from_fn(set.views(), |pair| set.pair(pair));
    pipeline::reconstruct(
        &graph,
        &photos(set),
        &set.intrinsics,
        &pipeline::Params::default(),
        &mut |_| Flow::Continue(()),
    )
}

#[test]
fn a_camera_that_only_turned_is_told_to_move() {
    let set = SyntheticSet::rotation_only(Scene::corner(), 3, 20.0, WIDTH, HEIGHT);
    let error = attempt(&set).expect_err("no depth can be recovered");
    let message = error.to_string();
    assert!(
        matches!(error, Error::NoUsablePair { ref reasons } if reasons.len() == 3),
        "{message}"
    );
    assert!(message.contains("barely moved"), "{message}");
    assert_eq!(error.stage(), Stage::Registration);
}

#[test]
fn a_flat_scene_is_reported_as_flat() {
    let set = SyntheticSet::orbit(Scene::plane(), 3, 30.0, WIDTH, HEIGHT);
    let error = attempt(&set).expect_err("the 8-point method cannot handle a plane");
    let message = error.to_string();
    assert!(matches!(error, Error::NoUsablePair { .. }), "{message}");
    assert!(message.contains("flat"), "{message}");
    if let Error::NoUsablePair { reasons } = &error {
        assert!(
            reasons.iter().all(|(_, reason)| reason.contains("flat")),
            "every pair should be blamed on the flat scene: {message}"
        );
    }
}

/// A mapping that finds nothing, as between photos of different scenes.
struct Nothing;

impl PairLookup for Nothing {
    fn a_to_b(&self, _: PhotoPx) -> Option<PhotoPx> {
        None
    }
    fn b_to_a(&self, _: PhotoPx) -> Option<PhotoPx> {
        None
    }
    fn coverage(&self) -> f32 {
        0.0
    }
    fn native_stride(&self) -> f32 {
        8.0
    }
    fn precision_px(&self) -> f32 {
        1.0
    }
}

#[test]
fn photos_with_nothing_in_common_do_not_connect() {
    let set = SyntheticSet::orbit(Scene::corner(), 3, 20.0, WIDTH, HEIGHT);
    let graph = PairGraph::from_fn(3, |_| Nothing);
    let error = pipeline::reconstruct(
        &graph,
        &photos(&set),
        &set.intrinsics,
        &pipeline::Params::default(),
        &mut |_| Flow::Continue(()),
    )
    .expect_err("nothing links the views");
    assert!(matches!(error, Error::DisconnectedViews { .. }), "{error}");
    assert!(error.to_string().contains("share too little"), "{error}");
}

#[test]
fn a_run_stops_when_asked() {
    let set = SyntheticSet::orbit(Scene::corner(), 3, 30.0, WIDTH, HEIGHT);
    let graph = PairGraph::from_fn(set.views(), |pair| set.pair(pair));
    let mut stages = Vec::new();
    let mut first_depth = None;
    let result = pipeline::reconstruct(
        &graph,
        &photos(&set),
        &set.intrinsics,
        &pipeline::Params::default(),
        &mut |event| {
            stages.push(event.stage);
            if event.stage == Stage::Depth {
                first_depth.get_or_insert(event.stage_fraction);
                Flow::Break(())
            } else {
                Flow::Continue(())
            }
        },
    );
    assert!(
        matches!(
            result,
            Err(Error::Cancelled {
                stage: Stage::Depth
            })
        ),
        "{:?}",
        result.err()
    );
    assert!(!stages.contains(&Stage::Fusion));
    assert!(stages.contains(&Stage::BundleAdjustment));
    // Depth reports as each view is solved, so the run stops partway through the stage
    // rather than only after it has already done all of its work.
    assert!(
        first_depth.is_some_and(|fraction| fraction < 1.0),
        "depth should report before it finishes, got {first_depth:?}"
    );
}
