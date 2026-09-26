//! The captures a reconstruction cannot be made from must fail with a reason a
//! photographer can act on, not produce a plausible-looking wrong model.

use std::sync::Arc;

use pixelmap::Photo;
use pixelmap_multiview::synthetic::{Scene, SyntheticSet};
use pixelmap_multiview::{
    pipeline, Degeneracy, DropReason, Error, Event, Flow, Focal, Model, Options, PairGraph,
    PairLookup, PhotoPx, Stage, ViewId,
};

const WIDTH: usize = 320;
const HEIGHT: usize = 240;

fn photos(set: &SyntheticSet) -> Vec<Arc<Photo>> {
    (0..set.views())
        .map(|v| Arc::new(set.render(ViewId(v as u32))))
        .collect()
}

fn attempt(set: &SyntheticSet) -> Result<Model, Error> {
    let graph = PairGraph::from_fn(set.views(), |pair| set.pair(pair));
    pipeline::reconstruct(
        &graph,
        &photos(set),
        &Options::new().focal(Focal::Pixels(set.intrinsics.fx)),
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
            reasons
                .iter()
                .all(|(_, reason)| matches!(reason, Degeneracy::Planar { .. })),
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
fn a_photo_that_cannot_be_placed_is_named_with_the_reason() {
    let set = SyntheticSet::orbit(Scene::corner(), 3, 30.0, WIDTH, HEIGHT);
    let graph = PairGraph::from_fn(set.views(), |pair| set.pair(pair));
    // No view outside the seed pair can ever see this many points.
    // Driving one stage to a chosen outcome needs the threshold itself, which `Options`
    // deliberately does not expose. `reconstruct_with_params` is hidden for exactly this.
    let params = pipeline::Params {
        sfm: pixelmap_multiview::sfm::Params {
            min_registration_points: usize::MAX,
            ..Default::default()
        },
        ..Default::default()
    };
    let mut messages = Vec::new();
    let error = pipeline::reconstruct_with_params(
        &graph,
        &photos(&set),
        &set.intrinsics,
        &params,
        &mut |event| {
            messages.push(event.message().into_owned());
            Flow::Continue(())
        },
    )
    .expect_err("only the seed pair can be placed");

    let message = error.to_string();
    let Error::RegistrationFailed {
        registered,
        left_out,
        ..
    } = &error
    else {
        panic!("expected a registration failure, got {message}");
    };
    assert_eq!(registered.len(), 2, "{message}");
    assert_eq!(left_out.len(), 1, "{message}");
    let (view, reason) = &left_out[0];
    assert!(!registered.contains(view), "{message}");
    assert!(
        matches!(reason, DropReason::TooFewPoints { .. }),
        "{message}"
    );
    assert!(
        message.contains(&format!("{view} was left out: {reason}")),
        "{message}"
    );
    // Reported as it happened too, so a caller showing progress sees it before the error.
    assert!(
        messages.iter().any(|m| m.contains("was left out")),
        "{messages:?}"
    );
}

#[test]
fn photos_with_nothing_in_common_do_not_connect() {
    let set = SyntheticSet::orbit(Scene::corner(), 3, 20.0, WIDTH, HEIGHT);
    let graph = PairGraph::from_fn(3, |_| Nothing);
    let error = pipeline::reconstruct(
        &graph,
        &photos(&set),
        &Options::new().focal(Focal::Pixels(set.intrinsics.fx)),
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
        &Options::new().focal(Focal::Pixels(set.intrinsics.fx)),
        &mut |event| {
            stages.push(event.stage());
            if let Event::Stage {
                stage: Stage::Depth,
                fraction,
                ..
            } = event
            {
                first_depth.get_or_insert(fraction);
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
