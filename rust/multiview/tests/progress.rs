//! The feedback a caller gets while a reconstruction runs.
//!
//! A web frontend drives this crate through the progress callback alone, so what the
//! callback sees is part of the contract: the overall fraction has to move forwards, and
//! every stage that takes noticeable time has to report from inside itself rather than
//! only once it is already done.

use std::sync::Arc;

use pixelmap::Photo;
use pixelmap_multiview::pipeline;
use pixelmap_multiview::synthetic::{Scene, SyntheticSet};
use pixelmap_multiview::{Event, Flow, PairGraph, Stage, ViewId};

fn photos(set: &SyntheticSet) -> Vec<Arc<Photo>> {
    (0..set.views())
        .map(|v| Arc::new(set.render(ViewId(v as u32))))
        .collect()
}

/// Every event of one successful run over a synthetic scene.
fn events_of_a_run() -> Vec<Event> {
    let set = SyntheticSet::orbit(Scene::corner(), 4, 36.0, 640, 480);
    let graph = PairGraph::from_fn(set.views(), |pair| set.pair(pair));
    let mut events = Vec::new();
    pipeline::reconstruct(
        &graph,
        &photos(&set),
        &set.intrinsics,
        &pipeline::Params::default(),
        &mut |event| {
            events.push(event);
            Flow::Continue(())
        },
    )
    .expect("the synthetic scene reconstructs");
    events
}

#[test]
fn the_overall_fraction_only_moves_forwards() {
    let events = events_of_a_run();
    assert!(events.len() > 20, "only {} events", events.len());

    let mut last = 0.0;
    for event in &events {
        let fraction = event.fraction();
        assert!(
            fraction >= last,
            "{} went backwards, from {last} to {fraction}: {}",
            event.stage,
            event.message
        );
        last = fraction;
    }
    assert!((last - 1.0).abs() < 1e-4, "a finished run ends at {last}");
}

#[test]
fn every_slow_stage_reports_before_it_finishes() {
    let events = events_of_a_run();
    // `reconstruct` starts from mappings that already exist, so it never runs pairwise
    // correspondence. These are the stages it does run that used to report only once, as
    // they finished.
    for stage in [
        Stage::Registration,
        Stage::BundleAdjustment,
        Stage::Depth,
        Stage::Fusion,
        Stage::Texture,
    ] {
        let partial = events
            .iter()
            .filter(|e| e.stage == stage && e.stage_fraction < 1.0)
            .count();
        assert!(partial > 0, "{stage} reported nothing before it finished");
    }
}

#[test]
fn only_the_events_that_finish_a_pair_carry_a_map() {
    // No pair is mapped here, so no event should carry one.
    assert!(events_of_a_run().iter().all(|e| e.map.is_none()));
}
