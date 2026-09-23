//! The feedback a caller gets while a reconstruction runs.
//!
//! A web frontend drives this crate through the event callback alone, so what the callback
//! sees is part of the contract: progress has to move forwards, every stage that takes
//! noticeable time has to report from inside itself rather than only once it is already
//! done, and the facts a caller needs have to be fields rather than prose to be parsed.

use std::sync::Arc;

use pixelmap::Photo;
use pixelmap_multiview::synthetic::{Scene, SyntheticSet};
use pixelmap_multiview::{pipeline, Event, Flow, Options, PairGraph, Stage, ViewId};

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
        &Options::new(),
        &mut |event| {
            events.push(event);
            Flow::Continue(())
        },
    )
    .expect("the synthetic scene reconstructs");
    events
}

#[test]
fn progress_only_moves_forwards() {
    let events = events_of_a_run();
    assert!(events.len() > 20, "only {} events", events.len());

    let mut last = 0.0;
    for event in &events {
        // A log line or a dropped view says nothing about progress, by design: it can
        // happen anywhere within a stage, and reporting the stage's start would drive a
        // progress bar backwards.
        let Some(progress) = event.progress() else {
            continue;
        };
        assert!(
            progress >= last,
            "{} went backwards, from {last} to {progress}: {}",
            event.stage(),
            event.message()
        );
        last = progress;
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
            .filter(|event| {
                matches!(
                    event,
                    Event::Stage { stage: s, fraction, .. } if *s == stage && *fraction < 1.0
                )
            })
            .count();
        assert!(partial > 0, "{stage} reported nothing before it finished");
    }
}

#[test]
fn no_pair_is_mapped_when_the_mappings_were_given() {
    // `reconstruct` is handed its correspondences, so it maps nothing and should say so by
    // reporting no pair at all, rather than by reporting pairs with nothing in them.
    assert!(!events_of_a_run()
        .iter()
        .any(|event| matches!(event, Event::PairStarted { .. } | Event::PairMapped { .. })));
}

#[test]
fn every_event_says_which_stage_it_came_from() {
    // The stage is what a caller groups a log by, so it is answerable for every event,
    // including the ones that carry no progress.
    for event in events_of_a_run() {
        let stage = event.stage();
        assert!(Stage::ALL.contains(&stage), "{stage} is not a known stage");
        assert!(!event.message().is_empty(), "{stage} reported nothing");
    }
}
