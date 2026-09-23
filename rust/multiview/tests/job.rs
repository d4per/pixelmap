//! The worker-thread wrapper: a run a caller can follow, poll and stop.
//!
//! Only built with the `threads` feature, which wasm leaves off — there `run` is driven
//! directly, inside a web worker, and everything here is unavailable by design.
//!
//! These exercise the plumbing `Job` adds — the thread, the event channel, the folded
//! status and the cancel token — rather than the reconstruction underneath, which the other
//! tests cover through `run` and `reconstruct`. A run is stopped rather than finished
//! because a full reconstruction through the real matcher costs seconds, and finishing one
//! would test `pipeline::run` all over again.

#![cfg(feature = "threads")]

use std::sync::Arc;

use pixelmap::{Photo, Quality};
use pixelmap_multiview::synthetic::{Scene, SyntheticSet};
use pixelmap_multiview::{Error, Event, Job, Options, Stage, ViewId};

const WIDTH: usize = 200;
const HEIGHT: usize = 150;

fn rendered() -> Vec<Arc<Photo>> {
    let set = SyntheticSet::orbit(Scene::corner(), 3, 20.0, WIDTH, HEIGHT);
    (0..set.views())
        .map(|v| Arc::new(set.render(ViewId(v as u32))))
        .collect()
}

fn job() -> Job {
    Job::start(rendered(), Options::new().quality(Quality::Low))
}

#[test]
fn a_job_says_how_much_work_it_is_before_doing_any() {
    let job = job();
    let first = job
        .events()
        .recv()
        .expect("a run reports before it does anything");
    match first {
        Event::Started { views, pairs } => {
            assert_eq!(views, 3);
            // Three photos make three pairs, and every pair is a solver run.
            assert_eq!(pairs, 3);
        }
        other => panic!("expected the run to announce itself, got {other:?}"),
    }
    job.cancel();
    assert!(matches!(job.join(), Err(Error::Cancelled { .. })));
}

#[test]
fn a_job_stops_when_asked() {
    let job = job();
    job.cancel();
    // Cancelling is not a failure of the photos, but it is how the run ends.
    match job.join() {
        Err(Error::Cancelled { stage }) => {
            assert!(Stage::ALL.contains(&stage), "{stage} is not a known stage");
        }
        other => panic!("expected a cancelled run, got {other:?}"),
    }
}

#[test]
fn a_cancel_token_stops_a_run_from_anywhere() {
    let job = job();
    let canceller = job.canceller();
    // The thread that decides to stop need not be the one holding the job.
    std::thread::spawn(move || canceller.cancel())
        .join()
        .expect("the canceller thread does not panic");
    assert!(matches!(job.join(), Err(Error::Cancelled { .. })));
}

#[test]
fn status_follows_the_events() {
    let job = job();
    let mut seen_progress = 0.0;
    let mut pairs_total = 0;

    for event in job.events() {
        let status = job.status();
        assert!(
            status.progress >= seen_progress,
            "status went backwards, from {seen_progress} to {}",
            status.progress
        );
        seen_progress = status.progress;
        if let Event::Started { pairs, .. } = event {
            pairs_total = pairs;
        }
        // Far enough in to have folded something in; no need to map every pair.
        if status.pairs_total > 0 && !status.message.is_empty() {
            job.cancel();
        }
    }

    let status = job.status();
    assert_eq!(status.pairs_total, pairs_total);
    assert!(
        status.pairs_total > 0,
        "the run never said how much work it is"
    );
    assert!(
        status.pairs_mapped <= status.pairs_total,
        "{} of {} pairs mapped",
        status.pairs_mapped,
        status.pairs_total
    );
    assert!(matches!(job.join(), Err(Error::Cancelled { .. })));
}

#[test]
fn dropping_a_job_does_not_hang() {
    // Nobody is waiting for this any more, so it is asked to stop rather than left to
    // grind through a reconstruction. What matters here is that dropping returns.
    drop(job());
}
