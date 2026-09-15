//! Bundle adjustment against synthetic ground truth.

use std::sync::Arc;

use nalgebra::{Rotation3, Vector3};
use pixelmap::{Quality, DEFAULT_SEED};
use pixelmap_multiview::rng::Rng;
use pixelmap_multiview::sfm::{self, SparseModel};
use pixelmap_multiview::synthetic::{Scene, SyntheticSet};
use pixelmap_multiview::tracks::{self, Track};
use pixelmap_multiview::twoview::{self, RelativePose};
use pixelmap_multiview::{
    align, ba, pairs, Error, Flow, FocalSource, Intrinsics, PairGraph, PairLookup, ViewId,
};

const SIZE: (usize, usize) = (640, 480);

fn relatives<L: PairLookup>(
    graph: &PairGraph<L>,
    intrinsics: &Intrinsics,
    size: (usize, usize),
) -> Vec<RelativePose> {
    graph
        .pairs()
        .enumerate()
        .filter_map(|(i, (pair, lookup))| {
            twoview::estimate(
                pair,
                lookup,
                intrinsics,
                size,
                &twoview::Params::default(),
                &mut Rng::new(DEFAULT_SEED).derive(i as u64),
            )
            .ok()
        })
        .collect()
}

/// Tracks and a registered model, as the pipeline would produce them.
fn register<L: PairLookup>(
    graph: &PairGraph<L>,
    intrinsics: &Intrinsics,
    size: (usize, usize),
) -> (Vec<Track>, SparseModel) {
    let tracks = tracks::build(graph, size, &tracks::Params::default());
    let model = sfm::reconstruct(
        &relatives(graph, intrinsics, size),
        &tracks,
        intrinsics,
        graph.views(),
        graph.precision_px() as f64,
        &sfm::Params::default(),
        &mut Rng::new(DEFAULT_SEED).derive(u64::MAX),
    )
    .expect("the views register");
    (tracks, model)
}

/// Aligns the model to the truth on its cameras and points together, and returns the
/// largest camera error and the median point error, as fractions of the camera spread.
fn errors(set: &SyntheticSet, tracks: &[Track], model: &SparseModel) -> (f64, f64) {
    let mut estimated = Vec::new();
    let mut expected = Vec::new();
    for view in model.registered() {
        estimated.push(model.cameras[view.index()].unwrap().centre());
        expected.push(set.poses[view.index()].centre());
    }
    let cameras = estimated.len();
    for point in &model.points {
        let track = &tracks[point.track];
        let anchor = track.observation(track.anchor).unwrap();
        if let Some(surface) = set.surface_point(track.anchor, anchor) {
            estimated.push(point.position.0);
            expected.push(surface.0);
        }
    }
    let similarity = align::umeyama(&estimated, &expected).expect("alignable");
    let extent = set
        .poses
        .iter()
        .flat_map(|a| {
            set.poses
                .iter()
                .map(move |b| (a.centre() - b.centre()).norm())
        })
        .fold(0.0, f64::max);
    let errors: Vec<f64> = estimated
        .iter()
        .zip(&expected)
        .map(|(e, x)| (similarity.apply(e) - x).norm() / extent)
        .collect();
    let camera = errors[..cameras].iter().copied().fold(0.0, f64::max);
    let mut points = errors[cameras..].to_vec();
    points.sort_by(f64::total_cmp);
    (camera, points[points.len() / 2])
}

#[test]
fn recovers_perturbed_cameras_and_points() {
    let set = SyntheticSet::orbit(Scene::corner(), 4, 36.0, SIZE.0, SIZE.1);
    let graph = PairGraph::from_fn(set.views(), |pair| set.pair(pair));
    let (tracks, mut model) = register(&graph, &set.intrinsics, SIZE);

    let seed_a = model.seed.a();
    let mut rng = Rng::new(11);
    let mut jitter = |amount: f64| {
        Vector3::new(
            rng.next_f64() - 0.5,
            rng.next_f64() - 0.5,
            rng.next_f64() - 0.5,
        ) * amount
    };
    for (view, camera) in model.cameras.iter_mut().enumerate() {
        if view == seed_a.index() {
            continue;
        }
        let pose = camera.as_mut().unwrap();
        pose.rotation = Rotation3::new(jitter(0.02)) * pose.rotation;
        pose.translation += jitter(0.05);
    }
    for point in &mut model.points {
        point.position.0 += jitter(0.05);
    }
    let (camera_before, points_before) = errors(&set, &tracks, &model);

    let adjusted = ba::adjust(
        &model,
        &tracks,
        &set.intrinsics,
        graph.precision_px() as f64,
        &ba::Params::default(),
    )
    .expect("bundle adjustment succeeds");
    let (camera_after, points_after) = errors(&set, &tracks, &adjusted.model);
    let report = &adjusted.report;

    assert!(report.final_median_px < 0.01, "{report:?}");
    assert!(
        camera_after < 1e-3 && points_after < 1e-3,
        "before: cameras {camera_before:.4}, points {points_before:.4}; after: cameras {camera_after:.5}, points {points_after:.5}"
    );
    assert_eq!(
        adjusted.model.cameras[seed_a.index()],
        model.cameras[seed_a.index()]
    );
    assert_eq!(adjusted.intrinsics, set.intrinsics);
}

#[test]
fn corrects_the_registration_of_matched_renders() {
    let set = SyntheticSet::orbit(Scene::corner(), 4, 30.0, 800, 600);
    let size = (set.width, set.height);
    let photos: Vec<_> = (0..set.views())
        .map(|v| Arc::new(set.render(ViewId(v as u32))))
        .collect();
    let graph = pairs::compute(&photos, Quality::Low, DEFAULT_SEED, &mut |_| {
        Flow::Continue(())
    })
    .expect("rendered photos map");
    let (tracks, model) = register(&graph, &set.intrinsics, size);
    let (camera_before, points_before) = errors(&set, &tracks, &model);

    let adjusted = ba::adjust(
        &model,
        &tracks,
        &set.intrinsics,
        graph.precision_px() as f64,
        &ba::Params::default(),
    )
    .expect("bundle adjustment succeeds");
    let (camera_after, points_after) = errors(&set, &tracks, &adjusted.model);

    assert!(
        camera_after < 0.01 && points_after < 0.01,
        "before: cameras {:.2}%, points {:.2}%; after: cameras {:.2}%, points {:.2}%; {:?}",
        camera_before * 100.0,
        points_before * 100.0,
        camera_after * 100.0,
        points_after * 100.0,
        adjusted.report
    );
    assert!(adjusted.report.final_median_px <= adjusted.report.initial_median_px);
}

#[test]
fn refines_a_wrong_focal_length() {
    let set = SyntheticSet::orbit(Scene::corner(), 4, 36.0, SIZE.0, SIZE.1);
    let graph = PairGraph::from_fn(set.views(), |pair| set.pair(pair));
    let mut wrong = set.intrinsics;
    wrong.fx *= 1.15;
    wrong.fy *= 1.15;
    wrong.source = FocalSource::Estimated;
    let (tracks, model) = register(&graph, &wrong, SIZE);

    let adjusted = ba::adjust(
        &model,
        &tracks,
        &wrong,
        graph.precision_px() as f64,
        &ba::Params {
            refine_focal: true,
            ..ba::Params::default()
        },
    )
    .expect("bundle adjustment succeeds");

    let focal_error = adjusted.intrinsics.fx / set.intrinsics.fx - 1.0;
    assert!(
        focal_error.abs() < 0.01,
        "focal length {:.1} px, true {:.1} px; {:?}",
        adjusted.intrinsics.fx,
        set.intrinsics.fx,
        adjusted.report
    );
    assert_eq!(adjusted.intrinsics.source, FocalSource::Refined);
    let (camera, points) = errors(&set, &tracks, &adjusted.model);
    assert!(
        camera < 0.01 && points < 0.01,
        "cameras {camera:.4}, points {points:.4}"
    );
}

#[test]
fn rejects_a_fit_that_stays_poor() {
    let set = SyntheticSet::orbit(Scene::corner(), 4, 36.0, SIZE.0, SIZE.1);
    let graph = PairGraph::from_fn(set.views(), |pair| set.pair(pair).with_noise(1.0));
    let (tracks, model) = register(&graph, &set.intrinsics, SIZE);

    let result = ba::adjust(
        &model,
        &tracks,
        &set.intrinsics,
        graph.precision_px() as f64,
        &ba::Params {
            max_median_error: 0.05,
            ..ba::Params::default()
        },
    );
    assert!(
        matches!(result, Err(Error::BundleAdjustment { .. })),
        "{result:?}"
    );
}
