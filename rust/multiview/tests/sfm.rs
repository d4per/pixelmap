//! Tracks and registration against exact synthetic correspondences: every camera and
//! every point has a known true position, so the reconstruction can be checked directly
//! once aligned for position and scale.

use nalgebra::Point3;
use pixelmap_multiview::calib::{FocalSource, Intrinsics};
use pixelmap_multiview::rng::Rng;
use pixelmap_multiview::sfm::{self, SparseModel};
use pixelmap_multiview::synthetic::{Scene, SyntheticPair, SyntheticSet};
use pixelmap_multiview::tracks::{self, Track};
use pixelmap_multiview::twoview::{self, RelativePose};
use pixelmap_multiview::{align, ba, PairGraph};

const WIDTH: usize = 640;
const HEIGHT: usize = 480;
const SIZE: (usize, usize) = (WIDTH, HEIGHT);

fn graph(set: &SyntheticSet, noise: f32, outliers: f32) -> PairGraph<SyntheticPair<'_>> {
    PairGraph::from_fn(set.views(), |pair| {
        set.pair(pair)
            .with_noise(noise)
            .with_outliers(outliers)
            .with_seed(u64::from(pair.a().0) * 16 + u64::from(pair.b().0))
    })
}

fn relatives(set: &SyntheticSet, graph: &PairGraph<SyntheticPair<'_>>) -> Vec<RelativePose> {
    graph
        .pairs()
        .enumerate()
        .filter_map(|(i, (pair, lookup))| {
            twoview::estimate(
                pair,
                lookup,
                &set.intrinsics,
                SIZE,
                &twoview::Params::default(),
                &mut Rng::new(i as u64),
            )
            .ok()
        })
        .collect()
}

fn reconstruct(
    set: &SyntheticSet,
    graph: &PairGraph<SyntheticPair<'_>>,
    tracks: &[Track],
) -> SparseModel {
    sfm::reconstruct(
        &relatives(set, graph),
        tracks,
        &set.intrinsics,
        set.views(),
        graph.precision_px() as f64,
        &sfm::Params::default(),
        &mut Rng::new(3),
    )
    .expect("the synthetic views register")
}

/// The fraction of non-anchor observations more than `tolerance_px` from where the point
/// truly appears.
fn misplaced_fraction(set: &SyntheticSet, tracks: &[Track], tolerance_px: f32) -> f64 {
    let (mut bad, mut total) = (0usize, 0usize);
    for track in tracks {
        let anchor = track.observation(track.anchor).unwrap();
        for &(view, p) in &track.observations {
            if view == track.anchor {
                continue;
            }
            total += 1;
            match set.transfer(track.anchor, view, anchor) {
                Some(truth) if (truth.0 - p.0).norm() <= tolerance_px => {}
                _ => bad += 1,
            }
        }
    }
    bad as f64 / total.max(1) as f64
}

/// Aligns the model to the truth by its camera centres, then checks cameras and points
/// against `tolerance`, a fraction of the spread of the true cameras.
fn assert_matches_truth(set: &SyntheticSet, tracks: &[Track], model: &SparseModel, tolerance: f64) {
    assert_eq!(
        model.registered().len(),
        set.views(),
        "{:?}",
        model.warnings
    );

    let estimated: Vec<Point3<f64>> = model.cameras.iter().map(|c| c.unwrap().centre()).collect();
    let expected: Vec<Point3<f64>> = set.poses.iter().map(|p| p.centre()).collect();
    let similarity = align::umeyama(&estimated, &expected).expect("cameras are not collinear");
    let extent = expected
        .iter()
        .flat_map(|a| expected.iter().map(move |b| (a - b).norm()))
        .fold(0.0, f64::max);

    for (view, (e, x)) in estimated.iter().zip(&expected).enumerate() {
        let error = (similarity.apply(e) - x).norm() / extent;
        assert!(
            error < tolerance,
            "view {view} is off by {:.3}% of the spread",
            error * 100.0
        );
    }

    assert!(
        model.points.len() > 500,
        "only {} points",
        model.points.len()
    );
    let mut errors: Vec<f64> = model
        .points
        .iter()
        .map(|p| {
            let track = &tracks[p.track];
            let surface = set
                .surface_point(track.anchor, track.observation(track.anchor).unwrap())
                .expect("tracks start on a surface");
            (similarity.apply(&p.position.0) - surface.0).norm() / extent
        })
        .collect();
    errors.sort_by(f64::total_cmp);
    let median = errors[errors.len() / 2];
    assert!(
        median < tolerance,
        "median point error {:.3}% of the spread",
        median * 100.0
    );
}

#[test]
fn registers_every_view_from_exact_matches() {
    for (name, scene) in [("sphere", Scene::sphere()), ("corner", Scene::corner())] {
        let set = SyntheticSet::orbit(scene, 4, 36.0, WIDTH, HEIGHT);
        let graph = graph(&set, 0.0, 0.0);
        let tracks = tracks::build(&graph, SIZE, &tracks::Params::default());
        let multi = tracks::require_enough(&tracks).expect("enough tracks");
        assert!(misplaced_fraction(&set, &tracks, 1.0) < 1e-3, "{name}");

        let model = reconstruct(&set, &graph, &tracks);
        assert!(model.warnings.is_empty(), "{name}: {:?}", model.warnings);
        assert_matches_truth(&set, &tracks, &model, 0.01);
        assert!(multi > 500, "{name}: {multi} tracks in 3+ views");
    }
}

#[test]
fn consistency_filter_rejects_injected_outliers() {
    let set = SyntheticSet::orbit(Scene::sphere(), 4, 36.0, WIDTH, HEIGHT);
    let graph = graph(&set, 0.5, 0.1);

    let unfiltered = tracks::build(
        &graph,
        SIZE,
        &tracks::Params {
            max_residual: f32::INFINITY,
            ..tracks::Params::default()
        },
    );
    let filtered = tracks::build(&graph, SIZE, &tracks::Params::default());

    let before = misplaced_fraction(&set, &unfiltered, 4.0);
    let after = misplaced_fraction(&set, &filtered, 4.0);
    assert!(
        before > 0.05,
        "only {:.2}% misplaced without the filter",
        before * 100.0
    );
    assert!(
        after < 0.002,
        "{:.3}% misplaced with the filter",
        after * 100.0
    );
    tracks::require_enough(&filtered).expect("enough tracks survive");
}

#[test]
fn registers_every_view_despite_noise_and_outliers() {
    let set = SyntheticSet::orbit(Scene::corner(), 4, 36.0, WIDTH, HEIGHT);
    let graph = graph(&set, 0.5, 0.1);
    let tracks = tracks::build(&graph, SIZE, &tracks::Params::default());
    let model = reconstruct(&set, &graph, &tracks);
    assert_matches_truth(&set, &tracks, &model, 0.02);
}

#[test]
fn a_wrong_estimated_focal_length_does_not_leave_views_out() {
    let views = 10;
    let set = SyntheticSet::orbit(Scene::corner(), views, 160.0, WIDTH, HEIGHT);
    let graph = graph(&set, 0.7, 0.1);
    let tracks = tracks::build(&graph, SIZE, &tracks::Params::default());
    // 25% off, as a focal length estimated without EXIF can be. Registering against it
    // skews the points, and views placed later disagree with them.
    let intrinsics =
        Intrinsics::from_focal(WIDTH as f64 * 1.25, WIDTH, HEIGHT, FocalSource::Estimated);
    let register = |adjust: ba::Params| {
        sfm::reconstruct(
            &relatives(&set, &graph),
            &tracks,
            &intrinsics,
            views,
            graph.precision_px() as f64,
            &sfm::Params {
                adjust,
                ..sfm::Params::default()
            },
            &mut Rng::new(3),
        )
        .expect("the synthetic views register")
    };

    let unadjusted = register(ba::Params {
        max_iterations: 0,
        ..sfm::Params::default().adjust
    });
    let adjusted = register(ba::Params {
        refine_focal: true,
        ..sfm::Params::default().adjust
    });
    assert!(
        adjusted.registered().len() > unadjusted.registered().len(),
        "{:?} placed with adjustment, {:?} without",
        adjusted.registered(),
        unadjusted.registered()
    );
    assert!(
        (adjusted.intrinsics.fx - WIDTH as f64).abs() < 0.02 * WIDTH as f64,
        "focal length refined to {}",
        adjusted.intrinsics.fx
    );
    assert!(adjusted.registrations.iter().all(|r| r.inlier_ratio > 0.6));
}
