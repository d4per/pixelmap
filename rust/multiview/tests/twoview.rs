//! Two-view geometry against exact synthetic correspondences: the matcher is taken out of
//! the picture, so any error here is the estimator's.

use nalgebra::{Rotation3, Vector3};
use pixelmap_multiview::rng::Rng;
use pixelmap_multiview::synthetic::{Scene, SyntheticSet};
use pixelmap_multiview::twoview::{self, Degeneracy, Params, RelativePose, Verdict};
use pixelmap_multiview::{PairId, PairLookup};

const WIDTH: usize = 640;
const HEIGHT: usize = 480;

fn estimate(set: &SyntheticSet, pair: PairId, lookup: &impl PairLookup) -> RelativePose {
    twoview::estimate(
        pair,
        lookup,
        &set.intrinsics,
        (set.width, set.height),
        &Params::default(),
        &mut Rng::new(1),
    )
    .expect("a pose is estimated")
}

fn rotation_error_deg(a: &Rotation3<f64>, b: &Rotation3<f64>) -> f64 {
    let cos = ((a * b.inverse()).matrix().trace() - 1.0) / 2.0;
    // `Rotation3::angle` does not clamp, and a near-perfect estimate lands just above 1.
    cos.clamp(-1.0, 1.0).acos().to_degrees()
}

fn direction_error_deg(a: &Vector3<f64>, b: &Vector3<f64>) -> f64 {
    a.angle(b).to_degrees()
}

#[test]
fn recovers_relative_pose_from_exact_matches() {
    for (name, scene) in [("sphere", Scene::sphere()), ("corner", Scene::corner())] {
        let set = SyntheticSet::orbit(scene, 3, 30.0, WIDTH, HEIGHT);
        for pair in PairId::all(3) {
            let estimate = estimate(&set, pair, &set.pair(pair));
            let truth = set.relative_pose(pair);

            let r = rotation_error_deg(&estimate.pose.rotation, &truth.rotation);
            let t = direction_error_deg(&estimate.pose.translation, &truth.translation);
            assert!(r < 0.1, "{name} {pair}: rotation off by {r}°");
            assert!(t < 1.0, "{name} {pair}: translation off by {t}°");
            assert_eq!(estimate.verdict, Verdict::Usable, "{name} {pair}");
            assert!(estimate.inlier_ratio > 0.95, "{name} {pair}: {estimate:?}");
            assert!((estimate.pose.translation.norm() - 1.0).abs() < 1e-9);
        }
    }
}

#[test]
fn survives_noise_and_outliers() {
    let set = SyntheticSet::orbit(Scene::sphere(), 3, 30.0, WIDTH, HEIGHT);
    let pair = PairId::all(3).next().unwrap();
    let lookup = set
        .pair(pair)
        .with_noise(0.5)
        .with_outliers(0.3)
        .with_seed(7);
    let estimate = estimate(&set, pair, &lookup);
    let truth = set.relative_pose(pair);

    let r = rotation_error_deg(&estimate.pose.rotation, &truth.rotation);
    let t = direction_error_deg(&estimate.pose.translation, &truth.translation);
    assert!(r < 0.5, "rotation off by {r}°");
    assert!(t < 3.0, "translation off by {t}°");
    assert!(
        (0.6..0.75).contains(&estimate.inlier_ratio),
        "inlier ratio {}",
        estimate.inlier_ratio
    );
    assert_eq!(estimate.verdict, Verdict::Usable);
}

#[test]
fn flags_a_camera_that_only_rotated() {
    let set = SyntheticSet::rotation_only(Scene::corner(), 3, 20.0, WIDTH, HEIGHT);
    let pair = PairId::all(3).next().unwrap();
    let estimate = estimate(&set, pair, &set.pair(pair));
    assert!(
        matches!(
            estimate.verdict,
            Verdict::Degenerate(Degeneracy::NoParallax { .. })
        ),
        "{:?}",
        estimate.verdict
    );
}

#[test]
fn flags_a_baseline_too_small_to_measure_depth() {
    let set = SyntheticSet::orbit(Scene::sphere(), 3, 1.0, WIDTH, HEIGHT);
    let pair = PairId::all(3).next().unwrap();
    let estimate = estimate(&set, pair, &set.pair(pair));
    assert!(
        matches!(
            estimate.verdict,
            Verdict::Degenerate(
                Degeneracy::NoParallax { .. }
                    | Degeneracy::SmallBaseline { .. }
                    | Degeneracy::Planar { .. }
            )
        ),
        "{:?}",
        estimate.verdict
    );
}

#[test]
fn is_deterministic() {
    let set = SyntheticSet::orbit(Scene::corner(), 2, 20.0, WIDTH, HEIGHT);
    let pair = PairId::all(2).next().unwrap();
    let lookup = set.pair(pair).with_noise(0.5).with_outliers(0.2);
    let first = estimate(&set, pair, &lookup);
    let second = estimate(&set, pair, &lookup);
    assert_eq!(first.pose, second.pose);
    assert_eq!(first.inliers, second.inliers);
}
