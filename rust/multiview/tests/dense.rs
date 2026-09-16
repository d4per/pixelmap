//! Dense depth and fusion against synthetic ground truth.

use std::sync::Arc;

use nalgebra::{Point3, Vector3};
use pixelmap::{Quality, DEFAULT_SEED};
use pixelmap_multiview::depth::{self, DepthMap};
use pixelmap_multiview::rng::Rng;
use pixelmap_multiview::synthetic::{Scene, Surface, SyntheticSet};
use pixelmap_multiview::{
    align, ba, fusion, pairs, sfm, tracks, twoview, Flow, PairGraph, PhotoPx, Pose, ViewId,
};

const SIZE: (usize, usize) = (640, 480);

fn true_cameras(set: &SyntheticSet) -> Vec<Option<Pose>> {
    set.poses.iter().copied().map(Some).collect()
}

/// The fraction of samples with a depth, and the median and 95th-percentile relative
/// depth error of those. A depth where the truth has no surface counts as infinitely
/// wrong.
fn depth_errors(set: &SyntheticSet, maps: &[DepthMap], scale: f64) -> (f64, f64, f64) {
    let mut errors = Vec::new();
    let mut total = 0;
    for map in maps {
        total += map.depth.len();
        for row in 0..map.rows {
            for column in 0..map.columns {
                let Some(d) = map.get(column, row) else {
                    continue;
                };
                let error = set
                    .depth(map.view, map.pixel(column, row))
                    .map_or(f64::INFINITY, |truth| {
                        (d as f64 * scale - truth).abs() / truth
                    });
                errors.push(error);
            }
        }
    }
    let valid = errors.len() as f64 / total as f64;
    errors.sort_by(f64::total_cmp);
    let at = |q: f64| errors[((errors.len() - 1) as f64 * q) as usize];
    (valid, at(0.5), at(0.95))
}

fn percentile(values: &mut [f64], q: f64) -> f64 {
    values.sort_by(f64::total_cmp);
    values[((values.len() - 1) as f64 * q) as usize]
}

#[test]
fn depth_maps_match_the_true_depth() {
    for scene in [Scene::corner(), Scene::sphere()] {
        let set = SyntheticSet::orbit(scene, 4, 36.0, SIZE.0, SIZE.1);
        let graph = PairGraph::from_fn(set.views(), |pair| set.pair(pair));
        let maps = depth::estimate(
            &graph,
            &true_cameras(&set),
            &set.intrinsics,
            SIZE,
            &depth::Params::default(),
        );
        assert_eq!(maps.len(), 4);

        let (valid, median, p95) = depth_errors(&set, &maps, 1.0);
        assert!(valid > 0.5, "only {:.0}% valid", valid * 100.0);
        assert!(
            median < 1e-3 && p95 < 0.01,
            "median {median}, 95th percentile {p95}"
        );
        depth::require_coverage(&maps).expect("enough depth");
    }
}

#[test]
fn filters_keep_outliers_out_of_the_depth_maps() {
    let set = SyntheticSet::orbit(Scene::corner(), 4, 36.0, SIZE.0, SIZE.1);
    let graph = PairGraph::from_fn(set.views(), |pair| {
        set.pair(pair)
            .with_noise(0.5)
            .with_outliers(0.1)
            .with_seed(u64::from(pair.a().0) * 16 + u64::from(pair.b().0))
    });
    let maps = depth::estimate(
        &graph,
        &true_cameras(&set),
        &set.intrinsics,
        SIZE,
        &depth::Params::default(),
    );

    let (valid, _, _) = depth_errors(&set, &maps, 1.0);
    let wrong = maps
        .iter()
        .flat_map(|map| {
            (0..map.rows)
                .flat_map(move |row| (0..map.columns).map(move |column| (map, column, row)))
        })
        .filter_map(|(map, column, row)| {
            let d = map.get(column, row)? as f64;
            let truth = set.depth(map.view, map.pixel(column, row));
            Some(truth.is_none_or(|t| (d - t).abs() / t > 0.05))
        })
        .filter(|&w| w)
        .count();
    let kept: usize = maps.iter().map(DepthMap::valid).sum();
    assert!(valid > 0.3, "only {:.0}% valid", valid * 100.0);
    assert!(
        (wrong as f64) < 0.005 * kept as f64,
        "{wrong} of {kept} depths are more than 5% off"
    );
}

#[test]
fn fuses_exact_depth_into_the_true_surface() {
    for (name, scene) in [("corner", Scene::corner()), ("sphere", Scene::sphere())] {
        let set = SyntheticSet::orbit(scene, 4, 36.0, SIZE.0, SIZE.1);
        let graph = PairGraph::from_fn(set.views(), |pair| set.pair(pair));
        let cameras = true_cameras(&set);
        let maps = depth::estimate(
            &graph,
            &cameras,
            &set.intrinsics,
            SIZE,
            &depth::Params::default(),
        );
        let fused = fusion::fuse(&maps, &cameras, &set.intrinsics, &fusion::Params::default())
            .expect("a surface");

        let mesh = &fused.mesh;
        assert!(
            mesh.triangles.len() > 1000,
            "{name}: {} triangles",
            mesh.triangles.len()
        );
        // The corner is one connected surface. The sphere stands in front of a separate
        // backdrop, so two large pieces are the right answer there.
        let required = if name == "corner" { 0.9 } else { 0.5 };
        assert!(
            fused.largest_component > required
                && fused.dropped_triangles * 20 < mesh.triangles.len(),
            "{name}: largest piece {:.0}%, {} triangles dropped of {}",
            fused.largest_component * 100.0,
            fused.dropped_triangles,
            mesh.triangles.len()
        );
        let mut distances: Vec<f64> = mesh
            .positions
            .iter()
            .map(|p| set.scene.distance(p))
            .collect();
        let median = percentile(&mut distances, 0.5);
        let p95 = percentile(&mut distances, 0.95);
        assert!(
            median < 0.5 * fused.voxel_size && p95 < 2.0 * fused.voxel_size,
            "{name}: median {median}, 95th percentile {p95}, voxel {}",
            fused.voxel_size
        );

        // Normals face the cameras.
        let centre = set.poses[0].centre();
        let facing = mesh
            .positions
            .iter()
            .zip(&mesh.normals)
            .filter(|(p, n)| n.dot(&(centre - *p)) > 0.0)
            .count();
        assert!(
            facing as f64 > 0.95 * mesh.positions.len() as f64,
            "{name}: {facing} facing"
        );
    }
}

#[test]
fn reconstructs_matched_renders_close_to_the_truth() {
    let set = SyntheticSet::orbit(Scene::corner(), 4, 30.0, 800, 600);
    let size = (set.width, set.height);
    let photos: Vec<_> = (0..set.views())
        .map(|v| Arc::new(set.render(ViewId(v as u32))))
        .collect();
    let graph = pairs::compute(&photos, Quality::Low, DEFAULT_SEED, &mut |_| {
        Flow::Continue(())
    })
    .expect("rendered photos map");
    let precision = graph.precision_px() as f64;

    let relatives: Vec<_> = graph
        .pairs()
        .enumerate()
        .filter_map(|(i, (pair, lookup))| {
            twoview::estimate(
                pair,
                lookup,
                &set.intrinsics,
                size,
                &twoview::Params::default(),
                &mut Rng::new(DEFAULT_SEED).derive(i as u64),
            )
            .ok()
        })
        .collect();
    let tracks = tracks::build(&graph, size, &tracks::Params::default());
    let model = sfm::reconstruct(
        &relatives,
        &tracks,
        &set.intrinsics,
        set.views(),
        precision,
        &sfm::Params::default(),
        &mut Rng::new(DEFAULT_SEED).derive(u64::MAX),
    )
    .expect("the views register");
    let adjusted = ba::adjust(
        &model,
        &tracks,
        &set.intrinsics,
        precision,
        &ba::Params::default(),
    )
    .expect("bundle adjustment succeeds");
    let model = adjusted.model;

    let maps = depth::estimate(
        &graph,
        &model.cameras,
        &set.intrinsics,
        size,
        &depth::Params::default(),
    );
    depth::require_coverage(&maps).expect("enough depth");
    let fused = fusion::fuse(
        &maps,
        &model.cameras,
        &set.intrinsics,
        &fusion::Params::default(),
    )
    .expect("a surface");

    // Align the reconstruction with the truth on cameras and sparse points, then measure
    // the mesh against the true surfaces.
    let mut estimated = Vec::new();
    let mut expected = Vec::new();
    for view in model.registered() {
        estimated.push(model.cameras[view.index()].unwrap().centre());
        expected.push(set.poses[view.index()].centre());
    }
    for point in &model.points {
        let track = &tracks[point.track];
        if let Some(surface) =
            set.surface_point(track.anchor, track.observation(track.anchor).unwrap())
        {
            estimated.push(point.position.0);
            expected.push(surface.0);
        }
    }
    let similarity = align::umeyama(&estimated, &expected).expect("alignable");
    let extent = expected[..set.views()]
        .iter()
        .flat_map(|a| expected[..set.views()].iter().map(move |b| (a - b).norm()))
        .fold(0.0, f64::max);

    let (valid, depth_median, depth_p95) = depth_errors(&set, &maps, similarity.scale);
    let mut distances: Vec<f64> = fused
        .mesh
        .positions
        .iter()
        .map(|p| set.scene.distance(&similarity.apply(p)) / extent)
        .collect();
    let median = percentile(&mut distances, 0.5);
    let p90 = percentile(&mut distances, 0.9);
    assert!(
        median < 0.01 && p90 < 0.03,
        "mesh off the truth by a median {:.2}% and 90th percentile {:.2}% of the camera spread; \
         depth {:.0}% valid, median error {:.2}%, 95th percentile {:.2}%",
        median * 100.0,
        p90 * 100.0,
        valid * 100.0,
        depth_median * 100.0,
        depth_p95 * 100.0
    );
    assert!(
        fused.largest_component > 0.5,
        "largest piece {:.0}%",
        fused.largest_component * 100.0
    );
}

/// How many views truly see the surface at pixel `p` of `view`, itself included.
fn true_views(set: &SyntheticSet, view: ViewId, p: PhotoPx) -> usize {
    let Some(point) = set.surface_point(view, p) else {
        return 0;
    };
    (0..set.views() as u32)
        .map(ViewId)
        .filter(|&v| set.project(v, &point).is_some() && set.visible(v, &point))
        .count()
}

#[test]
fn two_view_areas_get_depth() {
    // Adjacent views 12° apart with 2.5 px of matching noise: a depth from a single pair is
    // good to about 2%, a depth from both neighbours to about 1%.
    let set = SyntheticSet::orbit(Scene::corner(), 3, 24.0, SIZE.0, SIZE.1);
    let graph = PairGraph::from_fn(set.views(), |pair| {
        set.pair(pair)
            .with_noise(2.5)
            .with_seed(u64::from(pair.a().0) * 16 + u64::from(pair.b().0))
    });
    let (maps, stats) = depth::estimate_with_stats(
        &graph,
        &true_cameras(&set),
        &set.intrinsics,
        SIZE,
        &depth::Params::default(),
    );

    // Index 0: seen by exactly two views. Index 1: seen by all three.
    let mut samples = [0usize; 2];
    let mut valid = [0usize; 2];
    let mut errors: [Vec<f64>; 2] = [Vec::new(), Vec::new()];
    for map in &maps {
        for row in 0..map.rows {
            for column in 0..map.columns {
                let p = map.pixel(column, row);
                let seen_by = true_views(&set, map.view, p);
                if seen_by < 2 {
                    continue;
                }
                let group = seen_by.min(3) - 2;
                samples[group] += 1;
                if let Some(d) = map.get(column, row) {
                    valid[group] += 1;
                    if let Some(truth) = set.depth(map.view, p) {
                        errors[group].push((d as f64 - truth).abs() / truth);
                    }
                }
            }
        }
    }
    assert!(
        samples[0] > 1000,
        "the scene has areas seen by two views: {samples:?}"
    );
    let fraction = |group: usize| valid[group] as f64 / samples[group] as f64;
    assert!(
        fraction(0) >= 0.8 * fraction(1),
        "{:.0}% of the areas seen by two views have a depth, against {:.0}% of those seen by three; {:#?}",
        fraction(0) * 100.0,
        fraction(1) * 100.0,
        stats
    );
    let median = percentile(&mut errors[0], 0.5);
    assert!(
        median < 0.03,
        "median depth error {:.2}% where two views see the surface",
        median * 100.0
    );
}

#[test]
fn fuses_surfaces_seen_at_a_grazing_angle() {
    // A long floor, seen by cameras 0.6 units above it, looking along it: most of the floor
    // meets the rays at well under 20°.
    let floor = Scene::new(
        vec![Surface::Rectangle {
            origin: Point3::new(-6.0, 0.0, -3.0),
            u: Vector3::new(12.0, 0.0, 0.0),
            v: Vector3::new(0.0, 0.0, 16.0),
        }],
        Point3::new(0.0, 0.0, 3.0),
        6.0,
    );
    let poses = [-0.8, 0.0, 0.8]
        .map(|x| {
            Pose::look_at(
                &Point3::new(x, 0.6, -4.0),
                &Point3::new(0.0, 0.0, 3.0),
                &Vector3::y(),
            )
        })
        .to_vec();
    let set = SyntheticSet::with_poses(floor, poses, SIZE.0, SIZE.1);
    let graph = PairGraph::from_fn(set.views(), |pair| set.pair(pair));
    let cameras = true_cameras(&set);
    let maps = depth::estimate(
        &graph,
        &cameras,
        &set.intrinsics,
        SIZE,
        &depth::Params::default(),
    );
    let fused = fusion::fuse(&maps, &cameras, &set.intrinsics, &fusion::Params::default())
        .expect("a surface");

    // Index mesh vertices by cells two voxels wide, then ask of each depth sample whether a
    // vertex lies within two voxels of its true surface point.
    let cell = 2.0 * fused.voxel_size;
    let key = |p: &Point3<f64>| p.coords.map(|c| (c / cell).floor() as i64);
    let mut cells = std::collections::HashMap::<_, Vec<Point3<f64>>>::new();
    for p in &fused.mesh.positions {
        cells.entry(key(p)).or_default().push(*p);
    }
    let (mut samples, mut covered) = (0usize, 0usize);
    for map in &maps {
        for row in (0..map.rows).step_by(3) {
            for column in (0..map.columns).step_by(3) {
                if map.get(column, row).is_none() {
                    continue;
                }
                let Some(truth) = set.surface_point(map.view, map.pixel(column, row)) else {
                    continue;
                };
                samples += 1;
                let k = key(&truth.0);
                let near = (-1..=1).any(|dx| {
                    (-1..=1).any(|dy| {
                        (-1..=1).any(|dz| {
                            cells
                                .get(&(k + Vector3::new(dx, dy, dz)))
                                .is_some_and(|ps| ps.iter().any(|p| (p - truth.0).norm() <= cell))
                        })
                    })
                });
                covered += near as usize;
            }
        }
    }
    assert!(samples > 1000, "only {samples} floor samples");
    let fraction = covered as f64 / samples as f64;
    assert!(
        fraction > 0.9,
        "only {:.0}% of the floor's depth made it into the mesh",
        fraction * 100.0
    );

    let mut distances: Vec<f64> = fused
        .mesh
        .positions
        .iter()
        .map(|p| set.scene.distance(p))
        .collect();
    let median = percentile(&mut distances, 0.5);
    assert!(
        median < 0.5 * fused.voxel_size,
        "median distance {median}, voxel {}",
        fused.voxel_size
    );
}
