//! Rendered synthetic photos through the real matcher: checks that pixelmap's
//! correspondences reach the geometry stages in the coordinates they expect.

use std::sync::Arc;

use pixelmap::{Quality, DEFAULT_SEED};
use pixelmap_multiview::pairs::{self, MIN_PAIR_COVERAGE};
use pixelmap_multiview::rng::Rng;
use pixelmap_multiview::synthetic::{Scene, SyntheticSet};
use pixelmap_multiview::twoview::{self, Params};
 use pixelmap_multiview::{Error, Event, Flow, PairLookup, PhotoPx, Stage, ViewId};

fn rendered(set: &SyntheticSet) -> Vec<Arc<pixelmap::Photo>> {
    (0..set.views())
        .map(|v| Arc::new(set.render(ViewId(v as u32))))
        .collect()
}

#[test]
fn maps_rendered_views_and_recovers_their_motion() {
    let set = SyntheticSet::orbit(Scene::sphere(), 3, 24.0, 400, 300);
    let photos = rendered(&set);

    let mut events = 0;
    let graph = pairs::compute(&photos, Quality::Low, DEFAULT_SEED, &mut |event| {
        assert_eq!(event.stage(), Stage::Pairs);
        events += 1;
        Flow::Continue(())
    })
    .expect("rendered photos are valid input");
    assert!(events > 3);
    assert_eq!(
        pairs::require_connected(&graph, MIN_PAIR_COVERAGE).unwrap(),
        [ViewId(0), ViewId(1), ViewId(2)]
    );

    for (pair, mapping) in graph.pairs() {
        // pixelmap against the exact answer, at a sample of points.
        let truth = set.pair(pair);
        let (mut compared, mut close) = (0, 0);
        for y in (10..290).step_by(20) {
            for x in (10..390).step_by(20) {
                let p = pixelmap_multiview::PhotoPx::new(x as f32, y as f32);
                if let (Some(found), Some(exact)) = (mapping.a_to_b(p), truth.a_to_b(p)) {
                    compared += 1;
                    let error = (found.0 - exact.0).norm();
                    if error < 3.0 {
                        close += 1;
                    }
                }
            }
        }
        assert!(compared > 100, "{pair}: only {compared} points mapped");
        assert!(
            close as f32 > 0.8 * compared as f32,
            "{pair}: only {close} of {compared} lookups within 3 px of the truth"
        );

        let estimate = twoview::estimate(
            pair,
            mapping,
            &set.intrinsics,
            (set.width, set.height),
            &Params::default(),
            &mut Rng::new(DEFAULT_SEED),
        )
        .expect("a pose is estimated");
        let truth = set.relative_pose(pair);
        let relative = estimate.pose.rotation * truth.rotation.inverse();
        let r = ((relative.matrix().trace() - 1.0) / 2.0)
            .clamp(-1.0, 1.0)
            .acos()
            .to_degrees();
        let t = estimate
            .pose
            .translation
            .angle(&truth.translation)
            .to_degrees();
        assert!(r < 1.0, "{pair}: rotation off by {r}°; {estimate:?}");
        assert!(t < 5.0, "{pair}: translation off by {t}°; {estimate:?}");
    }
}

#[test]
fn hands_back_the_dense_map_of_every_pair() {
    let set = SyntheticSet::orbit(Scene::sphere(), 3, 24.0, 400, 300);
    let photos = rendered(&set);

    let mut maps = Vec::new();
    let graph = pairs::compute(&photos, Quality::Low, DEFAULT_SEED, &mut |event| {
        if let Event::PairMapped { pair, map, .. } = event {
            assert_eq!(map.pair, pair, "a map belongs to the pair it finished");
            maps.push(*map);
        }
        Flow::Continue(())
    })
    .expect("rendered photos are valid input");

    // One map per pair, in the order the pairs were mapped.
    let pairs: Vec<_> = graph.pairs().map(|(pair, _)| pair).collect();
    assert_eq!(maps.len(), pairs.len());
    assert_eq!(maps.iter().map(|m| m.pair).collect::<Vec<_>>(), pairs);

    for (map, (pair, mapping)) in maps.iter().zip(graph.pairs()) {
        let (columns, rows) = mapping.grid_dimensions();
        assert_eq!((map.columns, map.rows), (columns, rows));
        assert_eq!(map.cell_size, mapping.native_stride());
        assert_eq!(map.points.len(), columns * rows * 2);
        assert!(map.mapped() > 0, "{pair}: nothing mapped");
        assert!(
            map.coverage > 0.5,
            "{pair}: only {:.0}% covered",
            map.coverage * 100.0
        );

        // The grid must be in the coordinates of the photos handed in, not the lower
        // resolution the solver works at. Comparing every cell against the lookup at the
        // same place is what catches the working-scale conversion being wrong.
        let (mut compared, mut close) = (0, 0);
        for row in 0..rows {
            for column in 0..columns {
                let (Some(found), Some(expected)) = (
                    map.point(column, row),
                    mapping.a_to_b(map.pixel(column, row)),
                ) else {
                    continue;
                };
                compared += 1;
                if (found.0 - expected.0).norm() < 1.0 {
                    close += 1;
                }
            }
        }
        assert!(compared > 100, "{pair}: only {compared} cells to compare");
        assert!(
            close as f32 > 0.9 * compared as f32,
            "{pair}: only {close} of {compared} cells agree with the lookup"
        );
    }
}

#[test]
fn stops_when_asked() {
    let set = SyntheticSet::orbit(Scene::corner(), 3, 20.0, 200, 150);
    let photos = rendered(&set);
    let result = pairs::compute(
        &photos,
        Quality::Low,
        DEFAULT_SEED,
        &mut |_| Flow::Break(()),
    );
    assert!(matches!(
        result,
        Err(Error::Cancelled {
            stage: Stage::Pairs
        })
    ));
}

#[test]
fn a_mapping_looks_up_exactly_what_its_correspondence_did() {
    // `Mapping` drops the photos a `Correspondence` holds and reimplements its lookup; the
    // two must agree bit for bit, including where nothing is mapped.
    let set = SyntheticSet::orbit(Scene::corner(), 2, 20.0, 400, 300);
    let photos = rendered(&set);
    let correspondence = pixelmap::Correspondence::builder()
        .quality(Quality::Low)
        .seed(DEFAULT_SEED)
        .run(photos[0].clone(), photos[1].clone())
        .expect("rendered photos are valid input");
    let mapping = pairs::Mapping::from(correspondence.clone());

    let as_bits = |p: Option<(f32, f32)>| p.map(|(x, y)| (x.to_bits(), y.to_bits()));
    let (mut mapped, mut unmapped) = (0, 0);
    for y in -8..(300 * 4 + 8) {
        for x in -8..(400 * 4 + 8) {
            // Quarter pixels, reaching a little outside the photo on every side.
            let p = PhotoPx::new(x as f32 * 0.25 + 0.1, y as f32 * 0.25);
            let forward = correspondence.lookup(p.x(), p.y());
            assert_eq!(
                as_bits(mapping.a_to_b(p).map(|q| (q.x(), q.y()))),
                as_bits(forward),
                "a_to_b at {p:?}"
            );
            assert_eq!(
                as_bits(mapping.b_to_a(p).map(|q| (q.x(), q.y()))),
                as_bits(correspondence.lookup_back(p.x(), p.y())),
                "b_to_a at {p:?}"
            );
            if forward.is_some() {
                mapped += 1;
            } else {
                unmapped += 1;
            }
        }
    }
    assert!(mapped > 0 && unmapped > 0, "{mapped} mapped, {unmapped} not");
    assert_eq!(mapping.coverage(), correspondence.coverage());
}
