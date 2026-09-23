//! The whole pipeline on exact synthetic correspondences and rendered photos, checked on
//! the colour it puts on the mesh.

use std::sync::Arc;

use pixelmap::Photo;
use pixelmap_multiview::synthetic::{Scene, SyntheticSet};
use pixelmap_multiview::{align, pipeline, Flow, Model, Options, PairGraph, ViewId, World};

fn reconstruct(set: &SyntheticSet) -> (Vec<Arc<Photo>>, Model) {
    let photos: Vec<_> = (0..set.views())
        .map(|v| Arc::new(set.render(ViewId(v as u32))))
        .collect();
    let graph = PairGraph::from_fn(set.views(), |pair| set.pair(pair));
    let reconstruction = pipeline::reconstruct(
        &graph,
        &photos,
        &set.intrinsics,
        &Options::new(),
        &mut |_| Flow::Continue(()),
    )
    .expect("the synthetic scene reconstructs");
    (photos, reconstruction)
}

/// The similarity from the reconstruction's frame to the truth's.
fn alignment(set: &SyntheticSet, r: &Model) -> align::Similarity {
    let mut estimated = Vec::new();
    let mut expected = Vec::new();
    let adjusted = &r.diagnostics().adjusted;
    for view in adjusted.registered() {
        estimated.push(adjusted.cameras[view.index()].unwrap().centre());
        expected.push(set.poses[view.index()].centre());
    }
    for point in &adjusted.points {
        let track = &r.diagnostics().tracks[point.track];
        if let Some(surface) =
            set.surface_point(track.anchor, track.observation(track.anchor).unwrap())
        {
            estimated.push(point.position.0);
            expected.push(surface.0);
        }
    }
    align::umeyama(&estimated, &expected).expect("alignable")
}

fn difference(a: [u8; 3], b: [u8; 3]) -> f64 {
    a.iter()
        .zip(&b)
        .map(|(x, y)| (*x as f64 - *y as f64).abs())
        .sum::<f64>()
        / 3.0
}

fn percentile(values: &mut [f64], q: f64) -> f64 {
    values.sort_by(f64::total_cmp);
    values[((values.len() - 1) as f64 * q) as usize]
}

#[test]
fn colours_the_mesh_from_the_photos() {
    let set = SyntheticSet::orbit(Scene::corner(), 4, 36.0, 640, 480);
    let (_, r) = reconstruct(&set);
    let similarity = alignment(&set, &r);

    let mesh = &r.mesh;
    assert_eq!(r.texture.vertex_colours.len(), mesh.positions.len());
    let mut errors: Vec<f64> = mesh
        .positions
        .iter()
        .zip(&r.texture.vertex_colours)
        .map(|(p, &colour)| {
            let [red, green, blue, _] = set.scene.colour(&similarity.apply(p));
            difference(colour, [red, green, blue])
        })
        .collect();
    let median = percentile(&mut errors, 0.5);
    let p90 = percentile(&mut errors, 0.9);
    assert!(
        median < 12.0 && p90 < 30.0,
        "colour off by a median {median:.1} and 90th percentile {p90:.1} levels"
    );
}

#[test]
fn the_atlas_holds_the_chosen_photo_under_each_triangle() {
    let set = SyntheticSet::orbit(Scene::sphere(), 4, 36.0, 640, 480);
    let (photos, r) = reconstruct(&set);
    let texture = &r.texture;
    let mesh = &r.mesh;
    let atlas = &texture.atlas;
    assert!(texture.charts > 0);
    assert!(atlas.width().is_power_of_two() && atlas.height().is_power_of_two());
    assert_eq!(texture.face_texcoords.len(), mesh.triangles.len());

    let nearest = |photo: &Photo, x: f64, y: f64| -> Option<[u8; 3]> {
        let [r, g, b, _] = photo.pixel(x.round().max(0.0) as usize, y.round().max(0.0) as usize)?;
        Some([r, g, b])
    };
    let mut errors = Vec::new();
    for (f, triangle) in mesh.triangles.iter().enumerate().step_by(7) {
        let view = texture.face_views[f];
        let pose = r.diagnostics().adjusted.cameras[view.index()].unwrap();
        for (corner, &v) in triangle.iter().enumerate() {
            let [u, t] = texture.texcoords[texture.face_texcoords[f][corner] as usize];
            let ax = u as f64 * atlas.width() as f64 - 0.5;
            let ay = (1.0 - t as f64) * atlas.height() as f64 - 0.5;
            let Some(n) = pose.project(&World(mesh.positions[v as usize])) else {
                continue;
            };
            let p = r.intrinsics.denormalize(n);
            if let (Some(a), Some(b)) = (
                nearest(atlas, ax, ay),
                nearest(&photos[view.index()], p.x() as f64, p.y() as f64),
            ) {
                errors.push(difference(a, b));
            }
        }
    }
    assert!(errors.len() > 100);
    let median = percentile(&mut errors, 0.5);
    assert!(
        median < 6.0,
        "atlas and photo differ by a median {median:.1} levels"
    );
}
