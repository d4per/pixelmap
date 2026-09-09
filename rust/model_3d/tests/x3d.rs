//! Reconstruction from a synthetic correspondence field, exercised the way a downstream
//! user would: only through the public API of both crates.
//!
//! The mapping is built by hand rather than solved for, so the test stays fast and
//! depends on nothing but the geometry `Model3D` is supposed to recover.

use std::sync::Arc;

use pixelmap::{DensePhotoMap, Photo};
use pixelmap_model_3d::Model3D;

const WIDTH: usize = 200;
const HEIGHT: usize = 150;
const GRID_WIDTH: usize = 21;
const GRID_HEIGHT: usize = 16;

/// A flat photo. Only its dimensions matter here: `Model3D` reads the correspondence
/// grid, and carries the photo along as the texture.
fn photo() -> Arc<Photo> {
    let data = vec![128u8; WIDTH * HEIGHT * 4];
    Arc::new(Photo::from_rgba(WIDTH, HEIGHT, data).expect("buffer matches the dimensions"))
}

/// A correspondence field with real depth in it: a horizontal disparity that varies
/// quadratically across the frame, as a surface bulging towards the camera would produce.
///
/// The variation has to be non-linear. `Model3D` fits the affine part of the
/// correspondence and reads depth out of what is left over, so a pure translation — or
/// any disparity linear in `(x, y)` — is absorbed by that fit and leaves a zero
/// residual, which the reconstruction reports as a flat surface.
fn bulging_surface() -> DensePhotoMap {
    let mut map = DensePhotoMap::new(photo(), photo(), GRID_WIDTH, GRID_HEIGHT);
    let cell = map.grid_cell_size() as f32;

    for y in 0..GRID_HEIGHT {
        for x in 0..GRID_WIDTH {
            let u = x as f32 / (GRID_WIDTH - 1) as f32;
            let depth = 1.0 - 4.0 * (u - 0.5) * (u - 0.5);
            let disparity = 4.0 + 8.0 * depth;
            map.set_grid_coordinates(x, y, x as f32 * cell - disparity, y as f32 * cell);
        }
    }
    map
}

#[test]
fn builds_a_textured_grid_from_a_mapping() {
    let model = Model3D::new(&bulging_surface());

    assert_eq!(model.grid_width, GRID_WIDTH);
    assert_eq!(model.grid_height, GRID_HEIGHT);
    assert_eq!(model.photo.width(), WIDTH);

    let mut valid = 0;
    for y in 0..model.grid_height {
        for x in 0..model.grid_width {
            let p = model.get_texture_point(x, y);
            if p.x.is_nan() {
                continue;
            }
            valid += 1;
            assert!(p.x.is_finite() && p.y.is_finite() && p.z.is_finite());
            assert!((0.0..=1.0).contains(&p.u), "u out of range: {}", p.u);
            assert!((0.0..=1.0).contains(&p.v), "v out of range: {}", p.v);
            assert_eq!((p.grid_x, p.grid_y), (x, y));
        }
    }

    let cells = GRID_WIDTH * GRID_HEIGHT;
    assert!(
        valid * 2 > cells,
        "only {valid} of {cells} cells were reconstructed"
    );
}

#[test]
fn writes_a_textured_x3d_mesh() {
    let x3d = Model3D::new(&bulging_surface()).to_x3d();

    assert!(x3d.starts_with("<X3D"));
    assert!(x3d.ends_with("</X3D>"));
    assert!(x3d.contains("<IndexedFaceSet"));
    // The caller substitutes its own texture URL for this.
    assert!(x3d.contains("[photo_placeholder]"));

    let faces = between(&x3d, "coordIndex='", "'");
    assert!(!faces.trim().is_empty(), "no faces were emitted");
    // Two triangles per quad, each closed by the -1 separator.
    assert_eq!(faces.matches("-1").count() % 2, 0);

    let points: Vec<&str> = between(&x3d, "<Coordinate point='", "'")
        .split_whitespace()
        .collect();
    assert_eq!(points.len(), GRID_WIDTH * GRID_HEIGHT * 3);

    let texture: Vec<&str> = between(&x3d, "<TextureCoordinate point='", "'")
        .split_whitespace()
        .collect();
    assert_eq!(texture.len(), GRID_WIDTH * GRID_HEIGHT * 2);

    for value in points.iter().chain(texture.iter()) {
        assert!(
            value.parse::<f32>().is_ok_and(f32::is_finite),
            "X3D holds a value no viewer can read: {value}"
        );
    }
}

/// The text between the first `start` and the next `end` after it.
fn between<'a>(haystack: &'a str, start: &str, end: &str) -> &'a str {
    let rest = &haystack[haystack.find(start).expect("marker present") + start.len()..];
    &rest[..rest.find(end).expect("closing marker present")]
}
