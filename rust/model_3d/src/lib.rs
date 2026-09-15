//! Turning a correspondence map into 3D geometry.
//!
//! A dense correspondence between two views of a static scene constrains the shape of
//! that scene. [`Model3D::new`] solves for it, yielding a grid of points that each carry
//! the texture coordinates they came from, and [`Model3D::to_x3d`] writes that grid out
//! as an X3D mesh.
//!
//! # How depth is recovered
//!
//! The two photos are treated as what they are: pictures from two perspective cameras.
//!
//! 1. A **fundamental matrix** is fitted to the correspondences with RANSAC and refined
//!    on its inliers, which also discards matches that do not fit any single rigid
//!    camera motion.
//! 2. Assuming a focal length ([`Settings::focal_lengths`]) and a principal point at the
//!    centre of the frame, it becomes an **essential matrix**, which factors into the
//!    rotation and the direction of travel between the two cameras. Of its four
//!    factorizations, the one that puts the scene in front of both cameras wins, so the
//!    depth comes out the right way round for convex and concave scenes alike.
//! 3. Every consistent correspondence is **triangulated**. Wild depths are dropped, a
//!    3x3 median filter removes matching noise, and the result is centred and scaled so
//!    its longer image-plane extent spans `[-1, 1]` — with depth on the same scale, so
//!    the proportions are the triangulated ones.
//!
//! Two views do not always pin that geometry down: too few matches may agree on it, a
//! single homography may explain the map just as well (a flat scene, or a camera that only
//! turned), or the viewpoints may be too close for depth to rise above the noise. With
//! [`Projection::Auto`], the default, such pairs fall back to an **affine** camera model,
//! which reads a relative depth out of whatever an affine warp cannot explain.
//! [`Model3D::projection`] says which model was used, and
//! [`Model3D::two_view_geometry`] reports the perspective estimate either way.
//!
//! The correspondence itself comes from the [`pixelmap`] crate; this one only reads the
//! finished [`DensePhotoMap`], so nothing here is on the algorithm's hot path.
//!
#![doc = include_str!("../doc/model-3d.md")]
#![warn(missing_docs)]

mod affine;
mod perspective;
mod rng;
mod two_view;

pub use two_view::TwoViewGeometry;

use std::sync::Arc;

use pixelmap::{DensePhotoMap, Photo};

use two_view::PointPair;

/// The focal length [`Settings::default`] assumes for both cameras, as a multiple of the
/// larger image dimension: about a 28 mm lens in 35 mm terms, typical of a phone's main
/// camera.
pub const DEFAULT_FOCAL_LENGTH: f64 = 0.8;

/// The diagonal of a 36 x 24 mm frame, which is what a "35 mm equivalent" focal length
/// is relative to.
const FULL_FRAME_DIAGONAL_MM: f64 = 43.266_615_305_567_875;

/// In the affine model, faces with an edge longer than this in model units are torn.
const AFFINE_MAX_EDGE: f32 = 0.5;

/// In the perspective model, an edge is torn when it is this many times longer than the
/// grid spacing of a surface facing the camera at that depth. Six allows surfaces
/// inclined up to about 80° from the viewing direction, and tears at real depth
/// discontinuities such as the silhouette of a near object against a far background.
const PERSPECTIVE_MAX_STRETCH: f32 = 6.0;

/// Which camera model a reconstruction assumes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Projection {
    /// Perspective when the two views determine it well, affine otherwise.
    #[default]
    Auto,
    /// Always reconstruct under the perspective model. If no perspective geometry can be
    /// estimated at all, the model is empty.
    Perspective,
    /// Always use the affine model, which never looks at camera geometry. Its depth scale
    /// is a fixed guess and it assumes the middle of the scene is nearest the viewer.
    Affine,
}

/// How [`Model3D::with_settings`] reconstructs.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Settings {
    /// Which camera model to assume. Defaults to [`Projection::Auto`].
    pub projection: Projection,
    /// The focal lengths of the cameras that took photo 1 and photo 2, each as a multiple
    /// of the photo's larger dimension. Defaults to [`DEFAULT_FOCAL_LENGTH`] for both.
    ///
    /// A wrong value mostly stretches or squashes the model along the viewing direction.
    /// [`Settings::focal_length_from_35mm`] converts the 35 mm equivalent focal length
    /// most cameras record in their EXIF data.
    pub focal_lengths: [f64; 2],
    /// Seeds the RANSAC sampling, so the same mapping always gives the same model.
    pub seed: u64,
}

impl Default for Settings {
    fn default() -> Self {
        Settings {
            projection: Projection::Auto,
            focal_lengths: [DEFAULT_FOCAL_LENGTH; 2],
            seed: 0x3D_5EED,
        }
    }
}

impl Settings {
    /// Converts a 35 mm equivalent focal length, for a photo of `width` x `height`, into
    /// the ratio [`Settings::focal_lengths`] expects.
    ///
    /// Only the aspect ratio of the dimensions matters, so the original photo's size and
    /// the working resolution give the same answer.
    pub fn focal_length_from_35mm(millimetres: f64, width: usize, height: usize) -> f64 {
        let (w, h) = (width as f64, height as f64);
        millimetres / FULL_FRAME_DIAGONAL_MM * w.hypot(h) / w.max(h)
    }
}

/// Where the first camera sits in a perspective model.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Viewpoint {
    pub position: [f32; 3],
    /// Radians, across the smaller dimension of the photo.
    pub field_of_view: f32,
}

/// How to decide that a face spans a gap in the surface rather than the surface itself.
enum Tearing {
    /// The affine model's depth is compressed by a fixed ratio; undo that and compare
    /// against a fixed length.
    Affine { z_scale: f32 },
    /// The perspective model's grid spacing grows with depth, so compare against the
    /// spacing expected at each cell.
    Perspective { spacing: Vec<f32> },
}

/// A 3D model built from a 2D grid of correspondences. Each cell in the grid has a
/// position in 3D space (`x, y, z`) plus texture coordinates (`u, v`) mapping it
/// into an associated photo.
pub struct Model3D {
    /// The number of grid cells in the horizontal direction.
    pub grid_width: usize,

    /// The number of grid cells in the vertical direction.
    pub grid_height: usize,

    /// A reference-counted handle to the source `Photo` used for texturing.
    pub photo: Arc<Photo>,

    /// A flat storage of [`TexturePoint`]s, of length `grid_width * grid_height`,
    /// describing each cell's 3D location and texture coordinates.
    grid: Vec<TexturePoint>,

    projection: Projection,
    geometry: Option<TwoViewGeometry>,
    tearing: Tearing,
    viewpoint: Option<Viewpoint>,
}

/// Represents a single point's 3D position along with texture coordinates.
/// It also tracks which cell in the grid (`grid_x, grid_y`) it belongs to.
#[derive(Clone, Copy)]
pub struct TexturePoint {
    /// 3D x-coordinate of this point.
    pub x: f32,
    /// 3D y-coordinate of this point.
    pub y: f32,
    /// 3D z-coordinate of this point.
    pub z: f32,

    /// Horizontal texture coordinate (U axis), in [0, 1].
    pub u: f32,
    /// Vertical texture coordinate (V axis), in [0, 1].
    pub v: f32,

    /// The x-index in the grid this point belongs to.
    pub grid_x: usize,
    /// The y-index in the grid this point belongs to.
    pub grid_y: usize,
}

impl TexturePoint {
    /// Computes the Euclidean distance between `self` and `other`, using (x, y, z).
    ///
    /// # Returns
    /// The distance `sqrt((x1 - x2)^2 + (y1 - y2)^2 + (z1 - z2)^2)`.
    pub fn distance(&self, other: &TexturePoint) -> f32 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        let dz = self.z - other.z;
        (dx * dx + dy * dy + dz * dz).sqrt()
    }
}

/// Provides a default `TexturePoint` with `x, y, z = NaN`, `u, v = 0`, and grid indices = 0.
/// This indicates an invalid or uninitialized point.
impl Default for TexturePoint {
    fn default() -> Self {
        TexturePoint {
            x: f32::NAN,
            y: f32::NAN,
            z: f32::NAN,
            u: 0.0,
            v: 0.0,
            grid_x: 0,
            grid_y: 0,
        }
    }
}

impl Model3D {
    /// Reconstructs a model from a mapping with [`Settings::default`].
    ///
    /// Grid cells that were never mapped, or whose depth could not be recovered, hold a
    /// point with `NaN` coordinates.
    pub fn new(photo_mapping: &DensePhotoMap) -> Self {
        Self::with_settings(photo_mapping, &Settings::default())
    }

    /// Reconstructs a model from a mapping; see the [crate documentation](crate) for how.
    pub fn with_settings(photo_mapping: &DensePhotoMap, settings: &Settings) -> Self {
        if settings.projection == Projection::Affine {
            return Self::affine(photo_mapping, None);
        }

        let (cells, pairs) = correspondences(photo_mapping);
        let photo = photo_mapping.photo1();
        let estimate = two_view::estimate(&pairs, photo.width(), photo.height(), settings);

        let Some(estimate) = estimate else {
            return match settings.projection {
                Projection::Perspective => Self::empty(photo_mapping, Projection::Perspective),
                _ => Self::affine(photo_mapping, None),
            };
        };
        if settings.projection == Projection::Auto && !estimate.geometry.is_well_conditioned() {
            return Self::affine(photo_mapping, Some(estimate.geometry));
        }

        let surface = perspective::build(photo_mapping, &cells, &pairs, &estimate);
        let mut model = Self::empty(photo_mapping, Projection::Perspective);
        model.geometry = Some(estimate.geometry);
        if let Some(surface) = surface {
            model.grid = surface.grid;
            model.tearing = Tearing::Perspective {
                spacing: surface.spacing,
            };
            model.viewpoint = Some(surface.viewpoint);
        }
        model
    }

    fn empty(photo_mapping: &DensePhotoMap, projection: Projection) -> Self {
        let (grid_width, grid_height) = photo_mapping.grid_dimensions();
        Model3D {
            grid_width,
            grid_height,
            photo: photo_mapping.photo1().clone(),
            grid: vec![TexturePoint::default(); grid_width * grid_height],
            projection,
            geometry: None,
            tearing: Tearing::Affine { z_scale: 1.0 },
            viewpoint: None,
        }
    }

    fn affine(photo_mapping: &DensePhotoMap, geometry: Option<TwoViewGeometry>) -> Self {
        let (grid, z_scale) = affine::reconstruct(photo_mapping);
        Model3D {
            grid,
            geometry,
            tearing: Tearing::Affine { z_scale },
            ..Self::empty(photo_mapping, Projection::Affine)
        }
    }

    /// The camera model this reconstruction used: [`Projection::Perspective`] or
    /// [`Projection::Affine`], never [`Projection::Auto`].
    pub fn projection(&self) -> Projection {
        self.projection
    }

    /// The perspective geometry estimated between the two photos, if one was attempted
    /// and succeeded — including when [`Projection::Auto`] then judged it too weak and
    /// fell back to the affine model, in which case this says why.
    pub fn two_view_geometry(&self) -> Option<&TwoViewGeometry> {
        self.geometry.as_ref()
    }

    /// Retrieves a reference to the `TexturePoint` at grid cell `(x, y)`.
    ///
    /// # Panics
    /// If `(x, y)` is out of range, this will cause a panic due to a slice index out of bounds.
    pub fn get_texture_point(&self, x: usize, y: usize) -> &TexturePoint {
        &self.grid[y * self.grid_width + x]
    }

    /// Whether the mesh edge between grid cells `a` and `b` (flat indices) spans a gap in
    /// the surface rather than the surface itself.
    fn is_torn(&self, a: usize, b: usize, diagonal: bool) -> bool {
        let (pa, pb) = (&self.grid[a], &self.grid[b]);
        match &self.tearing {
            Tearing::Affine { z_scale } => {
                // The model's depth is deliberately compressed to `Z_DEPTH_RATIO` of the
                // image plane; stretch it back out so the test is equally sensitive in
                // all three directions.
                let dz = if *z_scale > 0.0 {
                    (pa.z - pb.z) / z_scale
                } else {
                    0.0
                };
                let (dx, dy) = (pa.x - pb.x, pa.y - pb.y);
                (dx * dx + dy * dy + dz * dz).sqrt() > AFFINE_MAX_EDGE
            }
            Tearing::Perspective { spacing } => {
                let steps = if diagonal {
                    std::f32::consts::SQRT_2
                } else {
                    1.0
                };
                let expected = 0.5 * (spacing[a] + spacing[b]) * steps;
                pa.distance(pb) > PERSPECTIVE_MAX_STRETCH * expected
            }
        }
    }

    /// Creates and returns an X3D string representing the 3D mesh of points.
    ///
    /// # Details
    /// - The `<IndexedFaceSet>` is built by iterating over each cell `(x, y)` and forming quads
    ///   (split into triangles) with the adjacent cells `(x+1, y)`, `(x, y+1)`, `(x+1, y+1)`.
    /// - Invalid points (with `NaN` coordinates) are skipped.
    /// - A quad with an edge that spans a gap in the surface — a depth discontinuity, or
    ///   an outlier — is skipped.
    /// - Texture coordinates and 3D positions are embedded in the X3D output.
    /// - A perspective model also carries a `<Viewpoint>` at the first camera, so a viewer
    ///   opens on the scene as photo 1 saw it.
    ///
    /// Replace `"[photo_placeholder]"` in the string with a real texture file URL if needed.
    pub fn to_x3d(&self) -> String {
        let mut result = String::new();
        result.push_str(
            r#"<X3D width="1000px" height="1000px">
    <head>
        <meta name='title' content='3D Model'/>
        <meta name='description' content='3D Model with texture'/>
    </head>
    <Scene>
"#,
        );
        if let Some(viewpoint) = &self.viewpoint {
            let [x, y, z] = viewpoint.position;
            result.push_str(&format!(
                "        <Viewpoint description='photo 1' position='{x} {y} {z}' \
                 fieldOfView='{}'></Viewpoint>\n",
                viewpoint.field_of_view
            ));
        }
        result.push_str(
            r#"        <Shape>
            <Appearance>
                <ImageTexture id="imagetexture" url='"#,
        );
        result.push_str("[photo_placeholder]");
        result.push_str(
            r#"'></ImageTexture>
            </Appearance>
            <IndexedFaceSet solid="false" ccw="true" colorPerVertex="false" coordIndex='"#,
        );

        // Each pair of adjacent cells forms two triangles, if valid.
        for y in 0..self.grid_height.saturating_sub(1) {
            for x in 0..self.grid_width.saturating_sub(1) {
                let i1 = y * self.grid_width + x;
                let (i2, i3, i4) = (i1 + 1, i1 + self.grid_width, i1 + self.grid_width + 1);

                // Skip faces if any point is invalid or the quad spans a gap.
                if [i1, i2, i3, i4].iter().any(|&i| self.grid[i].x.is_nan()) {
                    continue;
                }
                if self.is_torn(i1, i2, false)
                    || self.is_torn(i1, i3, false)
                    || self.is_torn(i2, i4, false)
                    || self.is_torn(i3, i4, false)
                    || self.is_torn(i1, i4, true)
                {
                    continue;
                }

                // Construct two triangles (p1->p2->p4 and p1->p4->p3).
                // X3D uses -1 as a face separator.
                result.push_str(&format!("{i1} {i2} {i4} -1 {i1} {i4} {i3} -1 "));
            }
        }

        result.push_str("'>\n<Coordinate point='");

        // Write out the 3D coordinates of every grid cell.
        for p in &self.grid {
            if p.x.is_nan() {
                result.push_str("0 0 0 ");
            } else {
                result.push_str(&format!("{} {} {} ", p.x, p.y, p.z));
            }
        }

        result.push_str("'></Coordinate>\n<TextureCoordinate point='");

        // Write out the (u, v) texture coordinates.
        for p in &self.grid {
            if p.x.is_nan() {
                result.push_str("0 0 ");
            } else {
                result.push_str(&format!("{} {} ", p.u, p.v));
            }
        }

        result.push_str(
            r#"'></TextureCoordinate>
            </IndexedFaceSet>
        </Shape>
    </Scene>
</X3D>"#,
        );
        result
    }
}

/// The mapped cells of `map`, as grid coordinates and the correspondences they carry.
fn correspondences(map: &DensePhotoMap) -> (Vec<(usize, usize)>, Vec<PointPair>) {
    let (grid_width, grid_height) = map.grid_dimensions();
    let cell = map.grid_cell_size() as f64;
    (0..grid_height)
        .flat_map(|gy| (0..grid_width).map(move |gx| (gx, gy)))
        .filter_map(|(gx, gy)| {
            let (x2, y2) = map.grid_coordinates(gx, gy);
            (!x2.is_nan() && !y2.is_nan()).then_some((
                (gx, gy),
                PointPair {
                    x1: gx as f64 * cell,
                    y1: gy as f64 * cell,
                    x2: x2 as f64,
                    y2: y2 as f64,
                },
            ))
        })
        .unzip()
}

/// The texture coordinates of grid cell `(gx, gy)`: the centre of the photo-1 pixel the
/// cell sits on, with `v` running up from the bottom of the image as X3D expects.
///
/// Derived from the pixel position rather than from the cell's index over the grid size:
/// the grid's last node generally falls short of the photo's last pixel, so dividing by
/// the grid size slid the texture across the mesh by up to a cell.
pub(crate) fn texture_coordinates(map: &DensePhotoMap, gx: usize, gy: usize) -> (f32, f32) {
    let photo = map.photo1();
    let cell = map.grid_cell_size() as f32;
    let u = (gx as f32 * cell + 0.5) / photo.width() as f32;
    let v = 1.0 - (gy as f32 * cell + 0.5) / photo.height() as f32;
    (u.clamp(0.0, 1.0), v.clamp(0.0, 1.0))
}

/// The `q`-quantile of `values` (nearest rank), reordering them. `values` must not be
/// empty.
pub(crate) fn quantile(values: &mut [f64], q: f64) -> f64 {
    values.sort_unstable_by(f64::total_cmp);
    values[((values.len() - 1) as f64 * q).round() as usize]
}

#[cfg(test)]
mod tests {
    use nalgebra::{Matrix3, Rotation3, Vector3};

    use super::*;

    const WIDTH: usize = 320;
    const HEIGHT: usize = 240;
    const CELL: usize = 8;
    const DISTANCE: f64 = 10.0;

    fn photo() -> Arc<Photo> {
        Arc::new(Photo::from_rgba(WIDTH, HEIGHT, vec![0u8; WIDTH * HEIGHT * 4]).unwrap())
    }

    fn grid_size() -> (usize, usize) {
        ((WIDTH - 1) / CELL + 1, (HEIGHT - 1) / CELL + 1)
    }

    /// The second camera: one unit to the right of the first, turned back towards the
    /// middle of the scene. Returns `(R, t)` with `X2 = R·X1 + t`.
    fn second_camera() -> (Matrix3<f64>, Vector3<f64>) {
        let angle = 1.0f64.atan2(DISTANCE);
        let rotation = Rotation3::from_axis_angle(&Vector3::y_axis(), angle).into_inner();
        let centre = Vector3::new(1.0, 0.0, 0.0);
        (rotation, -(rotation * centre))
    }

    /// Photographs a surface with both cameras and records the correspondences as a map.
    /// `depth(u, v)` gives the depth along camera 1's axis at normalized image position
    /// `(u, v)`, both in `[-1, 1]`.
    fn scene(depth: impl Fn(f64, f64) -> f64) -> DensePhotoMap {
        let (gw, gh) = grid_size();
        let mut map = DensePhotoMap::with_cell_size(photo(), photo(), gw, gh, CELL);
        let focal = DEFAULT_FOCAL_LENGTH * WIDTH as f64;
        let (cx, cy) = ((WIDTH - 1) as f64 / 2.0, (HEIGHT - 1) as f64 / 2.0);
        let (rotation, translation) = second_camera();
        for gy in 0..gh {
            for gx in 0..gw {
                let (px, py) = ((gx * CELL) as f64, (gy * CELL) as f64);
                let z = depth((px - cx) / cx, (py - cy) / cy);
                let point = Vector3::new((px - cx) / focal * z, (py - cy) / focal * z, z);
                let seen = rotation * point + translation;
                if seen.z <= 0.0 {
                    continue;
                }
                let x2 = focal * seen.x / seen.z + cx;
                let y2 = focal * seen.y / seen.z + cy;
                if (0.0..=(WIDTH - 1) as f64).contains(&x2)
                    && (0.0..=(HEIGHT - 1) as f64).contains(&y2)
                {
                    map.set_grid_coordinates(gx, gy, x2 as f32, y2 as f32);
                }
            }
        }
        map
    }

    fn bump(u: f64, v: f64) -> f64 {
        DISTANCE - 2.0 * (1.0 - (u * u + v * v)).max(0.0)
    }

    fn bowl(u: f64, v: f64) -> f64 {
        DISTANCE + 2.0 * (1.0 - (u * u + v * v)).max(0.0)
    }

    fn centre_and_corner_z(model: &Model3D) -> (f32, f32) {
        let (gw, gh) = grid_size();
        (
            model.get_texture_point(gw / 2, gh / 2).z,
            model.get_texture_point(3, 3).z,
        )
    }

    fn pearson(a: &[f64], b: &[f64]) -> f64 {
        let n = a.len() as f64;
        let (ma, mb) = (a.iter().sum::<f64>() / n, b.iter().sum::<f64>() / n);
        let cov: f64 = a.iter().zip(b).map(|(x, y)| (x - ma) * (y - mb)).sum();
        let va: f64 = a.iter().map(|x| (x - ma).powi(2)).sum();
        let vb: f64 = b.iter().map(|y| (y - mb).powi(2)).sum();
        cov / (va * vb).sqrt()
    }

    #[test]
    fn a_bump_comes_out_towards_the_viewer() {
        let model = Model3D::new(&scene(bump));
        assert_eq!(model.projection(), Projection::Perspective);
        let (centre, corner) = centre_and_corner_z(&model);
        assert!(
            centre > corner,
            "centre z {centre} behind corner z {corner}"
        );
    }

    /// The case the affine model's "the middle is nearest" guess gets wrong.
    #[test]
    fn a_concave_scene_is_not_turned_inside_out() {
        let model = Model3D::new(&scene(bowl));
        assert_eq!(model.projection(), Projection::Perspective);
        let (centre, corner) = centre_and_corner_z(&model);
        assert!(
            centre < corner,
            "centre z {centre} in front of corner z {corner}"
        );
    }

    #[test]
    fn depth_follows_the_true_surface() {
        let model = Model3D::new(&scene(bump));
        let (cx, cy) = ((WIDTH - 1) as f64 / 2.0, (HEIGHT - 1) as f64 / 2.0);
        let (mut got, mut truth) = (Vec::new(), Vec::new());
        for p in model.grid.iter().filter(|p| !p.x.is_nan()) {
            let (px, py) = ((p.grid_x * CELL) as f64, (p.grid_y * CELL) as f64);
            got.push(p.z as f64);
            // Nearer is larger z in the model.
            truth.push(-bump((px - cx) / cx, (py - cy) / cy));
        }
        let (gw, gh) = grid_size();
        assert!(got.len() * 10 > gw * gh * 9, "only {} points", got.len());
        let r = pearson(&got, &truth);
        assert!(r > 0.99, "depth correlation with the true surface was {r}");
    }

    #[test]
    fn the_relative_pose_is_recovered() {
        let model = Model3D::new(&scene(bump));
        let geometry = model.two_view_geometry().expect("geometry was estimated");
        let (rotation, translation) = second_camera();

        let error = geometry.rotation * rotation.transpose();
        let angle = ((error.trace() - 1.0) / 2.0).clamp(-1.0, 1.0).acos();
        assert!(angle.to_degrees() < 0.1, "rotation off by {angle} rad");

        let cos = geometry.translation.dot(&translation.normalize());
        assert!(
            cos > 0.5f64.to_radians().cos(),
            "translation off, cos = {cos}"
        );
        assert!(geometry.inlier_fraction > 0.99);
        assert!(geometry.is_well_conditioned());
    }

    #[test]
    fn outlying_matches_do_not_derail_the_estimate() {
        let mut map = scene(bump);
        let (gw, gh) = grid_size();
        // A deterministic scatter of a quarter of the cells to random places.
        let mut state = 12345u64;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 33) as f64 / (1u64 << 31) as f64
        };
        for gy in 0..gh {
            for gx in 0..gw {
                if next() < 0.25 {
                    let (x, y) = (next() * WIDTH as f64, next() * HEIGHT as f64);
                    map.set_grid_coordinates(gx, gy, x as f32, y as f32);
                }
            }
        }

        let model = Model3D::new(&map);
        assert_eq!(model.projection(), Projection::Perspective);
        let geometry = model.two_view_geometry().unwrap();
        let (rotation, _) = second_camera();
        let error = geometry.rotation * rotation.transpose();
        let angle = ((error.trace() - 1.0) / 2.0).clamp(-1.0, 1.0).acos();
        assert!(angle.to_degrees() < 0.5, "rotation off by {angle} rad");
        let (centre, corner) = centre_and_corner_z(&model);
        assert!(centre > corner);
    }

    #[test]
    fn a_planar_scene_falls_back_to_the_affine_model() {
        let model = Model3D::new(&scene(|u, _| DISTANCE + 1.5 * u));
        assert_eq!(model.projection(), Projection::Affine);
        let geometry = model
            .two_view_geometry()
            .expect("the estimate is still reported");
        assert!(!geometry.is_well_conditioned());
    }

    #[test]
    fn forcing_the_affine_model_skips_the_estimate() {
        let settings = Settings {
            projection: Projection::Affine,
            ..Settings::default()
        };
        let model = Model3D::with_settings(&scene(bump), &settings);
        assert_eq!(model.projection(), Projection::Affine);
        assert!(model.two_view_geometry().is_none());
        assert!(!model.to_x3d().contains("<Viewpoint"));
    }

    #[test]
    fn a_perspective_mesh_opens_at_the_first_camera() {
        let x3d = Model3D::new(&scene(bump)).to_x3d();
        assert!(x3d.contains("<Viewpoint"));
        let faces = x3d.split("coordIndex='").nth(1).unwrap();
        assert!(faces.starts_with(|c: char| c.is_ascii_digit()), "no faces");
    }

    #[test]
    fn texture_coordinates_address_pixel_centres() {
        let model = Model3D::new(&scene(bump));
        let p = model.get_texture_point(5, 4);
        assert_eq!(p.u, ((5 * CELL) as f32 + 0.5) / WIDTH as f32);
        assert_eq!(p.v, 1.0 - ((4 * CELL) as f32 + 0.5) / HEIGHT as f32);
    }

    #[test]
    fn focal_length_from_35mm() {
        // A 36 x 24 frame is its own reference: 36 mm across the long side is 1.0.
        let ratio = Settings::focal_length_from_35mm(36.0, 36, 24);
        assert!((ratio - 1.0).abs() < 1e-12, "{ratio}");
    }
}
