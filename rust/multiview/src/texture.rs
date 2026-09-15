//! Stage 8: colour from the photos onto the mesh.
//!
//! Two outputs, from the same view scoring:
//!
//! - **Vertex colours.** Each vertex is a blend of the views that see it best. Simple, and
//!   good enough for a viewer that cannot load textures.
//! - **A texture atlas.** Each triangle takes its colour from the single view that sees it
//!   best: the most head-on, from closest up, and actually visible. A few smoothing passes
//!   make neighbouring triangles agree on a view, so that there are fewer seams. Connected
//!   triangles that share a view form a chart. Each chart is the rectangle of its photo
//!   that covers it, padded so that mipmapping does not bleed across charts, and the
//!   charts are packed row by row into one image.
//!
//! Where neighbouring charts come from different photos, differences in exposure show up
//! as seams. Levelling colour across chart boundaries is left for later.

use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::BuildHasherDefault;
use std::sync::Arc;

use nalgebra::{Point3, Vector3};
use pixelmap::Photo;

use crate::calib::Intrinsics;
use crate::depth::DepthMap;
use crate::mesh::Mesh;
use crate::pose::Pose;
use crate::types::{PhotoPx, ViewId};

/// Tuning for [`build`].
#[derive(Clone, Debug)]
pub struct Params {
    /// How far a surface point may be from the depth a view measured there, as a fraction
    /// of that depth, and still count as visible.
    pub visibility_tolerance: f64,
    /// Views seeing a surface at a more grazing angle than this cosine are not used.
    pub min_cosine: f64,
    /// Passes of neighbourhood voting on each triangle's view.
    pub smoothing_passes: usize,
    /// How much each neighbour agreeing on a view counts, against a triangle's own
    /// preference (1 for its best view).
    pub smoothing_weight: f64,
    /// Pixels of photo kept around each chart.
    pub padding: usize,
    /// The largest atlas edge, in pixels. Charts are scaled down to fit.
    pub max_atlas_size: usize,
}

impl Default for Params {
    fn default() -> Self {
        Params {
            visibility_tolerance: 0.03,
            min_cosine: 0.1,
            smoothing_passes: 4,
            smoothing_weight: 0.3,
            padding: 3,
            max_atlas_size: 8192,
        }
    }
}

/// Colour for a mesh.
#[derive(Clone, Debug)]
pub struct Texture {
    /// One colour per mesh vertex.
    pub vertex_colours: Vec<[u8; 3]>,
    /// The view each triangle's texture comes from.
    pub face_views: Vec<ViewId>,
    /// Texture coordinates into [`Self::atlas`], OBJ convention: `v` grows upwards.
    pub texcoords: Vec<[f32; 2]>,
    /// For each triangle, the texture coordinates of its three corners.
    pub face_texcoords: Vec<[u32; 3]>,
    /// The packed charts.
    pub atlas: Photo,
    /// How many charts the atlas holds.
    pub charts: usize,
    /// Atlas pixels per photo pixel: below 1 when the charts had to be shrunk to fit.
    pub atlas_scale: f64,
}

/// Colours `mesh`, which must have normals, from `photos` taken by `cameras`. The depth
/// maps decide which views actually see which parts of the surface.
pub fn build(
    mesh: &Mesh,
    cameras: &[Option<Pose>],
    intrinsics: &Intrinsics,
    photos: &[Arc<Photo>],
    depth: &[DepthMap],
    params: &Params,
) -> Texture {
    let views: Vec<View> = cameras
        .iter()
        .enumerate()
        .filter_map(|(v, pose)| {
            let pose = (*pose)?;
            let id = ViewId(v as u32);
            Some(View {
                id,
                pose,
                centre: pose.centre(),
                photo: photos.get(v)?,
                depth: depth.iter().find(|m| m.view == id),
            })
        })
        .collect();

    let vertex_colours = mesh
        .positions
        .iter()
        .enumerate()
        .map(|(i, p)| {
            let normal = mesh.normals.get(i).copied().unwrap_or_else(Vector3::zeros);
            vertex_colour(&views, intrinsics, p, &normal, params)
        })
        .collect();

    let (labels, neighbours) = choose_views(mesh, &views, intrinsics, params);
    let charts = charts(mesh, &labels, &neighbours);
    let (texcoords, face_texcoords, atlas, atlas_scale) =
        pack(mesh, &views, intrinsics, &labels, &charts, params);

    Texture {
        vertex_colours,
        face_views: labels.iter().map(|&l| views[l].id).collect(),
        texcoords,
        face_texcoords,
        atlas,
        charts: charts.len(),
        atlas_scale,
    }
}

struct View<'a> {
    id: ViewId,
    pose: Pose,
    centre: Point3<f64>,
    photo: &'a Photo,
    depth: Option<&'a DepthMap>,
}

impl View<'_> {
    /// Where `point` lands in this view's photo, in pixels, and its depth. Not limited to
    /// the photo's bounds.
    fn project(&self, intrinsics: &Intrinsics, point: &Point3<f64>) -> Option<(f64, f64, f64)> {
        let c = self.pose.to_camera(point);
        if c.z <= 1e-9 {
            return None;
        }
        let x = intrinsics.fx * c.x / c.z + intrinsics.cx;
        let y = intrinsics.fy * c.y / c.z + intrinsics.cy;
        Some((x, y, c.z))
    }

    fn inside(&self, x: f64, y: f64) -> bool {
        x >= 0.0
            && y >= 0.0
            && x <= (self.photo.width() - 1) as f64
            && y <= (self.photo.height() - 1) as f64
    }

    /// How well this view sees a surface point with the given normal: the cosine of the
    /// viewing angle times pixels per unit length. Zero if the point is outside the photo
    /// or seen too obliquely. The flag says whether the view's depth map confirms the point
    /// is not hidden.
    fn score(
        &self,
        intrinsics: &Intrinsics,
        point: &Point3<f64>,
        normal: &Vector3<f64>,
        params: &Params,
    ) -> (f64, bool) {
        let Some((x, y, z)) = self.project(intrinsics, point) else {
            return (0.0, false);
        };
        if !self.inside(x, y) {
            return (0.0, false);
        }
        let to_camera = self.centre - point;
        let distance = to_camera.norm();
        let cosine = normal.dot(&to_camera) / distance;
        if cosine < params.min_cosine {
            return (0.0, false);
        }
        let visible = self
            .depth
            .and_then(|m| m.sample(PhotoPx::new(x as f32, y as f32)))
            .is_some_and(|d| (d as f64 - z).abs() <= params.visibility_tolerance * z);
        (cosine * intrinsics.fx / distance, visible)
    }
}

/// Each view's score for a surface point. Views whose depth maps do not confirm the point
/// count for nothing, unless none confirm it; then every view in front of it counts.
fn scores(
    views: &[View],
    intrinsics: &Intrinsics,
    point: &Point3<f64>,
    normal: &Vector3<f64>,
    params: &Params,
) -> Vec<f64> {
    let raw: Vec<(f64, bool)> = views
        .iter()
        .map(|v| v.score(intrinsics, point, normal, params))
        .collect();
    let any_visible = raw.iter().any(|&(s, visible)| visible && s > 0.0);
    raw.into_iter()
        .map(|(s, visible)| if visible || !any_visible { s } else { 0.0 })
        .collect()
}

fn vertex_colour(
    views: &[View],
    intrinsics: &Intrinsics,
    point: &Point3<f64>,
    normal: &Vector3<f64>,
    params: &Params,
) -> [u8; 3] {
    let scores = scores(views, intrinsics, point, normal, params);
    let best = scores.iter().copied().fold(0.0, f64::max);
    if best <= 0.0 {
        return [128, 128, 128];
    }
    let mut sum = [0.0; 3];
    let mut total = 0.0;
    for (view, &score) in views.iter().zip(&scores) {
        if score < 0.5 * best {
            continue;
        }
        let Some((x, y, _)) = view.project(intrinsics, point) else {
            continue;
        };
        let weight = score * score;
        for (s, c) in sum.iter_mut().zip(sample(view.photo, x, y)) {
            *s += weight * c;
        }
        total += weight;
    }
    sum.map(|s| (s / total).round().clamp(0.0, 255.0) as u8)
}

/// Bilinear sample of a photo at pixel position `(x, y)`, clamped to its bounds.
fn sample(photo: &Photo, x: f64, y: f64) -> [f64; 3] {
    let (w, h) = (photo.width(), photo.height());
    let x = x.clamp(0.0, (w - 1) as f64);
    let y = y.clamp(0.0, (h - 1) as f64);
    let (x0, y0) = (x.floor() as usize, y.floor() as usize);
    let (x1, y1) = ((x0 + 1).min(w - 1), (y0 + 1).min(h - 1));
    let (fx, fy) = (x - x0 as f64, y - y0 as f64);
    let at = |px: usize, py: usize| photo.pixel(px, py).unwrap_or([0; 4]);
    let (a, b, c, d) = (at(x0, y0), at(x1, y0), at(x0, y1), at(x1, y1));
    std::array::from_fn(|k| {
        let top = a[k] as f64 * (1.0 - fx) + b[k] as f64 * fx;
        let bottom = c[k] as f64 * (1.0 - fx) + d[k] as f64 * fx;
        top * (1.0 - fy) + bottom * fy
    })
}

type Deterministic = BuildHasherDefault<DefaultHasher>;

/// Each triangle's view, after smoothing, and each triangle's edge neighbours.
fn choose_views(
    mesh: &Mesh,
    views: &[View],
    intrinsics: &Intrinsics,
    params: &Params,
) -> (Vec<usize>, Vec<Vec<usize>>) {
    let face_scores: Vec<Vec<f64>> = mesh
        .triangles
        .iter()
        .map(|t| {
            let [a, b, c] = t.map(|i| mesh.positions[i as usize]);
            let centroid = Point3::from((a.coords + b.coords + c.coords) / 3.0);
            let normal = (b - a)
                .cross(&(c - a))
                .try_normalize(1e-300)
                .unwrap_or_else(Vector3::zeros);
            scores(views, intrinsics, &centroid, &normal, params)
        })
        .collect();

    let mut edges: HashMap<(u32, u32), Vec<usize>, Deterministic> = HashMap::default();
    for (f, t) in mesh.triangles.iter().enumerate() {
        for k in 0..3 {
            let (a, b) = (t[k], t[(k + 1) % 3]);
            edges.entry((a.min(b), a.max(b))).or_default().push(f);
        }
    }
    let mut neighbours = vec![Vec::new(); mesh.triangles.len()];
    for faces in edges.values() {
        for &f in faces {
            for &g in faces {
                if f != g {
                    neighbours[f].push(g);
                }
            }
        }
    }
    for list in &mut neighbours {
        list.sort_unstable();
        list.dedup();
    }

    let argmax = |values: &[f64]| {
        values
            .iter()
            .enumerate()
            .fold((0, f64::NEG_INFINITY), |best, (i, &v)| {
                if v > best.1 {
                    (i, v)
                } else {
                    best
                }
            })
            .0
    };
    let mut labels: Vec<usize> = face_scores.iter().map(|s| argmax(s)).collect();

    for _ in 0..params.smoothing_passes {
        let previous = labels.clone();
        for (f, scores) in face_scores.iter().enumerate() {
            let best = scores.iter().copied().fold(0.0, f64::max);
            if best <= 0.0 {
                continue;
            }
            let votes: Vec<f64> = scores
                .iter()
                .enumerate()
                .map(|(v, &s)| {
                    if s <= 0.0 {
                        return f64::NEG_INFINITY;
                    }
                    let agreeing = neighbours[f].iter().filter(|&&g| previous[g] == v).count();
                    s / best + params.smoothing_weight * agreeing as f64
                })
                .collect();
            labels[f] = argmax(&votes);
        }
    }
    (labels, neighbours)
}

/// Groups triangles into charts: connected through shared edges, with the same view.
/// Charts are numbered in order of their first triangle.
fn charts(mesh: &Mesh, labels: &[usize], neighbours: &[Vec<usize>]) -> Vec<Vec<usize>> {
    let mut chart_of = vec![usize::MAX; mesh.triangles.len()];
    let mut charts = Vec::new();
    for start in 0..mesh.triangles.len() {
        if chart_of[start] != usize::MAX {
            continue;
        }
        let id = charts.len();
        let mut members = vec![start];
        chart_of[start] = id;
        let mut next = 0;
        while next < members.len() {
            let f = members[next];
            next += 1;
            for &g in &neighbours[f] {
                if chart_of[g] == usize::MAX && labels[g] == labels[f] {
                    chart_of[g] = id;
                    members.push(g);
                }
            }
        }
        charts.push(members);
    }
    charts
}

/// The texture coordinates, per-face coordinate indices, atlas and scale.
type Packed = (Vec<[f32; 2]>, Vec<[u32; 3]>, Photo, f64);

fn pack(
    mesh: &Mesh,
    views: &[View],
    intrinsics: &Intrinsics,
    labels: &[usize],
    charts: &[Vec<usize>],
    params: &Params,
) -> Packed {
    // Each chart's rectangle in its photo.
    let rects: Vec<(usize, usize, usize, usize)> = charts
        .iter()
        .map(|faces| {
            let view = &views[labels[faces[0]]];
            let (w, h) = (view.photo.width() as f64, view.photo.height() as f64);
            let (mut min_x, mut min_y, mut max_x, mut max_y) =
                (f64::MAX, f64::MAX, f64::MIN, f64::MIN);
            for &f in faces {
                for &v in &mesh.triangles[f] {
                    if let Some((x, y, _)) = view.project(intrinsics, &mesh.positions[v as usize]) {
                        min_x = min_x.min(x);
                        min_y = min_y.min(y);
                        max_x = max_x.max(x);
                        max_y = max_y.max(y);
                    }
                }
            }
            if min_x > max_x {
                return (0, 0, 1, 1);
            }
            let pad = params.padding as f64;
            let x0 = (min_x - pad).floor().clamp(0.0, w - 1.0) as usize;
            let y0 = (min_y - pad).floor().clamp(0.0, h - 1.0) as usize;
            let x1 = (max_x + pad).ceil().clamp(0.0, w - 1.0) as usize;
            let y1 = (max_y + pad).ceil().clamp(0.0, h - 1.0) as usize;
            (x0, y0, x1 - x0 + 1, y1 - y0 + 1)
        })
        .collect();

    // Shelf packing into a power-of-two atlas, shrinking the charts until it fits.
    let mut scale = 1.0f64;
    let (width, height, offsets, sizes) = loop {
        let sizes: Vec<(usize, usize)> = rects
            .iter()
            .map(|&(_, _, w, h)| {
                (
                    ((w as f64 * scale).ceil() as usize).max(1),
                    ((h as f64 * scale).ceil() as usize).max(1),
                )
            })
            .collect();
        let area: usize = sizes.iter().map(|(w, h)| w * h).sum();
        let widest = sizes.iter().map(|s| s.0).max().unwrap_or(1);
        let width = widest
            .max((area as f64 * 1.1).sqrt().ceil() as usize)
            .next_power_of_two();
        let (offsets, used) = shelves(&sizes, width);
        let height = used.max(1).next_power_of_two();
        if width <= params.max_atlas_size && height <= params.max_atlas_size {
            break (width, height, offsets, sizes);
        }
        scale *= 0.9
            * (params.max_atlas_size as f64 / width.max(height) as f64)
                .sqrt()
                .min(1.0);
    };

    let mut rgba = vec![128u8; width * height * 4];
    for pixel in rgba.as_chunks_mut::<4>().0 {
        pixel[3] = 255;
    }
    for (chart, faces) in charts.iter().enumerate() {
        let photo = views[labels[faces[0]]].photo;
        let (x0, y0, _, _) = rects[chart];
        let (ox, oy) = offsets[chart];
        let (sw, sh) = sizes[chart];
        for ay in 0..sh {
            for ax in 0..sw {
                let px = x0 as f64 + (ax as f64 + 0.5) / scale - 0.5;
                let py = y0 as f64 + (ay as f64 + 0.5) / scale - 0.5;
                let colour = sample(photo, px, py);
                let i = ((oy + ay) * width + ox + ax) * 4;
                for (k, c) in colour.iter().enumerate() {
                    rgba[i + k] = c.round().clamp(0.0, 255.0) as u8;
                }
            }
        }
    }

    let mut texcoords = Vec::new();
    let mut index: HashMap<(u32, usize), u32, Deterministic> = HashMap::default();
    let mut face_texcoords = vec![[0u32; 3]; mesh.triangles.len()];
    for (chart, faces) in charts.iter().enumerate() {
        let view = &views[labels[faces[0]]];
        let (x0, y0, _, _) = rects[chart];
        let (ox, oy) = offsets[chart];
        for &f in faces {
            for (corner, &v) in mesh.triangles[f].iter().enumerate() {
                face_texcoords[f][corner] = *index.entry((v, chart)).or_insert_with(|| {
                    let (x, y) = view
                        .project(intrinsics, &mesh.positions[v as usize])
                        .map_or((x0 as f64, y0 as f64), |(x, y, _)| (x, y));
                    let u = (ox as f64 + (x - x0 as f64 + 0.5) * scale) / width as f64;
                    let t = (oy as f64 + (y - y0 as f64 + 0.5) * scale) / height as f64;
                    texcoords.push([u as f32, (1.0 - t) as f32]);
                    (texcoords.len() - 1) as u32
                });
            }
        }
    }

    let atlas =
        Photo::from_rgba(width, height, rgba).expect("buffer is exactly width × height × 4");
    (texcoords, face_texcoords, atlas, scale)
}

/// Places rectangles on shelves across an image `width` wide, tallest first. Returns each
/// rectangle's offset, in input order, and the height used.
fn shelves(sizes: &[(usize, usize)], width: usize) -> (Vec<(usize, usize)>, usize) {
    let mut order: Vec<usize> = (0..sizes.len()).collect();
    order.sort_by_key(|&i| (std::cmp::Reverse(sizes[i].1), i));
    let mut offsets = vec![(0, 0); sizes.len()];
    let (mut x, mut y, mut shelf) = (0, 0, 0);
    for i in order {
        let (w, h) = sizes[i];
        if x + w > width {
            y += shelf;
            x = 0;
            shelf = 0;
        }
        offsets[i] = (x, y);
        x += w;
        shelf = shelf.max(h);
    }
    (offsets, y + shelf)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shelves_do_not_overlap() {
        let sizes = [(30, 10), (50, 40), (20, 20), (60, 5), (10, 35)];
        let (offsets, height) = shelves(&sizes, 64);
        for (i, (&(x, y), &(w, h))) in offsets.iter().zip(&sizes).enumerate() {
            assert!(x + w <= 64 && y + h <= height);
            for (&(x2, y2), &(w2, h2)) in offsets.iter().zip(&sizes).skip(i + 1) {
                let apart = x + w <= x2 || x2 + w2 <= x || y + h <= y2 || y2 + h2 <= y;
                assert!(apart, "rectangles overlap");
            }
        }
    }

    #[test]
    fn samples_between_pixels() {
        let photo = Photo::from_rgba(2, 1, vec![0, 0, 0, 255, 200, 100, 50, 255]).unwrap();
        assert_eq!(sample(&photo, 0.5, 0.0), [100.0, 50.0, 25.0]);
        assert_eq!(sample(&photo, 5.0, -3.0), [200.0, 100.0, 50.0]);
    }
}
