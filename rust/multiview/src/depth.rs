//! Stage 6: a dense depth map per registered view.
//!
//! The sparse tracks are not the model. With the cameras known, every pixel the mappings
//! cover can be given a depth:
//!
//! 1. For each reference pixel, gather where the other views see it.
//! 2. Solve for the depth along the reference pixel's own ray that best fits those views:
//!    a linear least-squares start, then Gauss–Newton on reprojection error. A view that
//!    does not fit is dropped and the depth solved again.
//! 3. Keep the depth only if the remaining views fit and the rays meet at a usable angle.
//! 4. Cross-check the maps against each other: a depth survives only if another view's
//!    depth map puts the same surface in the same place.
//! 5. Remove small disconnected patches, then smooth lightly without crossing depth
//!    edges.
//!
//! A pair two-view geometry flagged as degenerate still contributes here. Rays from three
//! or four views condition a depth far better than two did.

use nalgebra::{Point3, Vector3};

use crate::calib::Intrinsics;
use crate::error::Error;
use crate::lookup::PairLookup;
use crate::pairs::PairGraph;
use crate::pose::Pose;
use crate::triangulate;
use crate::types::{Norm, PhotoPx, ViewId, World};

/// The fewest valid samples, as a fraction of all depth-map samples, that fusion will
/// work from.
pub const MIN_VALID_FRACTION: f64 = 0.15;

/// Tuning for [`estimate`].
#[derive(Clone, Debug)]
pub struct Params {
    /// Spacing of depth samples, in photo pixels. `None` uses a third of the mappings'
    /// native stride: sampling densely enough to follow the mapping without repeating it.
    pub stride: Option<usize>,
    /// The largest reprojection error a view may have, as a multiple of the mappings'
    /// precision. Also the pixel tolerance of the cross-view check.
    pub max_reprojection: f64,
    /// The widest angle between the reference ray and another view's ray must reach this,
    /// in degrees.
    pub min_angle_deg: f64,
    /// The largest relative depth difference the cross-view check accepts.
    pub max_depth_difference: f64,
    /// How many other depth maps must agree with a depth for it to survive.
    pub min_consistent_views: usize,
    /// Neighbouring samples whose depths differ by more than this fraction are on
    /// different surfaces: they are not connected when removing small patches, and not
    /// mixed when smoothing.
    pub max_depth_step: f64,
    /// Connected patches with fewer samples than this are removed.
    pub min_component: usize,
    /// Whether to apply the edge-preserving median filter.
    pub smooth: bool,
}

impl Default for Params {
    fn default() -> Self {
        Params {
            stride: None,
            max_reprojection: 2.0,
            min_angle_deg: 2.0,
            max_depth_difference: 0.02,
            min_consistent_views: 1,
            max_depth_step: 0.05,
            min_component: 64,
            smooth: true,
        }
    }
}

/// Depths on a regular grid over one view's photo.
#[derive(Clone, Debug, PartialEq)]
pub struct DepthMap {
    /// The view.
    pub view: ViewId,
    /// Grid columns.
    pub columns: usize,
    /// Grid rows.
    pub rows: usize,
    /// Photo pixels between neighbouring samples.
    pub stride: usize,
    /// Camera-frame z of the surface at each sample, row by row. `NaN` where unknown.
    pub depth: Vec<f32>,
}

impl DepthMap {
    /// The photo position of grid sample `(column, row)`.
    pub fn pixel(&self, column: usize, row: usize) -> PhotoPx {
        PhotoPx::new((column * self.stride) as f32, (row * self.stride) as f32)
    }

    /// The depth at grid sample `(column, row)`, if known.
    pub fn get(&self, column: usize, row: usize) -> Option<f32> {
        let d = self.depth[row * self.columns + column];
        d.is_finite().then_some(d)
    }

    /// The depth of the sample nearest photo position `p`, if known.
    pub fn sample(&self, p: PhotoPx) -> Option<f32> {
        let column = (p.x() / self.stride as f32).round();
        let row = (p.y() / self.stride as f32).round();
        if column < 0.0 || row < 0.0 {
            return None;
        }
        let (column, row) = (column as usize, row as usize);
        if column >= self.columns || row >= self.rows {
            return None;
        }
        self.get(column, row)
    }

    /// How many samples have a depth.
    pub fn valid(&self) -> usize {
        self.depth.iter().filter(|d| d.is_finite()).count()
    }

    /// The fraction of samples that have a depth.
    pub fn valid_fraction(&self) -> f64 {
        self.valid() as f64 / self.depth.len().max(1) as f64
    }
}

/// A depth map for every registered view in `cameras`, over photos of `size`.
pub fn estimate<L: PairLookup>(
    graph: &PairGraph<L>,
    cameras: &[Option<Pose>],
    intrinsics: &Intrinsics,
    (width, height): (usize, usize),
    params: &Params,
) -> Vec<DepthMap> {
    let precision = graph.precision_px() as f64;
    let native = graph
        .pairs()
        .map(|(_, lookup)| lookup.native_stride())
        .fold(0.0f32, f32::max);
    let stride = params
        .stride
        .unwrap_or_else(|| (native / 3.0).round() as usize)
        .max(1);
    let focal = (intrinsics.fx + intrinsics.fy) / 2.0;
    let threshold = params.max_reprojection * precision / focal;
    let min_angle = params.min_angle_deg.to_radians();
    let columns = (width - 1) / stride + 1;
    let rows = (height - 1) / stride + 1;

    let registered: Vec<(ViewId, Pose)> = cameras
        .iter()
        .enumerate()
        .filter_map(|(v, c)| Some((ViewId(v as u32), (*c)?)))
        .filter(|(v, _)| v.index() < graph.views())
        .collect();

    let raw: Vec<DepthMap> = registered
        .iter()
        .map(|&(view, pose)| {
            let to_world = pose.rotation.inverse();
            let centre = pose.centre();
            let mut depth = vec![f32::NAN; columns * rows];
            let mut rays = Vec::with_capacity(registered.len());
            for row in 0..rows {
                for column in 0..columns {
                    let p = PhotoPx::new((column * stride) as f32, (row * stride) as f32);
                    rays.clear();
                    for &(other, other_pose) in &registered {
                        if other == view {
                            continue;
                        }
                        if let Some(q) = graph.directed(view, other).and_then(|m| m.map(p)) {
                            rays.push((other_pose, intrinsics.normalize(q)));
                        }
                    }
                    let n = intrinsics.normalize(p);
                    let direction = to_world * Vector3::new(n.x(), n.y(), 1.0);
                    if let Some(z) =
                        depth_along_ray(&centre, &direction, &mut rays, threshold, min_angle)
                    {
                        depth[row * columns + column] = z as f32;
                    }
                }
            }
            DepthMap {
                view,
                columns,
                rows,
                stride,
                depth,
            }
        })
        .collect();

    consistency_filter(&raw, cameras, intrinsics, precision, params)
        .into_iter()
        .map(|map| {
            let map = remove_speckles(map, params);
            if params.smooth {
                smooth(map, params)
            } else {
                map
            }
        })
        .collect()
}

/// The fraction of all samples in `maps` that have a depth.
///
/// # Errors
/// [`Error::InsufficientDepth`] below [`MIN_VALID_FRACTION`].
pub fn require_coverage(maps: &[DepthMap]) -> Result<f64, Error> {
    let total: usize = maps.iter().map(|m| m.depth.len()).sum();
    let valid: usize = maps.iter().map(DepthMap::valid).sum();
    let fraction = valid as f64 / total.max(1) as f64;
    if fraction < MIN_VALID_FRACTION {
        return Err(Error::InsufficientDepth {
            valid_fraction: fraction,
            required: MIN_VALID_FRACTION,
        });
    }
    Ok(fraction)
}

/// The depth along the ray `centre + z · direction` that best fits `rays`, where
/// `direction` has unit z in the reference camera's frame, so that `z` is the reference
/// camera's depth. Drops the worst-fitting ray until the rest fit within `threshold`.
fn depth_along_ray(
    centre: &Point3<f64>,
    direction: &Vector3<f64>,
    rays: &mut Vec<(Pose, Norm)>,
    threshold: f64,
    min_angle: f64,
) -> Option<f64> {
    while !rays.is_empty() {
        let z = solve_depth(centre, direction, rays)?;
        if z <= 0.0 {
            return None;
        }
        let point = World(centre + direction * z);
        let (worst, error) = rays
            .iter()
            .enumerate()
            .map(|(i, (pose, n))| {
                let error = pose
                    .project(&point)
                    .map_or(f64::INFINITY, |p| (p.x() - n.x()).hypot(p.y() - n.y()));
                (i, error)
            })
            .max_by(|a, b| a.1.total_cmp(&b.1))?;
        if error <= threshold {
            let angle = rays
                .iter()
                .map(|(pose, _)| triangulate::triangulation_angle(centre, &pose.centre(), &point))
                .fold(0.0, f64::max);
            return (angle >= min_angle).then_some(z);
        }
        rays.swap_remove(worst);
    }
    None
}

fn solve_depth(
    centre: &Point3<f64>,
    direction: &Vector3<f64>,
    rays: &[(Pose, Norm)],
) -> Option<f64> {
    // In each view the point is A + z·B. Requiring it to project onto the observation
    // gives two equations linear in z.
    let (mut num, mut den) = (0.0, 0.0);
    for (pose, n) in rays {
        let a = pose.to_camera(centre).coords;
        let b = pose.rotation * direction;
        for (u, ai, bi) in [(n.x(), a.x, b.x), (n.y(), a.y, b.y)] {
            let alpha = u * a.z - ai;
            let beta = u * b.z - bi;
            num += alpha * beta;
            den += beta * beta;
        }
    }
    if den <= 1e-18 {
        return None;
    }
    let mut z = -num / den;

    // Then Gauss–Newton on the reprojection error itself.
    for _ in 0..5 {
        let (mut g, mut h) = (0.0, 0.0);
        for (pose, n) in rays {
            let c = pose.to_camera(&(centre + direction * z));
            if c.z <= 1e-9 {
                continue;
            }
            let b = pose.rotation * direction;
            let (px, py) = (c.x / c.z, c.y / c.z);
            let (jx, jy) = ((b.x - px * b.z) / c.z, (b.y - py * b.z) / c.z);
            g += jx * (px - n.x()) + jy * (py - n.y());
            h += jx * jx + jy * jy;
        }
        if h <= 1e-18 {
            break;
        }
        let step = g / h;
        z -= step;
        if step.abs() <= 1e-9 * z.abs() {
            break;
        }
    }
    z.is_finite().then_some(z)
}

/// Keeps a depth only if enough other maps see the same surface at the same place.
///
/// The reference sample is lifted into the world, projected into the other view, and
/// that view's depth there lifted back and projected into the reference. The two must
/// land within the pixel tolerance, at nearly the same depth. A mismatch means one of the
/// maps is wrong there, or that the point is occluded in the other view.
fn consistency_filter(
    maps: &[DepthMap],
    cameras: &[Option<Pose>],
    intrinsics: &Intrinsics,
    precision: f64,
    params: &Params,
) -> Vec<DepthMap> {
    let tolerance_px = params.max_reprojection * precision;
    maps.iter()
        .map(|map| {
            let pose = cameras[map.view.index()].expect("depth maps are of registered views");
            let to_world = pose.inverse();
            let mut filtered = map.clone();
            for row in 0..map.rows {
                for column in 0..map.columns {
                    let Some(d) = map.get(column, row) else {
                        continue;
                    };
                    let d = d as f64;
                    let p = map.pixel(column, row);
                    let n = intrinsics.normalize(p);
                    let world = to_world.to_camera(&Point3::new(n.x() * d, n.y() * d, d));

                    let agreeing = maps
                        .iter()
                        .filter(|other| other.view != map.view)
                        .filter(|other| {
                            let other_pose = cameras[other.view.index()]
                                .expect("depth maps are of registered views");
                            let c = other_pose.to_camera(&world);
                            if c.z <= 0.0 {
                                return false;
                            }
                            let q = intrinsics.denormalize(Norm::new(c.x / c.z, c.y / c.z));
                            let Some(ds) = other.sample(q) else {
                                return false;
                            };
                            let ds = ds as f64;
                            let nq = intrinsics.normalize(q);
                            let back = other_pose.inverse().to_camera(&Point3::new(
                                nq.x() * ds,
                                nq.y() * ds,
                                ds,
                            ));
                            let in_reference = pose.to_camera(&back);
                            if in_reference.z <= 0.0 {
                                return false;
                            }
                            let reprojected = intrinsics.denormalize(Norm::new(
                                in_reference.x / in_reference.z,
                                in_reference.y / in_reference.z,
                            ));
                            let pixels = (reprojected.0 - p.0).norm() as f64;
                            let difference = ((in_reference.z - d) / d).abs();
                            pixels <= tolerance_px && difference <= params.max_depth_difference
                        })
                        .count();
                    if agreeing < params.min_consistent_views {
                        filtered.depth[row * map.columns + column] = f32::NAN;
                    }
                }
            }
            filtered
        })
        .collect()
}

/// Whether two neighbouring depths belong to the same surface.
fn continuous(a: f32, b: f32, step: f64) -> bool {
    ((a - b).abs() as f64) <= step * a.max(b) as f64
}

/// Removes connected patches smaller than `params.min_component` samples.
fn remove_speckles(mut map: DepthMap, params: &Params) -> DepthMap {
    let (columns, rows) = (map.columns, map.rows);
    let mut visited = vec![false; map.depth.len()];
    let mut component = Vec::new();
    let mut stack = Vec::new();
    for start in 0..map.depth.len() {
        if visited[start] || !map.depth[start].is_finite() {
            continue;
        }
        visited[start] = true;
        component.clear();
        stack.push(start);
        while let Some(i) = stack.pop() {
            component.push(i);
            let (column, row) = (i % columns, i / columns);
            let neighbours = [
                (column > 0).then(|| i - 1),
                (column + 1 < columns).then(|| i + 1),
                (row > 0).then(|| i - columns),
                (row + 1 < rows).then(|| i + columns),
            ];
            for j in neighbours.into_iter().flatten() {
                if !visited[j]
                    && map.depth[j].is_finite()
                    && continuous(map.depth[i], map.depth[j], params.max_depth_step)
                {
                    visited[j] = true;
                    stack.push(j);
                }
            }
        }
        if component.len() < params.min_component {
            for &i in &component {
                map.depth[i] = f32::NAN;
            }
        }
    }
    map
}

/// A 3 × 3 median over the neighbours on the same surface as each sample.
fn smooth(map: DepthMap, params: &Params) -> DepthMap {
    let mut smoothed = map.clone();
    let mut window = Vec::with_capacity(9);
    for row in 0..map.rows {
        for column in 0..map.columns {
            let Some(d) = map.get(column, row) else {
                continue;
            };
            window.clear();
            for r in row.saturating_sub(1)..(row + 2).min(map.rows) {
                for c in column.saturating_sub(1)..(column + 2).min(map.columns) {
                    if let Some(n) = map.get(c, r) {
                        if continuous(d, n, params.max_depth_step) {
                            window.push(n);
                        }
                    }
                }
            }
            window.sort_by(f32::total_cmp);
            smoothed.depth[row * map.columns + column] = window[window.len() / 2];
        }
    }
    smoothed
}

#[cfg(test)]
mod tests {
    use super::*;

    fn map(depth: Vec<f32>, columns: usize) -> DepthMap {
        DepthMap {
            view: ViewId(0),
            columns,
            rows: depth.len() / columns,
            stride: 4,
            depth,
        }
    }

    #[test]
    fn samples_the_nearest_grid_point() {
        let m = map(vec![1.0, 2.0, 3.0, f32::NAN], 2);
        assert_eq!(m.sample(PhotoPx::new(1.9, 0.0)), Some(1.0));
        assert_eq!(m.sample(PhotoPx::new(2.1, 0.0)), Some(2.0));
        assert_eq!(m.sample(PhotoPx::new(0.0, 4.0)), Some(3.0));
        assert_eq!(m.sample(PhotoPx::new(4.0, 4.0)), None);
        assert_eq!(m.sample(PhotoPx::new(8.0, 0.0)), None);
        assert_eq!(m.valid(), 3);
    }

    #[test]
    fn removes_small_patches_and_keeps_edges_sharp() {
        let params = Params {
            min_component: 5,
            ..Params::default()
        };
        // A 3 × 4 surface at depth 1 next to a 1 × 4 strip at depth 2: the strip is too
        // small and goes, the surface stays.
        let mut depth = Vec::new();
        for _ in 0..4 {
            depth.extend([1.0, 1.0, 1.0, 2.0]);
        }
        let cleaned = remove_speckles(map(depth, 4), &params);
        assert_eq!(cleaned.valid(), 12);
        assert!(cleaned.get(3, 0).is_none());

        let mut step = Vec::new();
        for _ in 0..3 {
            step.extend([1.0, 1.0, 3.0, 3.0]);
        }
        let smoothed = smooth(map(step.clone(), 4), &params);
        assert_eq!(
            smoothed.depth, step,
            "smoothing must not blur across the edge"
        );
    }

    #[test]
    fn solves_depth_along_a_ray() {
        let target = Point3::new(0.3, -0.2, 4.0);
        let reference = Pose::identity();
        let others = [
            Pose::look_at(&Point3::new(1.0, 0.0, 0.0), &target, &Vector3::y()),
            Pose::look_at(&Point3::new(-0.8, 0.5, 0.2), &target, &Vector3::y()),
        ];
        let mut rays: Vec<(Pose, Norm)> = others
            .iter()
            .map(|pose| (*pose, pose.project(&World(target)).unwrap()))
            .collect();
        // A third view that lies.
        rays.push((others[0], Norm::new(0.4, 0.4)));

        let n = reference.project(&World(target)).unwrap();
        let direction = Vector3::new(n.x(), n.y(), 1.0);
        let z = depth_along_ray(
            &reference.centre(),
            &direction,
            &mut rays,
            1e-3,
            1f64.to_radians(),
        )
        .expect("the honest views agree");
        assert!((z - 4.0).abs() < 1e-6, "depth {z}");
        assert_eq!(rays.len(), 2, "the lying view was dropped");
    }
}
