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
use crate::event::{report, silent, Event, Flow, Stage};
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
    /// The largest relative depth difference the cross-view check accepts between two
    /// precise depths.
    pub max_depth_difference: f64,
    /// Less precise depths may differ by this many times their combined expected error,
    /// when that allows more than `max_depth_difference`. A depth from a single narrow pair
    /// is far less precise than one from several wide ones, and holding both to the same
    /// absolute tolerance throws away most of the parts only two photos see.
    pub depth_agreement_sigmas: f64,
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
            max_reprojection: 10.0,
            min_angle_deg: 7.0,
            max_depth_difference: 0.5,
            depth_agreement_sigmas: 5.0,
            min_consistent_views: 1,
            max_depth_step: 0.2,
            min_component: 64,
            smooth: true,
        }
    }
}

/// Depths on a regular grid over one view's photo.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
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
    /// The expected relative error of each depth, from the precision of the mappings and
    /// the angle at which the rays met. Meaningless where the depth is unknown.
    pub uncertainty: Vec<f32>,
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
        let (column, row) = self.nearest(p)?;
        self.get(column, row)
    }

    /// The depth at grid sample `(column, row)` and its expected relative error, if known.
    pub fn get_with_uncertainty(&self, column: usize, row: usize) -> Option<(f32, f32)> {
        let i = row * self.columns + column;
        let d = self.depth[i];
        d.is_finite().then(|| (d, self.uncertainty[i]))
    }

    /// [`Self::sample`], with the depth's expected relative error.
    pub fn sample_with_uncertainty(&self, p: PhotoPx) -> Option<(f32, f32)> {
        let (column, row) = self.nearest(p)?;
        self.get_with_uncertainty(column, row)
    }

    /// The index into [`Self::depth`] of the sample nearest photo position `p`, if `p` is
    /// over the grid.
    pub fn index_of(&self, p: PhotoPx) -> Option<usize> {
        self.nearest(p)
            .map(|(column, row)| row * self.columns + column)
    }

    /// The grid sample nearest photo position `p`, if `p` is over the grid.
    fn nearest(&self, p: PhotoPx) -> Option<(usize, usize)> {
        let column = (p.x() / self.stride as f32).round();
        let row = (p.y() / self.stride as f32).round();
        if column < 0.0 || row < 0.0 {
            return None;
        }
        let (column, row) = (column as usize, row as usize);
        (column < self.columns && row < self.rows).then_some((column, row))
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

/// Where one view's depth samples went, split by whether a sample mapped into one other
/// view or into several.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct DepthStats {
    /// The view.
    pub view: ViewId,
    /// Samples no other view's mapping reaches. Nothing can give these a depth.
    pub unmapped: usize,
    /// Samples mapped into exactly one other view.
    pub single: Counts,
    /// Samples mapped into two or more other views.
    pub multiple: Counts,
    /// What became of each sample, row by row over the view's depth-map grid.
    pub fates: Vec<Fate>,
    /// How many other views each sample mapped into, row by row.
    pub matched: Vec<u8>,
}

/// What became of one depth sample.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Fate {
    /// No other view's mapping reached it.
    Unmapped,
    /// The views did not fit one depth.
    RejectedByFit,
    /// The views fit, but their rays met at too narrow an angle to measure depth.
    RejectedByAngle,
    /// No other depth map agreed.
    RejectedByConsistency,
    /// Part of a small disconnected patch.
    RemovedAsSpeckle,
    /// It has a depth.
    Kept,
}

/// What became of a group of depth samples.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[non_exhaustive]
pub struct Counts {
    /// How many samples.
    pub samples: usize,
    /// Dropped because the views did not fit one depth.
    pub rejected_by_fit: usize,
    /// Dropped because the rays met at too narrow an angle.
    pub rejected_by_angle: usize,
    /// Dropped because no other depth map agreed.
    pub rejected_by_consistency: usize,
    /// Dropped as part of a small disconnected patch.
    pub removed_as_speckle: usize,
    /// Kept.
    pub kept: usize,
}

/// A depth map for every registered view in `cameras`, over photos of `size`.
pub fn estimate<L: PairLookup>(
    graph: &PairGraph<L>,
    cameras: &[Option<Pose>],
    intrinsics: &Intrinsics,
    size: (usize, usize),
    params: &Params,
) -> Vec<DepthMap> {
    estimate_with_stats(graph, cameras, intrinsics, size, params).0
}

/// [`estimate`], with an account of where each view's samples were lost.
pub fn estimate_with_stats<L: PairLookup>(
    graph: &PairGraph<L>,
    cameras: &[Option<Pose>],
    intrinsics: &Intrinsics,
    size: (usize, usize),
    params: &Params,
) -> (Vec<DepthMap>, Vec<DepthStats>) {
    // `silent` never breaks, so the only error this form could return cannot happen.
    estimate_with_progress(graph, cameras, intrinsics, size, params, &mut silent)
        .expect("a callback that never breaks cannot cancel")
}

/// [`estimate_with_stats`], reporting each view through `on_event`.
///
/// Depth is the first stage whose cost grows with the number of views squared, so it
/// reports as each view is solved, cross-checked and cleaned rather than only at the end.
///
/// # Errors
/// [`Error::Cancelled`] if `on_event` asks the run to stop.
pub fn estimate_with_progress<L: PairLookup>(
    graph: &PairGraph<L>,
    cameras: &[Option<Pose>],
    intrinsics: &Intrinsics,
    (width, height): (usize, usize),
    params: &Params,
    on_event: &mut dyn FnMut(Event) -> Flow,
) -> Result<(Vec<DepthMap>, Vec<DepthStats>), Error> {
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

    let mut results: Vec<(DepthMap, Vec<u8>, Vec<bool>)> = Vec::with_capacity(registered.len());
    for (index, &(view, pose)) in registered.iter().enumerate() {
        let to_world = pose.rotation.inverse();
        let centre = pose.centre();
        let mut depth = vec![f32::NAN; columns * rows];
        let mut uncertainty = vec![f32::NAN; columns * rows];
        let mut narrow = vec![false; columns * rows];
        let mut mapped = vec![0u8; columns * rows];
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
                let i = row * columns + column;
                mapped[i] = rays.len().min(u8::MAX as usize) as u8;
                let n = intrinsics.normalize(p);
                let direction = to_world * Vector3::new(n.x(), n.y(), 1.0);
                let fit = depth_along_ray(&centre, &direction, &mut rays, threshold, min_angle);
                narrow[i] = fit == Err(Rejection::Angle);
                if let Ok((z, angle)) = fit {
                    depth[i] = z as f32;
                    // The error grows as the rays meet more obliquely, and shrinks with
                    // more of them.
                    let count = rays.len().max(1) as f64;
                    uncertainty[i] = (precision / (focal * angle.sin() * count.sqrt())) as f32;
                }
            }
        }
        let map = DepthMap {
            view,
            columns,
            rows,
            stride,
            depth,
            uncertainty,
        };
        let (valid, total) = (map.valid(), map.depth.len());
        results.push((map, mapped, narrow));
        // Solving every view is the bulk of the stage; cross-checking and cleaning
        // share what is left.
        report(
            on_event,
            Stage::Depth,
            0.6 * (index + 1) as f32 / registered.len() as f32,
            format!("{view}: {valid} of {total} samples have a depth"),
        )?;
    }
    let mut raw = Vec::with_capacity(results.len());
    let mut mapped = Vec::with_capacity(results.len());
    let mut narrow = Vec::with_capacity(results.len());
    for (map, counts, angles) in results {
        raw.push(map);
        mapped.push(counts);
        narrow.push(angles);
    }

    let consistent = consistency_filter(&raw, cameras, intrinsics, precision, params, on_event)?;
    let view_count = raw.len();
    let mut maps = Vec::with_capacity(raw.len());
    let mut stats = Vec::with_capacity(raw.len());
    for (index, (((raw, consistent), mapped), narrow)) in raw
        .iter()
        .zip(consistent)
        .zip(&mapped)
        .zip(&narrow)
        .enumerate()
    {
        let cleaned = remove_speckles(consistent.clone(), params);
        let mut view_stats = DepthStats {
            view: raw.view,
            unmapped: 0,
            single: Counts::default(),
            multiple: Counts::default(),
            fates: Vec::with_capacity(mapped.len()),
            matched: mapped.clone(),
        };
        for (i, &rays) in mapped.iter().enumerate() {
            let fate = if rays == 0 {
                Fate::Unmapped
            } else if !raw.depth[i].is_finite() && narrow[i] {
                Fate::RejectedByAngle
            } else if !raw.depth[i].is_finite() {
                Fate::RejectedByFit
            } else if !consistent.depth[i].is_finite() {
                Fate::RejectedByConsistency
            } else if !cleaned.depth[i].is_finite() {
                Fate::RemovedAsSpeckle
            } else {
                Fate::Kept
            };
            view_stats.fates.push(fate);
            let counts = match rays {
                0 => {
                    view_stats.unmapped += 1;
                    continue;
                }
                1 => &mut view_stats.single,
                _ => &mut view_stats.multiple,
            };
            counts.samples += 1;
            match fate {
                Fate::RejectedByFit => counts.rejected_by_fit += 1,
                Fate::RejectedByAngle => counts.rejected_by_angle += 1,
                Fate::RejectedByConsistency => counts.rejected_by_consistency += 1,
                Fate::RemovedAsSpeckle => counts.removed_as_speckle += 1,
                Fate::Kept => counts.kept += 1,
                Fate::Unmapped => {}
            }
        }
        stats.push(view_stats);
        maps.push(if params.smooth {
            smooth(cleaned, params)
        } else {
            cleaned
        });
        report(
            on_event,
            Stage::Depth,
            0.9 + 0.1 * (index + 1) as f32 / view_count as f32,
            format!("{}: speckles removed", raw.view),
        )?;
    }
    Ok((maps, stats))
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
/// Returns the depth and the widest angle between the reference ray and another one.
/// Why a sample's rays gave it no depth.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum Rejection {
    /// No depth fits the views.
    Fit,
    /// The rays fit, but meet too narrowly.
    Angle,
}

fn depth_along_ray(
    centre: &Point3<f64>,
    direction: &Vector3<f64>,
    rays: &mut Vec<(Pose, Norm)>,
    threshold: f64,
    min_angle: f64,
) -> Result<(f64, f64), Rejection> {
    while !rays.is_empty() {
        let z = solve_depth(centre, direction, rays).ok_or(Rejection::Fit)?;
        if z <= 0.0 {
            return Err(Rejection::Fit);
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
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .ok_or(Rejection::Fit)?;
        if error <= threshold {
            let angle = rays
                .iter()
                .map(|(pose, _)| triangulate::triangulation_angle(centre, &pose.centre(), &point))
                .fold(0.0, f64::max);
            return if angle >= min_angle {
                Ok((z, angle))
            } else {
                Err(Rejection::Angle)
            };
        }
        rays.swap_remove(worst);
    }
    Err(Rejection::Fit)
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
    on_event: &mut dyn FnMut(Event) -> Flow,
) -> Result<Vec<DepthMap>, Error> {
    let tolerance_px = params.max_reprojection * precision;
    let mut filtered_maps = Vec::with_capacity(maps.len());
    for (index, map) in maps.iter().enumerate() {
        let pose = cameras[map.view.index()].expect("depth maps are of registered views");
        let to_world = pose.inverse();
        let mut filtered = map.clone();
        for row in 0..map.rows {
            for column in 0..map.columns {
                let Some((d, sigma)) = map.get_with_uncertainty(column, row) else {
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
                        let Some((ds, other_sigma)) = other.sample_with_uncertainty(q) else {
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
                        let allowed = params.max_depth_difference.max(
                            params.depth_agreement_sigmas
                                * (sigma as f64).hypot(other_sigma as f64),
                        );
                        // Lifting the other view's depth back into this one moves it only along the
                        // epipolar line, so the pixel distance measures the same disagreement
                        // as the depth difference. Relax it by the same factor.
                        let relaxation = allowed / params.max_depth_difference;
                        pixels <= tolerance_px * relaxation && difference <= allowed
                    })
                    .count();
                if agreeing < params.min_consistent_views {
                    filtered.depth[row * map.columns + column] = f32::NAN;
                }
            }
        }
        filtered_maps.push(filtered);
        report(
            on_event,
            Stage::Depth,
            0.6 + 0.3 * (index + 1) as f32 / maps.len() as f32,
            format!("{}: cross-checked against the other views", map.view),
        )?;
    }
    Ok(filtered_maps)
}

/// Whether two neighbouring depths, each with its expected relative error, belong to the
/// same surface: they differ by no more than `step`, or by three times their combined
/// error if that is larger.
fn continuous(a: (f32, f32), b: (f32, f32), step: f64) -> bool {
    let allowed = step.max(3.0 * (a.1 + b.1) as f64);
    ((a.0 - b.0).abs() as f64) <= allowed * a.0.max(b.0) as f64
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
                    && continuous(
                        (map.depth[i], map.uncertainty[i]),
                        (map.depth[j], map.uncertainty[j]),
                        params.max_depth_step,
                    )
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
            let Some(d) = map.get_with_uncertainty(column, row) else {
                continue;
            };
            window.clear();
            for r in row.saturating_sub(1)..(row + 2).min(map.rows) {
                for c in column.saturating_sub(1)..(column + 2).min(map.columns) {
                    if let Some(n) = map.get_with_uncertainty(c, r) {
                        if continuous(d, n, params.max_depth_step) {
                            window.push(n.0);
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
            uncertainty: vec![0.001; depth.len()],
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
        let (z, _angle) = depth_along_ray(
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
