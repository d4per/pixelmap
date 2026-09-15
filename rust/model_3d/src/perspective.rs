//! Lifting triangulated correspondences into a normalized, textured surface.

use pixelmap::DensePhotoMap;

use crate::two_view::{Estimate, PointPair};
use crate::{quantile, texture_coordinates, TexturePoint, Viewpoint};

/// Depths further than this many (robust) standard deviations from the median, in log
/// depth, are dropped before filtering. Loose on purpose: a scene with a near object in
/// front of a far wall is legitimately bimodal, and the median filter deals with
/// isolated spikes anyway. This only catches the wild ones, such as a wrong match that
/// happened to lie on its epipolar line.
const DEPTH_OUTLIER_SIGMAS: f64 = 8.0;
/// The median filter keeps a cell only if at least this many of its eight neighbours
/// have a depth too; fewer, and the cell is an island of noise.
const MIN_NEIGHBOURS: usize = 3;
/// The image-plane extent used to normalise the model ignores this fraction of points on
/// either side, so that a few stray far-away points cannot shrink the rest to a speck.
const EXTENT_PERCENTILE: f64 = 0.02;

/// A reconstructed surface, ready to be held by a [`crate::Model3D`].
pub(crate) struct Surface {
    pub grid: Vec<TexturePoint>,
    /// Per cell, the model-space distance between neighbouring grid nodes that a surface
    /// facing the camera would have at that depth. Used to tell a steep surface from a
    /// depth discontinuity when meshing.
    pub spacing: Vec<f32>,
    /// Where camera 1 ended up in model space.
    pub viewpoint: Viewpoint,
}

/// Triangulates every kept correspondence, cleans the depth grid, and maps the result
/// into the model's coordinate frame. `None` if fewer than three points survive or they
/// have no extent.
pub(crate) fn build(
    map: &DensePhotoMap,
    cells: &[(usize, usize)],
    pairs: &[PointPair],
    estimate: &Estimate,
) -> Option<Surface> {
    let (grid_width, grid_height) = map.grid_dimensions();
    let cell = map.grid_cell_size() as f64;
    let geometry = &estimate.geometry;

    // Depth is filtered in log space: matching noise in pixels becomes depth noise that
    // grows with the square of the depth, and the log evens that out across the scene.
    let mut log_depth = vec![f64::NAN; grid_width * grid_height];
    for ((&(gx, gy), pair), _) in cells
        .iter()
        .zip(pairs)
        .zip(&estimate.keep)
        .filter(|(_, &keep)| keep)
    {
        if let Some([_, _, z]) = geometry.triangulate(pair.x1, pair.y1, pair.x2, pair.y2) {
            log_depth[gy * grid_width + gx] = z.ln();
        }
    }
    reject_outliers(&mut log_depth);
    let log_depth = median_filter(&log_depth, grid_width, grid_height);

    // Camera 1's frame has y pointing down the image and z into the scene; X3D is
    // right-handed with y up and the viewer looking down -z. Flipping both gives a model
    // that, seen from the default direction, looks like photo 1.
    let focal = geometry.focal_lengths[0];
    let [cx, cy] = geometry.principal_point;
    let points: Vec<Option<[f64; 3]>> = log_depth
        .iter()
        .enumerate()
        .map(|(i, &lz)| {
            if lz.is_nan() {
                return None;
            }
            let z = lz.exp();
            let px = (i % grid_width) as f64 * cell;
            let py = (i / grid_width) as f64 * cell;
            Some([(px - cx) / focal * z, -(py - cy) / focal * z, -z])
        })
        .collect();

    let range = |axis: usize| {
        let mut values: Vec<f64> = points.iter().flatten().map(|p| p[axis]).collect();
        if values.len() < 3 {
            return None;
        }
        Some((
            quantile(&mut values, EXTENT_PERCENTILE),
            quantile(&mut values, 1.0 - EXTENT_PERCENTILE),
        ))
    };
    let (x_lo, x_hi) = range(0)?;
    let (y_lo, y_hi) = range(1)?;
    let (z_lo, z_hi) = range(2)?;

    // One scale for all three axes, set by the longer image-plane extent: the shape is
    // kept as triangulated, only its size and position are normalised.
    let extent = (x_hi - x_lo).max(y_hi - y_lo);
    if !(extent.is_finite() && extent > 0.0) {
        return None;
    }
    let scale = 2.0 / extent;
    let centre = [
        (x_lo + x_hi) / 2.0,
        (y_lo + y_hi) / 2.0,
        (z_lo + z_hi) / 2.0,
    ];

    let mut grid = vec![TexturePoint::default(); grid_width * grid_height];
    let mut spacing = vec![0.0f32; grid_width * grid_height];
    for (i, point) in points.iter().enumerate() {
        let Some(p) = point else {
            continue;
        };
        let (gx, gy) = (i % grid_width, i / grid_width);
        let (u, v) = texture_coordinates(map, gx, gy);
        grid[i] = TexturePoint {
            x: ((p[0] - centre[0]) * scale) as f32,
            y: ((p[1] - centre[1]) * scale) as f32,
            z: ((p[2] - centre[2]) * scale) as f32,
            u,
            v,
            grid_x: gx,
            grid_y: gy,
        };
        spacing[i] = (scale * cell * -p[2] / focal) as f32;
    }

    // X3D measures the field of view across the smaller dimension of the viewport.
    let photo = map.photo1();
    let smaller = photo.width().min(photo.height()) as f64;
    let viewpoint = Viewpoint {
        position: centre.map(|c| (-c * scale) as f32),
        field_of_view: (2.0 * (smaller / 2.0 / focal).atan()) as f32,
    };

    Some(Surface {
        grid,
        spacing,
        viewpoint,
    })
}

/// Drops values far from the median, measured in median absolute deviations.
fn reject_outliers(values: &mut [f64]) {
    let mut valid: Vec<f64> = values.iter().copied().filter(|v| !v.is_nan()).collect();
    if valid.len() < 3 {
        return;
    }
    let median = quantile(&mut valid, 0.5);
    let mut deviations: Vec<f64> = valid.iter().map(|v| (v - median).abs()).collect();
    // 1.4826 turns a median absolute deviation into a standard deviation for normal noise.
    let sigma = quantile(&mut deviations, 0.5) * 1.4826;
    if sigma <= 0.0 {
        return;
    }
    for v in values.iter_mut() {
        if (*v - median).abs() > DEPTH_OUTLIER_SIGMAS * sigma {
            *v = f64::NAN;
        }
    }
}

/// A 3x3 median over the set cells of a grid. Cells with too few set neighbours are
/// cleared rather than filtered.
fn median_filter(values: &[f64], width: usize, height: usize) -> Vec<f64> {
    let mut out = vec![f64::NAN; values.len()];
    let mut window = Vec::with_capacity(9);
    for y in 0..height {
        for x in 0..width {
            if values[y * width + x].is_nan() {
                continue;
            }
            window.clear();
            for ny in y.saturating_sub(1)..=(y + 1).min(height - 1) {
                for nx in x.saturating_sub(1)..=(x + 1).min(width - 1) {
                    let v = values[ny * width + nx];
                    if !v.is_nan() {
                        window.push(v);
                    }
                }
            }
            // The window includes the cell itself.
            if window.len() > MIN_NEIGHBOURS {
                out[y * width + x] = quantile(&mut window, 0.5);
            }
        }
    }
    out
}
