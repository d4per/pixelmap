//! Relative depth under an affine camera model: the fallback for pairs whose perspective
//! geometry cannot be pinned down.

use nalgebra::{DMatrix, Matrix2, SVD};
use pixelmap::DensePhotoMap;

use crate::{texture_coordinates, TexturePoint};

/// How deep the object is assumed to be, as a fraction of its smaller image-plane
/// extent.
///
/// An affine camera fixes shape only up to an unknown depth scale, so this has to be
/// assumed rather than measured. Purely empirical: `0.3` is what looks right on test
/// pairs.
pub(crate) const Z_DEPTH_RATIO: f64 = 0.3;

/// A residual spread smaller than this fraction of the image-plane extent counts as no
/// depth signal at all.
///
/// Without it, a scene whose two views really are related by an affine warp would have
/// its floating-point rounding noise stretched across the full depth range.
const FLAT_EPSILON: f64 = 1e-6;

/// Reconstructs relative depth under an affine camera, returning the grid of points and
/// the half-depth-range factor the depth was scaled by.
///
/// Two of the three coordinates are already known: a grid cell `(x, y)` sits at
/// pixel `(x, y) * grid_cell_size` in photo 1. Those become the model's `x` and `y`
/// directly, under a single shared scale so the photo's aspect ratio survives, and
/// only the depth has to be solved for.
///
/// The process, repeated for **two iterations** so the second one re-fits without
/// the outliers the first one found:
/// 1. Collects the valid cells (those with non-NaN coordinates) and takes the
///    bounding box of their photo-1 positions.
/// 2. Centers `[x1, y1, x2, y2]` on its column means and least-squares fits the
///    affine part, `[x2, y2] ~ [x1, y1] * W`. Whatever that cannot explain is the
///    depth signal: for an affine camera the residual is `depth * b` for one fixed
///    2-vector `b`, so an SVD of the residuals recovers `b` as its dominant
///    direction and the per-point depth as the projection onto it.
/// 3. Picks the sign of that direction so the middle of the object is nearer to the
///    viewer than its edges, since the SVD's own sign is arbitrary. This is a guess, and
///    the reason this model is only a fallback: it turns concave scenes inside out.
/// 4. Rescales the depth to [-1, 1] over its 10th/90th percentiles, drops cells
///    that land far outside that (they are invalidated for the next iteration), and
///    multiplies by a fixed fraction (`Z_DEPTH_RATIO`) of the smaller image-plane
///    extent.
pub(crate) fn reconstruct(photo_mapping: &DensePhotoMap) -> (Vec<TexturePoint>, f32) {
    let mut photo_mapping = photo_mapping.clone();
    let (map_width, map_height) = photo_mapping.grid_dimensions();
    let mut grid = vec![TexturePoint::default(); map_width * map_height];
    let mut result_z_scale = 1.0f32;

    let grid_cell_size = photo_mapping.grid_cell_size();

    // Perform two passes of cleanup and depth solving.
    for _ in 0..2 {
        // Collect valid points (non-NaN) into a vector for the fit.
        let valid_points: Vec<_> = (0..map_height)
            .flat_map(|y| (0..map_width).map(move |x| (x, y)))
            .filter_map(|(x, y)| {
                let (data_x, data_y) = photo_mapping.grid_coordinates(x, y);
                if !data_x.is_nan() && !data_y.is_nan() {
                    Some((
                        (x * grid_cell_size) as f64,
                        (y * grid_cell_size) as f64,
                        data_x as f64,
                        data_y as f64,
                    ))
                } else {
                    None
                }
            })
            .collect();

        // The affine fit needs more points than it has parameters.
        if valid_points.len() < 3 {
            break;
        }

        // Bounding box of the photo-1 positions. This is the object's extent in the
        // image plane, and it sets both the image-plane scale and the depth scale.
        let (mut x_min, mut x_max) = (f64::INFINITY, f64::NEG_INFINITY);
        let (mut y_min, mut y_max) = (f64::INFINITY, f64::NEG_INFINITY);
        for p in &valid_points {
            x_min = x_min.min(p.0);
            x_max = x_max.max(p.0);
            y_min = y_min.min(p.1);
            y_max = y_max.max(p.1);
        }
        let x_extent = x_max - x_min;
        let y_extent = y_max - y_min;
        let x_center = (x_min + x_max) / 2.0;
        let y_center = (y_min + y_max) / 2.0;

        // One scale for both image-plane axes, so the aspect ratio is preserved:
        // the longer of the two spans [-1, 1] and the shorter comes out shorter.
        let plane_extent = x_extent.max(y_extent);
        if plane_extent <= 0.0 {
            break;
        }
        let plane_scale = 2.0 / plane_extent;

        // Half the depth range, in the same units as the scaled image plane.
        let z_scale = Z_DEPTH_RATIO * x_extent.min(y_extent) * plane_scale / 2.0;

        // Build a DMatrix from the valid points: each row is [X, Y, dataX, dataY],
        // then center the columns so the affine fit needs no intercept term.
        let mut matrix = DMatrix::from_fn(valid_points.len(), 4, |i, j| match j {
            0 => valid_points[i].0,
            1 => valid_points[i].1,
            2 => valid_points[i].2,
            3 => valid_points[i].3,
            _ => unreachable!(),
        });
        let col_means = compute_column_means(&matrix);
        center_data(&mut matrix, &col_means);

        // Normal equations for [x2, y2] ~ [X, Y] * W. Both sides are two columns
        // wide, so the whole least-squares solve is a 2x2 inverse.
        let a = matrix.columns(0, 2);
        let b = matrix.columns(2, 2);
        let ata = Matrix2::new(
            a.column(0).dot(&a.column(0)),
            a.column(0).dot(&a.column(1)),
            a.column(1).dot(&a.column(0)),
            a.column(1).dot(&a.column(1)),
        );
        let atb = Matrix2::new(
            a.column(0).dot(&b.column(0)),
            a.column(0).dot(&b.column(1)),
            a.column(1).dot(&b.column(0)),
            a.column(1).dot(&b.column(1)),
        );
        // Singular only if the valid cells are collinear, which leaves no surface
        // to reconstruct.
        let Some(ata_inv) = ata.try_inverse() else {
            break;
        };
        let w = ata_inv * atb;

        // What the affine part could not explain. This is the depth signal.
        let residuals = DMatrix::from_fn(matrix.nrows(), 2, |i, j| {
            matrix[(i, 2 + j)] - (matrix[(i, 0)] * w[(0, j)] + matrix[(i, 1)] * w[(1, j)])
        });

        // The residuals of an affine camera lie along one direction; recover it.
        let svd = SVD::new(residuals.clone(), false, true);
        let v_t = svd.v_t.expect("V^T matrix not found");
        let mut direction = [v_t[(0, 0)], v_t[(0, 1)]];

        let mut depths: Vec<f64> = (0..residuals.nrows())
            .map(|i| residuals[(i, 0)] * direction[0] + residuals[(i, 1)] * direction[1])
            .collect();

        // The SVD's sign is arbitrary, so the surface can come out as an inverted
        // bowl. Assume the middle of the object is nearer to the viewer than its
        // edges (X3D is right-handed, so nearer means larger z) and flip if not.
        if x_extent > 0.0 && y_extent > 0.0 {
            let (mut inner_sum, mut inner_count) = (0.0, 0usize);
            let (mut outer_sum, mut outer_count) = (0.0, 0usize);
            for (p, &depth) in valid_points.iter().zip(depths.iter()) {
                let u = (p.0 - x_center) / (x_extent / 2.0);
                let v = (p.1 - y_center) / (y_extent / 2.0);
                if (u * u + v * v).sqrt() < 0.5 {
                    inner_sum += depth;
                    inner_count += 1;
                } else {
                    outer_sum += depth;
                    outer_count += 1;
                }
            }
            if inner_count > 0
                && outer_count > 0
                && inner_sum / (inner_count as f64) < outer_sum / (outer_count as f64)
            {
                direction = [-direction[0], -direction[1]];
                for depth in &mut depths {
                    *depth = -*depth;
                }
            }
        }

        // Determine min/max using the 10th and 90th percentile of the depth.
        let depth_column = DMatrix::from_column_slice(depths.len(), 1, &depths);
        let (min_values, max_values) = compute_min_max(&depth_column, 0.1, 0.9);
        let (z_low, z_high) = (min_values[0], max_values[0]);
        // Two views related by a pure affine warp carry no depth at all; without
        // this the rounding noise would be stretched over the whole depth range.
        let depth_spread = z_high - z_low;
        let is_flat = !depth_spread.is_finite() || depth_spread <= FLAT_EPSILON * plane_extent;

        result_z_scale = z_scale as f32;

        // Now update every cell in the DensePhotoMap with a 3D coordinate, or
        // invalidate it.
        for y in 0..map_height {
            for x in 0..map_width {
                let (x2, y2) = photo_mapping.grid_coordinates(x, y);
                if x2.is_nan() || y2.is_nan() {
                    continue;
                }

                // This cell's photo-1 position, and both pairs centered the same
                // way the fit was.
                let plane_x = (x * grid_cell_size) as f64;
                let plane_y = (y * grid_cell_size) as f64;
                let a_x = plane_x - col_means[0];
                let a_y = plane_y - col_means[1];
                let b_x = x2 as f64 - col_means[2];
                let b_y = y2 as f64 - col_means[3];

                // Project this cell's residual onto the depth direction.
                let r_x = b_x - (a_x * w[(0, 0)] + a_y * w[(1, 0)]);
                let r_y = b_y - (a_x * w[(0, 1)] + a_y * w[(1, 1)]);
                let z_normalized = if is_flat {
                    0.0
                } else {
                    rescale_value(r_x * direction[0] + r_y * direction[1], z_low, z_high)
                };

                // Only the depth can be an outlier now; x and y are exact grid
                // positions.
                if z_normalized.is_finite() && z_normalized.abs() < 3.0 {
                    let (u, v) = texture_coordinates(&photo_mapping, x, y);
                    grid[y * map_width + x] = TexturePoint {
                        x: ((plane_x - x_center) * plane_scale) as f32,
                        // Grid rows count downwards from the top of photo 1, X3D's
                        // y-axis points up.
                        y: (-(plane_y - y_center) * plane_scale) as f32,
                        z: (z_normalized * z_scale) as f32,
                        u,
                        v,
                        grid_x: x,
                        grid_y: y,
                    };
                } else {
                    // Invalidate this cell for the next iteration, and drop any
                    // value the previous iteration left in it.
                    grid[y * map_width + x] = TexturePoint::default();
                    photo_mapping.set_grid_coordinates(x, y, f32::NAN, f32::NAN);
                }
            }
        }
    }

    (grid, result_z_scale)
}

/// Computes column-wise means of a DMatrix.
fn compute_column_means(matrix: &DMatrix<f64>) -> Vec<f64> {
    (0..matrix.ncols())
        .map(|col| matrix.column(col).iter().sum::<f64>() / matrix.nrows() as f64)
        .collect()
}

/// Subtracts the given column means from each value in `matrix`, centering each column
/// around 0.
fn center_data(matrix: &mut DMatrix<f64>, col_means: &[f64]) {
    for (col, mean) in col_means.iter().enumerate() {
        for value in matrix.column_mut(col).iter_mut() {
            *value -= mean;
        }
    }
}

/// Finds the values at the specified `lower_percentile` and `upper_percentile` for each
/// column in `reduced_data`, as `(min_values, max_values)`.
fn compute_min_max(
    reduced_data: &DMatrix<f64>,
    lower_percentile: f64,
    upper_percentile: f64,
) -> (Vec<f64>, Vec<f64>) {
    let mut min_values = Vec::with_capacity(reduced_data.ncols());
    let mut max_values = Vec::with_capacity(reduced_data.ncols());

    for col in 0..reduced_data.ncols() {
        let mut values: Vec<f64> = reduced_data.column(col).iter().copied().collect();
        values.sort_by(|a, b| a.total_cmp(b));

        let lower_idx = (values.len() as f64 * lower_percentile) as usize;
        let upper_idx = (values.len() as f64 * upper_percentile) as usize;

        min_values.push(values[lower_idx]);
        max_values.push(values[upper_idx]);
    }
    (min_values, max_values)
}

/// Maps `value` from the range [min_val, max_val] to [-1, 1].
///
/// If `max_val == min_val`, this may produce invalid output (`NaN`).
fn rescale_value(value: f64, min_val: f64, max_val: f64) -> f64 {
    ((value - min_val) / (max_val - min_val)) * 2.0 - 1.0
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use pixelmap::Photo;

    use super::*;
    use crate::{Model3D, Projection, Settings};

    const GRID_WIDTH: usize = 17;
    const GRID_HEIGHT: usize = 9;
    const CELL: f64 = 10.0; // photo width 160 / (GRID_WIDTH - 1)

    /// A blank photo of the size the grid constants above assume.
    fn photo() -> Arc<Photo> {
        Arc::new(Photo::from_rgba(160, 80, vec![0u8; 160 * 80 * 4]).unwrap())
    }

    fn affine_model(map: &DensePhotoMap) -> Model3D {
        let settings = Settings {
            projection: Projection::Affine,
            ..Settings::default()
        };
        Model3D::with_settings(map, &settings)
    }

    /// A fully populated map whose two views are related by an exact affine warp, plus
    /// whatever `bump` adds to the horizontal correspondence.
    fn mapping(bump: impl Fn(f64, f64) -> f64) -> DensePhotoMap {
        let mut map = DensePhotoMap::new(photo(), photo(), GRID_WIDTH, GRID_HEIGHT);
        assert_eq!(map.grid_cell_size(), CELL as usize);
        for y in 0..GRID_HEIGHT {
            for x in 0..GRID_WIDTH {
                let (px, py) = (x as f64 * CELL, y as f64 * CELL);
                // Integer coefficients so the affine part is exact in f32 and the
                // no-bump case really does have a zero residual.
                let x2 = 2.0 * px + py + 5.0 + bump(px, py);
                let y2 = px + 2.0 * py + 3.0;
                map.set_grid_coordinates(x, y, x2 as f32, y2 as f32);
            }
        }
        map
    }

    /// Normalised distance from the centre of the grid, 0 at the middle and 1 at the
    /// middle of an edge.
    fn radius(px: f64, py: f64) -> f64 {
        let u =
            (px - (GRID_WIDTH - 1) as f64 * CELL / 2.0) / ((GRID_WIDTH - 1) as f64 * CELL / 2.0);
        let v =
            (py - (GRID_HEIGHT - 1) as f64 * CELL / 2.0) / ((GRID_HEIGHT - 1) as f64 * CELL / 2.0);
        (u * u + v * v).sqrt()
    }

    fn valid_points(model: &Model3D) -> Vec<TexturePoint> {
        (0..model.grid_height)
            .flat_map(|y| (0..model.grid_width).map(move |x| (x, y)))
            .map(|(x, y)| *model.get_texture_point(x, y))
            .filter(|p| !p.x.is_nan())
            .collect()
    }

    #[test]
    fn image_plane_keeps_the_photo_aspect_ratio() {
        let model = affine_model(&mapping(|px, py| 8.0 * (1.0 - radius(px, py)).max(0.0)));
        assert_eq!(model.projection(), Projection::Affine);
        let points = valid_points(&model);
        assert_eq!(points.len(), GRID_WIDTH * GRID_HEIGHT);

        // The grid is 160 x 80 pixels, so the long axis spans [-1, 1] and the short one
        // spans exactly half that.
        let xs: Vec<f32> = points.iter().map(|p| p.x).collect();
        let ys: Vec<f32> = points.iter().map(|p| p.y).collect();
        let x_span = xs.iter().cloned().fold(f32::MIN, f32::max)
            - xs.iter().cloned().fold(f32::MAX, f32::min);
        let y_span = ys.iter().cloned().fold(f32::MIN, f32::max)
            - ys.iter().cloned().fold(f32::MAX, f32::min);
        assert!((x_span - 2.0).abs() < 1e-5, "x span was {x_span}");
        assert!((y_span - 1.0).abs() < 1e-5, "y span was {y_span}");

        // x follows the column index, and y runs the other way: grid row 0 is the top
        // of the photo, which is the top of the model too.
        assert!(model.get_texture_point(0, 0).x < model.get_texture_point(GRID_WIDTH - 1, 0).x);
        assert!(model.get_texture_point(0, 0).y > model.get_texture_point(0, GRID_HEIGHT - 1).y);

        // The texture still tracks the geometry: v = 1 at the top.
        assert!(model.get_texture_point(0, 0).v > model.get_texture_point(0, GRID_HEIGHT - 1).v);
    }

    #[test]
    fn a_pure_affine_warp_has_no_depth() {
        let model = affine_model(&mapping(|_, _| 0.0));
        let points = valid_points(&model);
        assert_eq!(points.len(), GRID_WIDTH * GRID_HEIGHT);
        for p in &points {
            assert_eq!(p.z, 0.0, "flat scene produced z = {}", p.z);
        }
    }

    #[test]
    fn depth_scales_with_the_smaller_image_plane_extent() {
        let model = affine_model(&mapping(|px, py| 8.0 * (1.0 - radius(px, py)).max(0.0)));
        let mut zs: Vec<f32> = valid_points(&model).iter().map(|p| p.z).collect();
        zs.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // The depth is normalised over its own 10th/90th percentiles, so that is the
        // band the ratio applies to, measured against the 1.0-wide short axis.
        let low = zs[(zs.len() as f64 * 0.1) as usize];
        let high = zs[(zs.len() as f64 * 0.9) as usize];
        let expected = Z_DEPTH_RATIO as f32 * 1.0;
        assert!(
            (high - low - expected).abs() < 1e-3,
            "10-90 depth span was {}, expected {expected}",
            high - low
        );
    }

    #[test]
    fn the_middle_of_the_object_faces_the_viewer() {
        let bump = |px: f64, py: f64| 8.0 * (1.0 - radius(px, py)).max(0.0);
        let centre = |model: &Model3D| model.get_texture_point(GRID_WIDTH / 2, GRID_HEIGHT / 2).z;
        let corner = |model: &Model3D| model.get_texture_point(0, 0).z;

        // Either sign of the bump has to come out the same way round, since the SVD
        // direction it produces differs only by its sign.
        for sign in [1.0, -1.0] {
            let model = affine_model(&mapping(|px, py| sign * bump(px, py)));
            assert!(
                centre(&model) > corner(&model),
                "sign {sign}: centre z {} was not in front of corner z {}",
                centre(&model),
                corner(&model)
            );
        }
    }
}
