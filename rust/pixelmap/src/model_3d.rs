use std::rc::Rc;
use crate::photo::Photo;
use crate::dense_photo_map::DensePhotoMap;
use nalgebra::{DMatrix, DVector, SVD};

/// A 3D model built from a 2D grid of correspondences. Each cell in the grid has a
/// position in 3D space (`x, y, z`) plus texture coordinates (`u, v`) mapping it
/// into an associated photo.
pub struct Model3D {
    /// The number of grid cells in the horizontal direction.
    pub grid_width: usize,

    /// The number of grid cells in the vertical direction.
    pub grid_height: usize,

    /// A reference-counted handle to the source `Photo` used for texturing.
    pub photo: Rc<Photo>,

    /// A flat storage of [`TexturePoint`]s, of length `grid_width * grid_height`,
    /// describing each cell's 3D location and texture coordinates.
    grid: Vec<TexturePoint>,
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

    /// Horizontal texture coordinate (U axis), typically in [0, 1].
    pub u: f32,
    /// Vertical texture coordinate (V axis), typically in [0, 1].
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
    /// Constructs a new `Model3D` from a given [`DensePhotoMap`].
    ///
    /// The process:
    /// 1. Copies the `DensePhotoMap` and initializes a `Model3D` with matching grid dimensions.
    /// 2. For **two iterations**:
    ///    - Collects valid points from `DensePhotoMap` (those with non-NaN coordinates).
    ///    - Creates a data matrix from `[X, Y, dataX, dataY]` per valid cell.
    ///    - Centers the matrix columns around their mean.
    ///    - Performs SVD on the data, reducing from 4D to 3D by selecting the first 3 principal components.
    ///    - Rescales the resulting 3D coordinates to [-1, 1] based on 10th/90th percentiles.
    ///    - Updates the `TexturePoint`s in `Model3D`. If a point is out of range or NaN, that cell is invalidated in the next iteration.
    ///
    /// # Returns
    /// A `Model3D` whose grid cells now store `(x, y, z)` in 3D space, plus `(u, v)` texture coordinates.
    pub fn new(photo_mapping: &DensePhotoMap) -> Self {
        let mut photo_mapping = photo_mapping.clone();

        let mut result = Model3D {
            grid_width: photo_mapping.grid_width,
            grid_height: photo_mapping.grid_height,
            photo: photo_mapping.photo1.clone(),
            grid: vec![
                TexturePoint::default();
                photo_mapping.grid_width * photo_mapping.grid_height
            ],
        };

        let grid_cell_size = photo_mapping.get_grid_cell_size();

        // Perform two passes of cleanup and SVD-based embedding.
        for cleanup_iteration in 0..2 {
            // Collect valid points (non-NaN) into a vector for SVD.
            let valid_points: Vec<_> = (0..photo_mapping.grid_height)
                .flat_map(|y| {
                    (0..photo_mapping.grid_width).filter_map({
                        let value = photo_mapping.clone();
                        move |x| {
                            let (data_x, data_y) = value.get_grid_coordinates(x, y);
                            if !data_x.is_nan() {
                                Some((
                                    (x * grid_cell_size) as f64,
                                    (y * grid_cell_size) as f64,
                                    data_x as f64,
                                    data_y as f64,
                                ))
                            } else {
                                None
                            }
                        }
                    })
                })
                .collect();

            println!(
                "valid points size: {} iteration {}",
                valid_points.len(),
                cleanup_iteration
            );

            // Build a DMatrix from the valid points: each row is [X, Y, dataX, dataY].
            let mut matrix = DMatrix::from_fn(valid_points.len(), 4, |i, j| match j {
                0 => valid_points[i].0,
                1 => valid_points[i].1,
                2 => valid_points[i].2,
                3 => valid_points[i].3,
                _ => unreachable!(),
            });

            // Compute column means, then center the data by subtracting these means.
            let col_means = compute_column_means(&matrix);
            center_data(&mut matrix, &col_means);

            // Perform SVD on the centered data.
            let svd = SVD::new(matrix.clone(), true, true);
            let v_t = svd.v_t.expect("V^T matrix not found");
            // Keep the first 3 principal components (rows).
            let v_reduced = v_t.rows(0, 3);

            // Project original data onto the reduced dimension (3D).
            let reduced_data = matrix * v_reduced.transpose();

            // Determine min/max using the 10th and 90th percentile in each of the 3 new dimensions.
            let (min_values, max_values) = compute_min_max(&reduced_data, 0.1, 0.9);

            // Now update every cell in the DensePhotoMap with a 3D coordinate, or invalidate it.
            for y in 0..photo_mapping.grid_height {
                for x in 0..photo_mapping.grid_width {
                    let (x2, y2) = photo_mapping.get_grid_coordinates(x, y);
                    if x2.is_nan() {
                        continue;
                    }

                    // Build a row for this cell [X, Y, dataX, dataY].
                    let mut point_row = DMatrix::from_row_slice(
                        1,
                        4,
                        &[(x * grid_cell_size) as f64, (y * grid_cell_size) as f64, x2 as f64, y2 as f64],
                    );

                    // Apply centering (subtract the same col_means).
                    for col in 0..4 {
                        point_row[(0, col)] -= col_means[col];
                    }

                    // Project into 3D space.
                    let data_3d = point_row * v_reduced.transpose();

                    // Rescale each dimension to [-1, 1].
                    let xx = rescale_value(data_3d[0], min_values[0], max_values[0]);
                    let yy = rescale_value(data_3d[1], min_values[1], max_values[1]);
                    let zz = rescale_value(data_3d[2], min_values[2], max_values[2]);

                    // If valid and within bounds, store the 3D point; otherwise, mark as invalid.
                    if !xx.is_nan()
                        && !yy.is_nan()
                        && !zz.is_nan()
                        && xx.abs() < 3.0
                        && yy.abs() < 3.0
                        && zz.abs() < 3.0
                    {
                        result.grid[y * photo_mapping.grid_width + x] = TexturePoint {
                            x: xx as f32,
                            y: yy as f32,
                            z: zz as f32,
                            u: x as f32 / photo_mapping.grid_width as f32,
                            v: 1f32 - (y as f32 / photo_mapping.grid_height as f32),
                            grid_x: x,
                            grid_y: y,
                        };
                    } else {
                        // Invalidate this cell for the next iteration
                        photo_mapping.set_grid_coordinates(x, y, f32::NAN, f32::NAN);
                    }
                }
            }
        }

        result
    }

    /// Retrieves a reference to the `TexturePoint` at grid cell `(x, y)`.
    ///
    /// # Panics
    /// If `(x, y)` is out of range, this will cause a panic due to a slice index out of bounds.
    pub fn get_texture_point(&self, x: usize, y: usize) -> &TexturePoint {
        &self.grid[y * self.grid_width + x]
    }

    /// Creates and returns an X3D string representing the 3D mesh of points.
    ///
    /// # Details
    /// - The `<IndexedFaceSet>` is built by iterating over each cell `(x, y)` and forming quads
    ///   (split into triangles) with the adjacent cells `(x+1, y)`, `(x, y+1)`, `(x+1, y+1)`.
    /// - Invalid points (with `NaN` coordinates) are skipped.
    /// - If any pair of points is too far apart (distance > 0.5), that face is skipped.
    /// - Texture coordinates and 3D positions are embedded in the X3D output.
    ///
    /// Replace `"[photo_placeholder]"` in the string with a real texture file URL if needed.
    pub fn get_X3D(&self) -> String {
        let mut result = String::new();
        result.push_str(
            r#"<X3D width="1000px" height="1000px">
    <head>
        <meta name='title' content='3D Model'/>
        <meta name='description' content='3D Model with texture'/>
    </head>
    <Scene>
        <Shape>
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
        for y in 0..self.grid_height - 1 {
            for x in 0..self.grid_width - 1 {
                let p1 = self.get_texture_point(x, y);
                let p2 = self.get_texture_point(x + 1, y);
                let p3 = self.get_texture_point(x, y + 1);
                let p4 = self.get_texture_point(x + 1, y + 1);

                // Skip faces if any point is invalid or too far from the others.
                if p1.x.is_nan() || p2.x.is_nan() || p3.x.is_nan() || p4.x.is_nan() {
                    continue;
                }
                if p1.distance(p2) > 0.5 || p1.distance(p3) > 0.5 || p1.distance(p4) > 0.5 {
                    continue;
                }

                // Construct two triangles (p1->p2->p4 and p1->p4->p3).
                // X3D uses -1 as a face separator.
                result.push_str(&format!(
                    "{} {} {} -1 {} {} {} -1 ",
                    y * self.grid_width + x,
                    y * self.grid_width + x + 1,
                    (y + 1) * self.grid_width + x + 1,
                    y * self.grid_width + x,
                    (y + 1) * self.grid_width + x + 1,
                    (y + 1) * self.grid_width + x
                ));
            }
        }

        result.push_str("'>\n<Coordinate point='");

        // Write out the 3D coordinates of every grid cell.
        for y in 0..self.grid_height {
            for x in 0..self.grid_width {
                let p = self.get_texture_point(x, y);
                if p.x.is_nan() {
                    result.push_str("0 0 0 ");
                } else {
                    result.push_str(&format!("{} {} {} ", p.x, p.y, p.z));
                }
            }
        }

        result.push_str("'></Coordinate>\n<TextureCoordinate point='");

        // Write out the (u, v) texture coordinates.
        for y in 0..self.grid_height {
            for x in 0..self.grid_width {
                let p = self.get_texture_point(x, y);
                if p.x.is_nan() {
                    result.push_str("0 0 ");
                } else {
                    result.push_str(&format!("{} {} ", p.u, p.v));
                }
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

/// Computes column-wise means of a DMatrix.
///
/// # Returns
/// A `Vec<f64>` of length `matrix.ncols()`, where each entry is
/// the average of that column's values.
fn compute_column_means(matrix: &DMatrix<f64>) -> Vec<f64> {
    let mut means = Vec::with_capacity(matrix.ncols());
    for col in 0..matrix.ncols() {
        let sum: f64 = matrix.column(col).iter().sum();
        means.push(sum / matrix.nrows() as f64);
    }
    means
}

/// Subtracts the given column means from each value in `matrix`,
/// effectively centering each column around 0.
fn center_data(matrix: &mut DMatrix<f64>, col_means: &[f64]) {
    for col in 0..matrix.ncols() {
        for row in 0..matrix.nrows() {
            matrix[(row, col)] -= col_means[col];
        }
    }
}

/// Finds the values at the specified `lower_percentile` and `upper_percentile`
/// for each column in `reduced_data`.
///
/// # Returns
/// A tuple `(min_values, max_values)`, each a `Vec<f64>` of length `ncols`.
/// - `min_values[i]` is the `lower_percentile`-quantile in column `i`.
/// - `max_values[i]` is the `upper_percentile`-quantile in column `i`.
fn compute_min_max(
    reduced_data: &DMatrix<f64>,
    lower_percentile: f64,
    upper_percentile: f64,
) -> (Vec<f64>, Vec<f64>) {
    let mut min_values = Vec::with_capacity(reduced_data.ncols());
    let mut max_values = Vec::with_capacity(reduced_data.ncols());

    for col in 0..reduced_data.ncols() {
        let mut values: Vec<f64> = reduced_data.column(col).iter().copied().collect();
        values.sort_by(|a, b| a.partial_cmp(b).unwrap());

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
