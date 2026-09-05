//! The dense correspondence grid produced by a run, and the operations over it.

use crate::error::{DecodeError, Error};
use crate::photo::Photo;
use std::sync::Arc;

/// Marks a byte stream as a serialized mapping, so that arbitrary data is rejected
/// before any of it is interpreted as a grid dimension.
const MAGIC: &[u8; 4] = b"PXMP";

/// The encoding this build writes and is willing to read.
const FORMAT_VERSION: u16 = 1;

/// Magic, version, then `grid_width`, `grid_height` and `grid_cell_size` as `u64`s.
const HEADER_LEN: usize = MAGIC.len() + 2 + 3 * 8;

/// Represents a dense 2D mapping between two photos (`photo1` and `photo2`).
///
/// The map is stored in a grid of size `grid_width` × `grid_height`, with each cell
/// containing two floats describing how a point in `photo1` maps into coordinates
/// for `photo2`. This allows tasks like warp transformations, morphing, or alignment
/// between two images.
///
/// The fields are not public. `map_data.len() == grid_width * grid_height * 2` is an
/// invariant every read relies on, so letting a caller assign a new `grid_width` would
/// turn later lookups into out-of-bounds panics. Read them through
/// [`Self::grid_dimensions`], [`Self::grid_cell_size`], [`Self::photo1`] and
/// [`Self::photo2`] instead.
#[derive(Clone)]
pub struct DensePhotoMap {
    /// Reference-counted handle to the first photo.
    pub(crate) photo1: Arc<Photo>,

    /// Reference-counted handle to the second photo.
    pub(crate) photo2: Arc<Photo>,

    /// The number of columns in the mapping grid.
    pub(crate) grid_width: usize,

    /// The number of rows in the mapping grid.
    pub(crate) grid_height: usize,

    /// Internal storage for the mapping data, of length `grid_width * grid_height * 2`.
    /// Each cell stores (x2, y2) in consecutive slots. If a cell is empty, it holds `NaN`.
    map_data: Vec<f32>,

    /// The size (in pixels) each grid cell spans in `photo1`, computed during creation.
    grid_cell_size: usize,
}

impl std::fmt::Debug for DensePhotoMap {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DensePhotoMap")
            .field("grid_width", &self.grid_width)
            .field("grid_height", &self.grid_height)
            .field("grid_cell_size", &self.grid_cell_size)
            .field("coverage", &self.calculate_used_area())
            .finish_non_exhaustive()
    }
}

impl DensePhotoMap {
    /// Creates a new `DensePhotoMap` for the given photos and grid dimensions.
    ///
    /// # Parameters
    /// - `photo1`, `photo2`: Reference-counted handles to the two source photos.
    /// - `grid_width`, `grid_height`: How many columns and rows the map will have.
    ///
    /// # Returns
    /// A `DensePhotoMap` where all cells are initialized to `NaN`. The
    /// `grid_cell_size` is calculated from `photo1`’s width and `grid_width`.
    ///
    /// # Panics
    /// Panics if `grid_width < 2`. The cell size is `photo1.width / (grid_width - 1)`,
    /// which divides by zero at one column and underflows at none.
    pub fn new(
        photo1: Arc<Photo>,
        photo2: Arc<Photo>,
        grid_width: usize,
        grid_height: usize,
    ) -> Self {
        assert!(
            grid_width >= 2,
            "a mapping grid needs at least 2 columns, got {grid_width}"
        );
        let grid_cell_size = photo1.width / (grid_width - 1);
        DensePhotoMap {
            photo1,
            photo2,
            grid_width,
            grid_height,
            map_data: vec![f32::NAN; grid_width * grid_height * 2],
            grid_cell_size,
        }
    }

    /// Returns the pixel size each grid cell covers horizontally in `photo1`.
    pub fn grid_cell_size(&self) -> usize {
        self.grid_cell_size
    }

    /// The size of the mapping grid, as `(columns, rows)`.
    ///
    /// This is the resolution of the correspondence field itself, not of the photos; see
    /// [`Self::dimensions`] for the pixel dimensions the coordinates are expressed in.
    pub fn grid_dimensions(&self) -> (usize, usize) {
        (self.grid_width, self.grid_height)
    }

    /// The photo this mapping maps *from*.
    pub fn photo1(&self) -> &Arc<Photo> {
        &self.photo1
    }

    /// The photo this mapping maps *into*.
    pub fn photo2(&self) -> &Arc<Photo> {
        &self.photo2
    }

    /// Sets the mapped coordinates `(x2, y2)` in this map at grid location `(x1, y1)`.
    /// Typically, `(x1, y1)` indexes the grid, and `(x2, y2)` are the corresponding
    /// coordinates in `photo2`.
    ///
    /// # Parameters
    /// - `x1`, `y1`: Grid coordinate in the map (0 <= x1 < grid_width, 0 <= y1 < grid_height).
    /// - `x2`, `y2`: The mapped coordinate values stored in `map_data`.
    ///
    /// Out-of-range coordinates are ignored rather than written to a wrapped-around cell.
    pub fn set_grid_coordinates(&mut self, x1: usize, y1: usize, x2: f32, y2: f32) {
        if x1 >= self.grid_width || y1 >= self.grid_height {
            return;
        }
        let index = (y1 * self.grid_width + x1) * 2;
        self.map_data[index] = x2;
        self.map_data[index + 1] = y2;
    }

    /// Retrieves the mapped coordinates `(x2, y2)` from this map at grid location `(x1, y1)`.
    ///
    /// Returns `(NaN, NaN)` if the index is out of range or if the cell was
    /// never set (i.e., still contains `NaN`).
    ///
    /// `x1` is bounds-checked against `grid_width` in its own right, not just through the
    /// flat index: testing only the flat index lets `x1 == grid_width` address the first
    /// cell of the *next row*. Because [`Self::interpolated_point`] reads the corner
    /// at `x1 + 1`, that made every interpolation along the right edge silently blend
    /// with the far side of the image.
    pub fn grid_coordinates(&self, x1: usize, y1: usize) -> (f32, f32) {
        if x1 >= self.grid_width || y1 >= self.grid_height {
            return (f32::NAN, f32::NAN);
        }
        let index = (y1 * self.grid_width + x1) * 2;
        (self.map_data[index], self.map_data[index + 1])
    }

    /// Maps a pixel `(x1, y1)` from `photo1` to the corresponding location in `photo2`.
    /// Uses bilinear interpolation between the grid cells to get a smooth mapping.
    ///
    /// # Parameters
    /// - `x1`, `y1`: Floating-point pixel coordinates in `photo1`.
    ///
    /// # Returns
    /// A floating-point coordinate `(X, Y)` describing where that pixel maps in `photo2`.
    /// If any of the involved cells contain `NaN`, or the interpolation is invalid,
    /// returns `(NaN, NaN)`.
    pub fn map_photo_pixel(&self, x1: f32, y1: f32) -> (f32, f32) {
        self.interpolated_point(
            x1 / self.grid_cell_size as f32,
            y1 / self.grid_cell_size as f32,
        )
    }

    /// Interpolates the mapping for a fractional grid coordinate `(xin, yin)`.
    /// Looks up the four surrounding grid corners and performs bilinear interpolation.
    ///
    /// Valid only if all four corner cells are set (non-NaN) and pass distance checks.
    /// Returns `(NaN, NaN)` if interpolation fails or data is missing.
    pub fn interpolated_point(&self, xin: f32, yin: f32) -> (f32, f32) {
        let xxx = xin as usize;
        let yyy = yin as usize;
        let xr = xin - xxx as f32;
        let yr = yin - yyy as f32;

        // Get the four corner mappings.
        let (x1, y1) = self.grid_coordinates(xxx, yyy);
        let (x2, y2) = self.grid_coordinates(xxx + 1, yyy);
        let (x3, y3) = self.grid_coordinates(xxx + 1, yyy + 1);
        let (x4, y4) = self.grid_coordinates(xxx, yyy + 1);

        // If any corner is NaN, we cannot interpolate.
        if x1.is_nan() || x2.is_nan() || x3.is_nan() || x4.is_nan() {
            return (f32::NAN, f32::NAN);
        }

        // Check how far the center is from each corner; if it’s too large, discard.
        let cell_width = self.grid_cell_size as f32;
        let max_dist_sq = (cell_width * 3.0).powi(2);
        let center_x = (x1 + x2 + x3 + x4) / 4.0;
        let center_y = (y1 + y2 + y3 + y4) / 4.0;
        for &(cx, cy) in &[(x1, y1), (x2, y2), (x3, y3), (x4, y4)] {
            let dist_sq = (center_x - cx).powi(2) + (center_y - cy).powi(2);
            if dist_sq > max_dist_sq {
                return (f32::NAN, f32::NAN);
            }
        }

        // Bilinear interpolation between corners.
        let xx1 = x1 * (1.0 - xr) + x2 * xr;
        let yy1 = y1 * (1.0 - xr) + y2 * xr;

        let xx2 = x4 * (1.0 - xr) + x3 * xr;
        let yy2 = y4 * (1.0 - xr) + y3 * xr;

        let xt = xx1 * (1.0 - yr) + xx2 * yr;
        let yt = yy1 * (1.0 - yr) + yy2 * yr;

        (xt, yt)
    }

    /// Where the pixel at `(x, y)` in this map's source photo ends up in its target
    /// photo, or `None` if the algorithm could not map that point.
    ///
    /// Both the argument and the result are in *working-resolution* pixels — see
    /// [`crate::Correspondence::lookup`] for the same query in the coordinates of the
    /// photos you passed in.
    ///
    /// This is [`Self::map_photo_pixel`] with the `NaN` sentinel turned into a `None`, so
    /// that "no mapping here" cannot be mistaken for a coordinate.
    pub fn lookup(&self, x: f32, y: f32) -> Option<(f32, f32)> {
        let (mx, my) = self.map_photo_pixel(x, y);
        if mx.is_nan() || my.is_nan() {
            None
        } else {
            Some((mx, my))
        }
    }

    /// The dimensions, in working-resolution pixels, that [`Self::lookup`] takes and
    /// returns coordinates in.
    pub fn dimensions(&self) -> (usize, usize) {
        (self.photo1.width(), self.photo1.height())
    }

    /// Removes "outlier" mappings by checking consistency:
    /// - It maps `(x, y)` in `photo1` to `(mapped_x, mapped_y)`.
    /// - Then uses `other` to map `(mapped_x, mapped_y)` back into `photo1`.
    /// - If the round trip doesn't land near `(x, y)`, the cell is marked as invalid (set to `NaN`).
    ///
    /// # Parameters
    /// - `other`: Another `DensePhotoMap` presumably for the reverse transformation.
    /// - `max_dist`: Threshold for how far the round-trip mapping can deviate.
    pub fn remove_outliers(&mut self, other: &DensePhotoMap, max_dist: f32) {
        for y in 0..self.grid_height {
            for x in 0..self.grid_width {
                // Map forward
                let mapped = self.map_photo_pixel(
                    (x * self.grid_cell_size) as f32,
                    (y * self.grid_cell_size) as f32,
                );
                if mapped.0.is_nan() {
                    // Already invalid; set again to be sure
                    self.set_grid_coordinates(x, y, f32::NAN, f32::NAN);
                } else {
                    // Map back
                    let mapped_back = other.map_photo_pixel(mapped.0, mapped.1);
                    let dx = x as f32 - mapped_back.0 / self.grid_cell_size as f32;
                    let dy = y as f32 - mapped_back.1 / self.grid_cell_size as f32;

                    // If the round trip is too far, mark as invalid
                    if dx.is_nan() || dy.is_nan() || (dx * dx + dy * dy > max_dist) {
                        self.set_grid_coordinates(x, y, f32::NAN, f32::NAN);
                    }
                }
            }
        }
    }

    /// Calculates how many valid cells (non-NaN) exist in this map, expressed as
    /// a fraction of the total number of cells.
    ///
    /// # Returns
    /// A floating-point value in `[0.0, 1.0]` representing the ratio of valid
    /// cells to the total number of grid cells.
    pub fn calculate_used_area(&self) -> f32 {
        let mut count = 0usize;
        for y in 0..self.grid_height {
            for x in 0..self.grid_width {
                let mapped = self.grid_coordinates(x, y);
                if !mapped.0.is_nan() {
                    count += 1;
                }
            }
        }
        count as f32 / (self.grid_width * self.grid_height) as f32
    }

    /// Generates a new `Photo` by interpolating between the original coordinates
    /// `(x, y)` and the mapped coordinates `(x1, y1)`, blending them according
    /// to `interpolation_value` (clamped between 0 and 1).
    ///
    /// # Parameters
    /// - `interpolation_value`: Factor between `0.0` (use only original coordinates)
    ///   and `1.0` (use only mapped coordinates).
    /// - `detail_level`: Multiplier for how many sub-pixels to iterate over.
    ///   A larger `detail_level` could increase the resolution of the output,
    ///   but also the computation cost.
    ///
    /// # Returns
    /// A `Photo` with the same dimensions as `photo1`, but with possibly
    /// re-mapped and merged pixel data.
    ///
    /// # Notes
    /// - When mapped coordinates are invalid (`NaN`), this code currently
    ///   places a red pixel (`(255, 0, 0)`).
    /// - The logic uses the coordinates of `photo1` for indexing. If the mapped
    ///   point is out of range, it skips writing the pixel.
    pub fn interpolate_photo(&self, interpolation_value: f32, detail_level: usize) -> Photo {
        let interpolation_value = interpolation_value.clamp(0.0, 1.0);
        let photo1 = self.photo1.clone();

        // Prepare a blank RGBA buffer.
        let mut interpolated_img_data = vec![0u8; photo1.width * photo1.height * 4];

        for yi in 0..(photo1.height * detail_level) {
            let y = yi as f32 / detail_level as f32;
            for xi in 0..(photo1.width * detail_level) {
                let x = xi as f32 / detail_level as f32;
                let (x1, y1) = self.map_photo_pixel(x, y);

                // Interpolate the final coordinate.
                let x_interpolated = x * (1.0 - interpolation_value) + x1 * interpolation_value;
                let y_interpolated = y * (1.0 - interpolation_value) + y1 * interpolation_value;

                // Round to the nearest pixel in `photo1`.
                let xx1 = f32::round(x_interpolated) as usize;
                let yy1 = f32::round(y_interpolated) as usize;

                // Check bounds.
                if xx1 < photo1.width && yy1 < photo1.height {
                    // If invalid, paint red. Otherwise, use `photo1` pixel.
                    let (r, g, b) = if x1.is_nan() {
                        (255, 0, 0)
                    } else {
                        self.photo1.get_rgb(x as usize, y as usize)
                    };
                    let index = (yy1 * photo1.width + xx1) * 4;
                    if index < interpolated_img_data.len() {
                        interpolated_img_data[index] = r;
                        interpolated_img_data[index + 1] = g;
                        interpolated_img_data[index + 2] = b;
                        interpolated_img_data[index + 3] = 255; // alpha channel
                    }
                }
            }
        }

        Photo {
            img_data: interpolated_img_data,
            width: photo1.width,
            height: photo1.height,
        }
    }

    /// Repeatedly applies the `average_grid_points` method `iterations` times,
    /// effectively smoothing the grid by averaging neighbor values.
    pub fn smooth_grid_points_n_times(&self, iterations: usize) -> DensePhotoMap {
        let mut pm = self.clone();
        for _ in 0..iterations {
            pm = pm.average_grid_points();
        }
        pm
    }

    /// Creates a new `DensePhotoMap` where each cell is replaced by
    /// the average of its left, right, up, and down neighbors (if valid).
    ///
    /// This smooths out noisy mappings. Cells with invalid (`NaN`) neighbors
    /// are averaged only among the valid ones.
    pub fn average_grid_points(self) -> DensePhotoMap {
        let mut result = self.clone();
        let max_dist = self.grid_cell_size as f32 * 4.0;

        // Only average interior cells (1..width-1, 1..height-1).
        for y in 1..self.grid_height - 1 {
            for x in 1..self.grid_width - 1 {
                let a1 = self.grid_coordinates(x - 1, y);
                let a2 = self.grid_coordinates(x + 1, y);
                let b1 = self.grid_coordinates(x, y - 1);
                let b2 = self.grid_coordinates(x, y + 1);

                let center_x = (a1.0 + a2.0 + b1.0 + b2.0) / 4.0;
                let center_y = (a1.1 + a2.1 + b1.1 + b2.1) / 4.0;

                // Distances for neighbor validity checks.
                let aa = !(a1.0.is_nan() || a2.0.is_nan())
                    && ((a1.0 - center_x).powi(2) + (a1.1 - center_y).powi(2)).sqrt() < max_dist
                    && ((a2.0 - center_x).powi(2) + (a2.1 - center_y).powi(2)).sqrt() < max_dist;
                let bb = !(b1.0.is_nan() || b2.0.is_nan())
                    && ((b1.0 - center_x).powi(2) + (b1.1 - center_y).powi(2)).sqrt() < max_dist
                    && ((b2.0 - center_x).powi(2) + (b2.1 - center_y).powi(2)).sqrt() < max_dist;

                // If both horizontal neighbors (a1, a2) are valid, average them.
                // If both vertical neighbors (b1, b2) are valid, average them.
                // If both sets are valid, average all four.
                if aa && bb {
                    let avg_x = (a1.0 + a2.0 + b1.0 + b2.0) / 4.0;
                    let avg_y = (a1.1 + a2.1 + b1.1 + b2.1) / 4.0;
                    result.set_grid_coordinates(x, y, avg_x, avg_y);
                } else if aa {
                    let avg_x = (a1.0 + a2.0) / 2.0;
                    let avg_y = (a1.1 + a2.1) / 2.0;
                    result.set_grid_coordinates(x, y, avg_x, avg_y);
                } else if bb {
                    let avg_x = (b1.0 + b2.0) / 2.0;
                    let avg_y = (b1.1 + b2.1) / 2.0;
                    result.set_grid_coordinates(x, y, avg_x, avg_y);
                }
            }
        }
        result
    }

    /// Serializes the mapping to a byte vector, excluding the photos.
    ///
    /// The encoding is a header — the magic number `PXMP`, a `u16` format version, and
    /// the grid dimensions and cell size as `u64`s — followed by the grid itself as
    /// little-endian `f32` pairs. Empty cells are stored as `NaN`, exactly as they are
    /// held in memory. Read it back with [`DensePhotoMap::deserialize`], which pairs it
    /// with the two photos again.
    pub fn serialize(&self) -> Vec<u8> {
        let mut data = Vec::with_capacity(HEADER_LEN + self.map_data.len() * 4);
        data.extend_from_slice(MAGIC);
        data.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
        data.extend_from_slice(&(self.grid_width as u64).to_le_bytes());
        data.extend_from_slice(&(self.grid_height as u64).to_le_bytes());
        data.extend_from_slice(&(self.grid_cell_size as u64).to_le_bytes());
        for &val in &self.map_data {
            data.extend_from_slice(&val.to_le_bytes());
        }
        data
    }

    /// Reads back a mapping written by [`DensePhotoMap::serialize`], pairing it with the
    /// photos it describes.
    ///
    /// The two photos are not part of the encoding, so the caller supplies them; nothing
    /// checks that they are the ones the mapping was computed from.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Decode`] if `data` is not a mapping this build can read. The
    /// bytes may have come from a file or off a network, so every field is validated
    /// before it is used: no input, however malformed, makes this function panic.
    pub fn deserialize(data: &[u8], photo1: Arc<Photo>, photo2: Arc<Photo>) -> Result<Self, Error> {
        if data.len() < MAGIC.len() || &data[..MAGIC.len()] != MAGIC {
            return Err(DecodeError::NotAMapping.into());
        }

        // Every read below goes through `get`, so a truncated input is an error rather
        // than a slice index panic.
        let version = data
            .get(MAGIC.len()..MAGIC.len() + 2)
            .map(|b| u16::from_le_bytes([b[0], b[1]]))
            .ok_or(DecodeError::Truncated {
                expected: HEADER_LEN,
                actual: data.len(),
            })?;
        if version != FORMAT_VERSION {
            return Err(DecodeError::UnsupportedVersion {
                found: version,
                supported: FORMAT_VERSION,
            }
            .into());
        }

        let header = data
            .get(MAGIC.len() + 2..HEADER_LEN)
            .ok_or(DecodeError::Truncated {
                expected: HEADER_LEN,
                actual: data.len(),
            })?;
        let field = |i: usize| {
            let mut bytes = [0u8; 8];
            bytes.copy_from_slice(&header[i * 8..i * 8 + 8]);
            u64::from_le_bytes(bytes)
        };
        // A grid dimension wider than `usize` cannot be indexed on this target, whatever
        // the writer's word size was.
        let as_usize =
            |v: u64, reason| usize::try_from(v).map_err(|_| DecodeError::InvalidHeader { reason });
        let grid_width = as_usize(field(0), "grid width does not fit in a usize")?;
        let grid_height = as_usize(field(1), "grid height does not fit in a usize")?;
        let grid_cell_size = as_usize(field(2), "grid cell size does not fit in a usize")?;

        // `new` derives the cell size from `grid_width - 1`, so a grid narrower than two
        // columns is not something this type can represent.
        if grid_width < 2 {
            return Err(DecodeError::InvalidHeader {
                reason: "grid width must be at least 2",
            }
            .into());
        }
        if grid_height == 0 {
            return Err(DecodeError::InvalidHeader {
                reason: "grid height must be at least 1",
            }
            .into());
        }

        let floats = grid_width
            .checked_mul(grid_height)
            .and_then(|cells| cells.checked_mul(2))
            .ok_or(DecodeError::InvalidHeader {
                reason: "grid dimensions overflow",
            })?;
        let payload_len = floats.checked_mul(4).ok_or(DecodeError::InvalidHeader {
            reason: "grid dimensions overflow",
        })?;
        let expected = HEADER_LEN
            .checked_add(payload_len)
            .ok_or(DecodeError::InvalidHeader {
                reason: "grid dimensions overflow",
            })?;

        match data.len().cmp(&expected) {
            std::cmp::Ordering::Less => {
                return Err(DecodeError::Truncated {
                    expected,
                    actual: data.len(),
                }
                .into())
            }
            std::cmp::Ordering::Greater => {
                return Err(DecodeError::InvalidHeader {
                    reason: "trailing bytes after the declared grid",
                }
                .into())
            }
            std::cmp::Ordering::Equal => {}
        }

        let map_data = data[HEADER_LEN..]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        Ok(Self {
            photo1,
            photo2,
            grid_width,
            grid_height,
            map_data,
            grid_cell_size,
        })
    }
}
