use std::cell::Cell;
use crate::affine_transform::AffineTransform;
use crate::photo::Photo;
use std::rc::Rc;

/// The `CorrespondenceScoring` struct is responsible for evaluating the similarity between two photos
/// using pixel comparisons within a circular neighborhood. This similarity is computed based on color
/// differences between corresponding regions of the photos.
pub struct CorrespondenceScoring {
    /// A shared reference to the first photo.
    photo1: Rc<Photo>,
    /// A shared reference to the second photo.
    photo2: Rc<Photo>,
    /// Precomputed table of maximum x-offsets for each y-offset in the circular neighborhood.
    sqrt_table: Vec<isize>,
    /// Radius of the circular neighborhood used for comparisons.
    neighborhood_radius: isize,
    /// Tracks the total number of similarity score calculations performed.
    total_invocations: Cell<usize>,
}

impl CorrespondenceScoring {
    /// Creates a new `CorrespondenceScoring` instance.
    ///
    /// # Arguments
    /// * `photo1` - A shared reference to the first photo.
    /// * `photo2` - A shared reference to the second photo.
    /// * `neighborhood_radius` - The radius of the circular neighborhood used for comparisons.
    ///
    /// # Details
    /// This method precomputes a `sqrt_table` to optimize calculations of maximum x-offsets
    /// for each y-offset in the circular neighborhood.
    pub fn new(photo1: Rc<Photo>, photo2: Rc<Photo>, neighborhood_radius: isize) -> Self {
        let diameter = (2 * neighborhood_radius + 1) as usize;
        let mut sqrt_table = vec![0isize; diameter];
        let radius2 = neighborhood_radius * neighborhood_radius;

        // Precompute the square root values for the circular boundary.
        for y in -neighborhood_radius..=neighborhood_radius {
            let idx = (y + neighborhood_radius) as usize;
            let yy = y * y;
            sqrt_table[idx] = f32::sqrt((radius2 - yy) as f32).floor() as isize;
        }

        CorrespondenceScoring {
            photo1,
            photo2,
            sqrt_table,
            neighborhood_radius,
            total_invocations: Cell::new(0),
        }
    }

    /// Calculates the similarity score between two photos based on an affine transformation mapping.
    ///
    /// # Arguments
    /// * `circle_mapping` - An `AffineTransform` struct describing the mapping between the two photos.
    ///
    /// # Returns
    /// * A `f32` value representing the similarity score. A lower score indicates higher similarity.
    /// If no pixels are compared, returns `std::f32::MAX`.
    ///
    /// # Details
    /// The method evaluates the variance of color differences (red, green, blue) in the circular neighborhood
    /// defined by the affine transformation. The scores are accumulated and used to compute the total variance.
    pub fn calculate_similarity_score(&self, circle_mapping: &AffineTransform) -> f32 {
        let mut count = 0i64;
        let mut sum_dr = 0i64;
        let mut sum_dg = 0i64;
        let mut sum_db = 0i64;
        let mut sum_dr2 = 0i64;
        let mut sum_dg2 = 0i64;
        let mut sum_db2 = 0i64;

        let radius = self.neighborhood_radius;
        let x1 = circle_mapping.origin_x as isize;
        let y1 = circle_mapping.origin_y as isize;

        let photo1_width = self.photo1.width;
        let photo2_width = self.photo2.width;
        let photo1_data = &self.photo1.img_data;
        let photo2_data = &self.photo2.img_data;

        let kx1 = circle_mapping.a11;
        let kx2 = circle_mapping.a12;
        let ky1 = circle_mapping.a21;
        let ky2 = circle_mapping.a22;
        let x2_offset = circle_mapping.translate_x;
        let y2_offset = circle_mapping.translate_y;

        for y in -radius..=radius {
            let idx = (y + radius) as usize;
            let xx = self.sqrt_table[idx];
            let yy1 = y1 + y;

            for x in -xx..=xx {
                let xx1 = x1 + x;
                let pixel1index = (xx1 + yy1 * photo1_width as isize) * 4;
                if pixel1index < 0 || pixel1index + 3 >= photo1_data.len() as isize {
                    continue;
                }

                // Inline extrapolate point calculation
                let dx = x as f32;
                let dy = y as f32;
                let x2f = dx * kx1 + dy * kx2 + x2_offset;
                let y2f = dx * ky1 + dy * ky2 + y2_offset;
                let x2 = f32::round(x2f) as isize;
                let y2 = f32::round(y2f) as isize;

                let pixel2index = (x2 + y2 * photo2_width as isize) * 4;

                if pixel2index < 0 || pixel2index + 3 >= photo2_data.len() as isize {
                    continue;
                }
                let pixel1index = pixel1index as usize;
                let pixel2index = pixel2index as usize;

                let r1 = photo1_data[pixel1index] as i64;
                let g1 = photo1_data[pixel1index + 1] as i64;
                let b1 = photo1_data[pixel1index + 2] as i64;

                let r2 = photo2_data[pixel2index] as i64;
                let g2 = photo2_data[pixel2index + 1] as i64;
                let b2 = photo2_data[pixel2index + 2] as i64;

                let dr = r1 - r2;
                let dg = g1 - g2;
                let db = b1 - b2;

                sum_dr += dr;
                sum_dg += dg;
                sum_db += db;

                sum_dr2 += dr * dr;
                sum_dg2 += dg * dg;
                sum_db2 += db * db;

                count += 1;
            }
        }

        if count == 0 {
            return std::f32::MAX;
        }

        let mean_dr = sum_dr as f64 / count as f64;
        let mean_dg = sum_dg as f64 / count as f64;
        let mean_db = sum_db as f64 / count as f64;

        let variance_dr = sum_dr2 as f64 / count as f64 - mean_dr * mean_dr;
        let variance_dg = sum_dg2 as f64 / count as f64 - mean_dg * mean_dg;
        let variance_db = sum_db2 as f64 / count as f64 - mean_db * mean_db;

        let score = variance_dr + variance_dg + variance_db;

        self.total_invocations.set(self.total_invocations.get() + 1);
        score as f32
    }

    /// Returns the total number of similarity score calculations performed.
    pub fn get_num_comparisons(&self) -> usize {
        self.total_invocations.get()
    }

}
