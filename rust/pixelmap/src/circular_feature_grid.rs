use crate::circular_feature_descriptor::CircularFeatureDescriptor;
use crate::photo::Photo;

/// A grid that computes and stores `CircularFeatureDescriptor` values for each
/// (x, y) location in an image. Each descriptor captures the "center of mass"
/// of the R/G/B channels in a circular neighborhood around that point.
pub struct CircularFeatureGrid {
    /// A vector of circular feature descriptors, one per position in the grid.
    feature_descriptors: Vec<CircularFeatureDescriptor>,
}

impl CircularFeatureGrid {
    /// Creates a new `CircularFeatureGrid` without manually specifying rotation handling.
    pub fn new(photo: &Photo, width: usize, height: usize, circle_radius: usize) -> Self {
        // By default, we allow rotation-based computations (the `true` parameter).
        Self::new_with_rotation(photo, width, height, circle_radius, true)
    }

    /// Creates a new `CircularFeatureGrid`, optionally using rotation-based computations.
    ///
    /// - `photo` holds the pixel data.
    /// - `width`, `height` define the grid size in pixels.
    /// - `circle_radius` sets how large each neighborhood is.
    /// - `rotation` indicates whether advanced rotation alignment is used.
    /// Creates a new `CircularFeatureGrid`, optionally using rotation-based computations.
    ///
    /// - `photo` holds the pixel data.
    /// - `width`, `height` define the grid size in pixels.
    /// - `circle_radius` sets how large each neighborhood is.
    /// - `_rotation` is accepted for backwards compatibility; both settings always
    ///   computed the same descriptor.
    pub fn new_with_rotation(
        photo: &Photo,
        width: usize,
        height: usize,
        circle_radius: usize,
        _rotation: bool
    ) -> Self {
        let radius = circle_radius as isize;

        // Half-width of the disc for each row, computed once here instead of
        // being recomputed (with a `sqrt` and a `round`) for every descriptor.
        let row_half_width: Vec<isize> = (-radius..=radius)
            .map(|dy| (((radius * radius) as f64 - (dy * dy) as f64).sqrt().round()) as isize)
            .collect();

        let mut feature_descriptors =
            vec![CircularFeatureDescriptor::default(); width * height];

        Self::populate_feature_descriptors(
            &photo.img_data,
            width,
            height,
            radius,
            &row_half_width,
            &mut feature_descriptors,
        );

        CircularFeatureGrid { feature_descriptors }
    }

    /// Returns a reference to the vector of `CircularFeatureDescriptor` objects.
    pub fn get_infos(&self) -> &Vec<CircularFeatureDescriptor> {
        &self.feature_descriptors
    }

    /// Fills in the `feature_descriptors` for every position in the grid.
    ///
    /// Pixels at least `radius` from every edge cannot wrap, so they are handled by a
    /// branch- and division-free inner loop; only the border ring pays for the
    /// toroidal modulo. On a 400x225 image with radius 10 that fast path covers about
    /// 87% of the descriptors, and the two `%` it avoids were 64-bit divisions in the
    /// innermost loop.
    fn populate_feature_descriptors(
        data: &[u8],
        width: usize,
        height: usize,
        radius: isize,
        row_half_width: &[isize],
        out: &mut [CircularFeatureDescriptor],
    ) {
        let (w, h) = (width as isize, height as isize);
        for y in 0..h {
            let interior_row = y >= radius && y < h - radius;
            for x in 0..w {
                let interior = interior_row && x >= radius && x < w - radius;
                let sums = if interior {
                    Self::disc_sums_interior(data, w, x, y, radius, row_half_width)
                } else {
                    Self::disc_sums_wrapping(data, w, h, x, y, radius, row_half_width)
                };
                out[(x + y * w) as usize] = Self::finish_descriptor(x, y, sums);
            }
        }
    }

    /// Accumulates the disc sums for a centre that cannot touch an image edge.
    #[inline]
    fn disc_sums_interior(
        data: &[u8],
        w: isize,
        center_x: isize,
        center_y: isize,
        radius: isize,
        row_half_width: &[isize],
    ) -> DiscSums {
        let mut s = DiscSums::default();
        for dy in -radius..=radius {
            let half = row_half_width[(dy + radius) as usize];
            // Start of this row's span; every pixel of it is in bounds.
            let row_base = (((center_y + dy) * w + center_x - half) * 4) as usize;
            for (i, dx) in (-half..=half).enumerate() {
                let p = row_base + i * 4;
                // SAFETY: the centre is at least `radius` from every edge and
                // `half <= radius`, so `p + 2` is inside `data`.
                let (r, g, b) = unsafe {
                    (
                        *data.get_unchecked(p) as isize,
                        *data.get_unchecked(p + 1) as isize,
                        *data.get_unchecked(p + 2) as isize,
                    )
                };
                s.add(dx, dy, r, g, b);
            }
        }
        s
    }

    /// Accumulates the disc sums for a centre near an edge, wrapping toroidally.
    fn disc_sums_wrapping(
        data: &[u8],
        w: isize,
        h: isize,
        center_x: isize,
        center_y: isize,
        radius: isize,
        row_half_width: &[isize],
    ) -> DiscSums {
        let mut s = DiscSums::default();
        for dy in -radius..=radius {
            let half = row_half_width[(dy + radius) as usize];
            let wrapped_y = (center_y + dy + h) % h;
            let row = wrapped_y * w;
            for dx in -half..=half {
                let wrapped_x = (center_x + dx + w) % w;
                let p = ((wrapped_x + row) * 4) as usize;
                let (r, g, b) = (
                    data[p] as isize,
                    data[p + 1] as isize,
                    data[p + 2] as isize,
                );
                s.add(dx, dy, r, g, b);
            }
        }
        s
    }

    /// Turns accumulated disc sums into a `CircularFeatureDescriptor`, computing the
    /// per-channel "centre of mass" and aligning each channel to the combined angle.
    fn finish_descriptor(center_x: isize, center_y: isize, s: DiscSums) -> CircularFeatureDescriptor {
        let mut descriptor = CircularFeatureDescriptor::default();

        let (sum_red, sum_green, sum_blue) = (s.sum_red, s.sum_green, s.sum_blue);

        // Compute the "center of mass" for each color, i.e., X and Y offsets.
        let red_cm_x = if sum_red == 0 { 0.0 } else { s.wx_red as f32 / sum_red as f32 };
        let red_cm_y = if sum_red == 0 { 0.0 } else { s.wy_red as f32 / sum_red as f32 };
        let red_angle = if sum_red == 0 { 0.0 } else { red_cm_y.atan2(red_cm_x) };
        let red_radius = (red_cm_y * red_cm_y + red_cm_x * red_cm_x).sqrt();

        let green_cm_x = if sum_green == 0 { 0.0 } else { s.wx_green as f32 / sum_green as f32 };
        let green_cm_y = if sum_green == 0 { 0.0 } else { s.wy_green as f32 / sum_green as f32 };
        let green_angle = if sum_green == 0 { 0.0 } else { green_cm_y.atan2(green_cm_x) };
        let green_radius = (green_cm_y * green_cm_y + green_cm_x * green_cm_x).sqrt();

        let blue_cm_x = if sum_blue == 0 { 0.0 } else { s.wx_blue as f32 / sum_blue as f32 };
        let blue_cm_y = if sum_blue == 0 { 0.0 } else { s.wy_blue as f32 / sum_blue as f32 };
        let blue_angle = if sum_blue == 0 { 0.0 } else { blue_cm_y.atan2(blue_cm_x) };
        let blue_radius = (blue_cm_y * blue_cm_y + blue_cm_x * blue_cm_x).sqrt();

        // Compute the total color sum across R/G/B.
        let sum_all = sum_red + sum_green + sum_blue;
        let total_cm_x = if sum_all == 0 { 0.0 } else {
            (s.wx_red + s.wx_green + s.wx_blue) as f32 / sum_all as f32
        };
        let total_cm_y = if sum_all == 0 { 0.0 } else {
            (s.wy_red + s.wy_green + s.wy_blue) as f32 / sum_all as f32
        };
        let total_angle = if sum_all == 0 { 0.0 } else { total_cm_y.atan2(total_cm_x) };

        // Store the combined total center-of-mass and radius.
        descriptor.total_angle = total_angle;
        descriptor.total_radius = (total_cm_x * total_cm_x + total_cm_y * total_cm_y).sqrt();

        // Rotate each color channel so that the total color angle is the new "zero" angle.
        // This gives "fixed" coordinates, aligning each channel relative to the total angle.
        descriptor.aligned_red_x = (red_angle - total_angle).cos() * red_radius;
        descriptor.aligned_red_y = (red_angle - total_angle).sin() * red_radius;
        descriptor.aligned_green_x = (green_angle - total_angle).cos() * green_radius;
        descriptor.aligned_green_y = (green_angle - total_angle).sin() * green_radius;
        descriptor.aligned_blue_x = (blue_angle - total_angle).cos() * blue_radius;
        descriptor.aligned_blue_y = (blue_angle - total_angle).sin() * blue_radius;

        // Fill in descriptor metadata.
        descriptor.center_x = center_x as u16;
        descriptor.center_y = center_y as u16;
        descriptor.sum_red = sum_red as i32;
        descriptor.sum_green = sum_green as i32;
        descriptor.sum_blue = sum_blue as i32;

        // Store a quantized version of the fixed coordinates as the feature vector.
        descriptor.feature_vector[0] = f32::round(descriptor.aligned_red_x * 100.0) as i64;
        descriptor.feature_vector[1] = f32::round(descriptor.aligned_red_y * 100.0) as i64;
        descriptor.feature_vector[2] = f32::round(descriptor.aligned_green_x * 100.0) as i64;
        descriptor.feature_vector[3] = f32::round(descriptor.aligned_green_y * 100.0) as i64;
        descriptor.feature_vector[4] = f32::round(descriptor.aligned_blue_x * 100.0) as i64;
        descriptor.feature_vector[5] = f32::round(descriptor.aligned_blue_y * 100.0) as i64;

        descriptor
    }
}

/// Running colour and position-weighted sums over one circular neighbourhood.
#[derive(Default, Clone, Copy)]
struct DiscSums {
    sum_red: isize,
    sum_green: isize,
    sum_blue: isize,
    wx_red: isize,
    wy_red: isize,
    wx_green: isize,
    wy_green: isize,
    wx_blue: isize,
    wy_blue: isize,
}

impl DiscSums {
    #[inline(always)]
    fn add(&mut self, dx: isize, dy: isize, r: isize, g: isize, b: isize) {
        self.sum_red += r;
        self.sum_green += g;
        self.sum_blue += b;
        self.wx_red += dx * r;
        self.wy_red += dy * r;
        self.wx_green += dx * g;
        self.wy_green += dy * g;
        self.wx_blue += dx * b;
        self.wy_blue += dy * b;
    }
}
