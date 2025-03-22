use crate::circular_feature_descriptor::CircularFeatureDescriptor;
use crate::photo::Photo;

/// A grid that computes and stores `CircularFeatureDescriptor` values for each
/// (x, y) location in an image. Each descriptor captures the "center of mass"
/// of the R/G/B channels in a circular neighborhood around that point.
pub struct CircularFeatureGrid {
    /// The width (in pixels) of the grid/image.
    image_width: usize,
    /// The height (in pixels) of the grid/image.
    image_height: usize,
    /// The radius of the circular neighborhood used to compute features.
    circle_radius: usize,
    /// The maximum possible color value for a circular region. (Used for normalization if needed.)
    max_color_value: usize,
    /// A vector of circular feature descriptors, one per position in the grid.
    feature_descriptors: Vec<CircularFeatureDescriptor>,
    /// Raw RGB (or RGBA) data for the image being analyzed.
    image_pixel_data: Vec<u8>,
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
    pub fn new_with_rotation(
        photo: &Photo,
        width: usize,
        height: usize,
        circle_radius: usize,
        rotation: bool
    ) -> Self {
        // Copy the image data from the `photo`.
        let data = &photo.img_data;

        // Initialize the grid with default descriptors.
        let mut grid = CircularFeatureGrid {
            image_width: width,
            image_height: height,
            circle_radius,
            max_color_value: 0,
            feature_descriptors: vec![CircularFeatureDescriptor::default(); width * height],
            image_pixel_data: data.to_vec(),
        };

        // Populate the feature descriptors (either with or without rotation).
        grid.populate_feature_descriptors(rotation);

        // Calculate the maximum possible sum of color values for a circle of this radius.
        let mut pixel_count_in_circle = 0;
        for y in -(circle_radius as isize)..=(circle_radius as isize) {
            // For each row in the circle, figure out how many pixels are within the radius.
            let row_width = ((circle_radius * circle_radius) as f64 - (y * y) as f64).sqrt().round() as isize;
            for _x in -row_width..=row_width {
                pixel_count_in_circle += 1;
            }
        }
        // If each pixel can contribute up to 255 per channel, multiply by 255.
        grid.max_color_value = pixel_count_in_circle * 255;

        grid
    }

    /// Returns a reference to the vector of `CircularFeatureDescriptor` objects.
    pub fn get_infos(&self) -> &Vec<CircularFeatureDescriptor> {
        &self.feature_descriptors
    }

    /// Fills in the `feature_descriptors` for every position in the grid.
    fn populate_feature_descriptors(&mut self, rotation: bool) {
        for y in 0..self.image_height {
            for x in 0..self.image_width {
                // Compute the descriptor for the circular region around (x, y).
                // If `rotation` were used differently, we might adjust the call, but
                // here both branches do the same operation.
                self.feature_descriptors[x + y * self.image_width] =
                    if rotation {
                        self.compute_circular_feature_descriptor(x as isize, y as isize, self.circle_radius as isize)
                    } else {
                        self.compute_circular_feature_descriptor(x as isize, y as isize, self.circle_radius as isize)
                    };
            }
        }
    }

    /// Computes a `CircularFeatureDescriptor` for the circular region around (center_x, center_y).
    /// The radius specifies how large the neighborhood is in all directions.
    pub fn compute_circular_feature_descriptor(
        &self,
        center_x: isize,
        center_y: isize,
        radius: isize
    ) -> CircularFeatureDescriptor {
        // Accumulators for total R/G/B values.
        let mut sum_red = 0;
        let mut sum_green = 0;
        let mut sum_blue = 0;

        // Accumulators for weighted sums (used to compute "center of mass" per color).
        let mut sum_weighted_red_x = 0;
        let mut sum_weighted_red_y = 0;

        let mut sum_weighted_green_x = 0;
        let mut sum_weighted_green_y = 0;

        let mut sum_weighted_blue_x = 0;
        let mut sum_weighted_blue_y = 0;

        // Keep track of how many pixels we process inside the circular neighborhood.
        let mut pixel_count = 0;

        // Loop over each coordinate in the circular area around (center_x, center_y).
        for dy in -radius..=radius {
            // For each row, calculate how far we can extend in the x-direction.
            let row_radius = ((radius * radius) as f64 - (dy * dy) as f64).sqrt().round() as isize;
            for dx in -row_radius..=row_radius {
                // Wrap around edges using modulo operations (toroidal coordinates).
                let wrapped_x = (center_x + dx + self.image_width as isize) % self.image_width as isize;
                let wrapped_y = (center_y + dy + self.image_height as isize) % self.image_height as isize;

                // Index into the image buffer (assuming 4 bytes/pixel [RGBA] or 3 if not used).
                let pixel_index = 4 * (wrapped_x + wrapped_y * self.image_width as isize) as usize;

                // Extract R, G, and B color values.
                let red_val = self.image_pixel_data[pixel_index] as isize;
                let green_val = self.image_pixel_data[pixel_index + 1] as isize;
                let blue_val = self.image_pixel_data[pixel_index + 2] as isize;

                // Accumulate total color sums.
                sum_red += red_val;
                sum_green += green_val;
                sum_blue += blue_val;

                // Accumulate weighted sums (x- and y-coordinates times each color).
                sum_weighted_red_x += dx * red_val;
                sum_weighted_red_y += dy * red_val;

                sum_weighted_green_x += dx * green_val;
                sum_weighted_green_y += dy * green_val;

                sum_weighted_blue_x += dx * blue_val;
                sum_weighted_blue_y += dy * blue_val;

                pixel_count += 1;
            }
        }

        // Build the descriptor.
        let mut descriptor = CircularFeatureDescriptor::default();

        // Compute the "center of mass" for each color, i.e., X and Y offsets.
        let red_cm_x = if sum_red == 0 { 0.0 } else { sum_weighted_red_x as f32 / sum_red as f32 };
        let red_cm_y = if sum_red == 0 { 0.0 } else { sum_weighted_red_y as f32 / sum_red as f32 };
        let red_angle = if sum_red == 0 { 0.0 } else { red_cm_y.atan2(red_cm_x) };
        let red_radius = (red_cm_y * red_cm_y + red_cm_x * red_cm_x).sqrt();

        let green_cm_x = if sum_green == 0 { 0.0 } else { sum_weighted_green_x as f32 / sum_green as f32 };
        let green_cm_y = if sum_green == 0 { 0.0 } else { sum_weighted_green_y as f32 / sum_green as f32 };
        let green_angle = if sum_green == 0 { 0.0 } else { green_cm_y.atan2(green_cm_x) };
        let green_radius = (green_cm_y * green_cm_y + green_cm_x * green_cm_x).sqrt();

        let blue_cm_x = if sum_blue == 0 { 0.0 } else { sum_weighted_blue_x as f32 / sum_blue as f32 };
        let blue_cm_y = if sum_blue == 0 { 0.0 } else { sum_weighted_blue_y as f32 / sum_blue as f32 };
        let blue_angle = if sum_blue == 0 { 0.0 } else { blue_cm_y.atan2(blue_cm_x) };
        let blue_radius = (blue_cm_y * blue_cm_y + blue_cm_x * blue_cm_x).sqrt();

        // Compute the total color sum across R/G/B.
        let sum_all = sum_red + sum_green + sum_blue;
        let total_cm_x = if sum_all == 0 { 0.0 } else {
            (sum_weighted_red_x + sum_weighted_green_x + sum_weighted_blue_x) as f32 / sum_all as f32
        };
        let total_cm_y = if sum_all == 0 { 0.0 } else {
            (sum_weighted_red_y + sum_weighted_green_y + sum_weighted_blue_y) as f32 / sum_all as f32
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

    /// Set feature descriptors explicitly
    pub fn set_feature_descriptors(&mut self, descriptors: Vec<CircularFeatureDescriptor>) {
        self.feature_descriptors = descriptors;
    }
}
