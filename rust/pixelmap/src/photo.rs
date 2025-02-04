/// A basic representation of an image with RGBA pixel data.
/// Each pixel occupies 4 bytes: R, G, B, and A (alpha).
pub struct Photo {
    /// Pixel data stored in a 1D `Vec<u8>`, in RGBA format (4 bytes per pixel).
    pub img_data: Vec<u8>,
    /// The width (in pixels) of the image.
    pub width: usize,
    /// The height (in pixels) of the image.
    pub height: usize,
}

impl Default for Photo {
    /// Creates an empty `Photo` with zero width and height, and no image data.
    fn default() -> Photo {
        Photo {
            img_data: Vec::new(),
            width: 0,
            height: 0,
        }
    }
}

impl Photo {
    /// Returns the `(R, G, B)` components at the pixel coordinate `(x, y)`.
    ///
    /// If `(x, y)` is out of bounds, this method returns `(0, 0, 255)`, effectively a blue pixel.
    ///
    /// # Parameters
    /// - `x`: The x-coordinate of the pixel.
    /// - `y`: The y-coordinate of the pixel.
    ///
    /// # Returns
    /// A tuple `(r, g, b)` representing the red, green, and blue channels of the pixel.
    pub fn get_rgb(&self, x: usize, y: usize) -> (u8, u8, u8) {
        // Rust doesn't allow negative indices, so `x < 0 || y < 0` is redundant, but was in the original code.
        // For clarity, we keep the index checks for completeness.
        if x >= self.width || y >= self.height {
            (0, 0, 255) // Return blue if out of bounds
        } else {
            let index = (y * self.width + x) * 4;
            let r = self.img_data[index];
            let g = self.img_data[index + 1];
            let b = self.img_data[index + 2];
            (r, g, b)
        }
    }

    /// Produces a new `Photo` scaled proportionally to `new_width`.
    ///
    /// This method preserves the aspect ratio by computing a scale factor and rounding
    /// the new height accordingly. The pixel values in the resulting image are computed
    /// by averaging all corresponding pixels from the original image that fall into
    /// the region mapped by the new pixel.
    ///
    /// # Parameters
    /// - `new_width`: The desired new width of the scaled image. Must be greater than 0.
    ///
    /// # Returns
    /// A new `Photo` object with width = `new_width` and a proportionally scaled height.
    ///
    /// # Panics
    /// Panics if `new_width` is zero, since that would lead to a division by zero.
    pub fn get_scaled_proportional(&self, new_width: usize) -> Photo {
        // Check that the new width is not zero to avoid division by zero
        if new_width == 0 {
            panic!("The new width must be greater than 0");
        }

        // Compute the new height proportionally to the new width
        let scale_factor = new_width as f32 / self.width as f32;
        let new_height = (self.height as f32 * scale_factor).round() as usize;

        // Create a new vector to store the pixel data (RGBA) for the scaled image
        let mut new_img_data = vec![0u8; new_width * new_height * 4];

        // Iterate over each pixel in the new image
        for new_y in 0..new_height {
            for new_x in 0..new_width {
                // Calculate which portion of the original image this new pixel corresponds to
                let orig_x_start = ((new_x as f32) / scale_factor).round() as usize;
                let orig_y_start = ((new_y as f32) / scale_factor).round() as usize;
                let orig_x_end = (((new_x + 1) as f32) / scale_factor).round() as usize;
                let orig_y_end = (((new_y + 1) as f32) / scale_factor).round() as usize;

                // Ensure that the indices are within the original image's bounds
                let orig_x_end = orig_x_end.min(self.width - 1);
                let orig_y_end = orig_y_end.min(self.height - 1);

                // Accumulators for RGBA values, plus a pixel count
                let mut r_total: u32 = 0;
                let mut g_total: u32 = 0;
                let mut b_total: u32 = 0;
                let mut a_total: u32 = 0;
                let mut pixel_count: u32 = 0;

                // Iterate over the block of original pixels that map to this new pixel
                for orig_y in orig_y_start..=orig_y_end {
                    for orig_x in orig_x_start..=orig_x_end {
                        let orig_index = (orig_y * self.width + orig_x) * 4;
                        r_total += self.img_data[orig_index] as u32;
                        g_total += self.img_data[orig_index + 1] as u32;
                        b_total += self.img_data[orig_index + 2] as u32;
                        a_total += self.img_data[orig_index + 3] as u32;
                        pixel_count += 1;
                    }
                }

                // Compute the average color value for each channel
                let r_avg = (r_total / pixel_count) as u8;
                let g_avg = (g_total / pixel_count) as u8;
                let b_avg = (b_total / pixel_count) as u8;
                let a_avg = (a_total / pixel_count) as u8;

                // Store the pixel value in the new image
                let new_index = (new_y * new_width + new_x) * 4;
                new_img_data[new_index] = r_avg;
                new_img_data[new_index + 1] = g_avg;
                new_img_data[new_index + 2] = b_avg;
                new_img_data[new_index + 3] = a_avg; // Preserve the alpha channel
            }
        }

        // Return the scaled image as a new Photo structure
        Photo {
            img_data: new_img_data,
            width: new_width,
            height: new_height,
        }
    }
}
