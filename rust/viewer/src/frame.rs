//! A single animation frame, stored in the 0RGB layout that `minifb` expects.

use pixelmap::photo::Photo;

/// One frame of the animation.
pub struct Frame {
    /// Width of the frame in pixels.
    pub width: usize,
    /// Height of the frame in pixels.
    pub height: usize,
    /// Pixel data, one `0x00RRGGBB` word per pixel, row by row.
    pub pixels: Vec<u32>,
}

impl Frame {
    /// Converts a [`Photo`] (RGBA bytes) into a frame, dropping the alpha channel.
    pub fn from_photo(photo: &Photo) -> Frame {
        let pixels = photo
            .img_data
            .chunks_exact(4)
            .map(|px| (px[0] as u32) << 16 | (px[1] as u32) << 8 | px[2] as u32)
            .collect();

        Frame {
            width: photo.width,
            height: photo.height,
            pixels,
        }
    }

    /// Draws the frame centered in a `dst_width` x `dst_height` buffer, scaled to
    /// fit while keeping its aspect ratio. Pixels outside the frame keep whatever
    /// the caller left in `dst`.
    pub fn blit_fitted(&self, dst: &mut [u32], dst_width: usize, dst_height: usize) {
        if self.width == 0 || self.height == 0 || dst_width == 0 || dst_height == 0 {
            return;
        }

        // Largest integer size that fits inside the destination, aspect preserved.
        let scale = (dst_width as f32 / self.width as f32)
            .min(dst_height as f32 / self.height as f32);
        let draw_width = ((self.width as f32 * scale) as usize).clamp(1, dst_width);
        let draw_height = ((self.height as f32 * scale) as usize).clamp(1, dst_height);
        let offset_x = (dst_width - draw_width) / 2;
        let offset_y = (dst_height - draw_height) / 2;

        for y in 0..draw_height {
            let src_row = (y * self.height / draw_height) * self.width;
            let dst_row = (offset_y + y) * dst_width + offset_x;
            for x in 0..draw_width {
                dst[dst_row + x] = self.pixels[src_row + x * self.width / draw_width];
            }
        }
    }
}
