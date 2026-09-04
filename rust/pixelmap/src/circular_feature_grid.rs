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
    ///
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
    /// Along the interior of a row the disc is *slid* rather than re-walked: consecutive
    /// centres overlap in all but one column per disc row, so [`AbsSums`] can update the
    /// sums by subtracting the pixel leaving each of the `2r+1` rows and adding the one
    /// entering it. That is 2*(2r+1) pixel loads per descriptor instead of the whole
    /// disc — 42 instead of 317 at radius 10 — and, being integer arithmetic, gives
    /// bit-identical sums.
    ///
    /// Only the border ring, where the disc wraps toroidally and the overlap argument
    /// does not hold, still walks the full disc.
    fn populate_feature_descriptors(
        data: &[u8],
        width: usize,
        height: usize,
        radius: isize,
        row_half_width: &[isize],
        out: &mut [CircularFeatureDescriptor],
    ) {
        let (w, h) = (width as isize, height as isize);
        let interior_x = radius < w - radius;
        for y in 0..h {
            let interior_row = y >= radius && y < h - radius && interior_x;
            if !interior_row {
                for x in 0..w {
                    let sums = Self::disc_sums_wrapping(data, w, h, x, y, radius, row_half_width);
                    out[(x + y * w) as usize] = Self::finish_descriptor(x, y, sums);
                }
                continue;
            }
            for x in 0..radius {
                let sums = Self::disc_sums_wrapping(data, w, h, x, y, radius, row_half_width);
                out[(x + y * w) as usize] = Self::finish_descriptor(x, y, sums);
            }
            // Seed the running sums at the first interior centre of this row.
            let mut acc = AbsSums::default();
            for (i, &half) in row_half_width.iter().enumerate() {
                let py = y + i as isize - radius;
                for px in (radius - half)..=(radius + half) {
                    acc.add(px, py, data, w);
                }
            }
            for x in radius..(w - radius) {
                if x > radius {
                    // Slide one column right: one pixel out and one in per disc row.
                    for (i, &half) in row_half_width.iter().enumerate() {
                        let py = y + i as isize - radius;
                        acc.sub(x - 1 - half, py, data, w);
                        acc.add(x + half, py, data, w);
                    }
                }
                out[(x + y * w) as usize] =
                    Self::finish_descriptor(x, y, acc.to_disc_sums(x, y));
            }
            for x in (w - radius)..w {
                let sums = Self::disc_sums_wrapping(data, w, h, x, y, radius, row_half_width);
                out[(x + y * w) as usize] = Self::finish_descriptor(x, y, sums);
            }
        }
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

        let inv = |n: isize| if n == 0 { 0.0 } else { 1.0 / n as f32 };
        let (ir, ig, ib) = (inv(sum_red), inv(sum_green), inv(sum_blue));
        let red_cm_x = s.wx_red as f32 * ir;
        let red_cm_y = s.wy_red as f32 * ir;
        let green_cm_x = s.wx_green as f32 * ig;
        let green_cm_y = s.wy_green as f32 * ig;
        let blue_cm_x = s.wx_blue as f32 * ib;
        let blue_cm_y = s.wy_blue as f32 * ib;

        let sum_all = sum_red + sum_green + sum_blue;
        let ia = inv(sum_all);
        let total_cm_x = (s.wx_red + s.wx_green + s.wx_blue) as f32 * ia;
        let total_cm_y = (s.wy_red + s.wy_green + s.wy_blue) as f32 * ia;
        let total_radius = (total_cm_x * total_cm_x + total_cm_y * total_cm_y).sqrt();

        descriptor.total_angle = if sum_all == 0 { 0.0 } else { total_cm_y.atan2(total_cm_x) };

        // Rotating a channel's centre of mass by -total_angle needs no trigonometry:
        // cos(total_angle) and sin(total_angle) are total_cm_x/total_radius and
        // total_cm_y/total_radius by construction, so the rotation is one dot and one
        // cross product scaled by 1/total_radius. That replaces three `atan2`, three
        // `sin`, three `cos` and three `sqrt` per pixel with a single `sqrt`.
        let (c, sn) = if total_radius == 0.0 {
            (1.0, 0.0)
        } else {
            (total_cm_x / total_radius, total_cm_y / total_radius)
        };
        let rot = |x: f32, y: f32| (x * c + y * sn, y * c - x * sn);
        let (arx, ary) = rot(red_cm_x, red_cm_y);
        let (agx, agy) = rot(green_cm_x, green_cm_y);
        let (abx, aby) = rot(blue_cm_x, blue_cm_y);

        descriptor.center_x = center_x as u16;
        descriptor.center_y = center_y as u16;

        let q = |v: f32| f32::round(v * 100.0) as i16;
        descriptor.feature_vector = [q(arx), q(ary), q(agx), q(agy), q(abx), q(aby)];

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

/// The same sums as [`DiscSums`], but with the position moments kept in absolute image
/// coordinates so that they survive a shift of the disc.
///
/// `DiscSums` weights each pixel by its offset *from the centre*, which changes for every
/// retained pixel when the disc moves, so those moments cannot be updated incrementally.
/// The absolute moments `A = sum(px*v)` and `B = sum(py*v)` can be, and the relative ones
/// come back exactly as `Wx = A - x*S` and `Wy = B - y*S`.
#[derive(Default, Clone, Copy)]
struct AbsSums {
    s: [i64; 3],
    ax: [i64; 3],
    ay: [i64; 3],
}

impl AbsSums {
    #[inline(always)]
    fn rgb(data: &[u8], px: isize, py: isize, w: isize) -> (i64, i64, i64) {
        let p = ((py * w + px) * 4) as usize;
        // SAFETY: only reached for centres at least `radius` from every edge, and
        // `half <= radius`, so `px` is in `0..w` and `py` in `0..h`.
        unsafe {
            (
                *data.get_unchecked(p) as i64,
                *data.get_unchecked(p + 1) as i64,
                *data.get_unchecked(p + 2) as i64,
            )
        }
    }

    #[inline(always)]
    fn add(&mut self, px: isize, py: isize, data: &[u8], w: isize) {
        let (r, g, b) = Self::rgb(data, px, py, w);
        let (pxi, pyi) = (px as i64, py as i64);
        for (c, v) in [r, g, b].into_iter().enumerate() {
            self.s[c] += v;
            self.ax[c] += pxi * v;
            self.ay[c] += pyi * v;
        }
    }

    #[inline(always)]
    fn sub(&mut self, px: isize, py: isize, data: &[u8], w: isize) {
        let (r, g, b) = Self::rgb(data, px, py, w);
        let (pxi, pyi) = (px as i64, py as i64);
        for (c, v) in [r, g, b].into_iter().enumerate() {
            self.s[c] -= v;
            self.ax[c] -= pxi * v;
            self.ay[c] -= pyi * v;
        }
    }

    #[inline(always)]
    fn to_disc_sums(self, x: isize, y: isize) -> DiscSums {
        let (xi, yi) = (x as i64, y as i64);
        let rel = |c: usize| {
            (
                (self.ax[c] - xi * self.s[c]) as isize,
                (self.ay[c] - yi * self.s[c]) as isize,
            )
        };
        let (wx_red, wy_red) = rel(0);
        let (wx_green, wy_green) = rel(1);
        let (wx_blue, wy_blue) = rel(2);
        DiscSums {
            sum_red: self.s[0] as isize,
            sum_green: self.s[1] as isize,
            sum_blue: self.s[2] as isize,
            wx_red,
            wy_red,
            wx_green,
            wy_green,
            wx_blue,
            wy_blue,
        }
    }
}
