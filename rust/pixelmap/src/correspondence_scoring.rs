use std::cell::{Cell, RefCell};
use crate::affine_transform::AffineTransform;
use crate::photo::Photo;
use std::rc::Rc;

/// Number of fractional bits used by the fixed-point coordinate walk.
const FRAC: u32 = 16;
/// Added before the right shift to turn truncation into rounding.
const HALF: i32 = 1 << (FRAC - 1);

/// The pixels of one circular neighbourhood, resolved once and then scored
/// repeatedly at different offsets into `photo2`.
///
/// Both vectors are indexed in lockstep and never grow past `(2r+1)^2` entries,
/// so they stay resident in L1 across the five candidates of one
/// [`CorrespondenceScoring::optimize_translation`] call.
#[derive(Default)]
struct Samples {
    /// Packed `0x00BBGGRR` from `photo1`, one entry per surviving disc pixel.
    p1: Vec<u32>,
    /// Byte offset into `photo2.img_data` for the *unshifted* transform.
    p2_idx: Vec<u32>,
}

/// The `CorrespondenceScoring` struct is responsible for evaluating the similarity between two photos
/// using pixel comparisons within a circular neighborhood. This similarity is computed based on color
/// differences between corresponding regions of the photos.
pub struct CorrespondenceScoring {
    /// A shared reference to the first photo.
    photo1: Rc<Photo>,
    /// A shared reference to the second photo.
    photo2: Rc<Photo>,
    /// Precomputed table of maximum x-offsets for each y-offset in the circular neighborhood.
    sqrt_table: Vec<i32>,
    /// Radius of the circular neighborhood used for comparisons.
    neighborhood_radius: i32,
    /// Tracks the total number of similarity score calculations performed.
    total_invocations: Cell<usize>,
    /// Reused scratch for the disc resolved by the most recent call.
    scratch: RefCell<Samples>,
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
        let neighborhood_radius = neighborhood_radius as i32;
        let diameter = (2 * neighborhood_radius + 1) as usize;
        let mut sqrt_table = vec![0i32; diameter];
        let radius2 = neighborhood_radius * neighborhood_radius;

        // Precompute the square root values for the circular boundary.
        for y in -neighborhood_radius..=neighborhood_radius {
            let idx = (y + neighborhood_radius) as usize;
            let yy = y * y;
            sqrt_table[idx] = f32::sqrt((radius2 - yy) as f32).floor() as i32;
        }

        // The accumulators in `score_at` are i32. Worst case is
        // `pixels * 255^2`; assert we are nowhere near overflowing.
        debug_assert!(
            (diameter as i64) * (diameter as i64) * 65025 < i32::MAX as i64,
            "neighborhood_radius {neighborhood_radius} would overflow the i32 accumulators"
        );

        CorrespondenceScoring {
            photo1,
            photo2,
            sqrt_table,
            neighborhood_radius,
            total_invocations: Cell::new(0),
            scratch: RefCell::new(Samples::default()),
        }
    }

    /// Resolves the circular neighbourhood of `circle_mapping` into `self.scratch`.
    ///
    /// Walks the disc once, mapping each `photo1` pixel through the affine transform
    /// to a rounded `photo2` pixel, and keeps only the pairs where *both* sides are in
    /// bounds. `photo2` is additionally inset by one pixel on every side so that the
    /// caller may shift the sampled position by ±1 in either axis — which is exactly
    /// what [`Self::optimize_translation`] does — without leaving the buffer.
    ///
    /// Coordinates in `photo2` are walked in Q16.16 fixed point. Along a row the mapped
    /// position advances by exactly `a11` (resp. `a21`) per step, so the per-pixel work
    /// is one add and one shift per axis instead of two multiplies, two adds and a
    /// `round`. `is_scale_valid` bounds the matrix rows below 4.0 and `is_valid` bounds
    /// the translation to the image, so the mapped coordinate stays within roughly
    /// ±(width + 8r) — far inside the ±32767 that Q16.16 can represent.
    fn resolve_disc(&self, circle_mapping: &AffineTransform, s: &mut Samples) {
        s.p1.clear();
        s.p2_idx.clear();

        let w2 = self.photo2.width;
        let h2 = self.photo2.height;
        // The ±1 shift needs a one pixel margin on every side.
        if w2 < 3 || h2 < 3 {
            return;
        }

        let radius = self.neighborhood_radius;
        let x1 = circle_mapping.origin_x as i32;
        let y1 = circle_mapping.origin_y as i32;

        let w1 = self.photo1.width as i32;
        let h1 = self.photo1.height as i32;
        let photo1_data = &self.photo1.img_data;

        let (kx1, kx2) = (circle_mapping.a11, circle_mapping.a12);
        let (ky1, ky2) = (circle_mapping.a21, circle_mapping.a22);
        let x2_offset = circle_mapping.translate_x;
        let y2_offset = circle_mapping.translate_y;

        let step_x = (kx1 * 65536.0) as i32;
        let step_y = (ky1 * 65536.0) as i32;

        let x2_lo = 1i32;
        let x2_hi = w2 as i32 - 2;
        let y2_lo = 1i32;
        let y2_hi = h2 as i32 - 2;
        let w2i = w2 as i32;

        for y in -radius..=radius {
            let yy1 = y1 + y;
            if yy1 < 0 || yy1 >= h1 {
                continue;
            }
            let xx = self.sqrt_table[(y + radius) as usize];

            // Clamp the disc row to photo1 once, instead of testing every pixel.
            // Note this is a proper 2D clamp: the original code tested only the
            // flat index, which let a row straddling an image edge wrap onto its
            // neighbour.
            let x_lo = (-xx).max(-x1);
            let x_hi = xx.min(w1 - 1 - x1);
            if x_lo > x_hi {
                continue;
            }

            let dy = y as f32;
            let row_x2 = (x_lo as f32) * kx1 + dy * kx2 + x2_offset;
            let row_y2 = (x_lo as f32) * ky1 + dy * ky2 + y2_offset;
            let mut fx = (row_x2 * 65536.0) as i32;
            let mut fy = (row_y2 * 65536.0) as i32;

            let row1_base = ((yy1 * w1 + x1 + x_lo) as usize) * 4;

            for i in 0..=(x_hi - x_lo) {
                let px = (fx.wrapping_add(HALF)) >> FRAC;
                let py = (fy.wrapping_add(HALF)) >> FRAC;
                fx = fx.wrapping_add(step_x);
                fy = fy.wrapping_add(step_y);

                if px < x2_lo || px > x2_hi || py < y2_lo || py > y2_hi {
                    continue;
                }

                let i1 = row1_base + (i as usize) * 4;
                // SAFETY: `x1 + x_lo + i` is in `0..w1` and `yy1` is in `0..h1` by the
                // clamps above, so `i1 + 3` is inside `photo1_data`.
                let w = unsafe {
                    u32::from_le_bytes([
                        *photo1_data.get_unchecked(i1),
                        *photo1_data.get_unchecked(i1 + 1),
                        *photo1_data.get_unchecked(i1 + 2),
                        0,
                    ])
                };
                s.p1.push(w);
                s.p2_idx.push(((px + py * w2i) as u32) * 4);
            }
        }
    }

    /// Scores the resolved disc with every `photo2` sample shifted by `delta` bytes.
    ///
    /// Returns the sum of the variances of the per-channel differences — the same
    /// quantity the original scalar implementation computed, but accumulated in `i32`
    /// and combined exactly in `i64`:
    /// `n^2 * variance == n * sum(d^2) - sum(d)^2`, which avoids subtracting two
    /// nearly equal floats.
    fn score_at(&self, s: &Samples, delta: i32) -> f32 {
        let n = s.p1.len();
        if n == 0 {
            return f32::MAX;
        }
        let photo2_data = &self.photo2.img_data;

        let (mut sr, mut sg, mut sb) = (0i32, 0i32, 0i32);
        let (mut sr2, mut sg2, mut sb2) = (0i32, 0i32, 0i32);

        for i in 0..n {
            // SAFETY: `resolve_disc` insets photo2 by one pixel on every side, so
            // every `p2_idx` shifted by at most one pixel in either axis is still a
            // valid 4-byte-aligned pixel offset inside `photo2_data`.
            let (w1, w2) = unsafe {
                let a = *s.p1.get_unchecked(i);
                let j = (*s.p2_idx.get_unchecked(i) as i32 + delta) as usize;
                let b = u32::from_le_bytes([
                    *photo2_data.get_unchecked(j),
                    *photo2_data.get_unchecked(j + 1),
                    *photo2_data.get_unchecked(j + 2),
                    0,
                ]);
                (a, b)
            };

            let dr = (w1 & 0xFF) as i32 - (w2 & 0xFF) as i32;
            let dg = ((w1 >> 8) & 0xFF) as i32 - ((w2 >> 8) & 0xFF) as i32;
            let db = ((w1 >> 16) & 0xFF) as i32 - ((w2 >> 16) & 0xFF) as i32;

            sr += dr;
            sg += dg;
            sb += db;
            sr2 += dr * dr;
            sg2 += dg * dg;
            sb2 += db * db;
        }

        let n = n as i64;
        let num = n * sr2 as i64 - (sr as i64) * (sr as i64)
            + n * sg2 as i64 - (sg as i64) * (sg as i64)
            + n * sb2 as i64 - (sb as i64) * (sb as i64);
        (num as f64 / (n * n) as f64) as f32
    }

    /// Evaluates `circle_mapping` and its four ±1 pixel translations, returning the
    /// best `(score, transform)` of the five.
    ///
    /// The five candidates differ *only* in `translate_x`/`translate_y`, so they all
    /// cover the same `photo1` disc and their `photo2` samples differ by a constant
    /// byte offset: ∓4 for a one pixel shift in x, ∓`4 * photo2.width` for one in y.
    /// The disc is therefore resolved once and scored five times, instead of being
    /// re-walked — with all the affine arithmetic, rounding and bounds testing that
    /// entails — five times over.
    ///
    /// The search order matches the original: x first, then y relative to the x winner.
    pub fn optimize_translation(&self, cm: &AffineTransform) -> (f32, AffineTransform) {
        let mut s = self.scratch.borrow_mut();
        self.resolve_disc(cm, &mut s);
        self.total_invocations.set(self.total_invocations.get() + 5);

        let mut best = self.score_at(&s, 0);
        let mut delta = 0i32;
        let mut dx = 0.0f32;
        let mut dy = 0.0f32;

        let score_x1 = self.score_at(&s, -4);
        let score_x2 = self.score_at(&s, 4);
        if score_x1 < best && score_x1 < score_x2 {
            best = score_x1;
            delta = -4;
            dx = -1.0;
        } else if score_x2 < best {
            best = score_x2;
            delta = 4;
            dx = 1.0;
        }

        let w4 = (self.photo2.width as i32) * 4;
        let score_y1 = self.score_at(&s, delta - w4);
        let score_y2 = self.score_at(&s, delta + w4);
        if score_y1 < best && score_y1 < score_y2 {
            best = score_y1;
            dy = -1.0;
        } else if score_y2 < best {
            best = score_y2;
            dy = 1.0;
        }

        (
            best,
            AffineTransform {
                translate_x: cm.translate_x + dx,
                translate_y: cm.translate_y + dy,
                ..*cm
            },
        )
    }

    /// Returns the total number of similarity score calculations performed.
    pub fn get_num_comparisons(&self) -> usize {
        self.total_invocations.get()
    }
}
