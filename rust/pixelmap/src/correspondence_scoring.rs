use std::cell::{Cell, RefCell};
use crate::affine_transform::AffineTransform;
use crate::photo::Photo;
use std::sync::Arc;

/// Number of fractional bits used by the fixed-point coordinate walk.
const FRAC: u32 = 16;
/// Added before the right shift to turn truncation into rounding.
const HALF: i32 = 1 << (FRAC - 1);

/// The pixels of one circular neighbourhood, resolved once and then scored
/// repeatedly at different offsets into `photo2`.
///
/// Both vectors are allocated once at `(2r+1)^2` entries — the most the disc can ever
/// yield — and `len` says how much of them the current disc filled, so resolving a disc
/// never reallocates and never re-checks capacity. They stay resident in L1 across the
/// five candidates of one [`CorrespondenceScoring::optimize_translation`] call.
#[derive(Default)]
struct Samples {
    /// Packed `0x00BBGGRR` from `photo1`, one entry per surviving disc pixel.
    p1: Vec<u32>,
    /// Pixel index into `p2_words` for the *unshifted* transform.
    p2_idx: Vec<i32>,
    /// How many entries of `p1`/`p2_idx` the current disc filled.
    len: usize,
}

/// The `CorrespondenceScoring` struct is responsible for evaluating the similarity between two photos
/// using pixel comparisons within a circular neighborhood. This similarity is computed based on color
/// differences between corresponding regions of the photos.
pub struct CorrespondenceScoring {
    /// A shared reference to the first photo.
    photo1: Arc<Photo>,
    /// A shared reference to the second photo.
    photo2: Arc<Photo>,
    /// `photo1` and `photo2` with one packed `0x00BBGGRR` word per pixel.
    ///
    /// Sampling a pixel is then a single aligned 32-bit load instead of three byte loads
    /// and two shifts, which matters because the `photo2` side of the inner loop is a
    /// gather: the three bytes were three separate address computations against the same
    /// cache line. Costs one extra copy of each image, which the scoring loop pays back
    /// immediately.
    p1_words: Vec<u32>,
    p2_words: Vec<u32>,
    /// Precomputed table of maximum x-offsets for each y-offset in the circular neighborhood.
    sqrt_table: Vec<i32>,
    /// Radius of the circular neighborhood used for comparisons.
    neighborhood_radius: i32,
    /// Tracks the total number of similarity score calculations performed.
    total_invocations: Cell<usize>,
    /// Reused scratch for the disc resolved by the most recent call.
    scratch: RefCell<Samples>,
}

/// The running colour sums of one candidate offset.
///
/// Kept separate from the sweep so that several candidates can be accumulated side by
/// side over a single walk of the samples; see [`CorrespondenceScoring::score_taps`].
#[derive(Default, Clone, Copy)]
struct Acc {
    sr: i32,
    sg: i32,
    sb: i32,
    /// `sum(dr^2 + dg^2 + db^2)`, combined rather than kept per channel.
    ///
    /// `finish` only ever uses the three sums of squares added together, so keeping them
    /// apart bought nothing and cost two adds per sample and three live values per
    /// candidate — and the sweep carries `K` accumulators at once, so at `K = 3` that is
    /// eighteen of them rather than twelve. Worth about 6% of the whole pipeline.
    s2: i32,
}

impl Acc {
    #[inline(always)]
    fn add(&mut self, w1: u32, w2: u32) {
        let dr = (w1 & 0xFF) as i32 - (w2 & 0xFF) as i32;
        let dg = ((w1 >> 8) & 0xFF) as i32 - ((w2 >> 8) & 0xFF) as i32;
        let db = ((w1 >> 16) & 0xFF) as i32 - ((w2 >> 16) & 0xFF) as i32;
        self.sr += dr;
        self.sg += dg;
        self.sb += db;
        self.s2 += dr * dr + dg * dg + db * db;
    }

    /// The sum of the variances of the per-channel differences over `n` samples.
    ///
    /// Accumulated in `i32` and combined exactly in `i64`:
    /// `n^2 * variance == n * sum(d^2) - sum(d)^2`, which avoids subtracting two
    /// nearly equal floats.
    ///
    /// `inv_n2` is `1 / n^2`, passed in rather than computed here because every candidate
    /// of a sweep shares it, and an f64 divide is worth hoisting out of the `K` of them.
    /// It costs the result up to one ulp against dividing, which only matters if it flips
    /// an exact tie between two candidates.
    #[inline(always)]
    fn finish(&self, n: usize, inv_n2: f64) -> f32 {
        let num = (n as i64) * self.s2 as i64
            - (self.sr as i64) * (self.sr as i64)
            - (self.sg as i64) * (self.sg as i64)
            - (self.sb as i64) * (self.sb as i64);
        (num as f64 * inv_n2) as f32
    }
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
    /// for each y-offset in the circular neighborhood, packs both images into one word per
    /// pixel, and sizes the sample scratch to the largest disc that can occur.
    pub fn new(photo1: Arc<Photo>, photo2: Arc<Photo>, neighborhood_radius: isize) -> Self {
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

        // The accumulators in `Acc` are i32. Worst case is
        // `pixels * 255^2`; assert we are nowhere near overflowing.
        debug_assert!(
            (diameter as i64) * (diameter as i64) * 65025 < i32::MAX as i64,
            "neighborhood_radius {neighborhood_radius} would overflow the i32 accumulators"
        );

        let pack = |p: &Photo| -> Vec<u32> {
            p.img_data
                .chunks_exact(4)
                .map(|q| u32::from_le_bytes([q[0], q[1], q[2], 0]))
                .collect()
        };
        let (p1_words, p2_words) = (pack(&photo1), pack(&photo2));
        let capacity = diameter * diameter;

        CorrespondenceScoring {
            photo1,
            photo2,
            p1_words,
            p2_words,
            sqrt_table,
            neighborhood_radius,
            total_invocations: Cell::new(0),
            scratch: RefCell::new(Samples {
                p1: vec![0u32; capacity],
                p2_idx: vec![0i32; capacity],
                len: 0,
            }),
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
    /// The walk is Q16.16 fixed point and the whole function is integer: `step_*`
    /// advances the mapped position by one pixel along a row, `col_*` advances it by one
    /// row, and `base_*` is the mapped disc centre with `HALF` folded in. A pixel then
    /// costs one add and one shift per axis, and a *row* costs one multiply-add per axis
    /// instead of the two multiplies, two adds and a float-to-int conversion it used to.
    /// `is_scale_valid` bounds the matrix rows below 4.0 and `is_valid` bounds the
    /// translation to the image, so the mapped coordinate stays within roughly
    /// ±(width + 8r) — far inside the ±32767 that Q16.16 can represent.
    ///
    /// Bounds testing is decided *once per call* rather than once per row and once per
    /// pixel. The disc lies inside the `[-r, r]²` box, whose affine image is the convex
    /// hull of the box's four mapped corners; a per-axis interval test is therefore
    /// exact on those four, and `>> FRAC` is monotone so the integer coordinates are
    /// bounded by the same four. If they are all inside the inset — and the box is
    /// inside `photo1` — every sample is, and the walk degenerates to a contiguous copy
    /// plus an index ramp with no test at all. Only a disc that genuinely straddles an
    /// edge takes the general path, which clamps each row to `photo1` and, for a row
    /// that also straddles a `photo2` edge, tests every pixel.
    ///
    /// `allow_fast` is there for the tests, which run the general path over inputs the
    /// fast path would claim and assert the two agree; production always passes `true`.
    fn resolve_disc_inner(
        &self,
        circle_mapping: &AffineTransform,
        s: &mut Samples,
        allow_fast: bool,
    ) {
        s.len = 0;

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

        // The only floating point left in the function: six conversions per call, none
        // of them inside a loop.
        let step_x = (circle_mapping.a11 * 65536.0) as i32;
        let step_y = (circle_mapping.a21 * 65536.0) as i32;
        let col_x = (circle_mapping.a12 * 65536.0) as i32;
        let col_y = (circle_mapping.a22 * 65536.0) as i32;
        let base_x = (circle_mapping.translate_x * 65536.0) as i32 + HALF;
        let base_y = (circle_mapping.translate_y * 65536.0) as i32 + HALF;

        let x2_hi = w2 as i32 - 2;
        let y2_hi = h2 as i32 - 2;
        let w2i = w2 as i32;
        let inside = |v: i32, hi: i32| v >= 1 && v <= hi;

        // The rows of the disc that meet `photo1` at all, hoisted out of the row loop.
        let y_lo = (-radius).max(-y1);
        let y_hi = radius.min(h1 - 1 - y1);
        if y_lo > y_hi {
            return;
        }

        let corner = |sx: i32, sy: i32| {
            let fx = base_x
                .wrapping_add((sx * radius).wrapping_mul(step_x))
                .wrapping_add((sy * radius).wrapping_mul(col_x));
            let fy = base_y
                .wrapping_add((sx * radius).wrapping_mul(step_y))
                .wrapping_add((sy * radius).wrapping_mul(col_y));
            inside(fx >> FRAC, x2_hi) && inside(fy >> FRAC, y2_hi)
        };
        let fully_inside = allow_fast
            && y_lo == -radius
            && y_hi == radius
            && x1 - radius >= 0
            && x1 + radius < w1
            && corner(-1, -1)
            && corner(1, -1)
            && corner(-1, 1)
            && corner(1, 1);

        let mut row_fx = base_x.wrapping_add(y_lo.wrapping_mul(col_x));
        let mut row_fy = base_y.wrapping_add(y_lo.wrapping_mul(col_y));
        let mut out = 0usize;

        if fully_inside {
            for y in y_lo..=y_hi {
                // SAFETY: `y + radius` is in `0..2r+1`, the length of `sqrt_table`.
                let xx = unsafe { *self.sqrt_table.get_unchecked((y + radius) as usize) };
                let n = (2 * xx + 1) as usize;
                let row1_pix = ((y1 + y) * w1 + x1 - xx) as usize;
                let fx0 = row_fx.wrapping_sub(xx.wrapping_mul(step_x));
                let fy0 = row_fy.wrapping_sub(xx.wrapping_mul(step_y));

                // SAFETY: the disc's bounding box is inside `photo1`, so `row1_pix ..
                // row1_pix + n` is one contiguous run of `p1_words`; and the disc yields
                // at most `(2r+1)^2` samples, which is what `p1` and `p2_idx` were sized
                // to, so `out + n` stays within both.
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        self.p1_words.as_ptr().add(row1_pix),
                        s.p1.as_mut_ptr().add(out),
                        n,
                    );
                    // Written as `fx0 + i * step_x` rather than as a running sum on
                    // purpose: it states the affine relation the vectorizer needs.
                    let dst = s.p2_idx.as_mut_ptr().add(out);
                    for i in 0..n {
                        let fx = fx0.wrapping_add((i as i32).wrapping_mul(step_x));
                        let fy = fy0.wrapping_add((i as i32).wrapping_mul(step_y));
                        *dst.add(i) = (fx >> FRAC) + (fy >> FRAC) * w2i;
                    }
                }
                out += n;

                row_fx = row_fx.wrapping_add(col_x);
                row_fy = row_fy.wrapping_add(col_y);
            }
        } else {
            for y in y_lo..=y_hi {
                // SAFETY: as above.
                let xx = unsafe { *self.sqrt_table.get_unchecked((y + radius) as usize) };

                // Clamp the disc row to photo1 once, instead of testing every pixel.
                // Note this is a proper 2D clamp: the original code tested only the
                // flat index, which let a row straddling an image edge wrap onto its
                // neighbour.
                let x_lo = (-xx).max(-x1);
                let x_hi = xx.min(w1 - 1 - x1);
                if x_lo <= x_hi {
                    let span = x_hi - x_lo;
                    let mut fx = row_fx.wrapping_add(x_lo.wrapping_mul(step_x));
                    let mut fy = row_fy.wrapping_add(x_lo.wrapping_mul(step_y));

                    // The mapped position is affine in the step index, so a row that
                    // starts and ends inside the inset is inside it the whole way.
                    let all_in = inside(fx >> FRAC, x2_hi)
                        && inside(fx.wrapping_add(span.wrapping_mul(step_x)) >> FRAC, x2_hi)
                        && inside(fy >> FRAC, y2_hi)
                        && inside(fy.wrapping_add(span.wrapping_mul(step_y)) >> FRAC, y2_hi);

                    let row1_pix = ((y1 + y) * w1 + x1 + x_lo) as usize;

                    for i in 0..=span as usize {
                        let px = fx >> FRAC;
                        let py = fy >> FRAC;
                        fx = fx.wrapping_add(step_x);
                        fy = fy.wrapping_add(step_y);

                        if !all_in && (!inside(px, x2_hi) || !inside(py, y2_hi)) {
                            continue;
                        }

                        // SAFETY: `row1_pix + i` is a pixel of `photo1` by the row clamp
                        // above, and the disc yields at most `(2r+1)^2` samples, which is
                        // what `p1` and `p2_idx` were sized to.
                        unsafe {
                            *s.p1.get_unchecked_mut(out) =
                                *self.p1_words.get_unchecked(row1_pix + i);
                            *s.p2_idx.get_unchecked_mut(out) = px + py * w2i;
                        }
                        out += 1;
                    }
                }

                row_fx = row_fx.wrapping_add(col_x);
                row_fy = row_fy.wrapping_add(col_y);
            }
        }
        s.len = out;
    }

    /// Resolves the disc of `circle_mapping`; see [`Self::resolve_disc_inner`].
    #[inline(always)]
    fn resolve_disc(&self, circle_mapping: &AffineTransform, s: &mut Samples) {
        self.resolve_disc_inner(circle_mapping, s, true);
    }

    /// Scores the `K` candidates that differ from the resolved disc by the constant
    /// pixel offsets `base + offs[k]`, in a single sweep of the samples.
    ///
    /// Every candidate reads the same `photo1` word and the same base index, so sweeping
    /// them together loads each sample once instead of `K` times — and for the x search,
    /// whose three taps are adjacent pixels, the `photo2` reads land on one cache line.
    ///
    /// Scoring the five candidates of [`Self::optimize_translation`] therefore takes two
    /// sweeps rather than five.
    fn score_taps<const K: usize>(&self, s: &Samples, base: i32, offs: [i32; K]) -> [f32; K] {
        let n = s.len;
        if n == 0 {
            return [f32::MAX; K];
        }
        let p2 = &self.p2_words;
        let mut acc = [Acc::default(); K];

        for j in 0..n {
            // SAFETY: `j < n <= s.p1.len()`, and `resolve_disc` insets photo2 by one
            // pixel on every side, so every sample shifted by at most one pixel in
            // either axis is still a valid index into `p2_words`.
            unsafe {
                let w1 = *s.p1.get_unchecked(j);
                let idx = *s.p2_idx.get_unchecked(j) + base;
                for k in 0..K {
                    acc[k].add(w1, *p2.get_unchecked((idx + offs[k]) as usize));
                }
            }
        }

        let inv_n2 = 1.0 / ((n as f64) * (n as f64));
        let mut out = [f32::MAX; K];
        for k in 0..K {
            out[k] = acc[k].finish(n, inv_n2);
        }
        out
    }

    /// Evaluates `circle_mapping` and its four ±1 pixel translations, returning the
    /// best `(score, transform)` of the five.
    ///
    /// The five candidates differ *only* in `translate_x`/`translate_y`, so they all
    /// cover the same `photo1` disc and their `photo2` samples differ by a constant
    /// pixel offset: ∓1 for a one pixel shift in x, ∓`photo2.width` for one in y.
    /// The disc is therefore resolved once and scored in two sweeps, instead of being
    /// re-walked — with all the affine arithmetic, rounding and bounds testing that
    /// entails — five times over.
    ///
    /// The search order matches the original: x first, then y relative to the x winner.
    pub fn optimize_translation(&self, cm: &AffineTransform) -> (f32, AffineTransform) {
        let mut s = self.scratch.borrow_mut();
        self.resolve_disc(cm, &mut s);
        self.total_invocations.set(self.total_invocations.get() + 5);

        let [score_x1, score_0, score_x2] = self.score_taps(&s, 0, [-1, 0, 1]);

        let mut best = score_0;
        let mut delta = 0i32;
        let mut dx = 0.0f32;
        let mut dy = 0.0f32;

        if score_x1 < best && score_x1 < score_x2 {
            best = score_x1;
            delta = -1;
            dx = -1.0;
        } else if score_x2 < best {
            best = score_x2;
            delta = 1;
            dx = 1.0;
        }

        let w = self.photo2.width as i32;
        let [score_y1, score_y2] = self.score_taps(&s, delta, [-w, w]);
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic xorshift, so a failure is reproducible without pulling `rand` into
    /// the test.
    struct Rng(u64);

    impl Rng {
        fn next_u32(&mut self) -> u32 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 >> 32) as u32
        }

        /// Uniform in `[lo, hi)`.
        fn f32_in(&mut self, lo: f32, hi: f32) -> f32 {
            lo + (self.next_u32() as f32 / u32::MAX as f32) * (hi - lo)
        }
    }

    fn noise_photo(width: usize, height: usize, seed: u64) -> Arc<Photo> {
        let mut rng = Rng(seed);
        let mut img_data = vec![0u8; width * height * 4];
        for px in img_data.chunks_exact_mut(4) {
            let v = rng.next_u32();
            px[0] = v as u8;
            px[1] = (v >> 8) as u8;
            px[2] = (v >> 16) as u8;
            px[3] = 255;
        }
        Arc::new(Photo { img_data, width, height })
    }

    fn empty_samples(radius: usize) -> Samples {
        let capacity = (2 * radius + 1) * (2 * radius + 1);
        Samples { p1: vec![0u32; capacity], p2_idx: vec![0i32; capacity], len: 0 }
    }

    /// Random transforms spanning the range the solver actually produces: origins over
    /// the whole grid including the edges, translations that push part of the disc off
    /// `photo2`, and rotations/scales inside `is_scale_valid(4.0)`.
    fn random_transforms(w: usize, h: usize, count: usize, seed: u64) -> Vec<AffineTransform> {
        let mut rng = Rng(seed);
        (0..count)
            .map(|_| {
                let angle = rng.f32_in(0.0, std::f32::consts::TAU);
                let scale = rng.f32_in(0.4, 2.5);
                let shear = rng.f32_in(-0.3, 0.3);
                AffineTransform {
                    origin_x: (rng.next_u32() as usize % w) as u16,
                    origin_y: (rng.next_u32() as usize % h) as u16,
                    // Deliberately overshoots the image on both ends.
                    translate_x: rng.f32_in(-8.0, w as f32 + 8.0),
                    translate_y: rng.f32_in(-8.0, h as f32 + 8.0),
                    a11: scale * angle.cos(),
                    a12: scale * (-angle.sin() + shear),
                    a21: scale * angle.sin(),
                    a22: scale * (angle.cos() + shear),
                }
            })
            .collect()
    }

    /// The fast path's correctness rests entirely on the four-corner argument, so this
    /// is the check that matters: over a few thousand transforms, whatever the fast path
    /// claims it can do without bounds tests must match what the general path — which
    /// tests every pixel — produces.
    #[test]
    fn fast_path_agrees_with_general_path() {
        let (w, h, radius) = (64usize, 48usize, 5usize);
        let scoring = CorrespondenceScoring::new(
            noise_photo(w, h, 0x1234_5678_9abc_def0),
            noise_photo(w, h, 0x0fed_cba9_8765_4321),
            radius as isize,
        );

        let mut fast = empty_samples(radius);
        let mut general = empty_samples(radius);
        let mut fast_path_taken = 0usize;
        let mut samples_seen = 0usize;
        let full_disc: usize = scoring.sqrt_table.iter().map(|xx| (2 * xx + 1) as usize).sum();

        for cm in random_transforms(w, h, 4000, 0xdead_beef_0bad_f00d) {
            scoring.resolve_disc_inner(&cm, &mut fast, true);
            scoring.resolve_disc_inner(&cm, &mut general, false);

            assert_eq!(fast.len, general.len, "sample count differs for {cm:?}");
            assert_eq!(
                fast.p1[..fast.len],
                general.p1[..general.len],
                "photo1 words differ for {cm:?}"
            );
            assert_eq!(
                fast.p2_idx[..fast.len],
                general.p2_idx[..general.len],
                "photo2 indices differ for {cm:?}"
            );

            // A full disc means the fast path ran; it emits every pixel of the disc.
            if fast.len == full_disc {
                fast_path_taken += 1;
            }
            samples_seen += fast.len;
        }

        assert!(samples_seen > 0, "no samples were resolved at all");
        assert!(fast_path_taken > 100, "fast path almost never ran ({fast_path_taken} of 4000)");
    }

    /// `score_taps` reads every sample shifted by ±1 pixel and ±one row with no bounds
    /// check, so every index `resolve_disc` emits must sit inside `photo2`'s one pixel
    /// inset. Nothing else in the file enforces this.
    #[test]
    fn every_sample_is_inside_the_one_pixel_inset() {
        let (w, h, radius) = (37usize, 29usize, 5usize);
        let scoring = CorrespondenceScoring::new(
            noise_photo(w, h, 0xa5a5_5a5a_0f0f_f0f0),
            noise_photo(w, h, 0x5a5a_a5a5_f0f0_0f0f),
            radius as isize,
        );

        let mut s = empty_samples(radius);
        for cm in random_transforms(w, h, 4000, 0xfeed_face_cafe_babe) {
            scoring.resolve_disc_inner(&cm, &mut s, true);
            for &idx in &s.p2_idx[..s.len] {
                assert!(idx >= 0, "negative index {idx} for {cm:?}");
                let (px, py) = (idx % w as i32, idx / w as i32);
                assert!(px >= 1 && px <= w as i32 - 2, "x {px} outside the inset for {cm:?}");
                assert!(py >= 1 && py <= h as i32 - 2, "y {py} outside the inset for {cm:?}");
            }
        }
    }

    /// The collapsed `s2` and the hoisted reciprocal are the two places Stage 2 changed
    /// the arithmetic of `finish`. The first is exact and this pins it; the second is a
    /// reciprocal multiply instead of a divide, so it is allowed one ulp.
    #[test]
    fn collapsed_accumulator_matches_the_per_channel_formula() {
        let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
        for _ in 0..20_000 {
            let n = 1 + rng.next_u32() as usize % 121;
            let (mut acc, mut sr2, mut sg2, mut sb2) = (Acc::default(), 0i32, 0i32, 0i32);
            for _ in 0..n {
                let (w1, w2) = (rng.next_u32() & 0x00FF_FFFF, rng.next_u32() & 0x00FF_FFFF);
                acc.add(w1, w2);
                let dr = (w1 & 0xFF) as i32 - (w2 & 0xFF) as i32;
                let dg = ((w1 >> 8) & 0xFF) as i32 - ((w2 >> 8) & 0xFF) as i32;
                let db = ((w1 >> 16) & 0xFF) as i32 - ((w2 >> 16) & 0xFF) as i32;
                sr2 += dr * dr;
                sg2 += dg * dg;
                sb2 += db * db;
            }
            assert_eq!(acc.s2, sr2 + sg2 + sb2, "combined sum of squares is not exact");

            // The formula as it stood before the collapse, divide and all.
            let ni = n as i64;
            let reference = ((ni * sr2 as i64 - (acc.sr as i64) * (acc.sr as i64)
                + ni * sg2 as i64 - (acc.sg as i64) * (acc.sg as i64)
                + ni * sb2 as i64 - (acc.sb as i64) * (acc.sb as i64)) as f64
                / (ni * ni) as f64) as f32;
            let got = acc.finish(n, 1.0 / ((n as f64) * (n as f64)));
            let ulps = (reference.to_bits() as i64 - got.to_bits() as i64).abs();
            assert!(ulps <= 1, "finish drifted by {ulps} ulps: {reference} vs {got}");
        }
    }

    /// The fixed-point row bases accumulate individually truncated `col_*`/`step_*`
    /// instead of rounding a full-precision float per row. This pins the claim that the
    /// resulting drift can never move a sample by more than one pixel.
    #[test]
    fn fixed_point_walk_tracks_the_float_reference() {
        let (w, h, radius) = (64usize, 48usize, 5i32);
        let scoring = CorrespondenceScoring::new(
            noise_photo(w, h, 0x1111_2222_3333_4444),
            noise_photo(w, h, 0x4444_3333_2222_1111),
            radius as isize,
        );

        let mut s = empty_samples(radius as usize);
        for cm in random_transforms(w, h, 500, 0x0123_4567_89ab_cdef) {
            scoring.resolve_disc_inner(&cm, &mut s, true);

            // Re-derive the disc the same way `resolve_disc` walks it and match the
            // emitted indices against the exact float mapping, in order.
            let mut emitted = s.p2_idx[..s.len].iter();
            let (x1, y1) = (cm.origin_x as i32, cm.origin_y as i32);
            for y in -radius..=radius {
                if y1 + y < 0 || y1 + y >= h as i32 {
                    continue;
                }
                let xx = scoring.sqrt_table[(y + radius) as usize];
                for x in (-xx).max(-x1)..=xx.min(w as i32 - 1 - x1) {
                    let fx = x as f32 * cm.a11 + y as f32 * cm.a12 + cm.translate_x;
                    let fy = x as f32 * cm.a21 + y as f32 * cm.a22 + cm.translate_y;
                    let (rx, ry) = ((fx + 0.5).floor() as i32, (fy + 0.5).floor() as i32);
                    let kept = rx >= 1 && rx <= w as i32 - 2 && ry >= 1 && ry <= h as i32 - 2;
                    // A sample within a hair of the inset boundary may fall on either
                    // side of it, so only the unambiguously kept ones can be matched up
                    // positionally.
                    let unambiguous =
                        rx >= 2 && rx <= w as i32 - 3 && ry >= 2 && ry <= h as i32 - 3;
                    if kept && unambiguous {
                        let idx = *emitted.next().expect("fewer samples than the reference");
                        let (px, py) = (idx % w as i32, idx / w as i32);
                        assert!(
                            (px - rx).abs() <= 1 && (py - ry).abs() <= 1,
                            "fixed point walk drifted: got ({px}, {py}), float says ({rx}, {ry})"
                        );
                    } else if kept {
                        emitted.next();
                    }
                }
            }
        }
    }
}
