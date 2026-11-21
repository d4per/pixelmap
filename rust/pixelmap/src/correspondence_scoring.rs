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
///
/// The scorer itself holds nothing that changes, so that the bands of a pass can share
/// it across threads; each band brings one of these as its scratch, and its count of
/// comparisons with it.
#[derive(Default)]
pub(crate) struct Samples {
    /// The RGBA bytes of `photo1`'s surviving disc pixels, four per sample, copied as
    /// they are so that a disc row is a single `memcpy`.
    p1: Vec<u8>,
    /// Pixel index into `p2_words` for the *unshifted* transform.
    p2_idx: Vec<i32>,
    /// How many entries of `p1`/`p2_idx` the current disc filled.
    len: usize,
    /// How many pixels of the disc lie inside `photo1`: what there was to compare before
    /// any of it was lost to the bounds of `photo2`. `len` is the part of this that
    /// landed inside `photo2`.
    offered: usize,
    /// Comparisons made through this scratch so far, five per scored candidate.
    pub(crate) scored: usize,
}

impl Samples {
    /// Scratch for discs of the given radius.
    pub(crate) fn new(radius: usize) -> Self {
        let capacity = (2 * radius + 1) * (2 * radius + 1);
        Samples {
            p1: vec![0u8; 4 * capacity],
            p2_idx: vec![0i32; capacity],
            len: 0,
            offered: 0,
            scored: 0,
        }
    }
}

/// The least share of a disc that has to land inside `photo2` for a comparison to count,
/// as a fraction `(numerator, denominator)` of the pixels `photo1` offered.
///
/// A score is the variance of the colour differences over the samples that survive, and
/// says nothing about how many did. Left unchecked, that makes a transform better the
/// less of it can be checked: with a single sample the variance is exactly zero, the best
/// score there is, and with a handful it is usually far below any genuine match.
///
/// Such transforms arise where the mapping runs off the edge of `photo2`. The cells there
/// pile up against the edge, the matrices derived from them come out compressed, and a
/// compressed transform aimed at the edge keeps only the tip of its disc. It then beats
/// the genuine transform of every cell it is handed to, and because moving along the edge
/// changes nothing about it, it runs the length of a grid column or row. The round-trip
/// check deletes those cells afterwards, which leaves a straight unmapped band one cell
/// wide through an otherwise well mapped area.
///
/// One half amounts to requiring that the centre of the disc lands inside `photo2`. It is
/// measured against what `photo1` offered rather than against the full disc, because a
/// cell on the border of `photo1` has a smaller disc to begin with and must stay
/// mappable.
const MIN_COMPARED: (usize, usize) = (1, 2);

impl Samples {
    /// Whether enough of the disc was resolved for its score to mean anything; see
    /// [`MIN_COMPARED`].
    fn is_comparable(&self) -> bool {
        self.len > 0 && self.len * MIN_COMPARED.1 >= self.offered * MIN_COMPARED.0
    }
}

/// Everything the scorer reads from an [`AffineTransform`], in the form it works in: the
/// disc centre, and the Q16.16 steps and base of the walk through `photo2`.
///
/// [`CorrespondenceScoring::resolve_disc`] takes its inputs from here and from
/// nowhere else. Two transforms with equal `FixedTransform`s therefore resolve exactly
/// the same samples and score exactly the same, however their floats differ below the
/// fixed-point resolution. The solver relies on that to recognize a candidate a cell has
/// already evaluated without scoring it again; see
/// `CorrespondenceMappingAlgorithm::run_queue`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct FixedTransform {
    origin_x: u16,
    origin_y: u16,
    /// The mapped position advances by this for one pixel along a disc row.
    step_x: i32,
    step_y: i32,
    /// ...and by this for one disc row.
    col_x: i32,
    col_y: i32,
    /// The mapped disc centre, with `HALF` folded in so a shift rounds.
    base_x: i32,
    base_y: i32,
}

impl FixedTransform {
    /// The only floating point the scorer touches: six conversions per transform.
    pub(crate) fn new(cm: &AffineTransform) -> Self {
        FixedTransform {
            origin_x: cm.origin_x,
            origin_y: cm.origin_y,
            step_x: (cm.a11 * 65536.0) as i32,
            step_y: (cm.a21 * 65536.0) as i32,
            col_x: (cm.a12 * 65536.0) as i32,
            col_y: (cm.a22 * 65536.0) as i32,
            base_x: ((cm.translate_x * 65536.0) as i32).wrapping_add(HALF),
            base_y: ((cm.translate_y * 65536.0) as i32).wrapping_add(HALF),
        }
    }
}

/// The `CorrespondenceScoring` struct is responsible for evaluating the similarity between two photos
/// using pixel comparisons within a circular neighborhood. This similarity is computed based on color
/// differences between corresponding regions of the photos.
///
/// Pixels are read straight from the photos' RGBA bytes, four at a time as one
/// little-endian word, `0xAABBGGRR`: a single 32-bit load rather than three byte loads
/// and two shifts. The alpha byte comes along in the top eight bits, where [`Acc::add`]
/// never looks. This used to read a second, packed copy of each photo instead, which at
/// 3200 px cost 61 MB for the pair and bought no speed.
pub struct CorrespondenceScoring {
    /// A shared reference to the first photo.
    photo1: Arc<Photo>,
    /// A shared reference to the second photo.
    photo2: Arc<Photo>,
    /// Precomputed table of maximum x-offsets for each y-offset in the circular neighborhood.
    sqrt_table: Vec<i32>,
    /// The offset from the centre of every pixel of a full disc, `x` in `disc_dx` and `y`
    /// in `disc_dy`, in the order the walk visits them: row by row, left to right.
    disc_dx: Vec<i32>,
    disc_dy: Vec<i32>,
    /// Radius of the circular neighborhood used for comparisons.
    neighborhood_radius: i32,
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
    /// for each y-offset in the circular neighborhood, and lists the pixels of a full disc
    /// for the fast path.
    pub(crate) fn new(photo1: Arc<Photo>, photo2: Arc<Photo>, neighborhood_radius: isize) -> Self {
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

        // Row by row and left to right, the order both paths of `resolve_disc` emit a
        // full disc in.
        let (disc_dx, disc_dy): (Vec<i32>, Vec<i32>) = (-neighborhood_radius..=neighborhood_radius)
            .flat_map(|dy| {
                let xx = sqrt_table[(dy + neighborhood_radius) as usize];
                (-xx..=xx).map(move |dx| (dx, dy))
            })
            .unzip();

        CorrespondenceScoring {
            photo1,
            photo2,
            sqrt_table,
            disc_dx,
            disc_dy,
            neighborhood_radius,
        }
    }

    /// Resolves the circular neighbourhood of the transform `fixed` was made from into `s`.
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
    /// per row plus one index computation over the whole disc table, with no test at
    /// all. Only a disc that genuinely straddles an edge takes the general path, which
    /// clamps each row to `photo1` and, for a row that also straddles a `photo2` edge,
    /// tests every pixel.
    ///
    /// `allow_fast` is there for the tests, which run the general path over inputs the
    /// fast path would claim and assert the two agree; production always passes `true`.
    fn resolve_disc(&self, fixed: &FixedTransform, s: &mut Samples, allow_fast: bool) {
        s.len = 0;
        s.offered = 0;

        let w2 = self.photo2.width;
        let h2 = self.photo2.height;
        // The ±1 shift needs a one pixel margin on every side.
        if w2 < 3 || h2 < 3 {
            return;
        }

        let radius = self.neighborhood_radius;

        // Everything below is integer and comes from `fixed`, which is what makes a
        // `FixedTransform` a complete key for the result.
        let FixedTransform {
            origin_x,
            origin_y,
            step_x,
            step_y,
            col_x,
            col_y,
            base_x,
            base_y,
        } = *fixed;
        let x1 = origin_x as i32;
        let y1 = origin_y as i32;

        let w1 = self.photo1.width as i32;
        let h1 = self.photo1.height as i32;

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
        // Disc pixels inside `photo1`, counted by the general path as it clamps each row.
        // The fast path keeps every pixel of a full disc, so there it is simply `out`.
        let mut offered = 0usize;

        if fully_inside {
            for y in y_lo..=y_hi {
                // SAFETY: `y + radius` is in `0..2r+1`, the length of `sqrt_table`.
                let xx = unsafe { *self.sqrt_table.get_unchecked((y + radius) as usize) };
                let n = (2 * xx + 1) as usize;
                let row1_pix = ((y1 + y) * w1 + x1 - xx) as usize;

                // The disc's bounding box is inside `photo1`, so this row is one
                // contiguous run of its pixels; and the disc yields at most `(2r+1)^2`
                // samples, which is what `p1` was sized to.
                s.p1[4 * out..4 * (out + n)]
                    .copy_from_slice(&self.photo1.img_data[4 * row1_pix..4 * (row1_pix + n)]);
                out += n;
            }

            // The `photo2` indices in one flat pass over the disc table rather than one
            // short ramp per row: rows of at most 2r+1 pixels left a scalar remainder on
            // nearly every one, and a single loop over the whole disc vectorizes almost
            // without one. Each position is computed from the centre rather than advanced
            // row by row, but in wrapping arithmetic both reach the same integer,
            // `base + dy * col + dx * step`.
            for ((p2, &dx), &dy) in s.p2_idx[..out]
                .iter_mut()
                .zip(&self.disc_dx[..out])
                .zip(&self.disc_dy[..out])
            {
                let fx = base_x
                    .wrapping_add(dy.wrapping_mul(col_x))
                    .wrapping_add(dx.wrapping_mul(step_x));
                let fy = base_y
                    .wrapping_add(dy.wrapping_mul(col_y))
                    .wrapping_add(dx.wrapping_mul(step_y));
                *p2 = (fx >> FRAC) + (fy >> FRAC) * w2i;
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
                    offered += span as usize + 1;
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

                        // `row1_pix + i` is a pixel of `photo1` by the row clamp above,
                        // and the disc yields at most `(2r+1)^2` samples, which is what
                        // `p1` and `p2_idx` were sized to.
                        let from = 4 * (row1_pix + i);
                        s.p1[4 * out..4 * out + 4]
                            .copy_from_slice(&self.photo1.img_data[from..from + 4]);
                        // SAFETY: as above.
                        unsafe {
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
        s.offered = if fully_inside { out } else { offered };
    }

    /// [`Self::resolve_disc`] of an [`AffineTransform`], for the tests.
    #[cfg(test)]
    fn resolve_disc_inner(&self, cm: &AffineTransform, s: &mut Samples, allow_fast: bool) {
        self.resolve_disc(&FixedTransform::new(cm), s, allow_fast);
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
        let p2 = &self.photo2.img_data[..];
        let mut acc = [Acc::default(); K];

        for j in 0..n {
            // SAFETY: `j < n`, and `p1` holds the four bytes of each of the `n` samples;
            // and `resolve_disc` insets photo2 by one
            // pixel on every side, so every sample shifted by at most one pixel in
            // either axis is still a pixel of `photo2`: its four bytes are in `p2`.
            unsafe {
                let w1: [u8; 4] = s.p1.get_unchecked(4 * j..4 * j + 4).try_into().unwrap();
                let w1 = u32::from_le_bytes(w1);
                let idx = *s.p2_idx.get_unchecked(j) + base;
                for k in 0..K {
                    let at = 4 * (idx + offs[k]) as usize;
                    // A slice of exactly four bytes, so the conversion cannot fail and
                    // compiles to a single load.
                    let px: [u8; 4] = p2.get_unchecked(at..at + 4).try_into().unwrap();
                    acc[k].add(w1, u32::from_le_bytes(px));
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
    ///
    /// A transform that puts too little of its disc inside `photo2` is not scored at all
    /// and comes back with `f32::MAX`, which no cell accepts; see [`MIN_COMPARED`].
    ///
    /// `fixed` must be `FixedTransform::new(cm)`. The caller passes it in because the
    /// solver already holds it, as the key it recognizes repeated candidates by, and
    /// converting the transform twice per candidate was measurable. `s` is the caller's
    /// scratch; the comparisons are counted in it.
    pub(crate) fn optimize_translation(
        &self,
        cm: &AffineTransform,
        fixed: &FixedTransform,
        s: &mut Samples,
    ) -> (f32, AffineTransform) {
        debug_assert!(*fixed == FixedTransform::new(cm));
        self.resolve_disc(fixed, s, true);
        if !s.is_comparable() {
            return (f32::MAX, *cm);
        }
        s.scored += 5;

        let [score_x1, score_0, score_x2] = self.score_taps(s, 0, [-1, 0, 1]);

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
        let [score_y1, score_y2] = self.score_taps(s, delta, [-w, w]);
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
        Arc::new(Photo {
            img_data,
            width,
            height,
        })
    }

    fn empty_samples(radius: usize) -> Samples {
        Samples::new(radius)
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
        let full_disc: usize = scoring
            .sqrt_table
            .iter()
            .map(|xx| (2 * xx + 1) as usize)
            .sum();

        for cm in random_transforms(w, h, 4000, 0xdead_beef_0bad_f00d) {
            scoring.resolve_disc_inner(&cm, &mut fast, true);
            scoring.resolve_disc_inner(&cm, &mut general, false);

            assert_eq!(fast.len, general.len, "sample count differs for {cm:?}");
            assert_eq!(
                fast.offered, general.offered,
                "the two paths disagree on what photo1 offered for {cm:?}"
            );
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
        assert!(
            fast_path_taken > 100,
            "fast path almost never ran ({fast_path_taken} of 4000)"
        );
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
                assert!(
                    px >= 1 && px <= w as i32 - 2,
                    "x {px} outside the inset for {cm:?}"
                );
                assert!(
                    py >= 1 && py <= h as i32 - 2,
                    "y {py} outside the inset for {cm:?}"
                );
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
            assert_eq!(
                acc.s2,
                sr2 + sg2 + sb2,
                "combined sum of squares is not exact"
            );

            // The formula as it stood before the collapse, divide and all.
            let ni = n as i64;
            let reference = ((ni * sr2 as i64 - (acc.sr as i64) * (acc.sr as i64) + ni * sg2 as i64
                - (acc.sg as i64) * (acc.sg as i64)
                + ni * sb2 as i64
                - (acc.sb as i64) * (acc.sb as i64)) as f64
                / (ni * ni) as f64) as f32;
            let got = acc.finish(n, 1.0 / ((n as f64) * (n as f64)));
            let ulps = (reference.to_bits() as i64 - got.to_bits() as i64).abs();
            assert!(
                ulps <= 1,
                "finish drifted by {ulps} ulps: {reference} vs {got}"
            );
        }
    }

    fn identity_at(origin: (u16, u16), target: (f32, f32)) -> AffineTransform {
        AffineTransform {
            origin_x: origin.0,
            origin_y: origin.1,
            translate_x: target.0,
            translate_y: target.1,
            a11: 1.0,
            a12: 0.0,
            a21: 0.0,
            a22: 1.0,
        }
    }

    /// The transform behind the straight unmapped bands, as found in a real run: aimed
    /// 0.15 px short of the right edge of `photo2` and compressed, so that only the tip of
    /// its disc lands inside. One sample has a variance of zero, which used to make this
    /// the best-scoring transform a cell could be offered.
    #[test]
    fn a_disc_that_all_but_leaves_photo2_is_not_scored() {
        let (w, h, radius) = (64usize, 48usize, 5usize);
        let scoring = CorrespondenceScoring::new(
            noise_photo(w, h, 0x1357_9bdf_0246_8ace),
            noise_photo(w, h, 0xeca8_6420_fdb9_7531),
            radius as isize,
        );
        let sliver = AffineTransform {
            a11: 0.328,
            a21: 0.569,
            ..identity_at((30, 24), (w as f32 - 0.15, 24.0))
        };

        let mut s = empty_samples(radius);
        scoring.resolve_disc_inner(&sliver, &mut s, true);
        assert_eq!(
            (s.len, s.offered),
            (1, 81),
            "this is meant to be the one-sample case"
        );

        let mut probe = empty_samples(radius);
        let (score, returned) =
            scoring.optimize_translation(&sliver, &FixedTransform::new(&sliver), &mut probe);
        assert_eq!(score, f32::MAX, "a single sample was given a score");
        assert_eq!(returned, sliver);
        assert_eq!(
            probe.scored, 0,
            "nothing was compared, so nothing should be counted"
        );
    }

    /// The limit is on the share of the disc that is lost to `photo2`, so it must not
    /// touch a transform that is merely close to the edge...
    #[test]
    fn a_disc_centred_inside_photo2_is_still_scored() {
        let (w, h, radius) = (64usize, 48usize, 5usize);
        let scoring = CorrespondenceScoring::new(
            noise_photo(w, h, 0x1357_9bdf_0246_8ace),
            noise_photo(w, h, 0xeca8_6420_fdb9_7531),
            radius as isize,
        );
        // Centre three pixels in from the right edge: the disc is cut, but not in half.
        let near_edge = identity_at((30, 24), (w as f32 - 3.0, 24.0));

        let mut s = empty_samples(radius);
        scoring.resolve_disc_inner(&near_edge, &mut s, true);
        assert!(s.len < s.offered, "the edge should cut this disc");
        assert!(
            scoring
                .optimize_translation(
                    &near_edge,
                    &FixedTransform::new(&near_edge),
                    &mut empty_samples(5)
                )
                .0
                < f32::MAX
        );
    }

    /// ...nor a cell in the corner of `photo1`, whose disc is a quarter of the usual size
    /// before `photo2` is even considered.
    #[test]
    fn a_cell_on_the_border_of_photo1_is_still_scored() {
        let (w, h, radius) = (64usize, 48usize, 5usize);
        let scoring = CorrespondenceScoring::new(
            noise_photo(w, h, 0x1357_9bdf_0246_8ace),
            noise_photo(w, h, 0xeca8_6420_fdb9_7531),
            radius as isize,
        );
        let corner = identity_at((0, 0), (20.0, 20.0));

        let mut s = empty_samples(radius);
        scoring.resolve_disc_inner(&corner, &mut s, true);
        assert!(s.offered < 81 / 2, "a corner cell has a small disc");
        assert_eq!(s.len, s.offered, "all of it lands inside photo2");
        assert!(
            scoring
                .optimize_translation(
                    &corner,
                    &FixedTransform::new(&corner),
                    &mut empty_samples(5)
                )
                .0
                < f32::MAX
        );
    }

    /// The solver skips a candidate whose [`FixedTransform`] equals one a cell has already
    /// evaluated, on the grounds that it would score the same. This pins that: transforms
    /// that differ only below the fixed-point resolution resolve the same samples and
    /// come back with the same score and the same shift.
    #[test]
    fn equal_fixed_transforms_score_the_same() {
        let (w, h, radius) = (64usize, 48usize, 5usize);
        let scoring = CorrespondenceScoring::new(
            noise_photo(w, h, 0x2468_ace0_1357_9bdf),
            noise_photo(w, h, 0xfdb9_7531_0eca_8642),
            radius as isize,
        );
        let (mut a, mut b) = (empty_samples(radius), empty_samples(radius));
        let mut twins = 0usize;

        for cm in random_transforms(w, h, 4000, 0x0bad_cafe_dead_beef) {
            // The next float up: far below one Q16.16 unit for coordinates this small, so
            // most of these land on the same fixed-point value. Those that do not are
            // simply not twins and are skipped.
            let nudged = AffineTransform {
                translate_x: f32::from_bits(cm.translate_x.to_bits() + 1),
                a12: f32::from_bits(cm.a12.to_bits() + 1),
                ..cm
            };
            if FixedTransform::new(&cm) != FixedTransform::new(&nudged) {
                continue;
            }
            assert_ne!(cm, nudged, "the nudge has to change the floats");
            twins += 1;

            scoring.resolve_disc_inner(&cm, &mut a, true);
            scoring.resolve_disc_inner(&nudged, &mut b, true);
            assert_eq!(a.len, b.len, "sample count differs for {cm:?}");
            assert_eq!(a.p1[..a.len], b.p1[..b.len]);
            assert_eq!(a.p2_idx[..a.len], b.p2_idx[..b.len]);

            let (score, out) =
                scoring.optimize_translation(&cm, &FixedTransform::new(&cm), &mut empty_samples(5));
            let (score_twin, out_twin) = scoring.optimize_translation(
                &nudged,
                &FixedTransform::new(&nudged),
                &mut empty_samples(5),
            );
            assert_eq!(score.to_bits(), score_twin.to_bits());
            // Rounded, because `(t + 1.0) - t` is not always exactly 1.0 in `f32`.
            let shift = |from: &AffineTransform, to: &AffineTransform| {
                (
                    (to.translate_x - from.translate_x).round(),
                    (to.translate_y - from.translate_y).round(),
                )
            };
            assert_eq!(
                shift(&cm, &out),
                shift(&nudged, &out_twin),
                "the two picked different shifts"
            );
        }
        assert!(twins > 1000, "only {twins} of 4000 pairs shared a key");
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
