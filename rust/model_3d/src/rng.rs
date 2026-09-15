//! A small seeded generator for RANSAC's sampling.
//!
//! `pixelmap` keeps its own generator private, and pulling in `rand` for a handful of
//! index draws is not worth the dependency. What matters here is only that the stream is
//! a pure function of the seed, so that the same correspondence map always produces the
//! same model.

/// SplitMix64: tiny, fast, and statistically more than good enough to pick sample sets.
pub(crate) struct Rng(u64);

impl Rng {
    pub(crate) fn new(seed: u64) -> Self {
        Rng(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `0..n`. `n` must be non-zero.
    pub(crate) fn below(&mut self, n: usize) -> usize {
        ((self.next_u64() as u128 * n as u128) >> 64) as usize
    }

    /// Fills `out` with `k` distinct indices from `0..n`. Requires `k <= n`; RANSAC only
    /// ever asks for 4 or 8 out of many, so rejection is cheaper than a partial shuffle.
    pub(crate) fn sample_distinct(&mut self, n: usize, k: usize, out: &mut Vec<usize>) {
        debug_assert!(k <= n, "cannot draw {k} distinct indices from {n}");
        out.clear();
        while out.len() < k {
            let i = self.below(n);
            if !out.contains(&i) {
                out.push(i);
            }
        }
    }
}
