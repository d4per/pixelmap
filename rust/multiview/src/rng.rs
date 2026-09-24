//! The pipeline's own seeded random numbers.
//!
//! Every random choice (RANSAC samples, subsampling) draws from a generator derived from
//! one seed, so a whole reconstruction is as reproducible as the matcher underneath it.
//! This deliberately avoids `rand`: it would bring in `getrandom`, which needs a JavaScript
//! shim on wasm, and a seeded pipeline has no use for an entropy source.

/// SplitMix64. Small and fast, and more than good enough for picking RANSAC samples. Not
/// for anything cryptographic.
#[derive(Clone, Debug)]
pub struct Rng {
    state: u64,
}

impl Rng {
    /// A generator seeded with `seed`.
    pub fn new(seed: u64) -> Self {
        Rng { state: seed }
    }

    /// An independent generator for sub-task `stream`, such as one stage or one pair.
    ///
    /// Derived from this generator's current state without advancing it. Deriving from an
    /// untouched root means extra draws in one part of the pipeline never shift the
    /// numbers another part sees.
    pub fn derive(&self, stream: u64) -> Rng {
        Rng::new(mix(self.state ^ mix(stream.wrapping_add(GOLDEN_GAMMA))))
    }

    /// The next 64 uniformly distributed bits.
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(GOLDEN_GAMMA);
        mix(self.state)
    }

    /// Uniform in `[0, 1)`.
    pub fn next_f64(&mut self) -> f64 {
        unit(self.next_u64())
    }

    /// Uniform in `0..n`, by multiply-and-shift. The bias is at most `n / 2⁶⁴`, which is
    /// nothing at the sizes used here.
    ///
    /// # Panics
    /// If `n` is zero.
    pub fn below(&mut self, n: usize) -> usize {
        assert!(n > 0, "cannot draw from an empty range");
        ((self.next_u64() as u128 * n as u128) >> 64) as usize
    }

    /// Fills `out` with distinct indices below `n`, in no particular order.
    ///
    /// Meant for RANSAC's minimal samples, where `out` is a handful of elements and
    /// rejection is cheaper than shuffling.
    ///
    /// # Panics
    /// If `out` is longer than `n`.
    pub fn sample_distinct(&mut self, n: usize, out: &mut [usize]) {
        assert!(out.len() <= n, "cannot draw {} distinct of {n}", out.len());
        for i in 0..out.len() {
            out[i] = loop {
                let candidate = self.below(n);
                if !out[..i].contains(&candidate) {
                    break candidate;
                }
            };
        }
    }
}

/// Hashes three values into 64 well-mixed bits: randomness that has to be a pure function
/// of its inputs, such as noise keyed to a pixel position.
pub fn hash3(a: u64, b: u64, c: u64) -> u64 {
    mix(mix(mix(a.wrapping_add(GOLDEN_GAMMA)).wrapping_add(b)).wrapping_add(c))
}

/// Maps 64 random bits to a uniform value in `[0, 1)`.
pub fn unit(bits: u64) -> f64 {
    (bits >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
}

const GOLDEN_GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;

fn mix(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_the_reference_splitmix64_sequence() {
        let mut rng = Rng::new(0);
        assert_eq!(rng.next_u64(), 0xE220_A839_7B1D_CDAF);
    }

    #[test]
    fn is_deterministic_and_streams_are_independent() {
        let root = Rng::new(42);
        let draw = |mut r: Rng| (0..8).map(|_| r.next_u64()).collect::<Vec<_>>();
        assert_eq!(draw(root.derive(1)), draw(root.derive(1)));
        assert_ne!(draw(root.derive(1)), draw(root.derive(2)));
        assert_ne!(draw(root.derive(0)), draw(root.clone()));
    }

    #[test]
    fn draws_stay_in_range() {
        let mut rng = Rng::new(7);
        for _ in 0..10_000 {
            let f = rng.next_f64();
            assert!((0.0..1.0).contains(&f));
            assert!(rng.below(13) < 13);
        }
    }

    #[test]
    fn samples_are_distinct() {
        let mut rng = Rng::new(9);
        let mut sample = [0usize; 8];
        for _ in 0..1000 {
            rng.sample_distinct(10, &mut sample);
            let mut sorted = sample;
            sorted.sort_unstable();
            assert!(sorted.windows(2).all(|w| w[0] < w[1]));
            assert!(sorted[7] < 10);
        }
    }
}
