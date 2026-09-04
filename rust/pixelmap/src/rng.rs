//! A small deterministic PRNG, so the crate needs no `rand` dependency.
//!
//! The solver's only use for randomness is a Fisher-Yates shuffle whose stream has to be
//! reproducible from a seed (see [`crate::correspondence_mapping_algorithm`]). That does
//! not warrant a dependency, and `rand` is not free here: it pulls in `getrandom`, which
//! refuses to build for `wasm32-unknown-unknown` unless the *consumer* sets
//! `--cfg getrandom_backend="wasm_js"` in their own `.cargo/config.toml` — a file that is
//! not part of a published crate, so the breakage would land on them with no signal from
//! us.
//!
//! xoshiro256++ seeded through SplitMix64. Not cryptographic; it decides the order in
//! which equally good candidates are tried, nothing more.

/// A seeded xoshiro256++ generator.
pub(crate) struct Rng {
    s: [u64; 4],
}

/// SplitMix64, used to expand a single `u64` into the four words xoshiro needs.
/// Seeding xoshiro directly from a small integer leaves it in a low-entropy state that
/// takes a while to wash out; SplitMix64 is the author's recommended remedy.
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

impl Rng {
    /// Builds a generator whose stream is fully determined by `seed`.
    pub(crate) fn seed_from_u64(seed: u64) -> Self {
        let mut state = seed;
        Rng {
            s: [
                splitmix64(&mut state),
                splitmix64(&mut state),
                splitmix64(&mut state),
                splitmix64(&mut state),
            ],
        }
    }

    fn next_u64(&mut self) -> u64 {
        let result = self.s[0]
            .wrapping_add(self.s[3])
            .rotate_left(23)
            .wrapping_add(self.s[0]);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        result
    }

    /// Uniform in `[0, n)`, by Lemire's multiply-shift method: take the high half of a
    /// 128-bit product and reject only the rare low-half values that would bias the
    /// result. One multiply in the common case, no division.
    ///
    /// # Panics
    /// Debug-only: `n` must be non-zero.
    fn below(&mut self, n: u64) -> u64 {
        debug_assert!(n > 0, "below(0) has no valid result");
        let mut product = (self.next_u64() as u128) * (n as u128);
        let mut low = product as u64;
        if low < n {
            // Values below this threshold map to an over-represented bucket.
            let threshold = n.wrapping_neg() % n;
            while low < threshold {
                product = (self.next_u64() as u128) * (n as u128);
                low = product as u64;
            }
        }
        (product >> 64) as u64
    }

    /// Fisher-Yates, in place.
    pub(crate) fn shuffle<T>(&mut self, slice: &mut [T]) {
        for i in (1..slice.len()).rev() {
            let j = self.below(i as u64 + 1) as usize;
            slice.swap(i, j);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_same_seed_gives_the_same_stream() {
        let mut a = Rng::seed_from_u64(12345);
        let mut b = Rng::seed_from_u64(12345);
        for _ in 0..1000 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn different_seeds_diverge() {
        let mut a = Rng::seed_from_u64(1);
        let mut b = Rng::seed_from_u64(2);
        assert_ne!(
            (0..8).map(|_| a.next_u64()).collect::<Vec<_>>(),
            (0..8).map(|_| b.next_u64()).collect::<Vec<_>>()
        );
    }

    #[test]
    fn shuffle_is_a_permutation() {
        let mut rng = Rng::seed_from_u64(7);
        let mut values: Vec<u32> = (0..500).collect();
        rng.shuffle(&mut values);
        assert_ne!(values, (0..500).collect::<Vec<_>>(), "did not shuffle at all");
        values.sort_unstable();
        assert_eq!(values, (0..500).collect::<Vec<_>>(), "lost or duplicated elements");
    }

    #[test]
    fn shuffle_handles_degenerate_lengths() {
        let mut rng = Rng::seed_from_u64(7);
        rng.shuffle(&mut [0u8; 0]);
        let mut one = [42u8];
        rng.shuffle(&mut one);
        assert_eq!(one, [42]);
    }

    #[test]
    fn below_stays_in_range_and_covers_it() {
        let mut rng = Rng::seed_from_u64(99);
        let mut seen = [false; 6];
        for _ in 0..10_000 {
            let v = rng.below(6);
            assert!(v < 6, "below(6) returned {v}");
            seen[v as usize] = true;
        }
        assert!(seen.iter().all(|&s| s), "some values in [0, 6) were never produced");
    }
}
