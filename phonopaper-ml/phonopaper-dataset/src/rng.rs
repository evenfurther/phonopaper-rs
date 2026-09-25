//! Small, self-contained pseudo-random number generator.
//!
//! The generator is implemented in-crate (rather than depending on `rand`) so
//! that the dataset is reproducible **forever**: a semantic-version bump of an
//! external RNG crate can never change the generated images.
//!
//! The algorithm is `xoshiro256**` seeded through `SplitMix64`, both of which
//! are public-domain designs with fully specified integer arithmetic.  All
//! floating-point helpers use only `+ - * /` and integer→float conversions,
//! which are exactly rounded under IEEE 754 and therefore bit-identical on
//! every platform.

/// `SplitMix64` step: turn an arbitrary `u64` into a well-mixed one.
#[must_use]
pub fn splitmix64(state: u64) -> u64 {
    let mut z = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// A `xoshiro256**` pseudo-random generator.
#[derive(Debug, Clone)]
pub struct Rng {
    s: [u64; 4],
}

impl Rng {
    /// Create a generator from a single 64-bit seed.
    ///
    /// The four words of internal state are derived with `SplitMix64` so any
    /// seed (including `0`) yields a valid, well-mixed state.
    #[must_use]
    pub fn from_seed(seed: u64) -> Self {
        let a = splitmix64(seed);
        let b = splitmix64(a);
        let c = splitmix64(b);
        let d = splitmix64(c);
        Self { s: [a, b, c, d] }
    }

    /// Create the independent per-item stream `index` of a dataset seeded
    /// with `dataset_seed`.
    ///
    /// Every item gets its own stream, so the dataset can be generated in any
    /// order (or in parallel) and each image is a pure function of
    /// `(dataset_seed, index)`.
    #[must_use]
    pub fn for_item(dataset_seed: u64, index: u64) -> Self {
        Self::from_seed(
            splitmix64(dataset_seed) ^ splitmix64(index.wrapping_mul(0xD1B5_4A32_D192_ED03)),
        )
    }

    /// Next raw 64-bit output.
    pub fn next_u64(&mut self) -> u64 {
        let result = self.s[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        result
    }

    /// Uniform `f64` in `[0, 1)` with 53 bits of precision.
    pub fn next_f64(&mut self) -> f64 {
        #[expect(
            clippy::cast_precision_loss,
            reason = "the value is shifted to 53 bits so the conversion is exact"
        )]
        let mantissa = (self.next_u64() >> 11) as f64;
        mantissa * (1.0 / 9_007_199_254_740_992.0) // 2^-53
    }

    /// Uniform `f64` in `[lo, hi)`.
    pub fn range_f64(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.next_f64()
    }

    /// Uniform integer in the **inclusive** range `[lo, hi]`.
    ///
    /// # Panics
    ///
    /// Panics if `lo > hi`.
    pub fn range_u32(&mut self, lo: u32, hi: u32) -> u32 {
        assert!(lo <= hi, "range_u32: lo ({lo}) > hi ({hi})");
        let span = u64::from(hi - lo) + 1;
        // Lemire-style multiply-shift; the tiny bias is irrelevant here and
        // the arithmetic is fully deterministic.
        let offset = u32::try_from(((self.next_u64() >> 32) * span) >> 32)
            .expect("(x >> 32) * span >> 32 is strictly below span ≤ 2^32");
        lo + offset
    }

    /// Uniform integer in the inclusive range `[lo, hi]`, as `usize`.
    ///
    /// # Panics
    ///
    /// Panics if `lo > hi` or if the bounds do not fit in `u32`.
    pub fn range_usize(&mut self, lo: usize, hi: usize) -> usize {
        let lo32 = u32::try_from(lo).expect("range_usize: lo does not fit in u32");
        let hi32 = u32::try_from(hi).expect("range_usize: hi does not fit in u32");
        self.range_u32(lo32, hi32) as usize
    }

    /// `true` with probability `p`.
    pub fn chance(&mut self, p: f64) -> bool {
        self.next_f64() < p
    }

    /// Approximately standard-normal sample (mean 0, standard deviation 1).
    ///
    /// Uses the sum of twelve uniforms (central-limit construction) rather than
    /// Box–Muller so that no transcendental function is involved.  The result
    /// is bounded to `[-6, 6]`, which is perfectly adequate for image noise.
    pub fn gaussian(&mut self) -> f64 {
        let mut acc = 0.0;
        for _ in 0..12 {
            acc += self.next_f64();
        }
        acc - 6.0
    }

    /// Pick a uniformly random element of a non-empty slice.
    ///
    /// # Panics
    ///
    /// Panics if `items` is empty.
    pub fn pick<'a, T>(&mut self, items: &'a [T]) -> &'a T {
        assert!(!items.is_empty(), "pick: empty slice");
        &items[self.range_usize(0, items.len() - 1)]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_seed_same_stream() {
        let mut a = Rng::from_seed(42);
        let mut b = Rng::from_seed(42);
        for _ in 0..1000 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn different_items_differ() {
        let mut a = Rng::for_item(1, 0);
        let mut b = Rng::for_item(1, 1);
        assert_ne!(a.next_u64(), b.next_u64());
    }

    #[test]
    fn reference_values_are_stable() {
        // Pin the first outputs so an accidental algorithm change is caught.
        let mut r = Rng::from_seed(0);
        let first = [r.next_u64(), r.next_u64(), r.next_u64()];
        assert_eq!(
            first,
            [
                1_905_207_664_160_064_169,
                7_642_312_046_547_803_776,
                7_003_759_831_383_473_959
            ]
        );
    }

    #[test]
    fn f64_in_unit_interval() {
        let mut r = Rng::from_seed(7);
        for _ in 0..10_000 {
            let v = r.next_f64();
            assert!((0.0..1.0).contains(&v));
        }
    }

    #[test]
    fn range_u32_inclusive_bounds_hit() {
        let mut r = Rng::from_seed(3);
        let mut seen_lo = false;
        let mut seen_hi = false;
        for _ in 0..10_000 {
            let v = r.range_u32(5, 7);
            assert!((5..=7).contains(&v));
            seen_lo |= v == 5;
            seen_hi |= v == 7;
        }
        assert!(seen_lo && seen_hi);
    }

    #[test]
    fn gaussian_is_roughly_standard() {
        let mut r = Rng::from_seed(9);
        let n = 20_000;
        let (mut sum, mut sq) = (0.0, 0.0);
        for _ in 0..n {
            let g = r.gaussian();
            sum += g;
            sq += g * g;
        }
        let mean = sum / f64::from(n);
        let var = sq / f64::from(n) - mean * mean;
        assert!(mean.abs() < 0.05, "mean = {mean}");
        assert!((var - 1.0).abs() < 0.05, "var = {var}");
    }

    #[test]
    fn pick_returns_element() {
        let mut r = Rng::from_seed(1);
        let items = [10, 20, 30];
        for _ in 0..100 {
            assert!(items.contains(r.pick(&items)));
        }
    }
}
