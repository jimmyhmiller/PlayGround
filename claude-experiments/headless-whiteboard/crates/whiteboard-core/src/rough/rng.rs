//! Seeded pseudo-random generator for sketch generation.
//!
//! Rough.js seeds a tiny LCG-style generator and pulls `[0, 1)` doubles from it;
//! the exact constants below match Rough.js's `Random` implementation so that a
//! given seed reproduces the same sketch as upstream wherever possible. The
//! generator is intentionally simple and fully deterministic — no global state,
//! no time, no OS RNG — which is exactly what headless + snapshot testing needs.

/// Rough.js-compatible seeded RNG.
#[derive(Debug, Clone)]
pub struct RoughRng {
    seed: u32,
}

impl RoughRng {
    pub fn new(seed: u32) -> Self {
        RoughRng { seed }
    }

    /// Next pseudo-random `f64` in `[0, 1)`.
    ///
    /// Matches Rough.js: `seed = (2**31 - 1) & (seed * 1103515245 + 12345)`,
    /// returning `(seed & 0x3fffffff) / 0x40000000`.
    pub fn next_f64(&mut self) -> f64 {
        // Use wrapping arithmetic on u64 then mask to 31 bits, mirroring JS's
        // implicit double math followed by the `& 0x7fffffff` in Rough.js.
        let next = (self.seed as u64)
            .wrapping_mul(1103515245)
            .wrapping_add(12345);
        self.seed = (next & 0x7fff_ffff) as u32;
        ((self.seed & 0x3fff_ffff) as f64) / (0x4000_0000 as f64)
    }

    /// Next value in `[min, max)`.
    pub fn range(&mut self, min: f64, max: f64) -> f64 {
        min + (max - min) * self.next_f64()
    }

    /// A symmetric offset in `[-range, range)`, optionally scaled by roughness.
    /// Port of Rough.js's `_offset`.
    pub fn offset(&mut self, range: f64, roughness: f64) -> f64 {
        roughness * (self.next_f64() * 2.0 * range - range)
    }

    /// Offset within `[min, max)` scaled by roughness. Port of `_offsetOpt`.
    pub fn offset_opt(&mut self, min: f64, max: f64, roughness: f64) -> f64 {
        roughness * (self.range(min, max))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn values_in_unit_interval() {
        let mut r = RoughRng::new(42);
        for _ in 0..1000 {
            let v = r.next_f64();
            assert!((0.0..1.0).contains(&v), "out of range: {v}");
        }
    }

    #[test]
    fn deterministic_for_seed() {
        let first: Vec<f64> = {
            let mut r = RoughRng::new(7);
            (0..10).map(|_| r.next_f64()).collect()
        };
        let second: Vec<f64> = {
            let mut r = RoughRng::new(7);
            (0..10).map(|_| r.next_f64()).collect()
        };
        assert_eq!(first, second);
    }

    #[test]
    fn range_respects_bounds() {
        let mut r = RoughRng::new(99);
        for _ in 0..1000 {
            let v = r.range(-5.0, 5.0);
            assert!((-5.0..5.0).contains(&v));
        }
    }

    #[test]
    fn offset_is_symmetric_in_magnitude() {
        let mut r = RoughRng::new(3);
        for _ in 0..1000 {
            let v = r.offset(2.0, 1.0);
            assert!(v.abs() <= 2.0 + 1e-9);
        }
    }
}
