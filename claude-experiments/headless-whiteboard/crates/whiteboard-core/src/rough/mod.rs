//! Hand-drawn ("sketchy") shape generation — a Rust port of Rough.js.
//!
//! Reimplemented from Rough.js (MIT, © 2019 Preet Shihn). Phase 1 fills in the
//! drawable generators (line/ellipse/rectangle/polygon/curve) and the fill
//! patterns (hachure / cross-hatch / zigzag / dots) on top of the seeded RNG
//! established here.
//!
//! Determinism is the whole game: a given element `seed` must always produce the
//! same sketch, so renders are stable and snapshot-testable.

mod fills;
mod generator;
mod rng;

pub use fills::{cross_hatch, dots, fill_polygon, hachure, zigzag, FillLine};
pub use generator::{
    rough_ellipse, rough_line, rough_linear_path, rough_polygon, rough_rectangle, Drawable,
};
pub use rng::RoughRng;

/// Knobs controlling how rough a generated shape looks. Mirrors Rough.js's
/// `Options`, trimmed to the fields the whiteboard uses.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RoughOptions {
    /// Numerical roughness; 0 = precise, ~1 = default sketch, higher = wilder.
    pub roughness: f64,
    /// How far control points may bow out, scaled with size.
    pub bowing: f64,
    /// Maximum random offset applied to endpoints, in scene units.
    pub max_randomness_offset: f64,
    /// Seed feeding the RNG (an element's `seed`).
    pub seed: u32,
    /// Spacing between hachure fill lines.
    pub hachure_gap: f64,
    /// Angle (degrees) of hachure fill lines.
    pub hachure_angle: f64,
    /// Number of times each stroke is drawn (Rough.js's `disableMultiStroke`
    /// off ⇒ 2).
    pub stroke_passes: u32,
}

impl Default for RoughOptions {
    fn default() -> Self {
        RoughOptions {
            roughness: 1.0,
            bowing: 1.0,
            max_randomness_offset: 2.0,
            seed: 1,
            hachure_gap: 4.0,
            hachure_angle: -41.0,
            stroke_passes: 2,
        }
    }
}

impl RoughOptions {
    /// Build options for an element with the given `roughness` (0/1/2 in
    /// Excalidraw) and `seed`. A roughness of 0 yields essentially clean output.
    pub fn for_element(roughness: f64, seed: u32) -> Self {
        RoughOptions {
            roughness,
            seed,
            ..Default::default()
        }
    }

    /// A fresh RNG seeded from these options.
    pub fn rng(&self) -> RoughRng {
        RoughRng::new(self.seed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_seed_same_sequence() {
        let opts = RoughOptions::for_element(1.0, 12345);
        let mut a = opts.rng();
        let mut b = opts.rng();
        for _ in 0..16 {
            assert_eq!(a.next_f64(), b.next_f64());
        }
    }

    #[test]
    fn different_seed_diverges() {
        let mut a = RoughOptions::for_element(1.0, 1).rng();
        let mut b = RoughOptions::for_element(1.0, 2).rng();
        // Extremely unlikely to match across several draws.
        let same = (0..8).all(|_| a.next_f64() == b.next_f64());
        assert!(!same);
    }
}
