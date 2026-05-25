//! Color palette. Two flavors:
//!
//! - `color_for(name)`: Brendan-Gregg-style warm flame palette keyed by
//!   function name (red→yellow hue rotation).
//! - `color_for_blend(group_key, name)`: full hue rotation keyed by the
//!   group (e.g. service), with small saturation/value perturbation
//!   keyed by name. Different services land on visually distinct hues;
//!   spans within a service still vary so the flame texture survives.

use ahash::RandomState;
use std::hash::{BuildHasher, Hasher};

const PALETTE_SIZE: usize = 32;

const PALETTE: [[f32; 4]; PALETTE_SIZE] = generate_palette();

const fn generate_palette() -> [[f32; 4]; PALETTE_SIZE] {
    // Deterministic warm palette: rotate hue 0..60° (red→yellow), vary value/saturation.
    let mut out = [[0.0; 4]; PALETTE_SIZE];
    let mut i = 0;
    while i < PALETTE_SIZE {
        // Hue spans red (0°) to yellow (60°), occasionally pushing into orange-red.
        let t = i as f32 / PALETTE_SIZE as f32; // 0..1
        let hue_deg = 0.0 + t * 60.0;
        let sat = 0.65 + ((i % 3) as f32) * 0.07; // 0.65..0.79
        let val = 0.85 + ((i % 5) as f32) * 0.025; // 0.85..0.95
        let (r, g, b) = hsv_to_rgb(hue_deg, sat, val);
        out[i] = [r, g, b, 1.0];
        i += 1;
    }
    out
}

const fn hsv_to_rgb(h: f32, s: f32, v: f32) -> (f32, f32, f32) {
    // Const-fn-friendly HSV→RGB. h in degrees.
    let c = v * s;
    let h60 = h / 60.0;
    let h60_int = h60 as i32;
    let frac = h60 - h60_int as f32;
    let x = c * (1.0 - abs_const(frac * 2.0 - 1.0));
    let m = v - c;
    let (r1, g1, b1) = match h60_int {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };
    (r1 + m, g1 + m, b1 + m)
}

const fn abs_const(x: f32) -> f32 {
    if x < 0.0 { -x } else { x }
}

pub fn color_for(name: &str) -> [f32; 4] {
    // ahash with a fixed seed so colors are stable across runs.
    let mut hasher = RandomState::with_seeds(0xa11ce, 0xbeef, 0xcafe, 0xfade).build_hasher();
    hasher.write(name.as_bytes());
    let h = hasher.finish() as usize;
    PALETTE[h % PALETTE_SIZE]
}

fn hash_seeded(seeds: (u64, u64, u64, u64), bytes: &[u8]) -> u64 {
    let mut h = RandomState::with_seeds(seeds.0, seeds.1, seeds.2, seeds.3).build_hasher();
    h.write(bytes);
    h.finish()
}

/// Color = a base hue picked by `group_key` over the full 360° wheel, with
/// a small per-`name` perturbation on hue/saturation/value so spans within
/// a group remain visually distinguishable. Empty `group_key` falls through
/// to the warm-only `color_for(name)` (so non-OTel formats look unchanged).
pub fn color_for_blend(group_key: &str, name: &str) -> [f32; 4] {
    if group_key.is_empty() {
        return color_for(name);
    }
    // Group → base hue, full circle.
    let gh = hash_seeded((0xa11ce, 0xbeef, 0xcafe, 0xfade), group_key.as_bytes());
    // Golden-ratio hue stepping keeps consecutive groups visually far apart.
    let hue_base = ((gh % 360_000) as f32) / 1000.0;

    // Name → small jitter (±12° hue, slight S/V variation).
    let nh = hash_seeded((0xface, 0xb00c, 0x1234, 0x5678), name.as_bytes());
    let hue_jitter = ((nh & 0x3FF) as f32 / 1023.0 - 0.5) * 24.0;
    let sat = 0.62 + (((nh >> 10) & 0xFF) as f32 / 255.0) * 0.18; // 0.62..0.80
    let val = 0.82 + (((nh >> 18) & 0xFF) as f32 / 255.0) * 0.13; // 0.82..0.95

    let hue = (hue_base + hue_jitter).rem_euclid(360.0);
    let (r, g, b) = hsv_to_rgb_runtime(hue, sat, val);
    [r, g, b, 1.0]
}

fn hsv_to_rgb_runtime(h: f32, s: f32, v: f32) -> (f32, f32, f32) {
    let c = v * s;
    let h60 = h / 60.0;
    let h60_int = h60 as i32;
    let frac = h60 - h60_int as f32;
    let x = c * (1.0 - (frac * 2.0 - 1.0).abs());
    let m = v - c;
    let (r1, g1, b1) = match h60_int {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };
    (r1 + m, g1 + m, b1 + m)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stable_color_for_same_name() {
        assert_eq!(color_for("foo"), color_for("foo"));
    }

    #[test]
    fn palette_in_unit_range() {
        for c in PALETTE {
            for chan in &c[..3] {
                assert!(*chan >= 0.0 && *chan <= 1.0, "chan out of range: {chan}");
            }
        }
    }
}
