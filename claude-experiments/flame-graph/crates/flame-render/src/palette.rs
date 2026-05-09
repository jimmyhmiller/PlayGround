//! 32-entry hue-rotated palette in the spirit of Brendan Gregg's flamegraph.pl.
//! Names hash to a stable index. We swing through warm colors, biased toward
//! oranges/yellows for that classic flame look.

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
