//! OkLab / OkLCh ↔ linear sRGB conversions.
//!
//! Hand-rolled from Björn Ottosson's reference: <https://bottosson.github.io/posts/oklab/>.
//! ~60 lines of math is much lighter than pulling in the `palette` crate
//! just for two color spaces.
//!
//! ## Convention
//!
//! - L is on **[0, 100]** in our public API (theme.rhai authors say
//!   `oklch(70, 0.15, 280)`, not `oklch(0.70, ...)`). Internally we
//!   divide by 100 for the math.
//! - a and b are roughly **[-0.4, +0.4]** in OkLab.
//! - C (chroma) is roughly **[0, 0.4]**, h is **degrees [0, 360)**.
//! - The output is `bevy::color::LinearRgba` with alpha defaulted to 1.

use bevy::color::LinearRgba;

/// `oklab(L, a, b)` → linear sRGB, alpha = 1.
pub fn oklab_to_linear_srgb(l_pct: f32, a: f32, b: f32) -> LinearRgba {
    let l = l_pct * 0.01;
    let l_ = l + 0.3963377774 * a + 0.2158037573 * b;
    let m_ = l - 0.1055613458 * a - 0.0638541728 * b;
    let s_ = l - 0.0894841775 * a - 1.2914855480 * b;
    let l3 = l_ * l_ * l_;
    let m3 = m_ * m_ * m_;
    let s3 = s_ * s_ * s_;
    let r = 4.0767416621 * l3 - 3.3077115913 * m3 + 0.2309699292 * s3;
    let g = -1.2684380046 * l3 + 2.6097574011 * m3 - 0.3413193965 * s3;
    let b_lin = -0.0041960863 * l3 - 0.7034186147 * m3 + 1.7076147010 * s3;
    LinearRgba::new(
        r.clamp(0.0, 1.0),
        g.clamp(0.0, 1.0),
        b_lin.clamp(0.0, 1.0),
        1.0,
    )
}

/// `oklch(L, C, h_degrees)` → linear sRGB, alpha = 1.
pub fn oklch_to_linear_srgb(l_pct: f32, c: f32, h_degrees: f32) -> LinearRgba {
    let h = h_degrees.to_radians();
    let a = c * h.cos();
    let b = c * h.sin();
    oklab_to_linear_srgb(l_pct, a, b)
}

/// Linear sRGB → OkLab, returning `(L%, a, b)`.
pub fn linear_srgb_to_oklab(c: LinearRgba) -> (f32, f32, f32) {
    let r = c.red.max(0.0);
    let g = c.green.max(0.0);
    let b = c.blue.max(0.0);
    let l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b;
    let m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b;
    let s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b;
    let l_ = l.cbrt();
    let m_ = m.cbrt();
    let s_ = s.cbrt();
    let big_l = 0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_;
    let big_a = 1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_;
    let big_b = 0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_;
    (big_l * 100.0, big_a, big_b)
}

/// Linear sRGB → OkLCh, returning `(L%, C, h_degrees in [0, 360))`.
pub fn linear_srgb_to_oklch(c: LinearRgba) -> (f32, f32, f32) {
    let (l, a, b) = linear_srgb_to_oklab(c);
    let chroma = (a * a + b * b).sqrt();
    let mut h = b.atan2(a).to_degrees();
    if h < 0.0 {
        h += 360.0;
    }
    (l, chroma, h)
}

/// Perceptual-lightness difference between two colors, as OkLab L
/// (each in [0, 100]). Returns absolute difference. Good cheap
/// substitute for WCAG luminance contrast — a ΔL ≥ 50 is roughly
/// equivalent to WCAG AA for body text on background.
pub fn lightness_delta(a: LinearRgba, b: LinearRgba) -> f32 {
    let (la, _, _) = linear_srgb_to_oklab(a);
    let (lb, _, _) = linear_srgb_to_oklab(b);
    (la - lb).abs()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() < eps
    }

    #[test]
    fn white_roundtrip() {
        let white = LinearRgba::new(1.0, 1.0, 1.0, 1.0);
        let (l, a, b) = linear_srgb_to_oklab(white);
        assert!(approx_eq(l, 100.0, 0.5));
        assert!(approx_eq(a, 0.0, 0.01));
        assert!(approx_eq(b, 0.0, 0.01));
        let back = oklab_to_linear_srgb(l, a, b);
        assert!(approx_eq(back.red, 1.0, 0.01));
        assert!(approx_eq(back.green, 1.0, 0.01));
        assert!(approx_eq(back.blue, 1.0, 0.01));
    }

    #[test]
    fn oklch_hue_rotation_preserves_lightness() {
        // Two colors at the same L should be perceptually equally bright.
        let red = oklch_to_linear_srgb(70.0, 0.15, 30.0);
        let blue = oklch_to_linear_srgb(70.0, 0.15, 250.0);
        let (lr, _, _) = linear_srgb_to_oklab(red);
        let (lb, _, _) = linear_srgb_to_oklab(blue);
        assert!(approx_eq(lr, lb, 0.5), "L mismatch {} vs {}", lr, lb);
    }
}
