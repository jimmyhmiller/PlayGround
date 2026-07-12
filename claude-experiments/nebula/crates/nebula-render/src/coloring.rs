//! Turning algorithm outputs into node colors. Two families:
//!   - **categorical** for labels (components, communities, proper coloring):
//!     evenly-spaced hues via the golden angle so adjacent ids look distinct.
//!   - **sequential** for scalars (PageRank, degree, BFS distance): a perceptual
//!     colormap (a compact "turbo"-style approximation).

use crate::scene::pack_rgba;

/// Distinct color for a categorical label using golden-angle hue spacing.
pub fn categorical(label: u32) -> u32 {
    // Golden angle in [0,1). Multiplying the label spreads hues maximally.
    let h = (label as f32 * 0.61803398875).fract();
    // Vary saturation/value slightly by label so same-hue collisions still differ.
    let s = 0.62 + ((label / 3) % 3) as f32 * 0.12;
    let v = 0.80 + ((label / 7) % 3) as f32 * 0.07;
    let (r, g, b) = hsv_to_rgb(h, s.min(0.95), v.min(0.98));
    pack_rgba(r, g, b, 255)
}

/// Build a categorical color buffer from labels.
pub fn categorical_colors(labels: &[u32]) -> Vec<u32> {
    labels.iter().map(|&l| categorical(l)).collect()
}

/// Map a scalar buffer to a sequential colormap, auto-normalizing to [min,max].
/// `log_scale` compresses heavy-tailed distributions (degree, PageRank).
pub fn sequential_colors_f32(values: &[f32], log_scale: bool) -> Vec<u32> {
    if values.is_empty() {
        return Vec::new();
    }
    let xform = |v: f32| if log_scale { (v.max(0.0) + 1.0).ln() } else { v };
    let mut lo = f32::MAX;
    let mut hi = f32::MIN;
    for &v in values {
        if v.is_finite() {
            let t = xform(v);
            lo = lo.min(t);
            hi = hi.max(t);
        }
    }
    let span = (hi - lo).max(1e-9);
    values
        .iter()
        .map(|&v| {
            if !v.is_finite() {
                return pack_rgba(40, 40, 40, 255); // unreachable / NaN -> dim gray
            }
            let t = ((xform(v) - lo) / span).clamp(0.0, 1.0);
            let (r, g, b) = turbo(t);
            pack_rgba(r, g, b, 255)
        })
        .collect()
}

/// Same but for integer scalars (degree, distances).
pub fn sequential_colors_u32(values: &[u32], log_scale: bool) -> Vec<u32> {
    // u32::MAX is BFS's "unreachable" sentinel; map it to dim gray explicitly.
    let as_f: Vec<f32> = values
        .iter()
        .map(|&v| if v == u32::MAX { f32::INFINITY } else { v as f32 })
        .collect();
    // Reuse the f32 path but treat INFINITY as unreachable gray.
    let finite: Vec<f32> = as_f
        .iter()
        .copied()
        .map(|v| if v.is_infinite() { f32::NAN } else { v })
        .collect();
    sequential_colors_f32(&finite, log_scale)
}

/// Per-node size multipliers from a scalar (e.g. degree/PageRank), normalized so
/// hubs stand out without dwarfing everything. Returns values in [1, max_mult].
pub fn sizes_from_scalar(values: &[f32], max_mult: f32) -> Vec<f32> {
    if values.is_empty() {
        return Vec::new();
    }
    let mut lo = f32::MAX;
    let mut hi = f32::MIN;
    for &v in values {
        if v.is_finite() {
            let t = (v.max(0.0) + 1.0).ln();
            lo = lo.min(t);
            hi = hi.max(t);
        }
    }
    let span = (hi - lo).max(1e-9);
    values
        .iter()
        .map(|&v| {
            if !v.is_finite() {
                return 1.0;
            }
            let t = (((v.max(0.0) + 1.0).ln()) - lo) / span;
            1.0 + t.clamp(0.0, 1.0) * (max_mult - 1.0)
        })
        .collect()
}

fn hsv_to_rgb(h: f32, s: f32, v: f32) -> (u8, u8, u8) {
    let i = (h * 6.0).floor();
    let f = h * 6.0 - i;
    let p = v * (1.0 - s);
    let q = v * (1.0 - f * s);
    let t = v * (1.0 - (1.0 - f) * s);
    let (r, g, b) = match (i as i32).rem_euclid(6) {
        0 => (v, t, p),
        1 => (q, v, p),
        2 => (p, v, t),
        3 => (p, q, v),
        4 => (t, p, v),
        _ => (v, p, q),
    };
    (
        (r * 255.0).round() as u8,
        (g * 255.0).round() as u8,
        (b * 255.0).round() as u8,
    )
}

/// Compact polynomial approximation of the "turbo" colormap (Mikhailov 2019).
/// Input t in [0,1]. Good perceptual ordering, dark-to-bright.
fn turbo(t: f32) -> (u8, u8, u8) {
    let t = t.clamp(0.0, 1.0);
    // Polynomial fit coefficients.
    let r = 0.13572138 + t * (4.61539260 + t * (-42.66032258 + t * (132.13108234 + t * (-152.94239396 + t * 59.28637943))));
    let g = 0.09140261 + t * (2.19418839 + t * (4.84296658 + t * (-14.18503333 + t * (4.27729857 + t * 2.82956604))));
    let b = 0.10667330 + t * (12.64194608 + t * (-60.58204836 + t * (110.36276771 + t * (-89.90310912 + t * 27.34824973))));
    (
        (r.clamp(0.0, 1.0) * 255.0) as u8,
        (g.clamp(0.0, 1.0) * 255.0) as u8,
        (b.clamp(0.0, 1.0) * 255.0) as u8,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn categorical_distinct_for_neighbors() {
        assert_ne!(categorical(0), categorical(1));
        assert_ne!(categorical(1), categorical(2));
    }

    #[test]
    fn sequential_spans_range() {
        let vals: Vec<f32> = (0..100).map(|i| i as f32).collect();
        let cols = sequential_colors_f32(&vals, false);
        assert_eq!(cols.len(), 100);
        assert_ne!(cols[0], cols[99]);
    }

    #[test]
    fn turbo_endpoints_differ() {
        assert_ne!(turbo(0.0), turbo(1.0));
    }
}
