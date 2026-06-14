//! Font-backed text measurement and glyph rasterization for the tiny-skia
//! backend, using [`fontdue`].
//!
//! Two roles:
//! - [`FontSet::measure`] backs the crate's `FontMeasurer`, which implements
//!   [`whiteboard_core::text::TextMeasurer`] so the headless core can lay text
//!   out against real font metrics.
//! - [`FontSet::draw_run`] rasterizes a positioned [`TextRun`] onto a tiny-skia
//!   [`Pixmap`], glyph by glyph.
//!
//! Fonts are bundled (DejaVu Sans / Sans Mono) so rendering is self-contained and
//! deterministic — no system font discovery, identical output everywhere, which
//! is what snapshot tests need. DejaVu is under the Bitstream Vera license (see
//! `assets/` and `ATTRIBUTION.md`).

use fontdue::{Font, FontSettings};
use tiny_skia::{Pixmap, PremultipliedColorU8};
use whiteboard_core::render::Color;
use whiteboard_core::text::{FontFamily, FontSpec, TextMetrics, TextRun};

const DEJAVU_SANS: &[u8] = include_bytes!("../assets/DejaVuSans.ttf");
const DEJAVU_SANS_MONO: &[u8] = include_bytes!("../assets/DejaVuSansMono.ttf");

/// The bundled fonts, parsed once. Maps each [`FontFamily`] to a concrete font.
pub struct FontSet {
    sans: Font,
    mono: Font,
}

impl FontSet {
    /// Parse the bundled fonts. Panics only if the bundled bytes fail to parse,
    /// which is a build-time invariant (the fonts ship with the crate).
    pub fn new() -> Self {
        let settings = FontSettings::default();
        let sans = Font::from_bytes(DEJAVU_SANS, settings).expect("bundled DejaVuSans parses");
        let mono =
            Font::from_bytes(DEJAVU_SANS_MONO, settings).expect("bundled DejaVuSansMono parses");
        FontSet { sans, mono }
    }

    fn font_for(&self, family: &FontFamily) -> &Font {
        match family {
            // HandDrawn has no bundled sketch font; fall back to sans.
            FontFamily::HandDrawn | FontFamily::Normal | FontFamily::Custom(_) => &self.sans,
            FontFamily::Code => &self.mono,
        }
    }

    /// Measure one line of text.
    pub fn measure(&self, text: &str, font: &FontSpec) -> TextMetrics {
        let f = self.font_for(&font.family);
        let px = font.size as f32;
        let mut width = 0.0f32;
        let mut max_ascent = 0.0f32;
        let mut max_descent = 0.0f32;
        for ch in text.chars() {
            let m = f.metrics(ch, px);
            width += m.advance_width;
            // ymin is negative below baseline; ascent is the glyph top above it.
            let ascent = (m.bounds.height + m.bounds.ymin).max(0.0);
            let descent = (-m.bounds.ymin).max(0.0);
            max_ascent = max_ascent.max(ascent);
            max_descent = max_descent.max(descent);
        }
        // Use the font's line metrics for a stable ascent/descent even on lines
        // with no tall/deep glyphs, so multi-line spacing looks right.
        if let Some(lm) = f.horizontal_line_metrics(px) {
            max_ascent = max_ascent.max(lm.ascent);
            max_descent = max_descent.max(-lm.descent);
        }
        TextMetrics {
            width: width as f64,
            ascent: max_ascent as f64,
            descent: max_descent as f64,
        }
    }

    /// Rasterize a positioned text run onto `pixmap`. `run.origin` is the
    /// baseline-left position in device space (the caller has already applied any
    /// viewport transform to it).
    pub fn draw_run(&self, pixmap: &mut Pixmap, run: &TextRun, color: Color) {
        let f = self.font_for(&run.font.family);
        let px = run.font.size as f32;
        let mut pen_x = run.origin.x as f32;
        let baseline_y = run.origin.y as f32;

        for ch in run.text.chars() {
            let (metrics, bitmap) = f.rasterize(ch, px);
            if metrics.width > 0 && metrics.height > 0 {
                // Glyph top-left in device space: pen + xmin, baseline - (height + ymin).
                let gx = (pen_x + metrics.xmin as f32).round() as i32;
                let gy =
                    (baseline_y - (metrics.height as f32 + metrics.ymin as f32)).round() as i32;
                blit_coverage(pixmap, gx, gy, metrics.width, &bitmap, color);
            }
            pen_x += metrics.advance_width;
        }
    }
}

impl Default for FontSet {
    fn default() -> Self {
        FontSet::new()
    }
}

/// Alpha-blend a fontdue coverage bitmap (one u8 per pixel) onto the pixmap at
/// `(x, y)` using `color`, source-over.
fn blit_coverage(pixmap: &mut Pixmap, x: i32, y: i32, w: usize, coverage: &[u8], color: Color) {
    let pw = pixmap.width() as i32;
    let ph = pixmap.height() as i32;
    let h = coverage.len() / w.max(1);
    let data = pixmap.pixels_mut();

    for row in 0..h as i32 {
        let py = y + row;
        if py < 0 || py >= ph {
            continue;
        }
        for col in 0..w as i32 {
            let pxp = x + col;
            if pxp < 0 || pxp >= pw {
                continue;
            }
            let cov = coverage[(row as usize) * w + col as usize];
            if cov == 0 {
                continue;
            }
            let idx = (py * pw + pxp) as usize;
            data[idx] = blend_over(data[idx], color, cov);
        }
    }
}

/// Source-over blend of `src` (straight color) at coverage `cov` over `dst`
/// (premultiplied), producing a premultiplied result.
fn blend_over(dst: PremultipliedColorU8, src: Color, cov: u8) -> PremultipliedColorU8 {
    let sa = (src.a as u32 * cov as u32) / 255; // effective source alpha 0..255
    if sa == 0 {
        return dst;
    }
    let inv = 255 - sa;
    // src is straight; premultiply by sa. dst is already premultiplied.
    let blend = |s: u8, d: u8| -> u8 {
        let s_pm = (s as u32 * sa) / 255;
        ((s_pm + (d as u32 * inv) / 255).min(255)) as u8
    };
    let r = blend(src.r, dst.red());
    let g = blend(src.g, dst.green());
    let b = blend(src.b, dst.blue());
    let a = (sa + (dst.alpha() as u32 * inv) / 255).min(255) as u8;
    PremultipliedColorU8::from_rgba(r, g, b, a).unwrap_or(dst)
}

#[cfg(test)]
mod tests {
    use super::*;
    use whiteboard_core::text::FontFamily;

    #[test]
    fn measures_wider_for_longer_text() {
        let fs = FontSet::new();
        let font = FontSpec::new(FontFamily::Normal, 20.0);
        let a = fs.measure("i", &font);
        let b = fs.measure("WWWWW", &font);
        assert!(b.width > a.width);
        assert!(a.height() > 0.0);
    }

    #[test]
    fn empty_string_has_zero_width() {
        let fs = FontSet::new();
        let font = FontSpec::new(FontFamily::Normal, 20.0);
        assert_eq!(fs.measure("", &font).width, 0.0);
    }

    #[test]
    fn mono_and_sans_differ() {
        let fs = FontSet::new();
        let mono = fs.measure("il", &FontSpec::new(FontFamily::Code, 20.0));
        let sans = fs.measure("il", &FontSpec::new(FontFamily::Normal, 20.0));
        // Monospace advances are uniform; proportional 'i'/'l' are narrower.
        assert!(mono.width > sans.width);
    }
}
