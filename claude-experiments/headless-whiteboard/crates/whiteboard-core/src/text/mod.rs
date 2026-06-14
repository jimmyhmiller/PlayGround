//! Text styling, measurement, and layout.
//!
//! The core library cannot rasterize or measure glyphs — that requires fonts,
//! which only a real backend has. So measurement is **injected** via the
//! [`TextMeasurer`] trait. Everything the library needs to lay out and place
//! text it gets back through [`TextMetrics`].
//!
//! Layout (wrapping, container fitting) is added in Phase 1 on top of these
//! types; this module file establishes the shared vocabulary.

use serde::{Deserialize, Serialize};

/// Horizontal alignment of text within its bounding box / container.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum TextAlign {
    #[default]
    Left,
    Center,
    Right,
}

/// Vertical alignment of text within its container.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum VerticalAlign {
    Top,
    #[default]
    Middle,
    Bottom,
}

/// Which font family a text element uses. Excalidraw ships three; backends map
/// these to concrete fonts. `Custom` carries a backend-specific family name.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum FontFamily {
    /// Hand-drawn look (Excalidraw's "Virgil"/Excalifont).
    #[default]
    HandDrawn,
    /// Normal sans/serif (Excalidraw's "Helvetica"/Nunito).
    Normal,
    /// Monospace (Excalidraw's "Cascadia").
    Code,
    Custom(String),
}

/// Everything needed to identify a font for measurement and rendering.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FontSpec {
    pub family: FontFamily,
    pub size: f64,
    /// Line height as a multiple of font size (Excalidraw default ≈ 1.25).
    pub line_height: f64,
}

impl Default for FontSpec {
    fn default() -> Self {
        FontSpec {
            family: FontFamily::default(),
            size: 20.0,
            line_height: 1.25,
        }
    }
}

impl FontSpec {
    pub fn new(family: FontFamily, size: f64) -> Self {
        FontSpec {
            family,
            size,
            line_height: 1.25,
        }
    }

    /// Distance between baselines.
    pub fn line_spacing(&self) -> f64 {
        self.size * self.line_height
    }
}

/// Measured metrics for a single line of text in a given font.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TextMetrics {
    /// Advance width of the line.
    pub width: f64,
    /// Distance from baseline up to the top of the tallest glyph.
    pub ascent: f64,
    /// Distance from baseline down to the bottom of the lowest glyph.
    pub descent: f64,
}

impl TextMetrics {
    pub fn height(&self) -> f64 {
        self.ascent + self.descent
    }
}

/// Injected by the backend so the headless core can measure text without owning
/// fonts. Must be deterministic for a given `(text, font)` so layout and
/// snapshot tests are stable.
pub trait TextMeasurer {
    /// Measure a single line (no embedded newlines).
    fn measure(&self, text: &str, font: &FontSpec) -> TextMetrics;
}

/// A trivial measurer for tests and headless contexts with no real fonts:
/// assumes a fixed character advance and metrics proportional to font size.
/// Deterministic, dependency-free — never use it for production rendering.
#[derive(Debug, Clone, Copy)]
pub struct MonospaceMeasurer {
    /// Character advance as a fraction of font size.
    pub advance_ratio: f64,
}

impl Default for MonospaceMeasurer {
    fn default() -> Self {
        MonospaceMeasurer { advance_ratio: 0.6 }
    }
}

impl TextMeasurer for MonospaceMeasurer {
    fn measure(&self, text: &str, font: &FontSpec) -> TextMetrics {
        let chars = text.chars().count() as f64;
        TextMetrics {
            width: chars * font.size * self.advance_ratio,
            ascent: font.size * 0.8,
            descent: font.size * 0.2,
        }
    }
}

/// A laid-out run of text ready for a backend to draw: a single line at a known
/// origin (the baseline-left position), in a known font and alignment.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TextRun {
    pub text: String,
    pub font: FontSpec,
    /// Baseline-left origin in scene coordinates.
    pub origin: crate::geometry::Point,
    pub align: TextAlign,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn monospace_measurer_scales_with_length() {
        let m = MonospaceMeasurer::default();
        let f = FontSpec::new(FontFamily::Code, 10.0);
        let a = m.measure("ab", &f);
        let b = m.measure("abcd", &f);
        assert!((b.width - 2.0 * a.width).abs() < 1e-9);
        assert_eq!(a.height(), 10.0);
    }

    #[test]
    fn line_spacing_uses_line_height() {
        let f = FontSpec {
            size: 20.0,
            line_height: 1.25,
            ..Default::default()
        };
        assert_eq!(f.line_spacing(), 25.0);
    }
}
