//! Paint, color and stroke styling shared by elements and draw commands.

use serde::{Deserialize, Serialize};

/// An sRGB color with straight (non-premultiplied) alpha, 0..=255 per channel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Color {
    pub r: u8,
    pub g: u8,
    pub b: u8,
    pub a: u8,
}

impl Color {
    pub const TRANSPARENT: Color = Color {
        r: 0,
        g: 0,
        b: 0,
        a: 0,
    };
    pub const BLACK: Color = Color {
        r: 0,
        g: 0,
        b: 0,
        a: 255,
    };
    pub const WHITE: Color = Color {
        r: 255,
        g: 255,
        b: 255,
        a: 255,
    };

    pub const fn rgb(r: u8, g: u8, b: u8) -> Self {
        Color { r, g, b, a: 255 }
    }

    pub const fn rgba(r: u8, g: u8, b: u8, a: u8) -> Self {
        Color { r, g, b, a }
    }

    pub fn is_transparent(&self) -> bool {
        self.a == 0
    }

    /// Multiply this color's alpha by `opacity` (0.0..=1.0).
    pub fn with_opacity(self, opacity: f32) -> Color {
        let a = (self.a as f32 * opacity.clamp(0.0, 1.0)).round() as u8;
        Color { a, ..self }
    }

    /// Parse a CSS-style hex color: `#rgb`, `#rrggbb`, or `#rrggbbaa`. Also
    /// accepts the literal `"transparent"`.
    pub fn parse_hex(s: &str) -> Option<Color> {
        let s = s.trim();
        if s.eq_ignore_ascii_case("transparent") {
            return Some(Color::TRANSPARENT);
        }
        let hex = s.strip_prefix('#')?;
        let parse = |slice: &str| u8::from_str_radix(slice, 16).ok();
        match hex.len() {
            3 => {
                let r = parse(&hex[0..1])?;
                let g = parse(&hex[1..2])?;
                let b = parse(&hex[2..3])?;
                Some(Color::rgb(r * 17, g * 17, b * 17))
            }
            6 => Some(Color::rgb(
                parse(&hex[0..2])?,
                parse(&hex[2..4])?,
                parse(&hex[4..6])?,
            )),
            8 => Some(Color::rgba(
                parse(&hex[0..2])?,
                parse(&hex[2..4])?,
                parse(&hex[4..6])?,
                parse(&hex[6..8])?,
            )),
            _ => None,
        }
    }

    /// Render as `#rrggbb` or `#rrggbbaa` (when alpha < 255).
    pub fn to_hex(&self) -> String {
        if self.a == 255 {
            format!("#{:02x}{:02x}{:02x}", self.r, self.g, self.b)
        } else {
            format!("#{:02x}{:02x}{:02x}{:02x}", self.r, self.g, self.b, self.a)
        }
    }
}

/// How a closed shape's interior is filled. Mirrors Excalidraw's `fillStyle`.
///
/// The `Solid` variant fills flat; the others are hand-drawn fill patterns
/// produced by the rough generator (Phase 1).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum FillStyle {
    #[default]
    Hachure,
    CrossHatch,
    Solid,
    Zigzag,
    Dots,
}

/// Stroke dash pattern. Mirrors Excalidraw's `strokeStyle`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum StrokeStyle {
    #[default]
    Solid,
    Dashed,
    Dotted,
}

/// Line cap style.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum LineCap {
    #[default]
    Round,
    Butt,
    Square,
}

/// Line join style.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum LineJoin {
    #[default]
    Round,
    Miter,
    Bevel,
}

/// A concrete stroke description handed to backends. Dash patterns are resolved
/// to explicit on/off lengths so backends don't need to know `StrokeStyle`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Stroke {
    pub width: f64,
    pub cap: LineCap,
    pub join: LineJoin,
    /// Explicit dash on/off lengths in scene units; empty means solid.
    pub dash: Vec<f64>,
}

impl Default for Stroke {
    fn default() -> Self {
        Stroke {
            width: 1.0,
            cap: LineCap::Round,
            join: LineJoin::Round,
            dash: Vec::new(),
        }
    }
}

impl Stroke {
    pub fn solid(width: f64) -> Self {
        Stroke {
            width,
            ..Default::default()
        }
    }

    /// Resolve an Excalidraw `StrokeStyle` to explicit dash lengths, following
    /// Excalidraw's convention of scaling the pattern by stroke width.
    pub fn with_style(width: f64, style: StrokeStyle) -> Self {
        let dash = match style {
            StrokeStyle::Solid => Vec::new(),
            StrokeStyle::Dashed => vec![8.0, 8.0 + width * 2.0],
            StrokeStyle::Dotted => vec![1.5, 6.0 + width * 2.0],
        };
        Stroke {
            width,
            // Dotted relies on round caps to render dots; the others use round
            // caps too, matching Excalidraw's default stroke appearance.
            cap: LineCap::Round,
            join: LineJoin::Round,
            dash,
        }
    }
}

/// A paint source for fills and strokes. Solid color today; gradients are a
/// future extension that backends can ignore until they appear.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Paint {
    Solid(Color),
}

impl Paint {
    pub fn solid(color: Color) -> Self {
        Paint::Solid(color)
    }

    pub fn with_opacity(&self, opacity: f32) -> Paint {
        match self {
            Paint::Solid(c) => Paint::Solid(c.with_opacity(opacity)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_six_digit_hex() {
        let c = Color::parse_hex("#1e1e1e").unwrap();
        assert_eq!(c, Color::rgb(30, 30, 30));
    }

    #[test]
    fn parse_short_hex() {
        assert_eq!(Color::parse_hex("#fff").unwrap(), Color::WHITE);
    }

    #[test]
    fn parse_transparent() {
        assert!(Color::parse_hex("transparent").unwrap().is_transparent());
    }

    #[test]
    fn hex_round_trip() {
        let c = Color::rgba(18, 52, 86, 128);
        assert_eq!(Color::parse_hex(&c.to_hex()).unwrap(), c);
    }

    #[test]
    fn opacity_halves_alpha() {
        let c = Color::BLACK.with_opacity(0.5);
        assert_eq!(c.a, 128);
    }
}
