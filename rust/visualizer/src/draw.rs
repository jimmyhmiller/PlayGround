use serde::{Serialize, Deserialize};
use skia_safe::{Paint, Color4f};

#[derive(Serialize, Deserialize)]
pub struct Color {
    r: f32,
    g: f32,
    b: f32,
    a: f32,
}

impl Color {
    pub fn to_paint(&self) -> Paint {
        Paint::new(Color4f::new(self.r, self.g, self.b, self.a), None)
    }

    pub fn to_color4f(&self) -> Color4f {
        Color4f::new(self.r, self.g, self.b, self.a)
    }

    pub fn new(r: f32, g: f32, b: f32, a: f32) -> Color {
        Color { r, g, b, a }
    }

    pub fn parse_hex(hex: &str) -> Color {

        let mut start = 0;
        if hex.starts_with("#") {
            start = 1;
        }

        let r = i64::from_str_radix(&hex[start..start+2], 16).unwrap() as f32;
        let g = i64::from_str_radix(&hex[start+2..start+4], 16).unwrap() as f32;
        let b = i64::from_str_radix(&hex[start+4..start+6], 16).unwrap() as f32;
        return Color::new(r / 255.0, g / 255.0, b / 255.0, 1.0)
    }
}

