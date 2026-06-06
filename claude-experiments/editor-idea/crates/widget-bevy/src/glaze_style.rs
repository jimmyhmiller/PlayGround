//! Adapter: a `glaze::CompiledStyle` → the existing `protocol::Style`.
//!
//! This is the whole "Glaze produces the styling the renderer already
//! understands" seam. Colors are stored linear in Glaze; the renderer parses
//! sRGB hex back to linear, so we round-trip linear → sRGB `#rrggbb[aa]`.

use crate::protocol::{Border, Edges, ShaderSpec, Shadow, Style};
use glaze::{CompiledStyle, Dim, Dir, Layer, Rgba};

fn lin_to_srgb(c: f32) -> f32 {
    if c <= 0.003_130_8 {
        12.92 * c
    } else {
        1.055 * c.clamp(0.0, 1.0).powf(1.0 / 2.4) - 0.055
    }
}

/// linear-rgb → `#rrggbb` (or `#rrggbbaa` when not opaque).
pub fn hex(c: Rgba) -> String {
    let b8 = |x: f32| (lin_to_srgb(x).clamp(0.0, 1.0) * 255.0).round() as u8;
    let (r, g, b) = (b8(c.r), b8(c.g), b8(c.b));
    let a = (c.a.clamp(0.0, 1.0) * 255.0).round() as u8;
    if a == 255 {
        format!("#{r:02x}{g:02x}{b:02x}")
    } else {
        format!("#{r:02x}{g:02x}{b:02x}{a:02x}")
    }
}

fn dim_str(d: Dim) -> String {
    match d {
        Dim::Px(p) => format!("{p}"),
        Dim::Pct(p) => format!("{p}%"),
        Dim::Auto => "auto".into(),
    }
}

/// Convert a compiled Glaze style into a `protocol::Style`.
pub fn to_style(c: &CompiledStyle) -> Style {
    let mut s = Style::default();
    let p = c.box_.padding;
    s.padding = Some(Edges { top: p[0], right: p[1], bottom: p[2], left: p[3] });
    if c.box_.radius > 0.0 {
        s.radius = Some(format!("{}", c.box_.radius));
    }
    s.width = c.box_.width.map(dim_str);
    s.height = c.box_.height.map(dim_str);
    s.min_width = c.box_.min_width.map(dim_str);
    s.max_width = c.box_.max_width.map(dim_str);
    s.min_height = c.box_.min_height.map(dim_str);
    s.max_height = c.box_.max_height.map(dim_str);
    s.flex_grow = c.box_.flex_grow;
    s.flex_shrink = c.box_.flex_shrink;
    s.flex_direction = c.box_.flex_direction.map(|d| match d {
        Dir::Row => "row".to_string(),
        Dir::Column => "column".to_string(),
    });
    for layer in &c.layers {
        match layer {
            // last fill wins (a :hover overlay replaces the base fill)
            Layer::Fill(rgba) => s.background = Some(hex(*rgba)),
            Layer::Border { color, width } => {
                s.border = Some(Border { color: hex(*color), width: *width });
            }
            Layer::Shadow { color, blur, offset_y } => {
                s.shadow = Some(Shadow {
                    token: None,
                    color: Some(hex(*color)),
                    blur: Some(*blur),
                    offset_y: Some(*offset_y),
                });
            }
            Layer::Shader(cs) => {
                s.shader = Some(ShaderSpec { body: cs.wgsl_body.clone(), overlay: cs.overlay });
            }
        }
    }
    s
}
