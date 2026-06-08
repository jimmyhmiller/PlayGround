//! Scene constants, button table, and mesh constructors for the
//! calculator scene. CPU-only — builds (vertices, indices) tuples.

use crate::geometry::{
    MeshBuilder, Polygon, Vertex, ensure_ccw, ensure_cw, layout_text_centered, rounded_rect_ring,
};
use glam::Vec2;

pub const PAPER_W: f32 = 14.0;
pub const PAPER_H: f32 = 17.5;
pub const PAPER_R: f32 = 0.55;

pub const Z_FLOOR: f32 = -0.4;
pub const Z_PAPER: f32 = 0.56;
pub const Z_PANEL: f32 = 0.14;
pub const Z_DISPLAY: f32 = 0.07;
pub const Z_DISPLAY_INK: f32 = 0.01;

pub const PANEL_W: f32 = 10.6;
pub const PANEL_H: f32 = 13.6;
pub const PANEL_R: f32 = 0.65;

pub const DISPLAY_W: f32 = 9.4;
pub const DISPLAY_H_SIZE: f32 = 1.85;
pub const DISPLAY_R: f32 = 0.40;
pub const DISPLAY_Y: f32 = 4.85;
pub const DISPLAY_DIGIT_EM: f32 = 1.10;

pub const BTN_W: f32 = 1.85;
pub const BTN_H_SIZE: f32 = 1.55;
pub const BTN_R: f32 = 0.50;
pub const BTN_GAP_X: f32 = 0.25;
pub const BTN_GAP_Y: f32 = 0.25;
pub const BTN_TOP_ROW_Y: f32 = 2.45;
pub const BTN_GLYPH_EM: f32 = 0.85;

pub const Z_BTN_FACE_REST: f32 = 0.45;
pub const Z_BTN_FACE_PRESSED: f32 = 0.18;
pub const Z_BTN_BOTTOM: f32 = -0.5;
pub const Z_BTN_ENGRAVE_DELTA: f32 = 0.075;

const FONT_FALLBACKS: &[&str] = &[
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/Library/Fonts/Arial.ttf",
];

pub fn load_font() -> Vec<u8> {
    for path in FONT_FALLBACKS {
        if let Ok(b) = std::fs::read(path) {
            if ttf_parser::Face::parse(&b, 0).is_ok() {
                return b;
            }
        }
    }
    panic!("no usable font found");
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum WidgetId {
    Digit(u8),
    Add,
    Sub,
    Mul,
    Div,
    Equals,
    Decimal,
}

#[derive(Clone, Copy)]
pub enum BtnColor {
    Cream,
    Coral,
}

pub struct BtnSpec {
    pub id: WidgetId,
    pub glyph: &'static str,
    pub color: BtnColor,
    pub col: u8,
    pub row: u8,
}

pub const BUTTONS: &[BtnSpec] = &[
    BtnSpec { id: WidgetId::Digit(7), glyph: "7", color: BtnColor::Cream, col: 0, row: 0 },
    BtnSpec { id: WidgetId::Digit(8), glyph: "8", color: BtnColor::Cream, col: 1, row: 0 },
    BtnSpec { id: WidgetId::Digit(9), glyph: "9", color: BtnColor::Cream, col: 2, row: 0 },
    BtnSpec { id: WidgetId::Div,      glyph: "/", color: BtnColor::Coral, col: 3, row: 0 },
    BtnSpec { id: WidgetId::Digit(4), glyph: "4", color: BtnColor::Cream, col: 0, row: 1 },
    BtnSpec { id: WidgetId::Digit(5), glyph: "5", color: BtnColor::Cream, col: 1, row: 1 },
    BtnSpec { id: WidgetId::Digit(6), glyph: "6", color: BtnColor::Cream, col: 2, row: 1 },
    BtnSpec { id: WidgetId::Mul,      glyph: "x", color: BtnColor::Coral, col: 3, row: 1 },
    BtnSpec { id: WidgetId::Digit(1), glyph: "1", color: BtnColor::Cream, col: 0, row: 2 },
    BtnSpec { id: WidgetId::Digit(2), glyph: "2", color: BtnColor::Cream, col: 1, row: 2 },
    BtnSpec { id: WidgetId::Digit(3), glyph: "3", color: BtnColor::Cream, col: 2, row: 2 },
    BtnSpec { id: WidgetId::Sub,      glyph: "-", color: BtnColor::Coral, col: 3, row: 2 },
    BtnSpec { id: WidgetId::Digit(0), glyph: "0", color: BtnColor::Cream, col: 0, row: 3 },
    BtnSpec { id: WidgetId::Decimal,  glyph: ".", color: BtnColor::Cream, col: 1, row: 3 },
    BtnSpec { id: WidgetId::Equals,   glyph: "=", color: BtnColor::Coral, col: 2, row: 3 },
    BtnSpec { id: WidgetId::Add,      glyph: "+", color: BtnColor::Coral, col: 3, row: 3 },
];

pub fn btn_center(spec: &BtnSpec) -> Vec2 {
    let cx = (spec.col as f32 - 1.5) * (BTN_W + BTN_GAP_X);
    let cy = BTN_TOP_ROW_Y - spec.row as f32 * (BTN_H_SIZE + BTN_GAP_Y);
    Vec2::new(cx, cy)
}

pub fn build_paper_mesh() -> (Vec<Vertex>, Vec<u32>) {
    let mut mb = MeshBuilder::default();
    let exterior = ensure_ccw(rounded_rect_ring(0.0, 0.0, PAPER_W, PAPER_H, PAPER_R));
    let panel_hole = ensure_cw(rounded_rect_ring(0.0, 0.0, PANEL_W, PANEL_H, PANEL_R));

    mb.add_cap(
        &Polygon { exterior: exterior.clone(), holes: vec![panel_hole.clone()] },
        Z_PAPER,
        true,
    );
    mb.add_wall(&exterior, Z_PAPER, Z_FLOOR);
    mb.add_wall(&panel_hole, Z_PAPER, Z_PANEL);
    mb.into_vertices()
}

pub fn build_panel_mesh() -> (Vec<Vertex>, Vec<u32>) {
    let mut mb = MeshBuilder::default();
    let exterior = ensure_ccw(rounded_rect_ring(0.0, 0.0, PANEL_W, PANEL_H, PANEL_R));
    let display_hole = ensure_cw(rounded_rect_ring(
        0.0,
        DISPLAY_Y,
        DISPLAY_W,
        DISPLAY_H_SIZE,
        DISPLAY_R,
    ));
    mb.add_cap(
        &Polygon { exterior, holes: vec![display_hole.clone()] },
        Z_PANEL,
        true,
    );
    mb.add_wall(&display_hole, Z_PANEL, Z_DISPLAY);
    mb.into_vertices()
}

pub fn build_display_mesh(font: &[u8], display_text: &str) -> (Vec<Vertex>, Vec<u32>) {
    let mut mb = MeshBuilder::default();
    let exterior = ensure_ccw(rounded_rect_ring(
        0.0,
        DISPLAY_Y,
        DISPLAY_W,
        DISPLAY_H_SIZE,
        DISPLAY_R,
    ));

    let raw = layout_text_centered(font, display_text, Vec2::ZERO, DISPLAY_DIGIT_EM);
    let digit_polys: Vec<Polygon> = if raw.is_empty() {
        Vec::new()
    } else {
        let (mut xmin, mut xmax) = (f32::INFINITY, f32::NEG_INFINITY);
        for poly in &raw {
            for p in &poly.exterior {
                xmin = xmin.min(p.x);
                xmax = xmax.max(p.x);
            }
        }
        let half_w = (xmax - xmin) * 0.5;
        let target_right = DISPLAY_W * 0.5 - 0.55;
        let dx = target_right - half_w;
        let dy = DISPLAY_Y - 0.05;
        raw.into_iter().map(|p| p.translated(dx, dy)).collect()
    };

    let holes: Vec<Vec<Vec2>> = digit_polys
        .iter()
        .map(|p| ensure_cw(p.exterior.clone()))
        .collect();
    mb.add_cap(&Polygon { exterior, holes }, Z_DISPLAY, true);

    for poly in &digit_polys {
        let cavity_ring = ensure_cw(poly.exterior.clone());
        mb.add_wall(&cavity_ring, Z_DISPLAY, Z_DISPLAY_INK);
        for hole in &poly.holes {
            let mut island_ring = hole.clone();
            island_ring.reverse();
            mb.add_wall(&island_ring, Z_DISPLAY, Z_DISPLAY_INK);
            mb.add_cap(
                &Polygon::from_ring(ensure_ccw(island_ring)),
                Z_DISPLAY,
                true,
            );
        }
    }

    for poly in &digit_polys {
        mb.add_cap(
            &Polygon {
                exterior: poly.exterior.clone(),
                holes: poly.holes.clone(),
            },
            Z_DISPLAY_INK,
            true,
        );
    }

    mb.into_vertices()
}

pub fn build_button_mesh(font: &[u8], spec: &BtnSpec) -> (Vec<Vertex>, Vec<u32>) {
    let mut mb = MeshBuilder::default();
    let exterior = ensure_ccw(rounded_rect_ring(0.0, 0.0, BTN_W, BTN_H_SIZE, BTN_R));

    let glyph_polys = layout_text_centered(font, spec.glyph, Vec2::ZERO, BTN_GLYPH_EM);

    let glyph_holes: Vec<Vec<Vec2>> = glyph_polys
        .iter()
        .map(|p| ensure_cw(p.exterior.clone()))
        .collect();
    mb.add_cap(
        &Polygon { exterior: exterior.clone(), holes: glyph_holes },
        Z_BTN_FACE_REST,
        true,
    );

    mb.add_wall(&exterior, Z_BTN_FACE_REST, Z_BTN_BOTTOM);

    let engrave_z = Z_BTN_FACE_REST - Z_BTN_ENGRAVE_DELTA;
    for poly in &glyph_polys {
        let cavity_ring = ensure_cw(poly.exterior.clone());
        mb.add_wall(&cavity_ring, Z_BTN_FACE_REST, engrave_z);
        for hole in &poly.holes {
            let mut island = hole.clone();
            island.reverse();
            mb.add_wall(&island, Z_BTN_FACE_REST, engrave_z);
            mb.add_cap(
                &Polygon::from_ring(ensure_ccw(island)),
                Z_BTN_FACE_REST,
                true,
            );
        }
    }
    for poly in &glyph_polys {
        mb.add_cap(
            &Polygon {
                exterior: poly.exterior.clone(),
                holes: poly.holes.clone(),
            },
            engrave_z,
            true,
        );
    }

    mb.into_vertices()
}

/// Colours mirroring the StandardMaterial swatches in proper_mesh.rs.
/// Returned in linear sRGB-ish space (we'll apply gamma in the shader).
pub const COLOR_CREAM: [f32; 3] = [0.905, 0.860, 0.745];
pub const COLOR_TAUPE: [f32; 3] = [0.460, 0.405, 0.340];
pub const COLOR_OLIVE: [f32; 3] = [0.585, 0.620, 0.470];
pub const COLOR_CORAL: [f32; 3] = [0.860, 0.470, 0.380];
