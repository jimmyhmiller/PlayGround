//! Game-style scene: every visible piece is a static mesh built once
//! at startup and rendered via its own entity. Animation = transform
//! tween. No mesh rebuilds, no boolean ops, no heightmap rasterization.
//!
//! Entities:
//!   - Paper (rounded rect with rect-shaped hole for the panel)
//!   - Panel surface (rounded rect with hole for display)
//!   - Display (rounded rect with glyph holes for digits + engrave floor)
//!   - 16 buttons, each its own mesh: top face with glyph hole + outer
//!     wall + glyph wall + engrave floor
//!
//! Press animation lowers a button's transform.z. Mesh untouched.

use bevy::asset::RenderAssetUsages;
use bevy::input::ButtonInput;
use bevy::input::mouse::{MouseScrollUnit, MouseWheel};
use bevy::light::CascadeShadowConfigBuilder;
use bevy::mesh::{Indices, PrimitiveTopology};
use bevy::prelude::*;
use lyon::math::point as lyon_point;
use lyon::path::iterator::PathIterator;
use lyon::path::{Path as LyonPath, PathEvent};
use lyon::tessellation::{
    BuffersBuilder, FillOptions, FillRule, FillTessellator, FillVertex, VertexBuffers,
};
use std::collections::HashMap;
use ttf_parser::{Face, GlyphId, OutlineBuilder};

// ─── Constants ────────────────────────────────────────────────────────

const PAPER_W: f32 = 14.0;
const PAPER_H: f32 = 17.5;
const PAPER_R: f32 = 0.55;

const Z_FLOOR: f32 = -0.4;
const Z_PAPER: f32 = 0.56;
const Z_PANEL: f32 = 0.14;
const Z_DISPLAY: f32 = 0.07;
const Z_DISPLAY_INK: f32 = 0.01;

const PANEL_W: f32 = 10.6;
const PANEL_H: f32 = 13.6;
const PANEL_R: f32 = 0.65;

const DISPLAY_W: f32 = 9.4;
const DISPLAY_H_SIZE: f32 = 1.85;
const DISPLAY_R: f32 = 0.40;
const DISPLAY_Y: f32 = 4.85;
const DISPLAY_DIGIT_EM: f32 = 1.10;

const BTN_W: f32 = 1.85;
const BTN_H_SIZE: f32 = 1.55;
const BTN_R: f32 = 0.50;
const BTN_GAP_X: f32 = 0.25;
const BTN_GAP_Y: f32 = 0.25;
const BTN_TOP_ROW_Y: f32 = 2.45;
const BTN_GLYPH_EM: f32 = 0.85;

const Z_BTN_FACE_REST: f32 = 0.45;
const Z_BTN_FACE_PRESSED: f32 = 0.18;
const Z_BTN_BOTTOM: f32 = -0.5;
const Z_BTN_ENGRAVE_DELTA: f32 = 0.075;

const FONT_FALLBACKS: &[&str] = &[
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/Library/Fonts/Arial.ttf",
];

// ─── Polygon ──────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
struct Polygon {
    exterior: Vec<Vec2>,
    holes: Vec<Vec<Vec2>>,
}

impl Polygon {
    fn from_ring(ring: Vec<Vec2>) -> Self {
        Self { exterior: ring, holes: Vec::new() }
    }
    fn translated(&self, dx: f32, dy: f32) -> Self {
        let off = Vec2::new(dx, dy);
        Self {
            exterior: self.exterior.iter().map(|p| *p + off).collect(),
            holes: self.holes.iter().map(|h| h.iter().map(|p| *p + off).collect()).collect(),
        }
    }
    fn contains(&self, p: Vec2) -> bool {
        if !point_in_ring(p, &self.exterior) {
            return false;
        }
        for hole in &self.holes {
            if point_in_ring(p, hole) {
                return false;
            }
        }
        true
    }
}

fn point_in_ring(p: Vec2, ring: &[Vec2]) -> bool {
    let n = ring.len();
    if n < 3 {
        return false;
    }
    let mut inside = false;
    let mut j = n - 1;
    for i in 0..n {
        let (xi, yi) = (ring[i].x, ring[i].y);
        let (xj, yj) = (ring[j].x, ring[j].y);
        if (yi > p.y) != (yj > p.y) {
            let t = (xj - xi) * (p.y - yi) / (yj - yi) + xi;
            if p.x < t {
                inside = !inside;
            }
        }
        j = i;
    }
    inside
}

fn signed_area(pts: &[Vec2]) -> f32 {
    let mut a = 0.0;
    let n = pts.len();
    for i in 0..n {
        let j = (i + 1) % n;
        a += pts[i].x * pts[j].y - pts[j].x * pts[i].y;
    }
    a * 0.5
}

fn ensure_ccw(mut pts: Vec<Vec2>) -> Vec<Vec2> {
    if signed_area(&pts) < 0.0 {
        pts.reverse();
    }
    pts
}

fn ensure_cw(mut pts: Vec<Vec2>) -> Vec<Vec2> {
    if signed_area(&pts) > 0.0 {
        pts.reverse();
    }
    pts
}

// ─── Shape constructors ───────────────────────────────────────────────

fn rounded_rect_ring(cx: f32, cy: f32, w: f32, h: f32, r: f32) -> Vec<Vec2> {
    let r = r.min(w * 0.5).min(h * 0.5).max(0.0);
    if r < 1e-4 {
        return vec![
            Vec2::new(cx - w * 0.5, cy - h * 0.5),
            Vec2::new(cx + w * 0.5, cy - h * 0.5),
            Vec2::new(cx + w * 0.5, cy + h * 0.5),
            Vec2::new(cx - w * 0.5, cy + h * 0.5),
        ];
    }
    let segs = 24u32;
    let hw = w * 0.5 - r;
    let hh = h * 0.5 - r;
    let mut pts = Vec::with_capacity((segs as usize + 1) * 4);
    let corner = |center: Vec2, start: f32, pts: &mut Vec<Vec2>| {
        for i in 0..=segs {
            let a = start + (i as f32 / segs as f32) * std::f32::consts::FRAC_PI_2;
            pts.push(Vec2::new(center.x + r * a.cos(), center.y + r * a.sin()));
        }
    };
    corner(Vec2::new(cx + hw, cy + hh), 0.0, &mut pts);
    corner(Vec2::new(cx - hw, cy + hh), std::f32::consts::FRAC_PI_2, &mut pts);
    corner(Vec2::new(cx - hw, cy - hh), std::f32::consts::PI, &mut pts);
    corner(Vec2::new(cx + hw, cy - hh), 1.5 * std::f32::consts::PI, &mut pts);
    pts
}

// ─── Glyph outlines ───────────────────────────────────────────────────

#[derive(Clone, Copy)]
enum Op {
    Move(Vec2),
    Line(Vec2),
    Quad(Vec2, Vec2),
    Cubic(Vec2, Vec2, Vec2),
    Close,
}

struct OpCollector {
    ops: Vec<Op>,
    scale: f32,
    offset: Vec2,
}

impl OpCollector {
    fn t(&self, x: f32, y: f32) -> Vec2 {
        Vec2::new(x * self.scale + self.offset.x, y * self.scale + self.offset.y)
    }
}

impl OutlineBuilder for OpCollector {
    fn move_to(&mut self, x: f32, y: f32) { self.ops.push(Op::Move(self.t(x, y))); }
    fn line_to(&mut self, x: f32, y: f32) { self.ops.push(Op::Line(self.t(x, y))); }
    fn quad_to(&mut self, x1: f32, y1: f32, x: f32, y: f32) {
        self.ops.push(Op::Quad(self.t(x1, y1), self.t(x, y)));
    }
    fn curve_to(&mut self, x1: f32, y1: f32, x2: f32, y2: f32, x: f32, y: f32) {
        self.ops.push(Op::Cubic(self.t(x1, y1), self.t(x2, y2), self.t(x, y)));
    }
    fn close(&mut self) { self.ops.push(Op::Close); }
}

fn ops_to_path(ops: &[Op]) -> LyonPath {
    let mut pb = LyonPath::builder();
    let mut started = false;
    for op in ops {
        match *op {
            Op::Move(p) => {
                if started { pb.end(false); }
                pb.begin(lyon_point(p.x, p.y));
                started = true;
            }
            Op::Line(p) => { pb.line_to(lyon_point(p.x, p.y)); }
            Op::Quad(c, p) => { pb.quadratic_bezier_to(lyon_point(c.x, c.y), lyon_point(p.x, p.y)); }
            Op::Cubic(c1, c2, p) => {
                pb.cubic_bezier_to(
                    lyon_point(c1.x, c1.y),
                    lyon_point(c2.x, c2.y),
                    lyon_point(p.x, p.y),
                );
            }
            Op::Close => {
                if started { pb.end(true); started = false; }
            }
        }
    }
    if started { pb.end(false); }
    pb.build()
}

fn flatten_path_to_contours(path: &LyonPath, tol: f32) -> Vec<Vec<Vec2>> {
    let mut out: Vec<Vec<Vec2>> = Vec::new();
    let mut current: Vec<Vec2> = Vec::new();
    for evt in path.iter().flattened(tol) {
        match evt {
            PathEvent::Begin { at } => {
                current.clear();
                current.push(Vec2::new(at.x, at.y));
            }
            PathEvent::Line { to, .. } => { current.push(Vec2::new(to.x, to.y)); }
            PathEvent::Quadratic { .. } | PathEvent::Cubic { .. } => {}
            PathEvent::End { close, .. } => {
                if let (Some(first), Some(last)) =
                    (current.first().copied(), current.last().copied())
                {
                    if (first - last).length_squared() < 1e-12 && current.len() > 1 {
                        current.pop();
                    }
                }
                if close && current.len() >= 3 {
                    out.push(std::mem::take(&mut current));
                } else {
                    current.clear();
                }
            }
        }
    }
    out
}

fn build_glyph_polygons(contours: &[Vec<Vec2>]) -> Vec<Polygon> {
    let valid: Vec<&Vec<Vec2>> = contours.iter().filter(|c| c.len() >= 3).collect();
    let n = valid.len();
    if n == 0 { return Vec::new(); }
    let mut depths = vec![0usize; n];
    for i in 0..n {
        for j in 0..n {
            if i == j { continue; }
            if point_in_ring(valid[i][0], valid[j]) {
                depths[i] += 1;
            }
        }
    }
    let mut polys = Vec::new();
    for i in 0..n {
        if depths[i] % 2 != 0 { continue; }
        let outer = ensure_ccw(valid[i].clone());
        let mut holes: Vec<Vec<Vec2>> = Vec::new();
        for j in 0..n {
            if i == j || depths[j] != depths[i] + 1 { continue; }
            if point_in_ring(valid[j][0], &outer) {
                holes.push(ensure_cw(valid[j].clone()));
            }
        }
        polys.push(Polygon { exterior: outer, holes });
    }
    polys
}

fn outline_glyph_to_path(face: &Face, glyph_id: GlyphId, scale: f32, offset: Vec2) -> LyonPath {
    let mut col = OpCollector { ops: Vec::new(), scale, offset };
    face.outline_glyph(glyph_id, &mut col);
    ops_to_path(&col.ops)
}

fn layout_text_centered(font: &[u8], text: &str, origin: Vec2, em: f32) -> Vec<Polygon> {
    let face = match Face::parse(font, 0) {
        Ok(f) => f,
        Err(_) => return Vec::new(),
    };
    let upem = face.units_per_em() as f32;
    let scale = em / upem;
    let mut polys: Vec<Polygon> = Vec::new();
    let mut x = 0.0_f32;
    let mut bmin = Vec2::splat(f32::INFINITY);
    let mut bmax = Vec2::splat(f32::NEG_INFINITY);

    for ch in text.chars() {
        let Some(gid) = face.glyph_index(ch) else {
            x += em * 0.3;
            continue;
        };
        let advance = face.glyph_hor_advance(gid).unwrap_or(0) as f32 * scale;
        let path = outline_glyph_to_path(&face, gid, scale, Vec2::new(x, 0.0));
        let contours = flatten_path_to_contours(&path, 0.005);
        for c in &contours {
            for p in c { bmin = bmin.min(*p); bmax = bmax.max(*p); }
        }
        polys.extend(build_glyph_polygons(&contours));
        x += advance;
    }
    if !bmin.is_finite() { return polys; }
    let center = (bmin + bmax) * 0.5;
    let dx = origin.x - center.x;
    let dy = origin.y - center.y;
    polys.into_iter().map(|p| p.translated(dx, dy)).collect()
}

// ─── Mesh building (lyon for top faces, manual extrusion for walls) ──

#[derive(Default)]
struct MeshBuilder {
    positions: Vec<[f32; 3]>,
    normals: Vec<[f32; 3]>,
    indices: Vec<u32>,
}

impl MeshBuilder {
    /// Top (or bottom) cap face for a polygon-with-holes. Uses lyon's
    /// fill tessellator. `normal_up = true` puts the cap facing +Z.
    fn add_cap(&mut self, polygon: &Polygon, z: f32, normal_up: bool) {
        let mut pb = LyonPath::builder();
        if polygon.exterior.len() < 3 { return; }
        push_ring_to_path(&polygon.exterior, &mut pb);
        for hole in &polygon.holes {
            if hole.len() >= 3 { push_ring_to_path(hole, &mut pb); }
        }
        let path = pb.build();
        let mut buffers: VertexBuffers<[f32; 2], u32> = VertexBuffers::new();
        let mut tess = FillTessellator::new();
        let opts = FillOptions::default()
            .with_fill_rule(FillRule::NonZero)
            .with_tolerance(0.01);
        if tess
            .tessellate_path(
                &path,
                &opts,
                &mut BuffersBuilder::new(&mut buffers, |v: FillVertex| {
                    let p = v.position();
                    [p.x, p.y]
                }),
            )
            .is_err()
        {
            return;
        }
        let base = self.positions.len() as u32;
        let normal = if normal_up { [0.0, 0.0, 1.0] } else { [0.0, 0.0, -1.0] };
        for v in &buffers.vertices {
            self.positions.push([v[0], v[1], z]);
            self.normals.push(normal);
        }
        for chunk in buffers.indices.chunks_exact(3) {
            if normal_up {
                // Lyon emits y-down winding; flip for our +Z viewer.
                self.indices.push(base + chunk[0]);
                self.indices.push(base + chunk[2]);
                self.indices.push(base + chunk[1]);
            } else {
                self.indices.push(base + chunk[0]);
                self.indices.push(base + chunk[1]);
                self.indices.push(base + chunk[2]);
            }
        }
    }

    /// Extruded ring → vertical wall strip. `outward_normal = true`
    /// flips the wall's facing (for outer rings of a CCW polygon, you
    /// want outward; for hole rings of a CCW polygon — which we keep
    /// as CW — outward also points away from the polygon interior,
    /// which is the visible side).
    fn add_wall(&mut self, ring: &[Vec2], z_top: f32, z_bottom: f32) {
        let n = ring.len();
        if n < 3 { return; }
        for i in 0..n {
            let a = ring[i];
            let b = ring[(i + 1) % n];
            let edge = b - a;
            let elen = edge.length();
            if elen < 1e-7 { continue; }
            // For a CCW outer ring, the polygon interior is on the LEFT
            // of the walked edge — i.e., `(-edge.y, edge.x) / elen`. So
            // the outward face normal is `(edge.y, -edge.x) / elen`.
            // For a CW hole ring, this same formula gives the side
            // facing into the hole, which is the visible side. Either
            // way, this is correct.
            let nx = edge.y / elen;
            let ny = -edge.x / elen;
            let normal = [nx, ny, 0.0];
            let base = self.positions.len() as u32;
            self.positions.push([a.x, a.y, z_top]);
            self.positions.push([b.x, b.y, z_top]);
            self.positions.push([a.x, a.y, z_bottom]);
            self.positions.push([b.x, b.y, z_bottom]);
            for _ in 0..4 { self.normals.push(normal); }
            // Two triangles for the quad (top-left, top-right, bot-left, bot-right)
            self.indices.extend_from_slice(&[
                base, base + 1, base + 2,
                base + 1, base + 3, base + 2,
            ]);
        }
    }

    fn build(self) -> Mesh {
        let mut m = Mesh::new(
            PrimitiveTopology::TriangleList,
            RenderAssetUsages::RENDER_WORLD,
        );
        let n = self.positions.len();
        let uvs: Vec<[f32; 2]> = vec![[0.0, 0.0]; n];
        m.insert_attribute(Mesh::ATTRIBUTE_POSITION, self.positions);
        m.insert_attribute(Mesh::ATTRIBUTE_NORMAL, self.normals);
        m.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs);
        m.insert_indices(Indices::U32(self.indices));
        m
    }
}

fn push_ring_to_path(ring: &[Vec2], pb: &mut lyon::path::path::Builder) {
    if ring.len() < 3 { return; }
    pb.begin(lyon_point(ring[0].x, ring[0].y));
    for p in &ring[1..] {
        pb.line_to(lyon_point(p.x, p.y));
    }
    pb.end(true);
}

// ─── Buttons + UI ─────────────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum WidgetId {
    Digit(u8),
    Add,
    Sub,
    Mul,
    Div,
    Equals,
    Decimal,
}

#[derive(Clone, Copy)]
enum BtnColor { Cream, Coral }

struct BtnSpec {
    id: WidgetId,
    glyph: &'static str,
    color: BtnColor,
    col: u8,
    row: u8,
}

const BUTTONS: &[BtnSpec] = &[
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

fn btn_center(spec: &BtnSpec) -> Vec2 {
    let cx = (spec.col as f32 - 1.5) * (BTN_W + BTN_GAP_X);
    let cy = BTN_TOP_ROW_Y - spec.row as f32 * (BTN_H_SIZE + BTN_GAP_Y);
    Vec2::new(cx, cy)
}

// ─── Mesh constructors per scene piece ────────────────────────────────

fn build_paper_mesh() -> Mesh {
    let mut mb = MeshBuilder::default();
    let exterior = ensure_ccw(rounded_rect_ring(0.0, 0.0, PAPER_W, PAPER_H, PAPER_R));
    let panel_hole = ensure_cw(rounded_rect_ring(0.0, 0.0, PANEL_W, PANEL_H, PANEL_R));

    // Top face: paper rect with panel hole.
    mb.add_cap(
        &Polygon { exterior: exterior.clone(), holes: vec![panel_hole.clone()] },
        Z_PAPER,
        true,
    );
    // Outer wall down to floor.
    mb.add_wall(&exterior, Z_PAPER, Z_FLOOR);
    // Cavity wall (inside the panel cutout) down to panel surface.
    mb.add_wall(&panel_hole, Z_PAPER, Z_PANEL);
    mb.build()
}

fn build_panel_mesh() -> Mesh {
    let mut mb = MeshBuilder::default();
    let exterior = ensure_ccw(rounded_rect_ring(0.0, 0.0, PANEL_W, PANEL_H, PANEL_R));
    let display_hole = ensure_cw(
        rounded_rect_ring(0.0, DISPLAY_Y, DISPLAY_W, DISPLAY_H_SIZE, DISPLAY_R),
    );
    mb.add_cap(
        &Polygon { exterior, holes: vec![display_hole.clone()] },
        Z_PANEL,
        true,
    );
    // Cavity wall around the display, down to display surface.
    mb.add_wall(&display_hole, Z_PANEL, Z_DISPLAY);
    mb.build()
}

fn build_display_mesh(font: &[u8], display_text: &str) -> Mesh {
    let mut mb = MeshBuilder::default();
    let exterior = ensure_ccw(
        rounded_rect_ring(0.0, DISPLAY_Y, DISPLAY_W, DISPLAY_H_SIZE, DISPLAY_R),
    );

    // Layout digits, right-aligned in the display.
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

    // Display top: display_rect with each digit polygon as holes.
    // We add each digit's exterior as a hole; for digits with internal
    // holes (like "0"), those become engrave-floor "islands" we add
    // as separate caps below.
    let holes: Vec<Vec<Vec2>> = digit_polys
        .iter()
        .map(|p| ensure_cw(p.exterior.clone()))
        .collect();
    mb.add_cap(
        &Polygon { exterior, holes },
        Z_DISPLAY,
        true,
    );

    // Engrave walls (sides of each digit cavity).
    for poly in &digit_polys {
        let cavity_ring = ensure_cw(poly.exterior.clone());
        mb.add_wall(&cavity_ring, Z_DISPLAY, Z_DISPLAY_INK);
        // Inner holes of each digit (e.g. inside "0") become walls
        // going UP from the ink floor back to the display surface,
        // and an island cap at the display height.
        for hole in &poly.holes {
            // hole is CW. To wall it bottom→top, reverse for CCW.
            let mut island_ring = hole.clone();
            island_ring.reverse(); // now CCW
            mb.add_wall(&island_ring, Z_DISPLAY, Z_DISPLAY_INK);
            // Island cap (back at display surface height).
            mb.add_cap(
                &Polygon::from_ring(ensure_ccw(island_ring)),
                Z_DISPLAY,
                true,
            );
        }
    }

    // Engrave floor: filled digit shapes at ink height.
    for poly in &digit_polys {
        // Strip holes out of the floor shape (the holes are islands
        // raised back up by the loop above; the ink floor itself is
        // continuous under them and we don't render the floor under
        // an island, but the visual hidden-by-the-island makes that
        // moot).
        mb.add_cap(
            &Polygon { exterior: poly.exterior.clone(), holes: poly.holes.clone() },
            Z_DISPLAY_INK,
            true,
        );
    }

    mb.build()
}

/// Builds a button mesh with its top face at z = `Z_BTN_FACE_REST`
/// (face up) and bottom at `Z_BTN_BOTTOM`. The mesh sits centered at
/// (0, 0) — the entity transform places it on the panel.
fn build_button_mesh(font: &[u8], spec: &BtnSpec) -> Mesh {
    let mut mb = MeshBuilder::default();
    let exterior = ensure_ccw(rounded_rect_ring(0.0, 0.0, BTN_W, BTN_H_SIZE, BTN_R));

    // Glyph polygons centered on (0,0).
    let glyph_polys = layout_text_centered(font, spec.glyph, Vec2::ZERO, BTN_GLYPH_EM);

    // Top face: button rect with each glyph polygon's exterior as a hole.
    let glyph_holes: Vec<Vec<Vec2>> =
        glyph_polys.iter().map(|p| ensure_cw(p.exterior.clone())).collect();
    mb.add_cap(
        &Polygon { exterior: exterior.clone(), holes: glyph_holes },
        Z_BTN_FACE_REST,
        true,
    );

    // Outer side wall (face down to bottom).
    mb.add_wall(&exterior, Z_BTN_FACE_REST, Z_BTN_BOTTOM);

    // Engrave walls (inside the glyph) and engrave floor.
    let engrave_z = Z_BTN_FACE_REST - Z_BTN_ENGRAVE_DELTA;
    for poly in &glyph_polys {
        let cavity_ring = ensure_cw(poly.exterior.clone());
        mb.add_wall(&cavity_ring, Z_BTN_FACE_REST, engrave_z);
        // Letter holes ("0", "8") become islands raised back to face.
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
    // Engrave floor at engrave_z.
    for poly in &glyph_polys {
        mb.add_cap(
            &Polygon { exterior: poly.exterior.clone(), holes: poly.holes.clone() },
            engrave_z,
            true,
        );
    }

    mb.build()
}

// ─── Components & resources ───────────────────────────────────────────

#[derive(Component)]
struct Btn {
    id: WidgetId,
    home_z: f32,
    rect_world_center: Vec2,
}

#[derive(Resource, Default)]
struct UiState {
    pressed: Option<WidgetId>,
}

#[derive(Component)]
struct MainCam;

#[derive(Resource)]
struct CamCtl {
    yaw: f32,
    tilt: f32,
    dist: f32,
    target: Vec3,
}

impl CamCtl {
    fn default_pose() -> Self {
        Self { yaw: 0.0, tilt: 0.55, dist: 22.0, target: Vec3::new(0.0, -0.4, 0.0) }
    }
}

// ─── Setup ────────────────────────────────────────────────────────────

fn load_font() -> Vec<u8> {
    for path in FONT_FALLBACKS {
        if let Ok(b) = std::fs::read(path) {
            if Face::parse(&b, 0).is_ok() {
                return b;
            }
        }
    }
    panic!("no usable font found");
}

fn cream_mat(materials: &mut Assets<StandardMaterial>) -> Handle<StandardMaterial> {
    materials.add(StandardMaterial {
        base_color: Color::srgb(0.905, 0.860, 0.745),
        perceptual_roughness: 0.92,
        metallic: 0.0,
        reflectance: 0.30,
        ..default()
    })
}

fn taupe_mat(materials: &mut Assets<StandardMaterial>) -> Handle<StandardMaterial> {
    materials.add(StandardMaterial {
        base_color: Color::srgb(0.460, 0.405, 0.340),
        perceptual_roughness: 0.92,
        metallic: 0.0,
        reflectance: 0.25,
        ..default()
    })
}

fn olive_mat(materials: &mut Assets<StandardMaterial>) -> Handle<StandardMaterial> {
    materials.add(StandardMaterial {
        base_color: Color::srgb(0.585, 0.620, 0.470),
        perceptual_roughness: 0.92,
        metallic: 0.0,
        reflectance: 0.25,
        ..default()
    })
}

fn coral_mat(materials: &mut Assets<StandardMaterial>) -> Handle<StandardMaterial> {
    materials.add(StandardMaterial {
        base_color: Color::srgb(0.860, 0.470, 0.380),
        perceptual_roughness: 0.92,
        metallic: 0.0,
        reflectance: 0.30,
        ..default()
    })
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let font = load_font();

    // Camera (transform synced each frame from CamCtl).
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(0.0, 0.0, 22.0).looking_at(Vec3::ZERO, Vec3::Y),
        Projection::from(PerspectiveProjection {
            fov: 32f32.to_radians(),
            near: 0.1,
            far: 100.0,
            ..default()
        }),
        bevy::light::ShadowFilteringMethod::Gaussian,
        MainCam,
    ));

    // Lights — key + soft fill, key casts shadows.
    commands.spawn((
        DirectionalLight {
            color: Color::srgb(1.0, 0.96, 0.90),
            illuminance: 6_500.0,
            shadows_enabled: true,
            shadow_depth_bias: 0.04,
            shadow_normal_bias: 0.30,
            ..default()
        },
        CascadeShadowConfigBuilder {
            num_cascades: 1,
            minimum_distance: 0.05,
            first_cascade_far_bound: 80.0,
            maximum_distance: 80.0,
            ..default()
        }
        .build(),
        Transform::from_xyz(0.0, 0.0, 0.0)
            .looking_to(Vec3::new(0.30, 0.30, -0.90).normalize(), Vec3::Z),
    ));
    commands.spawn((
        DirectionalLight {
            color: Color::srgb(0.85, 0.88, 1.0),
            illuminance: 2_000.0,
            shadows_enabled: false,
            ..default()
        },
        Transform::from_xyz(0.0, 0.0, 0.0)
            .looking_to(Vec3::new(-0.40, 0.20, -0.85).normalize(), Vec3::Z),
    ));

    commands.insert_resource(GlobalAmbientLight {
        color: Color::WHITE,
        brightness: 240.0,
        affects_lightmapped_meshes: true,
    });

    // Materials.
    let cream = cream_mat(&mut materials);
    let taupe = taupe_mat(&mut materials);
    let olive = olive_mat(&mut materials);
    let coral = coral_mat(&mut materials);

    // Paper.
    let paper_mesh = meshes.add(build_paper_mesh());
    commands.spawn((Mesh3d(paper_mesh), MeshMaterial3d(cream.clone()), Transform::IDENTITY));

    // Panel.
    let panel_mesh = meshes.add(build_panel_mesh());
    commands.spawn((Mesh3d(panel_mesh), MeshMaterial3d(taupe), Transform::IDENTITY));

    // Display.
    let display_mesh = meshes.add(build_display_mesh(&font, "0"));
    commands.spawn((Mesh3d(display_mesh), MeshMaterial3d(olive), Transform::IDENTITY));

    // Buttons — each its own entity with its own mesh.
    for spec in BUTTONS {
        let mesh = meshes.add(build_button_mesh(&font, spec));
        let material = match spec.color {
            BtnColor::Cream => cream.clone(),
            BtnColor::Coral => coral.clone(),
        };
        let center = btn_center(spec);
        commands.spawn((
            Mesh3d(mesh),
            MeshMaterial3d(material),
            Transform::from_xyz(center.x, center.y, 0.0),
            Btn { id: spec.id, home_z: 0.0, rect_world_center: center },
        ));
    }

    commands.insert_resource(UiState::default());
    commands.insert_resource(CamCtl::default_pose());
}

// ─── Per-frame systems ────────────────────────────────────────────────

fn cursor_world_xy(
    windows: &Query<&Window>,
    cam_q: &Query<(&Camera, &GlobalTransform), With<MainCam>>,
) -> Option<Vec2> {
    let window = windows.iter().next()?;
    let cursor = window.cursor_position()?;
    let (camera, cam_xform) = cam_q.iter().next()?;
    let ray = camera.viewport_to_world(cam_xform, cursor).ok()?;
    let target_z = Z_BTN_FACE_REST;
    let dz = ray.direction.z;
    if dz.abs() < 1e-6 { return None; }
    let t = (target_z - ray.origin.z) / dz;
    if t < 0.0 { return None; }
    let p = ray.origin + ray.direction * t;
    Some(Vec2::new(p.x, p.y))
}

fn input_system(
    windows: Query<&Window>,
    cam_q: Query<(&Camera, &GlobalTransform), With<MainCam>>,
    mouse: Res<ButtonInput<MouseButton>>,
    mut ui: ResMut<UiState>,
    btn_q: Query<&Btn>,
) {
    let world = match cursor_world_xy(&windows, &cam_q) {
        Some(p) => p,
        None => return,
    };
    let half_w = BTN_W * 0.5;
    let half_h = BTN_H_SIZE * 0.5;
    let mut hovered: Option<WidgetId> = None;
    for btn in btn_q.iter() {
        let d = world - btn.rect_world_center;
        if d.x.abs() <= half_w && d.y.abs() <= half_h {
            hovered = Some(btn.id);
            break;
        }
    }
    if mouse.just_pressed(MouseButton::Left) {
        ui.pressed = hovered;
    }
    if mouse.just_released(MouseButton::Left) {
        ui.pressed = None;
    }
}

const TWEEN_TAU: f32 = 0.06;

fn animate_buttons(
    time: Res<Time>,
    ui: Res<UiState>,
    mut q: Query<(&Btn, &mut Transform)>,
) {
    let dt = time.delta_secs();
    let alpha = 1.0 - (-dt / TWEEN_TAU).exp();
    let press_drop = Z_BTN_FACE_PRESSED - Z_BTN_FACE_REST;
    for (btn, mut xf) in q.iter_mut() {
        let target = if Some(btn.id) == ui.pressed { btn.home_z + press_drop } else { btn.home_z };
        let cur = xf.translation.z;
        let next = cur + (target - cur) * alpha;
        xf.translation.z = if (next - target).abs() < 1e-4 { target } else { next };
    }
}

fn camera_input_system(
    keys: Res<ButtonInput<KeyCode>>,
    mut wheel: MessageReader<MouseWheel>,
    time: Res<Time>,
    mut cam: ResMut<CamCtl>,
) {
    let dt = time.delta_secs();
    if keys.pressed(KeyCode::ArrowLeft)  { cam.yaw -= 1.6 * dt; }
    if keys.pressed(KeyCode::ArrowRight) { cam.yaw += 1.6 * dt; }
    if keys.pressed(KeyCode::ArrowDown)  { cam.tilt = (cam.tilt + 1.2 * dt).clamp(0.0, 1.45); }
    if keys.pressed(KeyCode::ArrowUp)    { cam.tilt = (cam.tilt - 1.2 * dt).clamp(0.0, 1.45); }
    for ev in wheel.read() {
        let n = match ev.unit {
            MouseScrollUnit::Line => ev.y,
            MouseScrollUnit::Pixel => ev.y * 0.02,
        };
        if n.abs() < 1e-4 { continue; }
        let factor = (1.0 - n * 0.05).clamp(0.92, 1.08);
        cam.dist = (cam.dist * factor).clamp(4.0, 60.0);
    }
    if keys.just_pressed(KeyCode::KeyR) { *cam = CamCtl::default_pose(); }
    if keys.just_pressed(KeyCode::KeyT) {
        cam.tilt = if cam.tilt < 0.4 { 0.85 } else { 0.0 };
    }
}

fn sync_camera(cam: Res<CamCtl>, mut q: Query<&mut Transform, With<MainCam>>) {
    let Ok(mut t) = q.single_mut() else { return };
    let sin_t = cam.tilt.sin();
    let cos_t = cam.tilt.cos();
    let offset = Vec3::new(sin_t * cam.yaw.sin(), -sin_t * cam.yaw.cos(), cos_t) * cam.dist;
    let pos = cam.target + offset;
    let up = if cam.tilt < 0.05 { Vec3::Y } else { Vec3::Z };
    *t = Transform::from_translation(pos).looking_at(cam.target, up);
}

// ─── Main ─────────────────────────────────────────────────────────────

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "proper mesh calculator".into(),
                resolution: (1200u32, 1000u32).into(),
                ..default()
            }),
            ..default()
        }))
        .insert_resource(ClearColor(Color::srgb(0.10, 0.09, 0.13)))
        .insert_resource(bevy::light::DirectionalLightShadowMap { size: 4096 })
        .add_systems(Startup, setup)
        .add_systems(
            Update,
            (
                camera_input_system,
                sync_camera,
                input_system,
                animate_buttons,
            )
                .chain(),
        )
        .run();
}

// HashMap import not actually used; keep for symmetry with siblings.
#[allow(dead_code)]
fn _unused_hashmap_compat() -> HashMap<u8, u8> { HashMap::new() }
