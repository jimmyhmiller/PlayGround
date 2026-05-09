//! Heightmap-based recreation of the calculator scene from
//! `button_counter.rs`. The scene is a pure function of UiState; on
//! each change we re-rasterize all the shapes (panel, display, digits,
//! 16 buttons + glyphs) into a 1024² heightmap + layer-id texture.
//! No polygon clipping, no mesh rebuilds. Click-and-release on a
//! button sinks it briefly, then springs back.

use bevy::asset::RenderAssetUsages;
use bevy::image::{Image, ImageAddressMode, ImageFilterMode, ImageSampler, ImageSamplerDescriptor};
use bevy::input::ButtonInput;
use bevy::input::mouse::{MouseScrollUnit, MouseWheel};
use bevy::mesh::{Indices, PrimitiveTopology};
use bevy::prelude::*;
use bevy::render::render_resource::{
    AsBindGroup, Extent3d, ShaderType, TextureDimension, TextureFormat,
};
use bevy::shader::ShaderRef;
use lyon::math::point as lyon_point;
use lyon::path::iterator::PathIterator;
use lyon::path::{Path as LyonPath, PathEvent};
use std::collections::HashMap;
use ttf_parser::{Face, GlyphId, OutlineBuilder};

// ─── Constants ────────────────────────────────────────────────────────

const PAPER_W: f32 = 14.0;
const PAPER_H: f32 = 17.5;

/// Heightmap resolution. 1024² gives ~0.014 world-units per texel,
/// fine enough that button-glyph silhouettes look crisp from arm's
/// length but coarse enough that 1MB R16 + 1MB R8 stays cheap.
const HEIGHTMAP_RES: u32 = 1024;
/// Match grid to heightmap so each grid vertex samples its own texel.
/// Smaller than this and the silhouette of every shape staircases at
/// grid resolution rather than texel resolution.
const GRID_RES: u32 = 1024;

/// Maximum heightmap height in world units. heightmap_value 1.0 →
/// HEIGHT_SCALE world units of z displacement.
const HEIGHT_SCALE: f32 = 5.76;

// ─── Scene depths (heightmap-normalised heights, 0..1) ────────────────
//
// These mirror the per-layer `step` values from `button_counter.rs`,
// divided through by HEIGHT_SCALE so they live in [0,1].

const H_PAPER: f32 = 3.50 / HEIGHT_SCALE;
const H_PANEL: f32 = 0.85 / HEIGHT_SCALE;
const H_DISPLAY: f32 = 0.45 / HEIGHT_SCALE;
const H_DISPLAY_INK: f32 = 0.05 / HEIGHT_SCALE;
const H_BTN_IDLE: f32 = 2.80 / HEIGHT_SCALE;
const H_BTN_PRESS: f32 = 1.55 / HEIGHT_SCALE;
const H_ENGRAVE_DELTA: f32 = 0.40 / HEIGHT_SCALE;

// ─── Layer ids (palette indices) ──────────────────────────────────────

const ID_PAPER: u8 = 0;
const ID_PANEL: u8 = 1;
const ID_DISPLAY: u8 = 2;
const ID_DISPLAY_INK: u8 = 3;
const ID_BTN_CREAM: u8 = 4;
const ID_BTN_CORAL: u8 = 5;
const ID_ENGRAVE_CREAM: u8 = 6;
const ID_ENGRAVE_CORAL: u8 = 7;

// ─── Panel layout (matches button_counter.rs) ─────────────────────────

const PANEL_W: f32 = 10.6;
const PANEL_H: f32 = 13.6;
const PANEL_R: f32 = 0.65;

const DISPLAY_W: f32 = 9.4;
const DISPLAY_H: f32 = 1.85;
const DISPLAY_R: f32 = 0.40;
const DISPLAY_Y: f32 = 4.85;
const DISPLAY_DIGIT_EM: f32 = 1.10;

const BTN_W: f32 = 1.85;
const BTN_H: f32 = 1.55;
const BTN_R: f32 = 0.50;
const BTN_GAP_X: f32 = 0.25;
const BTN_GAP_Y: f32 = 0.25;
const BTN_TOP_ROW_Y: f32 = 2.45;
const BTN_GLYPH_EM: f32 = 0.85;

const FONT_FALLBACKS: &[&str] = &[
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/Library/Fonts/Arial.ttf",
];

// ─── Material ─────────────────────────────────────────────────────────

#[derive(ShaderType, Clone, Copy, Debug)]
struct PaperParams {
    paper_size: Vec2,
    height_scale: f32,
    _pad0: f32,
    inv_resolution: Vec2,
    _pad1: Vec2,
    light_dir: Vec3,
    _pad2: f32,
    palette: [Vec4; 8],
}

#[derive(Asset, AsBindGroup, TypePath, Clone)]
struct PaperMaterial {
    #[texture(0)]
    #[sampler(1)]
    heightmap: Handle<Image>,
    #[texture(2, sample_type = "u_int")]
    layer_id: Handle<Image>,
    #[uniform(3)]
    params: PaperParams,
}

impl Material for PaperMaterial {
    fn vertex_shader() -> ShaderRef {
        "shaders/displace.wgsl".into()
    }
    fn fragment_shader() -> ShaderRef {
        "shaders/displace.wgsl".into()
    }
}

// ─── Buttons + UI state ───────────────────────────────────────────────

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
enum BtnColor {
    Cream,
    Coral,
}

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
    let cy = BTN_TOP_ROW_Y - spec.row as f32 * (BTN_H + BTN_GAP_Y);
    Vec2::new(cx, cy)
}

#[derive(Resource)]
struct UiState {
    /// Animated face height per button, in normalised heightmap units.
    face_h: HashMap<WidgetId, f32>,
    pressed: Option<WidgetId>,
    display: String,
    /// Pre-computed glyph polygons per button (one per glyph), in local
    /// coords centered on the button center. Cached because glyph
    /// outlining isn't free.
    glyph_cache: HashMap<WidgetId, Vec<Polygon>>,
    /// Display digit polygons for the current `display` string,
    /// already translated into world space (right-aligned in display).
    /// Recomputed when `display` changes.
    display_glyphs: Vec<Polygon>,
    display_text: String,
}

impl UiState {
    fn new(font: &[u8]) -> Self {
        let mut glyph_cache = HashMap::new();
        for spec in BUTTONS {
            let center = btn_center(spec);
            let polys = layout_text_centered(font, spec.glyph, center, BTN_GLYPH_EM);
            glyph_cache.insert(spec.id, polys);
        }
        let face_h = BUTTONS.iter().map(|b| (b.id, H_BTN_IDLE)).collect();
        let mut s = Self {
            face_h,
            pressed: None,
            display: "0".into(),
            glyph_cache,
            display_glyphs: Vec::new(),
            display_text: String::new(),
        };
        let _ = s.refresh_display(font);
        s
    }

    fn refresh_display(&mut self, font: &[u8]) -> bool {
        if self.display == self.display_text {
            return false;
        }
        self.display_text = self.display.clone();
        // Right-align the glyphs inside the display rect.
        let raw = layout_text_centered(font, &self.display, Vec2::ZERO, DISPLAY_DIGIT_EM);
        if raw.is_empty() {
            self.display_glyphs = Vec::new();
            return true;
        }
        let mut xmin = f32::INFINITY;
        let mut xmax = f32::NEG_INFINITY;
        for poly in &raw {
            for [x, _] in poly.exterior_iter() {
                xmin = xmin.min(x);
                xmax = xmax.max(x);
            }
        }
        let glyph_w = xmax - xmin;
        let half_w = glyph_w * 0.5;
        let target_right = DISPLAY_W * 0.5 - 0.55;
        let dx = target_right - half_w;
        let dy = DISPLAY_Y - 0.05;
        self.display_glyphs = raw
            .into_iter()
            .map(|p| p.translated(dx, dy))
            .collect();
        true
    }
}

// ─── Polygon type (simple, with holes; not geo crate) ─────────────────

#[derive(Clone)]
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
            holes: self
                .holes
                .iter()
                .map(|h| h.iter().map(|p| *p + off).collect())
                .collect(),
        }
    }

    fn bbox(&self) -> (Vec2, Vec2) {
        let mut lo = Vec2::splat(f32::INFINITY);
        let mut hi = Vec2::splat(f32::NEG_INFINITY);
        for p in &self.exterior {
            lo = lo.min(*p);
            hi = hi.max(*p);
        }
        (lo, hi)
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

    fn exterior_iter(&self) -> impl Iterator<Item = [f32; 2]> + '_ {
        self.exterior.iter().map(|p| [p.x, p.y])
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
            let intersect_x = (xj - xi) * (p.y - yi) / (yj - yi) + xi;
            if p.x < intersect_x {
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
    let segs = 16u32;
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

fn rounded_rect(cx: f32, cy: f32, w: f32, h: f32, r: f32) -> Polygon {
    Polygon::from_ring(rounded_rect_ring(cx, cy, w, h, r))
}

fn btn_polygon(spec: &BtnSpec) -> Polygon {
    let c = btn_center(spec);
    rounded_rect(c.x, c.y, BTN_W, BTN_H, BTN_R)
}

// ─── Glyph outlines (ttf-parser + lyon flattening, hole-aware) ────────

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
    fn move_to(&mut self, x: f32, y: f32) {
        self.ops.push(Op::Move(self.t(x, y)));
    }
    fn line_to(&mut self, x: f32, y: f32) {
        self.ops.push(Op::Line(self.t(x, y)));
    }
    fn quad_to(&mut self, x1: f32, y1: f32, x: f32, y: f32) {
        self.ops.push(Op::Quad(self.t(x1, y1), self.t(x, y)));
    }
    fn curve_to(&mut self, x1: f32, y1: f32, x2: f32, y2: f32, x: f32, y: f32) {
        self.ops.push(Op::Cubic(self.t(x1, y1), self.t(x2, y2), self.t(x, y)));
    }
    fn close(&mut self) {
        self.ops.push(Op::Close);
    }
}

fn ops_to_path(ops: &[Op]) -> LyonPath {
    let mut pb = LyonPath::builder();
    let mut started = false;
    for op in ops {
        match *op {
            Op::Move(p) => {
                if started {
                    pb.end(false);
                }
                pb.begin(lyon_point(p.x, p.y));
                started = true;
            }
            Op::Line(p) => {
                pb.line_to(lyon_point(p.x, p.y));
            }
            Op::Quad(c, p) => {
                pb.quadratic_bezier_to(lyon_point(c.x, c.y), lyon_point(p.x, p.y));
            }
            Op::Cubic(c1, c2, p) => {
                pb.cubic_bezier_to(
                    lyon_point(c1.x, c1.y),
                    lyon_point(c2.x, c2.y),
                    lyon_point(p.x, p.y),
                );
            }
            Op::Close => {
                if started {
                    pb.end(true);
                    started = false;
                }
            }
        }
    }
    if started {
        pb.end(false);
    }
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
            PathEvent::Line { to, .. } => {
                current.push(Vec2::new(to.x, to.y));
            }
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
    if n == 0 {
        return Vec::new();
    }
    let mut depths = vec![0usize; n];
    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            if point_in_ring(valid[i][0], valid[j]) {
                depths[i] += 1;
            }
        }
    }
    let mut polygons = Vec::new();
    for i in 0..n {
        if depths[i] % 2 != 0 {
            continue;
        }
        let mut outer = valid[i].clone();
        if signed_area(&outer) < 0.0 {
            outer.reverse();
        }
        let mut holes: Vec<Vec<Vec2>> = Vec::new();
        for j in 0..n {
            if i == j || depths[j] != depths[i] + 1 {
                continue;
            }
            if point_in_ring(valid[j][0], &outer) {
                let mut hole = valid[j].clone();
                if signed_area(&hole) > 0.0 {
                    hole.reverse();
                }
                holes.push(hole);
            }
        }
        polygons.push(Polygon { exterior: outer, holes });
    }
    polygons
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
    let mut bounds_min = Vec2::splat(f32::INFINITY);
    let mut bounds_max = Vec2::splat(f32::NEG_INFINITY);

    for ch in text.chars() {
        let Some(gid) = face.glyph_index(ch) else {
            x += em * 0.3;
            continue;
        };
        let advance = face.glyph_hor_advance(gid).unwrap_or(0) as f32 * scale;
        let path = outline_glyph_to_path(&face, gid, scale, Vec2::new(x, 0.0));
        let contours = flatten_path_to_contours(&path, 0.005);
        for c in &contours {
            for p in c {
                bounds_min = bounds_min.min(*p);
                bounds_max = bounds_max.max(*p);
            }
        }
        polys.extend(build_glyph_polygons(&contours));
        x += advance;
    }
    if !bounds_min.is_finite() {
        return polys;
    }
    let center = (bounds_min + bounds_max) * 0.5;
    let dx = origin.x - center.x;
    let dy = origin.y - center.y;
    polys.into_iter().map(|p| p.translated(dx, dy)).collect()
}

// ─── Heightmap state ──────────────────────────────────────────────────

#[derive(Resource)]
struct Heightmap {
    pixels: Vec<u16>,
    ids: Vec<u8>,
    /// Cached background (paper + panel + display + display digits)
    /// — rebuilt only when the display string changes.
    bg_pixels: Vec<u16>,
    bg_ids: Vec<u8>,
    bg_dirty: bool,
    dirty: bool,
    height_handle: Handle<Image>,
    layer_handle: Handle<Image>,
}

const TEXEL_W: f32 = PAPER_W / HEIGHTMAP_RES as f32;
const TEXEL_H: f32 = PAPER_H / HEIGHTMAP_RES as f32;

/// Width of the boundary AA band, in texels, used to ramp height down
/// at every shape edge. Wider bands → walls become longer slopes
/// → fewer fringe artifacts at oblique view angles, at the cost of
/// softer silhouettes. Heightmap rendering of carved scenes needs the
/// walls to be slopes, not vertical faces — single-texel transitions
/// produce per-texel jitter along curved edges that reads as grass on
/// the walls. 4–6 texels is the sweet spot.
const BEVEL_TEXELS: f32 = 5.0;

impl Heightmap {
    fn new(height_handle: Handle<Image>, layer_handle: Handle<Image>) -> Self {
        let n = (HEIGHTMAP_RES * HEIGHTMAP_RES) as usize;
        Self {
            pixels: vec![0; n],
            ids: vec![ID_PAPER; n],
            bg_pixels: vec![0; n],
            bg_ids: vec![ID_PAPER; n],
            bg_dirty: true,
            dirty: true,
            height_handle,
            layer_handle,
        }
    }

    /// Rebuild the cached background. Cheap because it only happens
    /// when the display string changes.
    fn rebuild_background(&mut self, ui: &UiState) {
        let v = (H_PAPER.clamp(0.0, 1.0) * u16::MAX as f32) as u16;
        self.bg_pixels.fill(v);
        self.bg_ids.fill(ID_PAPER);
        rasterize_rounded_rect_aa(
            &mut self.bg_pixels,
            &mut self.bg_ids,
            0.0, 0.0, PANEL_W, PANEL_H, PANEL_R,
            H_PANEL, ID_PANEL,
        );
        rasterize_rounded_rect_aa(
            &mut self.bg_pixels,
            &mut self.bg_ids,
            0.0, DISPLAY_Y, DISPLAY_W, DISPLAY_H, DISPLAY_R,
            H_DISPLAY, ID_DISPLAY,
        );
        for poly in &ui.display_glyphs {
            rasterize_polygon_aa(
                &mut self.bg_pixels,
                &mut self.bg_ids,
                poly,
                H_DISPLAY_INK,
                ID_DISPLAY_INK,
            );
        }
        self.bg_dirty = false;
    }

    fn rebuild(&mut self, ui: &UiState) {
        if self.bg_dirty {
            self.rebuild_background(ui);
        }
        // Start from the cached background, then composite the
        // dynamic (animated) foreground on top.
        self.pixels.copy_from_slice(&self.bg_pixels);
        self.ids.copy_from_slice(&self.bg_ids);
        for spec in BUTTONS {
            let face_h = ui.face_h.get(&spec.id).copied().unwrap_or(H_BTN_IDLE);
            let face_id = match spec.color {
                BtnColor::Cream => ID_BTN_CREAM,
                BtnColor::Coral => ID_BTN_CORAL,
            };
            let engrave_id = match spec.color {
                BtnColor::Cream => ID_ENGRAVE_CREAM,
                BtnColor::Coral => ID_ENGRAVE_CORAL,
            };
            let c = btn_center(spec);
            rasterize_rounded_rect_aa(
                &mut self.pixels, &mut self.ids,
                c.x, c.y, BTN_W, BTN_H, BTN_R,
                face_h, face_id,
            );
            let engrave_h = (face_h - H_ENGRAVE_DELTA).max(0.0);
            if let Some(glyph_polys) = ui.glyph_cache.get(&spec.id) {
                for poly in glyph_polys {
                    rasterize_polygon_aa(
                        &mut self.pixels, &mut self.ids,
                        poly, engrave_h, engrave_id,
                    );
                }
            }
        }
        self.dirty = true;
    }
}

fn world_bbox_to_texels(lo: Vec2, hi: Vec2) -> (i32, i32, i32, i32) {
    let to_u = |x: f32| ((x + PAPER_W * 0.5) / PAPER_W) * HEIGHTMAP_RES as f32;
    let to_v = |y: f32| ((y + PAPER_H * 0.5) / PAPER_H) * HEIGHTMAP_RES as f32;
    (
        (to_u(lo.x).floor() as i32 - 1).max(0),
        (to_u(hi.x).ceil() as i32 + 1).min(HEIGHTMAP_RES as i32),
        (to_v(lo.y).floor() as i32 - 1).max(0),
        (to_v(hi.y).ceil() as i32 + 1).min(HEIGHTMAP_RES as i32),
    )
}

#[inline(always)]
fn write_blend(pixels: &mut [u16], ids: &mut [u8], i: usize, target_h_norm: f32, id: u8, coverage: f32) {
    if coverage <= 0.0 {
        return;
    }
    let prev_h = pixels[i] as f32 / u16::MAX as f32;
    let new_h = coverage * target_h_norm + (1.0 - coverage) * prev_h;
    pixels[i] = (new_h.clamp(0.0, 1.0) * u16::MAX as f32) as u16;
    if coverage >= 0.5 {
        ids[i] = id;
    }
}

/// Signed distance from a point to a rounded rectangle. Negative
/// inside, positive outside.
#[inline(always)]
fn sd_rounded_rect(p: Vec2, c: Vec2, w: f32, h: f32, r: f32) -> f32 {
    let d = (p - c).abs() - Vec2::new(w * 0.5 - r, h * 0.5 - r);
    Vec2::new(d.x.max(0.0), d.y.max(0.0)).length() + d.x.max(d.y).min(0.0) - r
}

fn rasterize_rounded_rect_aa(
    pixels: &mut [u16],
    ids: &mut [u8],
    cx: f32, cy: f32, w: f32, h: f32, r: f32,
    target_h_norm: f32, id: u8,
) {
    let texel_diag_half = (TEXEL_W * TEXEL_W + TEXEL_H * TEXEL_H).sqrt() * 0.5;
    // Half-width of the bevel ramp, in world units.
    let band_half = BEVEL_TEXELS * texel_diag_half;
    let lo = Vec2::new(cx - w * 0.5 - band_half * 2.0, cy - h * 0.5 - band_half * 2.0);
    let hi = Vec2::new(cx + w * 0.5 + band_half * 2.0, cy + h * 0.5 + band_half * 2.0);
    let (x0, x1, y0, y1) = world_bbox_to_texels(lo, hi);
    let center = Vec2::new(cx, cy);

    // Fast inside the interior-only rectangle (no rounded corner, no
    // edge blending needed); slope only the boundary band.
    let inner_w_half = (w * 0.5 - r - band_half).max(0.0);
    let inner_h_half = (h * 0.5 - r - band_half).max(0.0);
    let outer_w_half = w * 0.5 + band_half;
    let outer_h_half = h * 0.5 + band_half;

    let target_u16 = (target_h_norm.clamp(0.0, 1.0) * u16::MAX as f32) as u16;

    for py in y0..y1 {
        let wy = (py as f32 + 0.5) * TEXEL_H - PAPER_H * 0.5;
        let dy = (wy - cy).abs();
        if dy > outer_h_half {
            continue;
        }
        let row_base = (py as u32 * HEIGHTMAP_RES) as usize;
        for px in x0..x1 {
            let wx = (px as f32 + 0.5) * TEXEL_W - PAPER_W * 0.5;
            let dx = (wx - cx).abs();
            if dx > outer_w_half {
                continue;
            }
            let i = row_base + px as usize;
            // Definitely interior: cheap write, no distance calc.
            if dx < inner_w_half && dy < inner_h_half {
                pixels[i] = target_u16;
                ids[i] = id;
                continue;
            }
            // Boundary band: signed distance → coverage. The band is
            // BEVEL_TEXELS wide so the wall reads as a slope rather
            // than a 1-texel cliff.
            let d = sd_rounded_rect(Vec2::new(wx, wy), center, w, h, r);
            let coverage = (0.5 - d / (2.0 * band_half)).clamp(0.0, 1.0);
            write_blend(pixels, ids, i, target_h_norm, id, coverage);
        }
    }
}

/// Polygon rasterization with 4×4 supersampling at boundary texels.
/// Boundary detection by 4-corner agreement test. Uses the polygon's
/// even-odd contains() (with hole support).
fn rasterize_polygon_aa(
    pixels: &mut [u16],
    ids: &mut [u8],
    poly: &Polygon,
    target_h_norm: f32,
    id: u8,
) {
    let (lo, hi) = poly.bbox();
    let (x0, x1, y0, y1) = world_bbox_to_texels(
        Vec2::new(lo.x - 1.0, lo.y - 1.0),
        Vec2::new(hi.x + 1.0, hi.y + 1.0),
    );
    let target_u16 = (target_h_norm.clamp(0.0, 1.0) * u16::MAX as f32) as u16;

    for py in y0..y1 {
        let wy_c = (py as f32 + 0.5) * TEXEL_H - PAPER_H * 0.5;
        let row_base = (py as u32 * HEIGHTMAP_RES) as usize;
        for px in x0..x1 {
            let wx_c = (px as f32 + 0.5) * TEXEL_W - PAPER_W * 0.5;
            // Quick reject: 4-corner test. If all in or all out, no
            // need to supersample.
            let c00 = poly.contains(Vec2::new(wx_c - TEXEL_W * 0.5, wy_c - TEXEL_H * 0.5));
            let c01 = poly.contains(Vec2::new(wx_c - TEXEL_W * 0.5, wy_c + TEXEL_H * 0.5));
            let c10 = poly.contains(Vec2::new(wx_c + TEXEL_W * 0.5, wy_c - TEXEL_H * 0.5));
            let c11 = poly.contains(Vec2::new(wx_c + TEXEL_W * 0.5, wy_c + TEXEL_H * 0.5));
            let i = row_base + px as usize;
            if c00 && c01 && c10 && c11 {
                pixels[i] = target_u16;
                ids[i] = id;
                continue;
            }
            if !c00 && !c01 && !c10 && !c11 {
                // Could still cover sub-pixel features (very thin
                // glyph stems). Center test as a cheap catch.
                if !poly.contains(Vec2::new(wx_c, wy_c)) {
                    continue;
                }
            }
            // Mixed: 4×4 supersample.
            let mut count = 0u32;
            for sy in 0..4 {
                for sx in 0..4 {
                    let u_off = (sx as f32 + 0.5) / 4.0 - 0.5;
                    let v_off = (sy as f32 + 0.5) / 4.0 - 0.5;
                    let p = Vec2::new(wx_c + u_off * TEXEL_W, wy_c + v_off * TEXEL_H);
                    if poly.contains(p) {
                        count += 1;
                    }
                }
            }
            let coverage = count as f32 / 16.0;
            write_blend(pixels, ids, i, target_h_norm, id, coverage);
        }
    }
}

// ─── Scene rebuild ────────────────────────────────────────────────────


// ─── Camera + input ───────────────────────────────────────────────────

#[derive(Resource)]
struct CamCtl {
    yaw: f32,
    tilt: f32,
    dist: f32,
    target: Vec3,
}

impl CamCtl {
    fn default_pose() -> Self {
        Self { yaw: 0.0, tilt: 0.55, dist: 22.0, target: Vec3::ZERO }
    }
}

#[derive(Component)]
struct MainCam;

#[derive(Resource)]
struct PaperMaterialHandle(Handle<PaperMaterial>);

#[derive(Resource)]
struct LoadedFont(Vec<u8>);

// ─── Mesh + image construction ────────────────────────────────────────

fn build_grid_mesh(n: u32, size: Vec2) -> Mesh {
    let count = (n + 1) as usize;
    let mut positions = Vec::with_capacity(count * count);
    let mut normals = Vec::with_capacity(count * count);
    let mut uvs = Vec::with_capacity(count * count);
    let half = size * 0.5;
    for j in 0..=n {
        for i in 0..=n {
            let u = i as f32 / n as f32;
            let v = j as f32 / n as f32;
            let x = -half.x + u * size.x;
            let y = -half.y + v * size.y;
            positions.push([x, y, 0.0]);
            normals.push([0.0, 0.0, 1.0]);
            uvs.push([u, v]);
        }
    }
    let mut indices: Vec<u32> = Vec::with_capacity((n * n * 6) as usize);
    for j in 0..n {
        for i in 0..n {
            let a = j * (n + 1) + i;
            let b = a + 1;
            let c = a + (n + 1);
            let d = c + 1;
            indices.extend_from_slice(&[a, b, c, b, d, c]);
        }
    }
    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::RENDER_WORLD,
    );
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs);
    mesh.insert_indices(Indices::U32(indices));
    mesh
}

fn make_height_image() -> Image {
    let total = (HEIGHTMAP_RES * HEIGHTMAP_RES) as usize;
    let bytes: Vec<u8> = vec![0; total * 2];
    let mut image = Image::new_uninit(
        Extent3d {
            width: HEIGHTMAP_RES,
            height: HEIGHTMAP_RES,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        TextureFormat::R16Unorm,
        RenderAssetUsages::all(),
    );
    image.data = Some(bytes);
    image.sampler = ImageSampler::Descriptor(ImageSamplerDescriptor {
        address_mode_u: ImageAddressMode::ClampToEdge,
        address_mode_v: ImageAddressMode::ClampToEdge,
        address_mode_w: ImageAddressMode::ClampToEdge,
        mag_filter: ImageFilterMode::Linear,
        min_filter: ImageFilterMode::Linear,
        mipmap_filter: ImageFilterMode::Nearest,
        ..ImageSamplerDescriptor::linear()
    });
    image
}

fn make_layer_image() -> Image {
    let total = (HEIGHTMAP_RES * HEIGHTMAP_RES) as usize;
    let bytes: Vec<u8> = vec![ID_PAPER; total];
    let mut image = Image::new_uninit(
        Extent3d {
            width: HEIGHTMAP_RES,
            height: HEIGHTMAP_RES,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        TextureFormat::R8Uint,
        RenderAssetUsages::all(),
    );
    image.data = Some(bytes);
    image
}

fn palette() -> [Vec4; 8] {
    let cream = Vec3::new(0.905, 0.860, 0.745);
    let coral = Vec3::new(0.860, 0.470, 0.380);
    let cream_dark = Vec3::new(0.560, 0.495, 0.395);
    let coral_dark = Vec3::new(0.485, 0.235, 0.180);
    let taupe = Vec3::new(0.460, 0.405, 0.340);
    let display_olive = Vec3::new(0.585, 0.620, 0.470);
    let display_ink = Vec3::new(0.355, 0.380, 0.275);
    let mut p = [Vec4::ONE; 8];
    p[ID_PAPER as usize] = cream.extend(1.0);
    p[ID_PANEL as usize] = taupe.extend(1.0);
    p[ID_DISPLAY as usize] = display_olive.extend(1.0);
    p[ID_DISPLAY_INK as usize] = display_ink.extend(1.0);
    p[ID_BTN_CREAM as usize] = cream.extend(1.0);
    p[ID_BTN_CORAL as usize] = coral.extend(1.0);
    p[ID_ENGRAVE_CREAM as usize] = cream_dark.extend(1.0);
    p[ID_ENGRAVE_CORAL as usize] = coral_dark.extend(1.0);
    p
}

fn load_font_bytes() -> Vec<u8> {
    for path in FONT_FALLBACKS {
        if let Ok(b) = std::fs::read(path) {
            if Face::parse(&b, 0).is_ok() {
                return b;
            }
        }
    }
    panic!("no usable font found in {FONT_FALLBACKS:?}");
}

// ─── Setup ────────────────────────────────────────────────────────────

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<PaperMaterial>>,
    mut images: ResMut<Assets<Image>>,
    font: Res<LoadedFont>,
) {
    let height_handle = images.add(make_height_image());
    let layer_handle = images.add(make_layer_image());
    let mut hm = Heightmap::new(height_handle.clone(), layer_handle.clone());
    let ui = UiState::new(&font.0);
    hm.rebuild(&ui);
    commands.insert_resource(hm);
    commands.insert_resource(ui);

    let mesh_handle = meshes.add(build_grid_mesh(GRID_RES, Vec2::new(PAPER_W, PAPER_H)));

    let params = PaperParams {
        paper_size: Vec2::new(PAPER_W, PAPER_H),
        height_scale: HEIGHT_SCALE,
        _pad0: 0.0,
        inv_resolution: Vec2::splat(1.0 / HEIGHTMAP_RES as f32),
        _pad1: Vec2::ZERO,
        light_dir: Vec3::new(0.30, 0.30, -0.90).normalize(),
        _pad2: 0.0,
        palette: palette(),
    };

    let mat_handle = materials.add(PaperMaterial {
        heightmap: height_handle,
        layer_id: layer_handle,
        params,
    });
    commands.insert_resource(PaperMaterialHandle(mat_handle.clone()));

    commands.spawn((Mesh3d(mesh_handle), MeshMaterial3d(mat_handle), Transform::IDENTITY));

    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(0.0, 0.0, 22.0).looking_at(Vec3::ZERO, Vec3::Y),
        Projection::from(PerspectiveProjection {
            fov: 32f32.to_radians(),
            near: 0.1,
            far: 200.0,
            ..default()
        }),
        MainCam,
    ));
}

// ─── Per-frame systems ────────────────────────────────────────────────

fn upload_heightmap_if_dirty(
    mut hm: ResMut<Heightmap>,
    mut images: ResMut<Assets<Image>>,
    mut materials: ResMut<Assets<PaperMaterial>>,
    paper_mat: Res<PaperMaterialHandle>,
) {
    if !hm.dirty {
        return;
    }
    {
        let Some(image) = images.get_mut(&hm.height_handle) else { return };
        let mut bytes: Vec<u8> = Vec::with_capacity(hm.pixels.len() * 2);
        for &h in &hm.pixels {
            bytes.extend_from_slice(&h.to_le_bytes());
        }
        image.data = Some(bytes);
    }
    {
        let Some(image) = images.get_mut(&hm.layer_handle) else { return };
        image.data = Some(hm.ids.clone());
    }
    // Force the material's bind group to rebuild against the freshly
    // re-prepared GpuImages — without this Bevy holds onto the old
    // texture views.
    let _ = materials.get_mut(&paper_mat.0);
    hm.dirty = false;
}

fn cursor_world_xy(
    windows: &Query<&Window>,
    cam_q: &Query<(&Camera, &GlobalTransform), With<MainCam>>,
) -> Option<Vec2> {
    let window = windows.iter().next()?;
    let cursor = window.cursor_position()?;
    let (camera, cam_xform) = cam_q.iter().next()?;
    let ray = camera.viewport_to_world(cam_xform, cursor).ok()?;
    // Project onto the panel plane (button face surface).
    let target_z = H_BTN_IDLE * HEIGHT_SCALE;
    let dz = ray.direction.z;
    if dz.abs() < 1e-6 {
        return None;
    }
    let t = (target_z - ray.origin.z) / dz;
    if t < 0.0 {
        return None;
    }
    let p = ray.origin + ray.direction * t;
    Some(Vec2::new(p.x, p.y))
}

fn input_system(
    windows: Query<&Window>,
    cam_q: Query<(&Camera, &GlobalTransform), With<MainCam>>,
    mouse: Res<ButtonInput<MouseButton>>,
    mut ui: ResMut<UiState>,
) {
    let world = match cursor_world_xy(&windows, &cam_q) {
        Some(p) => p,
        None => return,
    };
    let mut hovered: Option<WidgetId> = None;
    for spec in BUTTONS {
        let rect = btn_polygon(spec);
        if rect.contains(world) {
            hovered = Some(spec.id);
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
    mut ui: ResMut<UiState>,
    mut hm: ResMut<Heightmap>,
    font: Res<LoadedFont>,
) {
    let dt = time.delta_secs();
    let alpha = 1.0 - (-dt / TWEEN_TAU).exp();
    let mut any_changed = false;
    let pressed = ui.pressed;
    let ids: Vec<WidgetId> = ui.face_h.keys().copied().collect();
    for id in ids {
        let target = if Some(id) == pressed { H_BTN_PRESS } else { H_BTN_IDLE };
        let cur = ui.face_h[&id];
        let next = cur + (target - cur) * alpha;
        let snapped = if (next - target).abs() < 1e-4 { target } else { next };
        if (snapped - cur).abs() > 1e-5 {
            ui.face_h.insert(id, snapped);
            any_changed = true;
        }
    }
    let display_changed = ui.refresh_display(&font.0);
    if display_changed {
        hm.bg_dirty = true;
    }
    if any_changed || display_changed {
        hm.rebuild(&ui);
    }
}

fn camera_input_system(
    keys: Res<ButtonInput<KeyCode>>,
    mut wheel: MessageReader<MouseWheel>,
    time: Res<Time>,
    mut cam: ResMut<CamCtl>,
) {
    let dt = time.delta_secs();
    let yaw_speed = 1.6;
    let tilt_speed = 1.2;
    if keys.pressed(KeyCode::ArrowLeft) {
        cam.yaw -= yaw_speed * dt;
    }
    if keys.pressed(KeyCode::ArrowRight) {
        cam.yaw += yaw_speed * dt;
    }
    if keys.pressed(KeyCode::ArrowDown) {
        cam.tilt = (cam.tilt + tilt_speed * dt).clamp(0.0, 1.45);
    }
    if keys.pressed(KeyCode::ArrowUp) {
        cam.tilt = (cam.tilt - tilt_speed * dt).clamp(0.0, 1.45);
    }
    for ev in wheel.read() {
        let normalized = match ev.unit {
            MouseScrollUnit::Line => ev.y,
            MouseScrollUnit::Pixel => ev.y * 0.02,
        };
        if normalized.abs() < 1e-4 {
            continue;
        }
        let factor = (1.0 - normalized * 0.05).clamp(0.92, 1.08);
        cam.dist = (cam.dist * factor).clamp(4.0, 60.0);
    }
    if keys.just_pressed(KeyCode::KeyR) {
        *cam = CamCtl::default_pose();
    }
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
    let font = load_font_bytes();
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "heightmap counter".into(),
                resolution: (1200u32, 1000u32).into(),
                ..default()
            }),
            ..default()
        }))
        .add_plugins(MaterialPlugin::<PaperMaterial>::default())
        .insert_resource(LoadedFont(font))
        .insert_resource(ClearColor(Color::srgb(0.10, 0.09, 0.13)))
        .insert_resource(CamCtl::default_pose())
        .add_systems(Startup, setup)
        .add_systems(
            Update,
            (
                camera_input_system,
                sync_camera,
                input_system,
                animate_buttons,
                upload_heightmap_if_dirty,
            )
                .chain(),
        )
        .run();
}
