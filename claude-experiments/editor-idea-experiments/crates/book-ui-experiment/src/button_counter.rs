//! Calculator scene built game-style: every visible piece is a static
//! mesh, spawned once at startup as its own Bevy entity. Press
//! animation = `Transform.translation.z` tween. Mesh data is *never*
//! rebuilt for animation, never rebuilt by polygon clipping. Mesh
//! rebuild only happens on debug-knob change (and there only for the
//! affected entities).
//!
//! Scene entities:
//!   - 1 paper (rounded paper rect with panel-shaped hole + walls)
//!   - 1 panel (rounded panel rect with display-shaped hole + walls)
//!   - 1 display (display rect with digit-glyph holes + engraved digits)
//!   - 16 buttons (each its own mesh: face with glyph hole, outer wall,
//!     glyph engrave wall, glyph engrave floor)
//!
//! `DebugKnobs` (press L) live-tunes layer depths, slope, and
//! animation targets; depth/slope knobs trigger an entity rebuild.

use bevy::asset::RenderAssetUsages;
use bevy::image::{
    Image, ImageAddressMode, ImageFilterMode, ImageSampler, ImageSamplerDescriptor,
};
use bevy::input::ButtonInput;
use bevy::light::{CascadeShadowConfigBuilder, GlobalAmbientLight};
use bevy::mesh::{Indices, PrimitiveTopology};
use bevy::prelude::*;
use bevy_egui::{EguiContexts, EguiPlugin, EguiPrimaryContextPass, egui};
use geo::{Coord, LineString, MultiPolygon, Polygon};
use serde::{Deserialize, Serialize};
use lyon::math::point as lyon_point;
use lyon::path::iterator::PathIterator;
use lyon::path::{Path as LyonPath, PathEvent};
use lyon::tessellation::{
    BuffersBuilder, FillOptions, FillRule, FillTessellator, FillVertex, VertexBuffers,
};
use ttf_parser::{Face, GlyphId, OutlineBuilder};

// ─── Dimensions / layout ─────────────────────────────────────────────

/// The single outer cream layer. Sized large so it extends past the
/// camera's view at any reasonable distance — there is no further-out
/// layer behind it; the ClearColor matches it for the bleed.
const PAPER_W: f32 = 30.0;
const PAPER_H: f32 = 30.0;
const PAPER_R: f32 = 1.20;

const PANEL_W: f32 = 10.6;
const PANEL_H: f32 = 13.6;
const PANEL_R: f32 = 0.65;

const DISPLAY_W: f32 = 9.4;
const DISPLAY_H: f32 = 1.85;
const DISPLAY_R: f32 = 0.40;
const DISPLAY_Y: f32 = 5.20;
const DISPLAY_DIGIT_EM: f32 = 1.10;

const BTN_W: f32 = 1.85;
const BTN_H: f32 = 1.55;
const BTN_R: f32 = 0.50;
const BTN_GAP_X: f32 = 0.25;
const BTN_GAP_Y: f32 = 0.25;
const BTN_TOP_ROW_Y: f32 = 2.45;
const BTN_GLYPH_EM: f32 = 0.85;

// Animation tuning constants (the depth/scale defaults live in
// DebugKnobs so they can be tweaked live).
const BTN_TWEEN_TAU: f32 = 0.06;
const BTN_SNAP_EPS: f32 = 0.001;

/// The inspiration shows clear serif numerals/symbols. Prefer Georgia
/// (warm transitional serif) → Times → Palatino, before falling back
/// to Arial sans.
const FONT_FALLBACKS: &[&str] = &[
    "/System/Library/Fonts/Supplemental/Georgia.ttf",
    "/Library/Fonts/Georgia.ttf",
    "/System/Library/Fonts/Times.ttc",
    "/Library/Fonts/Times New Roman.ttf",
    "/System/Library/Fonts/Palatino.ttc",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/Library/Fonts/Arial.ttf",
];

// ─── Polygon helpers (geo for boolean-free polygon-with-holes) ───────

fn rect_pts_ccw(cx: f32, cy: f32, w: f32, h: f32) -> Vec<(f64, f64)> {
    let hw = (w * 0.5) as f64;
    let hh = (h * 0.5) as f64;
    let cx = cx as f64;
    let cy = cy as f64;
    vec![
        (cx - hw, cy - hh),
        (cx + hw, cy - hh),
        (cx + hw, cy + hh),
        (cx - hw, cy + hh),
        (cx - hw, cy - hh),
    ]
}

fn rounded_rect_pts_ccw(cx: f32, cy: f32, w: f32, h: f32, r: f32) -> Vec<(f64, f64)> {
    let r_c = r.min(w * 0.5).min(h * 0.5).max(0.0);
    if r_c < 1e-4 {
        return rect_pts_ccw(cx, cy, w, h);
    }
    let segs = 24u32;
    let hw = (w * 0.5 - r_c) as f64;
    let hh = (h * 0.5 - r_c) as f64;
    let cx = cx as f64;
    let cy = cy as f64;
    let r = r_c as f64;
    let mut pts: Vec<(f64, f64)> = Vec::new();
    let mut corner = |center: (f64, f64), start: f64| {
        for i in 0..=segs {
            let a = start + (i as f64 / segs as f64) * std::f64::consts::FRAC_PI_2;
            pts.push((center.0 + r * a.cos(), center.1 + r * a.sin()));
        }
    };
    corner((cx + hw, cy + hh), 0.0);
    corner((cx - hw, cy + hh), std::f64::consts::FRAC_PI_2);
    corner((cx - hw, cy - hh), std::f64::consts::PI);
    corner((cx + hw, cy - hh), 1.5 * std::f64::consts::PI);
    pts.push(pts[0]);
    pts
}

fn polygon_filled(outer_ccw: Vec<(f64, f64)>) -> Polygon<f64> {
    Polygon::new(LineString::from(outer_ccw), vec![])
}

fn rounded_rect(cx: f32, cy: f32, w: f32, h: f32, r: f32) -> Polygon<f64> {
    polygon_filled(rounded_rect_pts_ccw(cx, cy, w, h, r))
}

fn point_in_polygon_xy(p: Vec2, poly: &Polygon<f64>) -> bool {
    use geo::Contains;
    poly.contains(&geo::Point::new(p.x as f64, p.y as f64))
}

// ─── Glyph outline → polygons ─────────────────────────────────────────

#[derive(Clone, Copy)]
enum Op {
    MoveTo(Vec2),
    LineTo(Vec2),
    QuadTo(Vec2, Vec2),
    CurveTo(Vec2, Vec2, Vec2),
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
    fn move_to(&mut self, x: f32, y: f32) { self.ops.push(Op::MoveTo(self.t(x, y))); }
    fn line_to(&mut self, x: f32, y: f32) { self.ops.push(Op::LineTo(self.t(x, y))); }
    fn quad_to(&mut self, x1: f32, y1: f32, x: f32, y: f32) {
        self.ops.push(Op::QuadTo(self.t(x1, y1), self.t(x, y)));
    }
    fn curve_to(&mut self, x1: f32, y1: f32, x2: f32, y2: f32, x: f32, y: f32) {
        self.ops.push(Op::CurveTo(self.t(x1, y1), self.t(x2, y2), self.t(x, y)));
    }
    fn close(&mut self) { self.ops.push(Op::Close); }
}

fn ops_to_path(ops: &[Op]) -> LyonPath {
    let mut pb = LyonPath::builder();
    let mut started = false;
    for op in ops {
        match *op {
            Op::MoveTo(p) => {
                if started { pb.end(false); }
                pb.begin(lyon_point(p.x, p.y));
                started = true;
            }
            Op::LineTo(p) => { pb.line_to(lyon_point(p.x, p.y)); }
            Op::QuadTo(c, p) => { pb.quadratic_bezier_to(lyon_point(c.x, c.y), lyon_point(p.x, p.y)); }
            Op::CurveTo(c1, c2, p) => {
                pb.cubic_bezier_to(
                    lyon_point(c1.x, c1.y),
                    lyon_point(c2.x, c2.y),
                    lyon_point(p.x, p.y),
                );
            }
            Op::Close => { if started { pb.end(true); started = false; } }
        }
    }
    if started { pb.end(false); }
    pb.build()
}

fn outline_glyph_to_path(face: &Face, glyph_id: GlyphId, scale: f32, offset: Vec2) -> LyonPath {
    let mut col = OpCollector { ops: Vec::new(), scale, offset };
    face.outline_glyph(glyph_id, &mut col);
    ops_to_path(&col.ops)
}

fn flatten_path_to_contours(path: &LyonPath, tol: f32) -> Vec<Vec<Vec2>> {
    let mut out = Vec::new();
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

fn signed_area(pts: &[Vec2]) -> f32 {
    let mut a = 0.0;
    let n = pts.len();
    for i in 0..n {
        let j = (i + 1) % n;
        a += pts[i].x * pts[j].y - pts[j].x * pts[i].y;
    }
    a * 0.5
}

fn point_in_ring(p: Vec2, ring: &[Vec2]) -> bool {
    let n = ring.len();
    if n == 0 { return false; }
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

fn linestring_from_vecs(pts: &[Vec2]) -> LineString<f64> {
    let mut coords: Vec<(f64, f64)> = pts.iter().map(|p| (p.x as f64, p.y as f64)).collect();
    if let Some(first) = coords.first().copied() {
        coords.push(first);
    }
    LineString::from(coords)
}

fn ensure_ccw(mut pts: Vec<Vec2>) -> Vec<Vec2> {
    if signed_area(&pts) < 0.0 { pts.reverse(); }
    pts
}

fn ensure_cw(mut pts: Vec<Vec2>) -> Vec<Vec2> {
    if signed_area(&pts) > 0.0 { pts.reverse(); }
    pts
}

fn build_glyph_polygons(contours: &[Vec<Vec2>]) -> Vec<Polygon<f64>> {
    let valid: Vec<&Vec<Vec2>> = contours.iter().filter(|c| c.len() >= 3).collect();
    let n = valid.len();
    if n == 0 { return Vec::new(); }
    let mut depths = vec![0usize; n];
    for i in 0..n {
        for j in 0..n {
            if i == j { continue; }
            if point_in_ring(valid[i][0], valid[j]) { depths[i] += 1; }
        }
    }
    let mut polygons = Vec::new();
    for i in 0..n {
        if depths[i] % 2 != 0 { continue; }
        let outer_pts = ensure_ccw(valid[i].clone());
        let mut holes: Vec<LineString<f64>> = Vec::new();
        for j in 0..n {
            if i == j || depths[j] != depths[i] + 1 { continue; }
            if point_in_ring(valid[j][0], &outer_pts) {
                let hole_pts = ensure_cw(valid[j].clone());
                holes.push(linestring_from_vecs(&hole_pts));
            }
        }
        polygons.push(Polygon::new(linestring_from_vecs(&outer_pts), holes));
    }
    polygons
}

fn layout_text_centered(font: &[u8], text: &str, origin: Vec2, em: f32) -> Vec<Polygon<f64>> {
    let face = match Face::parse(font, 0) {
        Ok(f) => f,
        Err(_) => return Vec::new(),
    };
    let upem = face.units_per_em() as f32;
    let scale = em / upem;
    let mut polys: Vec<Polygon<f64>> = Vec::new();
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
    polys.into_iter().map(|p| translate_polygon(&p, dx as f64, dy as f64)).collect()
}

fn translate_polygon(poly: &Polygon<f64>, dx: f64, dy: f64) -> Polygon<f64> {
    let shift = |c: &Coord<f64>| Coord { x: c.x + dx, y: c.y + dy };
    let ext: Vec<Coord<f64>> = poly.exterior().0.iter().map(shift).collect();
    let holes: Vec<LineString<f64>> = poly
        .interiors()
        .iter()
        .map(|h| LineString::from(h.0.iter().map(shift).collect::<Vec<_>>()))
        .collect();
    Polygon::new(LineString::from(ext), holes)
}

// ─── Mesh-builder helpers ─────────────────────────────────────────────

fn ring_to_coords(ring: &LineString<f64>) -> Vec<Coord<f64>> {
    let coords = &ring.0;
    let n = coords.len();
    if n == 0 { return Vec::new(); }
    let count = if n >= 2
        && (coords[0].x - coords[n - 1].x).abs() < 1e-9
        && (coords[0].y - coords[n - 1].y).abs() < 1e-9
    {
        n - 1
    } else {
        n
    };
    coords[..count].to_vec()
}

fn push_ring_to_path(ring: &LineString<f64>, pb: &mut lyon::path::path::Builder) -> bool {
    let coords = ring_to_coords(ring);
    if coords.len() < 3 { return false; }
    pb.begin(lyon_point(coords[0].x as f32, coords[0].y as f32));
    for c in &coords[1..] {
        pb.line_to(lyon_point(c.x as f32, c.y as f32));
    }
    pb.end(true);
    true
}

#[derive(Default)]
struct MeshBuilder {
    positions: Vec<[f32; 3]>,
    normals: Vec<[f32; 3]>,
    uvs: Vec<[f32; 2]>,
    indices: Vec<u32>,
}

impl MeshBuilder {
    fn add_cap(&mut self, region: &MultiPolygon<f64>, z: f32, normal_up: bool) {
        let mut tess = FillTessellator::new();
        let opts = FillOptions::default()
            .with_fill_rule(FillRule::NonZero)
            .with_tolerance(0.01);
        for poly in &region.0 {
            let mut pb = LyonPath::builder();
            if !push_ring_to_path(poly.exterior(), &mut pb) { continue; }
            for hole in poly.interiors() {
                push_ring_to_path(hole, &mut pb);
            }
            let path = pb.build();
            let mut buf: VertexBuffers<[f32; 2], u32> = VertexBuffers::new();
            if tess
                .tessellate_path(
                    &path,
                    &opts,
                    &mut BuffersBuilder::new(&mut buf, |v: FillVertex| {
                        let p = v.position();
                        [p.x, p.y]
                    }),
                )
                .is_err()
            {
                continue;
            }
            let base = self.positions.len() as u32;
            let normal = if normal_up { [0.0, 0.0, 1.0] } else { [0.0, 0.0, -1.0] };
            // Planar UV: paper texture tiles 1.5× across the paper width.
            let u_scale = 1.5 / PAPER_W;
            let v_scale = 1.5 / PAPER_H;
            for v in &buf.vertices {
                self.positions.push([v[0], v[1], z]);
                self.normals.push(normal);
                self.uvs.push([v[0] * u_scale + 0.5, v[1] * v_scale + 0.5]);
            }
            for chunk in buf.indices.chunks_exact(3) {
                if normal_up {
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
    }

    /// Extrude a closed ring as a vertical wall strip.
    /// `slope_inset` insets the bottom edge inward by this much (the
    /// "right" side of the walked edge), giving the wall a slight
    /// chamfer so adjacent walls don't z-fight.
    fn add_wall(
        &mut self,
        ring: &LineString<f64>,
        z_top: f32,
        z_bot: f32,
        slope_inset: f32,
    ) {
        let coords = ring_to_coords(ring);
        let n = coords.len();
        if n < 3 { return; }
        let bot_coords: Vec<Coord<f64>> = if slope_inset > 1e-9 {
            miter_offset_right(&coords, slope_inset as f64)
        } else {
            coords.clone()
        };
        let u_scale = 1.5 / PAPER_W;
        const WALL_V_SPAN: f32 = 0.15;
        let mut cum_len = vec![0.0_f32; n + 1];
        for i in 0..n {
            let j = (i + 1) % n;
            let dx = (coords[j].x - coords[i].x) as f32;
            let dy = (coords[j].y - coords[i].y) as f32;
            cum_len[i + 1] = cum_len[i] + (dx * dx + dy * dy).sqrt();
        }
        for i in 0..n {
            let j = (i + 1) % n;
            let p_top_i = Vec3::new(coords[i].x as f32, coords[i].y as f32, z_top);
            let p_top_j = Vec3::new(coords[j].x as f32, coords[j].y as f32, z_top);
            let p_bot_i = Vec3::new(bot_coords[i].x as f32, bot_coords[i].y as f32, z_bot);
            let p_bot_j = Vec3::new(bot_coords[j].x as f32, bot_coords[j].y as f32, z_bot);
            let edge = p_top_j - p_top_i;
            let down = p_bot_i - p_top_i;
            let n_face = down.cross(edge).try_normalize().unwrap_or(Vec3::Z);
            let normal = [n_face.x, n_face.y, n_face.z];
            let u_i = cum_len[i] * u_scale;
            let u_j = cum_len[i + 1] * u_scale;
            let wall_uvs = [[u_i, 0.0], [u_i, WALL_V_SPAN], [u_j, WALL_V_SPAN], [u_j, 0.0]];
            let base = self.positions.len() as u32;
            for (k, v) in [p_top_i, p_bot_i, p_bot_j, p_top_j].iter().enumerate() {
                self.positions.push([v.x, v.y, v.z]);
                self.normals.push(normal);
                self.uvs.push(wall_uvs[k]);
            }
            self.indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
        }
    }

    fn build(self) -> Mesh {
        let mut m = Mesh::new(
            PrimitiveTopology::TriangleList,
            RenderAssetUsages::RENDER_WORLD,
        );
        m.insert_attribute(Mesh::ATTRIBUTE_POSITION, self.positions);
        m.insert_attribute(Mesh::ATTRIBUTE_NORMAL, self.normals);
        m.insert_attribute(Mesh::ATTRIBUTE_UV_0, self.uvs);
        m.insert_indices(Indices::U32(self.indices));
        m
    }
}

const MITER_LIMIT: f64 = 4.0;

fn norm2_f64(x: f64, y: f64) -> (f64, f64) {
    let l = (x * x + y * y).sqrt();
    if l < 1e-12 { (0.0, 0.0) } else { (x / l, y / l) }
}

fn miter_offset_right(pts: &[Coord<f64>], inset: f64) -> Vec<Coord<f64>> {
    let n = pts.len();
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let p_prev = pts[(i + n - 1) % n];
        let p = pts[i];
        let p_next = pts[(i + 1) % n];
        let (e0x, e0y) = norm2_f64(p.x - p_prev.x, p.y - p_prev.y);
        let (e1x, e1y) = norm2_f64(p_next.x - p.x, p_next.y - p.y);
        let r0x = e0y;
        let r0y = -e0x;
        let r1x = e1y;
        let r1y = -e1x;
        let bx = r0x + r1x;
        let by = r0y + r1y;
        let blen = (bx * bx + by * by).sqrt();
        let (nx, ny) = if blen < 1e-9 { (r0x, r0y) } else { (bx / blen, by / blen) };
        let dot = nx * r0x + ny * r0y;
        let scale = if dot.abs() < 1e-9 { 1.0 } else { 1.0 / dot };
        let scale = scale.min(MITER_LIMIT).max(-MITER_LIMIT);
        out.push(Coord {
            x: p.x + nx * inset * scale,
            y: p.y + ny * inset * scale,
        });
    }
    out
}

// ─── Texture loading ──────────────────────────────────────────────────

fn load_mipped_image(path: &std::path::Path, is_srgb: bool) -> Image {
    let img = image::open(path)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", path.display()))
        .to_rgba8();
    let raw = img.width().min(img.height());
    let dim = if raw.is_power_of_two() { raw } else { raw.next_power_of_two() / 2 };
    let mut current = if img.width() == dim && img.height() == dim {
        img
    } else {
        image::imageops::resize(&img, dim, dim, image::imageops::FilterType::Triangle)
    };
    let mut data: Vec<u8> = current.as_raw().clone();
    let mut size = dim;
    let mut mip_count: u32 = 1;
    while size > 1 {
        let next = size / 2;
        current = image::imageops::resize(&current, next, next, image::imageops::FilterType::Triangle);
        data.extend_from_slice(current.as_raw());
        size = next;
        mip_count += 1;
    }
    let format = if is_srgb {
        bevy::render::render_resource::TextureFormat::Rgba8UnormSrgb
    } else {
        bevy::render::render_resource::TextureFormat::Rgba8Unorm
    };
    let mut image = Image::new_uninit(
        bevy::render::render_resource::Extent3d { width: dim, height: dim, depth_or_array_layers: 1 },
        bevy::render::render_resource::TextureDimension::D2,
        format,
        RenderAssetUsages::RENDER_WORLD,
    );
    image.data = Some(data);
    image.texture_descriptor.mip_level_count = mip_count;
    image.sampler = ImageSampler::Descriptor(ImageSamplerDescriptor {
        address_mode_u: ImageAddressMode::Repeat,
        address_mode_v: ImageAddressMode::Repeat,
        address_mode_w: ImageAddressMode::Repeat,
        mag_filter: ImageFilterMode::Linear,
        min_filter: ImageFilterMode::Linear,
        mipmap_filter: ImageFilterMode::Linear,
        anisotropy_clamp: 16,
        ..ImageSamplerDescriptor::linear()
    });
    image
}

// ─── Buttons / calc state ─────────────────────────────────────────────

#[derive(Hash, PartialEq, Eq, Clone, Copy, Debug)]
enum WidgetId {
    Digit(u8),
    Decimal,
    Add,
    Sub,
    Mul,
    Div,
    Equals,
}

#[derive(Clone, Copy, PartialEq, Eq)]
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
    BtnSpec { id: WidgetId::Div,      glyph: "\u{00F7}", color: BtnColor::Coral, col: 3, row: 0 },
    BtnSpec { id: WidgetId::Digit(4), glyph: "4", color: BtnColor::Cream, col: 0, row: 1 },
    BtnSpec { id: WidgetId::Digit(5), glyph: "5", color: BtnColor::Cream, col: 1, row: 1 },
    BtnSpec { id: WidgetId::Digit(6), glyph: "6", color: BtnColor::Cream, col: 2, row: 1 },
    BtnSpec { id: WidgetId::Mul,      glyph: "\u{00D7}", color: BtnColor::Coral, col: 3, row: 1 },
    BtnSpec { id: WidgetId::Digit(1), glyph: "1", color: BtnColor::Cream, col: 0, row: 2 },
    BtnSpec { id: WidgetId::Digit(2), glyph: "2", color: BtnColor::Cream, col: 1, row: 2 },
    BtnSpec { id: WidgetId::Digit(3), glyph: "3", color: BtnColor::Cream, col: 2, row: 2 },
    BtnSpec { id: WidgetId::Sub,      glyph: "\u{2212}", color: BtnColor::Coral, col: 3, row: 2 },
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

fn btn_polygon_world(spec: &BtnSpec) -> Polygon<f64> {
    let c = btn_center(spec);
    rounded_rect(c.x, c.y, BTN_W, BTN_H, BTN_R)
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum CalcOp { Add, Sub, Mul, Div }

#[derive(Resource)]
struct CalcState {
    display: String,
    accumulator: Option<f64>,
    pending_op: Option<CalcOp>,
    awaiting_operand: bool,
}

impl CalcState {
    fn new() -> Self {
        Self {
            display: "0".into(),
            accumulator: None,
            pending_op: None,
            awaiting_operand: false,
        }
    }

    fn press(&mut self, id: WidgetId) {
        match id {
            WidgetId::Digit(d) => {
                if self.awaiting_operand || self.display == "0" {
                    self.display = d.to_string();
                    self.awaiting_operand = false;
                } else if self.display.len() < 12 {
                    self.display.push_str(&d.to_string());
                }
            }
            WidgetId::Decimal => {
                if self.awaiting_operand {
                    self.display = "0.".into();
                    self.awaiting_operand = false;
                } else if !self.display.contains('.') {
                    self.display.push('.');
                }
            }
            WidgetId::Add => self.set_op(CalcOp::Add),
            WidgetId::Sub => self.set_op(CalcOp::Sub),
            WidgetId::Mul => self.set_op(CalcOp::Mul),
            WidgetId::Div => self.set_op(CalcOp::Div),
            WidgetId::Equals => {
                if let (Some(acc), Some(op)) = (self.accumulator, self.pending_op) {
                    let v = self.display.parse::<f64>().unwrap_or(0.0);
                    let r = apply_op(op, acc, v);
                    self.display = fmt_value(r);
                    self.accumulator = None;
                    self.pending_op = None;
                    self.awaiting_operand = true;
                }
            }
        }
    }

    fn set_op(&mut self, op: CalcOp) {
        let current = self.display.parse::<f64>().unwrap_or(0.0);
        self.accumulator = Some(match (self.accumulator, self.pending_op) {
            (Some(a), Some(po)) => apply_op(po, a, current),
            _ => current,
        });
        self.pending_op = Some(op);
        self.awaiting_operand = true;
        if let Some(a) = self.accumulator {
            self.display = fmt_value(a);
        }
    }
}

fn apply_op(op: CalcOp, a: f64, b: f64) -> f64 {
    match op {
        CalcOp::Add => a + b,
        CalcOp::Sub => a - b,
        CalcOp::Mul => a * b,
        CalcOp::Div => if b == 0.0 { 0.0 } else { a / b },
    }
}

fn fmt_value(v: f64) -> String {
    if !v.is_finite() { return "Err".into(); }
    if v == v.trunc() && v.abs() < 1e12 { format!("{}", v as i64) } else { format!("{:.4}", v) }
}

#[derive(Resource, Default)]
struct UiState {
    hovered: Option<WidgetId>,
    pressed: Option<WidgetId>,
}

#[derive(Resource)]
struct LoadedFont(Vec<u8>);

// ─── Debug knobs ──────────────────────────────────────────────────────

#[derive(Resource, Serialize, Deserialize)]
struct DebugKnobs {
    #[serde(default)]
    panel_open: bool,

    // Layer Z (in units of paper_thickness — multiplied through to world z).
    paper_thickness: f32,
    slope_inset: f32,
    paper_step: f32,
    panel_taupe_step: f32,
    display_olive_step: f32,
    display_ink_step: f32,

    // Per-layer slab thickness in WORLD units. Each layer is a slab:
    // top at its `step * paper_thickness`, bottom at `top - layer_thick`.
    // Walls extend from top to bottom around both the outer perimeter
    // and any holes (so layers can be freely raised / lowered without
    // exposing gaps).
    paper_layer_thick: f32,
    panel_layer_thick: f32,
    display_layer_thick: f32,

    // Button animation targets.
    btn_idle_step: f32,
    btn_hover_step: f32,
    btn_press_step: f32,
    btn_engrave_delta: f32,

    // Key directional light (from upper-* of screen). az = horizontal
    // sweep around +Z (degrees, 0 = east, 90 = north). elev = degrees
    // above the panel surface.
    key_illuminance: f32,
    key_az_deg: f32,
    key_elev_deg: f32,

    // Cool fill (no shadows).
    fill_illuminance: f32,
    fill_az_deg: f32,
    fill_elev_deg: f32,

    ambient_brightness: f32,
    shadow_normal_bias: f32,
    shadow_depth_bias: f32,
    /// Side length of the cascaded shadow texture (one of {1024, 2048, 4096, 8192}).
    #[serde(default = "default_shadow_map_size")]
    shadow_map_size: u32,
    /// World-units of cascade extent along the light direction. Smaller =
    /// more shadow-map texels per world unit (sharper edges) at the cost
    /// of features outside the cascade not casting shadows.
    #[serde(default = "default_cascade_max")]
    cascade_max: f32,
}

fn default_shadow_map_size() -> u32 { 4096 }
fn default_cascade_max() -> f32 { 30.0 }

impl Default for DebugKnobs {
    fn default() -> Self {
        // Each value sits roughly in the middle of its slider range so
        // there's headroom to push up *or* down from the start. Layer
        // ordering preserved (paper > panel > display > ink, buttons
        // between panel and paper).
        Self {
            panel_open: false,
            paper_thickness: 0.15,
            slope_inset: 0.05,
            paper_step: 6.0,
            panel_taupe_step: 4.0,
            display_olive_step: 3.0,
            display_ink_step: 2.0,
            btn_idle_step: 5.5,
            btn_hover_step: 5.7,
            btn_press_step: 5.0,
            btn_engrave_delta: 1.5,
            paper_layer_thick: 0.5,
            panel_layer_thick: 0.5,
            display_layer_thick: 0.3,
            key_illuminance: 8_000.0,
            key_az_deg: 135.0,
            key_elev_deg: 30.0,
            fill_illuminance: 2_000.0,
            fill_az_deg: -45.0,
            fill_elev_deg: 30.0,
            ambient_brightness: 250.0,
            shadow_normal_bias: 0.50,
            shadow_depth_bias: 0.05,
            shadow_map_size: 4096,
            cascade_max: 30.0,
        }
    }
}

// ─── Persistence ─────────────────────────────────────────────────────

fn knob_save_path() -> std::path::PathBuf {
    let home = std::env::var_os("HOME")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| std::path::PathBuf::from("."));
    home.join(".button_counter.json")
}

fn load_knobs() -> Option<DebugKnobs> {
    let path = knob_save_path();
    let bytes = std::fs::read(&path).ok()?;
    serde_json::from_slice(&bytes).ok()
}

fn save_knobs(knobs: &DebugKnobs) {
    let path = knob_save_path();
    if let Ok(json) = serde_json::to_vec_pretty(knobs) {
        let _ = std::fs::write(&path, json);
    }
}

#[derive(Resource, Default)]
struct PersistTimer {
    secs: f32,
    last_sig: u64,
}

fn knobs_full_signature(k: &DebugKnobs) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut h = DefaultHasher::new();
    // Hash everything that affects the look so we know when to save.
    for v in [
        k.paper_thickness, k.slope_inset, k.paper_step, k.panel_taupe_step,
        k.display_olive_step, k.display_ink_step, k.btn_idle_step,
        k.btn_hover_step, k.btn_press_step, k.btn_engrave_delta,
        k.paper_layer_thick, k.panel_layer_thick, k.display_layer_thick,
        k.key_illuminance, k.key_az_deg, k.key_elev_deg,
        k.fill_illuminance, k.fill_az_deg, k.fill_elev_deg,
        k.ambient_brightness, k.shadow_normal_bias, k.shadow_depth_bias,
        k.cascade_max,
    ] {
        v.to_bits().hash(&mut h);
    }
    k.shadow_map_size.hash(&mut h);
    h.finish()
}

fn persist_knobs_periodic(
    time: Res<Time>,
    knobs: Res<DebugKnobs>,
    mut timer: ResMut<PersistTimer>,
) {
    timer.secs += time.delta_secs();
    if timer.secs < 1.5 { return; }
    timer.secs = 0.0;
    let sig = knobs_full_signature(&knobs);
    if sig != timer.last_sig {
        timer.last_sig = sig;
        save_knobs(&knobs);
    }
}

/// Convert (azimuth_deg, elevation_deg) into the `light_dir` (where
/// the light *travels*, i.e. -direction-to-source).
fn polar_light_dir(az_deg: f32, elev_deg: f32) -> Vec3 {
    let az = az_deg.to_radians();
    let elev = elev_deg.to_radians();
    let dir_to_source = Vec3::new(elev.cos() * az.cos(), elev.cos() * az.sin(), elev.sin());
    -dir_to_source.normalize()
}

#[derive(Component)]
struct KeyLight;

#[derive(Component)]
struct FillLight;

fn sync_lights_from_knobs(
    knobs: Res<DebugKnobs>,
    mut ambient: ResMut<GlobalAmbientLight>,
    mut shadow_map: ResMut<bevy::light::DirectionalLightShadowMap>,
    mut key_q: Query<
        (&mut DirectionalLight, &mut Transform, &mut bevy::light::CascadeShadowConfig),
        (With<KeyLight>, Without<FillLight>),
    >,
    mut fill_q: Query<(&mut DirectionalLight, &mut Transform), (With<FillLight>, Without<KeyLight>)>,
) {
    if let Ok((mut light, mut xf, mut cascade)) = key_q.single_mut() {
        light.illuminance = knobs.key_illuminance;
        light.shadow_depth_bias = knobs.shadow_depth_bias;
        light.shadow_normal_bias = knobs.shadow_normal_bias;
        *xf = Transform::from_xyz(0.0, 0.0, 0.0)
            .looking_to(polar_light_dir(knobs.key_az_deg, knobs.key_elev_deg), Vec3::Z);
        *cascade = CascadeShadowConfigBuilder {
            num_cascades: 1,
            minimum_distance: 0.05,
            first_cascade_far_bound: knobs.cascade_max,
            maximum_distance: knobs.cascade_max,
            ..default()
        }
        .build();
    }
    if let Ok((mut light, mut xf)) = fill_q.single_mut() {
        light.illuminance = knobs.fill_illuminance;
        *xf = Transform::from_xyz(0.0, 0.0, 0.0)
            .looking_to(polar_light_dir(knobs.fill_az_deg, knobs.fill_elev_deg), Vec3::Z);
    }
    ambient.brightness = knobs.ambient_brightness;
    if shadow_map.size != knobs.shadow_map_size as usize {
        shadow_map.size = knobs.shadow_map_size as usize;
    }
}


// ─── Camera ───────────────────────────────────────────────────────────

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
        // Dead overhead. The 3D read comes from layer thickness +
        // beveled walls (slope_inset) + a low-angle directional light
        // that casts visible shadows of buttons onto the panel.
        Self { yaw: 0.0, tilt: 0.0, dist: 28.0, target: Vec3::ZERO }
    }
}

// ─── Per-entity components ────────────────────────────────────────────

#[derive(Component)]
struct Btn {
    id: WidgetId,
    target_face_z: f32,
    /// World-space hit rectangle (axis-aligned, in XY).
    hit_min: Vec2,
    hit_max: Vec2,
}

/// Marker to despawn-and-rebuild background entities on knob change.
#[derive(Component)]
struct BgEntity;

#[derive(Resource, Default)]
struct SceneRebuildMarker {
    last_knob_signature: u64,
    last_display: String,
}

// ─── Mesh constructors ────────────────────────────────────────────────

/// Emit a slab mesh for a polygon-with-holes region: top cap, bottom
/// cap, outer wall around the exterior ring, and an inner wall around
/// each hole. Lets the caller move the slab freely without exposing
/// gaps, because the slab carries its own walls everywhere it has
/// edges.
fn add_slab(
    mb: &mut MeshBuilder,
    region: &MultiPolygon<f64>,
    top_z: f32,
    thickness: f32,
    slope_inset: f32,
) {
    let bot_z = top_z - thickness.max(0.0);
    mb.add_cap(region, top_z, true);
    mb.add_cap(region, bot_z, false);
    for poly in &region.0 {
        mb.add_wall(poly.exterior(), top_z, bot_z, slope_inset);
        for hole in poly.interiors() {
            mb.add_wall(hole, top_z, bot_z, slope_inset);
        }
    }
}

/// The single outer cream layer — large rounded rect covering more
/// than the visible camera view, with a panel-shaped hole cut out of
/// it. Built as a slab so it carries its own thickness around the
/// outer perimeter and around the panel cavity.
fn build_paper_mesh(knobs: &DebugKnobs) -> Mesh {
    let mut mb = MeshBuilder::default();
    let paper = rounded_rect(0.0, 0.0, PAPER_W, PAPER_H, PAPER_R);
    let panel = rounded_rect(0.0, 0.0, PANEL_W, PANEL_H, PANEL_R);
    let mut panel_hole_coords: Vec<Coord<f64>> = panel.exterior().0.clone();
    panel_hole_coords.reverse();
    let region = MultiPolygon(vec![Polygon::new(
        paper.exterior().clone(),
        vec![LineString::from(panel_hole_coords)],
    )]);
    let z_paper = knobs.paper_step * knobs.paper_thickness;
    add_slab(&mut mb, &region, z_paper, knobs.paper_layer_thick, knobs.slope_inset);
    mb.build()
}

fn build_panel_mesh(knobs: &DebugKnobs) -> Mesh {
    let mut mb = MeshBuilder::default();
    let panel = rounded_rect(0.0, 0.0, PANEL_W, PANEL_H, PANEL_R);
    let display = rounded_rect(0.0, DISPLAY_Y, DISPLAY_W, DISPLAY_H, DISPLAY_R);
    let mut display_hole_coords: Vec<Coord<f64>> = display.exterior().0.clone();
    display_hole_coords.reverse();
    let region = MultiPolygon(vec![Polygon::new(
        panel.exterior().clone(),
        vec![LineString::from(display_hole_coords)],
    )]);
    let z_panel = knobs.panel_taupe_step * knobs.paper_thickness;
    add_slab(&mut mb, &region, z_panel, knobs.panel_layer_thick, knobs.slope_inset);
    mb.build()
}

fn display_digits(knobs: &DebugKnobs, font: &[u8], display_text: &str) -> Vec<Polygon<f64>> {
    let raw = layout_text_centered(font, display_text, Vec2::ZERO, DISPLAY_DIGIT_EM);
    if raw.is_empty() { return Vec::new(); }
    let _ = knobs;
    let (mut xmin, mut xmax) = (f64::INFINITY, f64::NEG_INFINITY);
    for poly in &raw {
        for c in poly.exterior().0.iter() {
            xmin = xmin.min(c.x);
            xmax = xmax.max(c.x);
        }
    }
    let half_w = (xmax - xmin) * 0.5;
    let target_right = (DISPLAY_W * 0.5 - 0.55) as f64;
    let dx = target_right - half_w;
    let dy = (DISPLAY_Y - 0.05) as f64;
    raw.iter().map(|p| translate_polygon(p, dx, dy)).collect()
}

/// Display surface = olive-colored mesh (display rect with digit
/// holes, engrave walls, glyph-hole island caps). Built as a slab —
/// it has its own outer wall extending down by `display_layer_thick`,
/// so when the display moves up/down independently of the panel the
/// gap is hidden by the slab's side.
fn build_display_face_mesh(knobs: &DebugKnobs, digits: &[Polygon<f64>]) -> Mesh {
    let mut mb = MeshBuilder::default();
    let display = rounded_rect(0.0, DISPLAY_Y, DISPLAY_W, DISPLAY_H, DISPLAY_R);
    let z_display = knobs.display_olive_step * knobs.paper_thickness;
    let z_ink = knobs.display_ink_step * knobs.paper_thickness;

    // Slab: display rect with digit-shaped holes, top + bottom + outer
    // wall + per-digit cavity walls (these double as the engrave walls
    // since the digit hole goes through the slab to the ink layer).
    let mut top_holes: Vec<LineString<f64>> = Vec::new();
    for poly in digits {
        let mut coords: Vec<Coord<f64>> = poly.exterior().0.clone();
        coords.reverse();
        top_holes.push(LineString::from(coords));
    }
    let region = MultiPolygon(vec![Polygon::new(display.exterior().clone(), top_holes)]);
    add_slab(&mut mb, &region, z_display, knobs.display_layer_thick, knobs.slope_inset);

    // Walls of digit cavities going down to the ink floor + island
    // caps for letter holes (e.g. inside "0"). These overlap the slab
    // inner walls but extend further down into the ink recess; the
    // inner-island caps land right at the display surface so the
    // visible center of "0" reads as olive.
    for poly in digits {
        mb.add_wall(poly.exterior(), z_display, z_ink, knobs.slope_inset);
        for hole in poly.interiors() {
            let mut coords: Vec<Coord<f64>> = hole.0.clone();
            coords.reverse();
            let island_ring = LineString::from(coords);
            mb.add_wall(&island_ring, z_display, z_ink, knobs.slope_inset);
            mb.add_cap(
                &MultiPolygon(vec![Polygon::new(island_ring, vec![])]),
                z_display,
                true,
            );
        }
    }
    mb.build()
}

/// Display engrave floor = dark-olive filled-digit caps at ink height.
fn build_display_engrave_mesh(knobs: &DebugKnobs, digits: &[Polygon<f64>]) -> Mesh {
    let mut mb = MeshBuilder::default();
    let z_ink = knobs.display_ink_step * knobs.paper_thickness;
    for poly in digits {
        mb.add_cap(&MultiPolygon(vec![poly.clone()]), z_ink, true);
    }
    mb.build()
}

/// Button face mesh — face top with glyph holes, outer wall, engrave
/// walls, and island caps for letter holes (e.g. inside "0"). Built
/// at z=0 in entity-local coords; transform handles world placement
/// and press animation. Material color = the button's face color
/// (cream / coral).
fn build_button_face_mesh(knobs: &DebugKnobs, font: &[u8], spec: &BtnSpec) -> Mesh {
    let mut mb = MeshBuilder::default();
    let face_rect = rounded_rect(0.0, 0.0, BTN_W, BTN_H, BTN_R);
    let glyph_polys = layout_text_centered(font, spec.glyph, Vec2::ZERO, BTN_GLYPH_EM);
    let z_face: f32 = 0.0;
    let z_engrave = z_face - knobs.btn_engrave_delta * knobs.paper_thickness;
    let z_bottom = z_face - (knobs.btn_idle_step - knobs.panel_taupe_step) * knobs.paper_thickness
        - 0.5;

    let mut face_holes: Vec<LineString<f64>> = Vec::new();
    for poly in &glyph_polys {
        let mut coords: Vec<Coord<f64>> = poly.exterior().0.clone();
        coords.reverse();
        face_holes.push(LineString::from(coords));
    }
    mb.add_cap(
        &MultiPolygon(vec![Polygon::new(face_rect.exterior().clone(), face_holes)]),
        z_face,
        true,
    );
    mb.add_wall(face_rect.exterior(), z_face, z_bottom, knobs.slope_inset);
    for poly in &glyph_polys {
        mb.add_wall(poly.exterior(), z_face, z_engrave, knobs.slope_inset);
        for hole in poly.interiors() {
            let mut coords: Vec<Coord<f64>> = hole.0.clone();
            coords.reverse();
            let island = LineString::from(coords);
            mb.add_wall(&island, z_face, z_engrave, knobs.slope_inset);
            mb.add_cap(
                &MultiPolygon(vec![Polygon::new(island, vec![])]),
                z_face,
                true,
            );
        }
    }
    mb.build()
}

/// Button engrave-floor mesh — filled glyph polygons at engrave_z.
/// Material color = the dark variant (cream_dark / coral_dark) so the
/// engraved letter reads as a darker recess against the face.
fn build_button_engrave_mesh(knobs: &DebugKnobs, font: &[u8], spec: &BtnSpec) -> Mesh {
    let mut mb = MeshBuilder::default();
    let glyph_polys = layout_text_centered(font, spec.glyph, Vec2::ZERO, BTN_GLYPH_EM);
    let z_engrave = -knobs.btn_engrave_delta * knobs.paper_thickness;
    for poly in &glyph_polys {
        mb.add_cap(&MultiPolygon(vec![poly.clone()]), z_engrave, true);
    }
    mb.build()
}

// ─── Materials ────────────────────────────────────────────────────────

#[derive(Resource)]
struct PaperTextures {
    diffuse: Handle<Image>,
    normal: Handle<Image>,
}

fn make_material(
    materials: &mut Assets<StandardMaterial>,
    tex: &PaperTextures,
    color: Color,
) -> Handle<StandardMaterial> {
    materials.add(StandardMaterial {
        base_color: color,
        base_color_texture: Some(tex.diffuse.clone()),
        normal_map_texture: Some(tex.normal.clone()),
        perceptual_roughness: 0.92,
        metallic: 0.0,
        reflectance: 0.32,
        ..default()
    })
}

// ─── Setup ────────────────────────────────────────────────────────────

fn load_font_bytes() -> Vec<u8> {
    for p in FONT_FALLBACKS {
        if let Ok(d) = std::fs::read(p) {
            if Face::parse(&d, 0).is_ok() {
                return d;
            }
        }
    }
    panic!("no font found in fallback list: {:?}", FONT_FALLBACKS);
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut images: ResMut<Assets<Image>>,
) {
    // Camera (transform synced from CamCtl each frame).
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(0.0, 0.0, 14.0).looking_at(Vec3::ZERO, Vec3::Y),
        Projection::from(PerspectiveProjection {
            fov: 32f32.to_radians(),
            near: 0.1,
            far: 100.0,
            ..default()
        }),
        bevy::light::ShadowFilteringMethod::Gaussian,
        MainCam,
    ));

    // Lights — directions/illuminance/biases all live in DebugKnobs so
    // the panel can drive them. The sync system below applies knob
    // values to the spawned lights each frame.
    let knobs0 = load_knobs().unwrap_or_default();
    commands.spawn((
        DirectionalLight {
            color: Color::srgb(1.0, 0.97, 0.92),
            illuminance: knobs0.key_illuminance,
            shadows_enabled: true,
            shadow_depth_bias: knobs0.shadow_depth_bias,
            shadow_normal_bias: knobs0.shadow_normal_bias,
            ..default()
        },
        CascadeShadowConfigBuilder {
            num_cascades: 1,
            minimum_distance: 0.05,
            first_cascade_far_bound: knobs0.cascade_max,
            maximum_distance: knobs0.cascade_max,
            ..default()
        }
        .build(),
        Transform::from_xyz(0.0, 0.0, 0.0)
            .looking_to(polar_light_dir(knobs0.key_az_deg, knobs0.key_elev_deg), Vec3::Z),
        KeyLight,
    ));
    commands.spawn((
        DirectionalLight {
            color: Color::srgb(0.86, 0.88, 1.0),
            illuminance: knobs0.fill_illuminance,
            shadows_enabled: false,
            ..default()
        },
        Transform::from_xyz(0.0, 0.0, 0.0)
            .looking_to(polar_light_dir(knobs0.fill_az_deg, knobs0.fill_elev_deg), Vec3::Z),
        FillLight,
    ));
    commands.insert_resource(GlobalAmbientLight {
        color: Color::WHITE,
        brightness: knobs0.ambient_brightness,
        affects_lightmapped_meshes: true,
    });

    // Paper textures.
    let diffuse_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("assets/textures/paper_diffuse.jpg");
    let normal_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("assets/textures/paper_normal.jpg");
    let paper_diffuse = images.add(load_mipped_image(&diffuse_path, true));
    let paper_normal = images.add(load_mipped_image(&normal_path, false));
    let tex = PaperTextures { diffuse: paper_diffuse, normal: paper_normal };

    let knobs = load_knobs().unwrap_or_default();
    let calc = CalcState::new();
    let font = LoadedFont(load_font_bytes());

    spawn_scene(&mut commands, &mut meshes, &mut materials, &tex, &knobs, &calc, &font.0);

    let init_sig = knobs_full_signature(&knobs);
    let knob_sig_for_rebuild = knob_signature(&knobs);
    commands.insert_resource(tex);
    commands.insert_resource(font);
    commands.insert_resource(calc);
    commands.insert_resource(knobs);
    commands.insert_resource(UiState::default());
    commands.insert_resource(CamCtl::default_pose());
    commands.insert_resource(SceneRebuildMarker {
        last_knob_signature: knob_sig_for_rebuild,
        last_display: "0".into(),
    });
    commands.insert_resource(PersistTimer { secs: 0.0, last_sig: init_sig });
}

fn spawn_scene(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardMaterial>,
    tex: &PaperTextures,
    knobs: &DebugKnobs,
    calc: &CalcState,
    font: &[u8],
) {
    // Material colors — tuned against the inspiration screenshot.
    let cream = Color::srgb(0.930, 0.870, 0.745);
    let coral = Color::srgb(0.910, 0.450, 0.350);
    let cream_dark = Color::srgb(0.560, 0.495, 0.395);
    let coral_dark = Color::srgb(0.550, 0.280, 0.210);
    let taupe = Color::srgb(0.500, 0.450, 0.385);
    let display_olive_color = Color::srgb(0.620, 0.650, 0.500);
    let display_ink_color = Color::srgb(0.300, 0.340, 0.240);

    // Paper — the single outer cream layer with a panel cutout.
    let paper_mat = make_material(materials, tex, cream);
    commands.spawn((
        Mesh3d(meshes.add(build_paper_mesh(knobs))),
        MeshMaterial3d(paper_mat),
        Transform::IDENTITY,
        BgEntity,
    ));

    // Panel.
    let panel_mat = make_material(materials, tex, taupe);
    commands.spawn((
        Mesh3d(meshes.add(build_panel_mesh(knobs))),
        MeshMaterial3d(panel_mat),
        Transform::IDENTITY,
        BgEntity,
    ));

    // Display surface (olive) + engrave floor (dark olive).
    let digits = display_digits(knobs, font, &calc.display);
    let display_face_mat = make_material(materials, tex, display_olive_color);
    commands.spawn((
        Mesh3d(meshes.add(build_display_face_mesh(knobs, &digits))),
        MeshMaterial3d(display_face_mat),
        Transform::IDENTITY,
        BgEntity,
    ));
    if !digits.is_empty() {
        let display_ink_mat = make_material(materials, tex, display_ink_color);
        commands.spawn((
            Mesh3d(meshes.add(build_display_engrave_mesh(knobs, &digits))),
            MeshMaterial3d(display_ink_mat),
            Transform::IDENTITY,
            BgEntity,
        ));
    }

    // Buttons — face entity + engrave-floor entity per button. Both
    // get a `Btn` component so the animator drives them together.
    let face_z_rest = knobs.btn_idle_step * knobs.paper_thickness;
    let half = Vec2::new(BTN_W * 0.5, BTN_H * 0.5);
    for spec in BUTTONS {
        let center = btn_center(spec);
        let face_color = match spec.color {
            BtnColor::Cream => cream,
            BtnColor::Coral => coral,
        };
        let engrave_color = match spec.color {
            BtnColor::Cream => cream_dark,
            BtnColor::Coral => coral_dark,
        };
        let face_mat = make_material(materials, tex, face_color);
        let engrave_mat = make_material(materials, tex, engrave_color);
        let face_mesh = meshes.add(build_button_face_mesh(knobs, font, spec));
        let engrave_mesh = meshes.add(build_button_engrave_mesh(knobs, font, spec));
        let btn_component = || Btn {
            id: spec.id,
            target_face_z: face_z_rest,
            hit_min: center - half,
            hit_max: center + half,
        };
        commands.spawn((
            Mesh3d(face_mesh),
            MeshMaterial3d(face_mat),
            Transform::from_xyz(center.x, center.y, face_z_rest),
            btn_component(),
            BgEntity,
        ));
        commands.spawn((
            Mesh3d(engrave_mesh),
            MeshMaterial3d(engrave_mat),
            Transform::from_xyz(center.x, center.y, face_z_rest),
            btn_component(),
            BgEntity,
        ));
    }
}

// ─── Per-frame systems ────────────────────────────────────────────────

fn cursor_world_xy(
    windows: &Query<&Window>,
    cam_q: &Query<(&Camera, &GlobalTransform), With<MainCam>>,
    knobs: &DebugKnobs,
) -> Option<Vec2> {
    let window = windows.iter().next()?;
    let cursor = window.cursor_position()?;
    let (camera, cam_xform) = cam_q.iter().next()?;
    let ray = camera.viewport_to_world(cam_xform, cursor).ok()?;
    let target_z = knobs.btn_idle_step * knobs.paper_thickness;
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
    knobs: Res<DebugKnobs>,
    mut state: ResMut<UiState>,
    mut calc: ResMut<CalcState>,
    btn_q: Query<&Btn>,
) {
    let world = match cursor_world_xy(&windows, &cam_q, &knobs) {
        Some(p) => p,
        None => {
            state.hovered = None;
            return;
        }
    };
    let mut hovered: Option<WidgetId> = None;
    for btn in btn_q.iter() {
        if world.x >= btn.hit_min.x && world.x <= btn.hit_max.x
            && world.y >= btn.hit_min.y && world.y <= btn.hit_max.y
        {
            hovered = Some(btn.id);
            break;
        }
    }
    state.hovered = hovered;
    if mouse.just_pressed(MouseButton::Left) {
        state.pressed = hovered;
    }
    if mouse.just_released(MouseButton::Left) {
        if let (Some(p), Some(h)) = (state.pressed, hovered) {
            if p == h {
                calc.press(h);
            }
        }
        state.pressed = None;
    }
}

fn target_face_z(state: &UiState, knobs: &DebugKnobs, id: WidgetId) -> f32 {
    let step = if state.pressed == Some(id) {
        knobs.btn_press_step
    } else if state.hovered == Some(id) && state.pressed.is_none() {
        knobs.btn_hover_step
    } else {
        knobs.btn_idle_step
    };
    step * knobs.paper_thickness
}

fn animate_buttons(
    time: Res<Time>,
    state: Res<UiState>,
    knobs: Res<DebugKnobs>,
    mut q: Query<(&mut Btn, &mut Transform)>,
) {
    let dt = time.delta_secs();
    let alpha = 1.0 - (-dt / BTN_TWEEN_TAU).exp();
    for (mut btn, mut xf) in q.iter_mut() {
        let target = target_face_z(&state, &knobs, btn.id);
        btn.target_face_z = target;
        let cur = xf.translation.z;
        let next = cur + (target - cur) * alpha;
        xf.translation.z = if (next - target).abs() < BTN_SNAP_EPS { target } else { next };
    }
}

fn camera_input_system(
    keys: Res<ButtonInput<KeyCode>>,
    mut wheel: MessageReader<bevy::input::mouse::MouseWheel>,
    time: Res<Time>,
    mut cam: ResMut<CamCtl>,
) {
    let dt = time.delta_secs();
    if keys.pressed(KeyCode::ArrowLeft)  { cam.yaw -= 1.6 * dt; }
    if keys.pressed(KeyCode::ArrowRight) { cam.yaw += 1.6 * dt; }
    if keys.pressed(KeyCode::ArrowDown)  { cam.tilt = (cam.tilt + 1.2 * dt).clamp(0.0, 1.45); }
    if keys.pressed(KeyCode::ArrowUp)    { cam.tilt = (cam.tilt - 1.2 * dt).clamp(0.0, 1.45); }
    use bevy::input::mouse::MouseScrollUnit;
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

fn debug_panel(
    mut contexts: EguiContexts,
    mut knobs: ResMut<DebugKnobs>,
    mut cam: ResMut<CamCtl>,
    keys: Res<ButtonInput<KeyCode>>,
) {
    if keys.just_pressed(KeyCode::KeyL) {
        knobs.panel_open = !knobs.panel_open;
    }
    if !knobs.panel_open { return; }
    let Ok(ctx) = contexts.ctx_mut() else { return };
    egui::Window::new("scene controls (L to hide)")
        .default_width(320.0)
        .show(ctx, |ui| {
            egui::CollapsingHeader::new("camera")
                .default_open(true)
                .show(ui, |ui| {
                    ui.add(egui::Slider::new(&mut cam.tilt, 0.0..=1.45).text("tilt (rad)"));
                    ui.add(egui::Slider::new(&mut cam.yaw, -3.14..=3.14).text("yaw (rad)"));
                    ui.add(egui::Slider::new(&mut cam.dist, 4.0..=80.0).text("distance"));
                });
            egui::CollapsingHeader::new("lighting")
                .default_open(true)
                .show(ui, |ui| {
                    ui.label("key light");
                    ui.add(egui::Slider::new(&mut knobs.key_illuminance, 0.0..=20_000.0).text("illuminance"));
                    ui.add(egui::Slider::new(&mut knobs.key_az_deg, -180.0..=180.0).text("azimuth°"));
                    ui.add(egui::Slider::new(&mut knobs.key_elev_deg, 0.0..=90.0).text("elevation°"));
                    ui.separator();
                    ui.label("fill light");
                    ui.add(egui::Slider::new(&mut knobs.fill_illuminance, 0.0..=10_000.0).text("illuminance"));
                    ui.add(egui::Slider::new(&mut knobs.fill_az_deg, -180.0..=180.0).text("azimuth°"));
                    ui.add(egui::Slider::new(&mut knobs.fill_elev_deg, 0.0..=90.0).text("elevation°"));
                    ui.separator();
                    ui.add(egui::Slider::new(&mut knobs.ambient_brightness, 0.0..=600.0).text("ambient"));
                    ui.add(egui::Slider::new(&mut knobs.shadow_normal_bias, 0.0..=2.0).text("shadow normal bias"));
                    ui.add(egui::Slider::new(&mut knobs.shadow_depth_bias, 0.0..=0.20).text("shadow depth bias"));
                    ui.separator();
                    ui.label("shadow map quality");
                    ui.add(egui::Slider::new(&mut knobs.cascade_max, 6.0..=80.0).text("cascade extent"));
                    egui::ComboBox::from_label("shadow map size")
                        .selected_text(format!("{}", knobs.shadow_map_size))
                        .show_ui(ui, |ui| {
                            for s in [1024u32, 2048, 4096, 8192] {
                                ui.selectable_value(&mut knobs.shadow_map_size, s, format!("{}", s));
                            }
                        });
                });
            egui::CollapsingHeader::new("thickness")
                .default_open(true)
                .show(ui, |ui| {
                    ui.label("global scale: world-units of z per 'step'");
                    ui.add(
                        egui::Slider::new(&mut knobs.paper_thickness, 0.01..=3.0)
                            .text("layer scale")
                            .logarithmic(true),
                    );
                    ui.add(
                        egui::Slider::new(&mut knobs.slope_inset, 0.0..=0.50)
                            .text("wall slope inset")
                            .logarithmic(true),
                    );
                    ui.separator();
                    ui.label("per-layer slab thickness (world units)");
                    ui.add(egui::Slider::new(&mut knobs.paper_layer_thick, 0.0..=2.0).text("paper slab"));
                    ui.add(egui::Slider::new(&mut knobs.panel_layer_thick, 0.0..=2.0).text("panel slab"));
                    ui.add(egui::Slider::new(&mut knobs.display_layer_thick, 0.0..=2.0).text("display slab"));
                });
            egui::CollapsingHeader::new("layer depths")
                .default_open(true)
                .show(ui, |ui| {
                    let pt = knobs.paper_thickness;
                    ui.label("each layer's z = step × layer scale");
                    let z_paper = knobs.paper_step * pt;
                    ui.add(egui::Slider::new(&mut knobs.paper_step, 0.0..=12.0)
                        .text(format!("paper  (z = {:.3})", z_paper)));
                    let z_panel = knobs.panel_taupe_step * pt;
                    ui.add(egui::Slider::new(&mut knobs.panel_taupe_step, 0.0..=12.0)
                        .text(format!("panel  (z = {:.3})", z_panel)));
                    let z_disp = knobs.display_olive_step * pt;
                    ui.add(egui::Slider::new(&mut knobs.display_olive_step, 0.0..=12.0)
                        .text(format!("display  (z = {:.3})", z_disp)));
                    let z_ink = knobs.display_ink_step * pt;
                    ui.add(egui::Slider::new(&mut knobs.display_ink_step, 0.0..=12.0)
                        .text(format!("display ink  (z = {:.3})", z_ink)));
                });
            egui::CollapsingHeader::new("button animation")
                .default_open(false)
                .show(ui, |ui| {
                    let pt = knobs.paper_thickness;
                    let z_idle = knobs.btn_idle_step * pt;
                    ui.add(egui::Slider::new(&mut knobs.btn_idle_step, 0.0..=12.0)
                        .text(format!("idle  (z = {:.3})", z_idle)));
                    let z_hover = knobs.btn_hover_step * pt;
                    ui.add(egui::Slider::new(&mut knobs.btn_hover_step, 0.0..=12.0)
                        .text(format!("hover  (z = {:.3})", z_hover)));
                    let z_press = knobs.btn_press_step * pt;
                    ui.add(egui::Slider::new(&mut knobs.btn_press_step, 0.0..=12.0)
                        .text(format!("press  (z = {:.3})", z_press)));
                    let z_engr = knobs.btn_engrave_delta * pt;
                    ui.add(egui::Slider::new(&mut knobs.btn_engrave_delta, 0.0..=4.0)
                        .text(format!("engrave delta  (Δz = {:.3})", z_engr)));
                });
        });
}

/// One-line "knob fingerprint" — when this changes, rebuild the
/// background and per-button meshes (button animation knobs are not
/// included; those just re-target the transforms).
fn knob_signature(k: &DebugKnobs) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut h = DefaultHasher::new();
    k.paper_thickness.to_bits().hash(&mut h);
    k.slope_inset.to_bits().hash(&mut h);
    k.panel_taupe_step.to_bits().hash(&mut h);
    k.display_olive_step.to_bits().hash(&mut h);
    k.display_ink_step.to_bits().hash(&mut h);
    k.paper_step.to_bits().hash(&mut h);
    k.paper_layer_thick.to_bits().hash(&mut h);
    k.panel_layer_thick.to_bits().hash(&mut h);
    k.display_layer_thick.to_bits().hash(&mut h);
    k.btn_engrave_delta.to_bits().hash(&mut h);
    // btn_idle_step affects the button mesh's `z_bottom` calc; include.
    k.btn_idle_step.to_bits().hash(&mut h);
    h.finish()
}

fn rebuild_scene_if_needed(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    tex: Res<PaperTextures>,
    knobs: Res<DebugKnobs>,
    font: Res<LoadedFont>,
    calc: Res<CalcState>,
    mut marker: ResMut<SceneRebuildMarker>,
    bg_q: Query<Entity, With<BgEntity>>,
) {
    let sig = knob_signature(&knobs);
    let display_changed = calc.display != marker.last_display;
    if sig == marker.last_knob_signature && !display_changed {
        return;
    }
    // Despawn all background+button entities and respawn from scratch.
    for e in bg_q.iter() {
        commands.entity(e).despawn();
    }
    spawn_scene(&mut commands, &mut meshes, &mut materials, &tex, &knobs, &calc, &font.0);
    marker.last_knob_signature = sig;
    marker.last_display = calc.display.clone();
}

// ─── Main ─────────────────────────────────────────────────────────────

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "button + counter".into(),
                resolution: (1200u32, 1000u32).into(),
                ..default()
            }),
            ..default()
        }))
        .add_plugins(EguiPlugin::default())
        // Match cream paper so the area beyond the paper edge blends
        // — overhead view exposes more of the clear color than a
        // tilted view would.
        .insert_resource(ClearColor(Color::srgb(0.930, 0.870, 0.745)))
        .insert_resource({
            let initial = load_knobs().unwrap_or_default();
            bevy::light::DirectionalLightShadowMap { size: initial.shadow_map_size as usize }
        })
        .add_systems(Startup, setup)
        .add_systems(
            Update,
            (
                camera_input_system,
                sync_camera,
                sync_lights_from_knobs,
                input_system,
                animate_buttons,
                rebuild_scene_if_needed,
                persist_knobs_periodic,
            )
                .chain(),
        )
        .add_systems(EguiPrimaryContextPass, debug_panel)
        .run();
}

#[allow(dead_code)]
fn _silence_unused() {
    let _ = point_in_polygon_xy(Vec2::ZERO, &rounded_rect(0.0, 0.0, 1.0, 1.0, 0.1));
    let _ = btn_polygon_world(&BUTTONS[0]);
}
