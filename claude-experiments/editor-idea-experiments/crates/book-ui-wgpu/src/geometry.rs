//! Polygon utilities, glyph outline → polygon conversion, and a
//! CPU-side mesh builder (positions + normals + indices). Pure logic;
//! no wgpu / windowing / Bevy.

use glam::Vec2;
use lyon::math::point as lyon_point;
use lyon::path::iterator::PathIterator;
use lyon::path::{Path as LyonPath, PathEvent};
use lyon::tessellation::{
    BuffersBuilder, FillOptions, FillRule, FillTessellator, FillVertex, VertexBuffers,
};
use ttf_parser::{Face, GlyphId, OutlineBuilder};

#[derive(Clone, Debug)]
pub struct Polygon {
    pub exterior: Vec<Vec2>,
    pub holes: Vec<Vec<Vec2>>,
}

impl Polygon {
    pub fn from_ring(ring: Vec<Vec2>) -> Self {
        Self { exterior: ring, holes: Vec::new() }
    }
    pub fn translated(&self, dx: f32, dy: f32) -> Self {
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
}

pub fn point_in_ring(p: Vec2, ring: &[Vec2]) -> bool {
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

pub fn ensure_ccw(mut pts: Vec<Vec2>) -> Vec<Vec2> {
    if signed_area(&pts) < 0.0 {
        pts.reverse();
    }
    pts
}

pub fn ensure_cw(mut pts: Vec<Vec2>) -> Vec<Vec2> {
    if signed_area(&pts) > 0.0 {
        pts.reverse();
    }
    pts
}

pub fn rounded_rect_ring(cx: f32, cy: f32, w: f32, h: f32, r: f32) -> Vec<Vec2> {
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

#[derive(Clone, Copy)]
pub enum Op {
    Move(Vec2),
    Line(Vec2),
    Quad(Vec2, Vec2),
    Cubic(Vec2, Vec2, Vec2),
    Close,
}

pub struct OpCollector {
    pub ops: Vec<Op>,
    pub scale: f32,
    pub offset: Vec2,
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

pub fn ops_to_path(ops: &[Op]) -> LyonPath {
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
                if started { pb.end(true); started = false; }
            }
        }
    }
    if started { pb.end(false); }
    pb.build()
}

pub fn flatten_path_to_contours(path: &LyonPath, tol: f32) -> Vec<Vec<Vec2>> {
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

pub fn build_glyph_polygons(contours: &[Vec<Vec2>]) -> Vec<Polygon> {
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

pub fn layout_text_centered(font: &[u8], text: &str, origin: Vec2, em: f32) -> Vec<Polygon> {
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
            for p in c {
                bmin = bmin.min(*p);
                bmax = bmax.max(*p);
            }
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

#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct Vertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
}

#[derive(Default)]
pub struct MeshBuilder {
    pub positions: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    pub indices: Vec<u32>,
}

impl MeshBuilder {
    /// Top (or bottom) cap face for a polygon-with-holes. `normal_up`
    /// puts the cap facing +Z.
    pub fn add_cap(&mut self, polygon: &Polygon, z: f32, normal_up: bool) {
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

    pub fn add_wall(&mut self, ring: &[Vec2], z_top: f32, z_bottom: f32) {
        let n = ring.len();
        if n < 3 { return; }
        for i in 0..n {
            let a = ring[i];
            let b = ring[(i + 1) % n];
            let edge = b - a;
            let elen = edge.length();
            if elen < 1e-7 { continue; }
            let nx = edge.y / elen;
            let ny = -edge.x / elen;
            let normal = [nx, ny, 0.0];
            let base = self.positions.len() as u32;
            self.positions.push([a.x, a.y, z_top]);
            self.positions.push([b.x, b.y, z_top]);
            self.positions.push([a.x, a.y, z_bottom]);
            self.positions.push([b.x, b.y, z_bottom]);
            for _ in 0..4 { self.normals.push(normal); }
            self.indices.extend_from_slice(&[
                base, base + 1, base + 2,
                base + 1, base + 3, base + 2,
            ]);
        }
    }

    pub fn into_vertices(self) -> (Vec<Vertex>, Vec<u32>) {
        let verts = self
            .positions
            .into_iter()
            .zip(self.normals.into_iter())
            .map(|(p, n)| Vertex { position: p, normal: n })
            .collect();
        (verts, self.indices)
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
