// Screen-space overlay: solid rectangles (panels) and bitmap-font text. Both are
// drawn as instanced quads pulled from storage buffers, positioned in pixels and
// converted to NDC here. Text samples an 8x8 monochrome font packed as one u32
// per glyph row (bit 0 = leftmost pixel).

struct Overlay {
    viewport: vec2<f32>,
    _pad: vec2<f32>,
};
@group(0) @binding(0) var<uniform> ov: Overlay;

struct Rect {
    pos: vec2<f32>,   // top-left in pixels
    size: vec2<f32>,  // pixels
    color: u32,       // packed RGBA8
    _p: u32,
};
struct Glyph {
    pos: vec2<f32>,   // top-left in pixels
    size: vec2<f32>,  // pixels (cell size)
    color: u32,
    ch: u32,          // ASCII code
};

struct Seg {
    a: vec2<f32>,     // endpoint pixels
    b: vec2<f32>,
    color: u32,
    thickness: f32,
};

@group(1) @binding(0) var<storage, read> rects: array<Rect>;
@group(1) @binding(1) var<storage, read> glyphs: array<Glyph>;
@group(1) @binding(2) var<storage, read> font: array<u32>; // 128 * 8 rows
@group(1) @binding(3) var<storage, read> segs: array<Seg>;

fn unpack(c: u32) -> vec4<f32> {
    return vec4<f32>(
        f32(c & 0xffu) / 255.0,
        f32((c >> 8u) & 0xffu) / 255.0,
        f32((c >> 16u) & 0xffu) / 255.0,
        f32((c >> 24u) & 0xffu) / 255.0,
    );
}

fn quad_corner(vi: u32) -> vec2<f32> {
    var c = array<vec2<f32>, 6>(
        vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 0.0), vec2<f32>(1.0, 1.0),
        vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 1.0),
    );
    return c[vi];
}

fn px_to_ndc(px: vec2<f32>) -> vec2<f32> {
    // Pixel origin top-left, y down -> NDC origin center, y up.
    return vec2<f32>(px.x / ov.viewport.x * 2.0 - 1.0, 1.0 - px.y / ov.viewport.y * 2.0);
}

// ---- Rects ----
struct RectOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) color: vec4<f32>,
};
@vertex
fn vs_rect(@builtin(vertex_index) vi: u32, @builtin(instance_index) inst: u32) -> RectOut {
    let r = rects[inst];
    let corner = quad_corner(vi);
    let px = r.pos + corner * r.size;
    var o: RectOut;
    o.clip = vec4<f32>(px_to_ndc(px), 0.0, 1.0);
    o.color = unpack(r.color);
    return o;
}
@fragment
fn fs_rect(in: RectOut) -> @location(0) vec4<f32> {
    return in.color;
}

// ---- Segments (oriented quads between two pixel endpoints) ----
struct SegOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) color: vec4<f32>,
};
@vertex
fn vs_seg(@builtin(vertex_index) vi: u32, @builtin(instance_index) inst: u32) -> SegOut {
    let s = segs[inst];
    let dir = s.b - s.a;
    let len = max(length(dir), 0.0001);
    let d = dir / len;
    let n = vec2<f32>(-d.y, d.x) * (s.thickness * 0.5);
    let corner = quad_corner(vi); // (u along, v across) in [0,1]
    let along = mix(s.a, s.b, corner.x);
    let px = along + n * (corner.y * 2.0 - 1.0);
    var o: SegOut;
    o.clip = vec4<f32>(px_to_ndc(px), 0.0, 1.0);
    o.color = unpack(s.color);
    return o;
}
@fragment
fn fs_seg(in: SegOut) -> @location(0) vec4<f32> {
    return in.color;
}

// ---- Glyphs ----
struct GlyphOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
    @location(2) @interpolate(flat) ch: u32,
};
@vertex
fn vs_glyph(@builtin(vertex_index) vi: u32, @builtin(instance_index) inst: u32) -> GlyphOut {
    let g = glyphs[inst];
    let corner = quad_corner(vi);
    let px = g.pos + corner * g.size;
    var o: GlyphOut;
    o.clip = vec4<f32>(px_to_ndc(px), 0.0, 1.0);
    o.uv = corner;
    o.color = unpack(g.color);
    o.ch = g.ch;
    return o;
}
@fragment
fn fs_glyph(in: GlyphOut) -> @location(0) vec4<f32> {
    let ch = in.ch & 127u;
    let gx = u32(clamp(in.uv.x, 0.0, 0.999) * 8.0);
    let gy = u32(clamp(in.uv.y, 0.0, 0.999) * 8.0);
    let row = font[ch * 8u + gy];
    let bit = (row >> gx) & 1u;
    if (bit == 0u) {
        discard;
    }
    return in.color;
}
