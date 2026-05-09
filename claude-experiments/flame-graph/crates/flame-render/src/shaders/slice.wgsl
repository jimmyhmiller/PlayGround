// Instanced rectangle rendering for flame-graph slices.
//
// Per-frame uniform: viewport size in pixels and a single global y-scroll offset.
// Per-instance attributes: pixel-space rect already computed CPU-side (so f64 ns
// arithmetic stays on the CPU and never has to round-trip through f32 GPU math).

struct Uniforms {
    viewport_size_px: vec2<f32>,
    // Hovered slice index in the instance buffer, or 0xFFFFFFFF if none.
    hovered: u32,
    _pad: u32,
};

struct SliceInstance {
    @location(0) rect_px: vec4<f32>, // x, y, w, h (top-left origin, pixels)
    @location(1) color: vec4<f32>,
    @location(2) instance_id: u32,
    @location(3) flags: u32,
};

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) local: vec2<f32>, // 0..1 within the rect
    @location(2) size_px: vec2<f32>,
    @location(3) is_hovered: f32,
    @location(4) is_selected: f32,
};

@group(0) @binding(0) var<uniform> u: Uniforms;

// 6 vertices per instance from vertex_index, no vertex buffer.
// Layout: triangle list covering the unit square (0,0)-(1,1).
const QUAD = array<vec2<f32>, 6>(
    vec2<f32>(0.0, 0.0),
    vec2<f32>(1.0, 0.0),
    vec2<f32>(0.0, 1.0),
    vec2<f32>(0.0, 1.0),
    vec2<f32>(1.0, 0.0),
    vec2<f32>(1.0, 1.0),
);

@vertex
fn vs_main(
    @builtin(vertex_index) vid: u32,
    inst: SliceInstance,
) -> VsOut {
    let corner = QUAD[vid];
    let rect = inst.rect_px;
    let pos_px = vec2<f32>(rect.x + rect.z * corner.x, rect.y + rect.w * corner.y);

    // Pixel-space → clip-space (top-left origin → NDC).
    let ndc = vec2<f32>(
         (pos_px.x / u.viewport_size_px.x) * 2.0 - 1.0,
        -((pos_px.y / u.viewport_size_px.y) * 2.0 - 1.0),
    );

    var out: VsOut;
    out.clip = vec4<f32>(ndc, 0.0, 1.0);
    out.color = inst.color;
    out.local = corner;
    out.size_px = rect.zw;
    out.is_hovered = select(0.0, 1.0, inst.instance_id == u.hovered);
    // bit 1 = selected
    out.is_selected = select(0.0, 1.0, (inst.flags & 2u) != 0u);
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    var color = in.color.rgb;

    // 1px border. Darken the outer ring; only when the slice is at least 3px wide
    // to avoid making tiny slices look like solid black.
    let edge_x_px = min(in.local.x * in.size_px.x, (1.0 - in.local.x) * in.size_px.x);
    let edge_y_px = min(in.local.y * in.size_px.y, (1.0 - in.local.y) * in.size_px.y);
    let edge_px = min(edge_x_px, edge_y_px);
    if (in.size_px.x >= 3.0 && in.size_px.y >= 3.0 && edge_px < 1.0) {
        color = color * 0.55;
    }

    if (in.is_hovered > 0.5) {
        color = mix(color, vec3<f32>(1.0), 0.25);
    }
    if (in.is_selected > 0.5) {
        // Selection ring: lighten + add a bright outline.
        color = mix(color, vec3<f32>(1.0), 0.45);
        if (in.size_px.x >= 4.0 && in.size_px.y >= 4.0 && edge_px < 2.0) {
            color = vec3<f32>(1.0, 0.95, 0.6);
        }
    }

    return vec4<f32>(color, 1.0);
}
