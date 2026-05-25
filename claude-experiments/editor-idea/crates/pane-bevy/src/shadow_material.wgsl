// Pane drop shadow — soft falloff outside a rounded rect.
//
// Each pane gets a sibling shadow quad rendered on the MAIN camera
// (layer 0), so the shadow can extend outside the per-pane camera's
// viewport clip. The mesh is sized larger than the pane by `blur`
// pixels on every side; this shader computes the signed distance to
// the rounded rect (in the mesh's pixel coords) and smoothly fades
// the shadow as distance increases.

#import bevy_sprite::mesh2d_vertex_output::VertexOutput

struct ShadowParams {
    // Mesh size in pixels (pane size + 2*blur on each side).
    mesh_size: vec2<f32>,
    // The actual pane rect (smaller than mesh by 2*blur in each axis).
    rect_size: vec2<f32>,
    corner_radius: f32,
    blur: f32,
    // Shadow color (linear RGB) and strength (a) at the rect edge.
    color: vec4<f32>,
    // Vertical offset of the shadow center inside the mesh. Positive
    // pushes the shadow down so it sits below the pane. 0 = centered.
    offset_y: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
}

@group(#{MATERIAL_BIND_GROUP}) @binding(0) var<uniform> params: ShadowParams;

fn rounded_rect_sdf(p: vec2<f32>, half_size: vec2<f32>, r: f32) -> f32 {
    let q = abs(p) - half_size + vec2<f32>(r);
    return length(max(q, vec2<f32>(0.0))) + min(max(q.x, q.y), 0.0) - r;
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    // Pixel coord centered on the mesh, then shifted by offset_y so
    // the rounded rect we test against sits HIGHER inside the mesh,
    // pushing the shadow extension downward visually.
    let p = (in.uv - vec2<f32>(0.5)) * params.mesh_size - vec2<f32>(0.0, params.offset_y);
    let half_rect = params.rect_size * 0.5;
    let r = min(params.corner_radius, min(half_rect.x, half_rect.y));
    let d = rounded_rect_sdf(p, half_rect, r);

    // Inside the rect: full opaque base. Outside: smooth fall-off.
    // smoothstep gives a soft gaussian-ish edge over [0..blur].
    let outside_t = clamp(d / max(params.blur, 0.001), 0.0, 1.0);
    let falloff = 1.0 - outside_t;
    // Square the falloff for a softer, more shadow-like profile.
    let a = params.color.a * falloff * falloff;
    return vec4<f32>(params.color.rgb, a);
}
