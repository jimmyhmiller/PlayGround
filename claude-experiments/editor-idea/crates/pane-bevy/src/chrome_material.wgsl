// Pane chrome — rounded rect SDF with optional border and focus glow.
//
// One material per pane, applied to a single mesh sized to the pane's
// rect. The fragment computes signed distance to the rounded rect and
// uses it to derive (in order, back to front):
//   1. fill         — opaque inside, transparent outside the rect
//   2. inner border — a band just inside the rect's edge
//   3. focus glow   — a soft inner ring that pulses when the pane has
//                     focus; mixed in over the fill
//
// All bands are anti-aliased by smoothstep on the distance field, so
// edges stay crisp at any DPI and any pane scale without an explicit
// MSAA pass.

#import bevy_sprite::mesh2d_vertex_output::VertexOutput

struct ChromeParams {
    // Pane size in pixels (matches the mesh size).
    size: vec2<f32>,
    // Corner radius, pixels.
    corner_radius: f32,
    // Border width, pixels. 0 disables the border band.
    border_width: f32,
    // Body fill color (linear RGB).
    bg: vec4<f32>,
    // Border color (linear RGB).
    border: vec4<f32>,
    // Focus glow color (linear RGB) and strength (a). When the pane
    // isn't focused, strength is 0 and this stage costs ~nothing.
    focus: vec4<f32>,
    // Width of the inner focus glow band, pixels. The glow fades from
    // `focus.rgb * focus.a` at the inside of the border to transparent
    // `focus_width` pixels deeper into the body.
    focus_width: f32,
    // Wall-clock-ish time in seconds. Drives the focus pulse so the
    // glow gently breathes when a pane has focus.
    time: f32,
    _pad0: f32,
    _pad1: f32,
}

@group(#{MATERIAL_BIND_GROUP}) @binding(0) var<uniform> params: ChromeParams;

// Standard rounded-rect SDF (Inigo Quilez). Returns < 0 inside, > 0
// outside, 0 right on the edge.
fn rounded_rect_sdf(p: vec2<f32>, half_size: vec2<f32>, r: f32) -> f32 {
    let q = abs(p) - half_size + vec2<f32>(r);
    return length(max(q, vec2<f32>(0.0))) + min(max(q.x, q.y), 0.0) - r;
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    // uv is [0,1] over the mesh. Convert to a centered pixel coord so
    // the SDF works in the natural "distance to the rect edge" units.
    let p = (in.uv - vec2<f32>(0.5)) * params.size;
    let half_size = params.size * 0.5;
    let r = min(params.corner_radius, min(half_size.x, half_size.y));
    let d = rounded_rect_sdf(p, half_size, r);

    // Coverage: 1 deep inside, fades to 0 across a 1px band at the
    // edge. AA without MSAA.
    let coverage = 1.0 - smoothstep(-0.5, 0.5, d);
    if (coverage <= 0.0) {
        return vec4<f32>(0.0);  // outside the rounded rect — transparent
    }

    // Subtle vertical gradient over the body: top is slightly lifted
    // (×1.06), bottom slightly recessed (×0.97). Reads as a top
    // light source without being obvious unless you look for it.
    let v = in.uv.y;  // 0 at top, 1 at bottom in this mesh
    let tone = mix(1.06, 0.97, v);
    var color = params.bg.rgb * tone;

    // Border band: from the rect edge inward by border_width. d is
    // negative inside, so the band lives in (-border_width .. 0).
    let bw = params.border_width;
    let border_coverage = select(
        0.0,
        1.0 - smoothstep(-bw - 0.5, -bw + 0.5, d),
        bw > 0.0,
    );
    color = mix(color, params.border.rgb, border_coverage);

    // Focus glow: an inner ring just past the border, fading inward.
    // Pulses gently when focused — 0.85..1.0 sinusoid at ~0.3 Hz so
    // it's noticeable but not distracting.
    let inset = -d - bw;
    let glow_t = clamp(1.0 - inset / max(params.focus_width, 0.001), 0.0, 1.0);
    let pulse = 0.85 + 0.15 * sin(params.time * 1.9);
    let glow_strength = params.focus.a * glow_t * glow_t * pulse;
    color = mix(color, params.focus.rgb, glow_strength);

    return vec4<f32>(color, coverage);
}
