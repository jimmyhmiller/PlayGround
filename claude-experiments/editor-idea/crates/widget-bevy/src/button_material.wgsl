// Widget button — rounded rect SDF with optional border and soft drop
// shadow. The mesh is oversized so the shadow can fall outside the
// button's interactive rect; the SDF inside the mesh defines what's
// "the button" vs "the shadow halo."

#import bevy_sprite::mesh2d_vertex_output::VertexOutput

struct ButtonParams {
    // Total mesh size in pixels (button + 2 × shadow_blur on every side).
    mesh_size: vec2<f32>,
    // The actual button rect inside the mesh (label + padding).
    button_size: vec2<f32>,
    corner_radius: f32,
    border_width: f32,
    // Body fill (linear RGB).
    bg: vec4<f32>,
    // Border color (linear RGB).
    border: vec4<f32>,
    // Shadow color + base alpha at the button edge.
    shadow_color: vec4<f32>,
    // How far the shadow fades, pixels.
    shadow_blur: f32,
    // Push shadow down by this many pixels (positive = below).
    shadow_offset_y: f32,
    _pad0: f32,
    _pad1: f32,
}

@group(#{MATERIAL_BIND_GROUP}) @binding(0) var<uniform> params: ButtonParams;

fn rounded_rect_sdf(p: vec2<f32>, half_size: vec2<f32>, r: f32) -> f32 {
    let q = abs(p) - half_size + vec2<f32>(r);
    return length(max(q, vec2<f32>(0.0))) + min(max(q.x, q.y), 0.0) - r;
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    // Mesh-centered coord, then shift the button "up" by the shadow
    // offset so the shadow extension reads as falling downward.
    let p_mesh = (in.uv - vec2<f32>(0.5)) * params.mesh_size;
    let p_button = p_mesh - vec2<f32>(0.0, params.shadow_offset_y);
    let half_button = params.button_size * 0.5;
    let r = min(params.corner_radius, min(half_button.x, half_button.y));
    let d = rounded_rect_sdf(p_button, half_button, r);

    // Inside the button: coverage = 1, AA at the edge.
    let inside_coverage = 1.0 - smoothstep(-0.5, 0.5, d);

    // Body fill + inner border.
    let bw = params.border_width;
    let border_coverage = select(
        0.0,
        1.0 - smoothstep(-bw - 0.5, -bw + 0.5, d),
        bw > 0.0,
    );
    var inside_color = mix(params.bg.rgb, params.border.rgb, border_coverage);

    // Shadow outside the button. Squared smoothstep for a softer
    // gaussian-ish falloff.
    let shadow_t = clamp(d / max(params.shadow_blur, 0.001), 0.0, 1.0);
    let shadow_falloff = 1.0 - shadow_t;
    let shadow_alpha = params.shadow_color.a * shadow_falloff * shadow_falloff;

    // Composite: button on top of shadow. Where the button is opaque,
    // the shadow contributes nothing (we just see the button).
    let button_a = inside_coverage;
    let shadow_only_a = shadow_alpha * (1.0 - button_a);
    let out_rgb = inside_color * button_a + params.shadow_color.rgb * shadow_only_a;
    let out_a = button_a + shadow_only_a;
    return vec4<f32>(out_rgb, out_a);
}
