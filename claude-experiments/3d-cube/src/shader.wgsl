struct Uniforms {
    view_proj: mat4x4<f32>,
    point_size: f32,
    screen_width: f32,
    screen_height: f32,
    _pad: f32,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

// --- Point cloud shader (instanced quads) ---

const QUAD_POSITIONS: array<vec2<f32>, 6> = array<vec2<f32>, 6>(
    vec2<f32>(-0.5, -0.5),
    vec2<f32>( 0.5, -0.5),
    vec2<f32>( 0.5,  0.5),
    vec2<f32>(-0.5, -0.5),
    vec2<f32>( 0.5,  0.5),
    vec2<f32>(-0.5,  0.5),
);

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) quad_uv: vec2<f32>,
};

@vertex
fn vs_main(
    @location(0) position: vec3<f32>,
    @location(1) color: vec4<f32>,
    @builtin(vertex_index) vertex_index: u32,
) -> VertexOutput {
    var out: VertexOutput;
    let center_clip = uniforms.view_proj * vec4<f32>(position, 1.0);
    let quad_pos = QUAD_POSITIONS[vertex_index % 6u];
    let pixel_offset = quad_pos * uniforms.point_size;
    let clip_offset = vec2<f32>(
        pixel_offset.x * 2.0 / uniforms.screen_width,
        pixel_offset.y * 2.0 / uniforms.screen_height,
    );
    out.clip_position = vec4<f32>(
        center_clip.x + clip_offset.x * center_clip.w,
        center_clip.y + clip_offset.y * center_clip.w,
        center_clip.z,
        center_clip.w,
    );
    out.color = color;
    out.quad_uv = quad_pos + 0.5;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let dist = length(in.quad_uv - vec2<f32>(0.5, 0.5));
    if dist > 0.5 {
        discard;
    }
    let alpha = in.color.a * smoothstep(0.5, 0.35, dist);
    return vec4<f32>(in.color.rgb, alpha);
}

// --- Wireframe cube ---

@vertex
fn vs_line(@location(0) position: vec3<f32>) -> @builtin(position) vec4<f32> {
    return uniforms.view_proj * vec4<f32>(position, 1.0);
}

@fragment
fn fs_line() -> @location(0) vec4<f32> {
    return vec4<f32>(0.15, 0.2, 0.3, 1.0);
}

// --- Minimap 2D shader ---
// Draws colored quads in NDC. Each quad vertex has a 2D position and a color.

struct MinimapVertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
};

@vertex
fn vs_minimap(
    @location(0) position: vec2<f32>,
    @location(1) color: vec4<f32>,
) -> MinimapVertexOutput {
    var out: MinimapVertexOutput;
    out.clip_position = vec4<f32>(position, 0.0, 1.0);
    out.color = color;
    return out;
}

@fragment
fn fs_minimap(in: MinimapVertexOutput) -> @location(0) vec4<f32> {
    return in.color;
}
