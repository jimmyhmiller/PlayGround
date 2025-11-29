// Surface rendering shader for external textures (video, canvas, etc.)

struct SurfaceUniforms {
    transform: mat3x3<f32>,
    opacity: f32,
    grayscale: f32,
    corner_radii: vec4<f32>,
    pad: vec2<f32>,
}

@group(0) @binding(0) var<uniform> globals: GlobalParams;
@group(0) @binding(1) var<uniform> surface_uniforms: SurfaceUniforms;
@group(0) @binding(2) var surface_texture: texture_2d<f32>;
@group(0) @binding(3) var surface_sampler: sampler;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) tex_coord: vec2<f32>,
    @location(1) local_pos: vec2<f32>,
    @location(2) bounds_size: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    // Generate unit quad vertices
    var positions = array<vec2<f32>, 4>(
        vec2<f32>(0.0, 0.0),
        vec2<f32>(1.0, 0.0),
        vec2<f32>(0.0, 1.0),
        vec2<f32>(1.0, 1.0),
    );

    let local_pos = positions[vertex_index];

    // Apply transform
    let transformed = surface_uniforms.transform * vec3<f32>(local_pos, 1.0);

    var output: VertexOutput;
    output.clip_position = to_device_position(transformed.xy, globals.viewport_size);
    output.tex_coord = local_pos;
    output.local_pos = local_pos;

    // Pass bounds size for corner radius calculations
    let size_x = length(surface_uniforms.transform[0].xy);
    let size_y = length(surface_uniforms.transform[1].xy);
    output.bounds_size = vec2<f32>(size_x, size_y);

    return output;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    // Sample texture
    var color = textureSample(surface_texture, surface_sampler, input.tex_coord);

    // Apply grayscale if needed
    if (surface_uniforms.grayscale > 0.5) {
        let luminance = dot(color.rgb, vec3<f32>(0.299, 0.587, 0.114));
        color = vec4<f32>(vec3<f32>(luminance), color.a);
    }

    // Apply opacity
    color = vec4<f32>(color.rgb * surface_uniforms.opacity, color.a * surface_uniforms.opacity);

    // Apply rounded corners
    let pixel_pos = input.local_pos * input.bounds_size;
    let corner_sdf = quad_sdf(
        pixel_pos,
        Bounds(vec2<f32>(0.0, 0.0), input.bounds_size),
        Corners(
            surface_uniforms.corner_radii.x,
            surface_uniforms.corner_radii.y,
            surface_uniforms.corner_radii.z,
            surface_uniforms.corner_radii.w
        )
    );

    // Smooth edge with antialiasing
    let alpha_multiplier = saturate(0.5 - corner_sdf);
    color = vec4<f32>(color.rgb, color.a * alpha_multiplier);

    return blend_color(color, 1.0, globals.premultiplied_alpha);
}
