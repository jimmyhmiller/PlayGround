// Path rendering shader

struct VertexInput {
    @location(0) position: vec2<f32>,
}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
}

struct PathUniforms {
    transform: mat3x3<f32>,
    fill_color: vec4<f32>,
    stroke_color: vec4<f32>,
    opacity: f32,
    filled: u32,
    stroked: u32,
    pad: u32,
}

@group(0) @binding(0) var<uniform> globals: GlobalParams;
@group(0) @binding(1) var<uniform> path_uniforms: PathUniforms;

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    // Apply transform
    let transformed = path_uniforms.transform * vec3<f32>(input.position, 1.0);

    var output: VertexOutput;
    output.position = to_device_position(transformed.xy, globals.viewport_size);
    return output;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    var color: vec4<f32>;

    if (path_uniforms.filled != 0u) {
        color = path_uniforms.fill_color;
    } else {
        color = path_uniforms.stroke_color;
    }

    // Apply opacity
    color = vec4<f32>(color.rgb * path_uniforms.opacity, color.a * path_uniforms.opacity);

    return blend_color(color, 1.0, globals.premultiplied_alpha);
}
