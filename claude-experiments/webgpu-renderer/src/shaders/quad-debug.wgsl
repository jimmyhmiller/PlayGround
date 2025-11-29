// Debug version of quad shader that outputs diagnostic colors

struct Quad {
    order: u32,
    border_style: u32,
    bounds: Bounds,
    content_mask: Bounds,
    // WGSL automatically adds padding here for alignment
    background: Background,
    border_color: Hsla,
    corner_radii: Corners,
    border_widths: Edges,
    transform: Transform,
    opacity: f32,
    pad: u32,
}

@group(0) @binding(0) var<uniform> globals: GlobalParams;
@group(0) @binding(1) var<storage, read> quads: array<Quad>;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) @interpolate(flat) quad_id: u32,
    @location(1) @interpolate(flat) debug_color: vec4<f32>,
}

@vertex
fn vs_main(
    @builtin(vertex_index) vertex_id: u32,
    @builtin(instance_index) instance_id: u32
) -> VertexOutput {
    // Generate unit quad vertices
    let unit_vertex = vec2<f32>(f32(vertex_id & 1u), 0.5 * f32(vertex_id & 2u));

    let quad = quads[instance_id];

    // Try to read from buffer
    let local_position = unit_vertex * quad.bounds.size + quad.bounds.origin;

    // No transform for debugging
    var out: VertexOutput;
    out.position = to_device_position(local_position, globals.viewport_size);
    out.quad_id = instance_id;

    // Output the solid color from quad data
    out.debug_color = hsla_to_rgba(quad.background.solid);

    return out;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    // Just return the debug color (background solid color)
    return input.debug_color;
}
