// Quad shader with HARDCODED VALUES (no storage buffer reads)
// Used to test if the pipeline works at all

@group(0) @binding(0) var<uniform> globals: GlobalParams;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
}

@vertex
fn vs_main(
    @builtin(vertex_index) vertex_id: u32,
    @builtin(instance_index) instance_id: u32
) -> VertexOutput {
    // Generate unit quad vertices
    let unit_vertex = vec2<f32>(f32(vertex_id & 1u), 0.5 * f32(vertex_id & 2u));

    // HARDCODED: quad from (100,100) to (300,300)
    let bounds_origin = vec2<f32>(100.0, 100.0);
    let bounds_size = vec2<f32>(200.0, 200.0);

    let local_position = unit_vertex * bounds_size + bounds_origin;

    var out: VertexOutput;
    out.position = to_device_position(local_position, globals.viewport_size);

    return out;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(1.0, 0.0, 0.0, 1.0); // RED
}
