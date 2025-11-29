// Path rasterization using Loop-Blinn algorithm for GPU curve rendering

struct PathRasterizationVertex {
    xy_position: vec2<f32>,
    st_position: vec2<f32>,  // Curve parameters for Loop-Blinn
    fill_color: Hsla,
    stroke_color: Hsla,
    stroke_width: f32,
    bounds: Bounds,
}

@group(0) @binding(0) var<uniform> globals: GlobalParams;
@group(0) @binding(1) var<storage, read> vertices: array<PathRasterizationVertex>;

struct PathRasterizationVarying {
    @builtin(position) position: vec4<f32>,
    @location(0) st_position: vec2<f32>,
    @location(1) @interpolate(flat) vertex_id: u32,
    @location(2) clip_distances: vec4<f32>,
}

@vertex
fn vs_path_rasterization(@builtin(vertex_index) vertex_id: u32) -> PathRasterizationVarying {
    let v = vertices[vertex_id];

    var out: PathRasterizationVarying;
    out.position = to_device_position(v.xy_position, globals.viewport_size);
    out.st_position = v.st_position;
    out.vertex_id = vertex_id;
    out.clip_distances = distance_from_clip_rect(v.xy_position, v.bounds);

    return out;
}

@fragment
fn fs_path_rasterization(input: PathRasterizationVarying) -> @location(0) vec4<f32> {
    let v = vertices[input.vertex_id];

    // Compute derivatives of st_position for Loop-Blinn algorithm
    // Must happen before branching for uniform control flow
    let dx = dpdx(input.st_position);
    let dy = dpdy(input.st_position);

    var alpha: f32;

    // Check if gradient is too small (degenerate triangle)
    if (length(vec2<f32>(dx.x, dy.x)) < 0.001) {
        // Gradient too small, treat as solid
        alpha = 1.0;
    } else {
        // Loop-Blinn implicit function: f(s,t) = s² - t
        // Points where f < 0 are inside the curve
        // Points where f > 0 are outside the curve

        // Compute gradient of implicit function
        // ∂f/∂x = 2s * ∂s/∂x - ∂t/∂x
        // ∂f/∂y = 2s * ∂s/∂y - ∂t/∂y
        let gradient = 2.0 * input.st_position.xx * vec2<f32>(dx.x, dy.x) - vec2<f32>(dx.y, dy.y);

        // Evaluate implicit function
        let f = input.st_position.x * input.st_position.x - input.st_position.y;

        // Compute signed distance to curve
        let distance = f / length(gradient);

        // Convert distance to alpha (0.5 pixel smooth transition)
        // Negative distance = inside curve (alpha = 1)
        // Positive distance = outside curve (alpha = 0)
        alpha = saturate(0.5 - distance);
    }

    // Use fill color for now (stroke support would require additional data)
    let fill_color = hsla_to_rgba(v.fill_color);

    // Output premultiplied alpha
    var color = vec4<f32>(fill_color.rgb * fill_color.a * alpha, fill_color.a * alpha);

    // Clip test (after derivative calculations)
    if (any(input.clip_distances < vec4<f32>(0.0))) {
        color = vec4<f32>(0.0);
    }

    return color;
}
