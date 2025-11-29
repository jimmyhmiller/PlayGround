// Shadow rendering shader with analytical Gaussian blur

struct Shadow {
    order: u32,
    blur_radius: f32,
    bounds: Bounds,
    corner_radii: Corners,
    content_mask: Bounds,
    color: Hsla,
    transform: Transform,
    opacity: f32,
    pad: u32,
}

@group(0) @binding(0) var<uniform> globals: GlobalParams;
@group(0) @binding(1) var<storage, read> shadows: array<Shadow>;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) @interpolate(flat) shadow_id: u32,
    @location(1) @interpolate(flat) color: vec4<f32>,
    @location(2) clip_distances: vec4<f32>,
}

// Gaussian function
fn gaussian(x: f32, sigma: f32) -> f32 {
    return exp(-(x * x) / (2.0 * sigma * sigma)) / (sqrt(2.0 * M_PI_F) * sigma);
}

// Error function approximation
fn erf(v: vec2<f32>) -> vec2<f32> {
    let s = sign(v);
    let a = abs(v);
    let r1 = 1.0 + (0.278393 + (0.230389 + (0.000972 + 0.078108 * a) * a) * a) * a;
    let r2 = r1 * r1;
    return s - s / (r2 * r2);
}

// Analytically integrated blur along X axis
fn blur_along_x(x: f32, y: f32, sigma: f32, corner: f32, half_size: vec2<f32>) -> f32 {
    let delta = min(half_size.y - corner - abs(y), 0.0);
    let curved = half_size.x - corner + sqrt(max(0.0, corner * corner - delta * delta));
    let integral = 0.5 + 0.5 * erf((x + vec2<f32>(-curved, curved)) * (sqrt(0.5) / sigma));
    return integral.y - integral.x;
}

@vertex
fn vs_main(
    @builtin(vertex_index) vertex_id: u32,
    @builtin(instance_index) instance_id: u32
) -> VertexOutput {
    let unit_vertex = vec2<f32>(f32(vertex_id & 1u), 0.5 * f32(vertex_id & 2u));
    var shadow = shadows[instance_id];

    // Expand bounds by blur margin (3 sigma covers 99.7% of Gaussian)
    let margin = 3.0 * shadow.blur_radius;
    shadow.bounds.origin -= vec2<f32>(margin);
    shadow.bounds.size += 2.0 * vec2<f32>(margin);

    let local_position = unit_vertex * shadow.bounds.size + shadow.bounds.origin;
    let position = apply_transform(local_position, shadow.transform);

    var out: VertexOutput;
    out.position = to_device_position(position, globals.viewport_size);
    out.shadow_id = instance_id;
    out.color = hsla_to_rgba(shadow.color);
    out.clip_distances = distance_from_clip_rect(position, shadow.content_mask);

    return out;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    // Clip test
    if (any(input.clip_distances < vec4<f32>(0.0))) {
        return vec4<f32>(0.0);
    }

    let shadow = shadows[input.shadow_id];
    let half_size = shadow.bounds.size / 2.0;
    let center = shadow.bounds.origin + half_size;
    let center_to_point = input.position.xy - center;

    let corner_radius = pick_corner_radius(center_to_point, shadow.corner_radii);

    // Only sample where Gaussian is significant
    let low = center_to_point.y - half_size.y;
    let high = center_to_point.y + half_size.y;
    let start = clamp(-3.0 * shadow.blur_radius, low, high);
    let end = clamp(3.0 * shadow.blur_radius, low, high);

    // Integrate using 4 sample points
    let step = (end - start) / 4.0;
    var y = start + step * 0.5;
    var alpha = 0.0;

    for (var i = 0; i < 4; i += 1) {
        let blur = blur_along_x(
            center_to_point.x,
            center_to_point.y - y,
            shadow.blur_radius,
            corner_radius,
            half_size
        );
        alpha += blur * gaussian(y, shadow.blur_radius) * step;
        y += step;
    }

    var final_color = blend_color(input.color, alpha, globals.premultiplied_alpha);

    // Apply opacity
    final_color = vec4<f32>(final_color.rgb * shadow.opacity, final_color.a * shadow.opacity);

    return final_color;
}
