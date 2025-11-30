// Quad rendering shader

struct Quad {
    order: u32,
    border_style: u32,
    bounds: Bounds,
    content_mask: Bounds,
    // NOTE: Padding must be EXPLICIT in storage buffer data!
    // 2 floats padding here for Background alignment (offset 40 -> 48)
    background: Background,
    border_color: Hsla,
    corner_radii: Corners,
    border_widths: Edges,
    transform: Transform,  // 32-byte alignment -> struct rounded to 256 bytes
    opacity: f32,
    pad: u32,
    // 2 floats padding at end for 32-byte struct alignment
    // Total: 64 floats = 256 bytes
}

@group(0) @binding(0) var<uniform> globals: GlobalParams;
@group(0) @binding(1) var<storage, read> quads: array<Quad>;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) @interpolate(flat) quad_id: u32,
    @location(1) @interpolate(flat) background_solid: vec4<f32>,
    @location(2) @interpolate(flat) background_color0: vec4<f32>,
    @location(3) @interpolate(flat) background_color1: vec4<f32>,
    @location(4) @interpolate(flat) border_color: vec4<f32>,
    @location(5) clip_distances: vec4<f32>,
}

@vertex
fn vs_main(
    @builtin(vertex_index) vertex_id: u32,
    @builtin(instance_index) instance_id: u32
) -> VertexOutput {
    // Generate unit quad vertices
    let unit_vertex = vec2<f32>(f32(vertex_id & 1u), 0.5 * f32(vertex_id & 2u));

    let quad = quads[instance_id];

    // Transform to screen space (apply bounds first, then transform)
    let local_position = unit_vertex * quad.bounds.size + quad.bounds.origin;
    let position = apply_transform(local_position, quad.transform);

    // Prepare gradient colors in vertex shader
    let gradient = prepare_gradient_colors(quad.background);

    var out: VertexOutput;
    out.position = to_device_position(position, globals.viewport_size);
    out.quad_id = instance_id;
    out.background_solid = gradient.solid;
    out.background_color0 = gradient.color0;
    out.background_color1 = gradient.color1;
    out.border_color = hsla_to_rgba(quad.border_color);
    out.clip_distances = distance_from_clip_rect(position, quad.content_mask);

    return out;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    // Get quad data from storage buffer
    let quad = quads[input.quad_id];

    // Evaluate background color (solid or gradient)
    let background_color = gradient_color(
        quad.background,
        input.position.xy,
        quad.bounds,
        input.background_solid,
        input.background_color0,
        input.background_color1
    );

    // Apply content mask clipping
    if (input.clip_distances.x < 0.0 || input.clip_distances.y < 0.0 ||
        input.clip_distances.z < 0.0 || input.clip_distances.w < 0.0) {
        discard;
    }

    // Compute SDF for rounded corners and borders
    let half_size = quad.bounds.size / 2.0;
    let center = quad.bounds.origin + half_size;
    let center_to_point = input.position.xy - center;

    let antialias_threshold = 0.5;

    let corner_radius = pick_corner_radius(center_to_point, quad.corner_radii);
    let corner_to_point = abs(center_to_point) - half_size;
    let corner_center_to_point = corner_to_point + corner_radius;

    // Width of nearest borders
    let border = vec2<f32>(
        select(quad.border_widths.right, quad.border_widths.left, center_to_point.x < 0.0),
        select(quad.border_widths.bottom, quad.border_widths.top, center_to_point.y < 0.0)
    );

    // Reduce 0-width borders for antialiasing
    let reduced_border = vec2<f32>(
        select(border.x, -antialias_threshold, border.x == 0.0),
        select(border.y, -antialias_threshold, border.y == 0.0)
    );

    // Outer SDF (quad boundary)
    let outer_sdf = quad_sdf_impl(corner_center_to_point, corner_radius);

    // Inner SDF (border inner edge)
    var inner_sdf = 0.0;

    let straight_border_inner_corner = corner_to_point + reduced_border;
    let is_beyond_inner_straight_border =
        straight_border_inner_corner.x > 0.0 ||
        straight_border_inner_corner.y > 0.0;

    let is_within_inner_straight_border =
        straight_border_inner_corner.x < -antialias_threshold &&
        straight_border_inner_corner.y < -antialias_threshold;

    let is_near_rounded_corner =
        corner_center_to_point.x >= 0.0 &&
        corner_center_to_point.y >= 0.0;

    // Fast path for interior
    if (is_within_inner_straight_border && !is_near_rounded_corner) {
        return blend_color(background_color, 1.0, globals.premultiplied_alpha);
    }

    // Compute inner SDF
    if (corner_center_to_point.x <= 0.0 || corner_center_to_point.y <= 0.0) {
        // Straight border region
        inner_sdf = -max(straight_border_inner_corner.x, straight_border_inner_corner.y);
    } else if (is_beyond_inner_straight_border) {
        inner_sdf = -1.0;
    } else if (reduced_border.x == reduced_border.y) {
        // Circular inner edge
        inner_sdf = -(outer_sdf + reduced_border.x);
    } else {
        // Elliptical inner edge
        let ellipse_radii = max(vec2<f32>(0.0), corner_radius - reduced_border);
        inner_sdf = quarter_ellipse_sdf(corner_center_to_point, ellipse_radii);
    }

    let border_sdf = max(inner_sdf, outer_sdf);

    var color = background_color;

    // Render border
    if (border_sdf < antialias_threshold) {
        var border_alpha = saturate(antialias_threshold - inner_sdf);

        // Apply dashed pattern if border style is dashed
        if (quad.border_style == 1u) {
            // Compute distance along border perimeter
            let abs_point = abs(center_to_point);
            var perimeter_distance: f32;

            // Determine which edge we're on and compute distance
            if (abs_point.x > abs_point.y) {
                // Left or right edge
                perimeter_distance = abs_point.y + select(half_size.y + half_size.x, 0.0, center_to_point.x < 0.0);
            } else {
                // Top or bottom edge
                perimeter_distance = abs_point.x + select(half_size.x, 2.0 * half_size.x + half_size.y, center_to_point.y < 0.0);
            }

            // Create dash pattern (8px dash, 4px gap)
            let dash_length = 8.0;
            let gap_length = 4.0;
            let pattern_length = dash_length + gap_length;
            let pattern_position = perimeter_distance % pattern_length;

            // Smooth transition at dash boundaries
            let dash_alpha = saturate((dash_length - pattern_position) * 2.0);
            border_alpha *= dash_alpha;
        }

        // Blend border over background
        let blended_border = over(background_color, input.border_color);
        color = mix(background_color, blended_border, border_alpha);
    }

    var final_color = blend_color(color, saturate(antialias_threshold - outer_sdf), globals.premultiplied_alpha);

    // Apply opacity
    final_color = vec4<f32>(final_color.rgb * quad.opacity, final_color.a * quad.opacity);

    return final_color;
}
