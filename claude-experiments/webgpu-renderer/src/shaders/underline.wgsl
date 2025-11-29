// Underline rendering shader (straight and wavy)

struct Underline {
    order: u32,
    pad: u32,
    bounds: Bounds,
    content_mask: Bounds,
    color: Hsla,
    thickness: f32,
    wavy: u32,
    transform: Transform,
    opacity: f32,
    pad2: u32,
}

@group(0) @binding(0) var<uniform> globals: GlobalParams;
@group(0) @binding(1) var<storage, read> underlines: array<Underline>;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) @interpolate(flat) underline_id: u32,
    @location(1) @interpolate(flat) color: vec4<f32>,
    @location(2) st: vec2<f32>,
    @location(3) clip_distances: vec4<f32>,
}

@vertex
fn vs_main(
    @builtin(vertex_index) vertex_id: u32,
    @builtin(instance_index) instance_id: u32
) -> VertexOutput {
    let unit_vertex = vec2<f32>(f32(vertex_id & 1u), 0.5 * f32(vertex_id & 2u));
    let underline = underlines[instance_id];

    let local_position = unit_vertex * underline.bounds.size + underline.bounds.origin;
    let position = apply_transform(local_position, underline.transform);

    var out: VertexOutput;
    out.position = to_device_position(position, globals.viewport_size);
    out.underline_id = instance_id;
    out.color = hsla_to_rgba(underline.color);
    out.st = unit_vertex; // Normalized [0,1] coordinates
    out.clip_distances = distance_from_clip_rect(position, underline.content_mask);

    return out;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    // Clip test
    if (any(input.clip_distances < vec4<f32>(0.0))) {
        return vec4<f32>(0.0);
    }

    let underline = underlines[input.underline_id];

    if (underline.wavy != 0u) {
        // Wavy underline
        const WAVE_FREQUENCY = 2.0;  // Waves per thickness
        const WAVE_HEIGHT_RATIO = 0.8;

        let half_thickness = underline.thickness / 2.0;
        let wave_amplitude = half_thickness * WAVE_HEIGHT_RATIO;
        let wave_freq = M_PI_F * WAVE_FREQUENCY / underline.thickness;

        // Compute wave at this x position
        let x = input.st.x * underline.bounds.size.x;
        let y = (input.st.y - 0.5) * underline.bounds.size.y;

        let wave = sin(x * wave_freq) * wave_amplitude;
        let dWave = cos(x * wave_freq) * wave_amplitude * wave_freq;

        // Distance from fragment to wave curve
        let distance = abs(y - wave) / sqrt(1.0 + dWave * dWave);

        let alpha = saturate(0.5 + half_thickness - distance);
        var final_color = blend_color(input.color, alpha, globals.premultiplied_alpha);
        final_color = vec4<f32>(final_color.rgb * underline.opacity, final_color.a * underline.opacity);
        return final_color;
    } else {
        // Straight underline
        let y = (input.st.y - 0.5) * underline.bounds.size.y;
        let distance = abs(y);
        let half_thickness = underline.thickness / 2.0;

        let alpha = saturate(0.5 + half_thickness - distance);
        var final_color = blend_color(input.color, alpha, globals.premultiplied_alpha);
        final_color = vec4<f32>(final_color.rgb * underline.opacity, final_color.a * underline.opacity);
        return final_color;
    }
}
