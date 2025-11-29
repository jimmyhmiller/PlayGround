// Sprite rendering shaders (monochrome and polychrome)

struct AtlasTile {
    texture_index: u32,
    texture_kind: u32,
    tile_id: u32,
    padding: u32,
    bounds: Bounds,
}

struct MonochromeSprite {
    order: u32,
    pad: u32,
    bounds: Bounds,
    content_mask: Bounds,
    color: Hsla,
    tile: AtlasTile,
    transform: Transform,
}

struct PolychromeSprite {
    order: u32,
    pad: u32,
    grayscale: u32,
    opacity: f32,
    bounds: Bounds,
    content_mask: Bounds,
    corner_radii: Corners,
    tile: AtlasTile,
    transform: Transform,
}

@group(0) @binding(0) var<uniform> globals: GlobalParams;
@group(0) @binding(1) var<uniform> gamma_ratios: vec4<f32>;
@group(0) @binding(2) var<uniform> grayscale_enhanced_contrast: f32;
@group(0) @binding(3) var<storage, read> sprites: array<MonochromeSprite>;
@group(0) @binding(4) var t_atlas: texture_2d<f32>;
@group(0) @binding(5) var s_atlas: sampler;

struct MonoVertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) @interpolate(flat) sprite_id: u32,
    @location(1) @interpolate(flat) color: vec4<f32>,
    @location(2) tile_uv: vec2<f32>,
    @location(3) clip_distances: vec4<f32>,
}

@vertex
fn vs_mono(
    @builtin(vertex_index) vertex_id: u32,
    @builtin(instance_index) instance_id: u32
) -> MonoVertexOutput {
    let unit_vertex = vec2<f32>(f32(vertex_id & 1u), 0.5 * f32(vertex_id & 2u));
    let sprite = sprites[instance_id];

    let local_position = unit_vertex * sprite.bounds.size + sprite.bounds.origin;
    let position = apply_transform(local_position, sprite.transform);

    // Compute texture coordinates
    let atlas_size = vec2<f32>(textureDimensions(t_atlas, 0));
    let tile_origin = vec2<f32>(sprite.tile.bounds.origin.x, sprite.tile.bounds.origin.y);
    let tile_size = vec2<f32>(sprite.tile.bounds.size.x, sprite.tile.bounds.size.y);
    let tile_uv = (tile_origin + unit_vertex * tile_size) / atlas_size;

    var out: MonoVertexOutput;
    out.position = to_device_position(position, globals.viewport_size);
    out.sprite_id = instance_id;
    out.color = hsla_to_rgba(sprite.color);
    out.tile_uv = tile_uv;
    out.clip_distances = distance_from_clip_rect(position, sprite.content_mask);

    return out;
}

@fragment
fn fs_mono(input: MonoVertexOutput) -> @location(0) vec4<f32> {
    // Sample atlas texture (must happen before branching for uniform control flow)
    let sample = textureSample(t_atlas, s_atlas, input.tile_uv).r;

    // Apply gamma correction for proper text rendering
    let alpha_corrected = apply_contrast_and_gamma_correction(
        sample,
        input.color.rgb,
        grayscale_enhanced_contrast,
        gamma_ratios
    );

    // Apply color tint
    var color = blend_color(input.color, alpha_corrected, globals.premultiplied_alpha);

    // Clip test (after sampling)
    if (any(input.clip_distances < vec4<f32>(0.0))) {
        color = vec4<f32>(0.0);
    }

    return color;
}

// Polychrome sprite shader (separate storage binding)
@group(0) @binding(3) var<storage, read> poly_sprites: array<PolychromeSprite>;

struct PolyVertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) @interpolate(flat) sprite_id: u32,
    @location(1) @interpolate(flat) grayscale: u32,
    @location(2) @interpolate(flat) opacity: f32,
    @location(3) tile_uv: vec2<f32>,
    @location(4) clip_distances: vec4<f32>,
    @location(5) local_pos: vec2<f32>,
}

@vertex
fn vs_poly(
    @builtin(vertex_index) vertex_id: u32,
    @builtin(instance_index) instance_id: u32
) -> PolyVertexOutput {
    let unit_vertex = vec2<f32>(f32(vertex_id & 1u), 0.5 * f32(vertex_id & 2u));
    let sprite = poly_sprites[instance_id];

    let local_position = unit_vertex * sprite.bounds.size + sprite.bounds.origin;
    let position = apply_transform(local_position, sprite.transform);

    // Compute texture coordinates
    let atlas_size = vec2<f32>(textureDimensions(t_atlas, 0));
    let tile_origin = vec2<f32>(sprite.tile.bounds.origin.x, sprite.tile.bounds.origin.y);
    let tile_size = vec2<f32>(sprite.tile.bounds.size.x, sprite.tile.bounds.size.y);
    let tile_uv = (tile_origin + unit_vertex * tile_size) / atlas_size;

    var out: PolyVertexOutput;
    out.position = to_device_position(position, globals.viewport_size);
    out.sprite_id = instance_id;
    out.grayscale = sprite.grayscale;
    out.opacity = sprite.opacity;
    out.tile_uv = tile_uv;
    out.clip_distances = distance_from_clip_rect(position, sprite.content_mask);
    out.local_pos = unit_vertex * sprite.bounds.size;

    return out;
}

@fragment
fn fs_poly(input: PolyVertexOutput) -> @location(0) vec4<f32> {
    let sprite = poly_sprites[input.sprite_id];

    // Sample atlas texture (must happen before branching for uniform control flow)
    var color = textureSample(t_atlas, s_atlas, input.tile_uv);

    // Optional grayscale conversion (REC. 601 luminance)
    if (input.grayscale != 0u) {
        let luminance = dot(color.rgb, vec3<f32>(0.299, 0.587, 0.114));
        color = vec4<f32>(vec3<f32>(luminance), color.a);
    }

    // Apply opacity
    color.a *= input.opacity;

    // Optional rounded corner clipping
    let has_corners = sprite.corner_radii.top_left > 0.0 ||
                     sprite.corner_radii.top_right > 0.0 ||
                     sprite.corner_radii.bottom_left > 0.0 ||
                     sprite.corner_radii.bottom_right > 0.0;

    if (has_corners) {
        let sdf = quad_sdf(input.local_pos,
            Bounds(vec2<f32>(0.0), sprite.bounds.size),
            sprite.corner_radii);
        let alpha_mask = saturate(0.5 - sdf);
        color.a *= alpha_mask;
    }

    var result = blend_color(color, 1.0, globals.premultiplied_alpha);

    // Clip test (after sampling)
    if (any(input.clip_distances < vec4<f32>(0.0))) {
        result = vec4<f32>(0.0);
    }

    return result;
}
