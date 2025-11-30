// Common shader functions and structures

struct GlobalParams {
    viewport_size: vec2<f32>,
    premultiplied_alpha: u32,
    pad: u32,
}

struct Bounds {
    origin: vec2<f32>,
    size: vec2<f32>,
}

struct Corners {
    top_left: f32,
    top_right: f32,
    bottom_right: f32,
    bottom_left: f32,
}

struct Edges {
    top: f32,
    right: f32,
    bottom: f32,
    left: f32,
}

struct Transform {
    m0: f32,  // scale x
    m1: f32,  // shear y
    m2: f32,  // shear x
    m3: f32,  // scale y
    m4: f32,  // translate x
    m5: f32,  // translate y
    pad0: f32,
    pad1: f32,
}

struct Hsla {
    h: f32,
    s: f32,
    l: f32,
    a: f32,
}

struct LinearColorStop {
    color: Hsla,
    percentage: f32,
    @size(12) _pad: array<f32, 3>,  // Explicit padding to 32-byte stride
}

struct Background {
    tag: u32,          // 0=Solid, 1=LinearGradient, 2=Pattern
    color_space: u32,  // 0=sRGB linear, 1=Oklab
    // Inline solid color fields to avoid struct alignment issues
    solid_h: f32,
    solid_s: f32,
    solid_l: f32,
    solid_a: f32,
    gradient_angle: f32,
    _pad0: f32,        // Explicit 1-float padding
    colors: array<LinearColorStop, 2>,
    pad: u32,
    @size(12) _pad: array<f32, 3>,  // Explicit padding to 112 bytes total
}

const M_PI_F: f32 = 3.1415926;

// Apply 2D affine transform to a point
fn apply_transform(point: vec2<f32>, transform: Transform) -> vec2<f32> {
    return vec2<f32>(
        transform.m0 * point.x + transform.m2 * point.y + transform.m4,
        transform.m1 * point.x + transform.m3 * point.y + transform.m5
    );
}

// Transform from pixel coordinates to NDC
fn to_device_position(position: vec2<f32>, viewport_size: vec2<f32>) -> vec4<f32> {
    let device_position = position / viewport_size * vec2<f32>(2.0, -2.0) + vec2<f32>(-1.0, 1.0);
    return vec4<f32>(device_position, 0.0, 1.0);
}

// Compute distances from clipping rectangle edges
fn distance_from_clip_rect(position: vec2<f32>, clip_bounds: Bounds) -> vec4<f32> {
    let tl = position - clip_bounds.origin;
    let br = clip_bounds.origin + clip_bounds.size - position;
    return vec4<f32>(tl.x, br.x, tl.y, br.y);
}

// sRGB to linear conversion
fn srgb_to_linear(srgb: vec3<f32>) -> vec3<f32> {
    let cutoff = srgb < vec3<f32>(0.04045);
    let higher = pow((srgb + vec3<f32>(0.055)) / vec3<f32>(1.055), vec3<f32>(2.4));
    let lower = srgb / vec3<f32>(12.92);
    return select(higher, lower, cutoff);
}

fn linear_to_srgb(linear: vec3<f32>) -> vec3<f32> {
    let cutoff = linear < vec3<f32>(0.0031308);
    let higher = vec3<f32>(1.055) * pow(linear, vec3<f32>(1.0 / 2.4)) - vec3<f32>(0.055);
    let lower = linear * vec3<f32>(12.92);
    return select(higher, lower, cutoff);
}

fn linear_to_srgba(color: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(linear_to_srgb(color.rgb), color.a);
}

fn srgba_to_linear(color: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(srgb_to_linear(color.rgb), color.a);
}

// Linear sRGB to Oklab
fn linear_srgb_to_oklab(color: vec4<f32>) -> vec4<f32> {
    let l = 0.4122214708 * color.r + 0.5363325363 * color.g + 0.0514459929 * color.b;
    let m = 0.2119034982 * color.r + 0.6806995451 * color.g + 0.1073969566 * color.b;
    let s = 0.0883024619 * color.r + 0.2817188376 * color.g + 0.6299787005 * color.b;

    let l_ = pow(l, 1.0 / 3.0);
    let m_ = pow(m, 1.0 / 3.0);
    let s_ = pow(s, 1.0 / 3.0);

    return vec4<f32>(
        0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_,
        1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_,
        0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_,
        color.a
    );
}

// Oklab to linear sRGB
fn oklab_to_linear_srgb(color: vec4<f32>) -> vec4<f32> {
    let l_ = color.r + 0.3963377774 * color.g + 0.2158037573 * color.b;
    let m_ = color.r - 0.1055613458 * color.g - 0.0638541728 * color.b;
    let s_ = color.r - 0.0894841775 * color.g - 1.2914855480 * color.b;

    let l = l_ * l_ * l_;
    let m = m_ * m_ * m_;
    let s = s_ * s_ * s_;

    return vec4<f32>(
        4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
        -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
        -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s,
        color.a
    );
}

// HSLA to linear RGBA conversion
fn hsla_to_rgba(hsla: Hsla) -> vec4<f32> {
    let h = hsla.h * 6.0;
    let s = hsla.s;
    let l = hsla.l;
    let a = hsla.a;

    let c = (1.0 - abs(2.0 * l - 1.0)) * s;
    let x = c * (1.0 - abs(h % 2.0 - 1.0));
    let m = l - c / 2.0;
    var color = vec3<f32>(m);

    if (h >= 0.0 && h < 1.0) {
        color.r += c;
        color.g += x;
    } else if (h >= 1.0 && h < 2.0) {
        color.r += x;
        color.g += c;
    } else if (h >= 2.0 && h < 3.0) {
        color.g += c;
        color.b += x;
    } else if (h >= 3.0 && h < 4.0) {
        color.g += x;
        color.b += c;
    } else if (h >= 4.0 && h < 5.0) {
        color.r += x;
        color.b += c;
    } else {
        color.r += c;
        color.b += x;
    }

    return vec4<f32>(color, a);
}

// Prepare gradient colors in vertex shader
struct GradientColors {
    solid: vec4<f32>,
    color0: vec4<f32>,
    color1: vec4<f32>,
}

fn prepare_gradient_colors(bg: Background) -> GradientColors {
    var result: GradientColors;

    if (bg.tag == 0u) {
        // Solid color
        let solid_hsla = Hsla(bg.solid_h, bg.solid_s, bg.solid_l, bg.solid_a);
        result.solid = hsla_to_rgba(solid_hsla);
    } else if (bg.tag == 1u) {
        // Linear gradient - convert to appropriate color space
        result.color0 = hsla_to_rgba(bg.colors[0].color);
        result.color1 = hsla_to_rgba(bg.colors[1].color);

        if (bg.color_space == 0u) {
            // sRGB space
            result.color0 = linear_to_srgba(result.color0);
            result.color1 = linear_to_srgba(result.color1);
        } else if (bg.color_space == 1u) {
            // Oklab space
            result.color0 = linear_srgb_to_oklab(result.color0);
            result.color1 = linear_srgb_to_oklab(result.color1);
        }
    } else if (bg.tag == 2u) {
        // Pattern - store colors directly
        let solid_hsla = Hsla(bg.solid_h, bg.solid_s, bg.solid_l, bg.solid_a);
        result.color0 = hsla_to_rgba(solid_hsla);
        result.color1 = hsla_to_rgba(bg.colors[0].color);
    } else if (bg.tag == 3u) {
        // Radial gradient - pack center and radius in solid_color
        result.solid = vec4<f32>(bg.solid_h, bg.solid_s, bg.solid_l, bg.solid_a);
        result.color0 = hsla_to_rgba(bg.colors[0].color);
        result.color1 = hsla_to_rgba(bg.colors[1].color);

        if (bg.color_space == 0u) {
            result.color0 = linear_to_srgba(result.color0);
            result.color1 = linear_to_srgba(result.color1);
        } else if (bg.color_space == 1u) {
            result.color0 = linear_srgb_to_oklab(result.color0);
            result.color1 = linear_srgb_to_oklab(result.color1);
        }
    } else if (bg.tag == 4u) {
        // Conic gradient - pack center in solid_color
        result.solid = vec4<f32>(bg.solid_h, bg.solid_s, bg.solid_l, bg.solid_a);
        result.color0 = hsla_to_rgba(bg.colors[0].color);
        result.color1 = hsla_to_rgba(bg.colors[1].color);

        if (bg.color_space == 0u) {
            result.color0 = linear_to_srgba(result.color0);
            result.color1 = linear_to_srgba(result.color1);
        } else if (bg.color_space == 1u) {
            result.color0 = linear_srgb_to_oklab(result.color0);
            result.color1 = linear_srgb_to_oklab(result.color1);
        }
    }

    return result;
}

// Evaluate gradient at fragment position
fn gradient_color(
    bg: Background,
    position: vec2<f32>,
    bounds: Bounds,
    solid_color: vec4<f32>,
    color0: vec4<f32>,
    color1: vec4<f32>
) -> vec4<f32> {
    if (bg.tag == 0u) {
        return solid_color;
    }

    if (bg.tag == 1u) {
        // Linear gradient
        let angle = bg.gradient_angle;
        let radians = (angle % 360.0 - 90.0) * M_PI_F / 180.0;
        var direction = vec2<f32>(cos(radians), sin(radians));

        let stop0_percentage = bg.colors[0].percentage;
        let stop1_percentage = bg.colors[1].percentage;

        // Normalize direction for non-square bounds
        if (bounds.size.x > bounds.size.y) {
            direction.y *= bounds.size.y / bounds.size.x;
        } else {
            direction.x *= bounds.size.x / bounds.size.y;
        }

        // Project position onto gradient axis
        let half_size = bounds.size / 2.0;
        let center = bounds.origin + half_size;
        let center_to_point = position - center;
        var t = dot(center_to_point, direction) / length(direction);

        // Convert to [0, 1] range
        if (abs(direction.x) > abs(direction.y)) {
            t = (t + half_size.x) / bounds.size.x;
        } else {
            t = (t + half_size.y) / bounds.size.y;
        }

        // Adjust for color stops
        t = (t - stop0_percentage) / (stop1_percentage - stop0_percentage);
        t = clamp(t, 0.0, 1.0);

        // Interpolate in chosen color space
        let interpolated = mix(color0, color1, t);

        if (bg.color_space == 0u) {
            return srgba_to_linear(interpolated);
        } else if (bg.color_space == 1u) {
            return oklab_to_linear_srgb(interpolated);
        } else {
            return interpolated;
        }
    }

    if (bg.tag == 2u) {
        let pattern_type = bg.color_space;
        let spacing = bg.colors[0].percentage;
        let local_pos = position - bounds.origin;

        if (pattern_type == 0u) {
            // Diagonal stripe pattern
            let angle = bg.gradient_angle;
            let radians = (angle % 360.0) * M_PI_F / 180.0;

            // Rotate position by pattern angle
            let rotated_x = local_pos.x * cos(radians) + local_pos.y * sin(radians);

            // Create alternating stripes
            let stripe_index = floor(rotated_x / spacing);
            let is_color1 = (i32(stripe_index) % 2) == 0;

            return select(color1, color0, is_color1);
        } else if (pattern_type == 1u) {
            // Dot pattern
            let dot_radius = spacing * 0.35;
            let grid_x = floor(local_pos.x / spacing);
            let grid_y = floor(local_pos.y / spacing);

            // Center of current grid cell
            let cell_center = vec2<f32>(
                (grid_x + 0.5) * spacing,
                (grid_y + 0.5) * spacing
            );

            // Distance from cell center
            let dist = length(local_pos - cell_center);

            // Smooth dot edge
            let dot_alpha = saturate((dot_radius - dist) * 2.0);
            return mix(color0, color1, dot_alpha);
        } else if (pattern_type == 2u) {
            // Checkerboard pattern
            let grid_x = floor(local_pos.x / spacing);
            let grid_y = floor(local_pos.y / spacing);
            let is_color1 = ((i32(grid_x) + i32(grid_y)) % 2) == 0;
            return select(color1, color0, is_color1);
        } else if (pattern_type == 3u) {
            // Grid pattern (lines)
            let line_width = spacing * 0.1; // 10% of spacing for line width
            let mod_x = local_pos.x % spacing;
            let mod_y = local_pos.y % spacing;

            // Check if we're on a grid line (with smooth antialiasing)
            let on_vertical_line = saturate((line_width - mod_x) * 2.0);
            let on_horizontal_line = saturate((line_width - mod_y) * 2.0);
            let grid_alpha = max(on_vertical_line, on_horizontal_line);

            return mix(color0, color1, grid_alpha);
        }
    }

    if (bg.tag == 3u) {
        // Radial gradient
        // Center and radius are packed in solid color (h, s, l, a)
        let center_x = solid_color.r;
        let center_y = solid_color.g;
        let radius = solid_color.b;

        // Compute distance from gradient center (in normalized space 0-1)
        let center = vec2<f32>(center_x, center_y);
        let pos_normalized = (position - bounds.origin) / bounds.size;
        let dist_from_center = length(pos_normalized - center);

        // Normalize by radius
        let t = clamp(dist_from_center / radius, 0.0, 1.0);

        // Apply color stops
        let stop0_percentage = bg.colors[0].percentage;
        let stop1_percentage = bg.colors[1].percentage;
        let adjusted_t = (t - stop0_percentage) / (stop1_percentage - stop0_percentage);
        let clamped_t = clamp(adjusted_t, 0.0, 1.0);

        // Interpolate in chosen color space
        let interpolated = mix(color0, color1, clamped_t);

        if (bg.color_space == 0u) {
            return srgba_to_linear(interpolated);
        } else if (bg.color_space == 1u) {
            return oklab_to_linear_srgb(interpolated);
        } else {
            return interpolated;
        }
    }

    if (bg.tag == 4u) {
        // Conic (angular/sweep) gradient
        // Center is packed in solid color (h, s components)
        let center_x = solid_color.r;
        let center_y = solid_color.g;
        let start_angle = bg.gradient_angle;

        // Compute angle from gradient center (in normalized space 0-1)
        let center = vec2<f32>(center_x, center_y);
        let pos_normalized = (position - bounds.origin) / bounds.size;
        let to_point = pos_normalized - center;

        // Compute angle in degrees (0-360)
        var angle = atan2(to_point.y, to_point.x) * 180.0 / M_PI_F;
        angle = (angle + 360.0) % 360.0; // Normalize to [0, 360)

        // Adjust by start angle
        angle = (angle - start_angle + 360.0) % 360.0;

        // Normalize to [0, 1]
        let t = angle / 360.0;

        // Apply color stops
        let stop0_percentage = bg.colors[0].percentage;
        let stop1_percentage = bg.colors[1].percentage;
        let adjusted_t = (t - stop0_percentage) / (stop1_percentage - stop0_percentage);
        let clamped_t = clamp(adjusted_t, 0.0, 1.0);

        // Interpolate in chosen color space
        let interpolated = mix(color0, color1, clamped_t);

        if (bg.color_space == 0u) {
            return srgba_to_linear(interpolated);
        } else if (bg.color_space == 1u) {
            return oklab_to_linear_srgb(interpolated);
        } else {
            return interpolated;
        }
    }

    return solid_color;
}

// Select corner radius based on quadrant
fn pick_corner_radius(center_to_point: vec2<f32>, radii: Corners) -> f32 {
    if (center_to_point.x < 0.0) {
        if (center_to_point.y < 0.0) {
            return radii.top_left;
        } else {
            return radii.bottom_left;
        }
    } else {
        if (center_to_point.y < 0.0) {
            return radii.top_right;
        } else {
            return radii.bottom_right;
        }
    }
}

// Signed distance field for quad with rounded corners
fn quad_sdf_impl(corner_center_to_point: vec2<f32>, corner_radius: f32) -> f32 {
    if (corner_radius == 0.0) {
        return max(corner_center_to_point.x, corner_center_to_point.y);
    } else {
        let signed_distance_to_inset_quad =
            length(max(vec2<f32>(0.0), corner_center_to_point)) +
            min(0.0, max(corner_center_to_point.x, corner_center_to_point.y));
        return signed_distance_to_inset_quad - corner_radius;
    }
}

fn quad_sdf(point: vec2<f32>, bounds: Bounds, corner_radii: Corners) -> f32 {
    let half_size = bounds.size / 2.0;
    let center = bounds.origin + half_size;
    let center_to_point = point - center;
    let corner_radius = pick_corner_radius(center_to_point, corner_radii);
    let corner_to_point = abs(center_to_point) - half_size;
    let corner_center_to_point = corner_to_point + corner_radius;
    return quad_sdf_impl(corner_center_to_point, corner_radius);
}

// Ellipse SDF approximation for borders with varying widths
fn quarter_ellipse_sdf(point: vec2<f32>, radii: vec2<f32>) -> f32 {
    let circle_vec = point / radii;
    let unit_circle_sdf = length(circle_vec) - 1.0;
    return unit_circle_sdf * (radii.x + radii.y) * -0.5;
}

// Porter-Duff over operator
fn over(below: vec4<f32>, above: vec4<f32>) -> vec4<f32> {
    let alpha = above.a + below.a * (1.0 - above.a);
    let color = (above.rgb * above.a + below.rgb * below.a * (1.0 - above.a)) / alpha;
    return vec4<f32>(color, alpha);
}

// Blend color with premultiplied alpha option
fn blend_color(color: vec4<f32>, alpha_factor: f32, premultiplied: u32) -> vec4<f32> {
    let alpha = color.a * alpha_factor;
    let multiplier = select(1.0, alpha, premultiplied != 0u);
    return vec4<f32>(color.rgb * multiplier, alpha);
}

// Gamma correction functions for text rendering
// Compute perceived brightness using REC. 601 coefficients
fn color_brightness(color: vec3<f32>) -> f32 {
    return dot(color, vec3<f32>(0.30, 0.59, 0.11));
}

// Compute enhanced contrast factor for light-on-dark text
fn light_on_dark_contrast(enhanced_contrast: f32, color: vec3<f32>) -> f32 {
    let brightness = color_brightness(color);
    let multiplier = saturate(4.0 * (0.75 - brightness));
    return enhanced_contrast * multiplier;
}

// Apply contrast enhancement to alpha
fn enhance_contrast(alpha: f32, k: f32) -> f32 {
    return alpha * (k + 1.0) / (alpha * k + 1.0);
}

// Apply alpha correction based on background brightness
fn apply_alpha_correction(a: f32, b: f32, g: vec4<f32>) -> f32 {
    let brightness_adjustment = g.x * b + g.y;
    let correction = brightness_adjustment * a + (g.z * b + g.w);
    return a + a * (1.0 - a) * correction;
}

// Apply both contrast and gamma correction to monochrome samples
fn apply_contrast_and_gamma_correction(
    sample: f32,
    color: vec3<f32>,
    enhanced_contrast_factor: f32,
    gamma_ratios: vec4<f32>
) -> f32 {
    let enhanced_contrast = light_on_dark_contrast(enhanced_contrast_factor, color);
    let brightness = color_brightness(color);
    let contrasted = enhance_contrast(sample, enhanced_contrast);
    return apply_alpha_correction(contrasted, brightness, gamma_ratios);
}
