struct Globals {
    view_proj: mat4x4<f32>,
    light_view_proj: mat4x4<f32>,
    camera_pos: vec4<f32>,
    key_light_dir: vec4<f32>,        // xyz = direction TO the light (world)
    key_light_color: vec4<f32>,      // rgb * a = irradiance
    fill_light_dir: vec4<f32>,
    fill_light_color: vec4<f32>,
    ambient: vec4<f32>,              // rgb = ambient color * brightness
    flags: vec4<f32>,                // x = tonemap (>0.5), y = gamma (>0.5),
                                     // z = shadow_bias_min, w = shadow_bias_max
};

struct Instance {
    model: mat4x4<f32>,
    color: vec4<f32>,
};

@group(0) @binding(0) var<uniform> globals: Globals;
@group(0) @binding(1) var shadow_map: texture_depth_2d;
@group(0) @binding(2) var shadow_sampler: sampler_comparison;
@group(1) @binding(0) var<uniform> instance: Instance;

struct VsIn {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
};

struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) light_clip: vec4<f32>,
};

@vertex
fn vs_main(in: VsIn) -> VsOut {
    var out: VsOut;
    let world_pos = instance.model * vec4<f32>(in.position, 1.0);
    out.world_pos = world_pos.xyz;
    // Assume model has no non-uniform scale/shear: rotate normal by
    // upper-left 3x3 of model. Translations are stripped because the
    // 4th column doesn't affect a w=0 multiplication.
    let n_world = (instance.model * vec4<f32>(in.normal, 0.0)).xyz;
    out.world_normal = n_world;
    out.clip_pos = globals.view_proj * world_pos;
    out.light_clip = globals.light_view_proj * world_pos;
    return out;
}

fn sample_shadow(light_clip: vec4<f32>, n_dot_l: f32) -> f32 {
    // Perspective divide. The key directional shadow camera is
    // orthographic, so w is 1 and the divide is a no-op, but we do it
    // anyway for correctness.
    let proj = light_clip.xyz / light_clip.w;
    // wgpu NDC: x,y in [-1,1], z in [0,1]. Texture coords need
    // y-flipped because shadow texture origin is top-left.
    let uv = vec2<f32>(proj.x * 0.5 + 0.5, 0.5 - proj.y * 0.5);
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
        return 1.0;
    }
    let bias_max = globals.flags.w;
    let bias_min = globals.flags.z;
    let bias = max(bias_max * (1.0 - n_dot_l), bias_min);
    let ref_depth = proj.z - bias;
    // 3x3 PCF.
    let tex_size = vec2<f32>(textureDimensions(shadow_map));
    let texel = 1.0 / tex_size;
    var sum = 0.0;
    for (var y: i32 = -1; y <= 1; y = y + 1) {
        for (var x: i32 = -1; x <= 1; x = x + 1) {
            let off = vec2<f32>(f32(x), f32(y)) * texel;
            sum = sum + textureSampleCompare(shadow_map, shadow_sampler, uv + off, ref_depth);
        }
    }
    return sum / 9.0;
}

// Approximate sRGB → linear. Cheap pow(c, 2.2); a piecewise-accurate
// version isn't worth the ALU for our flat-shaded look.
fn srgb_to_linear(c: vec3<f32>) -> vec3<f32> {
    return pow(max(c, vec3<f32>(0.0)), vec3<f32>(2.2));
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let n = normalize(in.world_normal);
    // Material colors come in as sRGB constants (matching Bevy's
    // Color::srgb); convert to linear so the sRGB surface's auto-encode
    // doesn't double-apply gamma.
    let base = srgb_to_linear(instance.color.rgb);
    let key_color = srgb_to_linear(globals.key_light_color.rgb);
    let fill_color = srgb_to_linear(globals.fill_light_color.rgb);
    let amb_color = srgb_to_linear(globals.ambient.rgb);

    let key_dir = normalize(globals.key_light_dir.xyz);
    let fill_dir = normalize(globals.fill_light_dir.xyz);

    let n_dot_l_key = max(dot(n, key_dir), 0.0);
    let n_dot_l_fill = max(dot(n, fill_dir), 0.0);

    let shadow = sample_shadow(in.light_clip, n_dot_l_key);

    let key = key_color * globals.key_light_color.a * n_dot_l_key * shadow;
    let fill = fill_color * globals.fill_light_color.a * n_dot_l_fill;
    let amb = amb_color * globals.ambient.a;

    let lit = base * (amb + key + fill);

    var color = lit;
    if (globals.flags.x > 0.5) {
        color = color / (color + vec3<f32>(1.0));
    }
    if (globals.flags.y > 0.5) {
        color = pow(max(color, vec3<f32>(0.0)), vec3<f32>(1.0 / 2.2));
    }
    return vec4<f32>(color, 1.0);
}
