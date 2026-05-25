// dust.wgsl — default dust + smear shader for the new dynamic
// material. Edit `~/.terminal-bevy/projects/.editor/shaders/dust.wgsl`
// on disk to iterate live. This file is just the bootstrap copy.
//
// The host introspects the `UserData` struct below; whatever fields
// exist here are addressable by scripts via `uniform_set("name", v)`.
// Engine-provided values (time, dt, resolution, mouse_world,
// focused_pane) are auto-populated by the runtime if they appear in
// the struct.
//
// Textures: declare `texture_2d<f32>` globals with whatever names you
// want. Scripts manipulate them by name via `mask_paint("name", ...)`.

#import bevy_sprite::mesh2d_vertex_output::VertexOutput

struct UserData {
    // Engine-populated each frame.
    time: f32,
    dt: f32,
    resolution: vec2<f32>,
    mouse_world: vec2<f32>,
    focused_pane: vec4<f32>,

    // Script-populated.
    dust_seconds: f32,
    dust_intensity: f32,
    windex_progress: f32,
    windex_active: f32,
    fg_muted: vec4<f32>,
}
@group(2) @binding(0) var<uniform> user: UserData;
@group(2) @binding(1) var samp: sampler;
@group(2) @binding(2) var wipe_mask: texture_2d<f32>;

fn hash21(p: vec2<f32>) -> f32 {
    let q = fract(p * vec2(123.34, 456.21));
    let r = q + dot(q, q + 78.233);
    return fract(r.x * r.y);
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    if (user.dust_intensity <= 0.001) {
        return vec4(0.0);
    }
    let px = in.uv * user.resolution;
    let hours = user.dust_seconds / 3600.0;
    let dust = clamp(sqrt(hours / 24.0), 0.0, 1.0) * user.dust_intensity;
    if (dust <= 0.001) {
        return vec4(0.0);
    }

    let n0 = hash21(floor(px * 0.50));
    let n1 = hash21(floor(px * 1.10) + 17.0);
    let n2 = hash21(floor(px * 2.30) + 53.0);
    let grain = n0 * 0.5 + n1 * 0.3 + n2 * 0.2;
    let drift = 0.5 + 0.5 * sin(user.time * 0.02 + px.x * 0.001);

    let wipe = clamp(textureSample(wipe_mask, samp, in.uv).r, 0.0, 1.0);

    var windex_clean = 0.0;
    if (user.windex_active > 0.5) {
        let p = user.windex_progress;
        let ease = p * p;
        let front_x = -220.0 + ease * (user.resolution.x + 440.0);
        let cleared = smoothstep(front_x + 12.0, front_x - 12.0, px.x);
        let fade = 1.0 - smoothstep(0.85, 1.0, p);
        windex_clean = cleared * fade;
    }

    let eff_wipe = clamp(max(wipe, windex_clean), 0.0, 1.0);
    let alpha = dust * grain * 0.55 * drift * (1.0 - eff_wipe);
    return vec4(user.fg_muted.rgb, alpha);
}
