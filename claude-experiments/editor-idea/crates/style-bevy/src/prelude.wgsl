// style-bevy shader prelude.
//
// User-authored background shaders `#import style_bevy::prelude::{world, proj, theme}`
// to read engine-provided data through these uniform blocks. The
// matching Rust structs (`WorldUniforms` / `ProjectUniforms` /
// `ThemeUniforms` in `shader.rs`) drive the bind group layout via
// `AsBindGroup` — keep field order and types in sync if you edit one.

#define_import_path style_bevy::prelude

struct WorldData {
    time: f32,
    camera_zoom: f32,
    resolution: vec2<f32>,
    mouse_world: vec2<f32>,
    padA: vec2<f32>,
    focused_pane: vec4<f32>,
    // Windex animation: 0..1 progress while a spray-and-wipe is
    // playing, else 0. `windex_active` is 1 while running, else 0.
    // Use these to drive a temporary visual effect; the persistent
    // wipe mask is NOT touched by Windex.
    windex_progress: f32,
    windex_active: u32,
    padB: vec2<f32>,
}

struct ProjectData {
    dust_seconds: f32,
    last_edit_seconds: f32,
    age_seconds: f32,
    padA: f32,
}

struct ThemeData {
    bg: vec4<f32>,
    fg: vec4<f32>,
    fg_muted: vec4<f32>,
    accent: vec4<f32>,
    caret: vec4<f32>,
    selection: vec4<f32>,
    warn: vec4<f32>,
    err: vec4<f32>,
    font_size: f32,
    line_height_ratio: f32,
    // Per-project multiplier on dust output. 0 disables dust on this
    // project; 1 is the default; >1 saturates fast. Set from the
    // theme.rhai token `dust_intensity`.
    dust_intensity: f32,
    padA: f32,
}

@group(#{MATERIAL_BIND_GROUP}) @binding(0) var<uniform> world: WorldData;
@group(#{MATERIAL_BIND_GROUP}) @binding(1) var<uniform> proj: ProjectData;
@group(#{MATERIAL_BIND_GROUP}) @binding(2) var<uniform> theme: ThemeData;
// Per-project wipe mask. R = how much dust has been cleaned at this
// UV. Sample via `textureSample(wipe_mask, wipe_sampler, in.uv).r` and
// use `1.0 - sample` to attenuate dust. The mask is UV-mapped across
// the whole canvas, regardless of window aspect.
@group(#{MATERIAL_BIND_GROUP}) @binding(3) var wipe_mask: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(4) var wipe_sampler: sampler;
