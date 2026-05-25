struct Globals {
    view_proj: mat4x4<f32>,
    light_view_proj: mat4x4<f32>,
    camera_pos: vec4<f32>,
    key_light_dir: vec4<f32>,
    key_light_color: vec4<f32>,
    fill_light_dir: vec4<f32>,
    fill_light_color: vec4<f32>,
    ambient: vec4<f32>,
    flags: vec4<f32>,
};

struct Instance {
    model: mat4x4<f32>,
    color: vec4<f32>,
};

@group(0) @binding(0) var<uniform> globals: Globals;
@group(1) @binding(0) var<uniform> instance: Instance;

@vertex
fn vs_main(@location(0) position: vec3<f32>) -> @builtin(position) vec4<f32> {
    let world = instance.model * vec4<f32>(position, 1.0);
    return globals.light_view_proj * world;
}
