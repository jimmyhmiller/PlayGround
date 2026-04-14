#import bevy_sprite::mesh2d_vertex_output::VertexOutput

const MAX_EXTRA_LIGHTS: u32 = 8u;

struct ExtraLight {
    pos: vec2<f32>,
    dir: vec2<f32>,
    cos_half_angle: f32,
    range: f32,
    intensity: f32,
    _pad: f32,
};

struct ConeLightParams {
    player_pos: vec2<f32>,
    aim_dir: vec2<f32>,
    cos_half_angle: f32,
    range: f32,
    ambient: f32,
    intensity: f32,
    extra_count: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
    extras: array<ExtraLight, 8>,
};

@group(2) @binding(0) var<uniform> params: ConeLightParams;

fn cone_contribution(world_pos: vec2<f32>, light_pos: vec2<f32>, light_dir: vec2<f32>, cos_half: f32, light_range: f32, light_intensity: f32) -> f32 {
    let to_pixel = world_pos - light_pos;
    let dist = length(to_pixel);

    if (dist >= light_range) {
        return 0.0;
    }

    let dir = to_pixel / max(dist, 0.0001);
    let aim = normalize(light_dir);
    let edge = cos_half;
    let angular = smoothstep(edge - 0.004, edge + 0.004, dot(dir, aim));
    let radial = 1.0 - smoothstep(light_range * 0.15, light_range, dist);
    return angular * radial * light_intensity;
}

@fragment
fn fragment(mesh: VertexOutput) -> @location(0) vec4<f32> {
    let world_pos = mesh.world_position.xy;

    // Player cone
    var total_light = cone_contribution(
        world_pos,
        params.player_pos,
        params.aim_dir,
        params.cos_half_angle,
        params.range,
        params.intensity,
    );

    // Extra lights (sentinels etc.)
    let count = min(params.extra_count, MAX_EXTRA_LIGHTS);
    for (var i = 0u; i < count; i = i + 1u) {
        let ex = params.extras[i];
        total_light += cone_contribution(
            world_pos,
            ex.pos,
            ex.dir,
            ex.cos_half_angle,
            ex.range,
            ex.intensity,
        );
    }

    let light = clamp(params.ambient + total_light, 0.0, 1.0);
    let darkness_alpha = 1.0 - light;
    return vec4<f32>(0.0, 0.0, 0.0, darkness_alpha);
}
