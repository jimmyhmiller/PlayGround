#import bevy_sprite::mesh2d_vertex_output::VertexOutput

struct ConeLightParams {
    player_pos: vec2<f32>,
    aim_dir: vec2<f32>,
    cos_half_angle: f32,
    range: f32,
    ambient: f32,
    intensity: f32,
};

@group(2) @binding(0) var<uniform> params: ConeLightParams;

@fragment
fn fragment(mesh: VertexOutput) -> @location(0) vec4<f32> {
    let world_pos = mesh.world_position.xy;
    let to_pixel = world_pos - params.player_pos;
    let dist = length(to_pixel);

    var cone: f32 = 0.0;
    if (dist < params.range) {
        let dir = to_pixel / max(dist, 0.0001);
        let aim = normalize(params.aim_dir);
        let dot_val = dot(dir, aim);
        if (dot_val > params.cos_half_angle) {
            // Soft edge on the angular boundary.
            let edge = params.cos_half_angle;
            let angular = smoothstep(edge, mix(edge, 1.0, 0.25), dot_val);
            // Radial falloff — brightest near the player, fading to range.
            let radial = 1.0 - smoothstep(0.0, params.range, dist);
            cone = angular * radial * params.intensity;
        }
    }

    let light = clamp(params.ambient + cone, 0.0, 1.0);
    let darkness_alpha = 1.0 - light;
    return vec4<f32>(0.0, 0.0, 0.0, darkness_alpha);
}
