// Reflective floor for the project-prism overview. A horizontal plane
// laid OVER the mirrored prism: where it's transparent the reflection
// shows; where it's opaque you see the floor's own color. The opacity
// ramps up with distance from the prism base, so the reflection fades out
// instead of reading as a hard mirror — a wet/shiny colored floor.
//
// params.xyz = floor base color, params.w = reflection strength (0..1).

#import bevy_pbr::forward_io::VertexOutput

@group(#{MATERIAL_BIND_GROUP}) @binding(0) var<uniform> params: vec4<f32>;

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    // Distance from the prism axis, in the floor plane.
    let d = length(in.world_position.xz);

    // 0 right under the prism, 1 far out.
    let fade = smoothstep(0.5, 14.0, d);

    let strength = clamp(params.w, 0.0, 1.0);
    // Near the base: partly transparent so the (dimmed) reflection shows.
    // Stronger `strength` => more transparent => brighter reflection.
    let near_alpha = 0.78 - 0.5 * strength;
    let far_alpha = 0.97;
    let a = mix(near_alpha, far_alpha, fade);

    // The floor color darkens toward the edges for depth, with a faint
    // brightening right at the base (a soft sheen where it meets the cube).
    let sheen = (1.0 - smoothstep(0.0, 3.0, d)) * 0.5;
    let col = params.xyz * mix(1.0, 0.35, fade) + params.xyz * sheen;

    return vec4<f32>(col, a);
}
