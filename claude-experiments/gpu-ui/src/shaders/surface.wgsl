// Surface shader with dynamic per-pixel lighting, shadows, and material properties.
// This is the core visual shader — every surface in the compositor goes through here.

struct GlobalUniforms {
    view_proj: mat4x4<f32>,
    camera_pos: vec4<f32>,
    ambient_light: vec4<f32>,
    lights: array<Light, 8>,
    num_lights: u32,
    time: f32,
    _pad: vec2<f32>,
};

struct Light {
    position: vec4<f32>,
    color_intensity: vec4<f32>,  // rgb + intensity
    radius: vec4<f32>,           // radius, ...
};

struct SurfaceUniforms {
    model: mat4x4<f32>,
    base_color: vec4<f32>,
    emissive: vec4<f32>,       // rgb + strength
    roughness: f32,
    metallic: f32,
    opacity: f32,
    _pad: f32,
};

@group(0) @binding(0)
var<uniform> globals: GlobalUniforms;

@group(1) @binding(0)
var<uniform> surface: SurfaceUniforms;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    let world_pos = surface.model * vec4<f32>(in.position, 1.0);
    out.world_pos = world_pos.xyz;
    out.clip_position = globals.view_proj * world_pos;

    // Transform normal to world space (using the model matrix, ignoring scale for now)
    let world_normal = (surface.model * vec4<f32>(in.normal, 0.0)).xyz;
    out.world_normal = normalize(world_normal);
    out.uv = in.uv;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let N = normalize(in.world_normal);
    let V = normalize(globals.camera_pos.xyz - in.world_pos);

    let base_color = surface.base_color.rgb;
    let roughness = surface.roughness;
    let metallic = surface.metallic;

    // PBR-inspired lighting (simplified Cook-Torrance)
    var lo = vec3<f32>(0.0);

    for (var i = 0u; i < globals.num_lights; i = i + 1u) {
        let light = globals.lights[i];
        let light_pos = light.position.xyz;
        let light_color = light.color_intensity.rgb;
        let light_intensity = light.color_intensity.w;
        let light_radius = light.radius.x;

        let L = light_pos - in.world_pos;
        let distance = length(L);
        let L_norm = L / distance;

        // Attenuation with smooth falloff
        let attenuation = light_intensity / (1.0 + distance * distance / (light_radius * light_radius));

        // Diffuse (Lambertian)
        let NdotL = max(dot(N, L_norm), 0.0);
        let diffuse = base_color * (1.0 - metallic) * NdotL;

        // Specular (Blinn-Phong approximation with roughness)
        let H = normalize(V + L_norm);
        let NdotH = max(dot(N, H), 0.0);
        let spec_power = max(2.0 / (roughness * roughness) - 2.0, 1.0);
        let specular_strength = mix(0.04, 1.0, metallic);
        let specular = vec3<f32>(specular_strength) * pow(NdotH, spec_power);

        lo = lo + (diffuse + specular) * light_color * attenuation;
    }

    // Ambient
    let ambient = globals.ambient_light.rgb * base_color;

    // Emissive
    let emissive = surface.emissive.rgb * surface.emissive.w;

    // Fresnel rim effect — subtle edge glow for that Compiz feel
    let fresnel = pow(1.0 - max(dot(N, V), 0.0), 3.0) * 0.15;
    let rim = vec3<f32>(0.3, 0.5, 0.8) * fresnel;

    let color = ambient + lo + emissive + rim;

    // Simple tone mapping
    let mapped = color / (color + vec3<f32>(1.0));

    return vec4<f32>(mapped, surface.opacity);
}
