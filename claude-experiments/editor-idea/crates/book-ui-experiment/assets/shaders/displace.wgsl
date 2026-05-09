#import bevy_pbr::{
    mesh_functions,
    forward_io::{Vertex, VertexOutput},
    view_transformations::position_world_to_clip,
}

@group(3) @binding(0) var heightmap_tex: texture_2d<f32>;
@group(3) @binding(1) var heightmap_samp: sampler;
@group(3) @binding(2) var layer_id_tex: texture_2d<u32>;

const N_PALETTE: u32 = 8u;

struct PaperParams {
    paper_size: vec2<f32>,
    height_scale: f32,
    _pad0: f32,
    inv_resolution: vec2<f32>,
    _pad1: vec2<f32>,
    light_dir: vec3<f32>,
    _pad2: f32,
    palette: array<vec4<f32>, N_PALETTE>,
};

@group(3) @binding(3) var<uniform> hm: PaperParams;

fn sample_height(uv: vec2<f32>) -> f32 {
    let h = textureSampleLevel(heightmap_tex, heightmap_samp, uv, 0.0).r;
    return h * hm.height_scale;
}

fn displaced_normal(uv: vec2<f32>) -> vec3<f32> {
    let du = vec2<f32>(hm.inv_resolution.x, 0.0);
    let dv = vec2<f32>(0.0, hm.inv_resolution.y);
    let hl = sample_height(uv - du);
    let hr = sample_height(uv + du);
    let hd = sample_height(uv - dv);
    let hu = sample_height(uv + dv);
    let world_dx = hm.paper_size.x * hm.inv_resolution.x * 2.0;
    let world_dy = hm.paper_size.y * hm.inv_resolution.y * 2.0;
    let dz_dx = (hr - hl) / world_dx;
    let dz_dy = (hu - hd) / world_dy;
    return normalize(vec3<f32>(-dz_dx, -dz_dy, 1.0));
}

@vertex
fn vertex(vertex: Vertex) -> VertexOutput {
    var out: VertexOutput;
    let world_from_local = mesh_functions::get_world_from_local(vertex.instance_index);

    let displaced_local = vec3<f32>(
        vertex.position.x,
        vertex.position.y,
        sample_height(vertex.uv),
    );
    out.world_position = mesh_functions::mesh_position_local_to_world(
        world_from_local,
        vec4<f32>(displaced_local, 1.0),
    );
    out.position = position_world_to_clip(out.world_position.xyz);

#ifdef VERTEX_NORMALS
    let local_normal = displaced_normal(vertex.uv);
    out.world_normal = mesh_functions::mesh_normal_local_to_world(
        local_normal,
        vertex.instance_index,
    );
#endif

#ifdef VERTEX_UVS_A
    out.uv = vertex.uv;
#endif

#ifdef VERTEX_OUTPUT_INSTANCE_INDEX
    out.instance_index = vertex.instance_index;
#endif
    return out;
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    let dims = vec2<f32>(textureDimensions(layer_id_tex));
    let pix = vec2<i32>(clamp(in.uv, vec2<f32>(0.0), vec2<f32>(0.999)) * dims);
    let id = textureLoad(layer_id_tex, pix, 0).r;
    let base = hm.palette[id].rgb;

    let n = normalize(in.world_normal);
    let l = normalize(-hm.light_dir);
    let key = max(dot(n, l), 0.0);
    let ambient = 0.32;
    let shade = ambient + key * 0.75;
    return vec4<f32>(base * shade, 1.0);
}
