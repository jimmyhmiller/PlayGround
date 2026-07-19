// Render pass for the density heatmap. Reads the tile counts written by the
// density compute pass as *read-only* (non-atomic) storage — WebGPU forbids
// writable/atomic storage in a vertex shader, so this is a separate module from
// density.wgsl even though it binds the same buffers.

struct DParams {
    tiles_x: u32,
    tiles_y: u32,
    tile_px: f32,
    num_nodes: u32,
    vw: f32,
    vh: f32,
    gamma: f32,
    _p: f32,
};

@group(0) @binding(0) var<uniform> dp: DParams;
@group(1) @binding(0) var<storage, read> counts: array<u32>;
@group(1) @binding(1) var<storage, read> maxcount: array<u32>;

struct Out {
    @builtin(position) clip: vec4<f32>,
    @location(0) color: vec4<f32>,
};

fn turbo(t: f32) -> vec3<f32> {
    let x = clamp(t, 0.0, 1.0);
    let r = 0.13572138 + x * (4.61539260 + x * (-42.66032258 + x * (132.13108234 + x * (-152.94239396 + x * 59.28637943))));
    let g = 0.09140261 + x * (2.19418839 + x * (4.84296658 + x * (-14.18503333 + x * (4.27729857 + x * 2.82956604))));
    let b = 0.10667330 + x * (12.64194608 + x * (-60.58204836 + x * (110.36276771 + x * (-89.90310912 + x * 27.34824973))));
    return clamp(vec3<f32>(r, g, b), vec3<f32>(0.0), vec3<f32>(1.0));
}

fn quad_corner(vi: u32) -> vec2<f32> {
    var c = array<vec2<f32>, 6>(
        vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 0.0), vec2<f32>(1.0, 1.0),
        vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 1.0),
    );
    return c[vi];
}

@vertex
fn vs_density(@builtin(vertex_index) vi: u32, @builtin(instance_index) inst: u32) -> Out {
    let tx = inst % dp.tiles_x;
    let ty = inst / dp.tiles_x;
    let c = counts[inst];
    var o: Out;
    if (c == 0u) {
        o.clip = vec4<f32>(2.0, 2.0, 2.0, 1.0); // empty tile -> offscreen
        o.color = vec4<f32>(0.0);
        return o;
    }
    let mx = max(maxcount[0], 1u);
    let t = pow(log(1.0 + f32(c)) / log(1.0 + f32(mx)), dp.gamma);

    let corner = quad_corner(vi);
    let px = (vec2<f32>(f32(tx), f32(ty)) + corner) * dp.tile_px;
    let ndc = vec2<f32>(px.x / dp.vw * 2.0 - 1.0, 1.0 - px.y / dp.vh * 2.0);
    o.clip = vec4<f32>(ndc, 0.0, 1.0);
    o.color = vec4<f32>(turbo(t), clamp(0.12 + 0.9 * t, 0.0, 1.0));
    return o;
}

@fragment
fn fs_density(in: Out) -> @location(0) vec4<f32> {
    return in.color;
}
