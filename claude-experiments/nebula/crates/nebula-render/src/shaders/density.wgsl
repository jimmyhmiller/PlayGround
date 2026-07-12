// Screen-space density aggregation (level-of-detail for huge graphs). Instead of
// drawing N node quads, one O(N) compute pass bins every node into a screen tile
// (atomic add), and the render pass draws one quad per *tile* colored by count.
// Rendering is then O(tiles), independent of N — the way to "view" a billion-node
// graph without rasterizing a billion primitives.

struct Camera {
    center: vec2<f32>,
    scale: vec2<f32>,
    viewport: vec2<f32>,
    zoom: f32,
    _pad: f32,
};

struct DParams {
    tiles_x: u32,
    tiles_y: u32,
    tile_px: f32,
    num_nodes: u32,
    vw: f32,
    vh: f32,
    gamma: f32,   // intensity exponent
    _p: f32,
};

@group(0) @binding(0) var<uniform> cam: Camera;
@group(0) @binding(1) var<uniform> dp: DParams;

@group(1) @binding(0) var<storage, read> positions: array<vec2<f32>>;
@group(1) @binding(1) var<storage, read_write> counts: array<atomic<u32>>;
@group(1) @binding(2) var<storage, read_write> maxcount: array<atomic<u32>>;

fn linear_index(wid: vec3<u32>, nwg: vec3<u32>, lidx: u32) -> u32 {
    let group = wid.x + wid.y * nwg.x + wid.z * nwg.x * nwg.y;
    return group * 256u + lidx;
}

@compute @workgroup_size(256)
fn clear_density(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_index) lidx: u32,
) {
    let i = linear_index(wid, nwg, lidx);
    let total = dp.tiles_x * dp.tiles_y;
    if (i < total) {
        atomicStore(&counts[i], 0u);
    }
    if (i == 0u) {
        atomicStore(&maxcount[0], 0u);
    }
}

@compute @workgroup_size(256)
fn accumulate_density(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_index) lidx: u32,
) {
    let i = linear_index(wid, nwg, lidx);
    if (i >= dp.num_nodes) {
        return;
    }
    let world = positions[i];
    let ndc = (world - cam.center) * cam.scale;
    // NDC -> screen pixels (origin top-left, y down).
    let sx = (ndc.x * 0.5 + 0.5) * dp.vw;
    let sy = (0.5 - ndc.y * 0.5) * dp.vh;
    if (sx < 0.0 || sy < 0.0 || sx >= dp.vw || sy >= dp.vh) {
        return;
    }
    let tx = u32(sx / dp.tile_px);
    let ty = u32(sy / dp.tile_px);
    if (tx >= dp.tiles_x || ty >= dp.tiles_y) {
        return;
    }
    let cell = ty * dp.tiles_x + tx;
    let prev = atomicAdd(&counts[cell], 1u);
    atomicMax(&maxcount[0], prev + 1u);
}
