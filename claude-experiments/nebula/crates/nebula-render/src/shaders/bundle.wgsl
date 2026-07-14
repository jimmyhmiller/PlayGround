// Edge bundling, compute side. Every edge, every frame, is binned by its
// (source cell, target cell) pair on a grid FIXED IN WORLD SPACE covering the
// whole layout, accumulating count, endpoint centroids, and endpoint colors
// per pair. The draw pass renders one line per occupied pair from centroid to
// centroid with brightness = count * edge_alpha — the same total light the
// individual lines would have deposited, at a fraction of the fill.
//
// Nothing here depends on the camera: the binning is a pure function of node
// positions. Zooming and panning therefore transform stable bundle geometry
// exactly like they transform nodes — camera motion cannot change the picture
// beyond the transform itself.
//
// Key property: a pair holding exactly one edge reconstructs that edge's
// endpoints EXACTLY (centroid of one point is the point), so sparse regions
// render true geometry; only visually indistinguishable dense flows aggregate.

struct BParams {
    cells_x: u32,
    cells_y: u32,
    num_edges: u32,
    // 0 = color bundles by accumulated endpoint node colors, 1 = fixed tint.
    mode: u32,
    tint: vec4<f32>,
    // World-space grid: corner and cell size, fixed for the graph's lifetime.
    origin: vec2<f32>,
    cell: f32,
    edge_alpha: f32,
};

@group(0) @binding(1) var<uniform> bp: BParams;

@group(1) @binding(0) var<storage, read> positions: array<vec2<f32>>;
@group(1) @binding(1) var<storage, read> colors: array<u32>;
@group(1) @binding(2) var<storage, read> edges: array<u32>;
// Per pair: edge count.
@group(1) @binding(3) var<storage, read_write> pair_count: array<atomic<u32>>;
// Per pair: 4 sums — endpoint offsets from their cell centers as cell
// fractions, x128 fixed point (a.x, a.y, b.x, b.y).
@group(1) @binding(4) var<storage, read_write> pair_geom: array<atomic<i32>>;
// Per pair: 6 sums — endpoint colors (ra,ga,ba, rb,gb,bb), 0..255 each.
@group(1) @binding(5) var<storage, read_write> pair_col: array<atomic<u32>>;
// Per pair: 2 sums — squared endpoint offsets |o|^2 (cell^2 units, x64 fixed),
// giving each end's spread so bundles draw as bands of the true width.
@group(1) @binding(6) var<storage, read_write> pair_spread: array<atomic<u32>>;

fn linear_index(wid: vec3<u32>, nwg: vec3<u32>, lidx: u32) -> u32 {
    let group = wid.x + wid.y * nwg.x + wid.z * nwg.x * nwg.y;
    return group * 256u + lidx;
}

@compute @workgroup_size(256)
fn clear_pairs(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_index) lidx: u32,
) {
    let i = linear_index(wid, nwg, lidx);
    let pairs = bp.cells_x * bp.cells_y * bp.cells_x * bp.cells_y;
    if (i >= pairs) {
        return;
    }
    atomicStore(&pair_count[i], 0u);
    for (var k = 0u; k < 4u; k = k + 1u) {
        atomicStore(&pair_geom[i * 4u + k], 0);
    }
    for (var k = 0u; k < 6u; k = k + 1u) {
        atomicStore(&pair_col[i * 6u + k], 0u);
    }
    atomicStore(&pair_spread[i * 2u], 0u);
    atomicStore(&pair_spread[i * 2u + 1u], 0u);
}

@compute @workgroup_size(256)
fn bin_edges(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_index) lidx: u32,
) {
    let e = linear_index(wid, nwg, lidx);
    if (e >= bp.num_edges) {
        return;
    }
    let na = edges[e * 2u];
    let nb = edges[e * 2u + 1u];
    let pa = positions[na];
    let pb = positions[nb];

    // World-space cell of each endpoint.
    let ax = clamp(i32(floor((pa.x - bp.origin.x) / bp.cell)), 0, i32(bp.cells_x) - 1);
    let ay = clamp(i32(floor((pa.y - bp.origin.y) / bp.cell)), 0, i32(bp.cells_y) - 1);
    let bx = clamp(i32(floor((pb.x - bp.origin.x) / bp.cell)), 0, i32(bp.cells_x) - 1);
    let by = clamp(i32(floor((pb.y - bp.origin.y) / bp.cell)), 0, i32(bp.cells_y) - 1);
    let ncells = bp.cells_x * bp.cells_y;
    let ia = u32(ay) * bp.cells_x + u32(ax);
    let ib = u32(by) * bp.cells_x + u32(bx);
    let pair = ia * ncells + ib;

    atomicAdd(&pair_count[pair], 1u);
    // Offsets from cell centers as cell fractions, x128 fixed point. Clamped
    // to +-2 cells so stray positions outside the world grid (their cell got
    // clamped above) can't overflow the accumulators.
    let ctra = bp.origin + (vec2<f32>(f32(ax), f32(ay)) + 0.5) * bp.cell;
    let ctrb = bp.origin + (vec2<f32>(f32(bx), f32(by)) + 0.5) * bp.cell;
    let oa = clamp((pa - ctra) / bp.cell, vec2<f32>(-2.0, -2.0), vec2<f32>(2.0, 2.0));
    let ob = clamp((pb - ctrb) / bp.cell, vec2<f32>(-2.0, -2.0), vec2<f32>(2.0, 2.0));
    atomicAdd(&pair_geom[pair * 4u + 0u], i32(round(oa.x * 128.0)));
    atomicAdd(&pair_geom[pair * 4u + 1u], i32(round(oa.y * 128.0)));
    atomicAdd(&pair_geom[pair * 4u + 2u], i32(round(ob.x * 128.0)));
    atomicAdd(&pair_geom[pair * 4u + 3u], i32(round(ob.y * 128.0)));
    atomicAdd(&pair_spread[pair * 2u], u32(round(dot(oa, oa) * 64.0)));
    atomicAdd(&pair_spread[pair * 2u + 1u], u32(round(dot(ob, ob) * 64.0)));
    if (bp.mode == 0u) {
        let colA = colors[na];
        let colB = colors[nb];
        atomicAdd(&pair_col[pair * 6u + 0u], colA & 0xffu);
        atomicAdd(&pair_col[pair * 6u + 1u], (colA >> 8u) & 0xffu);
        atomicAdd(&pair_col[pair * 6u + 2u], (colA >> 16u) & 0xffu);
        atomicAdd(&pair_col[pair * 6u + 3u], colB & 0xffu);
        atomicAdd(&pair_col[pair * 6u + 4u], (colB >> 8u) & 0xffu);
        atomicAdd(&pair_col[pair * 6u + 5u], (colB >> 16u) & 0xffu);
    }
}
