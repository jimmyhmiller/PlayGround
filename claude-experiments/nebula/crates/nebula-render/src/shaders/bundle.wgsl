// Edge bundling, compute side. Every edge, every frame: each edge is clipped
// to the visible world rect and binned by its (source cell, target cell) pair
// on a WORLD-SPACE grid, accumulating count, endpoint centroids, and endpoint
// colors per pair. The draw pass renders one line per occupied pair from
// centroid to centroid with brightness = count * edge_alpha — the same total
// light the individual lines would have deposited, at a fraction of the fill.
//
// The grid lives in world coordinates (origin quantized to whole cells) so
// camera motion never changes which cells an edge belongs to: zooming and
// panning transform stable bundle geometry instead of re-binning it. Level of
// detail comes from two power-of-two cell scales cross-faded by fractional
// zoom (see BParams.weight), so scale transitions are invisible too.
//
// Key property: a pair holding exactly one edge reconstructs that edge's
// endpoints EXACTLY (centroid of one point is the point), so sparse regions —
// and any zoomed-in view — render the true geometry; only visually
// indistinguishable dense flows aggregate.

struct Camera {
    center: vec2<f32>,
    scale: vec2<f32>,
    viewport: vec2<f32>,
    zoom: f32,
    _pad: f32,
};

struct BParams {
    cells_x: u32,
    cells_y: u32,
    num_edges: u32,
    // 0 = color bundles by accumulated endpoint node colors, 1 = fixed tint.
    mode: u32,
    tint: vec4<f32>,
    // World-space grid: corner, cell size (a power-of-two scale), and this
    // level's cross-fade weight.
    origin: vec2<f32>,
    cell: f32,
    weight: f32,
    // World-space clip rect (view + one cell of margin).
    clip_min: vec2<f32>,
    clip_max: vec2<f32>,
    edge_alpha: f32,
    _p0: f32,
    _p1: f32,
    _p2: f32,
};

@group(0) @binding(0) var<uniform> cam: Camera;
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
    let sa = positions[na];
    let sb = positions[nb];

    // Clip to the visible world rect (Liang–Barsky), so edges crossing the
    // view keep their true crossing geometry while fully offscreen edges
    // (which contribute no pixels) are skipped.
    let d = sb - sa;
    var t0 = 0.0;
    var t1 = 1.0;
    for (var axis = 0u; axis < 2u; axis = axis + 1u) {
        var p: f32;
        var q0: f32;
        var q1: f32;
        if (axis == 0u) {
            p = d.x; q0 = sa.x - bp.clip_min.x; q1 = bp.clip_max.x - sa.x;
        } else {
            p = d.y; q0 = sa.y - bp.clip_min.y; q1 = bp.clip_max.y - sa.y;
        }
        if (abs(p) < 1e-9) {
            if (q0 < 0.0 || q1 < 0.0) {
                return; // parallel and outside
            }
        } else {
            let ta = -q0 / p;
            let tb = q1 / p;
            t0 = max(t0, min(ta, tb));
            t1 = min(t1, max(ta, tb));
            if (t0 > t1) {
                return; // fully outside
            }
        }
    }
    let ca = sa + d * t0;
    let cb = sa + d * t1;

    // World-space cell of each (clipped) endpoint.
    let ax = clamp(i32(floor((ca.x - bp.origin.x) / bp.cell)), 0, i32(bp.cells_x) - 1);
    let ay = clamp(i32(floor((ca.y - bp.origin.y) / bp.cell)), 0, i32(bp.cells_y) - 1);
    let bx = clamp(i32(floor((cb.x - bp.origin.x) / bp.cell)), 0, i32(bp.cells_x) - 1);
    let by = clamp(i32(floor((cb.y - bp.origin.y) / bp.cell)), 0, i32(bp.cells_y) - 1);
    let ncells = bp.cells_x * bp.cells_y;
    let ia = u32(ay) * bp.cells_x + u32(ax);
    let ib = u32(by) * bp.cells_x + u32(bx);
    let pair = ia * ncells + ib;

    atomicAdd(&pair_count[pair], 1u);
    // Offsets from cell centers as cell fractions, x128 fixed point (scale
    // independent; per-edge quantization noise averages out over a bundle).
    let ctra = bp.origin + (vec2<f32>(f32(ax), f32(ay)) + 0.5) * bp.cell;
    let ctrb = bp.origin + (vec2<f32>(f32(bx), f32(by)) + 0.5) * bp.cell;
    let oa = (ca - ctra) / bp.cell;
    let ob = (cb - ctrb) / bp.cell;
    atomicAdd(&pair_geom[pair * 4u + 0u], i32(round(oa.x * 128.0)));
    atomicAdd(&pair_geom[pair * 4u + 1u], i32(round(oa.y * 128.0)));
    atomicAdd(&pair_geom[pair * 4u + 2u], i32(round(ob.x * 128.0)));
    atomicAdd(&pair_geom[pair * 4u + 3u], i32(round(ob.y * 128.0)));
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
