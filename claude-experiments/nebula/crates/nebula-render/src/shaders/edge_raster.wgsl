// Tiled compute software rasterizer for edges (runtime-toggleable alternative
// to the hardware LineList path).
//
// The naive per-edge version was bound by global atomic traffic: every DDA
// step did read-modify-writes to device memory. This version bins edges to
// 32x32 screen tiles, then one workgroup per tile accumulates all of its
// edges' pixels in on-chip workgroup memory and touches device memory once
// per pixel — the same reason tiled renderers (and 3DGS splatting) win for
// huge overlapping-primitive counts.
//
// Per batch of edges (batching bounds the pair-buffer size; integer adds
// commute, so batch boundaries cannot change the result):
//   tile_count  — walk each edge's supercover of crossed tiles, histogram
//                 into tile_counts.
//   tile_scan   — exclusive-scan tile_counts into tile_offsets; zeroes
//                 tile_counts so tile_emit can reuse them as cursors.
//   tile_emit   — re-walk, scatter edge ids into per-tile pair lists.
//   tile_raster — one workgroup per tile: load the tile's accum into shared
//                 memory, replay each listed edge's DDA steps that land in
//                 the tile (identical step formula => identical pixels),
//                 accumulate with workgroup atomics, store back, and zero
//                 the tile's counter for the next batch/frame.
//
// Determinism: all accumulation is integer addition, which commutes exactly,
// so pair order, batch split, and GPU scheduling cannot change the output.
// Saturation skipping (exact: sums are monotonic and the display clamps at
// 1.0) happens against the shared-memory copy, so earlier batches' saturation
// is honored.

struct Camera {
    center: vec2<f32>,
    scale: vec2<f32>,
    viewport: vec2<f32>,
    zoom: f32,
    _pad: f32,
};

struct RenderParams {
    base_radius_px: f32,
    min_radius_px: f32,
    max_radius_px: f32,
    size_gamma: f32,
    edge_alpha: f32,
    node_alpha: f32,
    _p0: f32,
    _p1: f32,
};

struct EdgeStyle {
    color: vec4<f32>,
    mode: u32,
    _p0: u32,
    _p1: u32,
    _p2: u32,
};

@group(0) @binding(0) var<uniform> cam: Camera;
@group(0) @binding(1) var<uniform> params: RenderParams;
@group(0) @binding(2) var<storage, read> positions: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read> colors: array<u32>;
@group(0) @binding(4) var<storage, read> edges: array<u32>;
@group(0) @binding(5) var<uniform> edge_style: EdgeStyle;

struct AccumDims {
    w: u32,
    h: u32,
    tiles_x: u32,
    tiles_y: u32,
};
struct Batch {
    start: u32,
    count: u32,
    _p0: u32,
    _p1: u32,
};
@group(1) @binding(0) var<uniform> dims: AccumDims;
@group(1) @binding(1) var<storage, read_write> accum: array<atomic<u32>>;
@group(1) @binding(2) var<storage, read_write> tile_counts: array<atomic<u32>>;
@group(1) @binding(3) var<storage, read_write> tile_offsets: array<u32>;
@group(1) @binding(4) var<storage, read_write> pairs: array<u32>;
@group(1) @binding(5) var<uniform> batch: Batch;

// Fixed-point scale for accumulation (max ~alpha*SCALE ≈ 500 per edge per
// channel; u32 overflows only past ~8M overlapping edges on one pixel).
const SCALE: f32 = 4096.0;
const TILE: u32 = 32u;
const WG: u32 = 256u;

fn unpack_color(c: u32) -> vec4<f32> {
    return vec4<f32>(
        f32(c & 0xffu) / 255.0,
        f32((c >> 8u) & 0xffu) / 255.0,
        f32((c >> 16u) & 0xffu) / 255.0,
        f32((c >> 24u) & 0xffu) / 255.0,
    );
}

fn clip_axis(p: f32, q: f32, t0: ptr<function, f32>, t1: ptr<function, f32>) -> bool {
    if (abs(p) < 1e-12) {
        return q >= 0.0;
    }
    let r = q / p;
    if (p < 0.0) {
        if (r > *t1) { return false; }
        if (r > *t0) { *t0 = r; }
    } else {
        if (r < *t0) { return false; }
        if (r < *t1) { *t1 = r; }
    }
    return true;
}

// Projected + viewport-clipped edge, in pixel space. `n` is the DDA step
// count; every pass derives pixels from the SAME a/seg/n so they agree
// exactly on which pixels an edge lights.
struct ClippedEdge {
    ok: bool,
    a: vec2<f32>,
    seg: vec2<f32>,
    t0: f32,
    t1: f32,
    n: u32,
};

fn clip_edge(e: u32) -> ClippedEdge {
    var out: ClippedEdge;
    out.ok = false;
    if (e * 2u + 1u >= arrayLength(&edges)) {
        return out;
    }
    let ia = edges[e * 2u];
    let ib = edges[e * 2u + 1u];
    let na = (positions[ia] - cam.center) * cam.scale;
    let nb = (positions[ib] - cam.center) * cam.scale;
    let wf = f32(dims.w);
    let hf = f32(dims.h);
    let pa = vec2<f32>((na.x * 0.5 + 0.5) * wf, (0.5 - na.y * 0.5) * hf);
    let pb = vec2<f32>((nb.x * 0.5 + 0.5) * wf, (0.5 - nb.y * 0.5) * hf);
    let d = pb - pa;
    if (max(abs(d.x), abs(d.y)) < 1e-6) {
        return out; // zero-length: hardware rasterizes nothing
    }
    var t0 = 0.0;
    var t1 = 1.0;
    if (!clip_axis(-d.x, pa.x, &t0, &t1)) { return out; }
    if (!clip_axis(d.x, (wf - 0.001) - pa.x, &t0, &t1)) { return out; }
    if (!clip_axis(-d.y, pa.y, &t0, &t1)) { return out; }
    if (!clip_axis(d.y, (hf - 0.001) - pa.y, &t0, &t1)) { return out; }
    if (t0 >= t1) {
        return out;
    }
    out.ok = true;
    out.a = pa + d * t0;
    out.seg = d * (t1 - t0);
    out.t0 = t0;
    out.t1 = t1;
    out.n = max(u32(ceil(max(abs(out.seg.x), abs(out.seg.y)))), 1u);
    return out;
}

fn global_thread(wg: vec3<u32>, nwg: vec3<u32>, li: u32) -> u32 {
    return (wg.y * nwg.x + wg.x) * WG + li;
}

fn tile_of(p: vec2<f32>) -> vec2<u32> {
    let wf = f32(dims.w);
    let hf = f32(dims.h);
    return vec2<u32>(
        u32(clamp(p.x, 0.0, wf - 1.0)) / TILE,
        u32(clamp(p.y, 0.0, hf - 1.0)) / TILE,
    );
}

// ---------------- Pass 1: per-tile histogram ----------------
// Walks the supercover of tiles the clipped segment crosses (one axis step at
// a time => exactly |dtx|+|dty|+1 tiles, a superset of the tiles the pixel
// walk visits — extra tiles just contribute nothing in tile_raster).

@compute @workgroup_size(256)
fn tile_count(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_index) li: u32,
) {
    let i = global_thread(wg, nwg, li);
    if (i >= batch.count) {
        return;
    }
    let ce = clip_edge(batch.start + i);
    if (!ce.ok) {
        return;
    }
    let ta = tile_of(ce.a);
    let tb = tile_of(ce.a + ce.seg);
    var tx = ta.x;
    var ty = ta.y;
    loop {
        atomicAdd(&tile_counts[ty * dims.tiles_x + tx], 1u);
        if (tx == tb.x && ty == tb.y) {
            break;
        }
        // Step one axis toward the endpoint: whichever tile boundary the
        // segment crosses first (compare boundary-crossing parameters).
        var step_x = false;
        if (ty == tb.y) {
            step_x = true;
        } else if (tx != tb.x) {
            let sx = select(-1.0, 1.0, tb.x > tx);
            let sy = select(-1.0, 1.0, tb.y > ty);
            let bx = f32(tx) * f32(TILE) + select(-0.001, f32(TILE), sx > 0.0);
            let by = f32(ty) * f32(TILE) + select(-0.001, f32(TILE), sy > 0.0);
            let ux = (bx - ce.a.x) / ce.seg.x;
            let uy = (by - ce.a.y) / ce.seg.y;
            step_x = ux <= uy;
        }
        if (step_x) {
            tx = u32(i32(tx) + select(-1, 1, tb.x > tx));
        } else {
            ty = u32(i32(ty) + select(-1, 1, tb.y > ty));
        }
    }
}

// ---------------- Pass 2: scan tile counts -> offsets ----------------
// Single workgroup, chunked. Also zeroes tile_counts so tile_emit can reuse
// them as write cursors.

var<workgroup> wg_scan: array<u32, 256u>;

@compute @workgroup_size(256)
fn tile_scan(@builtin(local_invocation_index) li: u32) {
    let ntiles = dims.tiles_x * dims.tiles_y;
    let per = (ntiles + WG - 1u) / WG;
    let start = li * per;
    var sum = 0u;
    for (var i = 0u; i < per; i++) {
        let idx = start + i;
        if (idx < ntiles) {
            let v = atomicLoad(&tile_counts[idx]);
            atomicStore(&tile_counts[idx], 0u);
            tile_offsets[idx] = sum;
            sum += v;
        }
    }
    wg_scan[li] = sum;
    workgroupBarrier();
    for (var s = 1u; s < WG; s = s << 1u) {
        var t = 0u;
        if (li >= s) {
            t = wg_scan[li - s];
        }
        workgroupBarrier();
        if (li >= s) {
            wg_scan[li] = wg_scan[li] + t;
        }
        workgroupBarrier();
    }
    let excl = wg_scan[li] - sum;
    for (var i = 0u; i < per; i++) {
        let idx = start + i;
        if (idx < ntiles) {
            tile_offsets[idx] += excl;
        }
    }
}

// ---------------- Pass 3: scatter (tile, edge) pairs ----------------

@compute @workgroup_size(256)
fn tile_emit(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_index) li: u32,
) {
    let i = global_thread(wg, nwg, li);
    if (i >= batch.count) {
        return;
    }
    let e = batch.start + i;
    let ce = clip_edge(e);
    if (!ce.ok) {
        return;
    }
    let ta = tile_of(ce.a);
    let tb = tile_of(ce.a + ce.seg);
    var tx = ta.x;
    var ty = ta.y;
    loop {
        let tile = ty * dims.tiles_x + tx;
        let slot = atomicAdd(&tile_counts[tile], 1u);
        pairs[tile_offsets[tile] + slot] = e;
        if (tx == tb.x && ty == tb.y) {
            break;
        }
        var step_x = false;
        if (ty == tb.y) {
            step_x = true;
        } else if (tx != tb.x) {
            let sx = select(-1.0, 1.0, tb.x > tx);
            let sy = select(-1.0, 1.0, tb.y > ty);
            let bx = f32(tx) * f32(TILE) + select(-0.001, f32(TILE), sx > 0.0);
            let by = f32(ty) * f32(TILE) + select(-0.001, f32(TILE), sy > 0.0);
            let ux = (bx - ce.a.x) / ce.seg.x;
            let uy = (by - ce.a.y) / ce.seg.y;
            step_x = ux <= uy;
        }
        if (step_x) {
            tx = u32(i32(tx) + select(-1, 1, tb.x > tx));
        } else {
            ty = u32(i32(ty) + select(-1, 1, tb.y > ty));
        }
    }
}

// ---------------- Pass 4: rasterize one tile per workgroup ----------------

const TILE_PX: u32 = 1024u; // TILE * TILE
var<workgroup> tile_px: array<atomic<u32>, 3072u>; // TILE*TILE*3

@compute @workgroup_size(256)
fn tile_raster(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_index) li: u32,
) {
    let tile = wg.y * dims.tiles_x + wg.x;
    let cnt = atomicLoad(&tile_counts[tile]);
    let px0 = wg.x * TILE; // tile origin in pixels
    let py0 = wg.y * TILE;

    // Load this tile's accum into shared memory (skipped for idle tiles).
    // Barriers stay in uniform control flow: work is guarded, barriers aren't.
    if (cnt != 0u) {
        for (var p = li; p < TILE_PX; p += WG) {
            let gx = px0 + (p % TILE);
            let gy = py0 + (p / TILE);
            if (gx < dims.w && gy < dims.h) {
                let g = (gy * dims.w + gx) * 3u;
                let l = p * 3u;
                atomicStore(&tile_px[l], atomicLoad(&accum[g]));
                atomicStore(&tile_px[l + 1u], atomicLoad(&accum[g + 1u]));
                atomicStore(&tile_px[l + 2u], atomicLoad(&accum[g + 2u]));
            }
        }
    }
    workgroupBarrier();

    if (cnt != 0u) {
        let base = tile_offsets[tile];
        let alpha = params.edge_alpha;
        let sat = u32(SCALE);
        let fx0 = f32(px0);
        let fy0 = f32(py0);
        let fx1 = f32(px0 + TILE) - 0.001;
        let fy1 = f32(py0 + TILE) - 0.001;
        for (var pr = li; pr < cnt; pr += WG) {
            let e = pairs[base + pr];
            let ce = clip_edge(e);
            if (!ce.ok) {
                continue;
            }
            // Restrict the replayed DDA to steps that can touch this tile
            // (conservative ±1; each step still bounds-checks exactly).
            var u0 = 0.0;
            var u1 = 1.0;
            if (!clip_axis(-ce.seg.x, ce.a.x - fx0, &u0, &u1)) { continue; }
            if (!clip_axis(ce.seg.x, fx1 - ce.a.x, &u0, &u1)) { continue; }
            if (!clip_axis(-ce.seg.y, ce.a.y - fy0, &u0, &u1)) { continue; }
            if (!clip_axis(ce.seg.y, fy1 - ce.a.y, &u0, &u1)) { continue; }
            if (u0 > u1) {
                continue;
            }
            let ia = edges[e * 2u];
            let ib = edges[e * 2u + 1u];
            var col_a = unpack_color(colors[ia]).rgb;
            var col_b = unpack_color(colors[ib]).rgb;
            if (edge_style.mode == 1u) {
                col_a = edge_style.color.rgb;
                col_b = edge_style.color.rgb;
            }
            let inv_n = 1.0 / f32(ce.n);
            let i_start = u32(max(floor(u0 * f32(ce.n)) - 1.0, 0.0));
            let i_end = min(ce.n, u32(ceil(u1 * f32(ce.n)) + 1.0));
            let wf = f32(dims.w);
            let hf = f32(dims.h);
            for (var i = i_start; i < i_end; i++) {
                let t = f32(i) * inv_n;
                let p = ce.a + ce.seg * t;
                let x = u32(clamp(p.x, 0.0, wf - 1.0));
                let y = u32(clamp(p.y, 0.0, hf - 1.0));
                // Exact ownership check: only this tile adds this pixel.
                if (x / TILE != wg.x || y / TILE != wg.y) {
                    continue;
                }
                let l = ((y - py0) * TILE + (x - px0)) * 3u;
                if (atomicLoad(&tile_px[l]) >= sat &&
                    atomicLoad(&tile_px[l + 1u]) >= sat &&
                    atomicLoad(&tile_px[l + 2u]) >= sat) {
                    continue;
                }
                let col = mix(col_a, col_b, mix(ce.t0, ce.t1, t));
                atomicAdd(&tile_px[l], u32(round(col.r * alpha * SCALE)));
                atomicAdd(&tile_px[l + 1u], u32(round(col.g * alpha * SCALE)));
                atomicAdd(&tile_px[l + 2u], u32(round(col.b * alpha * SCALE)));
            }
        }
    }
    workgroupBarrier();

    if (cnt != 0u) {
        // Flush shared back to global (exclusive tile ownership, plain store)
        // and reset the tile's counter for the next batch/frame.
        for (var p = li; p < TILE_PX; p += WG) {
            let gx = px0 + (p % TILE);
            let gy = py0 + (p / TILE);
            if (gx < dims.w && gy < dims.h) {
                let g = (gy * dims.w + gx) * 3u;
                let l = p * 3u;
                atomicStore(&accum[g], atomicLoad(&tile_px[l]));
                atomicStore(&accum[g + 1u], atomicLoad(&tile_px[l + 1u]));
                atomicStore(&accum[g + 2u], atomicLoad(&tile_px[l + 2u]));
            }
        }
        if (li == 0u) {
            atomicStore(&tile_counts[tile], 0u);
        }
    }
}
