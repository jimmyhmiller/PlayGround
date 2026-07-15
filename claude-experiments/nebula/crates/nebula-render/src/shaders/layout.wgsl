// GPU force-directed layout (Fruchterman–Reingold) with a uniform spatial grid
// for O(N) approximate repulsion. The passes each simulation step:
//
//   clear_grid  -> zero per-cell counters
//   count_grid  -> count how many nodes fall in each cell
//   scan_cells / scan_sums0 / scan_sums1 / add_sums0 / add_cells
//               -> exclusive prefix sum of the counts, giving each cell the
//                  offset of its run inside node_order
//   init_cursor -> per-cell insert cursor, starting at the run's offset
//   scatter_nodes
//               -> counting-sort node ids into node_order, grouped by cell
//   build_pyr_l0 / reduce_pyr (per level)
//               -> build the center-of-mass pyramid from the fine grid counts
//   forces      -> read positions (read-only), accumulate repulsion+attraction+
//                  gravity, write new velocities
//   integrate   -> advance positions from velocities
//
// Splitting force computation (reads positions) from integration (writes
// positions) makes the whole step race-free without double-buffering: within
// `forces` every invocation only reads positions and writes its own velocity.
//
// WHY THE SORT. `forces` is latency-bound, not bandwidth-bound: its gathers are
// only a few hundred MB per step but each one is a cache miss, and it stalls on
// them one at a time. Iterating nodes in *cell order* (thread g handles
// node_order[g], not node g) means a workgroup's threads sit in neighbouring
// cells, so they hit the same grid runs and the same pyramid entries and the
// cache absorbs the gathers. It computes exactly the same forces for exactly the
// same nodes — only which thread handles which node changes.
//
// The sort also replaces a fixed-capacity `grid_items` (dim^2 * cap entries,
// mostly empty) with an exact run per cell in an N-entry array.

struct Params {
    num_nodes: u32,
    grid_dim: u32,
    grid_cap: u32,
    num_levels: u32,
    world_size: f32,
    k: f32,
    repulsion: f32,
    attraction: f32,
    gravity: f32,
    damping: f32,
    dt: f32,
    max_speed: f32,
    // Global cooling: forces are scaled by alpha, which decays toward 0 so the
    // simulation converges and stops (d3-force style). Padded to 16 bytes.
    alpha: f32,
    _p1: f32,
    _p2: f32,
    _p3: f32,
    // COM pyramid level table: x = cell offset into pyr, y = dim.
    // Level 0 mirrors the fine grid; each level halves (power-of-two aligned).
    levels: array<vec4<u32>, 12>,
};

@group(0) @binding(0) var<uniform> params: Params;

@group(1) @binding(0) var<storage, read_write> positions: array<vec2<f32>>;
@group(1) @binding(1) var<storage, read_write> velocities: array<vec2<f32>>;
@group(1) @binding(2) var<storage, read> csr_offsets: array<u32>;
@group(1) @binding(3) var<storage, read> csr_targets: array<u32>;
@group(1) @binding(4) var<storage, read_write> grid_counts: array<atomic<u32>>;
// Node ids counting-sorted by cell: cell c owns the run
// node_order[cell_starts[c] .. cell_starts[c] + grid_counts[c]].
@group(1) @binding(5) var<storage, read_write> node_order: array<u32>;
// Exclusive prefix sum of grid_counts — where each cell's run begins.
@group(1) @binding(7) var<storage, read_write> cell_starts: array<u32>;
// Scratch for the scatter: each cell's next free slot, walked up from its start.
@group(1) @binding(8) var<storage, read_write> cell_cursor: array<atomic<u32>>;
// Per-block totals for the multi-level scan. Two levels packed back to back:
// [0, nblocks0) holds one total per 256-cell block, and [nblocks0, ...) holds one
// total per 256-block group. Two levels reach 256^3 = 16.7M cells, above any
// grid_dim we allow.
@group(1) @binding(9) var<storage, read_write> scan_sums: array<u32>;
// Center-of-mass pyramid for far-field repulsion (all levels packed).
//
// One 8-byte entry per cell: .x = center of mass, .y = mass (a node count).
// The far-field walk is memory-bound — it reads ~27 cells per level per node,
// scattered — so the entry is packed to keep that traffic down:
//
//   * COM is stored *cell-relative*, as two 16-bit fixed-point fractions of the
//     cell's own extent. A COM always lies inside its cell, so [0,1] per axis is
//     the exact range, and the quantization is cell_size/65535 — at the coarsest
//     level (4x4 of a ~100k world) that is sub-world-unit against a cell tens of
//     thousands of units across. Absolute f32 would spend 8 bytes to describe a
//     point we already know the neighborhood of.
//   * Mass is an exact u32 count (level 0 counts nodes; reduce sums children), so
//     it needs no float at all.
//
// Together: 12 bytes in two buffers -> 8 bytes in one, halving both the bytes and
// the number of scattered loads per cell.
@group(1) @binding(6) var<storage, read_write> pyr: array<vec2<u32>>;

// COM of a cell whose mass is spread evenly: the cell center. Level 0 uses this.
const COM_CELL_CENTER: u32 = 0x80008000u; // (0.5, 0.5) in 16-bit fixed point

// Decode a packed COM back to world space. Callers pass the cell's own (cx, cy)
// and side length, which they already have — recomputing them from a linear cell
// index would cost an integer divide per cell read.
fn pyr_com_at(cx: f32, cy: f32, cs: f32, packed: u32) -> vec2<f32> {
    let half = params.world_size * 0.5;
    let fx = f32(packed & 0xffffu) * (1.0 / 65535.0);
    let fy = f32(packed >> 16u) * (1.0 / 65535.0);
    return vec2<f32>((cx + fx) * cs - half, (cy + fy) * cs - half);
}

// Encode a world-space COM as a fraction of cell (cx, cy). The clamp guards only
// against float rounding at the cell boundary; the COM is inside the cell by
// construction.
fn pyr_pack_com(cx: f32, cy: f32, cs: f32, com: vec2<f32>) -> u32 {
    let half = params.world_size * 0.5;
    let fx = clamp((com.x + half) / cs - cx, 0.0, 1.0);
    let fy = clamp((com.y + half) / cs - cy, 0.0, 1.0);
    return u32(round(fx * 65535.0)) | (u32(round(fy * 65535.0)) << 16u);
}

// Which pyramid level a reduce_pyr pass writes (bound with a dynamic offset).
struct ReduceLevel {
    dst: u32,
};
@group(2) @binding(0) var<uniform> reduce_level: ReduceLevel;

// Linearize a (possibly 2D/3D) dispatch into a single global index. We dispatch
// in 2D once the workgroup count exceeds the 65535 per-dimension limit, so we
// cannot rely on global_invocation_id.x alone.
fn linear_index(wid: vec3<u32>, nwg: vec3<u32>, lidx: u32) -> u32 {
    let group = wid.x + wid.y * nwg.x + wid.z * nwg.x * nwg.y;
    return group * 256u + lidx;
}

fn cell_coord(p: vec2<f32>) -> vec2<i32> {
    let half = params.world_size * 0.5;
    let cs = params.world_size / f32(params.grid_dim);
    let dim = i32(params.grid_dim);
    let cx = clamp(i32(floor((p.x + half) / cs)), 0, dim - 1);
    let cy = clamp(i32(floor((p.y + half) / cs)), 0, dim - 1);
    return vec2<i32>(cx, cy);
}

@compute @workgroup_size(256)
fn clear_grid(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_index) lidx: u32,
) {
    let i = linear_index(wid, nwg, lidx);
    let cells = params.grid_dim * params.grid_dim;
    if (i >= cells) {
        return;
    }
    atomicStore(&grid_counts[i], 0u);
}

@compute @workgroup_size(256)
fn count_grid(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_index) lidx: u32,
) {
    let i = linear_index(wid, nwg, lidx);
    if (i >= params.num_nodes) {
        return;
    }
    let c = cell_coord(positions[i]);
    let cell = u32(c.y) * params.grid_dim + u32(c.x);
    atomicAdd(&grid_counts[cell], 1u);
}

// --- Multi-level exclusive prefix sum over the per-cell counts ---------------
// Standard three-stage scan: scan each 256-wide block locally, scan the block
// totals, then fold the scanned totals back in. `cells` can reach 4.19M, so the
// block totals need scanning too — hence two levels of sums.

fn num_blocks(n: u32) -> u32 {
    return (n + 255u) / 256u;
}

var<workgroup> scan_tmp: array<u32, 256>;

// Hillis–Steele scan across the workgroup. Every invocation must call this (the
// barriers require uniform control flow), so out-of-range threads pass v = 0.
// Returns (exclusive prefix for this lane, total for the whole block).
fn block_scan(lidx: u32, v: u32) -> vec2<u32> {
    scan_tmp[lidx] = v;
    workgroupBarrier();
    for (var off = 1u; off < 256u; off = off * 2u) {
        var add = 0u;
        if (lidx >= off) {
            add = scan_tmp[lidx - off];
        }
        workgroupBarrier(); // all reads land before any write
        scan_tmp[lidx] = scan_tmp[lidx] + add;
        workgroupBarrier(); // all writes land before the next round reads
    }
    let inclusive = scan_tmp[lidx];
    let total = scan_tmp[255];
    return vec2<u32>(inclusive - v, total);
}

// Level 0: scan the cell counts into cell_starts, emitting one total per block.
@compute @workgroup_size(256)
fn scan_cells(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_index) lidx: u32,
) {
    let cells = params.grid_dim * params.grid_dim;
    let i = linear_index(wid, nwg, lidx);
    var v = 0u;
    if (i < cells) {
        v = atomicLoad(&grid_counts[i]);
    }
    let r = block_scan(lidx, v);
    if (i < cells) {
        cell_starts[i] = r.x;
    }
    // The dispatch is tiled into 2D and can overshoot, so bound the sums write.
    let block = wid.x + wid.y * nwg.x + wid.z * nwg.x * nwg.y;
    if (lidx == 0u && block < num_blocks(cells)) {
        scan_sums[block] = r.y;
    }
}

// Level 1: scan those block totals in place, emitting one total per 256 blocks
// into the second region of scan_sums.
@compute @workgroup_size(256)
fn scan_sums0(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_index) lidx: u32,
) {
    let n0 = num_blocks(params.grid_dim * params.grid_dim);
    let i = linear_index(wid, nwg, lidx);
    var v = 0u;
    if (i < n0) {
        v = scan_sums[i];
    }
    let r = block_scan(lidx, v);
    if (i < n0) {
        scan_sums[i] = r.x;
    }
    let block = wid.x + wid.y * nwg.x + wid.z * nwg.x * nwg.y;
    if (lidx == 0u && block < num_blocks(n0)) {
        scan_sums[n0 + block] = r.y;
    }
}

// Level 2: scan the second region in place. It is at most 256 entries (256^3
// cells), so one workgroup finishes it and its grand total is just N.
@compute @workgroup_size(256)
fn scan_sums1(@builtin(local_invocation_index) lidx: u32) {
    let n0 = num_blocks(params.grid_dim * params.grid_dim);
    let n1 = num_blocks(n0);
    var v = 0u;
    if (lidx < n1) {
        v = scan_sums[n0 + lidx];
    }
    let r = block_scan(lidx, v);
    if (lidx < n1) {
        scan_sums[n0 + lidx] = r.x;
    }
}

// Fold level 2 back into level 1.
@compute @workgroup_size(256)
fn add_sums0(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_index) lidx: u32,
) {
    let n0 = num_blocks(params.grid_dim * params.grid_dim);
    let i = linear_index(wid, nwg, lidx);
    if (i >= n0) {
        return;
    }
    scan_sums[i] = scan_sums[i] + scan_sums[n0 + i / 256u];
}

// Fold level 1 back into the per-cell offsets. cell_starts is now the exclusive
// prefix sum of grid_counts across the whole grid.
@compute @workgroup_size(256)
fn add_cells(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_index) lidx: u32,
) {
    let cells = params.grid_dim * params.grid_dim;
    let i = linear_index(wid, nwg, lidx);
    if (i >= cells) {
        return;
    }
    cell_starts[i] = cell_starts[i] + scan_sums[i / 256u];
}

@compute @workgroup_size(256)
fn init_cursor(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_index) lidx: u32,
) {
    let i = linear_index(wid, nwg, lidx);
    let cells = params.grid_dim * params.grid_dim;
    if (i >= cells) {
        return;
    }
    atomicStore(&cell_cursor[i], cell_starts[i]);
}

// Place every node id into its cell's run. Order within a run is whatever the
// atomics decide, which is fine: the run is a set, and the near field sums it.
@compute @workgroup_size(256)
fn scatter_nodes(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_index) lidx: u32,
) {
    let i = linear_index(wid, nwg, lidx);
    if (i >= params.num_nodes) {
        return;
    }
    let c = cell_coord(positions[i]);
    let cell = u32(c.y) * params.grid_dim + u32(c.x);
    node_order[atomicAdd(&cell_cursor[cell], 1u)] = i;
}

// Pyramid level 0: one cell per fine-grid cell, mass = node count, COM = cell
// center (an approximation good enough for cells the near-field doesn't cover).
@compute @workgroup_size(256)
fn build_pyr_l0(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_index) lidx: u32,
) {
    let ci = linear_index(wid, nwg, lidx);
    let cells = params.grid_dim * params.grid_dim;
    if (ci >= cells) {
        return;
    }
    pyr[ci] = vec2<u32>(COM_CELL_CENTER, atomicLoad(&grid_counts[ci]));
}

// Reduce pyramid level dst-1 into level dst: each destination cell is the
// mass-weighted merge of its 2x2 children (dims are powers of two, so the
// mapping is exact).
@compute @workgroup_size(256)
fn reduce_pyr(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_index) lidx: u32,
) {
    let ci = linear_index(wid, nwg, lidx);
    let dst = reduce_level.dst;
    let doff = params.levels[dst].x;
    let ddim = params.levels[dst].y;
    if (ci >= ddim * ddim) {
        return;
    }
    let soff = params.levels[dst - 1u].x;
    let sdim = params.levels[dst - 1u].y;
    let scs = params.world_size / f32(sdim);
    let dcs = params.world_size / f32(ddim);
    let dx = ci % ddim;
    let dy = ci / ddim;
    var sum = vec2<f32>(0.0, 0.0);
    var mass = 0u;
    for (var sy = dy * 2u; sy < dy * 2u + 2u; sy = sy + 1u) {
        for (var sx = dx * 2u; sx < dx * 2u + 2u; sx = sx + 1u) {
            let child = pyr[soff + sy * sdim + sx];
            let m = child.y;
            if (m == 0u) {
                continue; // empty: its packed COM is meaningless
            }
            sum = sum + pyr_com_at(f32(sx), f32(sy), scs, child.x) * f32(m);
            mass = mass + m;
        }
    }
    // An empty parent's COM is never read (the far-field walk skips mass == 0),
    // so leave it at the cell origin rather than inventing a position.
    var packed = 0u;
    if (mass > 0u) {
        packed = pyr_pack_com(f32(dx), f32(dy), dcs, sum / f32(mass));
    }
    pyr[doff + ci] = vec2<u32>(packed, mass);
}

@compute @workgroup_size(256)
fn forces(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_index) lidx: u32,
) {
    let g = linear_index(wid, nwg, lidx);
    if (g >= params.num_nodes) {
        return;
    }
    // Walk nodes in cell order rather than index order — see WHY THE SORT above.
    // Same nodes, same forces; only the thread that computes each one changes.
    let i = node_order[g];
    let pi = positions[i];
    let k = params.k;
    let k2 = k * k;
    var force = vec2<f32>(0.0, 0.0);

    // --- Repulsion from nodes in the 3x3 neighborhood of cells ---------------
    // A cell may hold more nodes than `grid_cap`, in which case we only stored a
    // sample. We still know the *true* count, so we scale the sampled repulsion
    // by (true_count / sampled_count): a dense cell then repels with its full
    // mass instead of saturating, which is what stops the whole graph from
    // collapsing into a point.
    let c = cell_coord(pi);
    let dim = i32(params.grid_dim);
    for (var dy = -1; dy <= 1; dy = dy + 1) {
        for (var dx = -1; dx <= 1; dx = dx + 1) {
            let nx = c.x + dx;
            let ny = c.y + dy;
            if (nx < 0 || ny < 0 || nx >= dim || ny >= dim) {
                continue;
            }
            let cell = u32(ny) * params.grid_dim + u32(nx);
            let true_cnt = atomicLoad(&grid_counts[cell]);
            // The run is exact, but still only sample up to grid_cap of it: that
            // bounds the worst case when a transient pile-up puts thousands of
            // nodes in one cell. Occupancy averages ~1, so this almost never
            // binds — and raising the cap is now free, since node_order costs N
            // entries whatever the cap is.
            let cnt = min(true_cnt, params.grid_cap);
            let base = cell_starts[cell];
            var cell_rep = vec2<f32>(0.0, 0.0);
            for (var s = 0u; s < cnt; s = s + 1u) {
                let j = node_order[base + s];
                if (j == i) {
                    continue;
                }
                var delta = pi - positions[j];
                var d2 = dot(delta, delta);
                if (d2 < 0.0001) {
                    // Coincident: deterministic tiny nudge based on index parity.
                    delta = vec2<f32>(f32((i & 1u)) * 2.0 - 1.0, f32((i & 2u)) - 1.0);
                    d2 = 0.0001;
                }
                // FR repulsion: magnitude k^2/dist, direction +delta -> delta*k^2/d2.
                cell_rep = cell_rep + delta * (k2 / d2);
            }
            if (cnt > 0u) {
                force = force + cell_rep * (params.repulsion * f32(true_cnt) / f32(cnt));
            }
        }
    }

    // --- Far-field repulsion from the COM pyramid ----------------------------
    // Long-range global repulsion (what actually flattens sheets and separates
    // clusters), via fast-multipole-style interaction lists: at each level the
    // node interacts with the cells inside its parent's 3x3 neighborhood that
    // are NOT inside its own 3x3 neighborhood (those are refined one level
    // finer; level 0's own 3x3 is the exact near-field above). The coarsest
    // level covers the whole world outside its own 3x3. Every region of space
    // is thus counted exactly once, in ~27 cells per level.
    let half_w = params.world_size * 0.5;
    for (var lvl = 0u; lvl < params.num_levels; lvl = lvl + 1u) {
        let off = params.levels[lvl].x;
        let ldim = i32(params.levels[lvl].y);
        let lcs = params.world_size / f32(ldim);
        let cx = clamp(i32(floor((pi.x + half_w) / lcs)), 0, ldim - 1);
        let cy = clamp(i32(floor((pi.y + half_w) / lcs)), 0, ldim - 1);
        var x0: i32;
        var x1: i32;
        var y0: i32;
        var y1: i32;
        if (lvl == params.num_levels - 1u) {
            // Coarsest level: everything (outside the own 3x3).
            x0 = 0; x1 = ldim - 1; y0 = 0; y1 = ldim - 1;
        } else {
            // Parent's 3x3 neighborhood, expressed in this level's cells.
            let px = cx / 2;
            let py = cy / 2;
            x0 = 2 * (px - 1); x1 = 2 * (px + 1) + 1;
            y0 = 2 * (py - 1); y1 = 2 * (py + 1) + 1;
        }
        for (var y = max(y0, 0); y <= min(y1, ldim - 1); y = y + 1) {
            for (var x = max(x0, 0); x <= min(x1, ldim - 1); x = x + 1) {
                if (abs(x - cx) <= 1 && abs(y - cy) <= 1) {
                    continue; // covered one level finer (or by the near-field)
                }
                let entry = pyr[off + u32(y) * u32(ldim) + u32(x)];
                let mass = entry.y;
                if (mass == 0u) {
                    continue;
                }
                var delta = pi - pyr_com_at(f32(x), f32(y), lcs, entry.x);
                var d2 = dot(delta, delta);
                if (d2 < 1.0) {
                    d2 = 1.0;
                }
                force = force + delta * (params.repulsion * f32(mass) * k2 / d2);
            }
        }
    }

    // --- Attraction along edges (CSR neighbors) ------------------------------
    let start = csr_offsets[i];
    let end = csr_offsets[i + 1u];
    for (var e = start; e < end; e = e + 1u) {
        let j = csr_targets[e];
        let delta = pi - positions[j];
        let dist = max(length(delta), 0.001);
        // FR attraction: magnitude dist^2/k toward neighbor -> -delta*dist/k.
        force = force - delta * (params.attraction * dist / k);
    }

    // --- Gravity toward origin ----------------------------------------------
    force = force - pi * params.gravity;

    // --- Integrate velocity (positions advanced in a later pass) -------------
    // Scale the force by alpha: as alpha decays to ~0 the graph freezes in place.
    var vel = velocities[i] * params.damping + force * (params.dt * params.alpha);
    let sp = length(vel);
    if (sp > params.max_speed) {
        vel = vel * (params.max_speed / sp);
    }
    velocities[i] = vel;
}

@compute @workgroup_size(256)
fn integrate(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_index) lidx: u32,
) {
    let i = linear_index(wid, nwg, lidx);
    if (i >= params.num_nodes) {
        return;
    }
    positions[i] = positions[i] + velocities[i] * params.dt;
}
