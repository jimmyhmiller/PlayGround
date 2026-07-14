// GPU force-directed layout (Fruchterman–Reingold) with a uniform spatial grid
// for O(N) approximate repulsion. The passes each simulation step:
//
//   clear_grid  -> zero per-cell counters
//   build_grid  -> bucket each node into its cell (atomic append, capacity-clamped)
//   build_pyr_l0 / reduce_pyr (per level)
//               -> build the center-of-mass pyramid from the fine grid counts
//   forces      -> read positions (read-only), accumulate repulsion+attraction+
//                  gravity, write new velocities
//   integrate   -> advance positions from velocities
//
// Splitting force computation (reads positions) from integration (writes
// positions) makes the whole step race-free without double-buffering: within
// `forces` every invocation only reads positions and writes its own velocity.

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
    // COM pyramid level table: x = cell offset into pyr_com/pyr_mass, y = dim.
    // Level 0 mirrors the fine grid; each level halves (power-of-two aligned).
    levels: array<vec4<u32>, 12>,
};

@group(0) @binding(0) var<uniform> params: Params;

@group(1) @binding(0) var<storage, read_write> positions: array<vec2<f32>>;
@group(1) @binding(1) var<storage, read_write> velocities: array<vec2<f32>>;
@group(1) @binding(2) var<storage, read> csr_offsets: array<u32>;
@group(1) @binding(3) var<storage, read> csr_targets: array<u32>;
@group(1) @binding(4) var<storage, read_write> grid_counts: array<atomic<u32>>;
@group(1) @binding(5) var<storage, read_write> grid_items: array<u32>;
// Center-of-mass pyramid for far-field repulsion (all levels packed).
@group(1) @binding(6) var<storage, read_write> pyr_com: array<vec2<f32>>;
@group(1) @binding(7) var<storage, read_write> pyr_mass: array<f32>;

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
fn build_grid(
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
    let slot = atomicAdd(&grid_counts[cell], 1u);
    if (slot < params.grid_cap) {
        grid_items[cell * params.grid_cap + slot] = i;
    }
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
    let cs = params.world_size / f32(params.grid_dim);
    let half = params.world_size * 0.5;
    let cx = ci % params.grid_dim;
    let cy = ci / params.grid_dim;
    let cnt = f32(atomicLoad(&grid_counts[ci]));
    pyr_com[ci] = vec2<f32>(
        (f32(cx) + 0.5) * cs - half,
        (f32(cy) + 0.5) * cs - half,
    );
    pyr_mass[ci] = cnt;
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
    let dx = ci % ddim;
    let dy = ci / ddim;
    var sum = vec2<f32>(0.0, 0.0);
    var mass = 0.0;
    for (var sy = dy * 2u; sy < dy * 2u + 2u; sy = sy + 1u) {
        for (var sx = dx * 2u; sx < dx * 2u + 2u; sx = sx + 1u) {
            let m = pyr_mass[soff + sy * sdim + sx];
            sum = sum + pyr_com[soff + sy * sdim + sx] * m;
            mass = mass + m;
        }
    }
    if (mass > 0.0) {
        pyr_com[doff + ci] = sum / mass;
    } else {
        pyr_com[doff + ci] = vec2<f32>(0.0, 0.0);
    }
    pyr_mass[doff + ci] = mass;
}

@compute @workgroup_size(256)
fn forces(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_index) lidx: u32,
) {
    let i = linear_index(wid, nwg, lidx);
    if (i >= params.num_nodes) {
        return;
    }
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
            let cnt = min(true_cnt, params.grid_cap);
            let base = cell * params.grid_cap;
            var cell_rep = vec2<f32>(0.0, 0.0);
            for (var s = 0u; s < cnt; s = s + 1u) {
                let j = grid_items[base + s];
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
                let cell = off + u32(y) * u32(ldim) + u32(x);
                let mass = pyr_mass[cell];
                if (mass <= 0.0) {
                    continue;
                }
                var delta = pi - pyr_com[cell];
                var d2 = dot(delta, delta);
                if (d2 < 1.0) {
                    d2 = 1.0;
                }
                force = force + delta * (params.repulsion * mass * k2 / d2);
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
