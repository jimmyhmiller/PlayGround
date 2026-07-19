// Deterministic, order-preserving visible-edge compaction.
//
// Replaces the old per-vertex cull in vs_edge: a compute pre-pass tests each
// edge ONCE against the exact same NDC inequality, then writes the surviving
// (a, b) node-id pairs — in original edge order — into a compact index buffer
// plus a DrawIndexedIndirect arg block. The render then draws only visible
// edges, indexed, so the post-transform cache collapses repeated endpoints.
//
// Order preservation matters: additive blending on an 8-bit sRGB target rounds
// per blend op, so a nondeterministic (atomic-append) compaction order could
// wiggle dense pixels by an LSB frame-to-frame. The three passes below keep
// the exact original draw order, making the output bit-identical to drawing
// the full edge list:
//   cull_count — each workgroup owns a CHUNK of edges, counts visible ones.
//   cull_scan  — one workgroup exclusive-scans chunk counts into chunk offsets
//                and writes the total into the indirect args.
//   cull_emit  — re-tests visibility and scatters pairs at
//                chunk_offset + in-chunk rank (original order).

struct Camera {
    center: vec2<f32>,
    scale: vec2<f32>,
    viewport: vec2<f32>,
    zoom: f32,
    _pad: f32,
};

struct DrawArgs {
    index_count: u32,
    instance_count: u32,
    first_index: u32,
    base_vertex: i32,
    first_instance: u32,
};

@group(0) @binding(0) var<uniform> cam: Camera;
@group(0) @binding(1) var<storage, read> positions: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> edges: array<u32>;
@group(0) @binding(3) var<storage, read_write> out_indices: array<u32>;
@group(0) @binding(4) var<storage, read_write> draw_args: DrawArgs;
// Pass 1 writes per-chunk visible counts; pass 2 rewrites them in place as
// exclusive offsets (in visible-edge units); pass 3 reads the offsets.
@group(0) @binding(5) var<storage, read_write> chunk_counts: array<u32>;

const WG: u32 = 256u;
const PER_THREAD: u32 = 16u;
const CHUNK: u32 = 4096u; // WG * PER_THREAD

fn num_edges() -> u32 {
    return arrayLength(&edges) / 2u;
}

// EXACT same inequality the old vs_edge used: cull only when both endpoints
// are off the same side of NDC.
fn edge_visible(e: u32) -> bool {
    let a = edges[e * 2u];
    let b = edges[e * 2u + 1u];
    let pa = (positions[a] - cam.center) * cam.scale;
    let pb = (positions[b] - cam.center) * cam.scale;
    return !((pa.x < -1.0 && pb.x < -1.0) || (pa.x > 1.0 && pb.x > 1.0) ||
             (pa.y < -1.0 && pb.y < -1.0) || (pa.y > 1.0 && pb.y > 1.0));
}

var<workgroup> wg_scan: array<u32, 256u>;

// Inclusive Hillis-Steele scan over wg_scan. Callers read wg_scan[li] after.
fn workgroup_inclusive_scan(li: u32) {
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
}

@compute @workgroup_size(256)
fn cull_count(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_index) li: u32,
) {
    let chunk = wg.y * nwg.x + wg.x;
    let n = num_edges();
    let base = chunk * CHUNK + li * PER_THREAD;
    var cnt = 0u;
    for (var i = 0u; i < PER_THREAD; i++) {
        let e = base + i;
        if (e < n && edge_visible(e)) {
            cnt += 1u;
        }
    }
    wg_scan[li] = cnt;
    workgroupBarrier();
    // Tree reduce to wg_scan[0].
    var stride = WG / 2u;
    while (stride > 0u) {
        if (li < stride) {
            wg_scan[li] += wg_scan[li + stride];
        }
        workgroupBarrier();
        stride /= 2u;
    }
    if (li == 0u && chunk < arrayLength(&chunk_counts)) {
        chunk_counts[chunk] = wg_scan[0];
    }
}

@compute @workgroup_size(256)
fn cull_scan(@builtin(local_invocation_index) li: u32) {
    let nchunks = arrayLength(&chunk_counts);
    let per = (nchunks + WG - 1u) / WG;
    let start = li * per;
    // Exclusive scan of my slice in place; keep the slice total.
    var sum = 0u;
    for (var i = 0u; i < per; i++) {
        let idx = start + i;
        if (idx < nchunks) {
            let v = chunk_counts[idx];
            chunk_counts[idx] = sum;
            sum += v;
        }
    }
    wg_scan[li] = sum;
    workgroupBarrier();
    workgroup_inclusive_scan(li);
    // Add the exclusive prefix of the slice totals back onto my slice.
    let excl = wg_scan[li] - sum;
    for (var i = 0u; i < per; i++) {
        let idx = start + i;
        if (idx < nchunks) {
            chunk_counts[idx] += excl;
        }
    }
    if (li == WG - 1u) {
        draw_args.index_count = wg_scan[WG - 1u] * 2u;
        draw_args.instance_count = 1u;
        draw_args.first_index = 0u;
        draw_args.base_vertex = 0;
        draw_args.first_instance = 0u;
    }
}

@compute @workgroup_size(256)
fn cull_emit(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_index) li: u32,
) {
    let chunk = wg.y * nwg.x + wg.x;
    let n = num_edges();
    let base = chunk * CHUNK + li * PER_THREAD;
    var cnt = 0u;
    for (var i = 0u; i < PER_THREAD; i++) {
        let e = base + i;
        if (e < n && edge_visible(e)) {
            cnt += 1u;
        }
    }
    wg_scan[li] = cnt;
    workgroupBarrier();
    workgroup_inclusive_scan(li);
    let my_prefix = wg_scan[li] - cnt; // exclusive prefix within the chunk
    if (chunk >= arrayLength(&chunk_counts)) {
        return;
    }
    var w = chunk_counts[chunk] + my_prefix;
    for (var i = 0u; i < PER_THREAD; i++) {
        let e = base + i;
        if (e < n && edge_visible(e)) {
            out_indices[w * 2u] = edges[e * 2u];
            out_indices[w * 2u + 1u] = edges[e * 2u + 1u];
            w += 1u;
        }
    }
}
