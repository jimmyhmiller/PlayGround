// Edge bundling, draw side (separate module because the vertex stage may only
// bind these buffers read-only — same split as density.wgsl/density_render).
// One instance per (source cell, target cell) pair; empty pairs collapse to a
// degenerate offscreen line. Occupied pairs draw centroid->centroid with
// alpha = count * edge_alpha (additively blended, capped), so a bundle carries
// the same light its member edges would have.

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
    mode: u32,
    tint: vec4<f32>,
    origin: vec2<f32>,
    cell: f32,
    edge_alpha: f32,
};

@group(0) @binding(0) var<uniform> cam: Camera;
@group(0) @binding(1) var<uniform> bp: BParams;

@group(1) @binding(0) var<storage, read> pair_count: array<u32>;
@group(1) @binding(1) var<storage, read> pair_geom: array<i32>;
@group(1) @binding(2) var<storage, read> pair_col: array<u32>;
@group(1) @binding(3) var<storage, read> pair_spread: array<u32>;

struct BundleVsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) color: vec4<f32>,
};

// Centroid (world) of one end of a pair.
fn end_world(pair: u32, end: u32, n: f32) -> vec2<f32> {
    let ncells = bp.cells_x * bp.cells_y;
    let idx = select(pair / ncells, pair % ncells, end == 1u);
    let ctr = bp.origin
        + (vec2<f32>(f32(idx % bp.cells_x), f32(idx / bp.cells_x)) + 0.5) * bp.cell;
    let g = pair * 4u + end * 2u;
    let inv = bp.cell / (128.0 * n);
    return ctr + vec2<f32>(f32(pair_geom[g]) * inv, f32(pair_geom[g + 1u]) * inv);
}

// RMS spread of one end around its centroid, in world units.
fn end_spread(pair: u32, end: u32, n: f32) -> f32 {
    let g = pair * 4u + end * 2u;
    let mean = vec2<f32>(f32(pair_geom[g]), f32(pair_geom[g + 1u])) / (128.0 * n);
    let mean_sq = f32(pair_spread[pair * 2u + end]) / (64.0 * n);
    let var_cells = max(mean_sq - dot(mean, mean), 0.0);
    return sqrt(var_cells) * bp.cell;
}

// A bundle draws as a quad (triangle strip): the line between the two end
// centroids, widened at each end by that end's spread, with per-pixel alpha
// scaled down by the width — the bundle's light lands over the same band the
// individual edges would have covered instead of concentrating into 1px.
@vertex
fn vs_bundle(
    @builtin(vertex_index) vi: u32,
    @builtin(instance_index) pair: u32,
) -> BundleVsOut {
    var out: BundleVsOut;
    let n = pair_count[pair];
    if (n == 0u) {
        out.clip = vec4<f32>(2.0, 2.0, 2.0, 1.0);
        out.color = vec4<f32>(0.0);
        return out;
    }
    let nf = f32(n);
    let end = vi / 2u; // 0,0,1,1 across the strip
    let side = f32(vi & 1u) * 2.0 - 1.0; // -1,+1,-1,+1

    let wa = end_world(pair, 0u, nf);
    let wb = end_world(pair, 1u, nf);
    // Screen-space geometry (pixels).
    let to_px = 0.5 * cam.viewport * cam.scale; // world -> px scale (x, y)
    let pa = (wa - cam.center) * to_px;
    let pb = (wb - cam.center) * to_px;
    var dir = pb - pa;
    let len = length(dir);
    if (len < 1e-6) {
        dir = vec2<f32>(1.0, 0.0);
    } else {
        dir = dir / len;
    }
    let perp = vec2<f32>(-dir.y, dir.x);
    // This end's band width in pixels (RMS spread, floored to a thin line).
    let spread_px = end_spread(pair, end, nf) * abs(to_px.x);
    let w = max(spread_px, 1.5);
    let p = select(pa, pb, end == 1u) + perp * (w * 0.5 * side);
    let ndc = (select(wa, wb, end == 1u) - cam.center) * cam.scale
        + perp * (w * 0.5 * side) / (0.5 * cam.viewport);
    out.clip = vec4<f32>(ndc, 0.0, 1.0);

    var rgb: vec3<f32>;
    if (bp.mode == 1u) {
        rgb = bp.tint.rgb;
    } else {
        let c = pair * 6u + end * 3u;
        rgb = vec3<f32>(
            f32(pair_col[c]),
            f32(pair_col[c + 1u]),
            f32(pair_col[c + 2u]),
        ) / (255.0 * nf);
    }
    // Energy conservation: n edges of alpha spread across a w-px band.
    let alpha = min(1.0, nf * bp.edge_alpha * 1.5 / w);
    out.color = vec4<f32>(rgb, alpha);
    return out;
}

@fragment
fn fs_bundle(in: BundleVsOut) -> @location(0) vec4<f32> {
    return in.color;
}
