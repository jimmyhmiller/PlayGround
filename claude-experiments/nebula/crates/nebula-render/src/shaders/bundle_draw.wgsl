// Edge bundling, draw side (separate module because the vertex stage may only
// bind these buffers read-only — same split as density.wgsl/density_render).
// One instance per (source cell, target cell) pair; empty pairs collapse to a
// degenerate offscreen line. Occupied pairs draw centroid->centroid with
// alpha = count * edge_alpha * level weight (additively blended, capped), so a
// bundle carries the same light its member edges would have, split across the
// two cross-faded LOD scales.

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
    weight: f32,
    clip_min: vec2<f32>,
    clip_max: vec2<f32>,
    edge_alpha: f32,
    _p0: f32,
    _p1: f32,
    _p2: f32,
};

@group(0) @binding(0) var<uniform> cam: Camera;
@group(0) @binding(1) var<uniform> bp: BParams;

@group(1) @binding(0) var<storage, read> pair_count: array<u32>;
@group(1) @binding(1) var<storage, read> pair_geom: array<i32>;
@group(1) @binding(2) var<storage, read> pair_col: array<u32>;

struct BundleVsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) color: vec4<f32>,
};

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
    let ncells = bp.cells_x * bp.cells_y;
    let ia = pair / ncells;
    let ib = pair % ncells;
    let idx = select(ia, ib, vi == 1u);
    let ctr = bp.origin
        + (vec2<f32>(f32(idx % bp.cells_x), f32(idx / bp.cells_x)) + 0.5) * bp.cell;
    let g = pair * 4u + vi * 2u;
    let inv = bp.cell / (128.0 * f32(n));
    let world = ctr + vec2<f32>(f32(pair_geom[g]) * inv, f32(pair_geom[g + 1u]) * inv);

    let ndc = (world - cam.center) * cam.scale;
    out.clip = vec4<f32>(ndc, 0.0, 1.0);
    var rgb: vec3<f32>;
    if (bp.mode == 1u) {
        rgb = bp.tint.rgb;
    } else {
        let c = pair * 6u + vi * 3u;
        rgb = vec3<f32>(
            f32(pair_col[c]),
            f32(pair_col[c + 1u]),
            f32(pair_col[c + 2u]),
        ) / (255.0 * f32(n));
    }
    out.color = vec4<f32>(rgb, min(1.0, f32(n) * bp.edge_alpha) * bp.weight);
    return out;
}

@fragment
fn fs_bundle(in: BundleVsOut) -> @location(0) vec4<f32> {
    return in.color;
}
