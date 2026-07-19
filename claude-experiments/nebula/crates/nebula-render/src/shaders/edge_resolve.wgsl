// Fullscreen resolve for the compute edge rasterizer: converts the per-pixel
// fixed-point sums produced by edge_raster.wgsl to linear color and adds them
// onto the frame (blend One/One), before nodes draw on top — same layering as
// the hardware edges-then-nodes order.

struct AccumDims {
    w: u32,
    h: u32,
    _p0: u32,
    _p1: u32,
};
@group(0) @binding(0) var<uniform> dims: AccumDims;
@group(0) @binding(1) var<storage, read> accum: array<u32>;

const SCALE: f32 = 4096.0;

@vertex
fn vs_fullscreen(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
    // Single fullscreen triangle.
    let x = f32((vi << 1u) & 2u) * 2.0 - 1.0;
    let y = f32(vi & 2u) * 2.0 - 1.0;
    return vec4<f32>(x, y, 0.0, 1.0);
}

@fragment
fn fs_resolve(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let x = min(u32(pos.x), dims.w - 1u);
    let y = min(u32(pos.y), dims.h - 1u);
    let idx = (y * dims.w + x) * 3u;
    let r = f32(accum[idx]) / SCALE;
    let g = f32(accum[idx + 1u]) / SCALE;
    let b = f32(accum[idx + 2u]) / SCALE;
    // Alpha 0 with One/One blending: adds rgb, leaves dst alpha alone.
    return vec4<f32>(r, g, b, 0.0);
}
