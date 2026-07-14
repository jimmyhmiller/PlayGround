// Compute software rasterizer for edges — runtime-toggleable alternative to
// the hardware LineList path, aimed at the zoomed-out long-line case where a
// tile-based GPU pays for binning every line into every tile it crosses.
//
// Each thread takes one edge, projects + clips it to the viewport
// (Liang-Barsky, tracking the parameter range so endpoint-color interpolation
// matches the hardware's), then DDA-walks its pixels adding a fixed-point
// premultiplied color (rgb * edge_alpha * SCALE) into a per-pixel u32
// accumulator with atomics. Integer addition commutes exactly, so the result
// is bit-identical across frames regardless of GPU scheduling — deterministic
// by construction. edge_resolve.wgsl composites the sum onto the frame.

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
    _p0: u32,
    _p1: u32,
};
@group(1) @binding(0) var<uniform> dims: AccumDims;
@group(1) @binding(1) var<storage, read_write> accum: array<atomic<u32>>;

// Fixed-point scale for accumulation. Max contribution per edge per channel is
// ~alpha * SCALE ≈ 500, so u32 overflows only past ~8M overlapping edges on
// one pixel.
const SCALE: f32 = 4096.0;

fn unpack_color(c: u32) -> vec4<f32> {
    return vec4<f32>(
        f32(c & 0xffu) / 255.0,
        f32((c >> 8u) & 0xffu) / 255.0,
        f32((c >> 16u) & 0xffu) / 255.0,
        f32((c >> 24u) & 0xffu) / 255.0,
    );
}

// One Liang-Barsky boundary. Returns false if the segment is fully outside.
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

@compute @workgroup_size(256)
fn raster_edges(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_index) li: u32,
) {
    let e = (wg.y * nwg.x + wg.x) * 256u + li;
    if (e * 2u + 1u >= arrayLength(&edges)) {
        return;
    }
    let ia = edges[e * 2u];
    let ib = edges[e * 2u + 1u];

    let na = (positions[ia] - cam.center) * cam.scale;
    let nb = (positions[ib] - cam.center) * cam.scale;
    let wf = f32(dims.w);
    let hf = f32(dims.h);
    // NDC -> pixel space (y flipped).
    let pa = vec2<f32>((na.x * 0.5 + 0.5) * wf, (0.5 - na.y * 0.5) * hf);
    let pb = vec2<f32>((nb.x * 0.5 + 0.5) * wf, (0.5 - nb.y * 0.5) * hf);
    let d = pb - pa;

    // A zero-length line rasterizes to nothing on the hardware path.
    if (max(abs(d.x), abs(d.y)) < 1e-6) {
        return;
    }

    // Clip to the viewport, keeping the [t0, t1] parameter range for color.
    var t0 = 0.0;
    var t1 = 1.0;
    if (!clip_axis(-d.x, pa.x, &t0, &t1)) { return; }
    if (!clip_axis(d.x, (wf - 0.001) - pa.x, &t0, &t1)) { return; }
    if (!clip_axis(-d.y, pa.y, &t0, &t1)) { return; }
    if (!clip_axis(d.y, (hf - 0.001) - pa.y, &t0, &t1)) { return; }
    if (t0 >= t1) {
        return;
    }

    var col_a = unpack_color(colors[ia]).rgb;
    var col_b = unpack_color(colors[ib]).rgb;
    if (edge_style.mode == 1u) {
        col_a = edge_style.color.rgb;
        col_b = edge_style.color.rgb;
    }
    let alpha = params.edge_alpha;

    let a_px = pa + d * t0;
    let seg = d * (t1 - t0);
    // One pixel per major-axis step, endpoint-exclusive — approximates the
    // hardware diamond-exit rule. Sub-pixel edges still light one pixel.
    let n = max(u32(ceil(max(abs(seg.x), abs(seg.y)))), 1u);
    let inv_n = 1.0 / f32(n);
    // Once a pixel's sum reaches 1.0 in every channel the displayed value is
    // clamped and further adds cannot change it (the background only adds
    // more), so skipping them is exact. This converts the contended
    // read-modify-writes in dense saturated cores into cheap reads — an
    // optimization fixed-function blending cannot express.
    let sat = u32(SCALE);
    for (var i = 0u; i < n; i++) {
        let t = f32(i) * inv_n;
        let p = a_px + seg * t;
        let x = u32(clamp(p.x, 0.0, wf - 1.0));
        let y = u32(clamp(p.y, 0.0, hf - 1.0));
        let col = mix(col_a, col_b, mix(t0, t1, t));
        let idx = (y * dims.w + x) * 3u;
        if (atomicLoad(&accum[idx]) >= sat &&
            atomicLoad(&accum[idx + 1u]) >= sat &&
            atomicLoad(&accum[idx + 2u]) >= sat) {
            continue;
        }
        atomicAdd(&accum[idx], u32(round(col.r * alpha * SCALE)));
        atomicAdd(&accum[idx + 1u], u32(round(col.g * alpha * SCALE)));
        atomicAdd(&accum[idx + 2u], u32(round(col.b * alpha * SCALE)));
    }
}
