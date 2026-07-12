// Node + edge rendering. Nodes are instanced SDF circles (crisp at any zoom);
// edges are 1px lines with endpoint-blended colors and low alpha so dense
// regions read as luminous bundles rather than solid fill.

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

@group(0) @binding(0) var<uniform> cam: Camera;
@group(0) @binding(1) var<uniform> params: RenderParams;

@group(1) @binding(0) var<storage, read> positions: array<vec2<f32>>;
@group(1) @binding(1) var<storage, read> colors: array<u32>;
@group(1) @binding(2) var<storage, read> sizes: array<f32>;
@group(1) @binding(3) var<storage, read> edges: array<u32>;

fn unpack_color(c: u32) -> vec4<f32> {
    return vec4<f32>(
        f32(c & 0xffu) / 255.0,
        f32((c >> 8u) & 0xffu) / 255.0,
        f32((c >> 16u) & 0xffu) / 255.0,
        f32((c >> 24u) & 0xffu) / 255.0,
    );
}

// ---------------- Nodes ----------------

struct NodeVsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
};

@vertex
fn vs_node(@builtin(vertex_index) vi: u32, @builtin(instance_index) inst: u32) -> NodeVsOut {
    var corners = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0),
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, 1.0), vec2<f32>(-1.0, 1.0),
    );
    let corner = corners[vi];
    let world = positions[inst];

    let r = clamp(
        params.base_radius_px * pow(max(sizes[inst], 0.0001), params.size_gamma),
        params.min_radius_px,
        params.max_radius_px,
    );
    let radius_ndc = vec2<f32>(r * 2.0 / cam.viewport.x, r * 2.0 / cam.viewport.y);
    let center_ndc = (world - cam.center) * cam.scale;

    var out: NodeVsOut;
    out.clip = vec4<f32>(center_ndc + corner * radius_ndc, 0.0, 1.0);
    out.uv = corner;
    var col = unpack_color(colors[inst]);
    col.a = col.a * params.node_alpha;
    out.color = col;
    return out;
}

@fragment
fn fs_node(in: NodeVsOut) -> @location(0) vec4<f32> {
    let d = length(in.uv);
    // Anti-aliased circle edge using screen-space derivative of the SDF.
    let fw = fwidth(d);
    let alpha = 1.0 - smoothstep(1.0 - fw, 1.0, d);
    if (alpha <= 0.0) {
        discard;
    }
    return vec4<f32>(in.color.rgb, in.color.a * alpha);
}

// ---------------- Edges ----------------

struct EdgeVsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) color: vec4<f32>,
};

@vertex
fn vs_edge(@builtin(vertex_index) vi: u32) -> EdgeVsOut {
    let node = edges[vi];
    let world = positions[node];
    let ndc = (world - cam.center) * cam.scale;

    var out: EdgeVsOut;
    out.clip = vec4<f32>(ndc, 0.0, 1.0);
    var col = unpack_color(colors[node]);
    out.color = vec4<f32>(col.rgb, params.edge_alpha);
    return out;
}

@fragment
fn fs_edge(in: EdgeVsOut) -> @location(0) vec4<f32> {
    return in.color;
}
