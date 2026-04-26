// Packet cloud — single-draw-call instanced renderer for traveling
// packets. The mesh has no real geometry; vertex_index drives both
// quad-corner selection and instance lookup ("vertex pulling"). All
// interpolation happens GPU-side from the storage buffer of
// `PacketInstance` records — the CPU just rewrites the buffer each
// frame and uniform `now_real` advances.

#import bevy_sprite::mesh2d_view_bindings::view

struct PacketInstance {
    start: vec2<f32>,
    end: vec2<f32>,
    emit_real: f32,
    arrive_real: f32,
    color: vec4<f32>,
};

struct PacketClock {
    now_real: f32,
    active_count: u32,
    quad_radius: f32,
    _pad: f32,
};

@group(2) @binding(0) var<storage, read> instances: array<PacketInstance>;
@group(2) @binding(1) var<uniform> clock: PacketClock;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
};

// Two triangles per quad: corners 0..5 → (-1,-1), (1,-1), (1,1), (1,1), (-1,1), (-1,-1)
fn quad_corner(i: u32) -> vec2<f32> {
    switch (i) {
        case 0u, 5u: { return vec2(-1.0, -1.0); }
        case 1u:     { return vec2( 1.0, -1.0); }
        case 2u, 3u: { return vec2( 1.0,  1.0); }
        case 4u:     { return vec2(-1.0,  1.0); }
        default:     { return vec2(0.0); }
    }
}

@vertex
fn vertex(@builtin(vertex_index) vid: u32) -> VertexOutput {
    var out: VertexOutput;
    let instance_id = vid / 6u;
    let corner_id   = vid % 6u;

    // Out-of-range slots collapse to a degenerate clip-space point so
    // the GPU rasterizes nothing for them. Means we don't need to
    // resize the mesh as the packet count grows or shrinks — just
    // update `clock.active_count`.
    if (instance_id >= clock.active_count) {
        out.clip_position = vec4(2.0, 2.0, 2.0, 1.0);
        out.uv    = vec2(0.0);
        out.color = vec4(0.0);
        return out;
    }

    let inst   = instances[instance_id];
    let denom  = max(inst.arrive_real - inst.emit_real, 1e-6);
    let t      = clamp((clock.now_real - inst.emit_real) / denom, 0.0, 1.0);
    let center = mix(inst.start, inst.end, t);

    let corner = quad_corner(corner_id);
    let world  = center + corner * clock.quad_radius;

    // Push packets to a high world-z so they sort above gizmo arrows
    // and node meshes in Bevy's 2D transparent phase.
    out.clip_position = view.clip_from_world * vec4(world, 100.0, 1.0);
    out.uv    = corner;          // -1..1 across the quad
    out.color = inst.color;
    return out;
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    // Soft circular alpha: 1.0 inside (r < 0.85), feathered to 0 at r=1.0.
    let r = length(in.uv);
    if (r > 1.0) { discard; }
    // WGSL `smoothstep` requires edge0 < edge1; do the natural
    // 0→1 ramp at the rim and invert.
    let alpha = 1.0 - smoothstep(0.85, 1.0, r);
    return vec4(in.color.rgb, in.color.a * alpha);
}
