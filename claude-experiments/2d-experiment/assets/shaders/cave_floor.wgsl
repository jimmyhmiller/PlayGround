#import bevy_sprite::mesh2d_vertex_output::VertexOutput

// Size of a "big pixel" in world units. Larger = chunkier NES look.
const PIXEL_SIZE: f32 = 4.0;
// Logical tile size in big-pixels. Each tile gets its own rock pattern.
const TILE_PIXELS: f32 = 16.0;

fn hash2(p_in: vec2<f32>) -> f32 {
    let p = vec2<f32>(
        dot(p_in, vec2<f32>(127.1, 311.7)),
        dot(p_in, vec2<f32>(269.5, 183.3)),
    );
    return fract(sin(p.x + p.y) * 43758.5453);
}

// A small set of rock-tile "stamps" — each is a 16x16 bitmap-ish pattern
// generated procedurally from a tile seed, then sampled by local coord.
fn rock_stamp(tile_id: vec2<f32>, local: vec2<f32>) -> f32 {
    // Per-tile seeds for two scattered "lumps".
    let seed_a = hash2(tile_id);
    let seed_b = hash2(tile_id + vec2<f32>(13.0, 7.0));
    let seed_c = hash2(tile_id + vec2<f32>(3.0, 19.0));

    // Centers (in local tile space, 0..TILE_PIXELS).
    let c_a = vec2<f32>(3.0 + seed_a * 10.0, 3.0 + seed_b * 10.0);
    let c_b = vec2<f32>(2.0 + seed_c * 12.0, 9.0 + seed_a * 5.0);

    // Blocky distance (max-norm → square lumps, very 8-bit).
    let d_a = max(abs(local.x - c_a.x), abs(local.y - c_a.y));
    let d_b = max(abs(local.x - c_b.x), abs(local.y - c_b.y));

    // Lump radii vary slightly per tile.
    let r_a = 2.0 + floor(seed_b * 2.0);
    let r_b = 1.0 + floor(seed_c * 2.0);

    var v: f32 = 0.0;
    if (d_a < r_a) { v = max(v, 1.0 - d_a / (r_a + 1.0)); }
    if (d_b < r_b) { v = max(v, 0.8 - d_b / (r_b + 1.0)); }
    return v;
}

// Hard quantize into a 4-entry palette.
fn palette(i: i32) -> vec3<f32> {
    // Dark → light, cool cave-stone tones.
    if (i <= 0) { return vec3<f32>(0.055, 0.055, 0.075); }   // near-black
    if (i == 1) { return vec3<f32>(0.115, 0.115, 0.145); }   // deep slate
    if (i == 2) { return vec3<f32>(0.190, 0.180, 0.200); }   // mid stone
    return           vec3<f32>(0.290, 0.275, 0.285);         // highlight
}

@fragment
fn fragment(mesh: VertexOutput) -> @location(0) vec4<f32> {
    // 1. Snap to big-pixel grid — this is what gives the chunky 8-bit look.
    let world = floor(mesh.world_position.xy / PIXEL_SIZE);

    // 2. Figure out which logical tile we're in, and the local pixel within it.
    let tile_id = floor(world / TILE_PIXELS);
    let local = world - tile_id * TILE_PIXELS; // 0..TILE_PIXELS integer coords

    // 3. Base shade varies per tile (some tiles darker, some lighter).
    let tile_shade = hash2(tile_id + vec2<f32>(1.3, 5.7));
    var idx: i32 = 1;
    if (tile_shade < 0.35) { idx = 0; } else if (tile_shade > 0.75) { idx = 2; }

    // 4. Rock stamps bump the index up when present → highlights on the stone.
    let rock = rock_stamp(tile_id, local);
    if (rock > 0.55) { idx = min(idx + 2, 3); }
    else if (rock > 0.25) { idx = min(idx + 1, 3); }

    // 5. Sparse darker pits — random single pixels, NES-style speckle.
    let speckle = hash2(world + vec2<f32>(91.0, 17.0));
    if (speckle < 0.04) { idx = max(idx - 1, 0); }

    return vec4<f32>(palette(idx), 1.0);
}
