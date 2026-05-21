// Terminal grid renderer.
//
// One material per pane. The fragment shader samples a cell-data
// texture (one texel per cell, Rgba32Uint) to learn which atlas slot
// and which fg/bg colors to use, then samples the glyph atlas to get
// the glyph's alpha mask. Replaces the previous "one Sprite entity per
// cell" model — a single draw call per pane regardless of cell count.

#import bevy_sprite::mesh2d_vertex_output::VertexOutput

struct TermParams {
    // Grid dimensions (cells).
    cols: u32,
    rows: u32,
    // Atlas geometry.
    atlas_cols: u32,
    atlas_slot_w: u32,
    atlas_slot_h: u32,
    atlas_dim: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(#{MATERIAL_BIND_GROUP}) @binding(0) var<uniform> params: TermParams;
@group(#{MATERIAL_BIND_GROUP}) @binding(1) var atlas_tex: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(2) var atlas_sampler: sampler;
@group(#{MATERIAL_BIND_GROUP}) @binding(3) var cells_tex: texture_2d<u32>;

// Standard sRGB → linear, IEC 61966-2-1. The packed cell colors come
// from libghostty as 8-bit sRGB triples; Bevy's surface is sRGB-encoded
// and the GPU re-encodes linear → sRGB on write, so to round-trip the
// terminal's intended color we have to hand the GPU linear values. Pass
// sRGB through unchanged and colors come out the other side gamma-
// shifted (overbright / "washed out").
fn srgb_to_linear(c: f32) -> f32 {
    if (c <= 0.04045) {
        return c / 12.92;
    }
    return pow((c + 0.055) / 1.055, 2.4);
}

fn unpack_rgba(packed: u32) -> vec4<f32> {
    let r = f32((packed >> 24u) & 0xFFu) / 255.0;
    let g = f32((packed >> 16u) & 0xFFu) / 255.0;
    let b = f32((packed >> 8u) & 0xFFu) / 255.0;
    let a = f32(packed & 0xFFu) / 255.0;
    return vec4<f32>(
        srgb_to_linear(r),
        srgb_to_linear(g),
        srgb_to_linear(b),
        a,
    );
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    // Bevy's `Rectangle` mesh maps the top-left of the quad (world +Y)
    // to UV (0, 0) and the bottom-left to UV (0, 1). Terminal grid rows
    // are also indexed top-down (row 0 = top), so `uv.y` already matches
    // `row_f` directly — no flip. (Got this wrong once; the giveaway is
    // glyphs rendering vertically mirrored.)
    let u = in.uv.x;
    let v = in.uv.y;

    let col_f = u * f32(params.cols);
    let row_f = v * f32(params.rows);
    let col = u32(col_f);
    let row = u32(row_f);
    if (col >= params.cols || row >= params.rows) {
        // Outside the grid (shouldn't normally happen since the mesh is
        // sized to the grid) — write transparent.
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }

    let cell = textureLoad(cells_tex, vec2<i32>(i32(col), i32(row)), 0);
    let glyph_index = cell.r;
    let fg = unpack_rgba(cell.g);
    let bg = unpack_rgba(cell.b);

    // Cell-local UV in [0,1].
    let cell_uv = vec2<f32>(fract(col_f), fract(row_f));

    // Glyph atlas: slots laid out left-to-right, top-to-bottom.
    let atlas_col = glyph_index % params.atlas_cols;
    let atlas_row = glyph_index / params.atlas_cols;
    let slot_dim = vec2<f32>(f32(params.atlas_slot_w), f32(params.atlas_slot_h));
    let atlas_origin = vec2<f32>(f32(atlas_col), f32(atlas_row)) * slot_dim;
    let sample_px = atlas_origin + cell_uv * slot_dim;
    let sample_uv = sample_px / f32(params.atlas_dim);
    let glyph = textureSample(atlas_tex, atlas_sampler, sample_uv);

    // Glyph atlas is white-with-alpha; mix bg ← fg by alpha.
    return mix(bg, fg, glyph.a);
}
