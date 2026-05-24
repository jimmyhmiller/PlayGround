//! GPU-side terminal grid renderer.
//!
//! One `TermMaterial` per pane: a single fullscreen quad whose fragment
//! shader looks each fragment up in a per-cell data texture, then in
//! the shared glyph atlas, and outputs the final pixel. Replaces the
//! previous "spawn one Sprite per cell" model — every cell change is
//! just a texel write into the data texture, not an ECS component
//! mutation that has to propagate through extract / batch / draw.
//!
//! See `term_material.wgsl` for the shader.

use bevy::asset::{embedded_path, AssetPath, RenderAssetUsages};
use bevy::image::Image;
use bevy::prelude::*;
use bevy::render::render_resource::{AsBindGroup, Extent3d, ShaderType, TextureDimension, TextureFormat};
use bevy::shader::ShaderRef;
use bevy::sprite_render::{AlphaMode2d, Material2d, Material2dPlugin};
use bytemuck::{Pod, Zeroable};

/// Bevy plugin that registers `Material2dPlugin<TermMaterial>` and
/// embeds the WGSL shader.
pub struct TermMaterialPlugin;

impl Plugin for TermMaterialPlugin {
    fn build(&self, app: &mut App) {
        bevy::asset::embedded_asset!(app, "term_material.wgsl");
        app.add_plugins(Material2dPlugin::<TermMaterial>::default());
    }
}

/// One per pane. Holds uniform params (grid + atlas geometry), the
/// shared atlas texture, and a per-pane `cells` texture that the
/// terminal worker → sync_grid pipeline writes into.
#[derive(Asset, TypePath, AsBindGroup, Debug, Clone)]
pub struct TermMaterial {
    #[uniform(0)]
    pub params: TermParams,
    #[texture(1)]
    #[sampler(2)]
    pub atlas: Handle<Image>,
    /// `Rgba32Uint` (16 bytes/cell): (glyph_index, fg_packed, bg_packed, flags).
    /// One texel per cell of the terminal grid; the worker rewrites
    /// dirty texels via `Image::data` and re-uploads.
    #[texture(3, sample_type = "u_int")]
    pub cells: Handle<Image>,
}

impl Material2d for TermMaterial {
    fn fragment_shader() -> ShaderRef {
        ShaderRef::Path(
            AssetPath::from_path_buf(embedded_path!("term_material.wgsl"))
                .with_source("embedded"),
        )
    }

    fn alpha_mode(&self) -> AlphaMode2d {
        AlphaMode2d::Opaque
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, ShaderType, Pod, Zeroable)]
pub struct TermParams {
    pub cols: u32,
    pub rows: u32,
    pub atlas_cols: u32,
    /// Usable glyph area per slot (matches the rasterized bitmap).
    pub atlas_slot_w: u32,
    pub atlas_slot_h: u32,
    pub atlas_dim: u32,
    /// Stride between slot origins in the atlas (slot + transparent
    /// gutter). The shader needs this to find the next slot; sampling
    /// stays within `slot_w × slot_h`.
    pub atlas_stride_w: u32,
    pub atlas_stride_h: u32,
}

/// One texel in the `cells` texture (matches WGSL `vec4<u32>`).
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable, Default, PartialEq, Eq)]
pub struct GpuCell {
    pub glyph_index: u32,
    pub fg_packed: u32,
    pub bg_packed: u32,
    pub flags: u32,
}

impl GpuCell {
    /// A freshly-allocated cell with no glyph yet. `glyph_index: 0`
    /// points at the atlas's reserved blank slot (alpha=0 throughout),
    /// so any cell the worker hasn't painted in this frame renders as
    /// pure `default_bg` — not as a tofu box, which would be the case
    /// if slot 0 had any visible pixels.
    pub fn blank(default_bg: u32) -> Self {
        Self {
            glyph_index: 0,
            fg_packed: 0xFFFFFFFF,
            bg_packed: default_bg,
            flags: 0,
        }
    }
}

/// Pack an `RgbColor` (sRGB u8 triple) into a u32 the shader can
/// `unpack_rgba` directly: `RRGGBBAA`.
pub fn pack_rgb(r: u8, g: u8, b: u8) -> u32 {
    ((r as u32) << 24) | ((g as u32) << 16) | ((b as u32) << 8) | 0xFF
}

/// Build a fresh `Rgba32Uint` image sized for `cols × rows` cells.
/// Initially filled with a default-bg blank cell.
pub fn make_cells_image(cols: u32, rows: u32, default_bg: u32) -> Image {
    let cells = vec![GpuCell::blank(default_bg); (cols * rows) as usize];
    let bytes: Vec<u8> = bytemuck::cast_slice(&cells).to_vec();
    Image::new(
        Extent3d {
            width: cols.max(1),
            height: rows.max(1),
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        bytes,
        TextureFormat::Rgba32Uint,
        RenderAssetUsages::RENDER_WORLD | RenderAssetUsages::MAIN_WORLD,
    )
}

/// Overwrite a single cell in an existing cells image. Caller is
/// responsible for ensuring the image is large enough and for marking
/// the asset as changed so Bevy re-uploads it.
pub fn write_cell(image: &mut Image, cols: u32, col: u32, row: u32, cell: GpuCell) {
    let idx = (row * cols + col) as usize;
    let cells: &mut [GpuCell] = bytemuck::cast_slice_mut(
        image
            .data
            .as_mut()
            .expect("cells image must have CPU-side data"),
    );
    if let Some(slot) = cells.get_mut(idx) {
        *slot = cell;
    }
}

/// Read a cell without taking a `&mut`. Used by `sync_grid` to
/// compare-before-write.
pub fn read_cell(image: &Image, cols: u32, col: u32, row: u32) -> Option<GpuCell> {
    let idx = (row * cols + col) as usize;
    let cells: &[GpuCell] =
        bytemuck::cast_slice(image.data.as_ref()?.as_slice());
    cells.get(idx).copied()
}
