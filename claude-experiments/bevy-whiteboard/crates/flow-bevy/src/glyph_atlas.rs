//! Glyph atlas vendored from
//! `editor-idea/crates/terminal-bevy/src/atlas.rs` (Jimmy's terminal
//! atlas). Same design: pre-rasterise printable ASCII into a 1024²
//! RGBA atlas at startup, then lazily insert any other codepoints
//! the renderer asks for via `lookup_or_insert`. macOS gets a
//! CoreText-backed system cascade so chars missing from the primary
//! font (most unicode geometric / box-drawing / pictograph) fall
//! back to whatever font the OS has for that codepoint.
//!
//! Misses (codepoint not in any font, or atlas full) collapse to
//! slot 0 — a hollow tofu box drawn at startup.

use std::collections::HashMap;

use bevy::asset::RenderAssetUsages;
use bevy::image::{Image, TextureAtlasLayout};
use bevy::math::URect;
use bevy::prelude::*;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};

use swash::FontRef;
use swash::scale::{Render, ScaleContext, Source};
use swash::zeno::Format;

const ATLAS_DIM: u32 = 1024;
pub const DPI_SCALE: f32 = 2.0;

#[derive(Resource)]
pub struct GlyphAtlas {
    pub image: Handle<Image>,
    pub layout: Handle<TextureAtlasLayout>,
    slots: HashMap<char, u32>,
    slot_w: u32,
    slot_h: u32,
    cols_per_row: u32,
    max_slots: u32,
    next_slot: u32,
    baseline_atlas_y: u32,
    tofu_slot: u32,
    font_data: &'static [u8],
    system_fallback: SystemFallback,
    font_size_logical: f32,
    scale_context: ScaleContext,
}

/// System-cascade glyph fallback. Asks the OS "which font has this
/// codepoint?" and reads its bytes — same model real terminals use.
mod system_fallback {
    use std::collections::HashMap;
    use std::path::PathBuf;

    pub struct SystemFallback {
        char_cache: HashMap<char, Option<&'static [u8]>>,
        path_cache: HashMap<PathBuf, &'static [u8]>,
        #[cfg(target_os = "macos")]
        cascade_base: Option<core_text::font::CTFont>,
    }

    impl SystemFallback {
        pub fn new() -> Self {
            #[cfg(target_os = "macos")]
            let cascade_base = core_text::font::new_from_name("Menlo", 14.0).ok();
            Self {
                char_cache: HashMap::new(),
                path_cache: HashMap::new(),
                #[cfg(target_os = "macos")]
                cascade_base,
            }
        }

        pub fn font_bytes_for(&mut self, ch: char) -> Option<&'static [u8]> {
            if let Some(&cached) = self.char_cache.get(&ch) {
                return cached;
            }
            let result = self.lookup(ch);
            self.char_cache.insert(ch, result);
            result
        }

        #[cfg(target_os = "macos")]
        fn lookup(&mut self, ch: char) -> Option<&'static [u8]> {
            use core_foundation::base::{CFRange, TCFType};
            use core_foundation::string::{CFString, CFStringRef};
            use core_text::font::{CTFont, CTFontRef};

            unsafe extern "C" {
                fn CTFontCreateForString(
                    currentFont: CTFontRef,
                    string: CFStringRef,
                    range: CFRange,
                ) -> CTFontRef;
            }

            let base = self.cascade_base.as_ref()?;
            let s_buf = ch.to_string();
            let cfs = CFString::new(&s_buf);
            let utf16_len = s_buf.encode_utf16().count() as isize;
            let range = CFRange { location: 0, length: utf16_len };

            let fallback = unsafe {
                let r = CTFontCreateForString(
                    base.as_concrete_TypeRef(),
                    cfs.as_concrete_TypeRef(),
                    range,
                );
                if r.is_null() { return None; }
                CTFont::wrap_under_create_rule(r)
            };

            let url = fallback.url()?;
            let path = url.to_path()?;

            if let Some(&bytes) = self.path_cache.get(&path) {
                return Some(bytes);
            }
            let bytes = std::fs::read(&path).ok()?;
            let leaked: &'static [u8] = Box::leak(bytes.into_boxed_slice());
            self.path_cache.insert(path, leaked);
            Some(leaked)
        }

        #[cfg(not(target_os = "macos"))]
        fn lookup(&mut self, _ch: char) -> Option<&'static [u8]> { None }
    }
}

use system_fallback::SystemFallback;

impl GlyphAtlas {
    pub fn new(
        font_data: &'static [u8],
        font_size_logical: f32,
        cell_w_logical: f32,
        cell_h_logical: f32,
        images: &mut Assets<Image>,
        layouts: &mut Assets<TextureAtlasLayout>,
    ) -> Self {
        let slot_w = (cell_w_logical * DPI_SCALE).ceil() as u32;
        let slot_h = (cell_h_logical * DPI_SCALE).ceil() as u32;
        let cols_per_row = (ATLAS_DIM / slot_w).max(1);
        let rows_in_atlas = (ATLAS_DIM / slot_h).max(1);
        let max_slots = cols_per_row * rows_in_atlas;

        let pixel_count = (ATLAS_DIM * ATLAS_DIM) as usize;
        let mut data = vec![0u8; pixel_count * 4];

        let font = FontRef::from_index(font_data, 0).expect("font must parse");
        let metrics = font.metrics(&[]).scale(font_size_logical * DPI_SCALE);
        let baseline_atlas_y = metrics.ascent.round().max(0.0) as u32;

        write_tofu_into(&mut data, ATLAS_DIM, slot_rect_for(0, slot_w, slot_h, cols_per_row));

        let image = Image::new(
            Extent3d { width: ATLAS_DIM, height: ATLAS_DIM, depth_or_array_layers: 1 },
            TextureDimension::D2,
            data,
            TextureFormat::Rgba8UnormSrgb,
            RenderAssetUsages::RENDER_WORLD | RenderAssetUsages::MAIN_WORLD,
        );
        let image_handle = images.add(image);

        let mut layout = TextureAtlasLayout::new_empty(UVec2::splat(ATLAS_DIM));
        layout.add_texture(slot_rect_for(0, slot_w, slot_h, cols_per_row));
        let layout_handle = layouts.add(layout);

        let mut atlas = Self {
            image: image_handle,
            layout: layout_handle,
            slots: HashMap::new(),
            slot_w,
            slot_h,
            cols_per_row,
            max_slots,
            next_slot: 1,
            baseline_atlas_y,
            tofu_slot: 0,
            font_data,
            system_fallback: SystemFallback::new(),
            font_size_logical,
            scale_context: ScaleContext::new(),
        };

        for byte in b' '..=b'~' {
            atlas.lookup_or_insert(byte as char, images, layouts);
        }
        atlas
    }

    pub fn slot_w_logical(&self) -> f32 { self.slot_w as f32 / DPI_SCALE }
    pub fn slot_h_logical(&self) -> f32 { self.slot_h as f32 / DPI_SCALE }
    pub fn tofu_slot(&self) -> u32 { self.tofu_slot }

    pub fn lookup_or_insert(
        &mut self,
        ch: char,
        images: &mut Assets<Image>,
        layouts: &mut Assets<TextureAtlasLayout>,
    ) -> u32 {
        if let Some(&idx) = self.slots.get(&ch) {
            return idx;
        }
        if self.next_slot >= self.max_slots {
            self.slots.insert(ch, self.tofu_slot);
            return self.tofu_slot;
        }

        let raster = rasterize_in_font(
            &mut self.scale_context,
            self.font_data,
            self.font_size_logical,
            ch,
        )
        .or_else(|| {
            let bytes = self.system_fallback.font_bytes_for(ch)?;
            rasterize_in_font(
                &mut self.scale_context,
                bytes,
                self.font_size_logical,
                ch,
            )
        });

        let Some((raster_data, placement)) = raster else {
            self.slots.insert(ch, self.tofu_slot);
            return self.tofu_slot;
        };

        let slot = self.next_slot;
        let rect = slot_rect_for(slot, self.slot_w, self.slot_h, self.cols_per_row);

        let image = images.get_mut(&self.image).expect("atlas image asset");
        blit_glyph(
            image.data.as_mut().expect("atlas image data"),
            ATLAS_DIM,
            rect,
            self.baseline_atlas_y,
            &raster_data,
            placement.0, placement.1, placement.2, placement.3,
        );

        let layout = layouts.get_mut(&self.layout).expect("atlas layout asset");
        layout.add_texture(rect);

        self.slots.insert(ch, slot);
        self.next_slot += 1;
        slot
    }
}

fn rasterize_in_font(
    scale_context: &mut ScaleContext,
    font_data: &[u8],
    font_size_logical: f32,
    ch: char,
) -> Option<(Vec<u8>, (i32, i32, i32, i32))> {
    let font = FontRef::from_index(font_data, 0)?;
    let glyph_id = font.charmap().map(ch);
    if glyph_id == 0 {
        return None;
    }
    let mut scaler = scale_context
        .builder(font)
        .size(font_size_logical * DPI_SCALE)
        .hint(true)
        .build();
    let img = Render::new(&[Source::Outline])
        .format(Format::Alpha)
        .render(&mut scaler, glyph_id)?;
    Some((
        img.data,
        (
            img.placement.width as i32,
            img.placement.height as i32,
            img.placement.left,
            img.placement.top,
        ),
    ))
}

fn slot_rect_for(slot: u32, slot_w: u32, slot_h: u32, cols_per_row: u32) -> URect {
    let col = slot % cols_per_row;
    let row = slot / cols_per_row;
    let x = col * slot_w;
    let y = row * slot_h;
    URect { min: UVec2::new(x, y), max: UVec2::new(x + slot_w, y + slot_h) }
}

fn blit_glyph(
    data: &mut [u8],
    atlas_dim: u32,
    rect: URect,
    baseline_y_in_slot: u32,
    raster: &[u8],
    bitmap_w: i32,
    bitmap_h: i32,
    bitmap_left: i32,
    bitmap_top: i32,
) {
    if bitmap_w <= 0 || bitmap_h <= 0 { return; }
    let dst_x_origin = rect.min.x as i32 + bitmap_left;
    let dst_y_origin = rect.min.y as i32 + baseline_y_in_slot as i32 - bitmap_top;
    let stride = atlas_dim as usize * 4;
    let x_min = rect.min.x as i32;
    let x_max = rect.max.x as i32;
    let y_min = rect.min.y as i32;
    let y_max = rect.max.y as i32;

    for sy in 0..bitmap_h {
        let dy = dst_y_origin + sy;
        if dy < y_min || dy >= y_max { continue; }
        let row_start = (sy * bitmap_w) as usize;
        let dy_stride = dy as usize * stride;
        for sx in 0..bitmap_w {
            let dx = dst_x_origin + sx;
            if dx < x_min || dx >= x_max { continue; }
            let alpha = raster[row_start + sx as usize];
            if alpha == 0 { continue; }
            let p = dy_stride + dx as usize * 4;
            data[p] = 255;
            data[p + 1] = 255;
            data[p + 2] = 255;
            data[p + 3] = alpha;
        }
    }
}

fn write_tofu_into(data: &mut [u8], atlas_dim: u32, rect: URect) {
    let stride = atlas_dim as usize * 4;
    let inset = 1usize;
    let w = (rect.max.x - rect.min.x) as usize;
    let h = (rect.max.y - rect.min.y) as usize;
    if w < 2 * inset + 1 || h < 2 * inset + 1 { return; }
    let x0 = rect.min.x as usize;
    let y0 = rect.min.y as usize;
    for dy in inset..h - inset {
        let edge_y = dy == inset || dy == h - inset - 1;
        for dx in inset..w - inset {
            let edge_x = dx == inset || dx == w - inset - 1;
            if !(edge_x || edge_y) { continue; }
            let p = (y0 + dy) * stride + (x0 + dx) * 4;
            data[p] = 255;
            data[p + 1] = 255;
            data[p + 2] = 255;
            data[p + 3] = 200;
        }
    }
}
