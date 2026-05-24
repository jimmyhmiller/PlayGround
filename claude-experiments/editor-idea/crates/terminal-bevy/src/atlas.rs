//! Glyph atlas: rasterize each character on demand via `swash`, blit
//! its alpha bitmap into a fixed-size RGBA atlas texture, and hand back
//! a `TextureAtlasLayout` slot index. The renderer treats every cell
//! as a textured sprite that samples its glyph from the atlas — no
//! cosmic-text / Text2d / per-frame shaping in the hot path.
//!
//! Sized for a 1024×1024 atlas (4 MiB) — fits ~1700 glyph slots at our
//! cell dimensions, several orders of magnitude more than any real
//! terminal session reaches. We pre-populate printable ASCII at startup
//! so `cat` of an English file never faults a glyph; non-ASCII chars
//! get rasterized lazily on first sight (one full image re-upload per
//! novel char, which is fine since they arrive slowly).
//!
//! Slot 0 is reserved as a fully-transparent "blank" slot so the cells
//! texture's default-initialized state (all `glyph_index = 0`) samples to
//! alpha=0 and renders as solid background — without this you get a
//! screenful of tofu before the first publish lands. Slot 1 holds the
//! tofu glyph, used as the genuine font-missing / atlas-full fallback.

use std::collections::HashMap;

use bevy::asset::RenderAssetUsages;
use bevy::image::{Image, TextureAtlasLayout};
use bevy::math::URect;
use bevy::prelude::*;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};

use swash::scale::{Render, ScaleContext, Source};
use swash::zeno::Format;
use swash::FontRef;

/// Atlas texture is 1024×1024. Fits ~1700 glyphs at our cell size.
/// Picked smaller than the obvious 2048² choice because every novel-glyph
/// insertion re-uploads the whole image to the GPU; smaller = cheaper
/// fault.
const ATLAS_DIM: u32 = 1024;

/// Multiplier between logical pixels (sprite quad size) and atlas
/// pixels (rasterized glyph size). 2× keeps glyphs crisp on retina
/// displays without burning much memory.
pub const DPI_SCALE: f32 = 2.0;

/// Transparent gutter between slots in the atlas. Linear sampling at a
/// cell's edge would otherwise pick up the neighboring slot's pixels —
/// producing faint dots and speckles below glyphs whose atlas neighbor
/// has ink near its top row. With a 1px transparent gutter the worst
/// the sampler can do at an edge is read 50% glyph + 50% transparent.
pub const SLOT_PAD: u32 = 1;

#[derive(Resource)]
pub struct GlyphAtlas {
    pub image: Handle<Image>,
    pub layout: Handle<TextureAtlasLayout>,
    /// Char → atlas slot index. Built up over the session.
    slots: HashMap<char, u32>,
    /// Usable glyph area (the rect we blit into / sample from).
    slot_w: u32,
    slot_h: u32,
    /// Stride between slot origins in the atlas image (slot + SLOT_PAD).
    stride_w: u32,
    stride_h: u32,
    cols_per_row: u32,
    max_slots: u32,
    next_slot: u32,
    baseline_atlas_y: u32,
    /// Drawn as a hollow box for chars we couldn't rasterize. NOT slot
    /// 0 — slot 0 is the all-transparent "blank" slot used by the
    /// cells texture's default-initialized cells.
    tofu_slot: u32,
    /// `'static` font bytes. `swash::FontRef` borrows from this.
    font_data: &'static [u8],
    /// CoreText-backed per-codepoint cascade. Asks the OS which font
    /// owns each missing char; loads + leaks the bytes once per font.
    system_fallback: SystemFallback,
    font_size_logical: f32,
    scale_context: ScaleContext,
}

/// System-cascade glyph fallback. Replaces a hardcoded list of font
/// paths with the same per-codepoint lookup real terminals (Terminal.app,
/// Alacritty, kitty) use: ask the OS "which font has this codepoint?"
/// and load its bytes. Means we don't have to keep growing a list every
/// time some prompt theme uses a new symbol — the OS already indexes
/// every installed font's coverage.
///
/// On non-macOS this is a stub returning `None`; we'd add fontconfig /
/// DirectWrite paths analogously when those platforms ship.
mod system_fallback {
    use std::collections::HashMap;
    use std::path::PathBuf;

    pub struct SystemFallback {
        /// Cached per-char result so we make one OS query per unique
        /// codepoint, not one per render call.
        char_cache: HashMap<char, Option<&'static [u8]>>,
        /// Cached per-font-path bytes. A single font (e.g. Menlo) often
        /// covers many fallback chars; we load + leak once and reuse.
        path_cache: HashMap<PathBuf, &'static [u8]>,
        #[cfg(target_os = "macos")]
        cascade_base: Option<core_text::font::CTFont>,
    }

    impl SystemFallback {
        pub fn new() -> Self {
            #[cfg(target_os = "macos")]
            let cascade_base = {
                // Menlo is the canonical macOS terminal font, guaranteed
                // installed, and CT's cascade list off any mono font is
                // effectively the same — this is just the "current font"
                // CTFontCreateForString needs to anchor its query.
                core_text::font::new_from_name("Menlo", 14.0).ok()
            };
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

            // CTFontCreateForString isn't bound by the core-text crate
            // (its FFI declaration is commented out); declare it here.
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
            // CFRange is over UTF-16 code units, which is what
            // CFString stores internally.
            let utf16_len = s_buf.encode_utf16().count() as isize;
            let range = CFRange {
                location: 0,
                length: utf16_len,
            };

            let fallback = unsafe {
                let r = CTFontCreateForString(
                    base.as_concrete_TypeRef(),
                    cfs.as_concrete_TypeRef(),
                    range,
                );
                if r.is_null() {
                    return None;
                }
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
        fn lookup(&mut self, _ch: char) -> Option<&'static [u8]> {
            None
        }
    }
}

use system_fallback::SystemFallback;

impl GlyphAtlas {
    /// Allocate the atlas image, lay out the tofu fallback in slot 0,
    /// and pre-rasterize printable ASCII (32..=126).
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
        let stride_w = slot_w + SLOT_PAD;
        let stride_h = slot_h + SLOT_PAD;
        let cols_per_row = (ATLAS_DIM / stride_w).max(1);
        let rows_in_atlas = (ATLAS_DIM / stride_h).max(1);
        let max_slots = cols_per_row * rows_in_atlas;

        // Atlas image starts fully transparent; we mutate the bytes
        // directly during glyph insertion. Holding RenderAssetUsages so
        // both the main world and render world keep a copy — Bevy's
        // sprite pipeline will re-upload on data mutation.
        let pixel_count = (ATLAS_DIM * ATLAS_DIM) as usize;
        let mut data = vec![0u8; pixel_count * 4];

        // Compute baseline within an atlas cell from font metrics. swash
        // gives us ascent positive (above baseline) and descent negative.
        // The y-position within a cell where the glyph baseline sits is
        // approximately `ascent` atlas pixels from the cell top.
        let font = FontRef::from_index(font_data, 0).expect("font must parse");
        let metrics = font.metrics(&[]).scale(font_size_logical * DPI_SCALE);
        let baseline_atlas_y = metrics.ascent.round().max(0.0) as u32;

        // Slot 0 stays empty (all-transparent). Tofu lives at slot 1
        // so font-missing chars still show a visible box. Written into
        // `data` before the image is uploaded to avoid an extra GPU
        // round-trip.
        const BLANK_SLOT: u32 = 0;
        const TOFU_SLOT: u32 = 1;
        write_tofu_into(
            &mut data,
            ATLAS_DIM,
            slot_rect_for(TOFU_SLOT, slot_w, slot_h, stride_w, stride_h, cols_per_row),
        );

        let image = Image::new(
            Extent3d {
                width: ATLAS_DIM,
                height: ATLAS_DIM,
                depth_or_array_layers: 1,
            },
            TextureDimension::D2,
            data,
            // sRGB so Bevy's sprite shader (which expects sRGB texture
            // input) auto-converts on sample. Glyph pixels are pure
            // white-with-alpha so srgb↔linear is a no-op for the RGB
            // channel; using the right format keeps the alpha blending
            // path correct.
            TextureFormat::Rgba8UnormSrgb,
            RenderAssetUsages::RENDER_WORLD | RenderAssetUsages::MAIN_WORLD,
        );
        let image_handle = images.add(image);

        let mut layout = TextureAtlasLayout::new_empty(UVec2::splat(ATLAS_DIM));
        // Register slot 0 (blank, all-transparent) and slot 1 (tofu) so
        // indices line up with what the shader samples.
        layout.add_texture(slot_rect_for(BLANK_SLOT, slot_w, slot_h, stride_w, stride_h, cols_per_row));
        layout.add_texture(slot_rect_for(TOFU_SLOT, slot_w, slot_h, stride_w, stride_h, cols_per_row));
        let layout_handle = layouts.add(layout);

        let mut atlas = Self {
            image: image_handle,
            layout: layout_handle,
            slots: HashMap::new(),
            slot_w,
            slot_h,
            stride_w,
            stride_h,
            cols_per_row,
            max_slots,
            next_slot: 2, // 0 is blank, 1 is tofu
            baseline_atlas_y,
            tofu_slot: TOFU_SLOT,
            font_data,
            system_fallback: SystemFallback::new(),
            font_size_logical,
            scale_context: ScaleContext::new(),
        };

        // Pre-rasterize printable ASCII so the first frame of `cat` of
        // any English file finds every glyph already cached. We mutate
        // the image bytes directly here so the GPU upload happens once,
        // not 95 times.
        for byte in b' '..=b'~' {
            atlas.lookup_or_insert(byte as char, images, layouts);
        }
        atlas
    }

    pub fn tofu_slot(&self) -> u32 {
        self.tofu_slot
    }

    /// Atlas-space cell width in pixels (DPI-scaled). This is the
    /// usable glyph area, not the stride — see `stride_w` for layout.
    pub fn slot_w(&self) -> u32 {
        self.slot_w
    }
    /// Atlas-space cell height in pixels (DPI-scaled).
    pub fn slot_h(&self) -> u32 {
        self.slot_h
    }
    /// Distance between slot origins along X (slot_w + SLOT_PAD). The
    /// shader needs this to find a slot's atlas origin; sampling stays
    /// within `slot_w × slot_h`.
    pub fn stride_w(&self) -> u32 {
        self.stride_w
    }
    pub fn stride_h(&self) -> u32 {
        self.stride_h
    }
    /// How many slots fit along one row of the atlas.
    pub fn cols_per_row(&self) -> u32 {
        self.cols_per_row
    }
    /// Edge length of the atlas image (square).
    pub fn dim(&self) -> u32 {
        ATLAS_DIM
    }

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

        // Try primary font first; on miss, ask the OS via CoreText
        // which font owns this codepoint and rasterize from those bytes.
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
            // No font has this glyph — cache the miss so we don't retry.
            self.slots.insert(ch, self.tofu_slot);
            return self.tofu_slot;
        };

        let slot = self.next_slot;
        let rect = slot_rect_for(
            slot,
            self.slot_w,
            self.slot_h,
            self.stride_w,
            self.stride_h,
            self.cols_per_row,
        );

        let image = images.get_mut(&self.image).expect("atlas image asset");
        blit_glyph(
            image.data.as_mut().expect("atlas image data"),
            ATLAS_DIM,
            rect,
            self.baseline_atlas_y,
            &raster_data,
            placement.0,
            placement.1,
            placement.2,
            placement.3,
        );

        let layout = layouts.get_mut(&self.layout).expect("atlas layout asset");
        layout.add_texture(rect);

        self.slots.insert(ch, slot);
        self.next_slot += 1;
        slot
    }
}

/// Try to rasterize `ch` from `font_data`. Returns `(alpha_bytes,
/// (width, height, left, top))` matching swash's `Placement`. Returns
/// `None` if the font has no glyph for `ch` or scaling failed.
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

fn slot_rect_for(
    slot: u32,
    slot_w: u32,
    slot_h: u32,
    stride_w: u32,
    stride_h: u32,
    cols_per_row: u32,
) -> URect {
    let col = slot % cols_per_row;
    let row = slot / cols_per_row;
    let x = col * stride_w;
    let y = row * stride_h;
    URect {
        min: UVec2::new(x, y),
        max: UVec2::new(x + slot_w, y + slot_h),
    }
}

/// Blit `raster` (alpha bitmap from swash) into `data` at the slot
/// `rect`, baseline-aligned. White RGB with `alpha` from raster — the
/// sprite tint multiplies to give the final fg color.
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
    if bitmap_w <= 0 || bitmap_h <= 0 {
        return;
    }
    let dst_x_origin = rect.min.x as i32 + bitmap_left;
    let dst_y_origin = rect.min.y as i32 + baseline_y_in_slot as i32 - bitmap_top;
    let stride = atlas_dim as usize * 4;
    let x_min = rect.min.x as i32;
    let x_max = rect.max.x as i32;
    let y_min = rect.min.y as i32;
    let y_max = rect.max.y as i32;

    for sy in 0..bitmap_h {
        let dy = dst_y_origin + sy;
        if dy < y_min || dy >= y_max {
            continue;
        }
        let row_start = (sy * bitmap_w) as usize;
        let dy_stride = dy as usize * stride;
        for sx in 0..bitmap_w {
            let dx = dst_x_origin + sx;
            if dx < x_min || dx >= x_max {
                continue;
            }
            let alpha = raster[row_start + sx as usize];
            if alpha == 0 {
                continue;
            }
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
    if w < 2 * inset + 1 || h < 2 * inset + 1 {
        return;
    }
    let x0 = rect.min.x as usize;
    let y0 = rect.min.y as usize;
    for dy in inset..h - inset {
        let edge_y = dy == inset || dy == h - inset - 1;
        for dx in inset..w - inset {
            let edge_x = dx == inset || dx == w - inset - 1;
            if !(edge_x || edge_y) {
                continue;
            }
            let p = (y0 + dy) * stride + (x0 + dx) * 4;
            data[p] = 255;
            data[p + 1] = 255;
            data[p + 2] = 255;
            data[p + 3] = 200;
        }
    }
}
