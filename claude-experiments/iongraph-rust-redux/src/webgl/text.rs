//! Text rendering via Canvas2D textures
//!
//! Renders text to a 2D canvas, then uploads to WebGL as a texture atlas.
//! Uses row-based packing for efficient atlas utilization.

use std::collections::HashMap;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use web_sys::{
    CanvasRenderingContext2d, HtmlCanvasElement, WebGl2RenderingContext, WebGlTexture,
};

use super::scene::FontWeight;

/// Font configuration for text rendering
#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub struct FontConfig {
    pub family: String,
    pub size: u32,
    pub weight: FontWeight,
}

impl FontConfig {
    pub fn new(family: &str, size: u32, weight: FontWeight) -> Self {
        Self {
            family: family.to_string(),
            size,
            weight,
        }
    }

    pub fn monospace(size: u32) -> Self {
        Self::new("monospace", size, FontWeight::Normal)
    }

    pub fn monospace_bold(size: u32) -> Self {
        Self::new("monospace", size, FontWeight::Bold)
    }

    /// Get CSS font string
    pub fn to_css(&self) -> String {
        let weight = match self.weight {
            FontWeight::Normal => "normal",
            FontWeight::Bold => "bold",
        };
        format!("{} {}px {}", weight, self.size, self.family)
    }
}

impl Default for FontConfig {
    fn default() -> Self {
        Self::monospace(11)
    }
}

/// Key for looking up cached text
#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub struct TextCacheKey {
    pub text: String,
    pub font: FontConfig,
}

/// Cached rendered text block
#[derive(Debug, Clone)]
pub struct TextBlock {
    /// UV coordinates in atlas (x, y, width, height) normalized 0-1
    pub uv: [f32; 4],
    /// Pixel dimensions
    pub width: f32,
    pub height: f32,
}

/// Text atlas manager - renders text to Canvas2D and creates WebGL textures
pub struct TextAtlas {
    /// Canvas for rendering text
    canvas: HtmlCanvasElement,
    ctx: CanvasRenderingContext2d,

    /// WebGL texture containing the atlas
    texture: Option<WebGlTexture>,

    /// Current atlas dimensions (in physical pixels)
    atlas_width: u32,
    atlas_height: u32,

    /// Device pixel ratio for high-DPI rendering
    dpr: f64,

    /// Packing state - simple row-based packing (in physical pixels)
    current_row_y: u32,
    current_row_height: u32,
    current_x: u32,

    /// Cache of rendered text blocks
    cache: HashMap<TextCacheKey, TextBlock>,

    /// Whether atlas needs to be uploaded to GPU
    dirty: bool,
}

impl TextAtlas {
    /// Create a new text atlas
    pub fn new(gl: &WebGl2RenderingContext) -> Result<Self, JsValue> {
        let window = web_sys::window().ok_or("No window")?;
        let document = window.document().ok_or("No document")?;

        // Get device pixel ratio for high-DPI rendering
        let dpr = window.device_pixel_ratio().max(1.0);

        let canvas = document
            .create_element("canvas")?
            .dyn_into::<HtmlCanvasElement>()?;

        // Start with reasonable atlas size (scaled by DPR for high-DPI)
        let base_size = 2048;
        let atlas_width = (base_size as f64 * dpr) as u32;
        let atlas_height = (base_size as f64 * dpr) as u32;
        canvas.set_width(atlas_width);
        canvas.set_height(atlas_height);

        let ctx = canvas
            .get_context("2d")?
            .ok_or("No 2d context")?
            .dyn_into::<CanvasRenderingContext2d>()?;

        // Set up for grayscale text rendering
        ctx.set_text_baseline("top");

        // Create WebGL texture
        let texture = gl.create_texture();

        // Initialize texture
        if let Some(ref tex) = texture {
            gl.bind_texture(WebGl2RenderingContext::TEXTURE_2D, Some(tex));

            // Set texture parameters
            gl.tex_parameteri(
                WebGl2RenderingContext::TEXTURE_2D,
                WebGl2RenderingContext::TEXTURE_MIN_FILTER,
                WebGl2RenderingContext::LINEAR as i32,
            );
            gl.tex_parameteri(
                WebGl2RenderingContext::TEXTURE_2D,
                WebGl2RenderingContext::TEXTURE_MAG_FILTER,
                WebGl2RenderingContext::LINEAR as i32,
            );
            gl.tex_parameteri(
                WebGl2RenderingContext::TEXTURE_2D,
                WebGl2RenderingContext::TEXTURE_WRAP_S,
                WebGl2RenderingContext::CLAMP_TO_EDGE as i32,
            );
            gl.tex_parameteri(
                WebGl2RenderingContext::TEXTURE_2D,
                WebGl2RenderingContext::TEXTURE_WRAP_T,
                WebGl2RenderingContext::CLAMP_TO_EDGE as i32,
            );
        }

        Ok(Self {
            canvas,
            ctx,
            texture,
            atlas_width,
            atlas_height,
            dpr,
            current_row_y: 0,
            current_row_height: 0,
            current_x: 0,
            cache: HashMap::new(),
            dirty: true,
        })
    }

    /// Get or render a text block, returns UV coordinates and dimensions (in logical pixels)
    pub fn get_or_render(&mut self, text: &str, font: &FontConfig) -> TextBlock {
        let key = TextCacheKey {
            text: text.to_string(),
            font: font.clone(),
        };

        if let Some(cached) = self.cache.get(&key) {
            return cached.clone();
        }

        // Scale font size by DPR for high-DPI rendering
        let scaled_font_size = (font.size as f64 * self.dpr).round() as u32;
        let scaled_font = FontConfig::new(&font.family, scaled_font_size, font.weight.clone());

        // Set font and measure text (in physical pixels)
        self.ctx.set_font(&scaled_font.to_css());
        let metrics = self.ctx.measure_text(text).unwrap();
        let text_width = metrics.width().ceil() as u32;
        let text_height = (scaled_font_size as f32 * 1.4).ceil() as u32; // Line height

        // Add padding for better rendering (in physical pixels)
        let padding = (2.0 * self.dpr).ceil() as u32;
        let phys_width = text_width + padding * 2;
        let phys_height = text_height + padding * 2;

        // Allocate space in atlas (physical pixels)
        let (x, y) = self.allocate(phys_width, phys_height);

        // Clear area and render text
        self.ctx.set_fill_style(&JsValue::from_str("black"));
        self.ctx
            .fill_rect(x as f64, y as f64, phys_width as f64, phys_height as f64);

        self.ctx.set_fill_style(&JsValue::from_str("white"));
        self.ctx
            .fill_text(text, (x + padding) as f64, (y + padding) as f64)
            .unwrap();

        // UV coordinates in physical pixels (0-1 normalized)
        // Dimensions returned in logical pixels for positioning
        let logical_width = phys_width as f32 / self.dpr as f32;
        let logical_height = phys_height as f32 / self.dpr as f32;

        let block = TextBlock {
            uv: [
                x as f32 / self.atlas_width as f32,
                y as f32 / self.atlas_height as f32,
                phys_width as f32 / self.atlas_width as f32,
                phys_height as f32 / self.atlas_height as f32,
            ],
            width: logical_width,
            height: logical_height,
        };

        self.cache.insert(key, block.clone());
        self.dirty = true;

        block
    }

    /// Simple row-based packing allocator
    fn allocate(&mut self, width: u32, height: u32) -> (u32, u32) {
        // Check if fits in current row
        if self.current_x + width > self.atlas_width {
            // Move to next row
            self.current_row_y += self.current_row_height;
            self.current_x = 0;
            self.current_row_height = 0;
        }

        // Check if we need more vertical space
        if self.current_row_y + height > self.atlas_height {
            // Atlas is full - for now, just wrap around (could expand atlas instead)
            // This will cause visual glitches if atlas is truly full
            self.current_row_y = 0;
            self.current_x = 0;
            self.current_row_height = 0;
            self.cache.clear();
        }

        let x = self.current_x;
        let y = self.current_row_y;

        self.current_x += width;
        self.current_row_height = self.current_row_height.max(height);

        (x, y)
    }

    /// Upload atlas to GPU if dirty
    pub fn upload(&mut self, gl: &WebGl2RenderingContext) {
        if !self.dirty {
            return;
        }

        if let Some(ref tex) = self.texture {
            gl.bind_texture(WebGl2RenderingContext::TEXTURE_2D, Some(tex));

            // Upload canvas as texture using texImage2D with canvas source
            // WebGL2 method: tex_image_2d_with_u32_and_u32_and_html_canvas_element
            gl.tex_image_2d_with_u32_and_u32_and_html_canvas_element(
                WebGl2RenderingContext::TEXTURE_2D,
                0,
                WebGl2RenderingContext::RGBA as i32,
                WebGl2RenderingContext::RGBA,
                WebGl2RenderingContext::UNSIGNED_BYTE,
                &self.canvas,
            )
            .unwrap();
        }

        self.dirty = false;
    }

    /// Bind the atlas texture to a texture unit
    pub fn bind(&self, gl: &WebGl2RenderingContext, unit: u32) {
        gl.active_texture(WebGl2RenderingContext::TEXTURE0 + unit);
        gl.bind_texture(WebGl2RenderingContext::TEXTURE_2D, self.texture.as_ref());
    }

    /// Clear cache and reset allocation (call when switching graphs)
    pub fn clear_cache(&mut self) {
        self.cache.clear();
        self.current_row_y = 0;
        self.current_row_height = 0;
        self.current_x = 0;

        // Clear canvas
        self.ctx.set_fill_style(&JsValue::from_str("black"));
        self.ctx.fill_rect(
            0.0,
            0.0,
            self.atlas_width as f64,
            self.atlas_height as f64,
        );

        self.dirty = true;
    }

    /// Get texture handle for external use
    pub fn texture(&self) -> Option<&WebGlTexture> {
        self.texture.as_ref()
    }

    /// Measure text without caching (for layout calculations)
    pub fn measure_text(&self, text: &str, font: &FontConfig) -> (f32, f32) {
        self.ctx.set_font(&font.to_css());
        let metrics = self.ctx.measure_text(text).unwrap();
        let width = metrics.width() as f32;
        let height = font.size as f32 * 1.4;
        (width, height)
    }
}

/// Pre-computed character metrics for monospace fonts
pub struct MonospaceMetrics {
    pub char_width: f32,
    pub line_height: f32,
}

impl MonospaceMetrics {
    /// Standard metrics matching the SVG renderer
    pub fn standard() -> Self {
        Self {
            char_width: 7.0,
            line_height: 14.0,
        }
    }

    /// Measure a string width
    pub fn text_width(&self, text: &str) -> f32 {
        text.len() as f32 * self.char_width
    }
}
