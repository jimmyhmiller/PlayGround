//! The wasm32-only execution layer: replay [`CanvasOp`]s onto a real
//! [`web_sys::CanvasRenderingContext2d`].
//!
//! This module is compiled only for `target_arch = "wasm32"` (see the `cfg` in
//! `lib.rs`) because it constructs `web-sys` types that exist solely inside a
//! browser. It contains *no drawing decisions*: every choice was already made by
//! the host-tested [`crate::scene_to_ops`]. Here we only translate each
//! [`CanvasOp`] to the matching Canvas 2D call.

use std::collections::HashMap;

use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use web_sys::{CanvasRenderingContext2d, HtmlCanvasElement, HtmlImageElement, Path2d};

use whiteboard_core::render::RenderScene;

use crate::{scene_to_ops, CanvasOp};

/// A web-canvas backend. Construct it from a `<canvas>` element, register any
/// images you want [`crate::CanvasOp::DrawImage`] to resolve, then call
/// [`WebBackend::render`] with a [`RenderScene`].
#[wasm_bindgen]
pub struct WebBackend {
    ctx: CanvasRenderingContext2d,
    /// Resolves an [`crate::ImageId`] string to a loaded bitmap. Images are
    /// injected out-of-band because the headless core never holds pixels.
    images: HashMap<String, HtmlImageElement>,
}

#[wasm_bindgen]
impl WebBackend {
    /// Create a backend bound to the 2D context of `canvas`.
    ///
    /// # Errors
    /// Returns a `JsValue` error if the canvas has no obtainable 2D context.
    #[wasm_bindgen(constructor)]
    pub fn new(canvas: HtmlCanvasElement) -> Result<WebBackend, JsValue> {
        let ctx = canvas
            .get_context("2d")?
            .ok_or_else(|| JsValue::from_str("canvas has no 2d context"))?
            .dyn_into::<CanvasRenderingContext2d>()?;
        Ok(WebBackend {
            ctx,
            images: HashMap::new(),
        })
    }

    /// Register a bitmap so that a [`crate::CanvasOp::DrawImage`] with this id
    /// paints it. Ids with no registered image are drawn as a no-op.
    #[wasm_bindgen(js_name = setImage)]
    pub fn set_image(&mut self, id: String, image: HtmlImageElement) {
        self.images.insert(id, image);
    }

    /// Parse a [`RenderScene`] from JSON and render it. The JS-facing entry
    /// point on the backend (so a host can `new WebBackend(canvas); be.render(json)`).
    ///
    /// # Errors
    /// Returns a `JsValue` error if `json` does not deserialize into a scene.
    #[wasm_bindgen(js_name = render)]
    pub fn render_json(&self, json: &str) -> Result<(), JsValue> {
        let scene: RenderScene =
            serde_json::from_str(json).map_err(|e| JsValue::from_str(&e.to_string()))?;
        self.render_scene(&scene);
        Ok(())
    }
}

// A plain (non-`#[wasm_bindgen]`) impl block: these methods take core Rust types
// that don't implement `WasmDescribe`, so they must NOT be exported to JS. Rust
// callers in a wasm context use them directly.
impl WebBackend {
    /// Render a [`RenderScene`] by lowering it to [`CanvasOp`]s and executing each.
    pub fn render_scene(&self, scene: &RenderScene) {
        for op in scene_to_ops(scene) {
            self.exec(&op);
        }
    }

    /// Execute a single op. Errors from fallible Canvas calls are intentionally
    /// swallowed per-op: a single malformed path or font must not abort the
    /// whole frame. (The honest, fully-validated path lives in the pure layer;
    /// these `Result`s only surface browser-level rejections.)
    fn exec(&self, op: &CanvasOp) {
        match op {
            CanvasOp::Save => self.ctx.save(),
            CanvasOp::Restore => self.ctx.restore(),
            CanvasOp::Transform { a, b, c, d, e, f } => {
                let _ = self.ctx.transform(*a, *b, *c, *d, *e, *f);
            }
            CanvasOp::Clip { x, y, w, h } => {
                // The matching `Save` was already emitted by `PushClip` (and the
                // matching `Restore` will come from `PopClip`), so we only build
                // the clip rect here — no extra save/restore.
                self.ctx.begin_path();
                self.ctx.rect(*x, *y, *w, *h);
                self.ctx.clip();
            }
            CanvasOp::FillPath { data, fill } => {
                if let Ok(path) = Path2d::new_with_path_string(data) {
                    self.ctx.set_fill_style_str(fill);
                    self.ctx.fill_with_path_2d(&path);
                }
            }
            CanvasOp::StrokePath {
                data,
                stroke,
                line_width,
                dash,
                cap,
                join,
            } => {
                if let Ok(path) = Path2d::new_with_path_string(data) {
                    self.ctx.set_stroke_style_str(stroke);
                    self.ctx.set_line_width(*line_width);
                    self.ctx.set_line_cap(cap.keyword());
                    self.ctx.set_line_join(join.keyword());
                    self.set_dash(dash);
                    self.ctx.stroke_with_path(&path);
                }
            }
            CanvasOp::FillText {
                text,
                x,
                y,
                font,
                fill,
                align,
            } => {
                self.ctx.set_font(font);
                self.ctx.set_text_align(align);
                self.ctx.set_fill_style_str(fill);
                let _ = self.ctx.fill_text(text, *x, *y);
            }
            CanvasOp::DrawImage {
                id,
                x,
                y,
                w,
                h,
                opacity,
            } => self.exec_draw_image(id, *x, *y, *w, *h, *opacity),
        }
    }

    /// Set the canvas dash pattern from explicit on/off lengths (empty = solid).
    fn set_dash(&self, dash: &[f64]) {
        let arr = js_sys::Array::new();
        for d in dash {
            arr.push(&JsValue::from_f64(*d));
        }
        let _ = self.ctx.set_line_dash(&arr);
    }

    fn exec_draw_image(&self, id: &str, x: f64, y: f64, w: f64, h: f64, opacity: f32) {
        // Resolve via the injected image map. Unknown ids are a documented
        // no-op: nothing painted, rendering continues.
        let Some(image) = self.images.get(id) else {
            return;
        };
        self.ctx.save();
        self.ctx.set_global_alpha(opacity as f64);
        let _ = self
            .ctx
            .draw_image_with_html_image_element_and_dw_and_dh(image, x, y, w, h);
        self.ctx.restore();
    }
}

/// The wasm entry point a JS host calls: parse a [`RenderScene`] from JSON and
/// paint it onto `canvas`. Images are not resolved through this entry point
/// (there is no map to inject); use [`WebBackend`] directly for image support.
///
/// # Errors
/// Returns a `JsValue` error if the canvas has no 2D context or the JSON does
/// not deserialize into a [`RenderScene`].
#[wasm_bindgen]
pub fn render_scene_json(canvas: HtmlCanvasElement, json: &str) -> Result<(), JsValue> {
    let scene: RenderScene =
        serde_json::from_str(json).map_err(|e| JsValue::from_str(&e.to_string()))?;
    let backend = WebBackend::new(canvas)?;
    backend.render_scene(&scene);
    Ok(())
}

impl whiteboard_core::render::Backend for WebBackend {
    fn render(&mut self, scene: &RenderScene) {
        self.render_scene(scene);
    }
}
