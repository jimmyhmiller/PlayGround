//! WASM bindings for WebGL-based rendering
//!
//! Provides a WebGL viewer that renders directly to a canvas element
//! with full interactivity (pan, zoom, selection, hover).

#![cfg(all(feature = "webgl", target_arch = "wasm32"))]

use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use web_sys::{HtmlCanvasElement, MouseEvent, WheelEvent, KeyboardEvent};

use crate::compilers::ion::schema::IonJSON;
use crate::compilers::universal::pass_to_universal;
use crate::graph::{Graph, GraphOptions};
use crate::webgl::interaction::{EventHandler, KeyAction};
use crate::webgl::provider::WebGLLayoutProvider;
use crate::webgl::renderer::WebGLRenderer;
use crate::webgl::scene::Scene;

/// WebGL-based IonGraph viewer
///
/// Unlike the SVG viewer which returns strings, this viewer renders
/// directly to a canvas element and handles all interactivity.
#[wasm_bindgen]
pub struct WebGLIonGraphViewer {
    /// Parsed IonJSON data
    data: IonJSON,

    /// WebGL renderer
    renderer: WebGLRenderer,

    /// Event handler for interactivity
    event_handler: EventHandler,

    /// Currently rendered scene
    scene: Scene,

    /// Current function index
    current_func: usize,

    /// Current pass index
    current_pass: usize,
}

#[wasm_bindgen]
impl WebGLIonGraphViewer {
    /// Create a new WebGL viewer attached to a canvas
    ///
    /// # Arguments
    /// * `canvas_id` - ID of the canvas element to render to
    /// * `ion_json` - JSON string containing IonJSON data
    #[wasm_bindgen(constructor)]
    pub fn new(canvas_id: &str, ion_json: &str) -> Result<WebGLIonGraphViewer, JsValue> {
        // Parse JSON data
        let data: IonJSON = serde_json::from_str(ion_json)
            .map_err(|e| JsValue::from_str(&format!("JSON parse error: {}", e)))?;

        // Create WebGL renderer
        let renderer = WebGLRenderer::new(canvas_id)?;

        // Create event handler
        let (width, height) = renderer.size();
        let event_handler = EventHandler::new(width as f32, height as f32);

        let mut viewer = WebGLIonGraphViewer {
            data,
            renderer,
            event_handler,
            scene: Scene::new(),
            current_func: 0,
            current_pass: 0,
        };

        // Render initial pass
        if !viewer.data.functions.is_empty() && !viewer.data.functions[0].passes.is_empty() {
            viewer.render_current_pass()?;
        }

        Ok(viewer)
    }

    /// Render the current pass
    fn render_current_pass(&mut self) -> Result<(), JsValue> {
        if self.current_func >= self.data.functions.len() {
            return Err(JsValue::from_str("Invalid function index"));
        }
        let func = &self.data.functions[self.current_func];

        if self.current_pass >= func.passes.len() {
            return Err(JsValue::from_str("Invalid pass index"));
        }
        let pass = &func.passes[self.current_pass];

        // Convert to Universal IR
        let universal_ir = pass_to_universal(pass, &func.name);

        // Create WebGL layout provider
        let mut layout_provider = WebGLLayoutProvider::new();

        let options = GraphOptions {
            sample_counts: None,
            instruction_palette: None,
        };

        // Create and layout graph
        let mut graph = Graph::new(layout_provider, universal_ir, options);
        let (nodes_by_layer, layer_heights, track_heights) = graph.layout();
        graph.render(nodes_by_layer, layer_heights, track_heights);

        // Build scene from the graph container
        // We need to pass the root element to the provider since it wasn't stored
        let graph_container = graph.graph_container;
        layout_provider = graph.layout_provider;
        layout_provider.build_scene_from_root(&graph_container);
        self.scene = layout_provider.take_scene();

        // Update hit tester
        self.event_handler.update_hit_tester(&self.scene);

        // Fit view to content - account for content origin
        // The scene bounds might not start at (0, 0), so we need to include the full extent
        let content_width = self.scene.bounds.right() + 20.0;
        let content_height = self.scene.bounds.bottom() + 20.0;
        self.event_handler.viewport.fit_to_content(content_width, content_height, 20.0);

        // Render
        self.draw();

        Ok(())
    }

    /// Draw the current scene
    fn draw(&mut self) {
        self.renderer.render(
            &self.scene,
            &self.event_handler.viewport,
            &self.event_handler.interaction,
        );
    }

    /// Navigate to a specific function and pass
    pub fn navigate(&mut self, func_idx: usize, pass_idx: usize) -> Result<(), JsValue> {
        if func_idx >= self.data.functions.len() {
            return Err(JsValue::from_str(&format!(
                "Function index {} out of range",
                func_idx
            )));
        }

        let func = &self.data.functions[func_idx];
        if pass_idx >= func.passes.len() {
            return Err(JsValue::from_str(&format!(
                "Pass index {} out of range",
                pass_idx
            )));
        }

        self.current_func = func_idx;
        self.current_pass = pass_idx;
        self.event_handler.interaction.clear_selection();
        self.render_current_pass()
    }

    /// Go to next pass
    pub fn next_pass(&mut self) -> Result<(), JsValue> {
        let func = &self.data.functions[self.current_func];
        if self.current_pass + 1 < func.passes.len() {
            self.current_pass += 1;
            self.render_current_pass()?;
        }
        Ok(())
    }

    /// Go to previous pass
    pub fn prev_pass(&mut self) -> Result<(), JsValue> {
        if self.current_pass > 0 {
            self.current_pass -= 1;
            self.render_current_pass()?;
        }
        Ok(())
    }

    /// Go to next function
    pub fn next_function(&mut self) -> Result<(), JsValue> {
        if self.current_func + 1 < self.data.functions.len() {
            self.current_func += 1;
            self.current_pass = 0;
            self.render_current_pass()?;
        }
        Ok(())
    }

    /// Go to previous function
    pub fn prev_function(&mut self) -> Result<(), JsValue> {
        if self.current_func > 0 {
            self.current_func -= 1;
            self.current_pass = 0;
            self.render_current_pass()?;
        }
        Ok(())
    }

    /// Handle mouse down event
    pub fn on_mouse_down(&mut self, event: &MouseEvent) {
        self.event_handler.handle_mouse_down(
            event.offset_x() as f32,
            event.offset_y() as f32,
            event.button(),
            event.ctrl_key() || event.meta_key(),
            event.shift_key(),
        );

        if self.event_handler.take_needs_redraw() {
            self.draw();
        }
    }

    /// Handle mouse up event
    pub fn on_mouse_up(&mut self, event: &MouseEvent) {
        self.event_handler.handle_mouse_up(
            event.offset_x() as f32,
            event.offset_y() as f32,
            event.button(),
        );

        if self.event_handler.take_needs_redraw() {
            self.draw();
        }
    }

    /// Handle mouse move event
    pub fn on_mouse_move(&mut self, event: &MouseEvent) {
        self.event_handler.handle_mouse_move(
            event.offset_x() as f32,
            event.offset_y() as f32,
        );

        if self.event_handler.take_needs_redraw() {
            self.draw();
        }
    }

    /// Handle wheel event
    pub fn on_wheel(&mut self, event: &WheelEvent) {
        self.event_handler.handle_wheel(
            event.offset_x() as f32,
            event.offset_y() as f32,
            event.delta_x() as f32,
            event.delta_y() as f32,
            event.ctrl_key() || event.meta_key(),
        );

        if self.event_handler.take_needs_redraw() {
            self.draw();
        }
    }

    /// Handle key down event, returns true if navigation occurred
    pub fn on_key_down(&mut self, event: &KeyboardEvent) -> Result<bool, JsValue> {
        let action = self.event_handler.handle_key_down(
            &event.key(),
            event.ctrl_key() || event.meta_key(),
            event.shift_key(),
        );

        match action {
            KeyAction::NextPass => {
                self.next_pass()?;
                Ok(true)
            }
            KeyAction::PrevPass => {
                self.prev_pass()?;
                Ok(true)
            }
            KeyAction::NextFunction => {
                self.next_function()?;
                Ok(true)
            }
            KeyAction::PrevFunction => {
                self.prev_function()?;
                Ok(true)
            }
            KeyAction::Handled => {
                if self.event_handler.take_needs_redraw() {
                    self.draw();
                }
                Ok(false)
            }
            KeyAction::Unhandled => Ok(false),
        }
    }

    /// Handle canvas resize
    pub fn on_resize(&mut self, width: u32, height: u32) {
        self.renderer.resize(width, height);
        self.event_handler.handle_resize(width as f32, height as f32);
        self.draw();
    }

    /// Reset view to fit content
    pub fn reset_view(&mut self) {
        let content_width = self.scene.bounds.right() + 20.0;
        let content_height = self.scene.bounds.bottom() + 20.0;
        self.event_handler.viewport.fit_to_content(content_width, content_height, 20.0);
        self.draw();
    }

    /// Get current function index
    pub fn get_current_function(&self) -> usize {
        self.current_func
    }

    /// Get current pass index
    pub fn get_current_pass(&self) -> usize {
        self.current_pass
    }

    /// Get function count
    pub fn get_function_count(&self) -> usize {
        self.data.functions.len()
    }

    /// Get pass count for current function
    pub fn get_pass_count(&self) -> usize {
        if self.current_func < self.data.functions.len() {
            self.data.functions[self.current_func].passes.len()
        } else {
            0
        }
    }

    /// Get function name by index
    pub fn get_function_name(&self, func_idx: usize) -> Result<String, JsValue> {
        if func_idx >= self.data.functions.len() {
            return Err(JsValue::from_str("Function index out of range"));
        }
        Ok(self.data.functions[func_idx].name.clone())
    }

    /// Get pass name by index
    pub fn get_pass_name(&self, func_idx: usize, pass_idx: usize) -> Result<String, JsValue> {
        if func_idx >= self.data.functions.len() {
            return Err(JsValue::from_str("Function index out of range"));
        }
        let func = &self.data.functions[func_idx];
        if pass_idx >= func.passes.len() {
            return Err(JsValue::from_str("Pass index out of range"));
        }
        Ok(func.passes[pass_idx].name.clone())
    }

    /// Get currently selected block IDs
    pub fn get_selected_blocks(&self) -> Vec<String> {
        self.event_handler
            .interaction
            .selected_blocks
            .iter()
            .cloned()
            .collect()
    }

    /// Get hovered block ID
    pub fn get_hovered_block(&self) -> Option<String> {
        self.event_handler.interaction.hovered_block.clone()
    }

    /// Get zoom level
    pub fn get_zoom(&self) -> f32 {
        self.event_handler.viewport.scale
    }

    /// Set zoom level
    pub fn set_zoom(&mut self, scale: f32) {
        self.event_handler.viewport.scale = scale.clamp(0.1, 10.0);
        self.draw();
    }

    /// Debug: Get scene info
    pub fn debug_scene_info(&self) -> String {
        let mut block_ids: Vec<String> = Vec::new();
        for call in &self.scene.draw_calls {
            if let crate::webgl::scene::DrawCallOwner::Block { block_id } = &call.owner {
                if !block_ids.contains(block_id) {
                    block_ids.push(block_id.clone());
                }
            }
        }
        format!(
            "Scene: {} draw calls, {} blocks ({}), bounds: ({:.0}, {:.0}, {:.0}, {:.0})",
            self.scene.draw_calls.len(),
            block_ids.len(),
            block_ids.join(", "),
            self.scene.bounds.x,
            self.scene.bounds.y,
            self.scene.bounds.width,
            self.scene.bounds.height
        )
    }
}
