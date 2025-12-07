// WASM bindings for browser-based rendering

use wasm_bindgen::prelude::*;
use crate::graph::{Graph, GraphOptions};
use crate::compilers::ion::schema::IonJSON;
use crate::compilers::universal::pass_to_universal;
use crate::pure_svg_text_layout_provider::PureSVGTextLayoutProvider;
use crate::layout_provider::LayoutProvider;

// Enable better panic messages in WASM
#[wasm_bindgen(start)]
pub fn init() {
    console_error_panic_hook::set_once();
}

/// Stateful WASM viewer that caches parsed IonJSON data
#[wasm_bindgen]
pub struct IonGraphViewer {
    data: IonJSON,
}

#[wasm_bindgen]
impl IonGraphViewer {
    /// Create a new viewer instance with parsed IonJSON data
    ///
    /// # Arguments
    /// * `ion_json` - JSON string containing IonJSON data (parsed once and cached)
    ///
    /// # Returns
    /// IonGraphViewer instance on success, error message on failure
    #[wasm_bindgen(constructor)]
    pub fn new(ion_json: &str) -> Result<IonGraphViewer, JsValue> {
        let data: IonJSON = serde_json::from_str(ion_json)
            .map_err(|e| JsValue::from_str(&format!("JSON parse error: {}", e)))?;
        Ok(IonGraphViewer { data })
    }

    /// Render a specific pass to SVG string
    ///
    /// # Arguments
    /// * `func_idx` - Index of the function to render
    /// * `pass_idx` - Index of the pass to render
    ///
    /// # Returns
    /// SVG string on success, error message on failure
    pub fn render_pass(&self, func_idx: usize, pass_idx: usize) -> Result<String, JsValue> {
        // Validate function index
        if func_idx >= self.data.functions.len() {
            return Err(JsValue::from_str(&format!(
                "Function index {} out of range (max: {})",
                func_idx,
                self.data.functions.len() - 1
            )));
        }
        let func = &self.data.functions[func_idx];

        // Validate pass index
        if pass_idx >= func.passes.len() {
            return Err(JsValue::from_str(&format!(
                "Pass index {} out of range for function '{}' (max: {})",
                pass_idx,
                func.name,
                func.passes.len() - 1
            )));
        }
        let pass = &func.passes[pass_idx];

        // Convert to Universal IR
        let universal_ir = pass_to_universal(pass, &func.name);

        // Create layout provider and graph
        let mut layout_provider = PureSVGTextLayoutProvider::new();
        let options = GraphOptions {
            sample_counts: None,
            instruction_palette: None,
        };

        let mut graph = Graph::new(layout_provider, universal_ir, options);

        // Layout and render
        let (nodes_by_layer, layer_heights, track_heights) = graph.layout();
        graph.render(nodes_by_layer, layer_heights, track_heights);

        // Extract layout provider and create SVG
        layout_provider = graph.layout_provider;
        let mut svg_root = layout_provider.create_svg_element("svg");
        layout_provider.set_attribute(&mut svg_root, "xmlns", "http://www.w3.org/2000/svg");

        let width = (graph.size.x + 40.0).ceil() as i32;
        let height = (graph.size.y + 40.0).ceil() as i32;

        layout_provider.set_attribute(&mut svg_root, "width", &width.to_string());
        layout_provider.set_attribute(&mut svg_root, "height", &height.to_string());
        layout_provider.set_attribute(&mut svg_root, "viewBox", &format!("0 0 {} {}", width, height));
        layout_provider.append_child(&mut svg_root, graph.graph_container);

        Ok(layout_provider.to_svg_string(&svg_root))
    }

    /// Get the number of functions in the IonJSON data
    pub fn get_function_count(&self) -> usize {
        self.data.functions.len()
    }

    /// Get the number of passes for a specific function
    pub fn get_pass_count(&self, func_idx: usize) -> Result<usize, JsValue> {
        if func_idx >= self.data.functions.len() {
            return Err(JsValue::from_str(&format!(
                "Function index {} out of range (max: {})",
                func_idx,
                self.data.functions.len() - 1
            )));
        }

        Ok(self.data.functions[func_idx].passes.len())
    }

    /// Get function name by index
    pub fn get_function_name(&self, func_idx: usize) -> Result<String, JsValue> {
        if func_idx >= self.data.functions.len() {
            return Err(JsValue::from_str(&format!(
                "Function index {} out of range (max: {})",
                func_idx,
                self.data.functions.len() - 1
            )));
        }

        Ok(self.data.functions[func_idx].name.clone())
    }

    /// Get pass name by index
    pub fn get_pass_name(&self, func_idx: usize, pass_idx: usize) -> Result<String, JsValue> {
        if func_idx >= self.data.functions.len() {
            return Err(JsValue::from_str(&format!(
                "Function index {} out of range (max: {})",
                func_idx,
                self.data.functions.len() - 1
            )));
        }

        let func = &self.data.functions[func_idx];

        if pass_idx >= func.passes.len() {
            return Err(JsValue::from_str(&format!(
                "Pass index {} out of range for function '{}' (max: {})",
                pass_idx,
                func.name,
                func.passes.len() - 1
            )));
        }

        Ok(func.passes[pass_idx].name.clone())
    }
}
