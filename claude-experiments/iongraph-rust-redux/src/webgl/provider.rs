//! WebGL LayoutProvider implementation
//!
//! Implements the LayoutProvider trait for WebGL rendering, allowing the
//! existing Graph code to work unchanged with WebGL output.

use std::collections::HashMap;

use crate::layout_provider::{DOMRect, LayoutProvider, Vec2};

use super::element::{ElementId, Rect, WebGLElement};
use super::scene::{colors, z_order, DrawCall, DrawCallOwner, FontWeight, Primitive, Scene, TextAlign};
use super::tessellation::tessellate_stroke;
use super::text::MonospaceMetrics;

/// Instruction row data for rendering
struct InstructionRow {
    num_text: Option<String>,
    opcode_text: Option<String>,
    type_text: Option<String>,
    has_movable: bool,
    has_guard: bool,
    has_rob: bool,
}

/// WebGL-based LayoutProvider implementation
pub struct WebGLLayoutProvider {
    /// All elements by ID
    elements: HashMap<ElementId, WebGLElement>,

    /// Next element ID
    next_id: ElementId,

    /// Root element ID
    root: Option<ElementId>,

    /// Generated scene for rendering
    pub scene: Scene,

    /// Monospace font metrics for sizing
    metrics: MonospaceMetrics,

    /// Track current block being rendered (for ownership)
    current_block_id: Option<String>,
}

impl WebGLLayoutProvider {
    pub fn new() -> Self {
        Self {
            elements: HashMap::new(),
            next_id: 0,
            root: None,
            scene: Scene::new(),
            metrics: MonospaceMetrics::standard(),
            current_block_id: None,
        }
    }

    /// Get the generated scene
    pub fn scene(&self) -> &Scene {
        &self.scene
    }

    /// Take ownership of the scene
    pub fn take_scene(&mut self) -> Scene {
        std::mem::take(&mut self.scene)
    }

    /// Build scene from element tree
    /// Call this after Graph::render() to convert elements to draw calls
    /// Pass the graph_container element that owns the rendered content
    pub fn build_scene_from_root(&mut self, root: &WebGLElement) {
        self.scene.clear();
        self.build_scene_from_element(root, 0.0, 0.0);
        self.scene.sort_for_rendering();

        // Debug: summarize scene contents
        let mut rect_count = 0;
        let mut stroked_rect_count = 0;
        let mut text_count = 0;
        let mut path_count = 0;
        for call in &self.scene.draw_calls {
            match &call.primitive {
                super::scene::Primitive::Rect { .. } => rect_count += 1,
                super::scene::Primitive::StrokedRect { .. } => stroked_rect_count += 1,
                super::scene::Primitive::Text { .. } => text_count += 1,
                super::scene::Primitive::Path { .. } => path_count += 1,
                _ => {}
            }
        }
        web_sys::console::log_1(&format!(
            "Scene contents: {} rects, {} stroked rects, {} texts, {} paths",
            rect_count, stroked_rect_count, text_count, path_count
        ).into());
    }

    /// Legacy method - builds scene from stored root (may not work if root wasn't stored)
    pub fn build_scene(&mut self) {
        self.scene.clear();

        if let Some(root_id) = self.root {
            self.build_scene_recursive(root_id, 0.0, 0.0);
        }

        self.scene.sort_for_rendering();
    }

    fn build_scene_from_element(&mut self, element: &WebGLElement, parent_x: f32, parent_y: f32) {
        // Calculate absolute position
        let x = parent_x + element.bounds.x;
        let y = parent_y + element.bounds.y;

        // Handle specific element types based on CSS classes
        if element.has_class("ig-block") {
            self.build_block(element, x, y);
        } else if element.has_class("ig-arrow") {
            self.build_arrow(element);
        } else if element.has_class("ig-arrowhead") {
            self.build_arrowhead(element);
        }

        // Recurse to children (look them up in stored elements)
        for child_id in &element.children {
            if let Some(child) = self.elements.get(child_id) {
                let child = child.clone();
                self.build_scene_from_element(&child, x, y);
            }
        }
    }

    fn build_scene_recursive(&mut self, element_id: ElementId, parent_x: f32, parent_y: f32) {
        // Get element data (clone to avoid borrow issues)
        let element = match self.elements.get(&element_id) {
            Some(e) => e.clone(),
            None => return,
        };

        // Calculate absolute position
        let x = parent_x + element.bounds.x;
        let y = parent_y + element.bounds.y;

        // Handle specific element types based on CSS classes
        if element.has_class("ig-block") {
            self.build_block(&element, x, y);
        } else if element.has_class("ig-arrow") {
            self.build_arrow(&element);
        }

        // Recurse to children
        for child_id in &element.children {
            self.build_scene_recursive(*child_id, x, y);
        }
    }

    fn build_block(&mut self, element: &WebGLElement, x: f32, y: f32) {
        let width = element.bounds.width;
        let height = element.bounds.height;
        let block_id = element.block_id().unwrap_or("unknown").to_string();

        web_sys::console::log_1(&format!(
            "Building block {}: pos=({}, {}), size=({}, {})",
            block_id, x, y, width, height
        ).into());

        // Determine header color based on block type
        let header_color = if element.is_loop_header() {
            colors::LOOP_HEADER
        } else if element.is_backedge() {
            colors::BACKEDGE_BLOCK
        } else {
            colors::BLOCK_HEADER
        };

        // Block background
        self.scene.add(DrawCall::block(
            Primitive::Rect {
                rect: Rect::new(x, y, width, height),
                color: colors::BLOCK_BACKGROUND,
                corner_radius: 0.0,
            },
            z_order::BLOCK_BACKGROUND,
            &block_id,
        ));

        // Block border
        self.scene.add(DrawCall::block(
            Primitive::StrokedRect {
                rect: Rect::new(x, y, width, height),
                color: colors::BLACK,
                stroke_width: 1.0,
            },
            z_order::BLOCK_BORDER,
            &block_id,
        ));

        // Header background (assuming 28px header height like SVG version)
        let header_height = 28.0;
        self.scene.add(DrawCall::block(
            Primitive::Rect {
                rect: Rect::new(x, y, width, header_height),
                color: header_color,
                corner_radius: 0.0,
            },
            z_order::BLOCK_HEADER,
            &block_id,
        ));

        // Header text (block ID) - WHITE on dark header background
        self.scene.add(DrawCall::block(
            Primitive::Text {
                x: x + 8.0,
                y: y + 6.0,
                text: block_id.clone(),
                color: colors::WHITE,
                font_size: 11.0,
                font_weight: FontWeight::Bold,
                align: TextAlign::Left,
            },
            z_order::HEADER_TEXT,
            &block_id,
        ));

        // Render instruction content from child elements
        self.render_block_content(element, x, y, &block_id);
    }

    /// Render instruction content from block's child elements
    /// Adapted from pure_svg_text_layout_provider.rs
    fn render_block_content(&mut self, element: &WebGLElement, block_x: f32, block_y: f32, block_id: &str) {
        const CHARACTER_WIDTH: f32 = 7.0;
        const LINE_HEIGHT: f32 = 14.0;
        const PADDING: f32 = 8.0;
        const START_Y_OFFSET: f32 = 28.0 + PADDING + 4.0; // Match pure SVG: 28 + 8 + 4 = 40

        // Find all instruction rows (elements with class "ig-ins")
        let rows = self.find_instruction_rows(element);

        if rows.is_empty() {
            web_sys::console::log_1(&format!("Block {}: no instruction rows found", block_id).into());
            return;
        }

        // Measure maximum width of each column
        let mut max_num_width: f32 = 0.0;
        let mut max_opcode_width: f32 = 0.0;

        for row in &rows {
            if let Some(num_text) = &row.num_text {
                let width = num_text.chars().count() as f32 * CHARACTER_WIDTH;
                max_num_width = max_num_width.max(width);
            }
            if let Some(opcode_text) = &row.opcode_text {
                let width = opcode_text.chars().count() as f32 * CHARACTER_WIDTH;
                max_opcode_width = max_opcode_width.max(width);
            }
        }

        web_sys::console::log_1(&format!(
            "Block {}: {} rows, max_num_width={}, max_opcode_width={}",
            block_id, rows.len(), max_num_width, max_opcode_width
        ).into());

        // Log first few rows to debug
        for (i, row) in rows.iter().take(3).enumerate() {
            web_sys::console::log_1(&format!(
                "  Row {}: num={:?}, opcode={:?}, type={:?}",
                i, row.num_text, row.opcode_text, row.type_text
            ).into());
        }

        // Calculate column positions with proper spacing
        let num_x = block_x + PADDING;
        let opcode_x = block_x + PADDING + max_num_width + 8.0;
        let type_x = opcode_x + max_opcode_width + 8.0;

        // Start Y below header (matches pure SVG: 28 + 8 + 4 = 40)
        let mut current_y = block_y + START_Y_OFFSET;

        for row in &rows {
            // Determine text color based on attributes
            let text_color = if row.has_movable {
                [0.063, 0.282, 0.686, 1.0] // #1048af
            } else if row.has_rob {
                [0.267, 0.267, 0.267, 1.0] // #444
            } else {
                colors::BLACK
            };

            // Render instruction number (gray)
            if let Some(num_text) = &row.num_text {
                self.scene.add(DrawCall::block(
                    Primitive::Text {
                        x: num_x,
                        y: current_y,
                        text: num_text.clone(),
                        color: [0.467, 0.467, 0.467, 1.0], // #777
                        font_size: 11.0,
                        font_weight: FontWeight::Normal,
                        align: TextAlign::Left,
                    },
                    z_order::INSTRUCTION_TEXT,
                    block_id,
                ));
            }

            // Render opcode
            if let Some(opcode_text) = &row.opcode_text {
                self.scene.add(DrawCall::block(
                    Primitive::Text {
                        x: opcode_x,
                        y: current_y,
                        text: opcode_text.clone(),
                        color: text_color,
                        font_size: 11.0,
                        font_weight: FontWeight::Normal,
                        align: TextAlign::Left,
                    },
                    z_order::INSTRUCTION_TEXT,
                    block_id,
                ));
            }

            // Render type (blue) - skip if empty or "None"
            if let Some(type_text) = &row.type_text {
                if !type_text.is_empty() && type_text != "None" {
                    self.scene.add(DrawCall::block(
                        Primitive::Text {
                            x: type_x,
                            y: current_y,
                            text: type_text.clone(),
                            color: [0.063, 0.282, 0.686, 1.0], // #1048af blue
                            font_size: 11.0,
                            font_weight: FontWeight::Normal,
                            align: TextAlign::Left,
                        },
                        z_order::INSTRUCTION_TEXT,
                        block_id,
                    ));
                }
            }

            current_y += LINE_HEIGHT;
        }
    }

    /// Find instruction rows in the element tree
    fn find_instruction_rows(&self, element: &WebGLElement) -> Vec<InstructionRow> {
        let mut rows = Vec::new();
        self.find_instruction_rows_recursive(element, &mut rows);
        rows
    }

    fn find_instruction_rows_recursive(&self, element: &WebGLElement, rows: &mut Vec<InstructionRow>) {
        // Check if this element is an instruction row
        if element.has_class("ig-ins") {
            let mut row = InstructionRow {
                num_text: None,
                opcode_text: None,
                type_text: None,
                has_movable: element.has_class("ig-ins-att-Movable"),
                has_guard: element.has_class("ig-ins-att-Guard"),
                has_rob: element.has_class("ig-ins-att-RecoveredOnBailout"),
            };

            // Look through children (table cells) to find text
            for child_id in &element.children {
                if let Some(child) = self.elements.get(child_id) {
                    if child.has_class("ig-ins-num") {
                        row.num_text = child.text_content.clone();
                    } else if child.has_class("ig-ins-type") {
                        row.type_text = child.text_content.clone();
                    } else if child.node_type == "td"
                        && !child.has_class("ig-ins-num")
                        && !child.has_class("ig-ins-type")
                    {
                        // Opcode is the td cell without a specific class
                        // Convert <- to ← and -> to → for consistency
                        row.opcode_text = child
                            .text_content
                            .as_ref()
                            .map(|s| s.replace("<-", "\u{2190}").replace("->", "\u{2192}"));
                    }
                }
            }

            rows.push(row);
        }

        // Recurse into children
        for child_id in &element.children {
            if let Some(child) = self.elements.get(child_id) {
                self.find_instruction_rows_recursive(child, rows);
            }
        }
    }

    fn build_arrow(&mut self, element: &WebGLElement) {
        // Get the SVG path data
        if let Some(d) = element.get_attribute("d") {
            let tessellated = tessellate_stroke(d, 1.0);

            if !tessellated.is_empty() {
                self.scene.add(DrawCall::new(
                    Primitive::Path {
                        vertices: tessellated.vertices,
                        indices: tessellated.indices,
                        color: colors::ARROW_STROKE,
                    },
                    z_order::ARROW_PATH,
                    DrawCallOwner::Background,
                ));
            }
        }
    }

    fn build_arrowhead(&mut self, element: &WebGLElement) {
        // Arrowhead is a filled triangle defined by path "M 0 0 L -5 7.5 L 5 7.5 Z"
        // with transform "translate(x, y) rotate(deg)"
        if let Some(_d) = element.get_attribute("d") {
            if let Some(transform) = element.get_attribute("transform") {
                // Parse transform to get translate and rotate
                let (tx, ty, rot) = parse_transform(&transform);

                // Parse the triangle path and apply transform
                // The path is "M 0 0 L -5 7.5 L 5 7.5 Z" (a triangle)
                // We need to rotate and translate these points
                let size = 5.0f32;
                let half_height = size * 1.5;

                // Original triangle points (pointing down at 0 rotation)
                let p0 = [0.0f32, 0.0f32];
                let p1 = [-size, half_height];
                let p2 = [size, half_height];

                // Apply rotation (in degrees)
                let rot_rad = rot.to_radians();
                let cos_r = rot_rad.cos();
                let sin_r = rot_rad.sin();

                let rotate = |p: [f32; 2]| -> [f32; 2] {
                    [
                        p[0] * cos_r - p[1] * sin_r,
                        p[0] * sin_r + p[1] * cos_r,
                    ]
                };

                let p0_rot = rotate(p0);
                let p1_rot = rotate(p1);
                let p2_rot = rotate(p2);

                // Apply translation
                let vertices = [
                    [p0_rot[0] + tx, p0_rot[1] + ty],
                    [p1_rot[0] + tx, p1_rot[1] + ty],
                    [p2_rot[0] + tx, p2_rot[1] + ty],
                ];

                self.scene.add(DrawCall::new(
                    Primitive::Triangle {
                        points: [vertices[0], vertices[1], vertices[2]],
                        color: colors::BLACK,
                    },
                    z_order::ARROWHEAD,
                    DrawCallOwner::Background,
                ));
            }
        }
    }

    /// Store an element in the internal map
    fn store_element(&mut self, element: WebGLElement) -> ElementId {
        let id = element.id;
        self.elements.insert(id, element);
        id
    }
}

impl Default for WebGLLayoutProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl LayoutProvider for WebGLLayoutProvider {
    type Element = WebGLElement;

    fn create_element(&mut self, tag: &str) -> Box<Self::Element> {
        let id = self.next_id;
        self.next_id += 1;
        Box::new(WebGLElement::new(id, tag))
    }

    fn create_svg_element(&mut self, tag: &str) -> Box<Self::Element> {
        self.create_element(tag)
    }

    fn append_child(&mut self, parent: &mut Self::Element, child: Box<Self::Element>) {
        let child_id = child.id;

        // Store child in map with parent reference
        let mut child = *child;
        child.parent = Some(parent.id);
        self.store_element(child);

        // Add child ID to parent's children list
        parent.children.push(child_id);

        // If this is the first child of root, track it
        if self.root.is_none() && parent.node_type == "div" {
            self.root = Some(parent.id);
        }
    }

    fn set_attribute(&mut self, element: &mut Self::Element, name: &str, value: &str) {
        element.attributes.insert(name.to_string(), value.to_string());

        // Track block ID for ownership
        if name == "data-ig-block-id" {
            self.current_block_id = Some(value.to_string());
        }
    }

    fn set_inner_html(&mut self, element: &mut Self::Element, html: &str) {
        element.text_content = Some(html.to_string());
    }

    fn set_inner_text(&mut self, element: &mut Self::Element, text: &str) {
        element.text_content = Some(text.to_string());
    }

    fn add_class(&mut self, element: &mut Self::Element, class_name: &str) {
        element.class_list.insert(class_name.to_string());
    }

    fn add_classes(&mut self, element: &mut Self::Element, class_names: &[&str]) {
        for name in class_names {
            element.class_list.insert(name.to_string());
        }
    }

    fn remove_class(&mut self, element: &mut Self::Element, class_name: &str) {
        element.class_list.remove(class_name);
    }

    fn toggle_class(
        &mut self,
        element: &mut Self::Element,
        class_name: &str,
        force: Option<bool>,
    ) {
        match force {
            Some(true) => {
                element.class_list.insert(class_name.to_string());
            }
            Some(false) => {
                element.class_list.remove(class_name);
            }
            None => {
                if element.class_list.contains(class_name) {
                    element.class_list.remove(class_name);
                } else {
                    element.class_list.insert(class_name.to_string());
                }
            }
        }
    }

    fn set_style(&mut self, element: &mut Self::Element, property: &str, value: &str) {
        element.style.insert(property.to_string(), value.to_string());

        // Parse position values
        match property {
            "left" => {
                if let Ok(x) = value.trim_end_matches("px").parse::<f32>() {
                    element.bounds.x = x;
                }
            }
            "top" => {
                if let Ok(y) = value.trim_end_matches("px").parse::<f32>() {
                    element.bounds.y = y;
                }
            }
            "width" => {
                if let Ok(w) = value.trim_end_matches("px").parse::<f32>() {
                    element.bounds.width = w;
                }
            }
            "height" => {
                if let Ok(h) = value.trim_end_matches("px").parse::<f32>() {
                    element.bounds.height = h;
                }
            }
            _ => {}
        }
    }

    fn set_css_property(&mut self, element: &mut Self::Element, property: &str, value: &str) {
        self.set_style(element, property, value);
    }

    fn get_bounding_client_rect(&self, element: &Self::Element) -> DOMRect {
        DOMRect {
            x: element.bounds.x as f64,
            y: element.bounds.y as f64,
            width: element.bounds.width as f64,
            height: element.bounds.height as f64,
            left: element.bounds.x as f64,
            right: (element.bounds.x + element.bounds.width) as f64,
            top: element.bounds.y as f64,
            bottom: (element.bounds.y + element.bounds.height) as f64,
        }
    }

    fn get_client_width(&self, element: &Self::Element) -> f64 {
        element.bounds.width as f64
    }

    fn get_client_height(&self, element: &Self::Element) -> f64 {
        element.bounds.height as f64
    }

    fn add_event_listener(
        &mut self,
        _element: &mut Self::Element,
        _event_type: &str,
        _listener: Box<dyn Fn()>,
    ) {
        // Event listeners are handled externally by EventHandler
    }

    fn query_selector(&self, _parent: &Self::Element, _selector: &str) -> bool {
        false
    }

    fn query_selector_all(&self, _parent: &Self::Element, _selector: &str) -> usize {
        0
    }

    fn observe_resize(
        &mut self,
        _element: &Self::Element,
        _callback: Box<dyn Fn(Vec2)>,
    ) -> Box<dyn Fn()> {
        Box::new(|| {})
    }

    fn set_pointer_capture(&mut self, _element: &mut Self::Element, _pointer_id: i32) {}

    fn release_pointer_capture(&mut self, _element: &mut Self::Element, _pointer_id: i32) {}

    fn has_pointer_capture(&self, _element: &Self::Element, _pointer_id: i32) -> bool {
        false
    }

    fn calculate_block_size(
        &self,
        block_id: &str,
        num_instructions: usize,
        has_lir: bool,
        has_samples: bool,
    ) -> Vec2 {
        // Must match pure_svg_text_layout_provider.rs exactly!
        const CHARACTER_WIDTH: f64 = 7.0;
        const LINE_HEIGHT: f64 = 14.0;
        const BLOCK_PADDING: f64 = 8.0;
        const HEADER_HEIGHT: f64 = 30.0;
        const TABLE_HEADER_HEIGHT: f64 = LINE_HEIGHT + 4.0;

        // Calculate header width
        let header_text_len = 6 + block_id.len() + 20;
        let header_width = header_text_len as f64 * CHARACTER_WIDTH;

        // Calculate table width based on column estimates
        let avg_id_width = 3.0 * CHARACTER_WIDTH;
        let avg_opcode_width = 50.0 * CHARACTER_WIDTH;
        let avg_type_width = 25.0 * CHARACTER_WIDTH;
        let avg_sample_width = 6.0 * CHARACTER_WIDTH;

        let table_width = if has_lir && has_samples {
            avg_id_width + avg_opcode_width + avg_sample_width + avg_sample_width + BLOCK_PADDING * 4.0
        } else if has_lir {
            avg_id_width + avg_opcode_width + BLOCK_PADDING * 2.0
        } else {
            // MIR - need wider for long instruction names
            avg_id_width + avg_opcode_width + avg_type_width + BLOCK_PADDING * 3.0
        };

        let content_width = header_width.max(table_width);
        let width = content_width + BLOCK_PADDING * 2.0;

        // Calculate height
        let mut height = HEADER_HEIGHT;
        if has_lir && has_samples {
            height += TABLE_HEADER_HEIGHT;
        }
        height += num_instructions as f64 * LINE_HEIGHT;
        height += BLOCK_PADDING * 2.0;

        Vec2 {
            x: width.max(150.0),
            y: height.max(60.0),
        }
    }
}

/// Parse SVG transform attribute like "translate(100, 200) rotate(180)"
/// Returns (tx, ty, rotation_degrees)
fn parse_transform(transform: &str) -> (f32, f32, f32) {
    let mut tx = 0.0f32;
    let mut ty = 0.0f32;
    let mut rot = 0.0f32;

    // Parse translate(x, y)
    if let Some(start) = transform.find("translate(") {
        let rest = &transform[start + 10..];
        if let Some(end) = rest.find(')') {
            let coords = &rest[..end];
            let parts: Vec<&str> = coords.split(',').map(|s| s.trim()).collect();
            if parts.len() >= 2 {
                tx = parts[0].parse().unwrap_or(0.0);
                ty = parts[1].parse().unwrap_or(0.0);
            }
        }
    }

    // Parse rotate(deg)
    if let Some(start) = transform.find("rotate(") {
        let rest = &transform[start + 7..];
        if let Some(end) = rest.find(')') {
            let deg = &rest[..end];
            rot = deg.trim().parse().unwrap_or(0.0);
        }
    }

    (tx, ty, rot)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_element() {
        let mut provider = WebGLLayoutProvider::new();
        let elem = provider.create_element("div");
        assert_eq!(elem.node_type, "div");
        assert_eq!(elem.id, 0);

        let elem2 = provider.create_element("span");
        assert_eq!(elem2.id, 1);
    }

    #[test]
    fn test_set_style_position() {
        let mut provider = WebGLLayoutProvider::new();
        let mut elem = provider.create_element("div");

        provider.set_style(&mut elem, "left", "100px");
        provider.set_style(&mut elem, "top", "200px");

        assert_eq!(elem.bounds.x, 100.0);
        assert_eq!(elem.bounds.y, 200.0);
    }

    #[test]
    fn test_block_size_calculation() {
        let provider = WebGLLayoutProvider::new();

        let size = provider.calculate_block_size("block0", 10, false, false);
        assert!(size.x >= 150.0);
        assert!(size.y >= 60.0);
    }
}
