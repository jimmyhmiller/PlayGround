// Port of PureSVGTextLayoutProvider.ts

use crate::layout_provider::{DOMRect, Element, LayoutProvider, Vec2};
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::rc::Rc;

const CHARACTER_WIDTH: f64 = 7.0;
const LINE_HEIGHT: f64 = 14.0;
const PADDING: f64 = 8.0;

struct InstructionRow {
    num_text: Option<String>,
    opcode_text: Option<String>,
    type_text: Option<String>,
    has_movable: bool,
    has_guard: bool,
    has_rob: bool,
}

#[derive(Debug, Clone)]
pub struct SVGTextNode {
    pub node_type: String,
    pub attributes: Vec<(String, String)>,  // Changed from HashMap to preserve order
    pub children: Vec<Rc<RefCell<SVGTextNode>>>,
    pub text_content: Option<String>,
    pub class_list: HashSet<String>,
    pub style: HashMap<String, String>,
}

impl SVGTextNode {
    fn new(node_type: String) -> Self {
        SVGTextNode {
            node_type,
            attributes: Vec::new(),  // Changed from HashMap::new()
            children: Vec::new(),
            text_content: None,
            class_list: HashSet::new(),
            style: HashMap::new(),
        }
    }
}

impl Element for SVGTextNode {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

pub struct PureSVGTextLayoutProvider {
    pub root: Rc<RefCell<SVGTextNode>>,
}

impl PureSVGTextLayoutProvider {
    pub fn new() -> Self {
        PureSVGTextLayoutProvider {
            root: Rc::new(RefCell::new(SVGTextNode::new("svg".to_string()))),
        }
    }

    // Helper to get attribute value from Vec
    fn get_attribute<'a>(attributes: &'a Vec<(String, String)>, name: &str) -> Option<&'a String> {
        attributes.iter()
            .find(|(k, _)| k == name)
            .map(|(_, v)| v)
    }

    pub fn to_svg_string(&self, root: &SVGTextNode) -> String {
        let mut result = String::new();

        // Get width/height/viewBox from root attributes if present
        let width = Self::get_attribute(&root.attributes, "width").map(|s| s.as_str()).unwrap_or("");
        let height = Self::get_attribute(&root.attributes, "height").map(|s| s.as_str()).unwrap_or("");
        let viewbox = Self::get_attribute(&root.attributes, "viewBox").map(|s| s.as_str()).unwrap_or("");

        // Start with SVG opening tag
        result.push_str("<svg xmlns=\"http://www.w3.org/2000/svg\"");
        if !width.is_empty() {
            result.push_str(&format!(" width=\"{}\"", width));
        }
        if !height.is_empty() {
            result.push_str(&format!(" height=\"{}\"", height));
        }
        if !viewbox.is_empty() {
            result.push_str(&format!(" viewBox=\"{}\"", viewbox));
        }
        result.push_str(">\n");

        // Render the content (no CSS - all inline styles)
        for child in &root.children {
            result.push_str(&self.render_node(&child.borrow(), 1));
        }

        result.push_str("</svg>\n");
        result
    }

    fn escape_xml(&self, text: &str) -> String {
        // First convert <- to ← (left arrow) and -> to → (right arrow) to match TypeScript
        let text = text.replace("<-", "←").replace("->", "→");

        // Then escape XML characters
        text.replace('&', "&amp;")
            .replace('<', "&lt;")
            .replace('>', "&gt;")
            .replace('"', "&quot;")
            .replace('\'', "&apos;")
    }

    fn render_node(&self, node: &SVGTextNode, depth: usize) -> String {
        let indent = "  ".repeat(depth);

        // Special handling for block groups to render them properly
        if node.class_list.contains("ig-block") {
            return self.render_block_group(node, depth);
        }

        // Special handling for instructions
        if node.node_type == "tr" && node.class_list.contains("ig-instruction") {
            return self.render_instruction(node, depth);
        }

        // Special handling for arrows
        if node.node_type == "path" && node.class_list.contains("ig-arrow") {
            return self.render_arrow(node, depth);
        }

        let mut result = String::new();

        // Opening tag
        result.push_str(&indent);
        result.push('<');
        result.push_str(&node.node_type);

        // Add attributes
        for (key, value) in &node.attributes {
            result.push_str(&format!(" {}=\"{}\"", key, self.escape_xml(value)));
        }

        // Add classes
        if !node.class_list.is_empty() {
            let classes: Vec<_> = node.class_list.iter().cloned().collect();
            result.push_str(&format!(" class=\"{}\"", classes.join(" ")));
        }

        // Add inline styles
        if !node.style.is_empty() {
            let styles: Vec<_> = node.style.iter()
                .map(|(k, v)| format!("{}: {}", k, v))
                .collect();
            result.push_str(&format!(" style=\"{}\"", styles.join("; ")));
        }

        // Self-closing tags
        if node.children.is_empty() && node.text_content.is_none() {
            result.push_str("/>\n");
            return result;
        }

        result.push('>');

        // Text content
        if let Some(text) = &node.text_content {
            result.push_str(&self.escape_xml(text));
        }

        // Children
        if !node.children.is_empty() {
            result.push('\n');
            for child in &node.children {
                result.push_str(&self.render_node(&child.borrow(), depth + 1));
            }
            result.push_str(&indent);
        }

        // Closing tag
        result.push_str(&format!("</{}>\n", node.node_type));

        result
    }

    fn render_block_group(&self, node: &SVGTextNode, depth: usize) -> String {
        let indent = "  ".repeat(depth);
        let mut result = String::new();

        // Get block position from transform or default
        let x = node.style.get("left")
            .and_then(|s| s.trim_end_matches("px").parse::<f64>().ok())
            .unwrap_or(0.0);
        let y = node.style.get("top")
            .and_then(|s| s.trim_end_matches("px").parse::<f64>().ok())
            .unwrap_or(0.0);

        result.push_str(&format!("{}<g transform=\"translate({}, {})\">\n", indent, x, y));

        // Render background rect
        let width = Self::get_attribute(&node.attributes, "data-width")
            .and_then(|s| s.parse::<f64>().ok())
            .unwrap_or(200.0);
        let height = Self::get_attribute(&node.attributes, "data-height")
            .and_then(|s| s.parse::<f64>().ok())
            .unwrap_or(100.0);

        result.push_str(&format!("{}  <rect x=\"0\" y=\"0\" width=\"{}\" height=\"{}\" fill=\"#f9f9f9\" stroke=\"#0c0c0d\" stroke-width=\"1\"/>\n",
            indent, width, height));

        // Check if this is a loop header for background color
        let is_loop_header = node.class_list.contains("ig-block-att-loopheader");
        let header_bg = if is_loop_header { "#1fa411" } else { "#0c0c0d" };

        // Render header with background
        for child in &node.children {
            let child_ref = child.borrow();
            if child_ref.class_list.contains("ig-block-header") {
                if let Some(text) = &child_ref.text_content {
                    // Draw header background rectangle
                    result.push_str(&format!("{}  <rect x=\"0\" y=\"0\" width=\"{}\" height=\"28\" fill=\"{}\"/>\n",
                        indent, width, header_bg));
                    // Draw header text (centered)
                    result.push_str(&format!("{}  <text x=\"{}\" y=\"18\" font-family=\"monospace\" font-size=\"12\" fill=\"white\" font-weight=\"bold\" text-anchor=\"middle\">{}</text>\n",
                        indent, width / 2.0, self.escape_xml(text)));
                }
            }
        }

        // Render instructions with proper column layout
        let start_y = 28.0 + PADDING + 4.0; // Start below header
        for child in &node.children {
            let child_ref = child.borrow();
            if child_ref.class_list.contains("ig-instructions") {
                result.push_str(&self.render_instructions_table(&child_ref, depth, start_y));
            }
        }

        // Render edge labels (TypeScript lines 414-427)
        for child in &node.children {
            let child_ref = child.borrow();
            if child_ref.class_list.contains("ig-edge-label") {
                if let Some(text) = &child_ref.text_content {
                    // Get x position from left style (matches TypeScript line 421 getting from transform)
                    let label_x = child_ref.style.get("left")
                        .and_then(|s| s.trim_end_matches("px").parse::<f64>().ok())
                        .unwrap_or(0.0);
                    let label_indent = format!("{}  ", indent);
                    // Position at bottom of block with offset (TypeScript line 424-425)
                    let label_y = height + 12.0;
                    result.push_str(&format!("{}<text x=\"{}\" y=\"{}\" font-family=\"monospace\" font-size=\"9\" fill=\"#777\">{}</text>\n",
                        label_indent, label_x + 4.0, label_y, self.escape_xml(text)));
                }
            }
        }

        result.push_str(&format!("{}</g>\n", indent));
        result
    }

    fn render_instructions_table(&self, instructions_node: &SVGTextNode, depth: usize, start_y: f64) -> String {
        let indent = "  ".repeat(depth + 1);
        let mut result = String::new();
        let mut current_y = start_y;

        // Find all instruction rows
        let rows = self.find_instruction_rows(instructions_node);

        // Measure maximum width of each column
        let mut max_num_width = 0.0;
        let mut max_opcode_width = 0.0;

        for row in &rows {
            if let Some(num_text) = &row.num_text {
                let width = num_text.chars().count() as f64 * CHARACTER_WIDTH;
                max_num_width = if width > max_num_width { width } else { max_num_width };
            }
            if let Some(opcode_text) = &row.opcode_text {
                let width = opcode_text.chars().count() as f64 * CHARACTER_WIDTH;
                max_opcode_width = if width > max_opcode_width { width } else { max_opcode_width };
            }
        }

        // Calculate column positions with proper spacing
        let num_x = PADDING;
        let opcode_x = PADDING + max_num_width + 8.0;
        let type_x = opcode_x + max_opcode_width + 8.0;

        for row in &rows {
            // Determine text color and decoration based on attributes
            let mut text_color = "black";
            let mut text_decoration = "";

            if row.has_movable {
                text_color = "#1048af";
            }
            if row.has_rob {
                text_color = "#444";
            }
            if row.has_guard {
                text_decoration = " text-decoration=\"underline\"";
            }

            // Render instruction number (gray)
            if let Some(num_text) = &row.num_text {
                result.push_str(&format!("{}<text x=\"{}\" y=\"{}\" font-family=\"monospace\" font-size=\"11\" fill=\"#777\">{}</text>\n",
                    indent, num_x, current_y, self.escape_xml(num_text)));
            }

            // Render opcode (colored based on attributes)
            if let Some(opcode_text) = &row.opcode_text {
                result.push_str(&format!("{}<text x=\"{}\" y=\"{}\" font-family=\"monospace\" font-size=\"11\" fill=\"{}\"{}>{}</text>\n",
                    indent, opcode_x, current_y, text_color, text_decoration, self.escape_xml(opcode_text)));
            }

            // Render type (blue) - skip if empty or "None"
            if let Some(type_text) = &row.type_text {
                if !type_text.is_empty() && type_text != "None" {
                    result.push_str(&format!("{}<text x=\"{}\" y=\"{}\" font-family=\"monospace\" font-size=\"11\" fill=\"#1048af\">{}</text>\n",
                        indent, type_x, current_y, self.escape_xml(type_text)));
                }
            }

            current_y += LINE_HEIGHT;
        }

        result
    }

    fn find_instruction_rows(&self, node: &SVGTextNode) -> Vec<InstructionRow> {
        let mut rows = Vec::new();
        self.find_instruction_rows_recursive(node, &mut rows);
        rows
    }

    fn find_instruction_rows_recursive(&self, node: &SVGTextNode, rows: &mut Vec<InstructionRow>) {
        if node.class_list.contains("ig-ins") {
            let mut row = InstructionRow {
                num_text: None,
                opcode_text: None,
                type_text: None,
                has_movable: node.class_list.contains("ig-ins-att-Movable"),
                has_guard: node.class_list.contains("ig-ins-att-Guard"),
                has_rob: node.class_list.contains("ig-ins-att-RecoveredOnBailout"),
            };

            for child in &node.children {
                let child_ref = child.borrow();
                if child_ref.class_list.contains("ig-ins-num") {
                    row.num_text = child_ref.text_content.clone();
                } else if child_ref.class_list.contains("ig-ins-type") {
                    row.type_text = child_ref.text_content.clone();
                } else if child_ref.node_type == "td" && !child_ref.class_list.contains("ig-ins-num") && !child_ref.class_list.contains("ig-ins-type") {
                    // Convert <- to ← and -> to → for measurement consistency
                    row.opcode_text = child_ref.text_content.as_ref().map(|s| s.replace("<-", "←").replace("->", "→"));
                }
            }

            rows.push(row);
        }

        for child in &node.children {
            self.find_instruction_rows_recursive(&child.borrow(), rows);
        }
    }

    fn render_instruction(&self, node: &SVGTextNode, depth: usize) -> String {
        let indent = "  ".repeat(depth);
        let mut result = String::new();

        result.push_str(&format!("{}<g class=\"ig-instruction\">\n", indent));

        let mut x_offset = 0.0;
        for cell in &node.children {
            let cell_ref = cell.borrow();
            if cell_ref.node_type == "td" {
                if let Some(text) = &cell_ref.text_content {
                    result.push_str(&format!("{}  <text x=\"{}\" y=\"0\">{}</text>\n",
                        indent, x_offset, self.escape_xml(text)));
                    x_offset += text.len() as f64 * CHARACTER_WIDTH;
                }
            }
        }

        result.push_str(&format!("{}</g>\n", indent));
        result
    }

    fn render_arrow(&self, node: &SVGTextNode, depth: usize) -> String {
        let indent = "  ".repeat(depth);
        let mut result = String::new();

        result.push_str(&indent);
        result.push_str("<path");

        if let Some(d) = Self::get_attribute(&node.attributes, "d") {
            result.push_str(&format!(" d=\"{}\"", d));
        }

        let mut classes = vec!["ig-arrow"];
        for class in &node.class_list {
            if class != "ig-arrow" {
                classes.push(class);
            }
        }
        result.push_str(&format!(" class=\"{}\"", classes.join(" ")));

        result.push_str(" />\n");
        result
    }

    fn count_instructions(&self, element: &SVGTextNode) -> usize {
        let mut count = 0;
        if element.class_list.contains("ig-instruction") {
            count += 1;
        }
        for child in &element.children {
            count += self.count_instructions(&child.borrow());
        }
        count
    }

    fn measure_text(&self, text: &str) -> f64 {
        text.len() as f64 * CHARACTER_WIDTH
    }

    fn measure_element(&self, element: &SVGTextNode) -> (f64, f64) {
        // Check if this is a block element - use block-specific measurement
        if element.class_list.contains("ig-block") {
            return self.measure_block_element(element);
        }

        // Simple measurement based on content for non-block elements
        let mut width = 0.0;
        let mut height = LINE_HEIGHT;

        if let Some(text) = &element.text_content {
            width = self.measure_text(text);
        }

        // Add children dimensions
        for child in &element.children {
            let child_ref = child.borrow();
            let (child_width, child_height) = self.measure_element(&child_ref);
            width = width.max(child_width);
            height += child_height;
        }

        (width, height)
    }

    fn measure_block_element(&self, element: &SVGTextNode) -> (f64, f64) {
        // Match TypeScript's calculateBlockSize exactly (lines 240-278)

        // Find all instruction rows
        let rows = self.find_instruction_rows(element);

        // Measure maximum width of each column
        let mut max_num_width = 0.0;
        let mut max_opcode_width = 0.0;
        let mut max_type_width = 0.0;

        for row in &rows {
            if let Some(num_text) = &row.num_text {
                let width = num_text.chars().count() as f64 * CHARACTER_WIDTH;
                max_num_width = if width > max_num_width { width } else { max_num_width };
            }
            if let Some(opcode_text) = &row.opcode_text {
                let width = opcode_text.chars().count() as f64 * CHARACTER_WIDTH;
                max_opcode_width = if width > max_opcode_width { width } else { max_opcode_width };
            }
            if let Some(type_text) = &row.type_text {
                if type_text != "None" {
                    let width = type_text.chars().count() as f64 * CHARACTER_WIDTH;
                    max_type_width = if width > max_type_width { width } else { max_type_width };
                }
            }
        }

        // Get header text width
        let mut header_width = 0.0;
        for child in &element.children {
            let child_ref = child.borrow();
            if child_ref.class_list.contains("ig-block-header") {
                if let Some(text) = &child_ref.text_content {
                    header_width = text.chars().count() as f64 * CHARACTER_WIDTH;
                }
            }
        }

        // Calculate width: padding + num + gap + opcode + gap + type + padding
        let calculated_width = PADDING + max_num_width + 8.0 + max_opcode_width + 8.0 + max_type_width + PADDING;

        // Header width with padding
        let min_width_for_header = header_width + PADDING * 2.0;

        // Width is max of header width (with padding), calculated width, and minimum 150px
        let width = if min_width_for_header > calculated_width {
            if min_width_for_header > 150.0 { min_width_for_header } else { 150.0 }
        } else if calculated_width > 150.0 {
            calculated_width
        } else {
            150.0
        };

        // Calculate height: 30 (header area) + (num_rows * line_height) + padding * 2
        let height = 30.0 + (rows.len() as f64 * LINE_HEIGHT) + PADDING * 2.0;

        (width, height)
    }
}

impl Default for PureSVGTextLayoutProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl LayoutProvider for PureSVGTextLayoutProvider {
    type Element = SVGTextNode;

    fn create_element(&mut self, tag: &str) -> Box<Self::Element> {
        Box::new(SVGTextNode::new(tag.to_string()))
    }

    fn create_svg_element(&mut self, tag: &str) -> Box<Self::Element> {
        Box::new(SVGTextNode::new(tag.to_string()))
    }

    fn append_child(&mut self, parent: &mut Self::Element, child: Box<Self::Element>) {
        parent.children.push(Rc::new(RefCell::new(*child)));
    }

    fn set_attribute(&mut self, element: &mut Self::Element, name: &str, value: &str) {
        // Remove existing attribute with same name if present
        element.attributes.retain(|(k, _)| k != name);
        // Add new attribute (preserves insertion order)
        element.attributes.push((name.to_string(), value.to_string()));
    }

    fn set_inner_html(&mut self, element: &mut Self::Element, html: &str) {
        // For SVG text layout, we just store as text content
        element.text_content = Some(html.to_string());
    }

    fn set_inner_text(&mut self, element: &mut Self::Element, text: &str) {
        element.text_content = Some(text.to_string());
    }

    fn add_class(&mut self, element: &mut Self::Element, class_name: &str) {
        element.class_list.insert(class_name.to_string());
    }

    fn add_classes(&mut self, element: &mut Self::Element, class_names: &[&str]) {
        for class_name in class_names {
            element.class_list.insert(class_name.to_string());
        }
    }

    fn remove_class(&mut self, element: &mut Self::Element, class_name: &str) {
        element.class_list.remove(class_name);
    }

    fn toggle_class(&mut self, element: &mut Self::Element, class_name: &str, force: Option<bool>) {
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
    }

    fn set_css_property(&mut self, element: &mut Self::Element, property: &str, value: &str) {
        element.style.insert(property.to_string(), value.to_string());
    }

    fn get_bounding_client_rect(&self, element: &Self::Element) -> DOMRect {
        let (width, height) = self.measure_element(element);

        let x = element.style.get("left")
            .and_then(|s| s.trim_end_matches("px").parse::<f64>().ok())
            .unwrap_or(0.0);
        let y = element.style.get("top")
            .and_then(|s| s.trim_end_matches("px").parse::<f64>().ok())
            .unwrap_or(0.0);

        DOMRect {
            x,
            y,
            width,
            height,
            left: x,
            right: x + width,
            top: y,
            bottom: y + height,
        }
    }

    fn get_client_width(&self, element: &Self::Element) -> f64 {
        let (width, _) = self.measure_element(element);
        width
    }

    fn get_client_height(&self, element: &Self::Element) -> f64 {
        let (_, height) = self.measure_element(element);
        height
    }

    fn add_event_listener(&mut self, _element: &mut Self::Element, _event_type: &str, _listener: Box<dyn Fn()>) {
        // Noop for pure SVG
    }

    fn query_selector(&self, _parent: &Self::Element, _selector: &str) -> bool {
        // Simplified - just return false
        false
    }

    fn query_selector_all(&self, _parent: &Self::Element, _selector: &str) -> usize {
        // Simplified - just return 0
        0
    }

    fn observe_resize(&mut self, _element: &Self::Element, _callback: Box<dyn Fn(Vec2)>) -> Box<dyn Fn()> {
        // Noop for pure SVG - return empty cleanup function
        Box::new(|| {})
    }

    fn set_pointer_capture(&mut self, _element: &mut Self::Element, _pointer_id: i32) {
        // Noop for pure SVG
    }

    fn release_pointer_capture(&mut self, _element: &mut Self::Element, _pointer_id: i32) {
        // Noop for pure SVG
    }

    fn has_pointer_capture(&self, _element: &Self::Element, _pointer_id: i32) -> bool {
        // Noop for pure SVG
        false
    }

    fn calculate_block_size(&self, block_id: &str, num_instructions: usize, has_lir: bool, has_samples: bool) -> Vec2 {
        // Calculate block size based on content
        // Block structure:
        // - Header: "Block X (description)"
        // - Table with instructions
        // - Optional edge labels (if 2 successors)

        const BLOCK_PADDING: f64 = 8.0;  // Padding on each side
        const HEADER_HEIGHT: f64 = LINE_HEIGHT + 8.0;  // Header text + padding
        const TABLE_HEADER_HEIGHT: f64 = LINE_HEIGHT + 4.0;  // Table header for LIR with samples

        // Calculate header width
        // "Block " + block_id + potential description like " (loop header)"
        let header_text_len = 6 + block_id.len() + 20; // Max description length
        let header_width = header_text_len as f64 * CHARACTER_WIDTH;

        // Calculate table width
        // For MIR: [ID] [Opcode] [Type]
        // For LIR: [ID] [Opcode] [Total] [Self] (if has_samples)
        //          [ID] [Opcode] (if no samples)

        // Estimate based on typical instruction content
        // These need to be generous because instructions can be very long
        let avg_id_width = 3.0 * CHARACTER_WIDTH; // "123"
        let avg_opcode_width = 50.0 * CHARACTER_WIDTH; // Instructions can be quite long
        let avg_type_width = 25.0 * CHARACTER_WIDTH; // Types can also be long
        let avg_sample_width = 6.0 * CHARACTER_WIDTH; // "12345"

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

        // Each instruction is one line
        height += num_instructions as f64 * LINE_HEIGHT;

        // Add padding
        height += BLOCK_PADDING * 2.0;

        // Edge labels don't affect size (they're positioned absolutely)

        Vec2 {
            x: width.max(150.0), // Minimum width
            y: height.max(60.0), // Minimum height
        }
    }
}
