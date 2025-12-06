// HTML Layout Provider for generating interactive standalone HTML files

use crate::layout_provider::{DOMRect, Element, LayoutProvider, Vec2};
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::rc::Rc;

const CHARACTER_WIDTH: f64 = 7.2;
const LINE_HEIGHT: f64 = 16.0;
const PADDING: f64 = 8.0;

#[derive(Debug, Clone)]
pub enum HTMLNode {
    Div {
        classes: HashSet<String>,
        attributes: Vec<(String, String)>,
        styles: HashMap<String, String>,
        children: Vec<Rc<RefCell<HTMLNode>>>,
        text: Option<String>,
    },
    Table {
        classes: HashSet<String>,
        attributes: Vec<(String, String)>,
        children: Vec<Rc<RefCell<HTMLNode>>>,
    },
    Tr {
        classes: HashSet<String>,
        attributes: Vec<(String, String)>,
        children: Vec<Rc<RefCell<HTMLNode>>>,
    },
    Td {
        classes: HashSet<String>,
        attributes: Vec<(String, String)>,
        text: Option<String>,
    },
    Svg {
        classes: HashSet<String>,
        attributes: Vec<(String, String)>,
        children: Vec<Rc<RefCell<HTMLNode>>>,
    },
    Path {
        classes: HashSet<String>,
        attributes: Vec<(String, String)>,
        d: String,
    },
    G {
        classes: HashSet<String>,
        attributes: Vec<(String, String)>,
        children: Vec<Rc<RefCell<HTMLNode>>>,
    },
    Text {
        text: String,
    },
}

impl HTMLNode {
    fn new_div() -> Self {
        HTMLNode::Div {
            classes: HashSet::new(),
            attributes: Vec::new(),
            styles: HashMap::new(),
            children: Vec::new(),
            text: None,
        }
    }

    fn new_table() -> Self {
        HTMLNode::Table {
            classes: HashSet::new(),
            attributes: Vec::new(),
            children: Vec::new(),
        }
    }

    fn new_tr() -> Self {
        HTMLNode::Tr {
            classes: HashSet::new(),
            attributes: Vec::new(),
            children: Vec::new(),
        }
    }

    fn new_td() -> Self {
        HTMLNode::Td {
            classes: HashSet::new(),
            attributes: Vec::new(),
            text: None,
        }
    }

    fn new_svg() -> Self {
        HTMLNode::Svg {
            classes: HashSet::new(),
            attributes: Vec::new(),
            children: Vec::new(),
        }
    }

    fn new_path() -> Self {
        HTMLNode::Path {
            classes: HashSet::new(),
            attributes: Vec::new(),
            d: String::new(),
        }
    }

    fn new_g() -> Self {
        HTMLNode::G {
            classes: HashSet::new(),
            attributes: Vec::new(),
            children: Vec::new(),
        }
    }

    pub fn add_class(&mut self, class_name: &str) {
        match self {
            HTMLNode::Div { classes, .. }
            | HTMLNode::Table { classes, .. }
            | HTMLNode::Tr { classes, .. }
            | HTMLNode::Td { classes, .. }
            | HTMLNode::Svg { classes, .. }
            | HTMLNode::Path { classes, .. }
            | HTMLNode::G { classes, .. } => {
                classes.insert(class_name.to_string());
            }
            HTMLNode::Text { .. } => {}
        }
    }

    pub fn remove_class(&mut self, class_name: &str) {
        match self {
            HTMLNode::Div { classes, .. }
            | HTMLNode::Table { classes, .. }
            | HTMLNode::Tr { classes, .. }
            | HTMLNode::Td { classes, .. }
            | HTMLNode::Svg { classes, .. }
            | HTMLNode::Path { classes, .. }
            | HTMLNode::G { classes, .. } => {
                classes.remove(class_name);
            }
            HTMLNode::Text { .. } => {}
        }
    }

    pub fn set_attribute(&mut self, name: &str, value: &str) {
        match self {
            HTMLNode::Div { attributes, .. }
            | HTMLNode::Table { attributes, .. }
            | HTMLNode::Tr { attributes, .. }
            | HTMLNode::Td { attributes, .. }
            | HTMLNode::Svg { attributes, .. }
            | HTMLNode::Path { attributes, .. }
            | HTMLNode::G { attributes, .. } => {
                // Remove existing attribute with same name
                attributes.retain(|(k, _)| k != name);
                attributes.push((name.to_string(), value.to_string()));
            }
            HTMLNode::Text { .. } => {}
        }
    }

    pub fn set_style(&mut self, property: &str, value: &str) {
        match self {
            HTMLNode::Div { styles, .. } => {
                styles.insert(property.to_string(), value.to_string());
            }
            _ => {}
        }
    }

    pub fn set_text(&mut self, text: &str) {
        match self {
            HTMLNode::Div { text: t, .. } | HTMLNode::Td { text: t, .. } => {
                *t = Some(text.to_string());
            }
            HTMLNode::Text { text: t } => {
                *t = text.to_string();
            }
            _ => {}
        }
    }

    pub fn append_child(&mut self, child: Rc<RefCell<HTMLNode>>) {
        match self {
            HTMLNode::Div { children, .. }
            | HTMLNode::Table { children, .. }
            | HTMLNode::Tr { children, .. }
            | HTMLNode::Svg { children, .. }
            | HTMLNode::G { children, .. } => {
                children.push(child);
            }
            _ => {}
        }
    }

    fn escape_html(&self, text: &str) -> String {
        text.replace('&', "&amp;")
            .replace('<', "&lt;")
            .replace('>', "&gt;")
            .replace('"', "&quot;")
            .replace('\'', "&#39;")
    }

    pub fn to_html(&self, indent: usize) -> String {
        let indent_str = "  ".repeat(indent);
        let mut result = String::new();

        match self {
            HTMLNode::Div {
                classes,
                attributes,
                styles,
                children,
                text,
            } => {
                result.push_str(&indent_str);
                result.push_str("<div");

                if !classes.is_empty() {
                    let classes_vec: Vec<_> = classes.iter().cloned().collect();
                    result.push_str(&format!(" class=\"{}\"", classes_vec.join(" ")));
                }

                for (key, value) in attributes {
                    result.push_str(&format!(" {}=\"{}\"", key, self.escape_html(value)));
                }

                if !styles.is_empty() {
                    let styles_vec: Vec<_> = styles
                        .iter()
                        .map(|(k, v)| format!("{}: {}", k, v))
                        .collect();
                    result.push_str(&format!(" style=\"{}\"", styles_vec.join("; ")));
                }

                result.push('>');

                if let Some(text) = text {
                    result.push_str(&self.escape_html(text));
                }

                if !children.is_empty() {
                    result.push('\n');
                    for child in children {
                        result.push_str(&child.borrow().to_html(indent + 1));
                    }
                    result.push_str(&indent_str);
                }

                result.push_str("</div>\n");
            }
            HTMLNode::Table {
                classes,
                attributes,
                children,
            } => {
                result.push_str(&indent_str);
                result.push_str("<table");

                if !classes.is_empty() {
                    let classes_vec: Vec<_> = classes.iter().cloned().collect();
                    result.push_str(&format!(" class=\"{}\"", classes_vec.join(" ")));
                }

                for (key, value) in attributes {
                    result.push_str(&format!(" {}=\"{}\"", key, self.escape_html(value)));
                }

                result.push('>');

                if !children.is_empty() {
                    result.push('\n');
                    for child in children {
                        result.push_str(&child.borrow().to_html(indent + 1));
                    }
                    result.push_str(&indent_str);
                }

                result.push_str("</table>\n");
            }
            HTMLNode::Tr {
                classes,
                attributes,
                children,
            } => {
                result.push_str(&indent_str);
                result.push_str("<tr");

                if !classes.is_empty() {
                    let classes_vec: Vec<_> = classes.iter().cloned().collect();
                    result.push_str(&format!(" class=\"{}\"", classes_vec.join(" ")));
                }

                for (key, value) in attributes {
                    result.push_str(&format!(" {}=\"{}\"", key, self.escape_html(value)));
                }

                result.push('>');

                if !children.is_empty() {
                    result.push('\n');
                    for child in children {
                        result.push_str(&child.borrow().to_html(indent + 1));
                    }
                    result.push_str(&indent_str);
                }

                result.push_str("</tr>\n");
            }
            HTMLNode::Td {
                classes,
                attributes,
                text,
            } => {
                result.push_str(&indent_str);
                result.push_str("<td");

                if !classes.is_empty() {
                    let classes_vec: Vec<_> = classes.iter().cloned().collect();
                    result.push_str(&format!(" class=\"{}\"", classes_vec.join(" ")));
                }

                for (key, value) in attributes {
                    result.push_str(&format!(" {}=\"{}\"", key, self.escape_html(value)));
                }

                result.push('>');

                if let Some(text) = text {
                    result.push_str(&self.escape_html(text));
                }

                result.push_str("</td>\n");
            }
            HTMLNode::Svg {
                classes,
                attributes,
                children,
            } => {
                result.push_str(&indent_str);
                result.push_str("<svg");

                if !classes.is_empty() {
                    let classes_vec: Vec<_> = classes.iter().cloned().collect();
                    result.push_str(&format!(" class=\"{}\"", classes_vec.join(" ")));
                }

                for (key, value) in attributes {
                    result.push_str(&format!(" {}=\"{}\"", key, self.escape_html(value)));
                }

                result.push('>');

                if !children.is_empty() {
                    result.push('\n');
                    for child in children {
                        result.push_str(&child.borrow().to_html(indent + 1));
                    }
                    result.push_str(&indent_str);
                }

                result.push_str("</svg>\n");
            }
            HTMLNode::Path {
                classes,
                attributes,
                d,
            } => {
                result.push_str(&indent_str);
                result.push_str("<path");

                if !classes.is_empty() {
                    let classes_vec: Vec<_> = classes.iter().cloned().collect();
                    result.push_str(&format!(" class=\"{}\"", classes_vec.join(" ")));
                }

                result.push_str(&format!(" d=\"{}\"", d));

                for (key, value) in attributes {
                    result.push_str(&format!(" {}=\"{}\"", key, self.escape_html(value)));
                }

                result.push_str("/>\n");
            }
            HTMLNode::G {
                classes,
                attributes,
                children,
            } => {
                result.push_str(&indent_str);
                result.push_str("<g");

                if !classes.is_empty() {
                    let classes_vec: Vec<_> = classes.iter().cloned().collect();
                    result.push_str(&format!(" class=\"{}\"", classes_vec.join(" ")));
                }

                for (key, value) in attributes {
                    result.push_str(&format!(" {}=\"{}\"", key, self.escape_html(value)));
                }

                result.push('>');

                if !children.is_empty() {
                    result.push('\n');
                    for child in children {
                        result.push_str(&child.borrow().to_html(indent + 1));
                    }
                    result.push_str(&indent_str);
                }

                result.push_str("</g>\n");
            }
            HTMLNode::Text { text } => {
                result.push_str(&self.escape_html(text));
            }
        }

        result
    }
}

impl Element for HTMLNode {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

pub struct HTMLLayoutProvider {
    block_sizes: HashMap<String, Vec2>,
}

impl HTMLLayoutProvider {
    pub fn new() -> Self {
        HTMLLayoutProvider {
            block_sizes: HashMap::new(),
        }
    }
}

impl LayoutProvider for HTMLLayoutProvider {
    type Element = HTMLNode;

    fn create_element(&mut self, tag: &str) -> Box<Self::Element> {
        Box::new(match tag {
            "div" => HTMLNode::new_div(),
            "table" => HTMLNode::new_table(),
            "tr" => HTMLNode::new_tr(),
            "td" => HTMLNode::new_td(),
            "g" => HTMLNode::new_g(),
            _ => HTMLNode::new_div(), // Default to div
        })
    }

    fn create_svg_element(&mut self, tag: &str) -> Box<Self::Element> {
        Box::new(match tag {
            "svg" => HTMLNode::new_svg(),
            "path" => HTMLNode::new_path(),
            "g" => HTMLNode::new_div(), // Use div for HTML output, not SVG group
            _ => HTMLNode::new_svg(), // Default to svg
        })
    }

    fn append_child(&mut self, parent: &mut Self::Element, child: Box<Self::Element>) {
        parent.append_child(Rc::new(RefCell::new(*child)));
    }

    fn set_attribute(&mut self, element: &mut Self::Element, name: &str, value: &str) {
        element.set_attribute(name, value);
    }

    fn set_inner_html(&mut self, _element: &mut Self::Element, _html: &str) {
        // Not needed for HTML generation
    }

    fn set_inner_text(&mut self, element: &mut Self::Element, text: &str) {
        element.set_text(text);
    }

    fn add_class(&mut self, element: &mut Self::Element, class_name: &str) {
        element.add_class(class_name);
    }

    fn add_classes(&mut self, element: &mut Self::Element, class_names: &[&str]) {
        for class_name in class_names {
            element.add_class(class_name);
        }
    }

    fn remove_class(&mut self, element: &mut Self::Element, class_name: &str) {
        element.remove_class(class_name);
    }

    fn toggle_class(&mut self, element: &mut Self::Element, class_name: &str, force: Option<bool>) {
        if let Some(force_value) = force {
            if force_value {
                element.add_class(class_name);
            } else {
                element.remove_class(class_name);
            }
        } else {
            // Toggle based on current state - for now just add
            element.add_class(class_name);
        }
    }

    fn set_style(&mut self, element: &mut Self::Element, property: &str, value: &str) {
        element.set_style(property, value);
    }

    fn set_css_property(&mut self, element: &mut Self::Element, property: &str, value: &str) {
        element.set_style(property, value);
    }

    fn get_bounding_client_rect(&self, _element: &Self::Element) -> DOMRect {
        // For HTML generation, we don't need actual measurements
        // Return a dummy rect
        DOMRect {
            x: 0.0,
            y: 0.0,
            width: 0.0,
            height: 0.0,
            left: 0.0,
            right: 0.0,
            top: 0.0,
            bottom: 0.0,
        }
    }

    fn get_client_width(&self, _element: &Self::Element) -> f64 {
        0.0
    }

    fn get_client_height(&self, _element: &Self::Element) -> f64 {
        0.0
    }

    fn add_event_listener(
        &mut self,
        _element: &mut Self::Element,
        _event_type: &str,
        _listener: Box<dyn Fn()>,
    ) {
        // Event handlers not needed for static HTML
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
        _block_id: &str,
        num_instructions: usize,
        has_lir: bool,
        has_samples: bool,
    ) -> Vec2 {
        // Calculate width based on instruction table columns
        let num_width = CHARACTER_WIDTH * 4.0; // "###"
        let opcode_width = CHARACTER_WIDTH * 20.0; // Longest opcode
        let type_width = CHARACTER_WIDTH * 15.0; // Type info

        let mut total_width = num_width + opcode_width + type_width + PADDING * 3.0;

        if has_lir {
            let snapshot_width = CHARACTER_WIDTH * 12.0;
            total_width += snapshot_width;
        }

        if has_samples {
            let sample_width = CHARACTER_WIDTH * 10.0;
            total_width += sample_width * 2.0; // Total and Self columns
        }

        // Calculate height
        let header_height = LINE_HEIGHT + PADDING;
        let instructions_height = (num_instructions as f64) * LINE_HEIGHT + PADDING;
        let total_height = header_height + instructions_height;

        Vec2 {
            x: total_width,
            y: total_height,
        }
    }
}
