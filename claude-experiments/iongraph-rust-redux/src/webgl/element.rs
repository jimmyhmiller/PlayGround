//! WebGL element representation
//!
//! Unlike DOM-based rendering, WebGL elements are lightweight data structures
//! that record what to render. They're converted to draw calls during scene building.

use std::any::Any;
use std::collections::{HashMap, HashSet};

use crate::layout_provider::Element;

/// Unique identifier for scene elements
pub type ElementId = u32;

/// Rectangle bounds
#[derive(Debug, Clone, Copy, Default)]
pub struct Rect {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

impl Rect {
    pub fn new(x: f32, y: f32, width: f32, height: f32) -> Self {
        Self { x, y, width, height }
    }

    pub fn contains(&self, px: f32, py: f32) -> bool {
        px >= self.x && px <= self.x + self.width && py >= self.y && py <= self.y + self.height
    }

    pub fn right(&self) -> f32 {
        self.x + self.width
    }

    pub fn bottom(&self) -> f32 {
        self.y + self.height
    }
}

/// WebGL element - lightweight struct that records what to render.
///
/// This mirrors the DOM element interface expected by Graph but stores
/// data for WebGL rendering rather than actual DOM nodes.
#[derive(Debug, Clone)]
pub struct WebGLElement {
    /// Unique identifier for this element
    pub id: ElementId,

    /// Element type (div, svg, g, path, text, rect, etc.)
    pub node_type: String,

    /// Bounding rectangle in local coordinates
    pub bounds: Rect,

    /// CSS classes applied to this element
    pub class_list: HashSet<String>,

    /// Inline styles
    pub style: HashMap<String, String>,

    /// Element attributes (data-*, d for paths, etc.)
    pub attributes: HashMap<String, String>,

    /// Text content for text elements
    pub text_content: Option<String>,

    /// Child element IDs (stored separately in provider)
    pub children: Vec<ElementId>,

    /// Parent element ID
    pub parent: Option<ElementId>,

    /// Whether this element should be rendered
    pub visible: bool,

    /// Whether this element needs to be re-rendered
    pub dirty: bool,
}

impl WebGLElement {
    pub fn new(id: ElementId, node_type: &str) -> Self {
        Self {
            id,
            node_type: node_type.to_string(),
            bounds: Rect::default(),
            class_list: HashSet::new(),
            style: HashMap::new(),
            attributes: HashMap::new(),
            text_content: None,
            children: Vec::new(),
            parent: None,
            visible: true,
            dirty: true,
        }
    }

    /// Check if element has a specific class
    pub fn has_class(&self, class_name: &str) -> bool {
        self.class_list.contains(class_name)
    }

    /// Get an attribute value
    pub fn get_attribute(&self, name: &str) -> Option<&str> {
        self.attributes.get(name).map(|s| s.as_str())
    }

    /// Get a style value
    pub fn get_style(&self, property: &str) -> Option<&str> {
        self.style.get(property).map(|s| s.as_str())
    }

    /// Check if this is a block element
    pub fn is_block(&self) -> bool {
        self.has_class("ig-block")
    }

    /// Check if this is an arrow/path element
    pub fn is_arrow(&self) -> bool {
        self.has_class("ig-arrow")
    }

    /// Check if this is a loop header block
    pub fn is_loop_header(&self) -> bool {
        self.has_class("ig-block-att-loopheader")
    }

    /// Check if this is a backedge block
    pub fn is_backedge(&self) -> bool {
        self.has_class("ig-block-att-backedge")
    }

    /// Get the block ID if this is a block element
    pub fn block_id(&self) -> Option<&str> {
        self.get_attribute("data-ig-block-id")
    }
}

impl Element for WebGLElement {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rect_contains() {
        let rect = Rect::new(10.0, 20.0, 100.0, 50.0);
        assert!(rect.contains(50.0, 40.0));
        assert!(!rect.contains(5.0, 40.0));
        assert!(!rect.contains(50.0, 100.0));
    }

    #[test]
    fn test_element_classes() {
        let mut elem = WebGLElement::new(1, "div");
        elem.class_list.insert("ig-block".to_string());
        elem.class_list.insert("ig-block-att-loopheader".to_string());

        assert!(elem.has_class("ig-block"));
        assert!(elem.is_block());
        assert!(elem.is_loop_header());
        assert!(!elem.is_backedge());
    }
}
