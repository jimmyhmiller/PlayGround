// Port of LayoutProvider.ts

#[derive(Debug, Clone, Copy)]
pub struct Vec2 {
    pub x: f64,
    pub y: f64,
}

#[derive(Debug, Clone)]
pub struct DOMRect {
    pub x: f64,
    pub y: f64,
    pub width: f64,
    pub height: f64,
    pub left: f64,
    pub right: f64,
    pub top: f64,
    pub bottom: f64,
}

// We use a trait object approach for elements since we need dynamic dispatch
pub trait Element: std::fmt::Debug {
    fn as_any(&self) -> &dyn std::any::Any;
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any;
}

pub trait LayoutProvider {
    type Element: Element;

    // DOM element creation
    fn create_element(&mut self, tag: &str) -> Box<Self::Element>;
    fn create_svg_element(&mut self, tag: &str) -> Box<Self::Element>;

    // Element manipulation
    fn append_child(&mut self, parent: &mut Self::Element, child: Box<Self::Element>);
    fn set_attribute(&mut self, element: &mut Self::Element, name: &str, value: &str);
    fn set_inner_html(&mut self, element: &mut Self::Element, html: &str);
    fn set_inner_text(&mut self, element: &mut Self::Element, text: &str);

    // CSS classes
    fn add_class(&mut self, element: &mut Self::Element, class_name: &str);
    fn add_classes(&mut self, element: &mut Self::Element, class_names: &[&str]);
    fn remove_class(&mut self, element: &mut Self::Element, class_name: &str);
    fn toggle_class(&mut self, element: &mut Self::Element, class_name: &str, force: Option<bool>);

    // Style manipulation
    fn set_style(&mut self, element: &mut Self::Element, property: &str, value: &str);
    fn set_css_property(&mut self, element: &mut Self::Element, property: &str, value: &str);

    // Measurements
    fn get_bounding_client_rect(&self, element: &Self::Element) -> DOMRect;
    fn get_client_width(&self, element: &Self::Element) -> f64;
    fn get_client_height(&self, element: &Self::Element) -> f64;

    // Event handling (noop for SVG)
    fn add_event_listener(&mut self, element: &mut Self::Element, event_type: &str, listener: Box<dyn Fn()>);

    // Query selectors (return owned to avoid lifetime issues)
    fn query_selector(&self, parent: &Self::Element, selector: &str) -> bool;
    fn query_selector_all(&self, parent: &Self::Element, selector: &str) -> usize;

    // Resize observation (noop for SVG)
    fn observe_resize(&mut self, element: &Self::Element, callback: Box<dyn Fn(Vec2)>) -> Box<dyn Fn()>;

    // Pointer capture (noop for SVG)
    fn set_pointer_capture(&mut self, element: &mut Self::Element, pointer_id: i32);
    fn release_pointer_capture(&mut self, element: &mut Self::Element, pointer_id: i32);
    fn has_pointer_capture(&self, element: &Self::Element, pointer_id: i32) -> bool;

    // Calculate the size of a block based on its content
    fn calculate_block_size(&self, block_id: &str, num_instructions: usize, has_lir: bool, has_samples: bool) -> Vec2;
}
