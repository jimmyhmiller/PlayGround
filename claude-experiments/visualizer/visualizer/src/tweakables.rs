use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use wasm_bindgen::prelude::*;

struct Entry {
    value: f64,
    min: f64,
    max: f64,
    step: f64,
    default: f64,
    category: String,
}

/// A registry of named tweakable values, backed by HTML range sliders.
pub struct Tweakables {
    entries: Rc<RefCell<HashMap<String, Entry>>>,
    order: Vec<String>,
    panel_built: bool,
}

impl Tweakables {
    pub fn new() -> Self {
        Self {
            entries: Rc::new(RefCell::new(HashMap::new())),
            order: Vec::new(),
            panel_built: false,
        }
    }

    /// Register a tweakable value. Call this at setup time.
    pub fn register(&mut self, name: &str, default: f64, min: f64, max: f64, step: f64, category: &str) {
        let mut entries = self.entries.borrow_mut();
        entries.insert(name.to_string(), Entry {
            value: default,
            min,
            max,
            step,
            default,
            category: category.to_string(),
        });
        self.order.push(name.to_string());
    }

    /// Get the current value of a tweakable.
    pub fn get(&self, name: &str) -> f64 {
        self.entries.borrow().get(name).map(|e| e.value).unwrap_or_else(|| {
            log::warn!("Tweakable '{name}' not found");
            0.0
        })
    }

    /// Check if a name is registered as a tweakable.
    pub fn has(&self, name: &str) -> bool {
        self.entries.borrow().contains_key(name)
    }

    /// Clear all registered tweakables and remove the panel.
    pub fn clear(&mut self) {
        self.entries.borrow_mut().clear();
        self.order.clear();
        self.panel_built = false;
        // Remove existing panel from DOM
        if let Some(doc) = web_sys::window().and_then(|w| w.document()) {
            if let Some(panel) = doc.get_element_by_id("tweak-panel") {
                if let Some(parent) = panel.parent_node() {
                    let _ = parent.remove_child(&panel);
                }
            }
        }
    }

    /// Build the HTML panel. Call once after all values are registered.
    pub fn build_panel(&mut self) {
        if self.panel_built {
            return;
        }
        self.panel_built = true;

        let document = web_sys::window().unwrap().document().unwrap();

        // Create panel container
        let panel: web_sys::HtmlDivElement = document
            .create_element("div").unwrap()
            .dyn_into().unwrap();
        panel.set_id("tweak-panel");
        panel.style().set_css_text(
            "position: fixed; top: 0; right: 0; width: 260px; height: 100vh; \
             overflow-y: auto; background: #1a1b26; color: #c0caf5; \
             font-family: monospace; font-size: 12px; padding: 12px; \
             box-sizing: border-box; border-left: 1px solid #2a2b36; \
             z-index: 1000; user-select: none;"
        );

        // Title
        let title: web_sys::HtmlDivElement = document
            .create_element("div").unwrap()
            .dyn_into().unwrap();
        title.set_inner_html("tweaks");
        title.style().set_css_text(
            "font-size: 14px; font-weight: bold; margin-bottom: 12px; \
             color: #7aa2f7; text-transform: uppercase; letter-spacing: 2px;"
        );
        panel.append_child(&title).unwrap();

        // Group by category
        let entries = self.entries.borrow();
        let mut categories: Vec<String> = Vec::new();
        for name in &self.order {
            if let Some(entry) = entries.get(name) {
                if !categories.contains(&entry.category) {
                    categories.push(entry.category.clone());
                }
            }
        }
        drop(entries);

        for cat in &categories {
            // Category header
            let header: web_sys::HtmlDivElement = document
                .create_element("div").unwrap()
                .dyn_into().unwrap();
            header.set_inner_html(cat);
            header.style().set_css_text(
                "font-size: 11px; color: #565f89; margin-top: 12px; margin-bottom: 6px; \
                 text-transform: uppercase; letter-spacing: 1px;"
            );
            panel.append_child(&header).unwrap();

            let entries = self.entries.borrow();
            let names_in_cat: Vec<String> = self.order.iter()
                .filter(|n| entries.get(*n).map(|e| &e.category) == Some(cat))
                .cloned()
                .collect();
            drop(entries);

            for name in &names_in_cat {
                self.build_slider(&document, &panel, name);
            }
        }

        document.body().unwrap().append_child(&panel).unwrap();
    }

    fn build_slider(
        &self,
        document: &web_sys::Document,
        panel: &web_sys::HtmlDivElement,
        name: &str,
    ) {
        let entries = self.entries.borrow();
        let entry = entries.get(name).unwrap();

        let row: web_sys::HtmlDivElement = document
            .create_element("div").unwrap()
            .dyn_into().unwrap();
        row.style().set_css_text("margin-bottom: 8px;");

        // Label row with name and value display
        let label_row: web_sys::HtmlDivElement = document
            .create_element("div").unwrap()
            .dyn_into().unwrap();
        label_row.style().set_css_text(
            "display: flex; justify-content: space-between; margin-bottom: 2px;"
        );

        let label: web_sys::HtmlSpanElement = document
            .create_element("span").unwrap()
            .dyn_into().unwrap();
        label.set_inner_html(name);
        label.style().set_css_text("color: #a9b1d6;");

        let value_display: web_sys::HtmlSpanElement = document
            .create_element("span").unwrap()
            .dyn_into().unwrap();
        value_display.set_id(&format!("tweak-val-{name}"));
        value_display.set_inner_html(&format_value(entry.value, entry.step));
        value_display.style().set_css_text("color: #7aa2f7;");

        label_row.append_child(&label).unwrap();
        label_row.append_child(&value_display).unwrap();

        // Slider
        let slider: web_sys::HtmlInputElement = document
            .create_element("input").unwrap()
            .dyn_into().unwrap();
        slider.set_type("range");
        slider.set_min(&entry.min.to_string());
        slider.set_max(&entry.max.to_string());
        slider.set_step(&entry.step.to_string());
        slider.set_value(&entry.value.to_string());
        slider.style().set_css_text(
            "width: 100%; height: 4px; appearance: none; background: #2a2b36; \
             border-radius: 2px; outline: none; cursor: pointer; \
             accent-color: #7aa2f7;"
        );

        let entries_ref = self.entries.clone();
        let name_owned = name.to_string();
        let step = entry.step;
        let callback = Closure::wrap(Box::new(move |_event: web_sys::Event| {
            let document = web_sys::window().unwrap().document().unwrap();
            let slider: web_sys::HtmlInputElement = document
                .get_element_by_id(&format!("tweak-slider-{}", name_owned))
                .unwrap()
                .dyn_into()
                .unwrap();
            let val: f64 = slider.value().parse().unwrap_or(0.0);

            if let Some(entry) = entries_ref.borrow_mut().get_mut(&name_owned) {
                entry.value = val;
            }

            if let Some(display) = document.get_element_by_id(&format!("tweak-val-{}", name_owned)) {
                display.set_inner_html(&format_value(val, step));
            }
        }) as Box<dyn FnMut(_)>);

        slider.set_id(&format!("tweak-slider-{name}"));
        slider.add_event_listener_with_callback("input", callback.as_ref().unchecked_ref()).unwrap();
        callback.forget(); // leak the closure so it lives forever

        row.append_child(&label_row).unwrap();
        row.append_child(&slider).unwrap();
        panel.append_child(&row).unwrap();
    }
}

fn format_value(val: f64, step: f64) -> String {
    if step >= 1.0 {
        format!("{}", val as i64)
    } else if step >= 0.1 {
        format!("{:.1}", val)
    } else if step >= 0.01 {
        format!("{:.2}", val)
    } else {
        format!("{:.3}", val)
    }
}
