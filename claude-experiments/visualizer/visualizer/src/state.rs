use std::collections::HashMap;

/// Simple key-value state store for application data.
/// Tweakables are for user-adjustable params with sliders.
/// State is for values that drive the visualization programmatically.
pub struct State {
    values: HashMap<String, f64>,
    lists: HashMap<String, Vec<HashMap<String, f64>>>,
}

impl State {
    pub fn new() -> Self {
        Self {
            values: HashMap::new(),
            lists: HashMap::new(),
        }
    }

    pub fn set(&mut self, key: &str, value: f64) {
        self.values.insert(key.to_string(), value);
    }

    pub fn get(&self, key: &str) -> f64 {
        self.values.get(key).copied().unwrap_or(0.0)
    }

    pub fn set_list(&mut self, key: &str, list: Vec<HashMap<String, f64>>) {
        self.lists.insert(key.to_string(), list);
    }

    pub fn list_len(&self, key: &str) -> usize {
        self.lists.get(key).map(|l| l.len()).unwrap_or(0)
    }

    pub fn list_push(&mut self, key: &str, item: HashMap<String, f64>) {
        self.lists.entry(key.to_string()).or_default().push(item);
    }

    pub fn list_get(&self, list_key: &str, index: usize, field: &str) -> f64 {
        self.lists
            .get(list_key)
            .and_then(|l| l.get(index))
            .and_then(|m| m.get(field))
            .copied()
            .unwrap_or(0.0)
    }

    pub fn list_set(&mut self, list_key: &str, index: usize, field: &str, value: f64) {
        if let Some(list) = self.lists.get_mut(list_key) {
            if let Some(item) = list.get_mut(index) {
                item.insert(field.to_string(), value);
            }
        }
    }
}
