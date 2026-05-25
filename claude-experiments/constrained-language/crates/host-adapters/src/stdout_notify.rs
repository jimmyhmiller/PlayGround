//! NotifyUser adapter that prints to stdout. Useful as the terminal in the
//! agent loop's effect chain so the final answer is visible.

use std::sync::{Arc, Mutex};

use runtime::effect::{Adapter, AdapterResult};
use runtime::Value;

/// Prints `request.text` to stdout (one line) and records every line for
/// assertions. The request schema is expected to be a record with at least a
/// `text: string` field.
#[derive(Default, Clone)]
pub struct StdoutNotifyAdapter {
    lines: Arc<Mutex<Vec<String>>>,
}

impl StdoutNotifyAdapter {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn lines(&self) -> Vec<String> {
        self.lines.lock().unwrap().clone()
    }
}

impl Adapter for StdoutNotifyAdapter {
    fn fulfill(&mut self, request: Value, _emit_id: u64) -> AdapterResult {
        let text = request
            .get("text")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        println!("{text}");
        self.lines.lock().unwrap().push(text);
        AdapterResult::Ok(Value::String("ok".into()))
    }
}
