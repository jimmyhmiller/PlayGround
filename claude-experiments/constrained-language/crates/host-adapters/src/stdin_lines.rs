//! Stdin-lines pull generator. Each `next()` call blocks on a line from
//! stdin; the line value is substituted into the configured payload
//! template (replacing `$input` in any string positions). Returns `None`
//! on EOF (Ctrl-D).
//!
//! The generator owns no state besides the template; threads-per-stdin
//! semantics are managed by the runtime's `start_generators`.

use runtime::generator::Generator;
use runtime::Value;
use serde_json::Value as Json;
use std::io::{self, BufRead};

#[derive(Clone)]
pub struct StdinLinesGenerator {
    payload_template: Json,
}

impl StdinLinesGenerator {
    /// `payload_template` is a JSON value that may contain `$input`
    /// placeholders in string positions. Each stdin line is substituted in.
    /// If the template is `null`, the line is emitted as a bare string.
    pub fn new(payload_template: Json) -> Self {
        Self { payload_template }
    }
}

impl Generator for StdinLinesGenerator {
    fn next(&mut self) -> Option<Value> {
        let stdin = io::stdin();
        let mut line = String::new();
        match stdin.lock().read_line(&mut line) {
            Ok(0) => None, // EOF
            Ok(_) => {
                let line = line.trim_end_matches('\n').trim_end_matches('\r').to_string();
                if line.is_empty() {
                    // Skip empty lines but stay alive.
                    return self.next();
                }
                Some(substitute(&self.payload_template, &line))
            }
            Err(_) => None,
        }
    }
}

fn substitute(template: &Json, input: &str) -> Json {
    match template {
        Json::Null => Json::String(input.to_string()),
        Json::String(s) => Json::String(s.replace("$input", input)),
        Json::Array(items) => Json::Array(items.iter().map(|v| substitute(v, input)).collect()),
        Json::Object(map) => {
            let mut out = serde_json::Map::with_capacity(map.len());
            for (k, v) in map {
                out.insert(k.clone(), substitute(v, input));
            }
            Json::Object(out)
        }
        other => other.clone(),
    }
}
