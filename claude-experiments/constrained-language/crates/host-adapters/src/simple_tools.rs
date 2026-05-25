//! A trivial tool-execution adapter. Recognises a handful of named tools and
//! returns their `result` as a string. Matches the agent loop's `ExecuteTool`
//! effect request shape: `{ goal_id, tool, args }`.

use std::time::{SystemTime, UNIX_EPOCH};

use runtime::effect::{Adapter, AdapterResult};
use runtime::Value;

/// Tools shipped out of the box:
///
/// * `calc(args)`   — evaluate a math expression with `meval`. Returns the
///                    result formatted as a decimal string.
/// * `time()`       — return the current UTC unix-epoch seconds.
/// * `echo(args)`   — return the args verbatim. Useful as a smoke test.
///
/// Unknown tool names produce a `Failed` outcome.
#[derive(Default, Clone)]
pub struct SimpleToolsAdapter {}

impl SimpleToolsAdapter {
    pub fn new() -> Self {
        Self::default()
    }
}

impl Adapter for SimpleToolsAdapter {
    fn fulfill(&mut self, request: Value, _emit_id: u64) -> AdapterResult {
        let tool = request
            .get("tool")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let args = request
            .get("args")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        match tool.as_str() {
            "calc" => match meval::eval_str(&args) {
                Ok(n) => ok_result(format!("{n}")),
                Err(e) => AdapterResult::Failed {
                    reason: format!("calc error: {e}"),
                },
            },
            "time" => {
                let secs = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .map(|d| d.as_secs())
                    .unwrap_or(0);
                ok_result(secs.to_string())
            }
            "echo" => ok_result(args),
            other => AdapterResult::Failed {
                reason: format!("unknown tool `{other}`"),
            },
        }
    }
}

fn ok_result(s: String) -> AdapterResult {
    AdapterResult::Ok(serde_json::json!({ "result": s }))
}
