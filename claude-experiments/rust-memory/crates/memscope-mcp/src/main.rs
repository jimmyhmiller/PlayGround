//! A minimal **MCP** (Model Context Protocol) stdio server exposing memscope's
//! recording analysis as tools an AI agent can call directly: `marks`, `diff`,
//! `analyze`, `query`. It's a thin wrapper — each tool shells out to the
//! `memscope` CLI with `--json` and returns the structured result — so the agent
//! debugging loop (`marks` → `diff` → `analyze` → `query` → edit → re-`diff`)
//! lives inside a context window.
//!
//! Transport is newline-delimited JSON-RPC 2.0 over stdin/stdout, per the MCP
//! stdio transport. Point an MCP client at this binary; set `MEMSCOPE_BIN` if the
//! `memscope` CLI isn't a sibling of this executable or on `PATH`.

use std::io::{BufRead, Write};
use std::path::PathBuf;
use std::process::Command;

use serde_json::{json, Value};

const PROTOCOL_VERSION: &str = "2024-11-05";

fn main() {
    let stdin = std::io::stdin();
    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => break,
        };
        if line.trim().is_empty() {
            continue;
        }
        let req: Value = match serde_json::from_str(&line) {
            Ok(v) => v,
            Err(_) => continue, // ignore garbage
        };
        let id = req.get("id").cloned();
        let method = req.get("method").and_then(|m| m.as_str()).unwrap_or("");

        // Notifications (no id) are acknowledged silently.
        let response = match method {
            "initialize" => Some(ok(id, initialize_result())),
            "ping" => Some(ok(id, json!({}))),
            "tools/list" => Some(ok(id, json!({ "tools": tool_specs() }))),
            "tools/call" => Some(handle_call(id, req.get("params"))),
            _ if id.is_some() => Some(err(id, -32601, &format!("method not found: {method}"))),
            _ => None, // notification we don't care about
        };
        if let Some(resp) = response {
            let _ = writeln!(out, "{resp}");
            let _ = out.flush();
        }
    }
}

fn initialize_result() -> Value {
    json!({
        "protocolVersion": PROTOCOL_VERSION,
        "capabilities": { "tools": {} },
        "serverInfo": { "name": "memscope", "version": env!("CARGO_PKG_VERSION") }
    })
}

fn ok(id: Option<Value>, result: Value) -> Value {
    json!({ "jsonrpc": "2.0", "id": id.unwrap_or(Value::Null), "result": result })
}

fn err(id: Option<Value>, code: i64, message: &str) -> Value {
    json!({ "jsonrpc": "2.0", "id": id.unwrap_or(Value::Null), "error": { "code": code, "message": message } })
}

/// The tool definitions advertised to the client.
fn tool_specs() -> Value {
    json!([
        {
            "name": "memscope_marks",
            "description": "List the named checkpoints (memscope::mark) in a recording, with the live heap size and top types at each. Use first to see the shape of memory growth and pick two marks to diff.",
            "inputSchema": {
                "type": "object",
                "properties": { "file": { "type": "string", "description": "Path to a .mscope/.jsonl recording" } },
                "required": ["file"]
            }
        },
        {
            "name": "memscope_diff",
            "description": "Diff the live heap between two checkpoints (set-diff by type+site). Shows what grew, what shrank, and born/freed-in-window — born>0 with freed=0 is a leak. A/B are mark labels, or 'start'/'end' for the stream ends.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "file": { "type": "string", "description": "Path to a recording" },
                    "a": { "type": "string", "description": "Start checkpoint label, or 'start'" },
                    "b": { "type": "string", "description": "End checkpoint label, or 'end'" }
                },
                "required": ["file", "a", "b"]
            }
        },
        {
            "name": "memscope_analyze",
            "description": "Ranked memory findings over the whole recording: leaks, allocation churn, realloc-thrash, short-lived boxes. Each has a severity, a source location, and a fix class. Start here to find what's wrong.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "file": { "type": "string", "description": "Path to a recording" },
                    "top": { "type": "integer", "description": "Max findings to return (default 20)" }
                },
                "required": ["file"]
            }
        },
        {
            "name": "memscope_query",
            "description": "Drill into one finding: its full call stack ('stack'), a freed-allocation lifetime histogram ('lifetimes'), aggregate stats ('stats'), or every call site of a type ('sites'). Identify the target by site id or type label.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "file": { "type": "string", "description": "Path to a recording" },
                    "site": { "type": "integer", "description": "Site id (from a finding)" },
                    "type": { "type": "string", "description": "Type label, e.g. 'StringBuf<u8>' (alternative to site)" },
                    "field": { "type": "string", "enum": ["stack", "lifetimes", "stats", "sites"], "description": "What to return (default stats)" }
                },
                "required": ["file"]
            }
        }
    ])
}

/// Dispatch a `tools/call`: build the CLI argv, run it, wrap stdout as MCP content.
fn handle_call(id: Option<Value>, params: Option<&Value>) -> Value {
    let Some(params) = params else {
        return err(id, -32602, "missing params");
    };
    let name = params.get("name").and_then(|n| n.as_str()).unwrap_or("");
    let args = params.get("arguments").cloned().unwrap_or(json!({}));

    let argv = match build_argv(name, &args) {
        Ok(a) => a,
        Err(e) => return ok(id, tool_error(&e)),
    };

    match run_memscope(&argv) {
        Ok(stdout) => ok(id, tool_text(&stdout)),
        Err(e) => ok(id, tool_error(&e)),
    }
}

/// Map a tool name + arguments to a `memscope` CLI argv (always `--json`).
fn build_argv(name: &str, args: &Value) -> Result<Vec<String>, String> {
    let file = args
        .get("file")
        .and_then(|f| f.as_str())
        .ok_or("missing required argument: file")?
        .to_string();
    let s = |k: &str| args.get(k).and_then(|v| v.as_str()).map(String::from);
    let i = |k: &str| args.get(k).and_then(|v| v.as_i64());

    match name {
        "memscope_marks" => Ok(vec!["marks".into(), file, "--json".into()]),
        "memscope_diff" => {
            let a = s("a").ok_or("missing required argument: a")?;
            let b = s("b").ok_or("missing required argument: b")?;
            Ok(vec!["diff".into(), file, a, b, "--json".into()])
        }
        "memscope_analyze" => {
            let mut v = vec!["analyze".into(), file, "--json".into()];
            if let Some(top) = i("top") {
                v.push("--top".into());
                v.push(top.to_string());
            }
            Ok(v)
        }
        "memscope_query" => {
            let mut v = vec!["query".into(), file];
            match (i("site"), s("type")) {
                (Some(site), _) => {
                    v.push("--site".into());
                    v.push(site.to_string());
                }
                (None, Some(ty)) => {
                    v.push("--type".into());
                    v.push(ty);
                }
                (None, None) => return Err("query needs 'site' or 'type'".into()),
            }
            if let Some(field) = s("field") {
                v.push("--field".into());
                v.push(field);
            }
            v.push("--json".into());
            Ok(v)
        }
        other => Err(format!("unknown tool: {other}")),
    }
}

/// Locate the `memscope` CLI: `$MEMSCOPE_BIN`, else a sibling of this executable,
/// else `memscope` on `PATH`.
fn memscope_bin() -> String {
    if let Ok(p) = std::env::var("MEMSCOPE_BIN") {
        return p;
    }
    if let Ok(exe) = std::env::current_exe() {
        let sibling: PathBuf = exe.with_file_name("memscope");
        if sibling.exists() {
            return sibling.to_string_lossy().into_owned();
        }
    }
    "memscope".to_string()
}

fn run_memscope(argv: &[String]) -> Result<String, String> {
    let bin = memscope_bin();
    let output = Command::new(&bin)
        .args(argv)
        .output()
        .map_err(|e| format!("failed to run {bin}: {e}"))?;
    if output.status.success() {
        Ok(String::from_utf8_lossy(&output.stdout).into_owned())
    } else {
        let stderr = String::from_utf8_lossy(&output.stderr);
        Err(format!("memscope {}: {}", argv.first().map(String::as_str).unwrap_or(""), stderr.trim()))
    }
}

fn tool_text(text: &str) -> Value {
    json!({ "content": [ { "type": "text", "text": text } ], "isError": false })
}

fn tool_error(message: &str) -> Value {
    json!({ "content": [ { "type": "text", "text": message } ], "isError": true })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn analyze_argv_includes_json_and_top() {
        let argv = build_argv("memscope_analyze", &json!({ "file": "r.mscope", "top": 5 })).unwrap();
        assert_eq!(argv, ["analyze", "r.mscope", "--json", "--top", "5"]);
    }

    #[test]
    fn diff_argv_maps_a_b() {
        let argv =
            build_argv("memscope_diff", &json!({ "file": "r.mscope", "a": "warm", "b": "end" }))
                .unwrap();
        assert_eq!(argv, ["diff", "r.mscope", "warm", "end", "--json"]);
    }

    #[test]
    fn query_prefers_site_over_type() {
        let argv = build_argv(
            "memscope_query",
            &json!({ "file": "r.mscope", "site": 41, "type": "X", "field": "stack" }),
        )
        .unwrap();
        assert_eq!(argv, ["query", "r.mscope", "--site", "41", "--field", "stack", "--json"]);
    }

    #[test]
    fn query_by_type_when_no_site() {
        let argv =
            build_argv("memscope_query", &json!({ "file": "r.mscope", "type": "Vec<u8>" })).unwrap();
        assert_eq!(argv, ["query", "r.mscope", "--type", "Vec<u8>", "--json"]);
    }

    #[test]
    fn missing_file_is_an_error() {
        assert!(build_argv("memscope_marks", &json!({})).is_err());
    }

    #[test]
    fn unknown_tool_is_an_error() {
        assert!(build_argv("memscope_bogus", &json!({ "file": "r.mscope" })).is_err());
    }
}
