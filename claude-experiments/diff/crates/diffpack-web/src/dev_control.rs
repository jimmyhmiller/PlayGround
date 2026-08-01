//! Shared browser and Node hot-update control operations.

use std::collections::BTreeSet;
use std::io::{Read, Write};
use std::net::TcpStream;
use std::path::Path;
use std::time::Duration;

use diffpack_core::ModuleFormat;

use crate::dev_build::EnvBuild;
use crate::websocket::HmrHub;

pub fn push_client(
    client: &EnvBuild,
    changed_ids: &BTreeSet<String>,
    hub: &HmrHub,
    micro_chunk: Option<&Path>,
) -> String {
    if changed_ids.is_empty() {
        return "client: no change".to_string();
    }
    let reachable = client.reachable_ids();
    let located = match client
        .bundler
        .hmr_locate(&reachable, changed_ids, "client.js")
    {
        Ok(located) => located,
        Err(error) => {
            hub.broadcast_reload();
            return format!("client: locate failed ({error}); reloaded");
        }
    };
    if located.is_empty() {
        return "client: no located modules".to_string();
    }
    let ids = located
        .iter()
        .map(|location| location.runtime_id)
        .collect::<Vec<_>>();
    let chunks: BTreeSet<String> = if let Some(output_root) = micro_chunk {
        match client.bundler.write_hmr_chunk(
            &reachable,
            changed_ids,
            "client.js",
            client.options,
            ModuleFormat::BrowserEsm,
            &output_root.join("public/client.hmr.js"),
        ) {
            Ok(true) => std::iter::once("/client.hmr.js".to_string()).collect(),
            Ok(false) => return "client: no live changed module for micro-chunk".to_string(),
            Err(error) => {
                hub.broadcast_reload();
                return format!("client: micro-chunk render/write failed ({error}); reloaded");
            }
        }
    } else {
        located
            .iter()
            .map(|location| format!("/{}", location.chunk_file))
            .collect()
    };
    let message = format!(
        "{{\"type\":\"update\",\"ids\":[{}],\"chunks\":[{}]}}",
        ids.iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join(","),
        chunks
            .iter()
            .map(|chunk| json_string(chunk))
            .collect::<Vec<_>>()
            .join(","),
    );
    hub.send(&message);
    format!(
        "client: hmr update -> {} module(s) in {} chunk(s), {} browser(s)",
        ids.len(),
        chunks.len(),
        hub.client_count()
    )
}

pub fn json_string(value: &str) -> String {
    serde_json::to_string(value).unwrap_or_else(|_| "\"\"".to_string())
}

pub fn post_json(port: u16, path: &str, json: &str) -> Result<(), String> {
    let mut stream = TcpStream::connect(("127.0.0.1", port))
        .map_err(|error| format!("cannot reach 127.0.0.1:{port}{path}: {error}"))?;
    stream.set_read_timeout(Some(Duration::from_secs(30))).ok();
    let request = format!(
        "POST {path} HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{json}",
        json.len()
    );
    stream
        .write_all(request.as_bytes())
        .and_then(|()| stream.flush())
        .map_err(|error| format!("cannot send control request: {error}"))?;
    let mut response = Vec::new();
    stream
        .read_to_end(&mut response)
        .map_err(|error| format!("cannot read control response: {error}"))?;
    let response = String::from_utf8_lossy(&response);
    if response.starts_with("HTTP/1.1 200") || response.starts_with("HTTP/1.0 200") {
        return Ok(());
    }
    let (status, body) = response
        .split_once("\r\n\r\n")
        .map_or((response.as_ref(), ""), |(head, body)| {
            (head.lines().next().unwrap_or("<no status>"), body)
        });
    Err(format!("control endpoint returned {status}: {body}"))
}
