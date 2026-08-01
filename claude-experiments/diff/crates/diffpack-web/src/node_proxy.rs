//! Framework-neutral reverse proxy for Node-backed development adapters.

use std::io::{BufReader, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::dev_response::{
    REFRESH_RUNTIME_PATH, WS_PATH, forward as forward_to_node, inject_hmr, ssr_error_document,
};
use crate::http::{parse_request_line, read_body, read_head};
use crate::response::{accepts_gzip, if_none_match, write_file, write_javascript};
use crate::static_files::resolve as resolve_static_file;
use crate::websocket::{HmrHub, accept as ws_accept};

pub trait RequestGuard {}
impl<T> RequestGuard for T {}

/// Optional framework adapter that blocks a request until its route is built.
pub trait RouteGate: Send + Sync {
    fn incomplete(&self) -> bool;
    fn ensure(&self, path: &str) -> Result<Option<u16>, String>;
    fn serving_request(&self) -> Box<dyn RequestGuard + '_>;
}

pub fn serve(
    listener: TcpListener,
    node_port: u16,
    hub: HmrHub,
    refresh_runtime: Arc<String>,
    static_dir: Option<Arc<PathBuf>>,
    route_gate: Option<Arc<dyn RouteGate>>,
) {
    for connection in listener.incoming() {
        let Ok(stream) = connection else { continue };
        let hub = hub.clone();
        let refresh_runtime = Arc::clone(&refresh_runtime);
        let static_dir = static_dir.clone();
        let route_gate = route_gate.clone();
        let _ = std::thread::Builder::new()
            .name("diffpack-dev-conn".into())
            .spawn(move || {
                let _ = handle_connection(
                    stream,
                    node_port,
                    &hub,
                    &refresh_runtime,
                    static_dir.as_deref().map(|path| path.as_path()),
                    route_gate.as_deref(),
                );
            });
    }
}

fn handle_connection(
    mut stream: TcpStream,
    node_port: u16,
    hub: &HmrHub,
    refresh_runtime: &str,
    static_dir: Option<&Path>,
    route_gate: Option<&dyn RouteGate>,
) -> Result<(), String> {
    let mut reader = BufReader::new(
        stream
            .try_clone()
            .map_err(|error| format!("cannot clone client socket: {error}"))?,
    );
    let (request_line, headers) = read_head(&mut reader)?;
    let (method, target) = parse_request_line(&request_line)?;
    let path = target.split('?').next().unwrap_or(&target);

    if path == WS_PATH {
        if let Some((_, key)) = headers
            .iter()
            .find(|(name, _)| name.eq_ignore_ascii_case("sec-websocket-key"))
        {
            let accept = ws_accept(key.trim());
            let response = format!(
                "HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Accept: {accept}\r\n\r\n"
            );
            stream
                .write_all(response.as_bytes())
                .and_then(|()| stream.flush())
                .map_err(|error| format!("cannot complete websocket handshake: {error}"))?;
            hub.send_to(&stream, r#"{"type":"connected"}"#);
            hub.register(stream);
            return Ok(());
        }
        return Ok(());
    }

    if path == REFRESH_RUNTIME_PATH {
        write_javascript(&mut stream, refresh_runtime)?;
        return Ok(());
    }

    if let Some(file) = static_dir
        .filter(|_| method == "GET")
        .and_then(|dir| resolve_static_file(dir, path))
    {
        write_file(
            &mut stream,
            &file,
            if_none_match(&headers),
            accepts_gzip(&headers),
        )?;
        return Ok(());
    }

    let body = read_body(&mut reader, &headers)?;
    let node_port = match route_gate.filter(|gate| gate.incomplete()) {
        Some(gate) => match gate.ensure(path) {
            Ok(Some(port)) => port,
            Ok(None) => node_port,
            Err(error) => {
                let error = format!("diffpack dev: {error}");
                eprintln!("[dev] {error}");
                let document = ssr_error_document(&error);
                let response = format!(
                    "HTTP/1.1 500 Internal Server Error\r\nContent-Type: text/html; charset=utf-8\r\nCache-Control: no-store\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{document}",
                    document.len(),
                );
                stream
                    .write_all(response.as_bytes())
                    .map_err(|error| format!("cannot write build-error response: {error}"))?;
                stream.flush().ok();
                return Ok(());
            }
        },
        None => node_port,
    };
    let _in_flight = route_gate.map(|gate| gate.serving_request());
    let upstream = forward_to_node(node_port, &method, &target, &headers, &body)?;
    let response = inject_hmr(upstream);
    stream
        .write_all(&response)
        .map_err(|error| format!("cannot write response to client: {error}"))?;
    stream.flush().ok();
    Ok(())
}
