//! Static development server transport for browser-only applications.

use std::io::{BufReader, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use crate::dev_proxy::{self, ProxyRule};
use crate::dev_response::{REFRESH_RUNTIME_PATH, WS_PATH, inject_document};
use crate::http::{parse_request_line, read_body, read_head};
use crate::response::{
    accepts_gzip, if_none_match, write_file_with_method, write_javascript, write_response,
};
use crate::websocket::{HmrHub, accept};

/// Serve an emitted SPA, its development runtime, HMR socket, and configured proxies.
pub fn serve(
    listener: TcpListener,
    output_root: PathBuf,
    base: String,
    served_html: Arc<Mutex<String>>,
    hub: HmrHub,
    refresh_runtime: Arc<String>,
    proxy: Arc<Vec<ProxyRule>>,
) {
    for connection in listener.incoming() {
        let Ok(stream) = connection else { continue };
        let output_root = output_root.clone();
        let base = base.clone();
        let served_html = Arc::clone(&served_html);
        let hub = hub.clone();
        let refresh_runtime = Arc::clone(&refresh_runtime);
        let proxy = Arc::clone(&proxy);
        let _ = std::thread::Builder::new()
            .name("diffpack-dev-spa-conn".into())
            .spawn(move || {
                let _ = handle_connection(
                    stream,
                    &output_root,
                    &base,
                    &served_html,
                    &hub,
                    &refresh_runtime,
                    &proxy,
                );
            });
    }
}

fn handle_connection(
    mut stream: TcpStream,
    output_root: &Path,
    base: &str,
    served_html: &Arc<Mutex<String>>,
    hub: &HmrHub,
    refresh_runtime: &str,
    proxy: &[ProxyRule],
) -> Result<(), String> {
    let mut reader = BufReader::new(
        stream
            .try_clone()
            .map_err(|error| format!("cannot clone client socket: {error}"))?,
    );
    let (request_line, headers) = read_head(&mut reader)?;
    let (method, target) = parse_request_line(&request_line)?;
    let path = target.split('?').next().unwrap_or(&target);
    let head_only = method.eq_ignore_ascii_case("HEAD");

    if let Some(rule) = dev_proxy::match_rule(proxy, path) {
        let body = read_body(&mut reader, &headers)?;
        match dev_proxy::forward(rule, &method, &target, &headers, &body) {
            Ok(response) => stream
                .write_all(&response)
                .and_then(|()| stream.flush())
                .map_err(|error| format!("cannot write proxied response: {error}"))?,
            Err(error) => {
                eprintln!("[dev] proxy error for {path}: {error}");
                write_response(
                    &mut stream,
                    "502 Bad Gateway",
                    "text/plain; charset=utf-8",
                    error.as_bytes(),
                    head_only,
                )?;
            }
        }
        return Ok(());
    }

    if path == WS_PATH {
        if let Some((_, key)) = headers
            .iter()
            .find(|(name, _)| name.eq_ignore_ascii_case("sec-websocket-key"))
        {
            let response = format!(
                "HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Accept: {}\r\n\r\n",
                accept(key.trim())
            );
            stream
                .write_all(response.as_bytes())
                .and_then(|()| stream.flush())
                .map_err(|error| format!("cannot complete websocket handshake: {error}"))?;
            hub.send_to(&stream, r#"{"type":"connected"}"#);
            hub.register(stream);
        }
        return Ok(());
    }

    if path == REFRESH_RUNTIME_PATH {
        return write_javascript(&mut stream, refresh_runtime);
    }

    if let Some(file) = crate::static_files::resolve_with_base(output_root, base, path) {
        if file.is_file() {
            return write_file_with_method(
                &mut stream,
                &file,
                if_none_match(&headers),
                accepts_gzip(&headers),
                head_only,
            );
        }
        if crate::static_files::looks_like_file(path) {
            return write_response(
                &mut stream,
                "404 Not Found",
                "text/plain; charset=utf-8",
                b"not found",
                head_only,
            );
        }
    }

    let document = served_html.lock().unwrap().clone();
    write_response(
        &mut stream,
        "200 OK",
        "text/html; charset=utf-8",
        &inject_document(document.as_bytes()),
        head_only,
    )
}
