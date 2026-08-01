//! HTTP response adaptation for the development reverse proxy.

use std::io::{Read, Write};
use std::net::TcpStream;

use crate::hmr;
use crate::http::{decode_chunked, find_subsequence};
use crate::response::{html_escape, inject_html};

pub const REFRESH_RUNTIME_PATH: &str = "/__diffpack_hmr/refresh-runtime.js";
pub const WS_PATH: &str = "/__diffpack_hmr/ws";

/// A parsed upstream HTTP response with any chunked body decoded.
pub struct UpstreamResponse {
    pub status_line: String,
    pub headers: Vec<(String, String)>,
    pub body: Vec<u8>,
}

pub fn parse(raw: Vec<u8>) -> Result<UpstreamResponse, String> {
    let split = find_subsequence(&raw, b"\r\n\r\n")
        .ok_or_else(|| "malformed node response (no header terminator)".to_string())?;
    let head = std::str::from_utf8(&raw[..split])
        .map_err(|error| format!("non-utf8 response headers from node: {error}"))?;
    let mut lines = head.split("\r\n");
    let status_line = lines
        .next()
        .ok_or_else(|| "empty node response".to_string())?
        .to_string();
    let mut headers = Vec::new();
    let mut chunked = false;
    for line in lines {
        if let Some((name, value)) = line.split_once(':') {
            let name = name.trim().to_string();
            let value = value.trim().to_string();
            if name.eq_ignore_ascii_case("transfer-encoding")
                && value.to_ascii_lowercase().contains("chunked")
            {
                chunked = true;
            }
            headers.push((name, value));
        }
    }
    let raw_body = &raw[split + 4..];
    let body = if chunked {
        decode_chunked(raw_body)?
    } else {
        raw_body.to_vec()
    };
    Ok(UpstreamResponse {
        status_line,
        headers,
        body,
    })
}

/// Forward one HTTP request to a loopback runtime and return its decoded response.
pub fn forward(
    port: u16,
    method: &str,
    target: &str,
    headers: &[(String, String)],
    body: &[u8],
) -> Result<UpstreamResponse, String> {
    let mut upstream = TcpStream::connect(("127.0.0.1", port))
        .map_err(|error| format!("cannot reach node runtime on :{port}: {error}"))?;
    let mut request = format!("{method} {target} HTTP/1.1\r\n");
    for (name, value) in headers {
        if matches!(
            name.to_ascii_lowercase().as_str(),
            "connection" | "accept-encoding" | "content-length" | "transfer-encoding"
        ) {
            continue;
        }
        request.push_str(name);
        request.push_str(": ");
        request.push_str(value);
        request.push_str("\r\n");
    }
    request.push_str("Connection: close\r\nAccept-Encoding: identity\r\n");
    request.push_str(&format!("Content-Length: {}\r\n\r\n", body.len()));
    upstream
        .write_all(request.as_bytes())
        .and_then(|()| upstream.write_all(body))
        .and_then(|()| upstream.flush())
        .map_err(|error| format!("cannot send request to node: {error}"))?;
    let mut raw = Vec::new();
    upstream
        .read_to_end(&mut raw)
        .map_err(|error| format!("cannot read node response: {error}"))?;
    parse(raw)
}

/// Inject the dev client into HTML (or turn a plain-text 5xx into an overlay page)
/// and serialize the response with correct framing.
pub fn inject_hmr(mut response: UpstreamResponse) -> Vec<u8> {
    let content_type = response
        .headers
        .iter()
        .find(|(name, _)| name.eq_ignore_ascii_case("content-type"))
        .map(|(_, value)| value.to_ascii_lowercase())
        .unwrap_or_default();
    if content_type.contains("text/html") {
        response.body = inject_html(&response.body, &preamble());
    } else if is_server_error(&response.status_line) && content_type.contains("text/plain") {
        response.body = ssr_error_document(&String::from_utf8_lossy(&response.body)).into_bytes();
        set_content_type_html(&mut response.headers);
    }

    let mut out = Vec::new();
    out.extend_from_slice(response.status_line.as_bytes());
    out.extend_from_slice(b"\r\n");
    for (name, value) in &response.headers {
        if matches!(
            name.to_ascii_lowercase().as_str(),
            "content-length" | "transfer-encoding" | "connection" | "content-encoding"
        ) {
            continue;
        }
        out.extend_from_slice(name.as_bytes());
        out.extend_from_slice(b": ");
        out.extend_from_slice(value.as_bytes());
        out.extend_from_slice(b"\r\n");
    }
    out.extend_from_slice(format!("Content-Length: {}\r\n", response.body.len()).as_bytes());
    out.extend_from_slice(b"Connection: close\r\n\r\n");
    out.extend_from_slice(&response.body);
    out
}

pub fn ssr_error_document(error: &str) -> String {
    format!(
        "<!doctype html><html><head>{}</head><body><pre id=\"__diffpack_ssr_error\" style=\"display:none\">{}</pre><script>if(window.__diffpackOverlay)window.__diffpackOverlay.showBuild({{message:document.getElementById(\"__diffpack_ssr_error\").textContent}});</script></body></html>",
        preamble(),
        html_escape(error),
    )
}

pub fn inject_document(body: &[u8]) -> Vec<u8> {
    inject_html(body, &preamble())
}

pub fn preamble() -> String {
    format!(
        "<script src=\"{REFRESH_RUNTIME_PATH}\"></script><script>{}</script><script>{}</script>",
        hmr::client_script(WS_PATH),
        hmr::overlay_script(),
    )
}

fn is_server_error(status_line: &str) -> bool {
    status_line
        .split_whitespace()
        .nth(1)
        .and_then(|code| code.chars().next())
        == Some('5')
}

fn set_content_type_html(headers: &mut Vec<(String, String)>) {
    if let Some((_, value)) = headers
        .iter_mut()
        .find(|(name, _)| name.eq_ignore_ascii_case("content-type"))
    {
        *value = "text/html; charset=utf-8".to_string();
    } else {
        headers.push((
            "Content-Type".to_string(),
            "text/html; charset=utf-8".to_string(),
        ));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_plain_and_chunked_responses() {
        let plain = parse(b"HTTP/1.1 200 OK\r\nContent-Length: 5\r\n\r\nhello".to_vec()).unwrap();
        assert_eq!(plain.body, b"hello");
        let chunked = parse(b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n4\r\nWiki\r\n5\r\npedia\r\n0\r\n\r\n".to_vec()).unwrap();
        assert_eq!(chunked.body, b"Wikipedia");
    }

    #[test]
    fn leaves_non_html_responses_untouched() {
        let out = inject_hmr(UpstreamResponse {
            status_line: "HTTP/1.1 200 OK".into(),
            headers: vec![("Content-Type".into(), "application/javascript".into())],
            body: b"console.log(1)".to_vec(),
        });
        let text = String::from_utf8(out).unwrap();
        assert!(!text.contains("$RefreshRuntime$"));
        assert!(text.contains("Content-Length: 14"));
    }

    #[test]
    fn turns_plain_text_5xx_into_overlay_document() {
        let out = inject_hmr(UpstreamResponse {
            status_line: "HTTP/1.1 500 Internal Server Error".into(),
            headers: vec![("Content-Type".into(), "text/plain".into())],
            body: b"ReferenceError: boom".to_vec(),
        });
        let text = String::from_utf8(out).unwrap();
        assert!(text.contains("text/html"));
        assert!(text.contains("__diffpackOverlay"));
        assert!(text.contains("ReferenceError: boom"));
    }

    #[test]
    fn injected_document_orders_runtime_client_and_overlay_in_head() {
        let text = String::from_utf8(inject_document(
            b"<!doctype html><html><head><title>x</title></head><body></body></html>",
        ))
        .unwrap();
        let head = text.find("<head>").unwrap();
        let runtime = text.find(REFRESH_RUNTIME_PATH).unwrap();
        let client = text.find("WebSocket").unwrap();
        let title = text.find("<title>").unwrap();
        assert!(head < runtime && runtime < client && client < title);
        assert!(text.contains("__diffpackOverlay"));
        assert!(!text.contains("type=\"module\">"));

        let without_head =
            String::from_utf8(inject_document(b"<html><body><p>hi</p></body></html>")).unwrap();
        assert!(
            without_head.find(REFRESH_RUNTIME_PATH).unwrap()
                < without_head.find("</body>").unwrap()
        );
    }
}
