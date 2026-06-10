//! A minimal synchronous HTTP/1.1 reverse proxy.
//!
//! Forwards a request to an upstream `host:port` over a fresh TCP connection
//! (`Connection: close`, so no keep-alive bookkeeping), reads the whole
//! response into memory, and returns it as a [`Reply`]. This is deliberately
//! simple — it targets local services (`127.0.0.1:PORT`) for a personal
//! gateway, not high-throughput streaming.

use std::io::{Read, Write};
use std::net::TcpStream;
use std::time::Duration;

use crate::reply::Reply;

/// Hop-by-hop headers that must not be forwarded (RFC 7230 §6.1). We also drop
/// the original Host/Authorization handling below.
const HOP_BY_HOP: &[&str] = &[
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
];

/// Forward `method` + `path` (path already normalized) to `upstream`, passing
/// the client's headers (minus hop-by-hop) and body. Returns the upstream's
/// response as a Reply, or a 502 if the upstream can't be reached.
pub fn forward(
    upstream: &str,
    method: &str,
    path: &str,
    client_headers: &[tiny_http::Header],
    body: &[u8],
) -> Reply {
    match try_forward(upstream, method, path, client_headers, body) {
        Ok(reply) => reply,
        Err(e) => Reply::status(502, &format!("Bad Gateway: {e}")),
    }
}

fn try_forward(
    upstream: &str,
    method: &str,
    path: &str,
    client_headers: &[tiny_http::Header],
    body: &[u8],
) -> std::io::Result<Reply> {
    let mut stream = TcpStream::connect(upstream)?;
    stream.set_read_timeout(Some(Duration::from_secs(30)))?;
    stream.set_write_timeout(Some(Duration::from_secs(30)))?;

    // Build the request. Use HTTP/1.1 with explicit Host and Connection: close.
    let mut req = format!("{method} {path} HTTP/1.1\r\n");
    req.push_str(&format!("Host: {upstream}\r\n"));
    req.push_str("Connection: close\r\n");

    for h in client_headers {
        let name = h.field.as_str().as_str().to_ascii_lowercase();
        if name == "host" || name == "connection" || name == "content-length" {
            continue; // we set these ourselves
        }
        if HOP_BY_HOP.contains(&name.as_str()) {
            continue;
        }
        req.push_str(&format!("{}: {}\r\n", h.field.as_str(), h.value.as_str()));
    }
    // Forwarded-for marker so the upstream knows it's behind the gate.
    req.push_str("X-Forwarded-By: gatekeeper\r\n");
    req.push_str(&format!("Content-Length: {}\r\n", body.len()));
    req.push_str("\r\n");

    stream.write_all(req.as_bytes())?;
    if !body.is_empty() {
        stream.write_all(body)?;
    }
    stream.flush()?;

    // Read the entire response (Connection: close means EOF terminates it).
    let mut raw = Vec::new();
    stream.read_to_end(&mut raw)?;

    parse_response(&raw)
}

/// Parse a raw HTTP/1.1 response into a Reply. Splits headers from body at the
/// first CRLF CRLF; passes the body through verbatim and forwards response
/// headers except hop-by-hop ones (and Transfer-Encoding, since we already have
/// the full body and serve it with our own framing).
fn parse_response(raw: &[u8]) -> std::io::Result<Reply> {
    let split = find_header_end(raw)
        .ok_or_else(|| io_err("malformed upstream response: no header terminator"))?;
    let (head, body) = raw.split_at(split);
    let body = &body[4..]; // skip the CRLF CRLF

    let head_str = String::from_utf8_lossy(head);
    let mut lines = head_str.split("\r\n");

    let status_line = lines
        .next()
        .ok_or_else(|| io_err("empty upstream response"))?;
    // "HTTP/1.1 200 OK" -> 200
    let status: u16 = status_line
        .split_whitespace()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .ok_or_else(|| io_err("bad status line from upstream"))?;

    let mut reply = Reply::new(status, body.to_vec());
    for line in lines {
        if line.is_empty() {
            continue;
        }
        let Some((name, value)) = line.split_once(':') else {
            continue;
        };
        let name = name.trim();
        let lname = name.to_ascii_lowercase();
        if HOP_BY_HOP.contains(&lname.as_str()) || lname == "content-length" {
            // We re-frame with our own Content-Length via from_data.
            continue;
        }
        reply = reply.with_header(name, value.trim());
    }
    Ok(reply)
}

/// Find the index of the CRLF CRLF that ends the header block.
fn find_header_end(raw: &[u8]) -> Option<usize> {
    raw.windows(4).position(|w| w == b"\r\n\r\n")
}

fn io_err(msg: &str) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::InvalidData, msg)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_basic_response() {
        let raw = b"HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nContent-Length: 5\r\n\r\nhello";
        let reply = parse_response(raw).unwrap();
        assert_eq!(reply.status, 200);
        assert_eq!(reply.body, b"hello");
        assert!(reply
            .headers
            .iter()
            .any(|(k, v)| k == "Content-Type" && v == "text/plain"));
        // Content-Length is dropped (we re-frame).
        assert!(!reply.headers.iter().any(|(k, _)| k == "Content-Length"));
    }

    #[test]
    fn parse_404() {
        let raw = b"HTTP/1.1 404 Not Found\r\n\r\n";
        let reply = parse_response(raw).unwrap();
        assert_eq!(reply.status, 404);
        assert!(reply.body.is_empty());
    }
}
