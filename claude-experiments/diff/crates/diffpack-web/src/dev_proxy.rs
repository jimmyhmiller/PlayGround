//! Native development HTTP proxy transport.

use std::io::{Read, Write};
use std::net::TcpStream;

/// Framework-neutral HTTP/WebSocket proxy rule consumed by the Web dev server.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProxyRule {
    pub context: String,
    pub target: String,
    pub change_origin: bool,
    pub ws: bool,
}

pub fn match_rule<'a>(rules: &'a [ProxyRule], path: &str) -> Option<&'a ProxyRule> {
    rules.iter().find(|rule| {
        let context = rule.context.strip_prefix('^').unwrap_or(&rule.context);
        !context.is_empty() && path.starts_with(context)
    })
}

pub fn target_host_port(target: &str) -> Result<(String, u16), String> {
    let (scheme, rest) = target.split_once("://").unwrap_or(("http", target));
    let authority = rest.split(['/', '?']).next().unwrap_or(rest);
    let default_port = if matches!(scheme, "https" | "wss") {
        443
    } else {
        80
    };
    let (host, port) = match authority.rsplit_once(':') {
        Some((host, port)) => (
            host.to_string(),
            port.parse::<u16>()
                .map_err(|error| format!("bad proxy target port in {target:?}: {error}"))?,
        ),
        None => (authority.to_string(), default_port),
    };
    if host.is_empty() {
        return Err(format!("proxy target {target:?} has no host"));
    }
    Ok((host, port))
}

pub fn forward(
    rule: &ProxyRule,
    method: &str,
    path_and_query: &str,
    headers: &[(String, String)],
    body: &[u8],
) -> Result<Vec<u8>, String> {
    let (host, port) = target_host_port(&rule.target)?;
    let mut upstream = TcpStream::connect((host.as_str(), port)).map_err(|error| {
        format!(
            "dev proxy cannot reach {} ({host}:{port}): {error}",
            rule.target
        )
    })?;
    let mut request = format!("{method} {path_and_query} HTTP/1.1\r\n");
    let mut wrote_host = false;
    for (name, value) in headers {
        let lower = name.to_ascii_lowercase();
        if lower == "host" {
            wrote_host = true;
            if rule.change_origin {
                request.push_str(&format!("Host: {host}:{port}\r\n"));
                continue;
            }
        }
        if matches!(lower.as_str(), "connection" | "accept-encoding") {
            continue;
        }
        request.push_str(&format!("{name}: {value}\r\n"));
    }
    if !wrote_host {
        request.push_str(&format!("Host: {host}:{port}\r\n"));
    }
    request.push_str("Connection: close\r\nAccept-Encoding: identity\r\n");
    request.push_str(&format!("Content-Length: {}\r\n\r\n", body.len()));
    upstream
        .write_all(request.as_bytes())
        .and_then(|()| upstream.write_all(body))
        .and_then(|()| upstream.flush())
        .map_err(|error| format!("dev proxy cannot send to {}: {error}", rule.target))?;
    let mut response = Vec::new();
    upstream
        .read_to_end(&mut response)
        .map_err(|error| format!("dev proxy cannot read from {}: {error}", rule.target))?;
    Ok(response)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::http::read_head;
    use std::io::{BufReader, Write};
    use std::net::TcpListener;

    fn rule(context: &str, target: &str) -> ProxyRule {
        ProxyRule {
            context: context.into(),
            target: target.into(),
            change_origin: false,
            ws: false,
        }
    }

    #[test]
    fn matches_and_parses_targets() {
        assert!(match_rule(&[rule("^/api", "http://localhost")], "/api/x").is_some());
        assert_eq!(
            target_host_port("https://api.example.com/base").unwrap(),
            ("api.example.com".into(), 443)
        );
    }

    #[test]
    fn forwards_to_a_live_upstream() {
        let listener = TcpListener::bind(("127.0.0.1", 0)).unwrap();
        let port = listener.local_addr().unwrap().port();
        let handle = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut reader = BufReader::new(stream.try_clone().unwrap());
            let (line, _) = read_head(&mut reader).unwrap();
            stream
                .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\nok")
                .unwrap();
            line
        });
        let response = forward(
            &rule("/api", &format!("http://127.0.0.1:{port}")),
            "GET",
            "/api/x",
            &[],
            &[],
        )
        .unwrap();
        assert!(response.ends_with(b"ok"));
        assert!(handle.join().unwrap().contains("/api/x"));
    }
}
