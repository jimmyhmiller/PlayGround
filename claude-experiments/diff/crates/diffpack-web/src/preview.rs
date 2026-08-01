//! Framework-neutral production build preview server.

use std::io::{BufReader, Write};
use std::net::{TcpListener, TcpStream};
use std::path::Path;

use crate::http::{parse_request_line, read_head};
use crate::static_files::content_type;

pub fn preview(build_dir: &Path, port: u16) -> Result<(), String> {
    if !build_dir.join("index.html").is_file() {
        return Err(format!(
            "{} has no index.html — run `diffpack build <root>` first",
            build_dir.display()
        ));
    }
    let listener = TcpListener::bind(("127.0.0.1", port))
        .map_err(|error| format!("cannot bind preview port {port}: {error}"))?;
    println!(
        "diffpack preview serving {} on http://127.0.0.1:{port}",
        build_dir.display()
    );
    for connection in listener.incoming() {
        let Ok(stream) = connection else { continue };
        let build_dir = build_dir.to_path_buf();
        let _ = std::thread::Builder::new()
            .name("diffpack-preview-conn".into())
            .spawn(move || {
                let _ = handle_connection(stream, &build_dir);
            });
    }
    Ok(())
}

fn handle_connection(mut stream: TcpStream, build_dir: &Path) -> Result<(), String> {
    let mut reader = BufReader::new(
        stream
            .try_clone()
            .map_err(|error| format!("cannot clone preview socket: {error}"))?,
    );
    let (request_line, _) = read_head(&mut reader)?;
    let (method, target) = parse_request_line(&request_line)?;
    let head_only = method.eq_ignore_ascii_case("HEAD");
    let path = target.split('?').next().unwrap_or(&target);
    let relative = path.trim_start_matches('/');
    let traversal = relative
        .split('/')
        .any(|segment| matches!(segment, "." | ".."));
    if !traversal && !relative.is_empty() {
        let candidate = build_dir.join(relative);
        if candidate.is_file() {
            let bytes = std::fs::read(&candidate)
                .map_err(|error| format!("cannot read {}: {error}", candidate.display()))?;
            return write_response(
                &mut stream,
                "200 OK",
                content_type(&candidate),
                &bytes,
                head_only,
            );
        }
        if relative
            .rsplit('/')
            .next()
            .is_some_and(|last| last.contains('.'))
        {
            return write_response(
                &mut stream,
                "404 Not Found",
                "text/plain; charset=utf-8",
                b"not found",
                head_only,
            );
        }
    }
    let index = build_dir.join("index.html");
    let bytes = std::fs::read(&index)
        .map_err(|error| format!("cannot read {}: {error}", index.display()))?;
    write_response(
        &mut stream,
        "200 OK",
        "text/html; charset=utf-8",
        &bytes,
        head_only,
    )
}

fn write_response(
    stream: &mut TcpStream,
    status: &str,
    content_type: &str,
    body: &[u8],
    head_only: bool,
) -> Result<(), String> {
    let header = format!(
        "HTTP/1.1 {status}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    );
    stream
        .write_all(header.as_bytes())
        .map_err(|error| format!("cannot write preview response head: {error}"))?;
    if !head_only {
        stream
            .write_all(body)
            .map_err(|error| format!("cannot write preview response body: {error}"))?;
    }
    stream
        .flush()
        .map_err(|error| format!("cannot flush preview response: {error}"))
}
