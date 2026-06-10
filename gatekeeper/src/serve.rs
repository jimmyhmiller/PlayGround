//! Static file serving for a route mapped to a directory.
//!
//! `rest` is the request path after the matched route prefix. It has already
//! been normalized and traversal-rejected by [`crate::route::Router::normalize`],
//! so it contains no `..` components. As belt-and-suspenders we still
//! canonicalize the final path and confirm it stays within the served root —
//! if a symlink inside the root points outside, we refuse.

use std::path::Path;

use crate::reply::Reply;

/// Serve `rest` (e.g. "/index.html" or "" ) from under `root`.
pub fn serve(root: &Path, rest: &str) -> Reply {
    // Map the request remainder onto the filesystem. `rest` starts with '/' or
    // is empty; strip the leading slash so join() treats it as relative.
    let rel = rest.trim_start_matches('/');
    let mut path = root.join(rel);

    // Directory -> index.html (simple default; no autoindex, by design).
    if path.is_dir() {
        path = path.join("index.html");
    }

    // Canonicalize and confirm containment. canonicalize() also resolves
    // symlinks, so this catches a symlink escaping the root.
    let (canon_root, canon_path) = match (root.canonicalize(), path.canonicalize()) {
        (Ok(r), Ok(p)) => (r, p),
        // A missing file fails to canonicalize -> 404 (don't leak which part).
        _ => return Reply::status(404, "Not Found"),
    };
    if !canon_path.starts_with(&canon_root) {
        // Symlink or join escaped the root. Treat as not found.
        return Reply::status(404, "Not Found");
    }

    match std::fs::read(&canon_path) {
        Ok(bytes) => {
            let ct = content_type(&canon_path);
            Reply::new(200, bytes).with_header("Content-Type", ct)
        }
        Err(_) => Reply::status(404, "Not Found"),
    }
}

/// A minimal extension → MIME map. Unknown types fall back to octet-stream.
fn content_type(path: &Path) -> &'static str {
    match path.extension().and_then(|e| e.to_str()).unwrap_or("") {
        "html" | "htm" => "text/html; charset=utf-8",
        "css" => "text/css; charset=utf-8",
        "js" | "mjs" => "text/javascript; charset=utf-8",
        "json" => "application/json",
        "svg" => "image/svg+xml",
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        "gif" => "image/gif",
        "webp" => "image/webp",
        "ico" => "image/x-icon",
        "txt" | "md" => "text/plain; charset=utf-8",
        "wasm" => "application/wasm",
        "woff2" => "font/woff2",
        _ => "application/octet-stream",
    }
}

