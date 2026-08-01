//! Static-file URL containment and response metadata.

use std::path::{Path, PathBuf};

pub fn resolve(root: &Path, url_path: &str) -> Option<PathBuf> {
    let relative = url_path.trim_start_matches('/');
    if relative.is_empty()
        || relative
            .split('/')
            .any(|segment| segment.is_empty() || matches!(segment, "." | ".."))
    {
        return None;
    }
    let candidate = root.join(relative);
    candidate.is_file().then_some(candidate)
}

/// Resolves a URL beneath an emitted site root after stripping its configured
/// public base. The document root returns `None` for SPA fallback handling.
pub fn resolve_with_base(root: &Path, base: &str, url_path: &str) -> Option<PathBuf> {
    let relative = if base == "/" {
        url_path.trim_start_matches('/')
    } else if let Some(rest) = url_path.strip_prefix(base) {
        rest
    } else if url_path == base.trim_end_matches('/') {
        ""
    } else {
        url_path.trim_start_matches('/')
    };
    if relative.is_empty()
        || relative
            .split('/')
            .any(|segment| matches!(segment, "." | ".."))
    {
        return None;
    }
    Some(root.join(relative))
}

pub fn looks_like_file(url_path: &str) -> bool {
    url_path
        .rsplit('/')
        .next()
        .is_some_and(|segment| segment.contains('.'))
}

pub fn content_type(path: &Path) -> &'static str {
    match path.extension().and_then(|value| value.to_str()) {
        Some("js" | "mjs" | "cjs") => "application/javascript; charset=utf-8",
        Some("css") => "text/css; charset=utf-8",
        Some("html") => "text/html; charset=utf-8",
        Some("json" | "map") => "application/json; charset=utf-8",
        Some("svg") => "image/svg+xml",
        Some("png") => "image/png",
        Some("jpg" | "jpeg") => "image/jpeg",
        Some("gif") => "image/gif",
        Some("webp") => "image/webp",
        Some("avif") => "image/avif",
        Some("ico") => "image/x-icon",
        Some("woff2") => "font/woff2",
        Some("woff") => "font/woff",
        Some("ttf") => "font/ttf",
        Some("otf") => "font/otf",
        Some("wasm") => "application/wasm",
        Some("txt") => "text/plain; charset=utf-8",
        _ => "application/octet-stream",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn traversal_and_empty_segments_never_escape_the_root() {
        let root = tempfile::tempdir().unwrap();
        std::fs::write(root.path().join("app.js"), "x").unwrap();
        assert_eq!(
            resolve(root.path(), "/app.js"),
            Some(root.path().join("app.js"))
        );
        assert_eq!(resolve(root.path(), "/../app.js"), None);
        assert_eq!(resolve(root.path(), "/a//b"), None);
    }

    #[test]
    fn classifies_text_and_precompressed_assets() {
        assert_eq!(
            content_type(Path::new("app.mjs")),
            "application/javascript; charset=utf-8"
        );
        assert_eq!(content_type(Path::new("font.woff2")), "font/woff2");
    }

    #[test]
    fn base_resolution_distinguishes_routes_files_and_traversal() {
        let root = Path::new("/out");
        assert_eq!(
            resolve_with_base(root, "/app/", "/app/index.js"),
            Some(root.join("index.js"))
        );
        assert_eq!(resolve_with_base(root, "/app/", "/app"), None);
        assert_eq!(resolve_with_base(root, "/", "/../secret"), None);
        assert!(looks_like_file("/assets/app.js"));
        assert!(!looks_like_file("/users/42"));
    }
}
