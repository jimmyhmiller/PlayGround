//! Browser response policy shared by development and preview servers.

use std::collections::HashMap;
use std::io::Write;
use std::net::TcpStream;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};

pub struct GzipCache {
    entries: HashMap<PathBuf, (u64, u128, Arc<Vec<u8>>)>,
    total: usize,
}

impl GzipCache {
    const MAX_ENTRY: usize = 64 * 1024 * 1024;
    const MAX_TOTAL: usize = 256 * 1024 * 1024;

    fn global() -> &'static Mutex<Self> {
        static CACHE: OnceLock<Mutex<GzipCache>> = OnceLock::new();
        CACHE.get_or_init(|| {
            Mutex::new(Self {
                entries: HashMap::new(),
                total: 0,
            })
        })
    }

    pub fn get(path: &Path, len: u64, mtime: u128, bytes: &[u8]) -> Option<Arc<Vec<u8>>> {
        if let Ok(cache) = Self::global().lock()
            && let Some((cached_len, cached_mtime, compressed)) = cache.entries.get(path)
            && *cached_len == len
            && *cached_mtime == mtime
        {
            return Some(Arc::clone(compressed));
        }
        let compressed = Arc::new(gzip_fast(bytes)?);
        if compressed.len() <= Self::MAX_ENTRY
            && let Ok(mut cache) = Self::global().lock()
        {
            if cache.total + compressed.len() > Self::MAX_TOTAL {
                cache.entries.clear();
                cache.total = 0;
            }
            if let Some((_, _, previous)) = cache
                .entries
                .insert(path.to_path_buf(), (len, mtime, Arc::clone(&compressed)))
            {
                cache.total = cache.total.saturating_sub(previous.len());
            }
            cache.total += compressed.len();
        }
        Some(compressed)
    }
}

fn gzip_fast(bytes: &[u8]) -> Option<Vec<u8>> {
    let mut encoder = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::fast());
    encoder.write_all(bytes).ok()?;
    encoder.finish().ok()
}

pub fn write_javascript(stream: &mut TcpStream, body: &str) -> Result<(), String> {
    let response = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: application/javascript; charset=utf-8\r\nCache-Control: no-cache\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    );
    stream
        .write_all(response.as_bytes())
        .and_then(|()| stream.write_all(body.as_bytes()))
        .and_then(|()| stream.flush())
        .map_err(|error| format!("cannot write js response: {error}"))
}

pub fn write_response(
    stream: &mut TcpStream,
    status: &str,
    content_type: &str,
    body: &[u8],
    head_only: bool,
) -> Result<(), String> {
    let header = format!(
        "HTTP/1.1 {status}\r\nContent-Type: {content_type}\r\nCache-Control: no-cache\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    );
    stream
        .write_all(header.as_bytes())
        .and_then(|()| {
            if head_only {
                Ok(())
            } else {
                stream.write_all(body)
            }
        })
        .and_then(|()| stream.flush())
        .map_err(|error| format!("cannot write response: {error}"))
}

pub fn write_not_modified(stream: &mut TcpStream, etag: &str) -> Result<(), String> {
    let header = format!(
        "HTTP/1.1 304 Not Modified\r\nETag: {etag}\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n"
    );
    stream
        .write_all(header.as_bytes())
        .and_then(|()| stream.flush())
        .map_err(|error| format!("cannot write 304 response: {error}"))
}

/// Serve an emitted file with validation and optional memoized gzip encoding.
pub fn write_file(
    stream: &mut TcpStream,
    file: &Path,
    sent_etag: Option<&str>,
    gzip: bool,
) -> Result<(), String> {
    write_file_with_method(stream, file, sent_etag, gzip, false)
}

pub fn write_file_with_method(
    stream: &mut TcpStream,
    file: &Path,
    sent_etag: Option<&str>,
    gzip: bool,
    head_only: bool,
) -> Result<(), String> {
    let metadata = std::fs::metadata(file).ok();
    let etag = metadata.as_ref().and_then(file_validator);
    if let (Some(etag), Some(sent)) = (&etag, sent_etag)
        && etag_matches(sent, etag)
    {
        return write_not_modified(stream, etag);
    }
    let bytes =
        std::fs::read(file).map_err(|error| format!("cannot read {}: {error}", file.display()))?;
    let content_type = crate::static_files::content_type(file);
    let validator = etag
        .map(|etag| format!("ETag: {etag}\r\n"))
        .unwrap_or_default();
    let encoded = if gzip && compressible(content_type) {
        metadata
            .as_ref()
            .and_then(|metadata| {
                let mtime = metadata
                    .modified()
                    .ok()?
                    .duration_since(std::time::UNIX_EPOCH)
                    .ok()?
                    .as_nanos();
                GzipCache::get(file, metadata.len(), mtime, &bytes)
            })
            .filter(|compressed| compressed.len() < bytes.len())
    } else {
        None
    };
    let (body, encoding) = match &encoded {
        Some(compressed) => (
            compressed.as_slice(),
            "Content-Encoding: gzip\r\nVary: Accept-Encoding\r\n",
        ),
        None => (bytes.as_slice(), ""),
    };
    let header = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: {content_type}\r\n{validator}{encoding}Cache-Control: no-cache\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    );
    stream
        .write_all(header.as_bytes())
        .and_then(|()| {
            if head_only {
                Ok(())
            } else {
                stream.write_all(body)
            }
        })
        .and_then(|()| stream.flush())
        .map_err(|error| format!("cannot write file response: {error}"))
}

pub fn etag_matches(if_none_match: &str, etag: &str) -> bool {
    let strip = |tag: &str| tag.trim().trim_start_matches("W/").trim().to_string();
    let wanted = strip(etag);
    if_none_match
        .split(',')
        .any(|candidate| candidate.trim() == "*" || strip(candidate) == wanted)
}

pub fn if_none_match(headers: &[(String, String)]) -> Option<&str> {
    headers
        .iter()
        .find(|(name, _)| name.eq_ignore_ascii_case("if-none-match"))
        .map(|(_, value)| value.as_str())
}

pub fn file_validator(metadata: &std::fs::Metadata) -> Option<String> {
    let nanos = metadata
        .modified()
        .ok()?
        .duration_since(std::time::UNIX_EPOCH)
        .ok()?
        .as_nanos();
    Some(format!("W/\"{:x}-{nanos:x}\"", metadata.len()))
}

pub fn accepts_gzip(headers: &[(String, String)]) -> bool {
    headers
        .iter()
        .find(|(name, _)| name.eq_ignore_ascii_case("accept-encoding"))
        .is_some_and(|(_, value)| {
            value.split(',').any(|token| {
                let token = token.trim();
                token
                    .split(';')
                    .next()
                    .unwrap_or("")
                    .trim()
                    .eq_ignore_ascii_case("gzip")
                    && !token
                        .split(';')
                        .any(|part| part.trim().replace(' ', "") == "q=0")
            })
        })
}

pub fn compressible(content_type: &str) -> bool {
    let base = content_type.split(';').next().unwrap_or("").trim();
    base.starts_with("text/")
        || matches!(
            base,
            "application/javascript"
                | "application/json"
                | "application/manifest+json"
                | "image/svg+xml"
        )
}

pub fn html_escape(text: &str) -> String {
    text.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

/// Inserts browser bootstrap markup at the top of `head`, before `body` when
/// there is no head, or at the end as a final fallback.
pub fn inject_html(body: &[u8], snippet: &str) -> Vec<u8> {
    let Ok(html) = std::str::from_utf8(body) else {
        return body.to_vec();
    };
    if let Some(position) = find_case_insensitive(html, "<head>") {
        let at = position + "<head>".len();
        return [
            html[..at].as_bytes(),
            snippet.as_bytes(),
            html[at..].as_bytes(),
        ]
        .concat();
    }
    if let Some(position) = rfind_case_insensitive(html, "</body>") {
        return [
            html[..position].as_bytes(),
            snippet.as_bytes(),
            html[position..].as_bytes(),
        ]
        .concat();
    }
    [body, snippet.as_bytes()].concat()
}

fn find_case_insensitive(haystack: &str, needle: &str) -> Option<usize> {
    haystack
        .to_ascii_lowercase()
        .find(&needle.to_ascii_lowercase())
}

fn rfind_case_insensitive(haystack: &str, needle: &str) -> Option<usize> {
    haystack
        .to_ascii_lowercase()
        .rfind(&needle.to_ascii_lowercase())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weak_etags_and_lists_match() {
        assert!(etag_matches("W/\"aa\", \"bb\"", "\"bb\""));
        assert!(!etag_matches("\"aa\"", "\"bb\""));
    }

    #[test]
    fn validators_and_gzip_follow_file_versions_and_content_policy() {
        let directory = tempfile::tempdir().unwrap();
        let file = directory.path().join("chunk.js");
        let body = "export const value = 1;\n".repeat(400);
        std::fs::write(&file, &body).unwrap();
        let metadata = std::fs::metadata(&file).unwrap();
        let first = file_validator(&metadata).unwrap();
        let mtime = metadata
            .modified()
            .unwrap()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let compressed = GzipCache::get(&file, metadata.len(), mtime, body.as_bytes()).unwrap();
        assert!(compressed.len() < body.len());
        assert!(Arc::ptr_eq(
            &compressed,
            &GzipCache::get(&file, metadata.len(), mtime, body.as_bytes()).unwrap()
        ));
        std::fs::write(&file, format!("{body}// edit\n")).unwrap();
        assert_ne!(
            first,
            file_validator(&std::fs::metadata(&file).unwrap()).unwrap()
        );

        let header = |value: &str| vec![("Accept-Encoding".to_string(), value.to_string())];
        assert!(accepts_gzip(&header("br, gzip;q=0.8")));
        assert!(!accepts_gzip(&header("gzip;q=0")));
        assert!(compressible("application/javascript; charset=utf-8"));
        assert!(!compressible("image/png"));
    }

    #[test]
    fn injects_case_insensitive_head() {
        assert_eq!(
            inject_html(b"<HTML><HEAD></HEAD></HTML>", "x"),
            b"<HTML><HEAD>x</HEAD></HTML>"
        );
    }
}
