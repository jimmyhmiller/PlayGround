//! Safe request-path normalization and route matching.
//!
//! This module is the heart of the "can't accidentally expose something"
//! guarantee. Two properties matter:
//!
//! 1. **Normalization rejects traversal.** A request path is percent-decoded
//!    and walked component-by-component; any `..` (or absolute/prefix trickery)
//!    is rejected outright with 400, *before* matching. So no encoded
//!    `..%2f..%2f` can ever make one route's request reach another route's
//!    target.
//!
//! 2. **Prefix matching respects `/` boundaries.** `/admin` matches `/admin`
//!    and `/admin/x` but **not** `/administrator`. Longest matching prefix
//!    wins, so a more specific public subpath can sit under a private parent
//!    (e.g. public `/admin/docs` under private `/admin`) — intentional and
//!    safe, because the more specific route is chosen.
//!
//! Matching is case-sensitive (per the URI spec); `/Admin` and `/admin` are
//! different routes.

use percent_encoding::percent_decode_str;

use crate::config::Route;

/// Outcome of resolving a request path against the route table.
#[derive(Debug)]
pub enum Match<'a> {
    /// A route matched. Carries the route, its index in the (sorted) table, and
    /// the remaining path *after* the matched prefix (always starts with `/` or
    /// is empty), which a static handler uses as the file path and a proxy
    /// passes through.
    Route {
        route: &'a Route,
        rest: String,
        index: usize,
    },
    /// The (normalized) path matched no route.
    NoRoute,
    /// The path was malformed or attempted traversal — reject with 400.
    BadPath,
}

/// The access-control decision for a request — the safety-critical outcome,
/// separated from any I/O so it can be tested exhaustively.
#[derive(Debug, PartialEq, Eq)]
pub enum Decision {
    /// Reject: malformed/traversal path (400).
    BadPath,
    /// Reject: no route matched (the configured unmatched status).
    NoRoute,
    /// Reject: matched a private route but auth failed (401).
    Unauthorized,
    /// Allow: serve this route. `private` records whether auth was required
    /// (and therefore passed) — used by tests to assert no private route is
    /// ever allowed without auth.
    Allow { route_index: usize, private: bool },
}

/// A precompiled, validated route table. Routes are sorted longest-path-first
/// so the first match in iteration order is also the most specific.
pub struct Router {
    routes: Vec<Route>,
}

impl Router {
    pub fn new(mut routes: Vec<Route>) -> Self {
        // Longest path first → first match is the most specific (longest prefix).
        routes.sort_by(|a, b| b.path.len().cmp(&a.path.len()));
        Router { routes }
    }

    pub fn routes(&self) -> &[Route] {
        &self.routes
    }

    /// Normalize a raw request path (the URI path, possibly percent-encoded)
    /// into a safe absolute path, or `None` if it is malformed / attempts
    /// traversal. The returned path:
    /// - starts with `/`,
    /// - contains no `.` or `..` components,
    /// - has its percent-encoding decoded exactly once.
    pub fn normalize(raw: &str) -> Option<String> {
        // Decode percent-encoding once. Invalid UTF-8 in the decoded bytes is a
        // malformed path for our purposes.
        let decoded = percent_decode_str(raw).decode_utf8().ok()?;

        // A decoded path containing a NUL or a raw control char is suspect.
        if decoded.bytes().any(|b| b == 0) {
            return None;
        }
        // After decoding, a path MUST still be absolute.
        if !decoded.starts_with('/') {
            return None;
        }

        // Walk components manually rather than via std::path (which has
        // platform-specific semantics, e.g. backslash on Windows). We only
        // accept `/`-separated segments and reject `.`/`..` explicitly.
        let mut out = String::with_capacity(decoded.len());
        for seg in decoded.split('/') {
            if seg.is_empty() || seg == "." {
                // Collapse empty segments (from `//` or leading `/`) and `.`.
                continue;
            }
            if seg == ".." {
                // Any parent reference is a traversal attempt — reject the
                // whole request. We do not try to "resolve" it.
                return None;
            }
            // A decoded segment that still contains a `/`-like trick can't
            // happen here (we already split on `/`), and a backslash is just a
            // normal filename char on the target FS; ServeDir guards the rest.
            out.push('/');
            out.push_str(seg);
        }
        if out.is_empty() {
            out.push('/');
        }
        Some(out)
    }

    /// Resolve a raw request path to a route. Performs normalization first;
    /// returns [`Match::BadPath`] for malformed/traversal input.
    pub fn resolve(&self, raw_path: &str) -> Match<'_> {
        let Some(path) = Self::normalize(raw_path) else {
            return Match::BadPath;
        };
        for (i, route) in self.routes.iter().enumerate() {
            if let Some(rest) = prefix_match(&path, &route.path) {
                return Match::Route { route, rest, index: i };
            }
        }
        Match::NoRoute
    }

    /// The complete access-control decision for a request. `auth_ok` is whether
    /// a valid token was presented. This is the ONE place the default-deny
    /// property is enforced: a private route is allowed only when `auth_ok`.
    pub fn decide(&self, raw_path: &str, auth_ok: bool) -> Decision {
        match self.resolve(raw_path) {
            Match::BadPath => Decision::BadPath,
            Match::NoRoute => Decision::NoRoute,
            Match::Route { route, index, .. } => {
                if route.public || auth_ok {
                    Decision::Allow {
                        route_index: index,
                        private: !route.public,
                    }
                } else {
                    Decision::Unauthorized
                }
            }
        }
    }
}

/// Match `path` against route prefix `prefix` on `/` boundaries.
///
/// Returns the remainder after the prefix (always `""` or starting with `/`)
/// when it matches, else `None`. The root prefix `/` matches everything.
///
/// Examples (prefix `/admin`):
/// - `/admin`        -> Some("")
/// - `/admin/`       -> Some("/")
/// - `/admin/x/y`    -> Some("/x/y")
/// - `/administrator`-> None   (no `/` boundary after the prefix)
fn prefix_match(path: &str, prefix: &str) -> Option<String> {
    if prefix == "/" {
        // Root matches everything; the rest is the whole path.
        return Some(path.to_string());
    }
    let rest = path.strip_prefix(prefix)?;
    if rest.is_empty() {
        Some(String::new())
    } else if rest.starts_with('/') {
        Some(rest.to_string())
    } else {
        // e.g. prefix `/admin`, path `/administrator` -> "istrator", reject.
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_basic() {
        assert_eq!(Router::normalize("/a/b").as_deref(), Some("/a/b"));
        assert_eq!(Router::normalize("/").as_deref(), Some("/"));
        assert_eq!(Router::normalize("/a//b/").as_deref(), Some("/a/b"));
        assert_eq!(Router::normalize("/a/./b").as_deref(), Some("/a/b"));
    }

    #[test]
    fn normalize_rejects_traversal() {
        assert_eq!(Router::normalize("/a/../b"), None);
        assert_eq!(Router::normalize("/../etc/passwd"), None);
        // percent-encoded `..%2f..%2f`
        assert_eq!(Router::normalize("/static/..%2f..%2fsecret"), None);
        // double-encoded still decodes to `..` only once; `%252e` -> `%2e`, not `.`
        // so it is NOT treated as traversal (it becomes a literal segment).
        assert!(Router::normalize("/static/%252e%252e").is_some());
    }

    #[test]
    fn normalize_rejects_non_absolute_and_nul() {
        assert_eq!(Router::normalize("relative/path"), None);
        assert_eq!(Router::normalize("/a%00b"), None);
    }

    #[test]
    fn prefix_boundary() {
        assert_eq!(prefix_match("/admin", "/admin").as_deref(), Some(""));
        assert_eq!(prefix_match("/admin/", "/admin").as_deref(), Some("/"));
        assert_eq!(prefix_match("/admin/x/y", "/admin").as_deref(), Some("/x/y"));
        assert_eq!(prefix_match("/administrator", "/admin"), None);
        assert_eq!(prefix_match("/anything", "/").as_deref(), Some("/anything"));
    }

    #[test]
    fn longest_prefix_wins() {
        let r = Router::new(vec![
            route("/admin", false),
            route("/admin/docs", true),
            route("/", true),
        ]);
        // /admin/docs/x -> the docs route (public), not /admin (private)
        match r.resolve("/admin/docs/x") {
            Match::Route { route, .. } => {
                assert_eq!(route.path, "/admin/docs");
                assert!(route.public);
            }
            other => panic!("expected docs route, got {other:?}"),
        }
        // /admin/secret -> /admin (private)
        match r.resolve("/admin/secret") {
            Match::Route { route, .. } => {
                assert_eq!(route.path, "/admin");
                assert!(!route.public);
            }
            other => panic!("expected admin route, got {other:?}"),
        }
        // unrelated path -> root (public)
        match r.resolve("/whatever") {
            Match::Route { route, .. } => assert_eq!(route.path, "/"),
            other => panic!("expected root route, got {other:?}"),
        }
    }

    #[test]
    fn resolve_bad_path() {
        let r = Router::new(vec![route("/static", true)]);
        assert!(matches!(r.resolve("/static/..%2f..%2fx"), Match::BadPath));
    }

    fn route(path: &str, public: bool) -> Route {
        Route {
            path: path.to_string(),
            static_dir: Some(std::path::PathBuf::from("/tmp")),
            proxy: None,
            public,
        }
    }
}
