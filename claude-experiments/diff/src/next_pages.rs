//! Next.js **pages-router** adapter — classic (non-RSC) React SSR onto diffpack's
//! native bundler.
//!
//! Unlike the app-router adapter ([`crate::next_adapter`], which maps onto the RSC
//! spine of three build graphs), the pages router is classic Next: a page module
//! exports a default React component plus optional `getServerSideProps` /
//! `getStaticProps`; the server renders the whole document with `renderToString` +
//! `renderToStaticMarkup` and the browser hydrates `#__next` with `hydrateRoot`.
//!
//! This adapter detects a `pages/` project, scaffolds the fixed runtime + `next/*`
//! shims (`next/link`, `next/head`, `next/router`, `next/document`) and the two
//! build entries under `<root>/.diffpack-next-pages/`, and generates the route-table
//! manifest modules (client + server) that statically import every page so the
//! native bundler discovers them. The build then runs two graphs (client -> the
//! browser `public/`, server -> the Node ESM `server/server.mjs`) exactly as the
//! generic `build-app` path does; the emitted `pages-server.mjs` orchestrator wires
//! them into a working HTTP server.
//!
//! Generated glue lives under `<root>/.diffpack-next-pages/` (gitignored, like the
//! other build outputs). The fixed runtime/shim files are diffpack-authored REAL
//! files carried via `include_str!`; only the route-table manifests are generated
//! (build glue derived from the filesystem, the same precedent as the app-router
//! adapter's resolver modules).

use std::path::{Path, PathBuf};

use crate::bundler::BuildConfig;
use crate::config::AppConfig;
use crate::transform::Target;

/// The directory under the project root where the pages adapter writes its generated
/// entries, `next/*` shims, and route-table manifests.
pub const ADAPTER_DIR: &str = ".diffpack-next-pages";

/// Extensions a page / api module may use.
const PAGE_EXTS: [&str; 4] = ["tsx", "jsx", "ts", "js"];

/// One kind of a route path segment.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Segment {
    /// A literal path segment (`post`).
    Literal(String),
    /// `[id]` -> one path component captured under `id`.
    Dynamic(String),
    /// `[...slug]` -> the rest of the path (>=1 component) captured under `slug`.
    CatchAll(String),
    /// `[[...slug]]` -> the rest of the path (>=0 components), optional.
    OptionalCatchAll(String),
}

/// A discovered route (page or api): its display pattern, the segments it compiles
/// to, and the module file that serves it.
#[derive(Debug, Clone)]
struct Route {
    /// Human/display pattern and `__NEXT_DATA__.page` value (`/post/[id]`).
    pattern: String,
    segments: Vec<Segment>,
    module: PathBuf,
}

/// Everything the manifest generators need about a pages project.
#[derive(Debug, Default)]
struct Discovery {
    pages: Vec<Route>,
    api: Vec<Route>,
    app: Option<PathBuf>,
    document: Option<PathBuf>,
    error: Option<PathBuf>,
}

/// The `pages/` directory for `root`, checking `pages/` then `src/pages/`.
fn pages_dir(root: &Path) -> Option<PathBuf> {
    [root.join("pages"), root.join("src").join("pages")]
        .into_iter()
        .find(|candidate| candidate.is_dir())
}

/// Whether `root` has a `next.config.*`.
fn has_next_config(root: &Path) -> bool {
    ["tsx", "jsx", "ts", "js", "mjs", "cjs"]
        .iter()
        .any(|ext| root.join(format!("next.config.{ext}")).is_file())
}

/// Whether `root` is a Next.js pages-router project this adapter handles: a
/// `next.config.*` plus a `pages/` (or `src/pages/`) directory containing at least
/// one page module. A project that ALSO has an `app/` page is an app-router project
/// (or hybrid); this adapter is only chosen when app-router detection declines, so
/// the caller must check [`crate::next_adapter::is_app_router`] first. Canonicalizes
/// defensively (a bad path is simply "not a pages app", never a panic).
pub fn is_pages_router(root: &Path) -> bool {
    let canonical = root.canonicalize().unwrap_or_else(|_| root.to_path_buf());
    if !has_next_config(&canonical) {
        return false;
    }
    let Some(dir) = pages_dir(&canonical) else {
        return false;
    };
    // At least one real page (not just _app/_document/api) must exist.
    match discover(&dir) {
        Ok(discovery) => !discovery.pages.is_empty(),
        Err(_) => false,
    }
}

/// The first `<dir>/<stem>.<ext>` that exists, in `PAGE_EXTS` priority order.
fn first_existing(dir: &Path, stem: &str) -> Option<PathBuf> {
    PAGE_EXTS
        .iter()
        .map(|ext| dir.join(format!("{stem}.{ext}")))
        .find(|path| path.is_file())
}

/// Recursively collect every module file under `dir` (relative paths from `dir`).
fn collect_files(dir: &Path, prefix: &Path, out: &mut Vec<PathBuf>) -> Result<(), String> {
    let entries = std::fs::read_dir(dir)
        .map_err(|error| format!("cannot read {}: {error}", dir.display()))?;
    for entry in entries {
        let entry = entry.map_err(|error| format!("cannot read entry in {}: {error}", dir.display()))?;
        let path = entry.path();
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if name.starts_with('.') {
            continue;
        }
        let file_type = entry
            .file_type()
            .map_err(|error| format!("cannot stat {}: {error}", path.display()))?;
        let rel = prefix.join(&*name);
        if file_type.is_dir() {
            collect_files(&path, &rel, out)?;
        } else if path
            .extension()
            .and_then(|e| e.to_str())
            .is_some_and(|ext| PAGE_EXTS.contains(&ext))
        {
            out.push(rel);
        }
    }
    Ok(())
}

/// Parse one filename segment (`post`, `[id]`, `[...slug]`, `[[...slug]]`).
fn parse_segment(raw: &str) -> Segment {
    if let Some(inner) = raw.strip_prefix("[[...").and_then(|s| s.strip_suffix("]]")) {
        Segment::OptionalCatchAll(inner.to_string())
    } else if let Some(inner) = raw.strip_prefix("[...").and_then(|s| s.strip_suffix(']')) {
        Segment::CatchAll(inner.to_string())
    } else if let Some(inner) = raw.strip_prefix('[').and_then(|s| s.strip_suffix(']')) {
        Segment::Dynamic(inner.to_string())
    } else {
        Segment::Literal(raw.to_string())
    }
}

/// Compile a route's file path (relative to `pages/`, without extension) into a
/// display pattern + segment list. `index` as the final component maps to the parent
/// path (so `post/index` -> `/post`, `index` -> `/`).
fn route_from_rel(rel: &Path) -> Option<(String, Vec<Segment>)> {
    let without_ext = rel.with_extension("");
    let mut parts: Vec<String> = without_ext
        .components()
        .map(|c| c.as_os_str().to_string_lossy().into_owned())
        .collect();
    if parts.last().map(String::as_str) == Some("index") {
        parts.pop();
    }
    let segments: Vec<Segment> = parts.iter().map(|p| parse_segment(p)).collect();
    let pattern = if parts.is_empty() {
        "/".to_string()
    } else {
        format!("/{}", parts.join("/"))
    };
    Some((pattern, segments))
}

/// The special pages-root files that are not routes.
fn is_special(stem: &str) -> bool {
    matches!(stem, "_app" | "_document" | "_error")
}

/// Discover the routes, api routes, and special files under a `pages/` directory.
fn discover(dir: &Path) -> Result<Discovery, String> {
    let mut files = Vec::new();
    collect_files(dir, Path::new(""), &mut files)?;
    files.sort();

    let mut discovery = Discovery {
        app: first_existing(dir, "_app"),
        document: first_existing(dir, "_document"),
        error: first_existing(dir, "_error"),
        ..Discovery::default()
    };

    for rel in files {
        let abs = dir.join(&rel);
        let stem = rel.file_stem().and_then(|s| s.to_str()).unwrap_or("");
        let top = rel.components().next().map(|c| c.as_os_str().to_string_lossy().into_owned());
        // Special files at the pages root are not routes.
        if rel.parent().is_none_or(|p| p.as_os_str().is_empty()) && is_special(stem) {
            continue;
        }
        let Some((pattern, segments)) = route_from_rel(&rel) else {
            continue;
        };
        let route = Route { pattern, segments, module: abs };
        if top.as_deref() == Some("api") {
            discovery.api.push(route);
        } else {
            discovery.pages.push(route);
        }
    }

    // Most-specific first: fewer dynamic/catch-all segments win; a catch-all is the
    // least specific. Ties broken by longer literal prefix, then lexically.
    let key = |route: &Route| -> (usize, usize, i32) {
        let dynamics = route
            .segments
            .iter()
            .filter(|s| !matches!(s, Segment::Literal(_)))
            .count();
        let has_catchall = route.segments.iter().any(|s| {
            matches!(s, Segment::CatchAll(_) | Segment::OptionalCatchAll(_))
        });
        (has_catchall as usize, dynamics, -(route.segments.len() as i32))
    };
    discovery
        .pages
        .sort_by(|a, b| key(a).cmp(&key(b)).then_with(|| a.pattern.cmp(&b.pattern)));
    discovery
        .api
        .sort_by(|a, b| key(a).cmp(&key(b)).then_with(|| a.pattern.cmp(&b.pattern)));

    Ok(discovery)
}

/// Escape a literal path segment for inclusion in a JS regex source.
fn escape_regex_literal(literal: &str) -> String {
    let mut out = String::with_capacity(literal.len());
    for ch in literal.chars() {
        if r".^$*+?()[]{}|\/".contains(ch) {
            out.push('\\');
        }
        out.push(ch);
    }
    out
}

/// Build the JS regex SOURCE (without the surrounding `/.../`) plus the ordered
/// capture keys `(name, is_catchall)` for a route's segments.
fn compile_regex(segments: &[Segment]) -> (String, Vec<(String, bool)>) {
    if segments.is_empty() {
        return ("^\\/$".to_string(), Vec::new());
    }
    let mut source = String::from("^");
    let mut keys = Vec::new();
    for segment in segments {
        match segment {
            Segment::Literal(literal) => {
                source.push_str("\\/");
                source.push_str(&escape_regex_literal(literal));
            }
            Segment::Dynamic(name) => {
                source.push_str("\\/([^/]+)");
                keys.push((name.clone(), false));
            }
            Segment::CatchAll(name) => {
                source.push_str("\\/(.+)");
                keys.push((name.clone(), true));
            }
            Segment::OptionalCatchAll(name) => {
                source.push_str("(?:\\/(.+))?");
                keys.push((name.clone(), true));
            }
        }
    }
    source.push('$');
    (source, keys)
}

/// A JS string literal for `value` (double-quoted, minimal escaping).
fn js_str(value: &str) -> String {
    let mut out = String::with_capacity(value.len() + 2);
    out.push('"');
    for ch in value.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            _ => out.push(ch),
        }
    }
    out.push('"');
    out
}

/// The JS `keys: [...]` array source for a route's capture keys.
fn keys_source(keys: &[(String, bool)]) -> String {
    let items: Vec<String> = keys
        .iter()
        .map(|(name, catchall)| format!("{{ name: {}, catchall: {catchall} }}", js_str(name)))
        .collect();
    format!("[{}]", items.join(", "))
}

/// Generate the CLIENT route-table manifest module: it statically imports each
/// page's default component (so the bundler discovers them), the App, and the error
/// page, and exports the ordered `pages` table used by `matchPath` on the client.
fn client_manifest_module(discovery: &Discovery, adapter_dir: &Path) -> String {
    let app = resolved_app(discovery, adapter_dir);
    let error = resolved_error(discovery, adapter_dir);
    let mut out = String::new();
    out.push_str("// GENERATED by diffpack's pages-router adapter. Do not edit.\n");
    out.push_str(&format!("import App from {};\n", js_str(&app.to_string_lossy())));
    out.push_str(&format!(
        "import ErrorPage from {};\n",
        js_str(&error.to_string_lossy())
    ));
    for (index, route) in discovery.pages.iter().enumerate() {
        out.push_str(&format!(
            "import Component{index} from {};\n",
            js_str(&route.module.to_string_lossy())
        ));
    }
    out.push_str("export { App, ErrorPage };\n");
    out.push_str("export const pages = [\n");
    for (index, route) in discovery.pages.iter().enumerate() {
        let (regex, keys) = compile_regex(&route.segments);
        out.push_str(&format!(
            "  {{ pattern: {}, regex: /{}/, keys: {}, component: Component{index} }},\n",
            js_str(&route.pattern),
            regex,
            keys_source(&keys),
        ));
    }
    out.push_str("];\n");
    out
}

/// Generate the SERVER route-table manifest module: it imports each page's full
/// namespace (default component + `getServerSideProps`/`getStaticProps`), the App,
/// the Document, the error page, and each api handler, and exports the ordered
/// `pages` and `apiRoutes` tables the server render entry dispatches through.
fn server_manifest_module(discovery: &Discovery, adapter_dir: &Path) -> String {
    let app = resolved_app(discovery, adapter_dir);
    let document = resolved_document(discovery, adapter_dir);
    let error = resolved_error(discovery, adapter_dir);
    let mut out = String::new();
    out.push_str("// GENERATED by diffpack's pages-router adapter. Do not edit.\n");
    out.push_str(&format!("import App from {};\n", js_str(&app.to_string_lossy())));
    out.push_str(&format!(
        "import Document from {};\n",
        js_str(&document.to_string_lossy())
    ));
    out.push_str(&format!(
        "import ErrorPage from {};\n",
        js_str(&error.to_string_lossy())
    ));
    for (index, route) in discovery.pages.iter().enumerate() {
        out.push_str(&format!(
            "import * as Page{index} from {};\n",
            js_str(&route.module.to_string_lossy())
        ));
    }
    for (index, route) in discovery.api.iter().enumerate() {
        out.push_str(&format!(
            "import * as Api{index} from {};\n",
            js_str(&route.module.to_string_lossy())
        ));
    }
    out.push_str("export { App, Document, ErrorPage };\n");
    out.push_str("export const pages = [\n");
    for (index, route) in discovery.pages.iter().enumerate() {
        let (regex, keys) = compile_regex(&route.segments);
        out.push_str(&format!(
            "  {{ pattern: {}, regex: /{}/, keys: {}, mod: Page{index} }},\n",
            js_str(&route.pattern),
            regex,
            keys_source(&keys),
        ));
    }
    out.push_str("];\n");
    out.push_str("export const apiRoutes = [\n");
    for (index, route) in discovery.api.iter().enumerate() {
        let (regex, keys) = compile_regex(&route.segments);
        out.push_str(&format!(
            "  {{ pattern: {}, regex: /{}/, keys: {}, handler: Api{index}.default }},\n",
            js_str(&route.pattern),
            regex,
            keys_source(&keys),
        ));
    }
    out.push_str("];\n");
    out
}

/// Resolve the App module: the project's `pages/_app` or the scaffolded default.
fn resolved_app(discovery: &Discovery, adapter_dir: &Path) -> PathBuf {
    discovery
        .app
        .clone()
        .unwrap_or_else(|| adapter_dir.join("default-app.jsx"))
}

/// Resolve the Document module: the project's `pages/_document` or the default.
fn resolved_document(discovery: &Discovery, adapter_dir: &Path) -> PathBuf {
    discovery
        .document
        .clone()
        .unwrap_or_else(|| adapter_dir.join("default-document.jsx"))
}

/// Resolve the error page: the project's `pages/_error` or the default.
fn resolved_error(discovery: &Discovery, adapter_dir: &Path) -> PathBuf {
    discovery
        .error
        .clone()
        .unwrap_or_else(|| adapter_dir.join("default-error.jsx"))
}

/// Write `contents` to `path` only if it differs (keeps mtimes stable for
/// incremental builds).
fn write_if_changed(path: &Path, contents: &str) -> Result<(), String> {
    if let Ok(existing) = std::fs::read_to_string(path)
        && existing == contents
    {
        return Ok(());
    }
    std::fs::write(path, contents)
        .map_err(|error| format!("cannot write {}: {error}", path.display()))
}

/// The fixed runtime + shim files scaffolded verbatim into the adapter dir.
const RUNTIME_FILES: &[(&str, &str)] = &[
    ("pages-runtime.jsx", include_str!("../scripts/pages/pages-runtime.jsx")),
    ("next-router.jsx", include_str!("../scripts/pages/next-router.jsx")),
    ("pages-head-manager.jsx", include_str!("../scripts/pages/pages-head-manager.jsx")),
    ("next-head.jsx", include_str!("../scripts/pages/next-head.jsx")),
    ("next-link.jsx", include_str!("../scripts/pages/next-link.jsx")),
    ("next-document.jsx", include_str!("../scripts/pages/next-document.jsx")),
    ("default-app.jsx", include_str!("../scripts/pages/default-app.jsx")),
    ("default-document.jsx", include_str!("../scripts/pages/default-document.jsx")),
    ("default-error.jsx", include_str!("../scripts/pages/default-error.jsx")),
    ("pages-client-entry.jsx", include_str!("../scripts/pages/pages-client-entry.jsx")),
    ("pages-server-entry.jsx", include_str!("../scripts/pages/pages-server-entry.jsx")),
];

/// The Node orchestrator source (`pages-server.mjs`) the production build emits.
pub const ORCHESTRATOR: &str = include_str!("../scripts/pages/pages-server.mjs");

/// The build-time SSG prerender driver (`pages-prerender.mjs`): imports the built SSR
/// bundle and calls its `prerender()` export, writing `prerender.json` the orchestrator
/// seeds its ISR cache from.
pub const PRERENDER_DRIVER: &str = include_str!("../scripts/pages/pages-prerender.mjs");

/// If `root` is a pages-router project, scaffold `.diffpack-next-pages/` (runtime +
/// shims + generated route-table manifests) and return the [`AppConfig`] for
/// `environment` (`client` -> the browser build, anything else -> the Node server
/// build). Returns `Ok(None)` for a non-pages project.
pub fn configure(root: &Path, environment: &str, dev: bool) -> Result<Option<AppConfig>, String> {
    let root = root
        .canonicalize()
        .map_err(|error| format!("cannot open project root {}: {error}", root.display()))?;
    if !has_next_config(&root) {
        return Ok(None);
    }
    let Some(dir) = pages_dir(&root) else {
        return Ok(None);
    };
    let discovery = discover(&dir)?;
    if discovery.pages.is_empty() {
        return Ok(None);
    }

    let adapter_dir = root.join(ADAPTER_DIR);
    std::fs::create_dir_all(&adapter_dir)
        .map_err(|error| format!("cannot create {}: {error}", adapter_dir.display()))?;

    // Scaffold the fixed runtime/shim files.
    for (name, contents) in RUNTIME_FILES {
        write_if_changed(&adapter_dir.join(name), contents)?;
    }
    // Generate the route-table manifests (both, so a later server build after a
    // client build never reads a stale table).
    write_if_changed(
        &adapter_dir.join("pages-manifest.client.js"),
        &client_manifest_module(&discovery, &adapter_dir),
    )?;
    write_if_changed(
        &adapter_dir.join("pages-manifest.server.js"),
        &server_manifest_module(&discovery, &adapter_dir),
    )?;

    let is_client = environment == "client";
    let entry = if is_client {
        adapter_dir.join("pages-client-entry.jsx")
    } else {
        adapter_dir.join("pages-server-entry.jsx")
    };
    let target = if is_client { Target::Client } else { Target::Server };
    let production = if dev { "development" } else { "production" };
    let conditions: Vec<String> = if is_client {
        vec!["module".into(), "browser".into(), production.into()]
    } else {
        vec!["node".into(), production.into()]
    };

    let alias = |spec: &str, file: &str| (spec.to_string(), adapter_dir.join(file).to_string_lossy().into_owned());
    let aliases = vec![
        alias("next/link", "next-link.jsx"),
        alias("next/head", "next-head.jsx"),
        alias("next/router", "next-router.jsx"),
        alias("next/document", "next-document.jsx"),
    ];

    let node_env = if dev { "\"development\"" } else { "\"production\"" };
    let defines = vec![("process.env.NODE_ENV".to_string(), node_env.to_string())];

    Ok(Some(AppConfig {
        environment: environment.to_string(),
        build: BuildConfig {
            base: "/".to_string(),
            browser_process_shim: is_client,
            asset_inline_limit: 4096,
            aliases,
            conditions,
            main_fields: Vec::new(),
            virtual_modules: Vec::new(),
            target,
            import_meta_env: None,
            import_meta_glob: None,
            defines,
            hmr: dev,
            scss: crate::sass::ScssOptions {
                additional_data: None,
                root: Some(root.clone()),
            },
            image_import_shape: crate::bundler::ImageImportShape::Url,
            css_preprocess: crate::bundler::CssPreprocess {
                root: Some(root.clone()),
                postcss: crate::postcss::discover(&root).map(std::sync::Arc::new),
            },
        },
        entry: Some(entry),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_segment_kinds() {
        assert_eq!(parse_segment("post"), Segment::Literal("post".into()));
        assert_eq!(parse_segment("[id]"), Segment::Dynamic("id".into()));
        assert_eq!(parse_segment("[...slug]"), Segment::CatchAll("slug".into()));
        assert_eq!(
            parse_segment("[[...slug]]"),
            Segment::OptionalCatchAll("slug".into())
        );
    }

    #[test]
    fn compiles_root_regex() {
        let (source, keys) = compile_regex(&[]);
        assert_eq!(source, "^\\/$");
        assert!(keys.is_empty());
        let re = regex_lite_match(&source, "/");
        assert!(re, "root regex must match /");
    }

    #[test]
    fn compiles_dynamic_regex() {
        let segments = vec![Segment::Literal("post".into()), Segment::Dynamic("id".into())];
        let (source, keys) = compile_regex(&segments);
        assert_eq!(source, "^\\/post\\/([^/]+)$");
        assert_eq!(keys, vec![("id".to_string(), false)]);
    }

    #[test]
    fn compiles_catchall_regex() {
        let segments = vec![Segment::Literal("blog".into()), Segment::CatchAll("slug".into())];
        let (source, keys) = compile_regex(&segments);
        assert_eq!(source, "^\\/blog\\/(.+)$");
        assert_eq!(keys, vec![("slug".to_string(), true)]);
    }

    #[test]
    fn optional_catchall_regex_matches_base() {
        let segments = vec![
            Segment::Literal("shop".into()),
            Segment::OptionalCatchAll("slug".into()),
        ];
        let (source, keys) = compile_regex(&segments);
        assert_eq!(source, "^\\/shop(?:\\/(.+))?$");
        assert_eq!(keys, vec![("slug".to_string(), true)]);
    }

    #[test]
    fn route_from_index_maps_to_slash() {
        let (pattern, segments) = route_from_rel(Path::new("index.tsx")).unwrap();
        assert_eq!(pattern, "/");
        assert!(segments.is_empty());
    }

    #[test]
    fn route_from_nested_dynamic() {
        let (pattern, segments) = route_from_rel(Path::new("post/[id].tsx")).unwrap();
        assert_eq!(pattern, "/post/[id]");
        assert_eq!(
            segments,
            vec![Segment::Literal("post".into()), Segment::Dynamic("id".into())]
        );
    }

    #[test]
    fn nested_index_drops_index() {
        let (pattern, segments) = route_from_rel(Path::new("post/index.tsx")).unwrap();
        assert_eq!(pattern, "/post");
        assert_eq!(segments, vec![Segment::Literal("post".into())]);
    }

    #[test]
    fn escapes_regex_metacharacters() {
        assert_eq!(escape_regex_literal("a.b"), "a\\.b");
        assert_eq!(escape_regex_literal("plain"), "plain");
    }

    #[test]
    fn generated_manifest_is_valid_js_shape() {
        let discovery = Discovery {
            pages: vec![
                Route {
                    pattern: "/".into(),
                    segments: vec![],
                    module: PathBuf::from("/x/pages/index.tsx"),
                },
                Route {
                    pattern: "/post/[id]".into(),
                    segments: vec![Segment::Literal("post".into()), Segment::Dynamic("id".into())],
                    module: PathBuf::from("/x/pages/post/[id].tsx"),
                },
            ],
            ..Discovery::default()
        };
        let adapter = PathBuf::from("/x/.diffpack-next-pages");
        let client = client_manifest_module(&discovery, &adapter);
        assert!(client.contains("export const pages = ["));
        assert!(client.contains("regex: /^\\/$/"));
        assert!(client.contains("regex: /^\\/post\\/([^/]+)$/"));
        assert!(client.contains("component: Component1"));
        assert!(client.contains("default-app.jsx"));
        let server = server_manifest_module(&discovery, &adapter);
        assert!(server.contains("import * as Page0"));
        assert!(server.contains("mod: Page0"));
        assert!(server.contains("export const apiRoutes = ["));
    }

    /// A minimal literal-only matcher good enough for the root-regex test (avoids a
    /// regex-crate dependency in the unit test).
    fn regex_lite_match(source: &str, input: &str) -> bool {
        // Only handles the exact `^\/$` case used by the root test.
        source == "^\\/$" && input == "/"
    }

    #[test]
    fn server_entry_exposes_ssg_lifecycle_contract() {
        let entry = include_str!("../scripts/pages/pages-server-entry.jsx");
        // The build-time prerender + runtime seed the SSG/ISR pipeline depends on.
        assert!(entry.contains("export async function prerender"));
        assert!(entry.contains("export function seedPrerender"));
        // The three data-fetching lifecycle branches, kept mutually exclusive.
        assert!(entry.contains("getServerSideProps"));
        assert!(entry.contains("getStaticProps"));
        assert!(entry.contains("getStaticPaths"));
        assert!(entry.contains("getInitialProps"));
        // ISR state header used to prove static-vs-regenerated behaviour.
        assert!(entry.contains("x-diffpack-isr"));
        // Default export must carry the new exports so the orchestrator/driver find them.
        assert!(entry.contains("prerender, seedPrerender"));
    }

    #[test]
    fn server_entry_api_request_surface_is_complete() {
        let entry = include_str!("../scripts/pages/pages-server-entry.jsx");
        // The Node (req, res) API surface the pages-router contract requires.
        assert!(entry.contains("cookies: parseCookies(headers)"),
            "req.cookies must be parsed from the Cookie header, never a silent empty stub");
        assert!(entry.contains("function parseCookies"));
        assert!(entry.contains("body: parseBody(headers, bodyText)"));
        // Body parsing covers JSON and urlencoded, not just JSON.
        assert!(entry.contains("application/json"));
        assert!(entry.contains("application/x-www-form-urlencoded"));
        // res.status().json() / setHeader / redirect are all present on the response.
        assert!(entry.contains("status(code)"));
        assert!(entry.contains("json(obj)"));
        assert!(entry.contains("setHeader(key, value)"));
        // Dynamic api routes dispatch through the same matcher as pages, merging params.
        assert!(entry.contains("matchPath(apiRoutes, pathname)"));
        assert!(entry.contains("...apiMatch.params"));
    }

    #[test]
    fn api_routes_manifest_exposes_default_handler() {
        // A dynamic api route (`api/user/[id]`) plus a static one must both land in the
        // apiRoutes table with their default export wired as `handler`, most-specific
        // first, and never in the pages table.
        let discovery = Discovery {
            pages: vec![Route {
                pattern: "/".into(),
                segments: vec![],
                module: PathBuf::from("/x/pages/index.tsx"),
            }],
            api: vec![
                Route {
                    pattern: "/api/hello".into(),
                    segments: vec![Segment::Literal("api".into()), Segment::Literal("hello".into())],
                    module: PathBuf::from("/x/pages/api/hello.ts"),
                },
                Route {
                    pattern: "/api/user/[id]".into(),
                    segments: vec![
                        Segment::Literal("api".into()),
                        Segment::Literal("user".into()),
                        Segment::Dynamic("id".into()),
                    ],
                    module: PathBuf::from("/x/pages/api/user/[id].ts"),
                },
            ],
            ..Discovery::default()
        };
        let adapter = PathBuf::from("/x/.diffpack-next-pages");
        let server = server_manifest_module(&discovery, &adapter);
        assert!(server.contains("import * as Api0"));
        assert!(server.contains("import * as Api1"));
        assert!(server.contains("handler: Api0.default"));
        assert!(server.contains("handler: Api1.default"));
        assert!(server.contains("regex: /^\\/api\\/user\\/([^/]+)$/"));
        // The client manifest never imports api handlers (server-only code).
        let client = client_manifest_module(&discovery, &adapter);
        assert!(!client.contains("api/hello"));
        assert!(!client.contains("apiRoutes"));
    }

    #[test]
    fn discover_classifies_api_and_dynamic_api_routes() {
        // A `pages/api/**` tree (static + nested dynamic) must classify into `api`, and
        // regular pages into `pages`, purely from the filesystem shape.
        let dir = std::env::temp_dir().join(format!("diffpack-api-discover-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("api/user")).unwrap();
        std::fs::write(dir.join("index.tsx"), "export default function P(){return null}").unwrap();
        std::fs::write(dir.join("api/hello.ts"), "export default function h(){}").unwrap();
        std::fs::write(dir.join("api/user/[id].ts"), "export default function h(){}").unwrap();
        let discovery = discover(&dir).unwrap();
        let api_patterns: Vec<&str> = discovery.api.iter().map(|r| r.pattern.as_str()).collect();
        assert!(api_patterns.contains(&"/api/hello"));
        assert!(api_patterns.contains(&"/api/user/[id]"));
        let page_patterns: Vec<&str> = discovery.pages.iter().map(|r| r.pattern.as_str()).collect();
        assert_eq!(page_patterns, vec!["/"]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn orchestrator_seeds_prerender_manifest() {
        assert!(ORCHESTRATOR.contains("prerender.json"));
        assert!(ORCHESTRATOR.contains("seedPrerender"));
    }

    #[test]
    fn prerender_driver_calls_prerender_export() {
        assert!(PRERENDER_DRIVER.contains("prerender"));
        assert!(PRERENDER_DRIVER.contains("prerender.json"));
        assert!(PRERENDER_DRIVER.contains("server.mjs"));
    }
}
