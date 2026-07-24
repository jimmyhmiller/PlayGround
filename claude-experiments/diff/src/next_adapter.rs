//! Next.js **app-router** adapter — the mapping layer from Next's file conventions
//! onto diffpack's existing RSC spine (Slices A–E).
//!
//! diffpack's RSC machinery (three build graphs: `Target::Client` /
//! `Target::ReactServer` / `Target::Server`, two manifests, the `__webpack_*` seam,
//! the server-action resolver) is framework-neutral: it drives off directive-marked
//! modules (`"use client"` / `"use server"`) and a canonical `module_reference_id`,
//! not off any TanStack- or Next-specific entry. What a real `create-next-app`
//! project lacks is the *entries*: Next has no `src/client.tsx` / `src/server.ts` /
//! `src/rsc-entry.tsx`; its "entry" is the **app-router file convention**
//! (`app/layout.tsx` wrapping `app/page.tsx`). Next's own runtime composes
//! `<RootLayout><Page/></RootLayout>` and renders that tree — this module does the
//! same, natively, by generating the three RSC entries (+ a minimal `next/*` shim
//! layer) that the proven spine then builds unchanged.
//!
//! This is the index route (`app/page.tsx`) composed under the root layout
//! (`app/layout.tsx`), all `"use client"` islands discovered under `app/`, with
//! `next/font` (build-time macro rewrite + hoisted CSS), the static Metadata API
//! (`<title>`/`<meta>`), global + CSS-Module stylesheets (linked from the
//! react-server graph's compiled CSS), and client-side soft navigation (`next/link`
//! intercepts clicks; the client Router fetches the target route's flight over
//! `?__rsc=1` and diff-renders it), dynamic segments (`[param]`/`[...catchAll]`/
//! `[[...optional]]`, matched per-request with `params` delivered to the page +
//! `useParams`), the `loading` (Suspense) and `error` (a generated client
//! ErrorBoundary) conventions, and a real HTTP 404 (`app/not-found.tsx`, no index
//! fall-through) all handled. A per-request context (an `AsyncLocalStorage`
//! established by the render entry from the request the orchestrator sends on stdin)
//! backs faithful `next/headers` (`await cookies()`/`headers()`), server-side
//! `redirect()`/`notFound()` (digest → real HTTP 307/404 over the fd-3 control
//! channel), and the `next/navigation` client hooks (`useParams`/`usePathname`/
//! `useSearchParams` read React contexts fed identically on SSR + client), and a
//! faithful `next/image` (a `getImgProps` port: raster srcs under `public/` get a
//! responsive `srcset`/`sizes` pointing at build-emitted static variant files — no
//! image-optimization server — SVG/`unoptimized`/`data:` srcs render the raw src
//! with no `srcset`, and `priority` hoists a `<link rel="preload" as="image">`).
//! A `diffpack dev` topology (Slice K) also drives this adapter: [`is_app_router`]
//! dispatches a Next app to `dev_server::next`, which builds these same three graphs
//! via [`configure_dev`] (development React + HMR) and serves them with
//! state-preserving Fast Refresh for `"use client"` islands and a correct reload for
//! Server-Component edits. Parallel (`@slot`) / intercepting (`(.)`) routes remain the
//! documented remaining gaps (see `docs/RSC_NEXT_GAP.md`).
//!
//! Generated glue lives under `<root>/.diffpack-next/` (gitignored, like the other
//! build outputs). Generating entry/shim source as Rust strings follows the exact
//! precedent of [`crate::rsc::generate_action_resolver_module`] and
//! [`crate::server_fn::generate_resolver_module`]: diffpack-authored build glue, not
//! guest source hidden in a string.

use std::path::{Path, PathBuf};

use crate::bundler::BuildConfig;
use crate::config::AppConfig;
use crate::rsc::{detect_directive, RscDirective};
use crate::transform::Target;

/// The directory under the project root where the adapter writes its generated
/// entries and `next/*` shims.
pub const ADAPTER_DIR: &str = ".diffpack-next";

/// Module-file extensions the adapter recognizes for app-router convention files.
const MODULE_EXTS: [&str; 4] = ["tsx", "jsx", "ts", "js"];

/// Detects whether `root` is a Next.js app-router project this adapter handles: an
/// `app/` directory containing a `page.{tsx,jsx,ts,js}`, plus a `next.config.*`
/// (so we never mistake a non-Next `app/` folder for one). Returns the resolved
/// `app/page` path when so.
fn detect_app_router(root: &Path) -> Option<PathBuf> {
    let has_next_config = MODULE_EXTS
        .iter()
        .chain(["mjs"].iter())
        .any(|ext| root.join(format!("next.config.{ext}")).is_file());
    if !has_next_config {
        return None;
    }
    let app = root.join("app");
    if !app.is_dir() {
        return None;
    }
    first_existing(&app, "page")
}

/// Whether `root` is a Next.js app-router project this adapter handles. Public
/// wrapper over [`detect_app_router`] so the dev server can dispatch a Next app to
/// its own topology (three RSC graphs + the next orchestrator) before the
/// TanStack/SPA detection. Canonicalizes defensively (a bad path is simply "not a
/// Next app", never a panic).
pub fn is_app_router(root: &Path) -> bool {
    let canonical = root.canonicalize().unwrap_or_else(|_| root.to_path_buf());
    detect_app_router(&canonical).is_some()
}

/// The first `<dir>/<stem>.<ext>` that exists, in `MODULE_EXTS` priority order.
fn first_existing(dir: &Path, stem: &str) -> Option<PathBuf> {
    MODULE_EXTS
        .iter()
        .map(|ext| dir.join(format!("{stem}.{ext}")))
        .find(|path| path.is_file())
}

/// Walks `app/` (skipping the adapter's own output) for `"use client"` modules,
/// returning their canonical paths sorted for deterministic codegen.
/// A route's static metadata (the subset the adapter renders): `title` and
/// `description` from an `export const metadata = { ... }`.
#[derive(Debug, Default, Clone)]
struct RouteMetadata {
    title: Option<String>,
    description: Option<String>,
}

/// Reads `export const metadata = { title, description }` string literals from a
/// module (the Metadata API subset), for the adapter to render `<title>`/`<meta>`
/// into the document head. `generateMetadata()` and non-string values are not read
/// (a documented gap), never silently guessed.
fn scan_metadata(path: &Path, source: &str) -> RouteMetadata {
    use oxc_ast::ast::{Declaration, Expression, ObjectPropertyKind, PropertyKey, Statement};
    if !source.contains("metadata") {
        return RouteMetadata::default();
    }
    let allocator = oxc_allocator::Allocator::default();
    let source_type = oxc_span::SourceType::from_path(path).unwrap_or_default().with_module(true);
    let parsed = oxc_parser::Parser::new(&allocator, source, source_type).parse();
    let mut meta = RouteMetadata::default();
    for statement in &parsed.program.body {
        let Statement::ExportNamedDeclaration(export) = statement else { continue };
        let Some(Declaration::VariableDeclaration(var)) = &export.declaration else { continue };
        for decl in &var.declarations {
            if decl.id.get_binding_identifier().map(|i| i.name.as_str()) != Some("metadata") {
                continue;
            }
            let Some(Expression::ObjectExpression(object)) = &decl.init else { continue };
            for property in &object.properties {
                let ObjectPropertyKind::ObjectProperty(prop) = property else { continue };
                let key = match &prop.key {
                    PropertyKey::StaticIdentifier(ident) => ident.name.as_str(),
                    PropertyKey::StringLiteral(lit) => lit.value.as_str(),
                    _ => continue,
                };
                if let Expression::StringLiteral(value) = &prop.value {
                    match key {
                        "title" => meta.title = Some(value.value.to_string()),
                        "description" => meta.description = Some(value.value.to_string()),
                        _ => {}
                    }
                }
            }
        }
    }
    meta
}

/// The route metadata for the app, page overriding layout (as in Next).
fn app_metadata(page: &Path, layout: Option<&Path>) -> RouteMetadata {
    let mut meta = RouteMetadata::default();
    if let Some(layout) = layout
        && let Ok(source) = std::fs::read_to_string(layout) {
            meta = scan_metadata(layout, &source);
        }
    if let Ok(source) = std::fs::read_to_string(page) {
        let page_meta = scan_metadata(page, &source);
        if page_meta.title.is_some() {
            meta.title = page_meta.title;
        }
        if page_meta.description.is_some() {
            meta.description = page_meta.description;
        }
    }
    meta
}

/// How a route is served: prerendered to a static file at build time, or rendered
/// per-request. Mirrors Next's `○ Static` / `● SSG` / `ƒ Dynamic` legend.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RouteKind {
    /// No request reads, no dynamic segment — one `<route>.html` prerendered.
    Static,
    /// `export const dynamic = "force-static" | "error"` — prerendered (request reads,
    /// if any, hard-error at build; documented MVP boundary).
    ForceStatic,
    /// A dynamic segment with `generateStaticParams` — one `.html` per enumerated combo.
    Ssg,
    /// Rendered per request (force-dynamic, request-state reads, or a dynamic segment
    /// with no `generateStaticParams`). NOT prerendered.
    Dynamic,
}

impl RouteKind {
    /// The lowercase string the render entry's ROUTES table + the prerender plan use.
    fn as_str(self) -> &'static str {
        match self {
            RouteKind::Static => "static",
            RouteKind::ForceStatic => "forceStatic",
            RouteKind::Ssg => "ssg",
            RouteKind::Dynamic => "dynamic",
        }
    }
    /// Whether the prerenderer emits an on-disk `.html`/`.rsc` for this route.
    fn is_prerendered(self) -> bool {
        matches!(self, RouteKind::Static | RouteKind::ForceStatic | RouteKind::Ssg)
    }
}

/// The route-config exports a page source declares, read by [`classify_route`].
#[derive(Debug, Clone, Default)]
struct RouteConfig {
    has_generate_static_params: bool,
    /// `export const dynamic = "..."` value (`force-dynamic|force-static|error|auto`).
    dynamic_config: Option<String>,
    /// `export const dynamicParams = false` sets this false; default true.
    dynamic_params: bool,
    /// `export const revalidate = <v>` raw value (a WARN unless "0").
    revalidate: Option<String>,
    /// Reads request-scoped state at the top level (cookies/headers/draftMode or the
    /// `searchParams` prop) — forces per-request rendering.
    reads_request_state: bool,
}

/// Extracts the raw RHS of `export const <name> = <rhs>;` (first occurrence), trimmed
/// and with surrounding quotes stripped. A substring scan (same shape as
/// `scan_metadata`/`scan_next_font`); returns None if the export is absent.
fn extract_export_const(source: &str, name: &str) -> Option<String> {
    for prefix in [format!("export const {name}"), format!("export let {name}"), format!("export var {name}")] {
        if let Some(pos) = source.find(&prefix) {
            let after = &source[pos + prefix.len()..];
            // Require the next non-space char to be `=` (not another identifier char).
            let after = after.trim_start();
            let Some(rest) = after.strip_prefix('=') else { continue };
            let rest = rest.trim_start();
            // RHS runs to the first `;` or newline.
            let end = rest.find([';', '\n']).unwrap_or(rest.len());
            let raw = rest[..end].trim();
            let unquoted = raw
                .strip_prefix('"').and_then(|s| s.strip_suffix('"'))
                .or_else(|| raw.strip_prefix('\'').and_then(|s| s.strip_suffix('\'')))
                .or_else(|| raw.strip_prefix('`').and_then(|s| s.strip_suffix('`')))
                .unwrap_or(raw);
            return Some(unquoted.to_string());
        }
    }
    None
}

/// Removes `//` line comments and `/* */` block comments from JS/TS source while
/// PRESERVING string literals (single/double/backtick) — so a `next/headers` import
/// specifier still counts, but a `searchParams` mention in a comment does not
/// false-trigger. A small char state machine (not string-content-aware beyond quote
/// tracking; template-literal `${}` is treated as string content, which is fine for
/// the token scan). Keeps the code's structure; only comment spans become spaces.
fn strip_comments(source: &str) -> String {
    let bytes = source.as_bytes();
    let mut out = String::with_capacity(source.len());
    let mut i = 0;
    #[derive(PartialEq)]
    enum State {
        Code,
        Str(u8),
        Line,
        Block,
    }
    let mut state = State::Code;
    while i < bytes.len() {
        let c = bytes[i];
        let next = bytes.get(i + 1).copied();
        match state {
            State::Code => {
                if c == b'/' && next == Some(b'/') {
                    state = State::Line;
                    i += 2;
                } else if c == b'/' && next == Some(b'*') {
                    state = State::Block;
                    i += 2;
                } else if c == b'"' || c == b'\'' || c == b'`' {
                    state = State::Str(c);
                    out.push(c as char);
                    i += 1;
                } else {
                    // Push the full UTF-8 char (source may be non-ASCII).
                    let ch = source[i..].chars().next().unwrap();
                    out.push(ch);
                    i += ch.len_utf8();
                }
            }
            State::Str(quote) => {
                if c == b'\\' {
                    // Escape: copy this and the next byte verbatim.
                    out.push(c as char);
                    if let Some(n) = next {
                        out.push(n as char);
                        i += 2;
                    } else {
                        i += 1;
                    }
                } else {
                    let ch = source[i..].chars().next().unwrap();
                    out.push(ch);
                    i += ch.len_utf8();
                    if c == quote {
                        state = State::Code;
                    }
                }
            }
            State::Line => {
                if c == b'\n' {
                    out.push('\n');
                    state = State::Code;
                }
                i += 1;
            }
            State::Block => {
                if c == b'*' && next == Some(b'/') {
                    state = State::Code;
                    i += 2;
                } else {
                    if c == b'\n' {
                        out.push('\n');
                    }
                    i += 1;
                }
            }
        }
    }
    out
}

/// Scans a page module source for the app-router route-config exports + request-state
/// reads that determine whether the route can be statically prerendered. Substring
/// based (like `scan_metadata`), over comment-stripped source; conservative — an
/// ambiguous scan errs toward Dynamic (via `classify_route`), never toward a
/// wrongly-static route.
fn scan_route_config(raw_source: &str) -> RouteConfig {
    let source = &strip_comments(raw_source);
    let has_generate_static_params = source.contains("generateStaticParams");
    let dynamic_config = extract_export_const(source, "dynamic");
    let dynamic_params = match extract_export_const(source, "dynamicParams") {
        Some(v) => v.trim() != "false",
        None => true,
    };
    let revalidate = extract_export_const(source, "revalidate");
    // Request-scoped reads: next/headers (cookies/headers/draftMode) or the searchParams
    // prop. Any of these makes the route request-dependent → per-request render.
    let reads_request_state = source.contains("next/headers")
        || source.contains("searchParams");
    RouteConfig {
        has_generate_static_params,
        dynamic_config,
        dynamic_params,
        revalidate,
        reads_request_state,
    }
}

/// Classifies a route from whether its pattern has a dynamic segment + its config,
/// reproducing Next's static/dynamic decision for the fixture exactly. Precedence:
/// force-dynamic (or `revalidate:0`) > force-static/error > request-state reads >
/// dynamic-segment (gsp ? Ssg : Dynamic) > Static.
fn classify_route(has_dynamic_segment: bool, cfg: &RouteConfig) -> RouteKind {
    if cfg.dynamic_config.as_deref() == Some("force-dynamic")
        || cfg.revalidate.as_deref() == Some("0")
    {
        return RouteKind::Dynamic;
    }
    if matches!(cfg.dynamic_config.as_deref(), Some("force-static") | Some("error")) {
        return RouteKind::ForceStatic;
    }
    if cfg.reads_request_state {
        return RouteKind::Dynamic;
    }
    if has_dynamic_segment {
        return if cfg.has_generate_static_params {
            RouteKind::Ssg
        } else {
            RouteKind::Dynamic
        };
    }
    RouteKind::Static
}

/// A human-readable reason a route is served per-request (for the prerender manifest).
fn dynamic_reason(has_dynamic_segment: bool, cfg: &RouteConfig) -> String {
    if cfg.dynamic_config.as_deref() == Some("force-dynamic") {
        "force-dynamic".to_string()
    } else if cfg.revalidate.as_deref() == Some("0") {
        "revalidate: 0 (force-dynamic)".to_string()
    } else if cfg.reads_request_state {
        // A request-state read forces dynamic rendering even when the route exports
        // generateStaticParams (Next's own precedence: reading cookies/headers/searchParams
        // at the top opts the WHOLE route into dynamic). Name that precedence explicitly so
        // the manifest never falsely claims the route simply lacks generateStaticParams.
        if cfg.has_generate_static_params {
            "reads request state (cookies/headers/searchParams); dynamic despite generateStaticParams".to_string()
        } else {
            "reads request state (cookies/headers/searchParams); no generateStaticParams".to_string()
        }
    } else if has_dynamic_segment {
        "dynamic segment with no generateStaticParams".to_string()
    } else {
        "dynamic".to_string()
    }
}

/// A parsed URL path segment from an app-router directory name.
#[derive(Debug, Clone, PartialEq)]
enum Seg {
    Static(String),
    Dynamic(String),
    CatchAll(String),
    OptionalCatchAll(String),
}

/// The classification of one directory-name component of a route path.
enum SegParse {
    /// A URL segment (static or dynamic).
    Seg(Seg),
    /// A route group `(name)` — contributes no URL segment, but its layout applies.
    Group,
    /// A parallel `@slot` or intercepting `(.)`/`(..)`/`(...)` route — the whole
    /// route is skipped (documented gap; never silently mis-served).
    Skip,
}

/// Parses one app-router directory-name component into a URL segment classification.
/// `[x]`→Dynamic, `[...x]`→CatchAll, `[[...x]]`→OptionalCatchAll, `(group)`→omitted,
/// `@slot` / `(.)`-intercepts → the route is skipped.
fn parse_segment(comp: &str) -> SegParse {
    if comp.starts_with('@') {
        return SegParse::Skip; // parallel-route slot
    }
    if comp.starts_with("(.") {
        return SegParse::Skip; // intercepting route: (.)/(..)/(...)
    }
    if let Some(inner) = comp.strip_prefix("[[...").and_then(|s| s.strip_suffix("]]")) {
        return SegParse::Seg(Seg::OptionalCatchAll(inner.to_string()));
    }
    if let Some(inner) = comp.strip_prefix("[...").and_then(|s| s.strip_suffix(']')) {
        return SegParse::Seg(Seg::CatchAll(inner.to_string()));
    }
    if let Some(inner) = comp.strip_prefix('[').and_then(|s| s.strip_suffix(']')) {
        return SegParse::Seg(Seg::Dynamic(inner.to_string()));
    }
    if comp.starts_with('(') && comp.ends_with(')') {
        return SegParse::Group;
    }
    SegParse::Seg(Seg::Static(comp.to_string()))
}

/// Reconstructs a display URL path (`/blog/[slug]`) from a parsed segment list.
fn segments_display(segments: &[Seg]) -> String {
    if segments.is_empty() {
        return "/".to_string();
    }
    let mut out = String::new();
    for seg in segments {
        out.push('/');
        match seg {
            Seg::Static(v) => out.push_str(v),
            Seg::Dynamic(v) => {
                out.push('[');
                out.push_str(v);
                out.push(']');
            }
            Seg::CatchAll(v) => {
                out.push_str("[...");
                out.push_str(v);
                out.push(']');
            }
            Seg::OptionalCatchAll(v) => {
                out.push_str("[[...");
                out.push_str(v);
                out.push_str("]]");
            }
        }
    }
    out
}

/// Serializes a parsed segment list to the `{ k, v }` JS array the render entry's
/// `matchSegments` consumes (e.g. `[{ k: "static", v: "blog" }, { k: "dynamic", v: "slug" }]`).
fn segments_js(segments: &[Seg]) -> String {
    let mut out = String::from("[");
    for (i, seg) in segments.iter().enumerate() {
        if i > 0 {
            out.push_str(", ");
        }
        let (k, v) = match seg {
            Seg::Static(v) => ("static", v),
            Seg::Dynamic(v) => ("dynamic", v),
            Seg::CatchAll(v) => ("catchall", v),
            Seg::OptionalCatchAll(v) => ("optcatchall", v),
        };
        out.push_str(&format!("{{ k: {}, v: {} }}", js_str(k), js_str(v)));
    }
    out.push(']');
    out
}

/// One level of the app-router nesting chain (one directory from `app/` down to the
/// page's own directory): the layout/loading/error convention files present there.
/// `not-found` is collected only at the app root (see [`Discovered::app_not_found`]) —
/// per-route `not-found` boundaries need the request context (a later slice).
#[derive(Debug, Clone, Default)]
struct Level {
    layout: Option<PathBuf>,
    loading: Option<PathBuf>,
    error: Option<PathBuf>,
}

/// One app-router route: its display URL path + parsed segment pattern, the page
/// module, its root→leaf level chain (layouts + loading/error boundaries), and its
/// resolved static metadata (page overriding layout).
struct Route {
    url_path: String,
    segments: Vec<Seg>,
    page: PathBuf,
    levels: Vec<Level>,
    metadata: RouteMetadata,
    /// Static/SSG/Dynamic classification (whether + how it is prerendered).
    kind: RouteKind,
    /// `generateStaticParams` is exported by the page (an Ssg enumeration source).
    has_generate_static_params: bool,
    /// `dynamicParams` (default true): unlisted params 404 when false.
    dynamic_params: bool,
    /// The reason a Dynamic route is served per-request (for the prerender manifest).
    dynamic_reason: String,
    /// A `revalidate` value other than `0`/absent — surfaced as a build WARN (ISR is
    /// out of scope; the route is prerendered once and will not revalidate).
    revalidate_warn: Option<String>,
}

/// Serializes a parsed segment list to a JSON `[{"k","v"}]` array (the plan is
/// consumed by node via `JSON.parse`, so keys are quoted — unlike `segments_js`).
fn segments_json(segments: &[Seg]) -> String {
    let mut out = String::from("[");
    for (i, seg) in segments.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        let (k, v) = match seg {
            Seg::Static(v) => ("static", v),
            Seg::Dynamic(v) => ("dynamic", v),
            Seg::CatchAll(v) => ("catchall", v),
            Seg::OptionalCatchAll(v) => ("optcatchall", v),
        };
        out.push_str(&format!("{{\"k\":{},\"v\":{}}}", js_str(k), js_str(v)));
    }
    out.push(']');
    out
}

/// The on-disk file stem (no extension) a static route's `.html`/`.rsc` are written
/// to, mirroring the URL path: `/` → `index`, `/about` → `about`, `/a/b` → `a/b`.
fn route_file_stem(url_path: &str) -> String {
    let trimmed = url_path.trim_matches('/');
    if trimmed.is_empty() {
        "index".to_string()
    } else {
        trimmed.to_string()
    }
}

/// Re-runs app-router discovery + classification and writes the machine-readable
/// prerender plan (`<out_dir>/static/prerender-plan.json`) the node prerenderer
/// consumes: one entry per route carrying its `kind`, parsed `segments` (for param
/// substitution), the on-disk `file` stem for static routes, and — for Dynamic
/// routes — the `reason` they are skipped (never silently dropped). Returns the route
/// count. Emits a build WARN for any `revalidate` (ISR is out of scope). This does NOT
/// build a graph; it reads page sources only (native Rust).
pub fn write_prerender_plan(project_root: &Path, out_dir: &Path) -> Result<usize, String> {
    let root = project_root
        .canonicalize()
        .map_err(|error| format!("cannot open project root {}: {error}", project_root.display()))?;
    if detect_app_router(&root).is_none() {
        return Err(format!(
            "{} is not a Next.js app-router project (no app/ + next config); \
             `build-app <root> static` only prerenders app-router apps",
            root.display(),
        ));
    }
    let app_dir = root.join("app");
    let layout = first_existing(&app_dir, "layout");
    let layout_abs = layout.as_ref().map(|l| l.canonicalize().unwrap_or_else(|_| l.clone()));
    let disc = discover_routes(&app_dir, layout_abs.as_deref())?;

    let mut entries = String::from("[\n");
    for (i, route) in disc.routes.iter().enumerate() {
        if let Some(v) = &route.revalidate_warn {
            eprintln!(
                "WARN next SSG: route {} has `export const revalidate = {v}` — ISR is out of \
                 scope; it is prerendered ONCE and will not revalidate.",
                route.url_path,
            );
        }
        if i > 0 {
            entries.push_str(",\n");
        }
        let mut fields = format!(
            "  {{ \"path\": {}, \"kind\": {}, \"segments\": {}",
            js_str(&route.url_path),
            js_str(route.kind.as_str()),
            segments_json(&route.segments),
        );
        if route.kind.is_prerendered() {
            match route.kind {
                RouteKind::Ssg => {
                    fields.push_str(&format!(
                        ", \"hasGenerateStaticParams\": {}, \"dynamicParams\": {}",
                        route.has_generate_static_params, route.dynamic_params,
                    ));
                }
                _ => {
                    fields.push_str(&format!(", \"file\": {}", js_str(&route_file_stem(&route.url_path))));
                }
            }
        } else {
            fields.push_str(&format!(", \"reason\": {}", js_str(&route.dynamic_reason)));
        }
        fields.push_str(" }");
        entries.push_str(&fields);
    }
    entries.push_str("\n]\n");

    let static_dir = out_dir.join("static");
    std::fs::create_dir_all(&static_dir)
        .map_err(|error| format!("cannot create {}: {error}", static_dir.display()))?;
    let plan_path = static_dir.join("prerender-plan.json");
    std::fs::write(&plan_path, entries)
        .map_err(|error| format!("cannot write {}: {error}", plan_path.display()))?;
    Ok(disc.routes.len())
}

/// The full app-router discovery result: the matchable routes plus the app-root
/// convention files used for the default document (root layout, root metadata) and
/// the real 404 body (`app/not-found`).
struct Discovered {
    routes: Vec<Route>,
    root_layout: Option<PathBuf>,
    root_metadata: RouteMetadata,
    app_not_found: Option<PathBuf>,
}

fn scan_metadata_file(path: &Path) -> RouteMetadata {
    std::fs::read_to_string(path)
        .map(|source| scan_metadata(path, &source))
        .unwrap_or_default()
}

/// Discovers every app-router route under `app/`, now INCLUDING dynamic segments
/// (`[param]`, `[...catchAll]`, `[[...optional]]`) and the `loading`/`error`
/// boundary conventions along each route's nested-layout chain, plus the app-root
/// `not-found`. Parallel (`@slot`) and intercepting (`(.)`) routes are still skipped
/// (documented gap; never mis-served). Routes are sorted most-specific first so a
/// literal segment beats a dynamic one at match time.
fn discover_routes(app_dir: &Path, root_layout: Option<&Path>) -> Result<Discovered, String> {
    let mut routes = Vec::new();
    discover_routes_dir(app_dir, app_dir, root_layout, &mut Vec::new(), &mut routes)?;
    // Specificity: fewest catch-alls, then fewest dynamics, then MORE segments
    // (longer/more specific), then lexicographic — so `/blog/new` beats `/blog/[slug]`.
    routes.sort_by(|a, b| {
        let count = |r: &Route, f: fn(&Seg) -> bool| r.segments.iter().filter(|s| f(s)).count();
        let ca = count(a, |s| matches!(s, Seg::CatchAll(_) | Seg::OptionalCatchAll(_)));
        let cb = count(b, |s| matches!(s, Seg::CatchAll(_) | Seg::OptionalCatchAll(_)));
        let da = count(a, |s| matches!(s, Seg::Dynamic(_)));
        let db = count(b, |s| matches!(s, Seg::Dynamic(_)));
        ca.cmp(&cb)
            .then(da.cmp(&db))
            .then(b.segments.len().cmp(&a.segments.len()))
            .then(a.url_path.cmp(&b.url_path))
    });
    let root_metadata = root_layout.map(scan_metadata_file).unwrap_or_default();
    Ok(Discovered {
        routes,
        root_layout: root_layout.map(|l| l.to_path_buf()),
        root_metadata,
        app_not_found: first_existing(app_dir, "not-found").map(|p| p.canonicalize().unwrap_or(p)),
    })
}

fn discover_routes_dir(
    app_dir: &Path,
    dir: &Path,
    root_layout: Option<&Path>,
    level_chain: &mut Vec<Level>,
    routes: &mut Vec<Route>,
) -> Result<(), String> {
    let canon = |p: PathBuf| p.canonicalize().unwrap_or(p);
    // This directory's own conventions form a level for it and its descendants.
    level_chain.push(Level {
        layout: first_existing(dir, "layout").map(canon),
        loading: first_existing(dir, "loading").map(canon),
        error: first_existing(dir, "error").map(canon),
    });

    if let Some(page) = first_existing(dir, "page") {
        // URL segments = dir relative to app/, parsed per component.
        let rel = dir.strip_prefix(app_dir).unwrap_or(Path::new(""));
        let mut segments = Vec::new();
        let mut skip = false;
        for comp in rel.components().filter_map(|c| c.as_os_str().to_str()) {
            match parse_segment(comp) {
                SegParse::Seg(seg) => segments.push(seg),
                SegParse::Group => {}
                SegParse::Skip => {
                    skip = true;
                    break;
                }
            }
        }
        if !skip {
            let page_abs = page.canonicalize().unwrap_or_else(|_| page.clone());
            let metadata = app_metadata(&page_abs, root_layout);
            let page_source = std::fs::read_to_string(&page_abs).unwrap_or_default();
            let cfg = scan_route_config(&page_source);
            let has_dynamic_segment = segments.iter().any(|s| {
                matches!(s, Seg::Dynamic(_) | Seg::CatchAll(_) | Seg::OptionalCatchAll(_))
            });
            let kind = classify_route(has_dynamic_segment, &cfg);
            let url_path = segments_display(&segments);
            // Nested-gsp is out of scope: a route with >1 dynamic segment that also
            // exports generateStaticParams needs a BFS param merge we do not implement.
            // Hard-error naming the route rather than emit a wrong enumeration.
            if kind == RouteKind::Ssg {
                let dyn_seg_count = segments.iter().filter(|s| {
                    matches!(s, Seg::Dynamic(_) | Seg::CatchAll(_) | Seg::OptionalCatchAll(_))
                }).count();
                if dyn_seg_count > 1 {
                    return Err(format!(
                        "route {url_path}: nested generateStaticParams BFS merge not implemented \
                         (>1 dynamic segment with generateStaticParams). Mark it Dynamic or \
                         implement the merge.",
                    ));
                }
            }
            let revalidate_warn = match cfg.revalidate.as_deref() {
                Some("0") | None => None,
                Some(v) => Some(v.to_string()),
            };
            let dynamic_reason = dynamic_reason(has_dynamic_segment, &cfg);
            routes.push(Route {
                url_path,
                segments,
                page: page_abs,
                levels: level_chain.clone(),
                metadata,
                kind,
                has_generate_static_params: cfg.has_generate_static_params,
                dynamic_params: cfg.dynamic_params,
                dynamic_reason,
                revalidate_warn,
            });
        }
    }

    // Recurse into child route directories (skip the adapter's own output + dotdirs).
    let read = match std::fs::read_dir(dir) {
        Ok(read) => read,
        Err(_) => {
            level_chain.pop();
            return Ok(());
        }
    };
    let mut children: Vec<PathBuf> = read
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.is_dir())
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .map(|n| !n.starts_with('.') && n != ADAPTER_DIR)
                .unwrap_or(false)
        })
        .collect();
    children.sort();
    for child in children {
        discover_routes_dir(app_dir, &child, root_layout, level_chain, routes)?;
    }

    level_chain.pop();
    Ok(())
}

/// Whether any app module imports a stylesheet (`import "./globals.css"` or a CSS
/// Module `import styles from "./x.module.css"`). When so, the react-server build's
/// compiled+scoped CSS (`server.css`, preserved to `public/rsc.css`) is linked from
/// the document head. The react-server graph is authoritative for CSS-Module class
/// scoping, since Server Components render there — its `styles.x` values match the
/// classes in its emitted CSS.
fn app_has_css(app_dir: &Path) -> bool {
    fn walk(dir: &Path) -> bool {
        let Ok(read) = std::fs::read_dir(dir) else { return false };
        for entry in read.flatten() {
            let path = entry.path();
            if path.is_dir() {
                if walk(&path) {
                    return true;
                }
            } else if is_module_file(&path)
                && let Ok(source) = std::fs::read_to_string(&path) {
                    // A `.css`/`.scss`/`.sass` import specifier anywhere in the module.
                    if source.contains(".css\"") || source.contains(".css'")
                        || source.contains(".scss\"") || source.contains(".scss'")
                        || source.contains(".sass\"") || source.contains(".sass'")
                    {
                        return true;
                    }
                }
        }
        false
    }
    walk(app_dir)
}

/// The served path the react-server build's compiled CSS (`server/server.css`) is
/// preserved to (see `main.rs`), and the href the adapter links from `<head>`.
pub const RSC_CSS_URL: &str = "/rsc.css";

/// Collects every `next/font` usage across the app's module files (deduped), so the
/// adapter can generate one CSS block covering all fonts. Mirrors island scanning.
fn collect_app_fonts(app_dir: &Path) -> Result<Vec<crate::next_font::FontUsage>, String> {
    let mut usages = Vec::new();
    collect_fonts_dir(app_dir, &mut usages)?;
    Ok(usages)
}

fn collect_fonts_dir(dir: &Path, usages: &mut Vec<crate::next_font::FontUsage>) -> Result<(), String> {
    let read = match std::fs::read_dir(dir) {
        Ok(read) => read,
        Err(_) => return Ok(()),
    };
    for entry in read {
        let entry = entry.map_err(|error| format!("cannot read {}: {error}", dir.display()))?;
        let path = entry.path();
        let file_type = entry
            .file_type()
            .map_err(|error| format!("cannot stat {}: {error}", path.display()))?;
        if file_type.is_dir() {
            collect_fonts_dir(&path, usages)?;
        } else if is_module_file(&path)
            && let Ok(source) = std::fs::read_to_string(&path)
                && source.contains("next/font") {
                    for usage in crate::next_font::scan_next_font(&path, &source) {
                        if !usages.contains(&usage) {
                            usages.push(usage);
                        }
                    }
                }
    }
    Ok(())
}

fn scan_client_islands(app_dir: &Path) -> Result<Vec<PathBuf>, String> {
    let mut islands = Vec::new();
    scan_islands_dir(app_dir, &mut islands)?;
    islands.sort();
    islands.dedup();
    Ok(islands)
}

fn scan_islands_dir(dir: &Path, islands: &mut Vec<PathBuf>) -> Result<(), String> {
    let read = match std::fs::read_dir(dir) {
        Ok(read) => read,
        Err(_) => return Ok(()),
    };
    for entry in read {
        let entry = entry.map_err(|error| format!("cannot read {}: {error}", dir.display()))?;
        let path = entry.path();
        let file_type = entry
            .file_type()
            .map_err(|error| format!("cannot stat {}: {error}", path.display()))?;
        if file_type.is_dir() {
            if matches!(
                entry.file_name().to_str(),
                Some("node_modules" | ".diffpack-output" | ".diffpack-next" | ".git" | ".next")
            ) {
                continue;
            }
            scan_islands_dir(&path, islands)?;
        } else if is_module_file(&path) {
            let Ok(source) = std::fs::read_to_string(&path) else {
                continue;
            };
            if !source.contains("use client") {
                continue;
            }
            let canonical = std::fs::canonicalize(&path).unwrap_or(path);
            if detect_directive(&canonical, &source) == Some(RscDirective::Client) {
                islands.push(canonical);
            }
        }
    }
    Ok(())
}

fn is_module_file(path: &Path) -> bool {
    matches!(
        path.extension().and_then(|value| value.to_str()),
        Some("ts" | "tsx" | "js" | "jsx" | "mjs" | "cjs")
    )
}

/// A JS string literal for an absolute path (JSON-quoted; escapes backslashes and
/// quotes so Windows-style paths and odd characters survive).
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

/// If `root` is an app-router project, scaffold `.diffpack-next/` and return the
/// [`AppConfig`] for `environment` (`client` | `react-server` | `ssr`/anything
/// else server-like). Returns `Ok(None)` for a non-Next project so the caller
/// falls back to the TanStack `derive_config` path unchanged. This is the
/// PRODUCTION entry point (`build-app`): byte-identical to before the dev server
/// existed.
pub fn configure(root: &Path, environment: &str) -> Result<Option<AppConfig>, String> {
    configure_inner(root, environment, false)
}

/// The DEV variant of [`configure`] (the `diffpack dev` Next topology, Slice K):
/// same scaffold, but the returned config is switched to development —
/// `build.hmr = true`, `process.env.NODE_ENV` defined as `"development"` (so React's
/// development build, which alone exposes the Fast Refresh renderer hook, is
/// bundled), and the resolve `production` condition swapped for `development`. All
/// three graphs run in development so the react-server/SSR React matches the client
/// React at hydration (no dev/prod hydration split). React 19.2.4 selects dev/prod
/// purely from `NODE_ENV` (its `exports` has no `development` condition), so the
/// condition swap is inert for React itself and only affects packages that publish a
/// `development`/`production` exports map.
pub fn configure_dev(root: &Path, environment: &str) -> Result<Option<AppConfig>, String> {
    configure_inner(root, environment, true)
}

fn configure_inner(root: &Path, environment: &str, dev: bool) -> Result<Option<AppConfig>, String> {
    let root = root
        .canonicalize()
        .map_err(|error| format!("cannot open project root {}: {error}", root.display()))?;
    let Some(page) = detect_app_router(&root) else {
        return Ok(None);
    };
    let app_dir = root.join("app");
    let layout = first_existing(&app_dir, "layout");
    let mut islands = scan_client_islands(&app_dir)?;
    // `next/font` usages across the app: the transform (next_font.rs) rewrites the
    // calls to static objects and drops the import; here we generate the companion
    // CSS (Google @import + the CSS-variable classes) that the render entry injects
    // as a React-hoisted <style> into the document head.
    let font_css = crate::next_font::generate_css(&collect_app_fonts(&app_dir)?);
    // Whether to link the app's compiled stylesheet (globals.css + CSS Modules)
    // into the head. The CSS itself is the react-server build's `server.css`,
    // preserved to `public/rsc.css` by the build (main.rs).
    let has_css = app_has_css(&app_dir);

    let adapter_dir = root.join(ADAPTER_DIR);
    let shims_dir = adapter_dir.join("shims");
    std::fs::create_dir_all(&shims_dir)
        .map_err(|error| format!("cannot create {}: {error}", shims_dir.display()))?;

    // --- app-router route table (every route + its nested layout/boundary chain) --
    let _ = &page; // detection anchor; the full route set comes from discovery.
    let layout_abs = layout.as_ref().map(|l| l.canonicalize().unwrap_or_else(|_| l.clone()));
    let discovered = discover_routes(&app_dir, layout_abs.as_deref())?;

    // The generated client Error Boundary (a `"use client"` class component) wraps
    // each route level that has an `error.tsx`. Like the `next/link` shim it must be
    // BUNDLED + REGISTERED in the client + ssr graphs (scan_islands_dir skips
    // `.diffpack-next/`) so its client reference resolves; in the react-server graph
    // it stays a client reference. Keyed by the SAME canonical path the react-server
    // render imports it from → manifest ids match. Write it first so its path exists.
    let error_boundary = adapter_dir.join("error-boundary.tsx");
    write_if_changed(&error_boundary, error_boundary_module())?;
    let error_boundary_canon = error_boundary
        .canonicalize()
        .unwrap_or_else(|_| error_boundary.clone());
    if !islands.contains(&error_boundary_canon) {
        islands.push(error_boundary_canon.clone());
    }

    // The per-request context (Slice I / spec Slice 3). `request-context.ts` holds
    // the ONE `AsyncLocalStorage` instance the react-server render establishes and
    // `next/headers` (cookies/headers) reads — shared because both are bundled into
    // the single react-server graph (same absolute path from both importers).
    // `hooks-context.ts` holds the React contexts `useParams`/`usePathname`/
    // `useSearchParams` read, fed identically on SSR + client (NOT window globals, so
    // no hydration mismatch). `hooks-context` guards `createContext` so it loads
    // harmlessly in the react-server graph too (where it's imported via `next/navigation`
    // for `redirect`, but createContext is undefined under the react-server condition).
    let request_context = adapter_dir.join("request-context.ts");
    write_if_changed(&request_context, request_context_module())?;
    let request_context_canon = request_context
        .canonicalize()
        .unwrap_or_else(|_| request_context.clone());
    let hooks_context = adapter_dir.join("hooks-context.ts");
    write_if_changed(&hooks_context, hooks_context_module())?;
    let hooks_context_canon = hooks_context
        .canonicalize()
        .unwrap_or_else(|_| hooks_context.clone());

    write_if_changed(&adapter_dir.join("lazy.js"), lazy_module())?;
    write_if_changed(
        &adapter_dir.join("rsc-entry.tsx"),
        &rsc_entry_module(
            &discovered,
            &font_css,
            has_css,
            &error_boundary_canon,
            &request_context_canon,
        ),
    )?;
    // The `next/link` shim is a `"use client"` intercepting component. In the
    // react-server graph it stays a client reference (resolved to real code through
    // the seam); in the client + ssr graphs it must be BUNDLED and REGISTERED like
    // any island so its client reference resolves and it hydrates. Because
    // `scan_islands_dir` skips `.diffpack-next/`, pin it explicitly here (write the
    // file first so its canonical path exists), keyed by the SAME canonical path the
    // react-server render resolves the `next/link` alias to → manifest ids match.
    let link_shim = shims_dir.join("link.tsx");
    write_if_changed(&link_shim, next_link_shim())?;
    let link_canon = link_shim.canonicalize().unwrap_or_else(|_| link_shim.clone());
    if !islands.contains(&link_canon) {
        islands.push(link_canon);
    }
    write_if_changed(
        &adapter_dir.join("server.tsx"),
        &ssr_entry_module(&adapter_dir, &islands, &hooks_context_canon),
    )?;
    write_if_changed(
        &adapter_dir.join("client.tsx"),
        &client_entry_module(&adapter_dir, &islands, &hooks_context_canon),
    )?;
    // next/image (Slice J / gap 4.2): generate the variant manifest the shim reads.
    // Scanning `public/` is deterministic (no build-output dependency), so it runs in
    // every environment and the manifest agrees across the three graphs; the actual
    // variant files are emitted once, from the client build's public-copy step
    // (main.rs `emit_image_variants`), keyed by the same deterministic hash.
    let public_images = scan_public_images(&root)?;
    write_if_changed(
        &adapter_dir.join("image-manifest.ts"),
        &image_manifest_module(&public_images),
    )?;
    write_if_changed(&shims_dir.join("image.tsx"), next_image_shim())?;
    write_if_changed(
        &shims_dir.join("navigation.ts"),
        &next_navigation_shim(&hooks_context_canon),
    )?;
    write_if_changed(
        &shims_dir.join("headers.ts"),
        &next_headers_shim(&request_context_canon),
    )?;

    // --- per-environment config --------------------------------------------------
    let (entry, target, conditions): (PathBuf, Target, Vec<&str>) = match environment {
        "client" => (
            adapter_dir.join("client.tsx"),
            Target::Client,
            vec!["module", "browser", "production"],
        ),
        "react-server" => (
            adapter_dir.join("rsc-entry.tsx"),
            Target::ReactServer,
            vec!["react-server", "node", "production", "wasm", "unwasm"],
        ),
        _ => (
            adapter_dir.join("server.tsx"),
            Target::Server,
            vec!["node", "production", "wasm", "unwasm"],
        ),
    };
    // DEV: swap the `production` resolve condition for `development` (mirrors
    // config::set_web_development_mode). Inert for React 19.2.4 (no `development`
    // condition in its exports — it dispatches on NODE_ENV), meaningful only for
    // packages that publish a development/production exports map.
    let conditions: Vec<String> = conditions
        .into_iter()
        .map(|condition| {
            if dev && condition == "production" {
                "development".to_string()
            } else {
                condition.to_string()
            }
        })
        .collect();

    // `next/*` shims resolved as aliases (specifier -> shim file). Only the subset
    // this adapter faithfully implements; `next/font`, `next/headers` server APIs,
    // etc. are documented gaps and are intentionally NOT silently aliased to a
    // no-op (an app importing an unshimmed `next/*` fails at resolve, naming it).
    let alias = |spec: &str, file: &Path| (spec.to_string(), file.to_string_lossy().into_owned());
    let aliases = vec![
        alias("next/link", &shims_dir.join("link.tsx")),
        alias("next/image", &shims_dir.join("image.tsx")),
        alias("next/navigation", &shims_dir.join("navigation.ts")),
        alias("next/headers", &shims_dir.join("headers.ts")),
    ];

    // React's dev/prod dispatch define. Production bundles the production React
    // (small, no dev warnings); DEV bundles the development React whose renderer
    // exposes the Fast Refresh hook the island HMR path needs.
    let node_env = if dev { "\"development\"" } else { "\"production\"" };
    let defines = vec![(
        "process.env.NODE_ENV".to_string(),
        node_env.to_string(),
    )];

    Ok(Some(AppConfig {
        environment: environment.to_string(),
        build: BuildConfig {
            base: "/".to_string(),
            browser_process_shim: true,
            asset_inline_limit: 4096,
            aliases,
            conditions,
            virtual_modules: Vec::new(),
            target,
            import_meta_env: None,
            import_meta_glob: None,
            defines,
            hmr: dev,
            scss: crate::sass::ScssOptions {
                additional_data: None,
                root: Some(root.to_path_buf()),
            },
        },
        entry: Some(entry),
    }))
}

/// Write `contents` to `path` only if it differs (keeps mtimes stable so an
/// unchanged adapter run does not perturb incremental builds).
fn write_if_changed(path: &Path, contents: &str) -> Result<(), String> {
    if let Ok(existing) = std::fs::read_to_string(path)
        && existing == contents {
            return Ok(());
        }
    std::fs::write(path, contents)
        .map_err(|error| format!("cannot write {}: {error}", path.display()))
}

// --- generated module templates --------------------------------------------------

fn lazy_module() -> &'static str {
    "// Generated by diffpack's next app-router adapter. A trivially code-split\n\
     // module: its dynamic import forces the client/SSR builds onto the registry\n\
     // runtime, so the RSC `__webpack_*` seam has a require-able registry to map.\n\
     export const value = \"diffpack-next-adapter-lazy\";\n"
}

/// The generated client Error Boundary (`error.tsx` convention). A `"use client"`
/// class component: on a thrown error it renders `props.fallback` (the route's
/// `error.tsx`) with `{ error, reset }`, else `props.children`. In the react-server
/// graph the throwing children become a recoverable client-boundary subtree (so the
/// flight render completes); the SSR/browser React catches and renders the fallback.
fn error_boundary_module() -> &'static str {
    r#""use client";
// Generated by diffpack's next app-router adapter — the client Error Boundary that
// implements the app-router `error.tsx` convention. React error boundaries must be
// client components; this wraps each route level that has an `error.tsx`.
import { Component, createElement } from "react";

export default class ErrorBoundary extends Component {
  constructor(props) {
    super(props);
    this.state = { error: null };
  }
  static getDerivedStateFromError(error) {
    return { error };
  }
  render() {
    if (this.state.error) {
      return createElement(this.props.fallback, {
        error: this.state.error,
        reset: () => this.setState({ error: null }),
      });
    }
    return this.props.children;
  }
}
"#
}

/// The per-request context module (`request-context.ts`). Holds the ONE
/// `AsyncLocalStorage` instance the react-server render establishes (`requestAls.run`)
/// and `next/headers` reads (`requestAls.getStore()`). Because rsc-entry and the
/// `next/headers` shim are bundled into the SAME react-server graph, they share this
/// single instance (Next's `workUnitAsyncStorage` analogue). It lands ONLY in the
/// react-server graph — Server Components are the only code that reads the request
/// context, and only they run there.
fn request_context_module() -> &'static str {
    "// Generated by diffpack's next app-router adapter — the per-request AsyncLocalStorage\n\
     // that carries { url, headers, cookieHeader, params } from the HTTP request into\n\
     // async Server Components (next/headers cookies()/headers()). One shared instance\n\
     // across the react-server graph (rsc-entry establishes it; next/headers reads it).\n\
     import { AsyncLocalStorage } from \"node:async_hooks\";\n\
     export const requestAls = new AsyncLocalStorage();\n"
}

/// The client-hooks context module (`hooks-context.ts`). Three React contexts the
/// `next/navigation` client hooks (`useParams`/`usePathname`/`useSearchParams`) read,
/// fed identically by the SSR and client entries — NOT window globals, which don't
/// exist during SSR and would cause a hydration mismatch. `createContext` is guarded:
/// under the `react-server` condition it is undefined, but this module is still pulled
/// into the react-server graph (via `next/navigation` → `redirect`), where the dummy
/// contexts load harmlessly (they are never provided or consumed there).
fn hooks_context_module() -> &'static str {
    "// Generated by diffpack's next app-router adapter — the React contexts the\n\
     // next/navigation client hooks read (useParams/usePathname/useSearchParams). Fed\n\
     // identically on SSR + client (never window globals) so hydration matches. The\n\
     // createContext guard lets this load harmlessly in the react-server graph (where\n\
     // createContext is undefined and the contexts are never provided/consumed).\n\
     import * as React from \"react\";\n\
     const createContext =\n\
     \x20 React.createContext ||\n\
     \x20 function () {\n\
     \x20   return { Provider: function (props) { return props.children; } };\n\
     \x20 };\n\
     export const PathParamsContext = createContext({});\n\
     export const PathnameContext = createContext(\"/\");\n\
     export const SearchParamsContext = createContext(\"\");\n"
}

/// The react-server render/action entry (Target::ReactServer). Builds the app's
/// ROUTE TABLE (every static route + its nested layout chain + metadata), matches a
/// requested pathname, composes `<Layout0>…<LayoutN>[head, <Page/>]` for the matched
/// route, and renders it to a flight stream (`render <pathname>` op), or dispatches a
/// server action (`action` op). The orchestrator spawns this in its own child so its
/// react-server React never mixes with the SSR/browser React.
fn rsc_entry_module(
    disc: &Discovered,
    font_css: &str,
    has_css: bool,
    error_boundary: &Path,
    request_context: &Path,
) -> String {
    // Intern every referenced module (page/layout/loading/error/not-found + the
    // generated error boundary) to a stable `M<i>` default-import binding.
    let mut modules: Vec<String> = Vec::new();
    fn intern(modules: &mut Vec<String>, path: &Path) -> usize {
        let s = path.to_string_lossy().into_owned();
        modules.iter().position(|m| m == &s).unwrap_or_else(|| {
            modules.push(s);
            modules.len() - 1
        })
    }
    fn opt_id(modules: &mut Vec<String>, path: &Option<PathBuf>) -> String {
        match path {
            Some(p) => format!("M{}", intern(modules, p)),
            None => "null".to_string(),
        }
    }

    let error_boundary_id = format!("M{}", intern(&mut modules, error_boundary));

    // Namespace imports for Ssg routes (generateStaticParams is a NAMED export; the
    // default-only `M<i>` binding cannot reach it). Keyed by page path → `NS<i>`.
    let mut namespaces: Vec<String> = Vec::new();
    fn intern_ns(namespaces: &mut Vec<String>, path: &Path) -> usize {
        let s = path.to_string_lossy().into_owned();
        namespaces.iter().position(|m| m == &s).unwrap_or_else(|| {
            namespaces.push(s);
            namespaces.len() - 1
        })
    }
    let mut static_param_entries = String::new();

    let mut route_entries = String::new();
    for route in &disc.routes {
        let page_id = format!("M{}", intern(&mut modules, &route.page));
        let mut levels_js = String::new();
        for level in &route.levels {
            let layout_id = opt_id(&mut modules, &level.layout);
            let loading_id = opt_id(&mut modules, &level.loading);
            let error_id = opt_id(&mut modules, &level.error);
            levels_js.push_str(&format!(
                "{{ layout: {layout_id}, loading: {loading_id}, error: {error_id} }}, "
            ));
        }
        let title = route.metadata.title.as_deref().map(js_str).unwrap_or_else(|| "null".to_string());
        let description = route.metadata.description.as_deref().map(js_str).unwrap_or_else(|| "null".to_string());
        if route.kind == RouteKind::Ssg {
            let ns_id = format!("NS{}", intern_ns(&mut namespaces, &route.page));
            static_param_entries.push_str(&format!(
                "  {}: {ns_id},\n",
                js_str(&route.url_path),
            ));
        }
        route_entries.push_str(&format!(
            "  {{ path: {}, segments: {}, page: {page_id}, levels: [{levels_js}], title: {title}, description: {description}, kind: {}, hasGenerateStaticParams: {}, dynamicParams: {} }},\n",
            js_str(&route.url_path),
            segments_js(&route.segments),
            js_str(route.kind.as_str()),
            route.has_generate_static_params,
            route.dynamic_params,
        ));
    }

    let root_layout_id = opt_id(&mut modules, &disc.root_layout);
    let app_not_found_id = opt_id(&mut modules, &disc.app_not_found);
    let root_title = disc.root_metadata.title.as_deref().map(js_str).unwrap_or_else(|| "null".to_string());
    let root_description = disc.root_metadata.description.as_deref().map(js_str).unwrap_or_else(|| "null".to_string());

    let imports: String = modules
        .iter()
        .enumerate()
        .map(|(i, s)| format!("import M{i} from {};\n", js_str(s)))
        .collect();
    // Namespace imports (Ssg routes) — `generateStaticParams` is a named export.
    let ns_imports: String = namespaces
        .iter()
        .enumerate()
        .map(|(i, s)| format!("import * as NS{i} from {};\n", js_str(s)))
        .collect();

    let request_context_import = format!(
        "import {{ requestAls }} from {};\n",
        js_str(&request_context.to_string_lossy()),
    );

    // Head items (React 19 hoists <link>/<style>/<title>/<meta> into <head>).
    let css_push = if has_css {
        format!(
            "  items.push(createElement(\"link\", {{ rel: \"stylesheet\", href: {}, precedence: \"low\" }}));\n",
            js_str(RSC_CSS_URL)
        )
    } else {
        String::new()
    };
    let (font_const, font_push) = if font_css.trim().is_empty() {
        (String::new(), String::new())
    } else {
        (
            format!("const FONT_CSS = {};\n", js_str(font_css)),
            "  items.push(createElement(\"style\", { href: \"diffpack-next-font\", precedence: \"high\", dangerouslySetInnerHTML: { __html: FONT_CSS } }));\n".to_string(),
        )
    };

    format!(
        r##"// Generated by diffpack's next app-router adapter — the REACT-SERVER render
// entry (Target::ReactServer, bundled under the `react-server` export condition).
// It holds the app's ROUTE TABLE (each route's parsed segment pattern + its root→leaf
// level chain of layouts + loading/error boundaries), MATCHES a requested pathname
// (dynamic `[param]`/`[...catchAll]` segments captured into `params`), composes the
// matched route (boundaries inner→outer, layouts root-last), and renders it to a
// flight stream — or renders the real 404 tree for an unmatched path (status carried
// to the orchestrator over fd 3), or dispatches a server action (`action` op).
import {{ renderToReadableStream }} from "react-server-dom-webpack/server";
import {{ createElement, Fragment, Suspense }} from "react";
import {{ readFileSync, writeSync, statSync }} from "node:fs";
import {{ fileURLToPath }} from "node:url";
import {{ handleServerAction }} from "#diffpack-rsc-action-handler";
{request_context_import}{imports}{ns_imports}
{font_const}const ROUTES = [
{route_entries}];
// Ssg routes (a dynamic segment with generateStaticParams) → their module namespace,
// so the `staticparams` op can enumerate concrete param sets at build time.
const STATIC_PARAM_ROUTES = {{
{static_param_entries}}};
const ROOT_LAYOUT = {root_layout_id};
const APP_NOT_FOUND = {app_not_found_id};
const ERROR_BOUNDARY = {error_boundary_id};
const ROOT_META = {{ title: {root_title}, description: {root_description} }};

// The route's head elements (stylesheet + font + this route's metadata). React 19
// hoists these into <head> from anywhere in the tree.
function headItems(meta) {{
  const items = [];
{css_push}{font_push}  if (meta && meta.title) items.push(createElement("title", null, meta.title));
  if (meta && meta.description) items.push(createElement("meta", {{ name: "description", content: meta.description }}));
  return items;
}}

// Match `pathname` against a route's segment pattern, capturing dynamic params.
// Static matches one part exactly, Dynamic one part, CatchAll the (≥1) tail,
// OptionalCatchAll the (≥0) tail. Returns the params object or null.
function matchSegments(segments, parts) {{
  const params = {{}};
  let i = 0;
  for (let s = 0; s < segments.length; s += 1) {{
    const seg = segments[s];
    if (seg.k === "static") {{
      if (parts[i] !== seg.v) return null;
      i += 1;
    }} else if (seg.k === "dynamic") {{
      if (i >= parts.length) return null;
      params[seg.v] = decodeURIComponent(parts[i]);
      i += 1;
    }} else if (seg.k === "catchall") {{
      if (i >= parts.length) return null;
      params[seg.v] = parts.slice(i).map(decodeURIComponent);
      i = parts.length;
    }} else if (seg.k === "optcatchall") {{
      params[seg.v] = parts.slice(i).map(decodeURIComponent);
      i = parts.length;
    }} else {{
      return null;
    }}
  }}
  if (i !== parts.length) return null;
  return params;
}}

// Match a pathname to the most-specific route (ROUTES is pre-sorted). Returns
// `{{ route, params }}` or null (a real 404 — no index fallback).
function matchRoute(pathname) {{
  const parts = pathname.split("/").filter(Boolean);
  for (const route of ROUTES) {{
    const params = matchSegments(route.segments, parts);
    if (params) return {{ route, params }};
  }}
  return null;
}}

// The real-404 document: `app/not-found.tsx` (or a built-in default) wrapped in the
// root layout + head items, so an unknown path renders a full document, never the
// index tree.
function notFoundTree() {{
  const body = APP_NOT_FOUND
    ? createElement(APP_NOT_FOUND)
    : createElement("main", {{ id: "not-found" }}, "404 — This page could not be found.");
  let node = createElement(Fragment, null, ...headItems(ROOT_META), body);
  if (ROOT_LAYOUT) node = createElement(ROOT_LAYOUT, null, node);
  return node;
}}

// Compose the matched route: the page (with its `params`/`searchParams` promises),
// wrapped level-by-level leaf→root — each level's loading (Suspense) then error
// (client ErrorBoundary) then layout — with the head items injected inside the root
// layout. Returns `{{ tree, status, params }}`.
function documentTree(pathname) {{
  const m = matchRoute(pathname);
  if (!m) return {{ tree: notFoundTree(), status: 404, params: {{}} }};
  const {{ route, params }} = m;
  const paramsPromise = Promise.resolve(params);
  let node = createElement(route.page, {{ params: paramsPromise, searchParams: Promise.resolve({{}}) }});
  let headInjected = false;
  for (let i = route.levels.length - 1; i >= 0; i -= 1) {{
    const level = route.levels[i];
    if (level.loading) node = createElement(Suspense, {{ fallback: createElement(level.loading) }}, node);
    if (level.error) {{
      // React recovers a thrown Server-Component error via an error boundary only
      // ACROSS a Suspense boundary (Next pairs every `error.tsx` with a segment
      // Suspense). If this level had no `loading.tsx`, insert an empty-fallback one
      // so the throw is contained and the client `error.tsx` fallback renders.
      const inner = level.loading ? node : createElement(Suspense, {{ fallback: null }}, node);
      node = createElement(ERROR_BOUNDARY, {{ fallback: level.error }}, inner);
    }}
    if (i === 0) {{
      // Head items belong inside the root layout (React hoists them to <head>).
      node = createElement(Fragment, null, ...headItems(route), node);
      headInjected = true;
    }}
    if (level.layout) node = createElement(level.layout, {{ params: paramsPromise }}, node);
  }}
  if (!headInjected) node = createElement(Fragment, null, ...headItems(route), node);
  return {{ tree: node, status: 200, params }};
}}

// The status/params sidechannel to the orchestrator (fd 3). Guarded: a standalone
// run without fd 3 no-ops (a clear path, not a silent stub).
function writeMeta(meta) {{
  try {{
    writeSync(3, JSON.stringify(meta));
  }} catch {{
    // no fd 3 (standalone/action invocation) — nothing to report.
  }}
}}

async function streamToStdout(stream) {{
  const reader = stream.getReader();
  for (;;) {{
    const {{ done, value }} = await reader.read();
    if (done) break;
    process.stdout.write(Buffer.from(value));
  }}
}}

async function readStdin() {{
  const chunks = [];
  for await (const chunk of process.stdin) chunks.push(Buffer.from(chunk));
  return Buffer.concat(chunks).toString("utf8");
}}

// Drain a ReadableStream (Web) fully into a single Buffer.
async function drainToBuffer(stream) {{
  const reader = stream.getReader();
  const chunks = [];
  for (;;) {{
    const {{ done, value }} = await reader.read();
    if (done) break;
    chunks.push(Buffer.from(value));
  }}
  return Buffer.concat(chunks);
}}

// Render `pathname` to a flight BUFFER + control meta. Shared by the one-shot argv
// `render` op AND the persistent `serve` worker, so both paths render identically.
export async function renderRequest(pathname, bundlerConfig, reqCtx) {{
  const {{ tree, status, params }} = documentTree(pathname);
  // The request store: url/headers/cookie carried from the HTTP request + the matched
  // dynamic-segment params. `requestAls.run` MUST enclose BOTH the render call AND the
  // stream drain, or a late async Server Component loses the store.
  const store = {{
    url: new URL(reqCtx.url || ("http://localhost" + pathname), "http://localhost"),
    headers: new Headers(reqCtx.headers || []),
    cookieHeader: reqCtx.cookie || "",
    params,
  }};
  const control = {{}};
  const flight = await requestAls.run(store, async () => {{
    const stream = renderToReadableStream(tree, bundlerConfig, {{
      onError(error) {{
        const digest = (error && error.digest) || "";
        if (digest.startsWith("NEXT_REDIRECT;")) {{
          // NEXT_REDIRECT;<type>;<url>;<status>; — capture the target + status so the
          // orchestrator can issue a real HTTP redirect (do NOT SSR the errored tree).
          const parts = digest.split(";");
          control.redirect = parts.slice(2, -2).join(";");
          control.status = Number(parts[parts.length - 2]) || 307;
        }} else if (digest === "NEXT_HTTP_ERROR_FALLBACK;404") {{
          control.notFound = true;
          control.status = 404;
        }} else {{
          // A genuine error (recovered by an app-router error boundary, or fatal) —
          // log it; returning the digest marks it known to React so the stream drains.
          console.error("rsc-entry render onError:", error && error.stack ? error.stack : String(error));
        }}
        return digest || undefined;
      }},
    }});
    return await drainToBuffer(stream);
  }});
  return {{
    flight,
    status: control.status || status || 200,
    params,
    redirect: control.redirect,
    notFound: control.notFound,
  }};
}}

// Dispatch a server action, returning its result flight BUFFER.
export async function runAction(id, bundlerConfig, body) {{
  const request = new Request("http://diffpack.local/_action/", {{
    method: "POST",
    headers: {{ "x-diffpack-action-id": id, "content-type": "application/json" }},
    body,
  }});
  const response = await handleServerAction(request, bundlerConfig);
  if (!response.body) throw new Error("rsc-entry action: handler produced no response body");
  return await drainToBuffer(response.body);
}}

// Enumerate an Ssg route's concrete param sets by calling its generateStaticParams.
export async function staticParams(routePath) {{
  const ns = STATIC_PARAM_ROUTES[routePath];
  if (!ns) throw new Error(`rsc-entry staticparams: route ${{JSON.stringify(routePath)}} is not an Ssg route`);
  if (typeof ns.generateStaticParams !== "function")
    throw new Error(`rsc-entry staticparams: route ${{routePath}} has no generateStaticParams export`);
  const combos = await ns.generateStaticParams({{ params: Promise.resolve({{}}) }});
  if (!Array.isArray(combos))
    throw new Error(`rsc-entry staticparams: ${{routePath}} generateStaticParams did not return an array`);
  return combos;
}}

// The persistent DEV worker (`serve` op). Instead of the orchestrator spawning a
// fresh Node child per request (the cold-start cost dominates a server-component
// HMR edit), it spawns ONE `serve` worker that stays warm and answers requests over
// newline-delimited JSON on stdin/stdout. Fresh code after a `diffpack dev` re-emit:
// the worker RE-IMPORTS ITSELF with `?v=<mtime>` whenever this file changes on disk,
// giving a fresh module instance (fresh app code + its own inlined React) — the SAME
// process isolation the per-request child had, minus the per-request spawn. Requests
// carry the flight back base64-encoded (one JSON line per response).
async function serveLoop() {{
  const selfPath = fileURLToPath(import.meta.url.split("?")[0]);
  let cached = {{ mtime: 0, mod: null }};
  async function fresh() {{
    const mtime = statSync(selfPath).mtimeMs;
    if (mtime !== cached.mtime) {{
      // The bundle's registry runtime is a GLOBAL singleton keyed by entry path
      // (`globalThis["__diffpack_runtime:…"] ??= …`) with a per-module instance cache.
      // A bare re-import would share that runtime and keep returning the OLD cached
      // factories, so the fresh code never takes effect. Drop every diffpack runtime
      // first (this worker holds only the react-server one) so the re-imported bundle
      // builds a fresh runtime + fresh module cache and its new code actually runs.
      for (const key of Object.keys(globalThis)) {{
        if (key.indexOf("__diffpack_runtime:") === 0) delete globalThis[key];
      }}
      const m = await import(import.meta.url.split("?")[0] + "?v=" + mtime);
      // diffpack exposes the entry's named exports on the default namespace
      // (`export default __diffpackEntry`, whose members are renderRequest/runAction).
      const ns = m.default || m;
      const renderRequest = ns.renderRequest || m.renderRequest;
      const runAction = ns.runAction || m.runAction;
      if (typeof renderRequest !== "function" || typeof runAction !== "function") {{
        throw new Error("rsc-entry serve: re-imported bundle does not export renderRequest/runAction");
      }}
      cached = {{ mtime, mod: {{ renderRequest, runAction }} }};
    }}
    return cached.mod;
  }}
  const manifestCache = new Map();
  function manifest(path) {{
    let m = manifestCache.get(path);
    if (m === undefined) {{ m = JSON.parse(readFileSync(path, "utf8")); manifestCache.set(path, m); }}
    return m;
  }}
  const reply = (obj) => process.stdout.write(JSON.stringify(obj) + "\n");
  let buffer = "";
  process.stdin.setEncoding("utf8");
  process.stdin.on("data", async (chunk) => {{
    buffer += chunk;
    let nl;
    while ((nl = buffer.indexOf("\n")) >= 0) {{
      const line = buffer.slice(0, nl);
      buffer = buffer.slice(nl + 1);
      if (!line.trim()) continue;
      let req;
      try {{ req = JSON.parse(line); }} catch {{ continue; }}
      try {{
        const mod = await fresh();
        // A re-emit can change the manifest too; always re-read on the worker path.
        manifestCache.delete(req.manifestPath);
        if (req.op === "render") {{
          const r = await mod.renderRequest(req.pathname || "/", manifest(req.manifestPath), req.reqCtx || {{}});
          reply({{ id: req.id, flight: Buffer.from(r.flight).toString("base64"), status: r.status, params: r.params, redirect: r.redirect, notFound: r.notFound }});
        }} else if (req.op === "action") {{
          const flight = await mod.runAction(req.actionId, manifest(req.manifestPath), req.body || "");
          reply({{ id: req.id, flight: Buffer.from(flight).toString("base64"), status: 200 }});
        }} else {{
          reply({{ id: req.id, error: `unknown worker op ${{JSON.stringify(req.op)}}` }});
        }}
      }} catch (error) {{
        reply({{ id: req.id, error: error && error.stack ? error.stack : String(error) }});
      }}
    }}
  }});
  // Keep the process alive; resolves only if stdin closes (orchestrator gone).
  await new Promise((resolve) => process.stdin.on("end", resolve));
}}

async function main() {{
  const [op, ...rest] = process.argv.slice(2);
  if (op === "serve") {{
    await serveLoop();
    return;
  }}
  if (op === "render") {{
    const pathname = rest[0] || "/";
    const manifestPath = rest[1];
    if (!manifestPath) throw new Error("rsc-entry render: missing manifest path argument");
    const bundlerConfig = JSON.parse(readFileSync(manifestPath, "utf8"));
    // The per-request context arrives as JSON on stdin (the orchestrator writes it for
    // every render: {{ url, headers: [[k,v]...], cookie }}). A standalone invocation
    // with no stdin defaults to an empty context — cookies()/headers() then hard-error
    // (called outside a request), not silently empty.
    const ctxRaw = await readStdin();
    let reqCtx = {{}};
    if (ctxRaw && ctxRaw.trim()) {{
      try {{ reqCtx = JSON.parse(ctxRaw); }} catch {{ reqCtx = {{}}; }}
    }}
    const r = await renderRequest(pathname, bundlerConfig, reqCtx);
    process.stdout.write(r.flight);
    writeMeta({{ status: r.status, params: r.params, redirect: r.redirect, notFound: r.notFound }});
    return;
  }}
  if (op === "action") {{
    const id = rest[0];
    const manifestPath = rest[1];
    if (!id) throw new Error("rsc-entry action: missing action id argument");
    if (!manifestPath) throw new Error("rsc-entry action: missing manifest path argument");
    const bundlerConfig = JSON.parse(readFileSync(manifestPath, "utf8"));
    const body = await readStdin();
    const flight = await runAction(id, bundlerConfig, body);
    process.stdout.write(flight);
    return;
  }}
  if (op === "staticparams") {{
    // Print a JSON array of param objects to stdout.
    const combos = await staticParams(rest[0]);
    process.stdout.write(JSON.stringify(combos));
    return;
  }}
  throw new Error(`rsc-entry: unknown op ${{JSON.stringify(op)}}; expected "render", "action", "staticparams", or "serve"`);
}}

// Only the ORIGINAL entry process runs `main()`. A `serve`-mode re-import (which
// carries `?v=<mtime>` in its URL — see `serveLoop`) is loaded PURELY for its
// exported render functions; it must NOT re-enter main()/serveLoop or it would start
// a second stdin reader and recurse.
if (!import.meta.url.includes("?v=")) {{
  main().catch((error) => {{
    console.error(error && error.stack ? error.stack : String(error));
    process.exit(1);
  }});
}}
"##,
    )
}

/// Imports every `"use client"` island so the graph bundles + registers it under a
/// runtime id (pinned to a global so DCE keeps it).
fn island_pins(adapter_dir: &Path, islands: &[PathBuf]) -> String {
    let _ = adapter_dir;
    let mut out = String::new();
    for (index, island) in islands.iter().enumerate() {
        out.push_str(&format!(
            "import * as __island{index} from {};\n(globalThis).__diffpack_next_island_{index} = __island{index};\n",
            js_str(&island.to_string_lossy()),
        ));
    }
    out
}

/// The SSR-of-flight entry (Target::Server). Reconstructs the flight (client refs
/// resolved to THIS build's real islands via the seam) and renders the whole
/// app-router document (`<html>` and all — the RootLayout owns the document, as in
/// real Next) to HTML with react-dom, injecting the client bootstrap module and the
/// inlined flight via react-dom's bootstrap options so hydration matches exactly.
fn ssr_entry_module(adapter_dir: &Path, islands: &[PathBuf], hooks_context: &Path) -> String {
    let pins = island_pins(adapter_dir, islands);
    let lazy = js_str(&adapter_dir.join("lazy.js").to_string_lossy());
    let hooks_import = js_str(&hooks_context.to_string_lossy());
    format!(
        r#"// Generated by diffpack's next app-router adapter — the SSR-of-flight entry
// (Target::Server, node conditions: react + react-dom/server +
// react-server-dom-webpack/client as ONE React copy that also owns the client
// islands, bundled here as real code). It reconstructs the flight the react-server
// render produced and renders the FULL app-router document to HTML, wrapped in the
// app-router hooks contexts (params/pathname/searchParams) fed from the matched
// request so `useParams`/`usePathname`/`useSearchParams` resolve identically here
// and on the client (no hydration mismatch — the client entry feeds the SAME values).
import {{ createFromReadableStream }} from "react-server-dom-webpack/client";
import {{ renderToPipeableStream }} from "react-dom/server";
import {{ createElement }} from "react";
import {{ PathParamsContext, PathnameContext, SearchParamsContext }} from {hooks_import};
import {{ Writable }} from "node:stream";
{pins}
// Force a code split so the build uses the registry runtime the seam maps onto.
import({lazy}).then((module) => {{
  (globalThis).__diffpack_next_ssr_lazy = module.value;
}});

function installSeam() {{
  const runtimeKey = Object.keys(globalThis).find((key) => key.startsWith("__diffpack_runtime:"));
  if (!runtimeKey) {{
    throw new Error(
      "diffpack next ssr: no __diffpack_runtime:* registry on globalThis; the SSR bundle must use the registry runtime (a code split forces it)",
    );
  }}
  const runtime = globalThis[runtimeKey];
  const g = globalThis;
  g.__webpack_require__ = (id) => runtime.require(id);
  g.__webpack_require__.u = (c) => c;
  g.__webpack_chunk_load__ = () => Promise.resolve();
}}

// Reconstruct the flight and render the whole document to an HTML string. The
// client bootstrap module (`/client.js`) and the inlined flight are injected via
// react-dom's bootstrap options, so the served DOM (scripts included) is exactly
// what hydration on the browser expects — no mismatch.
export async function renderFlightToDocument(flightBytes, serverConsumerManifest, flightBase64, params, url) {{
  installSeam();
  const bytes = new Uint8Array(flightBytes);
  const stream = new ReadableStream({{
    start(controller) {{
      controller.enqueue(bytes);
      controller.close();
    }},
  }});
  const flightRoot = await createFromReadableStream(stream, {{
    serverConsumerManifest,
    callServer() {{
      throw new Error("diffpack next ssr: a server action was called during SSR");
    }},
  }});
  // Feed the app-router hooks contexts from the matched request (params + parsed url).
  // The client entry feeds the SAME values from the injected globals, so client
  // components using useParams/usePathname/useSearchParams hydrate without a mismatch.
  const pathname = (url && url.pathname) || "/";
  const search = (url && url.search) || "";
  const root = createElement(
    PathParamsContext.Provider,
    {{ value: params || {{}} }},
    createElement(
      PathnameContext.Provider,
      {{ value: pathname }},
      createElement(SearchParamsContext.Provider, {{ value: search }}, flightRoot),
    ),
  );
  return await new Promise((resolve, reject) => {{
    const parts = [];
    const sink = new Writable({{
      write(chunk, _enc, cb) {{
        parts.push(Buffer.from(chunk));
        cb();
      }},
    }});
    sink.on("finish", () => resolve(Buffer.concat(parts).toString("utf8")));
    sink.on("error", reject);
    const {{ pipe }} = renderToPipeableStream(root, {{
      bootstrapModules: ["/client.js"],
      bootstrapScriptContent:
        "window.__DIFFPACK_FLIGHT__ = " + JSON.stringify(flightBase64) + ";" +
        "window.__DIFFPACK_PARAMS__ = " + JSON.stringify(params || {{}}) + ";" +
        "window.__DIFFPACK_URL__ = " + JSON.stringify({{ pathname: pathname, search: search }}) + ";",
      onAllReady() {{
        pipe(sink);
      }},
      // A fatal shell error (before the first Suspense boundary) aborts the render.
      onShellError(error) {{
        reject(error);
      }},
      // Errors RECOVERED by an app-router error boundary land here too — log, do NOT
      // reject (the render still completes with the fallback rendered).
      onError(error) {{
        console.error("next-ssr onError:", error && error.message ? error.message : error);
      }},
    }});
  }});
}}
"#,
    )
}

/// The browser entry (Target::Client). Reconstructs the inlined flight and hydrates
/// the whole document (the RootLayout owns `<html>`). Imports each island so the
/// flight's client references resolve to real code in this build's registry.
fn client_entry_module(adapter_dir: &Path, islands: &[PathBuf], hooks_context: &Path) -> String {
    let pins = island_pins(adapter_dir, islands);
    let lazy = js_str(&adapter_dir.join("lazy.js").to_string_lossy());
    let hooks_import = js_str(&hooks_context.to_string_lossy());
    format!(
        r##"// Generated by diffpack's next app-router adapter — the BROWSER entry
// (Target::Client, with the RSC `__webpack_*` seam installed over its registry). It
// reconstructs the inlined flight, hydrates the server-rendered document, and runs a
// client Router that performs app-router SOFT NAVIGATION: `next/link` clicks (and
// `history` back/forward) fetch the target route's flight (`?__rsc=1`) and diff-render
// it into the live tree via `useTransition`, WITHOUT a full document load. The whole
// tree is wrapped in the app-router hooks contexts (params/pathname/searchParams) fed
// from the injected request globals — the SAME values the SSR entry fed, so
// useParams/usePathname/useSearchParams hydrate without a mismatch.
import {{ createFromReadableStream, createFromFetch }} from "react-server-dom-webpack/client";
import {{ hydrateRoot }} from "react-dom/client";
import {{ use, useState, useEffect, useTransition, createElement }} from "react";
import {{ PathParamsContext, PathnameContext, SearchParamsContext }} from {hooks_import};
import {{ callServer }} from "#diffpack-call-server";
{pins}
// Force a code split so the client build uses the registry runtime + the RSC seam.
import({lazy}).then((module) => {{
  (globalThis).__diffpack_next_client_lazy = module.value;
}});

// Fetch the target route's raw flight and return a thenable React can `use()`. The
// flight is resolved through the same `__webpack_*` client seam + `callServer`
// transport the action round-trip uses — no manifest needed.
function fetchTree(href) {{
  const sep = href.includes("?") ? "&" : "?";
  return createFromFetch(fetch(href + sep + "__rsc=1"), {{ callServer }});
}}

// The client Router: holds the current flight tree, and swaps it (inside a
// transition, keeping the old document visible until the new flight resolves) when
// navigation happens. React 19 reconciles the swapped `<html>/<head>/<body>` in place.
function Router({{ initialTree }}) {{
  const [tree, setTree] = useState(initialTree);
  const [, startTransition] = useTransition();
  useEffect(() => {{
    function navigate(to, options) {{
      const opts = options || {{}};
      const push = opts.push !== false;
      const href = typeof to === "string" ? to : to.href;
      const replace = opts.replace || (typeof to === "object" && to && to.replace);
      const next = fetchTree(href);
      startTransition(() => {{
        setTree(next);
        if (push) history[replace ? "replaceState" : "pushState"](null, "", href);
      }});
    }}
    window.__diffpack_navigate = navigate;
    const onpop = () => navigate(location.pathname + location.search, {{ push: false }});
    window.addEventListener("popstate", onpop);
    return () => window.removeEventListener("popstate", onpop);
  }}, []);
  return use(tree);
}}

function decodeFlight(base64) {{
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) bytes[i] = binary.charCodeAt(i);
  return bytes;
}}

function boot() {{
  const flightBase64 = window.__DIFFPACK_FLIGHT__;
  if (!flightBase64) {{
    throw new Error(
      "diffpack next client: window.__DIFFPACK_FLIGHT__ is missing; the server must inline the flight payload",
    );
  }}
  const bytes = decodeFlight(flightBase64);
  const stream = new ReadableStream({{
    start(controller) {{
      controller.enqueue(bytes);
      controller.close();
    }},
  }});
  const initialTree = createFromReadableStream(stream, {{ callServer }});
  // Feed the hooks contexts from the request globals the SSR bootstrap injected —
  // the SAME values the SSR entry rendered with, so hydration matches exactly.
  const params = window.__DIFFPACK_PARAMS__ || {{}};
  const urlInfo = window.__DIFFPACK_URL__ || {{ pathname: location.pathname, search: location.search }};
  const app = createElement(
    PathParamsContext.Provider,
    {{ value: params }},
    createElement(
      PathnameContext.Provider,
      {{ value: urlInfo.pathname }},
      createElement(
        SearchParamsContext.Provider,
        {{ value: urlInfo.search }},
        createElement(Router, {{ initialTree }}),
      ),
    ),
  );
  // The RootLayout owns the document, so we hydrate the whole document.
  hydrateRoot(document, app);
}}

boot();
"##,
    )
}

// --- next/* shims ----------------------------------------------------------------

fn next_link_shim() -> &'static str {
    r##""use client";
// `next/link` shim (diffpack next app-router adapter). A `"use client"` intercepting
// component: it renders the same server-reachable `<a href>`, but on the browser a
// plain left-click on an internal href is intercepted and handed to the client
// Router (`window.__diffpack_navigate`), which fetches the target route's flight
// (`?__rsc=1`) and diff-renders it WITHOUT a full document load. Modified clicks
// (meta/ctrl/shift/alt or a non-primary button), external/non-string hrefs, an
// already-`defaultPrevented` event, or the pre-hydration window (no
// `__diffpack_navigate`) all fall through to a real navigation — no `preventDefault`.
import { createElement } from "react";

export default function Link(props) {
  const { href, children, prefetch, replace, scroll, shallow, locale, onClick, ...rest } = props;
  const resolved = typeof href === "string" ? href : (href && href.pathname) || "#";
  function handleClick(event) {
    if (onClick) onClick(event);
    if (event.defaultPrevented) return;
    if (event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
    if (typeof href !== "string" || !href.startsWith("/")) return;
    if (typeof window === "undefined" || typeof window.__diffpack_navigate !== "function") return;
    event.preventDefault();
    window.__diffpack_navigate(resolved, { replace: !!replace });
  }
  return createElement("a", { href: resolved, onClick: handleClick, ...rest }, children);
}
"##
}

// --- next/image build-time variant emit + manifest (Slice J / gap 4.2) -----------
//
// Next's `<Image>` produces a responsive `srcset` of pre-optimized variants. There
// is no image-optimization server in this adapter: the optimization happens at BUILD
// time (pure-Rust `image` crate) and the output is plain static files under
// `public/_diffpack-image/`. The runtime shim (`next_image_shim`, a `getImgProps`
// port) reads the generated manifest to build the `srcset` pointing at those files.
//
// `deviceSizes`/`imageSizes` mirror Next's defaults (`next/dist/shared/lib/image-config`).

/// Next's default `deviceSizes`.
const IMAGE_DEVICE_SIZES: [u32; 8] = [640, 750, 828, 1080, 1200, 1920, 2048, 3840];
/// Next's default `imageSizes`.
const IMAGE_IMAGE_SIZES: [u32; 8] = [16, 32, 48, 64, 96, 128, 256, 384];

/// A raster src's build-optimization plan: intrinsic dimensions + the variant
/// widths to emit (all standard sizes `<=` intrinsic, plus the intrinsic itself so
/// full-resolution is always available). `unoptimized` entries (SVG and other
/// raster formats this build can't decode) carry no variants — the shim renders
/// their raw src with no `srcset`, exactly like Next handles SVGs.
#[derive(Debug, Clone)]
struct ImageEntry {
    /// The served URL, e.g. `/hero.png` (POSIX, leading slash).
    src: String,
    /// Path relative to `public/` (for reading the source at emit time).
    rel: PathBuf,
    /// Lowercased extension without the dot, e.g. `png`.
    ext: String,
    unoptimized: bool,
    width: u32,
    height: u32,
    /// Variant widths to emit (empty when `unoptimized`).
    variants: Vec<u32>,
}

/// A short, stable hash of a src URL — the variant file-name prefix. Deterministic
/// so the manifest (written in `configure`) and the emitted files (written from
/// `main.rs`) agree without sharing state.
fn image_hash(src: &str) -> String {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    src.hash(&mut hasher);
    format!("{:016x}", hasher.finish())
}

/// The served URL of one emitted variant.
fn image_variant_url(src: &str, width: u32, ext: &str) -> String {
    format!("/_diffpack-image/{}-{width}.{ext}", image_hash(src))
}

/// Variant widths for a raster of intrinsic `width`: every standard size strictly
/// below the intrinsic width (no upscaling) plus the intrinsic width itself.
fn variant_widths(intrinsic: u32) -> Vec<u32> {
    let mut all: Vec<u32> = IMAGE_IMAGE_SIZES
        .iter()
        .chain(IMAGE_DEVICE_SIZES.iter())
        .copied()
        .filter(|&w| w < intrinsic)
        .collect();
    all.sort_unstable();
    all.push(intrinsic);
    all
}

/// Scan `public/` for image files and build the manifest the runtime shim consumes.
/// PNG/JPEG get their intrinsic dimensions read and a responsive variant plan; other
/// image formats (SVG, GIF, WebP, AVIF — not decoded by this build's `image`
/// features) become `unoptimized` entries so the shim renders their raw src with no
/// `srcset` (never silently degrading, never throwing on a known image). A raster
/// the shim can't find an entry for at all is a hard error there (naming the src).
pub fn scan_public_images(root: &Path) -> Result<Vec<PublicImage>, String> {
    let public_dir = root.join("public");
    if !public_dir.is_dir() {
        return Ok(Vec::new());
    }
    let mut entries = Vec::new();
    scan_public_images_dir(&public_dir, &public_dir, &mut entries)?;
    entries.sort_by(|a, b| a.src.cmp(&b.src));
    Ok(entries.into_iter().map(PublicImage).collect())
}

fn scan_public_images_dir(
    base: &Path,
    dir: &Path,
    out: &mut Vec<ImageEntry>,
) -> Result<(), String> {
    let read = std::fs::read_dir(dir)
        .map_err(|error| format!("cannot read {}: {error}", dir.display()))?;
    for entry in read {
        let entry = entry.map_err(|error| format!("cannot read {}: {error}", dir.display()))?;
        let path = entry.path();
        let file_type = entry
            .file_type()
            .map_err(|error| format!("cannot stat {}: {error}", path.display()))?;
        if file_type.is_dir() {
            scan_public_images_dir(base, &path, out)?;
            continue;
        }
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| e.to_ascii_lowercase());
        let Some(ext) = ext else { continue };
        let is_image = matches!(
            ext.as_str(),
            "png" | "jpg" | "jpeg" | "svg" | "gif" | "webp" | "avif"
        );
        if !is_image {
            continue;
        }
        let rel = path
            .strip_prefix(base)
            .map_err(|_| format!("{} not under {}", path.display(), base.display()))?
            .to_path_buf();
        // POSIX-style served URL (leading slash, forward slashes).
        let src = format!(
            "/{}",
            rel.components()
                .map(|c| c.as_os_str().to_string_lossy())
                .collect::<Vec<_>>()
                .join("/")
        );
        let optimizable = matches!(ext.as_str(), "png" | "jpg" | "jpeg");
        if optimizable {
            match image::image_dimensions(&path) {
                Ok((width, height)) if width > 0 => {
                    let variants = variant_widths(width);
                    out.push(ImageEntry {
                        src,
                        rel,
                        ext: if ext == "jpg" { "jpeg".to_string() } else { ext },
                        unoptimized: false,
                        width,
                        height,
                        variants,
                    });
                }
                _ => {
                    // Undecodable/zero-size raster: register it unoptimized rather than
                    // throw at the shim (honest passthrough, no fake variants).
                    out.push(ImageEntry {
                        src,
                        rel,
                        ext,
                        unoptimized: true,
                        width: 0,
                        height: 0,
                        variants: Vec::new(),
                    });
                }
            }
        } else {
            out.push(ImageEntry {
                src,
                rel,
                ext,
                unoptimized: true,
                width: 0,
                height: 0,
                variants: Vec::new(),
            });
        }
    }
    Ok(())
}

/// Opaque handle over the internal [`ImageEntry`] so `main.rs` can drive the variant
/// emit without depending on the private shape.
pub struct PublicImage(ImageEntry);

/// Emit the downscaled raster variants for every optimizable image under `public/`
/// into `<out_public>/_diffpack-image/`. Called from the client-build public-copy
/// step (`main.rs`). Returns the number of variant files written. SVG/unoptimized
/// entries are skipped (their raw file is already copied by `copy_static_public`).
pub fn emit_image_variants(
    root: &Path,
    out_public: &Path,
    images: &[PublicImage],
) -> Result<usize, String> {
    let public_dir = root.join("public");
    let variant_dir = out_public.join("_diffpack-image");
    let mut written = 0usize;
    for PublicImage(entry) in images {
        if entry.unoptimized {
            continue;
        }
        let source = public_dir.join(&entry.rel);
        let decoded = image::open(&source)
            .map_err(|error| format!("cannot decode image {}: {error}", source.display()))?;
        std::fs::create_dir_all(&variant_dir)
            .map_err(|error| format!("cannot create {}: {error}", variant_dir.display()))?;
        for &w in &entry.variants {
            // Preserve aspect ratio; `resize` never upscales past the requested box,
            // and we only request widths `<=` intrinsic, so this is downscale-or-copy.
            let target_h = ((entry.height as u64 * w as u64) / entry.width.max(1) as u64).max(1);
            let variant = decoded.resize(w, target_h as u32, image::imageops::FilterType::Triangle);
            let dest = variant_dir.join(format!(
                "{}-{w}.{}",
                image_hash(&entry.src),
                entry.ext
            ));
            variant
                .save(&dest)
                .map_err(|error| format!("cannot write {}: {error}", dest.display()))?;
            written += 1;
        }
    }
    Ok(written)
}

/// Generate the `.diffpack-next/image-manifest.ts` module the `next/image` shim
/// imports: a default-exported map from served src URL to its variant plan. Always
/// written (an empty map when the app has no public images) so the shim's import
/// resolves in every graph.
fn image_manifest_module(images: &[PublicImage]) -> String {
    let mut body = String::from(
        "// GENERATED by diffpack next-adapter (Slice J / gap 4.2). Maps each public\n\
         // image src to its build-emitted responsive variants. The `next/image` shim\n\
         // reads this to build a real `srcset` pointing at static files under\n\
         // `/_diffpack-image/` — no image-optimization server.\nexport default {\n",
    );
    for PublicImage(entry) in images {
        if entry.unoptimized {
            body.push_str(&format!("  {}: {{ unoptimized: true }},\n", js_str(&entry.src)));
            continue;
        }
        let variants = entry
            .variants
            .iter()
            .map(|&w| format!("{}: {}", js_str(&w.to_string()), js_str(&image_variant_url(&entry.src, w, &entry.ext))))
            .collect::<Vec<_>>()
            .join(", ");
        body.push_str(&format!(
            "  {}: {{ width: {}, height: {}, variants: {{ {variants} }} }},\n",
            js_str(&entry.src),
            entry.width,
            entry.height,
        ));
    }
    body.push_str("};\n");
    body
}

fn next_image_shim() -> &'static str {
    r#"// `next/image` (diffpack next app-router adapter) — a faithful port of Next's
// `getImgProps`. Raster srcs with a build-emitted variant manifest entry get a real
// responsive `srcSet`/`sizes` pointing at static files under `/_diffpack-image/`
// (variants are emitted at BUILD time by the pure-Rust `image` crate — there is NO
// image-optimization server). SVG / `data:` / `blob:` / `unoptimized` srcs render
// the raw src with NO `srcSet` (byte-faithful to Next's SVG handling). `priority`
// renders a `<link rel="preload" as="image">` that React 19 hoists into <head>.
// A LOCAL raster with no manifest entry throws (naming the src) — never a silent
// degraded <img>. Runs in all three graphs (no directive; imported by Server
// Components). OUT of scope (documented in RSC_NEXT_GAP.md §4.2): static image
// imports (`import x from './x.png'`) and the blur placeholder.
import { createElement, Fragment } from "react";
import MANIFEST from "../image-manifest";

const DEVICE_SIZES = [640, 750, 828, 1080, 1200, 1920, 2048, 3840];
const IMAGE_SIZES = [16, 32, 48, 64, 96, 128, 256, 384];
const ALL_SIZES = [...IMAGE_SIZES, ...DEVICE_SIZES];

// Port of Next's getWidths: given a numeric `width` and/or a `sizes` string, pick
// the candidate widths (and whether the descriptor is `w` or `x`).
function getWidths(width, sizes) {
  if (sizes) {
    const re = /(^|\s)(1?\d?\d)vw/g;
    const percent = [];
    let m;
    while ((m = re.exec(sizes))) percent.push(parseInt(m[2], 10));
    if (percent.length) {
      const smallest = Math.min(...percent) * 0.01;
      return { widths: ALL_SIZES.filter((s) => s >= DEVICE_SIZES[0] * smallest), kind: "w" };
    }
    return { widths: ALL_SIZES, kind: "w" };
  }
  if (typeof width !== "number" || Number.isNaN(width)) return { widths: DEVICE_SIZES, kind: "w" };
  const seen = [];
  for (const w of [width, width * 2]) {
    const snapped = ALL_SIZES.find((p) => p >= w) || ALL_SIZES[ALL_SIZES.length - 1];
    if (!seen.includes(snapped)) seen.push(snapped);
  }
  return { widths: seen, kind: "x" };
}

function isRasterPath(s) {
  return /\.(png|jpe?g|webp|gif|avif)(\?|$)/i.test(s);
}

export default function Image(props) {
  const {
    src, alt, width, height, priority, loader, placeholder, blurDataURL,
    fill, quality, sizes, unoptimized, loading, fetchPriority, decoding,
    ...rest
  } = props;
  const rawSrc = typeof src === "string" ? src : (src && (src.src || src.default)) || "";
  const entry = MANIFEST[rawSrc];
  const isData = rawSrc.startsWith("data:") || rawSrc.startsWith("blob:");
  const isSvg = /\.svg(\?|$)/i.test(rawSrc);
  const forcedUnopt = Boolean(unoptimized) || isData || isSvg || (entry && entry.unoptimized);

  const numericWidth = typeof width === "number" ? width : Number(width);
  const imgLoading = priority ? undefined : (loading || "lazy");
  const imgDecoding = decoding || "async";
  const imgFetchPriority = priority ? "high" : fetchPriority;

  const baseImg = () =>
    createElement("img", {
      src: rawSrc,
      alt: alt || "",
      width,
      height,
      decoding: imgDecoding,
      loading: imgLoading,
      fetchPriority: imgFetchPriority,
      ...rest,
    });

  if (forcedUnopt) return baseImg();

  if (!entry) {
    if (isRasterPath(rawSrc) && rawSrc.startsWith("/")) {
      throw new Error(
        "next/image: no build-emitted variant manifest entry for raster src '" + rawSrc +
        "'. Put the image under public/ (png/jpeg) so diffpack emits its responsive " +
        "variants at build time, or pass the `unoptimized` prop."
      );
    }
    // External / unknown src (e.g. http(s)): no local optimization possible — render
    // the raw src (an honest passthrough, not a fake srcset).
    return baseImg();
  }

  // OPTIMIZED: build a responsive srcSet from the emitted variants.
  const { widths, kind } = getWidths(numericWidth, sizes);
  const intrinsic = entry.width;
  const chosen = widths.filter((w) => w <= intrinsic);
  if (chosen.length === 0) chosen.push(intrinsic);
  const parts = [];
  chosen.forEach((w, i) => {
    const url = entry.variants[String(w)];
    if (url) parts.push(url + " " + (kind === "w" ? w + "w" : (i + 1) + "x"));
  });
  const srcSet = parts.length ? parts.join(", ") : undefined;
  const largest = chosen[chosen.length - 1];
  const finalSrc = entry.variants[String(largest)] || entry.variants[String(intrinsic)] || rawSrc;

  const img = createElement("img", {
    src: finalSrc,
    srcSet,
    sizes,
    alt: alt || "",
    width,
    height,
    decoding: imgDecoding,
    loading: imgLoading,
    fetchPriority: imgFetchPriority,
    ...rest,
  });

  if (priority) {
    const link = createElement("link", {
      rel: "preload",
      as: "image",
      href: finalSrc,
      imageSrcSet: srcSet,
      imageSizes: sizes,
      fetchPriority: "high",
    });
    return createElement(Fragment, null, link, img);
  }
  return img;
}
"#
}

fn next_navigation_shim(hooks_context: &Path) -> String {
    let hooks_import = js_str(&hooks_context.to_string_lossy());
    format!(
        r#"// `next/navigation` shim (diffpack next app-router adapter). The client hooks
// (useParams/usePathname/useSearchParams) read the app-router hooks CONTEXTS, fed
// identically by the SSR and client entries — so they resolve on BOTH SSR and the
// browser with no hydration mismatch (NOT window globals, which don't exist during
// SSR). `redirect`/`notFound` on the SERVER throw Next's digest errors, which the
// react-server render's onError captures and turns into a real HTTP redirect / 404;
// on the client they fall back to browser navigation. This module is imported in all
// three graphs, so it uses `React.useContext` (undefined under the react-server
// condition, but the hooks are only ever CALLED inside client components).
import * as React from "react";
import {{ PathParamsContext, PathnameContext, SearchParamsContext }} from {hooks_import};

export function useRouter() {{
  return {{
    push(href) {{
      if (typeof window !== "undefined" && typeof window.__diffpack_navigate === "function") {{
        window.__diffpack_navigate(href, {{ replace: false }});
      }} else if (typeof window !== "undefined") {{
        window.location.assign(href);
      }}
    }},
    replace(href) {{
      if (typeof window !== "undefined" && typeof window.__diffpack_navigate === "function") {{
        window.__diffpack_navigate(href, {{ replace: true }});
      }} else if (typeof window !== "undefined") {{
        window.location.replace(href);
      }}
    }},
    back() {{ if (typeof window !== "undefined") window.history.back(); }},
    forward() {{ if (typeof window !== "undefined") window.history.forward(); }},
    refresh() {{ if (typeof window !== "undefined") window.location.reload(); }},
    prefetch() {{ /* no-op: this adapter has no prefetch cache */ }},
  }};
}}

export function usePathname() {{
  return React.useContext(PathnameContext);
}}

export function useSearchParams() {{
  return new URLSearchParams(React.useContext(SearchParamsContext) || "");
}}

export function useParams() {{
  return React.useContext(PathParamsContext) || {{}};
}}

export function redirect(href, type) {{
  if (typeof window === "undefined") {{
    // Server: throw Next's redirect digest; the react-server render's onError captures
    // it (NEXT_REDIRECT;<type>;<url>;<status>;) and the orchestrator issues a real 307.
    throw Object.assign(new Error("NEXT_REDIRECT"), {{
      digest: "NEXT_REDIRECT;" + (type || "replace") + ";" + href + ";307;",
    }});
  }}
  if (typeof window.__diffpack_navigate === "function") window.__diffpack_navigate(href, {{ replace: true }});
  else window.location.assign(href);
}}

export function permanentRedirect(href, type) {{
  if (typeof window === "undefined") {{
    throw Object.assign(new Error("NEXT_REDIRECT"), {{
      digest: "NEXT_REDIRECT;" + (type || "replace") + ";" + href + ";308;",
    }});
  }}
  if (typeof window.__diffpack_navigate === "function") window.__diffpack_navigate(href, {{ replace: true }});
  else window.location.assign(href);
}}

export function notFound() {{
  // Both server and client: throw Next's 404 digest. On the server the render's
  // onError captures it and the orchestrator serves the real 404 tree.
  throw Object.assign(new Error("NEXT_HTTP_ERROR_FALLBACK;404"), {{
    digest: "NEXT_HTTP_ERROR_FALLBACK;404",
  }});
}}
"#,
    )
}

fn next_headers_shim(request_context: &Path) -> String {
    let request_import = js_str(&request_context.to_string_lossy());
    format!(
        r#"// `next/headers` shim (diffpack next app-router adapter). These read the real
// per-request context the react-server render established (an AsyncLocalStorage
// carrying the request url/headers/cookie). They are `async` (Next 16 requires
// `await cookies()`/`await headers()`). Called OUTSIDE a request (no store), each
// HARD-ERRORS naming the missing context (repo no-silent-stub rule) rather than
// returning silently-empty values. Imported only by Server Components → lands only
// in the react-server graph (with node:async_hooks under the node condition).
import {{ requestAls }} from {request_import};

function parseCookieHeader(header) {{
  const map = new Map();
  (header || "").split(";").forEach(function (pair) {{
    const eq = pair.indexOf("=");
    if (eq === -1) return;
    const key = pair.slice(0, eq).trim();
    const value = pair.slice(eq + 1).trim();
    if (key) map.set(key, value);
  }});
  return map;
}}

export async function cookies() {{
  const store = requestAls.getStore();
  if (!store) {{
    // Tagged so the SSG prerenderer can distinguish a classifier gap (a route it
    // treated static that actually reads request state) from a generic render failure.
    throw Object.assign(new Error("diffpack next shim: cookies() was called outside a request context (no AsyncLocalStorage store) — call it inside a Server Component during a render"), {{ digest: "DIFFPACK_DYNAMIC_BAILOUT" }});
  }}
  const map = parseCookieHeader(store.cookieHeader);
  return {{
    get(name) {{ return map.has(name) ? {{ name: name, value: map.get(name) }} : undefined; }},
    getAll(name) {{
      const all = [];
      map.forEach(function (value, key) {{ if (!name || key === name) all.push({{ name: key, value: value }}); }});
      return all;
    }},
    has(name) {{ return map.has(name); }},
    size: map.size,
  }};
}}

export async function headers() {{
  const store = requestAls.getStore();
  if (!store) {{
    throw Object.assign(new Error("diffpack next shim: headers() was called outside a request context (no AsyncLocalStorage store) — call it inside a Server Component during a render"), {{ digest: "DIFFPACK_DYNAMIC_BAILOUT" }});
  }}
  return store.headers;
}}

export async function draftMode() {{
  // Faithful: this adapter threads no draft cookie, so draft mode is always disabled;
  // enabling it would need a mutable response cookie the adapter does not provide.
  return {{
    isEnabled: false,
    enable() {{ throw new Error("diffpack next shim: draftMode().enable() is not supported (no response-cookie plumbing)"); }},
    disable() {{ throw new Error("diffpack next shim: draftMode().disable() is not supported (no response-cookie plumbing)"); }},
  }};
}}
"#,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "diffpack-next-adapter-{}-{}",
            name,
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn parse_segment_classifies_conventions() {
        assert!(matches!(parse_segment("blog"), SegParse::Seg(Seg::Static(s)) if s == "blog"));
        assert!(matches!(parse_segment("[slug]"), SegParse::Seg(Seg::Dynamic(s)) if s == "slug"));
        assert!(matches!(parse_segment("[...rest]"), SegParse::Seg(Seg::CatchAll(s)) if s == "rest"));
        assert!(matches!(parse_segment("[[...rest]]"), SegParse::Seg(Seg::OptionalCatchAll(s)) if s == "rest"));
        assert!(matches!(parse_segment("(marketing)"), SegParse::Group));
        assert!(matches!(parse_segment("@modal"), SegParse::Skip));
        assert!(matches!(parse_segment("(.)photo"), SegParse::Skip));
    }

    #[test]
    fn classify_route_reproduces_next_fixture() {
        // Runs discovery + classification on the REAL fixture and asserts the kinds
        // match what `next build` reports (verified in docs/RSC_SSG_SPEC.md §0):
        // / and /about → Static; /blog/[slug], /go, /error-demo → Dynamic;
        // /products/[id] → Ssg.
        let fixture = Path::new(env!("CARGO_MANIFEST_DIR")).join("integration/next-app-router");
        let app = fixture.join("app");
        assert!(app.is_dir(), "fixture app dir missing at {}", app.display());
        let layout = first_existing(&app, "layout");
        let disc = discover_routes(&app, layout.as_deref()).unwrap();
        let kind_of = |path: &str| -> RouteKind {
            disc.routes
                .iter()
                .find(|r| r.url_path == path)
                .unwrap_or_else(|| panic!("route {path} not discovered: {:?}", disc.routes.iter().map(|r| &r.url_path).collect::<Vec<_>>()))
                .kind
        };
        assert_eq!(kind_of("/"), RouteKind::Static, "/ should be Static");
        assert_eq!(kind_of("/about"), RouteKind::Static, "/about should be Static");
        assert_eq!(kind_of("/blog/[slug]"), RouteKind::Dynamic, "/blog/[slug] reads cookies() → Dynamic");
        assert_eq!(kind_of("/go"), RouteKind::Dynamic, "/go is force-dynamic → Dynamic");
        assert_eq!(kind_of("/error-demo"), RouteKind::Dynamic, "/error-demo is force-dynamic → Dynamic");
        assert_eq!(kind_of("/products/[id]"), RouteKind::Ssg, "/products/[id] has generateStaticParams → Ssg");

        // PRECEDENCE on the real fixture: /blog/[slug] exports generateStaticParams AND
        // reads cookies(). `next build` classifies it ƒ Dynamic (the cookies read opts the
        // whole route into dynamic rendering) — NOT ● SSG — despite the export. diffpack
        // must reproduce that: the discovered route carries the generateStaticParams export
        // yet is Dynamic, and its reason names the precedence rather than falsely claiming
        // the route lacks generateStaticParams.
        let blog = disc.routes.iter().find(|r| r.url_path == "/blog/[slug]").unwrap();
        assert!(blog.has_generate_static_params, "/blog/[slug] fixture must export generateStaticParams (precedence exemplar)");
        assert_eq!(blog.kind, RouteKind::Dynamic, "/blog/[slug] stays Dynamic despite generateStaticParams (cookies read wins)");
        assert!(
            blog.dynamic_reason.contains("despite generateStaticParams"),
            "/blog/[slug] dynamic reason must name the precedence, got: {}",
            blog.dynamic_reason,
        );
        // Contrast: /products/[id] has generateStaticParams and NO request read → Ssg.
        let products = disc.routes.iter().find(|r| r.url_path == "/products/[id]").unwrap();
        assert!(products.has_generate_static_params && products.kind == RouteKind::Ssg);
    }

    #[test]
    fn classify_route_precedence() {
        // Unit-level precedence checks independent of the fixture.
        let base = RouteConfig { dynamic_params: true, ..Default::default() };
        // No dynamic segment, no reads → Static.
        assert_eq!(classify_route(false, &base), RouteKind::Static);
        // force-dynamic beats everything.
        let fd = RouteConfig { dynamic_config: Some("force-dynamic".into()), has_generate_static_params: true, ..base.clone() };
        assert_eq!(classify_route(true, &fd), RouteKind::Dynamic);
        // revalidate:0 → Dynamic.
        let rz = RouteConfig { revalidate: Some("0".into()), ..base.clone() };
        assert_eq!(classify_route(false, &rz), RouteKind::Dynamic);
        // request-state read → Dynamic even without a dynamic segment.
        let rr = RouteConfig { reads_request_state: true, ..base.clone() };
        assert_eq!(classify_route(false, &rr), RouteKind::Dynamic);
        // request-state read BEATS generateStaticParams even on a dynamic segment
        // (the /blog/[slug] case: cookies + gsp → Dynamic, matching next build's ƒ).
        let rr_gsp = RouteConfig { reads_request_state: true, has_generate_static_params: true, ..base.clone() };
        assert_eq!(classify_route(true, &rr_gsp), RouteKind::Dynamic);
        assert!(dynamic_reason(true, &rr_gsp).contains("despite generateStaticParams"));
        // dynamic segment + gsp → Ssg; without gsp → Dynamic.
        let gsp = RouteConfig { has_generate_static_params: true, ..base.clone() };
        assert_eq!(classify_route(true, &gsp), RouteKind::Ssg);
        assert_eq!(classify_route(true, &base), RouteKind::Dynamic);
        // force-static → ForceStatic.
        let fs = RouteConfig { dynamic_config: Some("force-static".into()), ..base.clone() };
        assert_eq!(classify_route(false, &fs), RouteKind::ForceStatic);
    }

    #[test]
    fn extract_export_const_reads_values() {
        assert_eq!(extract_export_const("export const dynamic = \"force-dynamic\";", "dynamic").as_deref(), Some("force-dynamic"));
        assert_eq!(extract_export_const("export const dynamicParams = false\n", "dynamicParams").as_deref(), Some("false"));
        assert_eq!(extract_export_const("export const revalidate = 60;", "revalidate").as_deref(), Some("60"));
        assert_eq!(extract_export_const("no exports here", "dynamic"), None);
    }

    #[test]
    fn literal_route_beats_dynamic_after_specificity_sort() {
        // A literal `/blog/new` must sort before `/blog/[slug]`, so matchRoute (which
        // returns the first match) never lets the dynamic segment shadow the literal.
        let root = scratch("specificity");
        std::fs::write(root.join("next.config.ts"), "export default {}\n").unwrap();
        let app = root.join("app");
        std::fs::create_dir_all(app.join("blog").join("[slug]")).unwrap();
        std::fs::create_dir_all(app.join("blog").join("new")).unwrap();
        std::fs::write(app.join("layout.tsx"), "export default function L({children}){return children}\n").unwrap();
        std::fs::write(app.join("page.tsx"), "export default function P(){return null}\n").unwrap();
        std::fs::write(app.join("blog").join("[slug]").join("page.tsx"), "export default function P(){return null}\n").unwrap();
        std::fs::write(app.join("blog").join("new").join("page.tsx"), "export default function P(){return null}\n").unwrap();
        let disc = discover_routes(&app, None).unwrap();
        let idx_new = disc.routes.iter().position(|r| r.url_path == "/blog/new").unwrap();
        let idx_slug = disc.routes.iter().position(|r| r.url_path == "/blog/[slug]").unwrap();
        assert!(idx_new < idx_slug, "literal /blog/new must precede /blog/[slug]: {:?}", disc.routes.iter().map(|r| &r.url_path).collect::<Vec<_>>());
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn non_next_project_returns_none() {
        // A directory with no next.config / app dir is not an app-router project;
        // the adapter must decline so the TanStack path handles it unchanged.
        let root = scratch("non-next");
        std::fs::create_dir_all(root.join("src")).unwrap();
        std::fs::write(root.join("src/client.tsx"), "export {}\n").unwrap();
        assert!(configure(&root, "client").unwrap().is_none());
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn app_router_project_scaffolds_and_configures_each_environment() {
        let root = scratch("app-router");
        std::fs::write(root.join("next.config.ts"), "export default {}\n").unwrap();
        let app = root.join("app");
        std::fs::create_dir_all(&app).unwrap();
        std::fs::write(
            app.join("layout.tsx"),
            "export default function RootLayout({children}){return children}\n",
        )
        .unwrap();
        std::fs::write(
            app.join("page.tsx"),
            "import {Counter} from './Counter';\nexport default function Home(){return null}\n",
        )
        .unwrap();
        std::fs::write(
            app.join("Counter.tsx"),
            "\"use client\";\nimport {useState} from 'react';\nexport function Counter(){const[n]=useState(0);return null}\n",
        )
        .unwrap();
        // A dynamic segment route (`app/blog/[slug]/page.tsx`) + its loading boundary,
        // so discovery must parse `[slug]` into a Dynamic segment and compose Suspense.
        let blog = app.join("blog").join("[slug]");
        std::fs::create_dir_all(&blog).unwrap();
        std::fs::write(
            blog.join("page.tsx"),
            "export default async function Post({params}){const {slug}=await params;return null}\n",
        )
        .unwrap();
        std::fs::write(
            blog.join("loading.tsx"),
            "export default function Loading(){return null}\n",
        )
        .unwrap();

        // client environment.
        let client = configure(&root, "client").unwrap().expect("app-router detected");
        assert_eq!(client.environment, "client");
        assert_eq!(client.build.target, Target::Client);
        let client_entry = client.entry.clone().unwrap();
        assert!(client_entry.ends_with(".diffpack-next/client.tsx"));
        // The generated client entry imports the discovered "use client" island.
        let client_src = std::fs::read_to_string(&client_entry).unwrap();
        assert!(client_src.contains("Counter.tsx"), "client entry pins the island");
        assert!(client_src.contains("hydrateRoot(document"), "hydrates the document");
        // Soft navigation (Slice G): the client entry runs a Router that exposes
        // window.__diffpack_navigate and fetches per-route flight via createFromFetch;
        // the `"use client"` next/link shim is pinned into the client graph so its
        // client reference resolves and it hydrates.
        assert!(client_src.contains("window.__diffpack_navigate"), "client entry installs the soft-nav router: {client_src}");
        assert!(client_src.contains("createFromFetch"), "client router fetches per-route flight: {client_src}");
        assert!(
            client_src.contains("shims/link.tsx") || client_src.contains("shims\\link.tsx"),
            "the next/link shim is pinned into the client graph: {client_src}"
        );

        // react-server environment: ReactServer target + react-server condition.
        let rs = configure(&root, "react-server").unwrap().unwrap();
        assert_eq!(rs.build.target, Target::ReactServer);
        assert!(rs.build.conditions.iter().any(|c| c == "react-server"));
        let rs_entry = rs.entry.clone().unwrap();
        assert!(rs_entry.ends_with(".diffpack-next/rsc-entry.tsx"));
        let rs_src = std::fs::read_to_string(&rs_entry).unwrap();
        // Builds the route table (page + layout modules interned) and composes each
        // route's nested-layout chain around its page, matched per requested pathname.
        assert!(rs_src.contains("import M0 from"), "{rs_src}");
        assert!(rs_src.contains("const ROUTES = ["), "{rs_src}");
        assert!(rs_src.contains("path: \"/\""), "{rs_src}");
        assert!(rs_src.contains("levels: ["), "{rs_src}");
        assert!(rs_src.contains("function documentTree(pathname)"), "{rs_src}");
        // Dynamic-segment matching (Slice H): the `[slug]` dir becomes a Dynamic
        // segment, matched at request time by the generated `matchRoute`.
        assert!(rs_src.contains("function matchRoute(pathname)"), "{rs_src}");
        assert!(
            rs_src.contains("{ k: \"static\", v: \"blog\" }, { k: \"dynamic\", v: \"slug\" }"),
            "the [slug] dir yields a Dynamic segment: {rs_src}"
        );
        // The loading boundary composes a <Suspense> and the error boundary is imported.
        assert!(rs_src.contains("Suspense"), "{rs_src}");
        assert!(rs_src.contains("ERROR_BOUNDARY"), "{rs_src}");
        // The real-404 path (no index fallback) renders the not-found tree.
        assert!(rs_src.contains("function notFoundTree()"), "{rs_src}");
        assert!(rs_src.contains("status: 404"), "{rs_src}");

        // ssr environment: Server target, node conditions, renderFlightToDocument.
        let ssr = configure(&root, "ssr").unwrap().unwrap();
        assert_eq!(ssr.build.target, Target::Server);
        let ssr_entry = ssr.entry.clone().unwrap();
        assert!(ssr_entry.ends_with(".diffpack-next/server.tsx"));
        let ssr_src = std::fs::read_to_string(&ssr_entry).unwrap();
        assert!(ssr_src.contains("renderFlightToDocument"));
        // The next/link shim is also pinned into the SSR graph so the soft-nav
        // link's client reference resolves during SSR-of-flight (hydration match).
        assert!(
            ssr_src.contains("shims/link.tsx") || ssr_src.contains("shims\\link.tsx"),
            "the next/link shim is pinned into the ssr graph: {ssr_src}"
        );

        // Slice I: per-request context wiring. The rsc-entry establishes the request
        // AsyncLocalStorage (requestAls.run) around the render+drain and captures a
        // server-side redirect() digest via onError.
        assert!(rs_src.contains("import { requestAls } from"), "rsc-entry imports the request-context ALS: {rs_src}");
        assert!(rs_src.contains("requestAls.run(store"), "rsc-entry wraps the render in requestAls.run: {rs_src}");
        assert!(rs_src.contains("NEXT_REDIRECT;"), "rsc-entry captures the redirect digest: {rs_src}");
        // The generated request-context + hooks-context modules exist.
        let adapter = root.join(".diffpack-next");
        let req_ctx = std::fs::read_to_string(adapter.join("request-context.ts")).unwrap();
        assert!(req_ctx.contains("AsyncLocalStorage"), "request-context holds the ALS: {req_ctx}");
        let hooks_ctx = std::fs::read_to_string(adapter.join("hooks-context.ts")).unwrap();
        assert!(hooks_ctx.contains("PathParamsContext"), "hooks-context exports PathParamsContext: {hooks_ctx}");
        assert!(hooks_ctx.contains("React.createContext ||"), "hooks-context guards createContext for the react-server graph: {hooks_ctx}");
        // The next/navigation shim reads the hooks CONTEXTS (not window) and redirect()
        // throws the NEXT_REDIRECT digest on the server.
        let nav = std::fs::read_to_string(adapter.join("shims").join("navigation.ts")).unwrap();
        assert!(nav.contains("React.useContext(PathParamsContext)"), "useParams reads PathParamsContext: {nav}");
        assert!(nav.contains("NEXT_REDIRECT;"), "server redirect() throws the redirect digest: {nav}");
        // The next/headers shim reads the real request context (async cookies/headers).
        let hdr = std::fs::read_to_string(adapter.join("shims").join("headers.ts")).unwrap();
        assert!(hdr.contains("requestAls.getStore()"), "cookies()/headers() read the request ALS: {hdr}");
        assert!(hdr.contains("export async function cookies"), "cookies() is async (Next 16): {hdr}");
        // Both the SSR and client entries feed the hooks contexts (Providers wrap the tree).
        assert!(ssr_src.contains("PathParamsContext.Provider"), "ssr entry provides the hooks contexts: {ssr_src}");
        assert!(client_src.contains("PathParamsContext.Provider"), "client entry provides the hooks contexts: {client_src}");

        // next/* shims aliased to real generated files.
        let aliased: std::collections::HashMap<_, _> = client.build.aliases.iter().cloned().collect();
        for spec in ["next/link", "next/image", "next/navigation", "next/headers"] {
            let target = aliased.get(spec).unwrap_or_else(|| panic!("{spec} aliased"));
            assert!(Path::new(target).is_file(), "{spec} shim file exists");
        }

        // Slice J: next/image is a getImgProps port reading a generated variant
        // manifest; the manifest module is always written (empty map when no public
        // images) so the shim's `../image-manifest` import resolves in every graph.
        let img_shim = std::fs::read_to_string(adapter.join("shims").join("image.tsx")).unwrap();
        assert!(img_shim.contains(r#"import MANIFEST from "../image-manifest""#), "image shim reads the variant manifest: {img_shim}");
        assert!(img_shim.contains("function getWidths"), "image shim ports getWidths: {img_shim}");
        assert!(img_shim.contains(r#"rel: "preload""#), "image shim hoists a priority preload link: {img_shim}");
        assert!(img_shim.contains("no build-emitted variant manifest entry"), "image shim throws (no silent stub) on a raster with no entry: {img_shim}");
        assert!(adapter.join("image-manifest.ts").is_file(), "the image variant manifest module is generated");

        // React dev/prod dispatch define is present (keeps React's dev build out).
        assert!(client
            .build
            .defines
            .iter()
            .any(|(k, v)| k == "process.env.NODE_ENV" && v == "\"production\""));
        assert!(!client.build.hmr, "production config never turns on HMR");
        assert!(is_app_router(&root), "is_app_router detects the scaffolded project");

        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn dev_config_switches_all_three_graphs_to_development() {
        // Slice K: `configure_dev` returns the SAME scaffold but switched to
        // development — HMR on, NODE_ENV=development (so React's dev build, which alone
        // exposes the Fast Refresh hook, is bundled), and `production`→`development` in
        // the resolve conditions — for every environment, so the react-server/SSR React
        // matches the client React at hydration.
        let root = scratch("app-router-dev");
        std::fs::write(root.join("next.config.ts"), "export default {}\n").unwrap();
        let app = root.join("app");
        std::fs::create_dir_all(&app).unwrap();
        std::fs::write(app.join("layout.tsx"), "export default function L({children}){return children}\n").unwrap();
        std::fs::write(app.join("page.tsx"), "export default function P(){return null}\n").unwrap();

        for environment in ["client", "react-server", "ssr"] {
            let prod = configure(&root, environment).unwrap().unwrap();
            let dev = configure_dev(&root, environment).unwrap().unwrap();
            assert!(!prod.build.hmr, "prod {environment} keeps HMR off");
            assert!(dev.build.hmr, "dev {environment} turns HMR on");
            assert!(
                dev.build.defines.iter().any(|(k, v)| k == "process.env.NODE_ENV" && v == "\"development\""),
                "dev {environment} defines NODE_ENV=development"
            );
            assert!(
                !dev.build.conditions.iter().any(|c| c == "production"),
                "dev {environment} swaps the production condition: {:?}",
                dev.build.conditions
            );
            // The entry/target are identical between prod and dev (same scaffold).
            assert_eq!(prod.build.target, dev.build.target);
            assert_eq!(prod.entry, dev.entry);
        }
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn image_variant_widths_no_upscale_plus_intrinsic() {
        // Every standard size strictly below the intrinsic width, then the intrinsic
        // itself (full-res always available; no upscaling past it).
        let w = variant_widths(1200);
        assert_eq!(*w.last().unwrap(), 1200, "intrinsic width is the max variant");
        assert!(w.iter().all(|&x| x <= 1200), "no variant exceeds the intrinsic width: {w:?}");
        assert!(w.contains(&640) && w.contains(&1080), "standard sizes below intrinsic are present: {w:?}");
        assert!(!w.contains(&1920), "no upscaling above intrinsic: {w:?}");
        // A tiny image still yields at least its own intrinsic width.
        assert_eq!(variant_widths(10), vec![10]);
    }

    #[test]
    fn image_variant_url_is_deterministic_and_hashed() {
        let a = image_variant_url("/hero.png", 640, "png");
        assert_eq!(a, image_variant_url("/hero.png", 640, "png"), "deterministic");
        assert!(a.starts_with("/_diffpack-image/"), "under the variant dir: {a}");
        assert!(a.ends_with("-640.png"), "carries the width + ext: {a}");
        assert_ne!(
            image_variant_url("/hero.png", 640, "png"),
            image_variant_url("/other.png", 640, "png"),
            "distinct srcs hash to distinct variant files"
        );
    }

    #[test]
    fn image_manifest_module_shapes_optimized_and_unoptimized() {
        let images = vec![
            PublicImage(ImageEntry {
                src: "/hero.png".into(),
                rel: PathBuf::from("hero.png"),
                ext: "png".into(),
                unoptimized: false,
                width: 1200,
                height: 300,
                variants: vec![640, 1200],
            }),
            PublicImage(ImageEntry {
                src: "/next.svg".into(),
                rel: PathBuf::from("next.svg"),
                ext: "svg".into(),
                unoptimized: true,
                width: 0,
                height: 0,
                variants: Vec::new(),
            }),
        ];
        let module = image_manifest_module(&images);
        assert!(module.contains("export default {"), "{module}");
        assert!(module.contains(r#""/hero.png": { width: 1200, height: 300, variants: {"#), "optimized raster entry: {module}");
        assert!(module.contains(r#""640": "/_diffpack-image/"#), "variant keyed by width: {module}");
        assert!(module.contains(r#""/next.svg": { unoptimized: true }"#), "svg is unoptimized (no srcset): {module}");
    }
}
