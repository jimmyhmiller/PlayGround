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

/// Module-file extensions the adapter recognizes for app-router convention files
/// (layout/loading/error/not-found/route/next.config).
const MODULE_EXTS: [&str; 4] = ["tsx", "jsx", "ts", "js"];

/// Extensions a `page` may use — the module set PLUS MDX/Markdown, so `page.mdx` /
/// `page.md` is a route exactly like `page.tsx`. Only `page` is MDX-eligible; the other
/// convention files stay on [`MODULE_EXTS`].
const PAGE_EXTS: [&str; 6] = ["tsx", "jsx", "ts", "js", "mdx", "md"];

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
    first_existing_page(&app)
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

/// The app's `instrumentation.{ts,tsx,js,jsx}` (project root or `src/`), or None. Next's
/// convention for the boot hook whose `register()` runs once at server startup
/// (OpenTelemetry / Sentry / etc.).
pub fn instrumentation_entry(root: &Path) -> Option<PathBuf> {
    let canonical = root.canonicalize().unwrap_or_else(|_| root.to_path_buf());
    first_existing(&canonical, "instrumentation")
        .or_else(|| first_existing(&canonical.join("src"), "instrumentation"))
        .map(|p| p.canonicalize().unwrap_or(p))
}

/// The generated instrumentation boot entry: it imports the app's `register()` and runs
/// it at module load. `register()` is CALLED (a top-level side effect the bundler keeps),
/// not re-exported, because the native bundler tree-shakes an unused entry export — so
/// bundling this wrapper and importing the result at boot runs `register()` exactly once,
/// before `server.listen`. A missing `register` is a hard error (never a silent no-op).
fn instrumentation_entry_module(user_path: &Path) -> String {
    let spec = js_str(&user_path.to_string_lossy());
    let display = js_str(&user_path.display().to_string());
    format!(
        "// Generated by diffpack's next app-router adapter — the instrumentation boot entry.\n\
         // Runs the app's instrumentation register() once at server startup (before listen);\n\
         // zero per-request cost. register() is called (not re-exported) so the bundler keeps it.\n\
         import {{ register }} from {spec};\n\
         if (typeof register !== \"function\") {{\n  \
           throw new Error(\"instrumentation file \" + {display} + \" exports no register() function\");\n\
         }}\n\
         await register();\n",
    )
}

/// If the app has an `instrumentation.{{ts,js}}`, write the generated boot-entry wrapper
/// (see [`instrumentation_entry_module`]) under `<root>/.diffpack-next/` and return its
/// path so the production build can bundle it natively to `<out>/instrumentation.mjs`.
/// Returns `Ok(None)` when the app has no instrumentation file.
pub fn write_instrumentation_wrapper(root: &Path) -> Result<Option<PathBuf>, String> {
    let Some(user_path) = instrumentation_entry(root) else {
        return Ok(None);
    };
    let canonical = root.canonicalize().unwrap_or_else(|_| root.to_path_buf());
    let adapter_dir = canonical.join(ADAPTER_DIR);
    std::fs::create_dir_all(&adapter_dir)
        .map_err(|error| format!("cannot create {}: {error}", adapter_dir.display()))?;
    let wrapper = adapter_dir.join("instrumentation-entry.mjs");
    std::fs::write(&wrapper, instrumentation_entry_module(&user_path))
        .map_err(|error| format!("cannot write {}: {error}", wrapper.display()))?;
    Ok(Some(wrapper))
}

/// The first `<dir>/<stem>.<ext>` that exists, in `MODULE_EXTS` priority order.
fn first_existing(dir: &Path, stem: &str) -> Option<PathBuf> {
    first_existing_ext(dir, stem, &MODULE_EXTS)
}

/// The first `<dir>/<stem>.<ext>` that exists, in the given extension priority order.
fn first_existing_ext(dir: &Path, stem: &str, exts: &[&str]) -> Option<PathBuf> {
    exts.iter()
        .map(|ext| dir.join(format!("{stem}.{ext}")))
        .find(|path| path.is_file())
}

/// The route `page` module (MDX-eligible: `page.{tsx,jsx,ts,js,mdx,md}`).
fn first_existing_page(dir: &Path) -> Option<PathBuf> {
    first_existing_ext(dir, "page", &PAGE_EXTS)
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

/// The extensions a metadata IMAGE file convention (`icon`/`favicon`/`apple-icon`/
/// `opengraph-image`/`twitter-image`) may use as a STATIC file. Code generators
/// (`.tsx`/`.jsx`/`.ts`/`.js` returning an `ImageResponse`) are a separate, heavy-dep
/// capability that is deliberately NOT supported here (see [`scan_metadata_images`]).
const METADATA_IMAGE_EXTS: [&str; 7] = ["ico", "png", "jpg", "jpeg", "gif", "svg", "webp"];

/// A Next metadata IMAGE file convention discovered at the app root (`app/icon.png`,
/// `app/favicon.ico`, `app/apple-icon.png`, `app/opengraph-image.jpg`,
/// `app/twitter-image.png`). Copied to the served `public/` output at build time and
/// head-linked (`<link rel>`/`<meta property>`) into every route — zero per-request cost.
#[derive(Debug, Clone)]
struct MetaImage {
    /// The convention family, which determines the head element emitted.
    kind: MetaImageKind,
    /// The source file (absolute).
    source: PathBuf,
    /// The served URL path (`/icon.png`), which is also the copied output filename.
    served: String,
    /// The image MIME type inferred from the extension (for `<link type>`).
    mime: &'static str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MetaImageKind {
    /// `favicon.ico` -> `<link rel="icon" sizes="any">`.
    Favicon,
    /// `icon.*` -> `<link rel="icon">`.
    Icon,
    /// `apple-icon.*` -> `<link rel="apple-touch-icon">`.
    AppleIcon,
    /// `opengraph-image.*` -> `<meta property="og:image">`.
    OpengraphImage,
    /// `twitter-image.*` -> `<meta name="twitter:image">`.
    TwitterImage,
}

/// The image MIME type for a metadata-image extension.
fn metadata_image_mime(ext: &str) -> &'static str {
    match ext.to_ascii_lowercase().as_str() {
        "ico" => "image/x-icon",
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        "gif" => "image/gif",
        "svg" => "image/svg+xml",
        "webp" => "image/webp",
        other => unreachable!("metadata_image_mime called on non-image ext {other}"),
    }
}

/// Reads `export const metadata = { title, description }` string literals from a
/// module (the Metadata API subset), for the adapter to render `<title>`/`<meta>`
/// into the document head. `generateMetadata()` and non-string values are not read
/// (a documented gap), never silently guessed.
fn scan_metadata(path: &Path, source: &str) -> RouteMetadata {
    use oxc_ast::ast::{Declaration, Expression, ObjectPropertyKind, PropertyKey, Statement};
    // MDX/Markdown page: metadata comes from `title`/`description` frontmatter.
    if crate::mdx::is_mdx_path(path) {
        let fm = crate::mdx::frontmatter(source);
        return RouteMetadata {
            title: fm.get("title").cloned(),
            description: fm.get("description").cloned(),
        };
    }
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

/// Whether a layout/page module exports any metadata the render-time resolver reads:
/// `metadata`, `generateMetadata`, `viewport`, or `generateViewport`. A substring scan
/// over comment-stripped source (MDX frontmatter counts as `metadata`).
fn module_exports_metadata(path: &Path) -> bool {
    if crate::mdx::is_mdx_path(path) {
        return true; // frontmatter title/description resolve as metadata
    }
    let Ok(raw) = std::fs::read_to_string(path) else { return false };
    let source = strip_comments(&raw);
    ["metadata", "generateMetadata", "viewport", "generateViewport"]
        .iter()
        .any(|name| {
            source.contains(&format!("export const {name}"))
                || source.contains(&format!("export let {name}"))
                || source.contains(&format!("export var {name}"))
                || source.contains(&format!("export function {name}"))
                || source.contains(&format!("export async function {name}"))
        })
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
    /// `export const revalidate = N` (N>0) on an otherwise-static route: prerendered at
    /// build time AND regenerated on demand once the cached copy is older than N seconds
    /// (stale-while-revalidate). Next's ISR.
    Isr,
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
            RouteKind::Isr => "isr",
            RouteKind::Dynamic => "dynamic",
        }
    }
    /// Whether the prerenderer emits an on-disk `.html`/`.rsc` for this route.
    fn is_prerendered(self) -> bool {
        matches!(
            self,
            RouteKind::Static | RouteKind::ForceStatic | RouteKind::Ssg | RouteKind::Isr
        )
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
    /// `export const runtime = "nodejs" | "edge"`. `edge` hard-errors at discovery
    /// (diffpack has no edge runtime); `nodejs` (the default) is inert.
    runtime: Option<String>,
    /// `export const fetchCache = <mode>` — the fetch Data Cache mode. Parsed and
    /// WARNed (diffpack has no fetch Data Cache; fetches are always live).
    fetch_cache: Option<String>,
    /// `export const preferredRegion = <region>` — serverless region hint. Parsed and
    /// WARNed (a single-node server has no region routing).
    preferred_region: Option<String>,
    /// `export const maxDuration = <seconds>` — serverless per-invocation timeout.
    /// Parsed and WARNed (the persistent worker has no per-request timeout; enforcing
    /// one would add a per-render timer, which would make diffpack slower than Next).
    max_duration: Option<String>,
    /// `export const experimental_ppr = <bool>` — Partial Prerendering opt-in. Parsed
    /// and WARNed (PPR is not implemented; the route is classified by the normal rules).
    experimental_ppr: Option<String>,
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
    // Remaining route-segment-config exports. Recognized + reported (never silently
    // ignored); see `validate_segment_config` for how each is honored or WARNed.
    let runtime = extract_export_const(source, "runtime");
    let fetch_cache = extract_export_const(source, "fetchCache");
    let preferred_region = extract_export_const(source, "preferredRegion");
    let max_duration = extract_export_const(source, "maxDuration");
    let experimental_ppr = extract_export_const(source, "experimental_ppr");
    RouteConfig {
        has_generate_static_params,
        dynamic_config,
        dynamic_params,
        revalidate,
        reads_request_state,
        runtime,
        fetch_cache,
        preferred_region,
        max_duration,
        experimental_ppr,
    }
}

/// Enforce the route-segment-config exports diffpack recognizes beyond the
/// static/dynamic set. `runtime = "edge"` is a HARD ERROR naming the route (diffpack
/// has no edge runtime; there is no correct way to serve it, so failing loudly beats
/// silently running it on Node with different semantics). The remaining exports
/// (`fetchCache`, `preferredRegion`, `maxDuration`, `experimental_ppr`) are advisory
/// for a native single-node server: each is reported with a build WARN explaining
/// precisely why diffpack cannot honor it, so the behavior is explicit rather than a
/// silent default. Returns `Err` only for the unsupported edge runtime.
fn validate_segment_config(url_path: &str, cfg: &RouteConfig) -> Result<(), String> {
    if let Some(runtime) = cfg.runtime.as_deref() {
        match runtime {
            "edge" | "experimental-edge" => {
                return Err(format!(
                    "route {url_path}: runtime = \"{runtime}\" is not supported (diffpack has no \
                     edge runtime). Remove the export to use the Node.js runtime, which diffpack \
                     serves natively.",
                ));
            }
            "nodejs" => {} // the default; inert.
            other => {
                return Err(format!(
                    "route {url_path}: runtime = \"{other}\" is not a recognized Next runtime \
                     (expected \"nodejs\" or \"edge\").",
                ));
            }
        }
    }
    if let Some(mode) = cfg.fetch_cache.as_deref() {
        eprintln!(
            "next segment config: route {url_path} exports `fetchCache = \"{mode}\"`, which \
             diffpack does not honor (it has no fetch Data Cache; fetches always hit the network). \
             The page still renders correctly; only fetch-level caching is unavailable.",
        );
    }
    if let Some(region) = cfg.preferred_region.as_deref() {
        eprintln!(
            "next segment config: route {url_path} exports `preferredRegion = \"{region}\"`, which \
             diffpack does not honor (a single-node server has no region routing). Advisory only.",
        );
    }
    if let Some(secs) = cfg.max_duration.as_deref() {
        eprintln!(
            "next segment config: route {url_path} exports `maxDuration = {secs}`, which diffpack \
             does not enforce (the persistent worker has no per-request timeout; enforcing one \
             would add per-render overhead). Advisory only.",
        );
    }
    if let Some(ppr) = cfg.experimental_ppr.as_deref() {
        eprintln!(
            "next segment config: route {url_path} exports `experimental_ppr = {ppr}`, but Partial \
             Prerendering is not implemented; the route is classified by the normal static/dynamic \
             rules instead.",
        );
    }
    Ok(())
}

/// `export const revalidate = <n>` as a positive integer number of seconds, or None
/// (absent, `false`, `0`, or non-numeric). `revalidate = 0` opts a route into
/// force-dynamic (handled in `classify_route`); it is NOT an ISR TTL, so it maps to None
/// here.
fn parse_revalidate(cfg: &RouteConfig) -> Option<u64> {
    match cfg.revalidate.as_deref() {
        Some(v) => v.trim().parse::<u64>().ok().filter(|n| *n > 0),
        None => None,
    }
}

/// Classifies a route from whether its pattern has a dynamic segment + its config,
/// reproducing Next's static/dynamic decision for the fixture exactly. Precedence:
/// force-dynamic (or `revalidate:0`) > force-static/error > request-state reads >
/// dynamic-segment (gsp ? Ssg : Dynamic) > `revalidate:N` ISR > Static.
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
    if parse_revalidate(cfg).is_some() {
        return RouteKind::Isr;
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
    /// A parallel `@slot` or intercepting `(.)`/`(..)`/`(...)` marker in the PRIMARY
    /// page path — that page is not a normal route. `@slots` are instead discovered as
    /// parallel-route slots (composed into their layout as named props); intercepts are
    /// soft-navigation only (they do not match on a hard render).
    Skip,
}

/// Parses one app-router directory-name component into a URL segment classification.
/// `[x]`→Dynamic, `[...x]`→CatchAll, `[[...x]]`→OptionalCatchAll, `(group)`→omitted;
/// `@slot` / `(.)`-intercepts →Skip in the primary path (see [`SegParse::Skip`]).
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

/// The slot name of a parallel-route directory (`@team` -> `team`), or None.
fn slot_name(comp: &str) -> Option<String> {
    comp.strip_prefix('@').map(|s| s.to_string())
}

/// Whether a directory name is an intercepting-route marker (`(.)`/`(..)`/`(...)`).
/// Intercepts are soft-navigation only, so on a hard render they do NOT match (the real
/// route renders); they are treated as Skip inside slot discovery.
fn is_intercept_marker(comp: &str) -> bool {
    comp.starts_with("(.")
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
    /// `template.tsx` — a layout-like wrapper that RE-MOUNTS on navigation (fresh state
    /// each URL). Composed just inside this level's layout, keyed by pathname.
    template: Option<PathBuf>,
    /// Parallel-route `@slot` subtrees hosted by THIS directory's layout (passed to it
    /// as named props). Empty for a normal level.
    slots: Vec<Slot>,
    /// The number of URL segments consumed from `app/` down to and including this
    /// directory — so a slot matches the below-level URL parts via `parts.slice(slotBase)`.
    part_offset: usize,
}

/// A parallel route `@slot` (e.g. `@team`): its name (the layout prop it fills), the
/// routes inside it, and an optional `default.tsx` fallback when none match.
#[derive(Debug, Clone)]
struct Slot {
    name: String,
    routes: Vec<SlotRoute>,
    default: Option<PathBuf>,
}

/// One matchable route inside a `@slot`: its segment pattern (RELATIVE to the slot
/// directory) + page + the slot-internal layout chain wrapping it.
#[derive(Debug, Clone)]
struct SlotRoute {
    segments: Vec<Seg>,
    page: PathBuf,
    levels: Vec<Level>,
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
    /// `export const revalidate = N` (N>0) in seconds — the ISR TTL. Emitted into the
    /// prerender plan so the orchestrator revalidates the cached page after N seconds.
    revalidate_seconds: Option<u64>,
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
        if let Some(secs) = route.revalidate_seconds {
            eprintln!(
                "next ISR: route {} has `revalidate = {secs}` — prerendered at build and \
                 regenerated on demand once older than {secs}s.",
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
            // ISR TTL (seconds). Applies to isr routes and to any static/ssg route that
            // also declares `revalidate` — the orchestrator revalidates on this TTL.
            if let Some(secs) = route.revalidate_seconds {
                fields.push_str(&format!(", \"revalidate\": {secs}"));
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
    /// `app/global-error.tsx` — the app-root error boundary that owns `<html>`. When a
    /// throw escapes every nested `error.tsx` (including one in the root layout), this
    /// component's own document replaces the whole tree. Discovered once, at the root.
    global_error: Option<PathBuf>,
    /// `route.*` HTTP endpoints (`app/api/**/route.ts`), most-specific first.
    handlers: Vec<RouteHandler>,
    /// Intercepting routes (`@slot/(.)…`): a soft-navigation to a matching target
    /// renders this page as an overlay instead of the full target document.
    intercepts: Vec<Intercept>,
    /// Metadata IMAGE file conventions (`app/icon.png`, `app/favicon.ico`, ...) at the
    /// app root: copied to `public/` at build and head-linked into every route.
    meta_images: Vec<MetaImage>,
}

/// An intercepting route (`app/@modal/(..)photo/[id]/page.tsx`): on a SOFT navigation to
/// a URL matching `target_segments`, this page renders as an overlay (a modal) over the
/// current page, with the URL masked to the target. A hard load of the target renders
/// the real full-page route instead (intercepts are soft-nav only).
#[derive(Debug, Clone)]
struct Intercept {
    /// The target URL pattern this intercepts (e.g. `[photo, [id]]` for `(..)photo/[id]`).
    target_segments: Vec<Seg>,
    /// The overlay page + its slot-internal layout chain.
    page: PathBuf,
    levels: Vec<Level>,
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
/// The HTTP methods a route handler (`route.{ts,js}`) may export.
const HTTP_METHODS: [&str; 7] = ["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"];

/// A Next app-router ROUTE HANDLER: a `route.{ts,tsx,js,jsx}` file whose exported
/// HTTP-method functions serve `<dir>` as an HTTP endpoint (e.g. `app/api/users/
/// route.ts` -> `/api/users`), instead of rendering a React page.
#[derive(Debug, Clone)]
struct RouteHandler {
    /// The matched URL path (`/api/users/[id]`).
    url_path: String,
    segments: Vec<Seg>,
    /// The handler module (absolute, canonical).
    file: PathBuf,
    /// The HTTP methods it exports, in canonical order.
    methods: Vec<String>,
}

/// Whether `source` has a top-level export named `name` (a `function`/`const`/`let`/
/// `var` declaration or a re-export `{ name }` / `{ x as name }`) — used to detect
/// which HTTP methods a `route.*` file implements.
fn exports_symbol(source: &str, name: &str) -> bool {
    for form in [
        format!("export async function {name}"),
        format!("export function {name}"),
        format!("export const {name}"),
        format!("export let {name}"),
        format!("export var {name}"),
        format!("export {{ {name}"),
        format!("as {name} }}"),
        format!("as {name},"),
    ] {
        if source.contains(&form) {
            return true;
        }
    }
    false
}

/// Discover every `route.*` HTTP endpoint under `app/`, most-specific first (same
/// specificity order as page routes).
fn discover_route_handlers(app_dir: &Path) -> Result<Vec<RouteHandler>, String> {
    let mut handlers = Vec::new();
    discover_route_handlers_dir(app_dir, app_dir, &mut handlers)?;
    handlers.sort_by(|a, b| {
        let count = |r: &RouteHandler, f: fn(&Seg) -> bool| r.segments.iter().filter(|s| f(s)).count();
        let ca = count(a, |s| matches!(s, Seg::CatchAll(_) | Seg::OptionalCatchAll(_)));
        let cb = count(b, |s| matches!(s, Seg::CatchAll(_) | Seg::OptionalCatchAll(_)));
        let da = count(a, |s| matches!(s, Seg::Dynamic(_)));
        let db = count(b, |s| matches!(s, Seg::Dynamic(_)));
        ca.cmp(&cb)
            .then(da.cmp(&db))
            .then(b.segments.len().cmp(&a.segments.len()))
            .then(a.url_path.cmp(&b.url_path))
    });
    Ok(handlers)
}

fn discover_route_handlers_dir(
    app_dir: &Path,
    dir: &Path,
    out: &mut Vec<RouteHandler>,
) -> Result<(), String> {
    if let Some(route_file) = first_existing(dir, "route") {
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
            let file = route_file.canonicalize().unwrap_or(route_file.clone());
            let source = std::fs::read_to_string(&file).unwrap_or_default();
            let methods: Vec<String> = HTTP_METHODS
                .iter()
                .filter(|m| exports_symbol(&source, m))
                .map(|m| m.to_string())
                .collect();
            if !methods.is_empty() {
                out.push(RouteHandler {
                    url_path: segments_display(&segments),
                    segments,
                    file,
                    methods,
                });
            }
        }
    }
    let read = match std::fs::read_dir(dir) {
        Ok(read) => read,
        Err(_) => return Ok(()),
    };
    for child in read.flatten().map(|e| e.path()).filter(|p| p.is_dir()) {
        let name = child.file_name().and_then(|n| n.to_str()).unwrap_or("");
        if name.starts_with('.') || name == "node_modules" {
            continue;
        }
        discover_route_handlers_dir(app_dir, &child, out)?;
    }
    Ok(())
}

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
        global_error: first_existing(app_dir, "global-error").map(|p| p.canonicalize().unwrap_or(p)),
        handlers: discover_route_handlers(app_dir)?,
        intercepts: discover_intercepts(app_dir)?,
        meta_images: scan_metadata_images(app_dir)?,
    })
}

/// A metadata FILE convention endpoint (`app/sitemap.ts` -> `/sitemap.xml`,
/// `app/robots.ts` -> `/robots.txt`, `app/manifest.ts` -> `/manifest.webmanifest`).
/// Each is served by the SAME route-handler machinery as a `route.ts` endpoint: a tiny
/// generated wrapper imports the user file's default export, calls it, and serializes
/// the result to XML / text / JSON with the right content-type. Only the app root is
/// scanned (Next's convention location for these three files).
struct MetaFileConvention {
    /// The convention file stem (`sitemap`/`robots`/`manifest`).
    stem: &'static str,
    /// The served URL (`/sitemap.xml`).
    url: &'static str,
    /// The generated wrapper's basename under `shims/`.
    wrapper: &'static str,
}

const META_FILE_CONVENTIONS: [MetaFileConvention; 3] = [
    MetaFileConvention { stem: "sitemap", url: "/sitemap.xml", wrapper: "metadata-sitemap.ts" },
    MetaFileConvention { stem: "robots", url: "/robots.txt", wrapper: "metadata-robots.ts" },
    MetaFileConvention { stem: "manifest", url: "/manifest.webmanifest", wrapper: "metadata-manifest.ts" },
];

/// Synthesize route handlers for the `sitemap`/`robots`/`manifest` file conventions
/// present at the app root. For each, write the shared serializer helper + a per-file
/// wrapper (importing the user's default export) under `shims_dir`, and return a
/// [`RouteHandler`] pointing at the wrapper. These flow unchanged through the existing
/// `ROUTE_HANDLERS` table + orchestrator `route` dispatch — no new server code path.
fn synthesize_metadata_file_handlers(
    app_dir: &Path,
    shims_dir: &Path,
) -> Result<Vec<RouteHandler>, String> {
    let mut handlers = Vec::new();
    let mut wrote_serializer = false;
    let serializer = shims_dir.join("metadata-serialize.ts");
    for conv in &META_FILE_CONVENTIONS {
        let Some(user_file) = first_existing(app_dir, conv.stem) else { continue };
        let user_file = user_file.canonicalize().unwrap_or(user_file);
        // `generateSitemaps` (multiple, id-partitioned sitemaps) is a distinct Next
        // feature (`/sitemap/[id].xml`) that this adapter does not synthesize. Fail
        // clearly rather than serve a single wrong `/sitemap.xml` (no silent stub).
        if conv.stem == "sitemap" {
            let src = std::fs::read_to_string(&user_file).unwrap_or_default();
            if exports_symbol(&strip_comments(&src), "generateSitemaps") {
                return Err(format!(
                    "diffpack next metadata: {} exports `generateSitemaps` (multiple id-partitioned sitemaps), which this adapter does not support yet. Use a single default-export sitemap() returning the full url array instead.",
                    user_file.display(),
                ));
            }
        }
        if !wrote_serializer {
            write_if_changed(&serializer, metadata_serialize_shim())?;
            wrote_serializer = true;
        }
        let serializer_canon = serializer.canonicalize().unwrap_or_else(|_| serializer.clone());
        let wrapper_path = shims_dir.join(conv.wrapper);
        let wrapper_src = metadata_file_wrapper(conv.stem, &user_file, &serializer_canon);
        write_if_changed(&wrapper_path, &wrapper_src)?;
        let wrapper_canon = wrapper_path.canonicalize().unwrap_or_else(|_| wrapper_path.clone());
        // The served URL is a single static path segment (e.g. `sitemap.xml`).
        let seg = conv.url.trim_start_matches('/').to_string();
        handlers.push(RouteHandler {
            url_path: conv.url.to_string(),
            segments: vec![Seg::Static(seg)],
            file: wrapper_canon,
            methods: vec!["GET".to_string()],
        });
    }
    Ok(handlers)
}

/// Scan the app ROOT for static metadata-image file conventions. Nested (segment-scoped)
/// images and code-based image generators (`opengraph-image.tsx` returning an
/// `ImageResponse`) are NOT supported here: each is reported as a hard error naming the
/// file (per the no-silent-drop rule) rather than being dropped from the head.
fn scan_metadata_images(app_dir: &Path) -> Result<Vec<MetaImage>, String> {
    // (stem, kind) in head-emit priority order. `icon` also matches `icon0`, `icon1`
    // in Next, but the base convention is a single `icon.*`; we support the base names.
    let families: [(&str, MetaImageKind); 5] = [
        ("favicon", MetaImageKind::Favicon),
        ("icon", MetaImageKind::Icon),
        ("apple-icon", MetaImageKind::AppleIcon),
        ("opengraph-image", MetaImageKind::OpengraphImage),
        ("twitter-image", MetaImageKind::TwitterImage),
    ];
    let mut images = Vec::new();
    for (stem, kind) in families {
        // A code-based generator at the app root (e.g. `opengraph-image.tsx`): the
        // dynamic-image path (satori/@vercel/og) is a heavy build-time-only capability
        // this adapter does not implement. Fail clearly, pointing at a static file.
        if kind != MetaImageKind::Favicon
            && let Some(generator) = first_existing(app_dir, stem)
        {
            return Err(format!(
                "diffpack next metadata: {} is a code-based image generator (returns an ImageResponse), which this adapter does not support (it needs the heavy @vercel/og dependency and is kept out of the request path). Provide a static {stem}.png/.jpg/.svg instead.",
                generator.display(),
            ));
        }
        let Some(src) = first_existing_ext(app_dir, stem, &METADATA_IMAGE_EXTS) else { continue };
        let ext = src
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_ascii_lowercase();
        let source = src.canonicalize().unwrap_or(src);
        let served = format!("/{stem}.{ext}");
        images.push(MetaImage {
            kind,
            source,
            served,
            mime: metadata_image_mime(&ext),
        });
    }
    Ok(images)
}

fn discover_routes_dir(
    app_dir: &Path,
    dir: &Path,
    root_layout: Option<&Path>,
    level_chain: &mut Vec<Level>,
    routes: &mut Vec<Route>,
) -> Result<(), String> {
    let canon = |p: PathBuf| p.canonicalize().unwrap_or(p);
    // The number of URL segments consumed from app/ down to and including this dir, so a
    // parallel @slot hosted here matches the below-level URL parts (parts.slice(offset)).
    let part_offset = dir
        .strip_prefix(app_dir)
        .unwrap_or(Path::new(""))
        .components()
        .filter_map(|c| c.as_os_str().to_str())
        .filter(|c| matches!(parse_segment(c), SegParse::Seg(_)))
        .count();
    // This directory's own conventions form a level for it and its descendants, plus any
    // parallel @slot subtrees it hosts (passed to its layout as named props).
    level_chain.push(Level {
        layout: first_existing(dir, "layout").map(canon),
        loading: first_existing(dir, "loading").map(canon),
        error: first_existing(dir, "error").map(canon),
        template: first_existing(dir, "template").map(canon),
        slots: discover_slots(dir)?,
        part_offset,
    });

    if let Some(page) = first_existing_page(dir) {
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
            // Route-segment-config exports beyond the static/dynamic set: `runtime =
            // "edge"` hard-errors here (no edge runtime); the advisory ones WARN. Done
            // before the route is recorded so an unsupported route never reaches codegen.
            validate_segment_config(&url_path, &cfg)?;
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
            let revalidate_seconds = parse_revalidate(&cfg);
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
                revalidate_seconds,
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
                // Skip dotdirs, the adapter output, and `@slot` dirs (discovered as
                // parallel-route slots on this level, not as primary routes).
                .map(|n| !n.starts_with('.') && n != ADAPTER_DIR && !n.starts_with('@'))
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

/// Discover the parallel-route `@slot` subtrees a directory hosts (each becomes a named
/// prop on that directory's layout). A `@slot` dir yields a [`Slot`] with its matchable
/// routes (relative to the slot dir) and an optional `default.tsx`.
fn discover_slots(dir: &Path) -> Result<Vec<Slot>, String> {
    let read = match std::fs::read_dir(dir) {
        Ok(read) => read,
        Err(_) => return Ok(Vec::new()),
    };
    let mut slot_dirs: Vec<PathBuf> = read
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.is_dir())
        .filter(|p| p.file_name().and_then(|n| n.to_str()).is_some_and(|n| n.starts_with('@')))
        .collect();
    slot_dirs.sort();
    let canon = |p: PathBuf| p.canonicalize().unwrap_or(p);
    let mut slots = Vec::new();
    for slot_dir in slot_dirs {
        let name = slot_dir
            .file_name()
            .and_then(|n| n.to_str())
            .and_then(slot_name)
            .unwrap_or_default();
        let mut routes = Vec::new();
        let mut level_chain = Vec::new();
        discover_slot_dir(&slot_dir, &slot_dir, &mut level_chain, &mut routes)?;
        slots.push(Slot {
            name,
            routes,
            default: first_existing(&slot_dir, "default").map(canon),
        });
    }
    Ok(slots)
}

/// Walk a `@slot` subtree collecting its matchable [`SlotRoute`]s (segments relative to
/// `slot_root`). Intercepting-route markers `(.)`/`(..)`/`(...)` and nested `@slots` do
/// NOT match on a hard render, so they are skipped here (soft-nav intercepts are a
/// separate concern).
fn discover_slot_dir(
    slot_root: &Path,
    dir: &Path,
    level_chain: &mut Vec<Level>,
    routes: &mut Vec<SlotRoute>,
) -> Result<(), String> {
    let canon = |p: PathBuf| p.canonicalize().unwrap_or(p);
    level_chain.push(Level {
        layout: first_existing(dir, "layout").map(canon),
        loading: first_existing(dir, "loading").map(canon),
        error: first_existing(dir, "error").map(canon),
        template: first_existing(dir, "template").map(canon),
        slots: Vec::new(),
        part_offset: 0,
    });

    if let Some(page) = first_existing_page(dir) {
        let rel = dir.strip_prefix(slot_root).unwrap_or(Path::new(""));
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
            routes.push(SlotRoute {
                segments,
                page: page.canonicalize().unwrap_or(page),
                levels: level_chain.clone(),
            });
        }
    }

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
            p.file_name().and_then(|n| n.to_str()).is_some_and(|n| {
                !n.starts_with('.') && n != ADAPTER_DIR && !n.starts_with('@') && !is_intercept_marker(n)
            })
        })
        .collect();
    children.sort();
    for child in children {
        discover_slot_dir(slot_root, &child, level_chain, routes)?;
    }
    level_chain.pop();
    Ok(())
}

/// Where an intercept marker resolves the target relative to the marker's own URL level:
/// `(.)` same level, `(..)` one up (`(..)(..)` n up), `(...)` from the app root.
enum InterceptBase {
    Up(usize),
    Root,
}

/// Discover intercepting routes: a directory whose name is an intercept marker
/// `(.)`/`(..)`/`(...)` followed by a target path (e.g. `@modal/(..)photo/[id]`). The
/// target URL is the marker's URL level (adjusted by its `..` depth) plus the marker
/// path, so an overlay defined at `app/gallery/@modal/(.)photo/[id]` intercepts
/// `/gallery/photo/[id]`. Slot-internal layouts are captured so the overlay is wrapped
/// exactly as its slot would render it.
fn discover_intercepts(app_dir: &Path) -> Result<Vec<Intercept>, String> {
    let mut out = Vec::new();
    let mut base_segments = Vec::new();
    collect_intercepts(app_dir, &mut base_segments, &mut out)?;
    Ok(out)
}

/// Walk the tree tracking `base_segments` (the URL segments from app/ to the current dir;
/// `@slots` and `(groups)` add none). On an intercept-marker dir, resolve the target base
/// and collect the overlay subtree.
fn collect_intercepts(dir: &Path, base_segments: &mut Vec<Seg>, out: &mut Vec<Intercept>) -> Result<(), String> {
    let read = match std::fs::read_dir(dir) {
        Ok(read) => read,
        Err(_) => return Ok(()),
    };
    let mut children: Vec<PathBuf> = read.flatten().map(|e| e.path()).filter(|p| p.is_dir()).collect();
    children.sort();
    for path in children {
        let name = match path.file_name().and_then(|n| n.to_str()) {
            Some(n) => n,
            None => continue,
        };
        if name.starts_with('.') || name == ADAPTER_DIR {
            continue;
        }
        if is_intercept_marker(name) {
            let base = match marker_base(name) {
                InterceptBase::Root => Vec::new(),
                InterceptBase::Up(n) => base_segments[..base_segments.len().saturating_sub(n)].to_vec(),
            };
            let mut level_chain = Vec::new();
            collect_intercept_dir(&path, base, true, out, &mut level_chain)?;
            continue;
        }
        // Track URL segments through normal dirs; @slots and groups add none.
        match parse_segment(name) {
            SegParse::Seg(seg) => {
                base_segments.push(seg);
                collect_intercepts(&path, base_segments, out)?;
                base_segments.pop();
            }
            SegParse::Group | SegParse::Skip => {
                collect_intercepts(&path, base_segments, out)?;
            }
        }
    }
    Ok(())
}

/// Walk an intercept-marker subtree collecting the overlay page + its levels. The
/// marker-root component's marker prefix is stripped for its own target segment; deeper
/// dirs contribute normal segments. `target_prefix` is the resolved base URL segments.
fn collect_intercept_dir(
    dir: &Path,
    target_prefix: Vec<Seg>,
    is_marker_root: bool,
    out: &mut Vec<Intercept>,
    level_chain: &mut Vec<Level>,
) -> Result<(), String> {
    let canon = |p: PathBuf| p.canonicalize().unwrap_or(p);
    level_chain.push(Level {
        layout: first_existing(dir, "layout").map(canon),
        loading: first_existing(dir, "loading").map(canon),
        error: first_existing(dir, "error").map(canon),
        template: first_existing(dir, "template").map(canon),
        slots: Vec::new(),
        part_offset: 0,
    });
    let name = dir.file_name().and_then(|n| n.to_str()).unwrap_or("");
    let comp = if is_marker_root { strip_intercept_marker(name) } else { name };
    let mut target = target_prefix.clone();
    match parse_segment(comp) {
        SegParse::Seg(seg) => target.push(seg),
        SegParse::Group | SegParse::Skip => {}
    }
    if let Some(page) = first_existing_page(dir) {
        out.push(Intercept {
            target_segments: target.clone(),
            page: page.canonicalize().unwrap_or(page),
            levels: level_chain.clone(),
        });
    }
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
        .filter(|p| p.file_name().and_then(|n| n.to_str()).is_some_and(|n| !n.starts_with('.') && n != ADAPTER_DIR))
        .collect();
    children.sort();
    for child in children {
        collect_intercept_dir(&child, target.clone(), false, out, level_chain)?;
    }
    level_chain.pop();
    Ok(())
}

/// The base level an intercept marker resolves against: `(.)` same, `(..)` one up
/// (`(..)(..)` n up), `(...)` the app root.
fn marker_base(comp: &str) -> InterceptBase {
    if comp.starts_with("(...)") {
        return InterceptBase::Root;
    }
    let mut s = comp;
    let mut up = 0;
    while let Some(rest) = s.strip_prefix("(..)") {
        up += 1;
        s = rest;
    }
    InterceptBase::Up(up)
}

/// Strip an intercept marker prefix from a directory-name component: `(..)photo` ->
/// `photo`, `(.)a` -> `a`, `(...)b` -> `b`.
fn strip_intercept_marker(comp: &str) -> &str {
    if let Some(rest) = comp.strip_prefix("(...)") {
        return rest;
    }
    let mut s = comp;
    while let Some(rest) = s.strip_prefix("(..)").or_else(|| s.strip_prefix("(.)")) {
        s = rest;
    }
    s
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

    // Evaluate `next.config.*` ONCE for this pass (redirects/rewrites/headers + images +
    // the basePath/assetPrefix/trailingSlash/i18n routing surface). The result feeds the
    // image-config module, the baked asset/base-path prefixes below, and (on the
    // react-server pass) the config manifest — so node is spawned a single time instead of
    // once per consumer (a build-time win over the previous two spawns).
    let next_config = run_next_config_eval(&root);
    let routing = Routing::from_eval(next_config.as_ref());
    let base_path = routing.base_path.clone();
    let asset_base = routing.asset_base();

    let adapter_dir = root.join(ADAPTER_DIR);
    let shims_dir = adapter_dir.join("shims");
    std::fs::create_dir_all(&shims_dir)
        .map_err(|error| format!("cannot create {}: {error}", shims_dir.display()))?;

    // --- app-router route table (every route + its nested layout/boundary chain) --
    let _ = &page; // detection anchor; the full route set comes from discovery.
    let layout_abs = layout.as_ref().map(|l| l.canonicalize().unwrap_or_else(|_| l.clone()));
    let mut discovered = discover_routes(&app_dir, layout_abs.as_deref())?;
    // Metadata FILE conventions (`app/sitemap.ts`/`robots.ts`/`manifest.ts`): synthesize
    // a wrapper + route-handler entry for each present one, so `/sitemap.xml`,
    // `/robots.txt`, `/manifest.webmanifest` are served through the SAME route-handler
    // dispatch as any `route.ts` endpoint. Distinct literal URLs, so appending (no
    // re-sort) preserves the most-specific-first invariant of the handler table.
    discovered
        .handlers
        .extend(synthesize_metadata_file_handlers(&app_dir, &shims_dir)?);

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
    write_if_changed(&request_context, &request_context_module())?;
    let request_context_canon = request_context
        .canonicalize()
        .unwrap_or_else(|_| request_context.clone());
    let hooks_context = adapter_dir.join("hooks-context.ts");
    write_if_changed(&hooks_context, hooks_context_module())?;
    let hooks_context_canon = hooks_context
        .canonicalize()
        .unwrap_or_else(|_| hooks_context.clone());

    // The SEGMENT boundary island (useSelectedLayoutSegment(s)): like the error boundary
    // it is a `"use client"` island wrapped around each layout in the react-server render,
    // BUNDLED + REGISTERED in the client + ssr graphs so it PROVIDES SelectedSegmentContext
    // there, and a client reference in the react-server graph. It imports the shared
    // hooks-context, so write it after that file exists. Keyed by its canonical path so the
    // manifest ids match across graphs.
    let segment_boundary = adapter_dir.join("segment-boundary.tsx");
    write_if_changed(&segment_boundary, &segment_boundary_module(&hooks_context_canon))?;
    let segment_boundary_canon = segment_boundary
        .canonicalize()
        .unwrap_or_else(|_| segment_boundary.clone());
    if !islands.contains(&segment_boundary_canon) {
        islands.push(segment_boundary_canon.clone());
    }

    write_if_changed(&adapter_dir.join("lazy.js"), lazy_module())?;
    // Middleware: `middleware.{ts,js}` at the project root or under `src/`.
    let middleware = first_existing(&root, "middleware")
        .or_else(|| first_existing(&root.join("src"), "middleware"))
        .map(|p| p.canonicalize().unwrap_or(p));
    write_if_changed(
        &adapter_dir.join("rsc-entry.tsx"),
        &rsc_entry_module(
            &discovered,
            &font_css,
            has_css,
            &error_boundary_canon,
            &segment_boundary_canon,
            &request_context_canon,
            middleware.as_deref(),
            &asset_base,
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
    write_if_changed(&link_shim, &next_link_shim(&base_path))?;
    let link_canon = link_shim.canonicalize().unwrap_or_else(|_| link_shim.clone());
    if !islands.contains(&link_canon) {
        islands.push(link_canon);
    }
    write_if_changed(
        &adapter_dir.join("server.tsx"),
        &ssr_entry_module(&adapter_dir, &islands, &hooks_context_canon, &asset_base),
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
    // The next.config `images` block (remote allow-list + loader), bundled into every
    // graph so the shim can allow/deny remote hosts and drive a custom/built-in loader.
    write_if_changed(
        &adapter_dir.join("image-config.ts"),
        &image_config_module(&images_from_eval(next_config.as_ref())),
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
    write_if_changed(
        &shims_dir.join("cache.ts"),
        &next_cache_shim(&request_context_canon),
    )?;
    write_if_changed(&shims_dir.join("server.ts"), next_server_shim())?;
    write_if_changed(&shims_dir.join("dynamic.ts"), next_dynamic_shim())?;

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
        alias("next/cache", &shims_dir.join("cache.ts")),
        alias("next/server", &shims_dir.join("server.ts")),
        alias("next/dynamic", &shims_dir.join("dynamic.ts")),
    ];

    // React's dev/prod dispatch define. Production bundles the production React
    // (small, no dev warnings); DEV bundles the development React whose renderer
    // exposes the Fast Refresh hook the island HMR path needs.
    let node_env = if dev { "\"development\"" } else { "\"production\"" };
    let defines = vec![(
        "process.env.NODE_ENV".to_string(),
        node_env.to_string(),
    )];

    // Evaluate `next.config` once (on the react-server pass) into the routing-rules
    // manifest the orchestrator applies (redirects/rewrites/headers). Best-effort: a
    // failing/absent config yields empty rules, never a build failure.
    if environment == "react-server" {
        write_next_config_manifest(&root, next_config.as_ref());
    }

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
/// Evaluate `next.config.*` into `.diffpack-output/next-config-manifest.json` (the
/// redirects/rewrites/headers rules the orchestrator applies), via node + the app's
/// own jiti. Best-effort: no config, a config that throws, or missing node all yield
/// an empty-rules manifest — never a build error.
/// Evaluate `next.config.*` via node (the app's own jiti) into the full config JSON
/// (redirects/rewrites/headers/images). None when there is no config, node is missing,
/// or it fails — every caller then falls back to its own defaults (best-effort, never a
/// build error). stderr from the eval is surfaced.
fn run_next_config_eval(root: &Path) -> Option<serde_json::Value> {
    let config = ["next.config.ts", "next.config.mjs", "next.config.js", "next.config.cjs"]
        .iter()
        .map(|f| root.join(f))
        .find(|p| p.exists())?;
    let loader = std::env::temp_dir().join("diffpack-next-config-eval.mjs");
    std::fs::write(&loader, include_str!("../scripts/rsc/next-config-eval.mjs")).ok()?;
    let out = std::process::Command::new("node")
        .arg(&loader)
        .arg(&config)
        .current_dir(root)
        .output()
        .ok()?;
    if !out.stderr.is_empty() {
        eprintln!("[next.config] {}", String::from_utf8_lossy(&out.stderr).trim());
    }
    if !out.status.success() || out.stdout.is_empty() {
        return None;
    }
    serde_json::from_slice(&out.stdout).ok()
}

/// The next.config `images` block (remote-host allow-list + loader), defaulted to Next's
/// stock values when there is no config.
fn default_images_json() -> serde_json::Value {
    serde_json::json!({
        "deviceSizes": null, "imageSizes": null, "remotePatterns": [], "domains": [],
        "loader": "default", "loaderFile": null, "path": "/_next/image",
        "qualities": null, "unoptimized": false
    })
}

/// The `images` block from a single `run_next_config_eval` result, defaulted to Next's
/// stock values when there is no config (or it lacks an `images` key).
fn images_from_eval(eval: Option<&serde_json::Value>) -> serde_json::Value {
    eval.and_then(|v| v.get("images").cloned())
        .unwrap_or_else(default_images_json)
}

/// The next.config routing surface diffpack BAKES into generated modules at build time:
/// `basePath` (a URL prefix on every page link + asset) and `assetPrefix` (a CDN/path
/// prefix on static assets only). `trailingSlash`/`i18n` are purely request-time routing
/// decisions applied by the orchestrator from the manifest, so they are not carried here.
/// Both fields default to "" (no prefix) when there is no config.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Routing {
    pub base_path: String,
    pub asset_prefix: String,
}

impl Routing {
    fn from_eval(eval: Option<&serde_json::Value>) -> Self {
        let field = |key: &str| {
            eval.and_then(|v| v.get(key))
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string()
        };
        Routing {
            base_path: field("basePath"),
            asset_prefix: field("assetPrefix"),
        }
    }

    /// The prefix baked onto STATIC-asset URLs (`client.js`, `rsc.css`): assetPrefix then
    /// basePath. A page link gets `base_path` alone.
    fn asset_base(&self) -> String {
        format!("{}{}", self.asset_prefix, self.base_path)
    }
}

/// Generate `.diffpack-next/image-config.ts` — the next.config `images` the `next/image`
/// shim imports (it runs in the browser too, so it cannot read a JSON file at runtime).
/// A `loaderFile` is imported so its default export is bundled into every graph as
/// `loaderFn`.
fn image_config_module(images: &serde_json::Value) -> String {
    let field = |key: &str, default: &str| -> String {
        images
            .get(key)
            .filter(|v| !v.is_null())
            .map(|v| serde_json::to_string(v).unwrap_or_else(|_| default.to_string()))
            .unwrap_or_else(|| default.to_string())
    };
    let loader_file = images.get("loaderFile").and_then(|v| v.as_str());
    let mut out = String::from(
        "// GENERATED by diffpack next-adapter — the next.config `images` block the\n\
         // next/image shim reads (remote-host allow-list + loader).\n",
    );
    if let Some(lf) = loader_file {
        out.push_str(&format!("import __loaderFile from {};\n", js_str(lf)));
    }
    out.push_str("export default {\n");
    out.push_str(&format!("  deviceSizes: {},\n", field("deviceSizes", "null")));
    out.push_str(&format!("  imageSizes: {},\n", field("imageSizes", "null")));
    out.push_str(&format!("  remotePatterns: {},\n", field("remotePatterns", "[]")));
    out.push_str(&format!("  domains: {},\n", field("domains", "[]")));
    out.push_str(&format!("  loader: {},\n", field("loader", "\"default\"")));
    out.push_str(&format!("  path: {},\n", field("path", "\"/_next/image\"")));
    out.push_str(&format!("  qualities: {},\n", field("qualities", "null")));
    out.push_str(&format!("  unoptimized: {},\n", field("unoptimized", "false")));
    out.push_str(&format!(
        "  loaderFn: {},\n",
        if loader_file.is_some() { "__loaderFile" } else { "null" }
    ));
    out.push_str("};\n");
    out
}

/// The empty-config manifest: well-formed so the orchestrator's `routing` reader always
/// finds every field (no config, a config that throws, or missing node all land here).
const EMPTY_CONFIG_MANIFEST: &str =
    r#"{"redirects":[],"rewrites":[],"headers":[],"basePath":"","assetPrefix":"","trailingSlash":false,"i18n":null}"#;

/// Persist the single `run_next_config_eval` result to
/// `.diffpack-output/next-config-manifest.json` (the redirects/rewrites/headers rules +
/// the basePath/assetPrefix/trailingSlash/i18n routing surface the orchestrator applies).
/// No re-spawn of node: the caller already evaluated the config once for this pass.
fn write_next_config_manifest(root: &Path, eval: Option<&serde_json::Value>) {
    let output = root.join(".diffpack-output");
    let _ = std::fs::create_dir_all(&output);
    let manifest_path = output.join("next-config-manifest.json");
    match eval {
        Some(value) => {
            let _ = std::fs::write(&manifest_path, value.to_string());
        }
        None => {
            let _ = std::fs::write(&manifest_path, EMPTY_CONFIG_MANIFEST);
        }
    }
}

fn write_if_changed(path: &Path, contents: &str) -> Result<(), String> {
    if let Ok(existing) = std::fs::read_to_string(path)
        && existing == contents {
            return Ok(());
        }
    std::fs::write(path, contents)
        .map_err(|error| format!("cannot write {}: {error}", path.display()))
}

// --- generated module templates --------------------------------------------------

/// The shared serializer for the metadata FILE conventions. Pure functions turning a
/// Next `MetadataRoute.Sitemap` array / `MetadataRoute.Robots` object into the exact
/// sitemap XML / robots.txt text Next emits. Written once under `shims/` and imported
/// by the per-file wrappers. (`manifest` needs no helper — it is `JSON.stringify`.)
fn metadata_serialize_shim() -> &'static str {
    r##"// Generated by diffpack's next app-router adapter. Serializers for the metadata
// FILE conventions (sitemap.xml / robots.txt). Pure, no dependencies.
function xmlEscape(value) {
  return String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&apos;");
}

function isoDate(value) {
  if (value == null) return null;
  if (value instanceof Date) return value.toISOString();
  return String(value);
}

// A Next `MetadataRoute.Sitemap` (array of { url, lastModified?, changeFrequency?,
// priority?, alternates?: { languages }, images? }) -> sitemap XML.
export function serializeSitemap(entries) {
  if (!Array.isArray(entries)) {
    throw new Error("diffpack next metadata: sitemap() must return an array of entries, got " + typeof entries);
  }
  const hasAlternates = entries.some((e) => e && e.alternates && e.alternates.languages);
  const hasImages = entries.some((e) => e && e.images && e.images.length);
  let out = '<?xml version="1.0" encoding="UTF-8"?>\n';
  out += '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"';
  if (hasAlternates) out += ' xmlns:xhtml="http://www.w3.org/1999/xhtml"';
  if (hasImages) out += ' xmlns:image="http://www.google.com/schemas/sitemap-image/1.1"';
  out += ">\n";
  for (const entry of entries) {
    if (!entry || entry.url == null) {
      throw new Error("diffpack next metadata: every sitemap entry needs a `url`");
    }
    out += "<url>\n";
    out += "<loc>" + xmlEscape(entry.url) + "</loc>\n";
    const lastmod = isoDate(entry.lastModified);
    if (lastmod != null) out += "<lastmod>" + xmlEscape(lastmod) + "</lastmod>\n";
    if (entry.changeFrequency != null) out += "<changefreq>" + xmlEscape(entry.changeFrequency) + "</changefreq>\n";
    if (entry.priority != null) out += "<priority>" + xmlEscape(entry.priority) + "</priority>\n";
    if (entry.alternates && entry.alternates.languages) {
      for (const [lang, href] of Object.entries(entry.alternates.languages)) {
        out += '<xhtml:link rel="alternate" hreflang="' + xmlEscape(lang) + '" href="' + xmlEscape(href) + '" />\n';
      }
    }
    if (entry.images) {
      for (const img of entry.images) {
        out += "<image:image>\n<image:loc>" + xmlEscape(img) + "</image:loc>\n</image:image>\n";
      }
    }
    out += "</url>\n";
  }
  out += "</urlset>\n";
  return out;
}

// A Next `MetadataRoute.Robots` ({ rules, sitemap?, host? }) -> robots.txt text.
export function serializeRobots(data) {
  if (!data || typeof data !== "object") {
    throw new Error("diffpack next metadata: robots() must return an object, got " + typeof data);
  }
  const lines = [];
  const emitRule = (rule) => {
    const agents = rule.userAgent == null ? ["*"] : Array.isArray(rule.userAgent) ? rule.userAgent : [rule.userAgent];
    for (const agent of agents) lines.push("User-Agent: " + agent);
    const emitPaths = (label, value) => {
      if (value == null) return;
      const arr = Array.isArray(value) ? value : [value];
      for (const p of arr) lines.push(label + ": " + p);
    };
    emitPaths("Allow", rule.allow);
    emitPaths("Disallow", rule.disallow);
    if (rule.crawlDelay != null) lines.push("Crawl-delay: " + rule.crawlDelay);
  };
  const rules = data.rules == null ? [] : Array.isArray(data.rules) ? data.rules : [data.rules];
  rules.forEach((rule, i) => {
    if (i > 0) lines.push("");
    emitRule(rule);
  });
  if (data.host != null) lines.push("Host: " + data.host);
  if (data.sitemap != null) {
    const maps = Array.isArray(data.sitemap) ? data.sitemap : [data.sitemap];
    for (const m of maps) lines.push("Sitemap: " + m);
  }
  return lines.join("\n") + "\n";
}
"##
}

/// A per-file wrapper for a metadata FILE convention: it imports the user file's default
/// export, calls it (awaiting — the export may be async), serializes the result, and
/// returns a `Response` with the right content-type. Because it exports `GET`, it plugs
/// straight into the existing route-handler dispatch (`H<i>.GET`).
fn metadata_file_wrapper(stem: &str, user_file: &Path, serializer: &Path) -> String {
    let user = js_str(&user_file.to_string_lossy());
    let ser = js_str(&serializer.to_string_lossy());
    let (import, body) = match stem {
        "sitemap" => (
            format!("import handler from {user};\nimport {{ serializeSitemap }} from {ser};\n"),
            "  const body = serializeSitemap(await handler());\n  return new Response(body, { status: 200, headers: { \"content-type\": \"application/xml\" } });".to_string(),
        ),
        "robots" => (
            format!("import handler from {user};\nimport {{ serializeRobots }} from {ser};\n"),
            "  const body = serializeRobots(await handler());\n  return new Response(body, { status: 200, headers: { \"content-type\": \"text/plain\" } });".to_string(),
        ),
        "manifest" => (
            format!("import handler from {user};\n"),
            "  const body = JSON.stringify(await handler());\n  return new Response(body, { status: 200, headers: { \"content-type\": \"application/manifest+json\" } });".to_string(),
        ),
        other => unreachable!("metadata_file_wrapper called with unknown stem {other}"),
    };
    format!(
        "// Generated by diffpack's next app-router adapter. Serves the `{stem}` metadata\n\
         // FILE convention through the standard route-handler dispatch.\n\
         {import}\nexport async function GET() {{\n{body}\n}}\n",
    )
}

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

/// The `SEGMENT_BOUNDARY` island (`segment-boundary.tsx`). A `"use client"` boundary the
/// react-server render wraps around EACH layout element, carrying that layout's active
/// child URL segments (`parts.slice(level.slotBase)`) as a plain serializable prop. Like
/// the error boundary it is pinned as an island, so it renders (and PROVIDES
/// `SelectedSegmentContext`) in BOTH the SSR and client graphs; nested layouts read their
/// own slice via context nesting (innermost wins), which is exactly what
/// `useSelectedLayoutSegment(s)` returns. React context does not cross the RSC/SSR
/// boundary and Server Components cannot call `useContext`, so this client island is the
/// mechanism that carries the value across (the same reason params/pathname are provided
/// in the SSR+client entries, not the react-server render).
fn segment_boundary_module(hooks_context: &Path) -> String {
    let hooks_import = js_str(&hooks_context.to_string_lossy());
    format!(
        r#""use client";
// Generated by diffpack's next app-router adapter — the client SEGMENT boundary that
// provides SelectedSegmentContext to each layout so useSelectedLayoutSegment(s) resolve.
import {{ createElement }} from "react";
import {{ SelectedSegmentContext }} from {hooks_import};

export default function SegmentBoundary(props) {{
  return createElement(SelectedSegmentContext.Provider, {{ value: props.segments }}, props.children);
}}
"#,
    )
}

/// The per-request context module (`request-context.ts`). Holds the ONE
/// `AsyncLocalStorage` instance the react-server render establishes (`requestAls.run`)
/// and `next/headers` reads (`requestAls.getStore()`). Because rsc-entry and the
/// `next/headers` shim are bundled into the SAME react-server graph, they share this
/// single instance (Next's `workUnitAsyncStorage` analogue). It lands ONLY in the
/// react-server graph — Server Components are the only code that reads the request
/// context, and only they run there.
fn request_context_module() -> String {
    // The draft-mode signing secret is baked ONCE per build into the react-server graph
    // so draftMode()'s HMAC sign/verify runs entirely inside the worker (the orchestrator
    // never sees the secret; it only forwards the cookie header it already forwards). All
    // workers of one deployment import this same module source, so a token one worker
    // signs verifies on any other. A fresh secret per build invalidates outstanding
    // `__prerender_bypass` cookies across redeploys, matching Next's preview-mode behavior.
    let secret = draft_secret();
    format!(
        "// Generated by diffpack's next app-router adapter — the per-request AsyncLocalStorage\n\
         // that carries {{ url, headers, cookieHeader, params, responseCookies }} from the HTTP\n\
         // request into async Server Components (next/headers cookies()/headers()/draftMode()).\n\
         // One shared instance across the react-server graph (rsc-entry establishes it; the\n\
         // next/headers shim reads it). DRAFT_SECRET signs the draftMode bypass cookie.\n\
         import {{ AsyncLocalStorage }} from \"node:async_hooks\";\n\
         export const requestAls = new AsyncLocalStorage();\n\
         export const DRAFT_SECRET = {};\n",
        js_str(&secret),
    )
}

/// A random 32-hex draft-mode signing secret, generated once per process (so every
/// call within a single build returns the SAME value, keeping `write_if_changed` a
/// no-op for the rest of the build) but fresh across separate build invocations.
fn draft_secret() -> String {
    use std::sync::OnceLock;
    static SECRET: OnceLock<String> = OnceLock::new();
    SECRET
        .get_or_init(|| {
            let mut buf = [0u8; 16];
            // The OS CSPRNG on macOS/Linux. Read-failure falls back to a time+pid mix so a
            // build never hard-fails on an exotic platform without /dev/urandom.
            if let Ok(mut f) = std::fs::File::open("/dev/urandom") {
                use std::io::Read;
                if f.read_exact(&mut buf).is_ok() {
                    return buf.iter().map(|b| format!("{b:02x}")).collect();
                }
            }
            use std::hash::{Hash, Hasher};
            let mut acc = String::new();
            let mut seed = std::process::id() as u64;
            for _ in 0..2 {
                let mut h = std::collections::hash_map::DefaultHasher::new();
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_nanos())
                    .unwrap_or(0)
                    .hash(&mut h);
                seed.hash(&mut h);
                let v = h.finish();
                seed = seed.wrapping_add(v).rotate_left(17);
                acc.push_str(&format!("{v:016x}"));
            }
            acc
        })
        .clone()
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
     export const SearchParamsContext = createContext(\"\");\n\
     // The active URL segments below the nearest layout (useSelectedLayoutSegment(s)).\n\
     // Provided per layout level by the SEGMENT_BOUNDARY island so a nested layout reads\n\
     // its own slice via context nesting (innermost wins). null outside any layout.\n\
     export const SelectedSegmentContext = createContext(null);\n\
     // useServerInsertedHTML: the SSR entry provides a per-request `push(callback)` here;\n\
     // on the client it stays null (no server HTML to flush) so the hook is a no-op.\n\
     export const ServerInsertedHTMLContext = createContext(null);\n"
}

/// The react-server render/action entry (Target::ReactServer). Builds the app's
/// ROUTE TABLE (every static route + its nested layout chain + metadata), matches a
/// requested pathname, composes `<Layout0>…<LayoutN>[head, <Page/>]` for the matched
/// route, and renders it to a flight stream (`render <pathname>` op), or dispatches a
/// server action (`action` op). The orchestrator spawns this in its own child so its
/// react-server React never mixes with the SSR/browser React.
#[allow(clippy::too_many_arguments)]
fn rsc_entry_module(
    disc: &Discovered,
    font_css: &str,
    has_css: bool,
    error_boundary: &Path,
    segment_boundary: &Path,
    request_context: &Path,
    middleware: Option<&Path>,
    asset_base: &str,
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
    // Serialize a level chain to JS, including each level's parallel @slots (recursively,
    // since a slot route carries its own level chain).
    fn emit_levels(modules: &mut Vec<String>, levels: &[Level]) -> String {
        let mut out = String::new();
        for level in levels {
            let layout_id = opt_id(modules, &level.layout);
            let loading_id = opt_id(modules, &level.loading);
            let error_id = opt_id(modules, &level.error);
            let template_id = opt_id(modules, &level.template);
            let slots_js = emit_slots(modules, &level.slots);
            out.push_str(&format!(
                "{{ layout: {layout_id}, loading: {loading_id}, error: {error_id}, template: {template_id}, slotBase: {}, slots: [{slots_js}] }}, ",
                level.part_offset,
            ));
        }
        out
    }
    fn emit_slots(modules: &mut Vec<String>, slots: &[Slot]) -> String {
        let mut out = String::new();
        for slot in slots {
            let default_id = opt_id(modules, &slot.default);
            let mut routes_js = String::new();
            for sr in &slot.routes {
                let page_id = format!("M{}", intern(modules, &sr.page));
                let levels_js = emit_levels(modules, &sr.levels);
                routes_js.push_str(&format!(
                    "{{ segments: {}, page: {page_id}, levels: [{levels_js}] }}, ",
                    segments_js(&sr.segments),
                ));
            }
            out.push_str(&format!(
                "{{ name: {}, default: {default_id}, routes: [{routes_js}] }}, ",
                js_str(&slot.name),
            ));
        }
        out
    }

    let error_boundary_id = format!("M{}", intern(&mut modules, error_boundary));
    let segment_boundary_id = format!("M{}", intern(&mut modules, segment_boundary));

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
    // A layout/page's module NAMESPACE (`NS<i>`) — its `metadata`/`generateMetadata`/
    // `viewport`/`generateViewport` named exports are read at render time. `null` when
    // the file exports no metadata (so the metadata chain skips it cheaply).
    fn meta_ns(namespaces: &mut Vec<String>, path: &Option<PathBuf>) -> String {
        match path {
            Some(p) if module_exports_metadata(p) => format!("NS{}", intern_ns(namespaces, p)),
            _ => "null".to_string(),
        }
    }
    let mut static_param_entries = String::new();

    let mut route_entries = String::new();
    for route in &disc.routes {
        let page_id = format!("M{}", intern(&mut modules, &route.page));
        let levels_js = emit_levels(&mut modules, &route.levels);
        // Metadata chain (root→leaf layouts) + the page's own metadata namespace, walked
        // at render time to resolve+merge the document <head> (title templates, openGraph,
        // twitter, robots, icons, alternates, viewport, …).
        let meta_chain = route
            .levels
            .iter()
            .map(|lvl| meta_ns(&mut namespaces, &lvl.layout))
            .collect::<Vec<_>>()
            .join(", ");
        let page_meta = meta_ns(&mut namespaces, &Some(route.page.clone()));
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
            "  {{ path: {}, segments: {}, page: {page_id}, levels: [{levels_js}], metaChain: [{meta_chain}], pageMeta: {page_meta}, title: {title}, description: {description}, kind: {}, hasGenerateStaticParams: {}, dynamicParams: {} }},\n",
            js_str(&route.url_path),
            segments_js(&route.segments),
            js_str(route.kind.as_str()),
            route.has_generate_static_params,
            route.dynamic_params,
        ));
    }

    // Intercepting routes: on a SOFT navigation to a matching target, the overlay page
    // renders instead of the full document (the client masks the URL + keeps the current
    // page mounted). One entry per `@slot/(.)…` intercept.
    let mut intercept_entries = String::new();
    for ic in &disc.intercepts {
        let page_id = format!("M{}", intern(&mut modules, &ic.page));
        let levels_js = emit_levels(&mut modules, &ic.levels);
        intercept_entries.push_str(&format!(
            "  {{ segments: {}, page: {page_id}, levels: [{levels_js}] }},\n",
            segments_js(&ic.target_segments),
        ));
    }

    // Route handlers (`route.ts` HTTP endpoints): namespace-import each file (so every
    // exported method is reachable) and build the ROUTE_HANDLERS match table.
    let mut handler_namespaces: Vec<String> = Vec::new();
    let mut handler_entries = String::new();
    for handler in &disc.handlers {
        let key = handler.file.to_string_lossy().into_owned();
        let hidx = handler_namespaces.iter().position(|m| m == &key).unwrap_or_else(|| {
            handler_namespaces.push(key);
            handler_namespaces.len() - 1
        });
        let methods_js = handler
            .methods
            .iter()
            .map(|m| format!("{m}: H{hidx}.{m}"))
            .collect::<Vec<_>>()
            .join(", ");
        handler_entries.push_str(&format!(
            "  {{ path: {}, segments: {}, methods: {{ {methods_js} }} }},\n",
            js_str(&handler.url_path),
            segments_js(&handler.segments),
        ));
    }
    let handler_imports: String = handler_namespaces
        .iter()
        .enumerate()
        .map(|(i, s)| format!("import * as H{i} from {};\n", js_str(s)))
        .collect();

    // Middleware: namespace-import it (named `middleware` or default export) so
    // `runMiddleware` can invoke it; `null` when the app has none.
    let (middleware_import, middleware_const) = match middleware {
        Some(path) => (
            format!("import * as __mw from {};\n", js_str(&path.to_string_lossy())),
            "const MIDDLEWARE = __mw.middleware || __mw.default || null;".to_string(),
        ),
        None => (String::new(), "const MIDDLEWARE = null;".to_string()),
    };

    let root_layout_id = opt_id(&mut modules, &disc.root_layout);
    let app_not_found_id = opt_id(&mut modules, &disc.app_not_found);
    // `global-error.tsx`: the app-root boundary owning <html>. `"use client"` (like
    // error.tsx) so it is registered as an island; here it is a default client
    // reference, wrapped OUTSIDE the whole document tree in documentTree.
    let global_error_id = opt_id(&mut modules, &disc.global_error);
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
            js_str(&format!("{asset_base}{RSC_CSS_URL}"))
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
    // Static metadata IMAGE conventions (app/icon.png, app/favicon.ico, ...): head
    // elements built once at BUILD time and pushed for every route (the files are
    // copied to public/ by the client build). Zero per-request cost.
    let mut meta_image_push = String::new();
    for img in &disc.meta_images {
        let href = js_str(&img.served);
        let ty = js_str(img.mime);
        let el = match img.kind {
            MetaImageKind::Favicon => format!(
                "  items.push(createElement(\"link\", {{ rel: \"icon\", href: {href}, type: {ty}, sizes: \"any\" }}));\n"
            ),
            MetaImageKind::Icon => format!(
                "  items.push(createElement(\"link\", {{ rel: \"icon\", href: {href}, type: {ty} }}));\n"
            ),
            MetaImageKind::AppleIcon => format!(
                "  items.push(createElement(\"link\", {{ rel: \"apple-touch-icon\", href: {href}, type: {ty} }}));\n"
            ),
            MetaImageKind::OpengraphImage => format!(
                "  items.push(createElement(\"meta\", {{ property: \"og:image\", content: {href} }}));\n"
            ),
            MetaImageKind::TwitterImage => format!(
                "  items.push(createElement(\"meta\", {{ name: \"twitter:image\", content: {href} }}));\n"
            ),
        };
        meta_image_push.push_str(&el);
    }

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
import {{ NextRequest }} from "next/server";
{request_context_import}{imports}{ns_imports}{handler_imports}{middleware_import}
{middleware_const}
{font_const}const ROUTES = [
{route_entries}];
// Intercepting routes: a soft-nav to a matching target renders the overlay page.
const INTERCEPTS = [
{intercept_entries}];
// Route handlers (`route.ts` HTTP endpoints): each entry's `methods` maps an HTTP
// method to its exported handler function; `handleRoute` matches a request path here.
const ROUTE_HANDLERS = [
{handler_entries}];
// Ssg routes (a dynamic segment with generateStaticParams) → their module namespace,
// so the `staticparams` op can enumerate concrete param sets at build time.
const STATIC_PARAM_ROUTES = {{
{static_param_entries}}};
const ROOT_LAYOUT = {root_layout_id};
const APP_NOT_FOUND = {app_not_found_id};
const ERROR_BOUNDARY = {error_boundary_id};
// The client SEGMENT boundary wrapped around each layout, providing SelectedSegmentContext
// (the layout's active child URL segments) so useSelectedLayoutSegment(s) resolve.
const SEGMENT_BOUNDARY = {segment_boundary_id};
// The app-root global-error boundary (owns <html>), or null when the app has none.
const GLOBAL_ERROR = {global_error_id};
const ROOT_META = {{ title: {root_title}, description: {root_description} }};

// The route's head elements (stylesheet + font + this route's metadata). React 19
// hoists these into <head> from anywhere in the tree.
function headItems(meta) {{
  const items = [];
{css_push}{font_push}{meta_image_push}  if (meta && meta.title) items.push(createElement("title", null, meta.title));
  if (meta && meta.description) items.push(createElement("meta", {{ name: "description", content: meta.description }}));
  return items;
}}

// --- Metadata API ------------------------------------------------------------------
// Resolve + merge metadata from the root layout down to the page (each may export a
// `metadata` object OR an async `generateMetadata`; likewise `viewport`/
// `generateViewport`), applying ancestor title templates, then render the <head> tags
// (React 19 hoists them into <head>). Runs at flight-render time so dynamic/async
// metadata works — no per-request cost beyond the render already happening.
async function resolveMetadata(route, params) {{
  const chain = [...(route.metaChain || []), route.pageMeta];
  const paramsP = Promise.resolve(params);
  const meta = {{}};
  let template = null; // an ancestor title.template applies to descendant string titles
  for (const ns of chain) {{
    if (!ns) continue;
    let m = null;
    if (typeof ns.generateMetadata === "function") {{
      m = await ns.generateMetadata({{ params: paramsP, searchParams: Promise.resolve({{}}) }}, Promise.resolve(meta));
    }} else if (ns.metadata) {{
      m = ns.metadata;
    }}
    if (m) template = mergeMetadata(meta, m, template);
    let vp = null;
    if (typeof ns.generateViewport === "function") vp = await ns.generateViewport({{ params: paramsP }});
    else if (ns.viewport) vp = ns.viewport;
    if (vp) {{
      meta.viewport = vp;
      if (vp.themeColor) meta.themeColor = vp.themeColor;
      if (vp.colorScheme) meta.colorScheme = vp.colorScheme;
    }}
  }}
  return meta;
}}

// Merge `m` into `acc` (mutating). Returns the title.template descendants inherit.
function mergeMetadata(acc, m, parentTemplate) {{
  let template = parentTemplate;
  if (m.title !== undefined) {{
    if (typeof m.title === "string") {{
      acc.title = parentTemplate ? parentTemplate.replace("%s", m.title) : m.title;
    }} else if (m.title && typeof m.title === "object") {{
      if (m.title.absolute != null) acc.title = m.title.absolute;
      else if (m.title.default != null) acc.title = m.title.default;
      if (m.title.template != null) template = m.title.template;
    }}
  }}
  for (const k of ["description", "applicationName", "generator", "referrer", "creator", "publisher", "category", "keywords", "authors", "robots", "icons", "openGraph", "twitter", "alternates", "metadataBase", "manifest", "themeColor", "colorScheme", "formatDetection", "verification"]) {{
    if (m[k] !== undefined) acc[k] = m[k];
  }}
  return template;
}}

// Build the <head> React elements from resolved metadata.
function metadataToHead(meta) {{
  const el = createElement;
  const items = [];
  const base = meta.metadataBase ? String(meta.metadataBase).replace(/\/$/, "") : "";
  const abs = (u) => {{
    if (u == null) return u;
    u = String(u);
    return /^https?:\/\//.test(u) || !base ? u : base + (u.startsWith("/") ? "" : "/") + u;
  }};
  let key = 0;
  const meta_ = (attrs) => items.push(el("meta", {{ key: "m" + key++, ...attrs }}));
  const link_ = (attrs) => items.push(el("link", {{ key: "l" + key++, ...attrs }}));
  if (meta.title != null) items.push(el("title", {{ key: "title" }}, String(meta.title)));
  if (meta.description != null) meta_({{ name: "description", content: String(meta.description) }});
  if (meta.keywords) meta_({{ name: "keywords", content: Array.isArray(meta.keywords) ? meta.keywords.join(", ") : String(meta.keywords) }});
  if (meta.applicationName) meta_({{ name: "application-name", content: meta.applicationName }});
  if (meta.generator) meta_({{ name: "generator", content: meta.generator }});
  if (meta.creator) meta_({{ name: "creator", content: meta.creator }});
  if (meta.publisher) meta_({{ name: "publisher", content: meta.publisher }});
  if (meta.authors) {{
    const arr = Array.isArray(meta.authors) ? meta.authors : [meta.authors];
    for (const a of arr) if (a && a.name) meta_({{ name: "author", content: a.name }});
  }}
  if (meta.robots) {{
    const r = meta.robots;
    const content = typeof r === "string" ? r : [r.index === false ? "noindex" : "index", r.follow === false ? "nofollow" : "follow", r.nocache ? "noarchive" : null].filter(Boolean).join(", ");
    meta_({{ name: "robots", content }});
  }}
  if (meta.alternates && meta.alternates.canonical) link_({{ rel: "canonical", href: abs(meta.alternates.canonical) }});
  if (meta.icons) {{
    const ic = meta.icons;
    const list = typeof ic === "string" ? [ic] : Array.isArray(ic) ? ic : ic.icon ? (Array.isArray(ic.icon) ? ic.icon : [ic.icon]) : [];
    for (const i of list) {{ const url = typeof i === "string" ? i : i.url; if (url) link_({{ rel: "icon", href: abs(url) }}); }}
  }}
  const og = meta.openGraph;
  if (og) {{
    const p = (prop, c) => c != null && items.push(el("meta", {{ key: "og" + key++, property: "og:" + prop, content: String(c) }}));
    p("title", og.title != null ? og.title : meta.title);
    p("description", og.description != null ? og.description : meta.description);
    p("url", abs(og.url));
    p("site_name", og.siteName);
    p("type", og.type || "website");
    const imgs = og.images ? (Array.isArray(og.images) ? og.images : [og.images]) : [];
    for (const im of imgs) p("image", abs(typeof im === "string" ? im : im.url));
  }}
  const tw = meta.twitter;
  if (tw) {{
    const p = (name, c) => c != null && meta_({{ name: "twitter:" + name, content: String(c) }});
    p("card", tw.card || "summary_large_image");
    p("title", tw.title != null ? tw.title : meta.title);
    p("description", tw.description != null ? tw.description : meta.description);
    const imgs = tw.images ? (Array.isArray(tw.images) ? tw.images : [tw.images]) : [];
    for (const im of imgs) p("image", abs(typeof im === "string" ? im : im.url));
  }}
  if (meta.manifest) link_({{ rel: "manifest", href: abs(meta.manifest) }});
  const vp = meta.viewport;
  if (vp) {{
    const content = typeof vp === "string" ? vp : Object.entries(vp)
      .filter(([k]) => ["width", "height", "initialScale", "minimumScale", "maximumScale", "userScalable", "viewportFit"].includes(k))
      .map(([k, v]) => (k === "initialScale" ? "initial-scale=" + v : k === "minimumScale" ? "minimum-scale=" + v : k === "maximumScale" ? "maximum-scale=" + v : k === "viewportFit" ? "viewport-fit=" + v : k + "=" + v))
      .join(", ");
    if (content) meta_({{ name: "viewport", content }});
  }}
  if (meta.themeColor) {{
    const tc = Array.isArray(meta.themeColor) ? meta.themeColor : [meta.themeColor];
    for (const t of tc) meta_({{ name: "theme-color", content: typeof t === "string" ? t : t.color }});
  }}
  if (meta.colorScheme) meta_({{ name: "color-scheme", content: meta.colorScheme }});
  return items;
}}

// Async Server Component: resolves the route metadata and renders the <head> tags. A
// build-time title/description fallback (from the ROUTE table) covers a route whose
// modules export no metadata.
async function MetadataHead({{ route, params }}) {{
  const meta = await resolveMetadata(route, params);
  if (meta.title == null && route.title != null) meta.title = route.title;
  if (meta.description == null && route.description != null) meta.description = route.description;
  return createElement(Fragment, null, ...metadataToHead(meta));
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

// Match a pathname to an intercepting route (soft-nav only). Returns
// `{{ intercept, params }}` or null.
function matchIntercept(pathname) {{
  const parts = pathname.split("/").filter(Boolean);
  for (const ic of INTERCEPTS) {{
    const params = matchSegments(ic.segments, parts);
    if (params) return {{ intercept: ic, params }};
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

// Wrap a page in its level chain (leaf→root: loading Suspense, error boundary,
// template, layout), sharing `params`. Used for a matched @slot route. `remountKey`
// (the pathname) keys each `template.tsx` so React remounts it on navigation (fresh
// state per URL) while same-position layouts keep their state — matching Next's
// Layout > Template > ErrorBoundary > Suspense(loading) > children order.
function composeLevels(page, levels, params, remountKey) {{
  const paramsPromise = Promise.resolve(params);
  let node = createElement(page, {{ params: paramsPromise, searchParams: Promise.resolve({{}}) }});
  for (let i = levels.length - 1; i >= 0; i -= 1) {{
    const level = levels[i];
    if (level.loading) node = createElement(Suspense, {{ fallback: createElement(level.loading) }}, node);
    if (level.error) {{
      const inner = level.loading ? node : createElement(Suspense, {{ fallback: null }}, node);
      node = createElement(ERROR_BOUNDARY, {{ fallback: level.error }}, inner);
    }}
    if (level.template) node = createElement(level.template, {{ key: remountKey }}, node);
    if (level.layout) node = createElement(level.layout, {{ params: paramsPromise }}, node);
  }}
  return node;
}}

// Match a level's parallel `@slots` against the below-level URL parts, returning a map
// of slot name -> React node (the matched slot route, else its default.tsx, else null).
// The layout receives these as named props alongside `children`.
function matchSlots(level, parts, outerParams, remountKey) {{
  const props = {{}};
  if (!level.slots || !level.slots.length) return props;
  const rest = parts.slice(level.slotBase);
  for (const slot of level.slots) {{
    let node = null;
    for (const sr of slot.routes) {{
      const sp = matchSegments(sr.segments, rest);
      if (sp) {{ node = composeLevels(sr.page, sr.levels, {{ ...outerParams, ...sp }}, remountKey); break; }}
    }}
    if (!node && slot.default) node = createElement(slot.default, {{ params: Promise.resolve(outerParams) }});
    props[slot.name] = node;
  }}
  return props;
}}

// Compose the matched route: the page (with its `params`/`searchParams` promises),
// wrapped level-by-level leaf→root — each level's loading (Suspense) then error
// (client ErrorBoundary) then layout — with the head items injected inside the root
// layout. A level hosting parallel `@slots` spreads the matched slot nodes as named
// props on its layout. On a SOFT navigation (`opts.softNav`) to an intercepting route's
// target, returns JUST the overlay tree (marked `intercept: true`) so the client renders
// it over the still-mounted current page. Returns `{{ tree, status, params, intercept }}`.
function documentTree(pathname, opts) {{
  if (opts && opts.softNav) {{
    const hit = matchIntercept(pathname);
    if (hit) {{
      return {{
        tree: composeLevels(hit.intercept.page, hit.intercept.levels, hit.params, pathname),
        status: 200,
        params: hit.params,
        intercept: true,
      }};
    }}
  }}
  const m = matchRoute(pathname);
  if (!m) return {{ tree: notFoundTree(), status: 404, params: {{}} }};
  const {{ route, params }} = m;
  const parts = pathname.split("/").filter(Boolean);
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
    // template.tsx re-mounts on navigation: keying it by pathname forces React to
    // unmount+remount it whenever the URL changes (fresh state), while same-position
    // layouts keep their state. It sits just inside this level's layout.
    if (level.template) node = createElement(level.template, {{ key: pathname }}, node);
    if (i === 0) {{
      // Head items belong inside the root layout (React hoists them to <head>).
      node = createElement(Fragment, null, ...headItems({{}}), createElement(Suspense, {{ fallback: null }}, createElement(MetadataHead, {{ route, params }})), node);
      headInjected = true;
    }}
    if (level.layout) {{
      const slotProps = matchSlots(level, parts, params, pathname);
      // Wrap the layout in the SEGMENT boundary carrying its active child URL segments
      // (parts.slice(level.slotBase)) so useSelectedLayoutSegment(s) read them via context;
      // nested layouts each get their own slice (innermost provider wins).
      node = createElement(
        SEGMENT_BOUNDARY,
        {{ segments: parts.slice(level.slotBase) }},
        createElement(level.layout, {{ params: paramsPromise, ...slotProps }}, node),
      );
    }}
  }}
  if (!headInjected) node = createElement(Fragment, null, ...headItems({{}}), createElement(Suspense, {{ fallback: null }}, createElement(MetadataHead, {{ route, params }})), node);
  // global-error.tsx (owns <html>): wrap the entire composed document — root layout
  // included — in the client error boundary so a throw escaping every nested error.tsx
  // (including one in the root layout) is replaced by global-error's own document.
  // React surfaces a thrown Server Component error to a client boundary only across a
  // Suspense boundary, so pair it with one (same rule the per-level error uses).
  if (GLOBAL_ERROR) {{
    node = createElement(ERROR_BOUNDARY, {{ fallback: GLOBAL_ERROR }}, createElement(Suspense, {{ fallback: null }}, node));
  }}
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

// Interpret a react-server render error's `digest` into control flow, mutating
// `control` in place. ONE definition, shared by the buffered `renderRequest` and the
// streaming `renderRequestStream` so both classify redirect / notFound / real errors
// identically. Returns the digest (marking the error known to React so the stream
// drains) or undefined.
function flightControlOnError(control, error) {{
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
    // A genuine error (recovered by an app-router error boundary, or fatal) — log it;
    // returning the digest marks it known to React so the stream drains.
    console.error("rsc-entry render onError:", error && error.stack ? error.stack : String(error));
  }}
  return digest || undefined;
}}

// Build the per-request store (url/headers/cookie + matched params). `requestAls.run`
// MUST enclose BOTH the render call AND the stream drain, or a late async Server
// Component loses the store.
function renderStore(pathname, reqCtx, params) {{
  return {{
    url: new URL(reqCtx.url || ("http://localhost" + pathname), "http://localhost"),
    headers: new Headers(reqCtx.headers || []),
    cookieHeader: reqCtx.cookie || "",
    params,
    // next/cache: the cache TAGS a page reads (unstable_cache / tagged fetch), captured so
    // the prerenderer can register the page under them (revalidateTag → pathname), and the
    // on-demand invalidations (revalidatePath/revalidateTag) collected during a render.
    tags: new Set(),
    revalidated: {{ tags: new Set(), paths: new Set() }},
    // next/headers response-cookie channel: cookies().set()/delete() and
    // draftMode().enable()/disable() push serialized Set-Cookie strings here; the worker
    // returns them and the orchestrator applies them to the HTTP response. `sealed` flips
    // true once the streaming shell has flushed (headers already sent), after which a
    // write throws a CLEAR error instead of silently vanishing.
    responseCookies: [],
    sealed: false,
  }};
}}

// Render `pathname` to a flight BUFFER + control meta. Shared by the one-shot argv
// `render` op AND the persistent `serve` worker, so both paths render identically.
export async function renderRequest(pathname, bundlerConfig, reqCtx) {{
  const {{ tree, status, params, intercept }} = documentTree(pathname, {{ softNav: !!reqCtx.softNav }});
  const store = renderStore(pathname, reqCtx, params);
  const control = {{}};
  const flight = await requestAls.run(store, async () => {{
    const stream = renderToReadableStream(tree, bundlerConfig, {{
      onError(error) {{ return flightControlOnError(control, error); }},
    }});
    return await drainToBuffer(stream);
  }});
  return {{
    flight,
    status: control.status || status || 200,
    params,
    redirect: control.redirect,
    notFound: control.notFound,
    intercept,
    // The cache tags this page read — the prerenderer records them so revalidateTag can
    // later map a tag back to this concrete pathname.
    tags: [...store.tags],
    // Set-Cookie strings a top-level cookies().set()/draftMode() write produced.
    setCookies: store.responseCookies,
  }};
}}

// Streaming variant of `renderRequest`: forwards flight chunks to `sink` as React
// produces them (the shell first, then each Suspense boundary as its async Server
// Component resolves) instead of buffering the whole flight. This is what makes TTFB
// fast for a page with a slow data dependency behind `<Suspense fallback={{loading}}>`.
//
// `sink.meta(m)` fires exactly ONCE, right after the first chunk is ready — by which
// point any TOP-LEVEL redirect()/notFound() (thrown before a Suspense boundary) has
// been captured in `control`, so the orchestrator can still issue a real HTTP redirect
// or 404 rather than stream. A redirect/notFound thrown BEHIND a Suspense boundary
// (after the shell has flushed) cannot unwind an already-streamed response; it is
// reported on `sink.end` and the orchestrator logs it loudly (never silently dropped).
export async function renderRequestStream(pathname, bundlerConfig, reqCtx, sink) {{
  const {{ tree, status, params, intercept }} = documentTree(pathname, {{ softNav: !!reqCtx.softNav }});
  const store = renderStore(pathname, reqCtx, params);
  const control = {{}};
  await requestAls.run(store, async () => {{
    const stream = renderToReadableStream(tree, bundlerConfig, {{
      onError(error) {{ return flightControlOnError(control, error); }},
    }});
    const reader = stream.getReader();
    let metaSent = false;
    const sendMeta = () => {{
      metaSent = true;
      // Snapshot the top-level response cookies (writes that ran before the shell
      // flushed) INTO the meta so the orchestrator can attach them to docHeaders.
      sink.meta({{ status: control.status || status || 200, params, redirect: control.redirect, notFound: control.notFound, intercept, setCookies: store.responseCookies.slice() }});
      // Headers are now on their way out: a cookies().set() behind a <Suspense> boundary
      // can no longer change the response, so seal the store — a later write HARD-ERRORS
      // (repo no-silent-stub rule) instead of silently disappearing.
      store.sealed = true;
    }};
    for (;;) {{
      const {{ done, value }} = await reader.read();
      if (done) break;
      if (!metaSent) {{
        // First chunk ready: `control` now reflects any top-level redirect/notFound.
        sendMeta();
        if (control.redirect || control.notFound) {{
          try {{ await reader.cancel(); }} catch {{}}
          break;
        }}
      }}
      sink.chunk(Buffer.from(value).toString("base64"));
    }}
    if (!metaSent) sendMeta();
    sink.end({{ status: control.status || status || 200, redirect: control.redirect, notFound: control.notFound, metaSent, tags: [...store.tags], setCookies: store.responseCookies.slice() }});
  }});
}}

// Dispatch a server action, returning its result flight BUFFER plus any on-demand cache
// invalidations the action requested (next/cache revalidatePath/revalidateTag). The action
// runs INSIDE a `requestAls` store so next/headers + next/cache resolve; the store's
// `revalidated` sets are drained into the reply so the orchestrator can bust the matching
// prerendered cache entries. `reqCtx` carries the request url/headers/cookie (optional; a
// standalone invocation may omit it).
export async function runAction(id, bundlerConfig, body, reqCtx) {{
  const ctx = reqCtx || {{}};
  const request = new Request("http://diffpack.local/_action/", {{
    method: "POST",
    headers: {{ "x-diffpack-action-id": id, "content-type": "application/json" }},
    body,
  }});
  const store = {{
    url: new URL(ctx.url || "http://diffpack.local/_action/", "http://localhost"),
    headers: new Headers(ctx.headers || []),
    cookieHeader: ctx.cookie || "",
    params: {{}},
    tags: new Set(),
    revalidated: {{ tags: new Set(), paths: new Set() }},
    // A Server Action is buffered (no shell has flushed), so cookies().set()/delete() and
    // draftMode().enable()/disable() are always safe here — collected and returned to the
    // orchestrator, which merges them into the action's 200 response.
    responseCookies: [],
    sealed: false,
  }};
  const flight = await requestAls.run(store, async () => {{
    const response = await handleServerAction(request, bundlerConfig);
    if (!response.body) throw new Error("rsc-entry action: handler produced no response body");
    return await drainToBuffer(response.body);
  }});
  return {{
    flight,
    revalidated: {{ tags: [...store.revalidated.tags], paths: [...store.revalidated.paths] }},
    setCookies: store.responseCookies,
  }};
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

// Dispatch a ROUTE HANDLER (`route.ts` HTTP endpoint). Matches `pathname` against the
// ROUTE_HANDLERS table, invokes the exported method function with a real `Request` and
// `{{ params }}` (inside the request store so cookies()/headers() work), and returns
// the Response serialized as `{{ status, headers, body(base64) }}`. Returns `null` when
// no handler path matches (the orchestrator then falls back to page rendering), or a
// 405 when the path matches but the method is not implemented.
export async function handleRoute(pathname, method, reqCtx) {{
  const parts = pathname.split("/").filter(Boolean);
  for (const entry of ROUTE_HANDLERS) {{
    const params = matchSegments(entry.segments, parts);
    if (!params) continue;
    const fn = entry.methods[method] || (method === "HEAD" ? entry.methods.GET : undefined);
    if (typeof fn !== "function") {{
      return {{ status: 405, headers: [["allow", Object.keys(entry.methods).join(", ")]], body: "" }};
    }}
    const url = reqCtx.url || ("http://localhost" + pathname);
    const bodyBytes =
      method === "GET" || method === "HEAD" || reqCtx.body == null
        ? undefined
        : reqCtx.bodyIsBase64
          ? Buffer.from(reqCtx.body, "base64")
          : reqCtx.body;
    const request = new Request(url, {{
      method,
      headers: new Headers(reqCtx.headers || []),
      body: bodyBytes,
    }});
    const store = {{
      url: new URL(url, "http://localhost"),
      headers: new Headers(reqCtx.headers || []),
      cookieHeader: reqCtx.cookie || "",
      params,
      tags: new Set(),
      revalidated: {{ tags: new Set(), paths: new Set() }},
      // A route handler runs fully buffered, so cookies().set()/draftMode() are always
      // safe — collected here and merged with the Response's own Set-Cookie headers.
      responseCookies: [],
      sealed: false,
    }};
    const res = await requestAls.run(store, () => fn(request, {{ params: Promise.resolve(params) }}));
    // next/cache invalidations a route handler requested (revalidatePath/revalidateTag) —
    // drained onto the result so the orchestrator can bust the matching cache entries.
    const revalidated = {{ tags: [...store.revalidated.tags], paths: [...store.revalidated.paths] }};
    if (!(res instanceof Response)) {{
      return {{ status: 200, headers: [], body: "", revalidated, setCookies: store.responseCookies }};
    }}
    const buf = Buffer.from(await res.arrayBuffer());
    // Set-Cookie may come from BOTH next/headers cookies().set (store.responseCookies) and
    // the Response's own headers. Extract the Response's set-cookies with getSetCookie
    // (which the plain Headers iterator would otherwise comma-join and corrupt) and return
    // ONE merged array; strip set-cookie from the plain header list to avoid a duplicate.
    const setCookies = store.responseCookies.slice();
    const headers = [];
    for (const [k, v] of res.headers) {{
      if (k.toLowerCase() !== "set-cookie") headers.push([k, v]);
    }}
    if (typeof res.headers.getSetCookie === "function") {{
      for (const c of res.headers.getSetCookie()) setCookies.push(c);
    }} else {{
      const sc = res.headers.get("set-cookie");
      if (sc) setCookies.push(sc);
    }}
    return {{
      status: res.status,
      headers,
      body: buf.toString("base64"),
      bodyIsBase64: true,
      revalidated,
      setCookies,
    }};
  }}
  return null;
}}

// The route-handler routes (segment patterns + methods) + whether the app has
// middleware — queried once at boot by the orchestrator so it can match locally and
// dispatch handler/middleware without a per-page-request round-trip.
export function routeManifest() {{
  return {{
    handlers: ROUTE_HANDLERS.map((entry) => ({{ segments: entry.segments, methods: Object.keys(entry.methods) }})),
    hasMiddleware: MIDDLEWARE != null,
  }};
}}

// Run the app's middleware (if any) for a request. Returns the middleware's Response
// serialized as `{{ status, headers, body(base64) }}` — the orchestrator reads Next's
// `x-middleware-*` protocol headers on it (next / redirect / rewrite) — or `null` when
// there is no middleware.
export async function runMiddleware(reqCtx) {{
  if (MIDDLEWARE == null) return null;
  const url = reqCtx.url || "http://localhost/";
  const request = new NextRequest(url, {{ method: reqCtx.method || "GET", headers: new Headers(reqCtx.headers || []) }});
  const store = {{
    url: new URL(url, "http://localhost"),
    headers: new Headers(reqCtx.headers || []),
    cookieHeader: reqCtx.cookie || "",
    params: {{}},
  }};
  const res = await requestAls.run(store, () => MIDDLEWARE(request, {{}}));
  if (!(res instanceof Response)) return {{ status: 200, headers: [["x-middleware-next", "1"]], body: "" }};
  const buf = Buffer.from(await res.arrayBuffer());
  return {{ status: res.status, headers: [...res.headers], body: buf.toString("base64"), bodyIsBase64: true }};
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
      const renderRequestStream = ns.renderRequestStream || m.renderRequestStream;
      const runAction = ns.runAction || m.runAction;
      const handleRoute = ns.handleRoute || m.handleRoute;
      const routeManifest = ns.routeManifest || m.routeManifest;
      const runMiddleware = ns.runMiddleware || m.runMiddleware;
      if (typeof renderRequest !== "function" || typeof runAction !== "function") {{
        throw new Error("rsc-entry serve: re-imported bundle does not export renderRequest/runAction");
      }}
      cached = {{ mtime, mod: {{ renderRequest, renderRequestStream, runAction, handleRoute, routeManifest, runMiddleware }} }};
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
          reply({{ id: req.id, flight: Buffer.from(r.flight).toString("base64"), status: r.status, params: r.params, redirect: r.redirect, notFound: r.notFound, tags: r.tags || [], setCookies: r.setCookies || [] }});
        }} else if (req.op === "render-stream") {{
          // Streaming render: one `streamMeta` line, then N `streamChunk` lines, then a
          // single `streamEnd` line — all sharing this request id. The orchestrator
          // routes them to the request's flight stream (see `callStream`).
          if (typeof mod.renderRequestStream !== "function") {{
            reply({{ id: req.id, error: "rsc-entry serve: bundle does not export renderRequestStream" }});
          }} else {{
            await mod.renderRequestStream(req.pathname || "/", manifest(req.manifestPath), req.reqCtx || {{}}, {{
              meta: (m) => reply({{ id: req.id, streamMeta: m }}),
              chunk: (b64) => reply({{ id: req.id, streamChunk: b64 }}),
              end: (m) => reply({{ id: req.id, streamEnd: m || {{}} }}),
            }});
          }}
        }} else if (req.op === "action") {{
          const a = await mod.runAction(req.actionId, manifest(req.manifestPath), req.body || "", req.reqCtx || {{}});
          reply({{ id: req.id, flight: Buffer.from(a.flight).toString("base64"), status: 200, revalidated: a.revalidated, setCookies: a.setCookies || [] }});
        }} else if (req.op === "route") {{
          const r = mod.handleRoute
            ? await mod.handleRoute(req.pathname || "/", req.method || "GET", req.reqCtx || {{}})
            : null;
          reply({{ id: req.id, routeResult: r, revalidated: r && r.revalidated }});
        }} else if (req.op === "routes") {{
          reply({{ id: req.id, routes: mod.routeManifest ? mod.routeManifest() : {{ handlers: [], hasMiddleware: false }} }});
        }} else if (req.op === "middleware") {{
          const r = mod.runMiddleware ? await mod.runMiddleware(req.reqCtx || {{}}) : null;
          reply({{ id: req.id, middlewareResult: r }});
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
    writeMeta({{ status: r.status, params: r.params, redirect: r.redirect, notFound: r.notFound, tags: r.tags || [] }});
    return;
  }}
  if (op === "action") {{
    const id = rest[0];
    const manifestPath = rest[1];
    if (!id) throw new Error("rsc-entry action: missing action id argument");
    if (!manifestPath) throw new Error("rsc-entry action: missing manifest path argument");
    const bundlerConfig = JSON.parse(readFileSync(manifestPath, "utf8"));
    const body = await readStdin();
    const {{ flight }} = await runAction(id, bundlerConfig, body);
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
fn ssr_entry_module(
    adapter_dir: &Path,
    islands: &[PathBuf],
    hooks_context: &Path,
    asset_base: &str,
) -> String {
    let pins = island_pins(adapter_dir, islands);
    let lazy = js_str(&adapter_dir.join("lazy.js").to_string_lossy());
    let hooks_import = js_str(&hooks_context.to_string_lossy());
    // The browser fetches the client bootstrap under the app's basePath/assetPrefix (the
    // orchestrator strips that prefix back off before the publicDir lookup).
    let client_js = js_str(&format!("{asset_base}/client.js"));
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
import {{ renderToPipeableStream, renderToStaticMarkup }} from "react-dom/server";
import {{ createElement }} from "react";
import {{ PathParamsContext, PathnameContext, SearchParamsContext, ServerInsertedHTMLContext }} from {hooks_import};
import {{ Writable, PassThrough }} from "node:stream";
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
  // useServerInsertedHTML: a per-request list of callbacks (each returns a React node).
  // The buffered path renders them once the tree is done and splices the HTML before
  // </head>. Zero cost unless a CSS-in-JS registry actually registers one.
  const inserted = [];
  const root = createElement(
    ServerInsertedHTMLContext.Provider,
    {{ value: (cb) => inserted.push(cb) }},
    createElement(
      PathParamsContext.Provider,
      {{ value: params || {{}} }},
      createElement(
        PathnameContext.Provider,
        {{ value: pathname }},
        createElement(SearchParamsContext.Provider, {{ value: search }}, flightRoot),
      ),
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
    sink.on("finish", () => {{
      let html = Buffer.concat(parts).toString("utf8");
      if (inserted.length) {{
        const extra = inserted.map((cb) => renderToStaticMarkup(cb())).join("");
        inserted.length = 0;
        const at = html.indexOf("</head>");
        html = at === -1 ? extra + html : html.slice(0, at) + extra + html.slice(at);
      }}
      resolve(html);
    }});
    sink.on("error", reject);
    const {{ pipe }} = renderToPipeableStream(root, {{
      bootstrapModules: [{client_js}],
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

// STREAMING SSR-of-flight. `flightChunks` is an async iterable of base64 flight chunks
// arriving from the react-server worker as boundaries resolve. We reconstruct the tree
// from those chunks (createFromReadableStream over a live stream — the shell resolves
// immediately, inner Suspense boundaries stay pending), render with
// `renderToPipeableStream`, and pipe the HTML to `res` starting at `onShellReady` so
// the shell + fallbacks reach the browser BEFORE the slow data resolves.
//
// Hydration uses Next's incremental-inline-flight approach: each flight chunk is also
// written into the document as a `<script>self.__DF_FLIGHT.push([1,"<b64>"])</script>`
// (a final `[0]` marks the end). The client rebuilds the flight stream from that append
// array, so it hydrates from the SAME bytes with no second network fetch — and it works
// regardless of whether client.js executes before or after the chunks arrive.
//
// All bytes go to `res` through ONE ordered async loop (React's HTML via a PassThrough,
// flight scripts interleaved AFTER each HTML chunk), so a flight script can never
// precede the doctype/shell.
export async function renderFlightToStream(flightChunks, serverConsumerManifest, params, url, res, headers, status) {{
  installSeam();
  const pathname = (url && url.pathname) || "/";
  const search = (url && url.search) || "";
  // Live byte stream feeding the SSR flight reconstruction + a queue of the inline
  // `<script>` tags carrying the same chunks for the client. One pump drives both.
  let byteController;
  const byteStream = new ReadableStream({{ start(c) {{ byteController = c; }} }});
  const scriptQueue = [];
  let pumpDone = false;
  const pump = (async () => {{
    for await (const b64 of flightChunks) {{
      const binary = Buffer.from(b64, "base64");
      byteController.enqueue(new Uint8Array(binary));
      scriptQueue.push(
        "<script>(self.__DF_FLIGHT=self.__DF_FLIGHT||[]).push([1," + JSON.stringify(b64) + "])</script>",
      );
    }}
    byteController.close();
    scriptQueue.push("<script>(self.__DF_FLIGHT=self.__DF_FLIGHT||[]).push([0])</script>");
    pumpDone = true;
  }})();
  // The initial model resolves from the shell rows (does NOT await the whole stream);
  // pending Suspense boundaries stay lazy and resolve as later chunks arrive.
  const flightRoot = await createFromReadableStream(byteStream, {{
    serverConsumerManifest,
    callServer() {{
      throw new Error("diffpack next ssr: a server action was called during SSR");
    }},
  }});
  // useServerInsertedHTML: per-request callbacks flushed into the byte stream AFTER each
  // HTML chunk (streaming registries push as boundaries resolve). Shell styles land with
  // the first chunk; late registrations flush after the shell (styled-components tolerates
  // this). Zero cost unless a registry actually registers one.
  const inserted = [];
  let insertedFlushed = 0;
  const root = createElement(
    ServerInsertedHTMLContext.Provider,
    {{ value: (cb) => inserted.push(cb) }},
    createElement(
      PathParamsContext.Provider,
      {{ value: params || {{}} }},
      createElement(
        PathnameContext.Provider,
        {{ value: pathname }},
        createElement(SearchParamsContext.Provider, {{ value: search }}, flightRoot),
      ),
    ),
  );
  await new Promise((resolve, reject) => {{
    const html = new PassThrough();
    let shellStarted = false;
    const {{ pipe }} = renderToPipeableStream(root, {{
      bootstrapModules: [{client_js}],
      // No inlined full flight here — it streams as __DF_FLIGHT scripts. Seed the array
      // (so it exists before client.js runs) + the hooks-context globals.
      bootstrapScriptContent:
        "self.__DF_FLIGHT=self.__DF_FLIGHT||[];" +
        "window.__DIFFPACK_PARAMS__ = " + JSON.stringify(params || {{}}) + ";" +
        "window.__DIFFPACK_URL__ = " + JSON.stringify({{ pathname: pathname, search: search }}) + ";",
      onShellReady() {{
        res.writeHead(status || 200, headers);
        pipe(html);
      }},
      onShellError(error) {{
        if (!shellStarted) {{
          try {{ res.writeHead(500, {{ "content-type": "text/html; charset=utf-8" }}); }} catch {{}}
          res.end("<!doctype html><p>Internal Server Error</p>");
        }}
        reject(error);
      }},
      onError(error) {{
        console.error("next-ssr stream onError:", error && error.message ? error.message : error);
      }},
    }});
    // Forward React's HTML to `res`, flushing any queued flight scripts AFTER each HTML
    // chunk (guarantees the shell precedes the first flight script). React ends `html`
    // only once every boundary is done, by which point the flight is fully drained.
    (async () => {{
      try {{
        for await (const chunk of html) {{
          shellStarted = true;
          res.write(chunk);
          while (insertedFlushed < inserted.length) {{
            res.write(renderToStaticMarkup(inserted[insertedFlushed++]()));
          }}
          while (scriptQueue.length) res.write(scriptQueue.shift());
        }}
        await pump;
        while (insertedFlushed < inserted.length) {{
          res.write(renderToStaticMarkup(inserted[insertedFlushed++]()));
        }}
        while (scriptQueue.length) res.write(scriptQueue.shift());
        res.end();
        resolve();
      }} catch (error) {{
        reject(error);
      }}
    }})();
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
import {{ createPortal }} from "react-dom";
import {{ use, useState, useEffect, useRef, useTransition, createElement, Fragment, Suspense }} from "react";
import {{ PathParamsContext, PathnameContext, SearchParamsContext }} from {hooks_import};
import {{ callServer }} from "#diffpack-call-server";
{pins}
// Force a code split so the client build uses the registry runtime + the RSC seam.
import({lazy}).then((module) => {{
  (globalThis).__diffpack_next_client_lazy = module.value;
}});

// Fetch a route's raw flight over `?__rsc=1`, returning the React-`use()`-able tree AND
// whether the server rendered it as an INTERCEPT overlay (x-diffpack-intercept header).
// The flight resolves through the same `__webpack_*` client seam + `callServer` transport
// the action round-trip uses — no manifest needed.
async function fetchFlight(href) {{
  const sep = href.includes("?") ? "&" : "?";
  const res = await fetch(href + sep + "__rsc=1");
  const intercept = res.headers.get("x-diffpack-intercept") === "1";
  const tree = createFromReadableStream(res.body, {{ callServer }});
  return {{ tree, intercept }};
}}

// Portals an overlay (intercept modal) flight into <body>, so it sits ABOVE the still
// mounted underlying page (a sibling after <html> would be invalid). Suspends on its own
// thenable; wrapped in a Suspense with a null fallback so the page stays visible while it
// loads.
function ModalPortal({{ thenable }}) {{
  return createPortal(use(thenable), document.body);
}}

// The client Router: holds the current document tree, and swaps it (inside a transition,
// keeping the old document visible until the new flight resolves) on navigation. An
// INTERCEPT soft-nav does NOT swap the document — it keeps the current page mounted (so
// its state/scroll survive) and renders the overlay via a body portal, masking the URL to
// the target. Back on a masked modal URL closes the overlay without refetching.
function Router({{ initialTree }}) {{
  const [tree, setTree] = useState(initialTree);
  const [modal, setModal] = useState(null); // {{ tree }} overlay, or null
  const [, startTransition] = useTransition();
  const modalOpen = useRef(false);
  const underlying = useRef(location.pathname + location.search);
  useEffect(() => {{
    // A bounded, single-use prefetch cache: next/link hover/focus (and
    // useRouter().prefetch) warm the target route's flight here; navigate() consumes the
    // pending entry if present (instant swap), else fetches. LRU-capped + evicted on use
    // so it cannot grow across a long session, and far less aggressive than Next's
    // viewport auto-prefetch (hover/focus only unless prefetch===true).
    const PREFETCH_MAX = 24;
    const prefetchCache = new Map(); // href -> Promise<{{ tree, intercept }}>
    function prefetch(href) {{
      if (typeof href !== "string" || !href.startsWith("/")) return;
      if (prefetchCache.has(href)) return;
      prefetchCache.set(href, fetchFlight(href));
      while (prefetchCache.size > PREFETCH_MAX) {{
        prefetchCache.delete(prefetchCache.keys().next().value);
      }}
    }}
    async function navigate(to, options) {{
      const opts = options || {{}};
      const push = opts.push !== false;
      const href = typeof to === "string" ? to : to.href;
      const replace = opts.replace || (typeof to === "object" && to && to.replace);
      // Consume a warmed prefetch (single-use) if one exists, else fetch now.
      let pending = prefetchCache.get(href);
      if (pending) prefetchCache.delete(href);
      const {{ tree: next, intercept }} = await (pending || fetchFlight(href));
      if (intercept) {{
        underlying.current = location.pathname + location.search;
        modalOpen.current = true;
        startTransition(() => {{
          setModal({{ tree: next }});
          if (push) history.pushState({{ __diffpackModal: true }}, "", href);
        }});
        return;
      }}
      modalOpen.current = false;
      startTransition(() => {{
        setModal(null);
        setTree(next);
        if (push) history[replace ? "replaceState" : "pushState"](null, "", href);
      }});
    }}
    // Close an open overlay (used by a modal's own close / router.back()).
    function closeModal() {{
      if (!modalOpen.current) return;
      modalOpen.current = false;
      setModal(null);
      history.pushState(null, "", underlying.current);
    }}
    // router.refresh(): a SOFT RSC refresh — re-fetch the CURRENT route's flight (bypassing
    // the prefetch cache) and swap the tree inside a transition, keeping the document
    // mounted so island state survives. Never a window.location.reload().
    async function refresh() {{
      const href = location.pathname + location.search;
      const {{ tree: next }} = await fetchFlight(href);
      startTransition(() => {{
        setModal(null);
        setTree(next);
      }});
    }}
    window.__diffpack_navigate = navigate;
    window.__diffpack_close_modal = closeModal;
    window.__diffpack_prefetch = prefetch;
    window.__diffpack_refresh = refresh;
    function onpop() {{
      // Leaving a masked modal URL: just close the overlay (the underlying page is still
      // mounted), no refetch.
      if (modalOpen.current && (location.pathname + location.search) === underlying.current) {{
        modalOpen.current = false;
        setModal(null);
        return;
      }}
      navigate(location.pathname + location.search, {{ push: false }});
    }}
    window.addEventListener("popstate", onpop);
    return () => window.removeEventListener("popstate", onpop);
  }}, []);
  return createElement(
    Fragment,
    null,
    use(tree),
    modal ? createElement(Suspense, {{ fallback: null }}, createElement(ModalPortal, {{ thenable: modal.tree }})) : null,
  );
}}

function decodeFlight(base64) {{
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) bytes[i] = binary.charCodeAt(i);
  return bytes;
}}

// Rebuild the flight as a live stream from the incremental `self.__DF_FLIGHT` append
// array the STREAMING SSR path writes into the document (`[1,b64]` chunks, `[0]` end).
// Replaying entries already pushed before this runs, then overriding `push`, captures
// every chunk regardless of whether client.js executed before or after they arrived.
function flightStreamFromDF() {{
  const q = (self.__DF_FLIGHT = self.__DF_FLIGHT || []);
  return new ReadableStream({{
    start(controller) {{
      let closed = false;
      const handle = (entry) => {{
        if (!entry) return;
        if (entry[0] === 1) controller.enqueue(decodeFlight(entry[1]));
        else if (entry[0] === 0 && !closed) {{ closed = true; controller.close(); }}
      }};
      for (const entry of q) handle(entry);
      q.length = 0;
      q.push = (entry) => {{ handle(entry); return 0; }};
    }},
  }});
}}

function boot() {{
  // Streaming render inlines the flight incrementally as __DF_FLIGHT; the buffered
  // render (notFound / error docs) inlines the whole flight as __DIFFPACK_FLIGHT__.
  let stream;
  if (self.__DF_FLIGHT) {{
    stream = flightStreamFromDF();
  }} else if (window.__DIFFPACK_FLIGHT__) {{
    const bytes = decodeFlight(window.__DIFFPACK_FLIGHT__);
    stream = new ReadableStream({{
      start(controller) {{
        controller.enqueue(bytes);
        controller.close();
      }},
    }});
  }} else {{
    throw new Error(
      "diffpack next client: no flight payload (neither the __DF_FLIGHT stream nor window.__DIFFPACK_FLIGHT__)",
    );
  }}
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

/// `next/server` shim: `NextResponse` (extends `Response` with `next`/`redirect`/
/// `rewrite`/`json` + a cookie jar) and `NextRequest` (adds `nextUrl` + `cookies`).
/// Middleware returns a `NextResponse`; the orchestrator reads Next's real
/// `x-middleware-*` protocol headers to decide continue / redirect / rewrite.
fn next_server_shim() -> &'static str {
    r##"// `next/server` shim (diffpack next app-router adapter).
function cookieJar(headers, isResponse) {
  return {
    get(name) {
      const raw = headers.get("cookie") || "";
      const hit = raw.split(";").map((c) => c.trim()).find((c) => c.startsWith(name + "="));
      return hit ? { name, value: decodeURIComponent(hit.slice(name.length + 1)) } : undefined;
    },
    getAll() {
      const raw = headers.get("cookie") || "";
      return raw.split(";").map((c) => c.trim()).filter(Boolean).map((c) => {
        const eq = c.indexOf("=");
        return { name: c.slice(0, eq), value: decodeURIComponent(c.slice(eq + 1)) };
      });
    },
    set(name, value, opts) {
      const parts = [`${name}=${encodeURIComponent(typeof name === "object" ? name.value : value)}`];
      const o = typeof name === "object" ? name : opts || {};
      if (o.path) parts.push(`Path=${o.path}`);
      if (o.maxAge != null) parts.push(`Max-Age=${o.maxAge}`);
      if (o.httpOnly) parts.push("HttpOnly");
      if (o.secure) parts.push("Secure");
      if (o.sameSite) parts.push(`SameSite=${o.sameSite}`);
      headers.append("set-cookie", parts.join("; "));
      return this;
    },
    delete(name) {
      headers.append("set-cookie", `${name}=; Max-Age=0`);
      return this;
    },
  };
}

export class NextResponse extends Response {
  get cookies() {
    return cookieJar(this.headers, true);
  }
  static next(init) {
    const headers = new Headers(init && init.headers);
    // Request-header overrides (NextResponse.next({ request: { headers } })) are
    // encoded for the orchestrator to apply to the downstream render.
    if (init && init.request && init.request.headers) {
      const reqHeaders = new Headers(init.request.headers);
      const names = [];
      for (const [k, v] of reqHeaders) {
        names.push(k);
        headers.set("x-middleware-request-" + k, v);
      }
      headers.set("x-middleware-override-headers", names.join(","));
    }
    headers.set("x-middleware-next", "1");
    return new NextResponse(null, { headers });
  }
  static redirect(url, init) {
    const status = typeof init === "number" ? init : (init && init.status) || 307;
    const headers = new Headers(init && typeof init === "object" ? init.headers : undefined);
    headers.set("location", String(url));
    return new NextResponse(null, { status, headers });
  }
  static rewrite(destination, init) {
    const headers = new Headers(init && init.headers);
    headers.set("x-middleware-rewrite", String(destination));
    return new NextResponse(null, { headers });
  }
  static json(body, init) {
    const headers = new Headers(init && init.headers);
    if (!headers.has("content-type")) headers.set("content-type", "application/json");
    return new NextResponse(JSON.stringify(body), { ...(init || {}), headers });
  }
}

export class NextRequest extends Request {
  constructor(input, init) {
    super(input, init);
    const url = typeof input === "string" ? input : input.url;
    this.nextUrl = new URL(url, "http://localhost");
  }
  get cookies() {
    return cookieJar(this.headers, false);
  }
}

export default NextResponse;
export const userAgent = (request) => ({ ua: (request.headers.get("user-agent") || "") });
"##
}

fn next_link_shim(base_path: &str) -> String {
    // `BASE_PATH` (next.config `basePath`) is baked in so the rendered `<a href>` AND the
    // soft-nav target both carry the prefix; the orchestrator strips it back off on the
    // `?__rsc=1` fetch. "use client" MUST stay the first statement, so the const follows it.
    format!(
        "\"use client\";\nconst BASE_PATH = {};\n{NEXT_LINK_SHIM_BODY}",
        js_str(base_path),
    )
}

const NEXT_LINK_SHIM_BODY: &str = r##"// `next/link` shim (diffpack next app-router adapter). A `"use client"` intercepting
// component: it renders the same server-reachable `<a href>`, but on the browser a
// plain left-click on an internal href is intercepted and handed to the client
// Router (`window.__diffpack_navigate`), which fetches the target route's flight
// (`?__rsc=1`) and diff-renders it WITHOUT a full document load. Modified clicks
// (meta/ctrl/shift/alt or a non-primary button), external/non-string hrefs, an
// already-`defaultPrevented` event, or the pre-hydration window (no
// `__diffpack_navigate`) all fall through to a real navigation — no `preventDefault`.
import { createElement, useEffect } from "react";

// Prepend the app's basePath to an internal (leading-slash) href, once — an href already
// carrying the prefix (e.g. reconstructed from router state) is left untouched. External
// / non-string hrefs pass through so `<a>` and the click guards stay basePath-agnostic.
function withBasePath(href) {
  if (!BASE_PATH || typeof href !== "string" || !href.startsWith("/")) return href;
  if (href === BASE_PATH || href.startsWith(BASE_PATH + "/")) return href;
  return BASE_PATH + href;
}

export default function Link(props) {
  const { href, children, prefetch, replace, scroll, shallow, locale, onClick, onMouseEnter, onFocus, ...rest } = props;
  const rawHref = typeof href === "string" ? href : (href && href.pathname) || "#";
  const resolved = withBasePath(rawHref);
  // Warm the client Router's prefetch cache for an internal href (the same ?__rsc=1 flight
  // a click fetches, moved earlier to hover/focus). `prefetch={false}` opts out; the
  // default prefetches on interaction; `prefetch={true}` also prefetches eagerly on mount.
  // No viewport observer (that would add per-link cost) — hover/focus is the trigger.
  function warmPrefetch() {
    if (prefetch === false) return;
    if (typeof href !== "string" || !href.startsWith("/")) return;
    if (typeof window === "undefined" || typeof window.__diffpack_prefetch !== "function") return;
    window.__diffpack_prefetch(resolved);
  }
  useEffect(() => {
    if (prefetch === true) warmPrefetch();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  function handleClick(event) {
    if (onClick) onClick(event);
    if (event.defaultPrevented) return;
    if (event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
    if (typeof href !== "string" || !href.startsWith("/")) return;
    if (typeof window === "undefined" || typeof window.__diffpack_navigate !== "function") return;
    event.preventDefault();
    window.__diffpack_navigate(resolved, { replace: !!replace });
  }
  function handleMouseEnter(event) {
    if (onMouseEnter) onMouseEnter(event);
    warmPrefetch();
  }
  function handleFocus(event) {
    if (onFocus) onFocus(event);
    warmPrefetch();
  }
  return createElement(
    "a",
    { href: resolved, onClick: handleClick, onMouseEnter: handleMouseEnter, onFocus: handleFocus, ...rest },
    children,
  );
}

// `useLinkStatus` (Next 15.3+/16): the pending state of an in-progress client
// navigation started by a parent `<Link>`. This adapter's soft-nav diff-renders the
// route synchronously via `__diffpack_navigate` and does not expose a per-link pending
// signal, so this returns the settled state — matching the common `{ pending }`
// destructure (a loading indicator simply never shows) rather than throwing on import.
export function useLinkStatus() {
  return { pending: false };
}
"##;

/// `next/dynamic` shim (`shims/dynamic.ts`). A lean reimplementation of Next's `dynamic()`
/// keyed on its public option shape (`{ loading, ssr }`), backed by `React.lazy`. `ssr:true`
/// (the default) wraps the lazy chunk in a `Suspense` with the `loading` fallback — valid in
/// the react-server, SSR, and client graphs. `ssr:false` renders the `loading` fallback on
/// the server AND the first client paint (a mounted-gate: `useState(false)` + `useEffect`), so
/// the SSR HTML and first hydration match, then swaps in the real chunk after mount — exactly
/// as Next requires (and, like Next, `ssr:false` inside a Server Component surfaces React's
/// react-server hook error). React is namespace-imported so the client hooks are not named
/// bindings that would fail to resolve under the `react-server` export condition; they are
/// only ever CALLED on the client `ssr:false` path.
fn next_dynamic_shim() -> &'static str {
    r#"// `next/dynamic` shim (diffpack next app-router adapter). dynamic(loader, { loading, ssr }).
import * as React from "react";
const { createElement, lazy, useState, useEffect, Suspense } = React;

// Normalize a next/dynamic loader (a () => import(...), a bare import promise, or a
// { default } module) to a Promise<{ default: Component }> for React.lazy.
function toLoadable(loader) {
  return function load() {
    const result = typeof loader === "function" ? loader() : loader;
    return Promise.resolve(result).then((mod) => {
      if (mod && mod.default) return mod;
      return { default: mod };
    });
  };
}

export default function dynamic(loader, options) {
  const opts = options || {};
  const Loading = opts.loading || null;
  const fallback = Loading
    ? createElement(Loading, { isLoading: true, pastDelay: true, error: null })
    : null;
  const LazyComponent = lazy(toLoadable(loader));
  if (opts.ssr === false) {
    // Client-only: gate on mount so server + first client paint both render `fallback`
    // (no hydration mismatch), then load the real chunk.
    return function DynamicClientOnly(props) {
      const [mounted, setMounted] = useState(false);
      useEffect(() => { setMounted(true); }, []);
      if (!mounted) return fallback;
      return createElement(Suspense, { fallback: fallback }, createElement(LazyComponent, props));
    };
  }
  return function DynamicComponent(props) {
    return createElement(Suspense, { fallback: fallback }, createElement(LazyComponent, props));
  };
}
"#
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
/// Copy the app-root metadata IMAGE file conventions (`app/icon.png`,
/// `app/favicon.ico`, `app/apple-icon.*`, `app/opengraph-image.*`,
/// `app/twitter-image.*`) into the served `public/` output at their served filename
/// (`/icon.png`, ...). A build-time copy — the head `<link>`/`<meta>` referencing them
/// is emitted by the react-server entry, so serving them is zero per-request cost
/// (they flow through the orchestrator's existing static-asset path). Returns the count
/// copied; a no-op for an app with no convention images. Reuses the SAME discovery
/// ([`scan_metadata_images`]) as the head-link emitter, so the copied files and the
/// linked URLs cannot drift.
pub fn emit_metadata_images(root: &Path, out_public: &Path) -> Result<usize, String> {
    let app_dir = root.join("app");
    if !app_dir.is_dir() {
        return Ok(0);
    }
    let images = scan_metadata_images(&app_dir)?;
    let mut written = 0usize;
    for img in &images {
        std::fs::create_dir_all(out_public)
            .map_err(|error| format!("cannot create {}: {error}", out_public.display()))?;
        let dest = out_public.join(img.served.trim_start_matches('/'));
        std::fs::copy(&img.source, &dest).map_err(|error| {
            format!("cannot copy metadata image {} -> {}: {error}", img.source.display(), dest.display())
        })?;
        written += 1;
    }
    Ok(written)
}

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
import CONFIG from "../image-config";

const DEVICE_SIZES = CONFIG.deviceSizes || [640, 750, 828, 1080, 1200, 1920, 2048, 3840];
const IMAGE_SIZES = CONFIG.imageSizes || [16, 32, 48, 64, 96, 128, 256, 384];
const ALL_SIZES = [...IMAGE_SIZES, ...DEVICE_SIZES];

// Built-in third-party loaders (Next's `images.loader` presets). Each returns a URL for
// a given { src, width, quality }.
function imgixLoader({ src, width, quality }) {
  const u = new URL((CONFIG.path || "https://example.com") + src);
  const p = u.searchParams;
  p.set("auto", p.getAll("auto").join(",") || "format");
  p.set("fit", p.get("fit") || "max");
  p.set("w", p.get("w") || String(width));
  if (quality) p.set("q", String(quality));
  return u.href;
}
function cloudinaryLoader({ src, width, quality }) {
  const params = ["f_auto", "c_limit", "w_" + width, "q_" + (quality || "auto")];
  return (CONFIG.path || "") + params.join(",") + src;
}
function akamaiLoader({ src, width }) {
  return (CONFIG.path || "") + src + "?imwidth=" + width;
}
function builtinLoader(name) {
  if (name === "imgix") return imgixLoader;
  if (name === "cloudinary") return cloudinaryLoader;
  if (name === "akamai") return akamaiLoader;
  return null; // "default" / "custom" have no built-in URL scheme here
}

// Port of Next's remote-pattern matcher: a remote src is allowed if its URL matches any
// `images.remotePatterns` entry (protocol/hostname-with-* and **-wildcards/port/pathname/
// search) or a legacy `images.domains` hostname.
function wildcardMatch(pattern, value) {
  if (!pattern) return true;
  // `**` matches across dots (any subdomain / any path depth); `*` matches one segment.
  const rx = "^" + pattern
    .replace(/[.+?^${}()|[\]\\]/g, "\\$&")
    .replace(/\*\*/g, " ")
    .replace(/\*/g, "[^.\\/]*")
    .replace(/ /g, ".*") + "$";
  return new RegExp(rx).test(value);
}
function matchRemotePattern(p, url) {
  if (p.protocol && p.protocol.replace(/:$/, "") !== url.protocol.replace(/:$/, "")) return false;
  if (p.hostname && !wildcardMatch(p.hostname, url.hostname)) return false;
  if (p.port && p.port !== url.port) return false;
  if (p.pathname && !wildcardMatch(p.pathname, url.pathname)) return false;
  if (p.search && p.search !== url.search) return false;
  return true;
}
function hasMatch(url) {
  if ((CONFIG.domains || []).includes(url.hostname)) return true;
  return (CONFIG.remotePatterns || []).some((p) => matchRemotePattern(p, url));
}
function hostnameNotConfigured(rawSrc) {
  return new Error(
    "next/image: hostname '" + new URL(rawSrc).hostname + "' is not configured under " +
    "images in next.config; add it to images.remotePatterns (or images.domains).",
  );
}

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
  const isRemote = /^https?:\/\//i.test(rawSrc);
  const forcedUnopt = Boolean(unoptimized) || CONFIG.unoptimized || isData || isSvg || (entry && entry.unoptimized);

  // Loader precedence, matching Next: the `loader` prop > a next.config `loaderFile`
  // (bundled as CONFIG.loaderFn) > a built-in named loader (imgix/cloudinary/akamai).
  const explicitLoader =
    typeof loader === "function" ? loader : CONFIG.loaderFn || builtinLoader(CONFIG.loader);

  // Build a loader-driven srcset (one loader call per candidate width).
  const loaderSrcSet = (fn) => {
    const { widths, kind } = getWidths(numericWidth, sizes);
    const parts = widths.map((w, i) => fn({ src: rawSrc, width: w, quality }) + " " + (kind === "w" ? w + "w" : (i + 1) + "x"));
    const finalSrc = fn({ src: rawSrc, width: widths[widths.length - 1], quality });
    return { srcSet: parts.join(", "), finalSrc };
  };

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

  // An <img> whose src/srcSet come from a loader (with the same base attrs as baseImg),
  // plus the `priority` preload link when requested.
  const loaderImg = (finalSrc, srcSet) => {
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
      const link = createElement("link", { rel: "preload", as: "image", href: finalSrc, imageSrcSet: srcSet, imageSizes: sizes });
      return createElement(Fragment, null, link, img);
    }
    return img;
  };

  if (forcedUnopt) return baseImg();

  // A configured loader (prop / loaderFile / imgix|cloudinary|akamai) drives the srcset
  // for ANY src, remote or local, exactly as Next routes all images through loaderFile.
  if (explicitLoader) {
    if (isRemote && !hasMatch(new URL(rawSrc)) && typeof loader !== "function" && !CONFIG.loaderFn) {
      // A built-in named loader still respects the remote allow-list.
      throw hostnameNotConfigured(rawSrc);
    }
    const { srcSet, finalSrc } = loaderSrcSet(explicitLoader);
    return loaderImg(finalSrc, srcSet);
  }

  if (!entry) {
    if (isRemote) {
      // A remote src with the DEFAULT loader (no optimizer here): allow it only if the
      // host is configured, then render the raw src with no srcset (honest — there is no
      // /_next/image server). A disallowed host is a clear hard error, matching Next.
      if (!hasMatch(new URL(rawSrc))) throw hostnameNotConfigured(rawSrc);
      return baseImg();
    }
    if (isRasterPath(rawSrc) && rawSrc.startsWith("/")) {
      throw new Error(
        "next/image: no build-emitted variant manifest entry for raster src '" + rawSrc +
        "'. Put the image under public/ (png/jpeg) so diffpack emits its responsive " +
        "variants at build time, or pass the `unoptimized` prop."
      );
    }
    // Unknown non-remote src: honest passthrough.
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
import {{ PathParamsContext, PathnameContext, SearchParamsContext, SelectedSegmentContext, ServerInsertedHTMLContext }} from {hooks_import};

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
    refresh() {{
      // SOFT RSC refresh: re-fetch the current route's flight and diff-render it (island
      // state survives). Pre-hydration (no client Router yet) falls back to a full reload.
      if (typeof window !== "undefined" && typeof window.__diffpack_refresh === "function") {{
        window.__diffpack_refresh();
      }} else if (typeof window !== "undefined") {{
        window.location.reload();
      }}
    }},
    prefetch(href) {{
      // Warm the client Router's prefetch cache for `href` (same ?__rsc=1 flight a click
      // fetches). No-op before hydration (no Router) or in a non-browser context.
      if (typeof window !== "undefined" && typeof window.__diffpack_prefetch === "function") {{
        window.__diffpack_prefetch(href);
      }}
    }},
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

// The active URL segments below the calling layout, provided by the SEGMENT_BOUNDARY island
// wrapped around each layout in the react-server render (parts.slice(level.slotBase)). A
// named parallelRouteKey (parallel-route slots) is NOT supported by this adapter — it throws
// a CLEAR error rather than silently returning the primary segment.
export function useSelectedLayoutSegments(parallelRouteKey) {{
  if (parallelRouteKey !== undefined) {{
    throw new Error("diffpack next shim: useSelectedLayoutSegments(parallelRouteKey) with a named parallel-route slot is not supported by this adapter");
  }}
  return React.useContext(SelectedSegmentContext) || [];
}}

export function useSelectedLayoutSegment(parallelRouteKey) {{
  if (parallelRouteKey !== undefined) {{
    throw new Error("diffpack next shim: useSelectedLayoutSegment(parallelRouteKey) with a named parallel-route slot is not supported by this adapter");
  }}
  const segments = React.useContext(SelectedSegmentContext) || [];
  return segments.length ? segments[0] : null;
}}

// useServerInsertedHTML: register a callback (returning a React node) the SSR entry renders
// into the streamed HTML (CSS-in-JS registries). The context value is the SSR entry's
// per-request push; on the client it is null, so the hook is a no-op there.
export function useServerInsertedHTML(callback) {{
  const addInsertedHTML = React.useContext(ServerInsertedHTMLContext);
  if (addInsertedHTML) addInsertedHTML(callback);
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
import {{ requestAls, DRAFT_SECRET }} from {request_import};
import {{ createHmac, randomBytes, timingSafeEqual }} from "node:crypto";

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

// Serialize a single Set-Cookie header value from a name/value + Next's CookieOptions
// (path, maxAge, expires, domain, httpOnly, secure, sameSite, priority, partitioned).
// One lean native serializer — no `cookie` / `@edge-runtime/cookies` dependency — shared
// by cookies().set()/delete() and draftMode() so every response emits identical output.
function serializeSetCookie(name, value, o) {{
  o = o || {{}};
  const parts = [name + "=" + encodeURIComponent(value == null ? "" : String(value))];
  if (o.path != null) parts.push("Path=" + o.path);
  if (o.maxAge != null) parts.push("Max-Age=" + Math.floor(o.maxAge));
  if (o.expires != null) parts.push("Expires=" + new Date(o.expires).toUTCString());
  if (o.domain != null) parts.push("Domain=" + o.domain);
  if (o.httpOnly) parts.push("HttpOnly");
  if (o.secure) parts.push("Secure");
  if (o.sameSite) {{
    const s = String(o.sameSite);
    parts.push("SameSite=" + (s.charAt(0).toUpperCase() + s.slice(1).toLowerCase()));
  }}
  if (o.priority) {{
    const p = String(o.priority);
    parts.push("Priority=" + (p.charAt(0).toUpperCase() + p.slice(1).toLowerCase()));
  }}
  if (o.partitioned) parts.push("Partitioned");
  return parts.join("; ");
}}

// A cookies().set()/delete() write is only meaningful where the adapter can still emit a
// Set-Cookie: a Server Action, a Route Handler, or a Server Component BEFORE the streaming
// shell flushes. Outside a store, after the shell sealed the store, or in a context with
// no response-cookie channel (middleware), this HARD-ERRORS (repo no-silent-stub rule)
// rather than dropping the write.
function pushResponseCookie(store, serialized, api) {{
  if (!store.responseCookies) {{
    throw new Error("diffpack next shim: " + api + " is only supported in a Server Component, Server Action, or Route Handler");
  }}
  if (store.sealed) {{
    throw new Error("diffpack next shim: " + api + " was called after the response shell already flushed (headers already sent) — set cookies at the top level of a Server Component, or in a Server Action / Route Handler");
  }}
  store.responseCookies.push(serialized);
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
    // Writes: `set(name, value, options)` or `set({{ name, value, ...options }})`.
    set(name, value, options) {{
      const isObj = typeof name === "object" && name !== null;
      const n = isObj ? name.name : name;
      const v = isObj ? name.value : value;
      const o = isObj ? name : (options || {{}});
      pushResponseCookie(store, serializeSetCookie(n, v, o), "cookies().set()");
      return this;
    }},
    // `delete(name)` or `delete({{ name, path, domain }})`: an immediately-expired cookie.
    delete(name) {{
      const isObj = typeof name === "object" && name !== null;
      const n = isObj ? name.name : name;
      const o = isObj ? name : {{}};
      pushResponseCookie(store, serializeSetCookie(n, "", {{ path: o.path, domain: o.domain, maxAge: 0, expires: new Date(0) }}), "cookies().delete()");
      return this;
    }},
  }};
}}

export async function headers() {{
  const store = requestAls.getStore();
  if (!store) {{
    throw Object.assign(new Error("diffpack next shim: headers() was called outside a request context (no AsyncLocalStorage store) — call it inside a Server Component during a render"), {{ digest: "DIFFPACK_DYNAMIC_BAILOUT" }});
  }}
  return store.headers;
}}

// Draft mode rides Next's real `__prerender_bypass` cookie, signed with a per-build
// DRAFT_SECRET (HMAC-SHA256 over a random nonce) so a visitor cannot forge it. Sign and
// verify both run here inside the react-server worker; the orchestrator only forwards the
// cookie header it already forwards. isEnabled lets a Server Component branch on preview.
const DRAFT_COOKIE = "__prerender_bypass";
function signDraftToken() {{
  const nonce = randomBytes(16).toString("hex");
  const mac = createHmac("sha256", DRAFT_SECRET).update(nonce).digest("hex");
  return nonce + "." + mac;
}}
function verifyDraftToken(token) {{
  if (typeof token !== "string") return false;
  const dot = token.indexOf(".");
  if (dot === -1) return false;
  const nonce = token.slice(0, dot);
  const mac = token.slice(dot + 1);
  const expected = createHmac("sha256", DRAFT_SECRET).update(nonce).digest("hex");
  if (mac.length !== expected.length) return false;
  try {{
    return timingSafeEqual(Buffer.from(mac), Buffer.from(expected));
  }} catch {{
    return false;
  }}
}}

export async function draftMode() {{
  const store = requestAls.getStore();
  if (!store) {{
    throw Object.assign(new Error("diffpack next shim: draftMode() was called outside a request context (no AsyncLocalStorage store) — call it inside a Server Component during a render, a Server Action, or a Route Handler"), {{ digest: "DIFFPACK_DYNAMIC_BAILOUT" }});
  }}
  const token = parseCookieHeader(store.cookieHeader).get(DRAFT_COOKIE);
  const isEnabled = verifyDraftToken(token);
  // httpOnly + Path=/ + SameSite=None; Secure so it survives a cross-site preview
  // navigation, mirroring Next's preview cookie attributes.
  const draftOpts = {{ path: "/", httpOnly: true, secure: true, sameSite: "none" }};
  return {{
    isEnabled,
    enable() {{
      pushResponseCookie(store, serializeSetCookie(DRAFT_COOKIE, signDraftToken(), draftOpts), "draftMode().enable()");
    }},
    disable() {{
      pushResponseCookie(store, serializeSetCookie(DRAFT_COOKIE, "", {{ ...draftOpts, maxAge: 0, expires: new Date(0) }}), "draftMode().disable()");
    }},
  }};
}}
"#,
    )
}

/// The `next/cache` shim (`shims/cache.ts`): on-demand cache invalidation
/// (`revalidatePath` / `revalidateTag`) plus `unstable_cache`. Semantics follow Next's
/// `next/cache` (`next/dist/server/web/spec-extension/revalidate-path` /
/// `revalidate-tag` / `unstable-cache`), reimplemented natively — Next's versions ride
/// its heavy incremental-cache + tag manifest runtime, exactly the per-request cost this
/// adapter avoids.
///
/// `revalidatePath` / `revalidateTag` COLLECT invalidations into the per-request
/// `requestAls` store (the SAME `AsyncLocalStorage` `next/headers` reads). The worker
/// returns `store.revalidated` on its action / route reply; the orchestrator maps tags
/// to the concrete cached pathnames (captured per page at prerender time in
/// `prerender-manifest.json`) and marks those entries stale, so its existing
/// stale-while-revalidate machinery regenerates them in the background. NOTHING runs on a
/// cache-hit request. Called with NO store, each HARD-ERRORS naming the missing context
/// (repo no-silent-stub rule) rather than silently no-op'ing.
///
/// `unstable_cache` is a lean per-worker memo (expiry + tag purge) that also records its
/// tags into `store.tags` so a tagged page is registered under those tags at prerender
/// time. Because the default worker pool is a single warm process, `revalidateTag` purges
/// the local memo synchronously — the same worker then regenerates the page and
/// recomputes the cached value, with no orchestrator broadcast needed.
fn next_cache_shim(request_context: &Path) -> String {
    let request_import = js_str(&request_context.to_string_lossy());
    format!(
        r#"// `next/cache` shim (diffpack next app-router adapter). revalidatePath /
// revalidateTag collect on-demand cache invalidations into the per-request
// AsyncLocalStorage store the react-server render / action / route establishes; the
// orchestrator reads them off the worker reply and marks the matching prerendered cache
// entries stale (its existing stale-while-revalidate machinery then regenerates them in
// the background — zero hot-path cost). unstable_cache is a lean per-worker memo
// (expiry + tag purge) that also records its tags into store.tags so a tagged page is
// registered under those tags at prerender time. Semantics follow Next's next/cache
// revalidatePath / revalidateTag / unstable_cache, natively reimplemented (Next's ride a
// heavy incremental-cache + tag manifest runtime this adapter deliberately avoids).
import {{ requestAls }} from {request_import};

// Per-worker unstable_cache memo. Module-global → shared by the render AND action module
// instances in the same warm worker, so a revalidateTag during an action purges the value
// a subsequent re-render would otherwise reuse. key -> {{ value, expires|null, tags:[] }}.
const __unstableCacheMemo = new Map();

function requireStore(api) {{
  const store = requestAls.getStore();
  if (!store) {{
    throw new Error(
      "diffpack next shim: " + api + " was called outside a request context (no " +
        "AsyncLocalStorage store) — call it inside a Server Action, Route Handler, or " +
        "during a render",
    );
  }}
  if (!store.revalidated) store.revalidated = {{ tags: new Set(), paths: new Set() }};
  if (!store.tags) store.tags = new Set();
  return store;
}}

// revalidatePath(path, type?): invalidate a prerendered page (or, with type "layout",
// its whole subtree). The raw path + type is recorded as `<type>:<path>`; the
// orchestrator maps it to concrete cached pathnames (exact for a page, prefix for a
// layout / dynamic route). Mirrors next/cache revalidatePath.
export function revalidatePath(path, type) {{
  if (typeof path !== "string" || !path) {{
    throw new Error("diffpack next shim: revalidatePath(path) requires a non-empty string path");
  }}
  const kind = type === "layout" ? "layout" : "page";
  requireStore("revalidatePath").revalidated.paths.add(kind + ":" + path);
}}

// revalidateTag(tag): invalidate every prerendered page that read `tag` (via
// unstable_cache tags or a tagged fetch). Also purges THIS worker's unstable_cache memo
// so the background re-render recomputes. Mirrors next/cache revalidateTag.
export function revalidateTag(tag) {{
  if (typeof tag !== "string" || !tag) {{
    throw new Error("diffpack next shim: revalidateTag(tag) requires a non-empty string tag");
  }}
  requireStore("revalidateTag").revalidated.tags.add(tag);
  for (const [key, entry] of __unstableCacheMemo) {{
    if (entry.tags && entry.tags.indexOf(tag) !== -1) __unstableCacheMemo.delete(key);
  }}
}}

// unstable_cache(fn, keyParts?, options?): memoize an async function per worker, keyed by
// keyParts + arguments. `options.tags` register the entry (and the current page) under
// those tags for revalidateTag; `options.revalidate` (seconds) is a soft TTL. Mirrors
// next/cache unstable_cache (a lean local memo, not Next's filesystem incremental cache).
export function unstable_cache(fn, keyParts, options) {{
  if (typeof fn !== "function") {{
    throw new Error("diffpack next shim: unstable_cache(fn, keyParts?, options?) requires a function");
  }}
  const opts = options || {{}};
  const tags = Array.isArray(opts.tags) ? opts.tags.slice() : [];
  const revalidate = typeof opts.revalidate === "number" ? opts.revalidate : null;
  const base = (Array.isArray(keyParts) ? keyParts : []).join(":");
  return async function (...args) {{
    // Register the tags on the current render/action store so a tagged page is
    // discoverable by the orchestrator (best-effort: outside a store this is a plain memo).
    const store = requestAls.getStore();
    if (store) {{
      if (!store.tags) store.tags = new Set();
      for (const t of tags) store.tags.add(t);
    }}
    const key = base + "|" + JSON.stringify(args) + "|" + tags.join(",");
    const now = Date.now();
    const hit = __unstableCacheMemo.get(key);
    if (hit && (hit.expires == null || hit.expires > now)) return hit.value;
    const value = await fn(...args);
    __unstableCacheMemo.set(key, {{
      value,
      expires: revalidate != null ? now + revalidate * 1000 : null,
      tags,
    }});
    return value;
  }};
}}

// cacheTag / cacheLife belong to the "use cache" directive slice, which diffpack's Next
// adapter has NOT implemented (the build hard-errors on a "use cache" module rather than
// silently dropping it). Exported here so an import resolves, but calling them hard-errors
// (repo no-silent-stub rule) — use unstable_cache({{ tags, revalidate }}) instead.
export function cacheTag() {{
  throw new Error(
    "diffpack next shim: cacheTag() (the \"use cache\" directive family) is not implemented; " +
      "use unstable_cache(fn, keyParts, {{ tags }}) for tag-based caching",
  );
}}
export function cacheLife() {{
  throw new Error(
    "diffpack next shim: cacheLife() (the \"use cache\" directive family) is not implemented; " +
      "use unstable_cache(fn, keyParts, {{ revalidate }}) for a TTL",
  );
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
        assert_eq!(kind_of("/slow"), RouteKind::Dynamic, "/slow is force-dynamic → Dynamic");
        // ISR: `export const revalidate = 2` on an otherwise-static route → Isr, and the
        // parsed TTL is carried through for the prerender plan / orchestrator.
        assert_eq!(kind_of("/isr"), RouteKind::Isr, "/isr has revalidate=2 → Isr");
        let isr = disc.routes.iter().find(|r| r.url_path == "/isr").unwrap();
        assert_eq!(isr.revalidate_seconds, Some(2), "/isr carries its revalidate TTL in seconds");

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
    fn parallel_routes_become_layout_slot_props() {
        // The fixture's /dashboard hosts @team and @analytics parallel slots. Discovery
        // must attach them to the dashboard directory's Level (not as separate routes),
        // and the generated react-server entry must compose them as named layout props.
        let fixture = Path::new(env!("CARGO_MANIFEST_DIR")).join("integration/next-app-router");
        let app = fixture.join("app");
        let layout = first_existing(&app, "layout");
        let disc = discover_routes(&app, layout.as_deref()).unwrap();

        // @slot dirs never become their own routes.
        assert!(
            !disc.routes.iter().any(|r| r.url_path.contains('@')),
            "no @slot route leaked into the primary table: {:?}",
            disc.routes.iter().map(|r| &r.url_path).collect::<Vec<_>>()
        );

        let dashboard = disc.routes.iter().find(|r| r.url_path == "/dashboard").expect("/dashboard route");
        // The dashboard-directory level (part_offset 1 = the "dashboard" segment) carries
        // the two slots; the team slot has a page and no default, analytics has a default.
        let level = dashboard
            .levels
            .iter()
            .find(|l| !l.slots.is_empty())
            .expect("a level hosts the @team/@analytics slots");
        assert_eq!(level.part_offset, 1, "dashboard level consumed one URL segment above its slots");
        let names: Vec<&str> = level.slots.iter().map(|s| s.name.as_str()).collect();
        assert!(names.contains(&"team") && names.contains(&"analytics"), "slots: {names:?}");
        let team = level.slots.iter().find(|s| s.name == "team").unwrap();
        assert!(!team.routes.is_empty() && team.default.is_none(), "team slot has a page, no default");
        let analytics = level.slots.iter().find(|s| s.name == "analytics").unwrap();
        assert!(analytics.default.is_some(), "analytics slot has a default.tsx fallback");

        // Codegen: the react-server entry emits the slot tables + the matcher/composer.
        let boundary = fixture.join(".diffpack-next/error-boundary.tsx");
        let seg_boundary = fixture.join(".diffpack-next/segment-boundary.tsx");
        let reqctx = fixture.join(".diffpack-next/request-context.ts");
        let rsc_src = rsc_entry_module(&disc, "", false, &boundary, &seg_boundary, &reqctx, None, "");
        assert!(rsc_src.contains("slotBase:"), "levels carry slotBase: {rsc_src}");
        assert!(rsc_src.contains(r#"name: "team""#) && rsc_src.contains(r#"name: "analytics""#), "slot tables emitted");
        assert!(rsc_src.contains("function matchSlots"), "the slot matcher is generated");
        assert!(rsc_src.contains("function composeLevels"), "the slot composer is generated");
        assert!(rsc_src.contains("...slotProps"), "matched slots are spread as layout props");
    }

    #[test]
    fn metadata_api_chain_and_resolver_codegen() {
        let fixture = Path::new(env!("CARGO_MANIFEST_DIR")).join("integration/next-app-router");
        let app = fixture.join("app");
        let layout = first_existing(&app, "layout");
        let disc = discover_routes(&app, layout.as_deref()).unwrap();
        let boundary = fixture.join(".diffpack-next/error-boundary.tsx");
        let seg_boundary = fixture.join(".diffpack-next/segment-boundary.tsx");
        let reqctx = fixture.join(".diffpack-next/request-context.ts");
        let rsc_src = rsc_entry_module(&disc, "", false, &boundary, &seg_boundary, &reqctx, None, "");
        // Each route carries a metadata namespace chain resolved at render time.
        assert!(rsc_src.contains("metaChain: ["), "routes carry a metadata chain: {rsc_src}");
        assert!(rsc_src.contains("async function resolveMetadata"), "the metadata resolver is generated");
        assert!(rsc_src.contains("async function MetadataHead"), "the async MetadataHead component is generated");
        assert!(rsc_src.contains("function mergeMetadata"), "metadata merge (title templates) is generated");
        // Full head coverage: openGraph, twitter, robots, canonical, viewport.
        for marker in ["og:", "twitter:", "\"robots\"", "canonical", "\"viewport\"", "theme-color"] {
            assert!(rsc_src.contains(marker), "metadata head covers {marker}: missing");
        }
        // module_exports_metadata detects the various export forms.
        assert!(!module_exports_metadata(&app.join("Counter.tsx")), "a plain island exports no metadata");
    }

    #[test]
    fn metadata_file_conventions_synthesize_route_handlers() {
        let root = scratch("meta-files");
        let app = root.join("app");
        std::fs::create_dir_all(&app).unwrap();
        std::fs::write(app.join("layout.tsx"), "export default function L({children}){return children}\n").unwrap();
        std::fs::write(app.join("page.tsx"), "export default function P(){return null}\n").unwrap();
        std::fs::write(app.join("sitemap.ts"), "export default function sitemap(){ return [{ url: \"https://x.com\" }]; }\n").unwrap();
        std::fs::write(app.join("robots.ts"), "export default function robots(){ return { rules: { userAgent: \"*\", allow: \"/\" } }; }\n").unwrap();
        std::fs::write(app.join("manifest.ts"), "export default function manifest(){ return { name: \"X\" }; }\n").unwrap();
        let shims = root.join(".diffpack-next/shims");
        std::fs::create_dir_all(&shims).unwrap();

        let handlers = synthesize_metadata_file_handlers(&app, &shims).unwrap();
        let urls: Vec<&str> = handlers.iter().map(|h| h.url_path.as_str()).collect();
        assert!(urls.contains(&"/sitemap.xml"), "sitemap handler synthesized: {urls:?}");
        assert!(urls.contains(&"/robots.txt"), "robots handler synthesized: {urls:?}");
        assert!(urls.contains(&"/manifest.webmanifest"), "manifest handler synthesized: {urls:?}");
        for h in &handlers {
            assert_eq!(h.methods, vec!["GET".to_string()], "convention handlers are GET-only");
            assert!(h.file.exists(), "the wrapper file was written: {:?}", h.file);
        }
        // The wrappers set the right content-type + call the user export.
        let sitemap_wrapper = std::fs::read_to_string(shims.join("metadata-sitemap.ts")).unwrap();
        assert!(sitemap_wrapper.contains("application/xml"), "sitemap wrapper serves XML");
        assert!(sitemap_wrapper.contains("serializeSitemap"), "sitemap wrapper serializes");
        let robots_wrapper = std::fs::read_to_string(shims.join("metadata-robots.ts")).unwrap();
        assert!(robots_wrapper.contains("text/plain"), "robots wrapper serves text");
        let manifest_wrapper = std::fs::read_to_string(shims.join("metadata-manifest.ts")).unwrap();
        assert!(manifest_wrapper.contains("application/manifest+json"), "manifest wrapper serves manifest json");
        assert!(manifest_wrapper.contains("JSON.stringify"), "manifest wrapper JSON-serializes");
        // The shared serializer helper is present.
        assert!(shims.join("metadata-serialize.ts").exists(), "serializer helper written");
    }

    #[test]
    fn generate_sitemaps_hard_errors() {
        let root = scratch("meta-gen-sitemaps");
        let app = root.join("app");
        std::fs::create_dir_all(&app).unwrap();
        std::fs::write(app.join("sitemap.ts"), "export function generateSitemaps(){ return [{ id: 0 }]; }\nexport default function sitemap(){ return []; }\n").unwrap();
        let shims = root.join(".diffpack-next/shims");
        std::fs::create_dir_all(&shims).unwrap();
        let err = synthesize_metadata_file_handlers(&app, &shims).unwrap_err();
        assert!(err.contains("generateSitemaps"), "clear hard error names the unsupported feature: {err}");
    }

    #[test]
    fn static_metadata_images_scanned_and_head_linked() {
        let root = scratch("meta-images");
        let app = root.join("app");
        std::fs::create_dir_all(&app).unwrap();
        std::fs::write(app.join("layout.tsx"), "export default function L({children}){return children}\n").unwrap();
        std::fs::write(app.join("page.tsx"), "export default function P(){return null}\n").unwrap();
        std::fs::write(app.join("icon.png"), [0u8]).unwrap();
        std::fs::write(app.join("favicon.ico"), [0u8]).unwrap();
        std::fs::write(app.join("apple-icon.png"), [0u8]).unwrap();
        std::fs::write(app.join("opengraph-image.jpg"), [0u8]).unwrap();
        std::fs::write(app.join("twitter-image.png"), [0u8]).unwrap();

        let images = scan_metadata_images(&app).unwrap();
        let served: Vec<&str> = images.iter().map(|i| i.served.as_str()).collect();
        for want in ["/favicon.ico", "/icon.png", "/apple-icon.png", "/opengraph-image.jpg", "/twitter-image.png"] {
            assert!(served.contains(&want), "image {want} discovered: {served:?}");
        }

        // The react-server entry emits the head links for every route.
        let disc = discover_routes(&app, first_existing(&app, "layout").as_deref()).unwrap();
        let boundary = root.join(".diffpack-next/error-boundary.tsx");
        let seg_boundary = root.join(".diffpack-next/segment-boundary.tsx");
        let reqctx = root.join(".diffpack-next/request-context.ts");
        let rsc_src = rsc_entry_module(&disc, "", false, &boundary, &seg_boundary, &reqctx, None, "");
        assert!(rsc_src.contains(r#"rel: "icon", href: "/icon.png""#), "icon link emitted: {rsc_src}");
        assert!(rsc_src.contains(r#"rel: "apple-touch-icon", href: "/apple-icon.png""#), "apple-touch-icon emitted");
        assert!(rsc_src.contains(r#"property: "og:image", content: "/opengraph-image.jpg""#), "og:image emitted");
        assert!(rsc_src.contains(r#"name: "twitter:image", content: "/twitter-image.png""#), "twitter:image emitted");
        assert!(rsc_src.contains(r#"rel: "icon", href: "/favicon.ico", type: "image/x-icon", sizes: "any""#), "favicon emitted");
    }

    #[test]
    fn code_based_image_generator_hard_errors() {
        let root = scratch("meta-image-gen");
        let app = root.join("app");
        std::fs::create_dir_all(&app).unwrap();
        std::fs::write(app.join("opengraph-image.tsx"), "export default function OG(){ return null; }\n").unwrap();
        let err = scan_metadata_images(&app).unwrap_err();
        assert!(err.contains("code-based image generator"), "clear hard error for dynamic image gen: {err}");
        assert!(err.contains("opengraph-image.tsx"), "error names the file: {err}");
    }

    #[test]
    fn intercepting_routes_target_and_overlay() {
        // The fixture's app/gallery/@modal/(.)photo/[id] intercepts /gallery/photo/[id].
        let fixture = Path::new(env!("CARGO_MANIFEST_DIR")).join("integration/next-app-router");
        let app = fixture.join("app");
        let layout = first_existing(&app, "layout");
        let disc = discover_routes(&app, layout.as_deref()).unwrap();

        // The intercept resolves its target RELATIVE to the marker's URL level: `(.)` at
        // gallery/@modal -> /gallery/photo/[id] (not /photo/[id]).
        let ic = disc
            .intercepts
            .iter()
            .find(|i| segments_display(&i.target_segments) == "/gallery/photo/[id]")
            .unwrap_or_else(|| panic!("gallery intercept not found: {:?}", disc.intercepts.iter().map(|i| segments_display(&i.target_segments)).collect::<Vec<_>>()));
        assert!(ic.page.to_string_lossy().contains("@modal"), "overlay page is the @modal intercept: {:?}", ic.page);
        // The full /gallery/photo/[id] route also exists (hard load renders the real page).
        assert!(disc.routes.iter().any(|r| r.url_path == "/gallery/photo/[id]"), "the real photo route exists for hard loads");

        // Codegen: the react-server entry emits INTERCEPTS + softNav-gated matching; the
        // client Router portals the overlay and masks the URL.
        let boundary = fixture.join(".diffpack-next/error-boundary.tsx");
        let seg_boundary = fixture.join(".diffpack-next/segment-boundary.tsx");
        let reqctx = fixture.join(".diffpack-next/request-context.ts");
        let rsc_src = rsc_entry_module(&disc, "", false, &boundary, &seg_boundary, &reqctx, None, "");
        assert!(rsc_src.contains("const INTERCEPTS = ["), "INTERCEPTS table emitted: {rsc_src}");
        assert!(rsc_src.contains("function matchIntercept"), "intercept matcher generated");
        assert!(rsc_src.contains("opts.softNav"), "intercept only on soft-nav");

        let islands = [app.join("Counter.tsx")];
        let hooks = fixture.join(".diffpack-next/hooks-context.ts");
        let client_src = client_entry_module(&fixture.join(".diffpack-next"), &islands, &hooks);
        assert!(client_src.contains("x-diffpack-intercept"), "client reads the intercept header");
        assert!(client_src.contains("createPortal"), "client portals the overlay over the page");
        assert!(client_src.contains("__diffpackModal"), "client masks the URL for the overlay");
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
        assert!(rs_src.contains("function documentTree(pathname, opts)"), "{rs_src}");
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
        // Streaming SSR: the entry also exports the streaming renderer and inlines the
        // flight incrementally as __DF_FLIGHT scripts (so the shell can flush first).
        assert!(ssr_src.contains("export async function renderFlightToStream"), "ssr entry exports the streaming renderer: {ssr_src}");
        assert!(ssr_src.contains("onShellReady"), "streaming SSR flushes at onShellReady: {ssr_src}");
        assert!(ssr_src.contains("__DF_FLIGHT"), "streaming SSR inlines the flight incrementally: {ssr_src}");
        // The client reconstructs the flight from the incremental __DF_FLIGHT stream.
        assert!(client_src.contains("flightStreamFromDF"), "client rebuilds flight from the __DF_FLIGHT stream: {client_src}");
        // The worker exposes the streaming render op end-to-end.
        assert!(rs_src.contains("export async function renderRequestStream"), "rsc-entry exports the streaming render: {rs_src}");
        assert!(rs_src.contains("render-stream"), "the serve worker handles the render-stream op: {rs_src}");
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
        // Draft mode + server-side cookie writes: cookies() exposes set()/delete() pushing
        // onto the store's response-cookie channel, and draftMode() signs/verifies the real
        // __prerender_bypass cookie against the baked DRAFT_SECRET (never the always-throws
        // "not supported" stub). The request-context module bakes the secret.
        assert!(hdr.contains("serializeSetCookie"), "headers shim carries a native Set-Cookie serializer: {hdr}");
        assert!(hdr.contains("pushResponseCookie(store"), "cookies().set()/delete() push onto the response-cookie channel: {hdr}");
        assert!(hdr.contains("cookies().set()") && hdr.contains("cookies().delete()"), "cookies() gains set()/delete(): {hdr}");
        assert!(hdr.contains("__prerender_bypass"), "draftMode() uses Next's real bypass cookie name: {hdr}");
        assert!(hdr.contains("DRAFT_SECRET") && hdr.contains("createHmac"), "draftMode() HMAC-signs the bypass token with the baked secret: {hdr}");
        assert!(!hdr.contains("draftMode().enable() is not supported"), "draftMode().enable() is implemented, not a throwing stub: {hdr}");
        assert!(req_ctx.contains("DRAFT_SECRET"), "request-context bakes the draft secret: {req_ctx}");
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

        // --- Navigation completeness cluster ----------------------------------------
        // hooks-context exports the two new contexts under the createContext guard.
        assert!(hooks_ctx.contains("SelectedSegmentContext = createContext(null)"), "hooks-context exports SelectedSegmentContext: {hooks_ctx}");
        assert!(hooks_ctx.contains("ServerInsertedHTMLContext = createContext(null)"), "hooks-context exports ServerInsertedHTMLContext: {hooks_ctx}");
        // The next/navigation shim exports the segment hooks + useServerInsertedHTML, and
        // useRouter().refresh/prefetch route through the client Router (NOT location.reload).
        assert!(nav.contains("export function useSelectedLayoutSegment("), "nav shim exports useSelectedLayoutSegment: {nav}");
        assert!(nav.contains("export function useSelectedLayoutSegments("), "nav shim exports useSelectedLayoutSegments: {nav}");
        assert!(nav.contains("export function useServerInsertedHTML("), "nav shim exports useServerInsertedHTML: {nav}");
        assert!(nav.contains("window.__diffpack_refresh"), "useRouter().refresh soft-refreshes via the Router: {nav}");
        assert!(!nav.contains("window.location.reload()") || nav.contains("__diffpack_refresh"), "refresh prefers soft refresh over reload: {nav}");
        assert!(nav.contains("window.__diffpack_prefetch"), "useRouter().prefetch warms the prefetch cache: {nav}");
        assert!(nav.contains("not supported by this adapter"), "a named parallelRouteKey throws a clear error (no silent default): {nav}");
        // The rsc-entry wraps each layout in the SEGMENT_BOUNDARY island carrying the
        // active child segments (parts.slice(level.slotBase)).
        assert!(rs_src.contains("const SEGMENT_BOUNDARY ="), "rsc-entry interns the SEGMENT_BOUNDARY island: {rs_src}");
        assert!(rs_src.contains("SEGMENT_BOUNDARY,") && rs_src.contains("segments: parts.slice(level.slotBase)"), "rsc-entry wraps layouts in SEGMENT_BOUNDARY with the active segments: {rs_src}");
        // The segment-boundary island is pinned into the client + ssr graphs.
        assert!(
            client_src.contains("segment-boundary.tsx") || client_src.contains("segment-boundary"),
            "segment-boundary island pinned into the client graph: {client_src}"
        );
        assert!(
            ssr_src.contains("segment-boundary.tsx") || ssr_src.contains("segment-boundary"),
            "segment-boundary island pinned into the ssr graph: {ssr_src}"
        );
        let seg_mod = std::fs::read_to_string(adapter.join("segment-boundary.tsx")).unwrap();
        assert!(seg_mod.contains("SelectedSegmentContext.Provider"), "segment boundary provides SelectedSegmentContext: {seg_mod}");
        assert!(seg_mod.starts_with("\"use client\""), "segment boundary is a client island: {seg_mod}");
        // The client entry has the bounded prefetch cache + exposes prefetch/refresh.
        assert!(client_src.contains("prefetchCache"), "client entry has a prefetch cache: {client_src}");
        assert!(client_src.contains("window.__diffpack_prefetch"), "client entry exposes __diffpack_prefetch: {client_src}");
        assert!(client_src.contains("window.__diffpack_refresh"), "client entry exposes __diffpack_refresh: {client_src}");
        // The ssr entry provides ServerInsertedHTMLContext + flushes via renderToStaticMarkup.
        assert!(ssr_src.contains("renderToStaticMarkup"), "ssr entry imports renderToStaticMarkup for inserted HTML: {ssr_src}");
        assert!(ssr_src.contains("ServerInsertedHTMLContext.Provider"), "ssr entry provides ServerInsertedHTMLContext: {ssr_src}");
        assert!(ssr_src.contains("</head>"), "ssr buffered path splices inserted HTML before </head>: {ssr_src}");
        // next/dynamic is aliased and its shim exists (React.lazy + ssr:false mounted-gate).
        let dyn_target = aliased.get("next/dynamic").expect("next/dynamic aliased");
        assert!(Path::new(dyn_target).is_file(), "next/dynamic shim file exists");
        let dyn_shim = std::fs::read_to_string(dyn_target).unwrap();
        assert!(dyn_shim.contains("export default function dynamic("), "dynamic shim exports dynamic(): {dyn_shim}");
        assert!(dyn_shim.contains("opts.ssr === false"), "dynamic shim honors ssr:false: {dyn_shim}");
        assert!(dyn_shim.contains("lazy(toLoadable"), "dynamic shim backs on React.lazy: {dyn_shim}");
        // next/link wires prefetch on hover/focus.
        let link = std::fs::read_to_string(adapter.join("shims").join("link.tsx")).unwrap();
        assert!(link.contains("__diffpack_prefetch"), "link shim warms the prefetch cache: {link}");
        assert!(link.contains("onMouseEnter: handleMouseEnter"), "link shim prefetches on hover: {link}");

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

    #[test]
    fn image_config_module_serializes_remote_and_loader() {
        let images = serde_json::json!({
            "deviceSizes": null, "imageSizes": null,
            "remotePatterns": [{ "protocol": "https", "hostname": "**.example.com" }],
            "domains": ["cdn.example.org"], "loader": "imgix",
            "loaderFile": "/abs/loader.js", "path": "/_next/image",
            "qualities": null, "unoptimized": false
        });
        let module = image_config_module(&images);
        assert!(module.contains(r#"import __loaderFile from "/abs/loader.js";"#), "loaderFile imported: {module}");
        assert!(module.contains(r#""hostname":"**.example.com""#), "remotePatterns serialized: {module}");
        assert!(module.contains(r#""cdn.example.org""#), "domains serialized: {module}");
        assert!(module.contains(r#"loader: "imgix""#), "named loader: {module}");
        assert!(module.contains("loaderFn: __loaderFile"), "loaderFn wired to the imported file: {module}");
    }

    #[test]
    fn image_config_defaults_when_no_loader_file() {
        let module = image_config_module(&default_images_json());
        assert!(!module.contains("import __loaderFile"), "no loaderFile import when none set: {module}");
        assert!(module.contains("loaderFn: null"), "loaderFn null by default: {module}");
        assert!(module.contains("remotePatterns: []"), "empty remote allow-list by default: {module}");
    }

    #[test]
    fn routing_extraction_from_eval() {
        // A full eval carries basePath/assetPrefix; Routing pulls exactly those, and
        // asset_base() composes assetPrefix + basePath for static-asset URLs.
        let eval = serde_json::json!({
            "redirects": [], "rewrites": [], "headers": [],
            "basePath": "/docs", "assetPrefix": "/cdn",
            "trailingSlash": true, "i18n": null,
        });
        let r = Routing::from_eval(Some(&eval));
        assert_eq!(r.base_path, "/docs");
        assert_eq!(r.asset_prefix, "/cdn");
        assert_eq!(r.asset_base(), "/cdn/docs", "assets sit under assetPrefix then basePath");

        // basePath alone: asset_base is just the basePath.
        let base_only = serde_json::json!({ "basePath": "/docs", "assetPrefix": "" });
        let r2 = Routing::from_eval(Some(&base_only));
        assert_eq!(r2.asset_base(), "/docs");

        // No config at all: every prefix empty, asset_base is "" (URLs stay `/client.js`).
        let empty = Routing::from_eval(None);
        assert_eq!(empty.base_path, "");
        assert_eq!(empty.asset_base(), "");
    }

    #[test]
    fn next_link_shim_bakes_base_path_and_prefixes_internal_hrefs() {
        // With a basePath, the shim bakes the const, defines withBasePath, and routes the
        // rendered href + soft-nav target through it (so both carry the prefix).
        let shim = next_link_shim("/docs");
        assert!(shim.starts_with("\"use client\";"), "use client stays first: {shim}");
        assert!(shim.contains(r#"const BASE_PATH = "/docs";"#), "basePath baked as a const: {shim}");
        assert!(shim.contains("function withBasePath"), "the prefix helper is generated");
        assert!(shim.contains("const resolved = withBasePath(rawHref);"), "href routed through withBasePath");
        assert!(shim.contains("href.startsWith(BASE_PATH + \"/\")"), "no double-prefix guard present");

        // No basePath: the const is empty, so withBasePath is an identity (href unchanged).
        let plain = next_link_shim("");
        assert!(plain.contains(r#"const BASE_PATH = "";"#), "empty basePath const: {plain}");
    }

    #[test]
    fn ssr_entry_bakes_asset_base_into_bootstrap_modules() {
        let dir = scratch("ssr-asset-base");
        let hooks = dir.join("hooks-context.ts");
        // With an asset base the browser bootstrap is fetched under the prefix.
        let with_prefix = ssr_entry_module(&dir, &[], &hooks, "/cdn/docs");
        assert!(
            with_prefix.contains(r#"bootstrapModules: ["/cdn/docs/client.js"]"#),
            "bootstrapModules carry the asset base (both render paths): {with_prefix}",
        );
        assert_eq!(
            with_prefix.matches(r#"bootstrapModules: ["/cdn/docs/client.js"]"#).count(),
            2,
            "both the buffered and streaming render paths are prefixed",
        );
        // Empty asset base keeps the bare `/client.js`.
        let plain = ssr_entry_module(&dir, &[], &hooks, "");
        assert!(plain.contains(r#"bootstrapModules: ["/client.js"]"#), "no prefix -> bare client.js");
    }

    #[test]
    fn rsc_entry_prefixes_stylesheet_href() {
        let fixture = Path::new(env!("CARGO_MANIFEST_DIR")).join("integration/next-app-router");
        let app = fixture.join("app");
        let layout = first_existing(&app, "layout");
        let disc = discover_routes(&app, layout.as_deref()).unwrap();
        let boundary = fixture.join(".diffpack-next/error-boundary.tsx");
        let seg_boundary = fixture.join(".diffpack-next/segment-boundary.tsx");
        let reqctx = fixture.join(".diffpack-next/request-context.ts");
        // has_css=true with an asset base: the stylesheet <link> href carries the prefix.
        let src = rsc_entry_module(&disc, "", true, &boundary, &seg_boundary, &reqctx, None, "/docs");
        assert!(src.contains(r#"href: "/docs/rsc.css""#), "stylesheet href prefixed by basePath: {src}");
    }

    #[test]
    fn config_manifest_round_trips_routing_surface() {
        let dir = scratch("config-manifest");
        let eval = serde_json::json!({
            "redirects": [], "rewrites": [], "headers": [],
            "basePath": "/docs", "assetPrefix": "", "trailingSlash": true,
            "i18n": { "locales": ["en", "fr"], "defaultLocale": "en" },
        });
        write_next_config_manifest(&dir, Some(&eval));
        let written = std::fs::read_to_string(dir.join(".diffpack-output/next-config-manifest.json")).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&written).unwrap();
        assert_eq!(parsed["basePath"], "/docs");
        assert_eq!(parsed["trailingSlash"], true);
        assert_eq!(parsed["i18n"]["defaultLocale"], "en");

        // No eval: a well-formed empty manifest (every routing field present so the
        // orchestrator's reader never sees an undefined).
        write_next_config_manifest(&dir, None);
        let empty = std::fs::read_to_string(dir.join(".diffpack-output/next-config-manifest.json")).unwrap();
        let ep: serde_json::Value = serde_json::from_str(&empty).unwrap();
        assert_eq!(ep["basePath"], "");
        assert_eq!(ep["trailingSlash"], false);
        assert!(ep["i18n"].is_null());
    }

    #[test]
    fn image_shim_supports_remote_hosts_and_loaders() {
        let shim = next_image_shim();
        assert!(shim.contains(r#"import CONFIG from "../image-config""#), "shim reads the images config: {shim}");
        assert!(shim.contains("function matchRemotePattern"), "shim ports the remote-pattern matcher");
        assert!(shim.contains("is not configured under"), "shim throws a clear hostname error for a disallowed remote host");
        assert!(shim.contains("imgixLoader") && shim.contains("cloudinaryLoader") && shim.contains("akamaiLoader"), "shim has the built-in loaders");
    }

    #[test]
    fn next_cache_shim_emits_revalidate_and_unstable_cache() {
        // The next/cache shim exports the on-demand revalidation API + unstable_cache,
        // imports the shared requestAls store, collects into store.revalidated/store.tags,
        // and HARD-ERRORS (never silently no-ops) when called outside a request context.
        let ctx = Path::new("/tmp/.diffpack-next/request-context.ts");
        let shim = next_cache_shim(ctx);
        // Imports the SAME per-request store next/headers reads (collection hook).
        assert!(shim.contains("import { requestAls }"), "cache shim imports requestAls: {shim}");
        // The three public next/cache APIs are exported.
        assert!(shim.contains("export function revalidatePath("), "revalidatePath exported");
        assert!(shim.contains("export function revalidateTag("), "revalidateTag exported");
        assert!(shim.contains("export function unstable_cache("), "unstable_cache exported");
        // Collection targets: store.revalidated.paths / .tags and store.tags.
        assert!(shim.contains("revalidated.paths.add"), "revalidatePath writes store.revalidated.paths");
        assert!(shim.contains("revalidated.tags.add"), "revalidateTag writes store.revalidated.tags");
        assert!(shim.contains("store.tags.add"), "unstable_cache registers its tags on the page store");
        // No-silent-stub: missing store hard-errors naming the context.
        assert!(
            shim.contains("was called outside a request context"),
            "cache shim hard-errors with no store: {shim}"
        );
        // The "use cache" family (cacheTag/cacheLife) is exported but hard-errors (the
        // directive slice is unbuilt) rather than silently succeeding.
        assert!(shim.contains("export function cacheTag(") && shim.contains("is not implemented"),
            "cacheTag exported and hard-errors");
        assert!(shim.contains("export function cacheLife("), "cacheLife exported");
        // revalidateTag purges the local unstable_cache memo (single-worker correctness).
        assert!(shim.contains("__unstableCacheMemo.delete"), "revalidateTag purges the worker memo");
    }

    #[test]
    fn next_cache_alias_and_shim_written_by_build() {
        // build_next_app must write shims/cache.ts AND alias next/cache to it (an app
        // importing next/cache resolves the faithful shim, not an unshimmed failure).
        let root = scratch("next-cache-alias");
        std::fs::create_dir_all(root.join("app")).unwrap();
        std::fs::write(root.join("app/page.tsx"), "export default function Page(){return null;}\n").unwrap();
        std::fs::write(root.join("app/layout.tsx"), "export default function L({children}){return children;}\n").unwrap();
        std::fs::write(root.join("next.config.js"), "module.exports = {};\n").unwrap();
        // The `client` environment writes the shims + alias vec without the react-server
        // config-eval node spawn, so the alias wiring is exercised without a child process.
        let cfg = configure(&root, "client").unwrap().unwrap();
        let has_alias = cfg
            .build
            .aliases
            .iter()
            .any(|(spec, file)| spec == "next/cache" && file.ends_with("cache.ts"));
        assert!(has_alias, "next/cache aliased to the shim: {:?}", cfg.build.aliases);
        let shim_path = root.join(".diffpack-next/shims/cache.ts");
        assert!(shim_path.is_file(), "shims/cache.ts written at {}", shim_path.display());
        let contents = std::fs::read_to_string(&shim_path).unwrap();
        assert!(contents.contains("export function revalidateTag("), "written shim has revalidateTag");
    }

    #[test]
    fn use_cache_directive_detected_and_hard_errors() {
        // A "use cache" prologue is recognized as its own directive (never confused with
        // use client/use server) so the build can hard-error instead of silently dropping
        // the module's caching semantics.
        use crate::rsc::{detect_directive, RscDirective};
        let path = Path::new("/tmp/cached.ts");
        assert_eq!(
            detect_directive(path, "\"use cache\";\nexport async function data(){return 1;}\n"),
            Some(RscDirective::Cache),
            "\"use cache\" prologue detected as the Cache directive"
        );
        // The react-server transform of such a module produces a CLEAR diagnostic, not a
        // silent pass-through.
        let result = crate::transform::transform_module(
            path,
            "\"use cache\";\nexport async function data(){return 1;}\n",
            Target::ReactServer,
        );
        assert!(!result.diagnostics.is_empty(), "\"use cache\" module yields a diagnostic");
        assert!(
            result.diagnostics[0].contains("use cache") && result.diagnostics[0].contains("not yet implemented"),
            "diagnostic names the unimplemented directive: {:?}",
            result.diagnostics
        );
    }

    #[test]
    fn scan_route_config_parses_segment_config_exports() {
        let src = "\
export const runtime = \"nodejs\";
export const fetchCache = \"force-no-store\";
export const preferredRegion = \"iad1\";
export const maxDuration = 15;
export const experimental_ppr = true;
export default function Page(){ return null; }
";
        let cfg = scan_route_config(src);
        assert_eq!(cfg.runtime.as_deref(), Some("nodejs"));
        assert_eq!(cfg.fetch_cache.as_deref(), Some("force-no-store"));
        assert_eq!(cfg.preferred_region.as_deref(), Some("iad1"));
        assert_eq!(cfg.max_duration.as_deref(), Some("15"));
        assert_eq!(cfg.experimental_ppr.as_deref(), Some("true"));
        // The nodejs runtime + the advisory configs validate without error (they only WARN).
        assert!(validate_segment_config("/x", &cfg).is_ok());
        // Absent configs stay None.
        let empty = scan_route_config("export default function P(){return null;}\n");
        assert!(empty.runtime.is_none() && empty.fetch_cache.is_none() && empty.max_duration.is_none());
    }

    #[test]
    fn edge_runtime_is_a_hard_error() {
        let src = "export const runtime = \"edge\";\nexport default function Page(){return null;}\n";
        let cfg = scan_route_config(src);
        let err = validate_segment_config("/edgy", &cfg).unwrap_err();
        assert!(
            err.contains("/edgy") && err.contains("edge") && err.contains("not supported"),
            "edge error must name the route + reason: {err}",
        );
        // Discovery surfaces it: a route exporting `runtime = "edge"` fails the build.
        let app = scratch("edge-runtime");
        std::fs::write(app.join("layout.tsx"), "export default function L({children}){return children;}\n").unwrap();
        let edge_dir = app.join("edgy");
        std::fs::create_dir_all(&edge_dir).unwrap();
        std::fs::write(edge_dir.join("page.tsx"), src).unwrap();
        let layout = first_existing(&app, "layout");
        match discover_routes(&app, layout.as_deref()) {
            Err(e) => assert!(e.contains("edge"), "discovery error must name the edge runtime: {e}"),
            Ok(_) => panic!("edge-runtime route must fail discovery"),
        }
    }

    #[test]
    fn template_and_global_error_discovered_and_composed() {
        let app = scratch("template-global-error");
        std::fs::write(app.join("layout.tsx"), "export default function L({children}){return children;}\n").unwrap();
        std::fs::write(app.join("page.tsx"), "export default function P(){return null;}\n").unwrap();
        std::fs::write(
            app.join("template.tsx"),
            "\"use client\";\nexport default function T({children}){return children;}\n",
        ).unwrap();
        std::fs::write(
            app.join("global-error.tsx"),
            "\"use client\";\nexport default function GE({error}){return null;}\n",
        ).unwrap();
        let layout = first_existing(&app, "layout");
        let disc = discover_routes(&app, layout.as_deref()).unwrap();
        assert!(disc.global_error.is_some(), "global-error.tsx discovered at app root");
        let root = disc.routes.iter().find(|r| r.url_path == "/").unwrap();
        assert!(
            root.levels.iter().any(|l| l.template.is_some()),
            "the app-root level carries the template",
        );

        // Codegen: the react-server entry emits the template id, GLOBAL_ERROR const, the
        // pathname remount key, and the global-error boundary wrapping the whole tree.
        let boundary = app.join("error-boundary.tsx");
        std::fs::write(&boundary, error_boundary_module()).unwrap();
        let seg_boundary = app.join("segment-boundary.tsx");
        let reqctx = app.join("request-context.ts");
        std::fs::write(&reqctx, request_context_module()).unwrap();
        let rsc = rsc_entry_module(&disc, "", false, &boundary, &seg_boundary, &reqctx, None, "");
        assert!(rsc.contains("template:"), "levels carry a template id");
        assert!(rsc.contains("const GLOBAL_ERROR ="), "GLOBAL_ERROR const emitted");
        assert!(rsc.contains("key: pathname"), "template is keyed by pathname for remount");
        assert!(rsc.contains("fallback: GLOBAL_ERROR"), "global-error wraps the document tree");
    }

    #[test]
    fn instrumentation_entry_detects_root_and_src() {
        // Root-level instrumentation.ts is found.
        let root = scratch("instrumentation-root");
        std::fs::write(root.join("instrumentation.ts"), "export function register(){}\n").unwrap();
        assert!(instrumentation_entry(&root).is_some(), "root instrumentation.ts detected");
        // A src/ instrumentation.js is found when the root has none.
        let root2 = scratch("instrumentation-src");
        let src = root2.join("src");
        std::fs::create_dir_all(&src).unwrap();
        std::fs::write(src.join("instrumentation.js"), "export function register(){}\n").unwrap();
        assert!(instrumentation_entry(&root2).is_some(), "src/ instrumentation.js detected");
        // No instrumentation file → None.
        let root3 = scratch("instrumentation-none");
        assert!(instrumentation_entry(&root3).is_none(), "no instrumentation → None");
    }
}
