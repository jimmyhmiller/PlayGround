//! Next.js **app-router** adapter — the mapping layer from Next's file conventions
//! onto diffpack's existing RSC spine (Slices A–E).
//!
//! diffpack's RSC machinery (three build graphs: `Target::Client` /
//! `Target::IsolatedServer` / `Target::Server`, two manifests, the `__webpack_*` seam,
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
//! (`app/layout.tsx`), all `"use client"` islands discovered across the PROJECT
//! (they routinely live outside the app dir — `src/components` beside `src/app`), with
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
//! via the development configuration entry point and serves them with
//! state-preserving Fast Refresh for `"use client"` islands and a correct reload for
//! Server-Component edits. Parallel (`@slot`) / intercepting (`(.)`) routes remain the
//! documented remaining gaps (see `docs/RSC_NEXT_GAP.md`).
//!
//! Generated glue lives under `<root>/.diffpack-next/` (gitignored, like the other
//! build outputs). Generating entry/shim source as Rust strings follows the exact
//! precedent of [`crate::rsc::generate_action_resolver_module`] and
//! the equivalent TanStack server-function generator: diffpack-authored build glue, not
//! guest source hidden in a string.

use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::path::{Path, PathBuf};

use rayon::prelude::*;

use crate::rsc::{RscDirective, detect_directive};
use diffpack_core::transform::Target;

pub type AppRouterAppConfig = diffpack_default_loader::driver_config::EnvironmentConfig;

pub type AppRouterBuildConfig = diffpack_default_loader::driver_config::BuildConfig;

/// The directory under the project root where the adapter writes its generated
/// entries and `next/*` shims.
pub use crate::APP_ADAPTER_DIR as ADAPTER_DIR;

/// The full pinned-island list (canonical paths) the last scaffold generated,
/// recorded so [`reconcile_async_islands`] can intersect it with the discovered
/// graph's async closure without re-walking the project.
const ISLANDS_FILE: &str = "islands.json";

/// Which pinned islands are ASYNC (top-level await, directly or transitively),
/// per environment: `{"client": [paths...], "ssr": [paths...]}`. The entry
/// generators pin the UNION eagerly (static `import * as`) and everything else
/// lazily (a never-called `require` thunk). Maintained by
/// [`reconcile_async_islands`] from each graph's real discovery, so it is a
/// recorded build fact; a stale set is caught after discovery and repaired by
/// regenerating the entries and rediscovering once.
const ASYNC_ISLANDS_FILE: &str = "async-islands.json";

/// The islands the REACT-SERVER graph actually references (canonical paths), recorded
/// by the dev server from that graph's own client-references manifest.
///
/// Pinning is what puts an island into the client and SSR graphs so its client reference
/// resolves, and the project walk that discovers islands cannot say whether any route
/// references one — so it pins every `"use client"` file in the tree. On cal.com that is
/// 231 islands, of which the whole app references 101 and a single route references 11;
/// the other pins compile a `"use client"` component (and its imports) into two graphs
/// for a flight that can never mention it. The react-server graph knows the exact set:
/// a `"use client"` module in it IS the client-reference boundary, so its reachable
/// client-directive modules are precisely the references a flight can carry.
///
/// Recorded rather than derived in-process because the three graphs are configured
/// independently; the dev server builds react-server first and writes this, and the
/// client/ssr `configure` passes read it. Absent (a build that has never built the
/// react-server graph — every production build, whose client graph is emitted first)
/// falls back to pinning the walk, which is what the file's absence means: "no better
/// information yet".
const REFERENCED_ISLANDS_FILE: &str = "referenced-islands.json";

/// The recorded [`REFERENCED_ISLANDS_FILE`] set, or `None` when it has never been
/// written (so the caller pins the project walk instead).
fn referenced_islands(adapter_dir: &Path) -> Option<BTreeSet<String>> {
    let text = std::fs::read_to_string(adapter_dir.join(REFERENCED_ISLANDS_FILE)).ok()?;
    serde_json::from_str::<BTreeSet<String>>(&text).ok()
}

/// Record which islands the react-server graph references, from that graph's own
/// client-references manifest (whose keys are exactly the canonical paths of its
/// reachable `"use client"` modules). Returns `true` when the recorded set CHANGED, so
/// the caller knows the client/ssr entries on disk were generated from a stale pin set.
pub fn write_referenced_islands(root: &Path, references_manifest: &Path) -> Result<bool, String> {
    let root = root.canonicalize().unwrap_or_else(|_| root.to_path_buf());
    let adapter_dir = root.join(ADAPTER_DIR);
    let text = std::fs::read_to_string(references_manifest).map_err(|error| {
        format!(
            "cannot read the react-server client-references manifest {}: {error}",
            references_manifest.display(),
        )
    })?;
    let manifest: BTreeMap<String, serde_json::Value> = serde_json::from_str(&text)
        .map_err(|error| format!("cannot parse {}: {error}", references_manifest.display()))?;
    let referenced: BTreeSet<&String> = manifest.keys().collect();
    let text = serde_json::to_string_pretty(&referenced)
        .map_err(|error| format!("cannot serialize the referenced-island set: {error}"))?;
    let path = adapter_dir.join(REFERENCED_ISLANDS_FILE);
    if std::fs::read_to_string(&path).ok().as_deref() == Some(text.as_str()) {
        return Ok(false);
    }
    std::fs::create_dir_all(&adapter_dir)
        .map_err(|error| format!("cannot create {}: {error}", adapter_dir.display()))?;
    std::fs::write(&path, text)
        .map_err(|error| format!("cannot write {}: {error}", path.display()))?;
    Ok(true)
}

/// The recorded eager-island union (all environments). Empty when never
/// recorded — the correct default: a lazily-pinned async island is caught by
/// the reconcile step before emit.
fn recorded_eager_islands(adapter_dir: &Path) -> BTreeSet<String> {
    let Ok(text) = std::fs::read_to_string(adapter_dir.join(ASYNC_ISLANDS_FILE)) else {
        return BTreeSet::new();
    };
    let Ok(map) = serde_json::from_str::<BTreeMap<String, BTreeSet<String>>>(&text) else {
        return BTreeSet::new();
    };
    map.into_values().flatten().collect()
}

/// The pinned-island list (canonical paths) the last scaffold recorded in
/// the adapter's islands file. Empty when the project has never been scaffolded.
pub fn recorded_islands(root: &Path) -> Vec<String> {
    let root = root.canonicalize().unwrap_or_else(|_| root.to_path_buf());
    let Ok(text) = std::fs::read_to_string(root.join(ADAPTER_DIR).join(ISLANDS_FILE)) else {
        return Vec::new();
    };
    serde_json::from_str::<Vec<String>>(&text).unwrap_or_default()
}

/// After a client/ssr graph discovery: intersect the pinned islands with the
/// graph's async closure and reconcile the result against
/// the adapter's async-islands file. Returns `true` when the recorded UNION changed —
/// meaning the entries on disk were generated from a stale eager set, and the
/// caller must re-run `configure` (which rewrites them) and rediscover. A
/// non-app-router project (no recorded island list) always reconciles to
/// unchanged.
pub fn reconcile_async_islands_from_tainted(
    root: &Path,
    environment: &str,
    tainted: &HashSet<String>,
) -> Result<bool, String> {
    let root = root.canonicalize().unwrap_or_else(|_| root.to_path_buf());
    let adapter_dir = root.join(ADAPTER_DIR);
    let islands_path = adapter_dir.join(ISLANDS_FILE);
    let Ok(islands_text) = std::fs::read_to_string(&islands_path) else {
        return Ok(false);
    };
    let islands: Vec<String> = serde_json::from_str(&islands_text)
        .map_err(|error| format!("cannot parse {}: {error}", islands_path.display()))?;
    let needed: BTreeSet<String> = islands
        .into_iter()
        .filter(|island| tainted.contains(island))
        .collect();
    let recorded_path = adapter_dir.join(ASYNC_ISLANDS_FILE);
    let mut map: BTreeMap<String, BTreeSet<String>> = std::fs::read_to_string(&recorded_path)
        .ok()
        .and_then(|text| serde_json::from_str(&text).ok())
        .unwrap_or_default();
    if map.get(environment) == Some(&needed)
        || (!map.contains_key(environment) && needed.is_empty())
    {
        return Ok(false);
    }
    let union_before: BTreeSet<String> = map.values().flatten().cloned().collect();
    map.insert(environment.to_string(), needed);
    let union_after: BTreeSet<String> = map.values().flatten().cloned().collect();
    let serialized = serde_json::to_string_pretty(&map)
        .map_err(|error| format!("cannot serialize the async-island set: {error}"))?;
    let staged = recorded_path.with_extension(format!("staged-{}", std::process::id()));
    std::fs::write(&staged, serialized)
        .map_err(|error| format!("cannot write {}: {error}", staged.display()))?;
    std::fs::rename(&staged, &recorded_path)
        .map_err(|error| format!("cannot publish {}: {error}", recorded_path.display()))?;
    // Only a UNION change invalidates the generated entries (they pin the
    // union); a per-environment shrink that leaves the union intact is recorded
    // but needs no rebuild.
    Ok(union_after != union_before)
}

/// Module-file extensions the adapter recognizes for app-router convention files
/// (layout/loading/error/not-found/route).
const MODULE_EXTS: [&str; 4] = ["tsx", "jsx", "ts", "js"];

/// Extensions a `next.config` may use — Next's own set, and the ONE definition of it in
/// diffpack (detection and the config eval both go through [`next_config_path`]). Note
/// `tsx`/`jsx` are deliberately absent: Next never loads a JSX config file.

/// Extensions a `page` may use — the module set PLUS MDX/Markdown, so `page.mdx` /
/// `page.md` is a route exactly like `page.tsx`. Only `page` is MDX-eligible; the other
/// convention files stay on [`MODULE_EXTS`].
const PAGE_EXTS: [&str; 6] = ["tsx", "jsx", "ts", "js", "mdx", "md"];

/// Every source extension the adapter's page/route discovery can compile (the superset of
/// [`PAGE_EXTS`] plus the CommonJS/ESM variants oxc parses). A next.config `pageExtensions`
/// is HONORED against this set: diffpack discovers pages using this fixed superset, which
/// equals Next's default (`tsx/ts/jsx/js`) plus what `@next/mdx` adds (`md/mdx`) — the
/// standard configuration, and exactly the mdx fixture's. A configured extension OUTSIDE
/// this set (e.g. a custom `.vue` page) means diffpack could never discover that page, so
/// it is a CLEAR hard error rather than a silently-missing route.
const SUPPORTED_PAGE_EXTS: [&str; 8] = ["tsx", "jsx", "ts", "js", "mjs", "cjs", "mdx", "md"];

/// Validate a next.config `pageExtensions` (from the config eval) against the extensions
/// diffpack's discovery supports. Present + all-supported: Ok (diffpack's superset covers
/// every page the config intends). Any unsupported extension: a hard error naming it. No
/// `pageExtensions` set (null/absent): Ok — the adapter uses its built-in [`PAGE_EXTS`].
fn validate_page_extensions(eval: Option<&serde_json::Value>) -> Result<(), String> {
    let Some(list) = eval
        .and_then(|v| v.get("pageExtensions"))
        .and_then(|v| v.as_array())
    else {
        return Ok(());
    };
    let unsupported: Vec<String> = list
        .iter()
        .filter_map(|v| v.as_str())
        .filter(|ext| !SUPPORTED_PAGE_EXTS.contains(ext))
        .map(|ext| ext.to_string())
        .collect();
    if unsupported.is_empty() {
        return Ok(());
    }
    Err(format!(
        "next.config `pageExtensions` includes {:?}, which diffpack's app-router page \
         discovery cannot compile (supported: {:?}). Remove the unsupported extension(s) or \
         author those pages in a supported extension.",
        unsupported, SUPPORTED_PAGE_EXTS,
    ))
}

/// Report the app's `@next/mdx` (`createMDX`) configuration to the build log.
///
/// `createMDX({ options: { remarkPlugins: [remarkGfm] } })` used to be read by nobody: the
/// app got plain CommonMark with no warning and no error, and the page simply rendered
/// differently from what the author wrote. The options are now read by the config eval and
/// stated here, every build, so their fate is visible: honoured natively, or handed to the
/// app's own MDX pipeline (`src/mdx_runner.mjs`) — which hard-errors, naming the plugins and
/// the file, if that pipeline is not installed.
///
/// Silent only when the app does not use `@next/mdx` at all.
pub(crate) fn report_mdx_config(eval: Option<&serde_json::Value>) {
    let config = crate::mdx::MdxConfig::from_eval(eval);
    if !config.configured {
        return;
    }
    let summary = config.summary();
    if config.unhonored_options().is_empty() {
        eprintln!(
            "[next.config] @next/mdx: {summary} — compiled by diffpack's native MDX compiler"
        );
    } else {
        eprintln!(
            "[next.config] @next/mdx: {summary} — .mdx/.md files are compiled with the app's \
             own @mdx-js/mdx pipeline so these run"
        );
    }
}

/// Browser aliases for the Node built-ins Next.js POLYFILLS for client bundles.
///
/// Next ships these under `next/dist/compiled/` and its client webpack config maps the
/// built-in specifier onto them, so `import { format } from "url"` in a page is valid
/// Next code and `next build` accepts it. Without this table diffpack's (correct)
/// "node built-in cannot be bundled for the browser" error rejects an app that Next
/// builds — which is what happened to `examples/with-shallow-routing`.
///
/// Only built-ins Next actually vendors are mapped, and only when the file is really
/// present in this app's `node_modules`: a missing one stays a hard error rather than
/// resolving to nothing. Built-ins Next does NOT polyfill (`fs`, `net`, `child_process`,
/// …) are deliberately absent and keep failing loudly.
///
/// The table mirrors the `isClient` `resolve.fallback` block of
/// `next/dist/build/webpack-config`, entry for entry. It applies to EVERY client
/// compilation — app router and pages router alike — because webpack's fallback is a
/// property of the client compiler, not of a router.
pub fn next_browser_polyfill_aliases(root: &Path) -> Vec<(String, String)> {
    // specifier -> Next's vendored package directory, mirroring next/dist/build/webpack-config.
    const POLYFILLS: &[(&str, &str)] = &[
        ("assert", "assert"),
        ("buffer", "buffer"),
        ("constants", "constants-browserify"),
        ("crypto", "crypto-browserify"),
        ("domain", "domain-browser"),
        ("events", "events"),
        ("http", "stream-http"),
        ("https", "https-browserify"),
        ("os", "os-browserify"),
        ("path", "path-browserify"),
        ("process", "process"),
        ("punycode", "punycode"),
        ("querystring", "querystring-es3"),
        ("stream", "stream-browserify"),
        ("string_decoder", "string_decoder"),
        // Next maps `sys` onto the same vendored `util` (`sys` is Node's deprecated
        // alias for it).
        ("sys", "util"),
        ("timers", "timers-browserify"),
        ("tty", "tty-browserify"),
        ("url", "native-url"),
        ("util", "util"),
        ("vm", "vm-browserify"),
        ("zlib", "browserify-zlib"),
    ];
    let Some(compiled) = next_compiled_dir(root) else {
        return Vec::new();
    };
    let mut aliases = Vec::new();
    for (specifier, vendored) in POLYFILLS {
        let dir = compiled.join(vendored);
        if !dir.is_dir() {
            continue;
        }
        aliases.push(((*specifier).to_string(), dir.to_string_lossy().into_owned()));
        // `node:`-prefixed form resolves to the same polyfill. webpack's fallback keys
        // are unprefixed, but `node:path` and `path` name the same module, and real
        // Next apps write both.
        aliases.push((
            format!("node:{specifier}"),
            dir.to_string_lossy().into_owned(),
        ));
    }
    aliases
}

/// The compile-time value of `process.env.NEXT_RUNTIME` for a build target, as a JS
/// literal ready for [`crate::vite_define`].
///
/// Next defines this in every compilation (`next/dist/build/define-env`:
/// `isEdgeServer ? 'edge' : isNodeServer ? 'nodejs' : ''`) and library code inside
/// `next/dist` branches on it to pick a Node-only implementation. `next/dist/shared/lib/
/// bloom-filter.js` is the canonical case:
///
/// ```js
/// if (process.env.NEXT_RUNTIME === 'nodejs') {
///   const gzipSize = require('next/dist/compiled/gzip-size').sync(filterData);
/// }
/// ```
///
/// Leaving the define out does not merely miss an optimization: the test stays
/// undecidable, [`crate::dead_branch`] cannot delete the branch, and
/// [`crate::parser`]'s unfiltered walk records the `require` as a real graph edge — so a
/// CLIENT bundle acquires `gzip-size` and, through it, `fs`/`stream`/`zlib`. `fs` has no
/// Next polyfill by design, so the build then fails on a module webpack never put in the
/// client graph at all. The define is what makes the branch dead, and deleting the
/// branch is what keeps the edge out.
///
/// The empty string is the correct client value, not a placeholder: `''` is falsy and
/// unequal to every runtime name, which is exactly how browser code is meant to read it.
/// The compile-time value of `process.browser` for a build target, as a JS literal.
///
/// Next defines it in every compilation (`next/dist/build/define-env.js`:
/// `'process.browser': isClient`), and it is the switch a large amount of
/// isomorphic library code branches on to reach for Node:
///
/// ```js
/// if (!process.browser && typeof window === 'undefined') {
///   var fs = require('fs');
/// }
/// ```
///
/// That is `next-i18next`'s `createConfig`, and it is the same mechanism as
/// [`next_runtime_define`]: without the define the test is undecidable,
/// [`crate::dead_branch`] cannot delete the branch, and [`crate::parser`]'s
/// unfiltered walk records the `require("fs")` as a real client-graph edge. `fs`
/// has no Next browser polyfill by design, so the build then fails on a module
/// webpack never put in the client graph at all.
pub(crate) fn process_browser_define(target: Target) -> &'static str {
    match target {
        Target::Client => "true",
        Target::Server | Target::IsolatedServer => "false",
    }
}

pub(crate) fn next_runtime_define(target: Target) -> &'static str {
    match target {
        // Both server graphs run under Node. diffpack has no separate edge *build*
        // target; a `runtime = "edge"` route handler is served from the server graph in
        // a WinterCG context, so `nodejs` remains the honest answer for the compilation.
        Target::Server | Target::IsolatedServer => "\"nodejs\"",
        Target::Client => "\"\"",
    }
}

/// `next/dist/compiled` for the `next` that would be RESOLVED from `root`.
///
/// Not `root/node_modules/next`: in a workspace monorepo (cal.com, and every
/// yarn/pnpm/npm workspace) the app lives at `apps/web` while `next` is hoisted to the
/// repository root's `node_modules`. Looking only at the app's own directory finds
/// nothing there, which used to mean "Next polyfills no built-ins" — a silent
/// difference in graph shape between a standalone app and the identical app inside a
/// workspace. Walk `node_modules` ancestors nearest-first, exactly as Node resolution
/// does.
fn next_compiled_dir(root: &Path) -> Option<PathBuf> {
    root.ancestors()
        .map(|dir| dir.join("node_modules").join("next"))
        .find(|next| next.join("package.json").is_file())
        .map(|next| next.join("dist").join("compiled"))
        .filter(|compiled| compiled.is_dir())
}

/// The app's `next.config.<ext>`, or None. A `next.config` is
/// OPTIONAL in Next.js, so its absence is not evidence that a project is not a Next app.
pub fn next_config_path(root: &Path) -> Option<PathBuf> {
    crate::next_config::next_config_path(root)
}

/// Whether `root` is a Next.js project at all — router-independent, and the guard that
/// keeps a non-Next `app/` (a TanStack `routesDirectory`, say) from being hijacked by the
/// Next adapters. Two signals, OR'd: a `next` dependency in `package.json`, or a
/// `next.config.*`. Both are needed — real apps commonly have no config (it is optional),
/// and diffpack's own hermetic fixtures have no `package.json`.
pub fn is_next_project(root: &Path) -> bool {
    let manifest = root.join("package.json");
    if let Ok(text) = std::fs::read_to_string(&manifest) {
        match serde_json::from_str::<serde_json::Value>(&text) {
            Ok(json) => {
                let declares_next = ["dependencies", "devDependencies"]
                    .iter()
                    .any(|field| json.get(field).and_then(|deps| deps.get("next")).is_some());
                if declares_next {
                    return true;
                }
            }
            // Never silently answer "not a Next app" because of a broken manifest: say so
            // loudly, then fall through to the config signal.
            Err(error) => eprintln!(
                "next detection: cannot parse {} ({error}); falling back to next.config detection",
                manifest.display(),
            ),
        }
    }
    next_config_path(root).is_some()
}

/// The app-router directory for `root`, checking `app/` then `src/app/` (Next's own
/// precedence: the root location wins). None when neither exists.
pub fn app_dir(root: &Path) -> Option<PathBuf> {
    [root.join("app"), root.join("src").join("app")]
        .into_iter()
        .find(|candidate| candidate.is_dir())
}

/// Detects whether `root` is a Next.js app-router project this adapter handles: a Next
/// project (see [`is_next_project`]) whose `app/` — or `src/app/` — contains at least one
/// `page`/`route` module ANYWHERE beneath it (Next has no requirement that the app root
/// itself be a route: `app/[lang]/page.tsx` alone is a valid app). Returns the resolved
/// page/route path when so.
fn detect_app_router(root: &Path) -> Option<PathBuf> {
    if !is_next_project(root) {
        return None;
    }
    first_page_under(&app_dir(root)?)
}

/// Whether `root` is a Next.js app-router project this adapter handles. Public
/// wrapper over the private detector so the dev server can dispatch a Next app to
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
/// under `<root>/.diffpack-next/` and return its
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

/// The first `page`/`route` module anywhere under `dir` (breadth of the whole subtree,
/// deterministic: children are sorted). This is DETECTION only, so it must never fail —
/// only decline; that is why it walks the tree itself instead of reusing
/// [`discover_routes`], which parses sources and can return `Err`. Skips dotdirs, the
/// adapter's own output, and `node_modules`.
fn first_page_under(dir: &Path) -> Option<PathBuf> {
    if let Some(page) = first_existing_page(dir).or_else(|| first_existing(dir, "route")) {
        return Some(page);
    }
    let mut children: Vec<PathBuf> = std::fs::read_dir(dir)
        .ok()?
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| path.is_dir())
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .map(|name| !name.starts_with('.') && name != ADAPTER_DIR && name != "node_modules")
                .unwrap_or(false)
        })
        .collect();
    children.sort();
    children.iter().find_map(|child| first_page_under(child))
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
    /// The source file (absolute): a static image to copy, or — when `generator` is
    /// true — a code-based `opengraph-image.tsx` (etc.) whose default export returns a
    /// `@vercel/og` `ImageResponse`, prerendered to a PNG at build time.
    source: PathBuf,
    /// The served URL path (`/icon.png`), which is also the copied/emitted output filename.
    served: String,
    /// The image MIME type inferred from the extension (for `<link type>`).
    mime: &'static str,
    /// True when `source` is a code-based ImageResponse generator (prerendered to
    /// `served` at build time via `@vercel/og`), rather than a static file to copy.
    generator: bool,
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
    let source_type = diffpack_core::parser::scan_source_type(path);
    let parsed = oxc_parser::Parser::new(&allocator, source, source_type).parse();
    let mut meta = RouteMetadata::default();
    for statement in &parsed.program.body {
        let Statement::ExportNamedDeclaration(export) = statement else {
            continue;
        };
        let Some(Declaration::VariableDeclaration(var)) = &export.declaration else {
            continue;
        };
        for decl in &var.declarations {
            if decl.id.get_binding_identifier().map(|i| i.name.as_str()) != Some("metadata") {
                continue;
            }
            let Some(Expression::ObjectExpression(object)) = &decl.init else {
                continue;
            };
            for property in &object.properties {
                let ObjectPropertyKind::ObjectProperty(prop) = property else {
                    continue;
                };
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
    let Ok(raw) = std::fs::read_to_string(path) else {
        return false;
    };
    let source = strip_comments(&raw);
    [
        "metadata",
        "generateMetadata",
        "viewport",
        "generateViewport",
    ]
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
        && let Ok(source) = std::fs::read_to_string(layout)
    {
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
    /// A layout/template ABOVE this page in the route's level chain that reads request
    /// state (`next/headers`). Next composes the whole segment tree for a route, so a
    /// read anywhere in it — the ROOT layout included — makes the entire route
    /// per-request even though the page's own source shows nothing. Holds the display
    /// path of the first such module (root→leaf) so the manifest reason can name it.
    request_state_module: Option<String>,
    /// `export const runtime = "nodejs" | "edge"`. `edge` opts the route into diffpack's
    /// lean WinterCG (edge-like) context (see [`RouteRuntime`] + [`validate_edge_module`]);
    /// `nodejs` (the default) is inert.
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
    for prefix in [
        format!("export const {name}"),
        format!("export let {name}"),
        format!("export var {name}"),
    ] {
        if let Some(pos) = source.find(&prefix) {
            let after = &source[pos + prefix.len()..];
            // Require the next non-space char to be `=` (not another identifier char).
            let after = after.trim_start();
            let Some(rest) = after.strip_prefix('=') else {
                continue;
            };
            let rest = rest.trim_start();
            // RHS runs to the first `;` or newline.
            let end = rest.find([';', '\n']).unwrap_or(rest.len());
            let raw = rest[..end].trim();
            let unquoted = raw
                .strip_prefix('"')
                .and_then(|s| s.strip_suffix('"'))
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
    let reads_request_state = source.contains("next/headers") || source.contains("searchParams");
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
        // Filled in by the caller, which is the only place that knows the level chain.
        request_state_module: None,
        runtime,
        fetch_cache,
        preferred_region,
        max_duration,
        experimental_ppr,
    }
}

/// The first layout/template in a route's root→leaf level chain that reads request-scoped
/// state (`next/headers`: cookies / headers / draftMode), or None if none does.
///
/// A route is not just its `page` module: Next composes the ENTIRE segment tree — every
/// enclosing `layout.tsx` and `template.tsx` — into one render. A request read anywhere in
/// that tree makes the whole route per-request, and the page's own source cannot show it.
/// (The root layout is the common case: one `headers()` there makes every route in the app
/// dynamic, which is exactly what `next build` reports.)
///
/// Layouts and templates receive `params` but NOT `searchParams`, so — unlike a page — only
/// the `next/headers` read counts here; a `searchParams` mention in a layout is not a
/// request read.
fn level_chain_request_state_read(levels: &[Level]) -> Option<PathBuf> {
    for level in levels {
        for module in [level.layout.as_ref(), level.template.as_ref()]
            .into_iter()
            .flatten()
        {
            let Ok(raw) = std::fs::read_to_string(module) else {
                continue;
            };
            if strip_comments(&raw).contains("next/headers") {
                return Some(module.clone());
            }
        }
    }
    None
}

/// Which runtime a route / route-handler / middleware declares. Next's default is
/// `nodejs`; `edge` (or the legacy alias `experimental-edge`) opts into a lean
/// WinterCG runtime. diffpack serves both from one native Node process — the edge
/// distinction is (1) the WinterCG global surface (already present on Node) advertised
/// via `globalThis.EdgeRuntime`, and (2) the ban on Node built-ins enforced at build
/// time by [`validate_edge_module`], so an edge route never silently reaches a Node-only
/// API it would fail on in a real edge deployment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RouteRuntime {
    Node,
    Edge,
}

impl RouteRuntime {
    /// Classify a `runtime` config value. Unknown values are a hard error (never a
    /// silent Node default) so a typo does not quietly change where code runs.
    fn from_config(url_path: &str, value: Option<&str>) -> Result<Self, String> {
        match value {
            None | Some("nodejs") => Ok(RouteRuntime::Node),
            Some("edge") | Some("experimental-edge") => Ok(RouteRuntime::Edge),
            Some(other) => Err(format!(
                "route {url_path}: runtime = \"{other}\" is not a recognized Next runtime \
                 (expected \"nodejs\" or \"edge\").",
            )),
        }
    }

    fn is_edge(self) -> bool {
        matches!(self, RouteRuntime::Edge)
    }
}

/// Node built-in modules a WinterCG (edge) runtime does NOT provide. Importing any of
/// these from an `edge` route/handler/middleware is rejected at BUILD time — mirroring
/// what a real edge deployment would reject — instead of silently running on Node with
/// different semantics. The set covers the filesystem, process/child-process, raw
/// networking, and other Node-host-only modules; the WinterCG-shaped ones an edge
/// runtime polyfills (buffer/events/util/stream/crypto/path/async_hooks/...) are
/// deliberately allowed.
const EDGE_FORBIDDEN_NODE_BUILTINS: [&str; 20] = [
    "fs",
    "fs/promises",
    "child_process",
    "worker_threads",
    "cluster",
    "net",
    "tls",
    "dgram",
    "dns",
    "dns/promises",
    "http",
    "https",
    "http2",
    "os",
    "vm",
    "v8",
    "inspector",
    "readline",
    "repl",
    "module",
];

/// Scan an `edge`-runtime module's source for imports of Node built-ins the WinterCG
/// runtime cannot provide, hard-erroring (naming the module + the offending specifier)
/// if any is found. This is the lean, native enforcement of the edge "no Node fs" (and
/// no process/net/...) contract: the check is SHALLOW (this module's own import/require
/// specifiers only, over comment-stripped source), documented as such — it catches the
/// direct, common mistake without a full graph walk. Returns `Ok(())` for a clean edge
/// module.
fn validate_edge_module(label: &str, source: &str) -> Result<(), String> {
    let stripped = strip_comments(source);
    for name in EDGE_FORBIDDEN_NODE_BUILTINS {
        // `node:<name>` is unambiguously the builtin wherever it appears as a specifier.
        // A bare `<name>` is only treated as the builtin in specifier position
        // (`from "..."`, `require("...")`, `import("...")`, side-effect `import "..."`).
        let quoted: [String; 2] = [format!("\"{name}\""), format!("'{name}'")];
        let node_quoted: [String; 2] = [format!("\"node:{name}\""), format!("'node:{name}'")];
        let hits_node = node_quoted.iter().any(|q| stripped.contains(q.as_str()));
        let hits_bare = quoted.iter().any(|q| {
            // Require a specifier keyword just before the quoted string.
            for kw in ["from ", "require(", "import(", "import "] {
                let needle = format!("{kw}{q}");
                if stripped.contains(&needle) {
                    return true;
                }
            }
            false
        });
        if hits_node || hits_bare {
            return Err(format!(
                "{label}: imports the Node built-in \"{name}\", which the edge (WinterCG) runtime \
                 does not provide. Remove `export const runtime = \"edge\"` to run it on the Node.js \
                 runtime, or rewrite it against Web APIs (fetch/Request/Response/URL/crypto).",
            ));
        }
    }
    Ok(())
}

/// Enforce the route-segment-config exports diffpack recognizes beyond the
/// static/dynamic set. `runtime = "edge"` selects diffpack's lean WinterCG context (no
/// longer a hard error): the caller runs [`validate_edge_module`] on the source to
/// enforce the no-Node-built-ins contract, and the served route advertises the edge
/// globals via `globalThis.EdgeRuntime`. An UNRECOGNIZED runtime is still a hard error
/// (never a silent Node default). The remaining exports (`fetchCache`,
/// `preferredRegion`, `maxDuration`, `experimental_ppr`) are advisory for a native
/// single-node server: each is reported with a build WARN explaining precisely why
/// diffpack cannot honor it, so the behavior is explicit rather than a silent default.
fn validate_segment_config(url_path: &str, cfg: &RouteConfig) -> Result<(), String> {
    // Classify the runtime (errors on an unrecognized value; `edge` is accepted here and
    // its module body is checked by the caller via `validate_edge_module`).
    RouteRuntime::from_config(url_path, cfg.runtime.as_deref())?;
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
    if matches!(
        cfg.dynamic_config.as_deref(),
        Some("force-static") | Some("error")
    ) {
        return RouteKind::ForceStatic;
    }
    if cfg.reads_request_state || cfg.request_state_module.is_some() {
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
            "reads request state (cookies/headers/searchParams); no generateStaticParams"
                .to_string()
        }
    } else if let Some(module) = cfg.request_state_module.as_deref() {
        // The page itself is static-looking; a layout/template wrapping it reads request
        // state, which is what makes the WHOLE route per-request. Name the module so the
        // manifest never leaves the reason unattributable to a file.
        format!(
            "a layout/template in this route's tree reads request state (next/headers): {module}"
        )
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
    if let Some(inner) = comp
        .strip_prefix("[[...")
        .and_then(|s| s.strip_suffix("]]"))
    {
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
    let root = project_root.canonicalize().map_err(|error| {
        format!(
            "cannot open project root {}: {error}",
            project_root.display()
        )
    })?;
    if detect_app_router(&root).is_none() {
        return Err(format!(
            "{} is not a Next.js app-router project (no `next` dependency / next.config, or \
             no app/ or src/app containing a page); \
             `build-app <root> static` only prerenders app-router apps",
            root.display(),
        ));
    }
    let app_dir = app_dir(&root).ok_or_else(|| {
        format!(
            "{}: app-router detected but no app/ or src/app directory",
            root.display()
        )
    })?;
    let layout = first_existing(&app_dir, "layout");
    let layout_abs = layout
        .as_ref()
        .map(|l| l.canonicalize().unwrap_or_else(|_| l.clone()));
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
                    fields.push_str(&format!(
                        ", \"file\": {}",
                        js_str(&route_file_stem(&route.url_path))
                    ));
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

/// Which of the app's routes a build COMPILES.
///
/// Production always compiles every route: the output has to be able to serve any of
/// them. Dev does not — a cold start compiles all 229 of cal.com's routes into three
/// whole-app graphs before it answers anything, and the browser then asks for exactly
/// one of them. [`Only`](RouteScope::Only) narrows the generated entries to a subset so
/// a cold start pays for the route being loaded; the dev server widens the scope as
/// routes are asked for and fills in the rest in the background.
///
/// The app-root conventions (root layout, `not-found`, `global-error`, middleware, the
/// metadata image files) are NOT route-scoped: every document is built from them, so
/// they are always compiled.
/// HTTP endpoints (`route.ts` handlers + `pages/api/**`) are scoped separately from
/// pages, because a page is USELESS without them: the document a browser renders
/// immediately calls the app's own API (cal.com's login page reads a next-auth session and
/// several tRPC queries), and an endpoint that is not compiled answers 404, which the page
/// reports as a broken app rather than as "still compiling".
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EndpointScope {
    /// Compile every endpoint. What a lazily-scoped dev build uses, so the first page it
    /// serves is functional and not just visible.
    All,
    /// Compile only these endpoint URL paths.
    Only(BTreeSet<String>),
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum RouteScope {
    /// Every discovered route, handler and pages-API endpoint.
    #[default]
    All,
    /// Only these page routes (the discovery spelling, e.g. `/booking/[uid]`), with
    /// endpoints scoped by `endpoints`.
    Only {
        pages: BTreeSet<String>,
        endpoints: EndpointScope,
    },
}

impl RouteScope {
    /// A scope holding exactly `pages`, with every HTTP endpoint compiled.
    pub fn pages<I: IntoIterator<Item = String>>(pages: I) -> Self {
        RouteScope::Only {
            pages: pages.into_iter().collect(),
            endpoints: EndpointScope::All,
        }
    }

    /// A scope holding exactly `pages` and exactly `endpoints`.
    pub fn pages_and_endpoints<P, E>(pages: P, endpoints: E) -> Self
    where
        P: IntoIterator<Item = String>,
        E: IntoIterator<Item = String>,
    {
        RouteScope::Only {
            pages: pages.into_iter().collect(),
            endpoints: EndpointScope::Only(endpoints.into_iter().collect()),
        }
    }

    /// Every page this scope compiles, or `None` for [`RouteScope::All`].
    pub fn compiled_pages(&self) -> Option<&BTreeSet<String>> {
        match self {
            RouteScope::All => None,
            RouteScope::Only { pages, .. } => Some(pages),
        }
    }

    /// Whether this scope compiles `url_path`, which is what decides if a request for it
    /// can be served now or has to wait for a wider build. The kind matters: pages and
    /// endpoints are scoped separately.
    pub fn includes(&self, url_path: &str, kind: PatternKind) -> bool {
        match kind {
            PatternKind::Page => self.includes_page(url_path),
            PatternKind::Endpoint => self.includes_endpoint(url_path),
        }
    }

    fn includes_page(&self, url_path: &str) -> bool {
        match self {
            RouteScope::All => true,
            RouteScope::Only { pages, .. } => pages.contains(url_path),
        }
    }

    fn includes_endpoint(&self, url_path: &str) -> bool {
        match self {
            RouteScope::All => true,
            RouteScope::Only {
                endpoints: EndpointScope::All,
                ..
            } => true,
            RouteScope::Only {
                endpoints: EndpointScope::Only(paths),
                ..
            } => paths.contains(url_path),
        }
    }

    /// A short label for the dev log.
    pub fn label(&self) -> String {
        match self {
            RouteScope::All => "all routes".to_string(),
            RouteScope::Only { pages, endpoints } => {
                let pages = match pages.len() {
                    0 => "no pages".to_string(),
                    1 => format!("page {}", pages.iter().next().expect("len 1")),
                    n => format!("{n} pages"),
                };
                match endpoints {
                    EndpointScope::All => format!("{pages} + all endpoints"),
                    EndpointScope::Only(paths) => format!("{pages} + {} endpoint(s)", paths.len()),
                }
            }
        }
    }
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
    /// `pages/api/**` HTTP endpoints of a HYBRID app (`app/` pages + pages-router API
    /// routes), most-specific first. Served after `app/**/route.ts`, which wins on a
    /// path both could answer — Next's own precedence.
    pages_api: Vec<PagesApiRoute>,
}

/// A pages-router API route (`pages/api/**`) in a project whose pages live in `app/`.
///
/// Next supports this hybrid shape and real apps lean on it: cal.com serves next-auth
/// (`pages/api/auth/[...nextauth].ts`) and its entire tRPC surface as pages API routes
/// while every page is app-router. Building only `app/` leaves those endpoints unserved,
/// so the client cannot read a session, cannot log in, and every data query 404s.
///
/// The endpoint contract differs from an app-router `route.ts`: a pages API route is a
/// single default-exported `(req, res)` function taking a Node request/response pair,
/// not per-method functions taking a Web `Request`. See `next_runtime/pages_api.js`.
#[derive(Debug, Clone)]
struct PagesApiRoute {
    /// The matched URL path (`/api/auth/[...nextauth]`).
    url_path: String,
    segments: Vec<Seg>,
    /// The handler module (absolute, canonical).
    file: PathBuf,
}

/// The file extensions Next treats as a route module, in `pageExtensions` default order.
const PAGES_API_EXTENSIONS: [&str; 4] = ["tsx", "ts", "jsx", "js"];

/// Parse one pages-router path component into a URL segment. The pages router has no
/// route groups and no parallel slots, so every component is a literal or a
/// `[param]`/`[...catchAll]`/`[[...optionalCatchAll]]` — it deliberately does NOT reuse
/// the app-router [`parse_segment`], which would swallow a directory literally named
/// `(x)` as a group.
fn parse_pages_segment(raw: &str) -> Seg {
    if let Some(inner) = raw.strip_prefix("[[...").and_then(|s| s.strip_suffix("]]")) {
        Seg::OptionalCatchAll(inner.to_string())
    } else if let Some(inner) = raw.strip_prefix("[...").and_then(|s| s.strip_suffix(']')) {
        Seg::CatchAll(inner.to_string())
    } else if let Some(inner) = raw.strip_prefix('[').and_then(|s| s.strip_suffix(']')) {
        Seg::Dynamic(inner.to_string())
    } else {
        Seg::Static(raw.to_string())
    }
}

/// The `pages/` directory of an app-router project, if it has one: `pages/` beside
/// `app/`, or `src/pages/` beside `src/app/`. Next looks for the two dirs as siblings,
/// which is why this is derived from the app dir rather than probed independently.
fn sibling_pages_dir(app_dir: &Path) -> Option<PathBuf> {
    let pages = app_dir.parent()?.join("pages");
    pages.is_dir().then_some(pages)
}

/// Discover every `pages/api/**` endpoint of a hybrid app, most-specific first (the same
/// specificity order route handlers and pages use).
fn discover_pages_api_routes(app_dir: &Path) -> Result<Vec<PagesApiRoute>, String> {
    let Some(api_dir) = sibling_pages_dir(app_dir).map(|pages| pages.join("api")) else {
        return Ok(Vec::new());
    };
    if !api_dir.is_dir() {
        return Ok(Vec::new());
    }
    let mut out = Vec::new();
    discover_pages_api_dir(&api_dir, &["api".to_string()], &mut out)?;
    out.sort_by(|a, b| {
        let count =
            |r: &PagesApiRoute, f: fn(&Seg) -> bool| r.segments.iter().filter(|s| f(s)).count();
        let ca = count(a, |s| {
            matches!(s, Seg::CatchAll(_) | Seg::OptionalCatchAll(_))
        });
        let cb = count(b, |s| {
            matches!(s, Seg::CatchAll(_) | Seg::OptionalCatchAll(_))
        });
        let da = count(a, |s| matches!(s, Seg::Dynamic(_)));
        let db = count(b, |s| matches!(s, Seg::Dynamic(_)));
        ca.cmp(&cb)
            .then(da.cmp(&db))
            .then(b.segments.len().cmp(&a.segments.len()))
            .then(a.url_path.cmp(&b.url_path))
    });
    Ok(out)
}

fn discover_pages_api_dir(
    dir: &Path,
    prefix: &[String],
    out: &mut Vec<PagesApiRoute>,
) -> Result<(), String> {
    let read = std::fs::read_dir(dir)
        .map_err(|error| format!("cannot read {}: {error}", dir.display()))?;
    let mut entries: Vec<PathBuf> = read
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| {
            !path
                .file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|name| name.starts_with('.'))
        })
        .collect();
    entries.sort();
    for path in entries {
        let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        if path.is_dir() {
            let mut nested = prefix.to_vec();
            nested.push(name.to_string());
            discover_pages_api_dir(&path, &nested, out)?;
            continue;
        }
        if !path
            .extension()
            .and_then(|e| e.to_str())
            .is_some_and(|ext| PAGES_API_EXTENSIONS.contains(&ext))
        {
            continue;
        }
        let stem = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or_default();
        // `pages/api/x/index.ts` serves `/api/x`, exactly as `pages/api/x.ts` does.
        let mut parts = prefix.to_vec();
        if stem != "index" {
            parts.push(stem.to_string());
        }
        let segments: Vec<Seg> = parts.iter().map(|part| parse_pages_segment(part)).collect();
        out.push(PagesApiRoute {
            url_path: segments_display(&segments),
            segments,
            file: path.canonicalize().unwrap_or(path.clone()),
        });
    }
    Ok(())
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
    /// `export const runtime = "edge"` — served in diffpack's lean WinterCG context
    /// (`globalThis.EdgeRuntime` advertised; Node built-ins rejected at build).
    edge: bool,
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
        let count =
            |r: &RouteHandler, f: fn(&Seg) -> bool| r.segments.iter().filter(|s| f(s)).count();
        let ca = count(a, |s| {
            matches!(s, Seg::CatchAll(_) | Seg::OptionalCatchAll(_))
        });
        let cb = count(b, |s| {
            matches!(s, Seg::CatchAll(_) | Seg::OptionalCatchAll(_))
        });
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
                let url_path = segments_display(&segments);
                // Route-handler runtime: `export const runtime = "edge"` opts into the
                // lean WinterCG context. Enforce the no-Node-built-ins contract on the
                // handler source (a real edge deployment would reject those imports).
                let runtime = RouteRuntime::from_config(
                    &url_path,
                    extract_export_const(&strip_comments(&source), "runtime").as_deref(),
                )?;
                if runtime.is_edge() {
                    validate_edge_module(
                        &format!("edge route handler {url_path} ({})", file.display()),
                        &source,
                    )?;
                    eprintln!(
                        "next edge runtime: route handler {url_path} exports `runtime = \"edge\"`; \
                         served in diffpack's WinterCG context (globalThis.EdgeRuntime set; Node \
                         built-ins rejected at build).",
                    );
                }
                out.push(RouteHandler {
                    url_path,
                    segments,
                    file,
                    methods,
                    edge: runtime.is_edge(),
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
        let ca = count(a, |s| {
            matches!(s, Seg::CatchAll(_) | Seg::OptionalCatchAll(_))
        });
        let cb = count(b, |s| {
            matches!(s, Seg::CatchAll(_) | Seg::OptionalCatchAll(_))
        });
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
        global_error: first_existing(app_dir, "global-error")
            .map(|p| p.canonicalize().unwrap_or(p)),
        handlers: discover_route_handlers(app_dir)?,
        intercepts: discover_intercepts(app_dir)?,
        meta_images: scan_metadata_images(app_dir)?,
        pages_api: discover_pages_api_routes(app_dir)?,
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
    MetaFileConvention {
        stem: "sitemap",
        url: "/sitemap.xml",
        wrapper: "metadata-sitemap.ts",
    },
    MetaFileConvention {
        stem: "robots",
        url: "/robots.txt",
        wrapper: "metadata-robots.ts",
    },
    MetaFileConvention {
        stem: "manifest",
        url: "/manifest.webmanifest",
        wrapper: "metadata-manifest.ts",
    },
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
        let Some(user_file) = first_existing(app_dir, conv.stem) else {
            continue;
        };
        let user_file = user_file.canonicalize().unwrap_or(user_file);
        if !wrote_serializer {
            write_if_changed(&serializer, metadata_serialize_shim())?;
            wrote_serializer = true;
        }
        let serializer_canon = serializer
            .canonicalize()
            .unwrap_or_else(|_| serializer.clone());
        // `generateSitemaps` (id-partitioned sitemaps): Next serves each partition at
        // `/sitemap/[id].xml` (e.g. `/sitemap/0.xml`), calling the default `sitemap({ id })`
        // once per id enumerated by `generateSitemaps()`. Synthesize a DYNAMIC route handler
        // (`/sitemap/[id]`) whose wrapper validates the requested id against the enumeration,
        // then serializes `sitemap({ id })` — plugging into the same dispatch as a `route.ts`.
        if conv.stem == "sitemap" {
            let src = std::fs::read_to_string(&user_file).unwrap_or_default();
            if exports_symbol(&strip_comments(&src), "generateSitemaps") {
                let wrapper_path = shims_dir.join("metadata-sitemap-id.ts");
                write_if_changed(
                    &wrapper_path,
                    &sitemap_id_wrapper(&user_file, &serializer_canon),
                )?;
                let wrapper_canon = wrapper_path
                    .canonicalize()
                    .unwrap_or_else(|_| wrapper_path.clone());
                handlers.push(RouteHandler {
                    url_path: "/sitemap/[id].xml".to_string(),
                    segments: vec![
                        Seg::Static("sitemap".to_string()),
                        Seg::Dynamic("id".to_string()),
                    ],
                    file: wrapper_canon,
                    methods: vec!["GET".to_string()],
                    edge: false,
                });
                continue;
            }
        }
        let wrapper_path = shims_dir.join(conv.wrapper);
        let wrapper_src = metadata_file_wrapper(conv.stem, &user_file, &serializer_canon);
        write_if_changed(&wrapper_path, &wrapper_src)?;
        let wrapper_canon = wrapper_path
            .canonicalize()
            .unwrap_or_else(|_| wrapper_path.clone());
        // The served URL is a single static path segment (e.g. `sitemap.xml`).
        let seg = conv.url.trim_start_matches('/').to_string();
        handlers.push(RouteHandler {
            url_path: conv.url.to_string(),
            segments: vec![Seg::Static(seg)],
            file: wrapper_canon,
            methods: vec!["GET".to_string()],
            edge: false,
        });
    }
    Ok(handlers)
}

/// Walk up from `start` to the nearest ancestor directory that contains a `node_modules`
/// folder, returning that directory (the effective project root for module resolution).
fn nearest_node_modules_root(start: &Path) -> Option<PathBuf> {
    let mut dir = Some(start);
    while let Some(d) = dir {
        if d.join("node_modules").is_dir() {
            return Some(d.to_path_buf());
        }
        dir = d.parent();
    }
    None
}

/// The module specifier a code-based image generator imports its `ImageResponse` from:
/// `next/og` (Next re-exports `@vercel/og`) or `@vercel/og` directly. Returns the first
/// recognized specifier found in the (comment-stripped) source, or None.
fn og_import_specifier(source: &str) -> Option<&'static str> {
    let stripped = strip_comments(source);
    for spec in ["next/og", "@vercel/og"] {
        for q in [format!("\"{spec}\""), format!("'{spec}'")] {
            if stripped.contains(&q) {
                return Some(match spec {
                    "next/og" => "next/og",
                    _ => "@vercel/og",
                });
            }
        }
    }
    None
}

/// Whether the `ImageResponse` provider a generator imports resolves in this project.
/// `@vercel/og` resolves when the package is installed; `next/og` resolves when `next`
/// is installed (Next bundles `@vercel/og` and exposes it as `next/og`). A filesystem
/// check (no Node spawn) — lean and run only when a generator is actually present.
fn og_provider_resolves(project_root: &Path, spec: &str) -> bool {
    let nm = project_root.join("node_modules");
    match spec {
        "@vercel/og" => nm.join("@vercel/og").join("package.json").is_file(),
        "next/og" => nm.join("next").join("package.json").is_file(),
        _ => false,
    }
}

/// Scan the app ROOT for metadata-image file conventions: STATIC image files (copied to
/// the served output) and code-based `ImageResponse` GENERATORS (`opengraph-image.tsx`
/// etc., prerendered to a PNG at build time via `@vercel/og`). A generator whose
/// `@vercel/og`/`next/og` provider does not resolve is a clear hard error (naming the
/// file + how to install/opt out). Nested (segment-scoped) images are not scanned here.
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
        // A code-based generator (e.g. `opengraph-image.tsx`) returning a `@vercel/og`
        // ImageResponse. `favicon` has no code-generator convention in Next.
        if kind != MetaImageKind::Favicon
            && let Some(generator) = first_existing(app_dir, stem)
        {
            let generator = generator.canonicalize().unwrap_or(generator);
            let source = std::fs::read_to_string(&generator).unwrap_or_default();
            let stripped = strip_comments(&source);
            // `generateImageMetadata` (multiple id-partitioned images) needs the request
            // context + a size manifest this build-time prerender does not model. Fail
            // clearly rather than emit one wrong image (no silent stub).
            if exports_symbol(&stripped, "generateImageMetadata") {
                return Err(format!(
                    "diffpack next metadata: {} exports `generateImageMetadata` (multiple \
                     id-partitioned images), which this adapter does not prerender yet. Use a \
                     single default-export {stem}() returning one ImageResponse instead.",
                    generator.display(),
                ));
            }
            let Some(spec) = og_import_specifier(&source) else {
                return Err(format!(
                    "diffpack next metadata: {} is a code-based image generator but imports its \
                     ImageResponse from neither `next/og` nor `@vercel/og`. diffpack prerenders \
                     the ImageResponse at build time via @vercel/og; import it from one of those, \
                     or provide a static {stem}.png/.jpg/.svg instead.",
                    generator.display(),
                ));
            };
            let root = nearest_node_modules_root(app_dir).unwrap_or_else(|| app_dir.to_path_buf());
            if !og_provider_resolves(&root, spec) {
                return Err(format!(
                    "diffpack next metadata: {} is a code-based image generator that imports \
                     `{spec}`, but {spec} does not resolve in this project. Install it (\
                     `@vercel/og`, or `next` for `next/og`) so diffpack can prerender the \
                     ImageResponse at build time, or provide a static {stem}.png/.jpg/.svg instead.",
                    generator.display(),
                ));
            }
            images.push(MetaImage {
                kind,
                source: generator,
                served: format!("/{stem}.png"),
                mime: "image/png",
                generator: true,
            });
            continue;
        }
        let Some(src) = first_existing_ext(app_dir, stem, &METADATA_IMAGE_EXTS) else {
            continue;
        };
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
            generator: false,
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
            let mut cfg = scan_route_config(&page_source);
            // A layout/template anywhere above this page reading `next/headers` makes the
            // whole route per-request — the page's own source cannot show that.
            cfg.request_state_module =
                level_chain_request_state_read(level_chain).map(|p| p.display().to_string());
            let has_dynamic_segment = segments.iter().any(|s| {
                matches!(
                    s,
                    Seg::Dynamic(_) | Seg::CatchAll(_) | Seg::OptionalCatchAll(_)
                )
            });
            let kind = classify_route(has_dynamic_segment, &cfg);
            let url_path = segments_display(&segments);
            // Route-segment-config exports beyond the static/dynamic set: an
            // unrecognized `runtime` hard-errors; the advisory ones WARN. Done before the
            // route is recorded so an unsupported route never reaches codegen.
            validate_segment_config(&url_path, &cfg)?;
            // `runtime = "edge"`: accept it (diffpack renders the page through its Node
            // RSC pipeline — output-identical to edge), but enforce the WinterCG
            // no-Node-built-ins contract on the page source so an edge page that would
            // fail in a real edge deployment fails loudly here instead.
            if RouteRuntime::from_config(&url_path, cfg.runtime.as_deref())?.is_edge() {
                validate_edge_module(
                    &format!("edge route {url_path} ({})", page_abs.display()),
                    &page_source,
                )?;
                eprintln!(
                    "next edge runtime: route {url_path} exports `runtime = \"edge\"`; diffpack \
                     renders it through its native RSC pipeline under the WinterCG global surface \
                     (fetch/Request/Response/URL/crypto), Node built-ins rejected at build.",
                );
            }
            // Nested-gsp is out of scope: a route with >1 dynamic segment that also
            // exports generateStaticParams needs a BFS param merge we do not implement.
            // Hard-error naming the route rather than emit a wrong enumeration.
            if kind == RouteKind::Ssg {
                let dyn_seg_count = segments
                    .iter()
                    .filter(|s| {
                        matches!(
                            s,
                            Seg::Dynamic(_) | Seg::CatchAll(_) | Seg::OptionalCatchAll(_)
                        )
                    })
                    .count();
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
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.starts_with('@'))
        })
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
                !n.starts_with('.')
                    && n != ADAPTER_DIR
                    && !n.starts_with('@')
                    && !is_intercept_marker(n)
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
fn collect_intercepts(
    dir: &Path,
    base_segments: &mut Vec<Seg>,
    out: &mut Vec<Intercept>,
) -> Result<(), String> {
    let read = match std::fs::read_dir(dir) {
        Ok(read) => read,
        Err(_) => return Ok(()),
    };
    let mut children: Vec<PathBuf> = read
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.is_dir())
        .collect();
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
                InterceptBase::Up(n) => {
                    base_segments[..base_segments.len().saturating_sub(n)].to_vec()
                }
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
    let comp = if is_marker_root {
        strip_intercept_marker(name)
    } else {
        name
    };
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
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| !n.starts_with('.') && n != ADAPTER_DIR)
        })
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

/// The served path the react-server build's compiled CSS (`server/server.css`) is
/// preserved to (see `main.rs`), and the href the adapter links from `<head>`.
pub const RSC_CSS_URL: &str = "/rsc.css";

/// The file name the react-server build gives its compiled stylesheet, next to the
/// render entry. Its EXISTENCE is the single fact both halves of the stylesheet
/// concept read: `main.rs` preserves exactly this file to the served
/// `public/rsc.css`, and the render entry links [`RSC_CSS_URL`] only when it is
/// there. Two derivations of the same fact cannot disagree; a source scan and an
/// emitted artifact can, and did — a `<link>` to a 404.
pub const RSC_EMITTED_CSS_FILE: &str = "server.css";

/// The served path of the CLIENT graph's compiled CSS, and the second stylesheet the
/// adapter links from `<head>`.
///
/// The react-server graph does NOT see a `"use client"` module's body — the proxy that
/// replaces it keeps only that module's OWN direct stylesheet imports (see
/// `crate::rsc::transform_use_client_server`). So CSS imported by a plain module that
/// only a client component reaches — cal.com's `packages/ui/components/editor/
/// Editor.tsx` doing `import "./stylesEditor.css"`, reached from a `"use client"`
/// wrapper — never enters `server.css`, and the page renders unstyled in exactly that
/// subtree. The CLIENT graph is complete by construction and already compiles that CSS
/// into `public/client.css`; it was emitted and served but nothing linked it.
///
/// Next does the same thing: the route's document links the Tailwind/app chunk and
/// then a separate chunk holding the client components' CSS (verified against
/// cal.com's reference `next start` build, which links exactly three stylesheets on the
/// event-type route, the last one carrying `.editor-container`/`.toolbar`/`glide`).
/// Linking it LAST matches that order, so a client component's CSS wins ties against
/// the app stylesheet on both toolchains.
pub const CLIENT_CSS_URL: &str = "/client.css";

/// Where the render entry looks for [`CLIENT_CSS_URL`]'s artifact, relative to its own
/// `import.meta.url`. Same shape as [`RSC_EMITTED_CSS_FILE`]'s guard — the LINK and the
/// ARTIFACT are one fact, so a `<link>` whose href 404s is not expressible — except the
/// artifact belongs to a different graph, so it sits in the served `public/` beside the
/// server dir rather than next to the entry (`src/server_runtime/index.mjs` derives the
/// orchestrator's `publicDir` from the server dir the same way).
pub const CLIENT_EMITTED_CSS_PATH: &str = "../public/client.css";

/// The module-level facts one project walk yields for the adapter.
struct ProjectScan {
    /// Every `"use client"` module, canonical + sorted + deduped: the islands pinned
    /// into the client and SSR graphs so their client references resolve.
    islands: Vec<PathBuf>,
    /// Every `next/font` usage (deduped), so the adapter generates one CSS block
    /// covering all fonts.
    fonts: Vec<crate::next_font::FontUsage>,
    /// The source text of every module the walk read, keyed by canonical path, handed
    /// to [`crate::project_graph`] so the reachability walk re-reads nothing.
    sources: std::collections::HashMap<PathBuf, String>,
}

/// Scans the WHOLE project — not just `app/` — for the module facts above, over the
/// same [`crate::rsc::walk_project_modules`] the `"use server"` action scan uses, so
/// the two halves of the RSC directive concept cannot diverge on root or skip-list.
///
/// Rooted at the project because a `"use client"` component, a `next/font` call, or a
/// stylesheet import is routinely a SIBLING of the app directory rather than a child
/// of it (`src/components/`, `src/lib/` next to `src/app/`; `components/` next to
/// `app/`). An app-rooted walk misses those, the island never enters the client graph,
/// it therefore gets no client-references-manifest entry, and the react-server render
/// dies with `Could not find the module "…" in the React Client Manifest`.
///
/// Honest cost of discovery-by-filesystem-walk rather than by graph reachability:
/// every `"use client"` file in the tree becomes a pin, including ones no route
/// imports (Next pins only what is reachable), and a `"use client"` module inside
/// `node_modules` is never seen at all — the walk must skip dependencies to stay
/// tractable. The over-approximation is what makes the manifest complete, so it
/// stays; what it must NOT do is make dead code fatal, which
/// [`drop_unreachable_unbuildable_islands`] is responsible for.
fn scan_project(root: &Path) -> Result<ProjectScan, String> {
    let mut scan = ProjectScan {
        islands: Vec::new(),
        fonts: Vec::new(),
        sources: std::collections::HashMap::new(),
    };
    crate::rsc::walk_project_modules(root, &mut |path, source| {
        if source.contains("use client")
            && detect_directive(path, source) == Some(RscDirective::Client)
        {
            scan.islands.push(path.to_path_buf());
        }
        if source.contains("next/font") {
            for usage in crate::next_font::scan_next_font(path, source)? {
                if !scan.fonts.contains(&usage) {
                    scan.fonts.push(usage);
                }
            }
        }
        scan.sources.insert(path.to_path_buf(), source.to_string());
        Ok(())
    })?;
    scan.islands.sort();
    scan.islands.dedup();
    Ok(scan)
}

/// What kind of thing a [`RoutePattern`] addresses, because pages and endpoints are
/// scoped separately (see [`EndpointScope`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PatternKind {
    /// An app-router page.
    Page,
    /// A `route.ts` handler or a `pages/api/**` endpoint.
    Endpoint,
}

/// One matchable URL pattern of the app, with the scope key needed to compile it.
///
/// This is the table the dev server matches an incoming request against BEFORE anything
/// is compiled, so it can decide which route a request needs and widen its
/// [`RouteScope`] to include it. Discovery is a directory walk, so having the full table
/// costs nothing; compiling what it names is the expensive part, and that stays lazy.
#[derive(Debug, Clone)]
pub struct RoutePattern {
    /// The discovery spelling (`/booking/[uid]`), which is the scope key.
    pub url_path: String,
    pub kind: PatternKind,
    /// Absolute source module that owns the route, when discovered from a project.
    /// Synthetic patterns created with [`RoutePattern::parse`] have no source.
    pub source_path: Option<PathBuf>,
    segments: Vec<Seg>,
}

impl RoutePattern {
    /// A pattern from its URL spelling (`/blog/[slug]`, `/api/trpc/[...trpc]`), for tests
    /// that need a pattern table without a project on disk. The segment grammar is the
    /// pages-router one, which is exactly the spelling `url_path` carries: route groups and
    /// parallel slots contribute no URL segment, so they never appear here.
    pub fn parse(url_path: &str, kind: PatternKind) -> RoutePattern {
        RoutePattern {
            url_path: url_path.to_string(),
            kind,
            source_path: None,
            segments: url_path
                .split('/')
                .filter(|part| !part.is_empty())
                .map(parse_pages_segment)
                .collect(),
        }
    }

    /// Whether `request_path` (a URL path, no query) matches this pattern. The same rule
    /// the generated entry's `matchSegments` applies, so Rust's pre-match and the
    /// orchestrator's real match agree on which route a request belongs to.
    pub fn matches(&self, request_path: &str) -> bool {
        let parts: Vec<&str> = request_path.split('/').filter(|p| !p.is_empty()).collect();
        let mut i = 0usize;
        for segment in &self.segments {
            match segment {
                Seg::Static(value) => {
                    if parts.get(i) != Some(&value.as_str()) {
                        return false;
                    }
                    i += 1;
                }
                Seg::Dynamic(_) => {
                    if i >= parts.len() {
                        return false;
                    }
                    i += 1;
                }
                Seg::CatchAll(_) => {
                    if i >= parts.len() {
                        return false;
                    }
                    i = parts.len();
                }
                Seg::OptionalCatchAll(_) => {
                    i = parts.len();
                }
            }
        }
        i == parts.len()
    }
}

/// Every matchable pattern of an app-router project, endpoints first and then pages —
/// the orchestrator's own precedence, so the pattern this returns for a request path is
/// the one that will actually serve it. Each table stays in discovery order, which is
/// most-specific-first.
///
/// Discovery only: no scaffolding is written, no config is evaluated, nothing is
/// compiled. `Ok(None)` for a project that is not app-router.
pub fn discover_route_patterns(root: &Path) -> Result<Option<Vec<RoutePattern>>, String> {
    let root = root
        .canonicalize()
        .map_err(|error| format!("cannot open project root {}: {error}", root.display()))?;
    if detect_app_router(&root).is_none() {
        return Ok(None);
    }
    let Some(app_dir) = app_dir(&root) else {
        return Ok(None);
    };
    let layout = first_existing(&app_dir, "layout").map(|l| l.canonicalize().unwrap_or(l));
    let discovered = discover_routes(&app_dir, layout.as_deref())?;
    let mut patterns = Vec::new();
    for handler in &discovered.handlers {
        patterns.push(RoutePattern {
            url_path: handler.url_path.clone(),
            kind: PatternKind::Endpoint,
            source_path: Some(handler.file.clone()),
            segments: handler.segments.clone(),
        });
    }
    for endpoint in &discovered.pages_api {
        patterns.push(RoutePattern {
            url_path: endpoint.url_path.clone(),
            kind: PatternKind::Endpoint,
            source_path: Some(endpoint.file.clone()),
            segments: endpoint.segments.clone(),
        });
    }
    for route in &discovered.routes {
        patterns.push(RoutePattern {
            url_path: route.url_path.clone(),
            kind: PatternKind::Page,
            source_path: Some(route.page.clone()),
            segments: route.segments.clone(),
        });
    }
    Ok(Some(patterns))
}

/// A route/handler pattern the current [`RouteScope`] does NOT compile: enough to
/// MATCH a request (so a request for it is a "not built yet", not a 404) and nothing
/// else. Carries no module reference, so publishing it compiles nothing.
#[derive(Debug, Clone)]
struct UnbuiltPattern {
    url_path: String,
    segments: Vec<Seg>,
}

/// Everything a scope leaves out, by kind. Empty for [`RouteScope::All`].
#[derive(Debug, Clone, Default)]
struct UnbuiltPatterns {
    routes: Vec<UnbuiltPattern>,
    handlers: Vec<UnbuiltPattern>,
    pages_api: Vec<UnbuiltPattern>,
}

/// The patterns `scope` excludes, computed BEFORE [`apply_route_scope`] drops them.
fn unbuilt_patterns(disc: &Discovered, scope: &RouteScope) -> UnbuiltPatterns {
    if *scope == RouteScope::All {
        return UnbuiltPatterns::default();
    }
    let excluded = |included: bool, url_path: &String, segments: &Vec<Seg>| {
        (!included).then(|| UnbuiltPattern {
            url_path: url_path.clone(),
            segments: segments.clone(),
        })
    };
    UnbuiltPatterns {
        routes: disc
            .routes
            .iter()
            .filter_map(|route| {
                excluded(
                    scope.includes_page(&route.url_path),
                    &route.url_path,
                    &route.segments,
                )
            })
            .collect(),
        handlers: disc
            .handlers
            .iter()
            .filter_map(|handler| {
                excluded(
                    scope.includes_endpoint(&handler.url_path),
                    &handler.url_path,
                    &handler.segments,
                )
            })
            .collect(),
        pages_api: disc
            .pages_api
            .iter()
            .filter_map(|route| {
                excluded(
                    scope.includes_endpoint(&route.url_path),
                    &route.url_path,
                    &route.segments,
                )
            })
            .collect(),
    }
}

/// Drop every route/handler/endpoint `scope` excludes, so the generated entries import
/// only the modules the in-scope routes need.
///
/// The app-root conventions (root layout, `not-found`, `global-error`, metadata images)
/// are deliberately untouched: every document is built from them, so scoping them would
/// only mean rebuilding on the first request. Intercepting routes go with their target —
/// an intercept whose target is out of scope cannot be reached.
fn apply_route_scope(disc: &mut Discovered, scope: &RouteScope) {
    if *scope == RouteScope::All {
        return;
    }
    disc.routes
        .retain(|route| scope.includes_page(&route.url_path));
    disc.handlers
        .retain(|handler| scope.includes_endpoint(&handler.url_path));
    disc.pages_api
        .retain(|route| scope.includes_endpoint(&route.url_path));
    // An intercept is addressed by the URL of the route it intercepts; keep it exactly
    // when that route is compiled.
    let in_scope_paths: BTreeSet<&str> = disc.routes.iter().map(|r| r.url_path.as_str()).collect();
    disc.intercepts.retain(|intercept| {
        in_scope_paths.contains(segments_display(&intercept.target_segments).as_str())
    });
}

/// Every module the app's ROUTES are built from: each route's page and its whole
/// nested layout/loading/error/template chain (parallel `@slot` subtrees included),
/// the root layout, `not-found`, `global-error`, every `route.*` handler, and each
/// intercepting route's page + chain. These are the entry points the react-server
/// graph is rooted at, and therefore the definition of "a route can reach it".
fn route_module_roots(disc: &Discovered) -> Vec<PathBuf> {
    fn push_levels(levels: &[Level], out: &mut Vec<PathBuf>) {
        for level in levels {
            for path in [&level.layout, &level.loading, &level.error, &level.template]
                .into_iter()
                .flatten()
            {
                out.push(path.clone());
            }
            for slot in &level.slots {
                if let Some(default) = &slot.default {
                    out.push(default.clone());
                }
                for route in &slot.routes {
                    out.push(route.page.clone());
                    push_levels(&route.levels, out);
                }
            }
        }
    }
    let mut roots = Vec::new();
    for path in [&disc.root_layout, &disc.app_not_found, &disc.global_error]
        .into_iter()
        .flatten()
    {
        roots.push(path.clone());
    }
    for route in &disc.routes {
        roots.push(route.page.clone());
        push_levels(&route.levels, &mut roots);
    }
    for handler in &disc.handlers {
        roots.push(handler.file.clone());
    }
    for intercept in &disc.intercepts {
        roots.push(intercept.page.clone());
        push_levels(&intercept.levels, &mut roots);
    }
    roots.sort();
    roots.dedup();
    roots
}

/// Removes from `islands` every `"use client"` module that (a) no route can reach and
/// (b) cannot be built, returning the exclusions for the caller to report.
///
/// THE RULE. Island discovery is a filesystem walk, so it finds files that are not
/// part of the application at all — a leftover component, an `examples/` sketch, a
/// `__tests__` helper. Pinning them is harmless while they compile, and that
/// over-approximation is what keeps the React Client Manifest complete. But a pin is a
/// hard build dependency, so before this an unreachable file with an unresolvable
/// import failed the WHOLE build: dead code was fatal.
///
/// The rule chosen is the narrowest one that fixes exactly that: **an island is
/// dropped only when it is both unbuildable and unreachable.**
///
/// * Unbuildable AND reachable from a route — left pinned. The bundler reports it as
///   the fatal, specifier-naming diagnostic it already does. A real broken import in
///   live code must never be downgraded.
/// * Buildable but unreachable — still pinned, exactly as before. Nothing changes for
///   dead code that compiles.
/// * Unbuildable and unreachable — dropped, and reported by the caller naming the
///   island, the module that carries the bad specifier, and the specifier. Never
///   silent: the app is told precisely what was excluded and why.
///
/// The classification is deliberately asymmetric in the safe direction. Failing to
/// resolve something the bundler could resolve only makes an *unreachable* island
/// eligible for a drop, which is what the rule wants anyway; a reachable island is
/// never dropped whatever the probe thinks.
fn drop_unreachable_unbuildable_islands(
    islands: &mut Vec<PathBuf>,
    scan: &ProjectScan,
    disc: &Discovered,
    aliases: &[(String, String)],
) -> Vec<(PathBuf, crate::project_graph::UnresolvedImport)> {
    let route_roots = route_module_roots(disc);
    let mut seeds = route_roots.clone();
    seeds.extend(islands.iter().cloned());
    let graph = crate::project_graph::ProjectImportGraph::build(&seeds, &scan.sources, aliases);
    let mut unbuildable: Vec<(PathBuf, crate::project_graph::UnresolvedImport)> = Vec::new();
    for island in islands.iter() {
        if let Some(reason) = graph.first_unresolved_from(island) {
            unbuildable.push((island.clone(), reason));
        }
    }
    if unbuildable.is_empty() {
        return Vec::new();
    }
    let reachable = graph.reachable_from(&route_roots);
    unbuildable.retain(|(island, _)| !reachable.contains(island));
    let dropped: std::collections::HashSet<PathBuf> = unbuildable
        .iter()
        .map(|(island, _)| island.clone())
        .collect();
    islands.retain(|island| !dropped.contains(island));
    unbuildable
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
/// [`AppRouterAppConfig`] for `environment` (`client` | `react-server` | `ssr`/anything
/// else server-like). Returns `Ok(None)` for a non-Next project so the caller
/// falls back to the TanStack `derive_config` path unchanged. This is the
/// PRODUCTION entry point (`build-app`): byte-identical to before the dev server
/// existed.
pub fn configure_app_router(
    root: &Path,
    environment: &str,
) -> Result<Option<AppRouterAppConfig>, String> {
    // Production compiles every route: the output must be able to serve any of them.
    configure_inner(root, environment, false, &RouteScope::All)
}

/// The development variant of [`configure_app_router`] (the `diffpack dev` Next topology):
/// same scaffold, but the returned config is switched to development —
/// `build.hmr = true`, `process.env.NODE_ENV` defined as `"development"` (so React's
/// development build, which alone exposes the Fast Refresh renderer hook, is
/// bundled), and the resolve `production` condition swapped for `development`. All
/// three graphs run in development so the react-server/SSR React matches the client
/// React at hydration (no dev/prod hydration split). React 19.2.4 selects dev/prod
/// purely from `NODE_ENV` (its `exports` has no `development` condition), so the
/// condition swap is inert for React itself and only affects packages that publish a
/// `development`/`production` exports map.
///
/// `scope` decides which routes the generated entries import (see [`RouteScope`]): dev
/// starts narrow and widens as routes are asked for, so a cold start compiles the route
/// being loaded rather than the whole app.
pub fn configure_app_router_dev(
    root: &Path,
    environment: &str,
    scope: &RouteScope,
) -> Result<Option<AppRouterAppConfig>, String> {
    configure_inner(root, environment, true, scope)
}

fn configure_inner(
    root: &Path,
    environment: &str,
    dev: bool,
    scope: &RouteScope,
) -> Result<Option<AppRouterAppConfig>, String> {
    let root = root
        .canonicalize()
        .map_err(|error| format!("cannot open project root {}: {error}", root.display()))?;
    let Some(page) = detect_app_router(&root) else {
        return Ok(None);
    };
    let app_dir = app_dir(&root).ok_or_else(|| {
        format!(
            "{}: app-router detected but no app/ or src/app directory",
            root.display()
        )
    })?;
    let layout = first_existing(&app_dir, "layout");
    // ONE walk of the project (NOT of `app/` — client components and fonts routinely
    // live in siblings like `src/components`) yields the `"use client"` islands and the
    // `next/font` usages. The font transform (next_font.rs) rewrites the calls to
    // static objects and drops the import; here we generate the companion CSS (Google
    // @import + the CSS-variable classes) the render entry injects as a React-hoisted
    // <style> into the document head. The app's own stylesheet is NOT a fact of this
    // walk: the head <link> is derived from the react-server build's emitted
    // `server.css` at render time (see `rsc_entry_module`), which is the same artifact
    // `main.rs` preserves to `public/rsc.css`.
    let scan_stage = diffpack_core::build_profile::stage("adapter/scan-project");
    let scan = scan_project(&root)?;
    drop(scan_stage);
    let mut islands = scan.islands.clone();

    // Evaluate `next.config.*` ONCE for this pass (redirects/rewrites/headers + images +
    // the basePath/assetPrefix/trailingSlash/i18n routing surface). The result feeds the
    // image-config module, the baked asset/base-path prefixes below, and (on the
    // react-server pass) the config manifest — so node is spawned a single time instead of
    // once per consumer (a build-time win over the previous two spawns).
    let config_eval_stage = diffpack_core::build_profile::stage("adapter/next-config-eval");
    let next_config = run_next_config_eval(&root);
    drop(config_eval_stage);
    // Honor next.config `pageExtensions` (e.g. what `@next/mdx` merges): a configured
    // extension diffpack cannot discover is a clear hard error, never a silently-missing
    // route. Supported extensions (incl. md/mdx) flow through the built-in discovery set.
    validate_page_extensions(next_config.as_ref())?;
    // State what `createMDX`'s options were and how they are honoured — they must never be
    // dropped in silence (see `report_mdx_config`).
    report_mdx_config(next_config.as_ref());
    let routing = Routing::from_eval(next_config.as_ref());
    let base_path = routing.base_path.clone();
    let asset_base = routing.asset_base();

    let adapter_dir = root.join(ADAPTER_DIR);
    let shims_dir = adapter_dir.join("shims");
    std::fs::create_dir_all(&shims_dir)
        .map_err(|error| format!("cannot create {}: {error}", shims_dir.display()))?;

    // The font CSS is generated HERE rather than beside the project walk because a
    // `next/font/local` face carries a real URL (`<assetPrefix>/_diffpack-font/...`),
    // which only exists once `next.config`'s asset base is known. The same pass records
    // the source files behind those URLs in the adapter directory, and the client emit
    // copies exactly that list — so the emitted @font-face and the emitted file cannot
    // point at different things.
    let fonts = crate::next_font::generate(&root, &scan.fonts, &asset_base)?;
    crate::next_font::write_font_manifest(&adapter_dir, &fonts.assets)?;

    // `next/*` shims resolved as aliases (specifier -> shim file). Only the subset
    // this adapter faithfully implements; `next/font`, `next/headers` server APIs,
    // etc. are documented gaps and are intentionally NOT silently aliased to a
    // no-op (an app importing an unshimmed `next/*` fails at resolve, naming it).
    // Built here rather than at the config below because the island reachability probe
    // must resolve specifiers exactly as the build will.
    // (`react-server-dom-webpack/*` joins this list once the environment's resolve
    // conditions are known — see the RSC-runtime alias below. It is NOT needed for the
    // reachability probe: app code never imports the flight runtime, only diffpack's
    // generated entries do.)
    let alias = |spec: &str, file: &Path| (spec.to_string(), file.to_string_lossy().into_owned());
    let mut aliases = vec![
        alias("next/link", &shims_dir.join("link.tsx")),
        alias("next/image", &shims_dir.join("image.tsx")),
        alias("next/navigation", &shims_dir.join("navigation.ts")),
        alias("next/headers", &shims_dir.join("headers.ts")),
        alias("next/cache", &shims_dir.join("cache.ts")),
        alias("next/server", &shims_dir.join("server.ts")),
        alias("next/dynamic", &shims_dir.join("dynamic.ts")),
        alias("next/script", &shims_dir.join("script.tsx")),
    ];
    // The SAME entry points spelled with their file extension. The `next` package has no
    // `exports` map, so `next/navigation` and `next/navigation.js` name one and the same
    // file on disk — and an alias table that covers only the bare spelling silently
    // splits the module in two, half of it diffpack's shim and half of it Next's real
    // implementation. That is not hypothetical: `nuqs` (a dependency of cal.com, used on
    // most of its pages) imports `useRouter` from `"next/navigation.js"`, which reached
    // Next's own `useRouter` — and it throws `invariant expected app router to be
    // mounted` against a context nothing here provides, taking the whole SSR render of
    // every affected page down to an empty document.
    let extensioned: Vec<(String, String)> = aliases
        .iter()
        .map(|(spec, file)| (format!("{spec}.js"), file.clone()))
        .collect();
    aliases.extend(extensioned);

    // Every file those aliases name is written HERE, before anything resolves a
    // specifier. The island reachability probe below resolves `next/*` exactly as the
    // build will, and a resolver reports a missing file as "does not resolve" — so on a
    // tree with no `.diffpack-next/` (a genuinely cold build) a probe that ran before
    // these writes would judge every `next/link` importer unbuildable and silently drop
    // real islands, leaving the client graph short of the references the react-server
    // graph emits. Writing first makes the cold build identical to the warm one.
    // `request-context.ts` / `hooks-context.ts` come along because the `next/headers`,
    // `next/cache` and `next/navigation` shims import them, and the image manifest +
    // config because the `next/image` shim does.
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
    // The next.config `images` block (remote allow-list + loader), bundled into every
    // graph so the shim can allow/deny remote hosts and drive a custom/built-in loader.
    // Resolved FIRST because it also decides whether this build pre-optimizes at all
    // (see `ImageOptimization`), which the scan below obeys. Persisted alongside as
    // JSON so the later out-of-`configure` steps (main.rs's variant emit, the dev
    // server's) read the same resolved block without re-spawning node.
    let images = images_from_eval(next_config.as_ref());
    write_if_changed(
        &adapter_dir.join(IMAGE_CONFIG_JSON),
        &serde_json::to_string_pretty(&images)
            .map_err(|error| format!("cannot serialize the next.config images block: {error}"))?,
    )?;
    write_if_changed(
        &adapter_dir.join("image-config.ts"),
        &image_config_module(&images),
    )?;
    // next/image (Slice J / gap 4.2): generate the variant manifest the shim reads.
    // Scanning `public/` is deterministic (no build-output dependency), so it runs in
    // every environment and the manifest agrees across the three graphs; the actual
    // variant files are emitted once, from the client build's public-copy step
    // (main.rs `emit_image_variants`), keyed by the same deterministic hash.
    let images_stage = diffpack_core::build_profile::stage("image/scan-public");
    let optimization = ImageOptimization::from_images(&images);
    let public_images = scan_public_images_with(&root, &optimization)?;
    drop(images_stage);
    write_if_changed(
        &adapter_dir.join("image-manifest.ts"),
        &image_manifest_module(&public_images),
    )?;
    let link_shim = shims_dir.join("link.tsx");
    write_if_changed(&link_shim, &next_link_shim(&base_path))?;
    let script_shim = shims_dir.join("script.tsx");
    write_if_changed(&script_shim, next_script_shim())?;
    write_if_changed(&shims_dir.join("image.tsx"), &next_image_shim(&asset_base))?;
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

    // --- app-router route table (every route + its nested layout/boundary chain) --
    let _ = &page; // detection anchor; the full route set comes from discovery.
    let layout_abs = layout
        .as_ref()
        .map(|l| l.canonicalize().unwrap_or_else(|_| l.clone()));
    let mut discovered = discover_routes(&app_dir, layout_abs.as_deref())?;
    // Metadata FILE conventions (`app/sitemap.ts`/`robots.ts`/`manifest.ts`): synthesize
    // a wrapper + route-handler entry for each present one, so `/sitemap.xml`,
    // `/robots.txt`, `/manifest.webmanifest` are served through the SAME route-handler
    // dispatch as any `route.ts` endpoint. Distinct literal URLs, so appending (no
    // re-sort) preserves the most-specific-first invariant of the handler table.
    discovered
        .handlers
        .extend(synthesize_metadata_file_handlers(&app_dir, &shims_dir)?);
    // The FULL route table is always discovered (a directory walk, no compilation), so
    // the served route set and the compiled one stay separate facts: the dev server can
    // tell "a route that exists but is not compiled yet" from "no such route" and keep
    // 404 behaviour identical while compiling lazily. Only what the generated entries
    // IMPORT is scoped — the patterns the scope leaves out are still published (as pure
    // data, no imports) so the orchestrator can match them and ask for a build.
    let unbuilt = unbuilt_patterns(&discovered, scope);
    apply_route_scope(&mut discovered, scope);

    // Island discovery over-approximates on purpose (see `scan_project`), and a pin is
    // a hard build dependency — so a `"use client"` file that is BOTH unbuildable and
    // unreachable from every route would otherwise fail the whole build for dead code.
    // Drop exactly those, and say so: the app is told which file was excluded, which
    // module carries the bad specifier, and what the specifier was. A reachable island
    // is never dropped — its broken import stays the bundler's fatal diagnostic.
    for (island, reason) in
        drop_unreachable_unbuildable_islands(&mut islands, &scan, &discovered, &aliases)
    {
        eprintln!(
            "next app-router: excluded the unreachable \"use client\" module {} — no route imports it, and it cannot be built: {} imports {:?}, which does not resolve. Import it from a route to make this a build error instead.",
            island.display(),
            reason.file.display(),
            reason.specifier,
        );
    }

    // The generated client Error Boundary (a `"use client"` class component) wraps
    // each route level that has an `error.tsx`. Like the `next/link` shim it must be
    // BUNDLED + REGISTERED in the client + ssr graphs (`scan_project` skips
    // `.diffpack-next/`) so its client reference resolves; in the react-server graph
    // it stays a client reference. Keyed by the SAME canonical path the react-server
    // render imports it from → manifest ids match. Write it first so its path exists.
    // The CONTROL boundary island: same island contract (bundled + registered in the
    // client/ssr graphs, a client reference in the react-server graph). It completes a
    // redirect() thrown once the shell had already flushed, and it owns the ONE
    // definition of "this error is app-router control flow" that the error boundary and
    // the client entry both read. Written first: both of those import from it.
    let control_boundary = adapter_dir.join("control-boundary.tsx");
    write_if_changed(&control_boundary, control_boundary_module())?;
    let control_boundary_canon = control_boundary
        .canonicalize()
        .unwrap_or_else(|_| control_boundary.clone());
    if !islands.contains(&control_boundary_canon) {
        islands.push(control_boundary_canon.clone());
    }
    let error_boundary = adapter_dir.join("error-boundary.tsx");
    write_if_changed(
        &error_boundary,
        &error_boundary_module(&control_boundary_canon),
    )?;
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
    // The SEGMENT boundary island (useSelectedLayoutSegment(s)): like the error boundary
    // it is a `"use client"` island wrapped around each layout in the react-server render,
    // BUNDLED + REGISTERED in the client + ssr graphs so it PROVIDES SelectedSegmentContext
    // there, and a client reference in the react-server graph. It imports the shared
    // hooks-context, so write it after that file exists. Keyed by its canonical path so the
    // manifest ids match across graphs.
    let segment_boundary = adapter_dir.join("segment-boundary.tsx");
    write_if_changed(
        &segment_boundary,
        &segment_boundary_module(&hooks_context_canon),
    )?;
    let segment_boundary_canon = segment_boundary
        .canonicalize()
        .unwrap_or_else(|_| segment_boundary.clone());
    if !islands.contains(&segment_boundary_canon) {
        islands.push(segment_boundary_canon.clone());
    }
    write_if_changed(&adapter_dir.join("lazy.js"), lazy_module())?;
    // Middleware: `middleware.{ts,js}` at the project root or under `src/`. Next
    // middleware ALWAYS runs on the edge (WinterCG) runtime, so enforce the
    // no-Node-built-ins contract on its source (a real deployment rejects those
    // imports) — the served middleware runs in diffpack's edge context.
    let middleware = first_existing(&root, "middleware")
        .or_else(|| first_existing(&root.join("src"), "middleware"))
        .map(|p| p.canonicalize().unwrap_or(p));
    if let Some(mw) = middleware.as_deref() {
        let mw_source = std::fs::read_to_string(mw).unwrap_or_default();
        validate_edge_module(
            &format!("middleware ({}) [edge runtime]", mw.display()),
            &mw_source,
        )?;
    }
    write_if_changed(
        &adapter_dir.join("rsc-entry.tsx"),
        &rsc_entry_module(
            &discovered,
            &unbuilt,
            &fonts,
            &error_boundary_canon,
            &segment_boundary_canon,
            &control_boundary_canon,
            &request_context_canon,
            middleware.as_deref(),
            &asset_base,
        ),
    )?;
    // The `next/link` shim is a `"use client"` intercepting component. In the
    // react-server graph it stays a client reference (resolved to real code through
    // the seam); in the client + ssr graphs it must be BUNDLED and REGISTERED like
    // any island so its client reference resolves and it hydrates. Because
    // `scan_project` skips `.diffpack-next/`, pin it explicitly here, keyed by the SAME
    // canonical path the react-server render resolves the `next/link` alias to →
    // manifest ids match. The file itself was written above the reachability probe (it
    // has to exist before ANY specifier resolves); the pin belongs AFTER the probe so
    // diffpack's own shims are never candidates for being dropped as unreachable.
    let link_canon = link_shim
        .canonicalize()
        .unwrap_or_else(|_| link_shim.clone());
    if !islands.contains(&link_canon) {
        islands.push(link_canon.clone());
    }
    // `next/script` is the same shape: a `"use client"` component the react-server graph
    // sees only as a client reference. Next's own `next/script` is CommonJS inside
    // `node_modules`, which `scan_project` cannot see (it skips dependencies), so without
    // this alias + pin the flight carries a client reference no client manifest has an
    // entry for. Pinned by canonical path so all three graphs agree on the id; the file
    // itself was written above the reachability probe.
    let script_canon = script_shim
        .canonicalize()
        .unwrap_or_else(|_| script_shim.clone());
    if !islands.contains(&script_canon) {
        islands.push(script_canon.clone());
    }
    // Native Next's official App Page entry always installs its built-in global
    // error boundary in the loader tree. Pin that client boundary into the same
    // browser graph when the native bridge is active so its Flight reference has
    // a real client module and chunk, just like an application global-error.tsx.
    if std::env::var_os("DIFFPACK_NATIVE_NEXT_OUTPUT").is_some()
        && let Ok(next_root) = crate::rsc_runtime_resolve::installed_package_root(&root, "next")
    {
        for relative in [
            "dist/client/components/builtin/global-error.js",
            "dist/client/components/client-page.js",
            "dist/client/components/client-segment.js",
            "dist/client/components/http-access-fallback/error-boundary.js",
            "dist/client/components/instant-validation/boundary.js",
            "dist/client/components/layout-router.js",
            "dist/client/components/render-from-template-context.js",
            "dist/lib/framework/boundary-components.js",
            "dist/lib/metadata/generate/icon-mark.js",
        ] {
            let builtin = next_root.join(relative);
            if builtin.is_file() && !islands.contains(&builtin) {
                islands.push(builtin);
            }
        }
    }
    // THE PIN SET. A pin exists for exactly one reason: to put an island into the client
    // and SSR graphs so the client reference the flight carries for it resolves. The
    // react-server graph knows precisely which those are — every `"use client"` module it
    // reaches IS a reference boundary — so when that set has been recorded
    // ([`REFERENCED_ISLANDS_FILE`]) it is authoritative, and the project walk's
    // over-approximation (231 islands on cal.com against 101 the app references and 11 a
    // single route does) stops being compiled into two graphs for nothing.
    //
    // diffpack's OWN generated islands are unioned in unconditionally rather than trusted
    // to appear: they are pinned by construction (the walk skips `.diffpack-next/`), they
    // cost five modules, and one missing from the recorded set would be a render-time
    // "Could not find the module in the React Client Manifest" rather than a build error.
    //
    // No recorded set (a production build, or the very first react-server build of a cold
    // tree) falls back to pinning the walk, which is the pre-existing behaviour.
    let adapter_pins = [
        control_boundary_canon.clone(),
        error_boundary_canon.clone(),
        segment_boundary_canon.clone(),
        link_canon.clone(),
        script_canon.clone(),
    ];
    // DEV ONLY. A production build must pin from its own walk: it is a single pass that
    // emits the client graph FIRST (nothing has recorded a referenced set yet), and reading
    // a file some earlier `diffpack dev` left behind would make the build's output depend on
    // whether a dev server had ever run in this tree. That is not hypothetical — it silently
    // shrank a production pin set to a dev route scope's, and the next/image gate caught it
    // as a hero image with no srcset candidates.
    let islands: Vec<PathBuf> = match referenced_islands(&adapter_dir).filter(|_| dev) {
        Some(referenced) => {
            let mut pinned: BTreeSet<PathBuf> = referenced.iter().map(PathBuf::from).collect();
            pinned.extend(adapter_pins.iter().cloned());
            pinned.into_iter().collect()
        }
        None => islands,
    };
    // Record the pinned-island list (canonical) and split the pins into lazy
    // thunks vs the recorded async set's eager imports — see `island_pins`.
    let canonical_islands: Vec<String> = islands
        .iter()
        .map(|island| {
            island
                .canonicalize()
                .unwrap_or_else(|_| island.clone())
                .to_string_lossy()
                .into_owned()
        })
        .collect();
    write_if_changed(
        &adapter_dir.join(ISLANDS_FILE),
        &serde_json::to_string_pretty(&canonical_islands)
            .map_err(|error| format!("cannot serialize the island list: {error}"))?,
    )?;
    let eager_islands = recorded_eager_islands(&adapter_dir);
    write_if_changed(
        &adapter_dir.join("server.tsx"),
        &ssr_entry_module(
            &adapter_dir,
            &islands,
            &eager_islands,
            &hooks_context_canon,
            &asset_base,
            &discovered.pages_api,
        ),
    )?;
    write_if_changed(
        &adapter_dir.join("client.tsx"),
        &client_entry_module(
            &adapter_dir,
            &islands,
            &eager_islands,
            &hooks_context_canon,
            // DEV splits the islands into per-island chunks so a page downloads the
            // islands it renders instead of every island in the app: cal.com's dev
            // `client.js` is 17.8 MB of which /auth/login needs a fraction.
            //
            // Production still pins statically, and NOT because it does not need this —
            // cal.com's production `client.js` is 8.1 MB for exactly the same reason, and
            // minification plus DCE is all that separates it from the dev number. It is
            // gated because the browser has to be told which chunks a route needs before
            // it hydrates, and production serves a STREAMING document, which has no place
            // to put that list yet (the react-server render discovers the references as it
            // serializes, so the list is only complete once the flight is). The buffered
            // dev document drains the flight first, so it has the list in time.
            // Per-island chunks: a page downloads the islands it renders instead of every
            // island in the app. Static pins put all 229 of cal.com's into one chunk —
            // 17.8 MB in dev, 8.1 MB minified in production, and every route paid for all
            // of it.
            //
            // Both modes. What makes this safe is that the document DECLARES the chunks
            // a route's client references live in and the browser entry loads them before
            // hydrating (`recordReferenceChunks` in the SSR entry, `loadDeclaredChunks` in
            // the browser entry) — the same property `next build` gets by emitting a
            // `<script>` per route chunk. Without it the split hydrates markup whose
            // islands have no handlers yet, which cal.com's own suite catches: a theme
            // option clicked before its island arrives leaves the form clean and its submit
            // button disabled forever.
            PinKind::DynamicChunk,
        ),
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
            Target::IsolatedServer,
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

    // The RSC runtime diffpack's own entries import. A real Next app does not depend on
    // `react-server-dom-webpack` — Next vendors it — so it is resolved from the copy
    // `next` ships, through that copy's `exports` under THESE conditions (the flight
    // writer exists only under `react-server`; the client half differs browser vs node).
    // An app that installs its own copy gets no alias and keeps it. See
    // `rsc_runtime_resolve` for what happens when neither exists.
    aliases.extend(crate::rsc_runtime_resolve::aliases(
        &root,
        &conditions,
        target == Target::Client,
    ));

    // React itself, from the copy `next` vendors — in EVERY layer, which is what
    // `next build` does. Unlike the flight runtime above, the app's own dependency
    // does not win: an App Router app is compiled against the React Next ships, and
    // an app on React 18 has no working `react-server` entry at all (React 18.2's
    // resolves to a module whose whole body throws). See `rsc_runtime_resolve`.
    aliases.extend(crate::rsc_runtime_resolve::react_aliases(
        &root,
        &conditions,
        target == Target::Client,
    ));

    // A CLIENT bundle may import the Node built-ins Next polyfills (`url`, `path`,
    // `process`, ...): Next's client webpack config maps them onto the copies it
    // vendors, so such an import is valid Next code that `next build` accepts. This is
    // the same `resolve.fallback` the pages adapter applies — it belongs to the client
    // compiler, not to a router. Server/react-server bundles keep the real built-ins.
    if target == Target::Client {
        aliases.extend(next_browser_polyfill_aliases(&root));
    }
    if dev {
        // Next's development websocket module names Turbopack's private HMR
        // client. Non-Turbopack compilers intentionally map that import to
        // Next's own no-op adapter in every compiler layer (see
        // `create-compiler-aliases.ts`). Diffpack carries its update transport
        // separately, so reproduce that contract instead of trying to host
        // Turbopack's runtime.
        if let Ok(next_root) = crate::rsc_runtime_resolve::installed_package_root(&root, "next") {
            let noop_hmr = next_root.join("dist/client/dev/noop-turbopack-hmr.js");
            aliases.push((
                "@vercel/turbopack-ecmascript-runtime/browser/dev/hmr-client/hmr-client.ts"
                    .to_string(),
                noop_hmr.to_string_lossy().into_owned(),
            ));
        }
    }

    // React's dev/prod dispatch define. Production bundles the production React
    // (small, no dev warnings); DEV bundles the development React whose renderer
    // exposes the Fast Refresh hook the island HMR path needs.
    let node_env = if dev {
        "\"development\""
    } else {
        "\"production\""
    };
    let mut defines = vec![
        ("process.env.NODE_ENV".to_string(), node_env.to_string()),
        (
            "process.env.NEXT_RUNTIME".to_string(),
            next_runtime_define(target).to_string(),
        ),
        (
            "process.browser".to_string(),
            process_browser_define(target).to_string(),
        ),
    ];
    if !dev {
        // Next's shared app entry-base is authored for Turbopack and references
        // these compiler intrinsics as free identifiers. A production build has
        // no hot-update cache to clear or apply; compile the optional hooks to
        // null exactly as the entry-base's declared contract permits.
        defines.extend([
            (
                "__turbopack_clear_chunk_cache__".to_string(),
                "null".to_string(),
            ),
            (
                "__turbopack_server_hmr_apply__".to_string(),
                "null".to_string(),
            ),
        ]);
    }
    // `NEXT_PUBLIC_*` is inlined in EVERY compilation, exactly as `next build`
    // does (see `next_public_env`). Server graphs additionally get the full
    // config environment at runtime via the manifest; the browser has only
    // these inlines, which is the whole point of the prefix.
    for (name, value) in next_public_env(&root, dev, next_config.as_ref()) {
        defines.push((
            format!("process.env.{name}"),
            serde_json::to_string(&value).expect("serializing an environment string cannot fail"),
        ));
    }

    // Evaluate `next.config` once (on the react-server pass) into the routing-rules
    // manifest the orchestrator applies (redirects/rewrites/headers). Best-effort: a
    // failing/absent config yields empty rules, never a build failure.
    if environment == "react-server" {
        write_next_config_manifest(&root, next_config.as_ref());
    }

    Ok(Some(AppRouterAppConfig {
        environment: environment.to_string(),
        build: AppRouterBuildConfig {
            base: "/".to_string(),
            browser_process_shim: true,
            asset_inline_limit: 4096,
            aliases,
            conditions,
            main_fields: Vec::new(),
            virtual_modules: Vec::new(),
            private_chunk_names: Vec::new(),
            target,
            source_policy: std::sync::Arc::new(crate::source_policy::NextSourcePolicy {
                defines,
                external_singletons: Vec::new(),
                ..Default::default()
            }),
            hmr: dev,
            // Next's own source-map policy, so `diffpack build-app` and `next build`
            // of the same app produce comparable artifacts (see
            // `default_source_maps`). The CLI's `--sourcemap` / `--no-sourcemap`
            // override this afterwards; with neither, the app's next.config decides.
            source_maps: default_source_maps(target, dev, next_config.as_ref()),
            scss: diffpack_default_loader::sass::ScssOptions {
                additional_data: None,
                root: Some(root.to_path_buf()),
            },
            // Next static-image imports (`import img from './x.png'`) yield the
            // `{ src, width, height, blurDataURL, variants }` object shape with
            // build-emitted responsive variants. ONLY the Next adapter opts in;
            // Vite/TanStack/generic builds keep bare-URL-string asset imports.
            // The ladder obeys the same next.config decision as the `public/` one:
            // an app with `images.unoptimized` (or its own loader) gets the object
            // WITHOUT `variants`, so nothing is encoded for URLs the emitted `<img>`
            // can never reference.
            image_import_shape: diffpack_default_loader::ImageImportShape::NextObject {
                responsive_variants: optimization == ImageOptimization::Enabled,
            },
            css_preprocess: diffpack_default_loader::CssPreprocess {
                root: Some(root.to_path_buf()),
                postcss: diffpack_default_loader::postcss::discover(&root).map(std::sync::Arc::new),
            },
            // Next compiles JSX in `.js`/`.mjs`/`.cjs` (its SWC loader enables jsx
            // for everything that is not a plain `.ts`).
            jsx_extensions: diffpack_core::parser::JsxExtensions::JsxInJavaScript,
            // Next does not expose Vite's `esbuild.jsx*` knobs; a Next app that
            // wants a different JSX runtime says so in its tsconfig, which
            // `jsx_config_for` reads per file.
            jsx: diffpack_core::transform::JsxConfig::default(),
            // next.config `serverExternalPackages`: never bundled into a server graph,
            // `require`d from node_modules at serve time instead. The list exists
            // because bundling these packages FAILS, so ignoring it turns a config the
            // app already wrote into a build error (cal.com's `rest-facade` reaches an
            // uninstalled `superagent-proxy` behind a runtime `if`).
            server_external_packages: server_external_packages(next_config.as_ref()),
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
pub(crate) fn run_next_config_eval(root: &Path) -> Option<serde_json::Value> {
    crate::next_config::run_next_config_eval(root)
}

/// Distinguishes concurrent config-eval payload files (each environment's build
/// evaluates the config, and builds can run in parallel).

/// The environment variables evaluating `next.config` ADDED or CHANGED, as reported by
/// `next-config-eval.mjs`.
///
/// `next dev` and `next build` load next.config in the same process that then compiles
/// and serves, so the config's side effects on `process.env` are part of the environment
/// the app runs under. Real apps depend on exactly that: cal.com's config is essentially
/// `dotenv.config({ path: "../../.env" })` plus a handful of computed variables, and its
/// `DATABASE_URL` exists nowhere else — without this the SSR server connected to a
/// default-named database and every data-backed route failed.
///
/// Diffpack evaluates the config in a CHILD process (so a config that throws cannot take
/// the build down), which is why the delta has to be carried back out and applied to
/// every process diffpack spawns. Removals are reported by the script but not applied:
/// unsetting a variable the real environment provided is not something the propagation
/// can do to an already-spawned parent, and no config in the wild deletes one.
/// Configuration environment values are read back from the persisted manifest for steps that
/// run AFTER the compile (the SSG prerenderer, the dev orchestrator) and therefore no
/// longer hold the evaluated config in memory. Empty when there is no manifest — an app
/// with no `next.config` has no side effects to propagate.
pub fn config_env_from_manifest(root: &Path) -> Vec<(String, String)> {
    config_env_from_output(&root.join(".diffpack-output"))
}

/// The same propagation keyed off the BUILD OUTPUT directory instead of the project root.
///
/// `diffpack start <output> [port]` is handed the output directory and nothing else — the
/// project root it was built from is not knowable from there (and may not even exist on
/// the serving host). Serving without this is what made every data-backed route fail:
/// cal.com's `DATABASE_URL` lives only in the `.env` its next.config `dotenv`-loads, so
/// the production server connected to a database named after the OS user and every
/// `prisma` call threw inside the Server Components render.
pub fn config_env_from_output(output: &Path) -> Vec<(String, String)> {
    let path = output.join("next-config-manifest.json");
    let Ok(text) = std::fs::read_to_string(&path) else {
        return Vec::new();
    };
    let Ok(value) = serde_json::from_str::<serde_json::Value>(&text) else {
        return Vec::new();
    };
    config_env(Some(&value))
}

/// Whether a Next build emits source maps for one graph BY DEFAULT.
///
/// This is Next's own policy, which diffpack follows so that `diffpack build-app`
/// and `next build` of the same app produce comparable artifacts — otherwise a
/// build-time comparison between them is measuring different work:
///
/// - **Server** graphs (`react-server`, `ssr`) get maps only when the app enables
///   Next's `experimental.serverSourceMaps`. Next defaults this off.
/// - **Browser** graphs get maps only when the app asks, via
///   `productionBrowserSourceMaps`. These ship to every visitor and publish the
///   app's source, so Next defaults them off and so does diffpack.
/// - In **dev** both graphs get maps: that is what `next dev` does, and it is the
///   setting a developer is actually debugging under.
pub(crate) fn default_source_maps(
    target: Target,
    dev: bool,
    next_config: Option<&serde_json::Value>,
) -> bool {
    if dev {
        return true;
    }
    match target {
        Target::Client => production_browser_source_maps(next_config),
        Target::Server | Target::IsolatedServer => server_source_maps(next_config),
    }
}

/// next.config `experimental.serverSourceMaps`, flattened by the config evaluator.
pub(crate) fn server_source_maps(eval: Option<&serde_json::Value>) -> bool {
    eval.and_then(|value| value.get("serverSourceMaps"))
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false)
}

/// next.config `productionBrowserSourceMaps`, as reported by `next-config-eval.mjs`.
/// Next's default is off, and so is the answer for an app with no config at all.
pub(crate) fn production_browser_source_maps(eval: Option<&serde_json::Value>) -> bool {
    eval.and_then(|value| value.get("productionBrowserSourceMaps"))
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false)
}

/// next.config `serverExternalPackages` (and the pre-15
/// `experimental.serverComponentsExternalPackages`), as reported by
/// `next-config-eval.mjs`. Empty when there is no config or it names none.
pub(crate) fn server_external_packages(eval: Option<&serde_json::Value>) -> Vec<String> {
    eval.and_then(|value| value.get("serverExternalPackages"))
        .and_then(|value| value.as_array())
        .map(|list| {
            list.iter()
                .filter_map(|entry| entry.as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_default()
}

/// Every `NEXT_PUBLIC_*` variable as `next build` would see it at compile time,
/// in Next's own precedence: the `@next/env` file stack (`.env` <
/// `.env.<mode>` < `.env.local` < `.env.<mode>.local`, never overriding a
/// variable the real environment already has), then the real process
/// environment, then — overriding everything, because in `next build` the
/// config runs in-process BEFORE compilation and assigns `process.env`
/// directly — the side effects `next.config` evaluation reported.
///
/// `next build` INLINES each of these as a `process.env.NEXT_PUBLIC_X` define
/// in every compilation (`next/dist/build/define-env`). diffpack was not
/// inlining any of them, so client code read them off the browser process shim
/// at runtime and got `undefined`: cal.com's footer rendered `v.undefined-sh`,
/// its `cityTimezones` query sent `CalComVersion: null` and got a 400, and
/// react-query's retries re-rendered the booker's timezone select forever —
/// the residual render loop under cal.com's own Playwright suite once the
/// styled-jsx hydration mismatch was fixed.
///
/// Only the `NEXT_PUBLIC_` prefix is collected: inlining anything else would
/// bake server secrets into the browser bundle, which Next never does.
fn next_public_env(
    root: &Path,
    dev: bool,
    eval: Option<&serde_json::Value>,
) -> Vec<(String, String)> {
    let mode = if dev { "development" } else { "production" };
    let mut merged: BTreeMap<String, String> = BTreeMap::new();
    for file_name in [
        ".env".to_string(),
        format!(".env.{mode}"),
        ".env.local".to_string(),
        format!(".env.{mode}.local"),
    ] {
        let path = root.join(&file_name);
        let Ok(text) = std::fs::read_to_string(&path) else {
            continue;
        };
        // A malformed env file is reported by the config-eval path already;
        // here it only loses the file's contribution to the inline set.
        let Ok(pairs) =
            diffpack_default_loader::env_file::parse(&text, &path.display().to_string())
        else {
            continue;
        };
        for (name, value) in pairs {
            merged.insert(name, value);
        }
    }
    for (name, value) in std::env::vars() {
        merged.insert(name, value);
    }
    for (name, value) in config_env(eval) {
        merged.insert(name, value);
    }
    merged
        .into_iter()
        .filter(|(name, _)| name.starts_with("NEXT_PUBLIC_"))
        .collect()
}

pub(crate) fn config_env(eval: Option<&serde_json::Value>) -> Vec<(String, String)> {
    let Some(map) = eval
        .and_then(|value| value.get("env"))
        .and_then(|v| v.as_object())
    else {
        return Vec::new();
    };
    let mut out = map
        .iter()
        .filter_map(|(key, value)| Some((key.clone(), value.as_str()?.to_string())))
        .collect::<Vec<_>>();
    out.sort();
    out
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

/// The machine-readable twin of the generated `image-config.ts`: the SAME resolved
/// `images` block, persisted so the later build steps that run outside `configure`
/// (`main.rs`'s public-image emit, the dev server's) can read what the app configured
/// without re-spawning node to re-evaluate `next.config`.
pub(crate) const IMAGE_CONFIG_JSON: &str = "image-config.json";

/// Whether this build pre-optimizes images at all — and, when it does not, the
/// next.config setting that says so.
///
/// `next build` runs NO image work at build time in two cases, and diffpack must not
/// either:
///
/// * `images.unoptimized` — Next's contract is that `<Image>` degrades to a plain
///   `<img src>` with no `srcset` and no `/_next/image` URL. Nothing can ever request
///   an optimized variant, so every emitted variant file is dead weight.
/// * `images.loader` other than `"default"`, or an `images.loaderFile` — the app's own
///   loader builds every URL, so diffpack's `/_next/image` optimizer (and therefore its
///   build-emitted variants behind it) is never reached.
///
/// This is not a heuristic and not a fallback: in both cases the emitted `srcset` is
/// decided by the shim from the SAME config, so "no variant is reachable" is a fact
/// about the generated HTML, not a guess.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ImageOptimization {
    /// Scan `public/` and emit the responsive variant ladder.
    Enabled,
    /// Off. Carries the next.config setting that turned it off, for the build log.
    Disabled(String),
}

impl ImageOptimization {
    /// Read the resolved `images` block and decide.
    pub(crate) fn from_images(images: &serde_json::Value) -> Self {
        if images
            .get("unoptimized")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false)
        {
            return ImageOptimization::Disabled("images.unoptimized is true".to_string());
        }
        if let Some(file) = images.get("loaderFile").and_then(serde_json::Value::as_str)
            && !file.is_empty()
        {
            return ImageOptimization::Disabled(format!("images.loaderFile is {file:?}"));
        }
        match images.get("loader").and_then(serde_json::Value::as_str) {
            Some(loader) if loader != "default" => {
                ImageOptimization::Disabled(format!("images.loader is {loader:?}"))
            }
            _ => ImageOptimization::Enabled,
        }
    }

    /// Read the decision back from the persisted `image-config.json` the adapter wrote
    /// for this project. A project with no adapter dir (a non-Next build) has no image
    /// config to honor, so optimization stays on and the existing behaviour is unchanged.
    pub fn for_project(root: &Path) -> Self {
        let path = root.join(ADAPTER_DIR).join(IMAGE_CONFIG_JSON);
        let Ok(text) = std::fs::read_to_string(&path) else {
            return ImageOptimization::Enabled;
        };
        match serde_json::from_str::<serde_json::Value>(&text) {
            Ok(images) => ImageOptimization::from_images(&images),
            // A corrupt file is a build defect, not something to silently ignore.
            Err(error) => {
                eprintln!(
                    "next/image: cannot read {} ({error}); building the full variant ladder",
                    path.display(),
                );
                ImageOptimization::Enabled
            }
        }
    }
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
    out.push_str(&format!(
        "  deviceSizes: {},\n",
        field("deviceSizes", "null")
    ));
    out.push_str(&format!("  imageSizes: {},\n", field("imageSizes", "null")));
    out.push_str(&format!(
        "  remotePatterns: {},\n",
        field("remotePatterns", "[]")
    ));
    out.push_str(&format!("  domains: {},\n", field("domains", "[]")));
    out.push_str(&format!("  loader: {},\n", field("loader", "\"default\"")));
    out.push_str(&format!("  path: {},\n", field("path", "\"/_next/image\"")));
    out.push_str(&format!("  qualities: {},\n", field("qualities", "null")));
    out.push_str(&format!(
        "  unoptimized: {},\n",
        field("unoptimized", "false")
    ));
    out.push_str(&format!(
        "  loaderFn: {},\n",
        if loader_file.is_some() {
            "__loaderFile"
        } else {
            "null"
        }
    ));
    out.push_str("};\n");
    out
}

/// The empty-config manifest: well-formed so the orchestrator's `routing` reader always
/// finds every field (no config, a config that throws, or missing node all land here).
const EMPTY_CONFIG_MANIFEST: &str = r#"{"redirects":[],"rewrites":[],"headers":[],"basePath":"","assetPrefix":"","trailingSlash":false,"i18n":null}"#;

/// Persist the single `run_next_config_eval` result to
/// `.diffpack-output/next-config-manifest.json` (the redirects/rewrites/headers rules +
/// the basePath/assetPrefix/trailingSlash/i18n routing surface the orchestrator applies).
/// No re-spawn of node: the caller already evaluated the config once for this pass.
fn write_next_config_manifest(root: &Path, eval: Option<&serde_json::Value>) {
    let output = root.join(".diffpack-output");
    let _ = std::fs::create_dir_all(&output);
    let manifest_path = output.join("next-config-manifest.json");
    let text = match eval {
        Some(value) => value.to_string(),
        None => EMPTY_CONFIG_MANIFEST.to_string(),
    };
    // Skip the write when the bytes already match: the artifact's mtime then
    // stays stable across a rebuild that reproduced it, which the dev warm
    // start relies on to prove "the rebuild changed nothing".
    if std::fs::read_to_string(&manifest_path).ok().as_deref() != Some(text.as_str()) {
        let _ = std::fs::write(&manifest_path, text);
    }
}

fn write_if_changed(path: &Path, contents: &str) -> Result<(), String> {
    if let Ok(existing) = std::fs::read_to_string(path)
        && existing == contents
    {
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

/// The wrapper for an id-partitioned sitemap (`generateSitemaps`). Served at
/// `/sitemap/[id].xml`: the `id` segment arrives as `"<id>.xml"` (Next's URL shape), so the
/// wrapper strips the `.xml` suffix, validates the id against `generateSitemaps()`'s
/// enumeration (a 404 for an unknown id, matching Next), coerces it back to the enumerated
/// value's type, then serializes `sitemap({ id })`. Exports `GET`, so it plugs straight into
/// the route-handler dispatch (which passes `{ params }`).
fn sitemap_id_wrapper(user_file: &Path, serializer: &Path) -> String {
    let user = js_str(&user_file.to_string_lossy());
    let ser = js_str(&serializer.to_string_lossy());
    format!(
        "// Generated by diffpack's next app-router adapter. Serves an id-partitioned\n\
         // `sitemap` (generateSitemaps) through the standard route-handler dispatch.\n\
         import handler, {{ generateSitemaps }} from {user};\n\
         import {{ serializeSitemap }} from {ser};\n\
         export async function GET(request, {{ params }}) {{\n  \
           const p = await params;\n  \
           const idStr = String(p.id).replace(/\\.xml$/, \"\");\n  \
           const maps = (await generateSitemaps()) || [];\n  \
           const match = maps.find((m) => String(m && m.id) === idStr);\n  \
           if (!match) return new Response(\"Not Found\", {{ status: 404 }});\n  \
           const body = serializeSitemap(await handler({{ id: match.id }}));\n  \
           return new Response(body, {{ status: 200, headers: {{ \"content-type\": \"application/xml\" }} }});\n\
         }}\n",
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
fn error_boundary_module(control_boundary: &Path) -> String {
    let control_import = js_str(&control_boundary.to_string_lossy());
    format!(
        r#""use client";
// Generated by diffpack's next app-router adapter — the client Error Boundary that
// implements the app-router `error.tsx` convention. React error boundaries must be
// client components; this wraps each route level that has an `error.tsx`.
import {{ Component, createElement }} from "react";
import {{ isControlFlowError }} from {control_import};

export default class ErrorBoundary extends Component {{
  constructor(props) {{
    super(props);
    this.state = {{ error: null }};
  }}
  static getDerivedStateFromError(error) {{
    return {{ error }};
  }}
  render() {{
    const error = this.state.error;
    if (error) {{
      // A redirect()/notFound() is CONTROL FLOW, not a failure. `error.tsx` must never
      // swallow it — showing an error page for a logged-out redirect is exactly what an
      // app-router error boundary is forbidden to do — so hand it to the control boundary
      // above, which completes the navigation. (Next's ErrorBoundaryHandler does the same.)
      if (isControlFlowError(error)) throw error;
      return createElement(this.props.fallback, {{
        error,
        reset: () => this.setState({{ error: null }}),
      }});
    }}
    return this.props.children;
  }}
}}
"#,
    )
}

/// The `CONTROL_BOUNDARY` island (`control-boundary.tsx`). A `"use client"` boundary
/// wrapped around the matched PAGE so a `redirect()` the page throws from BEHIND a
/// Suspense boundary still happens.
///
/// Once the shell has flushed, the HTTP status is already on the wire and no 307 can be
/// issued (that is Next's behaviour too: a route with a `loading.tsx` answers 200 and the
/// redirect travels in the stream). React hands the errored flight row to the client with
/// its `digest` attached; this boundary is where the digest is turned back into a real
/// navigation — the same job Next's `RedirectBoundary` does.
///
/// It is deliberately NOT a general error boundary: anything that is not a redirect digest
/// is re-thrown from `render`, so the app's own `error.tsx` / `global-error.tsx` chain sees
/// exactly the failure it saw before.
fn control_boundary_module() -> &'static str {
    r#""use client";
// Generated by diffpack's next app-router adapter — the client CONTROL boundary that
// completes a redirect() thrown after the response shell already flushed, plus the ONE
// definition of "this error is app-router control flow" (the error boundary and the
// client entry's onRecoverableError both import it from here).
import { Component, createElement } from "react";

// NEXT_REDIRECT;<type>;<url>;<status>; — the digest `next/navigation`'s redirect() throws.
// The URL itself may contain `;`, so the middle is rejoined rather than indexed.
export function redirectFromDigest(digest) {
  if (typeof digest !== "string" || !digest.startsWith("NEXT_REDIRECT;")) return null;
  const parts = digest.split(";");
  return { href: parts.slice(2, -2).join(";"), type: parts[1] || "replace" };
}

// notFound() thrown after the shell flushed. The status is already sent, so this cannot
// become a 404 — but it must never be silent either.
export function isNotFoundDigest(digest) {
  return digest === "NEXT_HTTP_ERROR_FALLBACK;404";
}

// Whether an error is app-router CONTROL FLOW rather than a failure. React wraps an error
// it recovers from (`new Error(<react code>, { cause: original })`) before handing it on,
// so the digest can sit a few `cause` hops down; the walk is bounded so a self-referential
// or cyclic cause chain terminates.
export function isControlFlowError(error) {
  for (let e = error, hops = 0; e && hops < 8; e = e.cause, hops += 1) {
    const digest = e.digest;
    if (redirectFromDigest(digest) || isNotFoundDigest(digest)) return true;
  }
  return false;
}

export default class ControlBoundary extends Component {
  constructor(props) {
    super(props);
    this.state = { error: null };
  }
  static getDerivedStateFromError(error) {
    return { error };
  }
  componentDidCatch(error) {
    const target = redirectFromDigest(error && error.digest);
    if (!target) return;
    const hard = () => {
      if (typeof window === "undefined") return;
      if (target.type === "push") window.location.assign(target.href);
      else window.location.replace(target.href);
    };
    // Catching the SAME target twice means the soft navigation did not actually replace
    // this subtree. Escalate to a real browser navigation instead of looping — the user
    // still lands where the server sent them.
    if (this.redirectedTo === target.href) {
      hard();
      return;
    }
    this.redirectedTo = target.href;
    // Prefer the client Router's soft navigation (keeps the document mounted); before it
    // has installed itself (a redirect caught during hydration) a real browser navigation
    // is always correct.
    const navigate = typeof window !== "undefined" && window.__diffpack_navigate;
    if (typeof navigate !== "function") {
      hard();
      return;
    }
    // This boundary is holding its subtree at `null` while the navigation runs. React
    // reuses a boundary instance that lands in the same position of the NEXT tree, so the
    // error state has to be cleared once the Router has swapped — otherwise the target
    // route renders into a boundary that is still showing nothing, and the user gets a
    // blank page after a redirect that "worked".
    Promise.resolve(navigate(target.href, { replace: target.type !== "push" })).then(
      () => this.setState({ error: null }),
      () => hard(),
    );
  }
  render() {
    const error = this.state.error;
    if (!error) return this.props.children;
    if (redirectFromDigest(error.digest)) {
      // The navigation is under way (componentDidCatch); render nothing in this subtree
      // rather than the half-rendered page it belongs to.
      return null;
    }
    if (isNotFoundDigest(error.digest)) {
      // notFound() from behind a Suspense boundary: the 200 is already sent, so this is
      // the app's `not-found.tsx` UI in Next. diffpack renders the not-found DOCUMENT
      // server-side (a real 404) only for a notFound() raised while the response still
      // had no bytes; reaching here means the throw came too late for that. Say so —
      // loudly, once — instead of showing a blank subtree with no explanation.
      console.error(
        "diffpack next: notFound() was thrown after the response shell had already flushed (" +
          (typeof location !== "undefined" ? location.pathname : "") +
          "). The 200 status cannot be withdrawn and the app's not-found.tsx is not part of " +
          "this document, so the built-in 404 body is shown instead.",
      );
      return createElement("main", { id: "not-found" }, "404 — This page could not be found.");
    }
    // Not ours: hand it to the app's own error boundaries unchanged.
    throw error;
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
    let secret = js_str(&draft_secret());
    format!(
        r#"// Generated by diffpack's next app-router adapter — the per-request AsyncLocalStorage
// that carries {{ url, headers, cookieHeader, params, responseCookies }} from the HTTP
// request into async Server Components (next/headers cookies()/headers()/draftMode()).
// One shared instance across the react-server graph (rsc-entry establishes it; the
// next/headers + next/cache shims read it). DRAFT_SECRET signs the draftMode bypass cookie.
import {{ AsyncLocalStorage }} from "node:async_hooks";
export const requestAls = new AsyncLocalStorage();
// The `"use cache"` COLLECTION scope: while a cached export runs, cacheTag()/cacheLife()
// (next/cache) record into the active scope here; the __diffpackUseCache wrapper reads
// the collected tags/TTL back out. Kept in this always-loaded singleton (not the
// next/cache shim) so a tagged fetch below can feed it without an import cycle.
export const cacheScopeAls = new AsyncLocalStorage();
export const DRAFT_SECRET = {secret};

// next/cache tagged fetch: `fetch(url, {{ next: {{ tags, revalidate }} }})`. Next patches
// global fetch so a tag on a fetch registers the reading page under that tag (bustable by
// revalidateTag) and `revalidate` bounds the datum's freshness. diffpack does the same
// natively: the tags are added to the current request store (page registration) AND, if a
// `"use cache"` scope is active, to that scope (so revalidateTag purges the memo); the
// numeric `revalidate` tightens the active cache scope's TTL. The non-standard `next`
// option is stripped before delegating so the platform fetch never sees it. Installed once.
if (!globalThis.__diffpackFetchPatched && typeof globalThis.fetch === "function") {{
  globalThis.__diffpackFetchPatched = true;
  const __diffpackOrigFetch = globalThis.fetch;
  globalThis.fetch = function (input, init) {{
    const next = init && init.next;
    if (next && typeof next === "object") {{
      if (Array.isArray(next.tags)) {{
        const store = requestAls.getStore();
        const scope = cacheScopeAls.getStore();
        for (const tag of next.tags) {{
          if (typeof tag === "string" && tag) {{
            if (store && store.tags) store.tags.add(tag);
            if (scope) scope.tags.add(tag);
          }}
        }}
      }}
      if (typeof next.revalidate === "number") {{
        const scope = cacheScopeAls.getStore();
        if (scope) {{
          scope.revalidate =
            scope.revalidate == null ? next.revalidate : Math.min(scope.revalidate, next.revalidate);
        }}
      }}
    }}
    if (init && typeof init === "object" && "next" in init) {{
      const rest = {{}};
      for (const key of Object.keys(init)) if (key !== "next") rest[key] = init[key];
      return __diffpackOrigFetch.call(this, input, rest);
    }}
    return __diffpackOrigFetch.call(this, input, init);
  }};
}}
"#,
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

/// The react-server render/action entry (Target::IsolatedServer). Builds the app's
/// ROUTE TABLE (every static route + its nested layout chain + metadata), matches a
/// requested pathname, composes `<Layout0>…<LayoutN>[head, <Page/>]` for the matched
/// route, and renders it to a flight stream (`render <pathname>` op), or dispatches a
/// server action (`action` op). The orchestrator spawns this in its own child so its
/// react-server React never mixes with the SSR/browser React.
#[allow(clippy::too_many_arguments)]
fn rsc_entry_module(
    disc: &Discovered,
    unbuilt: &UnbuiltPatterns,
    fonts: &crate::next_font::FontOutput,
    error_boundary: &Path,
    segment_boundary: &Path,
    control_boundary: &Path,
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
    let control_boundary_id = format!("M{}", intern(&mut modules, control_boundary));

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
        let title = route
            .metadata
            .title
            .as_deref()
            .map(js_str)
            .unwrap_or_else(|| "null".to_string());
        let description = route
            .metadata
            .description
            .as_deref()
            .map(js_str)
            .unwrap_or_else(|| "null".to_string());
        if route.kind == RouteKind::Ssg {
            let ns_id = format!("NS{}", intern_ns(&mut namespaces, &route.page));
            static_param_entries.push_str(&format!("  {}: {ns_id},\n", js_str(&route.url_path),));
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
        let hidx = handler_namespaces
            .iter()
            .position(|m| m == &key)
            .unwrap_or_else(|| {
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
            "  {{ path: {}, segments: {}, methods: {{ {methods_js} }}, edge: {} }},\n",
            js_str(&handler.url_path),
            segments_js(&handler.segments),
            handler.edge,
        ));
    }
    let handler_imports: String = handler_namespaces
        .iter()
        .enumerate()
        .map(|(i, s)| format!("import * as H{i} from {};\n", js_str(s)))
        .collect();

    // Pages-router API routes (`pages/api/**`) of a hybrid app: PATTERNS ONLY here. The
    // modules themselves are bundled into the SSR graph and invoked there (Next's
    // `api-node` layer has no `react-server` export condition — see the header of
    // src/next_runtime/pages_api.js). This entry still owns the table because it owns
    // route discovery, and `routeManifest` publishes it so the orchestrator knows which
    // paths to send to the SSR bundle instead of rendering a page for them.
    let mut pages_api_entries = String::new();
    for route in &disc.pages_api {
        pages_api_entries.push_str(&format!(
            "  {{ path: {}, segments: {} }},\n",
            js_str(&route.url_path),
            segments_js(&route.segments),
        ));
    }

    // The patterns this build did NOT compile (dev [`RouteScope`]). Pure data — no module
    // is referenced, so publishing them costs nothing — and they are what lets the
    // orchestrator tell a request for an uncompiled route from a request for a route that
    // does not exist.
    let unbuilt_entries = |patterns: &[UnbuiltPattern]| -> String {
        patterns
            .iter()
            .map(|pattern| {
                format!(
                    "  {{ path: {}, segments: {} }},\n",
                    js_str(&pattern.url_path),
                    segments_js(&pattern.segments),
                )
            })
            .collect()
    };
    let unbuilt_route_entries = unbuilt_entries(&unbuilt.routes);
    let unbuilt_handler_entries = unbuilt_entries(&unbuilt.handlers);
    let unbuilt_pages_api_entries = unbuilt_entries(&unbuilt.pages_api);

    // Middleware: namespace-import it (named `middleware` or default export) so
    // `runMiddleware` can invoke it; `null` when the app has none.
    let (middleware_import, middleware_const) = match middleware {
        Some(path) => (
            format!(
                "import * as __mw from {};\n",
                js_str(&path.to_string_lossy())
            ),
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
    let root_title = disc
        .root_metadata
        .title
        .as_deref()
        .map(js_str)
        .unwrap_or_else(|| "null".to_string());
    let root_description = disc
        .root_metadata
        .description
        .as_deref()
        .map(js_str)
        .unwrap_or_else(|| "null".to_string());
    // The metadata chain of the real-404 document: the root layout, then `not-found.tsx`'s
    // own `metadata`/`generateMetadata`. Next resolves this for a 404 exactly like any
    // other document — cal.com's not-found sets the `404: This page could not be found.`
    // title and `robots: noindex` there — so the 404 must carry a <head>, not just a body.
    let not_found_meta_chain = meta_ns(&mut namespaces, &disc.root_layout);
    let not_found_page_meta = meta_ns(&mut namespaces, &disc.app_not_found);

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
    //
    // The app stylesheet <link> is derived from what the build EMITTED, not from a
    // scan of the sources. `server.css` is written next to this entry exactly when the
    // react-server module graph compiled any CSS, and `main.rs` preserves exactly that
    // file to the served `public/rsc.css` — so one existence check decides both, and a
    // <link> whose href 404s is not expressible. A source scan cannot do this: it sees
    // CSS that the graph never compiles (a `"use client"` file no route imports; a
    // string that merely ends in `.css`) and links a stylesheet that was never emitted.
    let css_const = format!(
        "// Linked only when the react-server build emitted a stylesheet beside this entry.\n\
         const RSC_CSS_HREF = existsSync(new URL(\"./{RSC_EMITTED_CSS_FILE}\", import.meta.url)) ? {} : null;\n\
         // The CLIENT graph's stylesheet (see CLIENT_CSS_URL): CSS a `\"use client\"`\n\
         // module reaches only THROUGH another module never enters the react-server\n\
         // graph, so it exists nowhere in RSC_CSS_HREF. Linked last, like Next's\n\
         // client-component CSS chunk. Same emitted-artifact guard, in the served\n\
         // `public/` beside this server dir because it belongs to the other graph.\n\
         const CLIENT_CSS_HREF = existsSync(new URL({}, import.meta.url)) ? {} : null;\n",
        js_str(&format!("{asset_base}{RSC_CSS_URL}")),
        js_str(CLIENT_EMITTED_CSS_PATH),
        js_str(&format!("{asset_base}{CLIENT_CSS_URL}")),
    );
    let css_push = "  if (RSC_CSS_HREF) items.push(createElement(\"link\", { rel: \"stylesheet\", href: RSC_CSS_HREF, precedence: \"low\" }));\n".to_string();
    // Pushed AFTER the font block so React's precedence groups are created in the order
    // `low` (app) -> `high` (fonts) -> `client`, putting the client graph's CSS last in
    // <head> — the order the reference build produces.
    let client_css_push = "  if (CLIENT_CSS_HREF) items.push(createElement(\"link\", { rel: \"stylesheet\", href: CLIENT_CSS_HREF, precedence: \"client\" }));\n".to_string();
    let (font_const, mut font_push) = if fonts.css.trim().is_empty() {
        (String::new(), String::new())
    } else {
        (
            format!("const FONT_CSS = {};\n", js_str(&fonts.css)),
            "  items.push(createElement(\"style\", { href: \"diffpack-next-font\", precedence: \"high\", dangerouslySetInnerHTML: { __html: FONT_CSS } }));\n".to_string(),
        )
    };
    // `next/font/local` with `preload: true` (the default) gets the same
    // `<link rel="preload" as="font" crossorigin>` Next emits, so the face is requested
    // alongside the document instead of after the CSS parses. Google faces are reached
    // through an `@import`, which cannot be preloaded this way, so they get none.
    for href in &fonts.preloads {
        font_push.push_str(&format!(
            "  items.push(createElement(\"link\", {{ rel: \"preload\", href: {}, as: \"font\", type: {}, crossOrigin: \"\" }}));\n",
            js_str(href),
            js_str(font_mime(href)),
        ));
    }
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
// entry (Target::IsolatedServer, bundled under the `react-server` export condition).
// It holds the app's ROUTE TABLE (each route's parsed segment pattern + its root→leaf
// level chain of layouts + loading/error boundaries), MATCHES a requested pathname
// (dynamic `[param]`/`[...catchAll]` segments captured into `params`), composes the
// matched route (boundaries inner→outer, layouts root-last), and renders it to a
// flight stream — or renders the real 404 tree for an unmatched path (status carried
// to the orchestrator over fd 3), or dispatches a server action (`action` op).
import {{ renderToReadableStream }} from "react-server-dom-webpack/server";
import {{ createElement, Fragment, Suspense }} from "react";
import {{ readFileSync, writeSync, statSync, existsSync }} from "node:fs";
import {{ fileURLToPath, pathToFileURL }} from "node:url";
import {{ handleServerAction }} from "#diffpack-rsc-action-handler";
import {{ NextRequest }} from "next/server";
{request_context_import}{imports}{ns_imports}{handler_imports}{middleware_import}
{middleware_const}
{css_const}{font_const}const ROUTES = [
{route_entries}];
// Intercepting routes: a soft-nav to a matching target renders the overlay page.
const INTERCEPTS = [
{intercept_entries}];
// Route handlers (`route.ts` HTTP endpoints): each entry's `methods` maps an HTTP
// method to its exported handler function; `handleRoute` matches a request path here.
const ROUTE_HANDLERS = [
{handler_entries}];
// Pages-router API routes (`pages/api/**`) of a HYBRID app — `app/` renders the pages,
// these serve the HTTP endpoints under the pages-router `(req, res)` contract. PATTERNS
// ONLY: the modules live in the SSR graph (Next's `api-node` layer has no `react-server`
// condition) and run there. Published through `routeManifest` so the orchestrator can
// match them; matched only after ROUTE_HANDLERS, so an `app/**/route.ts` wins a path
// both could answer (Next's precedence).
const PAGES_API = [
{pages_api_entries}];
// Routes/handlers/endpoints this build did NOT compile, because the dev server's route
// scope left them out (see RouteScope). PATTERNS ONLY, referencing no module, so the
// tables exist without pulling a single uncompiled route into this graph. The
// orchestrator matches them AFTER the compiled tables above and asks the dev server to
// widen its scope, which is what separates "not built yet" from a real 404. Always empty
// in a production build.
const UNBUILT_ROUTES = [
{unbuilt_route_entries}];
const UNBUILT_HANDLERS = [
{unbuilt_handler_entries}];
const UNBUILT_PAGES_API = [
{unbuilt_pages_api_entries}];
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
// The client CONTROL boundary wrapped around the matched page: it turns a redirect()
// digest that reaches the client (thrown after the shell flushed, so no 307 was possible)
// back into a real navigation.
const CONTROL_BOUNDARY = {control_boundary_id};
// The app-root global-error boundary (owns <html>), or null when the app has none.
const GLOBAL_ERROR = {global_error_id};
const ROOT_META = {{ title: {root_title}, description: {root_description} }};
// The pseudo-route the real-404 document resolves its <head> from: the root layout's
// metadata, then `not-found.tsx`'s own. Same shape `resolveMetadata` walks for a matched
// route, so the 404 gets the identical title-template / robots / openGraph treatment.
const NOT_FOUND_ROUTE = {{ path: "/_not-found", metaChain: [{not_found_meta_chain}], pageMeta: {not_found_page_meta}, title: ROOT_META.title, description: ROOT_META.description }};

// The route's head elements (stylesheet + font + this route's metadata). React 19
// hoists these into <head> from anywhere in the tree.
function headItems(meta) {{
  const items = [];
{css_push}{font_push}{client_css_push}{meta_image_push}  if (meta && meta.title) items.push(createElement("title", null, meta.title));
  if (meta && meta.description) items.push(createElement("meta", {{ name: "description", content: meta.description }}));
  return items;
}}

// --- Metadata API ------------------------------------------------------------------
// Resolve + merge metadata from the root layout down to the page (each may export a
// `metadata` object OR an async `generateMetadata`; likewise `viewport`/
// `generateViewport`), applying ancestor title templates, then render the <head> tags
// (React 19 hoists them into <head>). Runs at flight-render time so dynamic/async
// metadata works — no per-request cost beyond the render already happening.
async function resolveMetadata(route, params, searchParams) {{
  const chain = [...(route.metaChain || []), route.pageMeta];
  const paramsP = Promise.resolve(params);
  const searchP = Promise.resolve(searchParams || {{}});
  const meta = {{}};
  let template = null; // an ancestor title.template applies to descendant string titles
  for (const ns of chain) {{
    if (!ns) continue;
    let m = null;
    if (typeof ns.generateMetadata === "function") {{
      m = await ns.generateMetadata({{ params: paramsP, searchParams: searchP }}, Promise.resolve(meta));
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
async function MetadataHead({{ route, params, searchParams }}) {{
  const meta = await resolveMetadata(route, params, searchParams);
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
  // The 404 is a real document: it resolves the metadata chain (root layout +
  // not-found.tsx) exactly like a matched route, so its <title>/robots/openGraph are the
  // ones the app declares. `headItems({{}})` carries the stylesheet/font/icon links only —
  // the title comes from the resolved chain, so it is never emitted twice.
  let node = createElement(
    Fragment,
    null,
    ...headItems({{}}),
    createElement(Suspense, {{ fallback: null }}, createElement(MetadataHead, {{ route: NOT_FOUND_ROUTE, params: {{}} }})),
    body,
  );
  if (ROOT_LAYOUT) node = createElement(ROOT_LAYOUT, {{ params: Promise.resolve({{}}) }}, node);
  return node;
}}

// Wrap a page in its level chain (leaf→root: loading Suspense, error boundary,
// template, layout), sharing `params`. Used for a matched @slot route. `remountKey`
// (the pathname) keys each `template.tsx` so React remounts it on navigation (fresh
// state per URL) while same-position layouts keep their state — matching Next's
// Layout > Template > ErrorBoundary > Suspense(loading) > children order.
function composeLevels(page, levels, params, remountKey, searchParams) {{
  const paramsPromise = Promise.resolve(params);
  // Same CONTROL boundary the main route composition installs, so a slot/intercept page
  // that redirect()s after its shell flushed navigates instead of rendering nothing.
  let node = createElement(
    CONTROL_BOUNDARY,
    null,
    createElement(page, {{ params: paramsPromise, searchParams: Promise.resolve(searchParams || {{}}) }}),
  );
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
// Resolve the matched route's own page component BEFORE the flight render starts, so a
// `redirect()` / `notFound()` it throws is known while the response still has no bytes.
//
// Next answers a top-level redirect with a real 307 because its shell — everything not
// inside a user `<Suspense>` — is complete before any byte leaves. Diffpack streams, and
// React's flight writer emits its first chunk as soon as the ROOT row is serializable:
// an async page is a pending row in that first chunk, so a redirect thrown while
// awaiting it landed AFTER the headers were already gone. A logged-in cal.com `/`
// redirects to `/event-types` from exactly there — the browser got a 200 with a
// half-rendered document instead of the redirect.
//
// Awaiting the page here does not weaken anything below it: what the page RETURNS
// (including its own `<Suspense>` boundaries and slow children) still streams normally.
// A page that throws an ordinary error is re-thrown INSIDE the tree so its segment's
// `error.tsx` boundary handles it exactly as before; the throw is only inspected here,
// never swallowed.
async function resolvePage(Page, props, control) {{
  // A client-reference page ("use client") is not callable here — render it as an element.
  if (typeof Page !== "function" || Page.$$typeof) return createElement(Page, props);
  let out;
  try {{
    out = Page(props);
    if (out && typeof out.then === "function") out = await out;
  }} catch (error) {{
    if (isControlThrow(error)) {{
      // redirect() / notFound() / a prerender bailout: record it NOW — that is the whole
      // point of resolving here — and re-throw inside the tree so React's boundaries and
      // the flight stream see the identical failure they saw before.
      flightControlOnError(control, error);
      return createElement(function DiffpackPageThrow() {{ throw error; }});
    }}
    // Any OTHER throw is handed back to React unrendered. A component that threw is
    // re-invoked by React anyway, and this keeps a page that legitimately cannot run
    // outside the renderer — a synchronous Server Component calling `use()`, whose
    // dispatcher only exists inside a render — on exactly the path it had before.
    return createElement(Page, props);
  }}
  return out;
}}

// Whether a throw is app-router CONTROL FLOW (redirect / notFound / a build-time
// request-state bailout) rather than a render failure. Keyed on the same digests
// `flightControlOnError` classifies, so the two can never disagree.
function isControlThrow(error) {{
  const digest = (error && error.digest) || "";
  return (
    digest.startsWith("NEXT_REDIRECT;") ||
    digest === "NEXT_HTTP_ERROR_FALLBACK;404" ||
    digest === "DIFFPACK_DYNAMIC_BAILOUT"
  );
}}

async function documentTree(pathname, opts, control) {{
  // The real-404 document, requested EXPLICITLY by the orchestrator (`reqCtx.notFound`)
  // after a render signalled notFound(). The requested pathname is kept as-is so
  // usePathname()/headers() still report the URL the visitor asked for.
  //
  // This must never be selected by a magic pathname: an app with a catch-all route
  // (cal.com's `app/[user]/page.tsx`) matches ANY sentinel too, so the "not-found"
  // render would render that catch-all page — which threw notFound() again and left the
  // 404 document with an errored, empty body.
  if (opts && opts.notFound) return {{ tree: notFoundTree(), status: 404, params: {{}} }};
  if (opts && opts.softNav) {{
    const hit = matchIntercept(pathname);
    if (hit) {{
      return {{
        tree: composeLevels(hit.intercept.page, hit.intercept.levels, hit.params, pathname, (opts && opts.searchParams) || {{}}),
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
  // The page is resolved BEFORE the flight render, so a top-level redirect()/notFound()
  // reaches `control` while the response still has no bytes.
  //
  // ONLY when the page is part of Next's SHELL. Next's shell is everything not inside a
  // `<Suspense>`, and a `loading.tsx` on any level of this route puts the page inside
  // one: Next flushes the loading fallback as the shell and a redirect()/notFound() the
  // page throws afterwards can no longer change the HTTP status — Next answers 200 and
  // the streamed flight makes the client Router navigate. Awaiting the page here in that
  // case would manufacture a redirect Next does not send (cal.com `/event-types` is
  // exactly this: `loading.tsx` present, Next 200, diffpack was answering 307).
  const searchParams = (opts && opts.searchParams) || {{}};
  const pageProps = {{ params: paramsPromise, searchParams: Promise.resolve(searchParams) }};
  const pageInShell = !route.levels.some((level) => level.loading);
  let node = pageInShell
    ? await resolvePage(route.page, pageProps, control || {{}})
    : createElement(route.page, pageProps);
  // The CONTROL boundary sits directly around the page (inside every layout, loading and
  // error boundary): a redirect() the page throws once the shell has flushed arrives on
  // the client as an errored flight row, and this is what turns it back into a real
  // navigation. Placed here — not around the Suspense — so the layouts stay mounted,
  // which is exactly where Next puts its RedirectBoundary.
  node = createElement(CONTROL_BOUNDARY, null, node);
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
      node = createElement(Fragment, null, ...headItems({{}}), createElement(Suspense, {{ fallback: null }}, createElement(MetadataHead, {{ route, params, searchParams }})), node);
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
    // One CONTROL boundary per level, OUTSIDE that level's layout. React's flight writer
    // errors the whole ROW a throw happened in, and an async layout owns its subtree's
    // row — so a boundary nested INSIDE that layout is destroyed together with it and
    // never renders. Only a boundary above the errored row survives to act, and putting
    // one at every level means the nearest surviving layouts stay mounted.
    node = createElement(CONTROL_BOUNDARY, null, node);
  }}
  if (!headInjected) node = createElement(Fragment, null, ...headItems({{}}), createElement(Suspense, {{ fallback: null }}, createElement(MetadataHead, {{ route, params, searchParams }})), node);
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
  }} else if (digest === "DIFFPACK_DYNAMIC_BAILOUT") {{
    // A request-state read during a build-time prerender. This is CONTROL FLOW, not a
    // render failure: the route is demoted to per-request rendering (Next's contract).
    // Keep the FIRST one — it is the read that actually decided the route's fate.
    if (!control.dynamicBailout) control.dynamicBailout = String((error && error.message) || digest);
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
// The route params for `pathname`, resolved WITHOUT composing the tree. The request
// store has to exist before `documentTree` runs — composition now calls the page (see
// `resolvePage`), and the page reads request state through the store — so the params it
// carries are looked up here first.
// The render options carried by the per-request context. ONE definition, so
// `matchParams` and `documentTree` can never be handed a different view of the same
// request (a not-found document that still resolved a catch-all route's params is
// exactly the kind of drift this prevents).
function renderOpts(reqCtx) {{
  return {{ softNav: !!reqCtx.softNav, notFound: !!reqCtx.notFound, searchParams: requestSearchParams(reqCtx) }};
}}

// The `searchParams` prop Next hands a PAGE Server Component (and generateMetadata):
// the request's query as `{{ key: string | string[] }}`, repeated keys as arrays. Layouts
// and templates get no searchParams, which is Next's contract, not an omission.
//
// `__rsc` is stripped: it is diffpack's own soft-navigation marker on the flight channel,
// never part of the app's query, and a page must not see the same URL differently
// depending on whether it was reached by a hard load or a client navigation.
function requestSearchParams(reqCtx) {{
  const url = new URL((reqCtx && reqCtx.url) || "http://localhost/", "http://localhost");
  const search = url.searchParams;
  search.delete("__rsc");
  const out = {{}};
  for (const key of new Set(search.keys())) {{
    const all = search.getAll(key);
    out[key] = all.length > 1 ? all : all[0];
  }}
  return out;
}}

function matchParams(pathname, opts) {{
  // The not-found document is not a route match: it has no params, even though a
  // catch-all route would happily match the pathname that produced the 404.
  if (opts && opts.notFound) return {{}};
  if (opts && opts.softNav) {{
    const hit = matchIntercept(pathname);
    if (hit) return hit.params;
  }}
  const m = matchRoute(pathname);
  return m ? m.params : {{}};
}}

function renderStore(pathname, reqCtx, params) {{
  return {{
    url: new URL(reqCtx.url || ("http://localhost" + pathname), "http://localhost"),
    headers: new Headers(reqCtx.headers || []),
    cookieHeader: reqCtx.cookie || "",
    params,
    // A BUILD-TIME prerender has no request at all. Reading request state under it is not
    // "read an empty header" — the value does not exist yet, so any answer would be a
    // fabrication that bakes into a static file. Next's contract is to DEMOTE the route to
    // per-request rendering instead; `cookies()`/`headers()`/`draftMode()` therefore throw
    // the DIFFPACK_DYNAMIC_BAILOUT digest here and the prerenderer records the route as
    // Dynamic. Deliberately false for `dynamic = "force-static"`, where Next's documented
    // behaviour IS to hand back empty values.
    prerender: !!reqCtx.prerender,
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
  const control = {{}};
  // `documentTree` awaits the matched page (see `resolvePage`), and the page reads
  // request state through the ALS store — so the store has to be established around the
  // composition, not just around the flight render.
  const store = renderStore(pathname, reqCtx, matchParams(pathname, renderOpts(reqCtx)));
  const {{ tree, status, params, intercept }} = await requestAls.run(store, () =>
    documentTree(pathname, renderOpts(reqCtx), control),
  );
  store.params = params;
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
    // Set only under a prerender (`reqCtx.prerender`): the route read request state, so it
    // is not statically prerenderable and the caller must record it Dynamic.
    dynamicBailout: control.dynamicBailout,
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
  const control = {{}};
  // See `renderRequest`: composing the tree now RUNS the page, which reads request state.
  const store = renderStore(pathname, reqCtx, matchParams(pathname, renderOpts(reqCtx)));
  const {{ tree, status, params, intercept }} = await requestAls.run(store, () =>
    documentTree(pathname, renderOpts(reqCtx), control),
  );
  store.params = params;
  await requestAls.run(store, async () => {{
    const stream = renderToReadableStream(tree, bundlerConfig, {{
      onError(error) {{ return flightControlOnError(control, error); }},
    }});
    const reader = stream.getReader();
    let metaSent = false;
    // What the meta ALREADY told the orchestrator. A redirect/notFound present here was
    // acted on (real 307/404); only one that appears afterwards is "too late", and the
    // orchestrator must distinguish the two or it warns about redirects it did honour.
    let metaControl = {{ redirect: undefined, notFound: undefined }};
    const sendMeta = () => {{
      metaSent = true;
      metaControl = {{ redirect: control.redirect, notFound: control.notFound }};
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
    // `lateControl`: a redirect()/notFound() that appeared only AFTER the meta went out —
    // i.e. thrown behind a Suspense boundary, once the shell had already flushed. That one
    // cannot change the response and the orchestrator reports it. A redirect/notFound the
    // meta already carried was acted on and must NOT be reported.
    const lateControl = !!((control.redirect && !metaControl.redirect) || (control.notFound && !metaControl.notFound));
    sink.end({{ status: control.status || status || 200, redirect: control.redirect, notFound: control.notFound, metaSent, lateControl, tags: [...store.tags], setCookies: store.responseCookies.slice() }});
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

// Enter diffpack's lean WinterCG (edge) context for an `runtime = "edge"` route/handler
// or for middleware (always edge in Next). Node already provides the full WinterCG global
// surface (fetch/Request/Response/URL/crypto/TextEncoder/…), so there is nothing to
// polyfill — we (1) advertise the edge marker `globalThis.EdgeRuntime` that edge code and
// libraries probe, and (2) assert the required globals are present, failing loudly if a
// host somehow lacks one. Node built-in imports are already rejected at BUILD time
// (`validate_edge_module`), so a served edge route never reaches a Node-only API. The
// marker is set idempotently and process-wide (one-process server) — documented.
function ensureEdgeContext() {{
  if (typeof globalThis.EdgeRuntime === "undefined") {{
    globalThis.EdgeRuntime = "diffpack-edge";
  }}
  for (const g of ["fetch", "Request", "Response", "URL", "crypto", "TextEncoder", "TextDecoder"]) {{
    if (typeof globalThis[g] === "undefined") {{
      throw new Error(`diffpack edge runtime: required WinterCG global \`${{g}}\` is missing from this host`);
    }}
  }}
}}

// Dispatch a ROUTE HANDLER (`route.ts` HTTP endpoint). Matches `pathname` against the
// ROUTE_HANDLERS table, invokes the exported method function with a real `NextRequest` and
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
    // Edge (WinterCG) handler: advertise the edge globals before invoking it.
    if (entry.edge) ensureEdgeContext();
    const url = reqCtx.url || ("http://localhost" + pathname);
    const bodyBytes =
      method === "GET" || method === "HEAD" || reqCtx.body == null
        ? undefined
        : reqCtx.bodyIsBase64
          ? Buffer.from(reqCtx.body, "base64")
          : reqCtx.body;
    // A `NextRequest`, not a bare `Request`: Next hands route handlers the former, and
    // reading the query off `request.nextUrl.searchParams` is the ordinary way to do it
    // (cal.com does in a dozen handlers). With a plain `Request` that is a read of
    // `undefined.searchParams` — the handler 500s on its first line.
    const request = new NextRequest(url, {{
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
  // No `app/**/route.ts` matched. A hybrid app's `pages/api/**` endpoints answer next —
  // but NOT from this graph: the orchestrator dispatches those to the SSR bundle's
  // `handlePagesApi` (Next's `api-node` layer, no `react-server` condition). Returning
  // null here is what makes it fall through to that path.
  return null;
}}

// The route-handler routes (segment patterns + methods) + whether the app has
// middleware — queried once at boot by the orchestrator so it can match locally and
// dispatch handler/middleware without a per-page-request round-trip. `pagesApi` carries
// the hybrid app's `pages/api/**` patterns so the orchestrator routes those requests to
// `handleRoute` too instead of rendering a page for them.
export function routeManifest() {{
  return {{
    handlers: ROUTE_HANDLERS.map((entry) => ({{ segments: entry.segments, methods: Object.keys(entry.methods) }})),
    pagesApi: PAGES_API.map((entry) => ({{ segments: entry.segments }})),
    hasMiddleware: MIDDLEWARE != null,
    // What this build left uncompiled, so the orchestrator can match a request against a
    // route that exists but is not built yet and ask for it instead of 404ing.
    unbuilt: {{
      routes: UNBUILT_ROUTES,
      handlers: UNBUILT_HANDLERS,
      pagesApi: UNBUILT_PAGES_API,
    }},
  }};
}}

// Run the app's middleware (if any) for a request. Returns the middleware's Response
// serialized as `{{ status, headers, body(base64) }}` — the orchestrator reads Next's
// `x-middleware-*` protocol headers on it (next / redirect / rewrite) — or `null` when
// there is no middleware.
export async function runMiddleware(reqCtx) {{
  if (MIDDLEWARE == null) return null;
  // Next middleware always runs on the edge runtime.
  ensureEdgeContext();
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
// newline-delimited JSON on stdin/stdout — the SAME process isolation the
// per-request child had, minus the per-request spawn. Requests carry the flight
// back base64-encoded (one JSON line per response).
//
// Fresh code after a `diffpack dev` edit arrives through the `invalidate` op, NOT by
// re-importing this file. Re-importing the entry (with a fresh `?v=<mtime>`) was the
// old mechanism and it is unfixable: the entry reaches its split chunks through
// `import("./server.chunk-N.mjs")`, a URL with no query, so Node's ESM cache hands
// the fresh entry the ALREADY-EVALUATED chunks. On any app whose graph is split that
// left the fresh runtime's id table disagreeing with the stale chunks'
// registrations, and the worker died with `Module is not loaded: <id>` on the first
// render after a server-component edit. Only a URL Node has never seen re-evaluates,
// which is what the dev server's per-edit micro-chunk is.
async function serveLoop() {{
  // This module instance IS the bundle as loaded; its own exports are the render
  // functions until a hot update replaces them.
  let cached = {{ renderRequest, renderRequestStream, runAction, handleRoute, routeManifest, runMiddleware }};
  // Serializes hot updates against renders: a render started after an `invalidate`
  // line arrived must see the new code, and a hot update must not interleave with a
  // render already resolving its modules.
  let applying = Promise.resolve();
  async function applyHotUpdate(req) {{
    if (!req.chunk) throw new Error("rsc-entry serve: an invalidate op needs a `chunk` path");
    // Importing the micro-chunk REGISTERS the changed modules' fresh factories into
    // the live runtime; `serverInvalidate` then drops the cache for them and every
    // importer up to the entry, re-runs exactly that path, and republishes the
    // entry's exports. React and every untouched dependency stay cached, so the
    // react-server React singleton survives.
    await import(pathToFileURL(req.chunk).href);
    const runtime = globalThis.__diffpack_hmr_runtime;
    if (!runtime || typeof runtime.serverInvalidate !== "function") {{
      throw new Error("rsc-entry serve: this bundle exposes no __diffpack_hmr_runtime.serverInvalidate; it was not emitted with HMR enabled, so a hot update cannot be applied");
    }}
    const invalidated = await runtime.serverInvalidate(req.ids || [], []);
    const fresh = globalThis.__diffpack_ssr_entry;
    if (!fresh) {{
      throw new Error("rsc-entry serve: the runtime did not republish globalThis.__diffpack_ssr_entry after serverInvalidate; the entry re-run failed");
    }}
    if (typeof fresh.renderRequest !== "function" || typeof fresh.runAction !== "function") {{
      throw new Error("rsc-entry serve: the hot-updated bundle does not export renderRequest/runAction");
    }}
    cached = {{
      renderRequest: fresh.renderRequest,
      renderRequestStream: fresh.renderRequestStream,
      runAction: fresh.runAction,
      handleRoute: fresh.handleRoute,
      routeManifest: fresh.routeManifest,
      runMiddleware: fresh.runMiddleware,
    }};
    return invalidated;
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
        if (req.op === "invalidate") {{
          // Chained so concurrent hot updates apply in arrival order; the render ops
          // below await the same chain, so they always observe the newest code.
          const settled = applying.then(() => applyHotUpdate(req), () => applyHotUpdate(req));
          applying = settled.then(() => {{}}, () => {{}});
          reply({{ id: req.id, invalidated: await settled }});
          continue;
        }}
        await applying;
        const mod = cached;
        // A re-emit can change the manifest too; always re-read on the worker path.
        manifestCache.delete(req.manifestPath);
        if (req.op === "render") {{
          const r = await mod.renderRequest(req.pathname || "/", manifest(req.manifestPath), req.reqCtx || {{}});
          reply({{ id: req.id, flight: Buffer.from(r.flight).toString("base64"), status: r.status, params: r.params, redirect: r.redirect, notFound: r.notFound, dynamicBailout: r.dynamicBailout, tags: r.tags || [], setCookies: r.setCookies || [] }});
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
    writeMeta({{ status: r.status, params: r.params, redirect: r.redirect, notFound: r.notFound, dynamicBailout: r.dynamicBailout, tags: r.tags || [] }});
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

// `main()` runs ONCE per process, ever. This module's body is not only executed by the
// original import: a dev hot update re-runs the entry factory in place
// (`serverInvalidate` → `__hmrRerun(__entryId)`) to pick up an edited Server Component,
// and a second `serveLoop()` would install a SECOND `process.stdin` reader — every
// request line then handled twice, so the worker renders the page twice and writes both
// flights under one reply id. That is not a crash: the orchestrator concatenates them
// and the SSR flight client dies decoding a duplicated row
// (`chunk.reason.enqueueModel is not a function`), with a flight exactly 2x its correct
// size. Keyed by this module's own URL so a process that legitimately hosts two
// different entries still starts each one. Strip search/hash first: importing the SAME
// entry through a cache-busting `server.mjs?v=...` URL must not turn it into a second
// worker. That was the old dev refresh mechanism, and stale orchestrators / pending
// refreshes can still have such an import in flight while the worker is coming up.
const __diffpackStarted = (globalThis.__diffpack_rsc_entry_started ??= new Set());
const __diffpackEntryUrl = new URL(import.meta.url);
__diffpackEntryUrl.search = "";
__diffpackEntryUrl.hash = "";
const __diffpackEntryKey = __diffpackEntryUrl.href;
if (!__diffpackStarted.has(__diffpackEntryKey)) {{
  __diffpackStarted.add(__diffpackEntryKey);
  main().catch((error) => {{
    console.error(error && error.stack ? error.stack : String(error));
    process.exit(1);
  }});
}}
"##,
    )
}

/// How a graph records its reachability pin to a lazy island.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PinKind {
    /// `require(path)` inside a never-called closure: a STATIC edge, so the island's
    /// factory lands in this graph's main chunk. Right for the SSR-of-flight bundle,
    /// which is one Node file the seam requires synchronously with
    /// `__webpack_chunk_load__` stubbed to `Promise.resolve()`.
    StaticRequire,
    /// `import(path)` inside a never-called closure: a DYNAMIC root, so the chunk
    /// planner gives the island its own chunk and the client-references manifest
    /// carries that chunk for `__webpack_chunk_load__`.
    ///
    /// This is what the browser entry wants and what the reference does. With static
    /// requires every island in the app was in `client.js`, so cal.com's login page
    /// downloaded the app store, the booker, every settings screen and all the payment
    /// components: 17.8 MB of JS on a page that needs a fraction of it, re-parsed on
    /// every full navigation.
    DynamicChunk,
}

/// Imports every `"use client"` island so the graph bundles + registers it under a
/// runtime id (pinned to a global so DCE keeps it).
fn island_pins(
    adapter_dir: &Path,
    islands: &[PathBuf],
    eager: &BTreeSet<String>,
    kind: PinKind,
) -> String {
    let _ = adapter_dir;
    // Reachability pins. Each island must be bundled and REGISTERED in this graph
    // so the RSC seam (`runtime.require(id)`) can resolve it while consuming a
    // flight — but nothing needs it EVALUATED before a render actually reaches
    // it. A `require` inside a never-called closure records the graph edge at
    // transform time without running the module at boot; evaluating every island
    // eagerly (469 on cal.com) was ~100% of the SSR bundle's boot time (425 ms
    // -> 9 ms measured). Lazy is also what the reference does: webpack's flight
    // consumer calls `requireModule` per client reference, on demand, in both
    // its SSR and browser runtimes.
    //
    // The exception is `eager`: islands whose evaluation is ASYNC (top-level
    // await, directly or transitively). The seam's require is synchronous, so an
    // async island must be evaluated-and-settled before any flight consumes it —
    // exactly what a static `import * as` guarantees (and the build's
    // async-closure guard enforces, by rejecting a bare `require` edge to an
    // async module). The set is recorded by [`reconcile_async_islands`] from the
    // discovered graph, so it is a build fact, not a guess; when it drifts, the
    // build regenerates this entry and rediscovers once.
    let mut out = String::new();
    let mut lazy: Vec<&PathBuf> = Vec::new();
    for (index, island) in islands.iter().enumerate() {
        let canonical = island.canonicalize().unwrap_or_else(|_| island.clone());
        if eager.contains(canonical.to_string_lossy().as_ref()) {
            out.push_str(&format!(
                "import * as __island{index} from {};\n(globalThis).__diffpack_next_island_{index} = __island{index};\n",
                js_str(&island.to_string_lossy()),
            ));
        } else {
            lazy.push(island);
        }
    }
    // The globalThis assignment keeps the thunk array (and so the recorded
    // edges) alive under export shaking.
    out.push_str("const __diffpackIslandPins = [\n");
    for island in lazy {
        let specifier = js_str(&island.to_string_lossy());
        match kind {
            PinKind::StaticRequire => out.push_str(&format!("  () => require({specifier}),\n")),
            PinKind::DynamicChunk => out.push_str(&format!("  () => import({specifier}),\n")),
        }
    }
    out.push_str("];\n(globalThis).__diffpack_next_island_pins = __diffpackIslandPins;\n");
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
    eager_islands: &BTreeSet<String>,
    hooks_context: &Path,
    asset_base: &str,
    pages_api: &[PagesApiRoute],
) -> String {
    let pins = island_pins(adapter_dir, islands, eager_islands, PinKind::StaticRequire);
    let lazy = js_str(&adapter_dir.join("lazy.js").to_string_lossy());
    // The base the browser fetches chunks from — the same one `client.js` is served under.
    let asset_base_json = js_str(asset_base);
    let hooks_import = js_str(&hooks_context.to_string_lossy());
    // The browser fetches the client bootstrap under the app's basePath/assetPrefix (the
    // orchestrator strips that prefix back off before the publicDir lookup).
    let client_js = js_str(&format!("{asset_base}/client.js"));
    // The streaming destination is real source (src/next_runtime/flight_sink.js) spliced
    // in verbatim so the node regression test can import the SAME code this entry runs.
    // It carries this module's `node:stream` import.
    let flight_sink = include_str!("next_runtime/flight_sink.js");
    // Pages-router API routes (`pages/api/**`) of a hybrid app live in THIS graph, not
    // the react-server one: Next compiles them in its `api-node` layer, without the
    // `react-server` export condition (see the header of src/next_runtime/pages_api.js
    // for the concrete failure that layering them wrong produces). Each is reached
    // through its own `import()` so it lands in its own chunk and a route only costs
    // module-init time when a request actually reaches it (cal.com has 45 of them
    // behind next-auth and tRPC).
    let pages_api_runtime = include_str!("next_runtime/pages_api.js");
    let mut pages_api_entries = String::new();
    for route in pages_api {
        pages_api_entries.push_str(&format!(
            "  {{ path: {}, segments: {}, load: () => import({}) }},\n",
            js_str(&route.url_path),
            segments_js(&route.segments),
            js_str(&route.file.to_string_lossy()),
        ));
    }
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
import {{ createElement, Fragment }} from "react";
import {{ PathParamsContext, PathnameContext, SearchParamsContext, ServerInsertedHTMLContext }} from {hooks_import};
{flight_sink}
// --- pages-router api runtime (src/next_runtime/pages_api.js, verbatim) -------------
{pages_api_runtime}
// --- end pages-router api runtime ---------------------------------------------------
// Pages-router API routes (`pages/api/**`) of a HYBRID app — `app/` renders the pages,
// these serve the HTTP endpoints under the pages-router `(req, res)` contract. The
// orchestrator matches a request against these patterns (published by the react-server
// entry's `routeManifest`, which knows the same build-time table) and calls
// `handlePagesApi` below, IN THIS PROCESS, because this is the graph whose React is the
// ordinary one.
const PAGES_API = [
{pages_api_entries}];
// Dispatch a pages-router API request. Same result shape an app-router `route.ts`
// handler returns, so the orchestrator serves both through one path; `null` means no
// `pages/api/**` route matched.
export async function handlePagesApi(pathname, method, reqCtx) {{
  return dispatchPagesApi(PAGES_API, pathname, method, reqCtx);
}}
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
  // Next PATCHES the NODE builds of its vendored react-server-dom-webpack to read
  // `globalThis.__next_require__` instead of `__webpack_require__` (its browser build
  // still reads `__webpack_require__`). diffpack resolves the flight runtime from that
  // vendored copy when the app has none, so the SSR-of-flight graph must answer to both
  // names — same registry, one alias. Installed unconditionally: an app that installs
  // the npm package reads `__webpack_require__` and never touches this.
  g.__next_require__ = g.__webpack_require__;
}}

// A chunk id's public URL. diffpack's chunk id IS its file name, and the browser serves it
// from the app's asset base — the same base the entry's own module tag is emitted with.
function chunkUrl(chunk) {{
  const base = {asset_base_json};
  return (base ? base.replace(/\/$/, "") : "") + "/" + String(chunk).replace(/^\.?\//, "");
}}

// RECORD WHICH CLIENT REFERENCES A RENDER RESOLVED, as the chunks the browser has to
// load for them.
//
// React resolves every client reference through `moduleMap[clientId]`
// (`resolveClientReference`), so a proxy over that object is an exact record of what this
// route rendered — no flight-wire parsing and no static over-approximation. The document
// then declares the list, and the browser entry loads it BEFORE hydrating.
//
// That ordering is the correctness requirement, not an optimisation: the RSC seam's
// `__webpack_require__` is synchronous, so an island whose chunk has not arrived yet is an
// island the user can see and click but that has no event handlers attached. cal.com's own
// suite catches exactly that — a theme option clicked before its island hydrates leaves the
// form clean and its submit button disabled forever. The reference has the same property by
// a different route: `next build` emits a `<script>` per route chunk into the document.
//
// `onChunks(fresh)` fires once per newly seen chunk id so a STREAMING document can declare
// them as the shell renders, instead of waiting for the whole flight.
function recordReferenceChunks(serverConsumerManifest, clientChunksById, onChunks) {{
  const byId = clientChunksById || {{}};
  const seen = new Set();
  const moduleMap = new Proxy(serverConsumerManifest.moduleMap, {{
    get(target, key) {{
      if (typeof key === "string") {{
        const fresh = [];
        // The manifest's flat `[chunkId, chunkFile, ...]`; the loader takes the id, and
        // diffpack's chunk id IS its file name, so the odd entries add nothing here.
        for (let i = 0; i < (byId[key] || []).length; i += 2) {{
          const chunk = byId[key][i];
          if (!seen.has(chunk)) {{
            seen.add(chunk);
            fresh.push(chunk);
          }}
        }}
        if (fresh.length > 0 && onChunks) onChunks(fresh);
      }}
      return target[key];
    }},
  }});
  return {{
    manifest: {{ ...serverConsumerManifest, moduleMap }},
    chunks: () => [...seen],
  }};
}}

// Reconstruct the flight and render the whole document to an HTML string. The
// client bootstrap module (`/client.js`) and the inlined flight are injected via
// react-dom's bootstrap options, so the served DOM (scripts included) is exactly
// what hydration on the browser expects — no mismatch.
export async function renderFlightToDocument(flightBytes, serverConsumerManifest, flightBase64, params, url, nonce, clientChunksById) {{
  installSeam();
  // Everything is rendered before a byte is sent on this path, so the recorded set is
  // complete and goes into the document as one list.
  const recorder = recordReferenceChunks(serverConsumerManifest, clientChunksById, null);
  serverConsumerManifest = recorder.manifest;
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
        // The SAME shape the client's `Router` hydrates with: its innermost provider
        // wraps a Fragment holding TWO children — the tree, and the intercept-modal
        // slot, which is a `null` SLOT (not an absent child) whenever no modal is open,
        // which is always the case at hydration. React's `useId` is derived from the
        // tree-id FORK a multi-child parent pushes, not from the rendered markup, so a
        // single-child `flightRoot` here and a two-child Fragment there made every
        // `useId` under the tree disagree across the seam: cal.com's base-ui inputs
        // server-rendered `base-ui-_R_2lmdbpi_` and client-rendered `base-ui-_R_amplf69_`,
        // which React reports as a hydration mismatch it "won't patch up" — leaving every
        // <label for> pointing at an id no input carries. Providers rendering no DOM is
        // why this was invisible in the markup and still wrong.
        createElement(
          SearchParamsContext.Provider,
          {{ value: search }},
          createElement(Fragment, null, flightRoot, null),
        ),
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
      // This route's client-reference chunks, as real module scripts AFTER the entry (which
      // creates the runtime they register into), then the boot call. Module scripts run in
      // document order and all before DOMContentLoaded, so hydration lands before DCL —
      // which is what an entry awaiting its own fetches could not give us.
      const nonceAttr = nonce ? " nonce=" + JSON.stringify(String(nonce)) : "";
      const moduleTag = (src) =>
        "<script type=\"module\"" + nonceAttr + " src=" + JSON.stringify(src) + "></script>";
      const tags =
        moduleTag({client_js}) +
        recorder.chunks().map((chunk) => moduleTag(chunkUrl(chunk))).join("") +
        "<script type=\"module\"" + nonceAttr + ">window.__DIFFPACK_BOOT_REQUESTED__=1;if(globalThis.__diffpackBoot)globalThis.__diffpackBoot()</script>";
      const bodyEnd = html.lastIndexOf("</body>");
      html = bodyEnd === -1 ? html + tags : html.slice(0, bodyEnd) + tags + html.slice(bodyEnd);
      if (inserted.length) {{
        const extra = inserted.map((cb) => renderToStaticMarkup(cb())).join("");
        inserted.length = 0;
        const at = html.indexOf("</head>");
        html = at === -1 ? extra + html : html.slice(0, at) + extra + html.slice(at);
      }}
      resolve(html);
    }});
    sink.on("error", reject);
    // react-dom's ready callbacks are NOT fire-once, so `pipe` (which throws
    // "React currently only supports piping to one writable stream." on a second call)
    // has to be guarded. Measured, not defensive: when the last work to finish is a
    // Suspense boundary that still holds abortable fallback tasks, `finishedTask`
    // decrements `allPendingTasks` FIRST, then aborts those fallback tasks — and each
    // abort re-enters `finishedTask`, whose own tail sees the counter already at 0 and
    // calls `completeAll` -> `onAllReady`. The outer frame's tail then calls it AGAIN.
    // (Stack captured on React 19.3.0-canary as vendored by Next 16, rendering
    // `integration/next-app-router`'s /error-demo.) Everything React itself passes for
    // `onAllReady` is a promise `resolve`, which absorbs the second call silently; ours
    // did not, and the throw came back out through the enclosing task's catch as a
    // RECOVERABLE error — logged once per request, and enough to mark an
    // already-completed boundary client-rendered had the destination not already closed.
    let piped = false;
    const {{ pipe }} = renderToPipeableStream(root, {{
      // Content-Security-Policy: the request's `script-src 'nonce-…'` value, so every
      // script react-dom emits (the bootstrap module + the inline bootstrap content)
      // carries it. Without it a strict-CSP app blocks its own hydration.
      nonce: nonce || undefined,
      // NO `bootstrapModules`. react-dom emits that tag with `async`, and an async module
      // script is unordered against the chunk scripts below it — so a chunk could execute
      // before the entry that creates the runtime it registers into, and throw. The entry is
      // emitted with the chunks instead, all plain (deferred, ORDERED) module scripts.
      bootstrapScriptContent:
        "window.__DIFFPACK_FLIGHT__ = " + JSON.stringify(flightBase64) + ";" +
        // Hydration is started by the document, after the chunk scripts below it.
        "window.__DIFFPACK_DEFER_BOOT__ = 1;" +
        "window.__DIFFPACK_PARAMS__ = " + JSON.stringify(params || {{}}) + ";" +
        "window.__DIFFPACK_URL__ = " + JSON.stringify({{ pathname: pathname, search: search }}) + ";",
      onAllReady() {{
        if (piped) return;
        piped = true;
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
// All bytes go to `res` through ONE ordered destination (`createFlightSink`), which
// forwards React's chunks untouched and injects the queued flight scripts ONLY at a
// react-dom flush-cycle boundary — react-dom's own `write()` boundaries fall every 2048
// bytes and routinely land inside an HTML token (see flight_sink.js).
export async function renderFlightToStream(flightChunks, serverConsumerManifest, params, url, res, headers, status, nonce, clientChunksById) {{
  installSeam();
  const pathname = (url && url.pathname) || "/";
  const search = (url && url.search) || "";
  // Live byte stream feeding the SSR flight reconstruction + a queue of the inline
  // `<script>` tags carrying the same chunks for the client. One pump drives both.
  let byteController;
  const byteStream = new ReadableStream({{ start(c) {{ byteController = c; }} }});
  const scriptQueue = [];
  // Content-Security-Policy: a strict-CSP app (middleware sets `script-src 'nonce-…'`)
  // blocks every inline script that does not carry the nonce — including the flight
  // chunks below, without which the client never hydrates. Same value react-dom stamps
  // on the bootstrap scripts via the `nonce` render option.
  const nonceAttr = nonce ? " nonce=" + JSON.stringify(String(nonce)) : "";
  let pumpDone = false;
  // Assigned once the destination exists; the pump nudges it so chunks that arrive while
  // React has nothing to flush still reach the client at the next macrotask boundary.
  let sink = null;
  // The chunks each resolved client reference lives in, declared into the SAME ordered
  // destination as the flight so they reach the browser as the shell renders rather than
  // after the whole stream. `client.js` is a module script, so it runs after the document
  // is parsed and therefore after every one of these — the list it reads is complete.
  const moduleTag = (src) =>
    "<script type=\"module\"" + nonceAttr + " src=" + JSON.stringify(src) + "></script>";
  // The entry first: every chunk registers into the runtime it creates. Queued before the
  // flight is decoded, which is when the first client reference can be discovered.
  scriptQueue.push(moduleTag({client_js}));
  const recorder = recordReferenceChunks(serverConsumerManifest, clientChunksById, (fresh) => {{
    for (const chunk of fresh) scriptQueue.push(moduleTag(chunkUrl(chunk)));
    if (sink) sink.scheduleDrain();
  }});
  serverConsumerManifest = recorder.manifest;
  const pump = (async () => {{
    for await (const b64 of flightChunks) {{
      const binary = Buffer.from(b64, "base64");
      byteController.enqueue(new Uint8Array(binary));
      scriptQueue.push(
        "<script" + nonceAttr + ">(self.__DF_FLIGHT=self.__DF_FLIGHT||[]).push([1," + JSON.stringify(b64) + "])</script>",
      );
      if (sink) sink.scheduleDrain();
    }}
    byteController.close();
    scriptQueue.push("<script" + nonceAttr + ">(self.__DF_FLIGHT=self.__DF_FLIGHT||[]).push([0])</script>");
    scriptQueue.push("<script type=\"module\"" + nonceAttr + ">window.__DIFFPACK_BOOT_REQUESTED__=1;if(globalThis.__diffpackBoot)globalThis.__diffpackBoot()</script>");
    if (sink) sink.scheduleDrain();
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
  // useServerInsertedHTML: per-request callbacks flushed into the byte stream at each
  // flush-cycle boundary (streaming registries push as boundaries resolve). Shell styles
  // land with the first boundary; late registrations flush after the shell
  // (styled-components tolerates this). Zero cost unless a registry registers one.
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
        // The SAME shape the client's `Router` hydrates with: its innermost provider
        // wraps a Fragment holding TWO children — the tree, and the intercept-modal
        // slot, which is a `null` SLOT (not an absent child) whenever no modal is open,
        // which is always the case at hydration. React's `useId` is derived from the
        // tree-id FORK a multi-child parent pushes, not from the rendered markup, so a
        // single-child `flightRoot` here and a two-child Fragment there made every
        // `useId` under the tree disagree across the seam: cal.com's base-ui inputs
        // server-rendered `base-ui-_R_2lmdbpi_` and client-rendered `base-ui-_R_amplf69_`,
        // which React reports as a hydration mismatch it "won't patch up" — leaving every
        // <label for> pointing at an id no input carries. Providers rendering no DOM is
        // why this was invisible in the markup and still wrong.
        createElement(
          SearchParamsContext.Provider,
          {{ value: search }},
          createElement(Fragment, null, flightRoot, null),
        ),
      ),
    ),
  );
  await new Promise((resolve, reject) => {{
    let shellStarted = false;
    sink = createFlightSink({{
      res,
      scriptQueue,
      renderInserted() {{
        const out = [];
        while (insertedFlushed < inserted.length) {{
          out.push(renderToStaticMarkup(inserted[insertedFlushed++]()));
        }}
        return out;
      }},
      onFirstWrite() {{
        shellStarted = true;
      }},
      // React ends the destination once every boundary is done; the pump's terminal
      // `push([0])` may still be in flight, so wait for it before closing `res`.
      beforeEnd: () => pump,
    }});
    sink.on("finish", resolve);
    sink.on("error", reject);
    // Guarded for the same reason the buffered path's `onAllReady` is (see there): a
    // react-dom ready callback can fire more than once, and the second `pipe` throws.
    // Here it would ALSO be a second `res.writeHead` on a response whose head has gone
    // out — ERR_HTTP_HEADERS_SENT, which this path is otherwise careful to never risk.
    let piped = false;
    const {{ pipe }} = renderToPipeableStream(root, {{
      nonce: nonce || undefined,
      // See the buffered path: react-dom would emit this `async`, which is unordered against
      // the chunk scripts. It is queued into the ordered destination instead, ahead of them.
      // No inlined full flight here — it streams as __DF_FLIGHT scripts. Seed the array
      // (so it exists before client.js runs) + the hooks-context globals.
      bootstrapScriptContent:
        "self.__DF_FLIGHT=self.__DF_FLIGHT||[];" +
        // Hydration is started by the document, after this route's chunk scripts.
        "window.__DIFFPACK_DEFER_BOOT__ = 1;" +
        "window.__DIFFPACK_PARAMS__ = " + JSON.stringify(params || {{}}) + ";" +
        "window.__DIFFPACK_URL__ = " + JSON.stringify({{ pathname: pathname, search: search }}) + ";",
      onShellReady() {{
        if (piped) return;
        piped = true;
        res.writeHead(status || 200, headers);
        pipe(sink);
      }},
      onShellError(error) {{
        // Only a response whose status line has NOT gone out can still be a 500;
        // `res.headersSent` is the authority (onShellReady may already have written
        // the head even if the sink has not seen a byte yet). Writing a header on a
        // sent response throws ERR_HTTP_HEADERS_SENT and kills the server process.
        if (!shellStarted && !res.headersSent) {{
          try {{
            res.writeHead(500, {{ "content-type": "text/html; charset=utf-8" }});
            res.end("<!doctype html><p>Internal Server Error</p>");
          }} catch {{}}
        }}
        reject(error);
      }},
      onError(error) {{
        console.error("next-ssr stream onError:", error && error.message ? error.message : error);
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
fn client_entry_module(
    adapter_dir: &Path,
    islands: &[PathBuf],
    eager_islands: &BTreeSet<String>,
    hooks_context: &Path,
    pins_kind: PinKind,
) -> String {
    let pins = island_pins(adapter_dir, islands, eager_islands, pins_kind);
    let lazy = js_str(&adapter_dir.join("lazy.js").to_string_lossy());
    let hooks_import = js_str(&hooks_context.to_string_lossy());
    // The ONE control-flow predicate, shared with the error boundary (see
    // `control_boundary_module`), so hydration's recoverable-error filter and the
    // boundaries can never disagree about what counts as a redirect/notFound.
    let control_import = js_str(&adapter_dir.join("control-boundary.tsx").to_string_lossy());
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
import {{ isControlFlowError }} from {control_import};
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
  // The matched route's dynamic params, sent by the orchestrator with the flight. A soft
  // navigation changes the route, so `useParams()` has to change with it, and the client
  // cannot derive `{{ uid: "…" }}` from a URL whose segment pattern only the server knows.
  const params = parseFlightParams(res.headers.get("x-diffpack-params"));
  // A server-side `redirect()` cannot be a 3xx on this channel — `fetch` follows those
  // transparently and the Router would never learn the URL changed — so the orchestrator
  // reports it as JSON. Handing that JSON to the flight reader (which is what happened
  // before this check existed) fails deep inside the reader and blanks the page: a login
  // whose callback lands on a redirecting route dead-ended on an empty document.
  if ((res.headers.get("content-type") || "").includes("application/json")) {{
    const payload = await res.json();
    if (payload && typeof payload.__redirect === "string") return {{ redirect: payload.__redirect }};
    throw new Error(
      "diffpack next client: unexpected JSON on the soft-navigation channel for " +
        href + ": " + JSON.stringify(payload).slice(0, 200),
    );
  }}
  // A soft navigation's flight is read to completion BEFORE it is handed to the Router,
  // and the resulting tree is awaited, so the Router only ever swaps in a tree that is
  // already renderable. This mirrors Next's own router, whose `fetchServerResponse`
  // awaits `createFromFetch` before the reducer applies the navigation.
  //
  // Handing the Router a live stream instead is what this replaced, and it made the
  // navigation's `startTransition` render suspend on rows that had not arrived yet. A
  // suspended navigation cannot commit ANYTHING — not the new tree, and not the
  // params/pathname/searchParams the Router provides alongside it — while `pushState`
  // has already run, so a lost or stalled tail leaves the tab permanently split: the
  // address bar on the new route, the DOM on the old one, with no error and no retry.
  // (Observed on cal.com: a second visit to `?tabName=recurring` rendered the previous
  // tab forever.) Nothing is given up by buffering: the Router renders the flight root
  // with no Suspense boundary of its own, so React could never commit a partially
  // arrived tree anyway — every successful navigation already waited for the last row.
  const bytes = new Uint8Array(await res.arrayBuffer());
  const tree = createFromReadableStream(
    new ReadableStream({{
      start(controller) {{
        controller.enqueue(bytes);
        controller.close();
      }},
    }}),
    {{ callServer }},
  );
  // Settle it here, where a rejection is a rejected navigation the caller can see, rather
  // than during a render that has no boundary to catch it.
  await tree;
  return {{ tree, intercept, params }};
}}

// Decode the `x-diffpack-params` header the orchestrator stamps on a soft-navigation
// flight. A PRESENT-but-malformed value is a server bug and throws, rather than being
// papered over with `{{}}`: silently empty params are exactly the failure this header
// exists to fix, and they resurface far away as "Required" on a zod schema. An ABSENT
// header is a different case and legitimately means "no params known": a `--static-export`
// host is a plain file server (scripts/rsc/next-static-serve.mjs) that hands back the
// prerendered `.rsc` with no idea which segments were dynamic.
function parseFlightParams(raw) {{
  if (raw == null) return EMPTY_PARAMS;
  let decoded;
  try {{
    decoded = JSON.parse(decodeURIComponent(raw));
  }} catch (error) {{
    throw new Error(
      "diffpack next client: the x-diffpack-params header on the soft-navigation channel is not " +
        "percent-encoded JSON (" + String(raw).slice(0, 120) + "): " + String(error),
    );
  }}
  if (decoded === null || typeof decoded !== "object" || Array.isArray(decoded)) {{
    throw new Error(
      "diffpack next client: the x-diffpack-params header must decode to an object, got " +
        JSON.stringify(decoded).slice(0, 120),
    );
  }}
  return decoded;
}}

// The route identity the app-router hooks read: `useParams()`, `usePathname()` and
// `useSearchParams()`. Recomputed on EVERY navigation (params come from the server with
// the flight; pathname/search are the href the navigation settled on, after redirects),
// so the three hooks describe the route currently rendered rather than the one the
// document was first loaded with.
const EMPTY_PARAMS = {{}};
function routeIdentity(href, params) {{
  const noHash = href.split("#")[0];
  const query = noHash.indexOf("?");
  return {{
    params: params || EMPTY_PARAMS,
    pathname: query === -1 ? noHash : noHash.slice(0, query),
    search: query === -1 ? "" : noHash.slice(query),
  }};
}}

// Follow a soft navigation through any server-side redirects, returning the final
// `{{ tree, intercept }}` plus the href it settled on. Bounded: a redirect cycle is a
// server bug, and looping forever would hang the tab with no diagnostic.
const MAX_REDIRECTS = 10;
async function fetchFlightFollowing(href, take) {{
  let target = href;
  for (let hop = 0; hop <= MAX_REDIRECTS; hop += 1) {{
    const result = await (take(target) || fetchFlight(target));
    if (!result.redirect) return {{ ...result, href: target }};
    target = result.redirect;
  }}
  throw new Error(
    "diffpack next client: more than " + MAX_REDIRECTS + " server redirects following " + href,
  );
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
function Router({{ initialTree, initialRoute }}) {{
  const [tree, setTree] = useState(initialTree);
  const [modal, setModal] = useState(null); // {{ tree }} overlay, or null
  // The route the hooks contexts describe. Swapped together with the tree so
  // useParams/usePathname/useSearchParams never lag a navigation behind.
  const [route, setRoute] = useState(initialRoute);
  const [, startTransition] = useTransition();
  const modalOpen = useRef(false);
  const underlying = useRef(location.pathname + location.search);
  // The route to restore when an intercept overlay closes (the underlying page stays
  // mounted, so its params/pathname/search must come back with it). `currentRoute`
  // mirrors the state for the navigate/close closures, which are created once (the
  // effect has no deps) and so cannot read the `route` state variable.
  const underlyingRoute = useRef(initialRoute);
  const currentRoute = useRef(initialRoute);
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
      const requested = typeof to === "string" ? to : to.href;
      const replace = opts.replace || (typeof to === "object" && to && to.replace);
      // Consume a warmed prefetch (single-use) if one exists, else fetch now — and follow
      // any server redirect, so the URL that lands in history is the one that rendered.
      const take = (target) => {{
        const warmed = prefetchCache.get(target);
        if (warmed) prefetchCache.delete(target);
        return warmed;
      }};
      const {{ tree: next, intercept, params, href }} = await fetchFlightFollowing(requested, take);
      const identity = routeIdentity(href, params);
      if (intercept) {{
        underlying.current = location.pathname + location.search;
        underlyingRoute.current = currentRoute.current;
        modalOpen.current = true;
        currentRoute.current = identity;
        startTransition(() => {{
          setModal({{ tree: next }});
          // The URL is masked to the overlay's target, so the hooks must describe it too.
          setRoute(identity);
          if (push) history.pushState({{ __diffpackModal: true }}, "", href);
        }});
        return;
      }}
      modalOpen.current = false;
      currentRoute.current = identity;
      startTransition(() => {{
        setModal(null);
        setTree(next);
        setRoute(identity);
        if (push) history[replace ? "replaceState" : "pushState"](null, "", href);
        // A back/forward navigation does not push, but if the server redirected, the
        // address bar still shows the URL that no longer renders — correct it in place.
        else if (href !== requested) history.replaceState(null, "", href);
      }});
    }}
    // Close an open overlay (used by a modal's own close / router.back()).
    function closeModal() {{
      if (!modalOpen.current) return;
      modalOpen.current = false;
      setModal(null);
      // The URL un-masks back to the still-mounted underlying page, so the hooks go back
      // to describing it.
      currentRoute.current = underlyingRoute.current;
      setRoute(underlyingRoute.current);
      history.pushState(null, "", underlying.current);
    }}
    // router.refresh(): a SOFT RSC refresh — re-fetch the CURRENT route's flight (bypassing
    // the prefetch cache) and swap the tree inside a transition, keeping the document
    // mounted so island state survives. Never a window.location.reload().
    async function refresh() {{
      const current = location.pathname + location.search;
      const {{ tree: next, params, href }} = await fetchFlightFollowing(current, () => undefined);
      // The route redirected since it was last rendered — land on the new URL.
      if (href !== current) history.replaceState(null, "", href);
      const identity = routeIdentity(href, params);
      currentRoute.current = identity;
      startTransition(() => {{
        setModal(null);
        setTree(next);
        setRoute(identity);
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
  // The app-router hooks contexts are provided HERE, inside the Router, so a soft
  // navigation re-provides them with the new route. Providers render no DOM, so this is
  // the same markup the SSR entry produced (which wraps the flight root in the identical
  // three providers) — hydration sees no difference.
  return createElement(
    PathParamsContext.Provider,
    {{ value: route.params }},
    createElement(
      PathnameContext.Provider,
      {{ value: route.pathname }},
      createElement(
        SearchParamsContext.Provider,
        {{ value: route.search }},
        createElement(
          Fragment,
          null,
          use(tree),
          modal ? createElement(Suspense, {{ fallback: null }}, createElement(ModalPortal, {{ thenable: modal.tree }})) : null,
        ),
      ),
    ),
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

// Every chunk the document declared for this route, loaded BEFORE anything renders.
//
// The RSC seam's `__webpack_require__` is synchronous, so a client reference whose chunk
// has not arrived cannot be rendered — and worse than failing, it hydrates as markup with
// no handlers attached, which looks alive and is not. The document lists them (see
// `recordReferenceChunks` in the SSR entry); this is one parallel wave of fetches, not a
// per-reference waterfall.
//
// A chunk that fails to load is a HARD error naming it: hydration would otherwise die
// further in, on a reference whose module never registered, with nothing left to point at.
// A LATE arrival (a Suspense boundary that resolved after the document was parsed) is
// loaded as it is announced and covered by React's own blocked-chunk path either way.
async function loadDeclaredChunks() {{
  const declared = window.__DIFFPACK_ROUTE_CHUNKS__;
  if (!Array.isArray(declared) || declared.length === 0) return;
  const load = globalThis.__webpack_chunk_load__;
  if (typeof load !== "function") {{
    throw new Error(
      "diffpack next client: the document declared route chunks but the RSC seam installed no __webpack_chunk_load__",
    );
  }}
  // Anything the document already loaded through its own `<script>` tags is registered
  // and resolves instantly here; this only has work to do for a chunk announced late.
  const pending = declared.map((chunk) => load(chunk));
  const push = declared.push.bind(declared);
  declared.push = (...chunks) => {{
    for (const chunk of chunks) Promise.resolve(load(chunk)).catch(() => {{}});
    return push(...chunks);
  }};
  await Promise.all(pending);
}}

async function boot() {{
  await loadDeclaredChunks();
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
  const params = window.__DIFFPACK_PARAMS__ || EMPTY_PARAMS;
  const urlInfo = window.__DIFFPACK_URL__ || {{ pathname: location.pathname, search: location.search }};
  // The Router owns the hooks contexts from here on: it re-provides them on every soft
  // navigation. This is only their INITIAL value, and it is the same one the SSR entry
  // rendered with, so hydration matches exactly.
  const app = createElement(Router, {{
    initialTree,
    initialRoute: {{ params, pathname: urlInfo.pathname, search: urlInfo.search }},
  }});
  // The RootLayout owns the document, so we hydrate the whole document.
  hydrateRoot(document, app, {{
    // A redirect()/notFound() that reaches the browser is CONTROL FLOW, not a failure:
    // the page threw it from behind a Suspense boundary, so no 307 was possible and the
    // control boundary is already completing the navigation. React's DEFAULT
    // onRecoverableError reports every recovered error through `reportError`, which
    // surfaces as an uncaught page error — an ordinary logged-out redirect would look
    // like a crash. Everything else keeps React's default reporting, unchanged.
    onRecoverableError(error, errorInfo) {{
      if (isControlFlowError(error) || isControlFlowError(errorInfo)) return;
      reportError(error);
    }},
  }});
}}

// WHO STARTS HYDRATION, AND WHEN.
//
// The document declares this route's chunks as real `<script type="module">` tags placed
// AFTER this entry, then calls `__diffpackBoot()` in one more module script. Module scripts
// execute in document order and all of them run before `DOMContentLoaded`, so hydration
// lands before DCL exactly as it did when everything was in one chunk — which cal.com's own
// suite depends on: its theme test reads `<html class>` right after DCL, and a page that
// hydrates later still says `light`.
//
// The chunks cannot simply precede this entry: each one registers into the runtime THIS
// module creates and throws if it is missing. So the order is entry, chunks, boot — which is
// also why the entry cannot just await them itself, since a fetch started here always
// resolves after DCL.
//
// A document that does not defer (no client references, or a build with no split chunks)
// gets the historical behaviour: boot immediately.
(globalThis).__diffpackBoot = boot;
// The boot tag and this entry are separate scripts, so either can run first: the tag sets
// `__DIFFPACK_BOOT_REQUESTED__` and calls `__diffpackBoot` if it is already defined, and
// this checks the flag in case the tag got there first. Nothing about the pair depends on
// document order — which is the property the chunk registry now has as well.
if (!window.__DIFFPACK_DEFER_BOOT__ || window.__DIFFPACK_BOOT_REQUESTED__) boot();
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

/// `next/script` shim (`shims/script.tsx`). A `"use client"` reimplementation of Next's
/// `<Script>`: `afterInteractive`/`lazyOnload` contribute only a `ReactDOM.preload` to the
/// server-rendered document and inject the real `<script>` after mount; `beforeInteractive`
/// additionally renders the `<script>` in place so it runs before hydration. Partytown
/// (`strategy="worker"`) is NOT implemented and throws, naming the prop.
///
/// The shim exists because Next's own `next/script` is a CommonJS barrel inside
/// `node_modules` wired to Next-internal singletons; aliasing to a project-local island is
/// what makes the client reference resolvable in all three graphs.
fn next_script_shim() -> &'static str {
    include_str!("next_runtime/next_script_shim.tsx")
}

/// `next/dynamic` shim (`shims/dynamic.ts`). A reimplementation of Next's app-router
/// `dynamic()` keyed on its public option shape (`{ loading, ssr }`), backed by `React.lazy`.
///
/// The load-bearing detail is WHEN a Suspense boundary is placed around the lazy chunk,
/// which diffpack now takes verbatim from Next's own `Loadable`
/// (`next/dist/shared/lib/lazy-dynamic/loadable.js`):
///
/// ```js
/// const hasSuspenseBoundary = !opts.ssr || !!opts.loading
/// const Wrap = hasSuspenseBoundary ? Suspense : Fragment
/// ```
///
/// So the DEFAULT (`ssr: true`, no `loading`) gets **no boundary at all** — see
/// `next_dynamic_matches_next_suspense_boundary_rule` for why that is a correctness rule
/// and not a detail. `ssr:false` renders the `loading` fallback on the server AND the first
/// client paint (a mounted-gate: `useState(false)` + `useEffect`), so the SSR HTML and first
/// hydration match, then swaps in the real chunk after mount — Next reaches the same
/// observable place by throwing `BailoutToCSR` under its own Suspense boundary (and, like
/// Next, `ssr:false` inside a Server Component surfaces React's react-server hook error).
///
/// React is namespace-imported so the client hooks are not named bindings that would fail to
/// resolve under the `react-server` export condition; they are only ever CALLED on the client
/// `ssr:false` path.
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
    // (no hydration mismatch), then load the real chunk. Next's own `ssr:false` path is
    // likewise inside a Suspense boundary (`hasSuspenseBoundary = !opts.ssr || ...`).
    return function DynamicClientOnly(props) {
      const [mounted, setMounted] = useState(false);
      useEffect(() => { setMounted(true); }, []);
      if (!mounted) return fallback;
      return createElement(Suspense, { fallback: fallback }, createElement(LazyComponent, props));
    };
  }
  // NEXT'S RULE, verbatim: `hasSuspenseBoundary = !opts.ssr || !!opts.loading`. With the
  // default `{ ssr: true }` and no `loading`, a dynamic component gets NO boundary of its
  // own — the lazy chunk suspends whatever update is rendering it, all the way up to
  // whatever real boundary the app put there.
  //
  // This is not cosmetic; it decides how many React COMMITS it takes to swap one dynamic
  // component for another, which the DOM can see. With no boundary, a transition that
  // swaps A for B cannot commit until B's chunk has loaded, so A's removal and B's
  // insertion land in ONE commit. Add a boundary and the same swap becomes TWO: A is
  // deleted and the `null` fallback committed at once, then B is inserted milliseconds
  // later — leaving the container observably EMPTY in between.
  //
  // cal.com is the proof. Its event-type tabs are `dynamic()` with no `loading`, rendered
  // into a `@formkit/auto-animate` container, and auto-animate answers a removal by
  // re-attaching the removed node as a ~250 ms exit-ghost. Under a boundary, a
  // MutationObserver on that container saw the tab swap as THREE batches — old panel out
  // (container left holding no live React child at all), then the new panel in ~290 ms
  // later, once the tab's chunk had arrived — and `data-testid`s from a ghost still
  // animating out coexisted with the same ids in a freshly inserted panel: Playwright's
  // strict-mode locator for `[data-testid=offer-seats-toggle]` resolved to 2 elements and
  // the run failed. Without the boundary the same swap is ONE batch, removal and insertion
  // together, and the container is never empty. (`EventSetupTab`, the one tab cal.com
  // declares WITH `loading: () => null`, keeps its boundary on both sides — the rule is
  // per-call, not global.)
  //
  // Next wraps in a `Fragment` here; a top-level unkeyed fragment is unwrapped by the
  // reconciler, so returning the element directly is the same tree.
  return function DynamicComponent(props) {
    if (!Loading) return createElement(LazyComponent, props);
    return createElement(Suspense, { fallback: fallback }, createElement(LazyComponent, props));
  };
}
"#
}

// --- next/image build-time variant emit + manifest (Slice J / gap 4.2) -----------
//
// Next's `<Image>` produces a responsive `srcset` of `/_next/image?url=&w=&q=` URLs.
// The shim (`next_image_shim`, a `getImgProps` port) emits that exact shape, but the
// optimization itself happens at BUILD time (pure-Rust `image` crate): every width in
// the ladder is written to `public/_diffpack-image/` and indexed in `variants.json`,
// which the orchestrator's `/_next/image` handler serves from directly. Runtime
// re-encoding is only for widths/qualities the build did not precompute.
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
    /// A tiny (~8px-wide) base64 `data:` URI for `placeholder="blur"`, generated
    /// natively at scan time from the decoded raster. `None` for `unoptimized`
    /// entries (no decodable raster to blur).
    blur_data_url: Option<String>,
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
pub fn variant_widths(intrinsic: u32) -> Vec<u32> {
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
    scan_public_images_with(root, &ImageOptimization::for_project(root))
}

/// [`scan_public_images`] with the optimization decision already in hand (the adapter
/// resolved `next.config` this pass and does not need to read it back off disk).
///
/// When optimization is off, every entry is registered `unoptimized` WITHOUT decoding
/// the file: the shim renders each one as a plain `<img src>` (no `srcset`), so nothing
/// downstream can consult an intrinsic size or a blur placeholder, and decoding hundreds
/// of rasters to fill fields no one reads is pure build time. This is exactly what
/// `next build` does — Next inspects `public/` not at all.
pub(crate) fn scan_public_images_with(
    root: &Path,
    optimization: &ImageOptimization,
) -> Result<Vec<PublicImage>, String> {
    let public_dir = root.join("public");
    if !public_dir.is_dir() {
        return Ok(Vec::new());
    }
    // Walk first (cheap directory reads), then decode the rasters in PARALLEL — the
    // decode plus blur encode is the whole cost of this scan, and it is per-file
    // independent. The result is re-sorted by `src` below, so the emitted manifest is
    // byte-identical to the sequential walk's.
    let mut files = Vec::new();
    scan_public_images_dir(&public_dir, &public_dir, &mut files)?;
    let mut entries = files
        .into_par_iter()
        .map(|(path, rel, src, ext)| public_image_entry(&path, rel, src, ext, optimization))
        .collect::<Result<Vec<_>, String>>()?;
    entries.sort_by(|a, b| a.src.cmp(&b.src));
    Ok(entries.into_iter().map(PublicImage).collect())
}

/// One `public/` image: decode it (when this build optimizes) into its intrinsic size,
/// variant plan and blur placeholder, or register it as a passthrough.
fn public_image_entry(
    path: &Path,
    rel: PathBuf,
    src: String,
    ext: String,
    optimization: &ImageOptimization,
) -> Result<ImageEntry, String> {
    let passthrough = |ext: String| ImageEntry {
        src: src.clone(),
        rel: rel.clone(),
        ext,
        unoptimized: true,
        width: 0,
        height: 0,
        variants: Vec::new(),
        blur_data_url: None,
    };
    let optimizable = matches!(ext.as_str(), "png" | "jpg" | "jpeg")
        && *optimization == ImageOptimization::Enabled;
    if !optimizable {
        return Ok(passthrough(ext));
    }
    // Decode ONCE here: this yields both intrinsic dimensions AND the blurDataURL (a
    // tiny downscale for `placeholder="blur"`). The full decode replaces the old
    // dimensions-only read; the marginal cost is one small resize/encode per
    // optimizable image, at build time.
    let out_ext = if ext == "jpg" { "jpeg" } else { ext.as_str() };
    match image::open(path) {
        Ok(decoded) if decoded.width() > 0 && decoded.height() > 0 => {
            let width = decoded.width();
            let height = decoded.height();
            Ok(ImageEntry {
                src,
                rel,
                ext: out_ext.to_string(),
                unoptimized: false,
                width,
                height,
                variants: variant_widths(width),
                blur_data_url: Some(generate_blur_data_url(&decoded, out_ext)?),
            })
        }
        // Undecodable/zero-size raster: register it unoptimized rather than throw at
        // the shim (honest passthrough, no fake variants).
        _ => Ok(passthrough(ext)),
    }
}

fn generate_blur_data_url(img: &image::DynamicImage, ext: &str) -> Result<String, String> {
    const TABLE: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let (width, height) = (img.width().max(1), img.height().max(1));
    let target_height = ((height as u64 * 8) / width as u64).max(1) as u32;
    let small = img.resize_exact(8, target_height, image::imageops::FilterType::Triangle);
    let (format, mime) = if matches!(ext, "jpeg" | "jpg") {
        (image::ImageFormat::Jpeg, "image/jpeg")
    } else {
        (image::ImageFormat::Png, "image/png")
    };
    let mut encoded = std::io::Cursor::new(Vec::new());
    small
        .write_to(&mut encoded, format)
        .map_err(|error| format!("cannot encode blur placeholder: {error}"))?;
    let bytes = encoded.into_inner();
    let mut base64 = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let bits = ((chunk[0] as u32) << 16)
            | ((chunk.get(1).copied().unwrap_or(0) as u32) << 8)
            | chunk.get(2).copied().unwrap_or(0) as u32;
        base64.push(TABLE[((bits >> 18) & 63) as usize] as char);
        base64.push(TABLE[((bits >> 12) & 63) as usize] as char);
        base64.push(if chunk.len() > 1 {
            TABLE[((bits >> 6) & 63) as usize] as char
        } else {
            '='
        });
        base64.push(if chunk.len() > 2 {
            TABLE[(bits & 63) as usize] as char
        } else {
            '='
        });
    }
    Ok(format!("data:{mime};base64,{base64}"))
}

/// Collect every image file under `public/` as `(path, rel, served src, lowercase ext)`.
/// Directory walking only — no decoding, so it stays sequential and cheap.
type PublicImageFile = (PathBuf, PathBuf, String, String);

fn scan_public_images_dir(
    base: &Path,
    dir: &Path,
    out: &mut Vec<PublicImageFile>,
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
        out.push((path, rel, src, ext));
    }
    Ok(())
}

/// Opaque handle over the internal image record so callers can drive the variant
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
/// as the head-link emitter, so the copied files and the
/// linked URLs cannot drift.
pub fn emit_metadata_images(root: &Path, out_public: &Path) -> Result<usize, String> {
    // A no-op (not an error) for a project with no app dir at all: main.rs calls this on
    // EVERY build, including non-Next ones.
    let Some(app_dir) = app_dir(root) else {
        return Ok(0);
    };
    let images = scan_metadata_images(&app_dir)?;
    let mut written = 0usize;
    for img in &images {
        std::fs::create_dir_all(out_public)
            .map_err(|error| format!("cannot create {}: {error}", out_public.display()))?;
        let dest = out_public.join(img.served.trim_start_matches('/'));
        if img.generator {
            prerender_og_image(root, &img.source, &dest)?;
        } else {
            std::fs::copy(&img.source, &dest).map_err(|error| {
                format!(
                    "cannot copy metadata image {} -> {}: {error}",
                    img.source.display(),
                    dest.display()
                )
            })?;
        }
        written += 1;
    }
    Ok(written)
}

/// Prerender a code-based `@vercel/og` ImageResponse GENERATOR (`opengraph-image.tsx`
/// etc.) to a PNG at build time. Transforms the generator to standalone ESM (TS stripped,
/// JSX lowered to `react/jsx-runtime`, imports untouched), writes it + the runner inside
/// the app tree (so Node resolves `@vercel/og`/`next/og`/`react` from the app's
/// node_modules), then runs the runner under Node to invoke the generator and capture the
/// rendered image bytes. Any transform/runtime failure is a hard error surfacing the
/// cause (never a silent or empty image). The satori/resvg rendering is @vercel/og's own
/// concern — diffpack drives it through the app's installed copy, adding no heavy dep.
fn transform_to_standalone_esm(path: &Path, source: &str) -> Result<String, String> {
    use oxc_allocator::Allocator;
    use oxc_codegen::Codegen;
    use oxc_parser::Parser;
    use oxc_semantic::SemanticBuilder;
    use oxc_transformer::{TransformOptions, Transformer};

    let allocator = Allocator::default();
    let source_type = diffpack_core::parser::source_type_for(
        path,
        diffpack_core::parser::JsxExtensions::JsxInJavaScript,
    );
    let parsed = Parser::new(&allocator, source, source_type).parse();
    let mut diagnostics: Vec<String> = parsed
        .diagnostics
        .into_iter()
        .map(|d| d.to_string())
        .collect();
    let mut program = parsed.program;
    let semantic = SemanticBuilder::new().build(&program);
    diagnostics.extend(semantic.diagnostics.into_iter().map(|d| d.to_string()));
    let transformed = Transformer::new(&allocator, path, &TransformOptions::default())
        .build_with_scoping(semantic.semantic.into_scoping(), &mut program);
    diagnostics.extend(transformed.diagnostics.into_iter().map(|d| d.to_string()));
    if !diagnostics.is_empty() {
        return Err(format!(
            "cannot transform {} for the @vercel/og prerender: {}",
            path.display(),
            diagnostics.join("; ")
        ));
    }
    Ok(Codegen::new().build(&program).code)
}

fn prerender_og_image(root: &Path, generator: &Path, dest: &Path) -> Result<(), String> {
    let source = std::fs::read_to_string(generator).map_err(|error| {
        format!(
            "cannot read og image generator {}: {error}",
            generator.display()
        )
    })?;
    let esm = transform_to_standalone_esm(generator, &source)?;
    // A private staging dir inside the app tree (node_modules resolves upward from here).
    let stage = root.join(ADAPTER_DIR).join("og");
    std::fs::create_dir_all(&stage)
        .map_err(|error| format!("cannot create {}: {error}", stage.display()))?;
    let stem = generator
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("opengraph-image");
    let gen_mjs = stage.join(format!("{stem}.mjs"));
    std::fs::write(&gen_mjs, esm)
        .map_err(|error| format!("cannot write {}: {error}", gen_mjs.display()))?;
    let runner = stage.join("og-prerender.mjs");
    std::fs::write(
        &runner,
        include_str!("../../../scripts/rsc/og-prerender.mjs"),
    )
    .map_err(|error| format!("cannot write {}: {error}", runner.display()))?;
    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|error| format!("cannot create {}: {error}", parent.display()))?;
    }
    let output = std::process::Command::new("node")
        .arg(&runner)
        .arg(&gen_mjs)
        .arg(dest)
        .current_dir(root)
        // An `opengraph-image` generator is app code: under `next build` it runs in the
        // process that loaded next.config, so it sees the environment that config
        // produced.
        .envs(config_env_from_manifest(root))
        .output()
        .map_err(|error| {
            format!(
                "cannot run the @vercel/og prerender for {} (is `node` on PATH?): {error}",
                generator.display(),
            )
        })?;
    if !output.status.success() {
        return Err(format!(
            "diffpack next metadata: prerendering the @vercel/og ImageResponse for {} failed:\n{}",
            generator.display(),
            String::from_utf8_lossy(&output.stderr).trim(),
        ));
    }
    Ok(())
}

pub fn emit_image_variants(
    root: &Path,
    out_public: &Path,
    images: &[PublicImage],
) -> Result<usize, String> {
    let public_dir = root.join("public");
    let variant_dir = out_public.join("_diffpack-image");
    let optimizable = images
        .iter()
        .filter(|PublicImage(entry)| !entry.unoptimized)
        .collect::<Vec<_>>();
    if !optimizable.is_empty() {
        std::fs::create_dir_all(&variant_dir)
            .map_err(|error| format!("cannot create {}: {error}", variant_dir.display()))?;
    }
    // Every image's ladder is independent — one decode plus N resize/encodes, all
    // writing distinct content-hashed file names — so the whole emit fans out across
    // the rayon pool. Sequentially this was the single largest phase of a production
    // build on an image-heavy app.
    let per_image = optimizable
        .into_par_iter()
        .map(|PublicImage(entry)| {
            let source = public_dir.join(&entry.rel);
            let decoded = image::open(&source)
                .map_err(|error| format!("cannot decode image {}: {error}", source.display()))?;
            for &w in &entry.variants {
                // Preserve aspect ratio; `resize` never upscales past the requested box,
                // and we only request widths `<=` intrinsic, so this is downscale-or-copy.
                let target_h =
                    ((entry.height as u64 * w as u64) / entry.width.max(1) as u64).max(1);
                let variant =
                    decoded.resize(w, target_h as u32, image::imageops::FilterType::Triangle);
                let dest =
                    variant_dir.join(format!("{}-{w}.{}", image_hash(&entry.src), entry.ext));
                variant
                    .save(&dest)
                    .map_err(|error| format!("cannot write {}: {error}", dest.display()))?;
            }
            let widths: serde_json::Map<String, serde_json::Value> = entry
                .variants
                .iter()
                .map(|&w| {
                    (
                        w.to_string(),
                        serde_json::Value::String(image_variant_url(&entry.src, w, &entry.ext)),
                    )
                })
                .collect();
            Ok((
                entry.src.clone(),
                serde_json::json!({ "width": entry.width, "widths": widths }),
                entry.variants.len(),
            ))
        })
        .collect::<Result<Vec<_>, String>>()?;
    let written = per_image.iter().map(|(_, _, count)| count).sum::<usize>();
    // Re-keyed in `images` order (already sorted by src), so the manifest is identical
    // whatever order the pool finished in.
    let mut served: serde_json::Map<String, serde_json::Value> = serde_json::Map::new();
    for (src, plan, _) in per_image {
        served.insert(src, plan);
    }
    // The orchestrator's `/_next/image` handler reads this to answer a request from a
    // build-emitted variant instead of re-optimizing at runtime. Written even when
    // empty so a missing file always means "this build emitted no variants" rather
    // than "the manifest step silently didn't run".
    std::fs::create_dir_all(&variant_dir)
        .map_err(|error| format!("cannot create {}: {error}", variant_dir.display()))?;
    let manifest_path = variant_dir.join(IMAGE_VARIANT_MANIFEST);
    std::fs::write(
        &manifest_path,
        serde_json::to_string_pretty(&serde_json::Value::Object(served))
            .map_err(|error| format!("cannot serialize the image variant manifest: {error}"))?,
    )
    .map_err(|error| format!("cannot write {}: {error}", manifest_path.display()))?;
    Ok(written)
}

/// File name of the build-emitted variant manifest the orchestrator's `/_next/image`
/// handler reads (under `<out>/public/_diffpack-image/`). Kept in one place so the
/// emitter here and the reader in `scripts/rsc/next-server.mjs` cannot drift silently.
pub(crate) const IMAGE_VARIANT_MANIFEST: &str = "variants.json";

/// Generate the `.diffpack-next/image-manifest.ts` module the `next/image` shim
/// imports: a default-exported map from served src URL to its variant plan. Always
/// written (an empty map when the app has no public images) so the shim's import
/// resolves in every graph.
fn image_manifest_module(images: &[PublicImage]) -> String {
    let mut body = String::from(
        "// GENERATED by diffpack next-adapter (Slice J / gap 4.2). Maps each public\n\
         // image src to its intrinsic size, blurDataURL and build-emitted responsive\n\
         // variants. The `next/image` shim reads the intrinsic/blur/unoptimized data\n\
         // from here; the variant FILES are served by the orchestrator behind Next's\n\
         // `/_next/image` URL (see `_diffpack-image/variants.json`).\nexport default {\n",
    );
    for PublicImage(entry) in images {
        if entry.unoptimized {
            body.push_str(&format!(
                "  {}: {{ unoptimized: true }},\n",
                js_str(&entry.src)
            ));
            continue;
        }
        let variants = entry
            .variants
            .iter()
            .map(|&w| {
                format!(
                    "{}: {}",
                    js_str(&w.to_string()),
                    js_str(&image_variant_url(&entry.src, w, &entry.ext))
                )
            })
            .collect::<Vec<_>>()
            .join(", ");
        let blur = entry
            .blur_data_url
            .as_ref()
            .map(|data| format!(", blurDataURL: {}", js_str(data)))
            .unwrap_or_default();
        body.push_str(&format!(
            "  {}: {{ width: {}, height: {}, variants: {{ {variants} }}{blur} }},\n",
            js_str(&entry.src),
            entry.width,
            entry.height,
        ));
    }
    body.push_str("};\n");
    body
}

fn next_image_shim(asset_base: &str) -> String {
    // ASSET_BASE (assetPrefix + basePath) is baked in so every emitted next/image URL —
    // build-time variant, static import, raw local src, and the runtime `/_next/image`
    // optimizer endpoint — carries the configured prefix. A local raster with no build
    // variant, or a remote src, falls back to that runtime optimizer.
    format!(
        "{NEXT_IMAGE_SHIM_HEADER}const ASSET_BASE = {};\n{NEXT_IMAGE_SHIM_BODY}",
        js_str(asset_base),
    )
}

const NEXT_IMAGE_SHIM_HEADER: &str = r#"// `next/image` (diffpack next app-router adapter) — a faithful port of Next's
// `getImgProps`. With Next's DEFAULT loader every optimizable src — a `/public`
// string, a static import, or an allow-listed remote — renders the optimizer URL
// shape `/_next/image?url=&w=&q=`, exactly as Next does; there is no "prefer a
// build-time file" branch, because Next has none. The pixels are still computed at
// BUILD time: the orchestrator answers those `/_next/image` requests from the
// build-emitted responsive variants (pure-Rust `image` crate) whenever one exists,
// and only shells out to the native optimizer for a width/quality the build did not
// precompute. SVG / `data:` / `blob:` / `unoptimized` srcs render the raw src with
// NO `srcSet` (byte-faithful to Next's SVG handling). `priority`/`preload`
// render a `<link rel="preload" as="image">` that React 19 hoists into <head>.
// Runs in all three graphs (no directive; imported by Server
// Components). Static image imports (`import x from './x.png'`) arrive as the
// build-emitted object `{ src, width, height, blurDataURL, variants }`. `placeholder="blur"`
// paints the build-generated blurDataURL as the img's own CSS background so the
// foreground image covers it on load — a zero-runtime approximation of Next (NO
// client JS); a blur requested with no resolvable blurDataURL is a hard error.
import { createElement, Fragment } from "react";
import MANIFEST from "../image-manifest";
import CONFIG from "../image-config";

"#;

const NEXT_IMAGE_SHIM_BODY: &str = r#"const DEVICE_SIZES = CONFIG.deviceSizes || [640, 750, 828, 1080, 1200, 1920, 2048, 3840];
const IMAGE_SIZES = CONFIG.imageSizes || [16, 32, 48, 64, 96, 128, 256, 384];
const ALL_SIZES = [...IMAGE_SIZES, ...DEVICE_SIZES];
const IMAGE_ENDPOINT = ASSET_BASE + "/_next/image";
// object-fit values that are not valid background-size values (Next's list).
const INVALID_BACKGROUND_SIZE_VALUES = ["-moz-initial", "fill", "none", "scale-down", undefined];

// Prepend the app's asset base (assetPrefix + basePath) to a LOCAL (leading-slash) URL,
// once — so build-emitted variant URLs and static image srcs resolve under the configured
// prefix. Remote/data/blob URLs and already-prefixed paths pass through untouched.
function withAssetBase(u) {
  if (!ASSET_BASE || typeof u !== "string" || !u.startsWith("/")) return u;
  if (u === ASSET_BASE || u.startsWith(ASSET_BASE + "/")) return u;
  return ASSET_BASE + u;
}
// The runtime optimizer URL for one candidate width (Next's default-loader shape:
// `/_next/image?url=&w=&q=`). The `url` param stays the RAW app-relative src (the
// orchestrator strips basePath before reading it); IMAGE_ENDPOINT carries the prefix.
function optimizerUrl(rawSrc, w, quality) {
  return IMAGE_ENDPOINT + "?url=" + encodeURIComponent(rawSrc) + "&w=" + w + "&q=" + (quality || 75);
}
// A responsive srcset pointing at the optimizer endpoint (one entry per candidate width)
// plus the largest-width finalSrc. This is the DEFAULT-loader path for EVERY optimizable
// src — local, static-import or remote — exactly as in Next.
function optimizerSrcSet(rawSrc, numericWidth, sizes, quality) {
  const { widths, kind } = getWidths(numericWidth, sizes);
  const parts = widths.map((w, i) => optimizerUrl(rawSrc, w, quality) + " " + (kind === "w" ? w + "w" : (i + 1) + "x"));
  const finalSrc = optimizerUrl(rawSrc, widths[widths.length - 1], quality);
  return { srcSet: parts.join(", "), finalSrc, kind };
}

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

export default function Image(props) {
  // Every prop Next's `getImgProps` consumes is destructured here so it can NEVER
  // leak through `...rest` onto the DOM <img> as a bogus attribute.
  const {
    src, alt, width, height, priority, loader, placeholder, blurDataURL,
    fill: fillProp, quality, sizes: sizesProp, unoptimized, loading, fetchPriority, decoding,
    style: styleProp, layout, objectFit, objectPosition, overrideSrc, preload,
    ...rest
  } = props;
  // Port of Next's legacy `layout` mapping: `layout="fill"` implies `fill`, and
  // intrinsic/responsive contribute style plus a default `sizes`.
  let fill = Boolean(fillProp);
  let sizes = sizesProp;
  let style = styleProp;
  if (layout) {
    if (layout === "fill") fill = true;
    const layoutToStyle = {
      intrinsic: { maxWidth: "100%", height: "auto" },
      responsive: { width: "100%", height: "auto" },
    };
    const layoutToSizes = { responsive: "100vw", fill: "100vw" };
    if (layoutToStyle[layout]) style = { ...style, ...layoutToStyle[layout] };
    if (layoutToSizes[layout] && !sizes) sizes = layoutToSizes[layout];
  }
  const isObjectSrc = src != null && typeof src === "object";
  const rawSrc = typeof src === "string" ? src : (src && (src.src || src.default)) || "";
  // A LOCAL src rendered raw (svg/data/unoptimized/unknown passthrough) still needs the
  // asset base prefix; remote/data/blob srcs are left untouched by withAssetBase.
  const displaySrc = withAssetBase(rawSrc);
  // A static image import carries its own dimensions/variants/blur; synthesize an
  // entry from it so the OPTIMIZED path below builds a srcSet from the embedded
  // variants directly, with NO MANIFEST lookup (the hashed /assets/ URL is not a
  // public-image key). Falls back to the public-image MANIFEST for string srcs.
  const entry =
    isObjectSrc && src.width && src.variants
      ? { width: src.width, height: src.height, variants: src.variants }
      : MANIFEST[rawSrc];
  // The blurDataURL resolves from (in order) the explicit prop, a static-import
  // object, or the public-image manifest entry.
  const resolvedBlur =
    blurDataURL || (isObjectSrc ? src.blurDataURL : undefined) || (entry && entry.blurDataURL);
  // `placeholder="blur"` with no resolvable blurDataURL is a hard error (naming the
  // src), never a silent no-op. Provide `blurDataURL`, use a static import, or a
  // public png/jpeg (diffpack auto-generates one at build — but only when next.config
  // leaves image optimization ON; with `images.unoptimized` or a custom loader the
  // build never decodes `public/`, exactly as `next build` never does).
  if (placeholder === "blur" && !resolvedBlur) {
    throw new Error(
      "next/image: placeholder=\"blur\" requires a blurDataURL for src '" + rawSrc +
      "'. Import the image statically (import img from './x.png') or pass the " +
      "`blurDataURL` prop; public png/jpeg get one generated automatically unless " +
      "next.config turns image optimization off (images.unoptimized / images.loader)."
    );
  }
  // Next's `getImgProps` img style, assembled in Next's exact order:
  //   Object.assign(fill ? {...positioning} : {}, showAltText ? {} : { color: "transparent" }, style)
  // `color: transparent` hides the alt text while the image loads; Next only drops it
  // once the <img> has ERRORED (`showAltText`), which a server-rendered <img> never
  // has — so it is unconditional here. Without it every next/image element differs
  // from Next on `color` (and on `border-*-color`, which inherits `currentColor`).
  const imgStyle = Object.assign(
    fill
      ? {
          position: "absolute",
          height: "100%",
          width: "100%",
          left: 0,
          top: 0,
          right: 0,
          bottom: 0,
          objectFit,
          objectPosition,
        }
      : {},
    { color: "transparent" },
    style,
  );
  // Next's placeholder background: the blurDataURL (placeholder="blur") or a caller
  // -supplied `data:` URI (placeholder="data:image/...", the shimmer pattern) painted
  // as the img's own background, which the foreground image covers on load. Zero
  // client JS. `backgroundSize`/`backgroundPosition` derive from the resolved
  // objectFit/objectPosition exactly as Next does.
  const backgroundImage =
    placeholder === "blur" && resolvedBlur
      ? 'url("' + resolvedBlur + '")'
      : placeholder && placeholder !== "empty" && placeholder !== "blur"
        ? 'url("' + placeholder + '")'
        : null;
  const backgroundSize = !INVALID_BACKGROUND_SIZE_VALUES.includes(imgStyle.objectFit)
    ? imgStyle.objectFit
    : imgStyle.objectFit === "fill"
      ? "100% 100%"
      : "cover";
  const placeholderStyle = backgroundImage
    ? {
        backgroundSize,
        backgroundPosition: imgStyle.objectPosition || "50% 50%",
        backgroundRepeat: "no-repeat",
        backgroundImage,
      }
    : {};
  // Next spreads the placeholder background OVER imgStyle (the placeholder wins).
  const finalStyle = { ...imgStyle, ...placeholderStyle };
  // Next tags every <img> it renders with data-nimg ("fill" or "1").
  const dataNimg = fill ? "fill" : "1";
  const isData = rawSrc.startsWith("data:") || rawSrc.startsWith("blob:");
  const isSvg = /\.svg$/i.test(rawSrc.split("?")[0]);
  const isRemote = /^https?:\/\//i.test(rawSrc);
  // A static-import object with no decodable variants (e.g. a format the build's
  // image crate can't optimize) renders unoptimized rather than throwing.
  const objectUnopt = isObjectSrc && !(src.width && src.variants);

  // Loader precedence, matching Next: the `loader` prop > a next.config `loaderFile`
  // (bundled as CONFIG.loaderFn) > a built-in named loader (imgix/cloudinary/akamai).
  // No explicit loader = Next's DEFAULT loader (the `/_next/image` optimizer).
  const explicitLoader =
    typeof loader === "function" ? loader : CONFIG.loaderFn || builtinLoader(CONFIG.loader);
  const isDefaultLoader = !explicitLoader;
  // Next's SVG special case is DEFAULT-LOADER ONLY (`get-img-props`: `isDefaultLoader &&
  // !config.dangerouslyAllowSVG && src.endsWith('.svg')`) — a configured loader still
  // gets to rewrite an SVG src.
  const svgUnopt = isSvg && isDefaultLoader && !CONFIG.dangerouslyAllowSVG;
  const forcedUnopt = Boolean(unoptimized) || CONFIG.unoptimized || isData || svgUnopt || objectUnopt || (entry && entry.unoptimized);

  // Build a loader-driven srcset (one loader call per candidate width).
  const loaderSrcSet = (fn) => {
    const { widths, kind } = getWidths(numericWidth, sizes);
    const parts = widths.map((w, i) => fn({ src: rawSrc, width: w, quality }) + " " + (kind === "w" ? w + "w" : (i + 1) + "x"));
    const finalSrc = fn({ src: rawSrc, width: widths[widths.length - 1], quality });
    return { srcSet: parts.join(", "), finalSrc, kind };
  };

  const numericWidth = typeof width === "number" ? width : Number(width);
  // Next's `isLazy` verbatim: `!priority && !preload && (loading === 'lazy' || loading
  // === undefined)`, forced false for data:/blob:. `loadingFinal = isLazy ? 'lazy' :
  // loading` — so `priority` does NOT erase an explicit `loading="eager"`.
  const isLazy = !priority && !preload && (loading === "lazy" || typeof loading === "undefined") && !isData;
  const imgLoading = isLazy ? "lazy" : loading;
  const imgDecoding = decoding || "async";
  // Next PASSES `fetchPriority` THROUGH — `priority` only drives lazy-loading and the
  // preload link, it never synthesizes fetchPriority="high" (`get-img-props`: the
  // returned props carry the caller's `fetchPriority` unchanged).
  const imgFetchPriority = fetchPriority;
  // Next's generateImgAttrs: a `w`-descriptor srcSet with no caller `sizes` gets
  // `sizes="100vw"`; an `x`-descriptor one (a fixed numeric width) gets none.
  const effectiveSizes = (kind) => (!sizes && kind === "w" ? "100vw" : sizes);

  const baseImg = () =>
    createElement("img", {
      src: overrideSrc || displaySrc,
      alt: alt || "",
      width,
      height,
      decoding: imgDecoding,
      loading: imgLoading,
      fetchPriority: imgFetchPriority,
      "data-nimg": dataNimg,
      ...rest,
      style: finalStyle,
    });

  // An <img> whose src/srcSet come from a loader (with the same base attrs as baseImg),
  // plus the `priority` preload link when requested.
  const loaderImg = (finalSrc, srcSet, kind) => {
    const imgSizes = effectiveSizes(kind);
    const img = createElement("img", {
      src: overrideSrc || finalSrc,
      srcSet,
      sizes: imgSizes,
      alt: alt || "",
      width,
      height,
      decoding: imgDecoding,
      loading: imgLoading,
      fetchPriority: imgFetchPriority,
      "data-nimg": dataNimg,
      ...rest,
      style: finalStyle,
    });
    if (priority || preload) {
      // Next's `ImagePreload` omits `href` whenever an `imageSrcSet` is present (an
      // href-only browser would preload the WRONG candidate) and carries the caller's
      // `fetchPriority`, not a synthesized "high".
      const link = createElement("link", {
        rel: "preload",
        as: "image",
        href: srcSet ? undefined : finalSrc,
        imageSrcSet: srcSet,
        imageSizes: imgSizes,
        fetchPriority: imgFetchPriority,
      });
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
    const { srcSet, finalSrc, kind } = loaderSrcSet(explicitLoader);
    return loaderImg(finalSrc, srcSet, kind);
  }

  // DEFAULT LOADER — Next's `/_next/image` optimizer. Next routes EVERY optimizable
  // image through it: a `/public` string src, a static import, and an allow-listed
  // remote alike (`get-img-props` -> `generateImgAttrs` -> `defaultLoader`, which is
  // unconditionally `${config.path}?url=&w=&q=`). There is no "prefer a build-time
  // file" branch in Next, so there is none here: the build-emitted responsive variants
  // are what the orchestrator SERVES those requests FROM (see `next-server.mjs`
  // `buildVariantFile`), so the pixels are still computed at build time and the URL
  // shape stays byte-faithful to Next.
  //
  // A remote host must be allow-listed — a disallowed host is a clear hard error, as in
  // Next. A local src the optimizer cannot resolve 404s there, naming the file; it is
  // never silently downgraded to a raw <img> here.
  if (isRemote && !hasMatch(new URL(rawSrc))) throw hostnameNotConfigured(rawSrc);
  const { srcSet, finalSrc, kind } = optimizerSrcSet(rawSrc, numericWidth, sizes, quality);
  return loaderImg(finalSrc, srcSet, kind);
}
"#;

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

// IDENTITY STABILITY. Next's navigation hooks return the SAME object across renders
// (`useRouter` reads a context, `useSearchParams`/`useParams` memoize on the context
// value), and real apps depend on it: any `useMemo`/`useEffect` that lists `router`,
// `searchParams` or `params` in its deps re-runs on EVERY render if the hook hands back
// a fresh object, and one such effect that setStates unconditionally is an infinite
// render loop. Returning a freshly built object per call is therefore not a cosmetic
// difference — it is a correctness difference. Hence: one router singleton (it is
// stateless — every method reads window globals), a `useMemo` on the raw search string,
// and shared empty fallbacks instead of fresh `{{}}` / `[]` literals.
const ROUTER = {{
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

export function useRouter() {{
  return ROUTER;
}}

export function usePathname() {{
  return React.useContext(PathnameContext);
}}

// The exact type `useSearchParams()` returns. Next exports the CLASS as well, because
// user code does `instanceof ReadonlyURLSearchParams` and because the mutators must
// refuse: the query string is owned by the router, so writing to the object returned by
// the hook would silently do nothing. Returning a plain `URLSearchParams` (and not
// exporting the name at all) made any module that imports it fail to load outright —
// `The requested module "next/navigation" does not provide an export named
// "ReadonlyURLSearchParams"`.
class ReadonlyURLSearchParamsError extends Error {{
  constructor() {{
    super("Method unavailable on `ReadonlyURLSearchParams`. The search params are owned by the router — navigate with router.push/replace to change them.");
  }}
}}
export class ReadonlyURLSearchParams extends URLSearchParams {{
  append() {{ throw new ReadonlyURLSearchParamsError(); }}
  delete() {{ throw new ReadonlyURLSearchParamsError(); }}
  set() {{ throw new ReadonlyURLSearchParamsError(); }}
  sort() {{ throw new ReadonlyURLSearchParamsError(); }}
}}

export function useSearchParams() {{
  const search = React.useContext(SearchParamsContext) || "";
  return React.useMemo(() => new ReadonlyURLSearchParams(search), [search]);
}}

// `redirect(href, RedirectType.push)` — the second argument's enum, re-exported by Next
// from next/navigation. The values are the strings the NEXT_REDIRECT digest carries.
export const RedirectType = {{ push: "push", replace: "replace" }};

// Next re-exports the CONTEXT beside the hook, so a CSS-in-JS registry can provide its
// own value. Same object the hook reads, or the two would not meet.
export {{ ServerInsertedHTMLContext }};

const EMPTY_PARAMS = {{}};
const EMPTY_SEGMENTS = [];

export function useParams() {{
  return React.useContext(PathParamsContext) || EMPTY_PARAMS;
}}

// The active URL segments below the calling layout, provided by the SEGMENT_BOUNDARY island
// wrapped around each layout in the react-server render (parts.slice(level.slotBase)). A
// named parallelRouteKey (parallel-route slots) is NOT supported by this adapter — it throws
// a CLEAR error rather than silently returning the primary segment.
export function useSelectedLayoutSegments(parallelRouteKey) {{
  if (parallelRouteKey !== undefined) {{
    throw new Error("diffpack next shim: useSelectedLayoutSegments(parallelRouteKey) with a named parallel-route slot is not supported by this adapter");
  }}
  return React.useContext(SelectedSegmentContext) || EMPTY_SEGMENTS;
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

// `forbidden()` / `unauthorized()` are Next's 403/401 interrupts, and they only exist
// behind `experimental.authInterrupts` — they need a `forbidden.tsx`/`unauthorized.tsx`
// convention this adapter does not discover. The EXPORTS exist (a missing name is a load
// failure for the whole module), but calling one refuses by name instead of throwing a
// digest no renderer here would turn into a 403/401 page.
export function forbidden() {{
  throw new Error("diffpack next shim: forbidden() needs experimental.authInterrupts and a forbidden.tsx convention, which this adapter does not implement. Render your own 403 UI, or redirect().");
}}

export function unauthorized() {{
  throw new Error("diffpack next shim: unauthorized() needs experimental.authInterrupts and an unauthorized.tsx convention, which this adapter does not implement. Render your own 401 UI, or redirect().");
}}

// The digests above are CONTROL FLOW, not failures: a `catch` that swallows one silently
// cancels the redirect / 404. `unstable_rethrow` is what user code calls first inside a
// catch to let them through.
export function unstable_rethrow(error) {{
  const digest = error && error.digest;
  if (typeof digest === "string" && (digest.startsWith("NEXT_REDIRECT;") || digest.startsWith("NEXT_HTTP_ERROR_FALLBACK;") || digest === "DIFFPACK_DYNAMIC_BAILOUT")) {{
    throw error;
  }}
}}

// True when the error means "this deployment does not know that server action id" — the
// deployment-skew case, where the browser holds a reference minted by an older build.
// diffpack's action resolver raises exactly that condition by name.
export function unstable_isUnrecognizedActionError(error) {{
  return !!error && typeof error.message === "string" && error.message.startsWith("diffpack rsc: no server action registered for id ");
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

// A request-state read reached during a BUILD-TIME prerender. There is no request, so
// there is no honest value to return; Next's contract is that the route stops being
// statically prerenderable and is served per-request instead. Carrying the same
// DIFFPACK_DYNAMIC_BAILOUT digest as the no-store case lets `flightControlOnError` classify
// it as control flow (like redirect()/notFound()) rather than a render failure, and the
// message names the API so the build log attributes the demotion to a real call.
function dynamicBailout(api) {{
  return Object.assign(
    new Error("diffpack next shim: " + api + " was called while prerendering — a build-time prerender has no request, so this route cannot be statically prerendered and is served per-request instead"),
    {{ digest: "DIFFPACK_DYNAMIC_BAILOUT" }},
  );
}}

export async function cookies() {{
  const store = requestAls.getStore();
  if (!store) {{
    // Tagged so the SSG prerenderer can distinguish a classifier gap (a route it
    // treated static that actually reads request state) from a generic render failure.
    throw Object.assign(new Error("diffpack next shim: cookies() was called outside a request context (no AsyncLocalStorage store) — call it inside a Server Component during a render"), {{ digest: "DIFFPACK_DYNAMIC_BAILOUT" }});
  }}
  if (store.prerender) throw dynamicBailout("cookies()");
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
  if (store.prerender) throw dynamicBailout("headers()");
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
  if (store.prerender) throw dynamicBailout("draftMode()");
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
import {{ requestAls, cacheScopeAls }} from {request_import};

// Per-worker unstable_cache memo. Module-global → shared by the render AND action module
// instances in the same warm worker, so a revalidateTag during an action purges the value
// a subsequent re-render would otherwise reuse. key -> {{ value, expires|null, tags:[] }}.
const __unstableCacheMemo = new Map();

// Per-worker `"use cache"` memo (same shape/lifetime as the unstable_cache memo). Keyed by
// the export's stable id + its serialized arguments; entries carry the cacheTag() tags and
// the cacheLife()/tagged-fetch TTL collected while the body ran, so revalidateTag purges
// exactly the entries that read the busted tag.
const __useCacheMemo = new Map();

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
  for (const [key, entry] of __useCacheMemo) {{
    if (entry.tags && entry.tags.indexOf(tag) !== -1) __useCacheMemo.delete(key);
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
    // Arguments that JSON cannot represent faithfully (a headers/cookies object, a class
    // instance, a function) MUST NOT share a memo entry: see __isKeyable for the
    // cross-session leak that produced. Uncacheable arguments run through, uncached.
    const argsKey = __cacheArgsKey(args);
    if (argsKey == null) return await fn(...args);
    const key = base + "|" + argsKey + "|" + tags.join(",");
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

// The active `"use cache"` collection scope, or a hard error naming the API — cacheTag /
// cacheLife are only meaningful while a cached export runs (inside the __diffpackUseCache
// wrapper's cacheScopeAls.run), never a silent no-op (repo no-silent-stub rule).
function requireCacheScope(api) {{
  const scope = cacheScopeAls.getStore();
  if (!scope) {{
    throw new Error(
      "diffpack next shim: " + api + " was called outside a use cache scope — call it inside " +
        "a function or module marked \"use cache\"",
    );
  }}
  return scope;
}}

// cacheTag(...tags): associate the surrounding cached value with one or more tags. A later
// revalidateTag(tag) purges the memoized value AND busts every prerendered page that read
// it. Mirrors next/cache cacheTag.
export function cacheTag(...tags) {{
  const scope = requireCacheScope("cacheTag");
  for (const tag of tags) {{
    if (typeof tag !== "string" || !tag) {{
      throw new Error("diffpack next shim: cacheTag(...tags) requires non-empty string tags");
    }}
    scope.tags.add(tag);
  }}
}}

// The named cacheLife profiles Next ships, resolved to a revalidate TTL in SECONDS (the
// only field diffpack's memo needs — `stale`/`expire` are client-hint knobs the page-level
// ISR model already subsumes). "max" means never revalidate on a timer (null).
const __cacheLifeProfiles = {{
  seconds: 1,
  minutes: 60,
  hours: 3600,
  days: 86400,
  weeks: 604800,
  max: null,
  default: 900,
}};

// cacheLife(profile): set the surrounding cached value's revalidate TTL, by named profile
// ("seconds".."weeks"/"max"/"default") or an explicit `{{ revalidate }}` (seconds) object.
// Mirrors next/cache cacheLife (diffpack maps it to the memo entry's soft expiry).
export function cacheLife(profile) {{
  const scope = requireCacheScope("cacheLife");
  let revalidate;
  if (typeof profile === "string") {{
    if (!(profile in __cacheLifeProfiles)) {{
      throw new Error(
        "diffpack next shim: cacheLife(\"" + profile + "\") is not a known profile (expected one " +
          "of " + Object.keys(__cacheLifeProfiles).join(", ") + ", or a {{ revalidate }} object)",
      );
    }}
    revalidate = __cacheLifeProfiles[profile];
  }} else if (profile && typeof profile === "object" && typeof profile.revalidate === "number") {{
    revalidate = profile.revalidate;
  }} else {{
    throw new Error(
      "diffpack next shim: cacheLife(profile) requires a named profile string or a " +
        "{{ revalidate: <seconds> }} object",
    );
  }}
  scope.revalidate = revalidate == null
    ? null
    : scope.revalidate == null
      ? revalidate
      : Math.min(scope.revalidate, revalidate);
}}

// Whether `JSON.stringify` represents `value` FAITHFULLY — i.e. two values that differ
// produce different JSON. It does not for anything JSON silently drops or flattens: a
// function, a symbol, a Map/Set, a class instance (`Headers`, `URL`, `RequestCookies`), a
// Proxy over one, a non-enumerable own property, NaN/Infinity. Those all stringify to
// `{{}}`/`null`, so distinct arguments collapse onto ONE cache key.
//
// This is not hypothetical. cal.com's `/event-types` page caches its data loader with
// `unstable_cache(fn, ["viewer.eventTypes.getUserEventGroups"], {{ revalidate: 3600 }})` and
// passes `await headers()` and `await cookies()` as arguments — the documented way to give
// a cached function request data. Both stringify to a constant, so the FIRST signed-in
// visitor's event types were served to every later visitor for an hour: a different user's
// name, slugs and links, cross-session. Caught by cal.com's own Playwright suite, where the
// second test's page rendered the first test's user.
//
// `undefined` is deliberately treated as faithful even though JSON turns an array hole into
// `null`: omitted trailing arguments are pervasive, and Next accepts the same collision in
// its own key (its source comments the coercion). Everything else fails closed.
function __isKeyable(value, budget) {{
  if (budget.n-- < 0) return false;
  if (value === null || value === undefined) return true;
  const type = typeof value;
  if (type === "boolean" || type === "string") return true;
  if (type === "number") return Number.isFinite(value);
  if (type !== "object") return false; // function, symbol, bigint
  if (Array.isArray(value)) {{
    for (const item of value) {{
      if (!__isKeyable(item, budget)) return false;
    }}
    return true;
  }}
  // An object that declares its own serialization is taken at its word.
  if (typeof value.toJSON === "function") return true;
  const proto = Object.getPrototypeOf(value);
  if (proto !== Object.prototype && proto !== null) return false;
  // Own properties JSON never writes (non-enumerable, or symbol-keyed) are lost.
  if (Object.getOwnPropertyNames(value).length !== Object.keys(value).length) return false;
  if (Object.getOwnPropertySymbols(value).length !== 0) return false;
  for (const key of Object.keys(value)) {{
    if (!__isKeyable(value[key], budget)) return false;
  }}
  return true;
}}

// A stable, throw-free key for a cached call's arguments. Returns null when the arguments
// cannot be serialized (a component's React-element children, a cyclic value) OR when
// serializing them would be LOSSY (see __isKeyable) — the caller then skips the memo for
// that call (never returning another caller's value) while STILL collecting tags.
function __cacheArgsKey(args) {{
  if (!__isKeyable(args, {{ n: 10000 }})) return null;
  try {{
    return JSON.stringify(args);
  }} catch {{
    return null;
  }}
}}

// The `"use cache"` boundary the react-server transform wraps each export in:
// __diffpackUseCache(realFn, "<moduleId>#<export>"). Returns a wrapper that memoizes the
// return per (id + arguments), runs the body inside a cacheTag()/cacheLife() collection
// scope, records the collected TTL as the entry's soft expiry, and ALWAYS propagates the
// entry's tags onto the current request store (hit or miss) so the reading page is
// registered under them for revalidateTag. Native reimplementation of Next's "use cache".
export function __diffpackUseCache(fn, id) {{
  if (typeof fn !== "function") {{
    throw new Error(
      "diffpack next shim: a \"use cache\" export (" + id + ") is not a function; a \"use cache\" " +
        "module may only export cached functions/components",
    );
  }}
  const propagate = (tags) => {{
    if (!tags || !tags.length) return;
    const store = requestAls.getStore();
    if (store) {{
      if (!store.tags) store.tags = new Set();
      for (const tag of tags) store.tags.add(tag);
    }}
  }};
  return async function (...args) {{
    const argsKey = __cacheArgsKey(args);
    const now = Date.now();
    if (argsKey != null) {{
      const key = id + "|" + argsKey;
      const hit = __useCacheMemo.get(key);
      if (hit && (hit.expires == null || hit.expires > now)) {{
        propagate(hit.tags);
        return hit.value;
      }}
      const scope = {{ tags: new Set(), revalidate: null }};
      const value = await cacheScopeAls.run(scope, () => fn.apply(this, args));
      const tags = [...scope.tags];
      __useCacheMemo.set(key, {{
        value,
        expires: scope.revalidate != null ? now + scope.revalidate * 1000 : null,
        tags,
      }});
      propagate(tags);
      return value;
    }}
    // Non-serializable arguments: run uncached (correctness over reuse) but still collect
    // and propagate the tags so the reading page stays bustable by revalidateTag.
    const scope = {{ tags: new Set(), revalidate: null }};
    const value = await cacheScopeAls.run(scope, () => fn.apply(this, args));
    propagate([...scope.tags]);
    return value;
  }};
}}
"#,
    )
}

/// The `type` a `<link rel="preload" as="font">` carries, from the emitted file's
/// extension. Next writes the same MIME strings.
fn font_mime(href: &str) -> &'static str {
    match href
        .rsplit('.')
        .next()
        .map(str::to_ascii_lowercase)
        .as_deref()
    {
        Some("woff2") => "font/woff2",
        Some("woff") => "font/woff",
        Some("ttf") => "font/ttf",
        Some("otf") => "font/otf",
        _ => "application/octet-stream",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn configure(root: &Path, environment: &str) -> Result<Option<AppRouterAppConfig>, String> {
        configure_app_router(root, environment)
    }

    fn configure_dev(
        root: &Path,
        environment: &str,
        scope: &RouteScope,
    ) -> Result<Option<AppRouterAppConfig>, String> {
        configure_app_router_dev(root, environment, scope)
    }

    // --- lazy route compilation: the pattern table + the scope ---------------------

    #[test]
    fn a_pattern_matches_the_same_paths_the_generated_entry_matcher_does() {
        let page = |spelling: &str| RoutePattern::parse(spelling, PatternKind::Page);
        assert!(page("/auth/login").matches("/auth/login"));
        assert!(!page("/auth/login").matches("/auth/login/extra"));
        assert!(!page("/auth/login").matches("/auth"));
        // A dynamic segment takes exactly one part.
        assert!(page("/booking/[uid]").matches("/booking/abc"));
        assert!(!page("/booking/[uid]").matches("/booking"));
        assert!(!page("/booking/[uid]").matches("/booking/a/b"));
        // A catch-all takes one or more; the optional form also takes none.
        assert!(page("/api/trpc/[...trpc]").matches("/api/trpc/viewer.me"));
        assert!(page("/api/trpc/[...trpc]").matches("/api/trpc/a/b/c"));
        assert!(!page("/api/trpc/[...trpc]").matches("/api/trpc"));
        assert!(page("/docs/[[...slug]]").matches("/docs"));
        assert!(page("/docs/[[...slug]]").matches("/docs/a/b"));
        // The root page matches only the root.
        assert!(page("/").matches("/"));
        assert!(!page("/").matches("/anything"));
    }

    #[test]
    fn a_scope_answers_pages_and_endpoints_separately() {
        let scope = RouteScope::pages(["/auth/login".to_string()]);
        assert!(scope.includes("/auth/login", PatternKind::Page));
        assert!(!scope.includes("/pro", PatternKind::Page));
        // `RouteScope::pages` compiles every endpoint: a page whose API 404s is a broken
        // app, not a page that is still compiling.
        assert!(scope.includes("/api/trpc/[...trpc]", PatternKind::Endpoint));
        assert!(scope.includes("/api/anything", PatternKind::Endpoint));

        let narrow = RouteScope::pages_and_endpoints(
            ["/auth/login".to_string()],
            ["/api/auth/[...nextauth]".to_string()],
        );
        assert!(narrow.includes("/api/auth/[...nextauth]", PatternKind::Endpoint));
        assert!(!narrow.includes("/api/trpc/[...trpc]", PatternKind::Endpoint));

        // `All` includes everything, of either kind.
        assert!(RouteScope::All.includes("/whatever", PatternKind::Page));
        assert!(RouteScope::All.includes("/whatever", PatternKind::Endpoint));
    }

    /// A `Discovered` with the given page/handler/endpoint spellings and all four app-root
    /// conventions present. Built fresh per assertion because `Discovered` is a build
    /// internal with no `Clone`.
    fn discovered_fixture(pages: &[&str], handlers: &[&str], endpoints: &[&str]) -> Discovered {
        Discovered {
            routes: pages
                .iter()
                .map(|url_path| Route {
                    url_path: (*url_path).to_string(),
                    segments: RoutePattern::parse(url_path, PatternKind::Page).segments,
                    page: PathBuf::from(format!("/app{url_path}/page.tsx")),
                    levels: Vec::new(),
                    metadata: RouteMetadata::default(),
                    kind: RouteKind::Dynamic,
                    has_generate_static_params: false,
                    dynamic_params: true,
                    dynamic_reason: String::new(),
                    revalidate_seconds: None,
                })
                .collect(),
            root_layout: Some(PathBuf::from("/app/layout.tsx")),
            root_metadata: RouteMetadata::default(),
            app_not_found: Some(PathBuf::from("/app/not-found.tsx")),
            global_error: Some(PathBuf::from("/app/global-error.tsx")),
            handlers: handlers
                .iter()
                .map(|url_path| RouteHandler {
                    url_path: (*url_path).to_string(),
                    segments: RoutePattern::parse(url_path, PatternKind::Endpoint).segments,
                    file: PathBuf::from(format!("/app{url_path}/route.ts")),
                    methods: vec!["GET".to_string()],
                    edge: false,
                })
                .collect(),
            intercepts: Vec::new(),
            meta_images: Vec::new(),
            pages_api: endpoints
                .iter()
                .map(|url_path| PagesApiRoute {
                    url_path: (*url_path).to_string(),
                    segments: RoutePattern::parse(url_path, PatternKind::Endpoint).segments,
                    file: PathBuf::from(format!("/pages{url_path}.ts")),
                })
                .collect(),
        }
    }

    /// The scope decides what the generated entries IMPORT; the app-root conventions are
    /// not route-scoped, because every document is built from them.
    #[test]
    fn applying_a_scope_drops_other_routes_but_never_the_app_root_conventions() {
        let pages = ["/auth/login", "/pro", "/booking/[uid]"];
        let handlers = ["/api/og"];
        let endpoints = ["/api/trpc/[trpc]"];

        // Pages narrowed, endpoints kept (the default lazy shape).
        let mut scoped = discovered_fixture(&pages, &handlers, &endpoints);
        apply_route_scope(&mut scoped, &RouteScope::pages(["/auth/login".to_string()]));
        assert_eq!(
            scoped
                .routes
                .iter()
                .map(|r| r.url_path.as_str())
                .collect::<Vec<_>>(),
            vec!["/auth/login"],
        );
        assert_eq!(scoped.handlers.len(), 1, "endpoints are not page-scoped");
        assert_eq!(scoped.pages_api.len(), 1, "pages API is not page-scoped");
        assert!(
            scoped.root_layout.is_some(),
            "the root layout is never scoped out"
        );
        assert!(
            scoped.app_not_found.is_some(),
            "not-found is never scoped out"
        );
        assert!(
            scoped.global_error.is_some(),
            "global-error is never scoped out"
        );

        // Endpoints narrowed too.
        let mut narrow = discovered_fixture(&pages, &handlers, &endpoints);
        apply_route_scope(
            &mut narrow,
            &RouteScope::pages_and_endpoints(["/pro".to_string()], Vec::<String>::new()),
        );
        assert!(narrow.handlers.is_empty());
        assert!(narrow.pages_api.is_empty());
        assert!(narrow.root_layout.is_some());

        // `All` changes nothing — the production path.
        let mut all = discovered_fixture(&pages, &handlers, &endpoints);
        apply_route_scope(&mut all, &RouteScope::All);
        assert_eq!(all.routes.len(), 3);
        assert_eq!(all.handlers.len(), 1);
        assert_eq!(all.pages_api.len(), 1);
    }

    #[test]
    fn the_unbuilt_table_names_exactly_what_the_scope_left_out() {
        let disc = discovered_fixture(
            &["/auth/login", "/pro"],
            &["/api/og"],
            &["/api/trpc/[trpc]"],
        );
        let unbuilt = unbuilt_patterns(&disc, &RouteScope::pages(["/auth/login".to_string()]));
        assert_eq!(
            unbuilt
                .routes
                .iter()
                .map(|p| p.url_path.as_str())
                .collect::<Vec<_>>(),
            vec!["/pro"],
            "the compiled route is not reported as unbuilt",
        );
        assert!(
            unbuilt.handlers.is_empty() && unbuilt.pages_api.is_empty(),
            "endpoints this scope compiles are not reported as unbuilt",
        );
        // A production build compiles everything, so nothing is ever reported unbuilt —
        // which is what keeps the generated entry identical to before this existed.
        let none = unbuilt_patterns(&disc, &RouteScope::All);
        assert!(none.routes.is_empty() && none.handlers.is_empty() && none.pages_api.is_empty());
    }

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

    /// The navigation hooks must be IDENTITY-STABLE across renders, exactly like Next's.
    /// `useRouter()`/`useSearchParams()`/`useParams()` feed `useMemo`/`useEffect`
    /// dependency arrays all over real apps; a fresh object per call re-runs every such
    /// effect on every render, and one effect that setStates unconditionally (cal.com's
    /// `useInitialFormValues` does) becomes an unbounded render loop that no interaction
    /// can outrun. Executed, not grepped: the shim runs against a React whose `useMemo`
    /// caches per hook slot the way React's does.
    #[test]
    fn navigation_hooks_are_identity_stable_across_renders() {
        if std::process::Command::new("node")
            .arg("--version")
            .output()
            .is_err()
        {
            return;
        }
        let dir = scratch("nav-hook-identity");
        let shim = next_navigation_shim(Path::new("./hooks-context.mjs"));
        // Swap the two real imports for a minimal React + context module: the point of
        // the test is the hook bodies, and a real React render would drag in react-dom.
        let body = shim
            .replace("import * as React from \"react\";", "")
            .replace(
                "import { PathParamsContext, PathnameContext, SearchParamsContext, SelectedSegmentContext, ServerInsertedHTMLContext } from \"./hooks-context.mjs\";",
                "",
            );
        assert!(
            !body.contains("from \"react\""),
            "the react import was replaced: {body}"
        );
        assert!(
            !body.contains("hooks-context"),
            "the hooks-context import was replaced: {body}"
        );
        let harness = format!(
            r##"const PathParamsContext = {{ name: "params" }};
const PathnameContext = {{ name: "pathname" }};
const SearchParamsContext = {{ name: "search" }};
const SelectedSegmentContext = {{ name: "segments" }};
const ServerInsertedHTMLContext = {{ name: "inserted" }};
const CTX = new Map([[SearchParamsContext, "?a=1"]]);
// React's own hook-slot memo: same slot + Object.is-equal deps reuses the value.
const slots = [];
let cursor = 0;
const React = {{
  useContext(c) {{ return CTX.get(c); }},
  useMemo(fn, deps) {{
    const i = cursor++;
    const prev = slots[i];
    if (prev && prev.deps.length === deps.length && prev.deps.every((d, j) => Object.is(d, deps[j]))) {{
      return prev.value;
    }}
    const value = fn();
    slots[i] = {{ deps, value }};
    return value;
  }},
}};
{body}
const render = () => {{ cursor = 0; return [useRouter(), useSearchParams(), useParams(), useSelectedLayoutSegments()]; }};
const first = render();
const second = render();
CTX.set(SearchParamsContext, "?a=2");
const third = render();
console.log(JSON.stringify({{
  router: first[0] === second[0],
  searchParams: first[1] === second[1],
  params: first[2] === second[2],
  segments: first[3] === second[3],
  searchParamsValue: second[1].get("a"),
  searchParamsRebuiltOnChange: third[1] !== second[1] && third[1].get("a") === "2",
  readonly: (() => {{ try {{ second[1].set("a", "9"); return false; }} catch {{ return true; }} }})(),
}}));
"##
        );
        let file = dir.join("nav-identity.mjs");
        std::fs::write(&file, harness).unwrap();
        let out = std::process::Command::new("node")
            .arg(&file)
            .output()
            .unwrap();
        assert!(
            out.status.success(),
            "next/navigation hook harness failed: {}",
            String::from_utf8_lossy(&out.stderr),
        );
        let got: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
        assert_eq!(
            got["router"],
            serde_json::json!(true),
            "useRouter is stable: {got}"
        );
        assert_eq!(
            got["searchParams"],
            serde_json::json!(true),
            "useSearchParams is stable: {got}"
        );
        assert_eq!(
            got["params"],
            serde_json::json!(true),
            "useParams is stable: {got}"
        );
        assert_eq!(
            got["segments"],
            serde_json::json!(true),
            "useSelectedLayoutSegments is stable: {got}"
        );
        assert_eq!(
            got["searchParamsValue"],
            serde_json::json!("1"),
            "the search string is parsed: {got}"
        );
        assert_eq!(
            got["searchParamsRebuiltOnChange"],
            serde_json::json!(true),
            "a changed query string produces a NEW object (stability must not become staleness): {got}"
        );
        assert_eq!(
            got["readonly"],
            serde_json::json!(true),
            "the memoized object still refuses mutation: {got}"
        );
    }

    /// `next/dynamic` must place a Suspense boundary EXACTLY where Next places one, and
    /// nowhere else. Next's app-router `Loadable`
    /// (`next/dist/shared/lib/lazy-dynamic/loadable.js`) decides with
    /// `hasSuspenseBoundary = !opts.ssr || !!opts.loading`, so the DEFAULT call —
    /// `dynamic(() => import(...))`, `ssr:true` and no `loading` — gets none.
    ///
    /// That is a correctness rule, not a styling one. Without a boundary, a not-yet-loaded
    /// chunk suspends the update that renders it, so a transition swapping dynamic A for
    /// dynamic B keeps A on screen and lands A's removal + B's insertion in ONE React
    /// commit. Wrap it in a boundary and the same swap takes TWO: A deleted and the `null`
    /// fallback committed immediately, B inserted milliseconds later, with the container
    /// observably EMPTY in between. cal.com renders its event-type tabs (plain `dynamic()`
    /// calls) into a `@formkit/auto-animate` container, which reacted to that gap by
    /// re-attaching the removed panel as a ~250 ms exit-ghost — duplicating every
    /// `data-testid` in the tab and hard-failing Playwright's strict-mode locators.
    ///
    /// Executed, not grepped: the shim's real body runs against a React stand-in.
    #[test]
    fn next_dynamic_matches_next_suspense_boundary_rule() {
        if std::process::Command::new("node")
            .arg("--version")
            .output()
            .is_err()
        {
            return;
        }
        let dir = scratch("next-dynamic-boundary");
        let body = next_dynamic_shim().replace("import * as React from \"react\";", "");
        assert!(
            !body.contains("from \"react\""),
            "the react import was replaced: {body}"
        );
        let harness = format!(
            r##"const SUSPENSE = {{ tag: "Suspense" }};
// The ssr:false path is the only one that calls hooks; mount it straight away so the
// post-mount tree is what we inspect.
const React = {{
  createElement: (type, props, ...children) => ({{ type, props: props || {{}}, children }}),
  lazy: (load) => ({{ tag: "lazy", load }}),
  Suspense: SUSPENSE,
  useState: () => [true, () => {{}}],
  useEffect: () => {{}},
}};
{body}
const loader = () => Promise.resolve({{ default: () => null }});
const shape = (el) => {{
  if (el === null) return "null";
  if (el.type === SUSPENSE) {{
    return "Suspense(" + (el.props.fallback === null ? "null" : "loading") + ")>" + shape(el.props.children === undefined ? el.children[0] : el.props.children);
  }}
  if (el.type && el.type.tag === "lazy") return "lazy";
  return "other";
}};
const Plain = dynamic(loader);
const WithLoading = dynamic(loader, {{ loading: () => null }});
const NoSsr = dynamic(loader, {{ ssr: false }});
const SsrTrueExplicit = dynamic(loader, {{ ssr: true }});
console.log(JSON.stringify({{
  plain: shape(Plain({{}})),
  withLoading: shape(WithLoading({{}})),
  noSsr: shape(NoSsr({{}})),
  ssrTrueExplicit: shape(SsrTrueExplicit({{}})),
}}));
"##
        );
        let file = dir.join("next-dynamic-boundary.mjs");
        std::fs::write(&file, harness).unwrap();
        let out = std::process::Command::new("node")
            .arg(&file)
            .output()
            .unwrap();
        assert!(
            out.status.success(),
            "next/dynamic harness failed: {}",
            String::from_utf8_lossy(&out.stderr),
        );
        let got: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
        assert_eq!(
            got["plain"],
            serde_json::json!("lazy"),
            "dynamic(loader) — Next's `!opts.ssr || !!opts.loading` is false — gets NO Suspense boundary, \
             so a swap between two of them is one commit, not two: {got}"
        );
        assert_eq!(
            got["ssrTrueExplicit"],
            serde_json::json!("lazy"),
            "an explicit ssr:true with no loading is the same case: {got}"
        );
        assert_eq!(
            got["withLoading"],
            serde_json::json!("Suspense(loading)>lazy"),
            "a `loading` component is what asks for a boundary, and it is the fallback: {got}"
        );
        assert_eq!(
            got["noSsr"],
            serde_json::json!("Suspense(null)>lazy"),
            "ssr:false is client-only and keeps its boundary: {got}"
        );
    }

    #[test]
    fn parse_segment_classifies_conventions() {
        assert!(matches!(parse_segment("blog"), SegParse::Seg(Seg::Static(s)) if s == "blog"));
        assert!(matches!(parse_segment("[slug]"), SegParse::Seg(Seg::Dynamic(s)) if s == "slug"));
        assert!(
            matches!(parse_segment("[...rest]"), SegParse::Seg(Seg::CatchAll(s)) if s == "rest")
        );
        assert!(
            matches!(parse_segment("[[...rest]]"), SegParse::Seg(Seg::OptionalCatchAll(s)) if s == "rest")
        );
        assert!(matches!(parse_segment("(marketing)"), SegParse::Group));
        assert!(matches!(parse_segment("@modal"), SegParse::Skip));
        assert!(matches!(parse_segment("(.)photo"), SegParse::Skip));
    }

    /// A route is its whole SEGMENT TREE, not just its `page`. A `layout.tsx`
    /// anywhere above the page reading `next/headers` makes the entire route
    /// per-request — the page's own source shows nothing.
    ///
    /// This is the dominant real shape, not a corner: cal.com's ROOT layout calls
    /// `headers()` (for the locale and the embed flags), which is why `next build`
    /// prerenders 2 of its ~161 app routes. Reading only the page said 79 routes were
    /// static, and every one of them then tried to render without a request.
    #[test]
    fn a_layout_that_reads_request_state_makes_every_route_below_it_dynamic() {
        let root = scratch("layout-request-state");
        let app = root.join("app");
        std::fs::create_dir_all(app.join("plain")).unwrap();
        std::fs::write(
            app.join("layout.tsx"),
            "import { headers } from \"next/headers\";\n\
             export default async function Root({ children }) {\n\
             const h = await headers();\n  return <html><body>{children}</body></html>;\n}\n",
        )
        .unwrap();
        // A page with NOTHING dynamic in it — the layout above is the only reason.
        std::fs::write(
            app.join("plain/page.tsx"),
            "export default function Plain() { return <p>plain</p>; }\n",
        )
        .unwrap();
        let layout = first_existing(&app, "layout");
        let disc = discover_routes(&app, layout.as_deref()).unwrap();
        let route = disc
            .routes
            .iter()
            .find(|r| r.url_path == "/plain")
            .expect("/plain discovered");
        assert_eq!(
            route.kind,
            RouteKind::Dynamic,
            "a layout reading next/headers makes the route per-request"
        );
        assert!(
            route.dynamic_reason.contains("layout.tsx"),
            "the reason must name the module that reads request state, got: {}",
            route.dynamic_reason
        );
    }

    /// The same tree without the request read stays Static — so the rule above is not
    /// just "any layout makes everything dynamic".
    #[test]
    fn a_layout_that_reads_nothing_leaves_its_routes_static() {
        let root = scratch("layout-no-request-state");
        let app = root.join("app");
        std::fs::create_dir_all(app.join("plain")).unwrap();
        std::fs::write(
            app.join("layout.tsx"),
            "export default function Root({ children }) {\n\
             return <html><body>{children}</body></html>;\n}\n",
        )
        .unwrap();
        std::fs::write(
            app.join("plain/page.tsx"),
            "export default function Plain() { return <p>plain</p>; }\n",
        )
        .unwrap();
        let layout = first_existing(&app, "layout");
        let disc = discover_routes(&app, layout.as_deref()).unwrap();
        let route = disc.routes.iter().find(|r| r.url_path == "/plain").unwrap();
        assert_eq!(route.kind, RouteKind::Static);
    }

    /// `dynamic = "force-static"` still wins: Next's documented behaviour there is that
    /// `cookies()`/`headers()` return EMPTY values rather than opting the route out, so
    /// a force-static page under a headers-reading layout is prerendered.
    #[test]
    fn force_static_survives_a_request_reading_layout() {
        let root = scratch("layout-force-static");
        let app = root.join("app");
        std::fs::create_dir_all(app.join("icons")).unwrap();
        std::fs::write(
            app.join("layout.tsx"),
            "import { headers } from \"next/headers\";\n\
             export default async function Root({ children }) { await headers(); return children; }\n",
        )
        .unwrap();
        std::fs::write(
            app.join("icons/page.tsx"),
            "export const dynamic = \"force-static\";\n\
             export default function Icons() { return <p>icons</p>; }\n",
        )
        .unwrap();
        let layout = first_existing(&app, "layout");
        let disc = discover_routes(&app, layout.as_deref()).unwrap();
        let route = disc.routes.iter().find(|r| r.url_path == "/icons").unwrap();
        assert_eq!(route.kind, RouteKind::ForceStatic);
    }

    #[test]
    fn classify_route_reproduces_next_fixture() {
        // Runs discovery + classification on the REAL fixture and asserts the kinds
        // match what `next build` reports (verified in docs/RSC_SSG_SPEC.md §0):
        // / and /about → Static; /blog/[slug], /go, /error-demo → Dynamic;
        // /products/[id] → Ssg.
        let fixture =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../integration/next-app-router");
        let app = fixture.join("app");
        assert!(app.is_dir(), "fixture app dir missing at {}", app.display());
        let layout = first_existing(&app, "layout");
        let disc = discover_routes(&app, layout.as_deref()).unwrap();
        let kind_of = |path: &str| -> RouteKind {
            disc.routes
                .iter()
                .find(|r| r.url_path == path)
                .unwrap_or_else(|| {
                    panic!(
                        "route {path} not discovered: {:?}",
                        disc.routes.iter().map(|r| &r.url_path).collect::<Vec<_>>()
                    )
                })
                .kind
        };
        assert_eq!(kind_of("/"), RouteKind::Static, "/ should be Static");
        assert_eq!(
            kind_of("/about"),
            RouteKind::Static,
            "/about should be Static"
        );
        assert_eq!(
            kind_of("/blog/[slug]"),
            RouteKind::Dynamic,
            "/blog/[slug] reads cookies() → Dynamic"
        );
        assert_eq!(
            kind_of("/go"),
            RouteKind::Dynamic,
            "/go is force-dynamic → Dynamic"
        );
        assert_eq!(
            kind_of("/error-demo"),
            RouteKind::Dynamic,
            "/error-demo is force-dynamic → Dynamic"
        );
        assert_eq!(
            kind_of("/products/[id]"),
            RouteKind::Ssg,
            "/products/[id] has generateStaticParams → Ssg"
        );
        assert_eq!(
            kind_of("/slow"),
            RouteKind::Dynamic,
            "/slow is force-dynamic → Dynamic"
        );
        // ISR: `export const revalidate = 2` on an otherwise-static route → Isr, and the
        // parsed TTL is carried through for the prerender plan / orchestrator.
        assert_eq!(
            kind_of("/isr"),
            RouteKind::Isr,
            "/isr has revalidate=2 → Isr"
        );
        let isr = disc.routes.iter().find(|r| r.url_path == "/isr").unwrap();
        assert_eq!(
            isr.revalidate_seconds,
            Some(2),
            "/isr carries its revalidate TTL in seconds"
        );

        // PRECEDENCE on the real fixture: /blog/[slug] exports generateStaticParams AND
        // reads cookies(). `next build` classifies it ƒ Dynamic (the cookies read opts the
        // whole route into dynamic rendering) — NOT ● SSG — despite the export. diffpack
        // must reproduce that: the discovered route carries the generateStaticParams export
        // yet is Dynamic, and its reason names the precedence rather than falsely claiming
        // the route lacks generateStaticParams.
        let blog = disc
            .routes
            .iter()
            .find(|r| r.url_path == "/blog/[slug]")
            .unwrap();
        assert!(
            blog.has_generate_static_params,
            "/blog/[slug] fixture must export generateStaticParams (precedence exemplar)"
        );
        assert_eq!(
            blog.kind,
            RouteKind::Dynamic,
            "/blog/[slug] stays Dynamic despite generateStaticParams (cookies read wins)"
        );
        assert!(
            blog.dynamic_reason.contains("despite generateStaticParams"),
            "/blog/[slug] dynamic reason must name the precedence, got: {}",
            blog.dynamic_reason,
        );
        // Contrast: /products/[id] has generateStaticParams and NO request read → Ssg.
        let products = disc
            .routes
            .iter()
            .find(|r| r.url_path == "/products/[id]")
            .unwrap();
        assert!(products.has_generate_static_params && products.kind == RouteKind::Ssg);
    }

    #[test]
    fn static_sibling_of_a_dynamic_segment_is_prerendered() {
        // A dynamic segment does NOT make its whole URL namespace dynamic. app/blog holds
        // BOTH [slug]/page.tsx (Dynamic: it reads cookies) and post/page.mdx (a plain static
        // page: no dynamic segment, no request read, no `export const dynamic`). Next
        // classifies the pair ƒ Dynamic + ○ Static, so /blog/post must be prerendered while
        // /blog/[slug] is skipped.
        //
        // This pins the pair that the SSG gate's old `ls static/blog/*.html` assertion
        // conflated: that glob read "no HTML under blog/", which was only ever equivalent to
        // "the dynamic route wrote nothing" while [slug] was the sole route in the namespace.
        // Adding the MDX sibling made a CORRECT prerender trip it.
        let fixture =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../integration/next-app-router");
        let app = fixture.join("app");
        let layout = first_existing(&app, "layout");
        let disc = discover_routes(&app, layout.as_deref()).unwrap();
        let route = |path: &str| {
            disc.routes
                .iter()
                .find(|r| r.url_path == path)
                .unwrap_or_else(|| {
                    panic!(
                        "route {path} not discovered: {:?}",
                        disc.routes.iter().map(|r| &r.url_path).collect::<Vec<_>>()
                    )
                })
        };
        let post = route("/blog/post");
        assert_eq!(
            post.kind,
            RouteKind::Static,
            "/blog/post is a static MDX sibling of a dynamic segment → Static (it must be prerendered)",
        );
        assert!(
            post.kind.is_prerendered(),
            "/blog/post must be prerendered, not skipped"
        );
        assert_eq!(
            route("/blog/[slug]").kind,
            RouteKind::Dynamic,
            "/blog/[slug] reads cookies() → Dynamic (it must NOT be prerendered)",
        );
        assert!(
            !route("/blog/[slug]").kind.is_prerendered(),
            "/blog/[slug] must be skipped at prerender",
        );
    }

    #[test]
    fn prerender_plan_never_emits_a_file_for_a_dynamic_route() {
        // Structural invariants of the emitted prerender-plan.json — the artifact the SSG
        // gate and the prerenderer both read. These hold for ANY fixture shape, so they do
        // not go stale as routes are added (which is exactly how the old path-glob assertion
        // broke): a dynamic route carries a reason and NO output file, and no prerendered
        // route's file path keeps a literal [bracket] segment.
        let fixture =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../integration/next-app-router");
        let out = scratch("prerender-plan");
        write_prerender_plan(&fixture, &out).unwrap();

        let text = std::fs::read_to_string(out.join("static/prerender-plan.json")).unwrap();
        let plan: Vec<serde_json::Value> = serde_json::from_str(&text).unwrap();
        assert!(!plan.is_empty(), "plan is empty");

        let mut saw_static_blog_sibling = false;
        let mut saw_dynamic_blog = false;
        for entry in &plan {
            let path = entry["path"].as_str().unwrap();
            let kind = entry["kind"].as_str().unwrap();
            if kind == "dynamic" {
                let reason = entry["reason"].as_str().unwrap_or("");
                assert!(
                    !reason.is_empty(),
                    "{path}: dynamic entry carries no reason (never drop a route silently)"
                );
                assert!(
                    entry.get("file").is_none(),
                    "{path}: dynamic entry names an output file {:?} — a dynamic route must produce nothing",
                    entry.get("file"),
                );
            }
            if let Some(file) = entry["file"].as_str() {
                assert!(
                    !file.contains('[') && !file.contains(']'),
                    "{path}: prerendered file {file:?} keeps a literal bracket segment",
                );
            }
            // The exact pair the stale glob conflated, as it lands in the plan.
            if path == "/blog/post" {
                assert_eq!(kind, "static", "/blog/post must be planned static");
                assert_eq!(
                    entry["file"].as_str(),
                    Some("blog/post"),
                    "/blog/post output file"
                );
                saw_static_blog_sibling = true;
            }
            if path == "/blog/[slug]" {
                assert_eq!(kind, "dynamic", "/blog/[slug] must be planned dynamic");
                saw_dynamic_blog = true;
            }
        }
        assert!(saw_static_blog_sibling, "/blog/post missing from the plan");
        assert!(saw_dynamic_blog, "/blog/[slug] missing from the plan");
        let _ = std::fs::remove_dir_all(&out);
    }

    #[test]
    fn parallel_routes_become_layout_slot_props() {
        // The fixture's /dashboard hosts @team and @analytics parallel slots. Discovery
        // must attach them to the dashboard directory's Level (not as separate routes),
        // and the generated react-server entry must compose them as named layout props.
        let fixture =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../integration/next-app-router");
        let app = fixture.join("app");
        let layout = first_existing(&app, "layout");
        let disc = discover_routes(&app, layout.as_deref()).unwrap();

        // @slot dirs never become their own routes.
        assert!(
            !disc.routes.iter().any(|r| r.url_path.contains('@')),
            "no @slot route leaked into the primary table: {:?}",
            disc.routes.iter().map(|r| &r.url_path).collect::<Vec<_>>()
        );

        let dashboard = disc
            .routes
            .iter()
            .find(|r| r.url_path == "/dashboard")
            .expect("/dashboard route");
        // The dashboard-directory level (part_offset 1 = the "dashboard" segment) carries
        // the two slots; the team slot has a page and no default, analytics has a default.
        let level = dashboard
            .levels
            .iter()
            .find(|l| !l.slots.is_empty())
            .expect("a level hosts the @team/@analytics slots");
        assert_eq!(
            level.part_offset, 1,
            "dashboard level consumed one URL segment above its slots"
        );
        let names: Vec<&str> = level.slots.iter().map(|s| s.name.as_str()).collect();
        assert!(
            names.contains(&"team") && names.contains(&"analytics"),
            "slots: {names:?}"
        );
        let team = level.slots.iter().find(|s| s.name == "team").unwrap();
        assert!(!team.routes.is_empty(), "team slot has a page");
        assert!(
            team.default.is_some(),
            "team slot has a default.tsx (Next 16 requires one per slot)"
        );
        let analytics = level.slots.iter().find(|s| s.name == "analytics").unwrap();
        assert!(
            analytics.default.is_some(),
            "analytics slot has a default.tsx fallback"
        );

        // Codegen: the react-server entry emits the slot tables + the matcher/composer.
        let boundary = fixture.join(".diffpack-next/error-boundary.tsx");
        let seg_boundary = fixture.join(".diffpack-next/segment-boundary.tsx");
        let ctl_boundary = fixture.join(".diffpack-next/control-boundary.tsx");
        let reqctx = fixture.join(".diffpack-next/request-context.ts");
        let rsc_src = rsc_entry_module(
            &disc,
            &UnbuiltPatterns::default(),
            &crate::next_font::FontOutput::default(),
            &boundary,
            &seg_boundary,
            &ctl_boundary,
            &reqctx,
            None,
            "",
        );
        assert!(
            rsc_src.contains("slotBase:"),
            "levels carry slotBase: {rsc_src}"
        );
        assert!(
            rsc_src.contains(r#"name: "team""#) && rsc_src.contains(r#"name: "analytics""#),
            "slot tables emitted"
        );
        assert!(
            rsc_src.contains("function matchSlots"),
            "the slot matcher is generated"
        );
        assert!(
            rsc_src.contains("function composeLevels"),
            "the slot composer is generated"
        );
        assert!(
            rsc_src.contains("...slotProps"),
            "matched slots are spread as layout props"
        );
    }

    #[test]
    fn metadata_api_chain_and_resolver_codegen() {
        let fixture =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../integration/next-app-router");
        let app = fixture.join("app");
        let layout = first_existing(&app, "layout");
        let disc = discover_routes(&app, layout.as_deref()).unwrap();
        let boundary = fixture.join(".diffpack-next/error-boundary.tsx");
        let seg_boundary = fixture.join(".diffpack-next/segment-boundary.tsx");
        let ctl_boundary = fixture.join(".diffpack-next/control-boundary.tsx");
        let reqctx = fixture.join(".diffpack-next/request-context.ts");
        let rsc_src = rsc_entry_module(
            &disc,
            &UnbuiltPatterns::default(),
            &crate::next_font::FontOutput::default(),
            &boundary,
            &seg_boundary,
            &ctl_boundary,
            &reqctx,
            None,
            "",
        );
        // Each route carries a metadata namespace chain resolved at render time.
        assert!(
            rsc_src.contains("metaChain: ["),
            "routes carry a metadata chain: {rsc_src}"
        );
        assert!(
            rsc_src.contains("async function resolveMetadata"),
            "the metadata resolver is generated"
        );
        assert!(
            rsc_src.contains("async function MetadataHead"),
            "the async MetadataHead component is generated"
        );
        assert!(
            rsc_src.contains("function mergeMetadata"),
            "metadata merge (title templates) is generated"
        );
        // Full head coverage: openGraph, twitter, robots, canonical, viewport.
        for marker in [
            "og:",
            "twitter:",
            "\"robots\"",
            "canonical",
            "\"viewport\"",
            "theme-color",
        ] {
            assert!(
                rsc_src.contains(marker),
                "metadata head covers {marker}: missing"
            );
        }
        // module_exports_metadata detects the various export forms.
        assert!(
            !module_exports_metadata(&app.join("Counter.tsx")),
            "a plain island exports no metadata"
        );
    }

    #[test]
    fn metadata_file_conventions_synthesize_route_handlers() {
        let root = scratch("meta-files");
        let app = root.join("app");
        std::fs::create_dir_all(&app).unwrap();
        std::fs::write(
            app.join("layout.tsx"),
            "export default function L({children}){return children}\n",
        )
        .unwrap();
        std::fs::write(
            app.join("page.tsx"),
            "export default function P(){return null}\n",
        )
        .unwrap();
        std::fs::write(
            app.join("sitemap.ts"),
            "export default function sitemap(){ return [{ url: \"https://x.com\" }]; }\n",
        )
        .unwrap();
        std::fs::write(app.join("robots.ts"), "export default function robots(){ return { rules: { userAgent: \"*\", allow: \"/\" } }; }\n").unwrap();
        std::fs::write(
            app.join("manifest.ts"),
            "export default function manifest(){ return { name: \"X\" }; }\n",
        )
        .unwrap();
        let shims = root.join(".diffpack-next/shims");
        std::fs::create_dir_all(&shims).unwrap();

        let handlers = synthesize_metadata_file_handlers(&app, &shims).unwrap();
        let urls: Vec<&str> = handlers.iter().map(|h| h.url_path.as_str()).collect();
        assert!(
            urls.contains(&"/sitemap.xml"),
            "sitemap handler synthesized: {urls:?}"
        );
        assert!(
            urls.contains(&"/robots.txt"),
            "robots handler synthesized: {urls:?}"
        );
        assert!(
            urls.contains(&"/manifest.webmanifest"),
            "manifest handler synthesized: {urls:?}"
        );
        for h in &handlers {
            assert_eq!(
                h.methods,
                vec!["GET".to_string()],
                "convention handlers are GET-only"
            );
            assert!(
                h.file.exists(),
                "the wrapper file was written: {:?}",
                h.file
            );
        }
        // The wrappers set the right content-type + call the user export.
        let sitemap_wrapper = std::fs::read_to_string(shims.join("metadata-sitemap.ts")).unwrap();
        assert!(
            sitemap_wrapper.contains("application/xml"),
            "sitemap wrapper serves XML"
        );
        assert!(
            sitemap_wrapper.contains("serializeSitemap"),
            "sitemap wrapper serializes"
        );
        let robots_wrapper = std::fs::read_to_string(shims.join("metadata-robots.ts")).unwrap();
        assert!(
            robots_wrapper.contains("text/plain"),
            "robots wrapper serves text"
        );
        let manifest_wrapper = std::fs::read_to_string(shims.join("metadata-manifest.ts")).unwrap();
        assert!(
            manifest_wrapper.contains("application/manifest+json"),
            "manifest wrapper serves manifest json"
        );
        assert!(
            manifest_wrapper.contains("JSON.stringify"),
            "manifest wrapper JSON-serializes"
        );
        // The shared serializer helper is present.
        assert!(
            shims.join("metadata-serialize.ts").exists(),
            "serializer helper written"
        );
    }

    #[test]
    fn generate_sitemaps_synthesizes_id_partitioned_route() {
        let root = scratch("meta-gen-sitemaps");
        let app = root.join("app");
        std::fs::create_dir_all(&app).unwrap();
        std::fs::write(
            app.join("sitemap.ts"),
            "export function generateSitemaps(){ return [{ id: 0 }, { id: 1 }]; }\n\
             export default function sitemap({ id }){ return [{ url: `https://x.com/${id}` }]; }\n",
        )
        .unwrap();
        let shims = root.join(".diffpack-next/shims");
        std::fs::create_dir_all(&shims).unwrap();
        let handlers = synthesize_metadata_file_handlers(&app, &shims).unwrap();
        // A single DYNAMIC handler at `/sitemap/[id].xml`, not the plain `/sitemap.xml`.
        let sm: Vec<&RouteHandler> = handlers
            .iter()
            .filter(|h| h.url_path.starts_with("/sitemap"))
            .collect();
        assert_eq!(
            sm.len(),
            1,
            "one id-partitioned sitemap handler: {:?}",
            handlers.iter().map(|h| &h.url_path).collect::<Vec<_>>()
        );
        let h = sm[0];
        assert_eq!(h.url_path, "/sitemap/[id].xml");
        assert_eq!(
            h.segments,
            vec![
                Seg::Static("sitemap".to_string()),
                Seg::Dynamic("id".to_string())
            ],
            "dynamic id segment under /sitemap",
        );
        assert_eq!(h.methods, vec!["GET".to_string()]);
        let wrapper = std::fs::read_to_string(shims.join("metadata-sitemap-id.ts")).unwrap();
        assert!(
            wrapper.contains("generateSitemaps"),
            "wrapper enumerates ids: {wrapper}"
        );
        assert!(
            wrapper.contains("await params"),
            "wrapper reads the id param: {wrapper}"
        );
        assert!(
            wrapper.contains(".xml$"),
            "wrapper strips the .xml suffix: {wrapper}"
        );
        assert!(
            wrapper.contains("status: 404"),
            "unknown id 404s: {wrapper}"
        );
        assert!(
            wrapper.contains("handler({ id: match.id })"),
            "calls sitemap({{id}}): {wrapper}"
        );
    }

    #[test]
    fn page_extensions_honored_or_hard_errored() {
        // Absent pageExtensions: Ok (adapter uses its built-in superset).
        assert!(validate_page_extensions(None).is_ok());
        assert!(validate_page_extensions(Some(&serde_json::json!({}))).is_ok());
        // The @next/mdx-merged default (tsx/ts/jsx/js + md/mdx) is fully supported.
        let ok = serde_json::json!({ "pageExtensions": ["tsx", "ts", "jsx", "js", "md", "mdx"] });
        assert!(validate_page_extensions(Some(&ok)).is_ok());
        // A restricting subset is still supported (superset covers it).
        let subset = serde_json::json!({ "pageExtensions": ["ts", "tsx"] });
        assert!(validate_page_extensions(Some(&subset)).is_ok());
        // An extension diffpack cannot compile is a clear hard error naming it.
        let bad = serde_json::json!({ "pageExtensions": ["tsx", "vue"] });
        let err = validate_page_extensions(Some(&bad)).unwrap_err();
        assert!(
            err.contains("pageExtensions") && err.contains("vue"),
            "names the bad ext: {err}"
        );
    }

    #[test]
    fn static_metadata_images_scanned_and_head_linked() {
        let root = scratch("meta-images");
        let app = root.join("app");
        std::fs::create_dir_all(&app).unwrap();
        std::fs::write(
            app.join("layout.tsx"),
            "export default function L({children}){return children}\n",
        )
        .unwrap();
        std::fs::write(
            app.join("page.tsx"),
            "export default function P(){return null}\n",
        )
        .unwrap();
        std::fs::write(app.join("icon.png"), [0u8]).unwrap();
        std::fs::write(app.join("favicon.ico"), [0u8]).unwrap();
        std::fs::write(app.join("apple-icon.png"), [0u8]).unwrap();
        std::fs::write(app.join("opengraph-image.jpg"), [0u8]).unwrap();
        std::fs::write(app.join("twitter-image.png"), [0u8]).unwrap();

        let images = scan_metadata_images(&app).unwrap();
        let served: Vec<&str> = images.iter().map(|i| i.served.as_str()).collect();
        for want in [
            "/favicon.ico",
            "/icon.png",
            "/apple-icon.png",
            "/opengraph-image.jpg",
            "/twitter-image.png",
        ] {
            assert!(
                served.contains(&want),
                "image {want} discovered: {served:?}"
            );
        }

        // The react-server entry emits the head links for every route.
        let disc = discover_routes(&app, first_existing(&app, "layout").as_deref()).unwrap();
        let boundary = root.join(".diffpack-next/error-boundary.tsx");
        let seg_boundary = root.join(".diffpack-next/segment-boundary.tsx");
        let ctl_boundary = root.join(".diffpack-next/control-boundary.tsx");
        let reqctx = root.join(".diffpack-next/request-context.ts");
        let rsc_src = rsc_entry_module(
            &disc,
            &UnbuiltPatterns::default(),
            &crate::next_font::FontOutput::default(),
            &boundary,
            &seg_boundary,
            &ctl_boundary,
            &reqctx,
            None,
            "",
        );
        assert!(
            rsc_src.contains(r#"rel: "icon", href: "/icon.png""#),
            "icon link emitted: {rsc_src}"
        );
        assert!(
            rsc_src.contains(r#"rel: "apple-touch-icon", href: "/apple-icon.png""#),
            "apple-touch-icon emitted"
        );
        assert!(
            rsc_src.contains(r#"property: "og:image", content: "/opengraph-image.jpg""#),
            "og:image emitted"
        );
        assert!(
            rsc_src.contains(r#"name: "twitter:image", content: "/twitter-image.png""#),
            "twitter:image emitted"
        );
        assert!(
            rsc_src.contains(
                r#"rel: "icon", href: "/favicon.ico", type: "image/x-icon", sizes: "any""#
            ),
            "favicon emitted"
        );
    }

    #[test]
    fn code_based_image_generator_requires_vercel_og() {
        // A generator importing `@vercel/og` whose provider does not resolve → clear
        // hard error naming the file + the missing dependency (no silent drop).
        let root = scratch("meta-image-gen");
        let app = root.join("app");
        std::fs::create_dir_all(&app).unwrap();
        std::fs::write(
            app.join("opengraph-image.tsx"),
            "import { ImageResponse } from \"@vercel/og\";\nexport default function OG(){ return new ImageResponse(null); }\n",
        ).unwrap();
        let err = scan_metadata_images(&app).unwrap_err();
        assert!(
            err.contains("@vercel/og") && err.contains("does not resolve"),
            "names the missing dep: {err}"
        );
        assert!(
            err.contains("opengraph-image.tsx"),
            "error names the file: {err}"
        );

        // A generator that imports neither next/og nor @vercel/og → clear hard error.
        let app2 = scratch("meta-image-gen-noimport").join("app");
        std::fs::create_dir_all(&app2).unwrap();
        std::fs::write(
            app2.join("opengraph-image.tsx"),
            "export default function OG(){ return null; }\n",
        )
        .unwrap();
        let err2 = scan_metadata_images(&app2).unwrap_err();
        assert!(
            err2.contains("next/og") && err2.contains("@vercel/og"),
            "explains the required import: {err2}"
        );

        // With @vercel/og present (a resolvable stub package), the generator is accepted
        // and recorded as a build-time-prerendered PNG head link.
        let ok_app = app.clone();
        let nm = root.join("node_modules").join("@vercel").join("og");
        std::fs::create_dir_all(&nm).unwrap();
        std::fs::write(nm.join("package.json"), "{\"name\":\"@vercel/og\"}\n").unwrap();
        let images = scan_metadata_images(&ok_app).unwrap();
        let og = images
            .iter()
            .find(|i| i.kind == MetaImageKind::OpengraphImage)
            .expect("og image recorded");
        assert!(og.generator, "recorded as a generator");
        assert_eq!(
            og.served, "/opengraph-image.png",
            "served as a prerendered PNG"
        );
        assert_eq!(og.mime, "image/png");
    }

    #[test]
    fn og_image_generator_prerenders_to_png() {
        // Exercises the full build-time @vercel/og prerender ORCHESTRATION: transform the
        // TSX generator to standalone ESM (TS stripped, JSX lowered to react/jsx-runtime,
        // the @vercel/og import preserved), run it under Node against a resolvable
        // ImageResponse provider, and capture the rendered bytes to a PNG. Uses stand-in
        // `@vercel/og` + `react/jsx-runtime` packages (same API shape as the real ones —
        // ImageResponse extends Response) so the pipeline is proven without the heavy dep;
        // satori's rasterization is @vercel/og's own concern, not diffpack's.
        if std::process::Command::new("node")
            .arg("--version")
            .output()
            .map(|o| !o.status.success())
            .unwrap_or(true)
        {
            eprintln!("SKIP og_image_generator_prerenders_to_png: node not found");
            return;
        }
        let root = scratch("og-prerender");
        let app = root.join("app");
        std::fs::create_dir_all(&app).unwrap();
        // Stand-in @vercel/og: ImageResponse extends the Web Response, emitting a minimal
        // valid PNG (signature + IHDR) regardless of the element (the real one rasterizes).
        let og = root.join("node_modules").join("@vercel").join("og");
        std::fs::create_dir_all(&og).unwrap();
        std::fs::write(
            og.join("package.json"),
            "{\"name\":\"@vercel/og\",\"type\":\"module\",\"exports\":{\".\":\"./index.js\"}}\n",
        )
        .unwrap();
        std::fs::write(
            og.join("index.js"),
            "export class ImageResponse extends Response {\n\
             \x20 constructor(element, opts) {\n\
             \x20   const png = Buffer.from('89504e470d0a1a0a0000000d494844520000000100000001', 'hex');\n\
             \x20   super(png, { status: 200, headers: { 'content-type': 'image/png' } });\n\
             \x20 }\n\
             }\n",
        ).unwrap();
        // Stand-in react/jsx-runtime (the automatic JSX runtime the transform targets).
        let react = root.join("node_modules").join("react");
        std::fs::create_dir_all(&react).unwrap();
        std::fs::write(react.join("package.json"), "{\"name\":\"react\",\"type\":\"module\",\"exports\":{\"./jsx-runtime\":\"./jsx-runtime.js\"}}\n").unwrap();
        std::fs::write(
            react.join("jsx-runtime.js"),
            "export function jsx(t,p){return {t,p};}\nexport function jsxs(t,p){return {t,p};}\nexport const Fragment=Symbol('F');\n",
        ).unwrap();
        // The generator: JSX + a preserved @vercel/og import + a default export function.
        let generator = app.join("opengraph-image.tsx");
        std::fs::write(
            &generator,
            "import { ImageResponse } from \"@vercel/og\";\n\
             export const size = { width: 1200, height: 630 };\n\
             export default function OG(): ImageResponse {\n\
             \x20 const label: string = \"Hello edge\";\n\
             \x20 return new ImageResponse(<div style={{ fontSize: 48 }}>{label}</div>);\n\
             }\n",
        )
        .unwrap();

        // scan records it as a generator (the provider resolves).
        let images = scan_metadata_images(&app).unwrap();
        let entry = images
            .iter()
            .find(|i| i.generator)
            .expect("generator recorded");
        assert_eq!(entry.served, "/opengraph-image.png");

        // Prerender it.
        let dest = root.join("out").join("opengraph-image.png");
        prerender_og_image(&root, &generator, &dest).expect("og prerender succeeds");
        let bytes = std::fs::read(&dest).expect("png written");
        assert!(bytes.len() >= 8, "non-empty image emitted");
        assert_eq!(
            &bytes[..8],
            &[0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a],
            "valid PNG signature"
        );
    }

    #[test]
    fn intercepting_routes_target_and_overlay() {
        // The fixture's app/gallery/@modal/(.)photo/[id] intercepts /gallery/photo/[id].
        let fixture =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../integration/next-app-router");
        let app = fixture.join("app");
        let layout = first_existing(&app, "layout");
        let disc = discover_routes(&app, layout.as_deref()).unwrap();

        // The intercept resolves its target RELATIVE to the marker's URL level: `(.)` at
        // gallery/@modal -> /gallery/photo/[id] (not /photo/[id]).
        let ic = disc
            .intercepts
            .iter()
            .find(|i| segments_display(&i.target_segments) == "/gallery/photo/[id]")
            .unwrap_or_else(|| {
                panic!(
                    "gallery intercept not found: {:?}",
                    disc.intercepts
                        .iter()
                        .map(|i| segments_display(&i.target_segments))
                        .collect::<Vec<_>>()
                )
            });
        assert!(
            ic.page.to_string_lossy().contains("@modal"),
            "overlay page is the @modal intercept: {:?}",
            ic.page
        );
        // The full /gallery/photo/[id] route also exists (hard load renders the real page).
        assert!(
            disc.routes
                .iter()
                .any(|r| r.url_path == "/gallery/photo/[id]"),
            "the real photo route exists for hard loads"
        );

        // Codegen: the react-server entry emits INTERCEPTS + softNav-gated matching; the
        // client Router portals the overlay and masks the URL.
        let boundary = fixture.join(".diffpack-next/error-boundary.tsx");
        let seg_boundary = fixture.join(".diffpack-next/segment-boundary.tsx");
        let ctl_boundary = fixture.join(".diffpack-next/control-boundary.tsx");
        let reqctx = fixture.join(".diffpack-next/request-context.ts");
        let rsc_src = rsc_entry_module(
            &disc,
            &UnbuiltPatterns::default(),
            &crate::next_font::FontOutput::default(),
            &boundary,
            &seg_boundary,
            &ctl_boundary,
            &reqctx,
            None,
            "",
        );
        assert!(
            rsc_src.contains("const INTERCEPTS = ["),
            "INTERCEPTS table emitted: {rsc_src}"
        );
        assert!(
            rsc_src.contains("function matchIntercept"),
            "intercept matcher generated"
        );
        assert!(
            rsc_src.contains("opts.softNav"),
            "intercept only on soft-nav"
        );

        let islands = [app.join("Counter.tsx")];
        let hooks = fixture.join(".diffpack-next/hooks-context.ts");
        let client_src = client_entry_module(
            &fixture.join(".diffpack-next"),
            &islands,
            &BTreeSet::new(),
            &hooks,
            PinKind::StaticRequire,
        );
        assert!(
            client_src.contains("x-diffpack-intercept"),
            "client reads the intercept header"
        );
        assert!(
            client_src.contains("createPortal"),
            "client portals the overlay over the page"
        );
        assert!(
            client_src.contains("__diffpackModal"),
            "client masks the URL for the overlay"
        );
    }

    #[test]
    fn nested_intercepting_routes_resolve_target() {
        // A NESTED intercept: a `(.)` marker inside a nested slot, plus a `(...)` root-based
        // marker inside another nested slot. Both must resolve their target relative to the
        // marker's URL level (not the filesystem depth), with the deeper `[id]` recursed in.
        let root = scratch("nested-intercepts");
        let app = root.join("app");
        // The real deep routes (hard loads).
        std::fs::create_dir_all(app.join("photos/[id]")).unwrap();
        std::fs::write(
            app.join("photos/[id]/page.tsx"),
            "export default function P(){return null}\n",
        )
        .unwrap();
        std::fs::create_dir_all(app.join("feed/photo/[id]")).unwrap();
        std::fs::write(
            app.join("feed/photo/[id]/page.tsx"),
            "export default function P(){return null}\n",
        )
        .unwrap();
        // (.) same level as the marker's URL level (feed) -> intercepts /feed/photo/[id].
        std::fs::create_dir_all(app.join("feed/@modal/(.)photo/[id]")).unwrap();
        std::fs::write(
            app.join("feed/@modal/(.)photo/[id]/page.tsx"),
            "export default function M(){return null}\n",
        )
        .unwrap();
        // (...) root-based marker nested under feed/@grid -> intercepts /photos/[id].
        std::fs::create_dir_all(app.join("feed/@grid/(...)photos/[id]")).unwrap();
        std::fs::write(
            app.join("feed/@grid/(...)photos/[id]/page.tsx"),
            "export default function G(){return null}\n",
        )
        .unwrap();

        let intercepts = discover_intercepts(&app).unwrap();
        let targets: Vec<String> = intercepts
            .iter()
            .map(|i| segments_display(&i.target_segments))
            .collect();
        // `(.)` at feed/@modal (same level as feed) -> /feed/photo/[id].
        assert!(
            targets.contains(&"/feed/photo/[id]".to_string()),
            "same-level intercept: {targets:?}"
        );
        // `(...)` -> resolved from the app root regardless of nesting depth.
        assert!(
            targets.contains(&"/photos/[id]".to_string()),
            "root-based intercept: {targets:?}"
        );
        // Each overlay page is the one inside the marker subtree, with its `[id]` recursed.
        for ic in &intercepts {
            assert!(
                matches!(ic.target_segments.last(), Some(Seg::Dynamic(d)) if d == "id"),
                "deep [id] recursed: {:?}",
                ic.target_segments
            );
        }
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn classify_route_precedence() {
        // Unit-level precedence checks independent of the fixture.
        let base = RouteConfig {
            dynamic_params: true,
            ..Default::default()
        };
        // No dynamic segment, no reads → Static.
        assert_eq!(classify_route(false, &base), RouteKind::Static);
        // force-dynamic beats everything.
        let fd = RouteConfig {
            dynamic_config: Some("force-dynamic".into()),
            has_generate_static_params: true,
            ..base.clone()
        };
        assert_eq!(classify_route(true, &fd), RouteKind::Dynamic);
        // revalidate:0 → Dynamic.
        let rz = RouteConfig {
            revalidate: Some("0".into()),
            ..base.clone()
        };
        assert_eq!(classify_route(false, &rz), RouteKind::Dynamic);
        // request-state read → Dynamic even without a dynamic segment.
        let rr = RouteConfig {
            reads_request_state: true,
            ..base.clone()
        };
        assert_eq!(classify_route(false, &rr), RouteKind::Dynamic);
        // request-state read BEATS generateStaticParams even on a dynamic segment
        // (the /blog/[slug] case: cookies + gsp → Dynamic, matching next build's ƒ).
        let rr_gsp = RouteConfig {
            reads_request_state: true,
            has_generate_static_params: true,
            ..base.clone()
        };
        assert_eq!(classify_route(true, &rr_gsp), RouteKind::Dynamic);
        assert!(dynamic_reason(true, &rr_gsp).contains("despite generateStaticParams"));
        // dynamic segment + gsp → Ssg; without gsp → Dynamic.
        let gsp = RouteConfig {
            has_generate_static_params: true,
            ..base.clone()
        };
        assert_eq!(classify_route(true, &gsp), RouteKind::Ssg);
        assert_eq!(classify_route(true, &base), RouteKind::Dynamic);
        // force-static → ForceStatic.
        let fs = RouteConfig {
            dynamic_config: Some("force-static".into()),
            ..base.clone()
        };
        assert_eq!(classify_route(false, &fs), RouteKind::ForceStatic);
    }

    #[test]
    fn extract_export_const_reads_values() {
        assert_eq!(
            extract_export_const("export const dynamic = \"force-dynamic\";", "dynamic").as_deref(),
            Some("force-dynamic")
        );
        assert_eq!(
            extract_export_const("export const dynamicParams = false\n", "dynamicParams")
                .as_deref(),
            Some("false")
        );
        assert_eq!(
            extract_export_const("export const revalidate = 60;", "revalidate").as_deref(),
            Some("60")
        );
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
        std::fs::write(
            app.join("layout.tsx"),
            "export default function L({children}){return children}\n",
        )
        .unwrap();
        std::fs::write(
            app.join("page.tsx"),
            "export default function P(){return null}\n",
        )
        .unwrap();
        std::fs::write(
            app.join("blog").join("[slug]").join("page.tsx"),
            "export default function P(){return null}\n",
        )
        .unwrap();
        std::fs::write(
            app.join("blog").join("new").join("page.tsx"),
            "export default function P(){return null}\n",
        )
        .unwrap();
        let disc = discover_routes(&app, None).unwrap();
        let idx_new = disc
            .routes
            .iter()
            .position(|r| r.url_path == "/blog/new")
            .unwrap();
        let idx_slug = disc
            .routes
            .iter()
            .position(|r| r.url_path == "/blog/[slug]")
            .unwrap();
        assert!(
            idx_new < idx_slug,
            "literal /blog/new must precede /blog/[slug]: {:?}",
            disc.routes.iter().map(|r| &r.url_path).collect::<Vec<_>>()
        );
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

    /// Writes a `package.json` declaring `next` under `field`.
    fn write_next_package_json(root: &Path, field: &str) {
        std::fs::write(
            root.join("package.json"),
            format!("{{\"name\":\"app\",\"{field}\":{{\"next\":\"16.2.11\"}}}}\n"),
        )
        .unwrap();
    }

    /// Writes `<dir>/layout.tsx` + `<dir>/page.tsx`.
    fn write_app_route(dir: &Path) {
        std::fs::create_dir_all(dir).unwrap();
        std::fs::write(
            dir.join("layout.tsx"),
            "export default function L({ children }) { return <html><body>{children}</body></html>; }\n",
        )
        .unwrap();
        std::fs::write(
            dir.join("page.tsx"),
            "export default function P() { return <p>hi</p>; }\n",
        )
        .unwrap();
    }

    /// Writes a minimal `next` install with the vendored polyfill packages the
    /// client fallback maps onto, under `<at>/node_modules/next`.
    fn write_vendored_next(at: &Path) {
        let next = at.join("node_modules").join("next");
        std::fs::create_dir_all(&next).unwrap();
        std::fs::write(next.join("package.json"), "{\"name\":\"next\"}\n").unwrap();
        for vendored in ["path-browserify", "process", "browserify-zlib"] {
            std::fs::create_dir_all(next.join("dist").join("compiled").join(vendored)).unwrap();
        }
    }

    #[test]
    fn the_client_polyfill_table_finds_next_hoisted_to_a_workspace_root() {
        // Every yarn/pnpm/npm workspace puts the app at `apps/web` and hoists `next`
        // to the repository root. Looking only at `<app>/node_modules/next` finds
        // nothing there, and the app silently loses the whole `resolve.fallback`
        // table that the identical STANDALONE app gets — so `import "node:path"`,
        // which `next build` accepts, becomes a fatal browser diagnostic.
        let workspace = scratch("hoisted-next-polyfills");
        write_vendored_next(&workspace);
        let app = workspace.join("apps").join("web");
        std::fs::create_dir_all(&app).unwrap();

        let aliases = next_browser_polyfill_aliases(&app);
        let mapped = |specifier: &str| {
            aliases
                .iter()
                .find(|(from, _)| from == specifier)
                .map(|(_, to)| to.clone())
        };
        let expected = workspace
            .join("node_modules/next/dist/compiled/path-browserify")
            .to_string_lossy()
            .into_owned();
        assert_eq!(mapped("path").as_deref(), Some(expected.as_str()));
        // Both spellings of the same built-in reach the same polyfill.
        assert_eq!(mapped("node:path").as_deref(), Some(expected.as_str()));
        assert!(mapped("process").is_some(), "{aliases:?}");
        assert!(mapped("zlib").is_some(), "{aliases:?}");
        // Built-ins Next does NOT polyfill stay unmapped and keep failing loudly.
        assert!(mapped("fs").is_none(), "{aliases:?}");
        assert!(mapped("child_process").is_none(), "{aliases:?}");
    }

    #[test]
    fn next_runtime_is_defined_per_target_the_way_next_defines_it() {
        // `next/dist/build/define-env`: edge -> 'edge', node server -> 'nodejs',
        // client -> ''. Library code inside `next/dist` branches on this to pick a
        // Node-only implementation (`bloom-filter.js` requires `gzip-size`, and with
        // it fs/stream/zlib, under `=== 'nodejs'`). Leaving it undefined keeps the
        // branch alive and drags that subtree into the CLIENT graph.
        assert_eq!(next_runtime_define(Target::Client), "\"\"");
        assert_eq!(next_runtime_define(Target::Server), "\"nodejs\"");
        assert_eq!(next_runtime_define(Target::IsolatedServer), "\"nodejs\"");
    }

    /// `next/dist/build/define-env.js`: `'process.browser': isClient`. It is the
    /// switch isomorphic library code branches on to reach for Node — `next-i18next`'s
    /// `createConfig` does `if (!process.browser && typeof window === 'undefined') {
    /// var fs = require('fs') }`. Undefined, the test stays undecidable, the branch
    /// survives, and the unfiltered dependency walk puts `fs` in the CLIENT graph,
    /// where Next has no polyfill for it by design.
    #[test]
    fn process_browser_is_defined_per_target_the_way_next_defines_it() {
        assert_eq!(process_browser_define(Target::Client), "true");
        assert_eq!(process_browser_define(Target::Server), "false");
        assert_eq!(process_browser_define(Target::IsolatedServer), "false");
    }

    /// Both Next routers must carry the define, in every environment.
    #[test]
    fn every_next_environment_defines_process_browser() {
        let root = scratch("process-browser-define");
        write_next_package_json(&root, "dependencies");
        write_app_route(&root.join("app"));
        for environment in ["client", "react-server", "ssr"] {
            let config = configure(&root, environment).unwrap().expect("configured");
            let value = config
                .build
                .source_policy
                .defines()
                .iter()
                .find(|(key, _)| key == "process.browser")
                .map(|(_, value)| value.clone())
                .unwrap_or_else(|| panic!("{environment} must define process.browser"));
            let expected = if environment == "client" {
                "true"
            } else {
                "false"
            };
            assert_eq!(value, expected, "{environment}");
        }
        std::fs::remove_dir_all(&root).ok();
    }

    /// The persistent `serve` worker must pick up an edit through the dev server's
    /// `invalidate` op (a micro-chunk import + `serverInvalidate`) and must NEVER go
    /// back to re-importing its own entry under a fresh `?v=<mtime>` URL.
    ///
    /// That was the original mechanism and it cannot be made to work: the entry
    /// reaches its split chunks through `import("./server.chunk-N.mjs")`, a URL with
    /// no query, so Node's ESM cache hands the freshly imported entry the
    /// ALREADY-EVALUATED chunks. On cal.com (69 server chunks) the fresh runtime's id
    /// table and the stale chunks' registrations disagreed and the worker died with
    /// `Module is not loaded: 9947` on the first render after a server-component edit
    /// — a "fix" that looked like freshness and was a crash. The regression this
    /// guards is someone reinstating an mtime poll.
    #[test]
    fn the_serve_worker_hot_updates_through_invalidate_and_never_re_imports_its_entry() {
        let fixture =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../integration/next-app-router");
        let app = fixture.join("app");
        let layout = first_existing(&app, "layout");
        let disc = discover_routes(&app, layout.as_deref()).unwrap();
        let source = rsc_entry_module(
            &disc,
            &UnbuiltPatterns::default(),
            &crate::next_font::FontOutput::default(),
            &fixture.join(".diffpack-next/error-boundary.tsx"),
            &fixture.join(".diffpack-next/segment-boundary.tsx"),
            &fixture.join(".diffpack-next/control-boundary.tsx"),
            &fixture.join(".diffpack-next/request-context.ts"),
            None,
            "",
        );
        assert!(
            source.contains("req.op === \"invalidate\""),
            "serveLoop must handle the dev server's `invalidate` op: {source}"
        );
        assert!(
            source.contains("runtime.serverInvalidate("),
            "an invalidate must drive the live runtime's serverInvalidate: {source}"
        );
        assert!(
            !source.contains("statSync(selfPath).mtimeMs"),
            "serveLoop must not poll its own mtime — an entry re-import cannot bust the chunk cache: {source}"
        );
        assert!(
            !source.contains("?v=\" + mtime"),
            "serveLoop must not re-import its own entry under a fresh query: {source}"
        );
    }

    /// The rsc entry's `main()` must start AT MOST ONCE PER PROCESS, and the guard must
    /// survive a hot re-run of the entry FACTORY — which is exactly what a dev
    /// server-component edit does (`serverInvalidate` re-runs the entry in place).
    ///
    /// The guard used to be `if (!import.meta.url.includes("?v="))`, which only saw the
    /// re-IMPORT case. On a hot re-run that test is true, so `main()` ran again and
    /// `serveLoop()` installed a SECOND `process.stdin` reader: every request line was
    /// then handled twice, the worker rendered the page twice, and both flights went
    /// out under one reply id. The orchestrator concatenated them, and the SSR flight
    /// client died decoding the duplicated row (`chunk.reason.enqueueModel is not a
    /// function`) on a flight exactly twice its correct size.
    #[test]
    fn the_rsc_entry_starts_main_at_most_once_per_process() {
        let fixture =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../integration/next-app-router");
        let app = fixture.join("app");
        let layout = first_existing(&app, "layout");
        let disc = discover_routes(&app, layout.as_deref()).unwrap();
        let source = rsc_entry_module(
            &disc,
            &UnbuiltPatterns::default(),
            &crate::next_font::FontOutput::default(),
            &fixture.join(".diffpack-next/error-boundary.tsx"),
            &fixture.join(".diffpack-next/segment-boundary.tsx"),
            &fixture.join(".diffpack-next/control-boundary.tsx"),
            &fixture.join(".diffpack-next/request-context.ts"),
            None,
            "",
        );
        assert!(
            source.contains("globalThis.__diffpack_rsc_entry_started"),
            "the entry must guard main() with process-level state that survives a factory re-run: {source}"
        );
        assert!(
            !source.contains("if (!import.meta.url.includes(\"?v=\"))"),
            "a URL-shaped guard does not see a hot re-run of the entry factory: {source}"
        );
    }

    #[test]
    fn app_router_detected_without_next_config() {
        // `next.config.*` is OPTIONAL in Next.js: a `next` dependency is enough.
        let root = scratch("no-config-app-router");
        write_next_package_json(&root, "dependencies");
        write_app_route(&root.join("app"));
        assert!(is_app_router(&root));
        assert!(configure(&root, "client").unwrap().is_some());
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn app_router_detected_from_dev_dependency() {
        let root = scratch("dev-dep-app-router");
        write_next_package_json(&root, "devDependencies");
        write_app_route(&root.join("app"));
        assert!(is_app_router(&root));
        std::fs::remove_dir_all(&root).ok();
    }

    /// Plants a project whose `"use client"` components are SIBLINGS of the app dir
    /// (`<base>/components`, `<base>/lib` beside `<base>/app`) — the shape of every
    /// real `with-<state-library>` example — plus one server module that imports a
    /// stylesheet from outside the app dir. `base` is `root` or `root/src`.
    fn write_islands_beside_app(root: &Path, base: &Path) {
        write_next_package_json(root, "dependencies");
        write_app_route(&base.join("app"));
        std::fs::create_dir_all(base.join("components")).unwrap();
        std::fs::create_dir_all(base.join("lib")).unwrap();
        std::fs::write(
            base.join("components").join("counter.tsx"),
            "\"use client\";\nimport \"./counter.css\";\nexport default function Counter() { return null; }\n",
        )
        .unwrap();
        std::fs::write(
            base.join("components").join("counter.css"),
            ".counter { color: red; }\n",
        )
        .unwrap();
        std::fs::write(
            base.join("lib").join("StoreProvider.tsx"),
            "\"use client\";\nexport default function StoreProvider() { return null; }\n",
        )
        .unwrap();
        std::fs::write(
            base.join("lib").join("store.ts"),
            "export const store = 1;\n",
        )
        .unwrap();
    }

    fn canon(path: PathBuf) -> PathBuf {
        path.canonicalize().unwrap()
    }

    #[test]
    fn client_islands_include_components_outside_the_app_dir() {
        // Island discovery is the ONLY thing that puts a `"use client"` module into the
        // client + SSR graphs, so a module it misses gets no client-references-manifest
        // entry and the react-server render dies with `Could not find the module "..."
        // in the React Client Manifest`. Client components living BESIDE the app dir
        // (not under it) are the common case, in both project layouts — asserted on the
        // generated entries, which is what the graphs actually build.
        for (name, sub) in [("islands-src-sibling", "src"), ("islands-root-sibling", "")] {
            let root = scratch(name);
            let base = if sub.is_empty() {
                root.clone()
            } else {
                root.join(sub)
            };
            write_islands_beside_app(&root, &base);
            assert!(configure(&root, "client").unwrap().is_some());

            let adapter_dir = root.canonicalize().unwrap().join(ADAPTER_DIR);
            let client = std::fs::read_to_string(adapter_dir.join("client.tsx")).unwrap();
            for island in ["components/counter.tsx", "lib/StoreProvider.tsx"] {
                let path = canon(base.join(island));
                assert!(
                    client.contains(&path.to_string_lossy().to_string()),
                    "{name}: {island} is not pinned into the client graph"
                );
            }
            assert!(
                !client.contains("lib/store.ts"),
                "{name}: a module without the directive is not an island"
            );
            // The SSR graph must pin the SAME set: it is where the flight's client
            // references are resolved during SSR-of-flight, so an island present in the
            // client graph and absent here has no ssrModuleMapping entry.
            let ssr = std::fs::read_to_string(adapter_dir.join("server.tsx")).unwrap();
            for island in ["components/counter.tsx", "lib/StoreProvider.tsx"] {
                let path = canon(base.join(island));
                assert!(
                    ssr.contains(&path.to_string_lossy().to_string()),
                    "{name}: {island} is not pinned into the SSR graph"
                );
            }

            std::fs::remove_dir_all(&root).ok();
        }
    }

    #[test]
    fn client_islands_skip_dependencies_and_build_output() {
        // Being rooted at the project (not the app dir) means the skip-list is what
        // keeps the walk tractable and keeps installed/generated/stale-build
        // `"use client"` files out of the pins.
        let root = scratch("islands-skips");
        write_next_package_json(&root, "dependencies");
        write_app_route(&root.join("app"));
        let island = "\"use client\";\nexport default function X() { return null; }\n";
        for skipped in [
            "node_modules/zustand",
            ".diffpack-output/public",
            ".diffpack-next/shims",
            "dist",
            ".output",
            ".next/static",
            ".git",
            // Exported/reported trees at the project root. Each holds a COPY of the
            // app's own modules, so a stale one would contribute a duplicate island —
            // a second client-reference id for a component that already has one.
            "out",
            "build",
            ".vercel/output",
            "coverage/lcov-report",
            "storybook-static",
        ] {
            let dir = root.join(skipped);
            std::fs::create_dir_all(&dir).unwrap();
            std::fs::write(dir.join("skipped-island.tsx"), island).unwrap();
        }
        // …but `out`/`build` are ordinary words: a source directory that happens to use
        // one further down the tree is REAL source and must still be scanned.
        let nested = root.join("src").join("build").join("widgets");
        std::fs::create_dir_all(&nested).unwrap();
        std::fs::write(nested.join("nested-island.tsx"), island).unwrap();

        let scan = scan_project(&root.canonicalize().unwrap()).unwrap();
        assert_eq!(
            scan.islands.len(),
            1,
            "only the nested src/build island may be found, got {:?}",
            scan.islands
        );
        assert!(
            scan.islands[0].ends_with("src/build/widgets/nested-island.tsx"),
            "a source directory named `build` below the root is not build output: {:?}",
            scan.islands
        );
        std::fs::remove_dir_all(&root).ok();
    }

    /// A minimal app-router project whose only route is `app/page.tsx` with `page_body`,
    /// plus whatever `extra` files (relative path, contents) the case needs.
    fn write_app_with(root: &Path, page_body: &str, extra: &[(&str, &str)]) {
        write_next_package_json(root, "dependencies");
        write_app_route(&root.join("app"));
        std::fs::write(root.join("app").join("page.tsx"), page_body).unwrap();
        for (relative, contents) in extra {
            let path = root.join(relative);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(&path, contents).unwrap();
        }
    }

    /// Whether the generated react-server entry would link the app stylesheet no matter
    /// what the build emits — the phantom-`<link>`-to-a-404 shape.
    fn links_the_stylesheet_unconditionally(adapter_dir: &Path) -> bool {
        let entry = std::fs::read_to_string(adapter_dir.join("rsc-entry.tsx")).unwrap();
        entry.lines().any(|line| {
            line.contains(RSC_CSS_URL)
                && !line.contains("existsSync(new URL(\"./server.css\", import.meta.url))")
        })
    }

    #[test]
    fn an_app_with_no_stylesheet_does_not_link_one() {
        // REGRESSION (FINDINGS 3a, reproduction 1). The verifier's app: NO stylesheet
        // anywhere, and the only occurrence of the string is a plain script constant.
        // The old substring scan for `.css"` flipped `has_css`, the entry baked in
        // `<link rel="stylesheet" href="/rsc.css">`, and `GET /rsc.css` was a 404
        // because the react-server build never emitted `server.css` to copy.
        let root = scratch("no-stylesheet-app");
        write_app_with(
            &root,
            "import { THEME } from \"../lib/theme\";\nexport default function P() { return <p>{THEME}</p>; }\n",
            &[("lib/theme.ts", "export const THEME = \"theme.css\";\n")],
        );
        assert!(configure(&root, "react-server").unwrap().is_some());
        let adapter_dir = root.canonicalize().unwrap().join(ADAPTER_DIR);
        assert!(
            !links_the_stylesheet_unconditionally(&adapter_dir),
            "an app with no stylesheet must not link one; the head <link> has to follow the emitted artifact",
        );
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn an_unreachable_islands_stylesheet_does_not_link_one() {
        // REGRESSION (FINDINGS 3a, reproduction 2). A `"use client"` component that
        // imports a REAL stylesheet but that no route imports: it is pinned as an
        // island (so the old scan flipped `has_css`) yet it never enters the
        // react-server graph, so its CSS is never compiled into `server.css` — the same
        // `<link>` to the same 404 by a second route.
        let root = scratch("unreachable-island-css");
        write_app_with(
            &root,
            "export default function P() { return <p>hi</p>; }\n",
            &[
                (
                    "components/orphan.tsx",
                    "\"use client\";\nimport \"./unused.css\";\nexport default function O() { return null; }\n",
                ),
                (
                    "components/unused.css",
                    ".orphan { color: rebeccapurple; }\n",
                ),
            ],
        );
        assert!(configure(&root, "react-server").unwrap().is_some());
        let adapter_dir = root.canonicalize().unwrap().join(ADAPTER_DIR);
        assert!(
            !links_the_stylesheet_unconditionally(&adapter_dir),
            "a stylesheet only an unreachable island imports is never compiled, so it must never be linked",
        );
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn dead_code_cannot_fail_the_build_but_live_code_still_does() {
        // REGRESSION (FINDINGS 3a, second consequence). Widening island discovery to the
        // whole project made every `"use client"` file a hard build dependency, so an
        // unreachable one with an unresolvable import failed the WHOLE build. The rule:
        // an island is dropped only when it is BOTH unbuildable AND unreachable —
        // reachable code keeps its fatal diagnostic.
        let orphan = "\"use client\";\nimport { helper } from \"./does-not-exist\";\nexport default function O() { return helper; }\n";

        let dead = scratch("dead-island-unresolvable");
        write_app_with(
            &dead,
            "export default function P() { return <p>hi</p>; }\n",
            &[("components/orphan.tsx", orphan)],
        );
        assert!(configure(&dead, "client").unwrap().is_some());
        let dead_client = std::fs::read_to_string(
            dead.canonicalize()
                .unwrap()
                .join(ADAPTER_DIR)
                .join("client.tsx"),
        )
        .unwrap();
        assert!(
            !dead_client.contains("components/orphan.tsx"),
            "an unbuildable module no route can reach must not become a build dependency: {dead_client}",
        );
        std::fs::remove_dir_all(&dead).ok();

        // The SAME module, imported by the page: still pinned, so the bundler still
        // reports `cannot resolve "./does-not-exist"` and fails the build.
        let live = scratch("live-island-unresolvable");
        write_app_with(
            &live,
            "import O from \"../components/orphan\";\nexport default function P() { return <O />; }\n",
            &[("components/orphan.tsx", orphan)],
        );
        assert!(configure(&live, "client").unwrap().is_some());
        let live_client = std::fs::read_to_string(
            live.canonicalize()
                .unwrap()
                .join(ADAPTER_DIR)
                .join("client.tsx"),
        )
        .unwrap();
        assert!(
            live_client.contains("components/orphan.tsx"),
            "a broken import in code a route reaches stays a hard build error: {live_client}",
        );
        std::fs::remove_dir_all(&live).ok();
    }

    #[test]
    fn dead_code_that_compiles_is_still_pinned() {
        // The over-approximation is what keeps the client manifest complete, so the drop
        // rule must be as narrow as it claims: an unreachable island that BUILDS is
        // still pinned, exactly as before.
        let root = scratch("dead-island-buildable");
        write_app_with(
            &root,
            "export default function P() { return <p>hi</p>; }\n",
            &[(
                "components/orphan.tsx",
                "\"use client\";\nexport default function O() { return null; }\n",
            )],
        );
        assert!(configure(&root, "client").unwrap().is_some());
        let client = std::fs::read_to_string(
            root.canonicalize()
                .unwrap()
                .join(ADAPTER_DIR)
                .join("client.tsx"),
        )
        .unwrap();
        assert!(
            client.contains("components/orphan.tsx"),
            "only UNBUILDABLE dead islands are dropped: {client}",
        );
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn the_island_scan_and_the_action_scan_share_one_walk() {
        // Two scanners over the same directive concept diverging on root or skip-list
        // IS the defect; this pins them to one walk. Both files sit at the same depth,
        // outside the app dir, where the app-rooted island scan used to see neither.
        let root = scratch("islands-actions-one-walk");
        write_next_package_json(&root, "dependencies");
        write_app_route(&root.join("app"));
        std::fs::create_dir_all(root.join("lib")).unwrap();
        std::fs::write(
            root.join("lib").join("island.tsx"),
            "\"use client\";\nexport default function X() { return null; }\n",
        )
        .unwrap();
        std::fs::write(
            root.join("lib").join("actions.ts"),
            "\"use server\";\nexport async function save() {}\n",
        )
        .unwrap();

        let canonical = root.canonicalize().unwrap();
        let islands = scan_project(&canonical).unwrap().islands;
        let actions = crate::rsc::scan_project_server_actions(&canonical).unwrap();
        assert_eq!(islands, vec![canon(root.join("lib").join("island.tsx"))]);
        assert_eq!(
            actions
                .iter()
                .map(|entry| entry.path.clone())
                .collect::<Vec<_>>(),
            vec![canon(root.join("lib").join("actions.ts"))]
        );
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn app_router_detected_under_src_app() {
        let root = scratch("src-app-router");
        write_next_package_json(&root, "dependencies");
        write_app_route(&root.join("src").join("app"));
        assert!(is_app_router(&root));
        assert_eq!(app_dir(&root), Some(root.join("src").join("app")));
        assert!(configure(&root, "client").unwrap().is_some());
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn root_app_wins_over_src_app() {
        // Next's own precedence when a project has both.
        let root = scratch("both-app-dirs");
        write_next_package_json(&root, "dependencies");
        write_app_route(&root.join("app"));
        write_app_route(&root.join("src").join("app"));
        assert_eq!(app_dir(&root), Some(root.join("app")));
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn app_router_detected_with_no_index_page() {
        // Next requires a route SOMEWHERE under app/, not an `app/page.*` specifically.
        let root = scratch("no-index-app-router");
        write_next_package_json(&root, "dependencies");
        write_app_route(&root.join("app").join("[lang]"));
        assert!(is_app_router(&root));
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn next_dep_without_app_dir_is_not_app_router() {
        // A pages-router project: app-router detection must decline so the caller's
        // `!is_app_router && is_pages_router` ordering hands it to the pages adapter.
        let root = scratch("pages-only");
        write_next_package_json(&root, "dependencies");
        std::fs::create_dir_all(root.join("pages")).unwrap();
        std::fs::write(
            root.join("pages/index.tsx"),
            "export default () => <p>hi</p>;\n",
        )
        .unwrap();
        assert!(!is_app_router(&root));
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn non_next_app_directory_is_not_app_router() {
        // A TanStack project whose `routesDirectory` is `app/` must NOT be hijacked.
        let root = scratch("non-next-app-dir");
        std::fs::write(
            root.join("package.json"),
            "{\"name\":\"app\",\"dependencies\":{\"vite\":\"7\"}}\n",
        )
        .unwrap();
        write_app_route(&root.join("app"));
        assert!(!is_app_router(&root));
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn first_page_under_skips_generated_and_vendor_dirs() {
        let root = scratch("first-page-under");
        let app = root.join("app");
        write_app_route(&app.join("node_modules").join("x"));
        write_app_route(&app.join(ADAPTER_DIR));
        write_app_route(&app.join("blog"));
        assert_eq!(
            first_page_under(&app),
            Some(app.join("blog").join("page.tsx"))
        );
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
        let client = configure(&root, "client")
            .unwrap()
            .expect("app-router detected");
        assert_eq!(client.environment, "client");
        assert_eq!(client.build.target, Target::Client);
        let client_entry = client.entry.clone().unwrap();
        assert!(client_entry.ends_with(".diffpack-next/client.tsx"));
        // The generated client entry imports the discovered "use client" island.
        let client_src = std::fs::read_to_string(&client_entry).unwrap();
        assert!(
            client_src.contains("Counter.tsx"),
            "client entry pins the island"
        );
        assert!(
            client_src.contains("hydrateRoot(document"),
            "hydrates the document"
        );
        // Soft navigation (Slice G): the client entry runs a Router that exposes
        // window.__diffpack_navigate and fetches per-route flight via createFromFetch;
        // the `"use client"` next/link shim is pinned into the client graph so its
        // client reference resolves and it hydrates.
        assert!(
            client_src.contains("window.__diffpack_navigate"),
            "client entry installs the soft-nav router: {client_src}"
        );
        assert!(
            client_src.contains("createFromFetch"),
            "client router fetches per-route flight: {client_src}"
        );
        assert!(
            client_src.contains("shims/link.tsx") || client_src.contains("shims\\link.tsx"),
            "the next/link shim is pinned into the client graph: {client_src}"
        );

        // react-server environment: IsolatedServer target + react-server condition.
        let rs = configure(&root, "react-server").unwrap().unwrap();
        assert_eq!(rs.build.target, Target::IsolatedServer);
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
        assert!(
            rs_src.contains("async function documentTree(pathname, opts, control)"),
            "{rs_src}"
        );
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
        assert!(
            ssr_src.contains("export async function renderFlightToStream"),
            "ssr entry exports the streaming renderer: {ssr_src}"
        );
        assert!(
            ssr_src.contains("onShellReady"),
            "streaming SSR flushes at onShellReady: {ssr_src}"
        );
        assert!(
            ssr_src.contains("__DF_FLIGHT"),
            "streaming SSR inlines the flight incrementally: {ssr_src}"
        );
        // The flight scripts may ONLY enter the byte stream at a react-dom flush-cycle
        // boundary: react-dom's own write() boundaries fall every 2048 bytes and land
        // inside HTML tokens, so injecting after an arbitrary chunk splits a tag.
        assert!(
            !ssr_src.contains("for await (const chunk of html)"),
            "renderFlightToStream must not re-read a PassThrough (flight scripts land mid-token): {ssr_src}"
        );
        assert!(
            ssr_src.contains("export function createFlightSink"),
            "the SSR entry carries src/next_runtime/flight_sink.js: {ssr_src}"
        );
        assert!(
            ssr_src.contains("sink.flush = () =>"),
            "flight scripts must be injected from react-dom's flushBuffered() hook: {ssr_src}"
        );
        assert!(
            ssr_src.contains("writev(chunks, cb)"),
            "the SSR destination must forward a corked writev burst without injecting between chunks: {ssr_src}"
        );
        assert!(
            ssr_src.contains("assertFlushHookFired"),
            "a react-dom without flushBuffered must be a hard error, not a silent fallback: {ssr_src}"
        );
        // Exactly TWO node:stream imports, and both belong to a spliced runtime file:
        // the flight sink's `Writable` and the pages-api runtime's `Duplex`. The
        // generated code around them must never add a third of its own — that is the
        // regression this counts.
        assert_eq!(
            ssr_src.matches("from \"node:stream\"").count(),
            2,
            "only the two spliced runtimes import node:stream: {ssr_src}"
        );
        // The client reconstructs the flight from the incremental __DF_FLIGHT stream.
        assert!(
            client_src.contains("flightStreamFromDF"),
            "client rebuilds flight from the __DF_FLIGHT stream: {client_src}"
        );
        // The worker exposes the streaming render op end-to-end.
        assert!(
            rs_src.contains("export async function renderRequestStream"),
            "rsc-entry exports the streaming render: {rs_src}"
        );
        assert!(
            rs_src.contains("render-stream"),
            "the serve worker handles the render-stream op: {rs_src}"
        );
        // The next/link shim is also pinned into the SSR graph so the soft-nav
        // link's client reference resolves during SSR-of-flight (hydration match).
        assert!(
            ssr_src.contains("shims/link.tsx") || ssr_src.contains("shims\\link.tsx"),
            "the next/link shim is pinned into the ssr graph: {ssr_src}"
        );

        // Slice I: per-request context wiring. The rsc-entry establishes the request
        // AsyncLocalStorage (requestAls.run) around the render+drain and captures a
        // server-side redirect() digest via onError.
        assert!(
            rs_src.contains("import { requestAls } from"),
            "rsc-entry imports the request-context ALS: {rs_src}"
        );
        assert!(
            rs_src.contains("requestAls.run(store"),
            "rsc-entry wraps the render in requestAls.run: {rs_src}"
        );
        assert!(
            rs_src.contains("NEXT_REDIRECT;"),
            "rsc-entry captures the redirect digest: {rs_src}"
        );
        // The generated request-context + hooks-context modules exist.
        let adapter = root.join(".diffpack-next");
        let req_ctx = std::fs::read_to_string(adapter.join("request-context.ts")).unwrap();
        assert!(
            req_ctx.contains("AsyncLocalStorage"),
            "request-context holds the ALS: {req_ctx}"
        );
        let hooks_ctx = std::fs::read_to_string(adapter.join("hooks-context.ts")).unwrap();
        assert!(
            hooks_ctx.contains("PathParamsContext"),
            "hooks-context exports PathParamsContext: {hooks_ctx}"
        );
        assert!(
            hooks_ctx.contains("React.createContext ||"),
            "hooks-context guards createContext for the react-server graph: {hooks_ctx}"
        );
        // The next/navigation shim reads the hooks CONTEXTS (not window) and redirect()
        // throws the NEXT_REDIRECT digest on the server.
        let nav = std::fs::read_to_string(adapter.join("shims").join("navigation.ts")).unwrap();
        assert!(
            nav.contains("React.useContext(PathParamsContext)"),
            "useParams reads PathParamsContext: {nav}"
        );
        assert!(
            nav.contains("NEXT_REDIRECT;"),
            "server redirect() throws the redirect digest: {nav}"
        );
        // The next/headers shim reads the real request context (async cookies/headers).
        let hdr = std::fs::read_to_string(adapter.join("shims").join("headers.ts")).unwrap();
        assert!(
            hdr.contains("requestAls.getStore()"),
            "cookies()/headers() read the request ALS: {hdr}"
        );
        assert!(
            hdr.contains("export async function cookies"),
            "cookies() is async (Next 16): {hdr}"
        );
        // Draft mode + server-side cookie writes: cookies() exposes set()/delete() pushing
        // onto the store's response-cookie channel, and draftMode() signs/verifies the real
        // __prerender_bypass cookie against the baked DRAFT_SECRET (never the always-throws
        // "not supported" stub). The request-context module bakes the secret.
        assert!(
            hdr.contains("serializeSetCookie"),
            "headers shim carries a native Set-Cookie serializer: {hdr}"
        );
        assert!(
            hdr.contains("pushResponseCookie(store"),
            "cookies().set()/delete() push onto the response-cookie channel: {hdr}"
        );
        assert!(
            hdr.contains("cookies().set()") && hdr.contains("cookies().delete()"),
            "cookies() gains set()/delete(): {hdr}"
        );
        assert!(
            hdr.contains("__prerender_bypass"),
            "draftMode() uses Next's real bypass cookie name: {hdr}"
        );
        assert!(
            hdr.contains("DRAFT_SECRET") && hdr.contains("createHmac"),
            "draftMode() HMAC-signs the bypass token with the baked secret: {hdr}"
        );
        assert!(
            !hdr.contains("draftMode().enable() is not supported"),
            "draftMode().enable() is implemented, not a throwing stub: {hdr}"
        );
        assert!(
            req_ctx.contains("DRAFT_SECRET"),
            "request-context bakes the draft secret: {req_ctx}"
        );
        // Both the SSR and client entries feed the hooks contexts (Providers wrap the tree).
        assert!(
            ssr_src.contains("PathParamsContext.Provider"),
            "ssr entry provides the hooks contexts: {ssr_src}"
        );
        assert!(
            client_src.contains("PathParamsContext.Provider"),
            "client entry provides the hooks contexts: {client_src}"
        );

        // next/* shims aliased to real generated files.
        let aliased: std::collections::HashMap<_, _> =
            client.build.aliases.iter().cloned().collect();
        for spec in ["next/link", "next/image", "next/navigation", "next/headers"] {
            let target = aliased
                .get(spec)
                .unwrap_or_else(|| panic!("{spec} aliased"));
            assert!(Path::new(target).is_file(), "{spec} shim file exists");
            // The `next` package has no `exports` map, so `next/x` and `next/x.js` are
            // the SAME file. An alias covering only the bare spelling splits the module
            // in two — half diffpack's shim, half Next's real implementation — and the
            // real `useRouter` then throws `invariant expected app router to be mounted`
            // during SSR. `nuqs` imports `"next/navigation.js"` exactly like this.
            let with_extension = aliased
                .get(&format!("{spec}.js"))
                .unwrap_or_else(|| panic!("{spec}.js must alias to the same shim"));
            assert_eq!(
                with_extension, target,
                "{spec}.js and {spec} must be one module"
            );
        }

        // Slice J: next/image is a getImgProps port reading a generated variant
        // manifest; the manifest module is always written (empty map when no public
        // images) so the shim's `../image-manifest` import resolves in every graph.
        let img_shim = std::fs::read_to_string(adapter.join("shims").join("image.tsx")).unwrap();
        assert!(
            img_shim.contains(r#"import MANIFEST from "../image-manifest""#),
            "image shim reads the variant manifest: {img_shim}"
        );
        assert!(
            img_shim.contains("function getWidths"),
            "image shim ports getWidths: {img_shim}"
        );
        assert!(
            img_shim.contains(r#"rel: "preload""#),
            "image shim hoists a priority preload link: {img_shim}"
        );
        // Every optimizable raster renders Next's `/_next/image` optimizer URL; the
        // build-emitted variants are what the orchestrator serves those requests FROM.
        assert!(
            img_shim.contains("function optimizerSrcSet"),
            "image shim builds the optimizer srcset: {img_shim}"
        );
        assert!(
            img_shim.contains(r#"const IMAGE_ENDPOINT = ASSET_BASE + "/_next/image";"#),
            "image shim points at the runtime optimizer endpoint: {img_shim}"
        );
        assert!(
            adapter.join("image-manifest.ts").is_file(),
            "the image variant manifest module is generated"
        );

        // React dev/prod dispatch define is present (keeps React's dev build out).
        assert!(
            client
                .build
                .source_policy
                .defines()
                .iter()
                .any(|(k, v)| k == "process.env.NODE_ENV" && v == "\"production\"")
        );
        assert!(!client.build.hmr, "production config never turns on HMR");
        assert!(
            is_app_router(&root),
            "is_app_router detects the scaffolded project"
        );

        // --- Navigation completeness cluster ----------------------------------------
        // hooks-context exports the two new contexts under the createContext guard.
        assert!(
            hooks_ctx.contains("SelectedSegmentContext = createContext(null)"),
            "hooks-context exports SelectedSegmentContext: {hooks_ctx}"
        );
        assert!(
            hooks_ctx.contains("ServerInsertedHTMLContext = createContext(null)"),
            "hooks-context exports ServerInsertedHTMLContext: {hooks_ctx}"
        );
        // The next/navigation shim exports the segment hooks + useServerInsertedHTML, and
        // useRouter().refresh/prefetch route through the client Router (NOT location.reload).
        assert!(
            nav.contains("export function useSelectedLayoutSegment("),
            "nav shim exports useSelectedLayoutSegment: {nav}"
        );
        assert!(
            nav.contains("export function useSelectedLayoutSegments("),
            "nav shim exports useSelectedLayoutSegments: {nav}"
        );
        assert!(
            nav.contains("export function useServerInsertedHTML("),
            "nav shim exports useServerInsertedHTML: {nav}"
        );
        assert!(
            nav.contains("window.__diffpack_refresh"),
            "useRouter().refresh soft-refreshes via the Router: {nav}"
        );
        assert!(
            !nav.contains("window.location.reload()") || nav.contains("__diffpack_refresh"),
            "refresh prefers soft refresh over reload: {nav}"
        );
        assert!(
            nav.contains("window.__diffpack_prefetch"),
            "useRouter().prefetch warms the prefetch cache: {nav}"
        );
        assert!(
            nav.contains("not supported by this adapter"),
            "a named parallelRouteKey throws a clear error (no silent default): {nav}"
        );
        // EXPORT SURFACE. A named export the shim omits is not a degraded feature — it is
        // a MODULE LOAD FAILURE for anything that imports it ("The requested module
        // \"next/navigation\" does not provide an export named ..."), which took down a
        // whole page render. This list is Next's own, read off
        // `next/dist/client/components/navigation.js`'s export table.
        for name in [
            "ReadonlyURLSearchParams",
            "RedirectType",
            "ServerInsertedHTMLContext",
            "forbidden",
            "notFound",
            "permanentRedirect",
            "redirect",
            "unauthorized",
            "unstable_isUnrecognizedActionError",
            "unstable_rethrow",
            "useParams",
            "usePathname",
            "useRouter",
            "useSearchParams",
            "useSelectedLayoutSegment",
            "useSelectedLayoutSegments",
            "useServerInsertedHTML",
        ] {
            assert!(
                nav.contains(&format!("export function {name}("))
                    || nav.contains(&format!("export class {name} "))
                    || nav.contains(&format!("export const {name} "))
                    || nav.contains(&format!("export {{ {name} }}")),
                "next/navigation must export {name} (a missing name fails the whole module load): {nav}"
            );
        }
        // The hook returns the READ-ONLY subclass, so `instanceof
        // ReadonlyURLSearchParams` holds in user code and a mutator refuses instead of
        // silently editing a copy the router never reads.
        assert!(
            nav.contains("new ReadonlyURLSearchParams(search)"),
            "useSearchParams returns a ReadonlyURLSearchParams: {nav}"
        );
        assert!(
            nav.contains("append() {{ throw new ReadonlyURLSearchParamsError(); }}")
                || nav.contains("append() { throw new ReadonlyURLSearchParamsError(); }"),
            "the read-only search params refuse mutation: {nav}"
        );
        // The rsc-entry wraps each layout in the SEGMENT_BOUNDARY island carrying the
        // active child segments (parts.slice(level.slotBase)).
        assert!(
            rs_src.contains("const SEGMENT_BOUNDARY ="),
            "rsc-entry interns the SEGMENT_BOUNDARY island: {rs_src}"
        );
        assert!(
            rs_src.contains("SEGMENT_BOUNDARY,")
                && rs_src.contains("segments: parts.slice(level.slotBase)"),
            "rsc-entry wraps layouts in SEGMENT_BOUNDARY with the active segments: {rs_src}"
        );
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
        assert!(
            seg_mod.contains("SelectedSegmentContext.Provider"),
            "segment boundary provides SelectedSegmentContext: {seg_mod}"
        );
        assert!(
            seg_mod.starts_with("\"use client\""),
            "segment boundary is a client island: {seg_mod}"
        );
        // The client entry has the bounded prefetch cache + exposes prefetch/refresh.
        assert!(
            client_src.contains("prefetchCache"),
            "client entry has a prefetch cache: {client_src}"
        );
        assert!(
            client_src.contains("window.__diffpack_prefetch"),
            "client entry exposes __diffpack_prefetch: {client_src}"
        );
        assert!(
            client_src.contains("window.__diffpack_refresh"),
            "client entry exposes __diffpack_refresh: {client_src}"
        );
        // The ssr entry provides ServerInsertedHTMLContext + flushes via renderToStaticMarkup.
        assert!(
            ssr_src.contains("renderToStaticMarkup"),
            "ssr entry imports renderToStaticMarkup for inserted HTML: {ssr_src}"
        );
        assert!(
            ssr_src.contains("ServerInsertedHTMLContext.Provider"),
            "ssr entry provides ServerInsertedHTMLContext: {ssr_src}"
        );
        assert!(
            ssr_src.contains("</head>"),
            "ssr buffered path splices inserted HTML before </head>: {ssr_src}"
        );
        // next/dynamic is aliased and its shim exists (React.lazy + ssr:false mounted-gate).
        let dyn_target = aliased.get("next/dynamic").expect("next/dynamic aliased");
        assert!(
            Path::new(dyn_target).is_file(),
            "next/dynamic shim file exists"
        );
        let dyn_shim = std::fs::read_to_string(dyn_target).unwrap();
        assert!(
            dyn_shim.contains("export default function dynamic("),
            "dynamic shim exports dynamic(): {dyn_shim}"
        );
        assert!(
            dyn_shim.contains("opts.ssr === false"),
            "dynamic shim honors ssr:false: {dyn_shim}"
        );
        assert!(
            dyn_shim.contains("lazy(toLoadable"),
            "dynamic shim backs on React.lazy: {dyn_shim}"
        );
        // next/link wires prefetch on hover/focus.
        let link = std::fs::read_to_string(adapter.join("shims").join("link.tsx")).unwrap();
        assert!(
            link.contains("__diffpack_prefetch"),
            "link shim warms the prefetch cache: {link}"
        );
        assert!(
            link.contains("onMouseEnter: handleMouseEnter"),
            "link shim prefetches on hover: {link}"
        );

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
        std::fs::write(
            app.join("layout.tsx"),
            "export default function L({children}){return children}\n",
        )
        .unwrap();
        std::fs::write(
            app.join("page.tsx"),
            "export default function P(){return null}\n",
        )
        .unwrap();

        for environment in ["client", "react-server", "ssr"] {
            let prod = configure(&root, environment).unwrap().unwrap();
            let dev = configure_dev(&root, environment, &RouteScope::All)
                .unwrap()
                .unwrap();
            assert!(!prod.build.hmr, "prod {environment} keeps HMR off");
            assert!(dev.build.hmr, "dev {environment} turns HMR on");
            assert!(
                dev.build
                    .source_policy
                    .defines()
                    .iter()
                    .any(|(k, v)| k == "process.env.NODE_ENV" && v == "\"development\""),
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

    /// Regression: a production build must pin from its OWN project walk and ignore any
    /// [`REFERENCED_ISLANDS_FILE`] a previous `diffpack dev` left in the tree. Reading it
    /// made the build's output depend on whether a dev server had ever run here — it
    /// silently shrank a production pin set to a dev route scope's, which surfaced as a
    /// `next/image` hero rendered with no srcset candidates.
    #[test]
    fn a_production_build_ignores_the_referenced_island_set_a_dev_run_recorded() {
        let root = scratch("app-router-referenced-islands");
        std::fs::write(root.join("next.config.ts"), "export default {}\n").unwrap();
        let app = root.join("app");
        std::fs::create_dir_all(&app).unwrap();
        std::fs::write(
            app.join("layout.tsx"),
            "export default function L({children}){return children}\n",
        )
        .unwrap();
        std::fs::write(
            app.join("page.tsx"),
            "import W from \"./widget\";\nexport default function P(){return <W/>}\n",
        )
        .unwrap();
        // A real island the project walk finds.
        std::fs::write(
            app.join("widget.tsx"),
            "\"use client\";\nexport default function W(){return null}\n",
        )
        .unwrap();

        // Scaffold once so `.diffpack-next/` exists, then plant a recorded set that names
        // NOTHING of the app — the shape a dev run with a narrow route scope leaves behind.
        configure(&root, "client").unwrap().unwrap();
        let adapter_dir = root.canonicalize().unwrap().join(ADAPTER_DIR);
        std::fs::write(
            adapter_dir.join(REFERENCED_ISLANDS_FILE),
            serde_json::to_string(&Vec::<String>::new()).unwrap(),
        )
        .unwrap();

        configure(&root, "client").unwrap().unwrap();
        let production_pins = recorded_islands(&root);
        assert!(
            production_pins
                .iter()
                .any(|pin| pin.ends_with("widget.tsx")),
            "the production build pinned its own walk, not the recorded set: {production_pins:?}",
        );

        // Dev, by contrast, is exactly the consumer of that recorded set: an empty one pins
        // only diffpack's own generated islands (the boundaries + the next/link and
        // next/script shims), never the app's.
        configure_dev(&root, "client", &RouteScope::All)
            .unwrap()
            .unwrap();
        let dev_pins = recorded_islands(&root);
        assert!(
            !dev_pins.iter().any(|pin| pin.ends_with("widget.tsx")),
            "dev honours the recorded referenced set: {dev_pins:?}",
        );
        assert!(
            dev_pins
                .iter()
                .any(|pin| pin.ends_with("error-boundary.tsx")),
            "diffpack's own generated islands are pinned regardless: {dev_pins:?}",
        );
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn image_variant_widths_no_upscale_plus_intrinsic() {
        // Every standard size strictly below the intrinsic width, then the intrinsic
        // itself (full-res always available; no upscaling past it).
        let w = variant_widths(1200);
        assert_eq!(
            *w.last().unwrap(),
            1200,
            "intrinsic width is the max variant"
        );
        assert!(
            w.iter().all(|&x| x <= 1200),
            "no variant exceeds the intrinsic width: {w:?}"
        );
        assert!(
            w.contains(&640) && w.contains(&1080),
            "standard sizes below intrinsic are present: {w:?}"
        );
        assert!(!w.contains(&1920), "no upscaling above intrinsic: {w:?}");
        // A tiny image still yields at least its own intrinsic width.
        assert_eq!(variant_widths(10), vec![10]);
    }

    #[test]
    fn image_variant_url_is_deterministic_and_hashed() {
        let a = image_variant_url("/hero.png", 640, "png");
        assert_eq!(
            a,
            image_variant_url("/hero.png", 640, "png"),
            "deterministic"
        );
        assert!(
            a.starts_with("/_diffpack-image/"),
            "under the variant dir: {a}"
        );
        assert!(a.ends_with("-640.png"), "carries the width + ext: {a}");
        assert_ne!(
            image_variant_url("/hero.png", 640, "png"),
            image_variant_url("/other.png", 640, "png"),
            "distinct srcs hash to distinct variant files"
        );
    }

    /// Regression: the emitted variant FILES and the `variants.json` the orchestrator
    /// reads must describe the same set. The rendered HTML now uses Next's
    /// `/_next/image` URL for every image, so this manifest is the ONLY thing that
    /// keeps a prerendered page off the runtime re-encode path — a drifted or missing
    /// entry silently costs a spawn per image instead of failing loudly.
    #[test]
    fn emitted_image_variants_are_indexed_by_the_manifest_the_server_reads() {
        let root = scratch("image-variant-manifest");
        let public = root.join("public");
        std::fs::create_dir_all(&public).unwrap();
        // A real 900x300 raster, so `variant_widths` plans several standard widths.
        image::RgbImage::from_fn(900, 300, |x, y| {
            image::Rgb([(x % 256) as u8, (y % 256) as u8, 7])
        })
        .save(public.join("hero.png"))
        .unwrap();
        let out_public = root.join("out").join("public");
        let images = scan_public_images(&root).unwrap();
        let written = emit_image_variants(&root, &out_public, &images).unwrap();
        assert!(written >= 2, "several widths emitted: {written}");

        let manifest_path = out_public
            .join("_diffpack-image")
            .join(IMAGE_VARIANT_MANIFEST);
        let manifest: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&manifest_path).unwrap()).unwrap();
        let entry = &manifest["/hero.png"];
        assert_eq!(entry["width"], 900, "intrinsic width recorded: {manifest}");
        let widths = entry["widths"].as_object().unwrap();
        assert_eq!(
            widths.len(),
            written,
            "every emitted variant is indexed: {manifest}"
        );
        assert!(
            widths.contains_key("900"),
            "the intrinsic width is indexed: {manifest}"
        );
        for (w, url) in widths {
            let url = url.as_str().unwrap();
            assert!(url.starts_with("/_diffpack-image/"), "served URL: {url}");
            assert!(
                url.ends_with(&format!("-{w}.png")),
                "width in the file name: {url}"
            );
            assert!(
                out_public.join(url.trim_start_matches('/')).is_file(),
                "manifest entry {url} points at a file that was actually emitted",
            );
        }
        std::fs::remove_dir_all(&root).ok();
    }

    /// Regression: `images.unoptimized` / a custom loader turn diffpack's build-time
    /// image optimization OFF, because under either setting the shim renders a plain
    /// `<img src>` (or an app-loader URL) and nothing can ever request a diffpack
    /// variant. This is the decision, tested against the `images` block alone.
    #[test]
    fn next_config_images_decide_whether_the_build_pre_optimizes() {
        assert_eq!(
            ImageOptimization::from_images(&default_images_json()),
            ImageOptimization::Enabled,
            "Next's stock images config pre-optimizes",
        );
        assert_eq!(
            ImageOptimization::from_images(&serde_json::json!({})),
            ImageOptimization::Enabled,
            "an app that configures no `images` block pre-optimizes",
        );
        // `unoptimized: true` — Next emits a plain <img src>, generating nothing.
        let disabled = ImageOptimization::from_images(&serde_json::json!({ "unoptimized": true }));
        match &disabled {
            ImageOptimization::Disabled(reason) => {
                assert!(
                    reason.contains("unoptimized"),
                    "the reason names the setting: {reason}"
                )
            }
            other => panic!("images.unoptimized must disable pre-optimization, got {other:?}"),
        }
        // `unoptimized: false` is the explicit stock value, not a disable.
        assert_eq!(
            ImageOptimization::from_images(&serde_json::json!({ "unoptimized": false })),
            ImageOptimization::Enabled,
        );
        // A named built-in loader replaces `/_next/image` entirely.
        match ImageOptimization::from_images(&serde_json::json!({ "loader": "cloudinary" })) {
            ImageOptimization::Disabled(reason) => {
                assert!(
                    reason.contains("cloudinary"),
                    "the reason names the loader: {reason}"
                )
            }
            other => panic!("a non-default loader must disable pre-optimization, got {other:?}"),
        }
        assert_eq!(
            ImageOptimization::from_images(&serde_json::json!({ "loader": "default" })),
            ImageOptimization::Enabled,
        );
        // A `loaderFile` does the same; a null/empty one does not.
        match ImageOptimization::from_images(&serde_json::json!({ "loaderFile": "./loader.js" })) {
            ImageOptimization::Disabled(reason) => {
                assert!(
                    reason.contains("loader.js"),
                    "the reason names the file: {reason}"
                )
            }
            other => panic!("a loaderFile must disable pre-optimization, got {other:?}"),
        }
        assert_eq!(
            ImageOptimization::from_images(&serde_json::json!({ "loaderFile": null })),
            ImageOptimization::Enabled,
        );
    }

    /// Regression for the dominant cost of an image-heavy production build: with
    /// optimization off, `public/` is neither decoded nor re-encoded and NOT ONE variant
    /// file is written — while every image still gets a manifest entry, so the shim's
    /// "no entry for this src" hard error cannot start firing.
    #[test]
    fn optimization_off_emits_no_variants_but_still_registers_every_public_image() {
        let root = scratch("image-unoptimized");
        let public = root.join("public");
        std::fs::create_dir_all(&public).unwrap();
        image::RgbImage::from_fn(900, 300, |x, y| {
            image::Rgb([(x % 256) as u8, (y % 256) as u8, 7])
        })
        .save(public.join("hero.png"))
        .unwrap();
        std::fs::write(public.join("logo.svg"), "<svg/>").unwrap();

        // Optimization ON is the control: the ladder is emitted as before.
        let on = scan_public_images_with(&root, &ImageOptimization::Enabled).unwrap();
        let on_written =
            emit_image_variants(&root, &root.join("out-on").join("public"), &on).unwrap();
        assert!(
            on_written >= 2,
            "the control build emits a ladder: {on_written}"
        );

        let off = scan_public_images_with(
            &root,
            &ImageOptimization::Disabled("images.unoptimized is true".to_string()),
        )
        .unwrap();
        assert_eq!(
            off.len(),
            on.len(),
            "every public image is still registered"
        );
        assert!(
            off.iter().all(|PublicImage(entry)| entry.unoptimized),
            "with optimization off every entry is a passthrough",
        );
        // The manifest the shim imports still names both srcs (no entry = hard error there).
        let module = image_manifest_module(&off);
        assert!(
            module.contains("\"/hero.png\""),
            "the raster is registered: {module}"
        );
        assert!(
            module.contains("\"/logo.svg\""),
            "the svg is registered: {module}"
        );

        let out_public = root.join("out-off").join("public");
        let written = emit_image_variants(&root, &out_public, &off).unwrap();
        assert_eq!(written, 0, "no variant file is written");
        let variant_dir = out_public.join("_diffpack-image");
        let files = std::fs::read_dir(&variant_dir)
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().to_string())
            .collect::<Vec<_>>();
        assert_eq!(
            files,
            vec![IMAGE_VARIANT_MANIFEST.to_string()],
            "only the (empty) manifest is written, so a missing file still means \
             `this build emitted no variants` rather than `the step did not run`",
        );
        let manifest: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(variant_dir.join(IMAGE_VARIANT_MANIFEST)).unwrap(),
        )
        .unwrap();
        assert_eq!(
            manifest,
            serde_json::json!({}),
            "the variant manifest is empty"
        );
        std::fs::remove_dir_all(&root).ok();
    }

    /// The adapter persists the resolved `images` block next to the generated
    /// `image-config.ts`, so the build steps that run after `configure` (the variant
    /// emit in `main.rs`, the dev server's) read the app's real setting instead of
    /// re-spawning node — and read the SAME value the bundled shim was generated from.
    #[test]
    fn configure_persists_the_images_block_for_the_later_build_steps() {
        let root = scratch("image-config-json");
        std::fs::write(
            root.join("next.config.ts"),
            "export default { images: { unoptimized: true } }\n",
        )
        .unwrap();
        let app = root.join("app");
        std::fs::create_dir_all(&app).unwrap();
        std::fs::write(
            app.join("layout.tsx"),
            "export default function L({children}){return children}\n",
        )
        .unwrap();
        std::fs::write(
            app.join("page.tsx"),
            "export default function P(){return null}\n",
        )
        .unwrap();
        // Without node the config cannot be evaluated at all; the decision then
        // defaults to Enabled, which this test is not about.
        if std::process::Command::new("node")
            .arg("--version")
            .output()
            .is_err()
        {
            std::fs::remove_dir_all(&root).ok();
            return;
        }
        configure(&root, "client").unwrap().unwrap();
        let persisted = root.join(ADAPTER_DIR).join(IMAGE_CONFIG_JSON);
        assert!(
            persisted.is_file(),
            "the resolved images block is persisted at {}",
            persisted.display()
        );
        let images: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&persisted).unwrap()).unwrap();
        assert_eq!(images["unoptimized"], serde_json::Value::Bool(true));
        assert_eq!(
            ImageOptimization::for_project(&root),
            ImageOptimization::from_images(&images),
            "the persisted file is what the later steps read",
        );
        // The generated shim config module agrees — one resolved block, two artifacts.
        let module =
            std::fs::read_to_string(root.join(ADAPTER_DIR).join("image-config.ts")).unwrap();
        assert!(
            module.contains("unoptimized: true"),
            "the shim config agrees: {module}"
        );
        std::fs::remove_dir_all(&root).ok();
    }

    /// A static asset must carry validators, so a repeat navigation revalidates into a
    /// bodiless `304` instead of re-downloading the whole client bundle, and must be
    /// gzipped when it is compressible and the client asked — the contract `next start`
    /// gives a `public/` file (`cache-control: public, max-age=0` + `ETag` +
    /// `Last-Modified`, gzip over 1 KB). Runs the orchestrator's REAL
    /// `serveStaticAsset` (sliced out of `next-server.mjs`) against a stub filesystem.
    #[test]
    fn next_server_serves_a_static_asset_with_validators_and_gzip() {
        if std::process::Command::new("node")
            .arg("--version")
            .output()
            .is_err()
        {
            return;
        }
        const SERVER: &str = include_str!("../../../scripts/rsc/next-server.mjs");
        let slice = |from: &str, to: &str| -> String {
            let start = SERVER
                .find(from)
                .unwrap_or_else(|| panic!("next-server.mjs still contains {from:?}"));
            let end = SERVER[start..]
                .find(to)
                .unwrap_or_else(|| panic!("next-server.mjs still contains {to:?} after {from:?}"))
                + start;
            SERVER[start..end].to_string()
        };
        let policy = slice("const STATIC_CACHE_CONTROL", "// Append Set-Cookie strings");
        let serve = slice(
            "function serveStaticAsset(",
            "// Which KIND of endpoint owns this path",
        );
        assert!(
            policy.contains("function acceptsGzip"),
            "sliced the policy block: {policy}"
        );

        let prelude = r#"import { createGzip, gunzipSync } from "node:zlib";
import { pipeline } from "node:stream";
import { Readable, Writable } from "node:stream";
import { extname } from "node:path";
const MIME = { ".js": "text/javascript", ".png": "image/png" };
const BODIES = {
  "/out/public/client.js": "console.log(1);".repeat(1000),
  "/out/public/tiny.js": "x",
  "/out/public/logo.png": "P".repeat(5000),
};
const MTIME = 1700000000123;
function statSync(p) {
  if (!(p in BODIES)) throw new Error("ENOENT " + p);
  return { size: Buffer.byteLength(BODIES[p]), mtimeMs: MTIME, isFile: () => true };
}
function createReadStream(p) { return Readable.from([Buffer.from(BODIES[p])]); }
function makeRes() {
  const chunks = [];
  const res = new Writable({ write(c, e, cb) { chunks.push(Buffer.from(c)); cb(); } });
  res.writeHead = (status, headers) => { res.status = status; res.headers = headers || {}; return res; };
  res.done = new Promise((resolve) => res.on("finish", resolve));
  res.body = () => Buffer.concat(chunks);
  return res;
}
async function run(file, headers) {
  const res = makeRes();
  serveStaticAsset({ headers }, res, file);
  await res.done;
  const h = res.headers;
  const raw = res.body();
  return {
    status: res.status,
    cacheControl: h["cache-control"] ?? null,
    etag: h.etag ?? null,
    lastModified: h["last-modified"] ?? null,
    encoding: h["content-encoding"] ?? null,
    vary: h.vary ?? null,
    contentLength: h["content-length"] ?? null,
    bodyBytes: raw.length,
    text: h["content-encoding"] === "gzip" ? gunzipSync(raw).toString() : raw.toString(),
  };
}
"#;
        let driver = r#"
const gz = await run("/out/public/client.js", { "accept-encoding": "gzip, deflate, br" });
// The SECOND request for the same asset must be answered from the kept frame — same
// bytes, and now with a declared length because the frame's size is known.
const gz2 = await run("/out/public/client.js", { "accept-encoding": "gzip" });
const plain = await run("/out/public/client.js", {});
const fresh = await run("/out/public/client.js", { "if-none-match": gz.etag });
const stale = await run("/out/public/client.js", { "if-none-match": 'W/"deadbeef-1"' });
const tiny = await run("/out/public/tiny.js", { "accept-encoding": "gzip" });
const png = await run("/out/public/logo.png", { "accept-encoding": "gzip" });
console.log(JSON.stringify({ gz, gz2, plain, fresh, stale, tiny, png }));
"#;
        let file = scratch("next-server-static-asset").join("serve.mjs");
        std::fs::write(&file, format!("{prelude}{policy}{serve}{driver}")).unwrap();
        let out = std::process::Command::new("node")
            .arg(&file)
            .output()
            .unwrap();
        assert!(
            out.status.success(),
            "static-asset serve failed: {}",
            String::from_utf8_lossy(&out.stderr),
        );
        let got: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();

        // Compressible + accepted -> gzip, and the bytes still decode to the file.
        assert_eq!(got["gz"]["status"], 200);
        assert_eq!(got["gz"]["encoding"], "gzip");
        assert_eq!(got["gz"]["vary"], "Accept-Encoding");
        assert_eq!(got["gz"]["cacheControl"], "public, max-age=0");
        assert_eq!(got["gz"]["text"], "console.log(1);".repeat(1000));
        assert!(
            got["gz"]["bodyBytes"].as_u64().unwrap() < got["plain"]["bodyBytes"].as_u64().unwrap(),
            "gzip is smaller than identity: {got}",
        );
        // Compressed at most once: the second request replays the kept frame verbatim,
        // which is the whole reason gzip is affordable on a single-event-loop server.
        assert_eq!(got["gz2"]["bodyBytes"], got["gz"]["bodyBytes"]);
        assert_eq!(got["gz2"]["text"], got["gz"]["text"]);
        assert_eq!(
            got["gz2"]["contentLength"], got["gz"]["bodyBytes"],
            "the replayed frame declares its length: {got}",
        );
        // A validator is always present, and the identity response declares its length.
        assert_eq!(got["plain"]["encoding"], serde_json::Value::Null);
        assert_eq!(got["plain"]["contentLength"], 15000);
        assert_eq!(got["plain"]["etag"], got["gz"]["etag"]);
        assert_eq!(got["plain"]["etag"], "W/\"3a98-18bcfe5687b\"");
        assert_eq!(
            got["plain"]["lastModified"],
            "Tue, 14 Nov 2023 22:13:20 GMT"
        );
        // The whole point: a matching validator costs no body at all.
        assert_eq!(got["fresh"]["status"], 304);
        assert_eq!(got["fresh"]["bodyBytes"], 0);
        assert_eq!(got["fresh"]["etag"], got["gz"]["etag"]);
        assert_eq!(
            got["stale"]["status"], 200,
            "a stale validator still gets the body: {got}"
        );
        // Below the threshold, and an already-compressed type, are sent as-is.
        assert_eq!(got["tiny"]["encoding"], serde_json::Value::Null);
        assert_eq!(got["tiny"]["vary"], serde_json::Value::Null);
        assert_eq!(got["png"]["encoding"], serde_json::Value::Null);
        assert_eq!(got["png"]["contentLength"], 5000);
        std::fs::remove_file(&file).ok();
    }

    /// Regression: `/_next/image?url=…&w=…` must be answered from a build-emitted
    /// variant whenever one exists — that is what keeps Next's URL shape from costing a
    /// runtime re-encode on every prerendered page. Runs the orchestrator's real
    /// `buildVariantFile` (sliced out of `next-server.mjs`, not a reimplementation)
    /// against a stub filesystem.
    #[test]
    fn next_server_answers_image_requests_from_build_emitted_variants() {
        if std::process::Command::new("node")
            .arg("--version")
            .output()
            .is_err()
        {
            return;
        }
        const SERVER: &str = include_str!("../../../scripts/rsc/next-server.mjs");
        let start = SERVER
            .find("const IMAGE_BUILD_QUALITY = 75;")
            .expect("next-server.mjs still defines the build-variant fast path");
        let end = SERVER[start..]
            .find("// Run the native optimizer")
            .expect("the build-variant block still ends before the native optimizer")
            + start;
        let region = &SERVER[start..end];
        assert!(
            region.contains("function buildVariantFile"),
            "sliced the right block: {region}"
        );

        let prelude = format!(
            r##"const publicDir = "/out/public";
const FILES = new Set([
  "/out/public/_diffpack-image/abc-640.jpeg",
  "/out/public/_diffpack-image/abc-1600.jpeg",
  "/out/public/assets/vercel-a1cdda59-1080.png",
  "/out/public/assets/vercel-a1cdda59-1600.png",
  "/out/public/hero-640-640.png",
  "/out/public/_diffpack-image/{manifest}",
]);
// What `readdirSync(publicDir + "/assets")` sees: the original plus its ladder, exactly
// as the client build emits them (the ladder tops out at the 1600px intrinsic width).
const ASSETS_DIR = ["vercel-a1cdda59.png", "vercel-a1cdda59-1080.png", "vercel-a1cdda59-1600.png"];
const MANIFEST = {{ "/cat.jpg": {{ width: 1600, widths: {{ "640": "/_diffpack-image/abc-640.jpeg", "1600": "/_diffpack-image/abc-1600.jpeg" }} }} }};
function join(...parts) {{ return parts.join("/").replace(/\/+/g, "/"); }}
function existsSync(p) {{ return FILES.has(p) || p === "/out/public/assets"; }}
function readFileSync() {{ return JSON.stringify(MANIFEST); }}
function readdirSync() {{ return ASSETS_DIR; }}
"##,
            manifest = IMAGE_VARIANT_MANIFEST,
        );
        let driver = r#"
const CASES = [
  ["/cat.jpg", 640, 75],
  ["/cat.jpg", 3840, 75],
  ["/cat.jpg", 640, 50],
  ["/assets/vercel-a1cdda59.png", 1080, 75],
  ["/assets/vercel-a1cdda59.png", 2048, 75],
  ["/assets/vercel-a1cdda59.png", 640, 75],
  ["/hero-640.png", 640, 75],
];
console.log(JSON.stringify(CASES.map(([s, w, q]) => buildVariantFile(s, w, q))));
"#;
        let file = scratch("image-build-variant-lookup").join("lookup.mjs");
        std::fs::write(&file, format!("{prelude}{region}{driver}")).unwrap();
        let out = std::process::Command::new("node")
            .arg(&file)
            .output()
            .unwrap();
        assert!(
            out.status.success(),
            "build-variant lookup failed: {}",
            String::from_utf8_lossy(&out.stderr),
        );
        let got: Vec<Option<String>> = serde_json::from_slice(&out.stdout).unwrap();
        assert_eq!(
            got[0].as_deref(),
            Some("/out/public/_diffpack-image/abc-640.jpeg"),
            "an exact indexed width is served from disk: {got:?}",
        );
        assert_eq!(
            got[1].as_deref(),
            Some("/out/public/_diffpack-image/abc-1600.jpeg"),
            "a width at/above intrinsic resolves to the intrinsic variant (the optimizer never upscales): {got:?}",
        );
        assert_eq!(
            got[2], None,
            "a non-default quality gets a real re-encode: {got:?}"
        );
        assert_eq!(
            got[3].as_deref(),
            Some("/out/public/assets/vercel-a1cdda59-1080.png"),
            "a static-import variant is found by its build naming convention: {got:?}",
        );
        assert_eq!(
            got[4].as_deref(),
            Some("/out/public/assets/vercel-a1cdda59-1600.png"),
            "a width above a static import's intrinsic resolves to the top of its ladder, so a prerendered page never re-encodes: {got:?}",
        );
        assert_eq!(
            got[5], None,
            "a width the ladder does not contain falls through: {got:?}"
        );
        assert_eq!(
            got[6], None,
            "a /public file literally named `hero-640.png` is NEVER mistaken for a variant of `hero.png`: {got:?}",
        );
        std::fs::remove_dir_all(file.parent().unwrap()).ok();
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
                blur_data_url: Some("data:image/png;base64,AAAA".into()),
            }),
            PublicImage(ImageEntry {
                src: "/next.svg".into(),
                rel: PathBuf::from("next.svg"),
                ext: "svg".into(),
                unoptimized: true,
                width: 0,
                height: 0,
                variants: Vec::new(),
                blur_data_url: None,
            }),
        ];
        let module = image_manifest_module(&images);
        assert!(module.contains("export default {"), "{module}");
        assert!(
            module.contains(r#""/hero.png": { width: 1200, height: 300, variants: {"#),
            "optimized raster entry: {module}"
        );
        assert!(
            module.contains(r#""640": "/_diffpack-image/"#),
            "variant keyed by width: {module}"
        );
        assert!(
            module.contains(r#"blurDataURL: "data:image/png;base64,AAAA""#),
            "blurDataURL serialized on optimized entry: {module}"
        );
        assert!(
            module.contains(r#""/next.svg": { unoptimized: true }"#),
            "svg is unoptimized (no srcset): {module}"
        );
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
        assert!(
            module.contains(r#"import __loaderFile from "/abs/loader.js";"#),
            "loaderFile imported: {module}"
        );
        assert!(
            module.contains(r#""hostname":"**.example.com""#),
            "remotePatterns serialized: {module}"
        );
        assert!(
            module.contains(r#""cdn.example.org""#),
            "domains serialized: {module}"
        );
        assert!(
            module.contains(r#"loader: "imgix""#),
            "named loader: {module}"
        );
        assert!(
            module.contains("loaderFn: __loaderFile"),
            "loaderFn wired to the imported file: {module}"
        );
    }

    #[test]
    fn image_config_defaults_when_no_loader_file() {
        let module = image_config_module(&default_images_json());
        assert!(
            !module.contains("import __loaderFile"),
            "no loaderFile import when none set: {module}"
        );
        assert!(
            module.contains("loaderFn: null"),
            "loaderFn null by default: {module}"
        );
        assert!(
            module.contains("remotePatterns: []"),
            "empty remote allow-list by default: {module}"
        );
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
        assert_eq!(
            r.asset_base(),
            "/cdn/docs",
            "assets sit under assetPrefix then basePath"
        );

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
        assert!(
            shim.starts_with("\"use client\";"),
            "use client stays first: {shim}"
        );
        assert!(
            shim.contains(r#"const BASE_PATH = "/docs";"#),
            "basePath baked as a const: {shim}"
        );
        assert!(
            shim.contains("function withBasePath"),
            "the prefix helper is generated"
        );
        assert!(
            shim.contains("const resolved = withBasePath(rawHref);"),
            "href routed through withBasePath"
        );
        assert!(
            shim.contains("href.startsWith(BASE_PATH + \"/\")"),
            "no double-prefix guard present"
        );

        // No basePath: the const is empty, so withBasePath is an identity (href unchanged).
        let plain = next_link_shim("");
        assert!(
            plain.contains(r#"const BASE_PATH = "";"#),
            "empty basePath const: {plain}"
        );
    }

    /// The browser bootstrap AND this route's client-reference chunks are fetched under the
    /// app's asset base, on both render paths.
    ///
    /// The entry is emitted by the document rather than by react-dom's `bootstrapModules`,
    /// which stamps `async` on its tag: an async module script is unordered against the
    /// chunk scripts, so a chunk could execute before the entry that creates the runtime it
    /// registers into and throw. So `bootstrapModules` must NOT come back, and the test says
    /// so — with all three theme tests in cal.com's suite as the evidence for why.
    #[test]
    fn ssr_entry_bakes_asset_base_into_the_module_scripts_it_emits() {
        let dir = scratch("ssr-asset-base");
        let hooks = dir.join("hooks-context.ts");
        let with_prefix = ssr_entry_module(&dir, &[], &BTreeSet::new(), &hooks, "/cdn/docs", &[]);
        assert_eq!(
            with_prefix
                .matches(r#"moduleTag("/cdn/docs/client.js")"#)
                .count(),
            2,
            "the entry tag carries the asset base on BOTH render paths: {with_prefix}",
        );
        assert!(
            with_prefix.contains(r#"const base = "/cdn/docs";"#),
            "chunk URLs resolve against the same base: {with_prefix}",
        );
        assert!(
            !with_prefix.contains("bootstrapModules:"),
            "the entry must not go back to react-dom's async bootstrap tag: {with_prefix}",
        );
        // Empty asset base keeps the bare `/client.js`.
        let plain = ssr_entry_module(&dir, &[], &BTreeSet::new(), &hooks, "", &[]);
        assert_eq!(
            plain.matches(r#"moduleTag("/client.js")"#).count(),
            2,
            "no prefix -> bare client.js: {plain}",
        );
    }

    #[test]
    fn rsc_entry_prefixes_stylesheet_href() {
        let fixture =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../integration/next-app-router");
        let app = fixture.join("app");
        let layout = first_existing(&app, "layout");
        let disc = discover_routes(&app, layout.as_deref()).unwrap();
        let boundary = fixture.join(".diffpack-next/error-boundary.tsx");
        let seg_boundary = fixture.join(".diffpack-next/segment-boundary.tsx");
        let ctl_boundary = fixture.join(".diffpack-next/control-boundary.tsx");
        let reqctx = fixture.join(".diffpack-next/request-context.ts");
        // With an asset base the stylesheet href carries the prefix — and it is still
        // only linked when the react-server build emitted a stylesheet beside the entry.
        let src = rsc_entry_module(
            &disc,
            &UnbuiltPatterns::default(),
            &crate::next_font::FontOutput::default(),
            &boundary,
            &seg_boundary,
            &ctl_boundary,
            &reqctx,
            None,
            "/docs",
        );
        assert!(
            src.contains(
                r#"const RSC_CSS_HREF = existsSync(new URL("./server.css", import.meta.url)) ? "/docs/rsc.css" : null;"#
            ),
            "stylesheet href prefixed by basePath, guarded by the emitted server.css: {src}",
        );
        // The <link> is pushed from that const, never unconditionally.
        assert!(
            src.contains(
                r#"  if (RSC_CSS_HREF) items.push(createElement("link", { rel: "stylesheet", href: RSC_CSS_HREF, precedence: "low" }));"#
            ),
            "the head <link> is conditional on the emitted stylesheet: {src}",
        );
        // Empty asset base keeps the bare `/rsc.css`.
        let plain = rsc_entry_module(
            &disc,
            &UnbuiltPatterns::default(),
            &crate::next_font::FontOutput::default(),
            &boundary,
            &seg_boundary,
            &ctl_boundary,
            &reqctx,
            None,
            "",
        );
        assert!(
            plain.contains(r#"? "/rsc.css" : null;"#),
            "no prefix -> bare /rsc.css: {plain}"
        );
    }

    /// REGRESSION. The react-server graph replaces a `"use client"` module with a proxy
    /// that keeps only that module's OWN direct stylesheet imports, so CSS the island
    /// reaches only THROUGH another module (cal.com: a `"use client"` wrapper -> plain
    /// `Editor.tsx` -> `import "./stylesEditor.css"`) is in NO stylesheet the document
    /// linked. It was compiled — into the client graph's `public/client.css` — and
    /// served, and nothing referenced it: the Lexical editor rendered 25px shorter than
    /// the reference build's, which moved the whole page and broke unrelated
    /// interactions. The document must link that stylesheet too, last, and under the
    /// same emitted-artifact guard so the href can never 404.
    #[test]
    fn rsc_entry_links_the_client_graphs_stylesheet_last() {
        let fixture =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../integration/next-app-router");
        let app = fixture.join("app");
        let layout = first_existing(&app, "layout");
        let disc = discover_routes(&app, layout.as_deref()).unwrap();
        let boundary = fixture.join(".diffpack-next/error-boundary.tsx");
        let seg_boundary = fixture.join(".diffpack-next/segment-boundary.tsx");
        let ctl_boundary = fixture.join(".diffpack-next/control-boundary.tsx");
        let reqctx = fixture.join(".diffpack-next/request-context.ts");
        let src = rsc_entry_module(
            &disc,
            &UnbuiltPatterns::default(),
            &crate::next_font::FontOutput::default(),
            &boundary,
            &seg_boundary,
            &ctl_boundary,
            &reqctx,
            None,
            "",
        );
        assert!(
            src.contains(&format!(
                r#"const CLIENT_CSS_HREF = existsSync(new URL("{CLIENT_EMITTED_CSS_PATH}", import.meta.url)) ? "{CLIENT_CSS_URL}" : null;"#
            )),
            "the client stylesheet is guarded by ITS emitted artifact in the served public dir: {src}",
        );
        // Exactly one place decides it, and the URL never appears outside that guard.
        for line in src.lines() {
            if !line.contains(CLIENT_CSS_URL) {
                continue;
            }
            assert!(
                line.contains(&format!(
                    "existsSync(new URL(\"{CLIENT_EMITTED_CSS_PATH}\", import.meta.url))"
                )),
                "the client stylesheet URL appears outside the emitted-artifact guard: {line}",
            );
        }
        // Pushed AFTER the app stylesheet, so React's precedence groups order the two
        // links `low` then `client` — client-component CSS wins ties, as in Next.
        let rsc_at = src
            .find("items.push(createElement(\"link\", { rel: \"stylesheet\", href: RSC_CSS_HREF")
            .expect("rsc link push");
        let client_at = src
            .find("items.push(createElement(\"link\", { rel: \"stylesheet\", href: CLIENT_CSS_HREF")
            .expect("client link push");
        assert!(
            client_at > rsc_at,
            "the client stylesheet must be linked after the app stylesheet"
        );
        // An asset base prefixes it exactly like every other static URL.
        let prefixed = rsc_entry_module(
            &disc,
            &UnbuiltPatterns::default(),
            &crate::next_font::FontOutput::default(),
            &boundary,
            &seg_boundary,
            &ctl_boundary,
            &reqctx,
            None,
            "/docs",
        );
        assert!(
            prefixed.contains(r#"? "/docs/client.css" : null;"#),
            "the client stylesheet href carries the asset base: {prefixed}",
        );
    }

    #[test]
    fn the_stylesheet_link_is_never_emitted_unconditionally() {
        // REGRESSION (FINDINGS 3a). The <link rel=stylesheet href=/rsc.css> used to be
        // baked in from a SUBSTRING scan for `.css"` over every project source, which
        // has no relationship to what the react-server build compiles: an app with no
        // stylesheet at all whose only `.css` is inside a string literal
        // (`export const THEME = "theme.css";`) served a document linking /rsc.css while
        // GET /rsc.css returned 404. The href may only appear inside the guard that
        // tests for the emitted artifact.
        let fixture =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../integration/next-app-router");
        let app = fixture.join("app");
        let layout = first_existing(&app, "layout");
        let disc = discover_routes(&app, layout.as_deref()).unwrap();
        let boundary = fixture.join(".diffpack-next/error-boundary.tsx");
        let seg_boundary = fixture.join(".diffpack-next/segment-boundary.tsx");
        let ctl_boundary = fixture.join(".diffpack-next/control-boundary.tsx");
        let reqctx = fixture.join(".diffpack-next/request-context.ts");
        let src = rsc_entry_module(
            &disc,
            &UnbuiltPatterns::default(),
            &crate::next_font::FontOutput::default(),
            &boundary,
            &seg_boundary,
            &ctl_boundary,
            &reqctx,
            None,
            "",
        );
        for line in src.lines() {
            if !line.contains(RSC_CSS_URL) {
                continue;
            }
            assert!(
                line.contains("existsSync(new URL(\"./server.css\", import.meta.url))"),
                "the stylesheet URL appears outside the emitted-artifact guard: {line}",
            );
        }
        assert_eq!(
            src.matches(RSC_CSS_URL).count(),
            1,
            "exactly one place decides the stylesheet link: {src}",
        );
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
        let written =
            std::fs::read_to_string(dir.join(".diffpack-output/next-config-manifest.json"))
                .unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&written).unwrap();
        assert_eq!(parsed["basePath"], "/docs");
        assert_eq!(parsed["trailingSlash"], true);
        assert_eq!(parsed["i18n"]["defaultLocale"], "en");

        // No eval: a well-formed empty manifest (every routing field present so the
        // orchestrator's reader never sees an undefined).
        write_next_config_manifest(&dir, None);
        let empty = std::fs::read_to_string(dir.join(".diffpack-output/next-config-manifest.json"))
            .unwrap();
        let ep: serde_json::Value = serde_json::from_str(&empty).unwrap();
        assert_eq!(ep["basePath"], "");
        assert_eq!(ep["trailingSlash"], false);
        assert!(ep["i18n"].is_null());
    }

    #[test]
    fn image_shim_supports_remote_hosts_and_loaders() {
        let shim = next_image_shim("");
        assert!(
            shim.contains(r#"import CONFIG from "../image-config""#),
            "shim reads the images config: {shim}"
        );
        assert!(
            shim.contains("function matchRemotePattern"),
            "shim ports the remote-pattern matcher"
        );
        assert!(
            shim.contains("is not configured under"),
            "shim throws a clear hostname error for a disallowed remote host"
        );
        assert!(
            shim.contains("imgixLoader")
                && shim.contains("cloudinaryLoader")
                && shim.contains("akamaiLoader"),
            "shim has the built-in loaders"
        );
    }

    #[test]
    fn image_shim_bakes_asset_base_and_routes_to_runtime_optimizer() {
        // With an asset base the shim bakes the const and points the runtime optimizer
        // endpoint under the prefix; build-variant + local raw URLs go through withAssetBase.
        let shim = next_image_shim("/cdn/docs");
        assert!(
            shim.contains(r#"const ASSET_BASE = "/cdn/docs";"#),
            "asset base baked as a const: {shim}"
        );
        assert!(
            shim.contains(r#"const IMAGE_ENDPOINT = ASSET_BASE + "/_next/image";"#),
            "optimizer endpoint under the asset base: {shim}"
        );
        assert!(
            shim.contains("function withAssetBase"),
            "the asset-base prefix helper is generated"
        );
        assert!(
            shim.contains("function optimizerSrcSet"),
            "the runtime-optimizer srcset builder is generated"
        );
        // The default-loader optimizer URL is Next's `/_next/image?url=&w=&q=` shape.
        assert!(
            shim.contains(
                r#"IMAGE_ENDPOINT + "?url=" + encodeURIComponent(rawSrc) + "&w=" + w + "&q=""#
            ),
            "optimizer URL uses Next's default-loader query shape: {shim}",
        );
        // EVERY default-loader src routes through the optimizer (Next has no
        // "prefer a build-time file" branch, so neither does the shim).
        assert!(
            shim.contains("optimizerSrcSet(rawSrc, numericWidth, sizes, quality)"),
            "the default loader routes through the optimizer: {shim}",
        );
        assert!(
            !shim.contains("entry.variants[String("),
            "no build-variant URL is ever rendered into the srcSet: {shim}",
        );
        assert!(
            !shim.contains("no build-emitted variant manifest entry for raster src"),
            "the old hard-error-on-missing-variant path is gone (optimizer handles it): {shim}",
        );

        // No asset base: the const is empty, so withAssetBase is an identity.
        let plain = next_image_shim("");
        assert!(
            plain.contains(r#"const ASSET_BASE = "";"#),
            "empty asset base const: {plain}"
        );
        assert!(
            plain.contains(r#"const IMAGE_ENDPOINT = ASSET_BASE + "/_next/image";"#),
            "optimizer endpoint still present with no prefix: {plain}"
        );
    }

    /// Run the generated `next/image` shim under node against a stub react
    /// `createElement` / manifest / config and return the props of every element it
    /// renders for `props`. The shim's three imports are the only substitutions —
    /// the component logic under test is the emitted source, byte for byte.
    fn render_image_shim(props: &str) -> serde_json::Value {
        let mut src = next_image_shim("");
        for (import, stub) in [
            (
                "import { createElement, Fragment } from \"react\";",
                "const Fragment = \"#fragment\";\nfunction createElement(type, props) { RENDERED.push({ type, props }); return { type, props }; }",
            ),
            (
                "import MANIFEST from \"../image-manifest\";",
                "const MANIFEST = STUB_MANIFEST;",
            ),
            (
                "import CONFIG from \"../image-config\";",
                "const CONFIG = STUB_CONFIG;",
            ),
        ] {
            assert!(src.contains(import), "shim still has `{import}`:\n{src}");
            src = src.replace(import, stub);
        }
        let prelude = r##"const RENDERED = [];
const STUB_CONFIG = { remotePatterns: [{ protocol: "https", hostname: "img.example.com" }] };
const STUB_MANIFEST = {
  "/hero.png": {
    width: 1000,
    height: 1000,
    variants: { "640": "/v-640.png", "750": "/v-750.png", "828": "/v-828.png", "1000": "/v-1000.png" },
    blurDataURL: "data:image/gif;base64,BLUR",
  },
};
"##;
        let driver = format!("\nImage({props});\nconsole.log(JSON.stringify(RENDERED));\n");
        // One scratch dir per CALL. `scratch` wipes the directory it hands back and
        // these tests run in parallel, so any name two calls can share is a race —
        // and naming the directory after the props hash was exactly such a name:
        // `image_shim_renders_next_srcset_sizes_and_dimensions` and
        // `image_shim_matches_next_img_style_assembly` pass byte-identical props, so
        // they collided and one `remove_dir_all` raced the other's `write` (an
        // intermittent `InvalidInput` from `fs::write`). A monotonic counter is
        // unique by construction, whatever the props say.
        static RENDER_SEQUENCE: std::sync::atomic::AtomicUsize =
            std::sync::atomic::AtomicUsize::new(0);
        let sequence = RENDER_SEQUENCE.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let file = scratch(&format!("image-shim-render-{sequence}")).join("shim.mjs");
        std::fs::write(&file, format!("{prelude}{src}{driver}")).unwrap();
        let out = std::process::Command::new("node")
            .arg(&file)
            .output()
            .unwrap();
        assert!(
            out.status.success(),
            "shim render failed: {}",
            String::from_utf8_lossy(&out.stderr),
        );
        serde_json::from_slice(&out.stdout).unwrap()
    }

    /// Regression: `next/image` renders `style={{ color: "transparent" }}` (Next hides
    /// the alt text until the <img> errors — which a server-rendered <img> never does).
    /// Without it EVERY next/image element differed from Next on the computed `color`
    /// and, because border colors inherit `currentColor`, on `border-*-color` too.
    #[test]
    fn image_shim_matches_next_img_style_assembly() {
        if std::process::Command::new("node")
            .arg("--version")
            .output()
            .is_err()
        {
            return;
        }
        // 1. A plain fixed-size image: color transparent, data-nimg="1", no `sizes`
        //    (the srcSet uses `x` descriptors, so Next emits no sizes attribute).
        let rendered =
            render_image_shim(r#"{ src: "/hero.png", alt: "Hero", width: 700, height: 700 }"#);
        let img = &rendered[0];
        assert_eq!(img["type"], "img");
        assert_eq!(img["props"]["style"]["color"], "transparent");
        assert_eq!(img["props"]["data-nimg"], "1");
        assert!(
            img["props"]["sizes"].is_null(),
            "no sizes for an x-descriptor srcSet: {img}"
        );

        // 2. The CALLER's style wins over `color: transparent` — Next assigns the
        //    caller's style LAST.
        let rendered = render_image_shim(
            r#"{ src: "/hero.png", alt: "Hero", width: 700, height: 700, style: { color: "red" } }"#,
        );
        assert_eq!(rendered[0]["props"]["style"]["color"], "red");

        // 3. `fill`: Next's absolute-positioning block, then color transparent, plus
        //    `data-nimg="fill"` and the default `sizes="100vw"` (w-descriptor srcSet).
        let rendered = render_image_shim(
            r#"{ src: "/hero.png", alt: "Hero", fill: true, objectFit: "cover" }"#,
        );
        let style = &rendered[0]["props"]["style"];
        assert_eq!(style["position"], "absolute");
        assert_eq!(style["width"], "100%");
        assert_eq!(style["height"], "100%");
        assert_eq!(style["objectFit"], "cover");
        assert_eq!(style["color"], "transparent");
        assert_eq!(rendered[0]["props"]["data-nimg"], "fill");
        assert_eq!(rendered[0]["props"]["sizes"], "100vw");

        // 4. The blur placeholder is spread OVER imgStyle (Next's order), and its
        //    background-size/position derive from the resolved objectFit/objectPosition.
        let rendered = render_image_shim(
            r#"{ src: "/hero.png", alt: "Hero", fill: true, objectFit: "contain", placeholder: "blur" }"#,
        );
        let style = &rendered[0]["props"]["style"];
        assert_eq!(
            style["color"], "transparent",
            "color survives the placeholder spread: {style}"
        );
        assert_eq!(style["backgroundSize"], "contain");
        assert_eq!(style["backgroundPosition"], "50% 50%");
        assert_eq!(
            style["backgroundImage"],
            "url(\"data:image/gif;base64,BLUR\")"
        );

        // 5. `unoptimized` still gets the style + data-nimg (Next tags every <img>).
        let rendered = render_image_shim(
            r#"{ src: "/hero.png", alt: "Hero", width: 700, height: 700, unoptimized: true }"#,
        );
        assert_eq!(rendered[0]["props"]["style"]["color"], "transparent");
        assert_eq!(rendered[0]["props"]["data-nimg"], "1");
    }

    /// Regression (e2e cluster "next/image serves build-time variants where Next serves
    /// its runtime optimizer URL"): with Next's DEFAULT loader every optimizable src
    /// renders the `/_next/image?url=&w=&q=` shape — a `/public` string src, a static
    /// image import, and an allow-listed remote alike. diffpack used to render its own
    /// build-emitted variant files (`/_diffpack-image/…`, `/assets/…-1080.png`), which
    /// is a URL shape Next never produces. The variants still exist; the orchestrator
    /// serves `/_next/image` FROM them.
    #[test]
    fn image_shim_renders_next_optimizer_urls_for_every_default_loader_src() {
        if std::process::Command::new("node")
            .arg("--version")
            .output()
            .is_err()
        {
            return;
        }
        // 1. A `/public` string src WITH a build-variant manifest entry: still the
        //    optimizer URL, never `/v-1000.png`.
        let rendered =
            render_image_shim(r#"{ src: "/hero.png", alt: "Hero", width: 700, height: 700 }"#);
        let props = &rendered[0]["props"];
        let src = props["src"].as_str().unwrap();
        assert_eq!(
            src, "/_next/image?url=%2Fhero.png&w=1920&q=75",
            "public string src renders Next's optimizer URL: {props}",
        );
        assert_eq!(
            props["srcSet"],
            "/_next/image?url=%2Fhero.png&w=750&q=75 1x, /_next/image?url=%2Fhero.png&w=1920&q=75 2x",
            "…and an x-descriptor optimizer srcSet: {props}",
        );
        assert!(
            !rendered.to_string().contains("/v-"),
            "no build-variant file URL is rendered: {rendered}",
        );

        // 2. A STATIC IMPORT object (`import x from './x.png'`) — the shape the build
        //    emits, variants and all. Next optimizes it through the same endpoint.
        let rendered = render_image_shim(
            r#"{ src: { src: "/assets/vercel-a1cdda59.png", width: 1600, height: 1600, variants: { "640": "/assets/vercel-a1cdda59-640.png" } }, alt: "V", width: 1000, height: 1000 }"#,
        );
        let props = &rendered[0]["props"];
        assert_eq!(
            props["src"], "/_next/image?url=%2Fassets%2Fvercel-a1cdda59.png&w=2048&q=75",
            "static import renders the optimizer URL, not its own variant file: {props}",
        );
        assert!(
            !props["srcSet"].as_str().unwrap().contains("-640.png"),
            "the static import's build variants stay out of the srcSet: {props}",
        );

        // 3. An allow-listed remote src is unchanged (it always used the optimizer).
        let rendered = render_image_shim(
            r#"{ src: "https://img.example.com/a.png", alt: "R", width: 200, height: 200 }"#,
        );
        assert!(
            rendered[0]["props"]["src"]
                .as_str()
                .unwrap()
                .starts_with("/_next/image?url=https%3A%2F%2Fimg.example.com"),
            "remote src still routes through the optimizer: {rendered}",
        );

        // 4. `unoptimized` and SVG keep the RAW src with no srcSet (Next's two
        //    default-loader escape hatches), so this is not "everything is a query URL".
        for props_js in [
            r#"{ src: "/hero.png", alt: "H", width: 700, height: 700, unoptimized: true }"#,
            r#"{ src: "/logo.svg", alt: "L", width: 700, height: 700 }"#,
        ] {
            let rendered = render_image_shim(props_js);
            let props = &rendered[0]["props"];
            assert!(
                !props["src"].as_str().unwrap().contains("/_next/image"),
                "unoptimized/svg keeps the raw src: {props_js} -> {props}",
            );
            assert!(
                props["srcSet"].is_null(),
                "…and gets no srcSet: {props_js} -> {props}"
            );
        }
    }

    /// Regression: `priority` must NOT synthesize `fetchPriority="high"`. Next's
    /// `getImgProps` passes the caller's `fetchPriority` through untouched — `priority`
    /// only turns off lazy loading and adds the preload link. diffpack rendered
    /// `fetchpriority="high"` on every priority image, an attribute Next never emits.
    #[test]
    fn image_shim_passes_fetch_priority_through_instead_of_synthesizing_high() {
        if std::process::Command::new("node")
            .arg("--version")
            .output()
            .is_err()
        {
            return;
        }
        // `priority` alone: no fetchPriority anywhere, and loading is dropped (not lazy).
        let rendered = render_image_shim(
            r#"{ src: "/hero.png", alt: "H", width: 700, height: 700, priority: true }"#,
        );
        let of_type = |kind: &str| -> serde_json::Value {
            rendered
                .as_array()
                .unwrap()
                .iter()
                .find(|e| e["type"] == kind)
                .unwrap_or_else(|| panic!("no <{kind}> rendered: {rendered}"))
                .clone()
        };
        let link = of_type("link");
        let img = of_type("img");
        assert!(
            link["props"]["fetchPriority"].is_null(),
            "the preload link carries no synthesized fetchPriority: {link}",
        );
        assert!(
            link["props"]["href"].is_null(),
            "Next omits the preload href when an imageSrcSet is present: {link}",
        );
        assert!(
            img["props"]["fetchPriority"].is_null(),
            "`priority` alone renders NO fetchPriority: {img}",
        );
        assert!(
            img["props"]["loading"].is_null(),
            "`priority` drops loading=lazy: {img}"
        );

        // An explicit fetchPriority is passed through verbatim, with and without priority.
        for (props_js, expected) in [
            (
                r#"{ src: "/hero.png", alt: "H", width: 700, height: 700, fetchPriority: "low" }"#,
                "low",
            ),
            (
                r#"{ src: "/hero.png", alt: "H", width: 700, height: 700, priority: true, fetchPriority: "auto" }"#,
                "auto",
            ),
        ] {
            let rendered = render_image_shim(props_js);
            let img = rendered
                .as_array()
                .unwrap()
                .iter()
                .find(|e| e["type"] == "img")
                .unwrap();
            assert_eq!(
                img["props"]["fetchPriority"], expected,
                "fetchPriority passed through: {img}"
            );
        }

        // Next's `loadingFinal = isLazy ? "lazy" : loading`: `priority` does not erase an
        // explicit `loading`, and a plain image is lazy.
        let rendered = render_image_shim(
            r#"{ src: "/hero.png", alt: "H", width: 700, height: 700, priority: true, loading: "eager" }"#,
        );
        let eager = rendered
            .as_array()
            .unwrap()
            .iter()
            .find(|e| e["type"] == "img")
            .unwrap();
        assert_eq!(eager["props"]["loading"], "eager");
        let rendered =
            render_image_shim(r#"{ src: "/hero.png", alt: "H", width: 700, height: 700 }"#);
        assert_eq!(rendered[0]["props"]["loading"], "lazy");
    }

    /// Regression: props `getImgProps` consumes must never leak through `...rest` onto
    /// the DOM <img> as bogus attributes (`objectfit="cover"`, `layout="fill"`, …).
    #[test]
    fn image_shim_does_not_leak_next_only_props_to_the_dom() {
        if std::process::Command::new("node")
            .arg("--version")
            .output()
            .is_err()
        {
            return;
        }
        let rendered = render_image_shim(
            r#"{ src: "/hero.png", alt: "Hero", layout: "fill", objectFit: "cover", objectPosition: "top", overrideSrc: "/override.png" }"#,
        );
        let props = &rendered[0]["props"];
        for leaked in [
            "layout",
            "objectFit",
            "objectPosition",
            "overrideSrc",
            "fill",
            "priority",
            "preload",
        ] {
            assert!(
                props.get(leaked).is_none(),
                "`{leaked}` leaked onto the DOM <img>: {props}",
            );
        }
        // `layout="fill"` implies fill (positioning + data-nimg + default sizes) and
        // `overrideSrc` replaces the rendered src, exactly as Next does.
        assert_eq!(props["data-nimg"], "fill");
        assert_eq!(props["style"]["position"], "absolute");
        assert_eq!(props["style"]["objectPosition"], "top");
        assert_eq!(props["sizes"], "100vw");
        assert_eq!(props["src"], "/override.png");
    }

    #[test]
    fn next_cache_shim_emits_revalidate_and_unstable_cache() {
        // The next/cache shim exports the on-demand revalidation API + unstable_cache,
        // imports the shared requestAls store, collects into store.revalidated/store.tags,
        // and HARD-ERRORS (never silently no-ops) when called outside a request context.
        let ctx = Path::new("/tmp/.diffpack-next/request-context.ts");
        let shim = next_cache_shim(ctx);
        // Imports the SAME per-request store next/headers reads (collection hook) AND the
        // "use cache" collection scope.
        assert!(
            shim.contains("import { requestAls, cacheScopeAls }"),
            "cache shim imports requestAls + cacheScopeAls: {shim}"
        );
        // The public next/cache APIs are exported.
        assert!(
            shim.contains("export function revalidatePath("),
            "revalidatePath exported"
        );
        assert!(
            shim.contains("export function revalidateTag("),
            "revalidateTag exported"
        );
        assert!(
            shim.contains("export function unstable_cache("),
            "unstable_cache exported"
        );
        // Collection targets: store.revalidated.paths / .tags and store.tags.
        assert!(
            shim.contains("revalidated.paths.add"),
            "revalidatePath writes store.revalidated.paths"
        );
        assert!(
            shim.contains("revalidated.tags.add"),
            "revalidateTag writes store.revalidated.tags"
        );
        assert!(
            shim.contains("store.tags.add"),
            "unstable_cache registers its tags on the page store"
        );
        // No-silent-stub: missing store hard-errors naming the context.
        assert!(
            shim.contains("was called outside a request context"),
            "cache shim hard-errors with no store: {shim}"
        );
        // The "use cache" family is now IMPLEMENTED: cacheTag/cacheLife collect into the
        // active cache scope and __diffpackUseCache is the memoizing boundary the transform
        // wraps each export in. Each hard-errors (never silently no-ops) outside a scope.
        assert!(
            shim.contains("export function cacheTag("),
            "cacheTag exported"
        );
        assert!(
            shim.contains("export function cacheLife("),
            "cacheLife exported"
        );
        assert!(
            shim.contains("export function __diffpackUseCache("),
            "cache boundary helper exported"
        );
        assert!(
            shim.contains("scope.tags.add"),
            "cacheTag records into the cache scope"
        );
        assert!(
            shim.contains("outside a use cache scope"),
            "cacheTag/cacheLife hard-error outside a cache scope: {shim}"
        );
        // revalidateTag purges BOTH local memos (single-worker correctness).
        assert!(
            shim.contains("__unstableCacheMemo.delete"),
            "revalidateTag purges the unstable_cache memo"
        );
        assert!(
            shim.contains("__useCacheMemo.delete"),
            "revalidateTag purges the use-cache memo"
        );
    }

    /// REGRESSION, executed by node against the SHIPPED shim source. A cached function's
    /// arguments are keyed by `JSON.stringify`, which represents a class instance, a Proxy,
    /// a function-valued property or a Map as `{}` — so two calls with COMPLETELY different
    /// arguments collapsed onto one memo entry and the first caller's value was returned to
    /// everyone.
    ///
    /// cal.com's `/event-types` page is exactly this shape: `unstable_cache(loader,
    /// ["viewer.eventTypes.getUserEventGroups"], { revalidate: 3600 })` called with
    /// `await headers()` and `await cookies()` — the documented way to hand a cached
    /// function its request data. The first signed-in visitor's event types (their name,
    /// slugs and booking links) were served to every later visitor for an hour. cal.com's
    /// own Playwright suite caught it: the second test's `/event-types` HTML carried the
    /// FIRST test's username, so its "preview" link opened a stranger's booking page.
    #[test]
    fn a_cached_call_never_reuses_a_value_across_unserializable_arguments() {
        if std::process::Command::new("node")
            .arg("--version")
            .output()
            .is_err()
        {
            return;
        }
        let dir = scratch("cache-args-key");
        // Stand-in for the per-request store module the shim imports.
        std::fs::write(
            dir.join("request-context.mjs"),
            "import { AsyncLocalStorage } from 'node:async_hooks';\n\
             export const requestAls = new AsyncLocalStorage();\n\
             export const cacheScopeAls = new AsyncLocalStorage();\n",
        )
        .unwrap();
        let shim = next_cache_shim(&dir.join("request-context.mjs"));
        std::fs::write(dir.join("cache.mjs"), shim).unwrap();
        // Two requests, each with its own cookies()-shaped object (a plain object whose
        // members are FUNCTIONS, exactly what the next/headers shim returns) and its own
        // Headers instance. Their JSON is `{"size":1}` / `{}` — identical.
        std::fs::write(
            dir.join("run.mjs"),
            r#"import { unstable_cache } from "./cache.mjs";
const cookiesFor = (token) => ({ get: (n) => ({ name: n, value: token }), size: 1 });
let calls = 0;
const load = unstable_cache(
  async (_headers, _cookies) => { calls += 1; return _cookies.get("session").value; },
  ["loader"],
  { revalidate: 3600 },
);
const a = await load(new Headers({ cookie: "session=alice" }), cookiesFor("alice"));
const b = await load(new Headers({ cookie: "session=bob" }), cookiesFor("bob"));
if (a !== "alice" || b !== "bob") {
  throw new Error(`cross-request leak: got ${a} / ${b} after ${calls} call(s)`);
}
if (calls !== 2) throw new Error(`expected both calls to run, ran ${calls}`);
// Genuinely serializable arguments still memoize.
let plainCalls = 0;
const plain = unstable_cache(async (n) => { plainCalls += 1; return n * 2; }, ["plain"]);
if ((await plain(21)) !== 42 || (await plain(21)) !== 42) throw new Error("plain value wrong");
if (plainCalls !== 1) throw new Error(`plain args must memoize, ran ${plainCalls}`);
if ((await plain(1)) !== 2 || plainCalls !== 2) throw new Error("distinct plain args must not share");
console.log("OK");
"#,
        )
        .unwrap();
        let out = std::process::Command::new("node")
            .arg(dir.join("run.mjs"))
            .output()
            .unwrap();
        assert!(
            String::from_utf8_lossy(&out.stdout).contains("OK"),
            "node run failed:\nstdout: {}\nstderr: {}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr),
        );
    }

    /// diffpack's DEFAULT source-map policy is Next's, so that a `diffpack build-app`
    /// and a `next build` of the same app produce comparable artifacts. Getting this
    /// backwards is not a cosmetic difference: emitting no server maps makes every
    /// production Server Component stack trace unreadable, and emitting browser maps
    /// an app never asked for publishes its source to every visitor.
    #[test]
    fn the_default_source_map_policy_is_next_s_own() {
        let asked = serde_json::json!({
            "productionBrowserSourceMaps": true,
            "serverSourceMaps": true
        });
        let silent = serde_json::json!({ "basePath": "" });

        for target in [Target::Server, Target::IsolatedServer] {
            assert!(!default_source_maps(target, false, Some(&silent)));
            assert!(!default_source_maps(target, false, None));
            assert!(default_source_maps(target, false, Some(&asked)));
        }

        assert!(
            !default_source_maps(Target::Client, false, Some(&silent)),
            "browser maps ship to every visitor, so an app that did not ask gets none"
        );
        assert!(
            !default_source_maps(Target::Client, false, None),
            "an app with no next.config at all has not asked either"
        );
        assert!(
            default_source_maps(Target::Client, false, Some(&asked)),
            "`productionBrowserSourceMaps: true` is the app asking, and is honored"
        );

        for target in [Target::Client, Target::Server, Target::IsolatedServer] {
            assert!(
                default_source_maps(target, true, Some(&silent)),
                "in dev BOTH graphs get maps ({target:?}) — that is what `next dev` does, \
                 and it is the setting a developer is actually debugging under"
            );
        }
    }

    /// Anything other than a literal `true` is Next's default of off — including the
    /// truthy-looking strings a hand-written config can end up with.
    #[test]
    fn production_browser_source_maps_reads_only_a_literal_true() {
        for value in [
            serde_json::json!({ "productionBrowserSourceMaps": false }),
            serde_json::json!({ "productionBrowserSourceMaps": "true" }),
            serde_json::json!({ "productionBrowserSourceMaps": 1 }),
            serde_json::json!({}),
        ] {
            assert!(!production_browser_source_maps(Some(&value)), "{value}");
        }
        assert!(production_browser_source_maps(Some(&serde_json::json!({
            "productionBrowserSourceMaps": true
        }))));
    }

    /// The policy is only as good as the field reaching Rust: the real config eval
    /// must report `productionBrowserSourceMaps`, present or absent.
    #[test]
    fn the_config_eval_reports_production_browser_source_maps() {
        if std::process::Command::new("node")
            .arg("--version")
            .output()
            .is_err()
        {
            return;
        }
        let asked = scratch("next-config-browser-maps-on");
        std::fs::write(
            asked.join("next.config.mjs"),
            "export default { productionBrowserSourceMaps: true };\n",
        )
        .unwrap();
        let eval = run_next_config_eval(&asked).expect("the config must evaluate");
        assert!(
            production_browser_source_maps(Some(&eval)),
            "the eval dropped the field: {eval}"
        );
        assert!(default_source_maps(Target::Client, false, Some(&eval)));

        let silent = scratch("next-config-browser-maps-off");
        std::fs::write(
            silent.join("next.config.mjs"),
            "export default { basePath: '' };\n",
        )
        .unwrap();
        let eval = run_next_config_eval(&silent).expect("the config must evaluate");
        assert!(
            !production_browser_source_maps(Some(&eval)),
            "a config that says nothing must not read as an opt-in: {eval}"
        );
    }

    /// Evaluating `next.config` mutates `process.env`, and under `next dev`/`next build`
    /// those mutations ARE the environment the app is compiled and served in — the config
    /// runs in the same process. cal.com's config is exactly that shape
    /// (`dotenv.config({ path: "../../.env" })` plus computed variables), and its
    /// `DATABASE_URL` exists nowhere else: without carrying the delta out of diffpack's
    /// config-eval child process, every data-backed route rendered against a
    /// default-named database that does not exist.
    ///
    /// The delta must be exactly what the config CHANGED — propagating the child's whole
    /// environment would overwrite the real one.
    #[test]
    fn next_config_env_side_effects_are_carried_out_of_the_eval_child_process() {
        if std::process::Command::new("node")
            .arg("--version")
            .output()
            .is_err()
        {
            return;
        }
        let root = scratch("next-config-env");
        // A config that loads variables from a file the way dotenv does, computes one
        // from another, and leaves everything else alone.
        std::fs::write(
            root.join("secrets.txt"),
            "DIFFPACK_TEST_DB=postgres://seeded\n",
        )
        .unwrap();
        std::fs::write(
            root.join("next.config.mjs"),
            "import { readFileSync } from 'node:fs';\n\
             for (const line of readFileSync(new URL('./secrets.txt', import.meta.url), 'utf8').split('\\n')) {\n\
               const [k, v] = line.split('=');\n\
               if (k) process.env[k] = v;\n\
             }\n\
             process.env.DIFFPACK_TEST_DERIVED = process.env.DIFFPACK_TEST_DB + '/derived';\n\
             export default { basePath: '' };\n",
        )
        .unwrap();

        let eval = run_next_config_eval(&root).expect("the config must evaluate");
        let env = config_env(Some(&eval));
        assert_eq!(
            env,
            vec![
                (
                    "DIFFPACK_TEST_DB".to_string(),
                    "postgres://seeded".to_string()
                ),
                (
                    "DIFFPACK_TEST_DERIVED".to_string(),
                    "postgres://seeded/derived".to_string()
                ),
            ],
            "only the variables the config set are reported"
        );
        assert!(
            !env.iter().any(|(key, _)| key == "PATH"),
            "an inherited, unmodified variable is not part of the delta: {env:?}"
        );

        // The prerenderer and the dev orchestrator run after the compile and read the
        // delta back off the persisted manifest.
        write_next_config_manifest(&root, Some(&eval));
        assert_eq!(
            config_env_from_manifest(&root),
            env,
            "the persisted manifest must carry the same environment"
        );
    }

    /// A `next.config` that PRINTS must not lose its config.
    ///
    /// The eval used to hand its JSON back on the child's stdout, so a single
    /// `console.log` in the config — cal.com logs which rewrite set it selected, and
    /// warning about unset variables is a common idiom — made the payload unparseable.
    /// Diffpack then fell back to the EMPTY config and served the app with none of its
    /// redirects, rewrites, headers, basePath or i18n, with nothing to show for it. The
    /// payload now goes to a private file, and the config's own output is re-pointed at
    /// stderr so it stays visible.
    #[test]
    fn a_next_config_that_prints_still_yields_its_full_config() {
        if std::process::Command::new("node")
            .arg("--version")
            .output()
            .is_err()
        {
            return;
        }
        let root = scratch("next-config-noisy");
        std::fs::write(
            root.join("next.config.mjs"),
            "console.log('[Phase] selecting rewrites');\n\
             process.stdout.write('raw bytes straight to stdout\\n');\n\
             export default {\n\
               basePath: '/app',\n\
               async redirects() {\n\
                 console.log('building redirects');\n\
                 return [{ source: '/old', destination: '/new', permanent: true }];\n\
               },\n\
             };\n",
        )
        .unwrap();

        let eval = run_next_config_eval(&root)
            .expect("a config that prints must still be read back in full");
        assert_eq!(eval["basePath"], "/app");
        assert_eq!(eval["redirects"][0]["source"], "/old");
        assert_eq!(eval["redirects"][0]["destination"], "/new");
    }

    /// `has` / `missing` conditions decide whether a next.config redirect, rewrite or
    /// header rule applies at all.
    ///
    /// Dropping them is not "unsupported", it is WRONG: a conditional rule then fires
    /// unconditionally. cal.com gates `/api/auth/:path*` -> `/404` on a `callbackUrl`
    /// query, so with the conditions ignored every auth API request 307'd to /404 and
    /// the whole client — session, tRPC — got HTML where it expected JSON.
    ///
    /// This runs the real matcher out of `next-server.mjs` (sliced, not reimplemented)
    /// against the four condition types Next supports.
    #[test]
    fn next_config_has_and_missing_conditions_gate_a_rule() {
        if std::process::Command::new("node")
            .arg("--version")
            .output()
            .is_err()
        {
            return;
        }
        const SERVER: &str = include_str!("../../../scripts/rsc/next-server.mjs");
        let start = SERVER
            .find("function safeParamName(")
            .expect("next-server.mjs still defines the has/missing matcher");
        let end = SERVER[start..]
            .find("// Apply next.config redirects")
            .expect("the matcher block still ends before applyNextConfig")
            + start;
        let region = &SERVER[start..end];
        assert!(
            region.contains("function matchHas("),
            "sliced the right block: {region}"
        );

        // `readCookie` lives earlier in the file and the matcher calls it; slice that
        // one too rather than restating it here.
        let cookie_start = SERVER
            .find("function readCookie(")
            .expect("next-server.mjs still defines readCookie");
        let cookie_end = SERVER[cookie_start..]
            .find("\n// Parse an Accept-Language")
            .expect("readCookie still ends before the Accept-Language parser")
            + cookie_start;
        let cookie_region = &SERVER[cookie_start..cookie_end];

        let driver = r#"
const req = (headers) => ({ headers });
const u = (s) => new URL("http://example.test" + s);
const out = {};
// cal.com's real rule: only redirect when a `callbackUrl` query is present AND is not
// an absolute URL.
const authRule = { has: [{ type: "query", key: "callbackUrl", value: "^(?!https?://).*$" }] };
out.noQuery = matchHas(authRule, req({}), u("/api/auth/session"));
out.relativeQuery = matchHas(authRule, req({}), u("/api/auth/signin?callbackUrl=/dashboard"));
out.absoluteQuery = matchHas(authRule, req({}), u("/api/auth/signin?callbackUrl=https://evil.test"));
// host, with a named capture that becomes a destination param
const orgRule = { has: [{ type: "host", value: "(?<orgslug>.*)\\.cal\\.local" }] };
out.hostMatch = matchHas(orgRule, req({ host: "acme.cal.local:3000" }), u("/"));
out.hostMiss = matchHas(orgRule, req({ host: "cal.local:3000" }), u("/"));
// presence-only header binds the value as a param
out.headerPresent = matchHas({ has: [{ type: "header", key: "x-tenant" }] }, req({ "x-tenant": "acme" }), u("/"));
// cookie + missing
out.cookieMissing = matchHas({ missing: [{ type: "cookie", key: "session" }] }, req({ cookie: "other=1" }), u("/"));
out.cookiePresent = matchHas({ missing: [{ type: "cookie", key: "session" }] }, req({ cookie: "session=abc" }), u("/"));
// no conditions at all: always applies, contributes nothing
out.unconditional = matchHas({}, req({}), u("/"));
console.log(JSON.stringify(out));
"#;
        let file = scratch("next-config-has").join("has.mjs");
        std::fs::write(&file, format!("{cookie_region}\n{region}\n{driver}")).unwrap();
        let out = std::process::Command::new("node")
            .arg(&file)
            .output()
            .unwrap();
        assert!(
            out.status.success(),
            "the has/missing matcher failed to run: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        let got: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
        assert!(
            got["noQuery"].is_null(),
            "an absent query must NOT match: {got}"
        );
        assert_eq!(got["relativeQuery"], serde_json::json!({}), "{got}");
        assert!(
            got["absoluteQuery"].is_null(),
            "a value the regex rejects must not match: {got}"
        );
        assert_eq!(
            got["hostMatch"],
            serde_json::json!({ "orgslug": "acme" }),
            "the port is stripped and named groups become params: {got}"
        );
        assert!(got["hostMiss"].is_null(), "{got}");
        assert_eq!(
            got["headerPresent"],
            serde_json::json!({ "xtenant": "acme" }),
            "a presence-only condition binds the value under Next's safe param name: {got}"
        );
        assert_eq!(got["cookieMissing"], serde_json::json!({}), "{got}");
        assert!(
            got["cookiePresent"].is_null(),
            "`missing` must reject when the cookie IS present: {got}"
        );
        assert_eq!(got["unconditional"], serde_json::json!({}), "{got}");
    }

    /// No `next.config` (or no manifest yet) means nothing to propagate — never a panic
    /// and never a partial environment.
    #[test]
    fn next_config_env_is_empty_without_a_config() {
        let root = scratch("next-config-env-absent");
        assert!(config_env_from_manifest(&root).is_empty());
        assert!(config_env(None).is_empty());
    }

    #[test]
    fn next_cache_alias_and_shim_written_by_build() {
        // build_next_app must write shims/cache.ts AND alias next/cache to it (an app
        // importing next/cache resolves the faithful shim, not an unshimmed failure).
        let root = scratch("next-cache-alias");
        std::fs::create_dir_all(root.join("app")).unwrap();
        std::fs::write(
            root.join("app/page.tsx"),
            "export default function Page(){return null;}\n",
        )
        .unwrap();
        std::fs::write(
            root.join("app/layout.tsx"),
            "export default function L({children}){return children;}\n",
        )
        .unwrap();
        std::fs::write(root.join("next.config.js"), "module.exports = {};\n").unwrap();
        // The `client` environment writes the shims + alias vec without the react-server
        // config-eval node spawn, so the alias wiring is exercised without a child process.
        let cfg = configure(&root, "client").unwrap().unwrap();
        let has_alias = cfg
            .build
            .aliases
            .iter()
            .any(|(spec, file)| spec == "next/cache" && file.ends_with("cache.ts"));
        assert!(
            has_alias,
            "next/cache aliased to the shim: {:?}",
            cfg.build.aliases
        );
        let shim_path = root.join(".diffpack-next/shims/cache.ts");
        assert!(
            shim_path.is_file(),
            "shims/cache.ts written at {}",
            shim_path.display()
        );
        let contents = std::fs::read_to_string(&shim_path).unwrap();
        assert!(
            contents.contains("export function revalidateTag("),
            "written shim has revalidateTag"
        );
    }

    #[test]
    fn use_cache_directive_detected_and_wrapped_in_cache_boundary() {
        // A "use cache" prologue is recognized as its own directive (never confused with
        // use client/use server) and the react-server transform wraps every export in the
        // __diffpackUseCache boundary rather than dropping the module's caching semantics.
        use crate::rsc::{RscDirective, detect_directive};
        let path = Path::new("/tmp/cached.ts");
        let src = "\"use cache\";\nexport async function data(){return 1;}\n";
        assert_eq!(
            detect_directive(path, src),
            Some(RscDirective::Cache),
            "\"use cache\" prologue detected as the Cache directive"
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
        assert!(
            empty.runtime.is_none() && empty.fetch_cache.is_none() && empty.max_duration.is_none()
        );
    }

    #[test]
    fn edge_runtime_is_accepted_and_classified() {
        // `runtime = "edge"` is a recognized runtime (no longer a hard error).
        let src =
            "export const runtime = \"edge\";\nexport default function Page(){return null;}\n";
        let cfg = scan_route_config(src);
        assert_eq!(cfg.runtime.as_deref(), Some("edge"));
        assert!(
            validate_segment_config("/edgy", &cfg).is_ok(),
            "edge runtime validates"
        );
        assert_eq!(
            RouteRuntime::from_config("/edgy", cfg.runtime.as_deref()).unwrap(),
            RouteRuntime::Edge
        );
        assert_eq!(
            RouteRuntime::from_config("/x", Some("experimental-edge")).unwrap(),
            RouteRuntime::Edge
        );
        assert_eq!(
            RouteRuntime::from_config("/x", Some("nodejs")).unwrap(),
            RouteRuntime::Node
        );
        assert_eq!(
            RouteRuntime::from_config("/x", None).unwrap(),
            RouteRuntime::Node
        );
        // An unrecognized runtime is still a hard error (never a silent Node default).
        let bad = RouteRuntime::from_config("/z", Some("deno")).unwrap_err();
        assert!(
            bad.contains("/z") && bad.contains("deno"),
            "unknown runtime names route + value: {bad}"
        );
        // A clean edge page discovers + classifies normally.
        let app = scratch("edge-runtime-ok");
        std::fs::write(
            app.join("layout.tsx"),
            "export default function L({children}){return children;}\n",
        )
        .unwrap();
        let edge_dir = app.join("edgy");
        std::fs::create_dir_all(&edge_dir).unwrap();
        std::fs::write(edge_dir.join("page.tsx"), src).unwrap();
        let layout = first_existing(&app, "layout");
        let disc = discover_routes(&app, layout.as_deref()).expect("clean edge page discovers");
        assert!(
            disc.routes.iter().any(|r| r.url_path == "/edgy"),
            "the edge route is recorded"
        );
    }

    #[test]
    fn edge_module_importing_node_builtin_hard_errors() {
        // The "no Node fs" contract: an edge module importing a Node built-in fails loudly.
        for (src, name) in [
            (
                "import fs from \"node:fs\";\nexport function GET(){}\n",
                "fs",
            ),
            (
                "import { readFile } from \"fs/promises\";\nexport function GET(){}\n",
                "fs/promises",
            ),
            (
                "const cp = require(\"child_process\");\nexport function GET(){}\n",
                "child_process",
            ),
            (
                "import net from \"net\";\nexport const runtime=\"edge\";\n",
                "net",
            ),
        ] {
            let err = validate_edge_module("edge route /x", src).unwrap_err();
            assert!(
                err.contains(name) && err.contains("edge"),
                "must name the builtin + edge: {err}"
            );
        }
        // WinterCG-safe imports (and Node built-ins the edge runtime polyfills) pass.
        for ok in [
            "import { NextResponse } from \"next/server\";\nexport function GET(){ return new Response(\"ok\"); }\n",
            "import { Buffer } from \"node:buffer\";\nexport function GET(){}\n",
            "import crypto from \"node:crypto\";\nexport function GET(){}\n",
            // A `fs` substring that is NOT an import specifier must not false-trigger.
            "const label = \"fs is fine as data\";\nexport function GET(){ return new Response(label); }\n",
        ] {
            assert!(
                validate_edge_module("edge route /ok", ok).is_ok(),
                "clean edge module passes: {ok}"
            );
        }
        // Discovery of an edge ROUTE HANDLER that imports fs fails the build, naming it.
        let app = scratch("edge-handler-fs");
        let api = app.join("api").join("edge");
        std::fs::create_dir_all(&api).unwrap();
        std::fs::write(
            api.join("route.ts"),
            "export const runtime = \"edge\";\nimport fs from \"node:fs\";\nexport function GET(){ return new Response(String(fs)); }\n",
        ).unwrap();
        let err = discover_route_handlers(&app).unwrap_err();
        assert!(
            err.contains("fs") && err.contains("edge"),
            "edge handler fs import fails discovery: {err}"
        );
    }

    #[test]
    fn template_and_global_error_discovered_and_composed() {
        let app = scratch("template-global-error");
        std::fs::write(
            app.join("layout.tsx"),
            "export default function L({children}){return children;}\n",
        )
        .unwrap();
        std::fs::write(
            app.join("page.tsx"),
            "export default function P(){return null;}\n",
        )
        .unwrap();
        std::fs::write(
            app.join("template.tsx"),
            "\"use client\";\nexport default function T({children}){return children;}\n",
        )
        .unwrap();
        std::fs::write(
            app.join("global-error.tsx"),
            "\"use client\";\nexport default function GE({error}){return null;}\n",
        )
        .unwrap();
        let layout = first_existing(&app, "layout");
        let disc = discover_routes(&app, layout.as_deref()).unwrap();
        assert!(
            disc.global_error.is_some(),
            "global-error.tsx discovered at app root"
        );
        let root = disc.routes.iter().find(|r| r.url_path == "/").unwrap();
        assert!(
            root.levels.iter().any(|l| l.template.is_some()),
            "the app-root level carries the template",
        );

        // Codegen: the react-server entry emits the template id, GLOBAL_ERROR const, the
        // pathname remount key, and the global-error boundary wrapping the whole tree.
        let boundary = app.join("error-boundary.tsx");
        let seg_boundary = app.join("segment-boundary.tsx");
        let ctl_boundary = app.join("control-boundary.tsx");
        std::fs::write(&boundary, error_boundary_module(&ctl_boundary)).unwrap();
        let reqctx = app.join("request-context.ts");
        std::fs::write(&reqctx, request_context_module()).unwrap();
        let rsc = rsc_entry_module(
            &disc,
            &UnbuiltPatterns::default(),
            &crate::next_font::FontOutput::default(),
            &boundary,
            &seg_boundary,
            &ctl_boundary,
            &reqctx,
            None,
            "",
        );
        assert!(rsc.contains("template:"), "levels carry a template id");
        assert!(
            rsc.contains("const GLOBAL_ERROR ="),
            "GLOBAL_ERROR const emitted"
        );
        assert!(
            rsc.contains("key: pathname"),
            "template is keyed by pathname for remount"
        );
        assert!(
            rsc.contains("fallback: GLOBAL_ERROR"),
            "global-error wraps the document tree"
        );
    }

    /// A BUILD-TIME prerender has no request. Reading `cookies()`/`headers()`/
    /// `draftMode()` under one must raise the dynamic bailout — Next's static→dynamic
    /// demotion — instead of handing back a fabricated empty value.
    ///
    /// Returning empty was silently wrong in the worst way: cal.com's settings pages ask
    /// `getServerSession(headers(), cookies())`, got "no session" from the empty answer,
    /// and `redirect("/auth/login")`ed — so a login redirect would have been baked into a
    /// static HTML file and served to every signed-in user. The store carries a
    /// `prerender` flag; the shims consult it; the prerenderer records the route Dynamic.
    #[test]
    fn a_request_state_read_under_a_prerender_raises_the_dynamic_bailout() {
        if std::process::Command::new("node")
            .arg("--version")
            .output()
            .is_err()
        {
            return;
        }
        let root = scratch("prerender-bailout");
        let reqctx = root.join("request-context.mjs");
        std::fs::write(&reqctx, request_context_module()).unwrap();
        let shim = root.join("headers.mjs");
        std::fs::write(&shim, next_headers_shim(&reqctx)).unwrap();
        let driver = root.join("driver.mjs");
        std::fs::write(
            &driver,
            format!(
                r#"import {{ requestAls }} from {reqctx};
import {{ cookies, headers, draftMode }} from {shim};
const base = {{ headers: new Headers([["x-a", "1"]]), cookieHeader: "k=v", responseCookies: [], tags: new Set() }};
const out = [];
for (const api of [["cookies", cookies], ["headers", headers], ["draftMode", draftMode]]) {{
  // Under a prerender: must throw the tagged bailout naming the API.
  await requestAls.run({{ ...base, prerender: true }}, async () => {{
    try {{ await api[1](); out.push(api[0] + ":NO-THROW"); }}
    catch (error) {{ out.push(api[0] + ":" + error.digest + ":" + (error.message.includes(api[0] + "()") ? "named" : "unnamed")); }}
  }});
}}
// Under a real request (and under force-static, which does NOT set the flag): the real value.
await requestAls.run({{ ...base, prerender: false }}, async () => {{
  const h = await headers();
  const c = await cookies();
  out.push("live:" + h.get("x-a") + ":" + c.get("k").value);
}});
console.log(out.join("|"));
"#,
                reqctx = js_str(&reqctx.to_string_lossy()),
                shim = js_str(&shim.to_string_lossy()),
            ),
        )
        .unwrap();
        let out = std::process::Command::new("node")
            .arg(&driver)
            .output()
            .unwrap();
        assert!(
            out.status.success(),
            "driver failed: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        let printed = String::from_utf8_lossy(&out.stdout).trim().to_string();
        assert_eq!(
            printed,
            "cookies:DIFFPACK_DYNAMIC_BAILOUT:named|headers:DIFFPACK_DYNAMIC_BAILOUT:named|draftMode:DIFFPACK_DYNAMIC_BAILOUT:named|live:1:v",
            "prerender reads bail out (tagged + naming the API); a real request still reads through"
        );
    }

    /// The prerenderer must DEMOTE a bailed-out route to Dynamic, not fail the build:
    /// `next build` prints "couldn't be rendered statically because it used `headers`"
    /// and serves the route on demand. Pinned on the shipped script's bytes.
    #[test]
    fn the_prerenderer_demotes_a_bailed_out_route_instead_of_failing() {
        const PRERENDER_MJS: &str = include_str!("../../../scripts/rsc/next-prerender.mjs");
        assert!(
            PRERENDER_MJS.contains("reqCtx: { prerender: !forceStatic }"),
            "the prerenderer tells the shims there is no request (except under force-static)"
        );
        assert!(
            PRERENDER_MJS
                .contains("if (dynamicBailout) return { demoted: String(dynamicBailout) };"),
            "a bailout demotes the route rather than dying"
        );
        assert!(
            PRERENDER_MJS.contains("dynamic.push({ path: d.path, reason: d.reason })"),
            "a demoted route lands in the SAME dynamic list a statically-classified one does"
        );
        assert!(
            PRERENDER_MJS.contains("could not be prerendered —"),
            "every demotion is reported by name (never a silent reclassification)"
        );
    }

    #[test]
    fn instrumentation_entry_detects_root_and_src() {
        // Root-level instrumentation.ts is found.
        let root = scratch("instrumentation-root");
        std::fs::write(
            root.join("instrumentation.ts"),
            "export function register(){}\n",
        )
        .unwrap();
        assert!(
            instrumentation_entry(&root).is_some(),
            "root instrumentation.ts detected"
        );
        // A src/ instrumentation.js is found when the root has none.
        let root2 = scratch("instrumentation-src");
        let src = root2.join("src");
        std::fs::create_dir_all(&src).unwrap();
        std::fs::write(
            src.join("instrumentation.js"),
            "export function register(){}\n",
        )
        .unwrap();
        assert!(
            instrumentation_entry(&root2).is_some(),
            "src/ instrumentation.js detected"
        );
        // No instrumentation file → None.
        let root3 = scratch("instrumentation-none");
        assert!(
            instrumentation_entry(&root3).is_none(),
            "no instrumentation → None"
        );
    }

    /// The emitted production orchestrator, as a string (the same bytes `build-app`
    /// writes to `<out>/next-server.mjs`).
    const NEXT_SERVER_MJS: &str = include_str!("../../../scripts/rsc/next-server.mjs");

    #[test]
    fn the_production_server_survives_a_failing_render() {
        // FINDINGS #22. The request handler's catch called `res.writeHead(500)` on a
        // response whose shell had already gone out; that throws ERR_HTTP_HEADERS_SENT
        // from an async handler, which Node turns into `process.exit(1)` — ONE bad
        // render killed the whole server. The catch must delegate to a reporter that
        // checks `res.headersSent`, and the process must not die on an unhandled
        // rejection either.
        let catch_block = NEXT_SERVER_MJS
            .rsplit_once("} catch (error) {")
            .expect("the request handler has a catch")
            .1;
        let catch_body = catch_block.split_once("\n  }").expect("catch block ends").0;
        assert!(
            catch_body.contains("failRequest(res, error)"),
            "the request handler's catch must go through failRequest: {catch_body}",
        );
        assert!(
            !catch_body.contains("res.writeHead("),
            "the catch must not write a status line unconditionally: {catch_body}",
        );
        assert!(
            NEXT_SERVER_MJS.contains("function failRequest(res, error)")
                && NEXT_SERVER_MJS.contains("res.headersSent || res.writableEnded"),
            "failRequest must guard on headersSent before writing a 500",
        );
        for event in ["uncaughtException", "unhandledRejection"] {
            assert!(
                NEXT_SERVER_MJS.contains(&format!("\"{event}\"")),
                "the server must install a process-level {event} handler",
            );
        }
        assert!(
            NEXT_SERVER_MJS.contains("process.on(event, (error)"),
            "the process-level handlers must log rather than exit",
        );
        // A socket error on either half of an in-flight streaming response must not
        // throw (an `error` event with no listener is fatal).
        assert!(
            NEXT_SERVER_MJS.contains("req.on(\"error\"")
                && NEXT_SERVER_MJS.contains("res.on(\"error\""),
            "the request/response sockets must have error listeners",
        );
    }

    #[test]
    fn the_streaming_renderer_reports_a_post_shell_error_without_writing_headers() {
        // The SSR entry's `onShellError` used to gate on its own `shellStarted` flag,
        // which can be false AFTER `onShellReady` wrote the head — writing a 500 there
        // throws the same ERR_HTTP_HEADERS_SENT. `res.headersSent` is the authority.
        let ssr = ssr_entry_module(
            Path::new("/app/.diffpack-next"),
            &[],
            &BTreeSet::new(),
            Path::new("/app/.diffpack-next/hooks-context.ts"),
            "",
            &[],
        );
        assert!(
            ssr.contains("if (!shellStarted && !res.headersSent) {"),
            "onShellError must check res.headersSent: {ssr}",
        );
    }

    /// Starting the react-server entry twice attaches two stdin readers. Both consume
    /// each request and publish a Flight under the same id, so the orchestrator joins
    /// the two byte streams and React fails in `resolveModelChunk` with
    /// `chunk.reason.enqueueModel is not a function`. A cache-busting query does not
    /// identify a different entry and therefore must not bypass the once guard.
    #[test]
    fn rsc_worker_once_key_ignores_cache_busting_url_parts() {
        let app = scratch("rsc-worker-once-key");
        std::fs::write(
            app.join("layout.tsx"),
            "export default function Layout({children}){return children;}\n",
        )
        .unwrap();
        std::fs::write(
            app.join("page.tsx"),
            "export default function Page(){return null;}\n",
        )
        .unwrap();
        let layout = first_existing(&app, "layout");
        let disc = discover_routes(&app, layout.as_deref()).unwrap();
        let rsc = rsc_entry_module(
            &disc,
            &UnbuiltPatterns::default(),
            &crate::next_font::FontOutput::default(),
            &app.join("error-boundary.tsx"),
            &app.join("segment-boundary.tsx"),
            &app.join("control-boundary.tsx"),
            &app.join("request-context.ts"),
            None,
            "",
        );
        assert!(
            rsc.contains("const __diffpackEntryUrl = new URL(import.meta.url);")
                && rsc.contains("__diffpackEntryUrl.search = \"\";")
                && rsc.contains("__diffpackEntryUrl.hash = \"\";")
                && rsc.contains("const __diffpackEntryKey = __diffpackEntryUrl.href;"),
            "the worker identity must be the query/hash-free entry URL: {rsc}",
        );
        assert!(
            rsc.contains("if (!__diffpackStarted.has(__diffpackEntryKey))")
                && rsc.contains("__diffpackStarted.add(__diffpackEntryKey);"),
            "the normalized key must guard main(), which installs the stdin reader: {rsc}",
        );
        assert!(
            !rsc.contains("__diffpackStarted.has(import.meta.url)")
                && !rsc.contains("__diffpackStarted.add(import.meta.url)"),
            "a raw cache-busted URL would start a second worker: {rsc}",
        );
    }

    /// Both renderers pipe AT MOST ONCE, because react-dom's ready callbacks are not
    /// fire-once. When the last work to finish is a Suspense boundary that still holds
    /// abortable fallback tasks (a boundary whose FALLBACK suspends), `finishedTask`
    /// decrements `allPendingTasks`, then aborts those fallback tasks — each abort
    /// re-enters `finishedTask`, whose tail sees the counter at 0 and calls
    /// `completeAll`; the outer frame's tail then calls it again. The second `pipe`
    /// throws "React currently only supports piping to one writable stream." into the
    /// enclosing task's catch, which React reports as RECOVERABLE — so the document
    /// still arrives and only a log line marks it. cal.com logged exactly that once per
    /// request.
    ///
    /// `scripts/rsc/tests/ssr-pipe-once.test.mjs` is the executable half: it reproduces
    /// the upstream double call against the vendored react-dom and shows this guard
    /// absorbing it. This test is what ties that conclusion to the SHIPPED entry.
    #[test]
    fn both_renderers_pipe_at_most_once_however_often_react_says_ready() {
        let ssr = ssr_entry_module(
            Path::new("/app/.diffpack-next"),
            &[],
            &BTreeSet::new(),
            Path::new("/app/.diffpack-next/hooks-context.ts"),
            "",
            &[],
        );
        // Every `pipe(sink)` in the entry stands behind a `piped` once-flag, and the flag
        // is SET BEFORE the call (a `pipe` that throws must not leave the door open).
        assert_eq!(
            ssr.matches("pipe(sink);").count(),
            2,
            "the buffered and streaming renderers are the only pipe sites: {ssr}",
        );
        assert_eq!(
            ssr.matches("if (piped) return;\n        piped = true;")
                .count(),
            2,
            "both pipe sites must be guarded by the once-flag, set before piping: {ssr}",
        );
        assert_eq!(
            ssr.matches("let piped = false;").count(),
            2,
            "each render declares its OWN flag (a shared one would break concurrent requests): {ssr}",
        );
        // The streaming path's guard also covers `res.writeHead`, where a second call is
        // not a log line but ERR_HTTP_HEADERS_SENT thrown at the request handler.
        let stream = ssr
            .find("export async function renderFlightToStream")
            .expect("the entry exports the streaming renderer");
        let guard = ssr[stream..]
            .find("if (piped) return;")
            .expect("the streaming renderer guards its ready callback");
        let head = ssr[stream..]
            .find("res.writeHead(status || 200, headers);")
            .expect("the streaming renderer writes the head from onShellReady");
        assert!(
            guard < head,
            "the once-guard must precede res.writeHead, not just the pipe: {ssr}",
        );
    }

    #[test]
    fn a_strict_csp_nonce_reaches_every_script_the_document_emits() {
        // `next-strict-csp`: middleware sets `script-src 'nonce-…'`, so an inline script
        // without the nonce is BLOCKED and the page never hydrates. react-dom nonces its
        // bootstrap scripts from the `nonce` render option; the injected `__DF_FLIGHT`
        // chunks are diffpack's own tags and must carry it too.
        let ssr = ssr_entry_module(
            Path::new("/app/.diffpack-next"),
            &[],
            &BTreeSet::new(),
            Path::new("/app/.diffpack-next/hooks-context.ts"),
            "",
            &[],
        );
        assert_eq!(
            ssr.matches("nonce: nonce || undefined").count(),
            2,
            "both the buffered and the streaming render must pass the nonce to react-dom: {ssr}",
        );
        assert!(
            ssr.contains("const nonceAttr = nonce ?"),
            "the streaming renderer must build a nonce attribute: {ssr}",
        );
        assert!(
            !ssr.contains("\"<script>(self.__DF_FLIGHT"),
            "no flight script may be emitted without the nonce attribute: {ssr}",
        );
        // `script-src` wins over `default-src` (the canonical recipe declares both).
        assert!(
            NEXT_SERVER_MJS.contains("function scriptNonceFromHeaders(headerPairs)")
                && NEXT_SERVER_MJS
                    .contains("directives.find((part) => part.startsWith(\"script-src\"))"),
            "the orchestrator must read the nonce off script-src first",
        );
    }

    #[test]
    fn next_script_is_shimmed_as_a_pinned_client_island() {
        // FINDINGS #23. Next's own `next/script` is a CommonJS barrel inside
        // node_modules that `scan_project` cannot see, so its client reference had no
        // manifest entry. It is aliased to a project-local `"use client"` shim instead,
        // exactly like `next/link`.
        let shim = next_script_shim();
        assert!(
            shim.starts_with("\"use client\";"),
            "the directive stays first: {shim}"
        );
        assert!(
            shim.contains("export default Script;"),
            "the shim has a default export"
        );
        assert!(
            shim.contains("ReactDOM.preload(src, preloadOptions(props))"),
            "afterInteractive contributes a preload to the document: {shim}",
        );
        assert!(
            shim.contains("strategy === \"worker\"") && shim.contains("is not implemented"),
            "Partytown must be a LOUD error, not a silent downgrade: {shim}",
        );
        // The client + ssr entries must both pin it, or the flight's client reference
        // resolves to nothing at render time.
        let islands = [PathBuf::from("/app/.diffpack-next/shims/script.tsx")];
        let hooks = PathBuf::from("/app/.diffpack-next/hooks-context.ts");
        let client = client_entry_module(
            Path::new("/app/.diffpack-next"),
            &islands,
            &BTreeSet::new(),
            &hooks,
            PinKind::StaticRequire,
        );
        let ssr = ssr_entry_module(
            Path::new("/app/.diffpack-next"),
            &islands,
            &BTreeSet::new(),
            &hooks,
            "",
            &[],
        );
        for (label, source) in [("client", &client), ("ssr", &ssr)] {
            assert!(
                source.contains("shims/script.tsx") || source.contains("shims\\script.tsx"),
                "{label} entry must pin the next/script shim: {source}",
            );
        }
    }

    // --- hybrid `pages/api/**` in an app-router build ---------------------------------

    /// Lay out a hybrid app: pages under `app/`, HTTP endpoints under `pages/api/**` —
    /// the shape cal.com has, where next-auth and every tRPC router are pages API routes.
    fn write_hybrid_app(root: &Path) -> PathBuf {
        let app = root.join("app");
        std::fs::create_dir_all(&app).unwrap();
        std::fs::write(
            app.join("layout.tsx"),
            "export default function L({children}){return children;}\n",
        )
        .unwrap();
        std::fs::write(
            app.join("page.tsx"),
            "export default function P(){return null;}\n",
        )
        .unwrap();
        for (rel, body) in [
            (
                "api/health.ts",
                "export default function h(req,res){res.json({ok:true});}\n",
            ),
            (
                "api/auth/[...nextauth].ts",
                "export default function h(req,res){res.json({});}\n",
            ),
            (
                "api/user/[id]/index.ts",
                "export default function h(req,res){res.json({});}\n",
            ),
            (
                "api/nested/deep/thing.js",
                "export default function h(req,res){res.end();}\n",
            ),
        ] {
            let file = root.join("pages").join(rel);
            std::fs::create_dir_all(file.parent().unwrap()).unwrap();
            std::fs::write(file, body).unwrap();
        }
        // A non-api page is NOT an api route: only `pages/api/**` is served here.
        let page = root.join("pages/router/index.tsx");
        std::fs::create_dir_all(page.parent().unwrap()).unwrap();
        std::fs::write(page, "export default function R(){return null;}\n").unwrap();
        app
    }

    /// An app-router project with a `pages/api/**` tree must discover those endpoints.
    /// Building only `app/` is what left cal.com with no `/api/auth/session`, no CSRF
    /// token and no tRPC — the login form rendered and could never submit.
    #[test]
    fn a_hybrid_apps_pages_api_routes_are_discovered() {
        let dir = scratch("hybrid-pages-api");
        let app = write_hybrid_app(&dir);
        let routes = discover_pages_api_routes(&app).unwrap();
        let paths: Vec<&str> = routes.iter().map(|r| r.url_path.as_str()).collect();
        assert!(paths.contains(&"/api/health"), "{paths:?}");
        // `pages/api/user/[id]/index.ts` serves `/api/user/[id]`, not `/api/user/[id]/index`.
        assert!(paths.contains(&"/api/user/[id]"), "{paths:?}");
        assert!(paths.contains(&"/api/nested/deep/thing"), "{paths:?}");
        assert!(paths.contains(&"/api/auth/[...nextauth]"), "{paths:?}");
        assert!(
            !paths.iter().any(|p| p.contains("router")),
            "a pages PAGE is not an api route: {paths:?}",
        );
        // Most-specific first: every literal route is ordered before the catch-all, or a
        // `[...nextauth]` at `/api/auth/**` would swallow sibling endpoints.
        let catch_all = paths
            .iter()
            .position(|p| *p == "/api/auth/[...nextauth]")
            .unwrap();
        let literal = paths.iter().position(|p| *p == "/api/health").unwrap();
        assert!(
            literal < catch_all,
            "literal must beat catch-all: {paths:?}"
        );
    }

    /// No `pages/` directory at all (the overwhelmingly common app-router project) must
    /// discover nothing and cost nothing.
    #[test]
    fn an_app_router_project_without_pages_has_no_pages_api_routes() {
        let dir = scratch("hybrid-pages-api-absent");
        let app = dir.join("app");
        std::fs::create_dir_all(&app).unwrap();
        assert!(discover_pages_api_routes(&app).unwrap().is_empty());
    }

    /// `pages/api/**` endpoints must be bundled and run in the SSR graph, NOT the
    /// react-server one. Next compiles them in its `api-node` layer, which has no
    /// `react-server` export condition; under that condition `react-dom/server` resolves
    /// to React's stub that only throws `react-dom/server is not supported in React
    /// Server Components`, and cal.com's `renderEmail` imports it on every booking — so
    /// with these routes in the wrong graph every `POST /api/book/event` answered 500.
    /// The react-server entry keeps the PATTERNS (it owns route discovery) and publishes
    /// them through `routeManifest` so the orchestrator knows which paths to send to the
    /// SSR bundle instead of rendering a page for them.
    #[test]
    fn pages_api_routes_run_in_the_ssr_graph_not_the_react_server_graph() {
        let dir = scratch("hybrid-pages-api-entry");
        let app = write_hybrid_app(&dir);
        let disc = discover_routes(&app, Some(&app.join("layout.tsx"))).unwrap();
        assert!(
            !disc.pages_api.is_empty(),
            "the hybrid app has pages api routes"
        );
        let boundary = dir.join("error-boundary.tsx");
        let seg_boundary = dir.join("segment-boundary.tsx");
        let ctl_boundary = dir.join("control-boundary.tsx");
        let reqctx = dir.join("request-context.ts");
        let source = rsc_entry_module(
            &disc,
            &UnbuiltPatterns::default(),
            &crate::next_font::FontOutput::default(),
            &boundary,
            &seg_boundary,
            &ctl_boundary,
            &reqctx,
            None,
            "",
        );
        assert!(
            source.contains("const PAGES_API = ["),
            "the table is emitted: {source}"
        );
        assert!(
            source.contains(r#"path: "/api/auth/[...nextauth]""#),
            "each endpoint is in the table",
        );
        assert!(
            source.contains("pagesApi: PAGES_API.map("),
            "routeManifest publishes the patterns so the orchestrator routes them",
        );
        // The react-server graph must NOT pull these modules in: no import() of an
        // endpoint, and no invocation runtime.
        assert!(
            !source.contains("load: () => import("),
            "the react-server entry must not import a pages/api module: {source}",
        );
        assert!(
            !source.contains("runPagesApiHandler"),
            "the pages-api runtime must not be spliced into the react-server entry: {source}",
        );
        assert!(
            source.contains("return null;"),
            "handleRoute returns null so the orchestrator falls through to the SSR bundle",
        );
        // An app-router `route.ts` must still be matched FIRST (Next's precedence).
        let handlers = source.find("const ROUTE_HANDLERS = [").unwrap();
        let pages_api = source.find("const PAGES_API = [").unwrap();
        assert!(
            handlers < pages_api,
            "ROUTE_HANDLERS is declared before PAGES_API"
        );

        // ...and the SSR entry is where they actually live and run.
        let hooks = dir.join("hooks-context.ts");
        let ssr = ssr_entry_module(&dir, &[], &BTreeSet::new(), &hooks, "", &disc.pages_api);
        assert!(
            ssr.contains("const PAGES_API = ["),
            "the SSR entry carries the table: {ssr}"
        );
        assert!(
            ssr.contains("load: () => import(") && ssr.contains("nextauth"),
            "each endpoint is loaded through its own import() so it gets its own chunk: {ssr}",
        );
        assert!(
            ssr.contains("export async function handlePagesApi("),
            "the SSR entry exports the dispatcher the orchestrator calls: {ssr}",
        );
        // The runtime is the real file, spliced verbatim — not a second copy.
        assert!(
            ssr.contains("export async function runPagesApiHandler("),
            "src/next_runtime/pages_api.js is spliced into the SSR entry: {ssr}",
        );
        // The orchestrator must dispatch a pages-api path to the SSR bundle, not the
        // react-server worker.
        const SERVER: &str = include_str!("../../../scripts/rsc/next-server.mjs");
        assert!(
            SERVER.contains(r#"if (endpointKind === "pages-api") {"#),
            "the orchestrator splits the two endpoint kinds",
        );
        assert!(
            SERVER.contains("getPagesApiHandler()"),
            "the orchestrator resolves the dispatcher from the SSR bundle",
        );
    }

    /// A page Server Component's `searchParams` prop must be the REQUEST'S QUERY. It was
    /// hard-coded to `Promise.resolve({})` for every render, so any page that branches on
    /// the query server-side rendered the no-query variant. cal.com's booker is exactly
    /// that page: `?rescheduleUid=…` is what makes it load the existing booking and show
    /// the reschedule form, so "can reschedule a booking" got the ordinary Confirm button
    /// forever. `generateMetadata` takes the same prop and had the same hole.
    ///
    /// `__rsc` must be stripped: it is diffpack's own marker on the soft-navigation
    /// channel, so leaving it in would make a page see a different query depending on
    /// whether it was reached by a hard load or a client navigation.
    #[test]
    fn a_page_server_component_receives_the_requests_search_params() {
        let dir = scratch("search-params-prop");
        let app = write_hybrid_app(&dir);
        let disc = discover_routes(&app, Some(&app.join("layout.tsx"))).unwrap();
        let source = rsc_entry_module(
            &disc,
            &UnbuiltPatterns::default(),
            &crate::next_font::FontOutput::default(),
            &dir.join("error-boundary.tsx"),
            &dir.join("segment-boundary.tsx"),
            &dir.join("control-boundary.tsx"),
            &dir.join("request-context.ts"),
            None,
            "",
        );
        // No render path may hand a page an empty query object any more.
        assert!(
            !source.contains("searchParams: Promise.resolve({})"),
            "no render path may hard-code an empty searchParams: {source}",
        );
        assert!(
            source.contains("search.delete(\"__rsc\");"),
            "the soft-navigation marker is not part of the app's query: {source}",
        );
        // Repeated keys arrive as arrays, single keys as strings — Next's shape.
        assert!(
            source.contains("out[key] = all.length > 1 ? all : all[0];"),
            "a repeated query key is an array: {source}",
        );
        assert!(
            source.contains("const pageProps = { params: paramsPromise, searchParams: Promise.resolve(searchParams) };"),
            "the matched page gets the request's query: {source}",
        );
        assert!(
            source.contains(
                "m = await ns.generateMetadata({ params: paramsP, searchParams: searchP }"
            ),
            "generateMetadata gets it too: {source}",
        );
        // Layouts and templates must NOT get one (Next's contract).
        assert!(
            source.contains("createElement(level.layout, { params: paramsPromise }"),
            "a layout receives params only: {source}",
        );
    }

    /// A SOFT navigation changes the route, so `useParams()`/`usePathname()`/
    /// `useSearchParams()` must change with it. They used to be provided ONCE, at boot,
    /// from the document's injected globals and never touched again — so after
    /// `router.push("/booking/<uid>")` the params context still described the page the
    /// tab was opened on. cal.com's booking-success page zod-parses `useParams()` and
    /// died with `uid: Required` on every booking made through the UI (a hard reload of
    /// the same URL rendered fine, which is what made it look like a server bug).
    /// The params cannot be derived on the client — only the server knows the segment
    /// pattern — so they travel with the flight on `x-diffpack-params`.
    #[test]
    fn a_soft_navigation_reprovides_the_route_identity_hooks() {
        let islands = [PathBuf::from("/app/.diffpack-next/shims/link.tsx")];
        let hooks = PathBuf::from("/app/.diffpack-next/hooks-context.ts");
        let client = client_entry_module(
            Path::new("/app/.diffpack-next"),
            &islands,
            &BTreeSet::new(),
            &hooks,
            PinKind::StaticRequire,
        );

        // The producer: BOTH orchestrator paths that can answer `?__rsc=1` stamp the
        // header — the live render and the prerendered `.rsc` served straight off disk.
        const SERVER: &str = include_str!("../../../scripts/rsc/next-server.mjs");
        assert_eq!(
            SERVER.matches("x-diffpack-params").count(),
            2,
            "the live render AND the prerender cache must both stamp the params header",
        );
        assert!(
            SERVER.contains(
                r#"rscHeaders["x-diffpack-params"] = encodeURIComponent(JSON.stringify(meta.params || {}));"#
            ),
            "the live soft-nav response carries the matched params",
        );
        assert!(
            SERVER.contains(r#"tags, params: e.params || {} }"#),
            "the prerender manifest's recorded params reach the cache entry",
        );
        const PRERENDER: &str = include_str!("../../../scripts/rsc/next-prerender.mjs");
        assert!(
            PRERENDER.contains("params: params || {}"),
            "the prerender records the params it rendered with",
        );

        // The consumer: params ride along through the redirect-following fetch, and the
        // Router swaps the whole route identity with the tree.
        assert!(
            client.contains(
                "const params = parseFlightParams(res.headers.get(\"x-diffpack-params\"));"
            ),
            "fetchFlight reads the header: {client}",
        );
        assert!(
            client.contains("return { tree, intercept, params };"),
            "the params travel with the tree: {client}",
        );
        assert!(
            client.contains("const identity = routeIdentity(href, params);"),
            "navigate() computes the new route identity from the href it settled on: {client}",
        );
        for setter in ["setTree(next);", "setRoute(identity);"] {
            assert!(
                client.contains(setter),
                "navigate() swaps {setter}: {client}"
            );
        }
        // The providers live INSIDE the Router (they are re-rendered per navigation),
        // not in boot() where they could only ever hold the document's initial values.
        let router_at = client
            .find("function Router({ initialTree, initialRoute })")
            .unwrap();
        let boot_at = client.find("function boot()").unwrap();
        let provider_at = client.find("PathParamsContext.Provider").unwrap();
        assert!(
            router_at < provider_at && provider_at < boot_at,
            "the hooks providers must be rendered by the Router, not by boot(): {client}",
        );
        assert!(
            client.contains("{ value: route.params }"),
            "the params provider reads the Router's live route: {client}",
        );
        // router.refresh() re-reads them too (a refresh can follow a redirect onto a
        // different route).
        assert!(
            client.contains("const { tree: next, params, href } = await fetchFlightFollowing(current, () => undefined);"),
            "refresh() picks up the refreshed route's params: {client}",
        );
        // A malformed header is a hard error, never silently empty params.
        assert!(
            client.contains("percent-encoded JSON (")
                && client.contains("must decode to an object, got"),
            "a malformed params header throws with a diagnosable message: {client}",
        );
    }

    /// A soft navigation must never swap in a flight that is still arriving.
    ///
    /// The Router renders the flight root with no Suspense boundary of its own, so a
    /// tree whose rows have not all landed suspends the navigation's transition — and a
    /// suspended transition commits NOTHING: not the tree, and not the params/pathname/
    /// searchParams the Router provides beside it. `pushState` has already run by then,
    /// so a stalled tail leaves the tab split for good: the address bar on the new
    /// route, the DOM on the old one, no error, no retry. cal.com hit this on a second
    /// visit to `/event-types/<id>?tabName=recurring`, which kept rendering the Advanced
    /// tab forever while the URL said `recurring`.
    ///
    /// So `fetchFlight` reads the response to completion and settles the tree before
    /// returning it, which is also what Next's router does — its `fetchServerResponse`
    /// awaits `createFromFetch` before the reducer applies the navigation.
    #[test]
    fn a_soft_navigation_swaps_in_a_settled_flight_never_a_live_stream() {
        let islands = [PathBuf::from("/app/.diffpack-next/shims/link.tsx")];
        let hooks = PathBuf::from("/app/.diffpack-next/hooks-context.ts");
        let client = client_entry_module(
            Path::new("/app/.diffpack-next"),
            &islands,
            &BTreeSet::new(),
            &hooks,
            PinKind::StaticRequire,
        );

        let fetch_flight = {
            let at = client.find("async function fetchFlight(href) {").unwrap();
            let rest = &client[at..];
            &rest[..rest.find("\n}\n").unwrap()]
        };
        assert!(
            fetch_flight.contains("await res.arrayBuffer()"),
            "the soft-nav flight is read to completion before it is handed to the Router: {fetch_flight}",
        );
        assert!(
            !fetch_flight.contains("createFromReadableStream(res.body"),
            "the live response body must NOT be what the Router renders: {fetch_flight}",
        );
        let create_at = fetch_flight.find("createFromReadableStream(").unwrap();
        let await_at = fetch_flight.find("\n  await tree;").unwrap();
        assert!(
            create_at < await_at,
            "the tree is settled after it is created and before it is returned: {fetch_flight}",
        );
        assert!(
            fetch_flight[await_at..].contains("return { tree, intercept, params };"),
            "the settled tree is the one that travels to navigate(): {fetch_flight}",
        );
        // The INITIAL document keeps streaming — the SSR HTML is already on screen, so
        // rows arriving late cost nothing there. Only the soft-nav channel is buffered.
        assert!(
            client.contains("function flightStreamFromDF()")
                && client.contains("controller.enqueue(decodeFlight(entry[1]))"),
            "hydration still consumes the document's incremental flight stream: {client}",
        );
    }

    /// A server-side `redirect()` reached over the SOFT-NAVIGATION channel must be
    /// followed. `fetch` swallows a 3xx, so the orchestrator reports the redirect as
    /// JSON — and the client Router used to hand that JSON straight to the flight
    /// reader, which fails deep inside it and leaves a blank page. That is where
    /// cal.com's login dead-ended: signing in navigates to the callback URL `/`, `/`
    /// redirects logged-in users to `/event-types`, and the browser sat on an empty `/`.
    #[test]
    fn the_client_router_follows_a_soft_navigation_redirect() {
        let islands = [PathBuf::from("/app/.diffpack-next/shims/link.tsx")];
        let hooks = PathBuf::from("/app/.diffpack-next/hooks-context.ts");
        let client = client_entry_module(
            Path::new("/app/.diffpack-next"),
            &islands,
            &BTreeSet::new(),
            &hooks,
            PinKind::StaticRequire,
        );
        // The producer side must still be the JSON contract this consumer reads.
        const SERVER: &str = include_str!("../../../scripts/rsc/next-server.mjs");
        assert!(
            SERVER.contains(
                r#"res.end(JSON.stringify({ __redirect: addBasePath(meta.redirect, localeSeg) }));"#
            ),
            "the orchestrator answers a soft-nav redirect with __redirect JSON",
        );
        assert!(
            client.contains(r#"if (payload && typeof payload.__redirect === "string") return { redirect: payload.__redirect };"#),
            "the client reads __redirect off the JSON instead of parsing it as flight: {client}",
        );
        assert!(
            client.contains(
                r#"if ((res.headers.get("content-type") || "").includes("application/json")) {"#
            ),
            "the JSON channel is detected by content-type: {client}",
        );
        // Unexpected JSON is a loud error, never a silent blank page.
        assert!(
            client.contains("unexpected JSON on the soft-navigation channel for"),
            "any other JSON payload throws with the href: {client}",
        );
        // Redirects are followed, bounded, and the URL that lands in history is the one
        // that actually rendered.
        assert!(
            client.contains("async function fetchFlightFollowing(href, take)")
                && client.contains("const MAX_REDIRECTS = 10;")
                && client.contains("more than \" + MAX_REDIRECTS + \" server redirects following "),
            "the follow loop is bounded and diagnosable: {client}",
        );
        assert!(
            client.contains(
                "const { tree: next, intercept, params, href } = await fetchFlightFollowing(requested, take);"
            ),
            "navigate() follows redirects: {client}",
        );
        assert!(
            client
                .contains(r#"else if (href !== requested) history.replaceState(null, "", href);"#),
            "a redirected back/forward navigation corrects the address bar: {client}",
        );
        assert!(
            client.contains("await fetchFlightFollowing(current, () => undefined);"),
            "router.refresh() follows redirects too: {client}",
        );
    }

    /// A `public/` DIRECTORY must never be served as a static asset. `existsSync` says
    /// yes for one, so the orchestrator wrote a 200 + content-type and then threw
    /// `EISDIR` reading it — headers already gone, body never written, page never
    /// rendered. cal.com ships `public/apps/`, so its real `/apps` page answered an
    /// empty 200. The test runs the orchestrator's own `isStaticFile` against a real
    /// directory, a real file and a missing path.
    #[test]
    fn a_public_directory_is_not_served_as_a_static_asset() {
        if std::process::Command::new("node")
            .arg("--version")
            .output()
            .is_err()
        {
            return;
        }
        const SERVER: &str = include_str!("../../../scripts/rsc/next-server.mjs");
        assert!(
            SERVER.contains(
                "if (name && filePath.startsWith(publicDir) && isStaticFile(filePath)) {"
            ),
            "the public-asset serve must gate on isStaticFile",
        );
        assert!(
            !SERVER
                .contains("if (name && existsSync(filePath) && filePath.startsWith(publicDir)) {"),
            "the existsSync-only gate must be gone",
        );
        let start = SERVER
            .find("function isStaticFile(filePath) {")
            .expect("next-server.mjs defines isStaticFile");
        let end = SERVER[start..]
            .find("\n}\n")
            .expect("isStaticFile is a complete function")
            + start
            + 3;

        let dir = scratch("public-dir-not-a-file");
        std::fs::create_dir_all(dir.join("apps")).unwrap();
        std::fs::write(dir.join("favicon.ico"), b"icon").unwrap();
        let driver = dir.join("driver.mjs");
        std::fs::write(
            &driver,
            format!(
                "import {{ statSync }} from \"node:fs\";\n{}\nconsole.log(JSON.stringify([\
                 isStaticFile({dir}+\"/apps\"), isStaticFile({dir}+\"/favicon.ico\"), isStaticFile({dir}+\"/missing\")]));\n",
                &SERVER[start..end],
                dir = js_str(&dir.to_string_lossy()),
            ),
        )
        .unwrap();
        let out = std::process::Command::new("node")
            .arg(&driver)
            .output()
            .unwrap();
        assert!(
            out.status.success(),
            "isStaticFile failed: {}",
            String::from_utf8_lossy(&out.stderr),
        );
        assert_eq!(
            serde_json::from_slice::<Vec<bool>>(&out.stdout).unwrap(),
            vec![false, true, false],
            "a directory is not a file; a real file is; a missing path is not",
        );
    }

    /// A `redirect()` an ASYNC page throws must reach the orchestrator as a real HTTP
    /// redirect. React's flight writer emits its first chunk as soon as the root row is
    /// serializable, so an async page is a pending row in that chunk and a redirect
    /// thrown while awaiting it used to land after the headers were already gone — a
    /// logged-in cal.com `/` answered 200 with a half-rendered document instead of
    /// redirecting to `/event-types`, and the login round-trip dead-ended there.
    /// Resolving the page BEFORE the render starts is what makes the shell complete
    /// first, exactly as Next's is.
    #[test]
    fn an_async_pages_redirect_is_known_before_any_flight_byte_is_produced() {
        let dir = scratch("async-page-redirect");
        let app = dir.join("app");
        std::fs::create_dir_all(&app).unwrap();
        std::fs::write(
            app.join("layout.tsx"),
            "export default function L({children}){return children;}\n",
        )
        .unwrap();
        std::fs::write(
            app.join("page.tsx"),
            "import { redirect } from \"next/navigation\";\n\
             export default async function P(){ await Promise.resolve(); redirect(\"/event-types\"); }\n",
        )
        .unwrap();
        let disc = discover_routes(&app, Some(&app.join("layout.tsx"))).unwrap();
        let source = rsc_entry_module(
            &disc,
            &UnbuiltPatterns::default(),
            &crate::next_font::FontOutput::default(),
            &dir.join("error-boundary.tsx"),
            &dir.join("segment-boundary.tsx"),
            &dir.join("control-boundary.tsx"),
            &dir.join("request-context.ts"),
            None,
            "",
        );
        // The page is invoked and awaited by the composer, not merely elementized.
        assert!(
            source.contains("async function resolvePage(Page, props, control)"),
            "the entry resolves the page before rendering: {source}",
        );
        assert!(
            source.contains("const pageInShell = !route.levels.some((level) => level.loading);")
                && source.contains("? await resolvePage(route.page, pageProps, control || {})"),
            "documentTree awaits the matched page when it is part of the shell: {source}",
        );
        // A throw from the page is recorded in `control` AND re-thrown inside the tree,
        // so an ordinary error still reaches the segment's error.tsx boundary.
        assert!(
            source.contains("flightControlOnError(control, error);")
                && source.contains("function DiffpackPageThrow() {"),
            "a control-flow throw is recorded and re-thrown inside the tree: {source}",
        );
        // Only redirect/notFound/bailout are intercepted. Any OTHER throw goes back to
        // React unrendered, so a page that cannot run outside the renderer (a sync
        // Server Component calling `use()`) keeps exactly the path it had before.
        assert!(
            source.contains("function isControlThrow(error)")
                && source.contains("if (isControlThrow(error)) {")
                && source.contains("    return createElement(Page, props);"),
            "a non-control throw is handed back to React: {source}",
        );
        // Both render entries compose INSIDE the request store, or the page's own
        // cookies()/headers() reads would run with no store at all.
        assert_eq!(
            source
                .matches("documentTree(pathname, renderOpts(reqCtx), control)")
                .count(),
            2,
            "renderRequest and renderRequestStream both pass control: {source}",
        );
        assert_eq!(
            source
                .matches(
                    "await requestAls.run(store, () =>\n    documentTree(pathname, renderOpts(reqCtx), control),\n  );",
                )
                .count(),
            2,
            "both compose inside the request store: {source}",
        );
    }

    /// An app-router `route.ts` handler is called by Next with a `NextRequest`, and
    /// reading the query off `request.nextUrl.searchParams` is the ordinary way to do it.
    /// Handing it a bare `Request` makes that a read of `undefined.searchParams`, so the
    /// handler 500s on its first line — which is what every cal.com endpoint that parses
    /// a query string did.
    #[test]
    fn a_route_handler_is_invoked_with_a_next_request() {
        let dir = scratch("route-handler-next-request");
        let app = dir.join("app");
        std::fs::create_dir_all(app.join("api/link")).unwrap();
        std::fs::write(
            app.join("layout.tsx"),
            "export default function L({children}){return children;}\n",
        )
        .unwrap();
        std::fs::write(
            app.join("page.tsx"),
            "export default function P(){return null;}\n",
        )
        .unwrap();
        std::fs::write(
            app.join("api/link/route.ts"),
            "export async function GET(req){ return Response.json({ q: req.nextUrl.searchParams.get(\"q\") }); }\n",
        )
        .unwrap();
        let disc = discover_routes(&app, Some(&app.join("layout.tsx"))).unwrap();
        assert_eq!(disc.handlers.len(), 1, "the route handler is discovered");
        let source = rsc_entry_module(
            &disc,
            &UnbuiltPatterns::default(),
            &crate::next_font::FontOutput::default(),
            &dir.join("error-boundary.tsx"),
            &dir.join("segment-boundary.tsx"),
            &dir.join("control-boundary.tsx"),
            &dir.join("request-context.ts"),
            None,
            "",
        );
        assert!(
            source.contains("const request = new NextRequest(url, {"),
            "handleRoute must build a NextRequest, not a bare Request: {source}",
        );
        assert!(
            !source.contains("const request = new Request(url, {"),
            "no bare-Request construction may survive: {source}",
        );
        // The class it constructs has to be in scope AND actually carry `nextUrl`.
        assert!(
            source.contains("import {{ NextRequest }} from \"next/server\";")
                || source.contains("import { NextRequest } from \"next/server\";"),
            "the entry imports NextRequest: {source}",
        );
        assert!(
            next_server_shim().contains("this.nextUrl = new URL(url,"),
            "the next/server shim's NextRequest carries nextUrl",
        );
    }

    /// The pages-api runtime itself, exercised through node: a catch-all endpoint sees
    /// its segments as an ARRAY (next-auth dispatches on `req.query.nextauth`), a
    /// urlencoded POST body is parsed, cookies are parsed, and BOTH `Set-Cookie` headers
    /// a handler writes come back as separate values rather than one comma-joined string
    /// (a joined pair is an unusable cookie, which is exactly how a login round-trip
    /// loses its session).
    #[test]
    fn the_pages_api_runtime_runs_a_node_style_handler() {
        if std::process::Command::new("node")
            .arg("--version")
            .output()
            .is_err()
        {
            return;
        }
        let dir = scratch("pages-api-runtime");
        let runtime = dir.join("pages_api.mjs");
        std::fs::write(&runtime, include_str!("next_runtime/pages_api.js")).unwrap();
        let driver = dir.join("driver.mjs");
        std::fs::write(
            &driver,
            r#"import { runPagesApiHandler } from "./pages_api.mjs";

function handler(req, res) {
  res.setHeader("Set-Cookie", ["a=1; Path=/", "b=2; Path=/"]);
  res.status(201).json({
    method: req.method,
    nextauth: req.query.nextauth,
    q: req.query.callbackUrl,
    body: req.body,
    cookie: req.cookies.session,
    url: req.url,
  });
}

const result = await runPagesApiHandler({
  routeLabel: "/api/auth/[...nextauth]",
  handler,
  config: undefined,
  pathname: "/api/auth/callback/credentials",
  method: "POST",
  reqCtx: {
    url: "http://localhost/api/auth/callback/credentials?callbackUrl=%2Fevent-types",
    headers: [["content-type", "application/x-www-form-urlencoded"]],
    cookie: "session=abc",
    body: Buffer.from("email=pro%40example.com&password=pro").toString("base64"),
    bodyIsBase64: true,
  },
  params: { nextauth: ["callback", "credentials"] },
});
console.log(JSON.stringify({
  status: result.status,
  setCookies: result.setCookies,
  payload: JSON.parse(Buffer.from(result.body, "base64").toString("utf8")),
}));
"#,
        )
        .unwrap();
        let out = std::process::Command::new("node")
            .arg(&driver)
            .output()
            .unwrap();
        assert!(
            out.status.success(),
            "the pages api runtime failed: {}",
            String::from_utf8_lossy(&out.stderr),
        );
        let got: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
        assert_eq!(got["status"], 201, "{got}");
        assert_eq!(
            got["setCookies"],
            serde_json::json!(["a=1; Path=/", "b=2; Path=/"]),
            "both Set-Cookie values survive as separate headers: {got}",
        );
        let payload = &got["payload"];
        assert_eq!(payload["method"], "POST", "{got}");
        assert_eq!(
            payload["nextauth"],
            serde_json::json!(["callback", "credentials"]),
            "a catch-all param reaches the handler as an array: {got}",
        );
        assert_eq!(
            payload["q"], "/event-types",
            "search params merge into req.query: {got}"
        );
        assert_eq!(
            payload["body"]["email"], "pro@example.com",
            "urlencoded body parsed: {got}"
        );
        assert_eq!(payload["cookie"], "abc", "req.cookies parsed: {got}");
        assert_eq!(
            payload["url"], "/api/auth/callback/credentials?callbackUrl=%2Fevent-types",
            "req.url keeps its query, as Next's does: {got}",
        );
    }

    /// `export const config = { api: { bodyParser: false } }` must leave the bytes
    /// unparsed AND readable off the request stream — a Stripe/webhook endpoint verifies
    /// a signature over the exact bytes, so a parsed body silently breaks it.
    #[test]
    fn a_pages_api_route_can_opt_out_of_the_body_parser_and_read_the_raw_stream() {
        if std::process::Command::new("node")
            .arg("--version")
            .output()
            .is_err()
        {
            return;
        }
        let dir = scratch("pages-api-raw-body");
        std::fs::write(
            dir.join("pages_api.mjs"),
            include_str!("next_runtime/pages_api.js"),
        )
        .unwrap();
        let driver = dir.join("driver.mjs");
        std::fs::write(
            &driver,
            r#"import { runPagesApiHandler } from "./pages_api.mjs";

async function handler(req, res) {
  const chunks = [];
  for await (const chunk of req) chunks.push(chunk);
  res.status(200).send(JSON.stringify({ raw: Buffer.concat(chunks).toString("utf8"), parsed: req.body }));
}

const result = await runPagesApiHandler({
  routeLabel: "/api/stripe/webhook",
  handler,
  config: { api: { bodyParser: false } },
  pathname: "/api/stripe/webhook",
  method: "POST",
  reqCtx: {
    url: "http://localhost/api/stripe/webhook",
    headers: [["content-type", "application/json"]],
    body: Buffer.from('{"id":"evt_1"}').toString("base64"),
    bodyIsBase64: true,
  },
  params: {},
});
console.log(Buffer.from(result.body, "base64").toString("utf8"));
"#,
        )
        .unwrap();
        let out = std::process::Command::new("node")
            .arg(&driver)
            .output()
            .unwrap();
        assert!(
            out.status.success(),
            "the raw-body path failed: {}",
            String::from_utf8_lossy(&out.stderr),
        );
        let got: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
        assert_eq!(
            got["raw"], "{\"id\":\"evt_1\"}",
            "the exact bytes reach the handler: {got}"
        );
        assert!(
            got["parsed"].is_null(),
            "bodyParser:false leaves req.body unset: {got}"
        );
    }

    /// `diffpack start <output>` is handed the OUTPUT directory, not the project root, so
    /// the env propagation has to be keyed off it. Without this the production server ran
    /// with none of the variables the app's next.config loads — cal.com's `DATABASE_URL`
    /// lives only there, so every Server Component that touched Prisma threw and each
    /// route degraded to an error shell.
    #[test]
    fn config_env_is_readable_from_the_build_output_directory_alone() {
        let dir = scratch("config-env-from-output");
        let output = dir.join(".diffpack-output");
        std::fs::create_dir_all(&output).unwrap();
        std::fs::write(
            output.join("next-config-manifest.json"),
            r#"{"env":{"DATABASE_URL":"postgresql://localhost/app","NEXTAUTH_URL":"http://localhost:3000"}}"#,
        )
        .unwrap();
        assert_eq!(
            config_env_from_output(&output),
            vec![
                (
                    "DATABASE_URL".to_string(),
                    "postgresql://localhost/app".to_string()
                ),
                (
                    "NEXTAUTH_URL".to_string(),
                    "http://localhost:3000".to_string()
                ),
            ],
        );
        // The project-root spelling still answers, so the build-time callers are unchanged.
        assert_eq!(
            config_env_from_manifest(&dir),
            config_env_from_output(&output)
        );
        // No manifest at all: nothing to propagate, and no error.
        assert!(config_env_from_output(&dir.join("nope")).is_empty());
    }

    /// `serverExternalPackages` (and its pre-15 `experimental.` spelling) must be read
    /// out of the evaluated config; both are merged, deduplicated and order-preserving.
    #[test]
    fn server_external_packages_are_read_from_both_config_spellings() {
        let eval = serde_json::json!({
            "serverExternalPackages": ["rest-facade", "jose"],
        });
        assert_eq!(
            server_external_packages(Some(&eval)),
            vec!["rest-facade".to_string(), "jose".to_string()],
        );
        assert!(server_external_packages(None).is_empty());
        assert!(server_external_packages(Some(&serde_json::json!({}))).is_empty());
    }

    /// REGRESSION (#51). The real-404 document must be selected by an explicit FLAG on
    /// the request context, never by a magic pathname handed to the ordinary router.
    ///
    /// The orchestrator used to ask for it by rendering the pathname
    /// `/__diffpack_notfound__`, and the react-server entry had no case for that string
    /// at all — it just happened to match nothing in a small app. cal.com has a catch-all
    /// `app/[user]/page.tsx`, which matches EVERY pathname including that sentinel, so the
    /// "not-found" render rendered the catch-all page; that page threw notFound() again
    /// and the 404 was served with an errored, completely empty body.
    #[test]
    fn the_not_found_document_is_requested_by_flag_and_a_catch_all_cannot_capture_it() {
        let dir = scratch("not-found-flag");
        let app = dir.join("app");
        std::fs::create_dir_all(app.join("[...slug]")).unwrap();
        std::fs::write(
            app.join("layout.tsx"),
            "export default function L({children}){return children;}\n",
        )
        .unwrap();
        std::fs::write(
            app.join("not-found.tsx"),
            "export default function NF(){return <p>gone</p>;}\n",
        )
        .unwrap();
        std::fs::write(
            app.join("[...slug]/page.tsx"),
            "import { notFound } from \"next/navigation\";\nexport default function P(){ notFound(); }\n",
        )
        .unwrap();
        let disc = discover_routes(&app, Some(&app.join("layout.tsx"))).unwrap();
        assert!(
            disc.app_not_found.is_some(),
            "the fixture has an app/not-found.tsx"
        );
        let source = rsc_entry_module(
            &disc,
            &UnbuiltPatterns::default(),
            &crate::next_font::FontOutput::default(),
            &dir.join("error-boundary.tsx"),
            &dir.join("segment-boundary.tsx"),
            &dir.join("control-boundary.tsx"),
            &dir.join("request-context.ts"),
            None,
            "",
        );
        // The flag short-circuits BEFORE any route matching, so the catch-all above can
        // never be what a not-found render renders.
        assert!(
            source.contains(
                "if (opts && opts.notFound) return { tree: notFoundTree(), status: 404, params: {} };",
            ),
            "the entry renders the not-found tree from the flag, before matchRoute: {source}",
        );
        // The flag reaches BOTH `matchParams` and `documentTree` through one definition,
        // so the store cannot end up carrying a catch-all's params for a 404 document.
        assert!(
            source.contains("function renderOpts(reqCtx) {")
                && source.contains(
                    "return { softNav: !!reqCtx.softNav, notFound: !!reqCtx.notFound, searchParams: requestSearchParams(reqCtx) };"
                ),
            "one definition of the render options: {source}",
        );
        assert_eq!(
            source
                .matches("matchParams(pathname, renderOpts(reqCtx))")
                .count(),
            2,
            "both render entries resolve params through renderOpts: {source}",
        );
        assert!(
            source.contains("if (opts && opts.notFound) return {};"),
            "a not-found document has no route params: {source}",
        );
        // And the sentinel pathname is gone from BOTH sides of the seam.
        assert!(
            !source.contains("__diffpack_notfound__"),
            "no magic not-found pathname in the react-server entry: {source}",
        );
        assert!(
            !NEXT_SERVER_MJS.contains("__diffpack_notfound__"),
            "no magic not-found pathname in the orchestrator",
        );
    }

    /// The orchestrator must ask for the not-found document at the REQUESTED pathname
    /// with `notFound: true`, and must refuse to serve the result if the render answers
    /// with another notFound()/redirect() — that means the entry ignored the flag and we
    /// would be about to serve some other route's document under a 404.
    #[test]
    fn the_orchestrator_asks_for_the_not_found_document_by_flag_and_verifies_the_reply() {
        let block = NEXT_SERVER_MJS
            .split_once("if (meta.notFound) {")
            .expect("the orchestrator has a notFound branch")
            .1
            .split_once("res.end(nfDoc);")
            .expect("the notFound branch ends by writing the document")
            .0;
        assert!(
            block.contains("[\"render\", url.pathname, clientManifestPath]")
                && block.contains("JSON.stringify({ ...reqCtxObj, notFound: true })"),
            "the not-found render is asked for by flag at the requested pathname: {block}",
        );
        assert!(
            block.contains("if (nf.notFound || nf.redirect) {")
                && block.contains("throw new Error("),
            "a not-found render that itself signals control flow is a hard error: {block}",
        );
        assert!(
            block.contains("res.writeHead(404, nfHeaders)"),
            "the not-found document is served 404: {block}",
        );
    }

    /// REGRESSION. Next's shell is everything NOT inside a `<Suspense>`, and the HTTP
    /// status is decided once the shell is complete. A `loading.tsx` on the route puts the
    /// page INSIDE a Suspense boundary: Next flushes the loading fallback and answers 200,
    /// and a redirect() the page throws afterwards travels in the stream.
    ///
    /// diffpack awaited the matched page unconditionally, which manufactured a 307 Next
    /// never sends — cal.com `/event-types` logged out is exactly this route shape (Next
    /// 200 + skeleton, diffpack 307) — and, worse, defeated `loading.tsx` on every route
    /// that has one by blocking the shell on the page.
    #[test]
    fn the_page_is_resolved_eagerly_only_when_it_is_part_of_the_shell() {
        let dir = scratch("shell-eager-resolve");
        let app = dir.join("app");
        std::fs::create_dir_all(app.join("slow")).unwrap();
        std::fs::write(
            app.join("layout.tsx"),
            "export default function L({children}){return children;}\n",
        )
        .unwrap();
        std::fs::write(
            app.join("page.tsx"),
            "export default function P(){return <p>root</p>;}\n",
        )
        .unwrap();
        std::fs::write(
            app.join("slow/loading.tsx"),
            "export default function Loading(){return <p>...</p>;}\n",
        )
        .unwrap();
        std::fs::write(
            app.join("slow/page.tsx"),
            "import { redirect } from \"next/navigation\";\n\
             export default async function P(){ await Promise.resolve(); redirect(\"/login\"); }\n",
        )
        .unwrap();
        let disc = discover_routes(&app, Some(&app.join("layout.tsx"))).unwrap();
        // The fixture really does carry a loading boundary on `/slow` and none on `/`.
        let slow = disc
            .routes
            .iter()
            .find(|r| {
                r.segments
                    .iter()
                    .any(|s| matches!(s, Seg::Static(name) if name == "slow"))
            })
            .expect("the /slow route was discovered");
        assert!(
            slow.levels.iter().any(|l| l.loading.is_some()),
            "the /slow route carries a loading.tsx level",
        );
        let source = rsc_entry_module(
            &disc,
            &UnbuiltPatterns::default(),
            &crate::next_font::FontOutput::default(),
            &dir.join("error-boundary.tsx"),
            &dir.join("segment-boundary.tsx"),
            &dir.join("control-boundary.tsx"),
            &dir.join("request-context.ts"),
            None,
            "",
        );
        assert!(
            source.contains("const pageInShell = !route.levels.some((level) => level.loading);"),
            "the shell test is the presence of a loading.tsx on the route: {source}",
        );
        assert!(
            source.contains("let node = pageInShell\n    ? await resolvePage(route.page, pageProps, control || {})\n    : createElement(route.page, pageProps);"),
            "only a shell page is awaited before the flight render starts: {source}",
        );
    }

    /// A redirect() thrown once the shell has flushed cannot become a 307, so it has to be
    /// completed on the CLIENT — that is what Next's RedirectBoundary does and what the
    /// `pageInShell` rule above makes routine. The boundary wraps the PAGE (inside every
    /// layout/loading/error boundary), and hands anything that is not a control digest
    /// back to the app's own error boundaries untouched.
    #[test]
    fn a_client_control_boundary_completes_a_redirect_that_reaches_the_browser() {
        let boundary = control_boundary_module();
        assert!(
            boundary.starts_with("\"use client\";"),
            "the control boundary is a client island"
        );
        assert!(
            boundary.contains("window.__diffpack_navigate")
                && boundary.contains("window.location.replace(target.href)")
                && boundary.contains("window.location.assign(target.href)"),
            "it navigates through the Router when present and the browser otherwise: {boundary}",
        );
        // REGRESSION. React reuses a boundary instance that lands in the same position of
        // the NEXT tree, so a boundary left in its error state renders the target route as
        // nothing: cal.com `/settings` reached `/auth/login` and showed a blank document.
        assert!(
            boundary.contains(") => this.setState({ error: null }),"),
            "the boundary clears its error once the Router has swapped: {boundary}",
        );
        // And catching the same target twice must escalate, never loop.
        assert!(
            boundary.contains("if (this.redirectedTo === target.href) {")
                && boundary.contains("this.redirectedTo = target.href;"),
            "a repeated catch of the same target escalates to a real navigation: {boundary}",
        );
        assert!(
            boundary.contains("    // Not ours: hand it to the app's own error boundaries unchanged.\n    throw error;"),
            "a non-control error is re-thrown to the app's error.tsx chain: {boundary}",
        );
        // The digest carries the target between `;` separators and the URL itself may
        // contain one, so the parse must rejoin the middle — a naive `split(";")[2]`
        // truncates `?a=1;b=2` and redirects to the wrong place. Run the real function.
        let dir = scratch("control-boundary-digest");
        let driver = dir.join("driver.mjs");
        let start = boundary
            .find("export function redirectFromDigest")
            .expect("the digest parser is exported");
        let end = boundary
            .find("// notFound() thrown after")
            .expect("the parser block ends");
        std::fs::write(
            &driver,
            format!(
                "{}\nconsole.log(JSON.stringify([\
                 redirectFromDigest(\"NEXT_REDIRECT;replace;/auth/login;307;\"),\
                 redirectFromDigest(\"NEXT_REDIRECT;push;/x?a=1;b=2;308;\"),\
                 redirectFromDigest(\"NEXT_HTTP_ERROR_FALLBACK;404\"),\
                 redirectFromDigest(undefined)]));\n",
                &boundary[start..end],
            ),
        )
        .unwrap();
        let out = std::process::Command::new("node")
            .arg(&driver)
            .output()
            .unwrap();
        assert!(
            out.status.success(),
            "the digest parser runs: {}",
            String::from_utf8_lossy(&out.stderr),
        );
        assert_eq!(
            String::from_utf8_lossy(&out.stdout).trim(),
            r#"[{"href":"/auth/login","type":"replace"},{"href":"/x?a=1;b=2","type":"push"},null,null]"#,
            "the redirect target survives a `;` in the URL and only redirect digests match",
        );
    }

    /// The boundary has to be WRAPPED AROUND THE PAGE by the react-server composition —
    /// both on the ordinary route path and on a parallel/intercepting slot route, or a
    /// redirect from a slot page renders nothing at all.
    #[test]
    fn the_control_boundary_wraps_every_composed_page() {
        let dir = scratch("control-boundary-composition");
        let app = dir.join("app");
        std::fs::create_dir_all(&app).unwrap();
        std::fs::write(
            app.join("layout.tsx"),
            "export default function L({children}){return children;}\n",
        )
        .unwrap();
        std::fs::write(
            app.join("page.tsx"),
            "export default function P(){return <p>hi</p>;}\n",
        )
        .unwrap();
        let disc = discover_routes(&app, Some(&app.join("layout.tsx"))).unwrap();
        let source = rsc_entry_module(
            &disc,
            &UnbuiltPatterns::default(),
            &crate::next_font::FontOutput::default(),
            &dir.join("error-boundary.tsx"),
            &dir.join("segment-boundary.tsx"),
            &dir.join("control-boundary.tsx"),
            &dir.join("request-context.ts"),
            None,
            "",
        );
        assert!(
            source.contains("const CONTROL_BOUNDARY = M"),
            "the island is interned: {source}"
        );
        assert!(
            source.contains("node = createElement(CONTROL_BOUNDARY, null, node);"),
            "documentTree wraps the page in the control boundary: {source}",
        );
        assert!(
            source.contains(
                "createElement(\n    CONTROL_BOUNDARY,\n    null,\n    createElement(page, {"
            ),
            "composeLevels (slot/intercept routes) wraps its page too: {source}",
        );
        // REGRESSION. A boundary only around the page is NOT enough: React's flight writer
        // errors the whole ROW a throw happened in, and an async LAYOUT owns that row, so a
        // boundary nested inside it is destroyed with it and never renders. cal.com
        // `/settings/my-account/profile` proved it — the redirect row arrived in the flight,
        // the boundary was in the tree, and the browser still sat on the settings page. A
        // boundary per level, outside each layout, is what survives to act.
        let loop_body = source
            .split_once("for (let i = route.levels.length - 1; i >= 0; i -= 1) {")
            .expect("documentTree composes level by level")
            .1
            .split_once("\n  }\n")
            .unwrap()
            .0;
        assert!(
            loop_body.contains("node = createElement(CONTROL_BOUNDARY, null, node);"),
            "every level is wrapped, outside its layout: {loop_body}",
        );
        let after_layout = loop_body
            .rsplit_once("SEGMENT_BOUNDARY,")
            .expect("the layout wrap is in the loop")
            .1;
        assert!(
            after_layout.contains("node = createElement(CONTROL_BOUNDARY, null, node);"),
            "the level's control boundary sits OUTSIDE its layout: {after_layout}",
        );
    }

    /// The orchestrator's "redirect after the shell flushed" report must fire ONLY for a
    /// redirect that appeared after the meta went out. A redirect the meta CARRIED was
    /// turned into a real 307 — reporting that one made every redirecting route log a
    /// scary "the response was already streamed and cannot be changed" line that was
    /// simply false.
    #[test]
    fn the_late_control_report_distinguishes_a_redirect_that_was_honoured() {
        let dir = scratch("late-control-report");
        let app = dir.join("app");
        std::fs::create_dir_all(&app).unwrap();
        std::fs::write(
            app.join("layout.tsx"),
            "export default function L({children}){return children;}\n",
        )
        .unwrap();
        std::fs::write(
            app.join("page.tsx"),
            "export default function P(){return <p>hi</p>;}\n",
        )
        .unwrap();
        let disc = discover_routes(&app, Some(&app.join("layout.tsx"))).unwrap();
        let source = rsc_entry_module(
            &disc,
            &UnbuiltPatterns::default(),
            &crate::next_font::FontOutput::default(),
            &dir.join("error-boundary.tsx"),
            &dir.join("segment-boundary.tsx"),
            &dir.join("control-boundary.tsx"),
            &dir.join("request-context.ts"),
            None,
            "",
        );
        assert!(
            source.contains("metaControl = { redirect: control.redirect, notFound: control.notFound };")
                && source.contains(
                    "const lateControl = !!((control.redirect && !metaControl.redirect) || (control.notFound && !metaControl.notFound));",
                ),
            "the stream end reports whether the control flow was LATE: {source}",
        );
        assert!(
            source.contains("lateControl, tags:"),
            "lateControl travels on the stream-end message: {source}",
        );
        assert!(
            NEXT_SERVER_MJS
                .contains("if (m && m.metaSent && m.lateControl && (m.redirect || m.notFound)) {"),
            "the orchestrator reports only a LATE redirect/notFound",
        );
    }

    /// A `redirect()`/`notFound()` thrown BEHIND a Suspense boundary is still a real
    /// 307/404 when the response is BUFFERED, because nothing has been sent yet.
    ///
    /// It reaches the orchestrator only on the stream's END meta, which for a streamed
    /// response is genuinely too late — the shell is already on the wire. The buffered dev
    /// path drains the whole flight before it renders a document, so acting on it there is
    /// both possible and required: cal.com's logged-out `/settings/my-account/profile`
    /// redirects from behind a boundary, and serving the errored flight instead answered 200
    /// with a broken document where the reference answers 307. React then reported the
    /// aborted render's component with no end time, which surfaced in the dev overlay as
    /// "Performance.measure: Given attribute end cannot be negative" — collateral damage
    /// that sent the first look at this bug in entirely the wrong direction.
    #[test]
    fn a_late_redirect_is_still_honoured_when_the_document_is_buffered() {
        assert!(
            NEXT_SERVER_MJS.contains("if (endMeta && endMeta.redirect) {"),
            "the buffered path turns a late redirect into a real 307",
        );
        assert!(
            NEXT_SERVER_MJS.contains("if (endMeta && endMeta.notFound) {"),
            "and a late notFound into a real 404 document",
        );
        // Both must be checked AFTER the flight is drained (that is what makes them
        // actionable) and BEFORE the document is rendered.
        let drained = NEXT_SERVER_MJS
            .find("for await (const b64 of flightChunks()) parts.push(")
            .expect("the buffered path drains the flight");
        let redirect = NEXT_SERVER_MJS
            .find("if (endMeta && endMeta.redirect) {")
            .expect("checked");
        let render = NEXT_SERVER_MJS[drained..]
            .find("getRenderFlightToDocument())(")
            .map(|at| at + drained)
            .expect("then renders the document");
        assert!(
            drained < redirect && redirect < render,
            "the late-control check sits between draining the flight and rendering the document",
        );
    }

    /// The real-404 document is a DOCUMENT: Next resolves its `<head>` from the root
    /// layout's metadata plus `not-found.tsx`'s own `metadata`/`generateMetadata`, which is
    /// where cal.com sets the `404: This page could not be found.` title and
    /// `robots: noindex`. diffpack's not-found tree carried a body and no metadata at all,
    /// so every 404 was served with no <title> and no robots directive.
    #[test]
    fn the_not_found_document_resolves_its_own_metadata_chain() {
        let dir = scratch("not-found-metadata");
        let app = dir.join("app");
        std::fs::create_dir_all(&app).unwrap();
        std::fs::write(
            app.join("layout.tsx"),
            "export const metadata = { title: { template: \"%s | Site\", default: \"Site\" } };\n\
             export default function L({children}){return <html><body>{children}</body></html>;}\n",
        )
        .unwrap();
        std::fs::write(
            app.join("page.tsx"),
            "export default function P(){return <p>hi</p>;}\n",
        )
        .unwrap();
        std::fs::write(
            app.join("not-found.tsx"),
            "export const metadata = { title: \"404\", robots: { index: false } };\n\
             export default function NF(){return <p>gone</p>;}\n",
        )
        .unwrap();
        let disc = discover_routes(&app, Some(&app.join("layout.tsx"))).unwrap();
        let source = rsc_entry_module(
            &disc,
            &UnbuiltPatterns::default(),
            &crate::next_font::FontOutput::default(),
            &dir.join("error-boundary.tsx"),
            &dir.join("segment-boundary.tsx"),
            &dir.join("control-boundary.tsx"),
            &dir.join("request-context.ts"),
            None,
            "",
        );
        // Both metadata-carrying modules are imported as namespaces and wired into the
        // 404's own chain, root layout first so its title template applies.
        assert!(
            source.contains(r#"const NOT_FOUND_ROUTE = { path: "/_not-found", metaChain: [NS"#),
            "the not-found pseudo-route carries a metadata chain: {source}",
        );
        let decl = source
            .split_once("const NOT_FOUND_ROUTE = ")
            .expect("NOT_FOUND_ROUTE is declared")
            .1
            .split_once('\n')
            .unwrap()
            .0;
        assert!(
            !decl.contains("metaChain: [null]") && !decl.contains("pageMeta: null"),
            "both the root layout and not-found.tsx contribute metadata: {decl}",
        );
        assert!(
            decl.contains("title: ROOT_META.title")
                && decl.contains("description: ROOT_META.description"),
            "the statically scanned root title/description remain the fallback: {decl}",
        );
        // notFoundTree renders that chain through the SAME MetadataHead a matched route
        // uses, and emits the link/font head items with no second <title>.
        assert!(
            source.contains("createElement(MetadataHead, { route: NOT_FOUND_ROUTE, params: {} })"),
            "the 404 head is resolved by MetadataHead: {source}",
        );
        let tree = source
            .split_once("function notFoundTree() {")
            .expect("notFoundTree is generated")
            .1
            .split_once("\n}")
            .unwrap()
            .0;
        assert!(
            tree.contains("...headItems({})") && !tree.contains("headItems(ROOT_META)"),
            "the 404 takes its title from the resolved chain, not a second static one: {tree}",
        );
    }

    /// A redirect()/notFound() that reaches the browser (thrown behind a Suspense boundary,
    /// so the status was already sent) is CONTROL FLOW. React recovers from it and reports
    /// it through `onRecoverableError`, whose default handler calls `reportError` — which
    /// surfaces as an UNCAUGHT page error. cal.com `/event-types` logged out did exactly
    /// that: the redirect worked and the browser still logged a React #520.
    ///
    /// React wraps the original error (`new Error(<code>, { cause })`) before recovering, so
    /// the digest is not on the error React hands over — the `cause` chain has to be walked.
    #[test]
    fn a_control_flow_error_recovered_by_react_is_not_reported_as_a_page_error() {
        let dir = scratch("recoverable-control-flow");
        let hooks = dir.join("hooks-context.ts");
        let client = client_entry_module(
            &dir,
            &[dir.join("Island.tsx")],
            &BTreeSet::new(),
            &hooks,
            PinKind::StaticRequire,
        );
        assert!(
            client.contains("onRecoverableError(error, errorInfo) {")
                && client.contains(
                    "if (isControlFlowError(error) || isControlFlowError(errorInfo)) return;"
                )
                && client.contains("      reportError(error);"),
            "hydrateRoot filters control flow out of the recoverable-error report: {client}",
        );
        // ONE definition of the predicate, imported from the control boundary — the error
        // boundary reads the SAME one, so they cannot disagree about what control flow is.
        assert!(
            client.contains("import { isControlFlowError } from \"")
                && client.contains("control-boundary.tsx\";"),
            "the client entry imports the shared predicate: {client}",
        );
        let error_boundary = error_boundary_module(&dir.join("control-boundary.tsx"));
        assert!(
            error_boundary.contains("import { isControlFlowError } from \"")
                && error_boundary.contains("if (isControlFlowError(error)) throw error;"),
            "error.tsx never swallows a redirect/notFound: {error_boundary}",
        );
        // Walking the cause chain is the whole point — assert it by RUNNING the predicate.
        let boundary = control_boundary_module();
        let start = boundary
            .find("export function redirectFromDigest")
            .expect("the digest parser is exported");
        let end = boundary
            .find("export default class ControlBoundary")
            .expect("the predicate block ends before the component");
        let driver = dir.join("driver.mjs");
        std::fs::write(
            &driver,
            format!(
                "{}\nconst wrap = (e) => Object.assign(new Error(\"react\"), {{ cause: e }});\n\
                 const redirect = Object.assign(new Error(\"x\"), {{ digest: \"NEXT_REDIRECT;replace;/auth/login;307;\" }});\n\
                 const nf = Object.assign(new Error(\"x\"), {{ digest: \"NEXT_HTTP_ERROR_FALLBACK;404\" }});\n\
                 const plain = new Error(\"a real bug\");\n\
                 const cycle = new Error(\"cycle\"); cycle.cause = cycle;\n\
                 console.log(JSON.stringify([\
                 isControlFlowError(redirect), isControlFlowError(wrap(redirect)), \
                 isControlFlowError(wrap(wrap(nf))), isControlFlowError(plain), \
                 isControlFlowError(wrap(plain)), isControlFlowError(undefined), \
                 isControlFlowError(cycle), \
                 redirectFromDigest(\"NEXT_REDIRECT;push;/x?a=1;b=2;308;\").href]));\n",
                &boundary[start..end],
            ),
        )
        .unwrap();
        let out = std::process::Command::new("node")
            .arg(&driver)
            .output()
            .unwrap();
        assert!(
            out.status.success(),
            "the predicate runs: {}",
            String::from_utf8_lossy(&out.stderr),
        );
        assert_eq!(
            String::from_utf8_lossy(&out.stdout).trim(),
            r#"[true,true,true,false,false,false,false,"/x?a=1;b=2"]"#,
            "control flow is recognised through the cause chain, a real error still reports, \
             a self-referential cause terminates, and a `;` in the URL survives the parse",
        );
    }
}
