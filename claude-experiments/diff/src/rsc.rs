//! React Server Components (RSC) — foundational directive detection.
//!
//! RSC and server actions are built on two module-level string directives that
//! sit in the directive prologue (before any other statement), exactly like
//! `"use strict"`:
//!
//! * **`"use client"`** marks a CLIENT BOUNDARY. In the React Server (RSC) graph,
//!   importing such a module must NOT pull its code onto the server — instead the
//!   server sees a *client reference* (a stable id + the client chunk that hosts
//!   the real module), which it serializes into the RSC payload; the client bundle
//!   contains the real module and resolves the reference on hydration.
//! * **`"use server"`** marks SERVER ACTIONS. Every export of such a module (or a
//!   function carrying the directive) becomes a server action: the client build
//!   gets a thin RPC stub keyed by a stable id, the server build keeps the real
//!   implementation and registers it under that id. This is the exact shape
//!   [`crate::server_fn`] already implements for TanStack's `createServerFn`,
//!   generalized to the bare directive.
//!
//! This module is the ATOM the rest of RSC support builds on: robustly detecting
//! which directive (if any) a module declares. The boundary transforms and the
//! client/server reference manifests are later slices (see docs/RSC_PLAN.md); they
//! all begin by asking this function what a module is.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use oxc_allocator::Allocator;
use oxc_ast::ast::{Declaration, Expression, ImportDeclarationSpecifier, Statement};
use oxc_ast_visit::{Visit, walk};
use oxc_parser::Parser;
use oxc_span::{GetSpan, Span};

use crate::server_fn::apply_edits;

/// The RSC module-level directive a source file declares, if any.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RscDirective {
    /// `"use client"` — a client boundary (server sees a client reference).
    Client,
    /// `"use server"` — server actions (client sees RPC stubs).
    Server,
    /// `"use cache"` — a cached component/function boundary (Next's Dynamic IO cache
    /// directive). diffpack does not yet implement the `"use cache"` transform slice;
    /// it is detected ONLY so the build can HARD-ERROR rather than silently drop the
    /// directive (which would mask a real caching-semantics gap, per the repo
    /// no-silent-stub rule). See [`crate::transform`] for the erroring path.
    Cache,
}

impl RscDirective {
    /// The canonical directive text.
    pub fn as_str(self) -> &'static str {
        match self {
            RscDirective::Client => "use client",
            RscDirective::Server => "use server",
            RscDirective::Cache => "use cache",
        }
    }
}

/// Detects a module's RSC directive from its directive prologue.
///
/// Only a string literal in the prologue (the leading run of bare string-literal
/// statements, per the ECMAScript "Directive Prologue" — so it may follow
/// `"use strict"`, but NOT any real statement, import, or expression) counts. A
/// `"use client"` string that appears later in the body, inside a function, or in
/// a comment is NOT a boundary and is correctly ignored. Whitespace and single vs
/// double quotes are handled by parsing the prologue rather than string-matching.
///
/// Gated on a cheap substring pre-check so non-RSC modules never pay a parse.
pub fn detect_directive(path: &Path, source: &str) -> Option<RscDirective> {
    if !source.contains("use client")
        && !source.contains("use server")
        && !source.contains("use cache")
    {
        return None;
    }
    let allocator = Allocator::default();
    let source_type = crate::parser::scan_source_type(path);
    let parsed = Parser::new(&allocator, source, source_type).parse();
    for directive in &parsed.program.directives {
        // `directive.directive` is the COOKED value (escapes resolved), so a
        // `"use client"` in any quote style is matched, while `"use client"`
        // resolves correctly too.
        match directive.directive.as_str() {
            "use client" => return Some(RscDirective::Client),
            "use server" => return Some(RscDirective::Server),
            "use cache" => return Some(RscDirective::Cache),
            _ => {}
        }
    }
    None
}

/// The react-server-dom module reference id for a `"use client"` (or
/// `"use server"`) module: the stable string that keys both the server-side
/// reference `$$id` (`"<moduleId>#<export>"`) and the client-references manifest.
/// The canonical absolute path is used verbatim (react-server-dom treats the id as
/// an opaque bundler string), so the server transform, the manifest, and the
/// client's `resolveClientReference` all agree on the same module.
pub fn module_reference_id(path: &Path) -> String {
    std::fs::canonicalize(path)
        .unwrap_or_else(|_| path.to_path_buf())
        .to_string_lossy()
        .into_owned()
}

/// Transforms a `"use client"` module for the REACT SERVER (RSC) build into its
/// client-reference surface, matching what `react-server-dom-webpack`'s
/// node-register produces (`module.exports = createClientModuleProxy(id)`): none
/// of the real component code runs on the server; each export becomes a client
/// reference the server serializes into the flight payload, and the client
/// resolves through the manifest.
///
/// Diffpack emits explicit ESM re-exports off the proxy (rather than replacing
/// `module.exports`) so the module keeps a static export surface the bundler's
/// reachability + the client manifest can enumerate. Returns `None` for a module
/// without the `"use client"` directive (the caller uses the source unchanged).
///
/// The proxy is imported from `react-server-dom-webpack/server`; the app supplies
/// it (it is a peer of using RSC at all), exactly as the reference plugin assumes.
///
/// The module's STYLESHEET imports are carried over as side-effect imports. A client
/// component's CSS belongs to the document, not to its client code (Next collects it
/// into the route's stylesheet so the first paint is already styled), and the
/// react-server graph is where the served stylesheet is compiled from — dropping
/// them with the body would silently delete the app's CSS from `server.css`, and with
/// it from the `/rsc.css` the document head links.
/// A `"use client"` module whose export surface cannot be enumerated is a HARD ERROR
/// (`Err`), never a proxy with no exports: such a module type-checks and bundles, then
/// throws `does not provide an export named "default"` on the first request that renders
/// it — a failure that names neither the file nor the reason.
pub fn transform_use_client_server(path: &Path, source: &str) -> Result<Option<String>, String> {
    if detect_directive(path, source) != Some(RscDirective::Client) {
        return Ok(None);
    }
    let exports = module_exports(path, source);
    if exports.is_empty() {
        return Err(format!(
            "{}: a \"use client\" module with no exports diffpack can enumerate.\n  \
             The react-server build replaces this module with client references, one per \
             export, so a module with none is a client boundary nothing can import.\n  \
             Add an `export` (or, if this is CommonJS, assign to `exports.<name>` / \
             `module.exports = {{ … }}` so the export surface is statically visible).",
            path.display()
        ));
    }
    let unrepresentable: Vec<&String> = exports
        .iter()
        .filter(|export| *export != "default" && !is_reexportable_identifier(export))
        .collect();
    if !unrepresentable.is_empty() {
        return Err(format!(
            "{}: \"use client\" module exports a name that is not a valid ESM binding: {}.\n  \
             The react-server build re-exports each export as a client reference, which \
             requires an identifier name.",
            path.display(),
            unrepresentable
                .iter()
                .map(|name| format!("{name:?}"))
                .collect::<Vec<_>>()
                .join(", ")
        ));
    }
    let id = module_reference_id(path);
    let mut out = String::new();
    for specifier in stylesheet_imports(path, source) {
        out.push_str(&format!("import {};\n", json_string(&specifier)));
    }
    out.push_str("import { createClientModuleProxy } from \"react-server-dom-webpack/server\";\n");
    out.push_str(&format!(
        "const __diffpack_client = createClientModuleProxy({});\n",
        json_string(&id)
    ));
    for export in &exports {
        if export == "default" {
            out.push_str("export default __diffpack_client.default;\n");
        } else {
            out.push_str(&format!(
                "export const {export} = __diffpack_client[{}];\n",
                json_string(export)
            ));
        }
    }
    Ok(Some(out))
}

/// The local binding a `"use cache"` module's default export is renamed to before it
/// is wrapped in the cache boundary (so the wrapped value is re-exported as `default`
/// without a naming collision with any module-local binding).
const USE_CACHE_DEFAULT_LOCAL: &str = "__diffpack_uc_default";

/// The name the `"use cache"` transform imports [`next_adapter`](crate::next_adapter)'s
/// `__diffpackUseCache` cache-boundary helper under (from the aliased `next/cache`
/// shim). The wrapper memoizes the export keyed by its arguments, runs it inside the
/// `cacheTag`/`cacheLife` collection scope, and propagates the collected tags onto the
/// current request store so a tagged page is bustable by `revalidateTag`.
const USE_CACHE_HELPER_LOCAL: &str = "__diffpack_uc";

/// Transforms a `"use cache"` module for the REACT SERVER (RSC) build into a set of
/// CACHED exports. The directive marks every export of the module as a cached async
/// function/component (Next's Dynamic-IO `"use cache"` file directive): each export's
/// return is memoized (keyed by its arguments), the body runs inside a `cacheTag()` /
/// `cacheLife()` collection scope, and the collected tags are recorded on the current
/// request so the page that read the cached value is registered under them for
/// `revalidateTag` — natively reimplemented on diffpack's existing next/cache tag
/// registry + prerender-cache invalidation.
///
/// The original bodies are kept VERBATIM (unlike `"use client"`, whose code must not
/// reach the server). The transform strips `export` off each local declaration, then
/// re-exports a `__diffpackUseCache`-wrapped view of the same binding, so recursion and
/// intra-module references resolve to the real (un-memoized) local exactly as before.
///
/// Returns `Ok(None)` for a module that is NOT a `"use cache"` boundary (caller uses
/// the source unchanged). Hard-errors (naming the construct) on `export ... from` /
/// `export *` re-exports, whose bindings are not local and so cannot be wrapped here.
pub fn transform_use_cache_server(path: &Path, source: &str) -> Result<Option<String>, String> {
    if detect_directive(path, source) != Some(RscDirective::Cache) {
        return Ok(None);
    }
    let allocator = Allocator::default();
    let source_type = crate::parser::scan_source_type(path);
    let parsed = Parser::new(&allocator, source, source_type).parse();
    let program = &parsed.program;
    let module_id = module_reference_id(path);

    // (local, exported) pairs to wrap in the footer, and the in-place source edits that
    // demote each `export` declaration to a plain local (or rename the default value to
    // a local const) so the footer can re-export the wrapped view without a collision.
    let mut wrapped: Vec<(String, String)> = Vec::new();
    let mut edits: Vec<(Span, String)> = Vec::new();

    for statement in &program.body {
        match statement {
            Statement::ExportNamedDeclaration(export) => {
                // `export type { ... }` / `export interface` carry no runtime binding —
                // types are erased in this graph, so leave them entirely untouched.
                if export.export_kind.is_type() {
                    continue;
                }
                if export.source.is_some() {
                    return Err(format!(
                        "'use cache' module {} re-exports with `export ... from`; a re-exported \
                         binding is not local and cannot be wrapped in a cache boundary. Export \
                         the cached function directly from this module.",
                        path.display()
                    ));
                }
                if let Some(declaration) = &export.declaration {
                    // Collect the declared name(s), then strip the leading `export ` so the
                    // binding stays module-local (the footer re-exports the wrapped view).
                    let mut collected = false;
                    match declaration {
                        Declaration::VariableDeclaration(var) => {
                            for decl in &var.declarations {
                                if let Some(ident) = decl.id.get_binding_identifier() {
                                    let name = ident.name.to_string();
                                    wrapped.push((name.clone(), name));
                                    collected = true;
                                }
                            }
                        }
                        Declaration::FunctionDeclaration(func) => {
                            if let Some(ident) = &func.id {
                                let name = ident.name.to_string();
                                wrapped.push((name.clone(), name));
                                collected = true;
                            }
                        }
                        Declaration::ClassDeclaration(class) => {
                            if let Some(ident) = &class.id {
                                let name = ident.name.to_string();
                                wrapped.push((name.clone(), name));
                                collected = true;
                            }
                        }
                        _ => {}
                    }
                    if collected {
                        // Remove exactly the `export ` keyword+whitespace before the decl.
                        edits.push((
                            Span::new(export.span().start, declaration.span().start),
                            String::new(),
                        ));
                    }
                }
                // `export { a, b as c }` (no declaration): the specifiers reference existing
                // module-local bindings — record them and drop the whole statement (the
                // footer re-exports the wrapped views under the same exported names).
                if export.declaration.is_none() && !export.specifiers.is_empty() {
                    let mut any = false;
                    for specifier in &export.specifiers {
                        if specifier.export_kind.is_type() {
                            continue;
                        }
                        wrapped.push((
                            specifier.local.name().to_string(),
                            specifier.exported.name().to_string(),
                        ));
                        any = true;
                    }
                    if any {
                        edits.push((export.span(), String::new()));
                    }
                }
            }
            Statement::ExportDefaultDeclaration(export) => {
                // Rename the default value to a local const so the footer can wrap it and
                // re-export the wrapped view as `default`. Wrapping the VALUE (not the
                // `export default` keyword) handles an expression, a named or anonymous
                // function/class declaration uniformly (each becomes an expression here).
                let value_span = export.declaration.span();
                let value = &source[value_span.start as usize..value_span.end as usize];
                edits.push((
                    export.span(),
                    format!("const {USE_CACHE_DEFAULT_LOCAL} = ({value});"),
                ));
                wrapped.push((USE_CACHE_DEFAULT_LOCAL.to_string(), "default".to_string()));
            }
            Statement::ExportAllDeclaration(_) => {
                return Err(format!(
                    "'use cache' module {} uses `export * from`; a wildcard re-export cannot be \
                     enumerated into cached references. Export each cached function directly.",
                    path.display()
                ));
            }
            _ => {}
        }
    }

    let import = format!(
        "import {{ __diffpackUseCache as {USE_CACHE_HELPER_LOCAL} }} from \"next/cache\";\n"
    );
    let mut rewritten = apply_edits(source, import, edits);

    // Footer: wrap each export in the cache boundary and re-export it under its original
    // exported name. `<local>` is the real (un-wrapped) binding; the id keys the memo.
    if !wrapped.is_empty() {
        rewritten.push('\n');
        for (index, (local, exported)) in wrapped.iter().enumerate() {
            let wrapped_local = format!("__diffpack_uc_w_{index}");
            let id = format!("{module_id}#{exported}");
            rewritten.push_str(&format!(
                "const {wrapped_local} = {USE_CACHE_HELPER_LOCAL}({local}, {});\n",
                json_string(&id)
            ));
            // `export { x as default }` and `export { x as foo }` are both valid clauses.
            rewritten.push_str(&format!("export {{ {wrapped_local} as {exported} }};\n"));
        }
    }
    Ok(Some(rewritten))
}

/// The subpath the client-side server-action transport (`callServer`) is imported
/// from; diffpack registers [`crate::rsc_runtime`]'s `call_server.js` under it as a
/// virtual module for the client build.
pub const CALL_SERVER_SPECIFIER: &str = "#diffpack-call-server";

/// The subpath the generated action resolver (`getServerActionById`) is imported
/// from; diffpack registers [`generate_action_resolver_module`]'s output under it
/// for the server build.
pub const ACTION_RESOLVER_SPECIFIER: &str = "#diffpack-rsc-action-resolver";

/// The subpath the server-side action dispatcher (`handleServerAction`) is imported
/// from; diffpack registers [`crate::rsc_runtime`]'s `action_handler.js` under it as
/// a virtual module for the server build.
pub const ACTION_HANDLER_SPECIFIER: &str = "#diffpack-rsc-action-handler";

/// The HTTP path the client `callServer` transport POSTs a server-action call to,
/// and the endpoint the emitted server runtime routes to `handleServerAction`.
pub const ACTION_ENDPOINT: &str = "/_action/";

/// The subpath the SSR pass imports the generated SSR consumer manifest
/// (Manifest #2, `serverConsumerManifest`) from; diffpack registers
/// [`ClientReferencesManifest::to_ssr_consumer_manifest_module`]'s output under it
/// for the server build, so `createFromReadableStream` can resolve the client
/// references embedded in a flight stream back to the real (SSR-graph) modules.
pub const SSR_CONSUMER_MANIFEST_SPECIFIER: &str = "#diffpack-rsc-ssr-consumer-manifest";

/// The stable server-reference id for a `"use server"` export named `name` in the
/// module at `path`: `"<moduleId>#<name>"`. The react-server protocol fixes this
/// shape (NOT server_fn's SHA-256): the client stub's `createServerReference` id,
/// the server's `registerServerReference` `$$id`, and the resolver's manifest key
/// all derive it from the same `module_reference_id`, so a client call lands on
/// exactly the handler the server registered — the same cross-graph id-agreement
/// invariant `server_fn` guarantees.
pub fn action_reference_id(path: &Path, name: &str) -> String {
    format!("{}#{}", module_reference_id(path), name)
}

/// A single `local => exported` name pair for a `"use server"` module export, so
/// the server registration can reference the real local binding while keying the
/// id by the name importers see (`export { local as exported }`).
struct ServerExport {
    local: String,
    exported: String,
}

/// Enumerates a `"use server"` module's exports as `(local, exported)` pairs plus
/// the default export's value span (if any). Hard-errors on an `export ... from`
/// re-export, whose binding is not local and so cannot be registered here (never a
/// silent pass-through that would ship an unregisterable server reference).
fn server_exports<'a>(
    path: &Path,
    program: &oxc_ast::ast::Program<'a>,
) -> Result<(Vec<ServerExport>, Option<Span>), String> {
    let mut named = Vec::new();
    let mut default_value: Option<Span> = None;
    for statement in &program.body {
        match statement {
            Statement::ExportNamedDeclaration(export) => {
                if export.source.is_some() {
                    return Err(format!(
                        "'use server' module {} re-exports with `export ... from`; a re-exported \
                         binding is not local and cannot be registered as a server reference. \
                         Export the action directly from this module.",
                        path.display()
                    ));
                }
                if let Some(declaration) = &export.declaration {
                    match declaration {
                        Declaration::VariableDeclaration(var) => {
                            for decl in &var.declarations {
                                if let Some(ident) = decl.id.get_binding_identifier() {
                                    let name = ident.name.to_string();
                                    named.push(ServerExport { local: name.clone(), exported: name });
                                }
                            }
                        }
                        Declaration::FunctionDeclaration(func) => {
                            if let Some(ident) = &func.id {
                                let name = ident.name.to_string();
                                named.push(ServerExport { local: name.clone(), exported: name });
                            }
                        }
                        Declaration::ClassDeclaration(class) => {
                            if let Some(ident) = &class.id {
                                let name = ident.name.to_string();
                                named.push(ServerExport { local: name.clone(), exported: name });
                            }
                        }
                        _ => {}
                    }
                }
                for specifier in &export.specifiers {
                    named.push(ServerExport {
                        local: specifier.local.name().to_string(),
                        exported: specifier.exported.name().to_string(),
                    });
                }
            }
            Statement::ExportDefaultDeclaration(export) => {
                default_value = Some(export.declaration.span());
            }
            Statement::ExportAllDeclaration(_) => {
                return Err(format!(
                    "'use server' module {} uses `export * from`; a wildcard re-export cannot be \
                     enumerated into named server references. Export each action directly.",
                    path.display()
                ));
            }
            _ => {}
        }
    }
    Ok((named, default_value))
}

/// Transforms a `"use server"` module for the REACT-SERVER graph into its
/// server-reference surface: keeps the real bodies verbatim, imports
/// `registerServerReference` from
/// `react-server-dom-webpack/server`, and registers each export under its
/// `action_reference_id`. Named exports are registered by a footer that references
/// the live local binding; the default export's value is wrapped in place. The
/// registered function's `$$id` is `"<moduleId>#<name>"`, matching the client stub
/// and the resolver key.
///
/// Returns `Ok(None)` for any module that is NOT a `"use server"` boundary, so the
/// caller uses the source unchanged. Hard-errors (naming the construct) on
/// re-exports, which cannot be registered locally.
pub fn transform_use_server_server(path: &Path, source: &str) -> Result<Option<String>, String> {
    if detect_directive(path, source) != Some(RscDirective::Server) {
        return Ok(None);
    }
    let allocator = Allocator::default();
    let source_type = crate::parser::scan_source_type(path);
    let parsed = Parser::new(&allocator, source, source_type).parse();
    let program = &parsed.program;
    let (named, default_value) = server_exports(path, program)?;
    let module_id = module_reference_id(path);

    // Wrap the default export's value in-place: `export default <value>` becomes
    // `export default __rsr(<value>, "<moduleId>", "default")`. Wrapping the value
    // (not the `export default` keyword) works uniformly for an expression, a named
    // function/class declaration, and an anonymous one (which becomes a function/
    // class expression in argument position).
    let mut edits: Vec<(Span, String)> = Vec::new();
    if let Some(span) = default_value {
        let value = &source[span.start as usize..span.end as usize];
        edits.push((
            span,
            format!(
                "__rsr({value}, {}, \"default\")",
                json_string(&module_id)
            ),
        ));
    }

    let import =
        "import { registerServerReference as __rsr } from \"react-server-dom-webpack/server\";\n"
            .to_string();
    let mut rewritten = apply_edits(source, import, edits);

    // Footer: register each NAMED export under its id, referencing the live local
    // binding. The id's name is the EXPORTED name (what importers and the client
    // stub key by); the registered value is the LOCAL binding.
    if !named.is_empty() {
        rewritten.push('\n');
        for export in &named {
            rewritten.push_str(&format!(
                "__rsr({}, {}, {});\n",
                export.local,
                json_string(&module_id),
                json_string(&export.exported),
            ));
        }
    }
    Ok(Some(rewritten))
}

/// Transforms a `"use server"` module for the CLIENT build into thin RPC stubs:
/// each export becomes `createServerReference("<moduleId>#<name>", callServer)`, so
/// none of the server-only bodies (or their server-only imports) ship to the
/// browser. The `callServer` transport is imported from [`CALL_SERVER_SPECIFIER`];
/// `createServerReference` from `react-server-dom-webpack/client`. Returns `None`
/// for a module without the `"use server"` directive.
///
/// The emitted id is byte-for-byte the id the server build registers
/// ([`action_reference_id`]), so a stub call round-trips to the real handler.
pub fn transform_use_server_client(path: &Path, source: &str) -> Option<String> {
    if detect_directive(path, source) != Some(RscDirective::Server) {
        return None;
    }
    let exports = module_exports(path, source);
    let module_id = module_reference_id(path);
    let mut out = String::new();
    out.push_str(
        "import { createServerReference } from \"react-server-dom-webpack/client\";\n",
    );
    out.push_str(&format!(
        "import {{ callServer }} from {};\n",
        json_string(CALL_SERVER_SPECIFIER)
    ));
    for export in &exports {
        let id = format!("{module_id}#{export}");
        if export == "default" {
            out.push_str(&format!(
                "export default createServerReference({}, callServer);\n",
                json_string(&id)
            ));
        } else {
            out.push_str(&format!(
                "export const {export} = createServerReference({}, callServer);\n",
                json_string(&id)
            ));
        }
    }
    Some(out)
}

/// A discovered `"use server"` action export: its module path (canonical), the
/// EXPORTED name importers key by, and its `action_reference_id`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ActionEntry {
    pub path: PathBuf,
    pub name: String,
    pub id: String,
}

/// Walks the project at `root` and calls `visit` once per readable module file with
/// its CANONICAL path and its source text, skipping installed dependencies, build
/// output, and VCS metadata.
///
/// This is the ONE project walk every directive/convention scan shares: the
/// `"use server"` action scan below, and the Next adapter's `"use client"` island /
/// `next/font` / stylesheet-import scan. They must see the same file set — a
/// `"use client"` module one scan finds and the other misses is exactly the
/// "Could not find the module ... in the React Client Manifest" class of bug, since
/// the react-server graph resolves real imports while island discovery is a
/// filesystem walk.
///
/// Skipped at ANY depth: `node_modules`, `.git`, and the generated/derived trees
/// (`dist`, `.output`, `.next`, `.diffpack-output`, `.diffpack-next`) — a
/// `"use client"` file inside one of those is a copy of a source file, and pinning
/// the copy registers a second module id for the same component.
/// Skipped only at the PROJECT ROOT ([`PROJECT_ROOT_ONLY_SKIPPED_DIRS`]): the
/// exported/reported trees whose names are ordinary words a source directory may
/// legitimately use further down (`src/build/`, `app/out/`).
pub fn walk_project_modules<F>(root: &Path, visit: &mut F) -> Result<(), String>
where
    F: FnMut(&Path, &str) -> Result<(), String>,
{
    walk_project_modules_inner(root, true, visit)
}

/// Build/report output trees skipped ONLY when they sit directly in the project
/// root. `next build && next export` writes `out/`, other toolchains write `build/`,
/// `vercel build` writes `.vercel/`, coverage and Storybook write `coverage/` and
/// `storybook-static/`. Each holds a COPY of the app's modules, so a stale one would
/// otherwise contribute duplicate `"use client"` islands (a second manifest id for a
/// component that already has one). They are not skipped deeper down because
/// `src/build/` or `lib/out/` is plausibly real source.
const PROJECT_ROOT_ONLY_SKIPPED_DIRS: [&str; 5] =
    ["out", "build", ".vercel", "coverage", "storybook-static"];

fn walk_project_modules_inner<F>(root: &Path, at_project_root: bool, visit: &mut F) -> Result<(), String>
where
    F: FnMut(&Path, &str) -> Result<(), String>,
{
    let read = match std::fs::read_dir(root) {
        Ok(read) => read,
        Err(_) => return Ok(()),
    };
    for entry in read {
        let entry = entry.map_err(|error| format!("cannot read {}: {error}", root.display()))?;
        let path = entry.path();
        let file_type = entry
            .file_type()
            .map_err(|error| format!("cannot stat {}: {error}", path.display()))?;
        if file_type.is_dir() {
            let name = entry.file_name();
            let name = name.to_str();
            if matches!(
                name,
                Some("node_modules" | ".diffpack-output" | ".diffpack-next" | ".git" | "dist" | ".output" | ".next")
            ) {
                continue;
            }
            if at_project_root
                && name.is_some_and(|name| PROJECT_ROOT_ONLY_SKIPPED_DIRS.contains(&name))
            {
                continue;
            }
            walk_project_modules_inner(&path, false, visit)?;
        } else if is_scannable_module(&path) {
            let Ok(source) = std::fs::read_to_string(&path) else {
                continue;
            };
            let canonical = std::fs::canonicalize(&path).unwrap_or(path);
            visit(&canonical, &source)?;
        }
    }
    Ok(())
}

/// Walks `root` (skipping `node_modules`, build output, and VCS dirs) for
/// `"use server"` modules, returning every exported action sorted by id. Only files
/// whose text contains `"use server"` AND whose directive prologue is `"use server"`
/// are enumerated (mirroring the transform's cheap gate). This is what lets the
/// generated action resolver map each id to a real module import.
pub fn scan_project_server_actions(root: &Path) -> Result<Vec<ActionEntry>, String> {
    let mut entries = Vec::new();
    walk_project_modules(root, &mut |path, source| {
        if !source.contains("use server") {
            return Ok(());
        }
        if detect_directive(path, source) != Some(RscDirective::Server) {
            return Ok(());
        }
        for name in module_exports(path, source) {
            let id = action_reference_id(path, &name);
            entries.push(ActionEntry {
                path: path.to_path_buf(),
                name,
                id,
            });
        }
        Ok(())
    })?;
    entries.sort_by(|left, right| left.id.cmp(&right.id));
    entries.dedup();
    Ok(entries)
}

fn is_scannable_module(path: &Path) -> bool {
    matches!(
        path.extension().and_then(|value| value.to_str()),
        Some("ts" | "tsx" | "js" | "jsx" | "mjs" | "cjs")
    )
}

/// Generates the [`ACTION_RESOLVER_SPECIFIER`] module source: a `manifest` mapping
/// each `"<moduleId>#<name>"` id to a dynamic import of its module plus the export
/// name, and `getServerActionById(id)` that resolves the module and returns the
/// real function export (which the server build has registered as a server
/// reference). A missing id, or a non-function export, is a hard, specific error
/// rather than a silent `undefined`.
pub fn generate_action_resolver_module(entries: &[ActionEntry]) -> String {
    let mut source = String::new();
    source
        .push_str("// Generated natively by Diffpack from the app's \"use server\" action exports.\n");
    source.push_str("const manifest = {\n");
    for entry in entries {
        source.push_str(&format!(
            "  {}: {{ importer: () => import({}), exportName: {} }},\n",
            json_string(&entry.id),
            json_string(&entry.path.to_string_lossy()),
            json_string(&entry.name),
        ));
    }
    source.push_str("};\n");
    source.push_str(
        r#"async function getServerActionById(id) {
  const info = manifest[id];
  if (!info) throw new Error("diffpack rsc: no server action registered for id " + id);
  const mod = await info.importer();
  const fn = mod[info.exportName];
  if (typeof fn !== "function")
    throw new Error("diffpack rsc: action id " + id + " is not a function export " + info.exportName);
  return fn;
}
export { getServerActionById };
export default { getServerActionById };
"#,
    );
    source
}

/// The embedded client-side server-action transport (`callServer`), registered as
/// the [`CALL_SERVER_SPECIFIER`] virtual module for the client build.
pub fn call_server_module_source() -> &'static str {
    include_str!("rsc_runtime/call_server.js")
}

/// The embedded server-side action dispatcher (`handleServerAction`), registered as
/// the [`ACTION_HANDLER_SPECIFIER`] virtual module for the server build.
pub fn action_handler_module_source() -> &'static str {
    include_str!("rsc_runtime/action_handler.js")
}

/// Collects a module's export names (named declarations, named export
/// specifiers, and `default`), so the `"use client"` re-export surface matches the
/// original module's exports one-for-one.
fn module_exports(path: &Path, source: &str) -> Vec<String> {
    let allocator = Allocator::default();
    let source_type = crate::parser::scan_source_type(path);
    let parsed = Parser::new(&allocator, source, source_type).parse();
    let mut names = Vec::new();
    let mut push = |name: String| {
        if !names.contains(&name) {
            names.push(name);
        }
    };
    for statement in &parsed.program.body {
        let Statement::ExportNamedDeclaration(export) = statement else {
            if matches!(
                statement,
                Statement::ExportDefaultDeclaration(_)
            ) {
                push("default".to_string());
            }
            continue;
        };
        if let Some(declaration) = &export.declaration {
            match declaration {
                Declaration::VariableDeclaration(var) => {
                    for decl in &var.declarations {
                        if let Some(ident) = decl.id.get_binding_identifier() {
                            push(ident.name.to_string());
                        }
                    }
                }
                Declaration::FunctionDeclaration(func) => {
                    if let Some(ident) = &func.id {
                        push(ident.name.to_string());
                    }
                }
                Declaration::ClassDeclaration(class) => {
                    if let Some(ident) = &class.id {
                        push(ident.name.to_string());
                    }
                }
                _ => {}
            }
        }
        for specifier in &export.specifiers {
            push(specifier.exported.name().to_string());
        }
    }
    // `push` borrows `names` mutably; the borrow ends with the last call above.
    if names.is_empty() {
        // No ESM export syntax at all: the module is (or was compiled to) CommonJS.
        // Published `"use client"` components in node_modules are overwhelmingly CJS
        // — `next/dist/client/script.js` is swc-compiled CJS — and a client proxy with
        // ZERO exports is a module that throws "does not provide an export named
        // default" the first time anything imports it. Recover the export surface
        // from the CJS assignment forms instead.
        let mut collector = CjsExportCollector { names: Vec::new() };
        collector.visit_program(&parsed.program);
        names = collector.names;
    }
    names
}

/// Whether `name` can be written as a bare `export const <name>` / `export { <name> }`
/// binding. A CJS module may export any string key (`exports["a-b"] = …`); such a name
/// has no ESM re-export form here, so it is reported by [`cjs_export_names`]'s caller
/// rather than silently emitted as invalid syntax.
fn is_reexportable_identifier(name: &str) -> bool {
    if name.is_empty() {
        return false;
    }
    let mut chars = name.chars();
    let first = chars.next().unwrap();
    if !(first.is_alphabetic() || first == '_' || first == '$') {
        return false;
    }
    if !chars.all(|c| c.is_alphanumeric() || c == '_' || c == '$') {
        return false;
    }
    // Reserved words cannot be a `const` binding name.
    !matches!(
        name,
        "await"
            | "break"
            | "case"
            | "catch"
            | "class"
            | "const"
            | "continue"
            | "debugger"
            | "default"
            | "delete"
            | "do"
            | "else"
            | "enum"
            | "export"
            | "extends"
            | "false"
            | "finally"
            | "for"
            | "function"
            | "if"
            | "import"
            | "in"
            | "instanceof"
            | "new"
            | "null"
            | "return"
            | "super"
            | "switch"
            | "this"
            | "throw"
            | "true"
            | "try"
            | "typeof"
            | "var"
            | "void"
            | "while"
            | "with"
            | "yield"
    )
}

/// Recovers the export names of a CommonJS module from its assignment forms. Covers
/// every shape a compiler (swc, tsc, babel) emits for `export … ` in CJS output:
///
/// * `exports.NAME = …` / `module.exports.NAME = …`
/// * `module.exports = { NAME: … }` (and swc's `0 && (module.exports = { … })` hint)
/// * `Object.defineProperty(exports, "NAME", …)`
/// * `_export(exports, { NAME: … })` — swc's helper — and `Object.assign(exports, {…})`
///
/// `__esModule` is a marker, never a real export, and is excluded.
struct CjsExportCollector {
    names: Vec<String>,
}

impl CjsExportCollector {
    fn push(&mut self, name: &str) {
        if name == "__esModule" || name.is_empty() {
            return;
        }
        if !self.names.iter().any(|kept| kept == name) {
            self.names.push(name.to_string());
        }
    }

    /// The keys of an object literal, for the `module.exports = {…}` / `_export(exports, {…})`
    /// forms. Spread elements carry no statically-known names and are skipped.
    fn push_object_keys(&mut self, object: &oxc_ast::ast::ObjectExpression<'_>) {
        for property in &object.properties {
            let oxc_ast::ast::ObjectPropertyKind::ObjectProperty(prop) = property else {
                continue;
            };
            if let Some(name) = prop.key.static_name() {
                self.push(&name);
            }
        }
    }
}

/// Whether `expression` is the CJS exports object: the `exports` identifier or the
/// `module.exports` member expression.
fn is_cjs_exports_object(expression: &Expression<'_>) -> bool {
    match expression {
        Expression::Identifier(ident) => ident.name == "exports",
        Expression::StaticMemberExpression(member) => {
            member.property.name == "exports"
                && matches!(&member.object, Expression::Identifier(ident) if ident.name == "module")
        }
        _ => false,
    }
}

impl<'a> Visit<'a> for CjsExportCollector {
    fn visit_assignment_expression(&mut self, assign: &oxc_ast::ast::AssignmentExpression<'a>) {
        match &assign.left {
            // `exports.NAME = …` / `module.exports.NAME = …`, and the whole-object
            // replacement `module.exports = { NAME: … }`.
            oxc_ast::ast::AssignmentTarget::StaticMemberExpression(member) => {
                if is_cjs_exports_object(&member.object) {
                    self.push(member.property.name.as_str());
                } else if member.property.name == "exports"
                    && matches!(&member.object, Expression::Identifier(ident) if ident.name == "module")
                    && let Expression::ObjectExpression(object) = &assign.right
                {
                    self.push_object_keys(object);
                }
            }
            // `exports["NAME"] = …` (a literal key; a computed one is not statically known)
            oxc_ast::ast::AssignmentTarget::ComputedMemberExpression(member) => {
                if is_cjs_exports_object(&member.object)
                    && let Expression::StringLiteral(literal) = &member.expression
                {
                    self.push(literal.value.as_str());
                }
            }
            // `exports = { NAME: … }`
            oxc_ast::ast::AssignmentTarget::AssignmentTargetIdentifier(ident)
                if ident.name == "exports" =>
            {
                if let Expression::ObjectExpression(object) = &assign.right {
                    self.push_object_keys(object);
                }
            }
            _ => {}
        }
        walk::walk_assignment_expression(self, assign);
    }

    fn visit_call_expression(&mut self, call: &oxc_ast::ast::CallExpression<'a>) {
        let args: Vec<&Expression<'a>> = call
            .arguments
            .iter()
            .map(|argument| argument.as_expression())
            .take_while(|argument| argument.is_some())
            .flatten()
            .collect();
        if args.len() >= 2 && is_cjs_exports_object(args[0]) {
            match args[1] {
                // `Object.defineProperty(exports, "NAME", …)`
                Expression::StringLiteral(literal) => self.push(literal.value.as_str()),
                // `_export(exports, { NAME: … })` / `Object.assign(exports, { NAME: … })`
                Expression::ObjectExpression(object) => self.push_object_keys(object),
                _ => {}
            }
        }
        walk::walk_call_expression(self, call);
    }
}

/// One module specifier a source file names, with the fact that decides whether the
/// build must resolve it: a `import type {...} from "./x"` (or `export type ... from`)
/// is ERASED before the bundler ever sees it, so it may legitimately name a `.d.ts`
/// that no resolver can find.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModuleImport {
    /// The specifier exactly as written (`"./x.css"`, `"@/lib/store"`, `"zustand"`).
    pub specifier: String,
    /// Type-only (`import type` / `export type`): erased, never resolved.
    pub type_only: bool,
}

/// EVERY module specifier `source` names, in source order and deduped: static
/// `import`/`export … from`, dynamic `import(...)`, and `require(...)` with a string
/// literal argument — anywhere in the module, not only at the top level.
///
/// This is the ONE specifier extraction the RSC-side scans share, so the stylesheet
/// carry-over ([`stylesheet_imports`]) and the project import graph
/// ([`crate::project_graph`]) cannot disagree about what a module depends on.
/// Non-literal forms (`import(variable)`, `require(name)`) are not specifiers a
/// bundler can follow and are not reported.
pub fn module_import_specifiers(path: &Path, source: &str) -> Vec<ModuleImport> {
    let allocator = Allocator::default();
    let source_type = crate::parser::scan_source_type(path);
    let parsed = Parser::new(&allocator, source, source_type).parse();
    let mut collector = ImportCollector {
        imports: Vec::new(),
    };
    collector.visit_program(&parsed.program);
    collector.imports
}

struct ImportCollector {
    imports: Vec<ModuleImport>,
}

impl ImportCollector {
    fn push(&mut self, specifier: &str, type_only: bool) {
        if let Some(existing) = self
            .imports
            .iter_mut()
            .find(|kept| kept.specifier == specifier)
        {
            // A specifier named both ways is a real (value) dependency.
            existing.type_only &= type_only;
            return;
        }
        self.imports.push(ModuleImport {
            specifier: specifier.to_string(),
            type_only,
        });
    }
}

impl<'a> Visit<'a> for ImportCollector {
    fn visit_import_declaration(&mut self, import: &oxc_ast::ast::ImportDeclaration<'a>) {
        let type_only = import.import_kind.is_type()
            || import.specifiers.as_ref().is_some_and(|specifiers| {
                !specifiers.is_empty()
                    && specifiers.iter().all(|specifier| match specifier {
                        ImportDeclarationSpecifier::ImportSpecifier(named) => {
                            named.import_kind.is_type()
                        }
                        _ => false,
                    })
            });
        self.push(import.source.value.as_str(), type_only);
        walk::walk_import_declaration(self, import);
    }

    fn visit_export_named_declaration(
        &mut self,
        export: &oxc_ast::ast::ExportNamedDeclaration<'a>,
    ) {
        if let Some(source) = &export.source {
            self.push(source.value.as_str(), export.export_kind.is_type());
        }
        walk::walk_export_named_declaration(self, export);
    }

    fn visit_export_all_declaration(&mut self, export: &oxc_ast::ast::ExportAllDeclaration<'a>) {
        self.push(export.source.value.as_str(), export.export_kind.is_type());
        walk::walk_export_all_declaration(self, export);
    }

    fn visit_import_expression(&mut self, import: &oxc_ast::ast::ImportExpression<'a>) {
        if let Expression::StringLiteral(literal) = &import.source {
            self.push(literal.value.as_str(), false);
        }
        walk::walk_import_expression(self, import);
    }

    fn visit_call_expression(&mut self, call: &oxc_ast::ast::CallExpression<'a>) {
        if let Expression::Identifier(callee) = &call.callee
            && callee.name == "require"
            && call.arguments.len() == 1
            && let Some(Expression::StringLiteral(literal)) =
                call.arguments[0].as_expression()
        {
            self.push(literal.value.as_str(), false);
        }
        walk::walk_call_expression(self, call);
    }
}

/// The stylesheet extensions diffpack's loader compiles into the emitted CSS
/// (`bundler::load_special_module`): plain CSS, Sass, Less and Stylus, in both their
/// global and `*.module.*` forms.
const STYLESHEET_EXTENSIONS: [&str; 6] = ["css", "scss", "sass", "less", "styl", "stylus"];

/// Whether a specifier names a stylesheet the build compiles. The loader query and
/// fragment (`./a.css?url`, `./a.css#x`) are not part of the extension.
pub fn is_stylesheet_specifier(specifier: &str) -> bool {
    let path = crate::resource_id::ResourceId::parse(specifier).path;
    matches!(
        Path::new(&path).extension().and_then(|value| value.to_str()),
        Some(extension) if STYLESHEET_EXTENSIONS.contains(&extension)
    )
}

/// The stylesheet specifiers `source` depends on, in source order and deduped. Read
/// by [`transform_use_client_server`], which discards the module body but must keep
/// its stylesheets in the react-server graph — that graph is what `server.css` (and
/// so the served `/rsc.css`) is compiled from, so a stylesheet dropped here is a
/// silently unstyled page.
///
/// Every form [`module_import_specifiers`] reports counts, including
/// `require("./x.css")` and `import("./x.css")`: the module body they sat in is being
/// deleted, so the only way the stylesheet survives at all is as a side-effect import
/// of the proxy. A dynamic import therefore becomes eager here — the styles apply
/// from first paint rather than on demand, which is the same direction Next takes
/// when it hoists a client component's CSS into the route's stylesheet.
fn stylesheet_imports(path: &Path, source: &str) -> Vec<String> {
    module_import_specifiers(path, source)
        .into_iter()
        .filter(|import| !import.type_only && is_stylesheet_specifier(&import.specifier))
        .map(|import| import.specifier)
        .collect()
}

/// JSON-encode a string as a JS string literal for embedding in generated code.
fn json_string(value: &str) -> String {
    serde_json::to_string(value).unwrap_or_else(|_| "\"\"".to_string())
}

/// The file name of the client-references manifest artifact the CLIENT build
/// writes and the react-server render reads as `renderToReadableStream`'s
/// `bundlerConfig` (Manifest #1 in docs/RSC_SPEC.md §1).
pub const CLIENT_REFERENCES_MANIFEST_FILE: &str = "client-references-manifest.json";

/// A server-like build's own client-references manifest file: the runtime ids +
/// hosting chunks it assigns to each `"use client"` module in ITS graph. Distinct
/// from [`CLIENT_REFERENCES_MANIFEST_FILE`] (the browser build's ids) so the
/// react-server render and the SSR builds do not clobber it, and so the SSR-of-flight
/// pass can join client ids (in the flight) with the SSR build's own ids to form the
/// divergent-id `ssrModuleMapping` (Manifest #2's `moduleMap`).
pub const SERVER_REFERENCES_MANIFEST_FILE: &str = "server-references-manifest.json";

/// The REACT-SERVER graph's own client-references manifest file. A production
/// `build-app` run emits the react-server graph and the SSR-of-flight graph into the
/// SAME output root and runs the ssr pass LAST, so under one shared file name the
/// react-server graph's manifest is silently overwritten and its contents lost.
///
/// That content is the one thing no other manifest records: the AUTHORITATIVE set of
/// `"use client"` modules a flight stream can carry. A client reference reaches the
/// wire only if the react-server graph resolved to that module, so this set — not the
/// browser graph's — is what the SSR consumer manifest must cover. The two sets are
/// legitimately different whenever a package's `exports` map sends the `browser` and
/// `node` conditions to different files: the browser graph then holds a `"use client"`
/// module the server graphs never see, and demanding an SSR twin for it would reject a
/// correct build.
pub const REACT_SERVER_REFERENCES_MANIFEST_FILE: &str = "react-server-references-manifest.json";

/// The node module that JOINS the three references manifests above into the
/// divergent-id `ssrModuleMapping`. Imported as a sibling by both the orchestrator
/// (`next-server.mjs`) and the render seam (`next-render-core.mjs`), so it is written
/// next to whichever of them lands in an output dir.
pub const SSR_MODULE_MAP_FILE: &str = "ssr-module-map.mjs";

#[cfg(test)]
mod ssr_module_map_tests {
    use super::*;
    use std::process::Command;

    /// Write the three manifests plus the shipped join module into `dir`, then run
    /// `script` against it under node. Returns `(stdout, stderr, ok)`.
    ///
    /// The module under test is the REAL file the binary embeds and writes next to
    /// every build's output, not a copy — the join rule has four consumers and this
    /// is the one place it is stated.
    fn run_join(
        client: &str,
        flight: &str,
        ssr: &str,
        script: &str,
    ) -> (String, String, bool) {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        fs::write(root.join(CLIENT_REFERENCES_MANIFEST_FILE), client).unwrap();
        fs::write(root.join(REACT_SERVER_REFERENCES_MANIFEST_FILE), flight).unwrap();
        fs::write(root.join(SERVER_REFERENCES_MANIFEST_FILE), ssr).unwrap();
        fs::write(
            root.join(SSR_MODULE_MAP_FILE),
            include_str!("../scripts/rsc/ssr-module-map.mjs"),
        )
        .unwrap();
        let runner = root.join("run.mjs");
        fs::write(
            &runner,
            format!(
                "import {{ loadServerConsumerManifest }} from \"./{SSR_MODULE_MAP_FILE}\";\n\
                 const outputDir = new URL(\".\", import.meta.url).pathname;\n{script}\n"
            ),
        )
        .unwrap();
        let output = Command::new("node").arg(&runner).output().unwrap();
        (
            String::from_utf8_lossy(&output.stdout).into_owned(),
            String::from_utf8_lossy(&output.stderr).into_owned(),
            output.status.success(),
        )
    }

    fn entry(id: usize) -> String {
        format!("{{\"id\":{id},\"chunks\":[],\"name\":\"*\"}}")
    }

    /// A `"use client"` module the BROWSER graph has and no server graph does is a
    /// correct build, not a broken one: a package whose `exports` sends `browser`
    /// and `node` to different files (`@sentry/nextjs`) contributes exactly this.
    /// Demanding an SSR twin for it rejected cal.com's whole prerender phase.
    ///
    /// It still gets a `moduleMap` entry — one that throws by name the instant
    /// anything resolves it — so "unreachable" can never quietly become "missing".
    #[test]
    fn a_browser_only_use_client_module_does_not_fail_the_join_but_cannot_be_resolved() {
        if Command::new("node").arg("--version").output().is_err() {
            return;
        }
        let client = format!(
            "{{\"/app/shared.js\":{},\"/pkg/browser-only.js\":{}}}",
            entry(1),
            entry(2)
        );
        let flight = format!("{{\"/app/shared.js\":{}}}", entry(7));
        let ssr = format!("{{\"/app/shared.js\":{}}}", entry(9));
        let (stdout, stderr, ok) = run_join(
            &client,
            &flight,
            &ssr,
            "const { moduleMap } = loadServerConsumerManifest(outputDir);\n\
             console.log(JSON.stringify(moduleMap[\"1\"]));\n\
             try { moduleMap[\"2\"][\"*\"]; console.log(\"NO THROW\"); }\n\
             catch (error) { console.log(\"threw:\" + error.message); }\n",
        );
        assert!(ok, "the join must succeed: {stderr}");
        assert!(
            stdout.contains("\"id\":9"),
            "the shared module maps to the SSR graph's id: {stdout}"
        );
        assert!(
            stdout.contains("threw:") && stdout.contains("/pkg/browser-only.js"),
            "resolving the browser-only module must throw, naming it: {stdout}"
        );
    }

    /// The check that DOES matter: a module the react-server graph can put on the
    /// wire but the SSR graph never bundled. That flight cannot be rendered to
    /// HTML, so it is a hard error naming the module and which graph lacks it.
    #[test]
    fn a_flight_reachable_module_missing_from_the_ssr_graph_is_a_hard_error() {
        if Command::new("node").arg("--version").output().is_err() {
            return;
        }
        let client = format!("{{\"/app/island.js\":{}}}", entry(1));
        let flight = format!("{{\"/app/island.js\":{}}}", entry(7));
        let (_, stderr, ok) = run_join(
            &client,
            &flight,
            "{}",
            "loadServerConsumerManifest(outputDir);\n",
        );
        assert!(!ok, "a flight-reachable module with no SSR twin must throw");
        assert!(stderr.contains("/app/island.js"), "{stderr}");
        assert!(stderr.contains("SSR graph"), "{stderr}");
    }
}

/// One `"use client"` module's entry in the client-references manifest
/// (`bundlerConfig[moduleId]`). The shape `react-server-dom-webpack`'s
/// `resolveClientReferenceMetadata` reads: `{ id, chunks, name }`.
///
/// * `id` — the CLIENT build's numeric runtime id (`runtime_ids[dense]`). It is
///   serialized raw into the flight and consumed in the browser as
///   `__webpack_require__(id)` (which diffpack maps onto its registry's
///   `require`). NOT the module path — the seam must not do a path→id lookup.
/// * `chunks` — a FLAT even-length `[chunkId, chunkFile, ...]` for
///   `__webpack_chunk_load__`. Empty when the module lands in the already-loaded
///   main entry chunk. diffpack uses the chunk file name as the chunk id too.
/// * `name` — `"*"`: the real export name arrives via the `$$id` split
///   (`"<moduleId>#<export>"`), so the entry's own name is only read on the
///   node-SSR-consumer path.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClientReferenceEntry {
    pub id: usize,
    pub chunks: Vec<String>,
    pub name: String,
}

/// Manifest #1 — the SERVER RENDER manifest (`bundlerConfig`), keyed by the bare
/// `module_reference_id` (canonical path), the same id the `$$id` prefix carries.
/// Because the moduleId is identical across graphs, this one manifest bridges the
/// react-server render and the client bundle (see docs/RSC_SPEC.md §1).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ClientReferencesManifest {
    pub entries: BTreeMap<String, ClientReferenceEntry>,
}

impl ClientReferencesManifest {
    /// The `{ "<moduleId>": { id, chunks, name } }` JSON value.
    pub fn to_json(&self) -> serde_json::Value {
        let mut map = serde_json::Map::new();
        for (module_id, entry) in &self.entries {
            map.insert(
                module_id.clone(),
                serde_json::json!({
                    "id": entry.id,
                    "chunks": entry.chunks,
                    "name": entry.name,
                }),
            );
        }
        serde_json::Value::Object(map)
    }

    /// Serializes to the `client-references-manifest.json` artifact.
    pub fn write(&self, path: &Path) -> Result<(), String> {
        let text = serde_json::to_string_pretty(&self.to_json())
            .map_err(|error| format!("cannot serialize client-references manifest: {error}"))?;
        // Skip the write when the bytes already match, keeping the artifact's
        // mtime stable across a rebuild that reproduced it — which is what lets
        // the dev warm start prove "the rebuild changed nothing" from mtimes.
        if fs::read_to_string(path).ok().as_deref() == Some(text.as_str()) {
            return Ok(());
        }
        fs::write(path, text)
            .map_err(|error| format!("cannot write {}: {error}", path.display()))
    }

    /// Reads the `client-references-manifest.json` artifact the client build
    /// produced. A missing or malformed file is a hard, specific error — never a
    /// silently-empty manifest — so the react-server render fails loudly when the
    /// client build has not run first.
    pub fn read(path: &Path) -> Result<Self, String> {
        let text = fs::read_to_string(path).map_err(|error| {
            format!(
                "cannot read client-references manifest {}: {error}; \
                 run the client build before the react-server render",
                path.display()
            )
        })?;
        let value: serde_json::Value = serde_json::from_str(&text)
            .map_err(|error| format!("cannot parse {}: {error}", path.display()))?;
        let object = value
            .as_object()
            .ok_or_else(|| format!("{}: manifest is not a JSON object", path.display()))?;
        let mut entries = BTreeMap::new();
        for (module_id, entry) in object {
            let id = entry["id"]
                .as_u64()
                .ok_or_else(|| format!("{}: `{module_id}` missing numeric `id`", path.display()))?
                as usize;
            let chunks = entry["chunks"]
                .as_array()
                .ok_or_else(|| format!("{}: `{module_id}` missing array `chunks`", path.display()))?
                .iter()
                .map(|chunk| {
                    chunk.as_str().map(str::to_string).ok_or_else(|| {
                        format!("{}: `{module_id}` has a non-string chunk", path.display())
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            let name = entry["name"]
                .as_str()
                .ok_or_else(|| format!("{}: `{module_id}` missing string `name`", path.display()))?
                .to_string();
            entries.insert(module_id.clone(), ClientReferenceEntry { id, chunks, name });
        }
        Ok(Self { entries })
    }

    /// Manifest #2 — the SSR CONSUMER manifest (`serverConsumerManifest`, the option
    /// `createFromReadableStream` reads on the node SSR pass), mechanically derived
    /// from this Manifest #1 (see docs/RSC_SPEC.md §1).
    ///
    /// `react-server-dom-webpack/client`'s `resolveClientReference` reads
    /// `moduleMap[metadata[0]]` where `metadata[0]` is the render-emitted numeric id
    /// (this manifest's `entry.id`, serialized raw into the flight), then
    /// `[metadata[2]]` (the export name from the `$$id` split) with a `"*"` fallback.
    /// Because the SSR graph shares diffpack's one runtime-id scheme (same
    /// `module_reference_id`, same registry), `moduleMap` is the IDENTITY map: keyed
    /// by the numeric id, a single `"*"` entry carrying the same `{id, chunks}`. The
    /// real export name arrives via the flight, so nesting per-export is unnecessary.
    ///
    /// `moduleLoading` is REQUIRED — the consumer reads `.prefix` off it
    /// unconditionally, and reading it off `undefined` crashes. `serverModuleMap`
    /// resolves `"use server"` references embedded in the flight; `null` when the
    /// rendered tree carries no server references (the caller supplies it when it
    /// does — same shape keyed by `"<moduleId>#<name>"`).
    pub fn to_ssr_consumer_manifest_json(
        &self,
        server_module_map: Option<serde_json::Value>,
    ) -> serde_json::Value {
        let mut module_map = serde_json::Map::new();
        for entry in self.entries.values() {
            let per_export = serde_json::json!({
                "*": {
                    "id": entry.id,
                    "chunks": entry.chunks,
                    "name": "*",
                }
            });
            // Keyed by the numeric render-emitted id (as a string object key), the
            // exact value `metadata[0]` carries in the flight.
            module_map.insert(entry.id.to_string(), per_export);
        }
        serde_json::json!({
            "moduleMap": serde_json::Value::Object(module_map),
            "serverModuleMap": server_module_map.unwrap_or(serde_json::Value::Null),
            "moduleLoading": { "prefix": "", "crossOrigin": serde_json::Value::Null },
        })
    }

    /// The [`SSR_CONSUMER_MANIFEST_SPECIFIER`] virtual-module source: the derived
    /// Manifest #2 exported as `serverConsumerManifest` (named + default), for the
    /// SSR pass to pass straight into `createFromReadableStream`.
    pub fn to_ssr_consumer_manifest_module(
        &self,
        server_module_map: Option<serde_json::Value>,
    ) -> String {
        let value = self.to_ssr_consumer_manifest_json(server_module_map);
        let json = serde_json::to_string_pretty(&value)
            .unwrap_or_else(|_| "{}".to_string());
        format!(
            "// Generated natively by Diffpack: the RSC SSR consumer manifest \
             (Manifest #2), derived from the client-references manifest.\n\
             export const serverConsumerManifest = {json};\n\
             export default serverConsumerManifest;\n"
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn detect(source: &str) -> Option<RscDirective> {
        detect_directive(Path::new("mod.tsx"), source)
    }

    #[test]
    fn detects_use_client_and_use_server() {
        assert_eq!(detect("\"use client\";\nexport function Widget() {}"), Some(RscDirective::Client));
        assert_eq!(detect("'use server'\nexport async function act() {}"), Some(RscDirective::Server));
    }

    #[test]
    fn allows_the_directive_after_use_strict() {
        assert_eq!(
            detect("\"use strict\";\n\"use client\";\nexport const A = () => null;"),
            Some(RscDirective::Client)
        );
    }

    #[test]
    fn ignores_a_directive_that_is_not_in_the_prologue() {
        // After a real statement — not a directive, so not a boundary.
        assert_eq!(detect("const x = 1;\n\"use client\";\nexport const A = () => null;"), None);
        // Inside a function body.
        assert_eq!(detect("export function f() {\n  \"use client\";\n  return 1;\n}"), None);
        // In a comment / string value, not a directive statement.
        assert_eq!(detect("// use client\nexport const A = 1;"), None);
        assert_eq!(detect("export const label = \"use client\";"), None);
    }

    #[test]
    fn plain_modules_are_none_without_parsing() {
        assert_eq!(detect("export const A = () => null;"), None);
    }

    fn client_server(source: &str) -> Option<String> {
        transform_use_client_server(Path::new("/app/src/Counter.tsx"), source)
            .expect("not a hard error")
    }

    #[test]
    fn use_client_server_transform_emits_client_module_proxy_reexports() {
        let out = client_server(
            "\"use client\";\nimport {useState} from 'react';\nexport function Counter(){const[n,s]=useState(0);return n;}\nexport const Label = () => 'x';\nexport default Counter;",
        )
        .expect("use client module transforms");
        // The real component code (useState) must NOT survive into the server build.
        assert!(!out.contains("useState"), "server build must not ship client code: {out}");
        // Each export is re-exported off react-server-dom's client module proxy.
        assert!(out.contains("createClientModuleProxy(\"/app/src/Counter.tsx\")"), "{out}");
        assert!(out.contains("export const Counter = __diffpack_client[\"Counter\"];"), "{out}");
        assert!(out.contains("export const Label = __diffpack_client[\"Label\"];"), "{out}");
        assert!(out.contains("export default __diffpack_client.default;"), "{out}");
        assert!(
            out.contains("from \"react-server-dom-webpack/server\""),
            "must import the proxy from react-server-dom: {out}"
        );
    }

    #[test]
    fn non_use_client_modules_are_left_unchanged() {
        assert_eq!(client_server("export const A = () => null;"), None);
        // A "use server" module is not a client boundary.
        assert_eq!(client_server("\"use server\";\nexport async function act(){}"), None);
    }

    #[test]
    fn module_reference_id_is_stable_and_pathlike() {
        // Non-existent path falls back to the given path verbatim (canonicalize fails).
        let id = module_reference_id(Path::new("/app/src/Widget.tsx"));
        assert_eq!(id, "/app/src/Widget.tsx");
    }

    #[test]
    fn the_server_transform_keys_a_client_reference_by_module_reference_id() {
        // The id is single-sourced; this pins that the react-server `$$id` prefix and
        // the client-references-manifest key stay the SAME string, for every layout a
        // real project uses (under `src/`, under `app/`, inside a dependency). A
        // disagreement here is the "Could not find the module ... in the React Client
        // Manifest" failure, and no filesystem-scan fix can paper over it.
        for path in [
            "/app/src/components/counter.tsx",
            "/app/app/components/counter.tsx",
            "/app/node_modules/jotai/esm/react.mjs",
        ] {
            let path = Path::new(path);
            let source = "\"use client\";\nexport default function C() { return null; }\n";
            let out = transform_use_client_server(path, source).unwrap().unwrap();
            let id = module_reference_id(path);
            assert!(
                out.contains(&format!("createClientModuleProxy({})", json_string(&id))),
                "{}: proxy id is not module_reference_id: {out}",
                path.display()
            );
            // The manifest keys by the same call, so the two agree by construction.
            let manifest = ClientReferencesManifest {
                entries: BTreeMap::from([(
                    module_reference_id(path),
                    ClientReferenceEntry { id: 1, chunks: Vec::new(), name: "*".to_string() },
                )]),
            };
            assert_eq!(manifest.to_json()[&id]["id"], serde_json::json!(1));
        }
    }

    #[test]
    fn the_client_reference_proxy_keeps_the_module_stylesheet_imports() {
        // The proxy discards the module body, but a client component's CSS belongs to
        // the document: the react-server graph is what `server.css` (served as
        // `/rsc.css`) is compiled from, so dropping these imports serves an unstyled
        // page while the head still links the stylesheet.
        let source = "\"use client\";\nimport \"./clock.css\";\nimport styles from \"./x.module.scss\";\nimport { useState } from \"react\";\nexport default function Clock() { return null; }\n";
        let out = transform_use_client_server(Path::new("/app/src/components/clock.tsx"), source)
            .unwrap()
            .unwrap();
        assert!(out.contains("import \"./clock.css\";"), "{out}");
        assert!(out.contains("import \"./x.module.scss\";"), "{out}");
        // Non-stylesheet imports (and the body) still must NOT reach the server.
        assert!(!out.contains("react\""), "{out}");
        assert!(!out.contains("useState"), "{out}");
    }

    #[test]
    fn the_proxy_keeps_every_stylesheet_form_and_every_stylesheet_language() {
        // The extraction used to be a static `import` declaration with a `.css`/`.scss`/
        // `.sass` extension and nothing else, so a `require`d, dynamically imported,
        // Less or Stylus stylesheet was silently deleted with the module body — an
        // unstyled page with no diagnostic.
        let source = "\"use client\";\n\
                      import \"./a.less\";\n\
                      import theme from \"./b.module.styl\";\n\
                      const c = require(\"./c.scss\");\n\
                      function open() { return import(\"./d.css\"); }\n\
                      export default function W() { return [theme, c, open]; }\n";
        let out = transform_use_client_server(Path::new("/app/src/w.tsx"), source).unwrap().unwrap();
        for stylesheet in ["./a.less", "./b.module.styl", "./c.scss", "./d.css"] {
            assert!(
                out.contains(&format!("import {};", json_string(stylesheet))),
                "{stylesheet} lost from the react-server graph: {out}",
            );
        }
    }

    #[test]
    fn a_type_only_stylesheet_import_is_not_a_stylesheet() {
        // `import type` is erased; carrying it over would emit an import of a module
        // the build never had.
        let source =
            "\"use client\";\nimport type Styles from \"./x.module.css\";\nexport default function W(): Styles { return null as never; }\n";
        let out = transform_use_client_server(Path::new("/app/src/w.tsx"), source).unwrap().unwrap();
        assert!(!out.contains("x.module.css"), "{out}");
    }

    #[test]
    fn a_commonjs_use_client_module_keeps_its_export_surface() {
        // FINDINGS #23. `next/dist/client/script.js` is swc-compiled CommonJS carrying
        // `"use client"`. The export scan only understood ESM syntax, so the proxy was
        // emitted with ZERO exports — a module that bundles fine and then throws
        // `The requested module "./dist/client/script" does not provide an export named
        // "default"` on the first render. Every CJS export form must be recovered.
        let source = "'use client';\n\
                      \"use strict\";\n\
                      Object.defineProperty(exports, \"__esModule\", { value: true });\n\
                      0 && (module.exports = { default: null, handleClientScriptLoad: null });\n\
                      function _export(target, all) { for (var name in all) Object.defineProperty(target, name, { get: all[name] }); }\n\
                      _export(exports, {\n\
                          default: function() { return _default; },\n\
                          handleClientScriptLoad: function() { return handleClientScriptLoad; },\n\
                          initScriptLoader: function() { return initScriptLoader; }\n\
                      });\n\
                      exports.extraNamed = 1;\n\
                      function _default() { return null; }\n";
        let out = transform_use_client_server(Path::new("/app/node_modules/next/dist/client/script.js"), source)
            .expect("a CJS use-client module is not a hard error")
            .expect("it IS a use client module");
        assert!(out.contains("export default __diffpack_client.default;"), "{out}");
        for named in ["handleClientScriptLoad", "initScriptLoader", "extraNamed"] {
            assert!(
                out.contains(&format!("export const {named} = __diffpack_client[")),
                "{named} is missing from the client-reference surface: {out}",
            );
        }
        // `__esModule` is a marker, never a real export.
        assert!(!out.contains("__esModule ="), "{out}");
    }

    #[test]
    fn a_use_client_module_with_no_enumerable_exports_is_a_hard_error() {
        // A proxy with no exports is a client boundary nothing can import: it must fail
        // the BUILD, naming the file, not a request months later with a message that
        // names neither the module nor the reason.
        let error = transform_use_client_server(
            Path::new("/app/node_modules/weird/index.js"),
            "\"use client\";\nconst x = 1;\nconsole.log(x);\n",
        )
        .expect_err("a use-client module with no exports must be a hard error");
        assert!(error.contains("/app/node_modules/weird/index.js"), "names the file: {error}");
        assert!(error.contains("use client"), "names the directive: {error}");
    }

    #[test]
    fn a_use_client_export_name_that_is_not_an_identifier_is_a_hard_error() {
        // `exports["data-x"]` has no `export const <name>` form; emitting one would be a
        // syntax error in the generated module, so it is rejected by name.
        let error = transform_use_client_server(
            Path::new("/app/node_modules/weird/index.js"),
            "\"use client\";\nexports[\"data-x\"] = 1;\n",
        )
        .expect_err("a non-identifier export name must be a hard error");
        assert!(error.contains("data-x"), "names the export: {error}");
    }

    #[test]
    fn esm_exports_still_win_over_the_commonjs_scan() {
        // The CJS recovery is a FALLBACK; a module with real ESM exports that also pokes
        // at a local `exports` object must not pick up the poking.
        let out = transform_use_client_server(
            Path::new("/app/src/w.tsx"),
            "\"use client\";\nexport default function W() { return null; }\nconst exports = {};\nexports.notAnExport = 1;\n",
        )
        .unwrap()
        .unwrap();
        assert!(out.contains("export default __diffpack_client.default;"), "{out}");
        assert!(!out.contains("notAnExport"), "{out}");
    }

    #[test]
    fn module_imports_cover_every_specifier_form() {
        let source = "import a from \"./a\";\n\
                      import type { T } from \"./t\";\n\
                      export { b } from \"./b\";\n\
                      export * from \"./c\";\n\
                      export type { U } from \"./u\";\n\
                      const d = require(\"./d\");\n\
                      const e = () => import(\"./e\");\n\
                      export default [a, d, e];\n";
        let imports = module_import_specifiers(Path::new("/app/src/x.ts"), source);
        let value: Vec<&str> = imports
            .iter()
            .filter(|import| !import.type_only)
            .map(|import| import.specifier.as_str())
            .collect();
        assert_eq!(value, ["./a", "./b", "./c", "./d", "./e"]);
        let type_only: Vec<&str> = imports
            .iter()
            .filter(|import| import.type_only)
            .map(|import| import.specifier.as_str())
            .collect();
        assert_eq!(type_only, ["./t", "./u"]);
    }

    #[test]
    fn client_references_manifest_serializes_the_verified_bundler_config_shape() {
        let mut entries = BTreeMap::new();
        entries.insert(
            "/app/src/Counter.tsx".to_string(),
            ClientReferenceEntry {
                id: 42,
                chunks: vec!["cc".to_string(), "client.chunk-3.js".to_string()],
                name: "*".to_string(),
            },
        );
        entries.insert(
            "/app/src/Widget.tsx".to_string(),
            ClientReferenceEntry {
                id: 7,
                chunks: Vec::new(),
                name: "*".to_string(),
            },
        );
        let manifest = ClientReferencesManifest { entries };
        let value = manifest.to_json();
        // Keyed by the bare moduleId (the `$$id` prefix); each entry is {id,chunks,name}.
        assert_eq!(value["/app/src/Counter.tsx"]["id"], serde_json::json!(42));
        assert_eq!(
            value["/app/src/Counter.tsx"]["chunks"],
            serde_json::json!(["cc", "client.chunk-3.js"])
        );
        assert_eq!(value["/app/src/Counter.tsx"]["name"], serde_json::json!("*"));
        // A module in the already-loaded main entry chunk has an empty `chunks`.
        assert_eq!(value["/app/src/Widget.tsx"]["chunks"], serde_json::json!([]));
    }

    #[test]
    fn ssr_consumer_manifest_is_the_identity_module_map_keyed_by_numeric_id() {
        let mut entries = BTreeMap::new();
        entries.insert(
            "/app/src/Counter.tsx".to_string(),
            ClientReferenceEntry {
                id: 0,
                chunks: Vec::new(),
                name: "*".to_string(),
            },
        );
        entries.insert(
            "/app/src/Widget.tsx".to_string(),
            ClientReferenceEntry {
                id: 3,
                chunks: vec!["client.chunk-2.js".to_string(), "client.chunk-2.js".to_string()],
                name: "*".to_string(),
            },
        );
        let manifest = ClientReferencesManifest { entries };
        let value = manifest.to_ssr_consumer_manifest_json(None);
        // moduleMap is keyed by the numeric render-emitted id (metadata[0]), each
        // carrying a single "*" entry with the same {id, chunks} — the identity map.
        assert_eq!(value["moduleMap"]["0"]["*"]["id"], serde_json::json!(0));
        assert_eq!(value["moduleMap"]["0"]["*"]["chunks"], serde_json::json!([]));
        assert_eq!(value["moduleMap"]["0"]["*"]["name"], serde_json::json!("*"));
        assert_eq!(
            value["moduleMap"]["3"]["*"]["chunks"],
            serde_json::json!(["client.chunk-2.js", "client.chunk-2.js"])
        );
        // moduleLoading is required (consumer reads `.prefix` unconditionally).
        assert_eq!(value["moduleLoading"]["prefix"], serde_json::json!(""));
        assert!(value["moduleLoading"].get("crossOrigin").is_some());
        // No server references in this tree.
        assert_eq!(value["serverModuleMap"], serde_json::Value::Null);
    }

    #[test]
    fn ssr_consumer_manifest_module_exports_the_manifest() {
        let mut entries = BTreeMap::new();
        entries.insert(
            "/app/src/Counter.tsx".to_string(),
            ClientReferenceEntry { id: 0, chunks: Vec::new(), name: "*".to_string() },
        );
        let manifest = ClientReferencesManifest { entries };
        let module = manifest.to_ssr_consumer_manifest_module(None);
        assert!(module.contains("export const serverConsumerManifest ="), "{module}");
        assert!(module.contains("export default serverConsumerManifest;"), "{module}");
        assert!(module.contains("\"moduleLoading\""), "{module}");
    }

    #[test]
    fn client_references_manifest_round_trips_through_json() {
        let mut entries = BTreeMap::new();
        entries.insert(
            "/app/src/Counter.tsx".to_string(),
            ClientReferenceEntry {
                id: 3,
                chunks: vec!["client.chunk-1.js".to_string(), "client.chunk-1.js".to_string()],
                name: "*".to_string(),
            },
        );
        let manifest = ClientReferencesManifest { entries };
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(CLIENT_REFERENCES_MANIFEST_FILE);
        manifest.write(&path).unwrap();
        assert_eq!(ClientReferencesManifest::read(&path).unwrap(), manifest);
    }

    #[test]
    fn a_missing_client_references_manifest_is_a_specific_error() {
        let dir = tempfile::tempdir().unwrap();
        let error =
            ClientReferencesManifest::read(&dir.path().join("nope.json")).unwrap_err();
        assert!(error.contains("client-references manifest"), "{error}");
        assert!(error.contains("client build"), "{error}");
    }

    #[test]
    fn non_use_server_modules_are_left_unchanged_by_both_transforms() {
        // Only a real `"use server"` prologue triggers either transform.
        assert_eq!(
            transform_use_server_server(
                Path::new("/app/src/Counter.tsx"),
                "\"use client\";\nexport function Counter(){ return null }"
            ),
            Ok(None)
        );
        assert_eq!(
            transform_use_server_server(Path::new("/app/src/plain.ts"), "export const a = 1;"),
            Ok(None)
        );
        assert_eq!(
            transform_use_server_client(Path::new("/app/src/plain.ts"), "export const a = 1;"),
            None
        );
    }

    #[test]
    fn use_cache_transform_wraps_named_exports_keeping_bodies() {
        let out = transform_use_cache_server(
            Path::new("/app/src/data.ts"),
            "\"use cache\";\nexport async function getData(id){ return id + 1 }\nexport const load = async () => 2;",
        )
        .unwrap()
        .expect("a use cache module transforms");
        // Imports the cache boundary helper from the aliased next/cache shim.
        assert!(
            out.contains("import { __diffpackUseCache as __diffpack_uc } from \"next/cache\";"),
            "{out}"
        );
        // The real bodies survive on the server (unlike use-client).
        assert!(out.contains("return id + 1"), "cached body preserved: {out}");
        assert!(out.contains("=> 2"), "{out}");
        // The `export` keyword is stripped from each declaration (binding stays local).
        assert!(out.contains("async function getData(id)"), "{out}");
        assert!(!out.contains("export async function getData"), "export stripped off decl: {out}");
        // Each export is wrapped and re-exported under its original name, keyed by id.
        assert!(out.contains("= __diffpack_uc(getData, \"/app/src/data.ts#getData\");"), "{out}");
        assert!(out.contains("= __diffpack_uc(load, \"/app/src/data.ts#load\");"), "{out}");
        assert!(out.contains("as getData };"), "{out}");
        assert!(out.contains("as load };"), "{out}");
    }

    #[test]
    fn use_cache_transform_wraps_the_default_export() {
        let out = transform_use_cache_server(
            Path::new("/app/src/page.tsx"),
            "\"use cache\";\nexport default async function Page(){ return null }",
        )
        .unwrap()
        .expect("default export transforms");
        // The default value is renamed to a local const, wrapped, and re-exported as default.
        assert!(out.contains("const __diffpack_uc_default = (async function Page(){ return null });"), "{out}");
        assert!(out.contains("= __diffpack_uc(__diffpack_uc_default, \"/app/src/page.tsx#default\");"), "{out}");
        assert!(out.contains("as default };"), "{out}");
    }

    #[test]
    fn use_cache_transform_wraps_renamed_specifiers_by_exported_name() {
        let out = transform_use_cache_server(
            Path::new("/app/src/data.ts"),
            "\"use cache\";\nasync function impl(){ return 1 }\nexport { impl as run };",
        )
        .unwrap()
        .expect("specifier export transforms");
        // Wrapped value is the LOCAL binding; the id + re-export use the EXPORTED name.
        assert!(out.contains("= __diffpack_uc(impl, \"/app/src/data.ts#run\");"), "{out}");
        assert!(out.contains("as run };"), "{out}");
        // The original specifier-only export statement is removed (no duplicate export).
        assert!(!out.contains("export { impl as run };"), "original specifier export removed: {out}");
    }

    #[test]
    fn use_cache_transform_hard_errors_on_reexport() {
        let err = transform_use_cache_server(
            Path::new("/app/src/data.ts"),
            "\"use cache\";\nexport { thing } from \"./other\";",
        )
        .unwrap_err();
        assert!(err.contains("export ... from") && err.contains("data.ts"), "{err}");

        let star = transform_use_cache_server(
            Path::new("/app/src/data.ts"),
            "\"use cache\";\nexport * from \"./other\";",
        )
        .unwrap_err();
        assert!(star.contains("export * from"), "{star}");
    }

    #[test]
    fn non_use_cache_modules_are_left_unchanged() {
        assert_eq!(
            transform_use_cache_server(Path::new("/app/src/plain.ts"), "export const a = 1;"),
            Ok(None)
        );
        // A use-client / use-server module is NOT a cache boundary.
        assert_eq!(
            transform_use_cache_server(
                Path::new("/app/src/c.tsx"),
                "\"use client\";\nexport function C(){ return null }"
            ),
            Ok(None)
        );
    }

    #[test]
    fn use_server_server_transform_registers_each_export_keeping_bodies() {
        let out = transform_use_server_server(
            Path::new("/app/src/actions.ts"),
            "\"use server\";\nexport async function increment(n){ return n + 1 }\nexport const decrement = async (n) => n - 1;",
        )
        .unwrap()
        .expect("a use server module transforms");
        // Imports the real writer's registration API.
        assert!(
            out.contains("import { registerServerReference as __rsr } from \"react-server-dom-webpack/server\";"),
            "{out}"
        );
        // The real bodies survive on the server.
        assert!(out.contains("return n + 1"), "server keeps the real body: {out}");
        assert!(out.contains("n - 1"), "{out}");
        // Each export is registered under its `<moduleId>#<name>` id via the live binding.
        assert!(
            out.contains("__rsr(increment, \"/app/src/actions.ts\", \"increment\");"),
            "{out}"
        );
        assert!(
            out.contains("__rsr(decrement, \"/app/src/actions.ts\", \"decrement\");"),
            "{out}"
        );
    }

    #[test]
    fn use_server_server_transform_wraps_the_default_export() {
        let out = transform_use_server_server(
            Path::new("/app/src/act.ts"),
            "\"use server\";\nexport default async function greet(name){ return \"hi \" + name }",
        )
        .unwrap()
        .expect("default export transforms");
        // The default value is wrapped in-place, registered under `#default`.
        assert!(
            out.contains("export default __rsr(async function greet(name){ return \"hi \" + name }, \"/app/src/act.ts\", \"default\")"),
            "{out}"
        );
    }

    #[test]
    fn use_server_server_transform_registers_renamed_specifiers_by_exported_name() {
        let out = transform_use_server_server(
            Path::new("/app/src/act.ts"),
            "\"use server\";\nasync function impl(){ return 1 }\nexport { impl as run };",
        )
        .unwrap()
        .expect("specifier export transforms");
        // Registered value is the LOCAL binding; the id name is the EXPORTED name.
        assert!(out.contains("__rsr(impl, \"/app/src/act.ts\", \"run\");"), "{out}");
    }

    #[test]
    fn use_server_client_transform_drops_bodies_and_emits_stubs() {
        let out = transform_use_server_client(
            Path::new("/app/src/actions.ts"),
            "\"use server\";\nimport { readFile } from 'node:fs/promises';\nexport async function increment(n){ await readFile('x'); return n + 1 }",
        )
        .expect("a use server module transforms for the client");
        // No server-only body or import ships to the browser.
        assert!(!out.contains("node:fs"), "server-only import must not ship: {out}");
        assert!(!out.contains("return n + 1"), "server body must not ship: {out}");
        assert!(!out.contains("readFile"), "{out}");
        // The stub imports the transport and creates a server reference by id.
        assert!(
            out.contains("import { createServerReference } from \"react-server-dom-webpack/client\";"),
            "{out}"
        );
        assert!(out.contains("import { callServer } from \"#diffpack-call-server\";"), "{out}");
        assert!(
            out.contains("export const increment = createServerReference(\"/app/src/actions.ts#increment\", callServer);"),
            "{out}"
        );
    }

    #[test]
    fn client_stub_and_server_registration_and_resolver_agree_on_the_action_id() {
        // The round-trip invariant: the id the client stub calls === the id the
        // server registers === the resolver's manifest key. All three derive from
        // `action_reference_id`, so they agree for the same source file.
        let path = Path::new("/app/src/actions.ts");
        let source =
            "\"use server\";\nexport async function increment(n){ return n + 1 }";
        let id = action_reference_id(path, "increment");
        assert_eq!(id, "/app/src/actions.ts#increment");

        let client = transform_use_server_client(path, source).unwrap();
        assert!(client.contains(&format!("createServerReference(\"{id}\", callServer)")), "{client}");

        let server = transform_use_server_server(path, source).unwrap().unwrap();
        assert!(server.contains(&"__rsr(increment, \"/app/src/actions.ts\", \"increment\");".to_string()), "{server}");

        // The resolver keys the same id and dispatches to the real export.
        let entries = vec![ActionEntry {
            path: path.to_path_buf(),
            name: "increment".to_string(),
            id: id.clone(),
        }];
        let resolver = generate_action_resolver_module(&entries);
        assert!(
            resolver.contains(&format!("\"{id}\": {{ importer: () => import(\"/app/src/actions.ts\"), exportName: \"increment\" }}")),
            "{resolver}"
        );
        assert!(resolver.contains("no server action registered"), "{resolver}");
        assert!(resolver.contains("export { getServerActionById }"), "{resolver}");
    }

    #[test]
    fn use_server_transforms_hard_error_on_re_exports() {
        // `export ... from` re-exports a non-local binding that cannot be registered.
        let error = transform_use_server_server(
            Path::new("/app/src/act.ts"),
            "\"use server\";\nexport { increment } from './other';",
        )
        .expect_err("re-export is unsupported");
        assert!(error.contains("re-export"), "error must name the construct: {error}");
        assert!(error.contains("/app/src/act.ts"), "{error}");

        let star = transform_use_server_server(
            Path::new("/app/src/act.ts"),
            "\"use server\";\nexport * from './other';",
        )
        .expect_err("wildcard re-export is unsupported");
        assert!(star.contains("export * from"), "{star}");
    }
}
