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
use oxc_ast::ast::{Declaration, Statement};
use oxc_parser::Parser;
use oxc_span::{GetSpan, Span, SourceType};

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
    let source_type = SourceType::from_path(path).unwrap_or_default().with_module(true);
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
pub fn transform_use_client_server(path: &Path, source: &str) -> Option<String> {
    if detect_directive(path, source) != Some(RscDirective::Client) {
        return None;
    }
    let exports = module_exports(path, source);
    let id = module_reference_id(path);
    let mut out = String::new();
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
    Some(out)
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
    let source_type = SourceType::from_path(path).unwrap_or_default().with_module(true);
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

/// Transforms a `"use server"` module for the SERVER build (both the normal SSR
/// server and the react-server graph) into its server-reference surface: keeps the
/// real bodies verbatim, imports `registerServerReference` from
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
    let source_type = SourceType::from_path(path).unwrap_or_default().with_module(true);
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

/// Walks `root` (skipping `node_modules`, build output, and VCS dirs) for
/// `"use server"` modules, returning every exported action sorted by id. Only files
/// whose text contains `"use server"` AND whose directive prologue is `"use server"`
/// are enumerated (mirroring the transform's cheap gate). This is what lets the
/// generated action resolver map each id to a real module import.
pub fn scan_project_server_actions(root: &Path) -> Result<Vec<ActionEntry>, String> {
    let mut entries = Vec::new();
    scan_actions_directory(root, &mut entries)?;
    entries.sort_by(|left, right| left.id.cmp(&right.id));
    entries.dedup();
    Ok(entries)
}

fn scan_actions_directory(dir: &Path, entries: &mut Vec<ActionEntry>) -> Result<(), String> {
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
            let name = entry.file_name();
            if matches!(
                name.to_str(),
                Some("node_modules" | ".diffpack-output" | ".diffpack-next" | ".git" | "dist" | ".output" | ".next")
            ) {
                continue;
            }
            scan_actions_directory(&path, entries)?;
        } else if is_action_module_file(&path) {
            let Ok(source) = std::fs::read_to_string(&path) else {
                continue;
            };
            if !source.contains("use server") {
                continue;
            }
            let canonical = std::fs::canonicalize(&path).unwrap_or(path);
            if detect_directive(&canonical, &source) != Some(RscDirective::Server) {
                continue;
            }
            for name in module_exports(&canonical, &source) {
                let id = action_reference_id(&canonical, &name);
                entries.push(ActionEntry {
                    path: canonical.clone(),
                    name,
                    id,
                });
            }
        }
    }
    Ok(())
}

fn is_action_module_file(path: &Path) -> bool {
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
    let source_type = SourceType::from_path(path).unwrap_or_default().with_module(true);
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
    names
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
