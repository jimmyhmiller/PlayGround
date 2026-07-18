//! Native TanStack Start server-function support (`createServerFn`).
//!
//! TanStack Start's `@tanstack/start-plugin-core` compiles every
//! `createServerFn(...).handler(fn)` call into two environment-specific shapes so
//! the browser never ships server code yet a data route still gets its data:
//!
//! * The **client** build replaces the handler argument with a thin RPC stub,
//!   `createClientRpc(functionId)` — invoking the server fn `fetch`es the
//!   server-fn HTTP endpoint (`/_serverFn/<functionId>`) and deserializes the
//!   result. The real handler body (and its server-only imports) is dropped.
//! * The **server** build keeps the real handler as the second `.handler` arg and
//!   prepends an *extracted* first arg, `createServerRpc({ id }, (opts) =>
//!   NAME.__executeServer(opts))`, so calling the server fn in-process (an SSR
//!   route loader) runs the real handler through the middleware chain. Each
//!   handler is also registered in a server-fn resolver module
//!   (`#tanstack-start-server-fn-resolver`) keyed by the same `functionId`, which
//!   the server runtime's `handleServerAction` dispatches through for HTTP calls.
//!
//! Diffpack replaces that plugin natively. This module performs the same rewrite
//! in Rust on the Oxc AST (source-to-source, so the result flows back through the
//! normal [`crate::transform::transform_module`] pipeline unchanged) and generates
//! the resolver module from a pre-scan of the app source. The rewrite is gated on
//! a cheap `source.contains("createServerFn")` string check before any parse, so
//! non-server-fn modules pay nothing and the incremental graph is preserved.
//!
//! `functionId` matches the reference contract's shape: the SHA-256 hex digest of
//! `"<module>--<name>_createServerFn_handler"` — deterministic, collision-safe,
//! and non-path-leaking (a hash, never the raw path). The client and server builds
//! (and the resolver pre-scan) all derive it from the same canonical module path,
//! so a client RPC call lands on exactly the handler the server registered.

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use oxc_allocator::Allocator;
use oxc_ast::ast::{
    Declaration, Expression, ImportDeclarationSpecifier, Program, Statement,
};
use oxc_parser::Parser;
use oxc_span::{GetSpan, SourceType, Span};

use crate::transform::Target;

/// TanStack Start's default server-fn HTTP base (`serverFns.base`), the path
/// prefix `createStartHandler` matches and `createClientRpc`/`createServerRpc`
/// build their URL from (`TSS_SERVER_FN_BASE`).
pub const SERVER_FN_BASE: &str = "/_serverFn/";

/// The subpath specifier the client RPC stub is imported from.
const CLIENT_RPC_SPECIFIER: &str = "@tanstack/start-client-core/client-rpc";

/// The subpath specifier the server RPC wrapper is imported from.
const SERVER_RPC_SPECIFIER: &str = "@tanstack/start-server-core/createServerRpc";

/// The virtual module `@tanstack/start-server-core`'s `getServerFnById` resolves
/// through (`VIRTUAL_MODULES.serverFnResolver`). Diffpack registers a natively
/// generated resolver under this specifier for the server build.
pub const RESOLVER_SPECIFIER: &str = "#tanstack-start-server-fn-resolver";

/// A `createServerFn(...).handler(fn)` declaration found in a module: the local
/// name it is bound to (`NAME` in `const NAME = ...`), the source span of the
/// handler's first argument (the function to replace/keep), and whether the
/// declaration is already a module export.
struct FoundServerFn {
    name: String,
    handler_arg: Span,
    exported: bool,
}

/// The deterministic `functionId` for a server function bound to `name` in the
/// module at `path`. Matches the reference contract's SHA-256-of-entryId shape,
/// with the canonical absolute module path standing in for the reference's
/// project-relative filename. The digest is opaque, so the id never leaks the
/// path; the canonical path makes the client build, the server build, and the
/// resolver pre-scan all agree on the id for the same source file.
pub fn function_id(path: &Path, name: &str) -> String {
    let canonical = std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
    let entry = format!(
        "{}--{name}_createServerFn_handler",
        canonical.to_string_lossy()
    );
    sha256_hex(entry.as_bytes())
}

/// Rewrites a module's `createServerFn(...).handler(fn)` calls for `target`,
/// returning the rewritten source, or `None` when the module defines no server
/// function (the common case; the caller then uses the source unchanged).
///
/// A `createServerFn().handler()` that is not bound to a top-level `const NAME`
/// is a hard error rather than a silent miscompile: the id (and the server
/// build's `__executeServer` reference) is derived from that name, so an
/// anonymous or nested server fn cannot be given a stable id here.
pub fn transform_server_fns(
    path: &Path,
    source: &str,
    target: Target,
) -> Result<Option<String>, String> {
    if !source.contains("createServerFn") {
        return Ok(None);
    }

    let allocator = Allocator::default();
    let source_type = SourceType::from_path(path)
        .unwrap_or_default()
        .with_module(true);
    let parsed = Parser::new(&allocator, source, source_type).parse();
    let program = &parsed.program;

    let locals = server_fn_locals(program);
    if locals.is_empty() {
        return Ok(None);
    }
    reject_untethered_server_fns(program, &locals, path)?;
    let found = find_server_fns(program, &locals);
    if found.is_empty() {
        return Ok(None);
    }

    let mut edits: Vec<(Span, String)> = Vec::new();
    for server_fn in &found {
        let id = function_id(path, &server_fn.name);
        match target {
            Target::Client => {
                // Replace the whole handler argument with the client RPC stub;
                // the real handler body (and its server imports) is dropped.
                edits.push((server_fn.handler_arg, format!("createClientRpc({})", quote(&id))));
            }
            Target::Server => {
                // Keep the real handler as the second `.handler` arg; prepend the
                // extracted first arg that runs it in-process via the middleware
                // chain. A zero-width insert at the argument start preserves the
                // original handler text verbatim as the second argument.
                let insert = Span::new(server_fn.handler_arg.start, server_fn.handler_arg.start);
                edits.push((
                    insert,
                    format!(
                        "createServerRpc({{ id: {} }}, (opts) => {}.__executeServer(opts)), ",
                        quote(&id),
                        server_fn.name
                    ),
                ));
            }
        }
    }

    let import = match target {
        Target::Client => format!("import {{ createClientRpc }} from {};\n", quote(CLIENT_RPC_SPECIFIER)),
        Target::Server => format!("import {{ createServerRpc }} from {};\n", quote(SERVER_RPC_SPECIFIER)),
    };
    let mut rewritten = apply_edits(source, import, edits);

    // The server-fn resolver imports each handler from its module by name to
    // dispatch HTTP calls, so a server fn that is only a local `const` (used
    // in-process, as `deferred.tsx`'s server fns are) must still be a module
    // export on the server build — otherwise the resolver's lookup misses. This
    // mirrors the reference plugin extracting each handler into an exported
    // binding. The client build needs no such export (its stub is called
    // directly), so this is server-only.
    if target == Target::Server {
        let unexported = found
            .iter()
            .filter(|server_fn| !server_fn.exported)
            .map(|server_fn| server_fn.name.clone())
            .collect::<Vec<_>>();
        if !unexported.is_empty() {
            rewritten.push_str(&format!("\nexport {{ {} }};\n", unexported.join(", ")));
        }
    }
    Ok(Some(rewritten))
}

/// Scans a module for its `createServerFn(...).handler(fn)` declarations,
/// returning `(name, functionId)` for each. Used by the resolver pre-scan; a
/// module with no server fn yields an empty vector.
pub fn scan_server_fns(path: &Path, source: &str) -> Vec<(String, String)> {
    if !source.contains("createServerFn") {
        return Vec::new();
    }
    let allocator = Allocator::default();
    let source_type = SourceType::from_path(path)
        .unwrap_or_default()
        .with_module(true);
    let parsed = Parser::new(&allocator, source, source_type).parse();
    let program = &parsed.program;
    let locals = server_fn_locals(program);
    if locals.is_empty() {
        return Vec::new();
    }
    find_server_fns(program, &locals)
        .into_iter()
        .map(|server_fn| {
            let id = function_id(path, &server_fn.name);
            (server_fn.name, id)
        })
        .collect()
}

/// A discovered server function: its module path (canonical), the local name it
/// is bound to (its module export), and its deterministic id.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ServerFnEntry {
    pub path: PathBuf,
    pub name: String,
    pub id: String,
}

/// Walks `root` (skipping `node_modules`, build output, and VCS dirs) for modules
/// that define server functions, returning every discovered function sorted by
/// id. Only files whose text contains `createServerFn` are parsed, mirroring the
/// transform's cheap string gate. This is the build-time equivalent of the
/// reference plugin compiling every server-fn module: it is what lets the
/// generated resolver map each id to a real module import.
pub fn scan_project_server_fns(root: &Path) -> Result<Vec<ServerFnEntry>, String> {
    let mut entries = Vec::new();
    scan_directory(root, &mut entries)?;
    entries.sort_by(|left, right| left.id.cmp(&right.id));
    entries.dedup();
    Ok(entries)
}

fn scan_directory(dir: &Path, entries: &mut Vec<ServerFnEntry>) -> Result<(), String> {
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
                Some("node_modules" | ".diffpack-output" | ".git" | "dist" | ".output")
            ) {
                continue;
            }
            scan_directory(&path, entries)?;
        } else if is_module_file(&path) {
            let Ok(source) = std::fs::read_to_string(&path) else {
                continue;
            };
            if !source.contains("createServerFn") {
                continue;
            }
            let canonical = std::fs::canonicalize(&path).unwrap_or(path);
            for (name, id) in scan_server_fns(&canonical, &source) {
                entries.push(ServerFnEntry {
                    path: canonical.clone(),
                    name,
                    id,
                });
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

/// Generates the `#tanstack-start-server-fn-resolver` module source: a `manifest`
/// mapping each `functionId` to a dynamic import of its module plus the export
/// name, and a `getServerFnById(id, access)` that resolves the module and returns
/// the handler's in-process `__executeServer` runner (what `handleServerAction`
/// dispatches through). A missing id is a hard, specific error rather than the
/// fake resolver's silent `undefined`.
pub fn generate_resolver_module(entries: &[ServerFnEntry]) -> String {
    let mut source = String::new();
    source.push_str("// Generated natively by Diffpack from the app's createServerFn declarations.\n");
    source.push_str("const manifest = {\n");
    for entry in entries {
        source.push_str(&format!(
            "  {}: {{ importer: () => import({}), functionName: {} }},\n",
            quote(&entry.id),
            quote(&entry.path.to_string_lossy()),
            quote(&entry.name),
        ));
    }
    source.push_str("};\n");
    source.push_str(
        r#"async function getServerFnById(id, access) {
  const info = manifest[id];
  if (!info) throw new Error("diffpack server-fn: no server function registered for id " + id);
  const mod = await info.importer();
  const fn = mod[info.functionName];
  if (!fn || typeof fn.__executeServer !== "function")
    throw new Error("diffpack server-fn: module for id " + id + " has no server handler export " + info.functionName);
  const action = (opts) => fn.__executeServer(opts);
  action.method = fn.method;
  return action;
}
export { getServerFnById };
export default { getServerFnById };
"#,
    );
    source
}

/// The set of local binding names imported as `createServerFn` from a
/// `@tanstack/` package. Matching by this set (not a bare name check) means a
/// user's own same-named local `createServerFn` is never rewritten.
fn server_fn_locals(program: &Program<'_>) -> HashSet<String> {
    let mut locals = HashSet::new();
    for statement in &program.body {
        let Statement::ImportDeclaration(declaration) = statement else {
            continue;
        };
        if !declaration.source.value.starts_with("@tanstack/") {
            continue;
        }
        let Some(specifiers) = &declaration.specifiers else {
            continue;
        };
        for specifier in specifiers {
            if let ImportDeclarationSpecifier::ImportSpecifier(specifier) = specifier
                && specifier.imported.name() == "createServerFn"
            {
                locals.insert(specifier.local.name.to_string());
            }
        }
    }
    locals
}

/// Every top-level `const NAME = createServerFn(...)...handler(fn)` in the module,
/// with the handler argument's span.
fn find_server_fns(program: &Program<'_>, locals: &HashSet<String>) -> Vec<FoundServerFn> {
    let mut found = Vec::new();
    for statement in &program.body {
        let (declaration, exported) = match statement {
            Statement::VariableDeclaration(declaration) => (Some(declaration.as_ref()), false),
            Statement::ExportNamedDeclaration(export) => match &export.declaration {
                Some(Declaration::VariableDeclaration(declaration)) => {
                    (Some(declaration.as_ref()), true)
                }
                _ => (None, false),
            },
            _ => (None, false),
        };
        let Some(declaration) = declaration else {
            continue;
        };
        for declarator in &declaration.declarations {
            let Some(name) = binding_name(declarator) else {
                continue;
            };
            let Some(init) = &declarator.init else {
                continue;
            };
            if let Some(handler_arg) = handler_argument_span(init, locals) {
                found.push(FoundServerFn {
                    name,
                    handler_arg,
                    exported,
                });
            }
        }
    }
    found
}

/// Hard-errors on a `createServerFn(...).handler(...)` chain that is not directly
/// the init of a top-level `const NAME = ...` — the only shape a stable id can be
/// derived for here. Never a silent pass-through that would ship a broken server
/// fn.
fn reject_untethered_server_fns(
    program: &Program<'_>,
    locals: &HashSet<String>,
    path: &Path,
) -> Result<(), String> {
    // Names bound directly to a server-fn chain: these are the supported shape.
    let mut tethered = HashSet::new();
    for statement in &program.body {
        let declaration = match statement {
            Statement::VariableDeclaration(declaration) => Some(declaration.as_ref()),
            Statement::ExportNamedDeclaration(export) => match &export.declaration {
                Some(Declaration::VariableDeclaration(declaration)) => Some(declaration.as_ref()),
                _ => None,
            },
            _ => None,
        };
        let Some(declaration) = declaration else {
            continue;
        };
        for declarator in &declaration.declarations {
            if let (Some(_), Some(init)) = (binding_name(declarator), declarator.init.as_ref())
                && handler_argument_span(init, locals).is_some()
            {
                tethered.insert(init.span());
            }
        }
    }
    // Any `.handler(` chain rooted at createServerFn whose enclosing expression is
    // not one of those tethered inits is unsupported.
    let mut checker = HandlerChecker {
        locals,
        tethered: &tethered,
        offending: false,
    };
    oxc_ast_visit::Visit::visit_program(&mut checker, program);
    if checker.offending {
        return Err(format!(
            "createServerFn().handler() in {} must be assigned to a top-level `const NAME = ...` \
             so Diffpack can give it a stable server-function id; an anonymous or nested server \
             function is not supported",
            path.display()
        ));
    }
    Ok(())
}

struct HandlerChecker<'s> {
    locals: &'s HashSet<String>,
    tethered: &'s HashSet<Span>,
    offending: bool,
}

impl<'a> oxc_ast_visit::Visit<'a> for HandlerChecker<'_> {
    fn visit_expression(&mut self, expression: &Expression<'a>) {
        if is_handler_chain(expression, self.locals) && !self.tethered.contains(&expression.span()) {
            self.offending = true;
        }
        oxc_ast_visit::walk::walk_expression(self, expression);
    }
}

/// The span of the first argument of the `.handler(fn)` call at the tail of a
/// `createServerFn(...)...handler(...)` chain, if `init` is such a chain.
fn handler_argument_span(init: &Expression<'_>, locals: &HashSet<String>) -> Option<Span> {
    let Expression::CallExpression(call) = init else {
        return None;
    };
    let Expression::StaticMemberExpression(member) = &call.callee else {
        return None;
    };
    if member.property.name != "handler" {
        return None;
    }
    let root = chain_root(&member.object)?;
    if !locals.contains(&root) {
        return None;
    }
    let argument = call.arguments.first()?.as_expression()?;
    Some(argument.span())
}

/// Whether `expression` is a `createServerFn(...)...handler(...)` chain rooted at
/// one of the tracked `createServerFn` locals.
fn is_handler_chain(expression: &Expression<'_>, locals: &HashSet<String>) -> bool {
    let Expression::CallExpression(call) = expression else {
        return false;
    };
    let Expression::StaticMemberExpression(member) = &call.callee else {
        return false;
    };
    if member.property.name != "handler" {
        return false;
    }
    chain_root(&member.object).is_some_and(|root| locals.contains(&root))
}

/// Walks a member/call chain down to its base callee identifier, returning its
/// name (`createServerFn(a).b().c` -> `createServerFn`).
fn chain_root(expression: &Expression<'_>) -> Option<String> {
    let mut current = expression;
    loop {
        match current {
            Expression::CallExpression(call) => match &call.callee {
                Expression::Identifier(identifier) => return Some(identifier.name.to_string()),
                Expression::StaticMemberExpression(member) => current = &member.object,
                _ => return None,
            },
            Expression::StaticMemberExpression(member) => current = &member.object,
            _ => return None,
        }
    }
}

/// The bound name of a single-identifier variable declarator (`const NAME = ...`).
fn binding_name(declarator: &oxc_ast::ast::VariableDeclarator<'_>) -> Option<String> {
    match &declarator.id {
        oxc_ast::ast::BindingPattern::BindingIdentifier(identifier) => {
            Some(identifier.name.to_string())
        }
        _ => None,
    }
}

/// Applies non-overlapping `(span, replacement)` edits to `source`, prepending
/// `preamble`. Insertions are zero-width spans.
fn apply_edits(source: &str, preamble: String, mut edits: Vec<(Span, String)>) -> String {
    edits.sort_by_key(|(span, _)| span.start);
    let mut output = preamble;
    let mut cursor = 0_usize;
    for (span, replacement) in edits {
        let start = span.start as usize;
        let end = span.end as usize;
        if start < cursor {
            continue;
        }
        output.push_str(&source[cursor..start]);
        output.push_str(&replacement);
        cursor = end;
    }
    output.push_str(&source[cursor..]);
    output
}

fn quote(value: &str) -> String {
    serde_json::to_string(value).expect("serializing a JavaScript string cannot fail")
}

/// A dependency-free SHA-256, producing the lowercase hex digest. Used for the
/// deterministic, collision-safe, non-path-leaking `functionId`.
fn sha256_hex(input: &[u8]) -> String {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];
    let mut h: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];

    let mut message = input.to_vec();
    let bit_len = (input.len() as u64) * 8;
    message.push(0x80);
    while message.len() % 64 != 56 {
        message.push(0);
    }
    message.extend_from_slice(&bit_len.to_be_bytes());

    for block in message.chunks_exact(64) {
        let mut w = [0u32; 64];
        for (index, word) in w.iter_mut().enumerate().take(16) {
            let start = index * 4;
            *word = u32::from_be_bytes([
                block[start],
                block[start + 1],
                block[start + 2],
                block[start + 3],
            ]);
        }
        for index in 16..64 {
            let s0 = w[index - 15].rotate_right(7)
                ^ w[index - 15].rotate_right(18)
                ^ (w[index - 15] >> 3);
            let s1 = w[index - 2].rotate_right(17)
                ^ w[index - 2].rotate_right(19)
                ^ (w[index - 2] >> 10);
            w[index] = w[index - 16]
                .wrapping_add(s0)
                .wrapping_add(w[index - 7])
                .wrapping_add(s1);
        }

        let mut v = h;
        for index in 0..64 {
            let s1 = v[4].rotate_right(6) ^ v[4].rotate_right(11) ^ v[4].rotate_right(25);
            let ch = (v[4] & v[5]) ^ ((!v[4]) & v[6]);
            let temp1 = v[7]
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[index])
                .wrapping_add(w[index]);
            let s0 = v[0].rotate_right(2) ^ v[0].rotate_right(13) ^ v[0].rotate_right(22);
            let maj = (v[0] & v[1]) ^ (v[0] & v[2]) ^ (v[1] & v[2]);
            let temp2 = s0.wrapping_add(maj);
            v[7] = v[6];
            v[6] = v[5];
            v[5] = v[4];
            v[4] = v[3].wrapping_add(temp1);
            v[3] = v[2];
            v[2] = v[1];
            v[1] = v[0];
            v[0] = temp1.wrapping_add(temp2);
        }
        for (slot, value) in h.iter_mut().zip(v) {
            *slot = slot.wrapping_add(value);
        }
    }

    let mut hex = String::with_capacity(64);
    for word in h {
        hex.push_str(&format!("{word:08x}"));
    }
    hex
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sha256_matches_known_vectors() {
        assert_eq!(
            sha256_hex(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn function_id_is_deterministic_and_opaque() {
        let path = Path::new("/app/src/utils/posts.tsx");
        let a = function_id(path, "fetchPosts");
        let b = function_id(path, "fetchPosts");
        assert_eq!(a, b);
        assert_eq!(a.len(), 64);
        assert!(a.chars().all(|c| c.is_ascii_hexdigit()));
        // Different names / paths give different ids.
        assert_ne!(a, function_id(path, "fetchPost"));
        assert_ne!(a, function_id(Path::new("/app/src/utils/users.tsx"), "fetchPosts"));
        // The raw path never appears in the id.
        assert!(!a.contains("posts"));
    }

    #[test]
    fn client_build_replaces_the_handler_with_an_rpc_stub() {
        let source = "import { createServerFn } from '@tanstack/react-start'\n\
             export const fetchPosts = createServerFn().handler(async () => {\n\
               const res = await fetch('https://example.com/posts')\n\
               return res.json()\n\
             })\n";
        let rewritten = transform_server_fns(Path::new("/app/src/utils/posts.tsx"), source, Target::Client)
            .unwrap()
            .expect("a module with a server fn is rewritten");
        assert!(
            rewritten.contains("import { createClientRpc } from \"@tanstack/start-client-core/client-rpc\";"),
            "{rewritten}"
        );
        let id = function_id(Path::new("/app/src/utils/posts.tsx"), "fetchPosts");
        assert!(rewritten.contains(&format!("createClientRpc(\"{id}\")")), "{rewritten}");
        // The server handler body is gone from the client build.
        assert!(!rewritten.contains("example.com/posts"), "{rewritten}");
        assert!(!rewritten.contains("res.json()"), "{rewritten}");
    }

    #[test]
    fn server_build_keeps_the_handler_and_registers_it() {
        let source = "import { createServerFn } from '@tanstack/react-start'\n\
             export const fetchPosts = createServerFn().handler(async () => {\n\
               return [1, 2, 3]\n\
             })\n";
        let path = Path::new("/app/src/utils/posts.tsx");
        let rewritten = transform_server_fns(path, source, Target::Server)
            .unwrap()
            .expect("server build rewrites the server fn");
        assert!(
            rewritten.contains("import { createServerRpc } from \"@tanstack/start-server-core/createServerRpc\";"),
            "{rewritten}"
        );
        let id = function_id(path, "fetchPosts");
        assert!(
            rewritten.contains(&format!("createServerRpc({{ id: \"{id}\" }}, (opts) => fetchPosts.__executeServer(opts)), async () =>")),
            "{rewritten}"
        );
        // The real handler body survives on the server.
        assert!(rewritten.contains("return [1, 2, 3]"), "{rewritten}");
    }

    #[test]
    fn a_local_same_named_create_server_fn_is_not_rewritten() {
        let source = "const createServerFn = () => ({ handler: (f) => f })\n\
             export const x = createServerFn().handler(async () => 1)\n";
        assert!(
            transform_server_fns(Path::new("/app/src/x.ts"), source, Target::Client)
                .unwrap()
                .is_none(),
            "a user's own createServerFn must not be treated as the tanstack helper"
        );
    }

    #[test]
    fn a_module_without_server_fns_is_left_alone() {
        assert!(
            transform_server_fns(Path::new("/app/src/x.ts"), "export const x = 1", Target::Client)
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn scan_extracts_names_and_ids() {
        let source = "import { createServerFn } from '@tanstack/react-start'\n\
             export const fetchPost = createServerFn({ method: 'POST' }).validator((d) => d).handler(async ({ data }) => data)\n\
             export const fetchPosts = createServerFn().handler(async () => [])\n";
        let path = Path::new("/app/src/utils/posts.tsx");
        let mut names = scan_server_fns(path, source);
        names.sort();
        assert_eq!(names.len(), 2);
        let ids: Vec<&str> = names.iter().map(|(_, id)| id.as_str()).collect();
        assert_eq!(ids[0].len(), 64);
        assert!(names.iter().any(|(name, _)| name == "fetchPosts"));
        assert!(names.iter().any(|(name, _)| name == "fetchPost"));
    }

    #[test]
    fn client_stub_and_server_registration_agree_on_the_id() {
        // The round-trip invariant: the id baked into the client RPC stub must be
        // exactly the id the server-fn resolver registers, or an HTTP call lands
        // on no handler. All three derivations (client transform, server
        // transform, resolver scan) share `function_id`, so they agree for the
        // same source file.
        let source = "import { createServerFn } from '@tanstack/react-start'\n\
             export const fetchPosts = createServerFn().handler(async () => [])\n";
        let path = Path::new("/app/src/utils/posts.tsx");

        let client = transform_server_fns(path, source, Target::Client)
            .unwrap()
            .unwrap();
        let registered = scan_server_fns(path, source);
        assert_eq!(registered.len(), 1);
        let (name, id) = &registered[0];
        assert_eq!(name, "fetchPosts");
        // The exact id the client stub fetches is the id the resolver registers.
        assert!(client.contains(&format!("createClientRpc(\"{id}\")")), "{client}");

        let server = transform_server_fns(path, source, Target::Server)
            .unwrap()
            .unwrap();
        assert!(server.contains(&format!("id: \"{id}\"")), "{server}");
    }

    #[test]
    fn server_build_exports_a_non_exported_server_fn_for_the_resolver() {
        // A server fn that is only a local `const` (deferred.tsx's shape) must
        // become a module export on the server build so the resolver can import
        // it by name for HTTP dispatch.
        let source = "import { createServerFn } from '@tanstack/react-start'\n\
             const slowServerFn = createServerFn().handler(async () => 1)\n\
             export const Route = { loader: () => slowServerFn() }\n";
        let path = Path::new("/app/src/routes/deferred.tsx");
        let server = transform_server_fns(path, source, Target::Server)
            .unwrap()
            .unwrap();
        assert!(server.contains("export { slowServerFn };"), "{server}");
        // The client build does not add the export (its stub is called directly).
        let client = transform_server_fns(path, source, Target::Client)
            .unwrap()
            .unwrap();
        assert!(!client.contains("export { slowServerFn };"), "{client}");
    }

    #[test]
    fn resolver_module_maps_ids_and_errors_on_unknown() {
        let entries = vec![ServerFnEntry {
            path: PathBuf::from("/app/src/utils/posts.tsx"),
            name: "fetchPosts".to_string(),
            id: "abc123".to_string(),
        }];
        let module = generate_resolver_module(&entries);
        assert!(module.contains("\"abc123\": { importer: () => import(\"/app/src/utils/posts.tsx\"), functionName: \"fetchPosts\" }"), "{module}");
        assert!(module.contains("__executeServer"), "{module}");
        assert!(module.contains("no server function registered"), "{module}");
        assert!(module.contains("export { getServerFnById }"), "{module}");
    }
}
