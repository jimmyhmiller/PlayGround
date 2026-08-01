//! TanStack Start environment-directive semantic transform.

use std::collections::HashMap;
use std::path::Path;

use diffpack_core::compiler::SemanticTransform;
use diffpack_core::transform::Target;
use oxc_allocator::{Allocator, TakeIn};
use oxc_ast::ast::{Expression, ImportDeclarationSpecifier, Program, Statement};
use oxc_ast_visit::{VisitMut, walk_mut};
use oxc_parser::Parser;
use oxc_semantic::Scoping;
use oxc_span::SourceType;
use oxc_syntax::symbol::SymbolId;

#[derive(Debug, Default, Clone, Copy)]
pub struct TanStackSemanticTransform;

impl SemanticTransform for TanStackSemanticTransform {
    fn apply<'a>(
        &self,
        allocator: &'a Allocator,
        program: &mut Program<'a>,
        scoping: &Scoping,
        target: Target,
        path: &Path,
    ) -> bool {
        apply_tanstack_env_transform(allocator, program, scoping, target, path)
    }
}

/// Which TanStack Start environment-directive helper an imported binding refers
/// to. These are `@tanstack/*` runtime stubs that a build tool is expected to
/// specialize per environment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EnvFn {
    ServerOnly,
    ClientOnly,
    Isomorphic,
    Middleware,
}

/// Specializes TanStack Start's environment-directive helpers for `target`,
/// mirroring `@tanstack/start-plugin-core`'s `handleEnvOnly` /
/// `handleCreateIsomorphicFn` compiler passes:
///
/// - `createServerOnlyFn(fn)` keeps `fn` on the server; on the client it becomes
///   a throwing stub (the reference to `fn` is dropped).
/// - `createClientOnlyFn(fn)` is the mirror image.
/// - `createIsomorphicFn().client(a).server(b)` collapses to `a` on the client
///   and `b` on the server (or `() => {}` when the chosen environment has no
///   implementation).
/// - `createMiddleware()...server(fn)` drops its `.server`/`.validator`/
///   `.inputValidator` calls on the client, severing references to server-only
///   code (e.g. an API route's `getRequestHeaders`).
///
/// Only helpers imported from a `@tanstack/` package are matched, resolved by
/// symbol so a same-named local binding is never rewritten. Returns whether any
/// rewrite happened, so the caller can rebuild scoping (the pass deletes
/// references, which the demand computation must observe to prune the
/// now-unused server imports). This is currently a no-op for `Target::Server`,
/// whose neutral runtime stubs already behave correctly under Node.
#[doc(hidden)]
pub fn apply_tanstack_env_transform<'a>(
    allocator: &'a Allocator,
    program: &mut Program<'a>,
    scoping: &Scoping,
    target: Target,
    path: &Path,
) -> bool {
    if target != Target::Client {
        return false;
    }
    // A `@tanstack/*` package bundles these environment-directive helpers as
    // *local* modules (`createServerOnlyFn` from `./envOnly.js`,
    // `createIsomorphicFn` from `./createIsomorphicFn.js`), and its own modules
    // import them by relative specifier rather than through the package name. The
    // reference TanStack plugin matches these helpers by their well-known names
    // regardless of import source; mirror that, but only inside a `@tanstack`
    // package so a same-named helper in the user's own app is never rewritten.
    let in_tanstack_package = path
        .components()
        .any(|component| component.as_os_str() == "@tanstack");
    let mut kinds: HashMap<SymbolId, EnvFn> = HashMap::new();
    for statement in &program.body {
        let Statement::ImportDeclaration(declaration) = statement else {
            continue;
        };
        let specifier = declaration.source.value.as_str();
        let is_directive_source = specifier.starts_with("@tanstack/")
            || (in_tanstack_package
                && (specifier.starts_with("./") || specifier.starts_with("../")));
        if !is_directive_source {
            continue;
        }
        let Some(specifiers) = &declaration.specifiers else {
            continue;
        };
        for specifier in specifiers {
            let ImportDeclarationSpecifier::ImportSpecifier(specifier) = specifier else {
                continue;
            };
            let kind = match specifier.imported.name().as_str() {
                "createServerOnlyFn" => EnvFn::ServerOnly,
                "createClientOnlyFn" => EnvFn::ClientOnly,
                "createIsomorphicFn" => EnvFn::Isomorphic,
                "createMiddleware" => EnvFn::Middleware,
                _ => continue,
            };
            kinds.insert(specifier.local.symbol_id(), kind);
        }
    }
    if kinds.is_empty() {
        return false;
    }
    let mut transform = EnvTransform {
        allocator,
        scoping,
        kinds,
        target,
        changed: false,
    };
    transform.visit_program(program);
    transform.changed
}

struct EnvTransform<'a, 's> {
    allocator: &'a Allocator,
    scoping: &'s Scoping,
    kinds: HashMap<SymbolId, EnvFn>,
    target: Target,
    changed: bool,
}

impl<'a> EnvTransform<'a, '_> {
    /// The [`EnvFn`] an identifier reference resolves to, if it is one of the
    /// tracked `@tanstack/*` imports.
    fn env_fn(&self, identifier: &oxc_ast::ast::IdentifierReference<'a>) -> Option<EnvFn> {
        let reference_id = identifier.reference_id.get()?;
        let symbol_id = self.scoping.get_reference(reference_id).symbol_id()?;
        self.kinds.get(&symbol_id).copied()
    }

    /// Parses a constant JavaScript expression into this module's arena. Used to
    /// synthesize the throwing / empty-arrow replacements.
    fn parse_expression(&self, source: &'static str) -> Expression<'a> {
        let parsed = Parser::new(self.allocator, source, SourceType::default()).parse();
        let mut program = parsed.program;
        match program.body.first_mut() {
            Some(Statement::ExpressionStatement(statement)) => {
                statement.expression.take_in(&self.allocator)
            }
            _ => unreachable!("env-transform replacement source must be a single expression"),
        }
    }

    fn throwing_stub(&self, function: &str, environment: &str) -> Expression<'a> {
        // A distinct constant per (function, environment) so the parser sees a
        // 'static string; the set is closed and tiny.
        let source = match (function, environment) {
            ("createServerOnlyFn", "server") => {
                "(() => { throw new Error(\"createServerOnlyFn() functions can only be called on the server!\") })"
            }
            ("createClientOnlyFn", "client") => {
                "(() => { throw new Error(\"createClientOnlyFn() functions can only be called on the client!\") })"
            }
            _ => unreachable!("no throwing stub for {function}/{environment}"),
        };
        self.parse_expression(source)
    }

    /// Rewrites `createServerOnlyFn(fn)` / `createClientOnlyFn(fn)`. Returns
    /// `true` if `expression` was a matching call (and was replaced).
    fn rewrite_env_only(&mut self, expression: &mut Expression<'a>) -> bool {
        let Expression::CallExpression(call) = expression else {
            return false;
        };
        let Expression::Identifier(callee) = &call.callee else {
            return false;
        };
        let kind = match self.env_fn(callee) {
            Some(kind @ (EnvFn::ServerOnly | EnvFn::ClientOnly)) => kind,
            _ => return false,
        };
        let keep = matches!(
            (kind, self.target),
            (EnvFn::ServerOnly, Target::Server) | (EnvFn::ClientOnly, Target::Client)
        );
        if keep {
            // Replace the whole call with its inner function argument.
            let Some(inner) = call
                .arguments
                .first_mut()
                .and_then(|argument| argument.as_expression_mut())
            else {
                return false;
            };
            *expression = inner.take_in(&self.allocator);
        } else {
            let (function, environment) = match kind {
                EnvFn::ServerOnly => ("createServerOnlyFn", "server"),
                EnvFn::ClientOnly => ("createClientOnlyFn", "client"),
                EnvFn::Isomorphic | EnvFn::Middleware => unreachable!(),
            };
            *expression = self.throwing_stub(function, environment);
        }
        true
    }

    /// Validates that `expression` is a complete
    /// `createIsomorphicFn()[.client(_)][.server(_)]` chain (read-only).
    fn is_isomorphic_chain(&self, expression: &Expression<'a>) -> bool {
        let Expression::CallExpression(call) = expression else {
            return false;
        };
        match &call.callee {
            Expression::Identifier(callee) => {
                self.env_fn(callee) == Some(EnvFn::Isomorphic) && call.arguments.is_empty()
            }
            Expression::StaticMemberExpression(member) => {
                matches!(member.property.name.as_str(), "client" | "server")
                    && self.is_isomorphic_chain(&member.object)
            }
            _ => false,
        }
    }

    /// Extracts the `.client` / `.server` implementation arguments from a
    /// validated isomorphic chain, consuming the chain.
    fn extract_isomorphic(
        &self,
        expression: &mut Expression<'a>,
        client: &mut Option<Expression<'a>>,
        server: &mut Option<Expression<'a>>,
    ) {
        let Expression::CallExpression(call) = expression else {
            return;
        };
        // Take the method argument before borrowing `callee`, so the two
        // disjoint field borrows never overlap.
        let argument = call
            .arguments
            .first_mut()
            .and_then(|argument| argument.as_expression_mut())
            .map(|argument| argument.take_in(&self.allocator));
        let Expression::StaticMemberExpression(member) = &mut call.callee else {
            return;
        };
        self.extract_isomorphic(&mut member.object, client, server);
        match member.property.name.as_str() {
            "client" => *client = argument,
            "server" => *server = argument,
            _ => {}
        }
    }

    /// Rewrites a full isomorphic chain to the target's implementation. Returns
    /// `true` if `expression` was such a chain.
    fn rewrite_isomorphic(&mut self, expression: &mut Expression<'a>) -> bool {
        // Only the outermost chain node (its callee is a `.client`/`.server`
        // member) is a rewrite point; the bare `createIsomorphicFn()` base is
        // left for its enclosing member call to consume.
        let is_chain_tail = matches!(
            expression,
            Expression::CallExpression(call)
                if matches!(&call.callee, Expression::StaticMemberExpression(_))
        );
        if !is_chain_tail || !self.is_isomorphic_chain(expression) {
            return false;
        }
        let mut client = None;
        let mut server = None;
        self.extract_isomorphic(expression, &mut client, &mut server);
        let chosen = match self.target {
            Target::Client => client,
            // IsolatedServer is server-like for the isomorphic chain. Unreachable in
            // practice (`apply_env_transform` returns early for non-`Client`), but
            // required for the match to be exhaustive.
            Target::Server | Target::IsolatedServer => server,
        };
        *expression = chosen.unwrap_or_else(|| self.parse_expression("(() => {})"));
        true
    }

    /// Whether `expression` is a `createMiddleware()[.method(_)]*` chain.
    fn is_middleware_chain(&self, expression: &Expression<'a>) -> bool {
        let Expression::CallExpression(call) = expression else {
            return false;
        };
        match &call.callee {
            Expression::Identifier(callee) => {
                self.env_fn(callee) == Some(EnvFn::Middleware) && call.arguments.is_empty()
            }
            Expression::StaticMemberExpression(member) => self.is_middleware_chain(&member.object),
            _ => false,
        }
    }

    /// Strips the environment-specific method calls from a validated
    /// `createMiddleware` chain, mirroring `handleCreateMiddleware`: on the
    /// client the `.server(...)`, `.validator(...)` and `.inputValidator(...)`
    /// calls are removed (severing their references to server-only code), while
    /// `.middleware(...)` and `.client(...)` are kept. Operates bottom-up so a
    /// stripped level is spliced out cleanly.
    fn strip_middleware(&mut self, expression: &mut Expression<'a>) {
        let Expression::CallExpression(call) = expression else {
            return;
        };
        let Expression::StaticMemberExpression(member) = &mut call.callee else {
            return;
        };
        self.strip_middleware(&mut member.object);
        let strip = matches!(
            member.property.name.as_str(),
            "server" | "validator" | "inputValidator"
        );
        if strip {
            let object = member.object.take_in(&self.allocator);
            *expression = object;
            self.changed = true;
        }
    }
}

impl<'a> VisitMut<'a> for EnvTransform<'a, '_> {
    fn visit_expression(&mut self, expression: &mut Expression<'a>) {
        if self.rewrite_env_only(expression) || self.rewrite_isomorphic(expression) {
            self.changed = true;
            // Descend into the replacement so a nested directive helper (e.g. an
            // isomorphic impl that itself calls a server-only fn) is handled too.
            self.visit_expression(expression);
            return;
        }
        if self.is_middleware_chain(expression) {
            // Strip the server-only method calls, then descend into what remains
            // (kept `.client`/`.middleware` arguments may contain their own
            // directive helpers). Re-visiting the whole node instead would loop,
            // since a stripped chain is still a chain.
            self.strip_middleware(expression);
            walk_mut::walk_expression(self, expression);
            return;
        }
        walk_mut::walk_expression(self, expression);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use diffpack_core::compiler::{PreparedSource, transform_prepared_module_in_language_with};
    use diffpack_core::parser::JsxExtensions;
    use diffpack_core::source_map::MapOrigin;
    use diffpack_core::transform::{
        DependencyDemand, ProjectConfig, SourceLanguage, TransformResult,
    };

    fn transform(source: &str, target: Target) -> TransformResult {
        transform_prepared_module_in_language_with(
            Path::new("mod.js"),
            PreparedSource {
                code: source,
                force_jsx: false,
                map_origin: MapOrigin::File,
            },
            target,
            false,
            JsxExtensions::default(),
            &ProjectConfig::default(),
            SourceLanguage::FromPath,
            false,
            &TanStackSemanticTransform,
        )
    }

    fn demand<'a>(result: &'a TransformResult, specifier: &str) -> Option<&'a DependencyDemand> {
        result
            .dependency_demands
            .iter()
            .find(|demand| demand.specifier == specifier)
    }

    #[test]
    fn client_neutralizes_server_only_fn_and_drops_its_server_import() {
        let source = r#"
            import { createServerOnlyFn } from "@tanstack/start-fn-stubs";
            import { getStartContext } from "@tanstack/start-storage-context";
            export const getStartContextServerOnly = createServerOnlyFn(getStartContext);
        "#;
        let client = transform(source, Target::Client);
        assert!(client.code.contains("can only be called on the server"));
        let storage = demand(&client, "@tanstack/start-storage-context").unwrap();
        assert!(!storage.all && storage.names.is_empty());

        let server = transform(source, Target::Server);
        assert_eq!(
            demand(&server, "@tanstack/start-storage-context")
                .unwrap()
                .names,
            ["getStartContext"]
        );
    }

    #[test]
    fn client_collapses_isomorphic_fn_to_client_impl() {
        let source = r#"
            import { createIsomorphicFn } from "@tanstack/start-fn-stubs";
            import { getStartContext } from "@tanstack/start-storage-context";
            export const getRouterInstance = createIsomorphicFn()
                .client(() => window.__TSR_ROUTER__)
                .server(() => getStartContext().getRouter());
        "#;
        let client = transform(source, Target::Client);
        assert!(client.code.contains("__TSR_ROUTER__"));
        assert!(!client.code.contains("getStartContext"));
        assert!(
            demand(&client, "@tanstack/start-storage-context")
                .is_none_or(|demand| !demand.all && demand.names.is_empty())
        );
    }

    #[test]
    fn same_named_local_binding_is_ignored() {
        let client = transform(
            "const createServerOnlyFn = (fn) => fn; export const value = createServerOnlyFn(() => 1);",
            Target::Client,
        );
        assert!(!client.code.contains("can only be called on the server"));
    }
}
