use std::collections::BTreeSet;
use std::path::Path;

use oxc_allocator::Allocator;
use oxc_ast::ast::{
    CallExpression, ExportAllDeclaration, ExportNamedDeclaration, Expression, ImportDeclaration,
    ImportExpression, Program,
};
use oxc_ast_visit::{
    Visit,
    walk::{walk_call_expression, walk_export_named_declaration},
};
use oxc_parser::Parser;
use oxc_span::SourceType;

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct ParseResult {
    pub dependencies: Vec<String>,
    pub errors: Vec<String>,
}

/// Which file extensions this project's toolchain lets JSX appear in.
///
/// Toolchains genuinely disagree, so this is an explicit compiler capability,
/// not a framework inference. `.ts`, `.mts`, and `.cts` remain JSX-free because
/// there `<T>x` is a type assertion rather than an element.
#[derive(Debug, Clone, Copy, Default, Eq, PartialEq)]
pub enum JsxExtensions {
    /// Only explicitly JSX-bearing `.jsx` and `.tsx` files may contain JSX.
    #[default]
    JsxAndTsxOnly,
    /// `.js`, `.mjs`, and `.cjs` are JSX-capable as well.
    JsxInJavaScript,
}

/// The parse options for `path` under this project's JSX rule. THE one place a
/// [`SourceType`] is derived from a path — `SourceType::from_path` must not be
/// called anywhere else (enforced by a grep gate in `check.sh`), because a second
/// copy of this rule can silently make one compiler stage reject valid input.
pub fn source_type_for(path: &Path, jsx: JsxExtensions) -> SourceType {
    let source_type = SourceType::from_path(path)
        .unwrap_or_default()
        .with_module(true);
    let js_extension = path
        .extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| matches!(extension, "js" | "mjs" | "cjs"));
    if jsx == JsxExtensions::JsxInJavaScript && js_extension {
        source_type.with_jsx(true)
    } else {
        source_type
    }
}

/// The parse options for an AUXILIARY scan of `path` — a directive-prologue
/// probe, an export enumeration, a `define`/dead-branch rewrite: anything that
/// inspects a module OUTSIDE the one parse whose diagnostics the build reports.
///
/// Deliberately the WIDEST rule rather than the project's own. An auxiliary scan
/// must never be *less* permissive than the module's real parse, or it answers
/// "nothing here" (no `"use client"`, no exports, no defines) for a file the
/// build is about to compile successfully — a silent wrong answer. Being *more*
/// permissive is harmless: JSX-enabled parsing accepts a strict superset of the
/// same source (no valid JavaScript expression begins with `<`), and if the
/// project's real rule rejects the file, the main parse fails the build anyway.
pub fn scan_source_type(path: &Path) -> SourceType {
    source_type_for(path, JsxExtensions::JsxInJavaScript)
}

#[derive(Default)]
struct DependencyVisitor {
    dependencies: Vec<String>,
    dynamic_dependencies: BTreeSet<String>,
    /// Specifiers seen at least once as a `require(...)` lexically inside a `try`
    /// block, in the same function as that `try`.
    guarded_requires: BTreeSet<String>,
    /// Specifiers seen at least once in a position where a resolution failure is
    /// NOT recoverable: a static `import`/`export … from`, a dynamic `import()`,
    /// or a bare `require(...)` outside any `try`.
    unguarded: BTreeSet<String>,
    /// How many enclosing `try` BLOCKS (not `catch`/`finally`) we are inside,
    /// counting only within the current function.
    try_depth: usize,
    /// Specifiers reached by at least one CommonJS `require(...)` call.
    require_syntax: BTreeSet<String>,
    /// Specifiers reached by at least one ESM form: a static `import` /
    /// `export … from`, or a dynamic `import()`.
    import_syntax: BTreeSet<String>,
    /// Specifiers reached by at least one reference that requires the target to be
    /// ALREADY EVALUATED when this module's body runs: a static `import`, an
    /// `export … from`, or a CommonJS `require(...)`. See
    /// [`collect_eager_dependencies`].
    eager_dependencies: BTreeSet<String>,
}

/// Which SYNTAX a module reaches each of its specifiers through.
///
/// This is not bookkeeping: it decides which export conditions the specifier
/// resolves under. `require("x")` resolves under `require`, `import "x"` under
/// `import`, and a package whose `exports` maps them to different files (the
/// near-universal dual-package shape `{"import": "./esm/index.mjs", "require":
/// "./index.js"}`) hands back genuinely different modules. Resolving a
/// `require` call site under `import` yields a Module namespace object where
/// the caller expects the CommonJS export — `class extends require("pg-pool")`
/// then throws `Class extends value [object Module] is not a constructor`.
#[derive(Debug, Default)]
pub struct DependencySyntax {
    pub require: BTreeSet<String>,
    pub import: BTreeSet<String>,
}

/// See [`DependencySyntax`]. Reports EVERY specifier under each syntax that
/// reaches it, including a specifier reached both ways.
pub fn collect_dependency_syntax(program: &Program<'_>) -> DependencySyntax {
    let mut visitor = DependencyVisitor::default();
    visitor.visit_program(program);
    DependencySyntax {
        require: visitor.require_syntax,
        import: visitor.import_syntax,
    }
}

impl DependencyVisitor {
    /// Runs `body` with the try-block depth reset, for a nested function body.
    ///
    /// A `try` does not guard code that merely *closes over* it: in
    /// `try { f = () => require("x") } catch {}` the `require` runs when `f` is
    /// called, long after the `catch` is out of scope. Resetting at every function
    /// boundary keeps "guarded" meaning what it says. The common real shape —
    /// `const dir = () => { try { return require("optional") } catch {} }` — puts
    /// the `try` inside the function and is unaffected.
    fn in_new_function(&mut self, body: impl FnOnce(&mut Self)) {
        let outer = std::mem::take(&mut self.try_depth);
        body(self);
        self.try_depth = outer;
    }
}

impl<'a> Visit<'a> for DependencyVisitor {
    fn visit_import_declaration(&mut self, declaration: &ImportDeclaration<'a>) {
        self.dependencies.push(declaration.source.value.to_string());
        self.unguarded.insert(declaration.source.value.to_string());
        self.import_syntax
            .insert(declaration.source.value.to_string());
        self.eager_dependencies
            .insert(declaration.source.value.to_string());
    }

    fn visit_import_expression(&mut self, expression: &ImportExpression<'a>) {
        if let Expression::StringLiteral(literal) = &expression.source {
            self.dependencies.push(literal.value.to_string());
            self.dynamic_dependencies.insert(literal.value.to_string());
            self.import_syntax.insert(literal.value.to_string());
            // A dynamic import rejects rather than throws, so a `try` around it only
            // catches when the call is awaited in that same `try`. That is not
            // decidable here, so `import()` never counts as guarded.
            self.unguarded.insert(literal.value.to_string());
        }
    }

    fn visit_export_all_declaration(&mut self, declaration: &ExportAllDeclaration<'a>) {
        self.dependencies.push(declaration.source.value.to_string());
        self.unguarded.insert(declaration.source.value.to_string());
        self.import_syntax
            .insert(declaration.source.value.to_string());
        self.eager_dependencies
            .insert(declaration.source.value.to_string());
    }

    fn visit_export_named_declaration(&mut self, declaration: &ExportNamedDeclaration<'a>) {
        if let Some(source) = &declaration.source {
            self.dependencies.push(source.value.to_string());
            self.unguarded.insert(source.value.to_string());
            self.import_syntax.insert(source.value.to_string());
            self.eager_dependencies.insert(source.value.to_string());
        }
        // `export` is a MODIFIER, not a scope: `export const p = import("./a")`
        // holds a dependency exactly as `const p = import("./a")` does. Handling
        // only the `from` clause and returning made every `import()` and
        // `require()` inside an exported declaration invisible to the graph — the
        // module was never discovered, and the emitted call fell through to the
        // registry's external path and threw MODULE_NOT_FOUND at the call site.
        walk_export_named_declaration(self, declaration);
    }

    fn visit_call_expression(&mut self, expression: &CallExpression<'a>) {
        if let Some(literal) = expression.common_js_require() {
            self.dependencies.push(literal.value.to_string());
            self.require_syntax.insert(literal.value.to_string());
            // `require(...)` returns the module's exports SYNCHRONOUSLY, so the target
            // must already be in the registry when the call runs — it can never be
            // deferred into a lazily-loaded chunk.
            self.eager_dependencies.insert(literal.value.to_string());
            if self.try_depth > 0 {
                self.guarded_requires.insert(literal.value.to_string());
            } else {
                self.unguarded.insert(literal.value.to_string());
            }
        }
        walk_call_expression(self, expression);
    }

    fn visit_try_statement(&mut self, statement: &oxc_ast::ast::TryStatement<'a>) {
        self.try_depth += 1;
        self.visit_block_statement(&statement.block);
        self.try_depth -= 1;
        // The `catch` and `finally` clauses run OUTSIDE the protection of their own
        // `try`, so a require in either is unguarded (unless some outer `try` still
        // applies, which the unchanged depth already expresses).
        if let Some(handler) = &statement.handler {
            self.visit_catch_clause(handler);
        }
        if let Some(finalizer) = &statement.finalizer {
            self.visit_block_statement(finalizer);
        }
    }

    fn visit_function(
        &mut self,
        function: &oxc_ast::ast::Function<'a>,
        flags: oxc_semantic::ScopeFlags,
    ) {
        self.in_new_function(|visitor| {
            oxc_ast_visit::walk::walk_function(visitor, function, flags);
        });
    }

    fn visit_arrow_function_expression(
        &mut self,
        arrow: &oxc_ast::ast::ArrowFunctionExpression<'a>,
    ) {
        self.in_new_function(|visitor| {
            oxc_ast_visit::walk::walk_arrow_function_expression(visitor, arrow);
        });
    }
}

pub fn parse_dependencies(path: &Path, source: &str) -> ParseResult {
    let allocator = Allocator::default();
    let parsed = Parser::new(&allocator, source, scan_source_type(path)).parse();

    ParseResult {
        dependencies: collect_dependencies(&parsed.program),
        errors: parsed
            .diagnostics
            .into_iter()
            .map(|error| error.to_string())
            .collect(),
    }
}

pub fn collect_dependencies(program: &Program<'_>) -> Vec<String> {
    let mut visitor = DependencyVisitor::default();
    visitor.visit_program(program);
    // First-occurrence SOURCE order, deduped. Import order is semantic — it is
    // the module execution order (and, through it, the CSS cascade order) — so
    // sorting here would silently reorder side effects.
    let mut seen = BTreeSet::new();
    visitor
        .dependencies
        .retain(|dependency| seen.insert(dependency.clone()));
    visitor.dependencies
}

pub fn collect_dynamic_dependencies(program: &Program<'_>) -> BTreeSet<String> {
    let mut visitor = DependencyVisitor::default();
    visitor.visit_program(program);
    visitor.dynamic_dependencies
}

/// The specifiers this module reaches through a reference that needs the target
/// ALREADY EVALUATED at the point this module's body runs: a static `import`, an
/// `export … from`, or a CommonJS `require(...)`.
///
/// The complement of [`collect_dynamic_dependencies`] is NOT this set, because the two
/// overlap: one module can reach one specifier both ways. The near-universal shape is a
/// barrel that re-exports a component AND lazily imports the same file —
///
/// ```js
/// export { default as Foo } from "./Foo";
/// export const FooLazy = dynamic(() => import("./Foo"));
/// ```
///
/// Reading only the `import()` there says "./Foo is a code-split boundary" and moves it
/// into its own chunk, which is exactly wrong: the `export … from` on the line above
/// resolves synchronously against the registry, and the chunk holding the target has not
/// been loaded. The result is `Module is not loaded: <id>` at first render. An edge is a
/// deferrable chunk boundary only when EVERY reference to it is an `import()`.
pub fn collect_eager_dependencies(program: &Program<'_>) -> BTreeSet<String> {
    let mut visitor = DependencyVisitor::default();
    visitor.visit_program(program);
    visitor.eager_dependencies
}

/// The specifiers this module treats as OPTIONAL: every reference to them is a
/// `require(...)` inside a `try` block of the same function.
///
/// This is the `try { require("bufferutil") } catch {}` idiom — near-universal in
/// packages with native or platform-specific accelerators (`ws`, `pg`, `sharp`,
/// `jsdom`). Node's own semantics are the contract: the `require` throws
/// `MODULE_NOT_FOUND` and the `catch` supplies a fallback. The author has already
/// written down that a resolution failure here is expected and handled, so failing
/// the BUILD over it rejects a program that runs correctly.
///
/// The condition is deliberately all-or-nothing. One unguarded reference anywhere in
/// the module means some code path does depend on the specifier resolving, and the
/// missing module stays a fatal build error.
pub fn collect_optional_dependencies(program: &Program<'_>) -> BTreeSet<String> {
    let mut visitor = DependencyVisitor::default();
    visitor.visit_program(program);
    visitor
        .guarded_requires
        .into_iter()
        .filter(|specifier| !visitor.unguarded.contains(specifier))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn optional_dependencies(source: &str) -> Vec<String> {
        let allocator = Allocator::default();
        let parsed = Parser::new(&allocator, source, scan_source_type(Path::new("m.js"))).parse();
        assert!(parsed.diagnostics.is_empty(), "{:?}", parsed.diagnostics);
        collect_optional_dependencies(&parsed.program)
            .into_iter()
            .collect()
    }

    fn dependencies_of(source: &str) -> Vec<String> {
        let allocator = Allocator::default();
        let parsed = Parser::new(&allocator, source, scan_source_type(Path::new("m.js"))).parse();
        assert!(parsed.diagnostics.is_empty(), "{:?}", parsed.diagnostics);
        collect_dependencies(&parsed.program)
    }

    fn syntax_of(source: &str) -> (Vec<String>, Vec<String>) {
        let allocator = Allocator::default();
        let parsed = Parser::new(&allocator, source, scan_source_type(Path::new("m.js"))).parse();
        assert!(parsed.diagnostics.is_empty(), "{:?}", parsed.diagnostics);
        let syntax = collect_dependency_syntax(&parsed.program);
        (
            syntax.require.into_iter().collect(),
            syntax.import.into_iter().collect(),
        )
    }

    /// `export` is a modifier, not a scope. Handling only the `from` clause and
    /// returning made every dependency inside an exported declaration invisible —
    /// `cal.com`'s `export const AppSetupPageMap = { alby: import("...") }` was
    /// never discovered, and the emitted `import()` threw MODULE_NOT_FOUND.
    #[test]
    fn a_dependency_inside_an_exported_declaration_is_collected() {
        assert_eq!(
            dependencies_of(r#"export const map = { a: import("./a") };"#),
            ["./a"]
        );
        assert_eq!(
            dependencies_of(r#"export const dep = require("./cjs");"#),
            ["./cjs"]
        );
        assert_eq!(
            dependencies_of(r#"export function load() { return import("./lazy"); }"#),
            ["./lazy"]
        );
        // The `from` clause still counts, and both are reported when a statement
        // has one and an exported declaration is present elsewhere.
        assert_eq!(
            dependencies_of("export { a } from \"./re\";\nexport const b = require(\"./cjs\");"),
            ["./re", "./cjs"]
        );
    }

    /// A dynamic `import()` inside an exported declaration is still DYNAMIC, so it
    /// keeps rooting its own chunk rather than being pulled into the main one.
    #[test]
    fn an_exported_declarations_dynamic_import_is_still_dynamic() {
        let allocator = Allocator::default();
        let source = r#"export const map = { a: import("./a") };"#;
        let parsed = Parser::new(&allocator, source, scan_source_type(Path::new("m.js"))).parse();
        let dynamic = collect_dynamic_dependencies(&parsed.program);
        assert!(dynamic.contains("./a"), "{dynamic:?}");
    }

    #[test]
    fn each_specifier_records_the_syntax_that_reaches_it() {
        let (require, import) = syntax_of(
            r#"
                import "./esm";
                export * from "./star";
                const cjs = require("./cjs");
                export const lazy = import("./dyn");
            "#,
        );
        assert_eq!(require, ["./cjs"]);
        assert_eq!(import, ["./dyn", "./esm", "./star"]);
    }

    #[test]
    fn a_specifier_reached_both_ways_is_reported_under_both() {
        let (require, import) = syntax_of(
            r#"
                const eager = require("dual");
                export const lazy = import("dual");
            "#,
        );
        assert_eq!(require, ["dual"]);
        assert_eq!(import, ["dual"]);
    }

    /// The lazy-component barrel: one specifier reached by BOTH an `export … from`
    /// and an `import()`. The static reference makes the target eager — the graph must
    /// see that, or the module is moved into a chunk that has not been fetched when the
    /// barrel's synchronous lookup runs.
    #[test]
    fn a_specifier_reached_both_statically_and_dynamically_is_eager() {
        let eager = eager_dependencies(
            r#"
                export { default as Foo } from "./Foo";
                export const FooLazy = dynamic(() => import("./Foo"));
                export const Bar = dynamic(() => import("./Bar"));
                const cjs = require("./cjs");
            "#,
        );
        assert!(
            eager.contains("./Foo"),
            "a static re-export is eager: {eager:?}"
        );
        assert!(eager.contains("./cjs"), "a require is eager: {eager:?}");
        assert!(
            !eager.contains("./Bar"),
            "an import()-ONLY specifier stays deferrable: {eager:?}"
        );
    }

    fn eager_dependencies(source: &str) -> BTreeSet<String> {
        let allocator = Allocator::default();
        let parsed = Parser::new(&allocator, source, scan_source_type(Path::new("m.js"))).parse();
        collect_eager_dependencies(&parsed.program)
    }

    #[test]
    fn a_require_inside_a_try_is_optional_and_one_outside_is_not() {
        // The `ws` / `pg` / `sharp` shape, plus its negation in the same module.
        let optional = optional_dependencies(
            r#"
                try { module.exports.fast = require("bufferutil"); } catch {}
                const always = require("./local.js");
            "#,
        );
        assert_eq!(optional, ["bufferutil"]);
    }

    #[test]
    fn one_unguarded_reference_disqualifies_a_specifier_entirely() {
        // A module that also needs the package on an unguarded path really does
        // depend on it resolving; the guarded copy must not launder that away.
        assert!(
            optional_dependencies(
                r#"
                    try { require("half-optional"); } catch {}
                    const needed = require("half-optional");
                "#,
            )
            .is_empty()
        );
    }

    #[test]
    fn a_try_does_not_guard_a_require_that_merely_closes_over_it() {
        // `f` runs after the `catch` is long gone, so the throw is uncaught.
        assert!(
            optional_dependencies(r#"try { var f = () => require("later"); } catch {}"#).is_empty()
        );
    }

    #[test]
    fn a_require_in_a_catch_or_finally_clause_is_not_guarded_by_its_own_try() {
        // Both clauses run OUTSIDE the protection of the `try` they belong to.
        assert!(optional_dependencies(r#"try { x(); } catch { require("in-catch"); }"#).is_empty());
        assert!(
            optional_dependencies(r#"try { x(); } finally { require("in-finally"); }"#).is_empty()
        );
    }

    #[test]
    fn a_try_wrapped_dynamic_import_or_static_import_is_never_optional() {
        // `import()` rejects rather than throws, so a surrounding `try` catches only
        // when the call is awaited there — not decidable from the specifier alone.
        // A static `import` is hoisted and cannot be guarded at all.
        assert!(optional_dependencies(r#"try { import("maybe"); } catch {}"#).is_empty());
        assert!(
            optional_dependencies(
                "import x from \"eager\";\ntry { require(\"eager\"); } catch {}\n"
            )
            .is_empty()
        );
    }

    #[test]
    fn a_try_inside_a_function_still_guards_its_own_require() {
        // sharp's actual shape: the `try` is in the function body, so resetting the
        // depth at the function boundary must not lose it.
        assert_eq!(
            optional_dependencies(
                r#"const dir = () => { try { return require("@img/sharp-libvips-dev/include"); } catch {} return ""; };"#,
            ),
            ["@img/sharp-libvips-dev/include"]
        );
    }

    #[test]
    fn extracts_static_reexport_and_literal_dynamic_dependencies() {
        let parsed = parse_dependencies(
            Path::new("example.ts"),
            r#"
                import { a } from "./a.js";
                export { b } from "./b.js";
                export * from "./c.js";
                const d = import("./d.js");
                const ignored = import(`./${name}.js`);
                const commonjs = require("./e.cjs");
            "#,
        );

        assert!(parsed.errors.is_empty(), "{:?}", parsed.errors);
        assert_eq!(
            parsed.dependencies,
            ["./a.js", "./b.js", "./c.js", "./d.js", "./e.cjs"]
        );
    }

    #[test]
    fn permissive_mode_enables_jsx_for_js_mjs_and_cjs() {
        for name in ["page.js", "page.mjs", "page.cjs", "page.jsx", "page.tsx"] {
            assert!(
                source_type_for(Path::new(name), JsxExtensions::JsxInJavaScript).is_jsx(),
                "{name} must be JSX-capable in permissive mode"
            );
        }
        // `<T>x` in a `.ts` module is a type assertion, not an element.
        let ts = source_type_for(Path::new("module.ts"), JsxExtensions::JsxInJavaScript);
        assert!(!ts.is_jsx(), "a .ts module must stay JSX-free");
        assert!(ts.is_typescript());
    }

    #[test]
    fn explicit_mode_enables_jsx_only_for_jsx_and_tsx() {
        for name in ["main.js", "main.mjs", "main.cjs", "main.ts"] {
            assert!(
                !source_type_for(Path::new(name), JsxExtensions::JsxAndTsxOnly).is_jsx(),
                "{name} must be plain under the explicit-extension rule"
            );
        }
        for name in ["main.jsx", "main.tsx"] {
            assert!(source_type_for(Path::new(name), JsxExtensions::JsxAndTsxOnly).is_jsx());
        }
    }

    #[test]
    fn an_auxiliary_scan_never_narrows_the_projects_own_rule() {
        for name in [
            "page.js", "page.mjs", "page.cjs", "page.jsx", "page.tsx", "page.ts",
        ] {
            let path = Path::new(name);
            for jsx in [JsxExtensions::JsxAndTsxOnly, JsxExtensions::JsxInJavaScript] {
                assert!(
                    scan_source_type(path).is_jsx() >= source_type_for(path, jsx).is_jsx(),
                    "{name}: the auxiliary scan is narrower than the project rule {jsx:?}"
                );
            }
        }
    }

    #[test]
    fn a_jsx_bearing_js_module_yields_its_dependencies() {
        let parsed = parse_dependencies(
            Path::new("pages/index.js"),
            r#"
                import Layout from "../components/Layout";
                export default function Home() {
                    return <Layout><h1>hi</h1></Layout>;
                }
            "#,
        );

        assert!(parsed.errors.is_empty(), "{:?}", parsed.errors);
        assert_eq!(parsed.dependencies, ["../components/Layout"]);
    }
}
