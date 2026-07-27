use std::collections::BTreeSet;
use std::path::Path;

use oxc_allocator::Allocator;
use oxc_ast::ast::{
    CallExpression, ExportAllDeclaration, ExportNamedDeclaration, Expression, ImportDeclaration,
    ImportExpression, Program,
};
use oxc_ast_visit::{Visit, walk::walk_call_expression};
use oxc_parser::Parser;
use oxc_span::SourceType;

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct ParseResult {
    pub dependencies: Vec<String>,
    pub errors: Vec<String>,
}

/// Which file extensions this project's toolchain lets JSX appear in.
///
/// Toolchains genuinely disagree, so this cannot be a global constant:
///
/// * Vite/esbuild parse `.js` as plain JavaScript on purpose — JSX there is a
///   syntax error, and a Vite app that wants it renames the file or configures
///   `esbuild.include`/`loader` (which diffpack does not honor; see the error
///   message in [`crate::transform`]).
/// * Next.js runs its SWC loader over `test: /\.(tsx|ts|js|cjs|mjs|jsx)$/` and
///   sets `[isTypeScript ? 'tsx' : 'jsx']: !isTSFile`, i.e. JSX is enabled for
///   everything that is not a plain `.ts`. Real Next apps rely on it heavily.
///
/// `.ts` is JSX-free under BOTH kinds: there `<T>x` is a type assertion, not an
/// element. `.mts`/`.cts` are likewise left as plain TypeScript — Next's loader
/// never sees those extensions.
#[derive(Debug, Clone, Copy, Default, Eq, PartialEq)]
pub enum JsxExtensions {
    /// Vite/esbuild/generic bundling: only `.jsx` and `.tsx` may contain JSX.
    #[default]
    JsxAndTsxOnly,
    /// Next.js: `.js`, `.mjs` and `.cjs` are JSX-capable as well.
    NextJs,
}

/// The parse options for `path` under this project's JSX rule. THE one place a
/// [`SourceType`] is derived from a path — `SourceType::from_path` must not be
/// called anywhere else (enforced by a grep gate in `check.sh`), because a second
/// copy of this rule is exactly how a Next page silently became a syntax error.
pub fn source_type_for(path: &Path, jsx: JsxExtensions) -> SourceType {
    let source_type = SourceType::from_path(path).unwrap_or_default().with_module(true);
    let js_extension = path
        .extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| matches!(extension, "js" | "mjs" | "cjs"));
    if jsx == JsxExtensions::NextJs && js_extension {
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
    source_type_for(path, JsxExtensions::NextJs)
}

#[derive(Default)]
struct DependencyVisitor {
    dependencies: Vec<String>,
    dynamic_dependencies: BTreeSet<String>,
}

impl<'a> Visit<'a> for DependencyVisitor {
    fn visit_import_declaration(&mut self, declaration: &ImportDeclaration<'a>) {
        self.dependencies.push(declaration.source.value.to_string());
    }

    fn visit_import_expression(&mut self, expression: &ImportExpression<'a>) {
        if let Expression::StringLiteral(literal) = &expression.source {
            self.dependencies.push(literal.value.to_string());
            self.dynamic_dependencies.insert(literal.value.to_string());
        }
    }

    fn visit_export_all_declaration(&mut self, declaration: &ExportAllDeclaration<'a>) {
        self.dependencies.push(declaration.source.value.to_string());
    }

    fn visit_export_named_declaration(&mut self, declaration: &ExportNamedDeclaration<'a>) {
        if let Some(source) = &declaration.source {
            self.dependencies.push(source.value.to_string());
        }
    }

    fn visit_call_expression(&mut self, expression: &CallExpression<'a>) {
        if let Some(literal) = expression.common_js_require() {
            self.dependencies.push(literal.value.to_string());
        }
        walk_call_expression(self, expression);
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
    visitor.dependencies.retain(|dependency| seen.insert(dependency.clone()));
    visitor.dependencies
}

pub fn collect_dynamic_dependencies(program: &Program<'_>) -> BTreeSet<String> {
    let mut visitor = DependencyVisitor::default();
    visitor.visit_program(program);
    visitor.dynamic_dependencies
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn next_enables_jsx_for_js_mjs_and_cjs() {
        for name in ["page.js", "page.mjs", "page.cjs", "page.jsx", "page.tsx"] {
            assert!(
                source_type_for(Path::new(name), JsxExtensions::NextJs).is_jsx(),
                "{name} must be JSX-capable under Next"
            );
        }
        // `<T>x` in a `.ts` module is a type assertion, not an element.
        let ts = source_type_for(Path::new("module.ts"), JsxExtensions::NextJs);
        assert!(!ts.is_jsx(), "a .ts module must stay JSX-free even under Next");
        assert!(ts.is_typescript());
    }

    #[test]
    fn vite_enables_jsx_only_for_jsx_and_tsx() {
        for name in ["main.js", "main.mjs", "main.cjs", "main.ts"] {
            assert!(
                !source_type_for(Path::new(name), JsxExtensions::JsxAndTsxOnly).is_jsx(),
                "{name} must be plain (non-JSX) under the Vite/esbuild rule"
            );
        }
        for name in ["main.jsx", "main.tsx"] {
            assert!(source_type_for(Path::new(name), JsxExtensions::JsxAndTsxOnly).is_jsx());
        }
    }

    #[test]
    fn an_auxiliary_scan_never_narrows_the_projects_own_rule() {
        for name in ["page.js", "page.mjs", "page.cjs", "page.jsx", "page.tsx", "page.ts"] {
            let path = Path::new(name);
            for jsx in [JsxExtensions::JsxAndTsxOnly, JsxExtensions::NextJs] {
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
