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
    let source_type = SourceType::from_path(path)
        .unwrap_or_default()
        .with_module(true);
    let parsed = Parser::new(&allocator, source, source_type).parse();

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
}
