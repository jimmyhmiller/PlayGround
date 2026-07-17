use std::collections::HashMap;
use std::path::Path;

use oxc_allocator::Allocator;
use oxc_ast::{
    AstKind,
    ast::{ExportDefaultDeclarationKind, ImportDeclarationSpecifier, ImportExpression, Statement},
};
use oxc_ast_visit::Visit;
use oxc_codegen::Codegen;
use oxc_parser::Parser;
use oxc_semantic::SemanticBuilder;
use oxc_span::{GetSpan, SourceType, Span};
use oxc_syntax::module_record::{ExportExportName, ExportLocalName, ModuleRecord};
use oxc_transformer::{TransformOptions, Transformer};

use crate::frontend_profile::{self, Phase};
use crate::parser::collect_dependencies;

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct TransformResult {
    pub code: String,
    pub diagnostics: Vec<String>,
    pub is_esm: bool,
    pub dependencies: Vec<String>,
}

pub fn transform_module(path: &Path, source: &str) -> TransformResult {
    if path
        .extension()
        .is_some_and(|extension| extension == "json")
    {
        return TransformResult {
            code: format!("module.exports = {source};\n"),
            diagnostics: Vec::new(),
            is_esm: false,
            dependencies: Vec::new(),
        };
    }

    let transform_started = frontend_profile::start();
    let allocator = Allocator::default();
    let source_type = SourceType::from_path(path)
        .unwrap_or_default()
        .with_module(true);
    let parsed = Parser::new(&allocator, source, source_type).parse();
    let mut diagnostics = parsed
        .diagnostics
        .into_iter()
        .map(|diagnostic| diagnostic.to_string())
        .collect::<Vec<_>>();
    let mut program = parsed.program;

    let semantic = SemanticBuilder::new()
        .with_excess_capacity(2.0)
        .with_enum_eval(true)
        .build(&program);
    diagnostics.extend(
        semantic
            .diagnostics
            .into_iter()
            .map(|diagnostic| diagnostic.to_string()),
    );
    let transformed = Transformer::new(&allocator, path, &TransformOptions::default())
        .build_with_scoping(semantic.semantic.into_scoping(), &mut program);
    diagnostics.extend(
        transformed
            .diagnostics
            .into_iter()
            .map(|diagnostic| diagnostic.to_string()),
    );

    let transformed_code = Codegen::new().build(&program).code;
    frontend_profile::finish(Phase::Transform, transform_started);
    let lower_started = frontend_profile::start();
    let (code, lower_diagnostics, is_esm, dependencies) =
        lower_module_syntax(path, &transformed_code);
    frontend_profile::finish(Phase::Lower, lower_started);
    diagnostics.extend(lower_diagnostics);
    TransformResult {
        code,
        diagnostics,
        is_esm,
        dependencies,
    }
}

fn lower_module_syntax(path: &Path, source: &str) -> (String, Vec<String>, bool, Vec<String>) {
    let allocator = Allocator::default();
    let source_type = SourceType::from_path(path)
        .unwrap_or_default()
        .with_typescript(false)
        .with_jsx(false)
        .with_module(true);
    let parsed = Parser::new(&allocator, source, source_type).parse();
    let mut diagnostics = parsed
        .diagnostics
        .into_iter()
        .map(|diagnostic| diagnostic.to_string())
        .collect::<Vec<_>>();
    let is_esm = parsed.module_record.has_module_syntax;
    let dependencies = collect_dependencies(&parsed.program);
    let mut dynamic_imports = DynamicImportCollector { edits: Vec::new() };
    dynamic_imports.visit_program(&parsed.program);
    if !is_esm {
        return (
            apply_edits(source.to_string(), dynamic_imports.edits),
            diagnostics,
            false,
            dependencies,
        );
    }

    let semantic = SemanticBuilder::new()
        .with_build_nodes(true)
        .build(&parsed.program);
    diagnostics.extend(
        semantic
            .diagnostics
            .into_iter()
            .map(|diagnostic| diagnostic.to_string()),
    );
    let semantic = semantic.semantic;
    let module_spans = parsed
        .program
        .body
        .iter()
        .filter_map(module_statement_span)
        .collect::<Vec<_>>();

    let mut edits = dynamic_imports.edits;
    let mut preamble_declarations = String::new();
    let mut preamble_exports = String::new();
    let mut import_bindings = HashMap::<String, String>::new();
    let mut import_index = 0_usize;
    let mut default_index = 0_usize;

    // First establish stable namespace slots and rewrite every use of an
    // imported binding into a property read. This preserves live bindings and
    // defers reads in cycles until the JavaScript expression is evaluated.
    for statement in &parsed.program.body {
        let Statement::ImportDeclaration(declaration) = statement else {
            continue;
        };
        let Some(specifiers) = &declaration.specifiers else {
            continue;
        };
        if specifiers.is_empty() {
            continue;
        }
        let dependency = format!("__diffpack_import_{import_index}");
        import_index += 1;
        preamble_declarations.push_str(&format!("let {dependency};\n"));
        for specifier in specifiers {
            let (local, expression, symbol_id) = match specifier {
                ImportDeclarationSpecifier::ImportDefaultSpecifier(specifier) => (
                    specifier.local.name.to_string(),
                    format!("__import({dependency},\"default\")"),
                    specifier.local.symbol_id(),
                ),
                ImportDeclarationSpecifier::ImportNamespaceSpecifier(specifier) => (
                    specifier.local.name.to_string(),
                    dependency.clone(),
                    specifier.local.symbol_id(),
                ),
                ImportDeclarationSpecifier::ImportSpecifier(specifier) => (
                    specifier.local.name.to_string(),
                    format!(
                        "__import({dependency},{})",
                        quote(&specifier.imported.name())
                    ),
                    specifier.local.symbol_id(),
                ),
            };
            import_bindings.insert(local.clone(), expression.clone());
            for reference in semantic.symbol_references(symbol_id) {
                let node = semantic.nodes().get_node(reference.node_id());
                let span = node.kind().span();
                if module_spans.iter().any(|module| contains(*module, span)) {
                    continue;
                }
                if let AstKind::ObjectProperty(property) =
                    semantic.nodes().parent_kind(reference.node_id())
                    && property.shorthand
                {
                    edits.push(Edit::replace(
                        property.span,
                        format!("{local}:{expression}"),
                    ));
                } else {
                    edits.push(Edit::replace(span, expression.clone()));
                }
            }
        }
    }

    import_index = 0;
    for statement in &parsed.program.body {
        match statement {
            Statement::ImportDeclaration(declaration) => {
                let request = quote(&declaration.source.value);
                let replacement = match &declaration.specifiers {
                    None => format!("require({request});"),
                    Some(specifiers) if specifiers.is_empty() => format!("require({request});"),
                    Some(_) => {
                        let dependency = format!("__diffpack_import_{import_index}");
                        import_index += 1;
                        format!("{dependency}=__toESM(require({request}));")
                    }
                };
                edits.push(Edit::replace(declaration.span, replacement));
            }
            Statement::ExportNamedDeclaration(declaration) => {
                let mut replacement = String::new();
                let mut replacement_span = declaration.span;
                if let Some(inner) = &declaration.declaration {
                    replacement_span = Span::new(declaration.span.start, inner.span().start);
                    preamble_exports.push_str(&local_exports_for_span(
                        &parsed.module_record,
                        declaration.span,
                    ));
                } else if let Some(request) = &declaration.source {
                    let dependency = format!("__diffpack_reexport_{import_index}");
                    import_index += 1;
                    preamble_declarations.push_str(&format!("let {dependency};\n"));
                    for specifier in &declaration.specifiers {
                        preamble_exports.push_str(&export_getter(
                            &specifier.exported.name(),
                            &format!("__import({dependency},{})", quote(&specifier.local.name())),
                        ));
                    }
                    replacement.push_str(&format!(
                        "{dependency}=__toESM(require({}));",
                        quote(&request.value)
                    ));
                } else {
                    for specifier in &declaration.specifiers {
                        let local = specifier.local.name();
                        let expression = import_bindings
                            .get(local.as_ref())
                            .map_or(local.as_ref(), String::as_str);
                        preamble_exports
                            .push_str(&export_getter(&specifier.exported.name(), expression));
                    }
                }
                edits.push(Edit::replace(replacement_span, replacement));
            }
            Statement::ExportDefaultDeclaration(declaration) => {
                let (prefix, local, body_start) = match &declaration.declaration {
                    ExportDefaultDeclarationKind::FunctionDeclaration(function)
                        if function.id.is_some() =>
                    {
                        let name = function.id.as_ref().unwrap().name.to_string();
                        (String::new(), name, function.span.start)
                    }
                    ExportDefaultDeclarationKind::ClassDeclaration(class) if class.id.is_some() => {
                        let name = class.id.as_ref().unwrap().name.to_string();
                        (String::new(), name, class.span.start)
                    }
                    other => {
                        let local = format!("__diffpack_default_{default_index}");
                        default_index += 1;
                        (format!("const {local}="), local, other.span().start)
                    }
                };
                preamble_exports.push_str(&export_getter("default", &local));
                edits.push(Edit::replace(
                    Span::new(declaration.span.start, body_start),
                    prefix,
                ));
            }
            Statement::ExportAllDeclaration(declaration) => {
                let request = quote(&declaration.source.value);
                let replacement = declaration.exported.as_ref().map_or_else(
                    || format!("__reExport(exports,__toESM(require({request})));"),
                    |exported| {
                        export_getter(&exported.name(), &format!("__toESM(require({request}))"))
                    },
                );
                edits.push(Edit::replace(declaration.span, replacement));
            }
            _ => {}
        }
    }

    let body = apply_edits(source.to_string(), edits);
    (
        format!(
            "exports=module.exports=__esmNamespace();\nObject.defineProperty(exports,\"__esModule\",{{value:true}});\n{preamble_declarations}{preamble_exports}{body}\n__seal(exports);"
        ),
        diagnostics,
        true,
        dependencies,
    )
}

fn module_statement_span(statement: &Statement<'_>) -> Option<Span> {
    match statement {
        Statement::ImportDeclaration(declaration) => Some(declaration.span),
        Statement::ExportNamedDeclaration(declaration) => declaration
            .declaration
            .as_ref()
            .map_or(Some(declaration.span), |inner| {
                Some(Span::new(declaration.span.start, inner.span().start))
            }),
        Statement::ExportDefaultDeclaration(declaration) => Some(Span::new(
            declaration.span.start,
            declaration.declaration.span().start,
        )),
        Statement::ExportAllDeclaration(declaration) => Some(declaration.span),
        _ => None,
    }
}

fn contains(outer: Span, inner: Span) -> bool {
    outer.start <= inner.start && inner.end <= outer.end
}

fn local_exports_for_span(module: &ModuleRecord<'_>, span: Span) -> String {
    let mut code = String::new();
    for entry in module
        .local_export_entries
        .iter()
        .filter(|entry| entry.statement_span == span && !entry.is_type)
    {
        let exported = match &entry.export_name {
            ExportExportName::Name(name) => name.name,
            ExportExportName::Default(_) => "default".into(),
            ExportExportName::Null => continue,
        };
        let local = match &entry.local_name {
            ExportLocalName::Name(name) | ExportLocalName::Default(name) => name.name,
            ExportLocalName::Null => continue,
        };
        if local != "*default*" {
            code.push_str(&export_getter(&exported, &local));
        }
    }
    code
}

fn export_getter(exported: &str, expression: &str) -> String {
    format!(
        "__export(exports,{},()=>{});\n",
        quote(exported),
        expression
    )
}

#[derive(Debug)]
struct DynamicImportCollector {
    edits: Vec<Edit>,
}

impl<'a> Visit<'a> for DynamicImportCollector {
    fn visit_import_expression(&mut self, expression: &ImportExpression<'a>) {
        if let oxc_ast::ast::Expression::StringLiteral(literal) = &expression.source {
            self.edits.push(Edit::replace(
                expression.span,
                format!(
                    "Promise.resolve().then(()=>require({}))",
                    quote(&literal.value)
                ),
            ));
        }
    }
}

#[derive(Debug)]
struct Edit {
    start: usize,
    end: usize,
    replacement: String,
}

impl Edit {
    fn replace(span: Span, replacement: String) -> Self {
        Self {
            start: span.start as usize,
            end: span.end as usize,
            replacement,
        }
    }
}

fn apply_edits(mut source: String, mut edits: Vec<Edit>) -> String {
    edits.sort_by_key(|edit| std::cmp::Reverse(edit.start));
    let mut previous_start = source.len();
    for edit in edits {
        assert!(
            edit.end <= previous_start,
            "overlapping module-lowering edits"
        );
        source.replace_range(edit.start..edit.end, &edit.replacement);
        previous_start = edit.start;
    }
    source
}

fn quote(value: &str) -> String {
    serde_json::to_string(value).expect("serializing a JavaScript string cannot fail")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strips_typescript_and_lowers_modules() {
        let transformed = transform_module(
            Path::new("entry.ts"),
            r#"
                import value, { named as local } from "./dep.js";
                export const answer: number = local;
                export default function named() { return value + answer; }
            "#,
        );

        assert!(
            transformed.diagnostics.is_empty(),
            "{:?}",
            transformed.diagnostics
        );
        assert!(!transformed.code.contains(": number"));
        assert!(!transformed.code.contains("import value"));
        assert!(!transformed.code.contains("export const"));
        assert!(transformed.code.contains("require(\"./dep.js\")"));
        assert!(transformed.code.contains("__export(exports,\"answer\""));
        assert!(transformed.code.contains("__export(exports,\"default\""));
    }

    #[test]
    fn lowers_literal_dynamic_import_into_the_single_chunk_runtime() {
        let transformed = transform_module(
            Path::new("entry.js"),
            "export const load = () => import('./lazy.js');",
        );
        assert!(
            transformed
                .code
                .contains("Promise.resolve().then(()=>require(\"./lazy.js\"))")
        );
    }

    #[test]
    fn lowers_jsx_to_javascript() {
        let transformed = transform_module(
            Path::new("component.jsx"),
            "export const Component = ({ name }) => <div>Hello {name}</div>;",
        );
        assert!(
            transformed.diagnostics.is_empty(),
            "{:?}",
            transformed.diagnostics
        );
        assert!(!transformed.code.contains("<div>"));
        assert!(transformed.code.contains("require(\"react/jsx-runtime\")"));
    }
}
