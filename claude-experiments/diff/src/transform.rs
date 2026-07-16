use std::path::Path;

use oxc_allocator::Allocator;
use oxc_ast::ast::{
    ExportDefaultDeclarationKind, ImportDeclarationSpecifier, ImportExpression, Statement,
};
use oxc_ast_visit::Visit;
use oxc_codegen::Codegen;
use oxc_parser::Parser;
use oxc_semantic::SemanticBuilder;
use oxc_span::{GetSpan, SourceType, Span};
use oxc_syntax::module_record::{ExportExportName, ExportLocalName, ModuleRecord};
use oxc_transformer::{TransformOptions, Transformer};

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct TransformResult {
    pub code: String,
    pub diagnostics: Vec<String>,
    pub is_esm: bool,
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
        };
    }

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
    let (code, lower_diagnostics, is_esm) = lower_module_syntax(path, &transformed_code);
    diagnostics.extend(lower_diagnostics);
    TransformResult {
        code,
        diagnostics,
        is_esm,
    }
}

fn lower_module_syntax(path: &Path, source: &str) -> (String, Vec<String>, bool) {
    let source = lower_dynamic_imports(path, source);
    let allocator = Allocator::default();
    let source_type = SourceType::from_path(path)
        .unwrap_or_default()
        .with_typescript(false)
        .with_jsx(false)
        .with_module(true);
    let parsed = Parser::new(&allocator, &source, source_type).parse();
    let diagnostics = parsed
        .diagnostics
        .into_iter()
        .map(|diagnostic| diagnostic.to_string())
        .collect::<Vec<_>>();
    let is_esm = parsed.module_record.has_module_syntax;
    if !is_esm {
        return (source, diagnostics, false);
    }

    let mut edits = Vec::new();
    let mut import_index = 0_usize;
    let mut default_index = 0_usize;
    for statement in &parsed.program.body {
        match statement {
            Statement::ImportDeclaration(declaration) => {
                let request = quote(&declaration.source.value);
                let replacement = match &declaration.specifiers {
                    None => format!("require({request});"),
                    Some(specifiers) if specifiers.is_empty() => format!("require({request});"),
                    Some(specifiers) => {
                        let dependency = format!("__diffpack_import_{import_index}");
                        import_index += 1;
                        let mut code = format!("const {dependency}=__toESM(require({request}));\n");
                        for specifier in specifiers {
                            match specifier {
                                ImportDeclarationSpecifier::ImportDefaultSpecifier(specifier) => {
                                    code.push_str(&format!(
                                        "const {}={dependency}.default;\n",
                                        specifier.local.name
                                    ));
                                }
                                ImportDeclarationSpecifier::ImportNamespaceSpecifier(specifier) => {
                                    code.push_str(&format!(
                                        "const {}={dependency};\n",
                                        specifier.local.name
                                    ));
                                }
                                ImportDeclarationSpecifier::ImportSpecifier(specifier) => {
                                    code.push_str(&format!(
                                        "const {}={dependency}[{}];\n",
                                        specifier.local.name,
                                        quote(&specifier.imported.name())
                                    ));
                                }
                            }
                        }
                        code
                    }
                };
                edits.push(Edit::replace(declaration.span, replacement));
            }
            Statement::ExportNamedDeclaration(declaration) => {
                let mut replacement = String::new();
                if let Some(inner) = &declaration.declaration {
                    replacement.push_str(slice(&source, inner.span()));
                    replacement.push('\n');
                    replacement.push_str(&local_exports_for_span(
                        &parsed.module_record,
                        declaration.span,
                    ));
                } else if let Some(request) = &declaration.source {
                    let dependency = format!("__diffpack_reexport_{import_index}");
                    import_index += 1;
                    replacement.push_str(&format!(
                        "const {dependency}=__toESM(require({}));\n",
                        quote(&request.value)
                    ));
                    for specifier in &declaration.specifiers {
                        replacement.push_str(&export_getter(
                            &specifier.exported.name(),
                            &format!("{dependency}[{}]", quote(&specifier.local.name())),
                        ));
                    }
                } else {
                    for specifier in &declaration.specifiers {
                        replacement.push_str(&export_getter(
                            &specifier.exported.name(),
                            specifier.local.name().as_ref(),
                        ));
                    }
                }
                edits.push(Edit::replace(declaration.span, replacement));
            }
            Statement::ExportDefaultDeclaration(declaration) => {
                let (body, local) = match &declaration.declaration {
                    ExportDefaultDeclarationKind::FunctionDeclaration(function)
                        if function.id.is_some() =>
                    {
                        let name = function.id.as_ref().unwrap().name.to_string();
                        (slice(&source, function.span).to_string(), name)
                    }
                    ExportDefaultDeclarationKind::ClassDeclaration(class) if class.id.is_some() => {
                        let name = class.id.as_ref().unwrap().name.to_string();
                        (slice(&source, class.span).to_string(), name)
                    }
                    other => {
                        let local = format!("__diffpack_default_{default_index}");
                        default_index += 1;
                        (
                            format!("const {local}=({});", slice(&source, other.span())),
                            local,
                        )
                    }
                };
                edits.push(Edit::replace(
                    declaration.span,
                    format!("{body}\n{}", export_getter("default", &local)),
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

    let body = apply_edits(source, edits);
    (
        format!("Object.defineProperty(exports,\"__esModule\",{{value:true}});\n{body}"),
        diagnostics,
        true,
    )
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

fn lower_dynamic_imports(path: &Path, source: &str) -> String {
    let allocator = Allocator::default();
    let source_type = SourceType::from_path(path)
        .unwrap_or_default()
        .with_typescript(false)
        .with_jsx(false)
        .with_module(true);
    let parsed = Parser::new(&allocator, source, source_type).parse();
    let mut collector = DynamicImportCollector { edits: Vec::new() };
    collector.visit_program(&parsed.program);
    apply_edits(source.to_string(), collector.edits)
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

fn slice(source: &str, span: Span) -> &str {
    &source[span.start as usize..span.end as usize]
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
