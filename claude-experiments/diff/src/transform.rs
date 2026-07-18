use std::collections::HashMap;
use std::path::Path;

use oxc_allocator::Allocator;
use oxc_ast::{
    ast::{
        BindingPattern, Declaration, ExportDefaultDeclarationKind, Expression,
        ImportDeclarationSpecifier, Statement, VariableDeclarationKind,
    },
    builder::{AstBuilder, NONE},
};
use oxc_ast_visit::{VisitMut, walk_mut};
use oxc_codegen::{Codegen, Context, Gen};
use oxc_ecmascript::BoundNames;
use oxc_parser::Parser;
use oxc_semantic::{Scoping, SemanticBuilder};
use oxc_span::{SPAN, SourceType};
use oxc_syntax::{operator::BinaryOperator, symbol::SymbolId};
use oxc_transformer::{TransformOptions, Transformer};

use crate::frontend_profile::{self, Phase};
use crate::parser::{collect_dependencies, collect_dynamic_dependencies};

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct TransformResult {
    pub code: String,
    pub diagnostics: Vec<String>,
    pub is_esm: bool,
    pub dependencies: Vec<String>,
    pub dependency_demands: Vec<DependencyDemand>,
    pub flat_module: Option<FlatModule>,
}

#[derive(Debug, Clone, Default, Eq, PartialEq)]
pub struct DependencyDemand {
    pub specifier: String,
    pub all: bool,
    pub names: Vec<String>,
    pub dynamic: bool,
}

#[derive(Debug, Clone, Default, Eq, PartialEq)]
pub struct FlatModule {
    pub code: String,
    pub declarations: Vec<String>,
    pub exports: Vec<String>,
    pub has_direct_effects: bool,
    pub import_replacements: Vec<(String, String)>,
    pub foldable: Option<FoldableModule>,
}

#[derive(Debug, Clone, Default, Eq, PartialEq)]
pub struct FoldableModule {
    pub constants: Vec<(String, FoldExpression)>,
    pub console_logs: Vec<FoldExpression>,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub enum FoldExpression {
    Number(u64),
    Reference(String),
    Add(Box<Self>, Box<Self>),
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
            dependency_demands: Vec::new(),
            flat_module: None,
        };
    }

    // A route file's heavy properties are split into virtual `?tsr-split`
    // modules and replaced with lazy imports before the module is lowered; this
    // is what turns each route's component into its own code-split chunk. Non-
    // route modules return `None` cheaply and take the source unchanged.
    let split = crate::route_split::split_reference_route(path, source);
    let source = split.as_deref().unwrap_or(source);

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

    frontend_profile::finish(Phase::Transform, transform_started);
    let lower_started = frontend_profile::start();
    let (code, is_esm, dependencies, dependency_demands, flat_module) =
        lower_module_ast(&allocator, &mut program, &transformed.scoping);
    frontend_profile::finish(Phase::Lower, lower_started);
    TransformResult {
        code,
        diagnostics,
        is_esm,
        dependencies,
        dependency_demands,
        flat_module,
    }
}

#[derive(Debug, Clone)]
enum ImportBinding {
    Namespace(String),
    Named { namespace: String, name: String },
}

fn lower_module_ast<'a>(
    allocator: &'a Allocator,
    program: &mut oxc_ast::ast::Program<'a>,
    scoping: &Scoping,
) -> (
    String,
    bool,
    Vec<String>,
    Vec<DependencyDemand>,
    Option<FlatModule>,
) {
    let dependencies = collect_dependencies(program);
    let dynamic_dependencies = collect_dynamic_dependencies(program);
    let mut dependency_demands = dependencies
        .iter()
        .map(|specifier| {
            (
                specifier.clone(),
                DependencyDemand {
                    specifier: specifier.clone(),
                    all: true,
                    names: Vec::new(),
                    dynamic: dynamic_dependencies.contains(specifier),
                },
            )
        })
        .collect::<HashMap<_, _>>();
    let is_esm = program.body.iter().any(|statement| {
        matches!(
            statement,
            Statement::ImportDeclaration(_)
                | Statement::ExportNamedDeclaration(_)
                | Statement::ExportDefaultDeclaration(_)
                | Statement::ExportAllDeclaration(_)
        )
    });

    let mut binding_expressions = HashMap::<SymbolId, ImportBinding>::new();
    let mut named_expressions = HashMap::<String, String>::new();
    let mut preamble_declarations = String::new();
    let mut preamble_exports = String::new();
    let mut import_index = 0_usize;
    let mut default_index = 0_usize;

    if is_esm {
        for statement in &program.body {
            match statement {
                Statement::ImportDeclaration(declaration) => {
                    let demand = dependency_demands
                        .entry(declaration.source.value.to_string())
                        .or_default();
                    demand.specifier = declaration.source.value.to_string();
                    demand.all = false;
                    demand.names.clear();
                    let Some(specifiers) = &declaration.specifiers else {
                        continue;
                    };
                    if specifiers.is_empty() {
                        continue;
                    }
                    let namespace = format!("__diffpack_import_{import_index}");
                    import_index += 1;
                    preamble_declarations.push_str(&format!("let {namespace};\n"));
                    for specifier in specifiers {
                        if !scoping
                            .get_resolved_reference_ids(specifier.local().symbol_id())
                            .is_empty()
                        {
                            match specifier {
                                ImportDeclarationSpecifier::ImportDefaultSpecifier(_) => {
                                    demand.names.push("default".into());
                                }
                                ImportDeclarationSpecifier::ImportNamespaceSpecifier(_) => {
                                    demand.all = true;
                                }
                                ImportDeclarationSpecifier::ImportSpecifier(specifier) => {
                                    demand.names.push(specifier.imported.name().to_string());
                                }
                            }
                        }
                        let (local, binding, expression) = match specifier {
                            ImportDeclarationSpecifier::ImportDefaultSpecifier(specifier) => {
                                let local = specifier.local.name.to_string();
                                (
                                    local,
                                    ImportBinding::Named {
                                        namespace: namespace.clone(),
                                        name: "default".into(),
                                    },
                                    format!("__import({namespace},\"default\")"),
                                )
                            }
                            ImportDeclarationSpecifier::ImportNamespaceSpecifier(specifier) => {
                                let local = specifier.local.name.to_string();
                                (
                                    local,
                                    ImportBinding::Namespace(namespace.clone()),
                                    namespace.clone(),
                                )
                            }
                            ImportDeclarationSpecifier::ImportSpecifier(specifier) => {
                                let local = specifier.local.name.to_string();
                                let imported = specifier.imported.name().to_string();
                                (
                                    local,
                                    ImportBinding::Named {
                                        namespace: namespace.clone(),
                                        name: imported.clone(),
                                    },
                                    format!("__import({namespace},{})", quote(&imported)),
                                )
                            }
                        };
                        binding_expressions.insert(specifier.local().symbol_id(), binding);
                        named_expressions.insert(local, expression);
                    }
                }
                Statement::ExportNamedDeclaration(declaration) => {
                    if let Some(source) = &declaration.source {
                        let demand = dependency_demands
                            .entry(source.value.to_string())
                            .or_default();
                        demand.specifier = source.value.to_string();
                        demand.all = false;
                        demand.names.extend(
                            declaration
                                .specifiers
                                .iter()
                                .map(|specifier| specifier.local.name().to_string()),
                        );
                    }
                    if let Some(inner) = &declaration.declaration {
                        inner.bound_names(&mut |identifier| {
                            preamble_exports
                                .push_str(&export_getter(&identifier.name, &identifier.name));
                        });
                    } else if declaration.source.is_some() {
                        let namespace = format!("__diffpack_reexport_{import_index}");
                        import_index += 1;
                        preamble_declarations.push_str(&format!("let {namespace};\n"));
                        for specifier in &declaration.specifiers {
                            preamble_exports.push_str(&export_getter(
                                &specifier.exported.name(),
                                &format!(
                                    "__import({namespace},{})",
                                    quote(&specifier.local.name())
                                ),
                            ));
                        }
                    } else {
                        for specifier in &declaration.specifiers {
                            let local = specifier.local.name();
                            let expression = named_expressions
                                .get(local.as_ref())
                                .map_or(local.as_ref(), String::as_str);
                            preamble_exports
                                .push_str(&export_getter(&specifier.exported.name(), expression));
                        }
                    }
                }
                Statement::ExportDefaultDeclaration(declaration) => {
                    let local = match &declaration.declaration {
                        ExportDefaultDeclarationKind::FunctionDeclaration(function)
                            if function.id.is_some() =>
                        {
                            function.id.as_ref().unwrap().name.to_string()
                        }
                        ExportDefaultDeclarationKind::ClassDeclaration(class)
                            if class.id.is_some() =>
                        {
                            class.id.as_ref().unwrap().name.to_string()
                        }
                        _ => {
                            let local = format!("__diffpack_default_{default_index}");
                            default_index += 1;
                            local
                        }
                    };
                    preamble_exports.push_str(&export_getter("default", &local));
                }
                Statement::ExportAllDeclaration(declaration) => {
                    let demand = dependency_demands
                        .entry(declaration.source.value.to_string())
                        .or_default();
                    demand.specifier = declaration.source.value.to_string();
                    demand.all = true;
                }
                _ => {}
            }
        }
    }

    let flat_module = build_flat_module(program, &dependencies, &dynamic_dependencies);

    AstModuleRewriter {
        builder: AstBuilder::new(allocator),
        scoping,
        bindings: &binding_expressions,
    }
    .visit_program(program);

    let mut codegen = Codegen::new();
    if is_esm {
        codegen.print_str(
            "exports=module.exports=__esmNamespace();\nObject.defineProperty(exports,\"__esModule\",{value:true});\n",
        );
        codegen.print_str(&preamble_declarations);
        codegen.print_str(&preamble_exports);
    }

    import_index = 0;
    default_index = 0;
    for statement in &program.body {
        match statement {
            Statement::ImportDeclaration(declaration) => {
                let request = quote(&declaration.source.value);
                let has_bindings = declaration
                    .specifiers
                    .as_ref()
                    .is_some_and(|specifiers| !specifiers.is_empty());
                if has_bindings {
                    let namespace = format!("__diffpack_import_{import_index}");
                    import_index += 1;
                    codegen.print_str(&format!(
                        "/*__diffpack_import:{request}__*/{namespace}=__toESM(require({request}));\n"
                    ));
                } else {
                    codegen.print_str(&format!(
                        "/*__diffpack_import:{request}__*/require({request});\n"
                    ));
                }
            }
            Statement::ExportNamedDeclaration(declaration) => {
                if let Some(inner) = &declaration.declaration {
                    let mut names = Vec::new();
                    inner.bound_names(&mut |identifier| names.push(identifier.name.to_string()));
                    let removable = declaration_is_obviously_pure(inner)
                        && declaration_bindings_are_locally_unused(inner, scoping);
                    if removable && !names.is_empty() {
                        codegen.print_str(&format!("/*__diffpack_decl:{}__*/\n", names.join(",")));
                    }
                    print_declaration(&mut codegen, inner);
                    if removable && !names.is_empty() {
                        codegen.print_str("/*__diffpack_decl_end__*/\n");
                    }
                } else if let Some(request) = &declaration.source {
                    let namespace = format!("__diffpack_reexport_{import_index}");
                    import_index += 1;
                    codegen.print_str(&format!(
                        "/*__diffpack_import:{request}__*/{namespace}=__toESM(require({request}));\n",
                        request = quote(&request.value)
                    ));
                }
            }
            Statement::ExportDefaultDeclaration(declaration) => {
                let is_named = matches!(
                    &declaration.declaration,
                    ExportDefaultDeclarationKind::FunctionDeclaration(function)
                        if function.id.is_some()
                ) || matches!(
                    &declaration.declaration,
                    ExportDefaultDeclarationKind::ClassDeclaration(class)
                        if class.id.is_some()
                );
                if !is_named {
                    codegen.print_str(&format!("const __diffpack_default_{default_index}="));
                    default_index += 1;
                }
                declaration
                    .declaration
                    .print(&mut codegen, Context::default());
                codegen.print_str("\n");
            }
            Statement::ExportAllDeclaration(declaration) => {
                let request = quote(&declaration.source.value);
                if let Some(exported) = &declaration.exported {
                    codegen.print_str(&export_getter(
                        &exported.name(),
                        &format!("__toESM(require({request}))"),
                    ));
                } else {
                    codegen.print_str(&format!(
                        "__reExport(exports,__toESM(require({request})));\n"
                    ));
                }
            }
            _ => {
                statement.print(&mut codegen, Context::default());
                codegen.print_str("\n");
            }
        }
    }
    if is_esm {
        codegen.print_str("__seal(exports);");
    }
    let mut dependency_demands = dependency_demands.into_values().collect::<Vec<_>>();
    for demand in &mut dependency_demands {
        demand.names.sort();
        demand.names.dedup();
    }
    dependency_demands.sort_by(|left, right| left.specifier.cmp(&right.specifier));
    let code = codegen.into_source_text();
    let flat_module = flat_module.map(|mut flat| {
        flat.code = derive_flat_code(&code, &flat.import_replacements);
        flat
    });
    (code, is_esm, dependencies, dependency_demands, flat_module)
}

fn print_declaration(codegen: &mut Codegen<'_>, declaration: &oxc_ast::ast::Declaration<'_>) {
    match declaration {
        oxc_ast::ast::Declaration::VariableDeclaration(declaration) => {
            declaration.print(codegen, Context::default());
        }
        oxc_ast::ast::Declaration::FunctionDeclaration(declaration) => {
            declaration.print(codegen, Context::default());
        }
        oxc_ast::ast::Declaration::ClassDeclaration(declaration) => {
            declaration.print(codegen, Context::default());
        }
        _ => {}
    }
    codegen.print_str("\n");
}

fn build_flat_module(
    program: &oxc_ast::ast::Program<'_>,
    dependencies: &[String],
    dynamic_dependencies: &std::collections::BTreeSet<String>,
) -> Option<FlatModule> {
    let foldable = build_foldable_module(program);
    let mut static_imports = Vec::new();
    let mut declarations = Vec::new();
    let mut exports = Vec::new();
    let mut has_direct_effects = false;
    let mut import_replacements = Vec::new();
    let mut binding_import_index = 0_usize;

    for statement in &program.body {
        match statement {
            Statement::ImportDeclaration(import) => {
                static_imports.push(import.source.value.to_string());
                if let Some(specifiers) = &import.specifiers {
                    let has_bindings = !specifiers.is_empty();
                    for specifier in specifiers {
                        match specifier {
                            ImportDeclarationSpecifier::ImportSpecifier(specifier)
                                if specifier.imported.name() == specifier.local.name =>
                            {
                                import_replacements.push((
                                    format!("__diffpack_import_{binding_import_index}"),
                                    specifier.imported.name().to_string(),
                                ));
                            }
                            _ => return None,
                        }
                    }
                    if has_bindings {
                        binding_import_index += 1;
                    }
                }
            }
            Statement::ExportNamedDeclaration(export) if export.source.is_none() => {
                if let Some(declaration) = &export.declaration {
                    let mut names = Vec::new();
                    declaration.bound_names(&mut |identifier| {
                        names.push(identifier.name.to_string());
                    });
                    declarations.extend(names.iter().cloned());
                    exports.extend(names.iter().cloned());
                    has_direct_effects |= !declaration_is_obviously_pure(declaration);
                } else {
                    for specifier in &export.specifiers {
                        if specifier.local.name() != specifier.exported.name() {
                            return None;
                        }
                        exports.push(specifier.exported.name().to_string());
                    }
                }
            }
            Statement::ExportNamedDeclaration(_)
            | Statement::ExportDefaultDeclaration(_)
            | Statement::ExportAllDeclaration(_) => return None,
            Statement::VariableDeclaration(declaration) => {
                declaration.bound_names(&mut |identifier| {
                    declarations.push(identifier.name.to_string());
                });
                has_direct_effects |= declaration.declarations.iter().any(|declarator| {
                    declarator
                        .init
                        .as_ref()
                        .is_some_and(|init| !expression_is_obviously_pure(init))
                });
            }
            Statement::FunctionDeclaration(declaration) => {
                declaration.bound_names(&mut |identifier| {
                    declarations.push(identifier.name.to_string());
                });
            }
            Statement::ClassDeclaration(declaration) => {
                declaration.bound_names(&mut |identifier| {
                    declarations.push(identifier.name.to_string());
                });
                has_direct_effects = true;
            }
            _ => {
                has_direct_effects = true;
            }
        }
    }
    if dependencies.iter().any(|dependency| {
        !static_imports.contains(dependency) && !dynamic_dependencies.contains(dependency)
    }) {
        return None;
    }
    declarations.sort();
    declarations.dedup();
    exports.sort();
    exports.dedup();
    Some(FlatModule {
        code: String::new(),
        declarations,
        exports,
        has_direct_effects,
        import_replacements,
        foldable,
    })
}

fn build_foldable_module(program: &oxc_ast::ast::Program<'_>) -> Option<FoldableModule> {
    let mut module = FoldableModule::default();
    for statement in &program.body {
        match statement {
            Statement::ImportDeclaration(_) => {}
            Statement::ExportNamedDeclaration(export) if export.source.is_none() => {
                let Declaration::VariableDeclaration(declaration) = export.declaration.as_ref()?
                else {
                    return None;
                };
                if declaration.kind != VariableDeclarationKind::Const {
                    return None;
                }
                for declarator in &declaration.declarations {
                    let BindingPattern::BindingIdentifier(identifier) = &declarator.id else {
                        return None;
                    };
                    module.constants.push((
                        identifier.name.to_string(),
                        fold_expression(declarator.init.as_ref()?)?,
                    ));
                }
            }
            Statement::ExpressionStatement(statement) => {
                let Expression::CallExpression(call) = &statement.expression else {
                    return None;
                };
                let Expression::StaticMemberExpression(member) = &call.callee else {
                    return None;
                };
                let Expression::Identifier(object) = &member.object else {
                    return None;
                };
                if object.name != "console"
                    || member.property.name != "log"
                    || call.arguments.len() != 1
                {
                    return None;
                }
                module
                    .console_logs
                    .push(fold_expression(call.arguments[0].as_expression()?)?);
            }
            Statement::EmptyStatement(_) => {}
            _ => return None,
        }
    }
    Some(module)
}

fn fold_expression(expression: &Expression<'_>) -> Option<FoldExpression> {
    match expression {
        Expression::NumericLiteral(number) => Some(FoldExpression::Number(number.value.to_bits())),
        Expression::Identifier(identifier) => {
            Some(FoldExpression::Reference(identifier.name.to_string()))
        }
        Expression::BinaryExpression(binary) if binary.operator == BinaryOperator::Addition => {
            Some(FoldExpression::Add(
                Box::new(fold_expression(&binary.left)?),
                Box::new(fold_expression(&binary.right)?),
            ))
        }
        Expression::ParenthesizedExpression(parenthesized) => {
            fold_expression(&parenthesized.expression)
        }
        _ => None,
    }
}

fn derive_flat_code(code: &str, replacements: &[(String, String)]) -> String {
    let mut flat = String::with_capacity(code.len());
    for line in code.lines() {
        if line.starts_with("exports=module.exports=__esmNamespace()")
            || line.starts_with("Object.defineProperty(exports,\"__esModule\"")
            || line.starts_with("let __diffpack_import_")
            || line.starts_with("/*__diffpack_export:")
            || line.starts_with("/*__diffpack_import:")
            || line == "__seal(exports);"
        {
            continue;
        }
        flat.push_str(line);
        flat.push('\n');
    }
    for (namespace, name) in replacements {
        flat = flat.replace(&format!("__import({namespace}, \"{name}\")"), name.as_str());
        flat = flat.replace(&format!("__import({namespace},\"{name}\")"), name.as_str());
    }
    flat
}

fn declaration_bindings_are_locally_unused(
    declaration: &oxc_ast::ast::Declaration<'_>,
    scoping: &Scoping,
) -> bool {
    let mut unused = true;
    declaration.bound_names(&mut |identifier| {
        unused &= scoping
            .get_resolved_reference_ids(identifier.symbol_id())
            .is_empty();
    });
    unused
}

fn declaration_is_obviously_pure(declaration: &oxc_ast::ast::Declaration<'_>) -> bool {
    match declaration {
        oxc_ast::ast::Declaration::FunctionDeclaration(_) => true,
        oxc_ast::ast::Declaration::VariableDeclaration(declaration) => {
            declaration.declarations.iter().all(|declarator| {
                declarator
                    .init
                    .as_ref()
                    .is_none_or(expression_is_obviously_pure)
            })
        }
        _ => false,
    }
}

fn expression_is_obviously_pure(expression: &oxc_ast::ast::Expression<'_>) -> bool {
    matches!(
        expression,
        oxc_ast::ast::Expression::BooleanLiteral(_)
            | oxc_ast::ast::Expression::NullLiteral(_)
            | oxc_ast::ast::Expression::NumericLiteral(_)
            | oxc_ast::ast::Expression::BigIntLiteral(_)
            | oxc_ast::ast::Expression::StringLiteral(_)
            | oxc_ast::ast::Expression::RegExpLiteral(_)
            | oxc_ast::ast::Expression::FunctionExpression(_)
            | oxc_ast::ast::Expression::ArrowFunctionExpression(_)
    )
}

struct AstModuleRewriter<'a, 's> {
    builder: AstBuilder<'a>,
    scoping: &'s Scoping,
    bindings: &'s HashMap<SymbolId, ImportBinding>,
}

#[allow(deprecated)]
impl<'a> AstModuleRewriter<'a, '_> {
    fn binding_expression(&self, binding: &ImportBinding) -> oxc_ast::ast::Expression<'a> {
        match binding {
            ImportBinding::Namespace(namespace) => self
                .builder
                .expression_identifier(SPAN, self.builder.ident(namespace)),
            ImportBinding::Named { namespace, name } => self.call(
                "__import",
                [
                    self.builder
                        .expression_identifier(SPAN, self.builder.ident(namespace)),
                    self.builder
                        .expression_string_literal(SPAN, self.builder.str(name), None),
                ],
            ),
        }
    }

    fn call<const N: usize>(
        &self,
        name: &str,
        arguments: [oxc_ast::ast::Expression<'a>; N],
    ) -> oxc_ast::ast::Expression<'a> {
        self.builder.expression_call(
            SPAN,
            self.builder
                .expression_identifier(SPAN, self.builder.ident(name)),
            NONE,
            self.builder
                .vec_from_iter(arguments.into_iter().map(oxc_ast::ast::Argument::from)),
            false,
        )
    }

    fn identifier_binding(
        &self,
        identifier: &oxc_ast::ast::IdentifierReference<'a>,
    ) -> Option<&ImportBinding> {
        let reference_id = identifier.reference_id.get()?;
        let symbol_id = self.scoping.get_reference(reference_id).symbol_id()?;
        self.bindings.get(&symbol_id)
    }
}

#[allow(deprecated)]
impl<'a> VisitMut<'a> for AstModuleRewriter<'a, '_> {
    fn visit_expression(&mut self, expression: &mut oxc_ast::ast::Expression<'a>) {
        if let oxc_ast::ast::Expression::Identifier(identifier) = expression
            && let Some(binding) = self.identifier_binding(identifier).cloned()
        {
            *expression = self.binding_expression(&binding);
            return;
        }
        if let oxc_ast::ast::Expression::ImportExpression(import) = expression
            && let oxc_ast::ast::Expression::StringLiteral(literal) = &import.source
        {
            *expression = self.call(
                "__dynamic",
                [
                    self.builder
                        .expression_identifier(SPAN, self.builder.ident("require")),
                    self.builder.expression_string_literal(
                        SPAN,
                        self.builder.str(&literal.value),
                        None,
                    ),
                ],
            );
            return;
        }
        walk_mut::walk_expression(self, expression);
    }

    fn visit_object_property(&mut self, property: &mut oxc_ast::ast::ObjectProperty<'a>) {
        if property.shorthand
            && let oxc_ast::ast::Expression::Identifier(identifier) = &property.value
            && self.identifier_binding(identifier).is_some()
        {
            property.shorthand = false;
        }
        walk_mut::walk_object_property(self, property);
    }
}

fn export_getter(exported: &str, expression: &str) -> String {
    format!(
        "/*__diffpack_export:{}__*/__export(exports,{},()=>{});\n",
        exported,
        quote(exported),
        expression
    )
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
                .contains("__dynamic(require, \"./lazy.js\")"),
            "{}",
            transformed.code
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

    #[test]
    fn records_imported_symbol_demand_without_scanning_generated_code() {
        let transformed = transform_module(
            Path::new("entry.js"),
            r#"
                import { used, unused } from "./values.js";
                import "./effects.js";
                export const answer = used;
            "#,
        );

        let values = transformed
            .dependency_demands
            .iter()
            .find(|demand| demand.specifier == "./values.js")
            .unwrap();
        assert!(!values.all);
        assert_eq!(values.names, ["used"]);
        assert!(!values.dynamic);

        let effects = transformed
            .dependency_demands
            .iter()
            .find(|demand| demand.specifier == "./effects.js")
            .unwrap();
        assert!(!effects.all);
        assert!(effects.names.is_empty());
        assert!(!effects.dynamic);
    }
}
