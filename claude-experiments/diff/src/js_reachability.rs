//! Shared module-level reachability over the Oxc AST.
//!
//! Both native transforms that prune a module to a live subset use this: route
//! splitting ([`crate::route_split`], which keeps a route property and its
//! transitive module-level dependencies) and the client server-fn transform
//! ([`crate::server_fn`], which drops the server-only code left dead once a
//! handler is replaced by an RPC stub). One implementation so the two cannot
//! drift on what "reachable" means.

use oxc_ast::ast::{Expression, ImportDeclarationSpecifier, Program, Statement};
use oxc_ast_visit::{Visit, walk};
use oxc_ecmascript::BoundNames;
use oxc_semantic::Scoping;
use oxc_span::{GetSpan, Span};
use oxc_syntax::symbol::SymbolId;

/// A top-level statement paired with the symbols it binds at module scope.
pub struct Item {
    pub span: Span,
    pub symbols: Vec<SymbolId>,
}

/// Every top-level statement of the program as an [`Item`], in source order.
pub fn top_level_items(program: &Program<'_>) -> Vec<Item> {
    let mut items = Vec::new();
    for statement in &program.body {
        let span = statement.span();
        let mut symbols = Vec::new();
        collect_statement_bindings(statement, &mut symbols);
        items.push(Item { span, symbols });
    }
    items
}

/// The index of the item that binds `symbol`, if any.
pub fn item_of(items: &[Item], symbol: SymbolId) -> Option<usize> {
    items.iter().position(|item| item.symbols.contains(&symbol))
}

/// The local binding symbol of an import specifier.
pub fn local_symbol(specifier: &ImportDeclarationSpecifier<'_>) -> Option<SymbolId> {
    let local = match specifier {
        ImportDeclarationSpecifier::ImportSpecifier(specifier) => &specifier.local,
        ImportDeclarationSpecifier::ImportDefaultSpecifier(specifier) => &specifier.local,
        ImportDeclarationSpecifier::ImportNamespaceSpecifier(specifier) => &specifier.local,
    };
    local.symbol_id.get()
}

fn collect_statement_bindings(statement: &Statement<'_>, symbols: &mut Vec<SymbolId>) {
    let mut push = |symbol: Option<SymbolId>| {
        if let Some(symbol) = symbol {
            symbols.push(symbol);
        }
    };
    match statement {
        Statement::ImportDeclaration(import) => {
            if let Some(specifiers) = &import.specifiers {
                for specifier in specifiers {
                    push(local_symbol(specifier));
                }
            }
        }
        Statement::VariableDeclaration(declaration) => {
            declaration.bound_names(&mut |identifier| push(identifier.symbol_id.get()));
        }
        Statement::FunctionDeclaration(declaration) => {
            declaration.bound_names(&mut |identifier| push(identifier.symbol_id.get()));
        }
        Statement::ClassDeclaration(declaration) => {
            declaration.bound_names(&mut |identifier| push(identifier.symbol_id.get()));
        }
        Statement::ExportNamedDeclaration(export) => {
            if let Some(declaration) = &export.declaration {
                declaration.bound_names(&mut |identifier| push(identifier.symbol_id.get()));
            }
        }
        _ => {}
    }
}

/// All resolved identifier references in the program, each with its source span
/// and the module-scope symbol it resolves to.
pub fn collect_references(program: &Program<'_>, scoping: &Scoping) -> Vec<(Span, SymbolId)> {
    let mut collector = ReferenceCollector {
        scoping,
        references: Vec::new(),
    };
    collector.visit_program(program);
    collector.references
}

struct ReferenceCollector<'s> {
    scoping: &'s Scoping,
    references: Vec<(Span, SymbolId)>,
}

impl<'a> Visit<'a> for ReferenceCollector<'_> {
    fn visit_identifier_reference(&mut self, identifier: &oxc_ast::ast::IdentifierReference<'a>) {
        if let Some(reference_id) = identifier.reference_id.get()
            && let Some(symbol) = self.scoping.get_reference(reference_id).symbol_id()
        {
            self.references.push((identifier.span, symbol));
        }
        walk::walk_identifier_reference(self, identifier);
    }
}

/// Whether `span` is fully contained within `region`.
pub fn within(span: Span, region: Span) -> bool {
    region.start <= span.start && span.end <= region.end
}

/// Appends `symbol` to `list` unless already present.
pub fn push_unique(list: &mut Vec<SymbolId>, symbol: SymbolId) {
    if !list.contains(&symbol) {
        list.push(symbol);
    }
}

/// Spans of the top-level *declaration* statements that become dead once
/// `removed_regions` are excised from the module.
///
/// This is source-level tree shaking for the pruning transforms: after a region
/// (a route property value, or a server-fn handler argument) is replaced, the
/// imports and declarations that were reachable *only* through it are garbage.
/// Reachability is computed from the module's live roots — exported declarations
/// and any statement that is not a removable declaration (side-effect imports,
/// bare expression statements, re-exports) — following only references that lie
/// outside every removed region. A declaration is returned (dead) only when every
/// symbol it binds is unreachable that way.
///
/// Soundness: only side-effect-free declarations are ever returned. Function and
/// class declarations and imports are inherently safe to drop when unused; a
/// `const`/`let`/`var` is returned only when every initializer is a pure
/// expression (a literal, identifier, closure, or composition of those), so a
/// declaration whose initializer could have a side effect is conservatively kept.
/// A bare side-effect import (`import './x.css'`) binds no symbol, is never a
/// removable declaration, and is always retained.
pub fn dead_declaration_spans(
    program: &Program<'_>,
    scoping: &Scoping,
    removed_regions: &[Span],
) -> Vec<Span> {
    let items = top_level_items(program);
    let refs = collect_references(program, scoping);

    // Removable candidates: pure-enough declarations that bind at least one
    // symbol. Everything else is a root — retained, and a reachability seed.
    let removable: Vec<bool> = program
        .body
        .iter()
        .zip(&items)
        .map(|(statement, item)| !item.symbols.is_empty() && is_removable_declaration(statement))
        .collect();

    // A symbol is reachable if a reachable item references it outside the removed
    // regions. Seed from every root item, then close over reachable removables.
    let mut reachable: Vec<SymbolId> = Vec::new();
    for (index, item) in items.iter().enumerate() {
        if !removable[index] {
            seed_item_refs(item, &refs, &items, removed_regions, &mut reachable);
        }
    }
    let mut cursor = 0;
    while cursor < reachable.len() {
        let symbol = reachable[cursor];
        cursor += 1;
        if let Some(index) = item_of(&items, symbol) {
            seed_item_refs(&items[index], &refs, &items, removed_regions, &mut reachable);
        }
    }

    items
        .iter()
        .enumerate()
        .filter(|(index, item)| {
            removable[*index] && item.symbols.iter().all(|symbol| !reachable.contains(symbol))
        })
        .map(|(_, item)| item.span)
        .collect()
}

/// Marks every module-scope symbol referenced by `item` outside the removed
/// regions as reachable.
fn seed_item_refs(
    item: &Item,
    refs: &[(Span, SymbolId)],
    items: &[Item],
    removed_regions: &[Span],
    reachable: &mut Vec<SymbolId>,
) {
    for (span, symbol) in refs {
        if within(*span, item.span)
            && !removed_regions.iter().any(|region| within(*span, *region))
            && item_of(items, *symbol).is_some()
        {
            push_unique(reachable, *symbol);
        }
    }
}

/// Whether a statement is a declaration that may be dropped when unused: an
/// import, or a `var`/`let`/`const` all of whose initializers are pure, or a
/// function/class declaration (inherently side-effect-free). An exported
/// declaration is deliberately excluded (it is a root, part of the module's
/// public surface).
fn is_removable_declaration(statement: &Statement<'_>) -> bool {
    match statement {
        Statement::ImportDeclaration(_)
        | Statement::FunctionDeclaration(_)
        | Statement::ClassDeclaration(_) => true,
        Statement::VariableDeclaration(declaration) => declaration
            .declarations
            .iter()
            .all(|declarator| declarator.init.as_ref().is_none_or(is_pure_expression)),
        _ => false,
    }
}

/// A conservative side-effect-free check: literals, identifiers, closures, and
/// compositions of those. Anything that could invoke user code (a call, `new`,
/// `await`, assignment, tagged template) is treated as impure so its declaration
/// is kept. Unknown shapes default to impure.
fn is_pure_expression(expression: &Expression<'_>) -> bool {
    match expression {
        Expression::BooleanLiteral(_)
        | Expression::NullLiteral(_)
        | Expression::NumericLiteral(_)
        | Expression::BigIntLiteral(_)
        | Expression::RegExpLiteral(_)
        | Expression::StringLiteral(_)
        | Expression::Identifier(_)
        | Expression::ThisExpression(_)
        | Expression::ArrowFunctionExpression(_)
        | Expression::FunctionExpression(_)
        | Expression::ClassExpression(_) => true,
        Expression::ParenthesizedExpression(inner) => is_pure_expression(&inner.expression),
        Expression::TemplateLiteral(template) => template.expressions.iter().all(is_pure_expression),
        Expression::UnaryExpression(unary) => is_pure_expression(&unary.argument),
        Expression::BinaryExpression(binary) => {
            is_pure_expression(&binary.left) && is_pure_expression(&binary.right)
        }
        Expression::LogicalExpression(logical) => {
            is_pure_expression(&logical.left) && is_pure_expression(&logical.right)
        }
        Expression::ConditionalExpression(conditional) => {
            is_pure_expression(&conditional.test)
                && is_pure_expression(&conditional.consequent)
                && is_pure_expression(&conditional.alternate)
        }
        Expression::ArrayExpression(array) => array.elements.iter().all(|element| match element {
            oxc_ast::ast::ArrayExpressionElement::SpreadElement(_) => false,
            oxc_ast::ast::ArrayExpressionElement::Elision(_) => true,
            other => other
                .as_expression()
                .is_some_and(is_pure_expression),
        }),
        Expression::ObjectExpression(object) => {
            object.properties.iter().all(|property| match property {
                oxc_ast::ast::ObjectPropertyKind::ObjectProperty(property) => {
                    is_pure_expression(&property.value)
                }
                oxc_ast::ast::ObjectPropertyKind::SpreadProperty(_) => false,
            })
        }
        Expression::StaticMemberExpression(member) => is_pure_expression(&member.object),
        Expression::ComputedMemberExpression(member) => {
            is_pure_expression(&member.object) && is_pure_expression(&member.expression)
        }
        _ => false,
    }
}
