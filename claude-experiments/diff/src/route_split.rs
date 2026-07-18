//! Native TanStack Router code splitting (`?tsr-split`).
//!
//! TanStack Router's Vite plugin rewrites every `createFileRoute(...)(...)` route
//! file so that heavy route properties (the `component`, and later
//! `errorComponent`/`pendingComponent`/`notFoundComponent`) are moved into a
//! virtual "split" module and replaced with a lazy import. That lazy import is a
//! code-split point: it becomes its own chunk, so a route's component is only
//! fetched when the route is entered. Diffpack replaces that plugin natively, so
//! it must perform the same rewrite in Rust on the Oxc AST — no Vite, no Babel.
//!
//! This module implements the `component` property, which is the shape every
//! route in the reference app uses. Two operations, both source-to-source so the
//! result flows back through the normal [`crate::transform::transform_module`]
//! pipeline unchanged:
//!
//! * [`split_reference_route`] rewrites the *reference* file: the `component`
//!   value becomes `lazyRouteComponent($$splitComponentImporter, 'component')`,
//!   where `$$splitComponentImporter = () => import('<file>?tsr-split=component')`,
//!   and the now-orphaned component definition (plus imports only it used) is
//!   removed.
//! * [`build_split_module`] synthesizes the *virtual* module
//!   `<file>?tsr-split=component`: the extracted component definition (and its
//!   transitive module-level dependencies), re-exported as `component`.
//!
//! The two halves are derived from the same original source, so a leaf edit to a
//! route file re-derives both deterministically — the incremental graph and the
//! low-memory thesis are preserved because the whole thing is a per-module
//! transform gated on cheap string checks before any parse.

use std::path::Path;

use oxc_allocator::Allocator;
use oxc_ast::ast::{
    BindingPattern, Declaration, ExportDefaultDeclarationKind, Expression,
    ImportDeclarationSpecifier, ObjectExpression, ObjectPropertyKind, Program, Statement,
};
use oxc_ast_visit::{Visit, walk};
use oxc_ecmascript::BoundNames;
use oxc_parser::Parser;
use oxc_semantic::{Scoping, SemanticBuilder};
use oxc_span::{GetSpan, SourceType, Span};
use oxc_syntax::symbol::SymbolId;

/// The one split property implemented natively today. The others
/// (`errorComponent`, `pendingComponent`, `notFoundComponent`) follow the same
/// shape but are deliberately not wired up yet; asking for one is a hard error
/// rather than a silent wrong value.
pub const COMPONENT_TARGET: &str = "component";

/// The framework package the lazy loader is imported from (react is the only
/// framework the reference app uses).
const ROUTER_PACKAGE: &str = "@tanstack/react-router";
/// The local importer binding TanStack generates, matched verbatim so the output
/// shape is identical.
const IMPORTER_IDENT: &str = "$$splitComponentImporter";
/// The lazy wrapper imported from the router package for a component split.
const LAZY_IDENT: &str = "lazyRouteComponent";

/// Rewrites a route file's reference half if it splits a `component`.
///
/// Returns the rewritten source when `path` is a route file (under a `routes`
/// directory, calling `createFileRoute`) with a splittable `component` property;
/// returns `None` for every other module so the caller uses the source
/// unchanged. Cheap string checks gate the parse, so non-route modules pay
/// nothing.
pub fn split_reference_route(path: &Path, source: &str) -> Option<String> {
    if !is_route_source(path, source) {
        return None;
    }

    let allocator = Allocator::default();
    let source_type = SourceType::from_path(path)
        .unwrap_or_default()
        .with_module(true);
    let parsed = Parser::new(&allocator, source, source_type).parse();
    let program = &parsed.program;
    let scoping = SemanticBuilder::new().build(program).semantic.into_scoping();

    let options = route_options(program)?;
    let component = object_property_value(options, COMPONENT_TARGET)?;
    if !is_splittable_value(component) {
        return None;
    }
    let value_span = component.span();

    // Determine what else must be removed alongside the component value. If the
    // component is a bare identifier bound to a standalone top-level declaration
    // that nothing else references, that declaration is dead once the value is
    // replaced and is removed; if the identifier is exported, TanStack does not
    // split it at all (removing it would break the export), so neither do we.
    let refs = collect_references(program, &scoping);
    let mut removed_regions = vec![value_span];
    if let Expression::Identifier(identifier) = &component {
        let name = identifier.name.as_str();
        if exported_names(program).iter().any(|exported| exported == name) {
            return None;
        }
        if let Some(declaration_span) =
            removable_declaration_span(program, name, value_span, &refs)
        {
            removed_regions.push(declaration_span);
        }
    }

    // An imported binding that is used nowhere outside the removed regions is
    // now dead; drop the whole import so its source is not pulled back into the
    // reference chunk (this is what keeps an imported component out of the main
    // bundle). A partially-used import is left intact.
    let used = used_symbols(&refs, &removed_regions);
    let mut edits = Vec::new();
    for (import_span, specifier_symbols) in import_statements(program) {
        if !specifier_symbols.is_empty()
            && specifier_symbols.iter().all(|symbol| !used.contains(symbol))
            && !within(import_span, value_span)
        {
            edits.push(Edit::delete(import_span));
        }
    }

    let split_url = split_url(path);
    edits.push(Edit::replace(
        value_span,
        format!("{LAZY_IDENT}({IMPORTER_IDENT}, '{COMPONENT_TARGET}')"),
    ));
    for region in removed_regions.iter().skip(1) {
        edits.push(Edit::delete(*region));
    }

    let mut preamble = String::new();
    if !binds_identifier(program, LAZY_IDENT) {
        preamble.push_str(&format!("import {{ {LAZY_IDENT} }} from '{ROUTER_PACKAGE}';\n"));
    }
    preamble.push_str(&format!(
        "const {IMPORTER_IDENT} = () => import('{split_url}');\n"
    ));

    Some(apply_edits(source, preamble, edits))
}

/// Synthesizes the virtual split module `<file>?tsr-split=<target>`.
///
/// The module exports the extracted route property under its canonical name, so
/// a dynamic `import()` of the split id yields `{ component }`. Only `component`
/// is implemented; any other target is a hard error naming what is missing.
pub fn build_split_module(path: &Path, source: &str, target: &str) -> Result<String, String> {
    if target != COMPONENT_TARGET {
        return Err(format!(
            "route split target `{target}` is not implemented (only `component` is); \
             requested for {}",
            path.display()
        ));
    }
    if !is_route_source(path, source) {
        return Err(format!(
            "`?tsr-split={target}` requested for {}, which is not a splittable route file",
            path.display()
        ));
    }

    let allocator = Allocator::default();
    let source_type = SourceType::from_path(path)
        .unwrap_or_default()
        .with_module(true);
    let parsed = Parser::new(&allocator, source, source_type).parse();
    let program = &parsed.program;
    let scoping = SemanticBuilder::new().build(program).semantic.into_scoping();

    let options = route_options(program).ok_or_else(|| {
        format!(
            "`?tsr-split={target}` requested for {}, but no createFileRoute options were found",
            path.display()
        )
    })?;
    let options_span = options.span();
    let component = object_property_value(options, COMPONENT_TARGET).ok_or_else(|| {
        format!(
            "`?tsr-split={target}` requested for {}, but the route has no `component` property",
            path.display()
        )
    })?;
    if !is_splittable_value(component) {
        return Err(format!(
            "`?tsr-split={target}` requested for {}, but the `component` value is not splittable",
            path.display()
        ));
    }
    let value_span = component.span();

    let refs = collect_references(program, &scoping);
    let items = top_level_items(program);

    // Root the reachability walk at the module-level bindings the component value
    // directly references. For `component: Home` that is `Home`; for an inline
    // `component: () => <.../>` it is whatever module-level names the arrow closes
    // over. The route options object itself is excluded from the dependency graph
    // (below) so that sibling properties like `loader` do not drag their imports
    // into the component chunk.
    let mut reachable: Vec<SymbolId> = Vec::new();
    for (span, symbol) in &refs {
        if within(*span, value_span) && item_of(&items, *symbol).is_some() {
            push_unique(&mut reachable, *symbol);
        }
    }

    // Transitive closure over module-level declarations, ignoring references made
    // inside the route options object.
    let mut cursor = 0;
    while cursor < reachable.len() {
        let symbol = reachable[cursor];
        cursor += 1;
        let Some(index) = item_of(&items, symbol) else {
            continue;
        };
        let item_span = items[index].span;
        for (span, referenced) in &refs {
            if within(*span, item_span)
                && !within(*span, options_span)
                && item_of(&items, *referenced).is_some()
            {
                push_unique(&mut reachable, *referenced);
            }
        }
    }

    // Emit the retained declarations in source order. The route declaration, if
    // reachable (a component that calls `Route.useLoaderData()` keeps it), is
    // emitted with its options object emptied to `{}` so no other property
    // survives.
    let mut output = String::new();
    for item in &items {
        if !item.symbols.iter().any(|symbol| reachable.contains(symbol)) {
            continue;
        }
        if within(options_span, item.span) {
            output.push_str(&slice(source, item.span.start, options_span.start));
            output.push_str("{}");
            output.push_str(&slice(source, options_span.end, item.span.end));
        } else {
            output.push_str(&slice(source, item.span.start, item.span.end));
        }
        output.push('\n');
    }

    // Re-export the component under its canonical name. A bare identifier is
    // re-exported directly; an inline expression is bound to a local first.
    match &component {
        Expression::Identifier(identifier) => {
            output.push_str(&format!(
                "export {{ {name} as {COMPONENT_TARGET} }};\n",
                name = identifier.name.as_str()
            ));
        }
        _ => {
            let value = slice(source, value_span.start, value_span.end);
            output.push_str(&format!(
                "const SplitComponent = {value};\nexport {{ SplitComponent as {COMPONENT_TARGET} }};\n"
            ));
        }
    }

    Ok(output)
}

/// Whether `path` sits under a `routes` directory and its source calls
/// `createFileRoute` (the only splittable route factory; `createRootRoute` is
/// intentionally excluded, matching TanStack).
fn is_route_source(path: &Path, source: &str) -> bool {
    path.components()
        .any(|component| component.as_os_str() == "routes")
        && source.contains("createFileRoute")
}

/// The dynamic-import specifier for a route's component split module, relative to
/// the route file so it resolves back to the same file with the split query.
fn split_url(path: &Path) -> String {
    let file = path
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_default();
    format!("./{file}?tsr-split={COMPONENT_TARGET}")
}

/// Finds the options object of a `createFileRoute(...)(<options>)` (or
/// `createFileRoute(<options>)`) call anywhere in the program.
fn route_options<'a, 'b>(program: &'b Program<'a>) -> Option<&'b ObjectExpression<'a>> {
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
            if let Some(Expression::CallExpression(call)) = &declarator.init
                && let Some(options) = create_file_route_options(call)
            {
                return Some(options);
            }
        }
    }
    None
}

/// Extracts the options object from a route-factory call. Accepts both the
/// curried `createFileRoute('/path')({ ... })` form and a bare
/// `createFileRoute({ ... })`.
fn create_file_route_options<'a, 'b>(
    call: &'b oxc_ast::ast::CallExpression<'a>,
) -> Option<&'b ObjectExpression<'a>> {
    let is_factory = match &call.callee {
        Expression::Identifier(identifier) => identifier.name == "createFileRoute",
        Expression::CallExpression(inner) => matches!(
            &inner.callee,
            Expression::Identifier(identifier) if identifier.name == "createFileRoute"
        ),
        _ => false,
    };
    if !is_factory {
        return None;
    }
    match call.arguments.first().and_then(|argument| argument.as_expression()) {
        Some(Expression::ObjectExpression(object)) => Some(object.as_ref()),
        _ => None,
    }
}

/// The value expression of an object property with the given key name.
fn object_property_value<'a, 'b>(
    object: &'b ObjectExpression<'a>,
    key: &str,
) -> Option<&'b Expression<'a>> {
    object.properties.iter().find_map(|property| match property {
        ObjectPropertyKind::ObjectProperty(property)
            if property.key.static_name().as_deref() == Some(key) =>
        {
            Some(&property.value)
        }
        _ => None,
    })
}

/// Whether a `component` value is worth splitting. TanStack skips `false`,
/// `null`, and `undefined`; everything else (an identifier or an inline
/// function) splits.
fn is_splittable_value(value: &Expression<'_>) -> bool {
    !matches!(
        value,
        Expression::BooleanLiteral(_) | Expression::NullLiteral(_)
    ) && !is_undefined(value)
}

fn is_undefined(value: &Expression<'_>) -> bool {
    matches!(value, Expression::Identifier(identifier) if identifier.name == "undefined")
}

/// A top-level binding and the source span of the statement that introduces it.
struct Item {
    span: Span,
    symbols: Vec<SymbolId>,
}

/// Every module-level statement that binds names, with the symbols it binds.
fn top_level_items(program: &Program<'_>) -> Vec<Item> {
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
fn item_of(items: &[Item], symbol: SymbolId) -> Option<usize> {
    items.iter().position(|item| item.symbols.contains(&symbol))
}

/// Import statements paired with the symbols of their local bindings.
fn import_statements(program: &Program<'_>) -> Vec<(Span, Vec<SymbolId>)> {
    program
        .body
        .iter()
        .filter_map(|statement| {
            let Statement::ImportDeclaration(import) = statement else {
                return None;
            };
            let mut symbols = Vec::new();
            if let Some(specifiers) = &import.specifiers {
                for specifier in specifiers {
                    if let Some(symbol) = local_symbol(specifier) {
                        symbols.push(symbol);
                    }
                }
            }
            Some((import.span, symbols))
        })
        .collect()
}

fn local_symbol(specifier: &ImportDeclarationSpecifier<'_>) -> Option<SymbolId> {
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

/// The span of a standalone, single-binding top-level declaration for `name`
/// that is only referenced within `value_span` (and is therefore dead once the
/// component value is replaced). Multi-declarator statements are left in place
/// to avoid removing an unrelated sibling binding.
fn removable_declaration_span(
    program: &Program<'_>,
    name: &str,
    value_span: Span,
    refs: &[(Span, SymbolId)],
) -> Option<Span> {
    for statement in &program.body {
        let (span, symbol) = match statement {
            Statement::FunctionDeclaration(function) => {
                let identifier = function.id.as_ref()?;
                if identifier.name != name {
                    continue;
                }
                (statement.span(), identifier.symbol_id.get())
            }
            Statement::VariableDeclaration(declaration) if declaration.declarations.len() == 1 => {
                let declarator = &declaration.declarations[0];
                let BindingPattern::BindingIdentifier(binding) = &declarator.id else {
                    continue;
                };
                if binding.name != name {
                    continue;
                }
                (statement.span(), binding.symbol_id.get())
            }
            _ => continue,
        };
        let Some(symbol) = symbol else { continue };
        // Removable only if no reference to it survives outside the value span.
        let used_elsewhere = refs
            .iter()
            .any(|(reference_span, referenced)| *referenced == symbol && !within(*reference_span, value_span));
        if used_elsewhere {
            return None;
        }
        return Some(span);
    }
    None
}

/// All resolved identifier references in the program, each with its source span
/// and the symbol it resolves to.
fn collect_references(program: &Program<'_>, scoping: &Scoping) -> Vec<(Span, SymbolId)> {
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

/// The set of symbols with at least one reference outside every removed region.
fn used_symbols(refs: &[(Span, SymbolId)], removed_regions: &[Span]) -> Vec<SymbolId> {
    let mut used = Vec::new();
    for (span, symbol) in refs {
        if removed_regions.iter().any(|region| within(*span, *region)) {
            continue;
        }
        push_unique(&mut used, *symbol);
    }
    used
}

/// Names exported by the module (declaration exports, named specifier exports,
/// and the default export identifier).
fn exported_names(program: &Program<'_>) -> Vec<String> {
    let mut names = Vec::new();
    for statement in &program.body {
        match statement {
            Statement::ExportNamedDeclaration(export) => {
                if let Some(declaration) = &export.declaration {
                    declaration.bound_names(&mut |identifier| names.push(identifier.name.to_string()));
                }
                for specifier in &export.specifiers {
                    names.push(specifier.local.name().to_string());
                }
            }
            Statement::ExportDefaultDeclaration(export) => {
                if let ExportDefaultDeclarationKind::FunctionDeclaration(function) =
                    &export.declaration
                    && let Some(identifier) = &function.id
                {
                    names.push(identifier.name.to_string());
                }
            }
            _ => {}
        }
    }
    names
}

/// Whether `name` is bound at module scope (an import or a declaration), used to
/// avoid inserting a duplicate `lazyRouteComponent` import.
fn binds_identifier(program: &Program<'_>, name: &str) -> bool {
    for statement in &program.body {
        match statement {
            Statement::ImportDeclaration(import) => {
                if let Some(specifiers) = &import.specifiers {
                    for specifier in specifiers {
                        let local = match specifier {
                            ImportDeclarationSpecifier::ImportSpecifier(specifier) => {
                                specifier.local.name.as_str()
                            }
                            ImportDeclarationSpecifier::ImportDefaultSpecifier(specifier) => {
                                specifier.local.name.as_str()
                            }
                            ImportDeclarationSpecifier::ImportNamespaceSpecifier(specifier) => {
                                specifier.local.name.as_str()
                            }
                        };
                        if local == name {
                            return true;
                        }
                    }
                }
            }
            Statement::VariableDeclaration(declaration) => {
                let mut found = false;
                declaration.bound_names(&mut |identifier| found |= identifier.name == name);
                if found {
                    return true;
                }
            }
            Statement::FunctionDeclaration(function)
                if function.id.as_ref().is_some_and(|id| id.name == name) =>
            {
                return true;
            }
            _ => {}
        }
    }
    false
}

/// Whether `span` is fully contained within `region`.
fn within(span: Span, region: Span) -> bool {
    region.start <= span.start && span.end <= region.end
}

fn slice(source: &str, start: u32, end: u32) -> String {
    source[start as usize..end as usize].to_string()
}

fn push_unique(list: &mut Vec<SymbolId>, symbol: SymbolId) {
    if !list.contains(&symbol) {
        list.push(symbol);
    }
}

/// A single edit to the source: a byte range and its replacement text (empty for
/// a deletion).
struct Edit {
    span: Span,
    replacement: String,
}

impl Edit {
    fn replace(span: Span, replacement: String) -> Self {
        Self { span, replacement }
    }

    fn delete(span: Span) -> Self {
        Self {
            span,
            replacement: String::new(),
        }
    }
}

/// Applies non-overlapping edits to `source`, prepending `preamble`.
fn apply_edits(source: &str, preamble: String, mut edits: Vec<Edit>) -> String {
    edits.sort_by_key(|edit| edit.span.start);
    let mut output = preamble;
    let mut cursor = 0_usize;
    for edit in edits {
        let start = edit.span.start as usize;
        let end = edit.span.end as usize;
        if start < cursor {
            // Overlapping/nested edit; skip it to keep the output well-formed.
            continue;
        }
        output.push_str(&source[cursor..start]);
        output.push_str(&edit.replacement);
        cursor = end;
    }
    output.push_str(&source[cursor..]);
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    fn route(source: &str) -> Option<String> {
        split_reference_route(Path::new("/app/src/routes/index.tsx"), source)
    }

    #[test]
    fn splits_a_local_component_out_of_the_reference_file() {
        let source = "import { createFileRoute } from '@tanstack/react-router'\n\
             export const Route = createFileRoute('/')({\n  component: Home,\n})\n\
             function Home() {\n  return <div>hi</div>\n}\n";
        let rewritten = route(source).expect("route with a component splits");
        assert!(rewritten.contains("import { lazyRouteComponent } from '@tanstack/react-router';"));
        assert!(rewritten
            .contains("const $$splitComponentImporter = () => import('./index.tsx?tsr-split=component');"));
        assert!(rewritten.contains("lazyRouteComponent($$splitComponentImporter, 'component')"));
        // The component definition is gone from the reference file.
        assert!(!rewritten.contains("function Home()"), "{rewritten}");
    }

    #[test]
    fn virtual_module_exports_the_component() {
        let source = "import { createFileRoute } from '@tanstack/react-router'\n\
             export const Route = createFileRoute('/')({\n  component: Home,\n})\n\
             function Home() {\n  return <div>hi</div>\n}\n";
        let module = build_split_module(Path::new("/app/src/routes/index.tsx"), source, "component")
            .expect("component split module builds");
        assert!(module.contains("function Home()"), "{module}");
        assert!(module.contains("export { Home as component };"), "{module}");
        // The route declaration is not reachable from Home, so it is dropped.
        assert!(!module.contains("createFileRoute"), "{module}");
    }

    #[test]
    fn virtual_module_keeps_the_route_when_the_component_uses_it() {
        let source = "import { Link, createFileRoute } from '@tanstack/react-router'\n\
             import { fetchPosts } from '../utils/posts'\n\
             export const Route = createFileRoute('/posts')({\n  \
             loader: async () => fetchPosts(),\n  component: PostsComponent,\n})\n\
             function PostsComponent() {\n  const posts = Route.useLoaderData()\n  \
             return <Link>{posts}</Link>\n}\n";
        let module =
            build_split_module(Path::new("/app/src/routes/posts.tsx"), source, "component").unwrap();
        // Route is reachable (PostsComponent calls Route.useLoaderData) so it is
        // kept, but with emptied options: no loader, so fetchPosts is dropped.
        assert!(module.contains("export const Route = createFileRoute('/posts')({})"), "{module}");
        assert!(!module.contains("fetchPosts"), "{module}");
        assert!(module.contains("Link"), "{module}");
        assert!(module.contains("export { PostsComponent as component };"), "{module}");
    }

    #[test]
    fn reference_file_drops_imports_only_the_component_used() {
        let source = "import { Link, createFileRoute } from '@tanstack/react-router'\n\
             import { heavy } from './heavy'\n\
             export const Route = createFileRoute('/posts')({\n  component: PostsComponent,\n})\n\
             function PostsComponent() {\n  return <Link>{heavy}</Link>\n}\n";
        let rewritten =
            split_reference_route(Path::new("/app/src/routes/posts.tsx"), source).unwrap();
        // `heavy` was used only by the component, so its import is removed.
        assert!(!rewritten.contains("./heavy"), "{rewritten}");
        // createFileRoute is still used by the Route, so its import stays.
        assert!(rewritten.contains("createFileRoute"), "{rewritten}");
    }

    #[test]
    fn non_route_files_are_left_alone() {
        assert!(split_reference_route(Path::new("/app/src/utils/posts.ts"), "export const x = 1").is_none());
        assert!(route("export const x = 1").is_none());
    }
}
