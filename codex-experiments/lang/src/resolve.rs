use crate::ast::*;
use crate::token::Span;
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone)]
pub struct ResolveError {
    pub message: String,
    pub span: Span,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ItemKind {
    Struct,
    Enum,
    Fn,
    ExternFn,
}

pub fn resolve_module(module: &Module) -> Result<(), Vec<ResolveError>> {
    resolve_modules(std::slice::from_ref(module))
}

pub fn resolve_modules(modules: &[Module]) -> Result<(), Vec<ResolveError>> {
    let mut errors = Vec::new();
    let mut items: HashMap<String, ItemKind> = HashMap::new();
    let mut enum_variants: HashMap<String, HashSet<String>> = HashMap::new();

    for module in modules {
        let module_path = module.path.clone().unwrap_or_default();
        for item in &module.items {
            match item {
                Item::Struct(s) => {
                    let name = full_item_name(&module_path, &s.name);
                    insert_item(&mut items, &mut errors, name, s.span, ItemKind::Struct);
                }
                Item::Enum(e) => {
                    let name = full_item_name(&module_path, &e.name);
                    insert_item(&mut items, &mut errors, name, e.span, ItemKind::Enum);
                }
                Item::Fn(f) => {
                    let name = full_item_name(&module_path, &f.name);
                    insert_item(&mut items, &mut errors, name, f.span, ItemKind::Fn);
                }
                Item::ExternFn(f) => {
                    let name = full_item_name(&module_path, &f.name);
                    insert_item(&mut items, &mut errors, name, f.span, ItemKind::ExternFn);
                }
                Item::Use(_) => {}
            }
        }
    }

    for module in modules {
        let module_path = module.path.clone().unwrap_or_default();
        for item in &module.items {
            if let Item::Enum(e) = item {
                let mut variants = HashSet::new();
                for v in &e.variants {
                    variants.insert(v.name.clone());
                }
                let key = full_item_name(&module_path, &e.name);
                enum_variants.insert(key, variants);
            }
        }
    }

    let builtins = builtin_names();

    for module in modules {
        for item in &module.items {
            match item {
                Item::Struct(s) => {
                    let tp: HashSet<String> = s.type_params.iter().cloned().collect();
                    for field in &s.fields {
                        resolve_type(&field.ty, &items, &builtins, &mut errors, &tp);
                    }
                }
                Item::Enum(e) => {
                    let tp: HashSet<String> = e.type_params.iter().cloned().collect();
                    for variant in &e.variants {
                        match &variant.kind {
                            EnumVariantKind::Unit => {}
                            EnumVariantKind::Tuple(types) => {
                                for ty in types {
                                    resolve_type(ty, &items, &builtins, &mut errors, &tp);
                                }
                            }
                            EnumVariantKind::Struct(fields) => {
                                for field in fields {
                                    resolve_type(&field.ty, &items, &builtins, &mut errors, &tp);
                                }
                            }
                        }
                    }
                }
                Item::Fn(f) => {
                    let empty_tp = HashSet::new();
                    for param in &f.params {
                        resolve_type(&param.ty, &items, &builtins, &mut errors, &empty_tp);
                    }
                    resolve_type(&f.ret_type, &items, &builtins, &mut errors, &empty_tp);
                    let mut scope = Scope::new();
                    for param in &f.params {
                        scope.insert(param.name.clone());
                    }
                    resolve_block(&f.body, &items, &builtins, &enum_variants, &mut errors, &mut scope);
                }
                Item::ExternFn(f) => {
                    let empty_tp = HashSet::new();
                    for param in &f.params {
                        resolve_type(&param.ty, &items, &builtins, &mut errors, &empty_tp);
                    }
                    resolve_type(&f.ret_type, &items, &builtins, &mut errors, &empty_tp);
                }
                Item::Use(_) => {}
            }
        }
    }

    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

fn insert_item(
    items: &mut HashMap<String, ItemKind>,
    errors: &mut Vec<ResolveError>,
    name: String,
    span: Span,
    kind: ItemKind,
) {
    if items.contains_key(&name) {
        errors.push(ResolveError {
            message: format!("duplicate item '{}'", name),
            span,
        });
    } else {
        items.insert(name, kind);
    }
}

fn resolve_type(ty: &Type, items: &HashMap<String, ItemKind>, builtins: &HashSet<String>, errors: &mut Vec<ResolveError>, type_params: &HashSet<String>) {
    match ty {
        Type::Path(path, type_args) => {
            if path.len() == 1 {
                let name = &path[0];
                if builtins.contains(name) || type_params.contains(name) {
                    for arg in type_args {
                        resolve_type(arg, items, builtins, errors, type_params);
                    }
                    return;
                }
            }
            let key = path_to_string(path);
            if !items.contains_key(&key) {
                errors.push(ResolveError {
                    message: format!("unknown type '{}'", key),
                    span: Span::new(0, 0),
                });
            }
            for arg in type_args {
                resolve_type(arg, items, builtins, errors, type_params);
            }
        }
        Type::RawPointer(inner) => resolve_type(inner, items, builtins, errors, type_params),
        Type::Tuple(tys) => {
            for t in tys {
                resolve_type(t, items, builtins, errors, type_params);
            }
        }
    }
}

fn resolve_block(
    block: &Block,
    items: &HashMap<String, ItemKind>,
    builtins: &HashSet<String>,
    enum_variants: &HashMap<String, HashSet<String>>,
    errors: &mut Vec<ResolveError>,
    scope: &mut Scope,
) {
    scope.push();
    for stmt in &block.stmts {
        match stmt {
            Stmt::Expr(expr, _) => resolve_expr(expr, items, builtins, enum_variants, errors, scope),
            Stmt::Return(expr, _) => {
                if let Some(expr) = expr {
                    resolve_expr(expr, items, builtins, enum_variants, errors, scope)
                }
            }
        }
    }
    if let Some(tail) = &block.tail {
        resolve_expr(tail, items, builtins, enum_variants, errors, scope);
    }
    scope.pop();
}

fn resolve_expr(
    expr: &Expr,
    items: &HashMap<String, ItemKind>,
    builtins: &HashSet<String>,
    enum_variants: &HashMap<String, HashSet<String>>,
    errors: &mut Vec<ResolveError>,
    scope: &mut Scope,
) {
    match expr {
        Expr::Let {
            name,
            value,
            ..
        } => {
            resolve_expr(value, items, builtins, enum_variants, errors, scope);
            scope.insert(name.clone());
        }
        Expr::If {
            cond,
            then_branch,
            else_branch,
            ..
        } => {
            resolve_expr(cond, items, builtins, enum_variants, errors, scope);
            resolve_block(then_branch, items, builtins, enum_variants, errors, scope);
            if let Some(else_branch) = else_branch {
                resolve_block(else_branch, items, builtins, enum_variants, errors, scope);
            }
        }
        Expr::While { cond, body, .. } => {
            resolve_expr(cond, items, builtins, enum_variants, errors, scope);
            resolve_block(body, items, builtins, enum_variants, errors, scope);
        }
        Expr::Match { scrutinee, arms, .. } => {
            resolve_expr(scrutinee, items, builtins, enum_variants, errors, scope);
            for arm in arms {
                scope.push();
                resolve_pattern(&arm.pattern, items, builtins, enum_variants, errors, scope);
                resolve_expr(&arm.body, items, builtins, enum_variants, errors, scope);
                scope.pop();
            }
        }
        Expr::Assign { target, value, .. } => {
            resolve_expr(target, items, builtins, enum_variants, errors, scope);
            resolve_expr(value, items, builtins, enum_variants, errors, scope);
        }
        Expr::Binary { left, right, .. } => {
            resolve_expr(left, items, builtins, enum_variants, errors, scope);
            resolve_expr(right, items, builtins, enum_variants, errors, scope);
        }
        Expr::Unary { expr, .. } => resolve_expr(expr, items, builtins, enum_variants, errors, scope),
        Expr::Call { callee, args, .. } => {
            resolve_expr(callee, items, builtins, enum_variants, errors, scope);
            for arg in args {
                resolve_expr(arg, items, builtins, enum_variants, errors, scope);
            }
        }
        Expr::Field { base, .. } => resolve_expr(base, items, builtins, enum_variants, errors, scope),
        Expr::Path(path, span) => resolve_path(path, *span, items, builtins, enum_variants, errors, scope, false),
        Expr::StructLit { path, fields, .. } => {
            if !path.is_empty() {
                let key = path_to_string(path);
                if matches!(items.get(&key), Some(ItemKind::Struct)) {
                    // Struct literal (possibly module-qualified).
                } else if path.len() >= 2 {
                    let (enum_name, variant) = enum_path_and_variant(path);
                    match items.get(&enum_name) {
                        Some(ItemKind::Enum) => {
                            if let Some(vars) = enum_variants.get(&enum_name) {
                                if !vars.contains(&variant) {
                                    errors.push(ResolveError {
                                        message: format!("unknown enum variant '{}'", variant),
                                        span: Span::new(0, 0),
                                    });
                                }
                            }
                        }
                        _ => errors.push(ResolveError {
                            message: format!("'{}' is not an enum", enum_name),
                            span: Span::new(0, 0),
                        }),
                    }
                } else {
                    errors.push(ResolveError {
                        message: format!("'{}' is not a struct", key),
                        span: Span::new(0, 0),
                    });
                }
            }
            for (_, expr) in fields {
                resolve_expr(expr, items, builtins, enum_variants, errors, scope);
            }
        }
        Expr::Tuple { items: elems, .. } => {
            for item in elems {
                resolve_expr(item, items, builtins, enum_variants, errors, scope);
            }
        }
        Expr::Literal(_, _) => {}
        Expr::Block(block) => resolve_block(block, items, builtins, enum_variants, errors, scope),
        Expr::Break { .. } | Expr::Continue { .. } => {}
    }
}

fn resolve_pattern(
    pattern: &Pattern,
    items: &HashMap<String, ItemKind>,
    builtins: &HashSet<String>,
    enum_variants: &HashMap<String, HashSet<String>>,
    errors: &mut Vec<ResolveError>,
    scope: &mut Scope,
) {
    match pattern {
        Pattern::Wildcard(_) => {}
        Pattern::Path(path, span) => {
            resolve_path(path, *span, items, builtins, enum_variants, errors, scope, true);
        }
        Pattern::Struct { path, fields, span } => {
            resolve_path(path, *span, items, builtins, enum_variants, errors, scope, true);
            for field in fields {
                if let Some(name) = &field.binding {
                    scope.insert(name.clone());
                }
            }
        }
    }
}

fn resolve_path(
    path: &[String],
    span: Span,
    items: &HashMap<String, ItemKind>,
    builtins: &HashSet<String>,
    enum_variants: &HashMap<String, HashSet<String>>,
    errors: &mut Vec<ResolveError>,
    scope: &Scope,
    in_pattern: bool,
) {
    if let Some(last) = path.last() {
        if path.len() >= 2 {
            let (enum_name, variant) = enum_path_and_variant(path);
            if let Some(variants) = enum_variants.get(&enum_name) {
                if variants.contains(&variant) {
                    return;
                }
            }
        }
        if path.len() == 1 {
            if scope.contains(last) {
                return;
            }
            if items.contains_key(last) || builtins.contains(last) {
                return;
            }
            let context = if in_pattern { "pattern" } else { "value" };
            errors.push(ResolveError {
                message: format!("unresolved {} '{}'", context, last),
                span,
            });
            return;
        }
        let key = path_to_string(path);
        if items.contains_key(&key) {
            return;
        }
        errors.push(ResolveError {
            message: format!("unresolved path '{}'", key),
            span,
        });
    }
}

fn builtin_names() -> HashSet<String> {
    let mut set = HashSet::new();
    for name in [
        "I8", "I16", "I32", "I64", "U8", "U16", "U32", "U64", "F32", "F64", "Bool", "Unit", "String",
    ] {
        set.insert(name.to_string());
    }
    set
}

fn path_to_string(path: &[String]) -> String {
    path.join("::")
}

fn full_item_name(module_path: &[String], name: &str) -> String {
    if module_path.is_empty() {
        name.to_string()
    } else {
        let mut parts = module_path.to_vec();
        parts.push(name.to_string());
        path_to_string(&parts)
    }
}

fn enum_path_and_variant(path: &[String]) -> (String, String) {
    let variant = path.last().cloned().unwrap_or_default();
    let enum_path = path[..path.len() - 1].to_vec();
    (path_to_string(&enum_path), variant)
}

struct Scope {
    frames: Vec<HashSet<String>>,
}

impl Scope {
    fn new() -> Self {
        Self { frames: vec![HashSet::new()] }
    }

    fn push(&mut self) {
        self.frames.push(HashSet::new());
    }

    fn pop(&mut self) {
        self.frames.pop();
    }

    fn insert(&mut self, name: String) {
        if let Some(frame) = self.frames.last_mut() {
            frame.insert(name);
        }
    }

    fn contains(&self, name: &str) -> bool {
        for frame in self.frames.iter().rev() {
            if frame.contains(name) {
                return true;
            }
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexer::Lexer;
    use crate::parser::Parser;

    fn parse(src: &str) -> Module {
        let tokens = Lexer::new(src).lex_all().unwrap();
        Parser::new(tokens).parse_module().unwrap()
    }

    #[test]
    fn resolve_reports_unresolved_value() {
        let src = r#"
            fn main() -> I64 {
                let x = 1;
                y
            }
        "#;
        let module = parse(src);
        let err = resolve_module(&module).unwrap_err();
        assert!(err.iter().any(|e| e.message.contains("unresolved")));
    }

    #[test]
    fn resolve_allows_locals_and_items() {
        let src = r#"
            struct Foo { x: I64 }
            fn main() -> I64 {
                let a: I64 = 1;
                let b = a;
                let c = Foo { x: b };
                b
            }
        "#;
        let module = parse(src);
        let result = resolve_module(&module);
        assert!(result.is_ok());
    }
}
