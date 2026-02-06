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
    let mut errors = Vec::new();
    let mut items: HashMap<String, ItemKind> = HashMap::new();
    let mut uses: HashSet<String> = HashSet::new();
    let mut enum_variants: HashMap<String, HashSet<String>> = HashMap::new();

    for item in &module.items {
        match item {
            Item::Struct(s) => insert_item(&mut items, &mut errors, s.name.clone(), s.span, ItemKind::Struct),
            Item::Enum(e) => insert_item(&mut items, &mut errors, e.name.clone(), e.span, ItemKind::Enum),
            Item::Fn(f) => insert_item(&mut items, &mut errors, f.name.clone(), f.span, ItemKind::Fn),
            Item::ExternFn(f) => insert_item(&mut items, &mut errors, f.name.clone(), f.span, ItemKind::ExternFn),
            Item::Use(u) => {
                if let Some(last) = u.path.last() {
                    if uses.contains(last) {
                        errors.push(ResolveError {
                            message: format!("duplicate use of '{}'", last),
                            span: u.span,
                        });
                    }
                    uses.insert(last.clone());
                }
            }
        }
    }

    for item in &module.items {
        if let Item::Enum(e) = item {
            let mut variants = HashSet::new();
            for v in &e.variants {
                variants.insert(v.name.clone());
            }
            enum_variants.insert(e.name.clone(), variants);
        }
    }

    let builtins = builtin_names();

    for item in &module.items {
        match item {
            Item::Struct(s) => {
                for field in &s.fields {
                    resolve_type(&field.ty, &items, &builtins, &mut errors);
                }
            }
            Item::Enum(e) => {
                for variant in &e.variants {
                    match &variant.kind {
                        EnumVariantKind::Unit => {}
                        EnumVariantKind::Tuple(types) => {
                            for ty in types {
                                resolve_type(ty, &items, &builtins, &mut errors);
                            }
                        }
                        EnumVariantKind::Struct(fields) => {
                            for field in fields {
                                resolve_type(&field.ty, &items, &builtins, &mut errors);
                            }
                        }
                    }
                }
            }
            Item::Fn(f) => {
                for param in &f.params {
                    resolve_type(&param.ty, &items, &builtins, &mut errors);
                }
                resolve_type(&f.ret_type, &items, &builtins, &mut errors);
                let mut scope = Scope::new();
                for param in &f.params {
                    scope.insert(param.name.clone());
                }
                resolve_block(&f.body, &items, &uses, &builtins, &enum_variants, &mut errors, &mut scope);
            }
            Item::ExternFn(f) => {
                for param in &f.params {
                    resolve_type(&param.ty, &items, &builtins, &mut errors);
                }
                resolve_type(&f.ret_type, &items, &builtins, &mut errors);
            }
            Item::Use(_) => {}
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

fn resolve_type(ty: &Type, items: &HashMap<String, ItemKind>, builtins: &HashSet<String>, errors: &mut Vec<ResolveError>) {
    match ty {
        Type::Path(path) => {
            if let Some(last) = path.last() {
                if builtins.contains(last) {
                    return;
                }
                if !items.contains_key(last) {
                    errors.push(ResolveError {
                        message: format!("unknown type '{}'", last),
                        span: Span::new(0, 0),
                    });
                }
            }
        }
        Type::RawPointer(inner) => resolve_type(inner, items, builtins, errors),
        Type::Tuple(tys) => {
            for t in tys {
                resolve_type(t, items, builtins, errors);
            }
        }
    }
}

fn resolve_block(
    block: &Block,
    items: &HashMap<String, ItemKind>,
    uses: &HashSet<String>,
    builtins: &HashSet<String>,
    enum_variants: &HashMap<String, HashSet<String>>,
    errors: &mut Vec<ResolveError>,
    scope: &mut Scope,
) {
    scope.push();
    for stmt in &block.stmts {
        match stmt {
            Stmt::Expr(expr, _) => resolve_expr(expr, items, uses, builtins, enum_variants, errors, scope),
            Stmt::Return(expr, _) => {
                if let Some(expr) = expr {
                    resolve_expr(expr, items, uses, builtins, enum_variants, errors, scope)
                }
            }
        }
    }
    if let Some(tail) = &block.tail {
        resolve_expr(tail, items, uses, builtins, enum_variants, errors, scope);
    }
    scope.pop();
}

fn resolve_expr(
    expr: &Expr,
    items: &HashMap<String, ItemKind>,
    uses: &HashSet<String>,
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
            resolve_expr(value, items, uses, builtins, enum_variants, errors, scope);
            scope.insert(name.clone());
        }
        Expr::If {
            cond,
            then_branch,
            else_branch,
            ..
        } => {
            resolve_expr(cond, items, uses, builtins, enum_variants, errors, scope);
            resolve_block(then_branch, items, uses, builtins, enum_variants, errors, scope);
            if let Some(else_branch) = else_branch {
                resolve_block(else_branch, items, uses, builtins, enum_variants, errors, scope);
            }
        }
        Expr::While { cond, body, .. } => {
            resolve_expr(cond, items, uses, builtins, enum_variants, errors, scope);
            resolve_block(body, items, uses, builtins, enum_variants, errors, scope);
        }
        Expr::Match { scrutinee, arms, .. } => {
            resolve_expr(scrutinee, items, uses, builtins, enum_variants, errors, scope);
            for arm in arms {
                scope.push();
                resolve_pattern(&arm.pattern, items, uses, builtins, enum_variants, errors, scope);
                resolve_expr(&arm.body, items, uses, builtins, enum_variants, errors, scope);
                scope.pop();
            }
        }
        Expr::Assign { target, value, .. } => {
            resolve_expr(target, items, uses, builtins, enum_variants, errors, scope);
            resolve_expr(value, items, uses, builtins, enum_variants, errors, scope);
        }
        Expr::Binary { left, right, .. } => {
            resolve_expr(left, items, uses, builtins, enum_variants, errors, scope);
            resolve_expr(right, items, uses, builtins, enum_variants, errors, scope);
        }
        Expr::Unary { expr, .. } => resolve_expr(expr, items, uses, builtins, enum_variants, errors, scope),
        Expr::Call { callee, args, .. } => {
            resolve_expr(callee, items, uses, builtins, enum_variants, errors, scope);
            for arg in args {
                resolve_expr(arg, items, uses, builtins, enum_variants, errors, scope);
            }
        }
        Expr::Field { base, .. } => resolve_expr(base, items, uses, builtins, enum_variants, errors, scope),
        Expr::Path(path, span) => resolve_path(path, *span, items, uses, builtins, enum_variants, errors, scope, false),
        Expr::StructLit { path, fields, .. } => {
            if path.len() >= 2 {
                let enum_name = &path[0];
                let variant = &path[1];
                match items.get(enum_name) {
                    Some(ItemKind::Enum) => {
                        if let Some(vars) = enum_variants.get(enum_name) {
                            if !vars.contains(variant) {
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
            } else if let Some(last) = path.last() {
                match items.get(last) {
                    Some(ItemKind::Struct) => {}
                    _ => errors.push(ResolveError {
                        message: format!("'{}' is not a struct", last),
                        span: Span::new(0, 0),
                    }),
                }
            }
            for (_, expr) in fields {
                resolve_expr(expr, items, uses, builtins, enum_variants, errors, scope);
            }
        }
        Expr::Literal(_, _) => {}
        Expr::Block(block) => resolve_block(block, items, uses, builtins, enum_variants, errors, scope),
    }
}

fn resolve_pattern(
    pattern: &Pattern,
    items: &HashMap<String, ItemKind>,
    uses: &HashSet<String>,
    builtins: &HashSet<String>,
    enum_variants: &HashMap<String, HashSet<String>>,
    errors: &mut Vec<ResolveError>,
    scope: &mut Scope,
) {
    match pattern {
        Pattern::Wildcard(_) => {}
        Pattern::Path(path, span) => {
            resolve_path(path, *span, items, uses, builtins, enum_variants, errors, scope, true);
        }
        Pattern::Struct { path, fields, span } => {
            resolve_path(path, *span, items, uses, builtins, enum_variants, errors, scope, true);
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
    uses: &HashSet<String>,
    builtins: &HashSet<String>,
    enum_variants: &HashMap<String, HashSet<String>>,
    errors: &mut Vec<ResolveError>,
    scope: &Scope,
    in_pattern: bool,
) {
    if let Some(last) = path.last() {
        if path.len() >= 2 {
            let enum_name = &path[0];
            let variant = &path[1];
            if let Some(variants) = enum_variants.get(enum_name) {
                if variants.contains(variant) {
                    return;
                }
            }
        }
        if path.len() == 1 {
            if scope.contains(last) {
                return;
            }
            if items.contains_key(last) || uses.contains(last) || builtins.contains(last) {
                return;
            }
            let context = if in_pattern { "pattern" } else { "value" };
            errors.push(ResolveError {
                message: format!("unresolved {} '{}'", context, last),
                span,
            });
            return;
        }
        if items.contains_key(last) || uses.contains(last) {
            return;
        }
        errors.push(ResolveError {
            message: format!("unresolved path '{}'", last),
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
