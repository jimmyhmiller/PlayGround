use crate::ast::*;
use crate::token::Span;
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone)]
pub struct QualifyError {
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

pub fn qualify_modules(modules: &mut [Module]) -> Result<(), Vec<QualifyError>> {
    let mut errors = Vec::new();
    let items = collect_global_items(modules, &mut errors);
    for module in modules.iter_mut() {
        let module_path = module.path.clone().unwrap_or_default();
        let use_map = build_use_map(&items, &module_path, module, &mut errors);
        let mut qualifier = Qualifier::new(module_path, use_map, &items, &mut errors);
        qualifier.qualify_module(module);
    }
    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

fn collect_global_items(modules: &[Module], errors: &mut Vec<QualifyError>) -> HashMap<String, ItemKind> {
    let mut items = HashMap::new();
    for module in modules {
        let module_path = module.path.clone().unwrap_or_default();
        for item in &module.items {
            let (name, kind, span) = match item {
                Item::Struct(s) => (&s.name, ItemKind::Struct, s.span),
                Item::Enum(e) => (&e.name, ItemKind::Enum, e.span),
                Item::Fn(f) => (&f.name, ItemKind::Fn, f.span),
                Item::ExternFn(f) => (&f.name, ItemKind::ExternFn, f.span),
                Item::Use(_) | Item::Link(_) => continue,
            };
            let full = full_item_name(&module_path, name);
            if items.contains_key(&full) {
                errors.push(QualifyError {
                    message: format!("duplicate item '{}'", full),
                    span,
                });
            } else {
                items.insert(full, kind);
            }
        }
    }
    items
}

fn build_use_map(
    items: &HashMap<String, ItemKind>,
    module_path: &[String],
    module: &Module,
    errors: &mut Vec<QualifyError>,
) -> HashMap<String, Vec<String>> {
    let mut map = HashMap::new();
    for item in &module.items {
        let use_decl = match item {
            Item::Use(u) => u,
            _ => continue,
        };
        let alias = use_decl.path.last().cloned().unwrap_or_default();
        if map.contains_key(&alias) {
            errors.push(QualifyError {
                message: format!("duplicate use alias '{}'", alias),
                span: use_decl.span,
            });
            continue;
        }
        let mut resolved: Option<Vec<String>> = None;
        let abs = path_to_string(&use_decl.path);
        if items.contains_key(&abs) {
            resolved = Some(use_decl.path.clone());
        } else if !module_path.is_empty() {
            let rel = module_path
                .iter()
                .cloned()
                .chain(use_decl.path.iter().cloned())
                .collect::<Vec<_>>();
            let rel_key = path_to_string(&rel);
            if items.contains_key(&rel_key) {
                resolved = Some(rel);
            }
        }
        match resolved {
            Some(path) => {
                map.insert(alias, path);
            }
            None => errors.push(QualifyError {
                message: format!("unresolved use path '{}'", abs),
                span: use_decl.span,
            }),
        }
    }
    map
}

struct Qualifier<'a> {
    module_path: Vec<String>,
    use_map: HashMap<String, Vec<String>>,
    items: &'a HashMap<String, ItemKind>,
    errors: &'a mut Vec<QualifyError>,
    locals: Vec<HashSet<String>>,
    type_params: HashSet<String>,
}

impl<'a> Qualifier<'a> {
    fn new(
        module_path: Vec<String>,
        use_map: HashMap<String, Vec<String>>,
        items: &'a HashMap<String, ItemKind>,
        errors: &'a mut Vec<QualifyError>,
    ) -> Self {
        Self {
            module_path,
            use_map,
            items,
            errors,
            locals: vec![HashSet::new()],
            type_params: HashSet::new(),
        }
    }

    fn qualify_module(&mut self, module: &mut Module) {
        for item in &mut module.items {
            match item {
                Item::Struct(s) => {
                    self.type_params = s.type_params.iter().cloned().collect();
                    for field in &mut s.fields {
                        self.qualify_type(&mut field.ty);
                    }
                    self.type_params.clear();
                }
                Item::Enum(e) => {
                    self.type_params = e.type_params.iter().cloned().collect();
                    for variant in &mut e.variants {
                        match &mut variant.kind {
                            EnumVariantKind::Unit => {}
                            EnumVariantKind::Tuple(types) => {
                                for ty in types {
                                    self.qualify_type(ty);
                                }
                            }
                            EnumVariantKind::Struct(fields) => {
                                for field in fields {
                                    self.qualify_type(&mut field.ty);
                                }
                            }
                        }
                    }
                    self.type_params.clear();
                }
                Item::Fn(f) => {
                    for param in &mut f.params {
                        self.qualify_type(&mut param.ty);
                    }
                    self.qualify_type(&mut f.ret_type);
                    self.locals_push();
                    for param in &f.params {
                        self.locals_insert(param.name.clone());
                    }
                    self.qualify_block(&mut f.body);
                    self.locals_pop();
                }
                Item::ExternFn(f) => {
                    for param in &mut f.params {
                        self.qualify_type(&mut param.ty);
                    }
                    self.qualify_type(&mut f.ret_type);
                }
                Item::Use(_) | Item::Link(_) => {}
            }
        }
    }

    fn qualify_block(&mut self, block: &mut Block) {
        self.locals_push();
        for stmt in &mut block.stmts {
            match stmt {
                Stmt::Expr(expr, _) => self.qualify_expr(expr),
                Stmt::Return(expr, _) => {
                    if let Some(expr) = expr {
                        self.qualify_expr(expr);
                    }
                }
            }
        }
        if let Some(tail) = &mut block.tail {
            self.qualify_expr(tail);
        }
        self.locals_pop();
    }

    fn qualify_expr(&mut self, expr: &mut Expr) {
        match expr {
            Expr::Let { name, ty, value, .. } => {
                if let Some(ty) = ty {
                    self.qualify_type(ty);
                }
                self.qualify_expr(value);
                self.locals_insert(name.clone());
            }
            Expr::If { cond, then_branch, else_branch, .. } => {
                self.qualify_expr(cond);
                self.qualify_block(then_branch);
                if let Some(else_branch) = else_branch {
                    self.qualify_block(else_branch);
                }
            }
            Expr::While { cond, body, .. } => {
                self.qualify_expr(cond);
                self.qualify_block(body);
            }
            Expr::Match { scrutinee, arms, .. } => {
                self.qualify_expr(scrutinee);
                for arm in arms {
                    self.locals_push();
                    self.qualify_pattern(&mut arm.pattern);
                    self.qualify_expr(&mut arm.body);
                    self.locals_pop();
                }
            }
            Expr::Assign { target, value, .. } => {
                self.qualify_expr(target);
                self.qualify_expr(value);
            }
            Expr::Binary { left, right, .. } => {
                self.qualify_expr(left);
                self.qualify_expr(right);
            }
            Expr::Unary { expr, .. } => self.qualify_expr(expr),
            Expr::Call { callee, args, .. } => {
                self.qualify_expr(callee);
                for arg in args {
                    self.qualify_expr(arg);
                }
            }
            Expr::Field { base, .. } => self.qualify_expr(base),
            Expr::Path(path, span) => self.qualify_value_path(path, *span),
            Expr::StructLit { path, fields, .. } => {
                self.qualify_value_path(path, Span::new(0, 0));
                for (_, expr) in fields {
                    self.qualify_expr(expr);
                }
            }
            Expr::Tuple { items, .. } => {
                for item in items {
                    self.qualify_expr(item);
                }
            }
            Expr::Literal(_, _) => {}
            Expr::Block(block) => self.qualify_block(block),
            Expr::Break { .. } | Expr::Continue { .. } => {}
        }
    }

    fn qualify_pattern(&mut self, pattern: &mut Pattern) {
        match pattern {
            Pattern::Wildcard(_) => {}
            Pattern::Path(path, span) => self.qualify_value_path(path, *span),
            Pattern::Struct { path, fields, .. } => {
                self.qualify_value_path(path, Span::new(0, 0));
                for field in fields {
                    if let Some(binding) = &field.binding {
                        self.locals_insert(binding.clone());
                    }
                }
            }
        }
    }

    fn qualify_type(&mut self, ty: &mut Type) {
        match ty {
            Type::Path(path, type_args) => {
                self.qualify_type_path(path);
                for arg in type_args {
                    self.qualify_type(arg);
                }
            }
            Type::RawPointer(inner) => self.qualify_type(inner),
            Type::Tuple(types) => {
                for t in types {
                    self.qualify_type(t);
                }
            }
        }
    }

    fn qualify_type_path(&mut self, path: &mut Vec<String>) {
        if path.len() == 1 {
            let name = &path[0];
            if is_builtin_type(name) {
                return;
            }
            if self.type_params.contains(name.as_str()) {
                return;
            }
        }
        self.qualify_path_common(path, true);
    }

    fn qualify_value_path(&mut self, path: &mut Vec<String>, span: Span) {
        if path.len() == 1 {
            let name = &path[0];
            if self.locals_contains(name) {
                return;
            }
        }
        self.qualify_path_common(path, true);
        if path.is_empty() {
            self.errors.push(QualifyError {
                message: "empty path".to_string(),
                span,
            });
        }
    }

    fn qualify_path_common(&mut self, path: &mut Vec<String>, allow_relative: bool) {
        if path.is_empty() {
            return;
        }
        if path.len() == 1 {
            let name = path[0].clone();
            if let Some(full) = self.use_map.get(&name) {
                *path = full.clone();
                return;
            }
            if allow_relative && !self.module_path.is_empty() {
                let full = self
                    .module_path
                    .iter()
                    .cloned()
                    .chain(std::iter::once(name.clone()))
                    .collect::<Vec<_>>();
                let key = path_to_string(&full);
                if self.items.contains_key(&key) {
                    *path = full;
                }
            }
            return;
        }
        let first = path[0].clone();
        if let Some(full) = self.use_map.get(&first) {
            let mut out = full.clone();
            out.extend_from_slice(&path[1..]);
            *path = out;
            return;
        }
        if allow_relative && !self.module_path.is_empty() {
            let mut candidate = self.module_path.clone();
            candidate.push(first.clone());
            let key = path_to_string(&candidate);
            if self.items.contains_key(&key) {
                let mut out = self.module_path.clone();
                out.extend_from_slice(path);
                *path = out;
            }
        }
    }

    fn locals_push(&mut self) {
        self.locals.push(HashSet::new());
    }

    fn locals_pop(&mut self) {
        self.locals.pop();
    }

    fn locals_insert(&mut self, name: String) {
        if let Some(frame) = self.locals.last_mut() {
            frame.insert(name);
        }
    }

    fn locals_contains(&self, name: &str) -> bool {
        for frame in self.locals.iter().rev() {
            if frame.contains(name) {
                return true;
            }
        }
        false
    }
}

fn is_builtin_type(name: &str) -> bool {
    matches!(
        name,
        "I8" | "I16" | "I32" | "I64" | "U8" | "U16" | "U32" | "U64" | "F32" | "F64" | "Bool" | "Unit" | "String"
    )
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

fn path_to_string(path: &[String]) -> String {
    path.join("::")
}
