use crate::ast::*;
use crate::token::Span;
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone)]
pub struct TypeError {
    pub message: String,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Ty {
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
    F32,
    F64,
    Bool,
    Unit,
    String,
    RawPtr(Box<Ty>),
    Struct(String),
    Enum(String),
    Tuple(Vec<Ty>),
}

#[derive(Debug, Clone)]
struct StructInfo {
    fields: Vec<(String, Ty)>,
}

#[derive(Debug, Clone)]
struct EnumInfo {
    variants: HashMap<String, VariantInfo>,
}

#[derive(Debug, Clone)]
struct FnSig {
    params: Vec<Ty>,
    ret: Ty,
    varargs: bool,
}

#[derive(Debug, Clone)]
enum VariantInfo {
    Unit,
    Tuple(Vec<Ty>),
    Struct(Vec<(String, Ty)>),
}

#[derive(Debug, Clone)]
struct TypeEnv {
    structs: HashMap<String, StructInfo>,
    enums: HashMap<String, EnumInfo>,
    fns: HashMap<String, FnSig>,
    extern_fns: HashMap<String, FnSig>,
}

pub fn typecheck_module(module: &Module) -> Result<(), Vec<TypeError>> {
    let mut errors = Vec::new();
    let env = build_env(module, &mut errors);
    if !errors.is_empty() {
        return Err(errors);
    }

    for item in &module.items {
        if let Item::Fn(f) = item {
            let sig = env.fns.get(&f.name).cloned();
            if let Some(sig) = sig {
                let ret_ty = sig.ret.clone();
                let mut checker = TypeChecker::new(&env, ret_ty.clone());
                for (param, ty) in f.params.iter().zip(sig.params.iter()) {
                    checker.locals_insert(param.name.clone(), ty.clone());
                }
                checker.check_block(&f.body, Some(&ret_ty));
                if let Some(errs) = checker.errors() {
                    errors.extend(errs);
                }
            }
        }
    }

    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

fn build_env(module: &Module, errors: &mut Vec<TypeError>) -> TypeEnv {
    let mut structs = HashMap::new();
    let mut enums = HashMap::new();
    let mut fns = HashMap::new();
    let mut extern_fns = HashMap::new();
    let mut struct_names = HashMap::new();
    let mut enum_names = HashMap::new();

    for item in &module.items {
        match item {
            Item::Struct(s) => {
                struct_names.insert(s.name.clone(), ());
            }
            Item::Enum(e) => {
                enum_names.insert(e.name.clone(), ());
            }
            _ => {}
        }
    }

    for item in &module.items {
        match item {
            Item::Struct(s) => {
                let mut fields = Vec::new();
                for field in &s.fields {
                    fields.push((field.name.clone(), lower_type(&field.ty, errors, &struct_names, &enum_names)));
                }
                structs.insert(s.name.clone(), StructInfo { fields });
            }
            Item::Enum(e) => {
                let mut variants = HashMap::new();
                for variant in &e.variants {
                    let info = match &variant.kind {
                        EnumVariantKind::Unit => VariantInfo::Unit,
                        EnumVariantKind::Tuple(types) => {
                            let mut payload = Vec::new();
                            for ty in types {
                                payload.push(lower_type(ty, errors, &struct_names, &enum_names));
                            }
                            VariantInfo::Tuple(payload)
                        }
                        EnumVariantKind::Struct(fields) => {
                            let mut out = Vec::new();
                            for field in fields {
                                let ty = lower_type(&field.ty, errors, &struct_names, &enum_names);
                                out.push((field.name.clone(), ty));
                            }
                            VariantInfo::Struct(out)
                        }
                    };
                    variants.insert(variant.name.clone(), info);
                }
                enums.insert(e.name.clone(), EnumInfo { variants });
            }
            Item::Fn(f) => {
                let params = f
                    .params
                    .iter()
                    .map(|p| lower_type(&p.ty, errors, &struct_names, &enum_names))
                    .collect();
                let ret = lower_type(&f.ret_type, errors, &struct_names, &enum_names);
                fns.insert(
                    f.name.clone(),
                    FnSig {
                        params,
                        ret,
                        varargs: false,
                    },
                );
            }
            Item::ExternFn(f) => {
                let params = f
                    .params
                    .iter()
                    .map(|p| lower_type(&p.ty, errors, &struct_names, &enum_names))
                    .collect();
                let ret = lower_type(&f.ret_type, errors, &struct_names, &enum_names);
                extern_fns.insert(
                    f.name.clone(),
                    FnSig {
                        params,
                        ret,
                        varargs: f.varargs,
                    },
                );
            }
            Item::Use(_) => {}
        }
    }

    TypeEnv {
        structs,
        enums,
        fns,
        extern_fns,
    }
}

fn lower_type(
    ty: &Type,
    errors: &mut Vec<TypeError>,
    struct_names: &HashMap<String, ()>,
    enum_names: &HashMap<String, ()>,
) -> Ty {
    match ty {
        Type::Path(path) => {
            if let Some(last) = path.last() {
                if let Some(b) = builtin_type(last) {
                    return b;
                }
                if enum_names.contains_key(last) {
                    return Ty::Enum(last.clone());
                }
                if struct_names.contains_key(last) {
                    return Ty::Struct(last.clone());
                }
                return Ty::Struct(last.clone());
            }
            errors.push(TypeError {
                message: "empty type path".to_string(),
                span: Span::new(0, 0),
            });
            Ty::Unit
        }
        Type::RawPointer(inner) => Ty::RawPtr(Box::new(lower_type(inner, errors, struct_names, enum_names))),
        Type::Tuple(tys) => Ty::Tuple(tys.iter().map(|t| lower_type(t, errors, struct_names, enum_names)).collect()),
    }
}

fn lower_type_env(ty: &Type, env: &TypeEnv, errors: &mut Vec<TypeError>) -> Ty {
    match ty {
        Type::Path(path) => {
            if let Some(last) = path.last() {
                if let Some(b) = builtin_type(last) {
                    return b;
                }
                if env.enums.contains_key(last) {
                    return Ty::Enum(last.clone());
                }
                if env.structs.contains_key(last) {
                    return Ty::Struct(last.clone());
                }
                return Ty::Struct(last.clone());
            }
            errors.push(TypeError {
                message: "empty type path".to_string(),
                span: Span::new(0, 0),
            });
            Ty::Unit
        }
        Type::RawPointer(inner) => Ty::RawPtr(Box::new(lower_type_env(inner, env, errors))),
        Type::Tuple(tys) => Ty::Tuple(tys.iter().map(|t| lower_type_env(t, env, errors)).collect()),
    }
}

fn builtin_type(name: &str) -> Option<Ty> {
    match name {
        "I8" => Some(Ty::I8),
        "I16" => Some(Ty::I16),
        "I32" => Some(Ty::I32),
        "I64" => Some(Ty::I64),
        "U8" => Some(Ty::U8),
        "U16" => Some(Ty::U16),
        "U32" => Some(Ty::U32),
        "U64" => Some(Ty::U64),
        "F32" => Some(Ty::F32),
        "F64" => Some(Ty::F64),
        "Bool" => Some(Ty::Bool),
        "Unit" => Some(Ty::Unit),
        "String" => Some(Ty::String),
        _ => None,
    }
}

struct TypeChecker<'a> {
    env: &'a TypeEnv,
    locals: Vec<HashMap<String, Ty>>,
    return_type: Ty,
    errors: Vec<TypeError>,
}

impl<'a> TypeChecker<'a> {
    fn new(env: &'a TypeEnv, return_type: Ty) -> Self {
        Self {
            env,
            locals: vec![HashMap::new()],
            return_type,
            errors: Vec::new(),
        }
    }

    fn errors(self) -> Option<Vec<TypeError>> {
        if self.errors.is_empty() {
            None
        } else {
            Some(self.errors)
        }
    }

    fn check_block(&mut self, block: &Block, expected: Option<&Ty>) -> Ty {
        self.locals_push();
        for stmt in &block.stmts {
            match stmt {
                Stmt::Expr(expr, _) => {
                    self.check_expr(expr, None);
                }
                Stmt::Return(expr, span) => {
                    let ret_ty = self.return_type.clone();
                    if let Some(expr) = expr {
                        let ty = self.check_expr(expr, Some(&ret_ty));
                        self.expect_type(&ret_ty, &ty, *span, "return type mismatch");
                    } else {
                        self.expect_type(&ret_ty, &Ty::Unit, *span, "return type mismatch");
                    }
                }
            }
        }
        let result = if let Some(tail) = &block.tail {
            self.check_expr(tail, expected)
        } else {
            Ty::Unit
        };
        if let Some(expected) = expected {
            self.expect_type(expected, &result, block.span, "block type mismatch");
        }
        self.locals_pop();
        result
    }

    fn check_expr(&mut self, expr: &Expr, expected: Option<&Ty>) -> Ty {
        match expr {
            Expr::Let {
                name,
                ty,
                value,
                ..
            } => {
                if let Some(annot) = ty {
                    let annot_ty = lower_type_env(annot, self.env, &mut self.errors);
                    let value_ty = self.check_expr(value, Some(&annot_ty));
                    self.expect_type(&annot_ty, &value_ty, expr_span(expr), "let type mismatch");
                    self.locals_insert(name.clone(), annot_ty);
                    Ty::Unit
                } else {
                    let value_ty = self.check_expr(value, None);
                    self.locals_insert(name.clone(), value_ty);
                    Ty::Unit
                }
            }
            Expr::If {
                cond,
                then_branch,
                else_branch,
                ..
            } => {
                let cond_ty = self.check_expr(cond, Some(&Ty::Bool));
                self.expect_type(&Ty::Bool, &cond_ty, expr_span(expr), "if condition must be Bool");
                let then_ty = self.check_block(then_branch, expected);
                if let Some(else_branch) = else_branch {
                    let else_ty = self.check_block(else_branch, expected);
                    self.expect_type(&then_ty, &else_ty, expr_span(expr), "if branch type mismatch");
                    then_ty
                } else {
                    Ty::Unit
                }
            }
            Expr::While { cond, body, .. } => {
                let cond_ty = self.check_expr(cond, Some(&Ty::Bool));
                self.expect_type(&Ty::Bool, &cond_ty, expr_span(expr), "while condition must be Bool");
                self.check_block(body, None);
                Ty::Unit
            }
            Expr::Match { scrutinee, arms, .. } => {
                let scrutinee_ty = self.check_expr(scrutinee, None);
                let mut result_ty: Option<Ty> = None;
                for arm in arms {
                    self.locals_push();
                    self.bind_pattern(&arm.pattern, &scrutinee_ty, arm.span);
                    let arm_ty = self.check_expr(&arm.body, expected);
                    self.locals_pop();
                    if let Some(prev) = &result_ty {
                        self.expect_type(prev, &arm_ty, arm.span, "match arm type mismatch");
                    } else {
                        result_ty = Some(arm_ty);
                    }
                }
                result_ty.unwrap_or(Ty::Unit)
            }
            Expr::Assign { target, value, .. } => {
                let target_ty = self.check_expr(target, None);
                let value_ty = self.check_expr(value, Some(&target_ty));
                self.expect_type(&target_ty, &value_ty, expr_span(expr), "assignment type mismatch");
                Ty::Unit
            }
            Expr::Binary { op, left, right, .. } => {
                let left_ty = self.check_expr(left, None);
                let right_ty = self.check_expr(right, Some(&left_ty));
                self.expect_type(&left_ty, &right_ty, expr_span(expr), "binary operand type mismatch");
                match op {
                    BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul | BinaryOp::Div => {
                        if is_numeric(&left_ty) {
                            left_ty
                        } else {
                            self.errors.push(TypeError {
                                message: "arithmetic operands must be numeric".to_string(),
                                span: expr_span(expr),
                            });
                            left_ty
                        }
                    }
                    BinaryOp::Rem => {
                        if is_int(&left_ty) {
                            left_ty
                        } else {
                            self.errors.push(TypeError {
                                message: "remainder operands must be integer".to_string(),
                                span: expr_span(expr),
                            });
                            left_ty
                        }
                    }
                    BinaryOp::Eq | BinaryOp::NotEq | BinaryOp::Lt | BinaryOp::LtEq | BinaryOp::Gt | BinaryOp::GtEq => Ty::Bool,
                    BinaryOp::AndAnd | BinaryOp::OrOr => {
                        self.expect_type(&Ty::Bool, &left_ty, expr_span(expr), "logical operands must be Bool");
                        Ty::Bool
                    }
                }
            }
            Expr::Unary { op, expr, .. } => {
                let ty = self.check_expr(expr, None);
                match op {
                    UnaryOp::Neg => {
                        if !is_numeric(&ty) {
                            self.errors.push(TypeError {
                                message: "negation requires numeric operand".to_string(),
                                span: expr_span(expr),
                            });
                        }
                        ty
                    }
                    UnaryOp::Not => {
                        self.expect_type(&Ty::Bool, &ty, expr_span(expr), "not requires Bool operand");
                        Ty::Bool
                    }
                }
            }
            Expr::Call { callee, args, .. } => {
                if let Expr::Path(path, span) = &**callee {
                    if let Some((ret, params, varargs)) = self.resolve_callable(path) {
                        self.check_call_args(args, &params, varargs, *span);
                        ret
                    } else {
                        self.errors.push(TypeError {
                            message: "unknown function".to_string(),
                            span: *span,
                        });
                        Ty::Unit
                    }
                } else {
                    self.errors.push(TypeError {
                        message: "call target must be a path in v0".to_string(),
                        span: expr_span(expr),
                    });
                    Ty::Unit
                }
            }
            Expr::Field { base, name, .. } => {
                let base_ty = self.check_expr(base, None);
                match base_ty {
                    Ty::Struct(s) => {
                        if let Some(info) = self.env.structs.get(&s) {
                            for (field, ty) in &info.fields {
                                if field == name {
                                    return ty.clone();
                                }
                            }
                            self.errors.push(TypeError {
                                message: format!("unknown field '{}'", name),
                                span: expr_span(expr),
                            });
                            Ty::Unit
                        } else {
                            self.errors.push(TypeError {
                                message: "unknown struct".to_string(),
                                span: expr_span(expr),
                            });
                            Ty::Unit
                        }
                    }
                    _ => {
                        self.errors.push(TypeError {
                            message: "field access requires struct".to_string(),
                            span: expr_span(expr),
                        });
                        Ty::Unit
                    }
                }
            }
            Expr::Path(path, span) => {
                if let Some(name) = path.last() {
                    if let Some(ty) = self.locals_get(name) {
                        return ty.clone();
                    }
                    if self.env.fns.contains_key(name) || self.env.extern_fns.contains_key(name) {
                        self.errors.push(TypeError {
                            message: "function used as value in v0".to_string(),
                            span: *span,
                        });
                        return Ty::Unit;
                    }
                }
                self.errors.push(TypeError {
                    message: "unknown value".to_string(),
                    span: *span,
                });
                Ty::Unit
            }
            Expr::StructLit { path, fields, .. } => {
                if path.len() >= 2 {
                    let enum_name = path[0].clone();
                    let variant = path[1].clone();
                    if let Some(info) = self.env.enums.get(&enum_name) {
                        if let Some(variant_info) = info.variants.get(&variant) {
                            if let VariantInfo::Struct(variant_fields) = variant_info {
                                for (field_name, field_ty) in variant_fields {
                                    let mut found = false;
                                    for (lit_name, lit_expr) in fields {
                                        if lit_name == field_name {
                                            let lit_ty = self.check_expr(lit_expr, Some(field_ty));
                                            self.expect_type(field_ty, &lit_ty, expr_span(expr), "variant field type mismatch");
                                            found = true;
                                            break;
                                        }
                                    }
                                    if !found {
                                        self.errors.push(TypeError {
                                            message: format!("missing field '{}'", field_name),
                                            span: expr_span(expr),
                                        });
                                    }
                                }
                                return Ty::Enum(enum_name);
                            }
                        }
                    }
                }

                let name = path.last().cloned().unwrap_or_default();
                if let Some(info) = self.env.structs.get(&name) {
                    for (field_name, field_ty) in &info.fields {
                        let mut found = false;
                        for (lit_name, lit_expr) in fields {
                            if lit_name == field_name {
                                let lit_ty = self.check_expr(lit_expr, Some(field_ty));
                                self.expect_type(field_ty, &lit_ty, expr_span(expr), "struct field type mismatch");
                                found = true;
                                break;
                            }
                        }
                        if !found {
                            self.errors.push(TypeError {
                                message: format!("missing field '{}'", field_name),
                                span: expr_span(expr),
                            });
                        }
                    }
                    Ty::Struct(name)
                } else {
                    self.errors.push(TypeError {
                        message: "unknown struct".to_string(),
                        span: expr_span(expr),
                    });
                    Ty::Unit
                }
            }
            Expr::Literal(lit, span) => match lit {
                Literal::Int(_) => self.number_literal(expected, true, *span),
                Literal::Float(_) => self.number_literal(expected, false, *span),
                Literal::Str(_) => {
                    if let Some(exp) = expected {
                        if exp != &Ty::String {
                            self.errors.push(TypeError {
                                message: "string literal type mismatch".to_string(),
                                span: *span,
                            });
                        }
                    }
                    Ty::String
                }
                Literal::Bool(_) => Ty::Bool,
            },
            Expr::Block(block) => self.check_block(block, expected),
        }
    }

    fn number_literal(&mut self, expected: Option<&Ty>, is_integer: bool, span: Span) -> Ty {
        if let Some(exp) = expected {
            if is_integer {
                if is_int(exp) {
                    return exp.clone();
                }
            } else if is_float(exp) {
                return exp.clone();
            }
            self.errors.push(TypeError {
                message: "numeric literal type mismatch".to_string(),
                span,
            });
            return exp.clone();
        }
        self.errors.push(TypeError {
            message: "numeric literal requires type context".to_string(),
            span,
        });
        if is_integer {
            Ty::I64
        } else {
            Ty::F64
        }
    }

    fn bind_pattern(&mut self, pattern: &Pattern, scrutinee: &Ty, span: Span) {
        match pattern {
            Pattern::Wildcard(_) => {}
            Pattern::Path(path, _) => {
                if path.len() >= 2 {
                    let enum_name = path[0].clone();
                    let variant_name = path[1].clone();
                    match scrutinee {
                        Ty::Enum(s) => {
                            if &enum_name != s {
                                self.errors.push(TypeError {
                                    message: "pattern enum mismatch".to_string(),
                                    span,
                                });
                            }
                        }
                        _ => {
                            self.errors.push(TypeError {
                                message: "match pattern requires enum scrutinee".to_string(),
                                span,
                            });
                        }
                    }
                    if let Some(info) = self.env.enums.get(&enum_name) {
                        if !info.variants.contains_key(&variant_name) {
                            self.errors.push(TypeError {
                                message: "unknown enum variant".to_string(),
                                span,
                            });
                        }
                    }
                }
            }
            Pattern::Struct { path, fields, .. } => {
                if path.len() < 2 {
                    self.errors.push(TypeError {
                        message: "struct pattern requires enum variant".to_string(),
                        span,
                    });
                    return;
                }
                let enum_name = path[0].clone();
                let variant_name = path[1].clone();
                match scrutinee {
                    Ty::Enum(s) => {
                        if &enum_name != s {
                            self.errors.push(TypeError {
                                message: "pattern enum mismatch".to_string(),
                                span,
                            });
                        }
                    }
                    _ => {
                        self.errors.push(TypeError {
                            message: "match pattern requires enum scrutinee".to_string(),
                            span,
                        });
                    }
                }
                let info = match self.env.enums.get(&enum_name) {
                    Some(i) => i,
                    None => return,
                };
                let variant = match info.variants.get(&variant_name) {
                    Some(v) => v,
                    None => {
                        self.errors.push(TypeError {
                            message: "unknown enum variant".to_string(),
                            span,
                        });
                        return;
                    }
                };
                let fields_def = match variant {
                    VariantInfo::Struct(f) => f,
                    _ => {
                        self.errors.push(TypeError {
                            message: "pattern requires struct variant".to_string(),
                            span,
                        });
                        return;
                    }
                };
                let mut seen = HashSet::new();
                for field in fields {
                    let mut found = None;
                    for (name, ty) in fields_def {
                        if name == &field.name {
                            found = Some(ty.clone());
                            break;
                        }
                    }
                    if let Some(ty) = found {
                        if let Some(bind) = &field.binding {
                            self.locals_insert(bind.clone(), ty);
                        }
                        seen.insert(field.name.clone());
                    } else {
                        self.errors.push(TypeError {
                            message: format!("unknown field '{}'", field.name),
                            span: field.span,
                        });
                    }
                }
                for (name, _) in fields_def {
                    if !seen.contains(name) {
                        self.errors.push(TypeError {
                            message: format!("missing field '{}'", name),
                            span,
                        });
                    }
                }
            }
        }
    }

    fn resolve_callable(&self, path: &[String]) -> Option<(Ty, Vec<Ty>, bool)> {
        if let Some(name) = path.last() {
            if let Some(sig) = self.env.fns.get(name) {
                return Some((sig.ret.clone(), sig.params.clone(), sig.varargs));
            }
            if let Some(sig) = self.env.extern_fns.get(name) {
                return Some((sig.ret.clone(), sig.params.clone(), sig.varargs));
            }
            if path.len() >= 2 {
                let enum_name = path[0].clone();
                let variant = path[1].clone();
                if let Some(info) = self.env.enums.get(&enum_name) {
                    if let Some(payload) = info.variants.get(&variant) {
                        if let VariantInfo::Tuple(args) = payload {
                            return Some((Ty::Enum(enum_name), args.clone(), false));
                        }
                    }
                }
            }
        }
        None
    }

    fn check_call_args(&mut self, args: &[Expr], params: &[Ty], varargs: bool, span: Span) {
        if !varargs && args.len() != params.len() {
            self.errors.push(TypeError {
                message: "argument count mismatch".to_string(),
                span,
            });
            return;
        }
        for (arg, param) in args.iter().zip(params.iter()) {
            let arg_ty = self.check_expr(arg, Some(param));
            self.expect_type(param, &arg_ty, span, "argument type mismatch");
        }
        if varargs && args.len() >= params.len() {
            for arg in args.iter().skip(params.len()) {
                self.check_expr(arg, None);
            }
        }
    }

    fn expect_type(&mut self, expected: &Ty, actual: &Ty, span: Span, msg: &str) {
        if expected != actual {
            self.errors.push(TypeError {
                message: msg.to_string(),
                span,
            });
        }
    }

    fn locals_push(&mut self) {
        self.locals.push(HashMap::new());
    }

    fn locals_pop(&mut self) {
        self.locals.pop();
    }

    fn locals_insert(&mut self, name: String, ty: Ty) {
        if let Some(frame) = self.locals.last_mut() {
            frame.insert(name, ty);
        }
    }

    fn locals_get(&self, name: &str) -> Option<&Ty> {
        for frame in self.locals.iter().rev() {
            if let Some(ty) = frame.get(name) {
                return Some(ty);
            }
        }
        None
    }
}

fn expr_span(expr: &Expr) -> Span {
    match expr {
        Expr::Let { span, .. }
        | Expr::If { span, .. }
        | Expr::While { span, .. }
        | Expr::Match { span, .. }
        | Expr::Assign { span, .. }
        | Expr::Binary { span, .. }
        | Expr::Unary { span, .. }
        | Expr::Call { span, .. }
        | Expr::Field { span, .. }
        | Expr::StructLit { span, .. }
        | Expr::Literal(_, span)
        | Expr::Path(_, span) => *span,
        Expr::Block(block) => block.span,
    }
}

fn is_int(ty: &Ty) -> bool {
    matches!(ty, Ty::I8 | Ty::I16 | Ty::I32 | Ty::I64 | Ty::U8 | Ty::U16 | Ty::U32 | Ty::U64)
}

fn is_float(ty: &Ty) -> bool {
    matches!(ty, Ty::F32 | Ty::F64)
}

fn is_numeric(ty: &Ty) -> bool {
    is_int(ty) || is_float(ty)
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
    fn typecheck_simple_function() {
        let src = r#"
            fn add(x: I64, y: I64) -> I64 {
                x + y
            }
        "#;
        let module = parse(src);
        let result = typecheck_module(&module);
        assert!(result.is_ok());
    }

    #[test]
    fn typecheck_if_mismatch() {
        let src = r#"
            fn bad(x: I64) -> I64 {
                if true { x } else { "no" }
            }
        "#;
        let module = parse(src);
        let err = typecheck_module(&module).unwrap_err();
        assert!(err.iter().any(|e| e.message.contains("if branch")));
    }

    #[test]
    fn typecheck_struct_literal_and_field() {
        let src = r#"
            struct User { id: I64, name: String }
            fn main() -> I64 {
                let u = User { id: 1, name: "a" };
                u.id
            }
        "#;
        let module = parse(src);
        let result = typecheck_module(&module);
        assert!(result.is_ok());
    }

    #[test]
    fn typecheck_match_enum() {
        let src = r#"
            enum Result { Ok(I64), Err(I64) }
            fn main() -> I64 {
                let r: Result = Result::Ok(1);
                match r { Result::Ok => 1, Result::Err => 2 }
            }
        "#;
        let module = parse(src);
        let result = typecheck_module(&module);
        assert!(result.is_ok());
    }

    #[test]
    fn typecheck_struct_variant_match_binding() {
        let src = r#"
            enum Opt { Some { value: I64 }, None {} }
            fn main() -> I64 {
                let o: Opt = Opt::Some { value: 7 };
                match o { Opt::Some { value } => value, Opt::None {} => 0 }
            }
        "#;
        let module = parse(src);
        let result = typecheck_module(&module);
        assert!(result.is_ok());
    }
}
