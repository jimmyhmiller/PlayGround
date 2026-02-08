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
    Struct(String, Vec<Ty>),
    Enum(String, Vec<Ty>),
    Tuple(Vec<Ty>),
    Param(std::string::String),
}

#[derive(Debug, Clone)]
struct StructInfo {
    type_params: Vec<std::string::String>,
    fields: Vec<(std::string::String, Ty)>,
}

#[derive(Debug, Clone)]
struct EnumInfo {
    type_params: Vec<std::string::String>,
    variants: HashMap<std::string::String, VariantInfo>,
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
    typecheck_modules(std::slice::from_ref(module))
}

pub fn typecheck_modules(modules: &[Module]) -> Result<(), Vec<TypeError>> {
    let mut errors = Vec::new();
    let env = build_env(modules, &mut errors);
    if !errors.is_empty() {
        return Err(errors);
    }

    for module in modules {
        let module_path = module.path.clone().unwrap_or_default();
        // Build per-module extern fns so each module sees its own declarations
        let mut mod_extern_fns = HashMap::new();
        let empty_tp: Vec<String> = Vec::new();
        for item in &module.items {
            if let Item::ExternFn(f) = item {
                let params: Vec<Ty> = f.params.iter()
                    .map(|p| lower_type_env(&p.ty, &env, &mut errors, &empty_tp))
                    .collect();
                let ret = lower_type_env(&f.ret_type, &env, &mut errors, &empty_tp);
                let name = full_item_name(&module_path, &f.name);
                mod_extern_fns.insert(name, FnSig { params, ret, varargs: f.varargs });
            }
        }
        let mod_env = TypeEnv {
            structs: env.structs.clone(),
            enums: env.enums.clone(),
            fns: env.fns.clone(),
            extern_fns: mod_extern_fns,
        };
        for item in &module.items {
            if let Item::Fn(f) = item {
                let full_name = full_item_name(&module_path, &f.name);
                let sig = mod_env.fns.get(&full_name).cloned();
                if let Some(sig) = sig {
                    let ret_ty = sig.ret.clone();
                    let mut checker = TypeChecker::new(&mod_env, ret_ty.clone());
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
    }

    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

fn build_env(modules: &[Module], errors: &mut Vec<TypeError>) -> TypeEnv {
    let mut structs = HashMap::new();
    let mut enums = HashMap::new();
    let mut fns = HashMap::new();
    let mut extern_fns = HashMap::new();
    let mut struct_names = HashMap::new();
    let mut enum_names = HashMap::new();

    for module in modules {
        let module_path = module.path.clone().unwrap_or_default();
        for item in &module.items {
            match item {
                Item::Struct(s) => {
                    struct_names.insert(full_item_name(&module_path, &s.name), ());
                }
                Item::Enum(e) => {
                    enum_names.insert(full_item_name(&module_path, &e.name), ());
                }
                _ => {}
            }
        }
    }

    for module in modules {
        let module_path = module.path.clone().unwrap_or_default();
        for item in &module.items {
            match item {
                Item::Struct(s) => {
                    let tp = &s.type_params;
                    let mut fields = Vec::new();
                    for field in &s.fields {
                        fields.push((field.name.clone(), lower_type(&field.ty, errors, &struct_names, &enum_names, tp)));
                    }
                    let name = full_item_name(&module_path, &s.name);
                    structs.insert(name, StructInfo { type_params: tp.clone(), fields });
                }
                Item::Enum(e) => {
                    let tp = &e.type_params;
                    let mut variants = HashMap::new();
                    for variant in &e.variants {
                        let info = match &variant.kind {
                            EnumVariantKind::Unit => VariantInfo::Unit,
                            EnumVariantKind::Tuple(types) => {
                                let mut payload = Vec::new();
                                for ty in types {
                                    payload.push(lower_type(ty, errors, &struct_names, &enum_names, tp));
                                }
                                VariantInfo::Tuple(payload)
                            }
                            EnumVariantKind::Struct(fields) => {
                                let mut out = Vec::new();
                                for field in fields {
                                    let ty = lower_type(&field.ty, errors, &struct_names, &enum_names, tp);
                                    out.push((field.name.clone(), ty));
                                }
                                VariantInfo::Struct(out)
                            }
                        };
                        variants.insert(variant.name.clone(), info);
                    }
                    let name = full_item_name(&module_path, &e.name);
                    enums.insert(name, EnumInfo { type_params: tp.clone(), variants });
                }
                Item::Fn(f) => {
                    let empty_tp: Vec<String> = Vec::new();
                    let params = f
                        .params
                        .iter()
                        .map(|p| lower_type(&p.ty, errors, &struct_names, &enum_names, &empty_tp))
                        .collect();
                    let ret = lower_type(&f.ret_type, errors, &struct_names, &enum_names, &empty_tp);
                    let name = full_item_name(&module_path, &f.name);
                    fns.insert(
                        name,
                        FnSig {
                            params,
                            ret,
                            varargs: false,
                        },
                    );
                }
                Item::ExternFn(f) => {
                    let empty_tp: Vec<String> = Vec::new();
                    let params = f
                        .params
                        .iter()
                        .map(|p| lower_type(&p.ty, errors, &struct_names, &enum_names, &empty_tp))
                        .collect();
                    let ret = lower_type(&f.ret_type, errors, &struct_names, &enum_names, &empty_tp);
                    let name = full_item_name(&module_path, &f.name);
                    extern_fns.insert(
                        name,
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
    type_param_env: &[String],
) -> Ty {
    match ty {
        Type::Path(path, type_args) => {
            if path.len() == 1 {
                let name = &path[0];
                if let Some(b) = builtin_type(name) {
                    return b;
                }
                if type_param_env.contains(name) {
                    return Ty::Param(name.clone());
                }
            }
            if let Some(last) = path.last() {
                if path.len() == 1 {
                    if let Some(b) = builtin_type(last) {
                        return b;
                    }
                }
                let key = path_to_string(path);
                let lowered_args: Vec<Ty> = type_args
                    .iter()
                    .map(|a| lower_type(a, errors, struct_names, enum_names, type_param_env))
                    .collect();
                if enum_names.contains_key(&key) {
                    return Ty::Enum(key, lowered_args);
                }
                if struct_names.contains_key(&key) {
                    return Ty::Struct(key, lowered_args);
                }
                return Ty::Struct(key, lowered_args);
            }
            errors.push(TypeError {
                message: "empty type path".to_string(),
                span: Span::new(0, 0),
            });
            Ty::Unit
        }
        Type::RawPointer(inner) => Ty::RawPtr(Box::new(lower_type(inner, errors, struct_names, enum_names, type_param_env))),
        Type::Tuple(tys) => Ty::Tuple(tys.iter().map(|t| lower_type(t, errors, struct_names, enum_names, type_param_env)).collect()),
    }
}

fn lower_type_env(ty: &Type, env: &TypeEnv, errors: &mut Vec<TypeError>, type_param_env: &[String]) -> Ty {
    match ty {
        Type::Path(path, type_args) => {
            if path.len() == 1 {
                let name = &path[0];
                if let Some(b) = builtin_type(name) {
                    return b;
                }
                if type_param_env.contains(name) {
                    return Ty::Param(name.clone());
                }
            }
            if let Some(last) = path.last() {
                if path.len() == 1 {
                    if let Some(b) = builtin_type(last) {
                        return b;
                    }
                }
                let key = path_to_string(path);
                let lowered_args: Vec<Ty> = type_args
                    .iter()
                    .map(|a| lower_type_env(a, env, errors, type_param_env))
                    .collect();
                if env.enums.contains_key(&key) {
                    return Ty::Enum(key, lowered_args);
                }
                if env.structs.contains_key(&key) {
                    return Ty::Struct(key, lowered_args);
                }
                return Ty::Struct(key, lowered_args);
            }
            errors.push(TypeError {
                message: "empty type path".to_string(),
                span: Span::new(0, 0),
            });
            Ty::Unit
        }
        Type::RawPointer(inner) => Ty::RawPtr(Box::new(lower_type_env(inner, env, errors, type_param_env))),
        Type::Tuple(tys) => Ty::Tuple(tys.iter().map(|t| lower_type_env(t, env, errors, type_param_env)).collect()),
    }
}

fn substitute(ty: &Ty, map: &HashMap<String, Ty>) -> Ty {
    match ty {
        Ty::Param(name) => map.get(name).cloned().unwrap_or_else(|| ty.clone()),
        Ty::Struct(name, args) => Ty::Struct(name.clone(), args.iter().map(|a| substitute(a, map)).collect()),
        Ty::Enum(name, args) => Ty::Enum(name.clone(), args.iter().map(|a| substitute(a, map)).collect()),
        Ty::RawPtr(inner) => Ty::RawPtr(Box::new(substitute(inner, map))),
        Ty::Tuple(tys) => Ty::Tuple(tys.iter().map(|t| substitute(t, map)).collect()),
        _ => ty.clone(),
    }
}

fn build_subst_map(type_params: &[String], type_args: &[Ty]) -> HashMap<String, Ty> {
    let mut map = HashMap::new();
    for (param, arg) in type_params.iter().zip(type_args.iter()) {
        map.insert(param.clone(), arg.clone());
    }
    map
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
    loop_depth: usize,
}

impl<'a> TypeChecker<'a> {
    fn new(env: &'a TypeEnv, return_type: Ty) -> Self {
        Self {
            env,
            locals: vec![HashMap::new()],
            return_type,
            errors: Vec::new(),
            loop_depth: 0,
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
                    let annot_ty = lower_type_env(annot, self.env, &mut self.errors, &[]);
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
                self.loop_depth += 1;
                self.check_block(body, None);
                self.loop_depth -= 1;
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
                    Ty::Struct(s, type_args) => {
                        if let Some(info) = self.env.structs.get(&s) {
                            let subst = build_subst_map(&info.type_params, &type_args);
                            for (field, ty) in &info.fields {
                                if field == name {
                                    return substitute(ty, &subst);
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
                    Ty::Tuple(items) => {
                        if let Ok(idx) = name.parse::<usize>() {
                            if idx < items.len() {
                                return items[idx].clone();
                            }
                        }
                        self.errors.push(TypeError {
                            message: format!("unknown tuple field '{}'", name),
                            span: expr_span(expr),
                        });
                        Ty::Unit
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
                    let key = path_to_string(path);
                    if self.env.fns.contains_key(&key) || self.env.extern_fns.contains_key(&key) {
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
                    let (enum_name, variant) = enum_path_and_variant(path);
                    if let Some(info) = self.env.enums.get(&enum_name) {
                        if let Some(variant_info) = info.variants.get(&variant) {
                            if let VariantInfo::Struct(variant_fields) = variant_info {
                                // Build substitution from expected type or empty
                                let type_args = match expected {
                                    Some(Ty::Enum(_, args)) => args.clone(),
                                    _ => vec![],
                                };
                                let subst = build_subst_map(&info.type_params, &type_args);
                                for (field_name, field_ty) in variant_fields {
                                    let concrete_ty = substitute(field_ty, &subst);
                                    let mut found = false;
                                    for (lit_name, lit_expr) in fields {
                                        if lit_name == field_name {
                                            let lit_ty = self.check_expr(lit_expr, Some(&concrete_ty));
                                            self.expect_type(&concrete_ty, &lit_ty, expr_span(expr), "variant field type mismatch");
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
                                return Ty::Enum(enum_name, type_args);
                            }
                        }
                    }
                }

                let name = path_to_string(path);
                if let Some(info) = self.env.structs.get(&name) {
                    // Build substitution from expected type or empty
                    let type_args = match expected {
                        Some(Ty::Struct(_, args)) => args.clone(),
                        _ => vec![],
                    };
                    let subst = build_subst_map(&info.type_params, &type_args);
                    for (field_name, field_ty) in &info.fields {
                        let concrete_ty = substitute(field_ty, &subst);
                        let mut found = false;
                        for (lit_name, lit_expr) in fields {
                            if lit_name == field_name {
                                let lit_ty = self.check_expr(lit_expr, Some(&concrete_ty));
                                self.expect_type(&concrete_ty, &lit_ty, expr_span(expr), "struct field type mismatch");
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
                    Ty::Struct(name, type_args)
                } else {
                    self.errors.push(TypeError {
                        message: "unknown struct".to_string(),
                        span: expr_span(expr),
                    });
                    Ty::Unit
                }
            }
            Expr::Tuple { items, .. } => {
                let expected_items = match expected {
                    Some(Ty::Tuple(tys)) => Some(tys),
                    Some(_) => {
                        self.errors.push(TypeError {
                            message: "tuple type mismatch".to_string(),
                            span: expr_span(expr),
                        });
                        None
                    }
                    None => None,
                };
                if let Some(tys) = expected_items {
                    if tys.len() != items.len() {
                        self.errors.push(TypeError {
                            message: "tuple arity mismatch".to_string(),
                            span: expr_span(expr),
                        });
                    }
                }
                let mut out = Vec::new();
                for (i, item) in items.iter().enumerate() {
                    let exp = expected_items.and_then(|tys| tys.get(i));
                    let ty = self.check_expr(item, exp);
                    out.push(ty);
                }
                Ty::Tuple(out)
            }
            Expr::Literal(lit, span) => match lit {
                Literal::Int(_) => self.number_literal(expected, true, *span),
                Literal::Char(_) => self.number_literal(expected, true, *span),
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
                Literal::Unit => Ty::Unit,
            },
            Expr::Block(block) => self.check_block(block, expected),
            Expr::Break { span, .. } => {
                if self.loop_depth == 0 {
                    self.errors.push(TypeError {
                        message: "break outside of loop".to_string(),
                        span: *span,
                    });
                }
                Ty::Unit
            }
            Expr::Continue { span, .. } => {
                if self.loop_depth == 0 {
                    self.errors.push(TypeError {
                        message: "continue outside of loop".to_string(),
                        span: *span,
                    });
                }
                Ty::Unit
            }
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
        if is_integer { Ty::I64 } else { Ty::F64 }
    }

    fn bind_pattern(&mut self, pattern: &Pattern, scrutinee: &Ty, span: Span) {
        match pattern {
            Pattern::Wildcard(_) => {}
            Pattern::Path(path, _) => {
                if path.len() >= 2 {
                    let (enum_name, variant_name) = enum_path_and_variant(path);
                    match scrutinee {
                        Ty::Enum(s, _) => {
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
                let (enum_name, variant_name) = enum_path_and_variant(path);
                let scrutinee_type_args = match scrutinee {
                    Ty::Enum(s, args) => {
                        if &enum_name != s {
                            self.errors.push(TypeError {
                                message: "pattern enum mismatch".to_string(),
                                span,
                            });
                        }
                        args.clone()
                    }
                    _ => {
                        self.errors.push(TypeError {
                            message: "match pattern requires enum scrutinee".to_string(),
                            span,
                        });
                        vec![]
                    }
                };
                let info = match self.env.enums.get(&enum_name) {
                    Some(i) => i,
                    None => return,
                };
                let subst = build_subst_map(&info.type_params, &scrutinee_type_args);
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
                            found = Some(substitute(ty, &subst));
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
        let key = path_to_string(path);
        if let Some(sig) = self.env.fns.get(&key) {
            return Some((sig.ret.clone(), sig.params.clone(), sig.varargs));
        }
        if let Some(sig) = self.env.extern_fns.get(&key) {
            return Some((sig.ret.clone(), sig.params.clone(), sig.varargs));
        }
        if path.len() >= 2 {
            let (enum_name, variant) = enum_path_and_variant(path);
            if let Some(info) = self.env.enums.get(&enum_name) {
                if let Some(payload) = info.variants.get(&variant) {
                    if let VariantInfo::Tuple(args) = payload {
                        // For now return with empty type args; caller provides expected type
                        return Some((Ty::Enum(enum_name, vec![]), args.clone(), false));
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
        if expected != actual && !types_compatible(expected, actual) {
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
        | Expr::Tuple { span, .. }
        | Expr::Literal(_, span)
        | Expr::Path(_, span)
        | Expr::Break { span }
        | Expr::Continue { span } => *span,
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

/// RawPointer<I8> is compatible with any struct/enum type at runtime
/// since all are pointer-sized. This allows vec_push/vec_get to be used
/// polymorphically without type aliases.
fn is_raw_ptr_i8(ty: &Ty) -> bool {
    matches!(ty, Ty::RawPtr(inner) if **inner == Ty::I8)
}

fn types_compatible(a: &Ty, b: &Ty) -> bool {
    if is_raw_ptr_i8(a) && matches!(b, Ty::Struct(..) | Ty::Enum(..) | Ty::RawPtr(..)) {
        return true;
    }
    if is_raw_ptr_i8(b) && matches!(a, Ty::Struct(..) | Ty::Enum(..) | Ty::RawPtr(..)) {
        return true;
    }
    false
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

    #[test]
    fn typecheck_tuple_and_field() {
        let src = r#"
            fn main() -> I64 {
                let t: (I64, I64) = (1, 2);
                t.1
            }
        "#;
        let module = parse(src);
        let result = typecheck_module(&module);
        assert!(result.is_ok());
    }
}
