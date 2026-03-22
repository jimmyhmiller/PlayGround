use std::collections::HashMap;

use crate::ast;
use crate::codegen::{ScalarType, VecType};

// ─── Error types ───

#[derive(Debug)]
pub struct TypeError {
    pub span: ast::Span,
    pub message: String,
}

impl std::fmt::Display for TypeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "error at {}: {}", self.span, self.message)
    }
}

// ─── Checked type ───

#[derive(Debug, Clone, PartialEq)]
enum CheckedType {
    Vec(VecType),
    Struct {
        name: String,
        fields: Vec<(String, VecType)>,
    },
    Ptr(ScalarType),
    /// Unresolved literal — gets resolved when combined with a concrete Vec.
    Literal(LitKind),
    Void,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum LitKind {
    Int,
    Float,
    Bool,
    Char,
}

impl CheckedType {
    fn as_vec(&self) -> Option<VecType> {
        match self {
            CheckedType::Vec(vt) => Some(*vt),
            _ => None,
        }
    }

    fn describe(&self) -> String {
        match self {
            CheckedType::Vec(vt) => format!("{:?}[{}]", vt.scalar, vt.width),
            CheckedType::Struct { name, .. } => format!("struct {}", name),
            CheckedType::Ptr(s) => format!("ptr[{:?}]", s),
            CheckedType::Literal(k) => match k {
                LitKind::Int => "int literal".into(),
                LitKind::Float => "float literal".into(),
                LitKind::Bool => "bool literal".into(),
                LitKind::Char => "char literal".into(),
            },
            CheckedType::Void => "void".into(),
        }
    }
}

// ─── Helpers ───

fn is_scalar_type_name(name: &str) -> bool {
    matches!(
        name,
        "f32" | "f64" | "i8" | "i16" | "i32" | "i64" | "u8" | "u16" | "u32" | "u64" | "bool"
    )
}

fn is_ptr_type_name(name: &str) -> bool {
    name == "ptr"
}

fn resolve_type(
    ty: &ast::Type,
    comptime_env: &HashMap<String, u64>,
    native_width: u64,
) -> VecType {
    let scalar = ScalarType::from_name(&ty.name);
    let width = ty
        .width
        .as_ref()
        .expect("type must have a width")
        .eval(comptime_env, native_width);
    VecType { scalar, width }
}

fn is_literal(expr: &ast::Expr) -> bool {
    match expr {
        ast::Expr::FloatLit(_)
        | ast::Expr::IntLit(_)
        | ast::Expr::BoolLit(_)
        | ast::Expr::CharLit(_) => true,
        ast::Expr::UnaryOp {
            op: ast::UnaryOp::Not,
            operand,
        }
        | ast::Expr::UnaryOp {
            op: ast::UnaryOp::Neg,
            operand,
        } => is_literal(operand),
        _ => false,
    }
}

fn lit_kind(expr: &ast::Expr) -> Option<LitKind> {
    match expr {
        ast::Expr::IntLit(_) => Some(LitKind::Int),
        ast::Expr::FloatLit(_) => Some(LitKind::Float),
        ast::Expr::BoolLit(_) => Some(LitKind::Bool),
        ast::Expr::CharLit(_) => Some(LitKind::Char),
        ast::Expr::UnaryOp { operand, .. } => lit_kind(operand),
        _ => None,
    }
}

/// Check if a literal kind is compatible with a scalar type.
fn lit_compatible(kind: LitKind, scalar: ScalarType) -> bool {
    match kind {
        LitKind::Int | LitKind::Char => scalar.is_int() || scalar == ScalarType::Bool,
        LitKind::Float => scalar.is_float(),
        LitKind::Bool => scalar == ScalarType::Bool || scalar.is_int(),
    }
}

// ─── Type checker ───

/// Signature of a user-defined function, for inter-function call checking.
#[derive(Clone)]
struct FnSig {
    params: Vec<(String, CheckedType)>,
    ret_ty: Option<CheckedType>,
}

struct TypeChecker {
    errors: Vec<TypeError>,
    vars: HashMap<String, CheckedType>,
    struct_defs: HashMap<String, ast::StructDef>,
    fn_sigs: HashMap<String, FnSig>,
    comptime_env: HashMap<String, u64>,
    native_width: u64,
    current_span: ast::Span,
    ret_ty: Option<VecType>,
}

impl TypeChecker {
    fn new(comptime_env: &HashMap<String, u64>, native_width: u64) -> Self {
        TypeChecker {
            errors: Vec::new(),
            vars: HashMap::new(),
            struct_defs: HashMap::new(),
            fn_sigs: HashMap::new(),
            comptime_env: comptime_env.clone(),
            native_width,
            current_span: ast::Span::dummy(),
            ret_ty: None,
        }
    }

    fn err(&mut self, msg: String) {
        self.errors.push(TypeError {
            span: self.current_span,
            message: msg,
        });
    }

    // ─── Function checking ───

    fn check_fn(&mut self, f: &ast::FnDef) {
        self.vars.clear();

        // Bind params
        for p in &f.params {
            let ty = if is_ptr_type_name(&p.ty.name) {
                let elem = match &p.ty.width {
                    Some(ast::Width::Param(name)) => ScalarType::from_name(name),
                    _ => {
                        self.err(format!("ptr type '{}' must have element type (e.g. ptr[u8])", p.name));
                        ScalarType::U8
                    }
                };
                CheckedType::Ptr(elem)
            } else if is_scalar_type_name(&p.ty.name) {
                CheckedType::Vec(resolve_type(&p.ty, &self.comptime_env, self.native_width))
            } else {
                // Struct param
                let sdef = self.struct_defs.get(&p.ty.name);
                if let Some(sd) = sdef {
                    let width = p.ty.width.as_ref()
                        .expect("struct type must have width")
                        .eval(&self.comptime_env, self.native_width);
                    let mut struct_env = self.comptime_env.clone();
                    if let Some(cp) = sd.comptime_params.first() {
                        struct_env.insert(cp.name.clone(), width);
                    }
                    let fields: Vec<(String, VecType)> = sd
                        .fields
                        .iter()
                        .map(|fd| {
                            (fd.name.clone(), resolve_type(&fd.ty, &struct_env, self.native_width))
                        })
                        .collect();
                    CheckedType::Struct {
                        name: p.ty.name.clone(),
                        fields,
                    }
                } else {
                    self.err(format!("undefined struct type: {}", p.ty.name));
                    CheckedType::Vec(VecType { scalar: ScalarType::I32, width: 1 })
                }
            };
            self.vars.insert(p.name.clone(), ty);
        }

        // Store return type
        self.ret_ty = f.ret_ty.as_ref().map(|rt| {
            if is_scalar_type_name(&rt.name) {
                resolve_type(rt, &self.comptime_env, self.native_width)
            } else {
                // Struct return: not validated yet
                VecType { scalar: ScalarType::I32, width: 1 }
            }
        });

        // Check body
        for stmt in &f.body {
            self.check_stmt(stmt);
        }
    }

    // ─── Statement checking ───

    fn check_stmt(&mut self, stmt: &ast::Stmt) {
        self.current_span = stmt.span;
        match &stmt.kind {
            ast::StmtKind::Assign { target, ty: ann_ty, value } => {
                match target {
                    ast::AssignTarget::Ident(name) => {
                        if let ast::Expr::StructLit {
                            name: sname,
                            width,
                            fields: field_exprs,
                        } = value
                        {
                            self.check_struct_lit(name, sname, width, field_exprs);
                        } else if let Some(ann) = ann_ty {
                            // Typed assignment
                            if is_scalar_type_name(&ann.name) {
                                let target_ty = resolve_type(ann, &self.comptime_env, self.native_width);
                                if is_literal(value) {
                                    if let Some(kind) = lit_kind(value) {
                                        if !lit_compatible(kind, target_ty.scalar) {
                                            self.err(format!(
                                                "{} not compatible with {:?}",
                                                CheckedType::Literal(kind).describe(),
                                                target_ty.scalar
                                            ));
                                        }
                                    }
                                    self.vars.insert(name.clone(), CheckedType::Vec(target_ty));
                                } else {
                                    let vty = self.infer_expr(value);
                                    if let CheckedType::Vec(inferred) = &vty {
                                        if inferred.width != target_ty.width {
                                            self.err(format!(
                                                "type annotation width {} doesn't match inferred width {}",
                                                target_ty.width, inferred.width
                                            ));
                                        }
                                    }
                                    self.vars.insert(name.clone(), CheckedType::Vec(target_ty));
                                }
                            } else {
                                self.err(format!("type annotation must be a scalar type, got: {}", ann.name));
                                let _ = self.infer_expr(value);
                            }
                        } else {
                            let vty = self.infer_expr(value);
                            self.vars.insert(name.clone(), vty);
                        }
                    }
                    ast::AssignTarget::Scatter { base, index, mask, .. } => {
                        let bty = self.infer_expr(base);
                        if !matches!(bty, CheckedType::Ptr(_)) {
                            self.err(format!("scatter target must be a pointer, got {}", bty.describe()));
                        }
                        let ity = self.infer_expr(index);
                        if let CheckedType::Vec(ivt) = &ity {
                            if !ivt.scalar.is_int() {
                                self.err("scatter index must be an integer vector".to_string());
                            }
                        }
                        if let Some(m) = mask {
                            let mty = self.infer_expr(m);
                            if let CheckedType::Vec(mvt) = &mty {
                                if mvt.scalar != ScalarType::Bool {
                                    self.err("scatter mask must be a bool vector".to_string());
                                }
                            }
                        }
                        let _ = self.infer_expr(value);
                    }
                    ast::AssignTarget::Destructure(_) => {
                        let _ = self.infer_expr(value);
                    }
                }
            }
            ast::StmtKind::Return(expr) => {
                let rty = self.infer_expr(expr);
                if let (Some(expected), CheckedType::Vec(actual)) = (&self.ret_ty, &rty) {
                    if expected.width != actual.width {
                        self.err(format!(
                            "return type width mismatch: expected {}[{}], got {:?}[{}]",
                            format!("{:?}", expected.scalar),
                            expected.width,
                            actual.scalar,
                            actual.width
                        ));
                    }
                }
            }
            ast::StmtKind::Expr(expr) => {
                let _ = self.infer_expr(expr);
            }
            ast::StmtKind::If {
                cond,
                then_body,
                else_body,
            } => {
                let cty = self.infer_expr(cond);
                if let CheckedType::Vec(vt) = &cty {
                    if vt.scalar != ScalarType::Bool {
                        self.err("if condition must be bool".to_string());
                    }
                    if vt.width != 1 {
                        self.err("if condition must be width 1 (scalar bool)".to_string());
                    }
                }
                for stmt in then_body {
                    self.check_stmt(stmt);
                }
                for stmt in else_body {
                    self.check_stmt(stmt);
                }
            }
            ast::StmtKind::While { cond, body } => {
                let cty = self.infer_expr(cond);
                if let CheckedType::Vec(vt) = &cty {
                    if vt.scalar != ScalarType::Bool {
                        self.err("while condition must be bool".to_string());
                    }
                    if vt.width != 1 {
                        self.err("while condition must be width 1 (scalar bool)".to_string());
                    }
                }
                for stmt in body {
                    self.check_stmt(stmt);
                }
            }
            ast::StmtKind::Stream {
                chunk_name,
                chunk_ty,
                buffer,
                carry,
                body,
                carry_updates,
            } => {
                // Check buffer is ptr
                if let Some(bty) = self.vars.get(buffer) {
                    if !matches!(bty, CheckedType::Ptr(_)) {
                        self.err(format!("stream buffer '{}' must be a pointer, got {}", buffer, bty.describe()));
                    }
                } else {
                    self.err(format!("undefined variable: {}", buffer));
                }

                // Save outer vars, create inner scope
                let outer_vars = self.vars.clone();
                let outer_ret = self.ret_ty.take();

                // Bind chunk
                let chunk_vty = resolve_type(chunk_ty, &self.comptime_env, self.native_width);
                self.vars.insert(chunk_name.clone(), CheckedType::Vec(chunk_vty));

                // Bind chunk_offset as i32[1]
                self.vars.insert(
                    "chunk_offset".into(),
                    CheckedType::Vec(VecType {
                        scalar: ScalarType::I32,
                        width: 1,
                    }),
                );

                // Bind carries
                let mut carry_types: Vec<(String, VecType)> = Vec::new();
                for c in carry {
                    let cty = resolve_type(&c.ty, &self.comptime_env, self.native_width);
                    // Check init expr compatibility
                    let init_ty = self.infer_expr(&c.init);
                    if let CheckedType::Vec(ivt) = &init_ty {
                        if ivt.width != cty.width {
                            self.err(format!(
                                "carry '{}' init width {} doesn't match declared width {}",
                                c.name, ivt.width, cty.width
                            ));
                        }
                    }
                    self.vars.insert(c.name.clone(), CheckedType::Vec(cty));
                    carry_types.push((c.name.clone(), cty));
                }

                // Check body
                for stmt in body {
                    self.check_stmt(stmt);
                }

                // Check carry updates
                for (cname, cexpr) in carry_updates {
                    let uty = self.infer_expr(cexpr);
                    if let Some((_, expected_vt)) = carry_types.iter().find(|(n, _)| n == cname) {
                        if let CheckedType::Vec(actual_vt) = &uty {
                            if actual_vt.width != expected_vt.width {
                                self.err(format!(
                                    "carry '{}' update width {} doesn't match declared width {}",
                                    cname, actual_vt.width, expected_vt.width
                                ));
                            }
                        }
                    }
                }

                // Restore outer scope, but bind carry outputs into it
                self.vars = outer_vars;
                self.ret_ty = outer_ret;
                for (cname, cty) in &carry_types {
                    self.vars.insert(cname.clone(), CheckedType::Vec(*cty));
                }
            }
        }
    }

    fn check_struct_lit(
        &mut self,
        var_name: &str,
        struct_name: &str,
        width: &ast::Width,
        field_exprs: &[(String, ast::Expr)],
    ) {
        let sdef = match self.struct_defs.get(struct_name) {
            Some(sd) => sd.clone(),
            None => {
                self.err(format!("undefined struct: {}", struct_name));
                return;
            }
        };

        let concrete_width = width.eval(&self.comptime_env, self.native_width);
        let mut struct_env = self.comptime_env.clone();
        if let Some(cp) = sdef.comptime_params.first() {
            struct_env.insert(cp.name.clone(), concrete_width);
        }

        let mut fields: Vec<(String, VecType)> = Vec::new();
        for fd in &sdef.fields {
            let expected_vty = resolve_type(&fd.ty, &struct_env, self.native_width);
            if let Some((_, expr)) = field_exprs.iter().find(|(n, _)| n == &fd.name) {
                let fty = self.infer_expr(expr);
                if let CheckedType::Vec(actual) = &fty {
                    if actual.width != expected_vty.width {
                        self.err(format!(
                            "struct '{}' field '{}' width mismatch: expected {}, got {}",
                            struct_name, fd.name, expected_vty.width, actual.width
                        ));
                    }
                }
                fields.push((fd.name.clone(), expected_vty));
            } else {
                self.err(format!(
                    "struct literal '{}' missing field '{}'",
                    struct_name, fd.name
                ));
                fields.push((fd.name.clone(), expected_vty));
            }
        }

        self.vars.insert(
            var_name.to_string(),
            CheckedType::Struct {
                name: struct_name.to_string(),
                fields,
            },
        );
    }

    // ─── Expression type inference ───

    fn infer_expr(&mut self, expr: &ast::Expr) -> CheckedType {
        match expr {
            ast::Expr::Ident(name) => {
                if let Some(ty) = self.vars.get(name) {
                    ty.clone()
                } else {
                    self.err(format!("undefined variable: {}", name));
                    // Return a dummy to keep checking
                    CheckedType::Vec(VecType {
                        scalar: ScalarType::I32,
                        width: 1,
                    })
                }
            }

            ast::Expr::IntLit(_) => CheckedType::Literal(LitKind::Int),
            ast::Expr::FloatLit(_) => CheckedType::Literal(LitKind::Float),
            ast::Expr::BoolLit(_) => CheckedType::Literal(LitKind::Bool),
            ast::Expr::CharLit(_) => CheckedType::Literal(LitKind::Char),

            ast::Expr::VecLit { elem_type, values } => {
                let scalar = ScalarType::from_name(elem_type);
                CheckedType::Vec(VecType {
                    scalar,
                    width: values.len() as u64,
                })
            }

            ast::Expr::BinOp { op, lhs, rhs } => self.infer_binop(*op, lhs, rhs),

            ast::Expr::UnaryOp { op, operand } => {
                if is_literal(expr) {
                    return CheckedType::Literal(lit_kind(expr).unwrap_or(LitKind::Int));
                }
                let oty = self.infer_expr(operand);
                match oty {
                    CheckedType::Vec(vt) => {
                        match op {
                            ast::UnaryOp::Not => {
                                if vt.scalar.is_float() {
                                    self.err("bitwise NOT (~) requires integer type".to_string());
                                }
                            }
                            ast::UnaryOp::Neg => {} // works on both int and float
                        }
                        CheckedType::Vec(vt)
                    }
                    CheckedType::Literal(k) => CheckedType::Literal(k),
                    other => {
                        self.err(format!("cannot apply unary op to {}", other.describe()));
                        other
                    }
                }
            }

            ast::Expr::Masked { mask, body, fallback } => {
                let mty = self.infer_expr(mask);
                if let CheckedType::Vec(mvt) = &mty {
                    if mvt.scalar != ScalarType::Bool {
                        self.err("mask must be a bool vector".to_string());
                    }
                }

                let fallback_expr = match fallback {
                    Some(f) => f.as_ref(),
                    None => {
                        self.err("masked expression must have a fallback".to_string());
                        return self.infer_expr(body);
                    }
                };

                // Handle literal broadcasting between body and fallback
                let (bty, _fty) = match (is_literal(body), is_literal(fallback_expr)) {
                    (true, false) => {
                        let fty = self.infer_expr(fallback_expr);
                        (fty.clone(), fty)
                    }
                    (false, true) => {
                        let bty = self.infer_expr(body);
                        (bty.clone(), bty)
                    }
                    _ => {
                        let bty = self.infer_expr(body);
                        let fty = self.infer_expr(fallback_expr);
                        (bty, fty)
                    }
                };
                bty
            }

            ast::Expr::Reduction { op: _, operand } => {
                let oty = self.infer_expr(operand);
                match oty {
                    CheckedType::Vec(vt) => CheckedType::Vec(VecType {
                        scalar: vt.scalar,
                        width: 1,
                    }),
                    _ => {
                        self.err("reduction operand must be a vector".to_string());
                        oty
                    }
                }
            }

            ast::Expr::Scan { op: _, operand, seed } => {
                let oty = self.infer_expr(operand);
                if let Some(s) = seed {
                    let sty = self.infer_expr(s);
                    if let (CheckedType::Vec(ovt), CheckedType::Vec(svt)) = (&oty, &sty) {
                        if svt.width != 1 {
                            self.err(format!("scan seed must be width 1, got {}", svt.width));
                        }
                        if svt.scalar != ovt.scalar {
                            self.err("scan seed scalar type must match operand".to_string());
                        }
                    }
                }
                oty
            }

            ast::Expr::Field { base, field } => {
                let bty = self.infer_expr(base);
                match bty {
                    CheckedType::Struct { name, fields } => {
                        if let Some((_, vt)) = fields.iter().find(|(n, _)| n == field) {
                            CheckedType::Vec(*vt)
                        } else {
                            self.err(format!("struct '{}' has no field '{}'", name, field));
                            CheckedType::Vec(VecType { scalar: ScalarType::I32, width: 1 })
                        }
                    }
                    _ => {
                        self.err(format!("field access on non-struct: {}", bty.describe()));
                        CheckedType::Vec(VecType { scalar: ScalarType::I32, width: 1 })
                    }
                }
            }

            ast::Expr::Call { func, args } => self.infer_call(func, args),

            ast::Expr::Gather { base, index, mask } => {
                let bty = self.infer_expr(base);
                let ity = self.infer_expr(index);
                if let Some(m) = mask {
                    let mty = self.infer_expr(m);
                    if let CheckedType::Vec(mvt) = &mty {
                        if mvt.scalar != ScalarType::Bool {
                            self.err("gather mask must be bool vector".to_string());
                        }
                    }
                }
                match (&bty, &ity) {
                    (CheckedType::Ptr(elem_scalar), CheckedType::Vec(idx_vt)) => {
                        if !idx_vt.scalar.is_int() {
                            self.err("gather index must be integer vector".to_string());
                        }
                        CheckedType::Vec(VecType {
                            scalar: *elem_scalar,
                            width: idx_vt.width,
                        })
                    }
                    _ => {
                        if !matches!(bty, CheckedType::Ptr(_)) {
                            self.err(format!("gather base must be a pointer, got {}", bty.describe()));
                        }
                        CheckedType::Vec(VecType { scalar: ScalarType::I32, width: 1 })
                    }
                }
            }

            ast::Expr::StructLit { name, width, fields } => {
                // Struct literal in expression position (e.g. return)
                let sdef = match self.struct_defs.get(name) {
                    Some(sd) => sd.clone(),
                    None => {
                        self.err(format!("undefined struct: {}", name));
                        return CheckedType::Vec(VecType { scalar: ScalarType::I32, width: 1 });
                    }
                };
                let concrete_width = width.eval(&self.comptime_env, self.native_width);
                let mut struct_env = self.comptime_env.clone();
                if let Some(cp) = sdef.comptime_params.first() {
                    struct_env.insert(cp.name.clone(), concrete_width);
                }
                let checked_fields: Vec<(String, VecType)> = sdef
                    .fields
                    .iter()
                    .map(|fd| {
                        let expected = resolve_type(&fd.ty, &struct_env, self.native_width);
                        if let Some((_, expr)) = fields.iter().find(|(n, _)| n == &fd.name) {
                            let _ = self.infer_expr(expr);
                        } else {
                            self.err(format!("struct '{}' missing field '{}'", name, fd.name));
                        }
                        (fd.name.clone(), expected)
                    })
                    .collect();
                CheckedType::Struct {
                    name: name.clone(),
                    fields: checked_fields,
                }
            }

            ast::Expr::Load { ty, ptr, offset, .. } => {
                let pty = self.infer_expr(ptr);
                if !matches!(pty, CheckedType::Ptr(_)) {
                    self.err(format!("load requires pointer, got {}", pty.describe()));
                }
                if let Some(off) = offset {
                    let _ = self.infer_expr(off);
                }
                CheckedType::Vec(resolve_type(ty, &self.comptime_env, self.native_width))
            }
        }
    }

    // ─── Binary op type inference ───

    fn infer_binop(
        &mut self,
        op: ast::BinOp,
        lhs: &ast::Expr,
        rhs: &ast::Expr,
    ) -> CheckedType {
        // Bit shifts: RHS must be an int literal
        if op == ast::BinOp::BitShl || op == ast::BinOp::BitShr {
            if !matches!(rhs, ast::Expr::IntLit(_)) {
                self.err("bit shift amount must be an integer literal".to_string());
            }
            let lty = self.infer_expr(lhs);
            if let CheckedType::Vec(vt) = &lty {
                if vt.scalar.is_float() {
                    self.err("bit shift requires integer type".to_string());
                }
            }
            return lty;
        }

        // Handle literal broadcasting
        let (lty, rty) = match (is_literal(lhs), is_literal(rhs)) {
            (true, true) => {
                // Both literals: return literal type
                let lk = lit_kind(lhs).unwrap_or(LitKind::Int);
                return CheckedType::Literal(lk);
            }
            (true, false) => {
                let rt = self.infer_expr(rhs);
                let lt = if let CheckedType::Vec(rvt) = &rt {
                    if let Some(kind) = lit_kind(lhs) {
                        if !lit_compatible(kind, rvt.scalar) {
                            self.err(format!(
                                "{} not compatible with {:?}",
                                CheckedType::Literal(kind).describe(),
                                rvt.scalar
                            ));
                        }
                    }
                    CheckedType::Vec(*rvt) // literal broadcasts to match
                } else {
                    self.infer_expr(lhs)
                };
                (lt, rt)
            }
            (false, true) => {
                let lt = self.infer_expr(lhs);
                let rt = if let CheckedType::Vec(lvt) = &lt {
                    if let Some(kind) = lit_kind(rhs) {
                        if !lit_compatible(kind, lvt.scalar) {
                            self.err(format!(
                                "{} not compatible with {:?}",
                                CheckedType::Literal(kind).describe(),
                                lvt.scalar
                            ));
                        }
                    }
                    CheckedType::Vec(*lvt) // literal broadcasts to match
                } else {
                    self.infer_expr(rhs)
                };
                (lt, rt)
            }
            (false, false) => {
                let lt = self.infer_expr(lhs);
                let rt = self.infer_expr(rhs);
                (lt, rt)
            }
        };

        // Both should be Vec by now
        let (lvt, rvt) = match (&lty, &rty) {
            (CheckedType::Vec(l), CheckedType::Vec(r)) => (*l, *r),
            _ => return lty, // Can't check further
        };

        // Auto-broadcast width-1
        let (lvt, rvt) = if lvt.width == 1 && rvt.width > 1 {
            (VecType { scalar: lvt.scalar, width: rvt.width }, rvt)
        } else if rvt.width == 1 && lvt.width > 1 {
            (lvt, VecType { scalar: rvt.scalar, width: lvt.width })
        } else {
            (lvt, rvt)
        };

        // Width check
        if lvt.width != rvt.width {
            self.err(format!(
                "width mismatch: {:?}[{}] vs {:?}[{}]",
                lvt.scalar, lvt.width, rvt.scalar, rvt.width
            ));
        }

        // Result type
        match op {
            ast::BinOp::Add
            | ast::BinOp::Sub
            | ast::BinOp::Mul
            | ast::BinOp::Div
            | ast::BinOp::And
            | ast::BinOp::Or
            | ast::BinOp::Xor => CheckedType::Vec(lvt),

            ast::BinOp::Gt
            | ast::BinOp::Lt
            | ast::BinOp::GtEq
            | ast::BinOp::LtEq
            | ast::BinOp::EqEq
            | ast::BinOp::NotEq => CheckedType::Vec(VecType {
                scalar: ScalarType::Bool,
                width: lvt.width,
            }),

            ast::BinOp::BitShl | ast::BinOp::BitShr => unreachable!(),
        }
    }

    // ─── Builtin call type inference ───

    fn infer_call(
        &mut self,
        func: &ast::Expr,
        args: &[ast::CallArg],
    ) -> CheckedType {
        let fname = match func {
            ast::Expr::Ident(n) => n.as_str(),
            ast::Expr::Field { base, field } => {
                // scan.add, scan.xor, etc.
                if let ast::Expr::Ident(base_name) = base.as_ref() {
                    if base_name == "scan" {
                        return self.infer_scan_call(field, args);
                    }
                }
                let _ = self.infer_expr(func);
                return CheckedType::Vec(VecType { scalar: ScalarType::I32, width: 1 });
            }
            _ => {
                let _ = self.infer_expr(func);
                return CheckedType::Vec(VecType { scalar: ScalarType::I32, width: 1 });
            }
        };

        match fname {
            "sqrt" => {
                self.check_arg_count(fname, args, 1);
                let aty = self.infer_arg(args, 0);
                if let CheckedType::Vec(vt) = &aty {
                    if !vt.scalar.is_float() {
                        self.err("sqrt requires float type".to_string());
                    }
                }
                aty
            }
            "popcount" => {
                self.check_arg_count(fname, args, 1);
                let aty = self.infer_arg(args, 0);
                if let CheckedType::Vec(vt) = &aty {
                    if vt.scalar != ScalarType::Bool {
                        self.err("popcount requires bool vector".to_string());
                    }
                }
                CheckedType::Vec(VecType { scalar: ScalarType::I32, width: 1 })
            }
            "compress" => {
                self.check_arg_count(fname, args, 2);
                let dty = self.infer_arg(args, 0);
                let mty = self.infer_arg(args, 1);
                if let CheckedType::Vec(mvt) = &mty {
                    if mvt.scalar != ScalarType::Bool {
                        self.err("compress mask must be bool vector".to_string());
                    }
                }
                dty
            }
            "store" | "store_at" => {
                if args.len() < 2 || args.len() > 3 {
                    self.err(format!("{} takes 2 or 3 arguments, got {}", fname, args.len()));
                }
                for i in 0..args.len() {
                    let _ = self.infer_arg(args, i);
                }
                CheckedType::Void
            }
            "iota" => {
                self.check_arg_count(fname, args, 1);
                if let Some(arg) = args.first() {
                    match &arg.value {
                        ast::Expr::IntLit(n) => {
                            return CheckedType::Vec(VecType {
                                scalar: ScalarType::I32,
                                width: *n as u64,
                            });
                        }
                        _ => self.err("iota argument must be an integer literal".to_string()),
                    }
                }
                CheckedType::Vec(VecType { scalar: ScalarType::I32, width: 1 })
            }
            "extract" => {
                self.check_arg_count(fname, args, 2);
                let aty = self.infer_arg(args, 0);
                if args.len() > 1 {
                    if !matches!(args[1].value, ast::Expr::IntLit(_)) {
                        self.err("extract lane must be an integer literal".to_string());
                    }
                }
                match aty {
                    CheckedType::Vec(vt) => CheckedType::Vec(VecType { scalar: vt.scalar, width: 1 }),
                    _ => aty,
                }
            }
            "compressstore" => {
                self.check_arg_count(fname, args, 4);
                for i in 0..args.len().min(4) {
                    let _ = self.infer_arg(args, i);
                }
                CheckedType::Void
            }
            "to_i32" | "to_i64" => {
                self.check_arg_count(fname, args, 1);
                let aty = self.infer_arg(args, 0);
                let target_scalar = if fname == "to_i32" { ScalarType::I32 } else { ScalarType::I64 };
                match aty {
                    CheckedType::Vec(vt) => CheckedType::Vec(VecType { scalar: target_scalar, width: vt.width }),
                    _ => aty,
                }
            }
            "to_bitmask" => {
                self.check_arg_count(fname, args, 1);
                let aty = self.infer_arg(args, 0);
                if let CheckedType::Vec(vt) = &aty {
                    if vt.scalar != ScalarType::Bool {
                        self.err("to_bitmask requires bool vector".to_string());
                    }
                }
                CheckedType::Vec(VecType { scalar: ScalarType::U64, width: 1 })
            }
            "from_bitmask" => {
                self.check_arg_count(fname, args, 2);
                let _ = self.infer_arg(args, 0);
                if let Some(arg) = args.get(1) {
                    match &arg.value {
                        ast::Expr::IntLit(n) => {
                            return CheckedType::Vec(VecType {
                                scalar: ScalarType::Bool,
                                width: *n as u64,
                            });
                        }
                        _ => self.err("from_bitmask width must be an integer literal".to_string()),
                    }
                }
                CheckedType::Vec(VecType { scalar: ScalarType::Bool, width: 64 })
            }
            "clmul" => {
                self.check_arg_count(fname, args, 2);
                let _ = self.infer_arg(args, 0);
                let _ = self.infer_arg(args, 1);
                CheckedType::Vec(VecType { scalar: ScalarType::U64, width: 1 })
            }
            "xor" => {
                self.check_arg_count(fname, args, 2);
                let aty = self.infer_arg(args, 0);
                let _ = self.infer_arg(args, 1);
                aty
            }
            "bit_shr" | "bit_shl" | "lane_shr" | "lane_shl" => {
                self.check_arg_count(fname, args, 2);
                let aty = self.infer_arg(args, 0);
                if args.len() > 1 {
                    if !matches!(args[1].value, ast::Expr::IntLit(_)) {
                        self.err(format!("{} amount must be an integer literal", fname));
                    }
                }
                aty
            }
            "tbl" => {
                self.check_arg_count(fname, args, 2);
                let tty = self.infer_arg(args, 0);
                let ity = self.infer_arg(args, 1);
                if let CheckedType::Vec(tvt) = &tty {
                    if tvt.width != 16 || tvt.scalar != ScalarType::U8 {
                        self.err("tbl table must be u8[16]".to_string());
                    }
                }
                ity // result type matches indices
            }
            "gather" => {
                self.check_arg_count(fname, args, 2);
                let bty = self.infer_arg(args, 0);
                let ity = self.infer_arg(args, 1);
                match (&bty, &ity) {
                    (CheckedType::Ptr(elem), CheckedType::Vec(ivt)) => {
                        CheckedType::Vec(VecType { scalar: *elem, width: ivt.width })
                    }
                    _ => {
                        if !matches!(bty, CheckedType::Ptr(_)) {
                            self.err("gather first argument must be a pointer".to_string());
                        }
                        ity
                    }
                }
            }
            "any" => {
                self.check_arg_count(fname, args, 1);
                let _ = self.infer_arg(args, 0);
                CheckedType::Vec(VecType { scalar: ScalarType::Bool, width: 1 })
            }
            "split_lo" | "split_hi" => {
                self.check_arg_count(fname, args, 1);
                let aty = self.infer_arg(args, 0);
                match aty {
                    CheckedType::Vec(vt) => CheckedType::Vec(VecType { scalar: vt.scalar, width: vt.width / 2 }),
                    _ => aty,
                }
            }
            "ctz" | "clear_lowest_bit" => {
                self.check_arg_count(fname, args, 1);
                self.infer_arg(args, 0)
            }
            "load_at" => {
                self.check_arg_count(fname, args, 2);
                let pty = self.infer_arg(args, 0);
                let _ = self.infer_arg(args, 1); // offset
                match pty {
                    CheckedType::Ptr(scalar) => CheckedType::Vec(VecType { scalar, width: 1 }),
                    _ => {
                        self.err("load_at first argument must be a pointer".to_string());
                        CheckedType::Vec(VecType { scalar: ScalarType::I32, width: 1 })
                    }
                }
            }
            "ptr_add" => {
                self.check_arg_count(fname, args, 2);
                let pty = self.infer_arg(args, 0);
                let _ = self.infer_arg(args, 1); // offset
                match pty {
                    CheckedType::Ptr(scalar) => CheckedType::Ptr(scalar),
                    _ => {
                        self.err("ptr_add first argument must be a pointer".to_string());
                        CheckedType::Ptr(ScalarType::U8)
                    }
                }
            }
            "bswap" => {
                self.check_arg_count(fname, args, 1);
                let aty = self.infer_arg(args, 0);
                match &aty {
                    CheckedType::Vec(vt) => {
                        match vt.scalar {
                            ScalarType::U16 | ScalarType::I16 |
                            ScalarType::U32 | ScalarType::I32 |
                            ScalarType::U64 | ScalarType::I64 => {},
                            _ => self.err("bswap requires u16/i16/u32/i32/u64/i64 element type".to_string()),
                        }
                    }
                    _ => self.err("bswap argument must be a vector".to_string()),
                }
                aty
            }
            _ => {
                // Check if it's a user-defined function
                if let Some(sig) = self.fn_sigs.get(fname).cloned() {
                    // Validate argument count
                    if args.len() != sig.params.len() {
                        self.err(format!(
                            "function '{}' takes {} argument{}, got {}",
                            fname, sig.params.len(),
                            if sig.params.len() == 1 { "" } else { "s" },
                            args.len()
                        ));
                    }
                    // Infer arg types (for side effects / error checking)
                    for a in args {
                        let _ = self.infer_expr(&a.value);
                    }
                    // Return the declared return type
                    sig.ret_ty.unwrap_or(CheckedType::Void)
                } else {
                    self.err(format!("unknown function: {}", fname));
                    for a in args {
                        let _ = self.infer_expr(&a.value);
                    }
                    CheckedType::Vec(VecType { scalar: ScalarType::I32, width: 1 })
                }
            }
        }
    }

    fn infer_scan_call(&mut self, op_name: &str, args: &[ast::CallArg]) -> CheckedType {
        match op_name {
            "add" | "xor" | "max" | "preceding_any" => {}
            _ => self.err(format!("unknown scan operation: scan.{}", op_name)),
        }
        if args.is_empty() || args.len() > 2 {
            self.err(format!("scan.{} takes 1 or 2 arguments, got {}", op_name, args.len()));
        }
        let oty = if !args.is_empty() {
            self.infer_expr(&args[0].value)
        } else {
            CheckedType::Vec(VecType { scalar: ScalarType::I32, width: 1 })
        };
        if args.len() == 2 {
            let sty = self.infer_expr(&args[1].value);
            if let (CheckedType::Vec(ovt), CheckedType::Vec(svt)) = (&oty, &sty) {
                if svt.width != 1 {
                    self.err(format!("scan seed must be width 1, got {}", svt.width));
                }
                if svt.scalar != ovt.scalar {
                    self.err("scan seed scalar type must match operand".to_string());
                }
            }
        }
        oty
    }

    fn check_arg_count(&mut self, name: &str, args: &[ast::CallArg], expected: usize) {
        if args.len() != expected {
            self.err(format!(
                "{} takes {} argument{}, got {}",
                name,
                expected,
                if expected == 1 { "" } else { "s" },
                args.len()
            ));
        }
    }

    fn infer_arg(&mut self, args: &[ast::CallArg], idx: usize) -> CheckedType {
        if let Some(arg) = args.get(idx) {
            self.infer_expr(&arg.value)
        } else {
            CheckedType::Vec(VecType { scalar: ScalarType::I32, width: 1 })
        }
    }
}

// ─── Public API ───

pub fn typecheck(
    items: &[ast::Item],
    comptime_env: &HashMap<String, u64>,
    native_width: u64,
) -> Result<(), Vec<TypeError>> {
    let mut checker = TypeChecker::new(comptime_env, native_width);

    // Collect struct defs first
    for item in items {
        if let ast::Item::Struct(sd) = item {
            checker.struct_defs.insert(sd.name.clone(), sd.clone());
        }
    }

    // Collect function signatures for inter-function call checking
    for item in items {
        if let ast::Item::Fn(f) = item {
            let params: Vec<(String, CheckedType)> = f.params.iter().map(|p| {
                let ty = if is_ptr_type_name(&p.ty.name) {
                    let elem = match &p.ty.width {
                        Some(ast::Width::Param(name)) => ScalarType::from_name(name),
                        _ => ScalarType::U8,
                    };
                    CheckedType::Ptr(elem)
                } else if is_scalar_type_name(&p.ty.name) {
                    CheckedType::Vec(resolve_type(&p.ty, &comptime_env, native_width))
                } else {
                    // Struct param — resolve to struct type
                    if let Some(sd) = checker.struct_defs.get(&p.ty.name) {
                        let width = p.ty.width.as_ref()
                            .expect("struct type must have width")
                            .eval(&comptime_env, native_width);
                        let mut struct_env = comptime_env.clone();
                        if let Some(cp) = sd.comptime_params.first() {
                            struct_env.insert(cp.name.clone(), width);
                        }
                        let fields = sd.fields.iter().map(|fld| {
                            (fld.name.clone(), resolve_type(&fld.ty, &struct_env, native_width))
                        }).collect();
                        CheckedType::Struct { name: p.ty.name.clone(), fields }
                    } else {
                        CheckedType::Vec(VecType { scalar: ScalarType::I32, width: 1 })
                    }
                };
                (p.name.clone(), ty)
            }).collect();

            let ret_ty = f.ret_ty.as_ref().map(|rt| {
                if is_ptr_type_name(&rt.name) {
                    let elem = match &rt.width {
                        Some(ast::Width::Param(name)) => ScalarType::from_name(name),
                        _ => ScalarType::U8,
                    };
                    CheckedType::Ptr(elem)
                } else if is_scalar_type_name(&rt.name) {
                    CheckedType::Vec(resolve_type(rt, &comptime_env, native_width))
                } else {
                    // Struct return
                    if let Some(sd) = checker.struct_defs.get(&rt.name) {
                        let width = rt.width.as_ref()
                            .expect("struct return must have width")
                            .eval(&comptime_env, native_width);
                        let mut struct_env = comptime_env.clone();
                        if let Some(cp) = sd.comptime_params.first() {
                            struct_env.insert(cp.name.clone(), width);
                        }
                        let fields = sd.fields.iter().map(|fld| {
                            (fld.name.clone(), resolve_type(&fld.ty, &struct_env, native_width))
                        }).collect();
                        CheckedType::Struct { name: rt.name.clone(), fields }
                    } else {
                        CheckedType::Vec(VecType { scalar: ScalarType::I32, width: 1 })
                    }
                }
            });

            checker.fn_sigs.insert(f.name.clone(), FnSig { params, ret_ty });
        }
    }

    for item in items {
        if let ast::Item::Fn(f) = item {
            checker.check_fn(f);
        }
    }

    if checker.errors.is_empty() {
        Ok(())
    } else {
        Err(checker.errors)
    }
}

// ─── Tests ───

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser;

    fn check(source: &str) -> Vec<String> {
        let items = parser::parse(source);
        match typecheck(&items, &HashMap::new(), 8) {
            Ok(()) => vec![],
            Err(errs) => errs.into_iter().map(|e| e.message).collect(),
        }
    }

    fn check_ok(source: &str) {
        let errs = check(source);
        assert!(errs.is_empty(), "expected no errors, got: {:?}", errs);
    }

    fn check_err(source: &str, expected_substr: &str) {
        let errs = check(source);
        assert!(
            errs.iter().any(|e| e.contains(expected_substr)),
            "expected error containing '{}', got: {:?}",
            expected_substr,
            errs
        );
    }

    // --- Basic passing ---

    #[test]
    fn test_simple_add() {
        check_ok("fn f(a: i32[4], b: i32[4]) -> i32[4] { return a + b }");
    }

    #[test]
    fn test_literal_broadcast() {
        check_ok("fn f(a: i32[4]) -> i32[4] { return a + 1 }");
    }

    #[test]
    fn test_char_literal_broadcast() {
        check_ok("fn f(chunk: u8[64]) -> bool[64] { return chunk == '\"' }");
    }

    #[test]
    fn test_comparison_produces_bool() {
        check_ok("fn f(a: i32[4], b: i32[4]) -> bool[4] { return a > b }");
    }

    #[test]
    fn test_width_1_broadcast() {
        check_ok("fn f(a: i32[4], b: i32[1]) -> i32[4] { return a + b }");
    }

    #[test]
    fn test_typed_assign() {
        check_ok("fn f() -> u64[1] { x: u64[1] = ~0\n return x }");
    }

    #[test]
    fn test_vec_literal() {
        check_ok("fn f() -> u8[4] { v = [u8: 1, 2, 3, 4]\n return v }");
    }

    #[test]
    fn test_vec_repeat() {
        check_ok("fn f() -> u8[4] { v = [u8: 0; 4]\n return v }");
    }

    #[test]
    fn test_masked_expression() {
        check_ok("fn f(a: i32[4], m: bool[4]) -> i32[4] { return [m] a + 1 : a }");
    }

    #[test]
    fn test_reduction() {
        check_ok("fn f(a: f32[8]) -> f32[1] { return +/ a }");
    }

    #[test]
    fn test_xor_operator() {
        check_ok("fn f(a: u64[1], b: u64[1]) -> u64[1] { return a ^ b }");
    }

    #[test]
    fn test_bit_shift() {
        check_ok("fn f(a: u8[64]) -> u8[64] { return a >> 4 }");
    }

    #[test]
    fn test_lane_shift_builtin() {
        check_ok("fn f(a: u8[64]) -> u8[64] { return lane_shr(a, 1) }");
    }

    // --- Error cases ---

    #[test]
    fn test_undefined_variable() {
        check_err(
            "fn f() -> i32[4] { return undefined_var }",
            "undefined variable: undefined_var",
        );
    }

    #[test]
    fn test_width_mismatch() {
        check_err(
            "fn f(a: i32[4], b: i32[8]) -> i32[4] { return a + b }",
            "width mismatch",
        );
    }

    #[test]
    fn test_sqrt_requires_float() {
        check_err(
            "fn f(a: i32[4]) -> i32[4] { return sqrt(a) }",
            "sqrt requires float",
        );
    }

    #[test]
    fn test_wrong_arg_count() {
        check_err(
            "fn f(a: f32[4], b: f32[4]) -> f32[4] { return sqrt(a, b) }",
            "sqrt takes 1 argument",
        );
    }

    #[test]
    fn test_unknown_function() {
        check_err(
            "fn f(a: i32[4]) -> i32[4] { return nonexistent(a) }",
            "unknown function: nonexistent",
        );
    }

    #[test]
    fn test_popcount_requires_bool() {
        check_err(
            "fn f(a: i32[4]) -> i32[1] { return popcount(a) }",
            "popcount requires bool",
        );
    }

    #[test]
    fn test_to_bitmask_requires_bool() {
        check_err(
            "fn f(a: i32[64]) -> u64[1] { return to_bitmask(a) }",
            "to_bitmask requires bool",
        );
    }

    #[test]
    fn test_float_literal_in_int_context() {
        check_err(
            "fn f(a: i32[4]) -> i32[4] { return a + 1.0 }",
            "not compatible",
        );
    }

    #[test]
    fn test_bitwise_not_on_float() {
        check_err(
            "fn f(a: f32[4]) -> f32[4] { return ~a }",
            "bitwise NOT (~) requires integer",
        );
    }

    #[test]
    fn test_field_on_non_struct() {
        check_err(
            "fn f(a: i32[4]) -> i32[4] { return a.x }",
            "field access on non-struct",
        );
    }

    // --- Stream validation ---

    #[test]
    fn test_stream_valid() {
        check_ok(r#"
            fn f(input: ptr[u8]) -> i32[1] {
                stream chunk: u8[8] over input carry (count: i32[1] = 0) {
                    is_a = chunk == 'a'
                    carry count = count + popcount(is_a)
                }
                return count
            }
        "#);
    }

    #[test]
    fn test_stream_buffer_must_be_ptr() {
        check_err(
            "fn f(input: i32[4]) -> i32[1] { stream chunk: u8[8] over input { x = chunk } return 0 }",
            "must be a pointer",
        );
    }

    // --- json_stage1 as integration test ---

    #[test]
    fn test_json_stage1_typechecks() {
        let source = std::fs::read_to_string("examples/json_stage1.simd")
            .expect("cannot read json_stage1.simd");
        let items = parser::parse(&source);
        let result = typecheck(&items, &HashMap::new(), 8);
        match result {
            Ok(()) => {}
            Err(errs) => {
                for e in &errs {
                    eprintln!("  {}", e);
                }
                panic!("json_stage1.simd had {} type errors", errs.len());
            }
        }
    }

    // --- load_at ---

    #[test]
    fn test_load_at_ok() {
        check_ok("fn f(buf: ptr[u8], off: i32[1]) -> u8[1] { return load_at(buf, off) }");
    }

    #[test]
    fn test_load_at_non_ptr() {
        check_err(
            "fn f(a: i32[4], off: i32[1]) -> i32[1] { return load_at(a, off) }",
            "load_at first argument must be a pointer",
        );
    }

    // --- ptr_add ---

    #[test]
    fn test_ptr_add_ok() {
        check_ok("fn f(buf: ptr[u8], off: i32[1]) -> u8[1] { p2 = ptr_add(buf, off)\n return load_at(p2, 0) }");
    }

    #[test]
    fn test_ptr_add_non_ptr() {
        check_err(
            "fn f(a: i32[4], off: i32[1]) { x = ptr_add(a, off) }",
            "ptr_add first argument must be a pointer",
        );
    }

    // --- bswap ---

    #[test]
    fn test_bswap_u32() {
        check_ok("fn f(a: u32[4]) -> u32[4] { return bswap(a) }");
    }

    #[test]
    fn test_bswap_u64_scalar() {
        check_ok("fn f(a: u64[1]) -> u64[1] { return bswap(a) }");
    }

    #[test]
    fn test_bswap_bool_err() {
        check_err(
            "fn f(a: bool[4]) -> bool[4] { return bswap(a) }",
            "bswap requires u16/i16/u32/i32/u64/i64",
        );
    }

    #[test]
    fn test_bswap_u8_err() {
        check_err(
            "fn f(a: u8[4]) -> u8[4] { return bswap(a) }",
            "bswap requires u16/i16/u32/i32/u64/i64",
        );
    }

    // --- inter-function calls ---

    #[test]
    fn test_inter_fn_call_ok() {
        check_ok(r#"
            fn add_one(x: i32[4]) -> i32[4] { return x + 1 }
            fn f(a: i32[4]) -> i32[4] { return add_one(a) }
        "#);
    }

    #[test]
    fn test_inter_fn_call_wrong_arg_count() {
        check_err(
            r#"
            fn add(x: i32[4], y: i32[4]) -> i32[4] { return x + y }
            fn f(a: i32[4]) -> i32[4] { return add(a) }
            "#,
            "takes 2 arguments, got 1",
        );
    }

    #[test]
    fn test_inter_fn_call_with_ptr() {
        check_ok(r#"
            fn read_byte(buf: ptr[u8], off: i32[1]) -> u8[1] { return load_at(buf, off) }
            fn f(data: ptr[u8]) -> u8[1] { return read_byte(data, 0) }
        "#);
    }
}
