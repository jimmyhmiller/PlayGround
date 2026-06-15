//! Type-checking + lowering to core IR.
//!
//! This pass walks the resolved surface AST, checks types, and emits the
//! monomorphic [`CoreProgram`]. For the v0 vertical slice it handles the
//! non-generic, non-heap subset: primitive scalars, arithmetic / comparison /
//! logic, `if`/`else`, `let`, blocks, direct function calls (incl. recursion),
//! and `return`. Structs, enums, generics+monomorphization, closures, and
//! traits are layered on in subsequent steps against this same scaffold.

use crate::ast::*;
use crate::core::*;
use crate::layout::LayoutRegistry;
use crate::lexer::Span;
use crate::types::{Prim, Ty, TyCtx, lower_type, suffix_prim};
use std::collections::HashMap;
use std::rc::Rc;

#[derive(Debug)]
pub struct LowerError {
    pub msg: String,
    pub span: Span,
}
type LResult<T> = Result<T, LowerError>;

fn err<T>(msg: impl Into<String>, span: Span) -> LResult<T> {
    Err(LowerError { msg: msg.into(), span })
}

/// A resolved function instantiation queued for lowering: the source `FnDef`,
/// the type-parameter substitution to apply, and the mangled name. Works for
/// free functions AND impl methods (methods aren't in `ctx.fns`, so we carry
/// the `FnDef` directly).
#[derive(Clone)]
struct Job {
    f: Rc<FnDef>,
    subst: HashMap<String, Ty>,
    mangled: String,
    /// For an impl method, the receiver's `self` type (already ground). `None`
    /// for a free function.
    self_ty: Option<Ty>,
}

/// Drives monomorphization: assigns each needed instantiation (keyed by its
/// mangled name) a [`FuncId`] and keeps a worklist of jobs to lower.
struct Mono {
    ids: HashMap<String, FuncId>,
    worklist: Vec<(FuncId, Job)>,
}

impl Mono {
    fn new() -> Self {
        Mono { ids: HashMap::new(), worklist: Vec::new() }
    }
    /// Get (or assign) the FuncId for a job (keyed by mangled name), queueing it
    /// for lowering the first time it's seen.
    fn intern(&mut self, job: Job, count: &mut u32) -> FuncId {
        if let Some(id) = self.ids.get(&job.mangled) {
            return *id;
        }
        let id = *count;
        *count += 1;
        self.ids.insert(job.mangled.clone(), id);
        self.worklist.push((id, job));
        id
    }
}

fn mangle(fq: &str, args: &[Ty]) -> String {
    if args.is_empty() {
        fq.to_string()
    } else {
        format!("{}${}", fq, args.iter().map(ty_mangle).collect::<Vec<_>>().join("_"))
    }
}

pub fn lower_program(globals: &crate::resolve::GlobalTable) -> LResult<CoreProgram> {
    let ctx = TyCtx::from_globals(globals);
    let mut reg = LayoutRegistry::new(&ctx);
    let mut prog = CoreProgram::default();
    let mut mono = Mono::new();
    let mut count: u32 = 0;

    // Seed from `main` (which must be non-generic).
    let main_fq = ctx.fns.keys().find(|k| k.rsplit("::").next().unwrap() == "main").cloned();
    let Some(main_fq) = main_fq else {
        return err("no `main` function found", Span::new(0, 0));
    };
    let main_f = ctx.fns[&main_fq].clone();
    let entry = mono.intern(
        Job { f: main_f, subst: HashMap::new(), mangled: main_fq, self_ty: None },
        &mut count,
    );
    prog.entry = Some(entry);

    // Lower instantiations until the worklist drains (transitively reachable).
    while let Some((id, job)) = mono.worklist.pop() {
        let lowered = lower_fn(&job, &ctx, &mut reg, &mut mono, &mut count)?;
        while prog.funcs.len() <= id as usize {
            prog.funcs.push(placeholder_fn());
        }
        prog.funcs[id as usize] = lowered;
    }

    prog.layouts = reg.layouts;
    prog.values = reg.values;
    Ok(prog)
}

fn placeholder_fn() -> CoreFn {
    CoreFn { name: String::new(), params: vec![], ret: Repr::Unit, locals: vec![], body: CoreBlock { stmts: vec![], tail: None } }
}

fn float_intrinsic(name: &str) -> Option<FloatIntrinsic> {
    Some(match name {
        "sqrt" => FloatIntrinsic::Sqrt,
        "abs" => FloatIntrinsic::Abs,
        "floor" => FloatIntrinsic::Floor,
        "ceil" => FloatIntrinsic::Ceil,
        _ => return None,
    })
}

/// The method-index key for a primitive receiver: `i64`, `f64`, etc. — the
/// literal name used in `impl Show for i64`.
fn prim_impl_key(ty: &Ty) -> String {
    match ty {
        Ty::Prim(p) => match p {
            Prim::I8 => "i8", Prim::I16 => "i16", Prim::I32 => "i32", Prim::I64 => "i64",
            Prim::U8 => "u8", Prim::U16 => "u16", Prim::U32 => "u32", Prim::U64 => "u64",
            Prim::F32 => "f32", Prim::F64 => "f64",
            Prim::Bool => "bool", Prim::Char => "char", Prim::Str => "String", Prim::Unit => "()",
        }.to_string(),
        _ => String::new(),
    }
}

fn ty_mangle(t: &Ty) -> String {
    match t {
        Ty::Prim(p) => format!("{:?}", p),
        Ty::Named { name, args } => if args.is_empty() { name.replace("::", ".") } else {
            format!("{}.{}", name.replace("::", "."), args.iter().map(ty_mangle).collect::<Vec<_>>().join("."))
        },
        Ty::Var(v) => format!("V{}", v),
        Ty::Array(e, n) => format!("A{}x{}", ty_mangle(e), n),
        Ty::Tuple(es) => format!("T{}", es.iter().map(ty_mangle).collect::<Vec<_>>().join("_")),
        Ty::Fn { params, ret } => format!("F{}_{}", params.iter().map(ty_mangle).collect::<Vec<_>>().join("_"), ty_mangle(ret)),
        Ty::Infer(n) => format!("I{}", n),
    }
}

struct FnLowerer<'a, 'r, 'm> {
    ctx: &'a TyCtx,
    reg: &'r mut LayoutRegistry<'a>,
    mono: &'m mut Mono,
    count: &'m mut u32,
    /// Type-parameter substitution for the current instantiation (empty if the
    /// enclosing function is non-generic).
    subst: HashMap<String, Ty>,
    /// Lexical scope: name → (LocalId, Ty).
    scope: Vec<HashMap<String, (LocalId, Ty)>>,
    /// All locals' reprs, indexed by LocalId.
    locals: Vec<Repr>,
    /// Parallel: each local's semantic Ty (for checking).
    local_tys: Vec<Ty>,
    ret_ty: Ty,
}

fn lower_fn<'a>(
    job: &Job,
    ctx: &'a TyCtx,
    reg: &mut LayoutRegistry<'a>,
    mono: &mut Mono,
    count: &mut u32,
) -> LResult<CoreFn> {
    let f = &job.f;
    let subst = &job.subst;
    let ret_ty = match &f.ret {
        Some(t) => ground_type(t, subst, ctx)?,
        None => Ty::unit(),
    };
    let mut lo = FnLowerer {
        ctx,
        reg,
        mono,
        count,
        subst: subst.clone(),
        scope: vec![HashMap::new()],
        locals: vec![],
        local_tys: vec![],
        ret_ty: ret_ty.clone(),
    };

    let mut params = Vec::new();
    // A method's implicit `self` parameter comes first.
    if f.has_self {
        let self_ty = job.self_ty.clone()
            .ok_or_else(|| LowerError { msg: "method without a self type".into(), span: f.span })?;
        let repr = lo.repr_of(&self_ty, f.span)?;
        let id = lo.fresh_local(repr.clone(), self_ty.clone());
        lo.bind("self", id, self_ty);
        params.push(repr);
    }
    for p in &f.params {
        let ty = ground_type(&p.ty, &lo.subst.clone(), lo.ctx)?;
        let repr = lo.repr_of(&ty, p.span)?;
        let id = lo.fresh_local(repr.clone(), ty.clone());
        lo.bind(&p.name, id, ty);
        params.push(repr);
    }

    let (body, body_ty) = lo.block_expected(&f.body, Some(&ret_ty))?;
    check_assignable(&body_ty, &ret_ty, f.body.span)?;
    let ret = lo.repr_of(&ret_ty, f.span)?;

    Ok(CoreFn { name: job.mangled.clone(), params, ret, locals: lo.locals, body })
}

/// Lower a surface type to a ground `Ty`, applying the instantiation subst.
fn ground_type(t: &Type, subst: &HashMap<String, Ty>, ctx: &TyCtx) -> LResult<Ty> {
    let gparams: Vec<String> = subst.keys().cloned().collect();
    let raw = lower_type(t, &gparams, ctx).map_err(conv)?;
    Ok(apply_subst(&raw, subst))
}

/// Substitute type variables in a `Ty` using `subst`.
fn apply_subst(ty: &Ty, subst: &HashMap<String, Ty>) -> Ty {
    match ty {
        Ty::Var(v) => subst.get(v).cloned().unwrap_or_else(|| ty.clone()),
        Ty::Named { name, args } => Ty::Named {
            name: name.clone(),
            args: args.iter().map(|a| apply_subst(a, subst)).collect(),
        },
        Ty::Tuple(es) => Ty::Tuple(es.iter().map(|e| apply_subst(e, subst)).collect()),
        Ty::Array(e, n) => Ty::Array(Box::new(apply_subst(e, subst)), *n),
        Ty::Fn { params, ret } => Ty::Fn {
            params: params.iter().map(|p| apply_subst(p, subst)).collect(),
            ret: Box::new(apply_subst(ret, subst)),
        },
        Ty::Prim(_) | Ty::Infer(_) => ty.clone(),
    }
}

fn conv(e: crate::types::TypeError) -> LowerError {
    LowerError { msg: e.msg, span: e.span }
}

impl<'a, 'r, 'm> FnLowerer<'a, 'r, 'm> {
    /// Repr of a type, via the shared layout registry.
    fn repr_of(&mut self, ty: &Ty, span: Span) -> LResult<Repr> {
        self.reg.repr(ty).map_err(|e| LowerError { msg: e.0, span })
    }

    fn fresh_local(&mut self, repr: Repr, ty: Ty) -> LocalId {
        let id = self.locals.len() as LocalId;
        self.locals.push(repr);
        self.local_tys.push(ty);
        id
    }
    fn bind(&mut self, name: &str, id: LocalId, ty: Ty) {
        self.scope.last_mut().unwrap().insert(name.to_string(), (id, ty));
    }
    fn lookup(&self, name: &str) -> Option<(LocalId, Ty)> {
        for s in self.scope.iter().rev() {
            if let Some((id, ty)) = s.get(name) {
                return Some((*id, ty.clone()));
            }
        }
        None
    }
    fn push_scope(&mut self) { self.scope.push(HashMap::new()); }
    fn pop_scope(&mut self) { self.scope.pop(); }

    fn block(&mut self, b: &Block) -> LResult<(CoreBlock, Ty)> {
        self.block_expected(b, None)
    }

    fn block_expected(&mut self, b: &Block, expected: Option<&Ty>) -> LResult<(CoreBlock, Ty)> {
        self.push_scope();
        let mut stmts = Vec::new();
        for s in &b.stmts {
            match s {
                Stmt::Let { pattern, ty, init, span } => {
                    let (init_expr, init_ty) = match init {
                        Some(e) => {
                            let expected = match ty {
                                Some(t) => Some(lower_type(t, &[], self.ctx).map_err(conv)?),
                                None => None,
                            };
                            self.expr(e, expected.as_ref())?
                        }
                        None => return err("let without initializer is not supported in v0", *span),
                    };
                    let name = match &pattern.kind {
                        PatternKind::Binding { name, .. } => name.clone(),
                        PatternKind::Wildcard => "_".to_string(),
                        _ => return err("only simple `let x =` patterns supported in v0", *span),
                    };
                    let repr = init_expr.repr.clone();
                    let id = self.fresh_local(repr, init_ty.clone());
                    self.bind(&name, id, init_ty);
                    stmts.push(CoreStmt::Let(id, init_expr));
                }
                Stmt::Expr(e) => {
                    let (ce, _) = self.expr(e, None)?;
                    stmts.push(CoreStmt::Expr(ce));
                }
                Stmt::Item(_) => return err("nested items not supported in v0", b.span),
            }
        }
        let (tail, ty) = match &b.tail {
            Some(e) => {
                let (ce, t) = self.expr(e, expected)?;
                (Some(ce), t)
            }
            None => (None, Ty::unit()),
        };
        self.pop_scope();
        Ok((CoreBlock { stmts, tail }, ty))
    }

    /// Check + lower an expression. `expected` guides literal defaulting.
    fn expr(&mut self, e: &Expr, expected: Option<&Ty>) -> LResult<(CoreExpr, Ty)> {
        match &*e.kind {
            ExprKind::Int(n, sfx) => {
                let prim = match expected {
                    Some(Ty::Prim(p)) if is_int_prim(*p) && *sfx == crate::lexer::NumSuffix::None => *p,
                    _ => suffix_prim(*sfx, false),
                };
                let sr = scalar_of(prim).unwrap();
                Ok((CoreExpr::new(CoreExprKind::ConstInt(*n, sr), Repr::Scalar(sr)), Ty::Prim(prim)))
            }
            ExprKind::Float(f, sfx) => {
                let prim = match expected {
                    Some(Ty::Prim(p)) if matches!(p, Prim::F32 | Prim::F64) && *sfx == crate::lexer::NumSuffix::None => *p,
                    _ => suffix_prim(*sfx, true),
                };
                let sr = scalar_of(prim).unwrap();
                Ok((CoreExpr::new(CoreExprKind::ConstFloat(*f, sr), Repr::Scalar(sr)), Ty::Prim(prim)))
            }
            ExprKind::Bool(b) => Ok((CoreExpr::new(CoreExprKind::ConstBool(*b), Repr::Scalar(ScalarRepr::Bool)), Ty::bool())),
            ExprKind::Char(c) => Ok((CoreExpr::new(CoreExprKind::ConstChar(*c), Repr::Scalar(ScalarRepr::Char)), Ty::Prim(Prim::Char))),
            ExprKind::Unit => Ok((CoreExpr::new(CoreExprKind::Unit, Repr::Unit), Ty::unit())),

            ExprKind::Path(path) if path.is_single() => {
                let name = path.last();
                if let Some((id, ty)) = self.lookup(name) {
                    let repr = self.repr_of(&ty, e.span)?;
                    Ok((CoreExpr::new(CoreExprKind::Local(id), repr), ty))
                } else if self.ctx.variants.contains_key(name) {
                    // An unqualified unit variant brought into scope (`None`).
                    self.variant_ctor(path, &[], expected, e.span)
                } else {
                    err(format!("unknown variable `{}`", name), e.span)
                }
            }

            ExprKind::Unary(op, inner) => {
                let (ci, ty) = self.expr(inner, expected)?;
                let repr = ci.repr.clone();
                Ok((CoreExpr::new(CoreExprKind::Un(*op, Box::new(ci)), repr), ty))
            }

            ExprKind::Binary(op, l, r) => self.binary(*op, l, r, e.span),

            ExprKind::Call(callee, args) => {
                // A variant constructor call (`Result::Err(x)`) needs `expected`
                // to infer type params the payload doesn't pin; intercept it
                // here where `expected` is in scope.
                if let ExprKind::Path(path) = &*callee.kind {
                    let key = path.segments.join("::");
                    if self.ctx.variants.contains_key(&key) || self.ctx.variants.contains_key(path.last()) {
                        return self.variant_ctor(path, args, expected, e.span);
                    }
                }
                self.call(callee, args, e.span)
            }

            ExprKind::If { cond, then_branch, else_branch } => {
                let (cc, ct) = self.expr(cond, Some(&Ty::bool()))?;
                check_assignable(&ct, &Ty::bool(), cond.span)?;
                let (tb, tt) = self.block_expected(then_branch, expected)?;
                // The else branch is checked against the then branch's type (or
                // the outer expectation if the then branch was itself inferred).
                let else_exp = expected.or(Some(&tt));
                let (eb, et) = match else_branch {
                    Some(e) => {
                        let (ce, t) = self.expr(e, else_exp)?;
                        (block_of(ce), t)
                    }
                    None => (CoreBlock { stmts: vec![], tail: None }, Ty::unit()),
                };
                check_assignable(&et, &tt, e.span)?;
                let repr = self.repr_of(&tt, e.span)?;
                Ok((
                    CoreExpr::new(CoreExprKind::If(Box::new(cc), Box::new(tb), Box::new(eb)), repr),
                    tt,
                ))
            }

            ExprKind::Block(b) => {
                let (cb, t) = self.block_expected(b, expected)?;
                let repr = self.repr_of(&t, e.span)?;
                Ok((CoreExpr::new(CoreExprKind::Block(Box::new(cb)), repr), t))
            }

            ExprKind::Return(v) => {
                let cv = match v {
                    Some(e) => {
                        let (ce, t) = self.expr(e, Some(&self.ret_ty.clone()))?;
                        check_assignable(&t, &self.ret_ty.clone(), e.span)?;
                        Some(Box::new(ce))
                    }
                    None => None,
                };
                // `return` has type "never"; represent as unit for the slice.
                Ok((CoreExpr::new(CoreExprKind::Return(cv), Repr::Unit), Ty::unit()))
            }

            ExprKind::While { cond, body } => {
                let (cc, ct) = self.expr(cond, Some(&Ty::bool()))?;
                check_assignable(&ct, &Ty::bool(), cond.span)?;
                let (cb, _) = self.block(body)?;
                // while => loop { if !cond { break } body }
                let break_blk = CoreBlock { stmts: vec![], tail: Some(CoreExpr::new(CoreExprKind::Break(None), Repr::Unit)) };
                let guard = CoreExpr::new(
                    CoreExprKind::If(
                        Box::new(negate(cc)),
                        Box::new(break_blk),
                        Box::new(CoreBlock { stmts: vec![], tail: None }),
                    ),
                    Repr::Unit,
                );
                let mut stmts = vec![CoreStmt::Expr(guard)];
                stmts.extend(cb.stmts);
                if let Some(t) = cb.tail {
                    stmts.push(CoreStmt::Expr(t));
                }
                let loop_body = CoreBlock { stmts, tail: None };
                Ok((CoreExpr::new(CoreExprKind::Loop(Box::new(loop_body)), Repr::Unit), Ty::unit()))
            }

            ExprKind::Loop { body } => {
                let (cb, _) = self.block(body)?;
                Ok((CoreExpr::new(CoreExprKind::Loop(Box::new(cb)), Repr::Unit), Ty::unit()))
            }

            ExprKind::Break(v) => {
                let cv = match v {
                    Some(e) => Some(Box::new(self.expr(e, None)?.0)),
                    None => None,
                };
                Ok((CoreExpr::new(CoreExprKind::Break(cv), Repr::Unit), Ty::unit()))
            }

            ExprKind::Continue => Ok((CoreExpr::new(CoreExprKind::Continue, Repr::Unit), Ty::unit())),

            ExprKind::Assign { target, op, value } => {
                // v0: only assignment to a simple local.
                let ExprKind::Path(p) = &*target.kind else {
                    return err("assignment target must be a variable in v0", e.span);
                };
                if !p.is_single() {
                    return err("assignment target must be a simple variable", e.span);
                }
                let Some((id, ty)) = self.lookup(p.last()) else {
                    return err(format!("unknown variable `{}`", p.last()), e.span);
                };
                let (cv, vt) = self.expr(value, Some(&ty))?;
                check_assignable(&vt, &ty, e.span)?;
                let final_val = match op {
                    None => cv,
                    Some(binop) => {
                        // x op= v  =>  x = x op v
                        let repr = cv.repr.clone();
                        let load = CoreExpr::new(CoreExprKind::Local(id), repr.clone());
                        CoreExpr::new(CoreExprKind::Bin(*binop, Box::new(load), Box::new(cv)), repr)
                    }
                };
                Ok((CoreExpr::new(CoreExprKind::Assign { local: id, value: Box::new(final_val) }, Repr::Unit), Ty::unit()))
            }

            ExprKind::Cast(inner, target) => {
                let (ci, from_ty) = self.expr(inner, None)?;
                let to_ty = lower_type(target, &[], self.ctx).map_err(conv)?;
                let from = ci.repr.clone();
                let to = self.repr_of(&to_ty, e.span)?;
                Ok((
                    CoreExpr::new(CoreExprKind::Cast { value: Box::new(ci), from, to: to.clone() }, to),
                    to_ty,
                ))
            }

            ExprKind::StructLit { path, fields, .. } => self.struct_lit(path, fields, e.span),

            ExprKind::Field { base, field } => self.field_access(base, field, e.span),

            ExprKind::Path(path) if !path.is_single() => {
                // A qualified path that's a unit enum variant: `Color::Red`,
                // `Option::None`.
                self.variant_ctor(path, &[], expected, e.span)
            }

            ExprKind::Match { scrutinee, arms } => self.match_expr(scrutinee, arms, e.span),

            ExprKind::Try(inner) => self.try_op(inner, e.span),

            ExprKind::MethodCall { recv, method, args, span } => {
                self.method_call(recv, method, args, *span)
            }

            other => err(format!("expression not supported in v0 slice: {:?}", disc(other)), e.span),
        }
    }

    /// A `Name { field: val, ... }` literal — struct OR an enum struct-variant.
    fn struct_lit(&mut self, path: &Path, fields: &[FieldInit], span: Span) -> LResult<(CoreExpr, Ty)> {
        let name = path.last();
        let canon = self.ctx.canon(name).unwrap_or_else(|| name.to_string());
        let Some(s) = self.ctx.structs.get(&canon).cloned() else {
            return err(format!("`{}` is not a struct", name), span);
        };
        let StructBody::Named(def_fields) = &s.body else {
            return err("struct literal on a non-record struct", span);
        };
        let gparams: Vec<String> = s.generics.params.iter().map(|p| p.name.clone()).collect();
        // Evaluate fields in declaration order, inferring the struct's type args
        // from the field values.
        let mut targ: HashMap<String, Ty> = HashMap::new();
        let mut cfields = Vec::with_capacity(def_fields.len());
        let mut field_checks = Vec::new();
        for df in def_fields {
            let init = fields.iter().find(|f| f.name == df.name)
                .ok_or_else(|| LowerError { msg: format!("missing field `{}`", df.name), span })?;
            let declared = lower_type(&df.ty, &gparams, self.ctx).map_err(conv)?;
            let val_expr = match &init.value {
                Some(v) => v.clone(),
                None => Expr { kind: Box::new(ExprKind::Path(Path::single(df.name.clone(), span))), span },
            };
            let (cv, vt) = self.expr(&val_expr, hint(&declared))?;
            unify_infer(&declared, &vt, &mut targ);
            field_checks.push((declared, vt));
            cfields.push(cv);
        }
        let mut struct_args = Vec::new();
        for gp in &gparams {
            match targ.get(gp) {
                Some(t) => struct_args.push(t.clone()),
                None => return err(format!("cannot infer `{}` for struct `{}`", gp, name), span),
            }
        }
        let subst: HashMap<String, Ty> = gparams.iter().cloned().zip(struct_args.iter().cloned()).collect();
        for (declared, actual) in &field_checks {
            check_assignable(actual, &apply_subst(declared, &subst), span)?;
        }
        let ty = Ty::Named { name: canon.clone(), args: struct_args };
        let repr = self.repr_of(&ty, span)?;
        let kind = match repr {
            Repr::Ref(lid) => CoreExprKind::New { layout: lid, fields: cfields },
            Repr::Value(vid) => CoreExprKind::MakeValue { value: vid, fields: cfields },
            _ => return err("struct must be a ref or value type", span),
        };
        Ok((CoreExpr::new(kind, repr), ty))
    }

    /// Construct an enum variant. `path` is `Enum::Variant`; `args` are payload
    /// values (empty for a unit variant). `expected` provides the enum's type
    /// args when they can't be inferred from the payload (e.g. `None`).
    fn variant_ctor(&mut self, path: &Path, args: &[Expr], expected: Option<&Ty>, span: Span) -> LResult<(CoreExpr, Ty)> {
        let key = path.segments.join("::");
        let last = path.last();
        let (enum_name, tag) = self.ctx.variants.get(&key)
            .or_else(|| self.ctx.variants.get(last))
            .cloned()
            .ok_or_else(|| LowerError { msg: format!("unknown variant `{}`", key), span })?;
        // Builtin Option/Result: enum_name is "Option"/"Result"; handle by name.
        let gparams: Vec<String> = self.ctx.enums.get(&enum_name)
            .map(|e| e.generics.params.iter().map(|p| p.name.clone()).collect())
            .unwrap_or_else(|| builtin_enum_params(&enum_name));

        // Declared payload types (may mention the enum's generic params).
        let decl_payload = self.variant_decl_payload(&enum_name, tag, &gparams)?;
        if args.len() != decl_payload.len() {
            return err(format!("variant `{}` expects {} args, got {}", last, decl_payload.len(), args.len()), span);
        }

        // Infer the enum's type args from (a) the expected type, (b) the payload.
        let mut targ: HashMap<String, Ty> = HashMap::new();
        if let Some(Ty::Named { name, args: eargs }) = expected {
            if name == &enum_name || name.rsplit("::").next() == enum_name.rsplit("::").next() {
                for (gp, a) in gparams.iter().zip(eargs) {
                    targ.insert(gp.clone(), a.clone());
                }
            }
        }
        let mut cargs = Vec::new();
        let mut actuals = Vec::new();
        for (a, decl) in args.iter().zip(&decl_payload) {
            let (ca, at) = self.expr(a, hint(decl))?;
            unify_infer(decl, &at, &mut targ);
            actuals.push((decl.clone(), at));
            cargs.push(ca);
        }
        // Resolve the enum's full type args.
        let mut enum_args = Vec::new();
        for gp in &gparams {
            match targ.get(gp) {
                Some(t) => enum_args.push(t.clone()),
                None => return err(format!("cannot infer `{}` for enum `{}`", gp, enum_name), span),
            }
        }
        let subst: HashMap<String, Ty> = gparams.iter().cloned().zip(enum_args.iter().cloned()).collect();
        for (decl, actual) in &actuals {
            check_assignable(actual, &apply_subst(decl, &subst), span)?;
        }

        let ty = Ty::Named { name: enum_name.clone(), args: enum_args };
        let repr = self.repr_of(&ty, span)?;
        let Repr::Ref(lid) = repr else {
            return err("value enums not yet supported in construction (pending)", span);
        };
        Ok((CoreExpr::new(CoreExprKind::MakeVariant { layout: lid, tag, fields: cargs }, repr), ty))
    }

    /// The declared payload types of `enum::variant[tag]`, with the enum's
    /// generic params left as `Ty::Var`. Handles builtin Option/Result.
    fn variant_decl_payload(&self, enum_name: &str, tag: u32, gparams: &[String]) -> LResult<Vec<Ty>> {
        if let Some(e) = self.ctx.enums.get(enum_name) {
            let variant = &e.variants[tag as usize];
            return match &variant.payload {
                VariantPayload::None => Ok(vec![]),
                VariantPayload::Tuple(tys) => tys.iter().map(|t| lower_type(t, gparams, self.ctx)).collect::<Result<_, _>>().map_err(conv),
                VariantPayload::Named(fs) => fs.iter().map(|f| lower_type(&f.ty, gparams, self.ctx)).collect::<Result<_, _>>().map_err(conv),
            };
        }
        // Builtin Option/Result.
        Ok(match (enum_name, tag) {
            ("Option", 0) => vec![],                       // None
            ("Option", 1) => vec![Ty::Var("T".into())],    // Some(T)
            ("Result", 0) => vec![Ty::Var("T".into())],    // Ok(T)
            ("Result", 1) => vec![Ty::Var("E".into())],    // Err(E)
            _ => return Err(LowerError { msg: format!("unknown variant of `{}`", enum_name), span: Span::new(0, 0) }),
        })
    }

    fn field_access(&mut self, base: &Expr, field: &FieldAccess, span: Span) -> LResult<(CoreExpr, Ty)> {
        let (cbase, bty) = self.expr(base, None)?;
        let FieldAccess::Named(fname) = field else {
            return err("tuple field access not yet supported in v0", span);
        };
        let Ty::Named { name, args: struct_args } = &bty else {
            return err("field access on a non-struct", span);
        };
        let s = self.ctx.structs.get(name).cloned()
            .ok_or_else(|| LowerError { msg: format!("`{}` is not a struct", name), span })?;
        let StructBody::Named(def_fields) = &s.body else {
            return err("field access on a non-record struct", span);
        };
        let idx = def_fields.iter().position(|f| &f.name == fname)
            .ok_or_else(|| LowerError { msg: format!("no field `{}` on `{}`", fname, name), span })?;
        let gparams: Vec<String> = s.generics.params.iter().map(|p| p.name.clone()).collect();
        let subst: HashMap<String, Ty> = gparams.iter().cloned().zip(struct_args.iter().cloned()).collect();
        let raw_fty = lower_type(&def_fields[idx].ty, &gparams, self.ctx).map_err(conv)?;
        let fty = apply_subst(&raw_fty, &subst);
        // The FieldLoc is in the base layout's field_map.
        let loc = match &cbase.repr {
            Repr::Ref(lid) => self.reg.layouts[*lid as usize].field_map[idx],
            Repr::Value(_vid) => return err("value-type field access not yet supported in v0", span),
            _ => return err("field access on non-aggregate", span),
        };
        let frepr = self.repr_of(&fty, span)?;
        Ok((CoreExpr::new(CoreExprKind::Field { base: Box::new(cbase), loc }, frepr), fty))
    }

    fn match_expr(&mut self, scrutinee: &Expr, arms: &[MatchArm], span: Span) -> LResult<(CoreExpr, Ty)> {
        let (cscrut, sty) = self.expr(scrutinee, None)?;
        let Ty::Named { name: enum_name, args: enum_args } = &sty else {
            return err("match on a non-enum is not yet supported in v0", span);
        };
        let enum_name = enum_name.clone();
        let enum_args = enum_args.clone();
        // Variant payload types, with the enum's concrete type args substituted.
        let variants: Vec<(u32, Vec<Ty>)> = self.enum_variants(&enum_name, &enum_args)?;
        let mut carms = Vec::new();
        let mut result_ty: Option<Ty> = None;
        for arm in arms {
            self.push_scope();
            let (tag, binds) = match &arm.pattern.kind {
                PatternKind::Wildcard => {
                    // wildcard: treat as a catch-all → lower as the remaining
                    // tags is complex; v0 requires explicit variant arms but
                    // allows a trailing `_`. Encode as tag = u32::MAX sentinel.
                    (u32::MAX, vec![])
                }
                PatternKind::Variant { path, payload } => {
                    let vlast = path.last();
                    let (tag, ptys) = self.variant_in(&enum_name, vlast, &variants, arm.pattern.span)?;
                    if payload.len() != ptys.len() {
                        return err(format!("variant `{}` binds {} fields, pattern has {}", vlast, ptys.len(), payload.len()), arm.pattern.span);
                    }
                    let mut binds = Vec::new();
                    for (p, pty) in payload.iter().zip(&ptys) {
                        let bname = match &p.kind {
                            PatternKind::Binding { name, .. } => name.clone(),
                            PatternKind::Wildcard => "_".to_string(),
                            _ => return err("nested patterns not yet supported in v0", p.span),
                        };
                        let prepr = self.repr_of(pty, p.span)?;
                        let id = self.fresh_local(prepr, pty.clone());
                        self.bind(&bname, id, pty.clone());
                        binds.push(id);
                    }
                    (tag, binds)
                }
                _ => return err("only variant / wildcard patterns supported in match (v0)", arm.pattern.span),
            };
            if arm.guard.is_some() {
                return err("match guards not yet supported in v0", arm.span);
            }
            let (body, bty) = self.expr(&arm.body, result_ty.as_ref())?;
            match &result_ty {
                Some(rt) => check_assignable(&bty, rt, arm.span)?,
                None => result_ty = Some(bty),
            }
            carms.push(CoreArm { tag, binds, body });
            self.pop_scope();
        }
        let rty = result_ty.unwrap_or_else(Ty::unit);
        let repr = self.repr_of(&rty, span)?;
        Ok((CoreExpr::new(CoreExprKind::Match { scrutinee: Box::new(cscrut), arms: carms }, repr), rty))
    }

    /// Resolve and lower `recv.method(args)` to a concrete monomorphic call,
    /// with `recv` passed as the leading `self` argument.
    fn method_call(&mut self, recv: &Expr, method: &str, args: &[Expr], span: Span) -> LResult<(CoreExpr, Ty)> {
        let (crecv, recv_ty) = self.expr(recv, None)?;
        let base = match &recv_ty {
            Ty::Named { name, .. } => name.rsplit("::").next().unwrap().to_string(),
            Ty::Prim(p) => format!("{:?}", p).to_lowercase(),
            _ => return err("method call on an unsupported receiver type", span),
        };
        // Look up the method (try the exact base, then the canonical struct/enum
        // name's last segment).
        let entry = self.ctx.methods.get(&(base.clone(), method.to_string())).cloned()
            .or_else(|| {
                // primitives: i64 etc are keyed by their literal name in impls.
                self.ctx.methods.get(&(prim_impl_key(&recv_ty), method.to_string())).cloned()
            })
            .ok_or_else(|| LowerError { msg: format!("no method `{}` on `{}`", method, base), span })?;

        // Infer the impl's generic params + Self by unifying the impl's written
        // self type against the receiver's concrete type.
        let impl_self = lower_type(&entry.self_ty, &entry.impl_generics, self.ctx).map_err(conv)?;
        let mut isubst: HashMap<String, Ty> = HashMap::new();
        unify_infer(&impl_self, &recv_ty, &mut isubst);
        // `Self` resolves to the concrete receiver type.
        isubst.insert("Self".to_string(), recv_ty.clone());

        let m = entry.method.clone();
        // Method-level generics inferred from args.
        let mgparams: Vec<String> = m.generics.params.iter().map(|p| p.name.clone()).collect();
        if args.len() != m.params.len() {
            return err(format!("method `{}` expects {} args, got {}", method, m.params.len(), args.len()), span);
        }
        let mut cargs = vec![crecv];
        let mut checks = Vec::new();
        for (a, p) in args.iter().zip(&m.params) {
            // declared param type may mention impl generics, Self, or method generics.
            let mut scope_params = entry.impl_generics.clone();
            scope_params.extend(mgparams.clone());
            scope_params.push("Self".to_string());
            let declared = lower_type(&p.ty, &scope_params, self.ctx).map_err(conv)?;
            let pre = apply_subst(&declared, &isubst);
            let (ca, at) = self.expr(a, hint(&pre))?;
            unify_infer(&pre, &at, &mut isubst);
            checks.push((declared.clone(), at));
            cargs.push(ca);
        }
        // Re-apply now that all type vars are known.
        for (declared, actual) in &checks {
            check_assignable(actual, &apply_subst(declared, &isubst), span)?;
        }

        // Build the full substitution (impl generics + method generics + Self).
        let subst = isubst.clone();
        // Mangle: <Self-type>::method$<args>.
        let self_mangle = ty_mangle(&recv_ty);
        let extra: Vec<Ty> = mgparams.iter().filter_map(|g| subst.get(g).cloned()).collect();
        let mangled = mangle(&format!("{}.{}", self_mangle, method), &extra);

        let ret_ty = match &m.ret {
            Some(t) => {
                let mut scope_params = entry.impl_generics.clone();
                scope_params.extend(mgparams.clone());
                scope_params.push("Self".to_string());
                let raw = lower_type(t, &scope_params, self.ctx).map_err(conv)?;
                apply_subst(&raw, &subst)
            }
            None => Ty::unit(),
        };

        let job = Job { f: m, subst, mangled, self_ty: Some(recv_ty) };
        let fid = self.mono.intern(job, self.count);
        let repr = self.repr_of(&ret_ty, span)?;
        Ok((CoreExpr::new(CoreExprKind::Call(fid, cargs), repr), ret_ty))
    }

    /// The `?` operator. `inner` evaluates to a `Result<T,E>` or `Option<T>`;
    /// `Ok(v)`/`Some(v)` yields `v`, `Err(e)`/`None` early-returns the value
    /// re-wrapped in the enclosing function's return type.
    fn try_op(&mut self, inner: &Expr, span: Span) -> LResult<(CoreExpr, Ty)> {
        let (cinner, ity) = self.expr(inner, None)?;
        let Ty::Named { name, args } = &ity else {
            return err("`?` operand must be a Result or Option", span);
        };
        let enum_name = name.clone();
        let kind = enum_name.rsplit("::").next().unwrap();
        let (ok_tag, err_tag) = match kind {
            "Result" | "Option" => (0u32, 1u32),
            _ => return err("`?` operand must be a Result or Option", span),
        };
        // Determine ok-payload type from the scrutinee's args.
        let variants = self.enum_variants(&enum_name, args)?;
        let ok_payload = variants.iter().find(|(t, _)| *t == ok_tag).map(|(_, p)| p.clone()).unwrap_or_default();
        let ok_ty = ok_payload.first().cloned().unwrap_or_else(Ty::unit);
        let ok_repr = self.repr_of(&ok_ty, span)?;

        // The enclosing fn's return type must be the same enum (so we can build
        // the Err/None to early-return). Re-wrap the err payload as the fn ret.
        let ret_ty = self.ret_ty.clone();
        let Ty::Named { name: rname, args: rargs } = &ret_ty else {
            return err("`?` requires the function to return a Result/Option", span);
        };
        if rname.rsplit("::").next() != Some(kind) {
            return err("`?` return-type mismatch with the function", span);
        }
        let ret_repr = self.repr_of(&ret_ty, span)?;
        let Repr::Ref(ret_lid) = ret_repr else {
            return err("`?` on a value-enum return is not supported", span);
        };

        // Ok arm: bind payload to a local; body = that local.
        let ok_local = self.fresh_local(ok_repr.clone(), ok_ty.clone());
        let ok_body = CoreExpr::new(CoreExprKind::Local(ok_local), ok_repr.clone());

        // Err arm: bind the err payload, rebuild Err(e)/None for the fn ret, return.
        let err_payload = variants.iter().find(|(t, _)| *t == err_tag).map(|(_, p)| p.clone()).unwrap_or_default();
        let (err_binds, err_fields) = if err_payload.is_empty() {
            (vec![], vec![]) // None
        } else {
            let ety = err_payload[0].clone();
            let erepr = self.repr_of(&ety, span)?;
            let elocal = self.fresh_local(erepr.clone(), ety);
            (vec![elocal], vec![CoreExpr::new(CoreExprKind::Local(elocal), erepr)])
        };
        // Build the return value: MakeVariant of the fn's return enum, err_tag.
        let _ = rargs;
        let rebuilt = CoreExpr::new(
            CoreExprKind::MakeVariant { layout: ret_lid, tag: err_tag, fields: err_fields },
            Repr::Ref(ret_lid),
        );
        let err_body = CoreExpr::new(CoreExprKind::Return(Some(Box::new(rebuilt))), Repr::Unit);

        let arms = vec![
            CoreArm { tag: ok_tag, binds: vec![ok_local], body: ok_body },
            CoreArm { tag: err_tag, binds: err_binds, body: err_body },
        ];
        let m = CoreExpr::new(
            CoreExprKind::Match { scrutinee: Box::new(cinner), arms },
            ok_repr,
        );
        Ok((m, ok_ty))
    }

    /// All (tag, payload-types) of an enum, with the enum's concrete type args
    /// substituted in (so a `match` on `Option<i64>` sees `Some(i64)`).
    fn enum_variants(&mut self, enum_name: &str, enum_args: &[Ty]) -> LResult<Vec<(u32, Vec<Ty>)>> {
        if let Some(e) = self.ctx.enums.get(enum_name).cloned() {
            let gparams: Vec<String> = e.generics.params.iter().map(|p| p.name.clone()).collect();
            let subst: HashMap<String, Ty> = gparams.iter().cloned().zip(enum_args.iter().cloned()).collect();
            let mut out = Vec::new();
            for (i, v) in e.variants.iter().enumerate() {
                let tys: Vec<Ty> = match &v.payload {
                    VariantPayload::None => vec![],
                    VariantPayload::Tuple(t) => t.iter().map(|t| {
                        lower_type(t, &gparams, self.ctx).map(|ty| apply_subst(&ty, &subst))
                    }).collect::<Result<_, _>>().map_err(conv)?,
                    VariantPayload::Named(f) => f.iter().map(|f| {
                        lower_type(&f.ty, &gparams, self.ctx).map(|ty| apply_subst(&ty, &subst))
                    }).collect::<Result<_, _>>().map_err(conv)?,
                };
                out.push((i as u32, tys));
            }
            Ok(out)
        } else {
            err(format!("unknown enum `{}` in match", enum_name), Span::new(0, 0))
        }
    }

    fn variant_in(&self, _enum_name: &str, vlast: &str, variants: &[(u32, Vec<Ty>)], span: Span) -> LResult<(u32, Vec<Ty>)> {
        if let Some((_, tag)) = self.ctx.variants.get(vlast).cloned() {
            if let Some((t, tys)) = variants.iter().find(|(i, _)| *i == tag) {
                return Ok((*t, tys.clone()));
            }
        }
        err(format!("unknown variant `{}`", vlast), span)
    }

    fn binary(&mut self, op: BinOp, l: &Expr, r: &Expr, span: Span) -> LResult<(CoreExpr, Ty)> {
        use BinOp::*;
        let (cl, lt) = self.expr(l, None)?;
        // Right side expects the left's type for arithmetic/compare.
        let (cr, rt) = self.expr(r, Some(&lt))?;
        match op {
            Add | Sub | Mul | Div | Rem | BitAnd | BitOr | BitXor | Shl | Shr => {
                check_assignable(&rt, &lt, span)?;
                if !is_numeric(&lt) {
                    return err("arithmetic on non-numeric type", span);
                }
                let repr = cl.repr.clone();
                Ok((CoreExpr::new(CoreExprKind::Bin(op, Box::new(cl), Box::new(cr)), repr), lt))
            }
            Eq | Ne | Lt | Le | Gt | Ge => {
                check_assignable(&rt, &lt, span)?;
                Ok((
                    CoreExpr::new(
                        CoreExprKind::Bin(op, Box::new(cl), Box::new(cr)),
                        Repr::Scalar(ScalarRepr::Bool),
                    ),
                    Ty::bool(),
                ))
            }
            And | Or => {
                check_assignable(&lt, &Ty::bool(), span)?;
                check_assignable(&rt, &Ty::bool(), span)?;
                Ok((
                    CoreExpr::new(
                        CoreExprKind::Bin(op, Box::new(cl), Box::new(cr)),
                        Repr::Scalar(ScalarRepr::Bool),
                    ),
                    Ty::bool(),
                ))
            }
        }
    }

    fn call(&mut self, callee: &Expr, args: &[Expr], span: Span) -> LResult<(CoreExpr, Ty)> {
        // v0 slice: only direct calls to named, non-generic functions.
        let ExprKind::Path(path) = &*callee.kind else {
            return err("only direct function calls supported in v0 slice", span);
        };
        let name = path.last();
        // Float intrinsics — only when the name isn't shadowed by a user fn.
        if let Some(intr) = float_intrinsic(name) {
            if !self.ctx.fns.keys().any(|k| k.rsplit("::").next().unwrap() == name) {
                if args.len() != 1 {
                    return err(format!("`{}` takes 1 argument", name), span);
                }
                let (ca, at) = self.expr(&args[0], None)?;
                if !matches!(at, Ty::Prim(Prim::F32 | Prim::F64)) {
                    return err(format!("`{}` requires a float argument", name), span);
                }
                let repr = ca.repr.clone();
                return Ok((CoreExpr::new(CoreExprKind::FloatIntrinsic(intr, Box::new(ca)), repr), at));
            }
        }
        // An enum variant constructor (`Option::Some(x)`, `Ok(v)`) is a call
        // syntactically; route it to variant construction.
        let key = path.segments.join("::");
        if self.ctx.variants.contains_key(&key) || self.ctx.variants.contains_key(name) {
            return self.variant_ctor(path, args, None, span);
        }
        let fq = self.ctx.fns.keys().find(|k| k.rsplit("::").next().unwrap() == name).cloned();
        let Some(fq) = fq else {
            return err(format!("unknown function `{}`", name), span);
        };
        let f = self.ctx.fns[&fq].clone();
        if args.len() != f.params.len() {
            return err(format!("`{}` expects {} args, got {}", name, f.params.len(), args.len()), span);
        }

        // Lower the argument expressions, learning each one's concrete type.
        // For a generic callee we infer the type-parameter assignment by
        // unifying declared param types (which mention `T`) against the actual
        // argument types.
        let gparams: Vec<String> = f.generics.params.iter().map(|p| p.name.clone()).collect();
        let mut targ: HashMap<String, Ty> = HashMap::new();
        let mut cargs = Vec::new();
        let mut arg_tys = Vec::new();
        for (a, p) in args.iter().zip(&f.params) {
            // The expected type may still contain `T`; lower with the callee's
            // generic params in scope (no subst yet — we're inferring it).
            let declared = lower_type(&p.ty, &gparams, self.ctx).map_err(conv)?;
            let (ca, at) = self.expr(a, hint(&declared))?;
            unify_infer(&declared, &at, &mut targ);
            arg_tys.push((declared, at.clone()));
            cargs.push(ca);
        }

        // Resolve the instantiation's concrete type args (in declared order).
        let mut inst_args = Vec::new();
        for gp in &gparams {
            match targ.get(gp) {
                Some(t) => inst_args.push(t.clone()),
                None => return err(format!("cannot infer type parameter `{}` for `{}`", gp, name), span),
            }
        }

        // Check each argument against its substituted declared type.
        let subst: HashMap<String, Ty> = gparams.iter().cloned().zip(inst_args.iter().cloned()).collect();
        for (declared, actual) in &arg_tys {
            check_assignable(actual, &apply_subst(declared, &subst), span)?;
        }

        let mangled = mangle(&fq, &inst_args);
        let job = Job { f: f.clone(), subst: subst.clone(), mangled, self_ty: None };
        let fid = self.mono.intern(job, self.count);

        let ret_ty = match &f.ret {
            Some(t) => {
                let raw = lower_type(t, &gparams, self.ctx).map_err(conv)?;
                apply_subst(&raw, &subst)
            }
            None => Ty::unit(),
        };
        let repr = self.repr_of(&ret_ty, span)?;
        Ok((CoreExpr::new(CoreExprKind::Call(fid, cargs), repr), ret_ty))
    }
}

/// A `Some(ty)` hint only when the declared type is already concrete (no free
/// type variables) — otherwise literal defaulting shouldn't be guided by `T`.
fn hint(declared: &Ty) -> Option<&Ty> {
    if has_var(declared) { None } else { Some(declared) }
}

fn has_var(t: &Ty) -> bool {
    match t {
        Ty::Var(_) => true,
        Ty::Named { args, .. } | Ty::Tuple(args) => args.iter().any(has_var),
        Ty::Array(e, _) => has_var(e),
        Ty::Fn { params, ret } => params.iter().any(has_var) || has_var(ret),
        Ty::Prim(_) | Ty::Infer(_) => false,
    }
}

/// Unify a declared type (which may contain `Ty::Var`) against a concrete
/// actual type, recording type-variable assignments into `out`.
fn unify_infer(declared: &Ty, actual: &Ty, out: &mut HashMap<String, Ty>) {
    match (declared, actual) {
        (Ty::Var(v), _) => { out.entry(v.clone()).or_insert_with(|| actual.clone()); }
        (Ty::Named { args: da, .. }, Ty::Named { args: aa, .. }) => {
            for (d, a) in da.iter().zip(aa) { unify_infer(d, a, out); }
        }
        (Ty::Tuple(de), Ty::Tuple(ae)) => {
            for (d, a) in de.iter().zip(ae) { unify_infer(d, a, out); }
        }
        (Ty::Array(de, _), Ty::Array(ae, _)) => unify_infer(de, ae, out),
        (Ty::Fn { params: dp, ret: dr }, Ty::Fn { params: ap, ret: ar }) => {
            for (d, a) in dp.iter().zip(ap) { unify_infer(d, a, out); }
            unify_infer(dr, ar, out);
        }
        _ => {}
    }
}

// ---- helpers ---------------------------------------------------------------

fn block_of(e: CoreExpr) -> CoreBlock {
    CoreBlock { stmts: vec![], tail: Some(e) }
}

/// Generic parameter names of the built-in `Option`/`Result` enums (used only
/// when they aren't user-declared; example programs declare their own).
fn builtin_enum_params(name: &str) -> Vec<String> {
    match name {
        "Option" => vec!["T".into()],
        "Result" => vec!["T".into(), "E".into()],
        _ => vec![],
    }
}

fn negate(e: CoreExpr) -> CoreExpr {
    let repr = e.repr.clone();
    CoreExpr::new(CoreExprKind::Un(UnOp::Not, Box::new(e)), repr)
}

fn scalar_of(p: Prim) -> Option<ScalarRepr> {
    Some(match p {
        Prim::I8 => ScalarRepr::I8, Prim::I16 => ScalarRepr::I16,
        Prim::I32 => ScalarRepr::I32, Prim::I64 => ScalarRepr::I64,
        Prim::U8 => ScalarRepr::U8, Prim::U16 => ScalarRepr::U16,
        Prim::U32 => ScalarRepr::U32, Prim::U64 => ScalarRepr::U64,
        Prim::F32 => ScalarRepr::F32, Prim::F64 => ScalarRepr::F64,
        Prim::Bool => ScalarRepr::Bool, Prim::Char => ScalarRepr::Char,
        Prim::Str | Prim::Unit => return None,
    })
}

fn is_int_prim(p: Prim) -> bool {
    matches!(p, Prim::I8 | Prim::I16 | Prim::I32 | Prim::I64 | Prim::U8 | Prim::U16 | Prim::U32 | Prim::U64)
}
fn is_numeric(ty: &Ty) -> bool {
    matches!(ty, Ty::Prim(p) if is_int_prim(*p) || matches!(p, Prim::F32 | Prim::F64))
}

fn check_assignable(actual: &Ty, expected: &Ty, span: Span) -> LResult<()> {
    if actual == expected || actual.is_unit() && expected.is_unit() {
        Ok(())
    } else {
        err(format!("type mismatch: expected {:?}, found {:?}", expected, actual), span)
    }
}

fn disc(e: &ExprKind) -> &'static str {
    match e {
        ExprKind::MethodCall { .. } => "method-call",
        ExprKind::Field { .. } => "field",
        ExprKind::Index { .. } => "index",
        ExprKind::StructLit { .. } => "struct-literal",
        ExprKind::Match { .. } => "match",
        ExprKind::While { .. } => "while",
        ExprKind::Loop { .. } => "loop",
        ExprKind::For { .. } => "for",
        ExprKind::Closure { .. } => "closure",
        ExprKind::Try(_) => "try",
        _ => "expr",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexer::lex;
    use crate::parser::parse_module;
    use crate::resolve::resolve_module;

    fn lower(src: &str) -> Result<CoreProgram, LowerError> {
        let m = parse_module(&lex(src).unwrap()).unwrap();
        let r = resolve_module(m).unwrap();
        lower_program(&r.globals)
    }

    #[test]
    fn lowers_fib() {
        let prog = lower(include_str!("../examples/fib.gcr")).unwrap();
        assert!(prog.entry.is_some());
        // fib + main
        assert_eq!(prog.funcs.len(), 2);
        let main = &prog.funcs[prog.entry.unwrap() as usize];
        assert_eq!(main.ret, Repr::Scalar(ScalarRepr::I64));
    }

    #[test]
    fn arithmetic_and_if() {
        let prog = lower("fn main() -> i64 { let x = 3; if x < 5 { x * 2 } else { 0 } }").unwrap();
        assert_eq!(prog.funcs.len(), 1);
    }

    #[test]
    fn type_mismatch_caught() {
        let e = lower("fn main() -> i64 { true }");
        assert!(e.is_err());
    }

    #[test]
    fn unsigned_literal() {
        let prog = lower("fn main() -> u32 { let x = 5u32; x + 1 }").unwrap();
        let main = &prog.funcs[0];
        assert_eq!(main.ret, Repr::Scalar(ScalarRepr::U32));
    }
}
