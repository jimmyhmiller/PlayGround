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
    /// Fully-formed lifted closure functions, keyed by their FuncId, to be
    /// installed into `prog.funcs` by the driver.
    closures: Vec<(FuncId, CoreFn)>,
}

impl Mono {
    fn new() -> Self {
        Mono { ids: HashMap::new(), worklist: Vec::new(), closures: Vec::new() }
    }
    /// Allocate a fresh FuncId for a lifted closure function (not name-keyed).
    fn fresh_closure_id(&mut self, count: &mut u32) -> FuncId {
        let id = *count;
        *count += 1;
        id
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
    // Lifted closure functions accumulate in `mono.closures` as bodies are
    // lowered; install them too.
    loop {
        while let Some((id, job)) = mono.worklist.pop() {
            let lowered = lower_fn(&job, &ctx, &mut reg, &mut mono, &mut count)?;
            install_fn(&mut prog, id, lowered);
        }
        if let Some((id, cf)) = mono.closures.pop() {
            install_fn(&mut prog, id, cf);
        } else {
            break;
        }
    }

    prog.layouts = reg.layouts;
    prog.values = reg.values;
    Ok(prog)
}

fn install_fn(prog: &mut CoreProgram, id: FuncId, f: CoreFn) {
    while prog.funcs.len() <= id as usize {
        prog.funcs.push(placeholder_fn());
    }
    prog.funcs[id as usize] = f;
}

fn placeholder_fn() -> CoreFn {
    CoreFn { name: String::new(), params: vec![], ret: Repr::Unit, locals: vec![], body: CoreBlock { stmts: vec![], tail: None }, closure_captures: vec![] }
}

/// Collect free variable names referenced in `e`: single-segment path uses that
/// are not in `params` (the closure's own params) and not in `bound` (names
/// bound by inner `let`/closures/match arms). Order-preserving, may repeat.
fn collect_free_vars(e: &Expr, params: &[&str], bound: &mut Vec<String>, out: &mut Vec<String>) {
    match &*e.kind {
        ExprKind::Path(p) if p.is_single() => {
            let n = p.last();
            if !params.contains(&n) && !bound.iter().any(|b| b == n) {
                out.push(n.to_string());
            }
        }
        ExprKind::Path(_) | ExprKind::Int(..) | ExprKind::Float(..) | ExprKind::Str(_)
        | ExprKind::Char(_) | ExprKind::Bool(_) | ExprKind::Unit | ExprKind::Continue => {}
        ExprKind::Call(c, args) => {
            collect_free_vars(c, params, bound, out);
            for a in args { collect_free_vars(a, params, bound, out); }
        }
        ExprKind::MethodCall { recv, args, .. } => {
            collect_free_vars(recv, params, bound, out);
            for a in args { collect_free_vars(a, params, bound, out); }
        }
        ExprKind::Field { base, .. } => collect_free_vars(base, params, bound, out),
        ExprKind::Index { base, index } => { collect_free_vars(base, params, bound, out); collect_free_vars(index, params, bound, out); }
        ExprKind::Unary(_, x) | ExprKind::Cast(x, _) | ExprKind::Try(x) => collect_free_vars(x, params, bound, out),
        ExprKind::Binary(_, l, r) => { collect_free_vars(l, params, bound, out); collect_free_vars(r, params, bound, out); }
        ExprKind::Assign { target, value, .. } => { collect_free_vars(target, params, bound, out); collect_free_vars(value, params, bound, out); }
        ExprKind::StructLit { fields, .. } => {
            for f in fields { if let Some(v) = &f.value { collect_free_vars(v, params, bound, out); } }
        }
        ExprKind::Tuple(es) => for x in es { collect_free_vars(x, params, bound, out); },
        ExprKind::Array(a) => match a {
            ArrayLit::Elems(es) => for x in es { collect_free_vars(x, params, bound, out); },
            ArrayLit::Repeat(v, n) => { collect_free_vars(v, params, bound, out); collect_free_vars(n, params, bound, out); }
        },
        ExprKind::Block(b) => collect_free_in_block(b, params, bound, out),
        ExprKind::If { cond, then_branch, else_branch } => {
            collect_free_vars(cond, params, bound, out);
            collect_free_in_block(then_branch, params, bound, out);
            if let Some(e) = else_branch { collect_free_vars(e, params, bound, out); }
        }
        ExprKind::Match { scrutinee, arms } => {
            collect_free_vars(scrutinee, params, bound, out);
            for arm in arms {
                let mark = bound.len();
                bind_pattern_names(&arm.pattern, bound);
                if let Some(g) = &arm.guard { collect_free_vars(g, params, bound, out); }
                collect_free_vars(&arm.body, params, bound, out);
                bound.truncate(mark);
            }
        }
        ExprKind::While { cond, body } => { collect_free_vars(cond, params, bound, out); collect_free_in_block(body, params, bound, out); }
        ExprKind::Loop { body } => collect_free_in_block(body, params, bound, out),
        ExprKind::For { pat, iter, body } => {
            collect_free_vars(iter, params, bound, out);
            let mark = bound.len();
            bind_pattern_names(pat, bound);
            collect_free_in_block(body, params, bound, out);
            bound.truncate(mark);
        }
        ExprKind::Closure { params: cps, body, .. } => {
            // Nested closure: its own params shadow; treat as bound.
            let mark = bound.len();
            for cp in cps { bound.push(cp.name.clone()); }
            collect_free_vars(body, params, bound, out);
            bound.truncate(mark);
        }
        ExprKind::Return(v) | ExprKind::Break(v) => { if let Some(x) = v { collect_free_vars(x, params, bound, out); } }
        ExprKind::Range { lo, hi, .. } => {
            if let Some(l) = lo { collect_free_vars(l, params, bound, out); }
            if let Some(h) = hi { collect_free_vars(h, params, bound, out); }
        }
    }
}

fn collect_free_in_block(b: &Block, params: &[&str], bound: &mut Vec<String>, out: &mut Vec<String>) {
    let mark = bound.len();
    for s in &b.stmts {
        match s {
            Stmt::Let { pattern, init, .. } => {
                if let Some(e) = init { collect_free_vars(e, params, bound, out); }
                bind_pattern_names(pattern, bound);
            }
            Stmt::Expr(e) => collect_free_vars(e, params, bound, out),
            Stmt::Item(_) => {}
        }
    }
    if let Some(t) = &b.tail { collect_free_vars(t, params, bound, out); }
    bound.truncate(mark);
}

fn bind_pattern_names(p: &Pattern, bound: &mut Vec<String>) {
    match &p.kind {
        PatternKind::Binding { name, .. } => bound.push(name.clone()),
        PatternKind::Variant { payload, .. } => for sp in payload { bind_pattern_names(sp, bound); },
        PatternKind::Struct { fields, .. } => for (_, sp) in fields { bind_pattern_names(sp, bound); },
        PatternKind::Tuple(ps) => for sp in ps { bind_pattern_names(sp, bound); },
        PatternKind::Wildcard | PatternKind::Literal(_) => {}
    }
}

/// A human-readable rendering of a type for diagnostics.
fn ty_display(ty: &Ty) -> String {
    match ty {
        Ty::Prim(p) => crate::types::prim_name(*p).to_string(),
        Ty::Named { name, args } => {
            let base = name.rsplit("::").next().unwrap();
            if args.is_empty() { base.to_string() }
            else { format!("{}<{}>", base, args.iter().map(ty_display).collect::<Vec<_>>().join(", ")) }
        }
        Ty::Var(v) => v.clone(),
        Ty::Array(e, n) => format!("[{}; {}]", ty_display(e), n),
        Ty::Tuple(es) => format!("({})", es.iter().map(ty_display).collect::<Vec<_>>().join(", ")),
        Ty::Fn { params, ret } => format!("fn({}) -> {}", params.iter().map(ty_display).collect::<Vec<_>>().join(", "), ty_display(ret)),
        Ty::Infer(_) => "_".to_string(),
    }
}

fn array_elem_ty(ty: &Ty) -> Option<Ty> {
    match ty {
        Ty::Named { name, args } if name.rsplit("::").next() == Some("Array") && !args.is_empty() => Some(args[0].clone()),
        _ => None,
    }
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
    // Point a body/return type mismatch at the tail expression (or the function
    // body if there's no tail), so the caret lands on the offending value.
    let body_err_span = f.body.tail.as_ref().map(|t| t.span).unwrap_or(f.body.span);
    check_assignable(&body_ty, &ret_ty, body_err_span)?;
    let ret = lo.repr_of(&ret_ty, f.span)?;

    Ok(CoreFn { name: job.mangled.clone(), params, ret, locals: lo.locals, body, closure_captures: vec![] })
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
                                Some(t) => Some(ground_type(t, &self.subst.clone(), self.ctx)?),
                                None => None,
                            };
                            let (ce, at) = self.expr(e, expected.as_ref())?;
                            // A `let x: T = e` whose `e` doesn't match `T` is an
                            // error pointing at `e` (not at later uses of `x`).
                            if let Some(exp) = &expected {
                                check_assignable(&at, exp, e.span)?;
                            }
                            (ce, at)
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
            ExprKind::Str(s) => {
                let repr = self.repr_of(&Ty::Prim(Prim::Str), e.span)?;
                let Repr::Ref(lid) = repr else { return err("String repr is not a reference", e.span) };
                Ok((CoreExpr::new(CoreExprKind::ConstStr(s.clone()), Repr::Ref(lid)), Ty::Prim(Prim::Str)))
            }
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
                self.call(callee, args, expected, e.span)
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

            ExprKind::For { pat, iter, body } => {
                // Only integer ranges `lo..hi` / `lo..=hi` in v0. Desugar to a
                // counter `while` loop and re-lower.
                let ExprKind::Range { lo: Some(lo), hi: Some(hi), inclusive } = &*iter.kind else {
                    return err("`for` only supports integer ranges `lo..hi` in v0", iter.span);
                };
                let PatternKind::Binding { name, .. } = &pat.kind else {
                    return err("`for` pattern must be a simple binding in v0", pat.span);
                };
                let sp = e.span;
                let var = || Expr { kind: Box::new(ExprKind::Path(Path::single(name.clone(), sp))), span: sp };
                // cond: x < hi   (or x <= hi for inclusive)
                let cmp = if *inclusive { BinOp::Le } else { BinOp::Lt };
                let cond = Expr { kind: Box::new(ExprKind::Binary(cmp, var(), (*hi).clone())), span: sp };
                // body + `x = x + 1;`
                let incr = Expr {
                    kind: Box::new(ExprKind::Assign {
                        target: var(),
                        op: None,
                        value: Expr {
                            kind: Box::new(ExprKind::Binary(BinOp::Add, var(),
                                Expr { kind: Box::new(ExprKind::Int(1, crate::lexer::NumSuffix::None)), span: sp })),
                            span: sp,
                        },
                    }),
                    span: sp,
                };
                let mut wbody = body.clone();
                wbody.stmts.push(Stmt::Expr(incr));
                let while_expr = Expr { kind: Box::new(ExprKind::While { cond, body: wbody }), span: sp };
                // { let mut x = lo; while ... { ... } }
                let outer = Block {
                    stmts: vec![
                        Stmt::Let {
                            pattern: Pattern { kind: PatternKind::Binding { is_mut: true, name: name.clone() }, span: sp },
                            ty: None,
                            init: Some((*lo).clone()),
                            span: sp,
                        },
                        Stmt::Expr(while_expr),
                    ],
                    tail: None,
                    span: sp,
                };
                self.expr(&Expr { kind: Box::new(ExprKind::Block(outer)), span: sp }, None)
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
                // `a[i] = v` desugars to `array_set(a, i, v)` (compound forms
                // expand to `a[i] = a[i] op v`).
                if let ExprKind::Index { base, index } = &*target.kind {
                    let val_expr = match op {
                        None => value.clone(),
                        Some(binop) => Expr {
                            kind: Box::new(ExprKind::Binary(*binop, target.clone(), value.clone())),
                            span: e.span,
                        },
                    };
                    return self.array_set_op(base, index, &val_expr, e.span);
                }
                // `s.field = v` → store into the heap object's field slot.
                if let ExprKind::Field { base, field } = &*target.kind {
                    return self.assign_field(base, field, op, value, e.span);
                }
                // Otherwise: assignment to a simple local.
                let ExprKind::Path(p) = &*target.kind else {
                    return err("assignment target must be a variable, field, or index", e.span);
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

            ExprKind::Index { base, index } => self.array_get_op(base, index, e.span),

            ExprKind::Tuple(elems) => {
                let mut cfields = Vec::with_capacity(elems.len());
                let mut tys = Vec::with_capacity(elems.len());
                for el in elems {
                    let (ce, t) = self.expr(el, None)?;
                    cfields.push(ce);
                    tys.push(t);
                }
                let ty = Ty::Tuple(tys);
                let repr = self.repr_of(&ty, e.span)?;
                let Repr::Value(vid) = repr else { return err("tuple must be a value type", e.span) };
                Ok((CoreExpr::new(CoreExprKind::MakeValue { value: vid, fields: cfields }, Repr::Value(vid)), ty))
            }

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

            ExprKind::Closure { params, ret, body } => {
                self.closure(params, ret, body, expected, e.span)
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
            // Apply the enclosing function's instantiation substitution so a
            // field type like `Array<T>` (where T is a still-generic param of the
            // surrounding fn) becomes the concrete `Array<i64>` the field value
            // (e.g. `array_new(..)`) needs as its expected type.
            let field_hint = apply_subst(&declared, &self.subst);
            let (cv, vt) = self.expr(&val_expr, hint(&field_hint))?;
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
        match repr {
            Repr::Ref(lid) => Ok((CoreExpr::new(CoreExprKind::MakeVariant { layout: lid, tag, fields: cargs }, repr), ty)),
            Repr::Value(vid) => {
                Ok((CoreExpr::new(CoreExprKind::MakeValueVariant { value: vid, tag, fields: cargs }, repr), ty))
            }
            _ => err("enum constructor produced a non-aggregate repr", span),
        }
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
        let fname = match field {
            FieldAccess::Named(n) => n,
            FieldAccess::Tuple(i) => {
                // `t.0` — tuple element access (value aggregate).
                let Ty::Tuple(elems) = &bty else {
                    return err("tuple index on a non-tuple", span);
                };
                let elem_ty = elems.get(*i as usize).cloned()
                    .ok_or_else(|| LowerError { msg: format!("tuple has no element {}", i), span })?;
                let frepr = self.repr_of(&elem_ty, span)?;
                return Ok((
                    CoreExpr::new(CoreExprKind::Field { base: Box::new(cbase), loc: FieldLoc::ValueField { index: *i } }, frepr),
                    elem_ty,
                ));
            }
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
        // The FieldLoc is in the base layout's field_map for reference structs;
        // value structs store fields in declaration order in the LLVM aggregate.
        let loc = match &cbase.repr {
            Repr::Ref(lid) => self.reg.layouts[*lid as usize].field_map[idx],
            Repr::Value(_vid) => FieldLoc::ValueField { index: idx as u32 },
            _ => return err("field access on non-aggregate", span),
        };
        let frepr = self.repr_of(&fty, span)?;
        Ok((CoreExpr::new(CoreExprKind::Field { base: Box::new(cbase), loc }, frepr), fty))
    }

    /// Lower `base.field = value` to a `SetField` (reference structs only).
    fn assign_field(&mut self, base: &Expr, field: &FieldAccess, op: &Option<BinOp>, value: &Expr, span: Span) -> LResult<(CoreExpr, Ty)> {
        let (cbase, bty) = self.expr(base, None)?;
        let FieldAccess::Named(fname) = field else {
            return err("tuple field assignment not supported in v0", span);
        };
        let Ty::Named { name, args: struct_args } = &bty else {
            return err("field assignment on a non-struct", span);
        };
        let s = self.ctx.structs.get(name).cloned()
            .ok_or_else(|| LowerError { msg: format!("`{}` is not a struct", name), span })?;
        if s.is_value {
            return err("value structs are immutable; rebuild instead of assigning a field", span);
        }
        let StructBody::Named(def_fields) = &s.body else {
            return err("field assignment on a non-record struct", span);
        };
        let idx = def_fields.iter().position(|f| &f.name == fname)
            .ok_or_else(|| LowerError { msg: format!("no field `{}` on `{}`", fname, name), span })?;
        let gparams: Vec<String> = s.generics.params.iter().map(|p| p.name.clone()).collect();
        let subst: HashMap<String, Ty> = gparams.iter().cloned().zip(struct_args.iter().cloned()).collect();
        let fty = apply_subst(&lower_type(&def_fields[idx].ty, &gparams, self.ctx).map_err(conv)?, &subst);
        let loc = match &cbase.repr {
            Repr::Ref(lid) => self.reg.layouts[*lid as usize].field_map[idx],
            _ => return err("field assignment on non-reference", span),
        };
        // Compound assign: read the current field first.
        let new_val = match op {
            None => { let (cv, vt) = self.expr(value, Some(&fty))?; check_assignable(&vt, &fty, value.span)?; cv }
            Some(binop) => {
                let frepr = self.repr_of(&fty, span)?;
                let cur = CoreExpr::new(CoreExprKind::Field { base: Box::new(cbase.clone()), loc }, frepr.clone());
                let (cv, vt) = self.expr(value, Some(&fty))?;
                check_assignable(&vt, &fty, value.span)?;
                CoreExpr::new(CoreExprKind::Bin(*binop, Box::new(cur), Box::new(cv)), frepr)
            }
        };
        Ok((CoreExpr::new(CoreExprKind::SetField { base: Box::new(cbase), loc, value: Box::new(new_val) }, Repr::Scalar(ScalarRepr::I64)), Ty::i64()))
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

        // Exhaustiveness: every variant must be covered, or a wildcard arm
        // (tag == u32::MAX) must be present. A non-exhaustive match is an error.
        let has_wildcard = carms.iter().any(|a| a.tag == u32::MAX);
        if !has_wildcard {
            let covered: std::collections::HashSet<u32> = carms.iter().map(|a| a.tag).collect();
            let missing: Vec<&str> = variants.iter()
                .filter(|(t, _)| !covered.contains(t))
                .filter_map(|(t, _)| {
                    self.ctx.enums.get(&enum_name)
                        .and_then(|e| e.variants.get(*t as usize))
                        .map(|v| v.name.as_str())
                })
                .collect();
            if !missing.is_empty() {
                return err(
                    format!("non-exhaustive match on `{}`: missing variant(s) {} (add the arms or a `_` wildcard)",
                        enum_name.rsplit("::").next().unwrap(), missing.join(", ")),
                    span,
                );
            }
        }

        let rty = result_ty.unwrap_or_else(Ty::unit);
        let repr = self.repr_of(&rty, span)?;
        // Value enums match on the inline aggregate's tag; reference enums on
        // the heap object's tag word.
        let node = match &cscrut.repr {
            Repr::Value(_) => CoreExprKind::ValueMatch { scrutinee: Box::new(cscrut), arms: carms },
            _ => CoreExprKind::Match { scrutinee: Box::new(cscrut), arms: carms },
        };
        Ok((CoreExpr::new(node, repr), rty))
    }

    /// `array_new(len)` — allocate a varlen array. Element type from `expected`.
    fn array_new(&mut self, len: &Expr, expected: Option<&Ty>, span: Span) -> LResult<(CoreExpr, Ty)> {
        let elem_ty = match expected {
            Some(Ty::Named { name, args }) if name.rsplit("::").next() == Some("Array") && !args.is_empty() => args[0].clone(),
            _ => return err("array_new requires an expected `Array<T>` type (annotate the let binding)", span),
        };
        let elem = self.repr_of(&elem_ty, span)?;
        let lid = self.reg.array_for(&elem);
        let (clen, lt) = self.expr(len, Some(&Ty::i64()))?;
        check_assignable(&lt, &Ty::i64(), len.span)?;
        let ty = Ty::Named { name: "Array".into(), args: vec![elem_ty] };
        Ok((CoreExpr::new(CoreExprKind::ArrayNew { layout: lid, len: Box::new(clen), elem }, Repr::Ref(lid)), ty))
    }

    fn array_len_op(&mut self, arr: &Expr, span: Span) -> LResult<(CoreExpr, Ty)> {
        let (ca, _) = self.expr(arr, None)?;
        Ok((CoreExpr::new(CoreExprKind::ArrayLen(Box::new(ca)), Repr::Scalar(ScalarRepr::I64)), Ty::i64()))
    }

    fn array_get_op(&mut self, arr: &Expr, idx: &Expr, span: Span) -> LResult<(CoreExpr, Ty)> {
        let (ca, aty) = self.expr(arr, None)?;
        let elem_ty = array_elem_ty(&aty).ok_or_else(|| LowerError { msg: "array_get on a non-array".into(), span })?;
        let elem = self.repr_of(&elem_ty, span)?;
        let (ci, it) = self.expr(idx, Some(&Ty::i64()))?;
        check_assignable(&it, &Ty::i64(), idx.span)?;
        Ok((CoreExpr::new(CoreExprKind::ArrayGet { array: Box::new(ca), index: Box::new(ci), elem: elem.clone() }, elem), elem_ty))
    }

    fn array_set_op(&mut self, arr: &Expr, idx: &Expr, val: &Expr, span: Span) -> LResult<(CoreExpr, Ty)> {
        let (ca, aty) = self.expr(arr, None)?;
        let elem_ty = array_elem_ty(&aty).ok_or_else(|| LowerError { msg: "array_set on a non-array".into(), span })?;
        let elem = self.repr_of(&elem_ty, span)?;
        let (ci, it) = self.expr(idx, Some(&Ty::i64()))?;
        check_assignable(&it, &Ty::i64(), idx.span)?;
        let (cv, vt) = self.expr(val, Some(&elem_ty))?;
        check_assignable(&vt, &elem_ty, val.span)?;
        Ok((CoreExpr::new(CoreExprKind::ArraySet { array: Box::new(ca), index: Box::new(ci), value: Box::new(cv), elem }, Repr::Scalar(ScalarRepr::I64)), Ty::i64()))
    }

    /// Lower a closure `|params| body`. Free variables that refer to enclosing
    /// locals are captured into a heap env object; the body is lifted into a
    /// top-level function `(env, params...) -> ret`. The closure value is the
    /// env reference, with the code pointer stored in its raw section.
    fn closure(
        &mut self,
        params: &[ClosureParam],
        ret: &Option<Type>,
        body: &Expr,
        expected: Option<&Ty>,
        span: Span,
    ) -> LResult<(CoreExpr, Ty)> {
        // Param types: from annotations, else from the `expected` fn type.
        let exp_params: Vec<Ty> = match expected {
            Some(Ty::Fn { params: p, .. }) => p.clone(),
            _ => vec![],
        };
        let mut param_tys = Vec::new();
        for (i, p) in params.iter().enumerate() {
            let ty = match &p.ty {
                Some(t) => ground_type(t, &self.subst.clone(), self.ctx)?,
                None => exp_params.get(i).cloned()
                    .ok_or_else(|| LowerError { msg: format!("cannot infer type of closure param `{}`", p.name), span: p.span })?,
            };
            param_tys.push(ty);
        }

        // Find captured free variables (names referenced in the body that are
        // bound in the current scope and not shadowed by a closure param).
        let mut free = Vec::new();
        let param_names: Vec<&str> = params.iter().map(|p| p.name.as_str()).collect();
        collect_free_vars(body, &param_names, &mut Vec::new(), &mut free);
        // Resolve each free name to (LocalId, Ty); dedup, keep only locals.
        let mut captures: Vec<(String, LocalId, Ty)> = Vec::new();
        let mut seen = std::collections::HashSet::new();
        for name in &free {
            if seen.contains(name) { continue; }
            if let Some((id, ty)) = self.lookup(name) {
                seen.insert(name.clone());
                captures.push((name.clone(), id, ty));
            }
        }

        // Capture reprs + the env layout.
        let mut cap_reprs = Vec::new();
        for (_, _, ty) in &captures {
            cap_reprs.push(self.repr_of(ty, span)?);
        }
        let key = format!("closure@{}.{}", span.start, span.end);
        let env_lid = self.reg.closure_env(&key, &cap_reprs);

        // Build the capture value expressions (loads of the enclosing locals).
        let capture_exprs: Vec<CoreExpr> = captures.iter().zip(&cap_reprs)
            .map(|((_, id, _), r)| CoreExpr::new(CoreExprKind::Local(*id), r.clone()))
            .collect();

        // Determine the return type. `None` ⇒ infer it from the body.
        let declared_ret = match ret {
            Some(t) => Some(ground_type(t, &self.subst.clone(), self.ctx)?),
            None => match expected {
                Some(Ty::Fn { ret, .. }) if !ret.is_unit() => Some((**ret).clone()),
                _ => None,
            },
        };

        // Lower the lifted function body in a fresh local/scope context that
        // binds the captures (loaded from env) and the params (from args).
        let code_id = self.mono.fresh_closure_id(self.count);
        let (lifted, ret_ty) = self.lower_lifted(&captures, &cap_reprs, params, &param_tys, body, declared_ret, env_lid, code_id, span)?;
        self.mono.closures.push((code_id, lifted));

        let fn_ty = Ty::Fn { params: param_tys, ret: Box::new(ret_ty) };
        Ok((
            CoreExpr::new(
                CoreExprKind::MakeClosure { code: code_id, env: env_lid, captures: capture_exprs },
                Repr::Ref(env_lid),
            ),
            fn_ty,
        ))
    }

    /// Build the lifted `CoreFn` for a closure: params are the env (implicit,
    /// added by codegen) followed by the closure's value params. Captures are
    /// loaded from the env at the top.
    #[allow(clippy::too_many_arguments)]
    fn lower_lifted(
        &mut self,
        captures: &[(String, LocalId, Ty)],
        cap_reprs: &[Repr],
        params: &[ClosureParam],
        param_tys: &[Ty],
        body: &Expr,
        declared_ret: Option<Ty>,
        env_lid: LayoutId,
        code_id: FuncId,
        span: Span,
    ) -> LResult<(CoreFn, Ty)> {
        // Swap in a fresh body-lowering context. The return type defaults to
        // the declared one if any (so `return` inside the body checks), else
        // unit until the body tail tells us otherwise.
        let saved_scope = std::mem::replace(&mut self.scope, vec![HashMap::new()]);
        let saved_locals = std::mem::take(&mut self.locals);
        let saved_local_tys = std::mem::take(&mut self.local_tys);
        let saved_ret = std::mem::replace(&mut self.ret_ty, declared_ret.clone().unwrap_or_else(Ty::unit));

        // Bind captures first as locals; codegen initializes these from the env.
        let mut cap_locals = Vec::new();
        for ((name, _, ty), repr) in captures.iter().zip(cap_reprs) {
            let id = self.fresh_local(repr.clone(), ty.clone());
            self.bind(name, id, ty.clone());
            cap_locals.push(id);
        }
        // Then params.
        let mut param_reprs = Vec::new();
        for (p, ty) in params.iter().zip(param_tys) {
            let repr = self.repr_of(ty, p.span)?;
            let id = self.fresh_local(repr.clone(), ty.clone());
            self.bind(&p.name, id, ty.clone());
            param_reprs.push(repr);
        }

        let (body_block, body_ty) = self.block_for_closure(body, declared_ret.as_ref())?;
        let ret_ty = match &declared_ret {
            Some(t) => { check_assignable(&body_ty, t, span)?; t.clone() }
            None => body_ty,
        };
        self.ret_ty = ret_ty.clone();
        let ret_repr = self.repr_of(&ret_ty, span)?;

        let locals = std::mem::replace(&mut self.locals, saved_locals);
        self.scope = saved_scope;
        self.local_tys = saved_local_tys;
        self.ret_ty = saved_ret;

        let _ = code_id;
        // Compute each capture's ABSOLUTE byte offset within the env (past the
        // 16-byte Full header). Pointer captures occupy the leading pointer
        // slots; scalar captures sit in the raw section after the 8-byte code
        // pointer. This must mirror MakeClosure codegen exactly.
        const HDR: u64 = 16;
        let n_ptr = cap_reprs.iter().filter(|r| matches!(r, Repr::Ref(_))).count() as u64;
        let raw_base = HDR + n_ptr * 8;
        let mut closure_captures = Vec::new();
        let mut ptr_idx = 0u64;
        let mut raw_off = 8u64; // skip the code pointer at raw offset 0
        for (cap_local, repr) in cap_locals.iter().zip(cap_reprs) {
            let offset = match repr {
                Repr::Ref(_) => { let o = HDR + ptr_idx * 8; ptr_idx += 1; o }
                Repr::Scalar(s) => {
                    let sz = (s.bits().max(8) / 8) as u64;
                    raw_off = raw_off.div_ceil(sz) * sz;
                    let o = raw_base + raw_off;
                    raw_off += sz;
                    o
                }
                _ => raw_base + raw_off,
            };
            closure_captures.push(ClosureCapture { local: *cap_local, offset });
        }
        Ok((CoreFn {
            name: format!("__closure_{}", env_lid),
            params: param_reprs,
            ret: ret_repr,
            locals,
            body: CoreBlock { stmts: body_block.stmts, tail: body_block.tail },
            closure_captures,
        }, ret_ty))
    }

    /// Lower a closure body (which is an arbitrary expression, possibly a block).
    fn block_for_closure(&mut self, body: &Expr, expected: Option<&Ty>) -> LResult<(CoreBlock, Ty)> {
        match &*body.kind {
            ExprKind::Block(b) => self.block_expected(b, expected),
            _ => {
                let (ce, t) = self.expr(body, expected)?;
                Ok((CoreBlock { stmts: vec![], tail: Some(ce) }, t))
            }
        }
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

    /// Whether an argument's own type can only be inferred from sibling
    /// arguments — currently: a bare enum-variant constructor (`None`, `Ok(x)`,
    /// `Some(x)`), whose enum type args may not be pinnable from itself. Such
    /// args are lowered in phase 2 after concrete args pin the type vars.
    fn is_inference_dependent_arg(&self, a: &Expr) -> bool {
        let path = match &*a.kind {
            ExprKind::Path(p) => p,
            ExprKind::Call(c, _) => match &*c.kind { ExprKind::Path(p) => p, _ => return false },
            _ => return false,
        };
        let key = path.segments.join("::");
        self.ctx.variants.contains_key(&key) || self.ctx.variants.contains_key(path.last())
    }

    fn call(&mut self, callee: &Expr, args: &[Expr], expected: Option<&Ty>, span: Span) -> LResult<(CoreExpr, Ty)> {
        // v0 slice: only direct calls to named, non-generic functions.
        let ExprKind::Path(path) = &*callee.kind else {
            return err("only direct function calls supported in v0 slice", span);
        };
        let name = path.last();
        // Calling a local that holds a closure value (`fn(...) -> _`).
        if path.is_single() {
            if let Some((id, Ty::Fn { params: fp, ret: fret })) = self.lookup(name) {
                if args.len() != fp.len() {
                    return err(format!("closure expects {} args, got {}", fp.len(), args.len()), span);
                }
                let crepr = self.repr_of(&Ty::Fn { params: fp.clone(), ret: fret.clone() }, span)?;
                let callee = CoreExpr::new(CoreExprKind::Local(id), crepr);
                let mut cargs = Vec::new();
                for (a, pt) in args.iter().zip(&fp) {
                    let (ca, at) = self.expr(a, Some(pt))?;
                    check_assignable(&at, pt, a.span)?;
                    cargs.push(ca);
                }
                let ret_repr = self.repr_of(&fret, span)?;
                return Ok((
                    CoreExpr::new(CoreExprKind::CallClosure { callee: Box::new(callee), args: cargs }, ret_repr),
                    (*fret).clone(),
                ));
            }
        }
        // Array intrinsics — built-in varlen arrays. The element type comes
        // from `expected` (for new) or the array's own type (get/set/len).
        if !self.ctx.fns.keys().any(|k| k.rsplit("::").next().unwrap() == name) {
            match name {
                "array_new" if args.len() == 1 => return self.array_new(&args[0], expected, span),
                "array_len" if args.len() == 1 => return self.array_len_op(&args[0], span),
                "array_get" if args.len() == 2 => return self.array_get_op(&args[0], &args[1], span),
                "array_set" if args.len() == 3 => return self.array_set_op(&args[0], &args[1], &args[2], span),
                _ => {}
            }
        }
        // print_int / print_float intrinsics → runtime extern, return i64.
        if (name == "print_int" || name == "print_float")
            && !self.ctx.fns.keys().any(|k| k.rsplit("::").next().unwrap() == name)
        {
            if args.len() != 1 {
                return err(format!("`{}` takes 1 argument", name), span);
            }
            let (ca, _) = self.expr(&args[0], None)?;
            return Ok((
                CoreExpr::new(CoreExprKind::Print(Box::new(ca)), Repr::Scalar(ScalarRepr::I64)),
                Ty::i64(),
            ));
        }
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
        // Seed inference from the EXPECTED return type, if any. This lets
        // `let v: Vec<i64> = vec_new()` infer `T=i64` from the annotation when no
        // argument constrains it (the only way to type a no-arg generic ctor).
        if !gparams.is_empty() {
            if let Some(exp) = expected {
                let ret_decl = match &f.ret {
                    Some(t) => lower_type(t, &gparams, self.ctx).map_err(conv)?,
                    None => Ty::unit(),
                };
                unify_infer(&ret_decl, exp, &mut targ);
            }
        }
        // Two-pass argument lowering so type variables inferred from CONCRETE
        // arguments guide inference for var-dependent ones. Phase 1 lowers args
        // whose declared type has no free var (and args of a non-generic callee).
        // Phase 2 lowers the rest with the resolved subst as the expected hint —
        // this is what makes `unwrap_or(None, x)` infer `T` from `x`.
        let declared_tys: Vec<Ty> = f.params.iter()
            .map(|p| lower_type(&p.ty, &gparams, self.ctx).map_err(conv))
            .collect::<Result<_, _>>()?;
        let mut lowered: Vec<Option<(CoreExpr, Ty)>> = (0..args.len()).map(|_| None).collect();
        // Phase 1: lower every argument that is NOT a bare enum-variant
        // constructor. Their concrete types pin type variables. (A variant
        // constructor like `None` may need a type var that only another argument
        // can supply, so it's deferred to phase 2.)
        for (i, (a, declared)) in args.iter().zip(&declared_tys).enumerate() {
            if self.is_inference_dependent_arg(a) { continue; }
            let (ca, at) = self.expr(a, hint(declared))?;
            unify_infer(declared, &at, &mut targ);
            lowered[i] = Some((ca, at));
        }
        // Phase 2: the deferred (variant-constructor) args, now with the type
        // vars inferred from phase 1 substituted into their expected type.
        for (i, (a, declared)) in args.iter().zip(&declared_tys).enumerate() {
            if lowered[i].is_some() { continue; }
            let hinted = apply_subst(declared, &targ);
            let (ca, at) = self.expr(a, hint(&hinted))?;
            unify_infer(declared, &at, &mut targ);
            lowered[i] = Some((ca, at));
        }
        let mut cargs = Vec::new();
        let mut arg_tys = Vec::new();
        for slot in lowered {
            let (ca, at) = slot.expect("every argument lowered");
            arg_tys.push((/* declared placeholder set below */ at.clone(), at));
            cargs.push(ca);
        }
        // Re-pair arg_tys with their declared types for the assignability check.
        for (slot, declared) in arg_tys.iter_mut().zip(&declared_tys) {
            slot.0 = declared.clone();
        }

        // Resolve the instantiation's concrete type args (in declared order).
        let mut inst_args = Vec::new();
        for gp in &gparams {
            match targ.get(gp) {
                Some(t) => inst_args.push(t.clone()),
                None => return err(format!("cannot infer type parameter `{}` for `{}`", gp, name), span),
            }
        }

        // Check declared trait bounds (`fn f<T: Show>`): the concrete type each
        // param resolves to must implement every bound trait. This catches
        // `f(unrelated_value)` at the CALL with a clear bound error instead of a
        // confusing "no method" error deep inside the (substituted) body.
        for (gp, conc) in f.generics.params.iter().zip(&inst_args) {
            for bound in &gp.bounds {
                let trait_name = bound.path.last();
                if !self.ctx.type_implements_trait(conc, trait_name) {
                    return err(
                        format!("the trait bound `{}: {}` is not satisfied (required by `{}`)",
                            ty_display(conc), trait_name, name),
                        span,
                    );
                }
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
        err(format!("type mismatch: expected `{}`, found `{}`", ty_display(expected), ty_display(actual)), span)
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
