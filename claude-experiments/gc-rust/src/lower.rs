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

/// Levenshtein edit distance between two short identifiers (for "did you mean?"
/// suggestions). Bounded use only — fine for identifier-length strings.
fn edit_distance(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    let mut cur = vec![0usize; b.len() + 1];
    for i in 1..=a.len() {
        cur[0] = i;
        for j in 1..=b.len() {
            let cost = if a[i - 1] == b[j - 1] { 0 } else { 1 };
            cur[j] = (prev[j] + 1).min(cur[j - 1] + 1).min(prev[j - 1] + cost);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[b.len()]
}

/// Pick the closest candidate name to `name` (within a small edit distance), for
/// a "did you mean `x`?" hint. Returns `None` if nothing is close enough.
fn closest<'a>(name: &str, candidates: impl Iterator<Item = &'a str>) -> Option<String> {
    // Allow up to ~1/3 of the name's length in edits (min 1, max 3).
    let budget = (name.len() / 3).clamp(1, 3);
    let mut best: Option<(usize, String)> = None;
    for c in candidates {
        if c == "_" || c.is_empty() { continue; }
        let d = edit_distance(name, c);
        if d <= budget && best.as_ref().map_or(true, |(bd, _)| d < *bd) {
            best = Some((d, c.to_string()));
        }
    }
    best.map(|(_, s)| s)
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
    // GC-safety normalization: let-bind every GC temporary so the collector can
    // find it (no live pointer is ever stranded in a register across a safepoint).
    crate::anf::anf_program(&mut prog);
    Ok(prog)
}

/// Type-check every non-generic top-level function (and method) independently,
/// collecting ALL errors rather than stopping at the first. This is the
/// multi-error pass: unlike `lower_program` (which is demand-driven from `main`
/// and halts on the first `?`), this checks each function in isolation, so one
/// broken function doesn't hide errors in the others.
///
/// Generic functions are skipped here — they can only be checked once
/// instantiated with concrete type args, which the demand-driven `lower_program`
/// does. So this catches the common case (broken concrete functions) up front;
/// `lower_program` still reports any error reachable only through a generic
/// instantiation.
pub fn check_program(globals: &crate::resolve::GlobalTable) -> Result<(), Vec<LowerError>> {
    let ctx = TyCtx::from_globals(globals);
    let mut errors = Vec::new();

    // Free functions: check each non-generic one in a throwaway lowering context.
    let mut names: Vec<&String> = ctx.fns.keys().collect();
    names.sort(); // deterministic error order
    for fq in names {
        let f = &ctx.fns[fq];
        if !f.generics.params.is_empty() {
            continue; // generic — needs instantiation; left to lower_program
        }
        let mut reg = LayoutRegistry::new(&ctx);
        let mut mono = Mono::new();
        let mut count = 0u32;
        let job = Job { f: f.clone(), subst: HashMap::new(), mangled: (*fq).clone(), self_ty: None };
        if let Err(e) = lower_fn(&job, &ctx, &mut reg, &mut mono, &mut count) {
            errors.push(e);
        }
    }

    if errors.is_empty() { Ok(()) } else { Err(errors) }
}

fn install_fn(prog: &mut CoreProgram, id: FuncId, f: CoreFn) {
    while prog.funcs.len() <= id as usize {
        prog.funcs.push(placeholder_fn());
    }
    prog.funcs[id as usize] = f;
}

fn placeholder_fn() -> CoreFn {
    CoreFn { name: String::new(), params: vec![], ret: Repr::Unit, locals: vec![], body: CoreBlock { stmts: vec![], tail: None }, closure_captures: vec![], is_extern: false }
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
        Ty::ExternFn { params, ret } => format!("extern fn({}) -> {}", params.iter().map(ty_display).collect::<Vec<_>>().join(", "), ty_display(ret)),
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
            Prim::RawPtr => "RawPtr",
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
        Ty::ExternFn { params, ret } => format!("XF{}_{}", params.iter().map(ty_mangle).collect::<Vec<_>>().join("_"), ty_mangle(ret)),
        Ty::Infer(n) => format!("I{}", n),
    }
}

/// Is `repr` allowed to cross the FFI boundary? Scalars are blittable; a value
/// struct is blittable iff it is a plain struct (not a value enum) whose fields
/// are all transitively blittable. Managed heap (`Ref`) types and `Unit` are
/// not. See `docs/ffi.md`.
/// Expected argument kind for a runtime-call intrinsic (atomics etc.).
#[derive(Clone, Copy)]
enum ArgKind { Int, Ptr }

fn repr_is_blittable(repr: &Repr, values: &[ValueLayout]) -> bool {
    match repr {
        Repr::Scalar(_) => true,
        Repr::Value(vid) => {
            let v = &values[*vid as usize];
            // Value enums carry a tag union whose layout we don't yet expose as
            // a C-stable type; only plain value structs cross.
            v.variants.is_none() && v.fields.iter().all(|f| repr_is_blittable(f, values))
        }
        Repr::Ref(_) | Repr::Unit => false,
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
    /// Lexical scope: name → (LocalId, Ty, is_mut). `is_mut` drives the
    /// immutable-by-default mutability discipline (see `docs/mutability.md`).
    scope: Vec<HashMap<String, (LocalId, Ty, bool)>>,
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

    // Foreign `extern "C"` declaration: no body to lower. Enforce the Phase-A
    // blittable-only rule (scalars only cross the boundary — never a managed
    // GC pointer), then emit a body-less `CoreFn` named after the bare C symbol.
    // See `docs/ffi.md`.
    if f.is_extern {
        let mut reg_lo = FnLowerer {
            ctx, reg, mono, count,
            subst: subst.clone(),
            scope: vec![HashMap::new()],
            locals: vec![],
            local_tys: vec![],
            ret_ty: ret_ty.clone(),
        };
        let mut params = Vec::new();
        for p in &f.params {
            let ty = ground_type(&p.ty, subst, ctx)?;
            let repr = reg_lo.repr_of(&ty, p.span)?;
            // Scalars cross by value; `#[repr(C)]` value structs of transitively
            // blittable fields cross by pointer (caller passes a stack alloca).
            // A managed heap (`Ref`) type never crosses. See `docs/ffi.md`.
            if !repr_is_blittable(&repr, &reg_lo.reg.values) {
                return err(
                    format!(
                        "extern function `{}` parameter `{}` has type `{}`, but only scalar \
                         types and value structs of scalar fields may cross the FFI boundary \
                         (a managed/heap type cannot — see docs/ffi.md)",
                        f.name, p.name, ty_display(&ty),
                    ),
                    p.span,
                );
            }
            params.push(repr);
        }
        let ret = reg_lo.repr_of(&ret_ty, f.span)?;
        if !matches!(ret, Repr::Scalar(_) | Repr::Unit) {
            return err(
                format!(
                    "extern function `{}` returns `{}`, but only scalar types (or no return) \
                     may cross the FFI boundary (see docs/ffi.md)",
                    f.name, ty_display(&ret_ty),
                ),
                f.span,
            );
        }
        return Ok(CoreFn {
            name: f.name.clone(),
            params,
            ret,
            locals: vec![],
            body: CoreBlock { stmts: vec![], tail: None },
            closure_captures: vec![],
            is_extern: true,
        });
    }

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
        lo.bind_mut("self", id, self_ty, f.self_is_mut);
        params.push(repr);
    }
    for p in &f.params {
        let ty = ground_type(&p.ty, &lo.subst.clone(), lo.ctx)?;
        let repr = lo.repr_of(&ty, p.span)?;
        let id = lo.fresh_local(repr.clone(), ty.clone());
        lo.bind_mut(&p.name, id, ty, p.is_mut);
        params.push(repr);
    }

    let (body, body_ty) = lo.block_expected(&f.body, Some(&ret_ty))?;
    // Point a body/return type mismatch at the tail expression (or the function
    // body if there's no tail), so the caret lands on the offending value.
    let body_err_span = f.body.tail.as_ref().map(|t| t.span).unwrap_or(f.body.span);
    check_assignable(&body_ty, &ret_ty, body_err_span)?;
    let ret = lo.repr_of(&ret_ty, f.span)?;

    Ok(CoreFn { name: job.mangled.clone(), params, ret, locals: lo.locals, body, closure_captures: vec![], is_extern: false })
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
        Ty::ExternFn { params, ret } => Ty::ExternFn {
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
    /// Bind an immutable local (the default). Use `bind_mut` for `let mut`,
    /// `mut` params, and `mut self`.
    fn bind(&mut self, name: &str, id: LocalId, ty: Ty) {
        self.bind_mut(name, id, ty, false);
    }
    fn bind_mut(&mut self, name: &str, id: LocalId, ty: Ty, is_mut: bool) {
        self.scope.last_mut().unwrap().insert(name.to_string(), (id, ty, is_mut));
    }
    fn lookup(&self, name: &str) -> Option<(LocalId, Ty)> {
        for s in self.scope.iter().rev() {
            if let Some((id, ty, _)) = s.get(name) {
                return Some((*id, ty.clone()));
            }
        }
        None
    }
    /// Is the binding `name` mutable in the current scope? `None` if unbound.
    fn lookup_mut(&self, name: &str) -> Option<bool> {
        for s in self.scope.iter().rev() {
            if let Some((_, _, is_mut)) = s.get(name) {
                return Some(*is_mut);
            }
        }
        None
    }

    /// The root binding name of an assignment target's access path:
    /// `a` for `a`, `a.b`, `a.b.c`, `a[i]`, `a.b[i].c`. Used to enforce
    /// transitive (deep) immutability: mutating any place requires the root of
    /// its path to be a `mut` binding. Returns `None` if the path isn't rooted
    /// in a simple variable (e.g. a function-call result).
    fn path_root_name(e: &Expr) -> Option<String> {
        match &*e.kind {
            ExprKind::Path(p) if p.is_single() => Some(p.last().to_string()),
            ExprKind::Field { base, .. } => Self::path_root_name(base),
            ExprKind::Index { base, .. } => Self::path_root_name(base),
            _ => None,
        }
    }

    /// Enforce the immutable-by-default rule for an assignment to `target`:
    /// the root binding of the access path must be declared `mut`. `what`
    /// describes the place for the error message ("variable", "field of",
    /// "element of").
    fn check_mutable_root(&self, target: &Expr, what: &str, span: Span) -> LResult<()> {
        match Self::path_root_name(target) {
            Some(root) => match self.lookup_mut(&root) {
                Some(true) => Ok(()),
                Some(false) => {
                    let fix = if root == "self" {
                        "declare the receiver `mut self`".to_string()
                    } else {
                        format!("declare it `let mut {}`, or take it as `mut {}`", root, root)
                    };
                    err(format!("cannot assign to {} immutable binding `{}` ({})", what, root, fix), span)
                }
                None => err(format!("unknown variable `{}`", root), span),
            },
            // No simple root (e.g. assigning into a temporary) — reject; there's
            // nothing to make `mut`.
            None => err(
                format!("cannot assign to {} a temporary value (it is not rooted in a `mut` binding)", what),
                span,
            ),
        }
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
                    let (name, is_mut) = match &pattern.kind {
                        PatternKind::Binding { name, is_mut } => (name.clone(), *is_mut),
                        PatternKind::Wildcard => ("_".to_string(), false),
                        _ => return err("only simple `let x =` patterns supported in v0", *span),
                    };
                    match init {
                        Some(e) => {
                            let expected = match ty {
                                Some(t) => Some(ground_type(t, &self.subst.clone(), self.ctx)?),
                                None => None,
                            };
                            let (init_expr, init_ty) = {
                                let (ce, at) = self.expr(e, expected.as_ref())?;
                                // A `let x: T = e` whose `e` doesn't match `T` is
                                // an error pointing at `e`.
                                if let Some(exp) = &expected {
                                    check_assignable(&at, exp, e.span)?;
                                }
                                (ce, at)
                            };
                            let repr = init_expr.repr.clone();
                            let id = self.fresh_local(repr, init_ty.clone());
                            self.bind_mut(&name, id, init_ty, is_mut);
                            stmts.push(CoreStmt::Let(id, init_expr));
                        }
                        // Deferred initialization: `let mut x: T;`. Requires a type
                        // annotation (nothing to infer from) and `mut` (the binding
                        // will be assigned later — see docs/mutability.md). The slot
                        // is zero-initialized; a later `x = ...` fills it.
                        None => {
                            let Some(t) = ty else {
                                return err("`let` without an initializer needs a type annotation (`let mut x: T;`)", *span);
                            };
                            if !is_mut {
                                return err("a deferred-initialization `let` must be `mut` (it is assigned later): write `let mut`", *span);
                            }
                            let dty = ground_type(t, &self.subst.clone(), self.ctx)?;
                            let repr = self.repr_of(&dty, *span)?;
                            let id = self.fresh_local(repr.clone(), dty.clone());
                            self.bind_mut(&name, id, dty.clone(), is_mut);
                            // Zero-initialize so the slot is well-defined before the
                            // first assignment (and GC-safe for ref slots → null).
                            let zero = self.zero_value(&repr);
                            stmts.push(CoreStmt::Let(id, zero));
                        }
                    }
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
                    // Suggest a similarly-spelled in-scope local or known fn.
                    let in_scope: Vec<String> = self.scope.iter()
                        .flat_map(|s| s.keys().cloned())
                        .chain(self.ctx.fns.keys().map(|k| k.rsplit("::").next().unwrap().to_string()))
                        .collect();
                    let hint = closest(name, in_scope.iter().map(|s| s.as_str()))
                        .map(|s| format!(" (did you mean `{}`?)", s))
                        .unwrap_or_default();
                    err(format!("unknown variable `{}`{}", name, hint), e.span)
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
                // `return` diverges — type never (assignable to any context).
                Ok((CoreExpr::new(CoreExprKind::Return(cv), Repr::Unit), never_ty()))
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
                // A `loop` diverges from the value's standpoint: it runs forever
                // or exits via `return`/`break`. Type it as never so it can stand
                // as a function body of any return type (e.g. the `swap` retry
                // loop). Never is assignable to anything, including unit.
                Ok((CoreExpr::new(CoreExprKind::Loop(Box::new(cb)), Repr::Unit), never_ty()))
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
                    // (`array_set_op` enforces the mutable-root rule for both this
                    // desugar and direct `array_set(..)` calls.)
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
                // Immutable-by-default: reassigning a non-`mut` binding is an error.
                if self.lookup_mut(p.last()) != Some(true) {
                    return err(
                        format!("cannot reassign immutable binding `{}` (declare it `let mut {}`)", p.last(), p.last()),
                        e.span,
                    );
                }
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
        // Transitive immutability: `base.field = v` requires the root of `base`'s
        // access path to be a `mut` binding (see docs/mutability.md).
        self.check_mutable_root(base, "field of", span)?;
        let (cbase, bty) = self.expr(base, None)?;
        let FieldAccess::Named(fname) = field else {
            // `t.0 = v`. Tuples are value aggregates (immutable, like value
            // structs): rebuild the tuple instead of assigning an element.
            return err("tuples are immutable value types; rebuild the tuple instead of assigning `.N` (e.g. `t = (new0, t.1)`)", span);
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

        // Dispatch: the fast tag-switch path handles a pure enum match with only
        // variant/wildcard arms and no guards. Anything else — a scalar/string
        // scrutinee, literal patterns, or guards — lowers to a sequential
        // if-else decision chain over the existing core primitives.
        let is_enum = matches!(&sty, Ty::Named { name, .. } if self.ctx.enums.contains_key(name));
        let any_guard = arms.iter().any(|a| a.guard.is_some());
        // The switch path handles only variant arms with simple (binding/wildcard)
        // payloads and an optional trailing wildcard. A bare binding arm (named
        // catch-all), literal, tuple, or struct pattern routes to the chain.
        let switchable = arms.iter().all(|a| match &a.pattern.kind {
            PatternKind::Wildcard => true,
            PatternKind::Variant { payload, .. } => payload.iter().all(|p| matches!(
                &p.kind, PatternKind::Binding { .. } | PatternKind::Wildcard
            )),
            _ => false,
        });
        if !is_enum || any_guard || !switchable {
            return self.match_chain(cscrut, sty, arms, span);
        }

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

    /// Lower a match as a sequential if-else decision chain. Handles scalar /
    /// string scrutinees, literal patterns, bare-binding catch-alls, and guards
    /// — anything the fast enum tag-switch can't. Built entirely from existing
    /// core primitives (`Let`, `If`, `Bin(Eq)`, `StrEq`), so no codegen changes.
    fn match_chain(&mut self, cscrut: CoreExpr, sty: Ty, arms: &[MatchArm], span: Span) -> LResult<(CoreExpr, Ty)> {
        // Bind the scrutinee to a fresh local so we evaluate it once and can test
        // it repeatedly.
        let srepr = cscrut.repr.clone();
        let sid = self.fresh_local(srepr.clone(), sty.clone());
        let scrut_local = |this: &Self| CoreExpr::new(CoreExprKind::Local(sid), this.locals[sid as usize].clone());

        // Enum info, if the scrutinee is an enum (for variant patterns in a
        // guarded match).
        let enum_info: Option<(String, Vec<Ty>, Vec<(u32, Vec<Ty>)>)> = match &sty {
            Ty::Named { name, args } if self.ctx.enums.contains_key(name) => {
                let vs = self.enum_variants(name, args)?;
                Some((name.clone(), args.clone(), vs))
            }
            _ => None,
        };

        let mut result_ty: Option<Ty> = None;
        // Build the chain bottom-up: start with the "no arm matched" tail.
        // For an exhaustive match the final arm is a catch-all; otherwise we
        // require coverage via a wildcard/binding (checked below).
        let mut saw_irrefutable = false;
        // Variant tags covered by an UNGUARDED variant arm (for enum
        // exhaustiveness when there's no wildcard).
        let mut covered_tags: std::collections::HashSet<u32> = std::collections::HashSet::new();
        struct Built { cond: Option<CoreExpr>, binds: Vec<(LocalId, CoreExpr)>, body: CoreExpr }
        let mut built: Vec<Built> = Vec::new();

        for arm in arms {
            self.push_scope();
            // Track an unguarded variant arm's tag for exhaustiveness.
            if arm.guard.is_none() {
                if let (PatternKind::Variant { path, .. }, Some((en, _, vs))) = (&arm.pattern.kind, &enum_info) {
                    if let Ok((tag, _)) = self.variant_in(en, path.last(), vs, arm.pattern.span) {
                        covered_tags.insert(tag);
                    }
                }
            }
            // Compute the match condition and the bindings this pattern introduces.
            let (cond, mut binds, irrefutable) =
                self.pattern_test(&arm.pattern, &scrut_local(self), &sty, &enum_info, arm.pattern.span)?;
            // A guard ANDs onto the condition (after bindings are in scope).
            let mut full_cond = cond;
            if let Some(g) = &arm.guard {
                // Bind names before lowering the guard so it can reference them.
                // The binds reference the scrutinee local; emit them as lets
                // inside the arm — but the guard needs them too, so we lower the
                // guard with a temporary view of the scope.
                for (id, init) in &binds {
                    // already bound in scope by pattern_test; nothing to do here.
                    let _ = (id, init);
                }
                let (cg, gt) = self.expr(g, Some(&Ty::bool()))?;
                check_assignable(&gt, &Ty::bool(), g.span)?;
                full_cond = Some(match full_cond {
                    Some(c) => CoreExpr::new(CoreExprKind::Bin(BinOp::And, Box::new(c), Box::new(cg)), Repr::Scalar(ScalarRepr::Bool)),
                    None => cg,
                });
            }
            if full_cond.is_none() {
                saw_irrefutable = true;
            }
            // An unguarded irrefutable pattern (wildcard/binding) makes coverage.
            let covers = irrefutable && arm.guard.is_none();
            let (body, bty) = self.expr(&arm.body, result_ty.as_ref())?;
            match &result_ty {
                Some(rt) => check_assignable(&bty, rt, arm.span)?,
                None => result_ty = Some(bty),
            }
            // Move bindings out for emission.
            let arm_binds = std::mem::take(&mut binds);
            built.push(Built { cond: full_cond, binds: arm_binds, body });
            self.pop_scope();
            if covers { saw_irrefutable = true; }
        }

        // Exhaustive if there's an irrefutable (wildcard/binding) arm, OR the
        // scrutinee is an enum and every variant tag is covered by an unguarded
        // variant arm.
        let enum_exhaustive = match &enum_info {
            Some((_, _, variants)) => variants.iter().all(|(t, _)| covered_tags.contains(t)),
            None => false,
        };
        if !saw_irrefutable && !enum_exhaustive {
            return err(
                "non-exhaustive match: add a `_` wildcard or binding arm to cover all cases",
                span,
            );
        }
        // The chain's innermost else is reached only if NO arm matched. When the
        // match is exhaustive purely via enum-variant coverage (no wildcard), the
        // LAST arm is guaranteed to match if reached, so we drop its condition —
        // it becomes the unconditional base of the chain. (Its pattern bindings
        // still apply.) This avoids a dangling empty `else` that would produce a
        // malformed PHI in codegen.
        if !saw_irrefutable && enum_exhaustive {
            if let Some(last) = built.last_mut() {
                last.cond = None;
            }
        }

        let rty = result_ty.unwrap_or_else(Ty::unit);
        let repr = self.repr_of(&rty, span)?;

        // Fold the arms into nested ifs, from the last arm up. Each arm with
        // bindings is wrapped as `{ let binds...; if cond { body } else { rest } }`
        // so the pattern bindings are in scope for BOTH the condition (guard) and
        // the body. The innermost "else" is unreachable (the exhaustiveness check
        // guarantees a prior irrefutable arm covers it).
        let unit_repr = repr.clone();
        let mut chain: Option<CoreExpr> = None;
        for b in built.into_iter().rev() {
            let bind_stmts: Vec<CoreStmt> = b.binds.into_iter().map(|(id, init)| CoreStmt::Let(id, init)).collect();
            let arm_expr = match b.cond {
                None => {
                    // Irrefutable arm: just `{ let binds...; body }`.
                    CoreExpr::new(
                        CoreExprKind::Block(Box::new(CoreBlock { stmts: bind_stmts, tail: Some(b.body) })),
                        unit_repr.clone(),
                    )
                }
                Some(cond) => {
                    let then_block = CoreBlock { stmts: vec![], tail: Some(b.body) };
                    let else_block = match chain.take() {
                        Some(e) => CoreBlock { stmts: vec![], tail: Some(e) },
                        None => CoreBlock { stmts: vec![], tail: None },
                    };
                    let iff = CoreExpr::new(
                        CoreExprKind::If(Box::new(cond), Box::new(then_block), Box::new(else_block)),
                        unit_repr.clone(),
                    );
                    // Wrap the `if` in a block that first emits the pattern binds,
                    // so the guard (inside `cond`) can reference them.
                    if bind_stmts.is_empty() {
                        iff
                    } else {
                        CoreExpr::new(
                            CoreExprKind::Block(Box::new(CoreBlock { stmts: bind_stmts, tail: Some(iff) })),
                            unit_repr.clone(),
                        )
                    }
                }
            };
            chain = Some(arm_expr);
        }

        // Emit `{ let scrut = <cscrut>; <chain> }`.
        let chain = chain.unwrap();
        let block = CoreBlock {
            stmts: vec![CoreStmt::Let(sid, cscrut)],
            tail: Some(chain),
        };
        Ok((CoreExpr::new(CoreExprKind::Block(Box::new(block)), repr), rty))
    }

    /// Compute the boolean test for a pattern against `scrut` (already a core
    /// expr referencing the scrutinee local), plus any bindings it introduces
    /// (each binding is bound into the current scope AND returned as a
    /// (LocalId, init-expr) so the caller can emit the `let`). Returns
    /// `(cond, binds, irrefutable)` where `cond == None` means "always matches".
    fn pattern_test(
        &mut self,
        pat: &Pattern,
        scrut: &CoreExpr,
        sty: &Ty,
        enum_info: &Option<(String, Vec<Ty>, Vec<(u32, Vec<Ty>)>)>,
        span: Span,
    ) -> LResult<(Option<CoreExpr>, Vec<(LocalId, CoreExpr)>, bool)> {
        match &pat.kind {
            PatternKind::Wildcard => Ok((None, vec![], true)),
            PatternKind::Binding { name, is_mut } => {
                // Bind the whole scrutinee to `name`.
                let id = self.fresh_local(scrut.repr.clone(), sty.clone());
                self.bind_mut(name, id, sty.clone(), *is_mut);
                Ok((None, vec![(id, scrut.clone())], true))
            }
            PatternKind::Literal(lit) => {
                let cond = self.literal_eq(scrut, lit, sty, span)?;
                Ok((Some(cond), vec![], false))
            }
            PatternKind::Variant { path, payload } => {
                let Some((enum_name, _eargs, variants)) = enum_info else {
                    return err("variant pattern in a match on a non-enum", span);
                };
                let vlast = path.last();
                let (tag, ptys) = self.variant_in(enum_name, vlast, variants, span)?;
                if payload.len() != ptys.len() {
                    return err(format!("variant `{}` binds {} fields, pattern has {}", vlast, ptys.len(), payload.len()), span);
                }
                // tag check: load the enum tag and compare. For a reference enum
                // the tag lives in the object; we expose it via a dedicated core
                // read. Reuse the existing Match machinery is overkill here, so
                // compare against an `EnumTag` read.
                let tag_read = CoreExpr::new(CoreExprKind::EnumTag(Box::new(scrut.clone())), Repr::Scalar(ScalarRepr::I32));
                let tag_const = CoreExpr::new(CoreExprKind::ConstInt(tag as u64, ScalarRepr::I32), Repr::Scalar(ScalarRepr::I32));
                let cond = CoreExpr::new(CoreExprKind::Bin(BinOp::Eq, Box::new(tag_read), Box::new(tag_const)), Repr::Scalar(ScalarRepr::Bool));
                // Bind payload fields (only simple bindings supported in the chain
                // for now; nested patterns within a guarded/literal match are a
                // later step).
                // Reprs of all payload fields (the physical slot of each field
                // depends on the full list — ptr and raw payloads are separate).
                let payload_reprs: Vec<Repr> = ptys.iter().map(|t| self.repr_of(t, span)).collect::<LResult<_>>()?;
                let mut binds = Vec::new();
                for (i, (p, pty)) in payload.iter().zip(&ptys).enumerate() {
                    let prepr = payload_reprs[i].clone();
                    let field = CoreExpr::new(
                        CoreExprKind::EnumPayload {
                            scrutinee: Box::new(scrut.clone()),
                            field: i as u32,
                            repr: prepr.clone(),
                            payload_reprs: payload_reprs.clone(),
                        },
                        prepr.clone(),
                    );
                    match &p.kind {
                        PatternKind::Wildcard => {}
                        PatternKind::Binding { name, is_mut } => {
                            let id = self.fresh_local(prepr, pty.clone());
                            self.bind_mut(name, id, pty.clone(), *is_mut);
                            binds.push((id, field));
                        }
                        _ => return err("nested patterns inside a guarded/scalar match are not yet supported", p.span),
                    }
                }
                Ok((Some(cond), binds, false))
            }
            PatternKind::Tuple(_) => err("tuple patterns are not yet supported in v0", span),
            PatternKind::Struct { .. } => err("struct patterns are not yet supported in v0", span),
        }
    }

    /// Build a boolean equality test `scrut == <literal>` for a literal pattern.
    fn literal_eq(&mut self, scrut: &CoreExpr, lit: &LitPattern, sty: &Ty, span: Span) -> LResult<CoreExpr> {
        let bool_repr = Repr::Scalar(ScalarRepr::Bool);
        match lit {
            LitPattern::Int(n) => {
                let sr = match &scrut.repr { Repr::Scalar(s) if !s.is_float() => *s, _ => return err("integer literal pattern against a non-integer scrutinee", span) };
                let k = CoreExpr::new(CoreExprKind::ConstInt(*n, sr), Repr::Scalar(sr));
                Ok(CoreExpr::new(CoreExprKind::Bin(BinOp::Eq, Box::new(scrut.clone()), Box::new(k)), bool_repr))
            }
            LitPattern::Bool(b) => {
                if !matches!(sty, Ty::Prim(Prim::Bool)) { return err("bool literal pattern against a non-bool scrutinee", span); }
                let k = CoreExpr::new(CoreExprKind::ConstBool(*b), bool_repr.clone());
                Ok(CoreExpr::new(CoreExprKind::Bin(BinOp::Eq, Box::new(scrut.clone()), Box::new(k)), bool_repr))
            }
            LitPattern::Char(c) => {
                if !matches!(sty, Ty::Prim(Prim::Char)) { return err("char literal pattern against a non-char scrutinee", span); }
                let k = CoreExpr::new(CoreExprKind::ConstChar(*c), Repr::Scalar(ScalarRepr::Char));
                Ok(CoreExpr::new(CoreExprKind::Bin(BinOp::Eq, Box::new(scrut.clone()), Box::new(k)), bool_repr))
            }
            LitPattern::Str(s) => {
                if !matches!(sty, Ty::Prim(Prim::Str)) { return err("string literal pattern against a non-string scrutinee", span); }
                let lid = match self.repr_of(&Ty::Prim(Prim::Str), span)? { Repr::Ref(l) => l, _ => return err("internal: String repr", span) };
                let k = CoreExpr::new(CoreExprKind::ConstStr(s.clone()), Repr::Ref(lid));
                Ok(CoreExpr::new(CoreExprKind::StrEq(Box::new(scrut.clone()), Box::new(k)), bool_repr))
            }
        }
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
        // Mutating an array element (whether via `a[i] = v` or `array_set(a, ..)`)
        // requires the array's access-path root to be a `mut` binding.
        self.check_mutable_root(arr, "element of", span)?;
        let (ca, aty) = self.expr(arr, None)?;
        let elem_ty = array_elem_ty(&aty).ok_or_else(|| LowerError { msg: "array_set on a non-array".into(), span })?;
        let elem = self.repr_of(&elem_ty, span)?;
        let (ci, it) = self.expr(idx, Some(&Ty::i64()))?;
        check_assignable(&it, &Ty::i64(), idx.span)?;
        let (cv, vt) = self.expr(val, Some(&elem_ty))?;
        check_assignable(&vt, &elem_ty, val.span)?;
        Ok((CoreExpr::new(CoreExprKind::ArraySet { array: Box::new(ca), index: Box::new(ci), value: Box::new(cv), elem }, Repr::Scalar(ScalarRepr::I64)), Ty::i64()))
    }

    /// Lower the built-in `String` intrinsics. Each argument must be a `String`.
    fn str_intrinsic(&mut self, name: &str, args: &[Expr], span: Span) -> LResult<(CoreExpr, Ty)> {
        let str_ty = Ty::Prim(Prim::Str);
        // Lower an argument and require it to be a String.
        let mut str_arg = |this: &mut Self, e: &Expr| -> LResult<CoreExpr> {
            let (ce, t) = this.expr(e, Some(&str_ty))?;
            check_assignable(&t, &str_ty, e.span)?;
            Ok(ce)
        };
        match name {
            "print_str" => {
                if args.len() != 1 { return err("`print_str` takes 1 argument", span); }
                let a = str_arg(self, &args[0])?;
                Ok((CoreExpr::new(CoreExprKind::PrintStr(Box::new(a)), Repr::Scalar(ScalarRepr::I64)), Ty::i64()))
            }
            "print" => {
                if args.len() != 1 { return err("`print` takes 1 argument", span); }
                let a = str_arg(self, &args[0])?;
                Ok((CoreExpr::new(CoreExprKind::PrintStrRaw(Box::new(a)), Repr::Scalar(ScalarRepr::I64)), Ty::i64()))
            }
            "char_to_str" => {
                if args.len() != 1 { return err("`char_to_str` takes 1 argument", span); }
                let (cp, cpt) = self.expr(&args[0], Some(&Ty::i64()))?;
                check_assignable(&cpt, &Ty::i64(), args[0].span)?;
                let layout = self.string_layout_id(span)?;
                Ok((
                    CoreExpr::new(CoreExprKind::StrFromChar { layout, cp: Box::new(cp) }, Repr::Ref(layout)),
                    str_ty,
                ))
            }
            "str_len" => {
                if args.len() != 1 { return err("`str_len` takes 1 argument", span); }
                let a = str_arg(self, &args[0])?;
                Ok((CoreExpr::new(CoreExprKind::StrLen(Box::new(a)), Repr::Scalar(ScalarRepr::I64)), Ty::i64()))
            }
            "str_eq" => {
                if args.len() != 2 { return err("`str_eq` takes 2 arguments", span); }
                let a = str_arg(self, &args[0])?;
                let b = str_arg(self, &args[1])?;
                Ok((CoreExpr::new(CoreExprKind::StrEq(Box::new(a), Box::new(b)), Repr::Scalar(ScalarRepr::Bool)), Ty::bool()))
            }
            "str_concat" => {
                if args.len() != 2 { return err("`str_concat` takes 2 arguments", span); }
                let a = str_arg(self, &args[0])?;
                let b = str_arg(self, &args[1])?;
                let layout = self.string_layout_id(span)?;
                Ok((
                    CoreExpr::new(CoreExprKind::StrConcat { layout, a: Box::new(a), b: Box::new(b) }, Repr::Ref(layout)),
                    str_ty,
                ))
            }
            "str_get" => {
                if args.len() != 2 { return err("`str_get` takes 2 arguments", span); }
                let s = str_arg(self, &args[0])?;
                let (i, it) = self.expr(&args[1], Some(&Ty::i64()))?;
                check_assignable(&it, &Ty::i64(), args[1].span)?;
                Ok((CoreExpr::new(CoreExprKind::StrGet(Box::new(s), Box::new(i)), Repr::Scalar(ScalarRepr::I64)), Ty::i64()))
            }
            "str_to_float" => {
                if args.len() != 1 { return err("`str_to_float` takes 1 argument", span); }
                let s = str_arg(self, &args[0])?;
                Ok((CoreExpr::new(CoreExprKind::StrToFloat(Box::new(s)), Repr::Scalar(ScalarRepr::F64)), Ty::Prim(Prim::F64)))
            }
            "float_bits" => {
                if args.len() != 1 { return err("`float_bits` takes 1 argument", span); }
                let (f, ft) = self.expr(&args[0], Some(&Ty::Prim(Prim::F64)))?;
                check_assignable(&ft, &Ty::Prim(Prim::F64), args[0].span)?;
                Ok((CoreExpr::new(CoreExprKind::FloatBits(Box::new(f)), Repr::Scalar(ScalarRepr::I64)), Ty::i64()))
            }
            "read_file" => {
                if args.len() != 1 { return err("`read_file` takes 1 argument", span); }
                let p = str_arg(self, &args[0])?;
                let layout = self.string_layout_id(span)?;
                Ok((
                    CoreExpr::new(CoreExprKind::ReadFile { layout, path: Box::new(p) }, Repr::Ref(layout)),
                    str_ty,
                ))
            }
            "str_hash" => {
                if args.len() != 1 { return err("`str_hash` takes 1 argument", span); }
                let s = str_arg(self, &args[0])?;
                Ok((CoreExpr::new(CoreExprKind::StrHash(Box::new(s)), Repr::Scalar(ScalarRepr::I64)), Ty::i64()))
            }
            "str_substring" => {
                if args.len() != 3 { return err("`str_substring` takes 3 arguments", span); }
                let s = str_arg(self, &args[0])?;
                let (st, stt) = self.expr(&args[1], Some(&Ty::i64()))?;
                check_assignable(&stt, &Ty::i64(), args[1].span)?;
                let (en, ent) = self.expr(&args[2], Some(&Ty::i64()))?;
                check_assignable(&ent, &Ty::i64(), args[2].span)?;
                let layout = self.string_layout_id(span)?;
                Ok((
                    CoreExpr::new(CoreExprKind::StrSubstring { layout, s: Box::new(s), start: Box::new(st), end: Box::new(en) }, Repr::Ref(layout)),
                    str_ty,
                ))
            }
            "to_string" => {
                if args.len() != 1 { return err("`to_string` takes 1 argument", span); }
                let (v, vt) = self.expr(&args[0], None)?;
                let is_float = match &vt {
                    Ty::Prim(Prim::F32 | Prim::F64) => true,
                    Ty::Prim(p) if matches!(p, Prim::I8 | Prim::I16 | Prim::I32 | Prim::I64
                        | Prim::U8 | Prim::U16 | Prim::U32 | Prim::U64 | Prim::Bool | Prim::Char) => false,
                    _ => return err("`to_string` requires an integer or float argument", args[0].span),
                };
                // Widen the numeric value so the runtime sees i64 / f64.
                let v = self.widen_to_string_arg(v, &vt, is_float);
                let layout = self.string_layout_id(span)?;
                Ok((
                    CoreExpr::new(CoreExprKind::StrFromNum { layout, is_float, v: Box::new(v) }, Repr::Ref(layout)),
                    str_ty,
                ))
            }
            _ => err(format!("unknown string intrinsic `{}`", name), span),
        }
    }

    /// Lower the built-in reflection intrinsics. The first argument is always a
    /// heap (`Ref`) value — reflection reads its object header (`type_id`) and the
    /// metadata table. Field intrinsics take a second `i64` field index and lower
    /// to `RuntimeCall`s into the `ai_reflect_*` externs.
    ///
    /// Surface:
    ///   `type_id_of(x) -> i64`, `type_name_of(x) -> String`
    ///   `field_count(x) -> i64`
    ///   `field_name(x, i) -> String`, `field_kind(x, i) -> i64`, `field_i64(x, i) -> i64`
    fn reflect_intrinsic(&mut self, name: &str, args: &[Expr], span: Span) -> LResult<(CoreExpr, Ty)> {
        let arity = if matches!(name, "type_id_of" | "type_name_of" | "field_count") { 1 } else { 2 };
        if args.len() != arity {
            return err(format!("`{}` takes {} argument(s)", name, arity), span);
        }
        let (obj, oty) = self.expr(&args[0], None)?;
        if !matches!(obj.repr, Repr::Ref(_)) {
            return err(
                format!("`{}` requires a heap (reference) value, got `{:?}`", name, oty),
                span,
            );
        }
        // Field intrinsics take an i64 index as the second argument.
        let idx = if arity == 2 {
            let (i, it) = self.expr(&args[1], Some(&Ty::i64()))?;
            check_assignable(&it, &Ty::i64(), args[1].span)?;
            Some(i)
        } else {
            None
        };
        let i64r = Repr::Scalar(ScalarRepr::I64);
        match name {
            "type_id_of" => Ok((
                CoreExpr::new(CoreExprKind::TypeIdOf(Box::new(obj)), i64r),
                Ty::i64(),
            )),
            "type_name_of" => {
                let layout = self.string_layout_id(span)?;
                Ok((
                    CoreExpr::new(
                        CoreExprKind::TypeNameOf { layout, obj: Box::new(obj) },
                        Repr::Ref(layout),
                    ),
                    Ty::Prim(Prim::Str),
                ))
            }
            "field_count" => Ok((
                CoreExpr::new(
                    CoreExprKind::RuntimeCall {
                        func: "ai_reflect_field_count",
                        args: vec![obj],
                        ret: i64r.clone(),
                    },
                    i64r,
                ),
                Ty::i64(),
            )),
            "field_kind" | "field_i64" => {
                let func = if name == "field_kind" { "ai_reflect_field_kind" } else { "ai_reflect_field_i64" };
                Ok((
                    CoreExpr::new(
                        CoreExprKind::RuntimeCall {
                            func,
                            args: vec![obj, idx.unwrap()],
                            ret: i64r.clone(),
                        },
                        i64r,
                    ),
                    Ty::i64(),
                ))
            }
            "field_name" => {
                let layout = self.string_layout_id(span)?;
                // ai_reflect_field_name(thread, i64 str_type_id, ptr obj, i64 i)
                let str_tid = CoreExpr::new(
                    CoreExprKind::ConstInt(layout as u64, ScalarRepr::I64),
                    Repr::Scalar(ScalarRepr::I64),
                );
                Ok((
                    CoreExpr::new(
                        CoreExprKind::RuntimeCall {
                            func: "ai_reflect_field_name",
                            args: vec![str_tid, obj, idx.unwrap()],
                            ret: Repr::Ref(layout),
                        },
                        Repr::Ref(layout),
                    ),
                    Ty::Prim(Prim::Str),
                ))
            }
            _ => err(format!("unknown reflection intrinsic `{}`", name), span),
        }
    }

    /// A zero/null value of the given repr, for deferred-init `let` slots.
    fn zero_value(&self, repr: &Repr) -> CoreExpr {
        match repr {
            Repr::Scalar(s) if s.is_float() => CoreExpr::new(CoreExprKind::ConstFloat(0.0, *s), repr.clone()),
            Repr::Scalar(s) => CoreExpr::new(CoreExprKind::ConstInt(0, *s), repr.clone()),
            Repr::Unit => CoreExpr::new(CoreExprKind::Unit, Repr::Unit),
            _ => CoreExpr::new(CoreExprKind::ConstZero(repr.clone()), repr.clone()),
        }
    }

    /// The layout id of the built-in `String` reference type.
    fn string_layout_id(&mut self, span: Span) -> LResult<LayoutId> {
        match self.repr_of(&Ty::Prim(Prim::Str), span)? {
            Repr::Ref(l) => Ok(l),
            _ => err("internal: String is not a reference type", span),
        }
    }

    /// Widen a numeric `to_string` argument to the i64/f64 the runtime expects.
    /// Codegen's `StrFromNum` passes the value straight through, so we coerce
    /// here via a `Cast` core node when the source repr is narrower.
    fn widen_to_string_arg(&self, v: CoreExpr, ty: &Ty, is_float: bool) -> CoreExpr {
        let target = if is_float { Ty::Prim(Prim::F64) } else { Ty::i64() };
        if ty == &target {
            return v;
        }
        let from = v.repr.clone();
        let to = if is_float { Repr::Scalar(ScalarRepr::F64) } else { Repr::Scalar(ScalarRepr::I64) };
        CoreExpr::new(CoreExprKind::Cast { value: Box::new(v), from, to: to.clone() }, to)
    }

    /// Lower a closure `|params| body`. Free variables that refer to enclosing
    /// locals are captured into a heap env object; the body is lifted into a
    /// top-level function `(env, params...) -> ret`. The closure value is the
    /// env reference, with the code pointer stored in its raw section.
    /// Send/Sync check for `Thread::spawn`: every value the closure `arg`
    /// captures must be `Sync` (it is shared with the parent thread). Non-closure
    /// args are left to the normal type check. See `docs/threads.md`.
    fn check_spawn_captures_sync(&self, arg: &Expr) -> LResult<()> {
        let ExprKind::Closure { params, body, .. } = &*arg.kind else { return Ok(()) };
        let param_names: Vec<&str> = params.iter().map(|p| p.name.as_str()).collect();
        let mut free = Vec::new();
        collect_free_vars(body, &param_names, &mut Vec::new(), &mut free);
        for name in &free {
            if let Some((_, ty)) = self.lookup(name) {
                if !self.ctx.is_sync(&ty) {
                    return err(
                        format!(
                            "cannot capture `{}: {}` in a spawned thread — it is not `Sync` \
                             (a shared mutable reference would race). Wrap it in `Atom<{}>` (or \
                             build it from `Atom`/`AtomicI64` fields), or capture an immutable value.",
                            name, ty_display(&ty), ty_display(&ty),
                        ),
                        arg.span,
                    );
                }
            }
        }
        Ok(())
    }

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
                // An inline value-aggregate capture lives in the raw section like
                // a scalar; advance `raw_off` by its size so a following capture
                // doesn't alias it. (Must mirror MakeClosure codegen.)
                Repr::Value(vid) => {
                    let sz = self.reg.values[*vid as usize].size as u64;
                    raw_off = raw_off.div_ceil(8) * 8; // 8-align value aggregates
                    let o = raw_base + raw_off;
                    raw_off += sz;
                    o
                }
                Repr::Unit => raw_base + raw_off,
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
            is_extern: false,
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

        // Transitive immutability: calling a `mut self` method requires the
        // receiver to be rooted in a `mut` binding (see docs/mutability.md).
        if entry.method.self_is_mut {
            self.check_mutable_root(recv, "the receiver of a `mut self` method through", span)?;
        }

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
            checks.push((declared.clone(), at, a.span));
            cargs.push(ca);
        }
        // Re-apply now that all type vars are known, pointing each error at the
        // argument that caused it.
        for (declared, actual, arg_span) in &checks {
            check_assignable(actual, &apply_subst(declared, &isubst), *arg_span)?;
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

    /// Lower an associated (self-less) function call `Type::func(args)`. Like a
    /// method call but with no receiver: impl-level and method-level generics
    /// (and `Self`) are inferred from the expected return type and the arguments.
    fn assoc_call(
        &mut self,
        base: &str,
        entry: &crate::types::MethodEntry,
        args: &[Expr],
        expected: Option<&Ty>,
        span: Span,
    ) -> LResult<(CoreExpr, Ty)> {
        let m = entry.method.clone();
        if args.len() != m.params.len() {
            return err(
                format!("`{}::{}` expects {} args, got {}", base, m.name, m.params.len(), args.len()),
                span,
            );
        }
        // SEND/SYNC: `Thread::spawn(closure)` shares the closure's captures with
        // the parent thread, so each must be `Sync`. We check at THIS call site
        // (the closure literal is the argument here; by the time the prelude
        // wrapper forwards it to the `thread_spawn` intrinsic it's just a param).
        // See `docs/threads.md`.
        if base == "Thread" && m.name == "spawn" && !args.is_empty() {
            self.check_spawn_captures_sync(&args[0])?;
        }
        let mgparams: Vec<String> = m.generics.params.iter().map(|p| p.name.clone()).collect();
        // All generic names in scope for the signature: impl generics + method
        // generics + Self.
        let mut scope_params = entry.impl_generics.clone();
        scope_params.extend(mgparams.clone());
        scope_params.push("Self".to_string());

        // `Self` is the impl's self type with its impl-generics left as vars.
        let impl_self = lower_type(&entry.self_ty, &entry.impl_generics, self.ctx).map_err(conv)?;
        let mut subst: HashMap<String, Ty> = HashMap::new();
        subst.insert("Self".to_string(), impl_self.clone());

        // Seed inference from the expected return type, if concrete.
        if let Some(exp) = expected {
            if let Some(ret) = &m.ret {
                let ret_decl = lower_type(ret, &scope_params, self.ctx).map_err(conv)?;
                unify_infer(&ret_decl, exp, &mut subst);
            }
        }

        // Lower args, inferring impl/method generics from their concrete types.
        let mut cargs = Vec::new();
        let mut checks = Vec::new();
        for (a, p) in args.iter().zip(&m.params) {
            let declared = lower_type(&p.ty, &scope_params, self.ctx).map_err(conv)?;
            let pre = apply_subst(&declared, &subst);
            let (ca, at) = self.expr(a, hint(&pre))?;
            unify_infer(&pre, &at, &mut subst);
            checks.push((declared, at, a.span));
            cargs.push(ca);
        }
        for (declared, actual, arg_span) in &checks {
            check_assignable(actual, &apply_subst(declared, &subst), *arg_span)?;
        }

        let ret_ty = match &m.ret {
            Some(t) => {
                let raw = lower_type(t, &scope_params, self.ctx).map_err(conv)?;
                apply_subst(&raw, &subst)
            }
            None => Ty::unit(),
        };

        // Mangle: <Self-type>::func$<impl-and-method-generic-args>.
        let self_concrete = apply_subst(&impl_self, &subst);
        let extra: Vec<Ty> = entry.impl_generics.iter().chain(mgparams.iter())
            .filter_map(|g| subst.get(g).cloned())
            .collect();
        let mangled = mangle(&format!("{}.{}", ty_mangle(&self_concrete), m.name), &extra);

        let job = Job { f: m, subst, mangled, self_ty: None };
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

    fn variant_in(&self, enum_name: &str, vlast: &str, variants: &[(u32, Vec<Ty>)], span: Span) -> LResult<(u32, Vec<Ty>)> {
        // Resolve the variant by name WITHIN this enum — variant names are
        // per-enum, not global (two enums may share a variant name, e.g.
        // `TokKind::Not` and `UnOp::Not`). Looking it up in the global
        // `ctx.variants` map (keyed partly by bare name) conflated same-named
        // variants across enums; scope it to `enum_name`'s own declaration.
        if let Some(edef) = self.ctx.enums.get(enum_name) {
            if let Some(idx) = edef.variants.iter().position(|v| v.name == vlast) {
                let tag = idx as u32;
                if let Some((t, tys)) = variants.iter().find(|(i, _)| *i == tag) {
                    return Ok((*t, tys.clone()));
                }
            }
        }
        err(format!("unknown variant `{}` of `{}`", vlast, enum_name), span)
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
    /// If `a` is an `as_c_bytes(x)` call, lower it to an `AsCBytes` node (a
    /// `RawPtr` to a stack copy of `x`'s contents). `x` may be a `String` or a
    /// scalar `Array<T>`. Returns `Ok(None)` if `a` is not `as_c_bytes`. Errors
    /// if the declared parameter is not `RawPtr` or if `x` is not a String/scalar
    /// array. When the extern parameter is `mut`, the copy is also written BACK
    /// into the array after the call (copy-out, e.g. `read(fd, buf, n)`). Only
    /// called for direct extern-call arguments — see the caller and `docs/ffi.md`.
    fn try_as_c_bytes_arg(&mut self, a: &Expr, declared: &Ty, param_is_mut: bool) -> LResult<Option<(CoreExpr, Ty)>> {
        let ExprKind::Call(callee, cargs) = &*a.kind else { return Ok(None) };
        let ExprKind::Path(p) = &*callee.kind else { return Ok(None) };
        if p.last() != "as_c_bytes" { return Ok(None); }
        // Don't shadow a user function of the same name.
        if self.ctx.fns.keys().any(|k| k.rsplit("::").next().unwrap() == "as_c_bytes") {
            return Ok(None);
        }
        if cargs.len() != 1 {
            return err("`as_c_bytes` takes 1 argument (a String or scalar Array)", a.span);
        }
        if !matches!(declared, Ty::Prim(Prim::RawPtr)) {
            return err(
                "`as_c_bytes` produces a `RawPtr`; the extern parameter here is not `RawPtr`",
                a.span,
            );
        }
        let (cs, sty) = self.expr(&cargs[0], None)?;
        // String: bytes (+ NUL); element stride 1.
        if matches!(sty, Ty::Prim(Prim::Str)) {
            return Ok(Some((
                CoreExpr::new(
                    CoreExprKind::AsCBytes {
                        src: Box::new(cs), elem: ScalarRepr::U8, is_string: true, copy_out: false,
                    },
                    Repr::Scalar(ScalarRepr::Ptr),
                ),
                Ty::Prim(Prim::RawPtr),
            )));
        }
        // Scalar Array<T>: contiguous T elements, stride = sizeof(T).
        if let Some(elem_ty) = array_elem_ty(&sty) {
            let elem_repr = self.repr_of(&elem_ty, cargs[0].span)?;
            let Repr::Scalar(elem) = elem_repr else {
                return err(
                    format!("`as_c_bytes` requires a String or an array of scalars, found `Array<{}>`",
                        ty_display(&elem_ty)),
                    cargs[0].span,
                );
            };
            return Ok(Some((
                CoreExpr::new(
                    CoreExprKind::AsCBytes {
                        src: Box::new(cs), elem, is_string: false, copy_out: param_is_mut,
                    },
                    Repr::Scalar(ScalarRepr::Ptr),
                ),
                Ty::Prim(Prim::RawPtr),
            )));
        }
        err(
            format!("`as_c_bytes` requires a `String` or a scalar `Array`, found `{}`", ty_display(&sty)),
            cargs[0].span,
        )
    }

    /// If `declared` is an `extern fn(..)` callback type and `a` names a (non-
    /// generic) gc-rust function, resolve it and emit a `CallbackPtr` (a `RawPtr`
    /// to a synthesized C trampoline). Returns `Ok(None)` if `declared` is not a
    /// callback type. Errors if `a` is not a plain function name, the function is
    /// generic/unknown, or its signature doesn't match. See `docs/ffi.md`.
    fn try_callback_arg(&mut self, a: &Expr, declared: &Ty) -> LResult<Option<(CoreExpr, Ty)>> {
        let Ty::ExternFn { params: cb_params, ret: cb_ret } = declared else { return Ok(None) };
        // The argument must be a bare path naming a function.
        let ExprKind::Path(path) = &*a.kind else {
            return err("a callback argument must be a named function", a.span);
        };
        let name = path.last();
        let fq = self.ctx.use_aliases.get(name)
            .filter(|fq| self.ctx.fns.contains_key(*fq))
            .cloned()
            .or_else(|| self.ctx.fns.keys().find(|k| k.rsplit("::").next().unwrap() == name).cloned());
        let Some(fq) = fq else {
            return err(format!("unknown function `{}` for callback", name), a.span);
        };
        let f = self.ctx.fns[&fq].clone();
        if !f.generics.params.is_empty() {
            return err(format!("callback function `{}` cannot be generic", name), a.span);
        }
        // Check the function's signature matches the expected callback type.
        let gparams: Vec<String> = Vec::new();
        let f_params: Vec<Ty> = f.params.iter()
            .map(|p| lower_type(&p.ty, &gparams, self.ctx).map_err(conv))
            .collect::<Result<_, _>>()?;
        let f_ret = match &f.ret {
            Some(t) => lower_type(t, &gparams, self.ctx).map_err(conv)?,
            None => Ty::unit(),
        };
        if f_params.len() != cb_params.len() {
            return err(
                format!("callback `{}` takes {} argument(s), but the expected type takes {}",
                    name, f_params.len(), cb_params.len()),
                a.span,
            );
        }
        for (fp, cp) in f_params.iter().zip(cb_params.iter()) {
            check_assignable(fp, cp, a.span)?;
        }
        check_assignable(&f_ret, cb_ret, a.span)?;
        let mangled = mangle(&fq, &[]);
        let job = Job { f, subst: HashMap::new(), mangled, self_ty: None };
        let fid = self.mono.intern(job, self.count);
        Ok(Some((
            CoreExpr::new(CoreExprKind::CallbackPtr(fid), Repr::Scalar(ScalarRepr::Ptr)),
            declared.clone(),
        )))
    }

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
        // Associated (self-less) function call: `Type::func(args)`. The path's
        // second-to-last segment names the type; look up an impl item of that
        // name with NO `self`. (Methods with `self` go through `method_call`.)
        if path.segments.len() >= 2 {
            let base = path.segments[path.segments.len() - 2].clone();
            if let Some(entry) = self.ctx.methods.get(&(base.clone(), name.to_string())).cloned() {
                if !entry.method.has_self {
                    return self.assoc_call(&base, &entry, args, expected, span);
                }
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
        // String intrinsics — built-in `String` ops backed by runtime externs.
        // Skipped if shadowed by a user-defined function of the same name.
        if !self.ctx.fns.keys().any(|k| k.rsplit("::").next().unwrap() == name) {
            match name {
                "print_str" | "print" | "str_len" | "str_eq" | "str_concat"
                | "str_get" | "str_substring" | "to_string" | "str_to_float" | "str_hash"
                | "read_file" | "float_bits" | "char_to_str" => {
                    return self.str_intrinsic(name, args, span);
                }
                "type_id_of" | "type_name_of" | "field_count" | "field_name" | "field_kind"
                | "field_i64" => {
                    return self.reflect_intrinsic(name, args, span);
                }
                _ => {}
            }
        }
        // `as_c_bytes` is only valid as a DIRECT argument to an extern call (it's
        // intercepted there). Reaching it through the general expression path
        // means it was used somewhere else — reject it, because the stack copy it
        // produces is only valid for the duration of that one call.
        if name == "as_c_bytes"
            && !self.ctx.fns.keys().any(|k| k.rsplit("::").next().unwrap() == "as_c_bytes")
        {
            return err(
                "`as_c_bytes(s)` may only appear as a direct argument to an `extern \"C\"` call \
                 (its pointer is valid only for that call — see docs/ffi.md)",
                span,
            );
        }
        // Atom<T> intrinsics. `atom_load(a) -> T` (atomic), `atom_cas(a, old,
        // new) -> bool` (atomic CAS + write barrier). The atom is an ordinary
        // heap object `{ value: T }`; swap!'s retry loop is in the prelude over
        // these (so old/new/atom are frame roots → GC-safe). See docs/threads.md.
        if (name == "atom_load" || name == "atom_cas")
            && !self.ctx.fns.keys().any(|k| k.rsplit("::").next().unwrap() == name)
        {
            // First arg is the atom; recover its element type T from `Atom<T>`.
            let (catom, atom_ty) = self.expr(&args[0], None)?;
            let elem_ty = match &atom_ty {
                Ty::Named { name: n, args: ta } if n.rsplit("::").next() == Some("Atom") && !ta.is_empty() => ta[0].clone(),
                _ => return err(format!("`{}` requires an `Atom<T>`, found `{}`", name, ty_display(&atom_ty)), args[0].span),
            };
            let elem = self.repr_of(&elem_ty, args[0].span)?;
            if name == "atom_load" {
                if args.len() != 1 { return err("`atom_load` takes 1 argument", span); }
                return Ok((
                    CoreExpr::new(CoreExprKind::AtomLoad { atom: Box::new(catom), elem: elem.clone() }, elem),
                    elem_ty,
                ));
            }
            // atom_cas(a, old, new) -> bool
            if args.len() != 3 { return err("`atom_cas` takes 3 arguments (atom, old, new)", span); }
            let (cold, oldt) = self.expr(&args[1], Some(&elem_ty))?;
            check_assignable(&oldt, &elem_ty, args[1].span)?;
            let (cnew, newt) = self.expr(&args[2], Some(&elem_ty))?;
            check_assignable(&newt, &elem_ty, args[2].span)?;
            return Ok((
                CoreExpr::new(
                    CoreExprKind::AtomCas { atom: Box::new(catom), old: Box::new(cold), new: Box::new(cnew) },
                    Repr::Scalar(ScalarRepr::Bool),
                ),
                Ty::bool(),
            ));
        }
        // Channel intrinsics. `chan_new(cap) -> RawPtr` (control block);
        // `chan_send(buf, ctrl, v) -> i64`; `chan_recv(buf, ctrl) -> T`. `buf` is
        // the on-heap element Array<T>; the GC traces queued values via it.
        if name == "chan_new"
            && !self.ctx.fns.keys().any(|k| k.rsplit("::").next().unwrap() == "chan_new")
        {
            if args.len() != 1 { return err("`chan_new` takes 1 argument (capacity)", span); }
            let (ccap, _) = self.expr(&args[0], Some(&Ty::i64()))?;
            return Ok((
                CoreExpr::new(CoreExprKind::RuntimeCall { func: "ai_chan_new", args: vec![ccap], ret: Repr::Scalar(ScalarRepr::Ptr) }, Repr::Scalar(ScalarRepr::Ptr)),
                Ty::Prim(Prim::RawPtr),
            ));
        }
        if name == "chan_send"
            && !self.ctx.fns.keys().any(|k| k.rsplit("::").next().unwrap() == "chan_send")
        {
            if args.len() != 3 { return err("`chan_send` takes 3 arguments (buf, ctrl, value)", span); }
            let (cbuf, _) = self.expr(&args[0], None)?;
            let (cctrl, _) = self.expr(&args[1], None)?;
            let (cval, _) = self.expr(&args[2], None)?;
            return Ok((
                CoreExpr::new(CoreExprKind::ChanSend { buf: Box::new(cbuf), ctrl: Box::new(cctrl), value: Box::new(cval) }, Repr::Scalar(ScalarRepr::I64)),
                Ty::i64(),
            ));
        }
        if name == "chan_recv"
            && !self.ctx.fns.keys().any(|k| k.rsplit("::").next().unwrap() == "chan_recv")
        {
            if args.len() != 2 { return err("`chan_recv` takes 2 arguments (buf, ctrl)", span); }
            let (cbuf, bty) = self.expr(&args[0], None)?;
            let (cctrl, _) = self.expr(&args[1], None)?;
            let elem_ty = array_elem_ty(&bty)
                .ok_or_else(|| LowerError { msg: format!("`chan_recv` buffer must be an Array, found `{}`", ty_display(&bty)), span: args[0].span })?;
            let elem = self.repr_of(&elem_ty, span)?;
            return Ok((
                CoreExpr::new(CoreExprKind::ChanRecv { buf: Box::new(cbuf), ctrl: Box::new(cctrl), elem: elem.clone() }, elem),
                elem_ty,
            ));
        }
        if (name == "chan_sender_clone" || name == "chan_sender_drop")
            && !self.ctx.fns.keys().any(|k| k.rsplit("::").next().unwrap() == name)
        {
            if args.len() != 1 { return err(format!("`{}` takes 1 argument (ctrl)", name), span); }
            let (cctrl, _) = self.expr(&args[0], None)?;
            let func = if name == "chan_sender_clone" { "ai_chan_sender_clone" } else { "ai_chan_sender_drop" };
            return Ok((
                CoreExpr::new(CoreExprKind::RuntimeCall { func, args: vec![cctrl], ret: Repr::Unit }, Repr::Unit),
                Ty::unit(),
            ));
        }
        // AtomicI64 intrinsics → runtime externs. The handle is a RawPtr.
        // `atomic_i64_new(v)->RawPtr`, `..._load(a)->i64`, `..._store(a,v)->i64`,
        // `..._fetch_add(a,d)->i64`, `..._cas(a,expected,new)->i64` (1/0).
        {
            let atomic_intr: Option<(&'static str, &[ArgKind], Repr, Ty)> = match name {
                "atomic_i64_new" => Some(("ai_atomic_i64_new", &[ArgKind::Int], Repr::Scalar(ScalarRepr::Ptr), Ty::Prim(Prim::RawPtr))),
                "atomic_i64_load" => Some(("ai_atomic_i64_load", &[ArgKind::Ptr], Repr::Scalar(ScalarRepr::I64), Ty::i64())),
                "atomic_i64_store" => Some(("ai_atomic_i64_store", &[ArgKind::Ptr, ArgKind::Int], Repr::Scalar(ScalarRepr::I64), Ty::i64())),
                "atomic_i64_fetch_add" => Some(("ai_atomic_i64_fetch_add", &[ArgKind::Ptr, ArgKind::Int], Repr::Scalar(ScalarRepr::I64), Ty::i64())),
                "atomic_i64_cas" => Some(("ai_atomic_i64_cas_marker", &[ArgKind::Ptr, ArgKind::Int, ArgKind::Int], Repr::Scalar(ScalarRepr::I64), Ty::i64())),
                _ => None,
            };
            if let Some((func, kinds, ret, ret_ty)) = atomic_intr {
                if !self.ctx.fns.keys().any(|k| k.rsplit("::").next().unwrap() == name) {
                    let func = if func == "ai_atomic_i64_cas_marker" { "ai_atomic_i64_compare_and_set" } else { func };
                    if args.len() != kinds.len() {
                        return err(format!("`{}` takes {} argument(s)", name, kinds.len()), span);
                    }
                    let mut cargs = Vec::new();
                    for (a, k) in args.iter().zip(kinds) {
                        let (ca, at) = self.expr(a, None)?;
                        match k {
                            ArgKind::Int if !matches!(at, Ty::Prim(Prim::I64)) =>
                                return err(format!("`{}` argument must be i64, found `{}`", name, ty_display(&at)), a.span),
                            ArgKind::Ptr if !matches!(at, Ty::Prim(Prim::RawPtr)) =>
                                return err(format!("`{}` argument must be a RawPtr handle, found `{}`", name, ty_display(&at)), a.span),
                            _ => {}
                        }
                        cargs.push(ca);
                    }
                    return Ok((CoreExpr::new(CoreExprKind::RuntimeCall { func, args: cargs, ret: ret.clone() }, ret), ret_ty));
                }
            }
        }
        // `thread_spawn(closure)` → spawn an OS thread; returns a RawPtr handle.
        // The closure must be a no-arg `fn() -> i64` (M2 restricts results to i64).
        if name == "thread_spawn"
            && !self.ctx.fns.keys().any(|k| k.rsplit("::").next().unwrap() == "thread_spawn")
        {
            if args.len() != 1 {
                return err("`thread_spawn` takes 1 argument (a `fn() -> i64` closure)", span);
            }
            // SEND/SYNC: a spawned closure shares its captures with the parent
            // thread, so every captured value must be `Sync` (safe to share). A
            // captured mutable reference type is NOT Sync → reject it, pointing at
            // a concrete fix (use Atom/AtomicI64 or build from Sync parts). See
            // `docs/threads.md`. (Deeply-immutable values are Sync automatically,
            // so the common case is unaffected.)
            self.check_spawn_captures_sync(&args[0])?;
            let (cf, ft) = self.expr(&args[0], None)?;
            match &ft {
                Ty::Fn { params, ret } if params.is_empty() && matches!(**ret, Ty::Prim(Prim::I64)) => {}
                _ => return err(
                    format!("`thread_spawn` requires a `fn() -> i64` closure, found `{}`", ty_display(&ft)),
                    args[0].span,
                ),
            }
            return Ok((
                CoreExpr::new(CoreExprKind::ThreadSpawn(Box::new(cf)), Repr::Scalar(ScalarRepr::Ptr)),
                Ty::Prim(Prim::RawPtr),
            ));
        }
        // `thread_join(handle)` → block; returns the thread's i64 result.
        if name == "thread_join"
            && !self.ctx.fns.keys().any(|k| k.rsplit("::").next().unwrap() == "thread_join")
        {
            if args.len() != 1 {
                return err("`thread_join` takes 1 argument (a thread handle)", span);
            }
            let (ch, ht) = self.expr(&args[0], None)?;
            if !matches!(ht, Ty::Prim(Prim::RawPtr)) {
                return err(
                    format!("`thread_join` requires a thread handle (RawPtr), found `{}`", ty_display(&ht)),
                    args[0].span,
                );
            }
            return Ok((
                CoreExpr::new(CoreExprKind::ThreadJoin(Box::new(ch)), Repr::Scalar(ScalarRepr::I64)),
                Ty::i64(),
            ));
        }
        // `thread_sleep(ms)` / `thread_yield()` / `thread_current_id()`.
        if name == "thread_sleep"
            && !self.ctx.fns.keys().any(|k| k.rsplit("::").next().unwrap() == "thread_sleep")
        {
            if args.len() != 1 { return err("`thread_sleep` takes 1 argument (millis)", span); }
            let (cm, _) = self.expr(&args[0], Some(&Ty::i64()))?;
            return Ok((
                CoreExpr::new(CoreExprKind::ThreadSleep(Box::new(cm)), Repr::Scalar(ScalarRepr::I64)),
                Ty::i64(),
            ));
        }
        if name == "thread_yield"
            && !self.ctx.fns.keys().any(|k| k.rsplit("::").next().unwrap() == "thread_yield")
        {
            if !args.is_empty() { return err("`thread_yield` takes no arguments", span); }
            return Ok((
                CoreExpr::new(CoreExprKind::ThreadYield, Repr::Scalar(ScalarRepr::I64)),
                Ty::i64(),
            ));
        }
        if name == "thread_current_id"
            && !self.ctx.fns.keys().any(|k| k.rsplit("::").next().unwrap() == "thread_current_id")
        {
            if !args.is_empty() { return err("`thread_current_id` takes no arguments", span); }
            return Ok((
                CoreExpr::new(CoreExprKind::ThreadCurrentId, Repr::Scalar(ScalarRepr::I64)),
                Ty::i64(),
            ));
        }
        // `ptr_read_i64(p)` → load an i64 through a RawPtr (for FFI callbacks).
        if name == "ptr_read_i64"
            && !self.ctx.fns.keys().any(|k| k.rsplit("::").next().unwrap() == "ptr_read_i64")
        {
            if args.len() != 1 {
                return err("`ptr_read_i64` takes 1 argument (a RawPtr)", span);
            }
            let (cp, pt) = self.expr(&args[0], None)?;
            if !matches!(pt, Ty::Prim(Prim::RawPtr)) {
                return err(
                    format!("`ptr_read_i64` requires a `RawPtr`, found `{}`", ty_display(&pt)),
                    args[0].span,
                );
            }
            return Ok((
                CoreExpr::new(CoreExprKind::PtrReadI64(Box::new(cp)), Repr::Scalar(ScalarRepr::I64)),
                Ty::i64(),
            ));
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
        // Resolution order: a `use` alias for this name wins (it disambiguates
        // same-named items across modules); otherwise fall back to the unique
        // last-segment match.
        let fq = self.ctx.use_aliases.get(name)
            .filter(|fq| self.ctx.fns.contains_key(*fq))
            .cloned()
            .or_else(|| self.ctx.fns.keys().find(|k| k.rsplit("::").next().unwrap() == name).cloned());
        let Some(fq) = fq else {
            let names: Vec<&str> = self.ctx.fns.keys().map(|k| k.rsplit("::").next().unwrap()).collect();
            let hint = closest(name, names.into_iter())
                .map(|s| format!(" (did you mean `{}`?)", s))
                .unwrap_or_default();
            return err(format!("unknown function `{}`{}", name, hint), span);
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
            // FFI `as_c_bytes(s)` is legal ONLY as a direct argument to an extern
            // call whose corresponding parameter is `RawPtr`. Intercept it here;
            // any other use falls through to `self.expr`, which rejects it.
            if f.is_extern {
                let param_is_mut = f.params.get(i).map(|p| p.is_mut).unwrap_or(false);
                if let Some((ca, at)) = self.try_as_c_bytes_arg(a, declared, param_is_mut)? {
                    unify_infer(declared, &at, &mut targ);
                    lowered[i] = Some((ca, at));
                    continue;
                }
                if let Some((ca, at)) = self.try_callback_arg(a, declared)? {
                    lowered[i] = Some((ca, at));
                    continue;
                }
            }
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
        for (i, slot) in lowered.into_iter().enumerate() {
            let (ca, at) = slot.expect("every argument lowered");
            // Carry each argument's own span so a type error underlines that
            // argument, not the whole call.
            arg_tys.push((/* declared placeholder */ at.clone(), at, args[i].span));
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

        // Check each argument against its substituted declared type, pointing the
        // error at the argument's own span.
        let subst: HashMap<String, Ty> = gparams.iter().cloned().zip(inst_args.iter().cloned()).collect();
        for (declared, actual, arg_span) in &arg_tys {
            check_assignable(actual, &apply_subst(declared, &subst), *arg_span)?;
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
        Ty::Fn { params, ret } | Ty::ExternFn { params, ret } => params.iter().any(has_var) || has_var(ret),
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
        Prim::RawPtr => ScalarRepr::Ptr,
        Prim::Str | Prim::Unit => return None,
    })
}

fn is_int_prim(p: Prim) -> bool {
    matches!(p, Prim::I8 | Prim::I16 | Prim::I32 | Prim::I64 | Prim::U8 | Prim::U16 | Prim::U32 | Prim::U64)
}
fn is_numeric(ty: &Ty) -> bool {
    matches!(ty, Ty::Prim(p) if is_int_prim(*p) || matches!(p, Prim::F32 | Prim::F64))
}

/// The "never" type, produced by diverging expressions (`return`, and a `loop`
/// with no value-carrying `break`). It is assignable to ANY expected type — a
/// `loop {}` whose only exits are `return` can stand as a function body of any
/// return type. Encoded as a reserved `Ty::Infer` sentinel to avoid a new `Ty`
/// variant rippling through every match. See `check_assignable` / `is_never`.
fn never_ty() -> Ty { Ty::Infer(u32::MAX) }
fn is_never(t: &Ty) -> bool { matches!(t, Ty::Infer(n) if *n == u32::MAX) }

fn check_assignable(actual: &Ty, expected: &Ty, span: Span) -> LResult<()> {
    if is_never(actual) || actual == expected || actual.is_unit() && expected.is_unit() {
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

    // ---- Mutability discipline (docs/mutability.md) ----------------------
    // Immutable by default for ALL values; `mut` is required at every binder.

    fn err_msg(src: &str) -> String {
        lower(src).err().expect("expected a lowering error").msg
    }

    #[test]
    fn associated_fn_resolves() {
        // `Type::func` with no `self` resolves to the impl's associated function.
        assert!(lower(
            "struct Foo { x: i64 } \
             impl Foo { fn make(n: i64) -> Foo { Foo { x: n } } fn get(self) -> i64 { self.x } } \
             fn main() -> i64 { let f = Foo::make(42); f.get() }",
        ).is_ok());
    }

    #[test]
    fn generic_associated_fn_infers_from_arg() {
        // `Box::wrap(99)` infers `T = i64` from the argument.
        assert!(lower(
            "struct Box<T> { val: T } \
             impl<T> Box<T> { fn wrap(x: T) -> Box<T> { Box { val: x } } fn get(self) -> T { self.val } } \
             fn main() -> i64 { let b = Box::wrap(99); b.get() }",
        ).is_ok());
    }

    #[test]
    fn reassign_immutable_let_rejected() {
        let m = err_msg("fn main() -> i64 { let x = 5; x = 6; x }");
        assert!(m.contains("immutable"), "{m}");
    }

    #[test]
    fn reassign_let_mut_ok() {
        assert!(lower("fn main() -> i64 { let mut x = 5; x = 6; x }").is_ok());
    }

    #[test]
    fn field_assign_through_immutable_rejected() {
        let m = err_msg(
            "struct P { x: i64 } fn main() -> i64 { let p = P { x: 1 }; p.x = 9; p.x }",
        );
        assert!(m.contains("immutable"), "{m}");
    }

    #[test]
    fn field_assign_through_mut_ok() {
        assert!(lower(
            "struct P { x: i64 } fn main() -> i64 { let mut p = P { x: 1 }; p.x = 9; p.x }",
        )
        .is_ok());
    }

    #[test]
    fn mut_self_method_on_immutable_receiver_rejected() {
        let m = err_msg(
            "struct C { n: i64 } \
             impl C { fn bump(mut self) -> i64 { self.n = self.n + 1; self.n } } \
             fn main() -> i64 { let c = C { n: 0 }; c.bump() }",
        );
        assert!(m.contains("immutable"), "{m}");
    }

    #[test]
    fn mut_self_method_on_mut_receiver_ok() {
        assert!(lower(
            "struct C { n: i64 } \
             impl C { fn bump(mut self) -> i64 { self.n = self.n + 1; self.n } } \
             fn main() -> i64 { let mut c = C { n: 0 }; c.bump() }",
        )
        .is_ok());
    }

    #[test]
    fn plain_self_cannot_mutate_fields() {
        let m = err_msg(
            "struct C { n: i64 } \
             impl C { fn bad(self) -> i64 { self.n = self.n + 1; self.n } } \
             fn main() -> i64 { let mut c = C { n: 0 }; c.bad() }",
        );
        assert!(m.contains("mut self"), "{m}");
    }

    #[test]
    fn assign_through_immutable_param_rejected() {
        let m = err_msg(
            "struct P { x: i64 } fn set(p: P) -> i64 { p.x = 5; p.x } \
             fn main() -> i64 { let q = P { x: 1 }; set(q) }",
        );
        assert!(m.contains("immutable"), "{m}");
    }

    #[test]
    fn assign_through_mut_param_ok() {
        assert!(lower(
            "struct P { x: i64 } fn set(mut p: P) -> i64 { p.x = 5; p.x } \
             fn main() -> i64 { let q = P { x: 1 }; set(q) }",
        )
        .is_ok());
    }

    // ---- Phase 2: match completion, deferred let, tuple immutability ------

    #[test]
    fn scalar_match_without_wildcard_rejected() {
        let m = err_msg("fn f(n: i64) -> i64 { match n { 0 => 1, 1 => 2 } } fn main() -> i64 { f(0) }");
        assert!(m.contains("non-exhaustive"), "{m}");
    }

    #[test]
    fn enum_missing_variant_rejected() {
        let m = err_msg("enum E { A, B, C } fn f(e: E) -> i64 { match e { E::A => 1, E::B => 2 } } fn main() -> i64 { f(E::A) }");
        assert!(m.contains("non-exhaustive"), "{m}");
    }

    #[test]
    fn guarded_variant_does_not_cover() {
        // `E::B if ...` is guarded, so it doesn't make B covered: still non-exhaustive.
        let m = err_msg("enum E { A, B } fn f(e: E) -> i64 { match e { E::A => 1, E::B if true => 2 } } fn main() -> i64 { f(E::A) }");
        assert!(m.contains("non-exhaustive"), "{m}");
    }

    #[test]
    fn immutable_deferred_let_rejected() {
        let m = err_msg("fn main() -> i64 { let x: i64; x = 1; x }");
        assert!(m.contains("mut"), "{m}");
    }

    #[test]
    fn deferred_let_needs_type_annotation() {
        let m = err_msg("fn main() -> i64 { let mut x; x = 1; x }");
        assert!(m.contains("type annotation"), "{m}");
    }

    #[test]
    fn tuple_field_assignment_rejected() {
        let m = err_msg("fn main() -> i64 { let mut t = (3, 4); t.0 = 9; t.0 }");
        assert!(m.contains("immutable") || m.contains("rebuild"), "{m}");
    }

    // ---- Phase 5: diagnostics quality ------------------------------------

    #[test]
    fn unknown_variable_suggests_close_name() {
        let m = err_msg("fn main() -> i64 { let count = 5; cont + 1 }");
        assert!(m.contains("did you mean `count`?"), "{m}");
    }

    #[test]
    fn unknown_function_suggests_close_name() {
        let m = err_msg("fn compute() -> i64 { 5 } fn main() -> i64 { comput() }");
        assert!(m.contains("did you mean `compute`?"), "{m}");
    }

    #[test]
    fn no_suggestion_when_nothing_close() {
        // A wildly different name shouldn't get a bogus suggestion.
        let m = err_msg("fn main() -> i64 { let count = 5; zzzzzzzz + 1 }");
        assert!(!m.contains("did you mean"), "{m}");
    }

    #[test]
    fn arg_type_error_points_at_argument() {
        // The error span should be the argument `true`, not the whole call. We
        // can't see the span text here, but lower returns the span; verify the
        // message is the arg type mismatch (the span fix is covered in tests/).
        let e = lower("fn f(x: i64) -> i64 { x } fn main() -> i64 { f(true) }").err().unwrap();
        assert!(e.msg.contains("expected `i64`, found `bool`"), "{}", e.msg);
        // The span must be inside the `true` token region, not span 0.
        assert!(e.span.start > 0);
    }

    #[test]
    fn edit_distance_works() {
        assert_eq!(edit_distance("count", "cont"), 1);
        assert_eq!(edit_distance("vec_push", "vec_pushh"), 1);
        assert_eq!(edit_distance("abc", "abc"), 0);
        assert_eq!(edit_distance("abc", "xyz"), 3);
    }

    // ---- multiple-error reporting (check_program) ------------------------

    fn check_errs(src: &str) -> Vec<LowerError> {
        let m = parse_module(&lex(src).unwrap()).unwrap();
        let r = resolve_module(m).unwrap();
        check_program(&r.globals).err().unwrap_or_default()
    }

    #[test]
    fn check_program_collects_all_errors() {
        // Three independently-broken functions — all three should be reported,
        // not just the first.
        let src = "fn a() -> i64 { true } \
                   fn b() -> i64 { nope + 1 } \
                   fn c() -> i64 { let x = 5; x = 6; x } \
                   fn main() -> i64 { 0 }";
        let errs = check_errs(src);
        assert_eq!(errs.len(), 3, "expected 3 errors, got: {:?}", errs.iter().map(|e| &e.msg).collect::<Vec<_>>());
    }

    #[test]
    fn check_program_ok_when_clean() {
        let src = "fn a() -> i64 { 1 } fn b() -> i64 { a() + 1 } fn main() -> i64 { b() }";
        assert!(check_errs(src).is_empty());
    }

    #[test]
    fn check_program_skips_generics_but_catches_concrete() {
        // A generic fn is skipped by the eager pass (needs instantiation), but a
        // broken concrete fn is still caught.
        let src = "fn id<T>(x: T) -> T { x } \
                   fn broken() -> i64 { true } \
                   fn main() -> i64 { id(0) }";
        let errs = check_errs(src);
        assert_eq!(errs.len(), 1);
        assert!(errs[0].msg.contains("found `bool`"));
    }
}
