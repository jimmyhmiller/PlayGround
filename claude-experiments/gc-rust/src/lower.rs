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

#[derive(Debug)]
pub struct LowerError {
    pub msg: String,
    pub span: Span,
}
type LResult<T> = Result<T, LowerError>;

fn err<T>(msg: impl Into<String>, span: Span) -> LResult<T> {
    Err(LowerError { msg: msg.into(), span })
}

/// A function instantiation: a fully-qualified name plus concrete type
/// arguments (empty for a non-generic function). This is the unit of
/// monomorphization — each distinct instantiation is lowered once.
#[derive(Clone, PartialEq, Eq, Hash)]
struct Inst {
    fq: String,
    args: Vec<Ty>,
}

impl Inst {
    fn mangled(&self) -> String {
        if self.args.is_empty() {
            self.fq.clone()
        } else {
            let a: Vec<String> = self.args.iter().map(ty_mangle).collect();
            format!("{}${}", self.fq, a.join("_"))
        }
    }
}

/// Drives monomorphization: assigns each needed instantiation a [`FuncId`] and
/// keeps a worklist of instantiations still to be lowered.
struct Mono {
    ids: HashMap<Inst, FuncId>,
    worklist: Vec<(FuncId, Inst)>,
}

impl Mono {
    fn new() -> Self {
        Mono { ids: HashMap::new(), worklist: Vec::new() }
    }
    /// Get (or assign) the FuncId for an instantiation, queueing it for lowering
    /// the first time it's seen. `count` is the current number of allocated ids.
    fn intern(&mut self, inst: Inst, count: &mut u32) -> FuncId {
        if let Some(id) = self.ids.get(&inst) {
            return *id;
        }
        let id = *count;
        *count += 1;
        self.ids.insert(inst.clone(), id);
        self.worklist.push((id, inst));
        id
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
    let entry = mono.intern(Inst { fq: main_fq, args: vec![] }, &mut count);
    prog.entry = Some(entry);

    // Lower instantiations until the worklist drains (transitively reachable).
    while let Some((id, inst)) = mono.worklist.pop() {
        let f = ctx.fns.get(&inst.fq).cloned()
            .ok_or_else(|| LowerError { msg: format!("unknown function `{}`", inst.fq), span: Span::new(0, 0) })?;
        let subst = build_fn_subst(&f, &inst.args);
        let lowered = lower_fn(&f, &inst, &subst, &ctx, &mut reg, &mut mono, &mut count)?;
        // Ensure prog.funcs is large enough.
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

fn build_fn_subst(f: &FnDef, args: &[Ty]) -> HashMap<String, Ty> {
    f.generics.params.iter().map(|p| p.name.clone()).zip(args.iter().cloned()).collect()
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

struct FnLowerer<'a, 'r> {
    ctx: &'a TyCtx,
    func_ids: &'a HashMap<String, FuncId>,
    reg: &'r mut LayoutRegistry<'a>,
    /// Lexical scope: name → (LocalId, Ty).
    scope: Vec<HashMap<String, (LocalId, Ty)>>,
    /// All locals' reprs, indexed by LocalId.
    locals: Vec<Repr>,
    /// Parallel: each local's semantic Ty (for checking).
    local_tys: Vec<Ty>,
    ret_ty: Ty,
}

fn lower_fn<'a>(
    f: &FnDef,
    ctx: &'a TyCtx,
    func_ids: &'a HashMap<String, FuncId>,
    reg: &mut LayoutRegistry<'a>,
) -> LResult<CoreFn> {
    let generics: Vec<String> = vec![];
    let ret_ty = match &f.ret {
        Some(t) => lower_type(t, &generics, ctx).map_err(conv)?,
        None => Ty::unit(),
    };
    let mut lo = FnLowerer {
        ctx,
        func_ids,
        reg,
        scope: vec![HashMap::new()],
        locals: vec![],
        local_tys: vec![],
        ret_ty: ret_ty.clone(),
    };

    let mut params = Vec::new();
    for p in &f.params {
        let ty = lower_type(&p.ty, &generics, lo.ctx).map_err(conv)?;
        let repr = lo.repr_of(&ty, p.span)?;
        let id = lo.fresh_local(repr.clone(), ty.clone());
        lo.bind(&p.name, id, ty);
        params.push(repr);
    }

    let (body, body_ty) = lo.block(&f.body)?;
    check_assignable(&body_ty, &ret_ty, f.body.span)?;
    let ret = lo.repr_of(&ret_ty, f.span)?;

    Ok(CoreFn { name: f.name.clone(), params, ret, locals: lo.locals, body })
}

fn conv(e: crate::types::TypeError) -> LowerError {
    LowerError { msg: e.msg, span: e.span }
}

impl<'a, 'r> FnLowerer<'a, 'r> {
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
                let (ce, t) = self.expr(e, None)?;
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
                    self.variant_ctor(path, &[], e.span)
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

            ExprKind::Call(callee, args) => self.call(callee, args, e.span),

            ExprKind::If { cond, then_branch, else_branch } => {
                let (cc, ct) = self.expr(cond, Some(&Ty::bool()))?;
                check_assignable(&ct, &Ty::bool(), cond.span)?;
                let (tb, tt) = self.block(then_branch)?;
                let (eb, et) = match else_branch {
                    Some(e) => {
                        let (ce, t) = self.expr(e, None)?;
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
                let (cb, t) = self.block(b)?;
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
                self.variant_ctor(path, &[], e.span)
            }

            ExprKind::Match { scrutinee, arms } => self.match_expr(scrutinee, arms, e.span),

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
        if !s.generics.params.is_empty() {
            return err("generic struct literals need the monomorphizer (pending)", span);
        }
        // Evaluate fields in declaration order.
        let mut cfields = Vec::with_capacity(def_fields.len());
        for df in def_fields {
            let init = fields.iter().find(|f| f.name == df.name)
                .ok_or_else(|| LowerError { msg: format!("missing field `{}`", df.name), span })?;
            let fty = lower_type(&df.ty, &[], self.ctx).map_err(conv)?;
            let val_expr = match &init.value {
                Some(v) => v.clone(),
                None => Expr { kind: Box::new(ExprKind::Path(Path::single(df.name.clone(), span))), span },
            };
            let (cv, vt) = self.expr(&val_expr, Some(&fty))?;
            check_assignable(&vt, &fty, span)?;
            cfields.push(cv);
        }
        let ty = Ty::Named { name: canon.clone(), args: vec![] };
        let repr = self.repr_of(&ty, span)?;
        let kind = match repr {
            Repr::Ref(lid) => CoreExprKind::New { layout: lid, fields: cfields },
            Repr::Value(vid) => CoreExprKind::MakeValue { value: vid, fields: cfields },
            _ => return err("struct must be a ref or value type", span),
        };
        Ok((CoreExpr::new(kind, repr), ty))
    }

    /// Construct an enum variant. `path` is `Enum::Variant`; `args` are payload
    /// values (empty for a unit variant).
    fn variant_ctor(&mut self, path: &Path, args: &[Expr], span: Span) -> LResult<(CoreExpr, Ty)> {
        let key = path.segments.join("::");
        let last = path.last();
        let (enum_name, tag) = self.ctx.variants.get(&key)
            .or_else(|| self.ctx.variants.get(last))
            .cloned()
            .ok_or_else(|| LowerError { msg: format!("unknown variant `{}`", key), span })?;
        let e = self.ctx.enums.get(&enum_name).cloned().unwrap();
        if !e.generics.params.is_empty() {
            return err("generic enum construction needs the monomorphizer (pending)", span);
        }
        let variant = &e.variants[tag as usize];
        let payload_tys: Vec<Ty> = match &variant.payload {
            VariantPayload::None => vec![],
            VariantPayload::Tuple(tys) => tys.iter().map(|t| lower_type(t, &[], self.ctx)).collect::<Result<_, _>>().map_err(conv)?,
            VariantPayload::Named(fs) => fs.iter().map(|f| lower_type(&f.ty, &[], self.ctx)).collect::<Result<_, _>>().map_err(conv)?,
        };
        if args.len() != payload_tys.len() {
            return err(format!("variant `{}` expects {} args, got {}", last, payload_tys.len(), args.len()), span);
        }
        let mut cargs = Vec::new();
        for (a, t) in args.iter().zip(&payload_tys) {
            let (ca, at) = self.expr(a, Some(t))?;
            check_assignable(&at, t, a.span)?;
            cargs.push(ca);
        }
        let ty = Ty::Named { name: enum_name.clone(), args: vec![] };
        let repr = self.repr_of(&ty, span)?;
        let Repr::Ref(lid) = repr else {
            return err("value enums not yet supported in construction (pending)", span);
        };
        Ok((CoreExpr::new(CoreExprKind::MakeVariant { layout: lid, tag, fields: cargs }, repr), ty))
    }

    fn field_access(&mut self, base: &Expr, field: &FieldAccess, span: Span) -> LResult<(CoreExpr, Ty)> {
        let (cbase, bty) = self.expr(base, None)?;
        let FieldAccess::Named(fname) = field else {
            return err("tuple field access not yet supported in v0", span);
        };
        let Ty::Named { name, .. } = &bty else {
            return err("field access on a non-struct", span);
        };
        let s = self.ctx.structs.get(name).cloned()
            .ok_or_else(|| LowerError { msg: format!("`{}` is not a struct", name), span })?;
        let StructBody::Named(def_fields) = &s.body else {
            return err("field access on a non-record struct", span);
        };
        let idx = def_fields.iter().position(|f| &f.name == fname)
            .ok_or_else(|| LowerError { msg: format!("no field `{}` on `{}`", fname, name), span })?;
        let fty = lower_type(&def_fields[idx].ty, &[], self.ctx).map_err(conv)?;
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
        let Ty::Named { name: enum_name, .. } = &sty else {
            return err("match on a non-enum is not yet supported in v0", span);
        };
        // Resolve enum (user or builtin Option/Result).
        let variants: Vec<(u32, Vec<Ty>)> = self.enum_variants(enum_name)?;
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
                    let (tag, ptys) = self.variant_in(enum_name, vlast, &variants, arm.pattern.span)?;
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

    /// All (tag, payload-types) of an enum (user or builtin).
    fn enum_variants(&mut self, enum_name: &str) -> LResult<Vec<(u32, Vec<Ty>)>> {
        if let Some(e) = self.ctx.enums.get(enum_name).cloned() {
            if !e.generics.params.is_empty() {
                return err("generic match needs the monomorphizer (pending)", Span::new(0, 0));
            }
            let mut out = Vec::new();
            for (i, v) in e.variants.iter().enumerate() {
                let tys: Vec<Ty> = match &v.payload {
                    VariantPayload::None => vec![],
                    VariantPayload::Tuple(t) => t.iter().map(|t| lower_type(t, &[], self.ctx)).collect::<Result<_, _>>().map_err(conv)?,
                    VariantPayload::Named(f) => f.iter().map(|f| lower_type(&f.ty, &[], self.ctx)).collect::<Result<_, _>>().map_err(conv)?,
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
        // An enum variant constructor (`Option::Some(x)`, `Ok(v)`) is a call
        // syntactically; route it to variant construction.
        let key = path.segments.join("::");
        if self.ctx.variants.contains_key(&key) || self.ctx.variants.contains_key(name) {
            return self.variant_ctor(path, args, span);
        }
        let fq = self.ctx.fns.keys().find(|k| k.rsplit("::").next().unwrap() == name).cloned();
        let Some(fq) = fq else {
            return err(format!("unknown function `{}`", name), span);
        };
        let f = self.ctx.fns[&fq].clone();
        if !f.generics.params.is_empty() {
            return err("generic function calls not yet supported (monomorphizer pending)", span);
        }
        let Some(&fid) = self.func_ids.get(&fq) else {
            return err(format!("function `{}` has no id", fq), span);
        };
        if args.len() != f.params.len() {
            return err(format!("`{}` expects {} args, got {}", name, f.params.len(), args.len()), span);
        }
        let mut cargs = Vec::new();
        for (a, p) in args.iter().zip(&f.params) {
            let pty = lower_type(&p.ty, &[], self.ctx).map_err(conv)?;
            let (ca, at) = self.expr(a, Some(&pty))?;
            check_assignable(&at, &pty, a.span)?;
            cargs.push(ca);
        }
        let ret_ty = match &f.ret {
            Some(t) => lower_type(t, &[], self.ctx).map_err(conv)?,
            None => Ty::unit(),
        };
        let repr = self.repr_of(&ret_ty, span)?;
        Ok((CoreExpr::new(CoreExprKind::Call(fid, cargs), repr), ret_ty))
    }
}

// ---- helpers ---------------------------------------------------------------

fn block_of(e: CoreExpr) -> CoreBlock {
    CoreBlock { stmts: vec![], tail: Some(e) }
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
