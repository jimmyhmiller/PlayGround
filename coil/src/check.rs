//! Type + convention checks, **bidirectional elaboration**, and **generic
//! inference** — the front end's typing pass, which now runs *before*
//! monomorphization and drives it.
//!
//! Three jobs:
//!
//! 1. **Typing & convention checks** — field/index typing, bit-struct access,
//!    exhaustive `match`, convention well-formedness, every named type exists.
//!
//! 2. **Literal elaboration** — integer literals start flexible (default `i64`)
//!    and adopt the concrete `iN`/`uN` of their context (the other operand of an
//!    op, the other `if`/`match` branch, a store/return/call/construct target).
//!    Where a literal needs a non-default type the checker wraps it in an
//!    explicit `(cast :iN …)`, so codegen stays untouched. A literal that
//!    doesn't fit its inferred type is an error.
//!
//! 3. **Generic type-argument inference** — the checker is generic-aware: it
//!    type-checks polymorphic bodies with the type parameters as opaque types,
//!    resolves generic struct/sum applications (`(Pair i64 i64)`), and at each
//!    generic call site either reads the explicit `[T …]` type arguments or
//!    *infers* them by unifying the declared parameter types against the actual
//!    argument types. The inferred arguments are written back onto the call, so
//!    the following monomorphization pass only ever sees explicit type args and
//!    stays a pure specializer.
//!
//! `check` returns the elaborated (but still generic) `Program`; `lib.rs` feeds
//! that to `mono::monomorphize`, then codegen.

use std::collections::{HashMap, HashSet};

use crate::ast::*;
use crate::span::{Diag, Span};

struct Sig {
    /// Generic type parameters (empty for an ordinary function/extern).
    type_params: Vec<String>,
    params: Vec<Type>,
    ret: Type,
    /// Calls to externs erase pointer regions at the boundary (see `arg_ok`).
    is_extern: bool,
    /// A C variadic extern: extra trailing arguments past `params` are allowed.
    variadic: bool,
    /// The calling convention's name, and whether a function pointer can be
    /// taken to it (only native conventions, for now).
    cc: String,
    fnptr_ok: bool,
}

struct StructInfo {
    type_params: Vec<String>,
    fields: Vec<(String, Type)>,
    /// `:layout bits` structs are accessed via get/set!, not field.
    is_bits: bool,
}

struct SumInfo {
    type_params: Vec<String>,
    variants: Vec<SumVariant>,
}

/// Everything `synth` needs besides the local variable environment.
struct Cx {
    sigs: HashMap<String, Sig>,
    structs: HashMap<String, StructInfo>,
    sums: HashMap<String, SumInfo>,
    /// Variant name -> its sum type's name (variant construction routes here).
    variant_to_sum: HashMap<String, String>,
    /// Named scalar constants, consulted when a `Var` misses the local env. The
    /// stored `Expr` is the literal a reference elaborates to; the `Type` is its
    /// reported type (the declared one, or the literal default for an untyped
    /// const).
    consts: HashMap<String, (Expr, Type)>,
    /// Stack of enclosing loops (innermost last), for typing `break`/`continue`:
    /// scoping, label resolution, and unifying the value type across break sites.
    loops: std::cell::RefCell<Vec<LoopFrame>>,
}

/// One enclosing-loop frame while type-checking its body.
struct LoopFrame {
    label: Option<String>,
    /// The unified type of the values seen at this loop's `break` sites so far
    /// (`None` until the first break). The loop's own type is this, or `Never` if
    /// there is no break.
    break_ty: Option<Type>,
    /// The loop's contextual expected type, if it was checked in checking mode
    /// (e.g. a loop in tail position against the function return type). Pushed into
    /// each `break` value so a bare `(Ok v)`/`(Err e)`/literal adopts it — this is
    /// what makes `try!`'s `(return-from :try (Err e))` infer the Result type.
    expected: Option<Type>,
}

pub fn check(program: &Program) -> Result<Program, Diag> {
    // ---- type tables --------------------------------------------------------
    let mut structs: HashMap<String, StructInfo> = HashMap::new();
    for sd in &program.structs {
        let info = StructInfo {
            type_params: sd.type_params.clone(),
            fields: sd.fields.clone(),
            is_bits: matches!(sd.layout, Layout::Bits(_)),
        };
        if structs.insert(sd.name.clone(), info).is_some() {
            return Err(format!("struct '{}' defined twice", sd.name).into());
        }
    }
    let mut sums: HashMap<String, SumInfo> = HashMap::new();
    let mut variant_to_sum: HashMap<String, String> = HashMap::new();
    for sd in &program.sums {
        let info = SumInfo {
            type_params: sd.type_params.clone(),
            variants: sd.variants.clone(),
        };
        if sums.insert(sd.name.clone(), info).is_some() {
            return Err(format!("sum '{}' defined twice", sd.name).into());
        }
        for v in &sd.variants {
            if variant_to_sum.insert(v.name.clone(), sd.name.clone()).is_some() {
                return Err(format!("variant '{}' is declared in two sum types", v.name).into());
            }
        }
    }
    // ---- signatures (functions + externs) ----------------------------------
    let mut sigs: HashMap<String, Sig> = HashMap::new();
    for f in &program.funcs {
        let native = program.conventions.get(&f.cc).is_some_and(|c| !c.is_shim());
        let ftps: HashSet<String> = f.type_params.iter().cloned().collect();
        sigs.insert(
            f.name.clone(),
            Sig {
                type_params: f.type_params.clone(),
                params: f
                    .params
                    .iter()
                    .map(|p| param_ref_type(&p.ty, &structs, &sums, &ftps))
                    .collect(),
                ret: f.ret.clone(),
                is_extern: false,
                variadic: false,
                cc: f.cc.clone(),
                // a generic function has no single address, so no fnptr.
                fnptr_ok: native && f.type_params.is_empty(),
            },
        );
    }
    for e in &program.externs {
        if sigs.contains_key(&e.name) {
            return Err(format!("'{}' is declared more than once", e.name).into());
        }
        let conv = program
            .conventions
            .get(&e.cc)
            .ok_or_else(|| format!("extern '{}': unknown convention '{}'", e.name, e.cc))?;
        if conv.is_shim() {
            return Err(format!(
                "extern '{}': shim conventions for externs are not supported yet",
                e.name
            ).into());
        }
        sigs.insert(
            e.name.clone(),
            Sig {
                type_params: vec![],
                params: e.params.clone(),
                ret: e.ret.clone(),
                is_extern: true,
                variadic: e.variadic,
                cc: e.cc.clone(),
                fnptr_ok: true,
            },
        );
    }

    // ---- named constants ----------------------------------------------------
    // Each const becomes a (literal Expr, reported Type) entry. An untyped const
    // reports the literal's default type and stays freely re-inferable at the use
    // site (it IS the literal); a typed const pins the type and fit-checks now.
    let mut consts: HashMap<String, (Expr, Type)> = HashMap::new();
    for c in &program.consts {
        if cx_sig_or_const(&sigs, &consts, &c.name) {
            return Err(format!("'{}' is declared more than once", c.name).into());
        }
        let entry = const_entry(c)?;
        consts.insert(c.name.clone(), entry);
    }

    let cx = Cx {
        sigs,
        structs,
        sums,
        variant_to_sum,
        consts,
        loops: std::cell::RefCell::new(Vec::new()),
    };

    // Validate each declared const type against the type tables (must be a real,
    // concrete type — no type parameters at top level).
    for c in &program.consts {
        if let Some(ty) = &c.ty {
            validate_type(ty, &cx, &HashSet::new())
                .map_err(|e| format!("const '{}': {e}", c.name))?;
        }
    }

    // ---- validate struct/sum field types (each with its own params in scope)
    for sd in &program.structs {
        let tps: HashSet<String> = sd.type_params.iter().cloned().collect();
        let mut seen = HashSet::new();
        for (fname, fty) in &sd.fields {
            if !seen.insert(fname) {
                return Err(format!("struct '{}': duplicate field '{fname}'", sd.name).into());
            }
            validate_type(fty, &cx, &tps)
                .map_err(|e| format!("struct '{}' field '{fname}': {e}", sd.name))?;
        }
    }
    for sd in &program.sums {
        let tps: HashSet<String> = sd.type_params.iter().cloned().collect();
        for v in &sd.variants {
            for (fname, fty) in &v.fields {
                validate_type(fty, &cx, &tps).map_err(|e| {
                    format!("sum '{}' variant '{}' field '{fname}': {e}", sd.name, v.name)
                })?;
            }
        }
    }

    // ---- elaborate every function body -------------------------------------
    let mut funcs: Vec<Func> = Vec::with_capacity(program.funcs.len());
    for f in &program.funcs {
        let tps: HashSet<String> = f.type_params.iter().cloned().collect();

        // convention well-formedness
        let conv = program
            .conventions
            .get(&f.cc)
            .ok_or_else(|| format!("function '{}': unknown convention '{}'", f.name, f.cc))?;
        if conv.is_shim() {
            if conv.ret.is_none() {
                return Err(format!(
                    "function '{}': shim convention '{}' needs a :ret register",
                    f.name, f.cc
                ).into());
            }
            if conv.params.len() < f.params.len() {
                return Err(format!(
                    "function '{}': convention '{}' provides {} param registers but the \
                     function has {} parameters",
                    f.name,
                    f.cc,
                    conv.params.len(),
                    f.params.len()
                ).into());
            }
        }

        for p in &f.params {
            validate_type(&p.ty, &cx, &tps)
                .map_err(|e| format!("function '{}' param '{}': {e}", f.name, p.name))?;
        }
        // `void` is a valid return type (validate_type rejects it elsewhere).
        if f.ret != Type::Void {
            validate_type(&f.ret, &cx, &tps)
                .map_err(|e| format!("function '{}' return type: {e}", f.name))?;
        }

        // Inside the body, a struct parameter is an (immutable) reference and a
        // `(mut T)` parameter a mutable one; scalars/sums/pointers are unchanged.
        let mut env: HashMap<String, Type> = f
            .params
            .iter()
            .map(|p| (p.name.clone(), param_ref_type(&p.ty, &cx.structs, &cx.sums, &tps)))
            .collect();

        let mut body: Vec<Expr> = Vec::with_capacity(f.body.len());
        let n = f.body.len();
        for (i, e) in f.body.iter().enumerate() {
            if i + 1 == n {
                // tail position: check against the declared return type, so a
                // literal or constructor in the body's tail adopts it.
                let (ee, et) = synth(e, Some(&f.ret), &mut env, &cx, &tps, &f.name)?;
                let et_str = ty_str(&et);
                let ee = coerce(ee, et, &f.ret, false, &f.name, "body").map_err(|_| {
                    format!(
                        "function '{}': body has type {} but the declared return type is {}",
                        f.name,
                        et_str,
                        ty_str(&f.ret)
                    )
                })?;
                body.push(ee);
            } else {
                let (ee, _) = synth(e, None, &mut env, &cx, &tps, &f.name)?;
                body.push(ee);
            }
        }
        if n == 0 && f.ret != Type::Int(64, true) && f.ret != Type::Void {
            return Err(format!(
                "function '{}': body has type i64 but the declared return type is {}",
                f.name,
                ty_str(&f.ret)
            ).into());
        }

        // Codegen sees only pointers: erase the reference tier in the output
        // signature (a struct/`mut` param becomes a `(ptr …)` parameter).
        let out_params: Vec<Param> = f
            .params
            .iter()
            .map(|p| Param {
                name: p.name.clone(),
                ty: erase_refs(&param_ref_type(&p.ty, &cx.structs, &cx.sums, &tps)),
            })
            .collect();
        funcs.push(Func {
            name: f.name.clone(),
            type_params: f.type_params.clone(),
            cc: f.cc.clone(),
            params: out_params,
            ret: erase_refs(&f.ret),
            body,
            span: f.span,
        });
    }

    // ---- static-assert conditions ------------------------------------------
    let empty_tps = HashSet::new();
    let mut asserts: Vec<StaticAssert> = Vec::with_capacity(program.asserts.len());
    for a in &program.asserts {
        let mut env: HashMap<String, Type> = HashMap::new();
        let (cond, t) = synth(&a.cond, None, &mut env, &cx, &empty_tps, "static-assert")?;
        if !is_cond(&t) {
            return Err(format!(
                "static-assert: condition must be a bool or integer, got {}",
                ty_str(&t)
            ).into());
        }
        asserts.push(StaticAssert {
            cond,
            msg: a.msg.clone(),
        });
    }

    Ok(Program {
        conventions: program.conventions.clone(),
        structs: program.structs.clone(),
        sums: program.sums.clone(),
        externs: program.externs.clone(),
        funcs,
        asserts,
        // Consts are fully erased into the elaborated bodies above; carried
        // through inert so the Program stays round-trippable.
        consts: program.consts.clone(),
    })
}

/// Validate that a type is well-formed: every named type exists (or is an
/// in-scope type parameter), and every generic application has the right arity.
fn validate_type(t: &Type, cx: &Cx, tps: &HashSet<String>) -> Result<(), Diag> {
    match t {
        Type::Never => Ok(()),   // synthesized only; not user-writable
        // `void` is ONLY a return type — never a parameter, field, or component
        // type. Return positions validate it separately (skipping this).
        Type::Void => Err("'void' is only valid as a return type".to_string().into()),
        Type::Int(..) => Ok(()),
        Type::Float(..) | Type::Bool => Ok(()),
        Type::Ptr(p) => validate_type(p, cx, tps),
        Type::Ref(_, p) => validate_type(p, cx, tps),
        Type::Array(e, _) => validate_type(e, cx, tps),
        Type::Slice(e) => validate_type(e, cx, tps),
        Type::Vec(e, n) => {
            if *n == 0 {
                return Err("vec must have a positive lane count".to_string().into());
            }
            match &**e {
                Type::Int(..) | Type::Float(..) => Ok(()),
                Type::Struct(p) if tps.contains(p) => Ok(()), // opaque param; checked at mono
                other => Err(format!("vec element must be a scalar int/float, got {}", ty_str(other)).into()),
            }
        }
        Type::Struct(name) => {
            if tps.contains(name) {
                Ok(()) // an in-scope, opaque type parameter
            } else if let Some(si) = cx.structs.get(name) {
                if si.type_params.is_empty() {
                    Ok(())
                } else {
                    Err(format!(
                        "generic type '{name}' expects {} type arguments, got 0",
                        si.type_params.len()
                    ).into())
                }
            } else if let Some(si) = cx.sums.get(name) {
                if si.type_params.is_empty() {
                    Ok(())
                } else {
                    Err(format!(
                        "generic type '{name}' expects {} type arguments, got 0",
                        si.type_params.len()
                    ).into())
                }
            } else {
                Err(format!("unknown type '{name}'").into())
            }
        }
        Type::Fn(_, params, ret) => {
            for p in params {
                validate_type(p, cx, tps)?;
            }
            validate_type(ret, cx, tps)
        }
        Type::App(name, args) => {
            for a in args {
                validate_type(a, cx, tps)?;
            }
            let arity = cx
                .structs
                .get(name)
                .map(|s| s.type_params.len())
                .or_else(|| cx.sums.get(name).map(|s| s.type_params.len()))
                .ok_or_else(|| format!("unknown generic type '{name}'"))?;
            if arity != args.len() {
                return Err(format!(
                    "generic type '{name}' expects {arity} type arguments, got {}",
                    args.len()
                ).into());
            }
            Ok(())
        }
    }
}

/// Synthesize the type of `e` and return an elaborated copy with literal
/// coercions and inferred type arguments inserted. `tps` is the set of type
/// parameters in scope (opaque types) for the function being checked.
///
/// Thin wrapper over [`synth_inner`] that attaches `e`'s source span to any
/// error this frame raises. Because `with_span` only fills a *spanless* diag,
/// and recursion goes through `synth` (so each child error already carries the
/// child's span), the innermost offending expression's span wins as the error
/// bubbles up — and every error a frame raises itself gets that frame's span.
fn synth(
    e: &Expr,
    expected: Option<&Type>,
    env: &mut HashMap<String, Type>,
    cx: &Cx,
    tps: &HashSet<String>,
    fname: &str,
) -> Result<(Expr, Type), Diag> {
    synth_inner(e, expected, env, cx, tps, fname).map_err(|d| d.with_span(e.span))
}

fn synth_inner(
    e: &Expr,
    expected: Option<&Type>,
    env: &mut HashMap<String, Type>,
    cx: &Cx,
    tps: &HashSet<String>,
    fname: &str,
) -> Result<(Expr, Type), Diag> {
    match &e.kind {
        ExprKind::Int(n) => Ok((Expr::new(ExprKind::Int(*n), e.span), Type::Int(64, true))),
        ExprKind::Float(x) => Ok((Expr::new(ExprKind::Float(*x), e.span), Type::Float(64))),
        ExprKind::Bool(b) => Ok((Expr::new(ExprKind::Bool(*b), e.span), Type::Bool)),
        // "…" is a (slice u8) view over a static byte global (length known now).
        ExprKind::Str(s) => Ok((Expr::new(ExprKind::Str(s.clone()), e.span), Type::Slice(Box::new(Type::Int(8, false))))),
        // c"…" is a (ptr i8) to a NUL-terminated global, for FFI.
        ExprKind::CStr(s) => Ok((Expr::new(ExprKind::CStr(s.clone()), e.span), Type::Ptr(Box::new(Type::Int(8, true))))),
        ExprKind::Zeroed(ty) => {
            validate_type(ty, cx, tps).map_err(|e| format!("in '{fname}': zeroed: {e}"))?;
            Ok((Expr::new(ExprKind::Zeroed(ty.clone()), e.span), ty.clone()))
        }
        ExprKind::Borrow { mutable, place } => {
            // Borrow a place as a reference. The place must be an lvalue (a
            // variable/field/index whose type is a ref or pointer); a mutable
            // borrow additionally requires the place itself be writable. The
            // reference is erased to the underlying pointer expression.
            let (pe, pt) = synth(place, None, env, cx, tps, fname)?;
            let pointee = place_pointee(&pt)
                .cloned()
                .ok_or_else(|| format!("in '{fname}': cannot borrow a non-place of type {}", ty_str(&pt)))?;
            if *mutable && !is_writable(&pt) {
                return Err(format!(
                    "in '{fname}': cannot take a mutable borrow of immutable '{}'",
                    ty_str(&pt)
                ).into());
            }
            Ok((pe, Type::Ref(*mutable, Box::new(pointee))))
        }
        ExprKind::Var(name) => {
            // Locals shadow consts (locals are checked first), so a parameter or
            // `let` named the same as a const is unaffected. A const reference
            // elaborates to its literal inline — zero runtime cost, and an untyped
            // const re-enters width inference exactly like the literal would.
            if let Some(t) = env.get(name) {
                return Ok((Expr::new(ExprKind::Var(name.clone()), e.span), t.clone()));
            }
            if let Some((lit, ty)) = cx.consts.get(name) {
                return Ok((lit.clone(), ty.clone()));
            }
            Err(format!("in '{fname}': unbound variable '{name}'").into())
        }
        ExprKind::LlvmIr { result, args, body } => {
            // The raw-IR escape hatch: the form's type *is* the declared result
            // (the checker trusts the annotation; the LLVM verifier checks the
            // body). Operands are synthesized so they compose like any value.
            validate_type(result, cx, tps)
                .map_err(|e| format!("in '{fname}': llvm-ir result type: {e}"))?;
            let eargs = args
                .iter()
                .map(|a| {
                    let (e, t) = synth(a, None, env, cx, tps, fname)?;
                    // forbid-use: a void value can't be an llvm-ir operand. This
                    // path discards the type, bypassing the coerce gate — enforce
                    // the void-check on the synthesized operand type directly.
                    if t == Type::Void {
                        return Err(format!(
                            "in '{fname}': llvm-ir operand uses a void value \
                             (a (-> void) call yields nothing)"
                        ).into());
                    }
                    Ok(e)
                })
                .collect::<Result<Vec<_>, Diag>>()?;
            Ok((
                Expr::new(ExprKind::LlvmIr { result: result.clone(), args: eargs, body: body.clone() }, e.span),
                result.clone(),
            ))
        }
        ExprKind::Bin { op, lhs, rhs } => {
            let (le, lt) = synth(lhs, None, env, cx, tps, fname)?;
            let (re, rt) = synth(rhs, None, env, cx, tps, fname)?;
            let (mut sides, t) =
                unify_branches(vec![(le, lt), (re, rt)], fname, "arithmetic")?;
            if !numeric(&t, tps) {
                return Err(format!(
                    "in '{fname}': arithmetic requires integers, got {}",
                    ty_str(&t)
                ).into());
            }
            let rhs = sides.pop().unwrap();
            let lhs = sides.pop().unwrap();
            Ok((
                Expr::new(ExprKind::Bin {
                    op: *op,
                    lhs: Box::new(lhs),
                    rhs: Box::new(rhs),
                }, e.span),
                t,
            ))
        }
        ExprKind::Not(x) => {
            let (xe, xt) = synth(x, None, env, cx, tps, fname)?;
            if !numeric(&xt, tps) {
                return Err(format!(
                    "in '{fname}': inot requires an integer, got {}",
                    ty_str(&xt)
                ).into());
            }
            Ok((Expr::new(ExprKind::Not(Box::new(xe)), e.span), xt))
        }
        ExprKind::Cmp { op, lhs, rhs } => {
            let (le, lt) = synth(lhs, None, env, cx, tps, fname)?;
            let (re, rt) = synth(rhs, None, env, cx, tps, fname)?;
            let (mut sides, t) =
                unify_branches(vec![(le, lt), (re, rt)], fname, "comparison")?;
            if !numeric(&t, tps) {
                return Err(format!(
                    "in '{fname}': comparison requires integers, got {}",
                    ty_str(&t)
                ).into());
            }
            let rhs = sides.pop().unwrap();
            let lhs = sides.pop().unwrap();
            Ok((
                Expr::new(ExprKind::Cmp {
                    op: *op,
                    lhs: Box::new(lhs),
                    rhs: Box::new(rhs),
                }, e.span),
                Type::Bool,
            ))
        }
        // The structured-loop primitive. The body is typed as statements (its
        // value is unused — a loop only yields a value by `break`ing). The loop's
        // type is the value type unified across its break sites, or i64.
        ExprKind::Loop { label, body } => {
            if let Some(l) = label {
                if cx.loops.borrow().iter().any(|f| f.label.as_deref() == Some(l)) {
                    return Err(format!(
                        "in '{fname}': loop label ':{l}' shadows an enclosing loop with the same label"
                    ).into());
                }
            }
            cx.loops.borrow_mut().push(LoopFrame {
                label: label.clone(),
                break_ty: None,
                expected: expected.cloned(),
            });
            let mut body_e = Vec::with_capacity(body.len());
            for e in body {
                match synth(e, None, env, cx, tps, fname) {
                    Ok((ee, _)) => body_e.push(ee),
                    Err(err) => {
                        cx.loops.borrow_mut().pop();
                        return Err(err);
                    }
                }
            }
            let frame = cx.loops.borrow_mut().pop().expect("loop frame present");
            // When the loop was checked against an expected type, that IS its type
            // (every break was checked/coerced to it). Otherwise it's the unified
            // break-value type, or Never if there is no break.
            let loop_ty = expected.cloned().or(frame.break_ty).unwrap_or(Type::Never);
            Ok((Expr::new(ExprKind::Loop { label: label.clone(), body: body_e }, e.span), loop_ty))
        }
        // Exit a loop with an optional value; the value type is unified into the
        // target loop's frame. `break` itself diverges (type i64, never used).
        ExprKind::Break { label, value } => {
            // Resolve the target loop and read its contextual expected type (used
            // to bidirectionally check the break value).
            let (idx, target_expected) = {
                let loops = cx.loops.borrow();
                if loops.is_empty() {
                    return Err(format!("in '{fname}': break outside of a loop").into());
                }
                let idx = match label {
                    Some(l) => loops.iter().rposition(|f| f.label.as_deref() == Some(l)).ok_or_else(
                        || format!("in '{fname}': break to unknown loop label ':{l}'"),
                    )?,
                    None => loops.len() - 1,
                };
                (idx, loops[idx].expected.clone())
            };
            // Check the value against the loop's expected type when there is one,
            // so a bare (Ok v)/(Err e)/literal adopts it; else synthesize.
            let (val_e, vty) = match value {
                Some(v) => match &target_expected {
                    Some(exp) => {
                        let ve = check_to(v, exp, env, cx, tps, fname, "break value")?;
                        (Some(Box::new(ve)), exp.clone())
                    }
                    None => {
                        let (ve, vt) = synth(v, None, env, cx, tps, fname)?;
                        (Some(Box::new(ve)), vt)
                    }
                },
                None => (None, Type::i64()),
            };
            // A Never-typed break value is divergent (its `value` already broke,
            // e.g. `(break :s (do … (break :s v)))`): it never yields that value, so
            // it must NOT constrain the loop's break type — mirror the Never-skip in
            // unify_branches. (Missing this regressed block/return-from/defer, whose
            // wrapper's outer break value is exactly such a divergent `do`.)
            if vty != Type::Never {
                let mut loops = cx.loops.borrow_mut();
                match &loops[idx].break_ty {
                    None => loops[idx].break_ty = Some(vty),
                    Some(prev) if *prev != vty => {
                        return Err(format!(
                            "in '{fname}': loop breaks with different value types ({} vs {})",
                            ty_str(prev),
                            ty_str(&vty)
                        ).into());
                    }
                    Some(_) => {}
                }
            }
            Ok((Expr::new(ExprKind::Break { label: label.clone(), value: val_e }, e.span), Type::Never))
        }
        ExprKind::Continue { label } => {
            let loops = cx.loops.borrow();
            if loops.is_empty() {
                return Err(format!("in '{fname}': continue outside of a loop").into());
            }
            if let Some(l) = label {
                if !loops.iter().any(|f| f.label.as_deref() == Some(l)) {
                    return Err(format!("in '{fname}': continue to unknown loop label ':{l}'").into());
                }
            }
            Ok((Expr::new(ExprKind::Continue { label: label.clone() }, e.span), Type::Never))
        }
        ExprKind::If { cond, then, els } => {
            let (ce, ct) = synth(cond, None, env, cx, tps, fname)?;
            if !is_cond(&ct) {
                return Err(format!(
                    "in '{fname}': if condition must be a bool or integer, got {}",
                    ty_str(&ct)
                ).into());
            }
            match expected {
                // checking mode: push the expected type into both branches, so a
                // literal or constructor in either arm adopts it.
                Some(exp) => {
                    let then_e = check_to(then, exp, env, cx, tps, fname, "if branch")?;
                    let els_e = check_to(els, exp, env, cx, tps, fname, "if branch")?;
                    Ok((
                        Expr::new(ExprKind::If {
                            cond: Box::new(ce),
                            then: Box::new(then_e),
                            els: Box::new(els_e),
                        }, e.span),
                        exp.clone(),
                    ))
                }
                // synthesis mode: synthesize both branches and reconcile them.
                None => {
                    let (te, tt) = synth(then, None, env, cx, tps, fname)?;
                    let (ee, et) = synth(els, None, env, cx, tps, fname)?;
                    let (mut branches, t) =
                        unify_branches(vec![(te, tt), (ee, et)], fname, "if branches")?;
                    let els = branches.pop().unwrap();
                    let then = branches.pop().unwrap();
                    Ok((
                        Expr::new(ExprKind::If {
                            cond: Box::new(ce),
                            then: Box::new(then),
                            els: Box::new(els),
                        }, e.span),
                        t,
                    ))
                }
            }
        }
        ExprKind::Do(es) => {
            let mut out = Vec::with_capacity(es.len());
            let mut last = Type::Int(64, true);
            let n = es.len();
            for (i, e) in es.iter().enumerate() {
                if i + 1 == n {
                    match expected {
                        Some(exp) => {
                            out.push(check_to(e, exp, env, cx, tps, fname, "do result")?);
                            last = exp.clone();
                        }
                        None => {
                            let (ee, et) = synth(e, None, env, cx, tps, fname)?;
                            last = et;
                            out.push(ee);
                        }
                    }
                } else {
                    let (ee, _) = synth(e, None, env, cx, tps, fname)?;
                    out.push(ee);
                }
            }
            Ok((Expr::new(ExprKind::Do(out), e.span), last))
        }
        ExprKind::Let { binds, body } => {
            let saved = env.clone();
            let mut new_binds = Vec::with_capacity(binds.len());
            for (name, mutable, val) in binds {
                let (ve, vt) = synth(val, None, env, cx, tps, fname)?;
                // A `(-> void)` call yields no value to bind (no-silent-wrong).
                if vt == Type::Void {
                    return Err(format!(
                        "in '{fname}': cannot bind '{name}' to a void value (a (-> void) \
                         call yields nothing)"
                    ).into());
                }
                match &vt {
                    // binding to an existing place is an *alias* (no new storage);
                    // it's immutable unless asked for `(mut name)`, and you can't
                    // make a mutable alias of an immutable place.
                    Type::Ref(src_mut, pointee) => {
                        if *mutable && !src_mut {
                            return Err(format!(
                                "in '{fname}': cannot make a mutable alias '{name}' of an \
                                 immutable reference"
                            ).into());
                        }
                        env.insert(name.clone(), Type::Ref(*mutable, pointee.clone()));
                        new_binds.push((name.clone(), false, ve));
                    }
                    // `(mut name)` or an aggregate value becomes a fresh stack
                    // place (its fields are addressable, it can be borrowed). The
                    // name is bound to an `alloc-stack` whose value is a `do` that
                    // stores the init value into the slot and yields the slot
                    // pointer — so the store happens *at the binding's position*,
                    // before any later binding (or the body) reads the place.
                    // (Deferring the stores to body-start would mis-order a later
                    // binding that consumes this one, e.g. `[s (f) t (g s)]`.)
                    _ if *mutable || is_place_value_type(&vt, cx) => {
                        env.insert(name.clone(), Type::Ref(*mutable, Box::new(vt.clone())));
                        let slot = format!("{name}.slot");
                        let init = Expr::new(ExprKind::Do(vec![
                            Expr::new(ExprKind::Store {
                                ptr: Box::new(Expr::new(ExprKind::Var(slot.clone()), e.span)),
                                val: Box::new(ve),
                            }, e.span),
                            Expr::new(ExprKind::Var(slot.clone()), e.span),
                        ]), e.span);
                        // Inner alias binding holds the alloc; the outer name is the
                        // slot pointer after the store. Both lower trivially.
                        new_binds.push((
                            slot.clone(),
                            false,
                            Expr::new(ExprKind::Alloc {
                                storage: Storage::Stack,
                                ty: erase_refs(&vt),
                            }, e.span),
                        ));
                        new_binds.push((name.clone(), false, init));
                    }
                    // scalars/pointers/sums stay ordinary immutable SSA values.
                    _ => {
                        env.insert(name.clone(), vt);
                        new_binds.push((name.clone(), false, ve));
                    }
                }
            }
            let mut out = Vec::with_capacity(body.len());
            let mut last = Type::Int(64, true);
            let n = body.len();
            for (i, e) in body.iter().enumerate() {
                if i + 1 == n {
                    match expected {
                        Some(exp) => {
                            out.push(check_to(e, exp, env, cx, tps, fname, "let body")?);
                            last = exp.clone();
                        }
                        None => {
                            let (ee, et) = synth(e, None, env, cx, tps, fname)?;
                            last = et;
                            out.push(ee);
                        }
                    }
                } else {
                    let (ee, _) = synth(e, None, env, cx, tps, fname)?;
                    out.push(ee);
                }
            }
            *env = saved; // bindings are lexical
            Ok((
                Expr::new(ExprKind::Let {
                    binds: new_binds,
                    body: out,
                }, e.span),
                last,
            ))
        }
        ExprKind::Call {
            func,
            type_args,
            args,
        } => {
            // Variant construction (the parser emits it as a call to the variant
            // name) routes here so we can type and infer its sum's type args.
            if cx.variant_to_sum.contains_key(func) {
                return synth_construct(func, type_args, args, expected, env, cx, tps, fname, e.span);
            }
            let sig = cx
                .sigs
                .get(func)
                .ok_or_else(|| format!("in '{fname}': call to undefined function '{func}'"))?;
            let arity_ok = if sig.variadic {
                args.len() >= sig.params.len()
            } else {
                args.len() == sig.params.len()
            };
            if !arity_ok {
                return Err(format!(
                    "in '{fname}': '{func}' expects {}{} args, got {}",
                    if sig.variadic { "at least " } else { "" },
                    sig.params.len(),
                    args.len()
                ).into());
            }
            if sig.type_params.is_empty() {
                // non-generic: each parameter type is known up front, so check the
                // argument against it (a literal/constructor argument adopts it).
                if !type_args.is_empty() {
                    return Err(format!(
                        "in '{fname}': '{func}' is not generic but got type arguments"
                    ).into());
                }
                let mut new_args = Vec::with_capacity(args.len());
                for (i, a) in args.iter().enumerate() {
                    // Variadic extra arguments have no declared parameter type:
                    // synthesize and pass them through (C default-promotion rules).
                    if i >= sig.params.len() {
                        let (ae, at) = synth(a, None, env, cx, tps, fname)?;
                        // forbid-use: a void value can't fill a variadic slot. This
                        // path discards the type, so it bypasses the coerce gate —
                        // the void-check must be enforced on the synthesized type too.
                        if at == Type::Void {
                            return Err(format!(
                                "in '{fname}': variadic argument uses a void value \
                                 (a (-> void) call yields nothing)"
                            ).into());
                        }
                        new_args.push(ae);
                        continue;
                    }
                    let want = sig.params[i].clone();
                    let is_mut_borrow = matches!(&a.kind, ExprKind::Borrow { mutable: true, .. });
                    // A by-reference parameter takes a value (or place) of its
                    // pointee type — impose the pointee as the expected type so a
                    // literal or bare constructor argument adopts it (it is then
                    // borrowed if it is a place, or spilled if it is an rvalue).
                    let exp = match &want {
                        Type::Ref(_, inner) => Some(&**inner),
                        _ => Some(&want),
                    };
                    let (ae, at) = synth(a, exp, env, cx, tps, fname)?;
                    let ae = coerce_arg(
                        is_mut_borrow,
                        ae,
                        at,
                        &want,
                        sig.is_extern,
                        fname,
                        &format!("argument {} to '{func}'", i + 1),
                    )?;
                    new_args.push(ae);
                }
                return Ok((
                    Expr::new(ExprKind::Call {
                        func: func.clone(),
                        type_args: vec![],
                        args: new_args,
                    }, e.span),
                    sig.ret.clone(),
                ));
            }

            // generic: synthesize the arguments, then infer the type parameters
            // from them — and from the expected return type, if there is one.
            //
            // An argument that is ITSELF a generic call may be unable to infer its
            // own type parameters in isolation — e.g. `(pick (empty-slice) 42)`,
            // where `pick`'s `T` is fixed by the sibling `42`, not by the
            // `(empty-slice)` argument. Such an argument is DEFERRED: synthesize
            // the rest, solve the substitution, then re-synthesize the deferred
            // ones with the now-known parameter type pushed in as the expected
            // type (bidirectional inference through a nested generic call).
            let mut arglist: Vec<Option<(Expr, Type)>> = Vec::with_capacity(args.len());
            let mut deferred: Vec<(usize, Diag)> = Vec::new();
            for (i, a) in args.iter().enumerate() {
                let saved = env.clone();
                match synth(a, None, env, cx, tps, fname) {
                    Ok(pair) => arglist.push(Some(pair)),
                    Err(e) => {
                        *env = saved; // a failed synth may have left the env dirty
                        arglist.push(None);
                        deferred.push((i, e));
                    }
                }
            }
            let subst = solve_type_args(
                &sig.type_params,
                &sig.params,
                type_args,
                &arglist,
                &sig.ret,
                expected,
                cx,
                tps,
                fname,
                func,
            )?;
            // Re-synthesize each deferred argument now that the parameter type is
            // concrete; if it STILL can't be inferred, the substitution didn't
            // fix it, so report the argument's original inference error.
            for (i, orig_err) in deferred {
                let want = subst_apply(&sig.params[i], &subst);
                let pair = synth(&args[i], Some(&want), env, cx, tps, fname)
                    .map_err(|_| orig_err)?;
                arglist[i] = Some(pair);
            }
            let mut new_args = Vec::with_capacity(args.len());
            for (i, a) in args.iter().enumerate() {
                let (ae, at) = arglist[i].take().expect("every argument synthesized");
                let want = subst_apply(&sig.params[i], &subst);
                let is_mut_borrow = matches!(&a.kind, ExprKind::Borrow { mutable: true, .. });
                let ae = coerce_arg(
                    is_mut_borrow,
                    ae,
                    at,
                    &want,
                    sig.is_extern,
                    fname,
                    &format!("argument {} to '{func}'", i + 1),
                )?;
                new_args.push(ae);
            }
            let out_type_args: Vec<Type> = sig
                .type_params
                .iter()
                .map(|p| subst[p].clone())
                .collect();
            let ret = subst_apply(&sig.ret, &subst);
            Ok((
                Expr::new(ExprKind::Call {
                    func: func.clone(),
                    type_args: out_type_args,
                    args: new_args,
                }, e.span),
                ret,
            ))
        }
        ExprKind::Alloc { storage, ty } => {
            validate_type(ty, cx, tps).map_err(|e| format!("in '{fname}': alloc: {e}"))?;
            Ok((
                Expr::new(ExprKind::Alloc {
                    storage: *storage,
                    ty: ty.clone(),
                }, e.span),
                Type::Ptr(Box::new(ty.clone())),
            ))
        }
        ExprKind::BitGet { ptr, field } => {
            let (pe, fty) = bit_field_type(ptr, field, env, cx, tps, fname)?;
            Ok((
                Expr::new(ExprKind::BitGet {
                    ptr: Box::new(pe),
                    field: field.clone(),
                }, e.span),
                fty,
            ))
        }
        ExprKind::BitSet { ptr, field, val } => {
            let (pe, fty) = bit_field_type(ptr, field, env, cx, tps, fname)?;
            let (ve, vt) = synth(val, Some(&fty), env, cx, tps, fname)?;
            let ve = coerce(
                ve,
                vt,
                &fty,
                false,
                fname,
                &format!("set! into bitfield '{field}'"),
            )?;
            Ok((
                Expr::new(ExprKind::BitSet {
                    ptr: Box::new(pe),
                    field: field.clone(),
                    val: Box::new(ve),
                }, e.span),
                fty,
            ))
        }
        ExprKind::Field { ptr, field } => {
            let (pe, pt) = synth(ptr, None, env, cx, tps, fname)?;
            let pointee = place_pointee(&pt).cloned().ok_or_else(|| {
                format!(
                    "in '{fname}': field access needs a pointer or reference, got {}",
                    ty_str(&pt)
                )
            })?;
            let sname = struct_name(&pointee).ok_or_else(|| {
                format!(
                    "in '{fname}': field access needs a pointer to a struct, got (ptr {})",
                    ty_str(&pointee)
                )
            })?;
            if cx.structs.get(sname).is_some_and(|s| s.is_bits) {
                return Err(format!(
                    "in '{fname}': '{sname}' is a :layout bits struct; use (get p {field}) / (set! p {field} v)"
                ).into());
            }
            let fields = struct_fields(&pointee, cx)
                .map_err(|e| format!("in '{fname}': {e}"))?;
            let fty = fields
                .iter()
                .find(|(n, _)| n == field)
                .map(|(_, t)| t.clone())
                .ok_or_else(|| format!("in '{fname}': struct '{sname}' has no field '{field}'"))?;
            // a field of a place is itself a place, inheriting its mutability.
            Ok((
                Expr::new(ExprKind::Field {
                    ptr: Box::new(pe),
                    field: field.clone(),
                }, e.span),
                replace_pointee(&pt, fty),
            ))
        }
        ExprKind::Load(p) => {
            let (pe, pt) = synth(p, None, env, cx, tps, fname)?;
            match place_pointee(&pt) {
                Some(pointee) => Ok((Expr::new(ExprKind::Load(Box::new(pe)), e.span), pointee.clone())),
                None => Err(format!(
                    "in '{fname}': load expects a pointer or reference, got {}",
                    ty_str(&pt)
                ).into()),
            }
        }
        ExprKind::Store { ptr, val } => {
            let (pe, pt) = synth(ptr, None, env, cx, tps, fname)?;
            let pointee = match place_pointee(&pt) {
                Some(p) => p.clone(),
                None => {
                    return Err(format!(
                        "in '{fname}': store! expects a pointer or reference, got {}",
                        ty_str(&pt)
                    ).into())
                }
            };
            if !is_writable(&pt) {
                return Err(format!(
                    "in '{fname}': cannot store! through immutable reference of type {} \
                     (declare it `(mut …)` to make it writable)",
                    ty_str(&pt)
                ).into());
            }
            let (ve, vt) = synth(val, Some(&pointee), env, cx, tps, fname)?;
            let ve = coerce(ve, vt, &pointee, false, fname, "store! value")?;
            Ok((
                Expr::new(ExprKind::Store {
                    ptr: Box::new(pe),
                    val: Box::new(ve),
                }, e.span),
                pointee,
            ))
        }
        ExprKind::Index { ptr, idx } => {
            let (pe, pt) = synth(ptr, None, env, cx, tps, fname)?;
            let (ie, it) = synth(idx, None, env, cx, tps, fname)?;
            let ie = coerce(ie, it, &Type::Int(64, true), false, fname, "index")
                .map_err(|_| format!("in '{fname}': index must be i64"))?;
            match place_pointee(&pt) {
                Some(pointee) => {
                    let elem = match pointee {
                        Type::Array(elem, _) => (**elem).clone(),
                        p => p.clone(),
                    };
                    // an element of a place is a place, inheriting its mutability.
                    Ok((
                        Expr::new(ExprKind::Index {
                            ptr: Box::new(pe),
                            idx: Box::new(ie),
                        }, e.span),
                        replace_pointee(&pt, elem),
                    ))
                }
                None => Err(format!(
                    "in '{fname}': index expects a pointer or reference, got {}",
                    ty_str(&pt)
                ).into()),
            }
        }
        ExprKind::Cast { ty, expr } => {
            validate_type(ty, cx, tps).map_err(|e| format!("in '{fname}': cast target: {e}"))?;
            let (ee, et) = synth(expr, None, env, cx, tps, fname)?;
            match (ty, &et) {
                // int<->int width change, ptr<->ptr reinterpret, int<->ptr
                // (null/MMIO/tagged pointers), and int<->float / float<->float.
                (Type::Int(..), Type::Int(..))
                | (Type::Ptr(..), Type::Ptr(..))
                | (Type::Int(..), Type::Ptr(..))
                | (Type::Ptr(..), Type::Int(..))
                | (Type::Float(..), Type::Float(..))
                | (Type::Float(..), Type::Int(..))
                | (Type::Int(..), Type::Float(..)) => Ok((
                    Expr::new(ExprKind::Cast {
                        ty: ty.clone(),
                        expr: Box::new(ee),
                    }, e.span),
                    ty.clone(),
                )),
                _ => Err(format!(
                    "in '{fname}': cast only converts among int, float, and ptr (got {} to {})",
                    ty_str(&et),
                    ty_str(ty)
                ).into()),
            }
        }
        ExprKind::SizeOf(ty) => {
            validate_type(ty, cx, tps).map_err(|e| format!("in '{fname}': sizeof: {e}"))?;
            Ok((Expr::new(ExprKind::SizeOf(ty.clone()), e.span), Type::Int(64, true)))
        }
        ExprKind::AlignOf(ty) => {
            validate_type(ty, cx, tps).map_err(|e| format!("in '{fname}': alignof: {e}"))?;
            Ok((Expr::new(ExprKind::AlignOf(ty.clone()), e.span), Type::Int(64, true)))
        }
        ExprKind::OffsetOf(ty, field) => {
            validate_type(ty, cx, tps).map_err(|e| format!("in '{fname}': offsetof: {e}"))?;
            let fields = struct_fields(ty, cx)
                .map_err(|_| format!("in '{fname}': offsetof needs a struct type, got {}", ty_str(ty)))?;
            if !fields.iter().any(|(n, _)| n == field) {
                let name = struct_name(ty).unwrap_or("?");
                return Err(format!(
                    "in '{fname}': offsetof: struct '{name}' has no field '{field}'"
                ).into());
            }
            Ok((Expr::new(ExprKind::OffsetOf(ty.clone(), field.clone()), e.span), Type::Int(64, true)))
        }
        ExprKind::Free(p) => {
            let (pe, pt) = synth(p, None, env, cx, tps, fname)?;
            match pt {
                Type::Ptr(_) => Ok((Expr::new(ExprKind::Free(Box::new(pe)), e.span), Type::Int(64, true))),
                other => Err(format!(
                    "in '{fname}': free expects a pointer, got {}",
                    ty_str(&other)
                ).into()),
            }
        }
        ExprKind::Construct { sum, variant, args } => {
            // The parser never emits Construct (it uses Call); kept for safety.
            synth_construct(variant, &[], args, None, env, cx, tps, fname, e.span)
                .map(|(e, _)| (e, Type::Struct(sum.clone())))
        }
        ExprKind::Match { scrut, arms } => {
            let (mut se, st0) = synth(scrut, None, env, cx, tps, fname)?;
            // A by-reference sum (e.g. a sum parameter, now passed by immutable
            // reference) reads as its value here: load it so the match sees a
            // value scrutinee, exactly as `coerce` reads a ref as its value.
            let st = match &st0 {
                Type::Ref(_, inner) if sum_variants(inner, cx).is_some() => {
                    se = Expr::new(ExprKind::Load(Box::new(se)), e.span);
                    (**inner).clone()
                }
                _ => st0,
            };
            let (sumname, variants) = sum_variants(&st, cx).ok_or_else(|| {
                format!("in '{fname}': match expects a sum value, got {}", ty_str(&st))
            })?;
            if arms.is_empty() {
                return Err(format!("in '{fname}': empty match").into());
            }
            let mut covered: HashSet<&str> = HashSet::new();
            // In checking mode each arm body is checked against `expected`; in
            // synthesis mode they are synthesized and reconciled afterwards.
            let mut checked_arms: Vec<Arm> = Vec::with_capacity(arms.len());
            let mut syn_bodies: Vec<(Expr, Type)> = Vec::with_capacity(arms.len());
            let mut meta: Vec<(String, Vec<String>)> = Vec::with_capacity(arms.len());
            for arm in arms {
                let v = variants.iter().find(|v| v.name == arm.variant).ok_or_else(|| {
                    format!("in '{fname}': sum '{sumname}' has no variant '{}'", arm.variant)
                })?;
                if !covered.insert(arm.variant.as_str()) {
                    return Err(format!("in '{fname}': duplicate match arm for '{}'", arm.variant).into());
                }
                if arm.binds.len() != v.fields.len() {
                    return Err(format!(
                        "in '{fname}': arm '{}' binds {} name(s) but the variant has {} field(s)",
                        arm.variant,
                        arm.binds.len(),
                        v.fields.len()
                    ).into());
                }
                let saved = env.clone();
                for (b, (_, fty)) in arm.binds.iter().zip(&v.fields) {
                    env.insert(b.clone(), fty.clone());
                }
                match expected {
                    Some(exp) => {
                        let be = check_to(&arm.body, exp, env, cx, tps, fname, "match arm")?;
                        checked_arms.push(Arm {
                            variant: arm.variant.clone(),
                            binds: arm.binds.clone(),
                            body: be,
                        });
                    }
                    None => {
                        let (be, bt) = synth(&arm.body, None, env, cx, tps, fname)?;
                        syn_bodies.push((be, bt));
                        meta.push((arm.variant.clone(), arm.binds.clone()));
                    }
                }
                *env = saved;
            }
            if covered.len() != variants.len() {
                return Err(format!(
                    "in '{fname}': non-exhaustive match on '{sumname}' ({} of {} variants)",
                    covered.len(),
                    variants.len()
                ).into());
            }
            match expected {
                Some(exp) => Ok((
                    Expr::new(ExprKind::Match {
                        scrut: Box::new(se),
                        arms: checked_arms,
                    }, e.span),
                    exp.clone(),
                )),
                None => {
                    let (coerced, t) = unify_branches(syn_bodies, fname, "match arms")?;
                    let new_arms = meta
                        .into_iter()
                        .zip(coerced)
                        .map(|((variant, binds), body)| Arm { variant, binds, body })
                        .collect();
                    Ok((
                        Expr::new(ExprKind::Match {
                            scrut: Box::new(se),
                            arms: new_arms,
                        }, e.span),
                        t,
                    ))
                }
            }
        }
        ExprKind::FnPtrOf(name) => {
            let sig = cx
                .sigs
                .get(name)
                .ok_or_else(|| format!("in '{fname}': fnptr-of unknown function '{name}'"))?;
            if !sig.type_params.is_empty() {
                return Err(format!(
                    "in '{fname}': cannot take a function pointer to generic function '{name}'"
                ).into());
            }
            if !sig.fnptr_ok {
                return Err(format!(
                    "in '{fname}': cannot take a function pointer to '{name}' \
                     (shim-convention functions are not supported yet)"
                ).into());
            }
            Ok((
                Expr::new(ExprKind::FnPtrOf(name.clone()), e.span),
                Type::Fn(sig.cc.clone(), sig.params.clone(), Box::new(sig.ret.clone())),
            ))
        }
        ExprKind::CallPtr { fp, args } => {
            let (fe, ft) = synth(fp, None, env, cx, tps, fname)?;
            match ft {
                Type::Fn(_, params, ret) => {
                    if params.len() != args.len() {
                        return Err(format!(
                            "in '{fname}': function pointer expects {} args, got {}",
                            params.len(),
                            args.len()
                        ).into());
                    }
                    let mut new_args = Vec::with_capacity(args.len());
                    for (i, a) in args.iter().enumerate() {
                        let (ae, at) = synth(a, Some(&params[i]), env, cx, tps, fname)?;
                        let ae = coerce(
                            ae,
                            at,
                            &params[i],
                            false,
                            fname,
                            &format!("call-ptr argument {}", i + 1),
                        )?;
                        new_args.push(ae);
                    }
                    Ok((
                        Expr::new(ExprKind::CallPtr {
                            fp: Box::new(fe),
                            args: new_args,
                        }, e.span),
                        *ret,
                    ))
                }
                other => Err(format!(
                    "in '{fname}': call-ptr expects a function pointer, got {}",
                    ty_str(&other)
                ).into()),
            }
        }
        // Produced by `coerce_arg` as part of elaboration; never an input to
        // synthesis (the checker runs once and `SpillRef` only appears in its
        // output).
        ExprKind::SpillRef(_) => {
            unreachable!("SpillRef is checker-produced and is never re-synthesized")
        }
    }
}

/// Type and elaborate a variant construction `(Variant [targs?] args…)`. The
/// sum's type arguments come from (in priority) explicit `[targs]`, the expected
/// type pushed in from context (checking mode), or inference from the field
/// argument types; they're written back onto the call so monomorphization can
/// specialize it. When the type arguments are known *before* the arguments
/// (explicit or from context), each argument is itself checked in checking mode,
/// so nested constructors and literals get their expected type too.
#[allow(clippy::too_many_arguments)]
fn synth_construct(
    variant: &str,
    type_args: &[Type],
    args: &[Expr],
    expected: Option<&Type>,
    env: &mut HashMap<String, Type>,
    cx: &Cx,
    tps: &HashSet<String>,
    fname: &str,
    span: Span,
) -> Result<(Expr, Type), Diag> {
    let sumname = cx.variant_to_sum[variant].clone();
    let si = &cx.sums[&sumname];
    let v = si
        .variants
        .iter()
        .find(|v| v.name == variant)
        .expect("variant_to_sum points at a real variant");
    if v.fields.len() != args.len() {
        return Err(format!(
            "in '{fname}': variant '{variant}' takes {} field(s), got {}",
            v.fields.len(),
            args.len()
        ).into());
    }
    let tparams = &si.type_params;
    let tpset: HashSet<String> = tparams.iter().cloned().collect();
    let field_types: Vec<Type> = v.fields.iter().map(|(_, t)| t.clone()).collect();

    // Phase 1: seed the substitution *without* the argument types — from explicit
    // type arguments, else from the expected type when it pins this sum.
    let mut subst: HashMap<String, Type> = HashMap::new();
    if !type_args.is_empty() {
        if type_args.len() != tparams.len() {
            return Err(format!(
                "in '{fname}': '{variant}' expects {} type arguments, got {}",
                tparams.len(),
                type_args.len()
            ).into());
        }
        for ta in type_args {
            validate_type(ta, cx, tps).map_err(|e| format!("in '{fname}': type argument: {e}"))?;
        }
        subst = tparams.iter().cloned().zip(type_args.iter().cloned()).collect();
    } else if let Some(targs) =
        expected.and_then(|exp| expected_sum_targs(exp, &sumname, tparams.len()))
    {
        subst = tparams.iter().cloned().zip(targs).collect();
    }

    let new_args = if tparams.iter().all(|p| subst.contains_key(p)) {
        // Fully determined: check each argument against its field type.
        let mut out = Vec::with_capacity(args.len());
        for (i, a) in args.iter().enumerate() {
            let want = subst_apply(&field_types[i], &subst);
            out.push(check_to(a, &want, env, cx, tps, fname, &format!("'{variant}' field {}", i + 1))?);
        }
        out
    } else {
        // Infer the remaining parameters from the argument types.
        let mut arglist: Vec<(Expr, Type)> = Vec::with_capacity(args.len());
        for a in args {
            arglist.push(synth(a, None, env, cx, tps, fname)?);
        }
        for (decl, (_, at)) in field_types.iter().zip(&arglist) {
            unify(decl, at, &tpset, &mut subst, fname)?;
        }
        for p in tparams {
            if !subst.contains_key(p) {
                return Err(format!(
                    "in '{fname}': cannot infer type argument '{p}' for '{variant}'; \
                     provide it explicitly: ({variant} [<types>] ...)"
                ).into());
            }
        }
        let mut out = Vec::with_capacity(args.len());
        for (i, (ae, at)) in arglist.into_iter().enumerate() {
            let want = subst_apply(&field_types[i], &subst);
            out.push(coerce(ae, at, &want, false, fname, &format!("'{variant}' field {}", i + 1))?);
        }
        out
    };

    if tparams.is_empty() {
        // concrete sum: leave type_args empty, result is the plain sum type.
        Ok((
            Expr::new(
                ExprKind::Call {
                    func: variant.to_string(),
                    type_args: vec![],
                    args: new_args,
                },
                span,
            ),
            Type::Struct(sumname),
        ))
    } else {
        let out_targs: Vec<Type> = tparams.iter().map(|p| subst[p].clone()).collect();
        Ok((
            Expr::new(
                ExprKind::Call {
                    func: variant.to_string(),
                    type_args: out_targs.clone(),
                    args: new_args,
                },
                span,
            ),
            Type::App(sumname, out_targs),
        ))
    }
}

/// If `expected` pins the named sum (`(S a b …)`, or a bare concrete `S`), return
/// its type arguments — used to seed a constructor's substitution from context.
fn expected_sum_targs(expected: &Type, sumname: &str, arity: usize) -> Option<Vec<Type>> {
    match expected {
        Type::App(n, targs) if n == sumname && targs.len() == arity => Some(targs.clone()),
        Type::Struct(n) if n == sumname && arity == 0 => Some(vec![]),
        _ => None,
    }
}

/// Synthesize `e` against an expected type (checking mode), then coerce the
/// result to it. The single boundary primitive: a literal/constructor in `e`
/// adopts `want`, and anything else is coerced (or reported as a mismatch).
fn check_to(
    e: &Expr,
    want: &Type,
    env: &mut HashMap<String, Type>,
    cx: &Cx,
    tps: &HashSet<String>,
    fname: &str,
    what: &str,
) -> Result<Expr, Diag> {
    let (ee, et) = synth(e, Some(want), env, cx, tps, fname)?;
    coerce(ee, et, want, false, fname, what)
}

/// Resolve a generic *function call's* type-parameter substitution. With explicit
/// `type_args` it uses them; otherwise it infers each parameter by unifying the
/// declared parameter types against the actual argument types, seeded by the
/// expected return type (so a parameter that appears only in the return can be
/// recovered from context).
#[allow(clippy::too_many_arguments)]
fn solve_type_args(
    type_params: &[String],
    declared: &[Type],
    type_args: &[Type],
    arglist: &[Option<(Expr, Type)>],
    ret_tmpl: &Type,
    expected: Option<&Type>,
    cx: &Cx,
    tps: &HashSet<String>,
    fname: &str,
    what: &str,
) -> Result<HashMap<String, Type>, Diag> {
    if !type_args.is_empty() {
        if type_args.len() != type_params.len() {
            return Err(format!(
                "in '{fname}': '{what}' expects {} type arguments, got {}",
                type_params.len(),
                type_args.len()
            ).into());
        }
        for ta in type_args {
            validate_type(ta, cx, tps).map_err(|e| format!("in '{fname}': type argument: {e}"))?;
        }
        return Ok(type_params.iter().cloned().zip(type_args.iter().cloned()).collect());
    }
    let tpset: HashSet<String> = type_params.iter().cloned().collect();
    let mut subst: HashMap<String, Type> = HashMap::new();
    if let Some(exp) = expected {
        unify(ret_tmpl, exp, &tpset, &mut subst, fname)?; // seed from context
    }
    // A `None` slot is a DEFERRED argument (one that couldn't infer its own type
    // parameters in isolation); it contributes nothing here and is re-synthesized
    // by the caller once the substitution is known.
    for (decl, slot) in declared.iter().zip(arglist) {
        if let Some((_, at)) = slot {
            unify(decl, at, &tpset, &mut subst, fname)?;
        }
    }
    for p in type_params {
        if !subst.contains_key(p) {
            return Err(format!(
                "in '{fname}': cannot infer type argument '{p}' for '{what}'; \
                 provide it explicitly: ({what} [<types>] ...)"
            ).into());
        }
    }
    Ok(subst)
}

/// Unify a declared type (which may mention type parameters in `tpset`) against
/// an actual concrete type, binding type parameters in `subst`. Structural where
/// shapes line up; mismatched shapes are deferred to the later coercion step.
fn unify(
    decl: &Type,
    actual: &Type,
    tpset: &HashSet<String>,
    subst: &mut HashMap<String, Type>,
    fname: &str,
) -> Result<(), Diag> {
    // forbid-use: `void` is return-position-only, so it can NEVER be a type
    // argument — inferring a type parameter to void would make a field/parameter
    // void (a no-silent-wrong hole, and a codegen panic on basic_ty(Void)). This is
    // the inference-path analogue of the coerce/synth void-checks: reject a void
    // value where a real type is required, here at the unification boundary.
    if *decl == Type::Void || *actual == Type::Void {
        return Err(format!(
            "in '{fname}': a void value cannot be used as a type argument \
             (a (-> void) call yields nothing)"
        ).into());
    }
    if let Type::Struct(p) = decl {
        if tpset.contains(p) {
            match subst.get(p) {
                Some(prev) if prev != actual => {
                    return Err(format!(
                        "in '{fname}': conflicting types for parameter '{p}' ({} vs {})",
                        ty_str(prev),
                        ty_str(actual)
                    ).into());
                }
                Some(_) => {}
                None => {
                    subst.insert(p.clone(), actual.clone());
                }
            }
            return Ok(());
        }
    }
    match (decl, actual) {
        (Type::Ptr(a), Type::Ptr(b)) => unify(a, b, tpset, subst, fname),
        // A reference on either side — a by-reference parameter, or a place
        // argument — unifies its pointee against the other side. So a by-ref
        // aggregate parameter matches a value, a place, or a raw-pointer
        // argument of the pointee type (it is read-as-value, borrowed, or
        // spilled). Subsumes the (ref,ref)/(ref,ptr)/(ptr,ref) place cases.
        (Type::Ref(_, a), b) => unify(a, b, tpset, subst, fname),
        (a, Type::Ref(_, b)) => unify(a, b, tpset, subst, fname),
        (Type::Array(a, _), Type::Array(b, _)) => unify(a, b, tpset, subst, fname),
        (Type::Slice(a), Type::Slice(b)) => unify(a, b, tpset, subst, fname),
        (Type::App(n1, a1), Type::App(n2, a2)) if n1 == n2 && a1.len() == a2.len() => {
            for (x, y) in a1.iter().zip(a2) {
                unify(x, y, tpset, subst, fname)?;
            }
            Ok(())
        }
        (Type::Fn(_, p1, r1), Type::Fn(_, p2, r2)) if p1.len() == p2.len() => {
            for (x, y) in p1.iter().zip(p2) {
                unify(x, y, tpset, subst, fname)?;
            }
            unify(r1, r2, tpset, subst, fname)
        }
        // shapes differ (or concrete): nothing to bind here — the coercion step
        // will report a genuine mismatch if there is one.
        _ => Ok(()),
    }
}

/// Erase the reference tier to plain pointers: a `Ref` is represented as a
/// `Ptr` at the machine level, so once the const-correctness check has run the
/// elaborated program codegen sees carries only `Ptr`.
fn erase_refs(t: &Type) -> Type {
    match t {
        Type::Ref(_, p) | Type::Ptr(p) => Type::Ptr(Box::new(erase_refs(p))),
        Type::Array(e, n) => Type::Array(Box::new(erase_refs(e)), *n),
        Type::Fn(cc, ps, r) => Type::Fn(
            cc.clone(),
            ps.iter().map(erase_refs).collect(),
            Box::new(erase_refs(r)),
        ),
        _ => t.clone(),
    }
}

/// How a parameter's written type is seen inside the body: every aggregate — a
/// struct, a sum, or a fixed array (concrete or generic-instance) — is passed by
/// **immutable reference**; `(mut T)` is a mutable reference; scalars, pointers,
/// and bare type parameters are unchanged. A `(ref T)`/`(mut T)` written by the
/// user is already a reference and passes through. (Externs are never routed
/// through here, so the C ABI for by-value aggregate FFI is untouched.)
fn param_ref_type(
    t: &Type,
    structs: &HashMap<String, StructInfo>,
    sums: &HashMap<String, SumInfo>,
    tps: &HashSet<String>,
) -> Type {
    let is_aggregate = |name: &String| {
        !tps.contains(name) && (structs.contains_key(name) || sums.contains_key(name))
    };
    match t {
        Type::Ref(..) | Type::Ptr(..) => t.clone(),
        Type::Struct(name) if is_aggregate(name) => Type::Ref(false, Box::new(t.clone())),
        Type::App(name, _) if is_aggregate(name) => Type::Ref(false, Box::new(t.clone())),
        Type::Array(..) => Type::Ref(false, Box::new(t.clone())),
        _ => t.clone(),
    }
}

/// Coerce a call argument to a (possibly reference) parameter type. References
/// require the argument to be a *place*; a `(mut T)` parameter additionally
/// requires the argument be passed mutably — written `(mut x)`, or a raw pointer.
fn coerce_arg(
    is_mut_borrow: bool,
    ae: Expr,
    at: Type,
    want: &Type,
    is_extern: bool,
    fname: &str,
    what: &str,
) -> Result<Expr, Diag> {
    match want {
        Type::Ref(want_mut, wp) => {
            // A place argument is borrowed directly; an rvalue of the pointee
            // type is spilled to a fresh stack slot first (only for an immutable
            // reference — a mutable one needs a real, writable place).
            let ap = match place_pointee(&at) {
                Some(ap) => ap,
                None => {
                    if *want_mut {
                        return Err(format!(
                            "in '{fname}': {what} is a (mut {}) parameter; pass a mutable place as (mut …)",
                            ty_str(wp)
                        ).into());
                    }
                    if &at != &**wp {
                        return Err(format!(
                            "in '{fname}': {what} expects a reference to {}, got {}",
                            ty_str(wp),
                            ty_str(&at)
                        ).into());
                    }
                    // Spill the temporary and pass a pointer to it.
                    let sp = ae.span;
                    return Ok(Expr::new(ExprKind::SpillRef(Box::new(ae)), sp));
                }
            };
            if ap != &**wp {
                return Err(format!(
                    "in '{fname}': {what} has type {} but expected a reference to {}",
                    ty_str(&at),
                    ty_str(wp)
                ).into());
            }
            if *want_mut {
                let explicitly_mut = is_mut_borrow || matches!(at, Type::Ptr(_));
                if !explicitly_mut {
                    return Err(format!(
                        "in '{fname}': {what} is a (mut {}) parameter; pass a mutable place as (mut …)",
                        ty_str(wp)
                    ).into());
                }
                if !is_writable(&at) {
                    return Err(format!(
                        "in '{fname}': {what} cannot be mutably borrowed (it is immutable)"
                    ).into());
                }
            }
            Ok(ae) // the borrow was already erased to the underlying pointer
        }
        // A by-value struct/sum parameter (an `extern`/`c`-cc function taking an
        // aggregate by value) accepts the value directly, or a *place* holding it
        // — a `let`-bound aggregate is materialized into a stack place, so a bare
        // `x` referring to one is a `(ref S)`. The place's pointer is passed
        // through; codegen reads the aggregate from it and applies the C ABI
        // coercion. (Without this, by-value aggregate arguments would be
        // unusable, since every aggregate `let` binding is a reference.)
        Type::Struct(sname) => {
            if &at == want {
                return Ok(ae); // a bare aggregate value
            }
            if let Some(pointee) = place_pointee(&at) {
                if struct_name(pointee) == Some(sname.as_str()) {
                    return Ok(ae); // pass the place pointer; codegen loads + coerces
                }
            }
            coerce(ae, at, want, is_extern, fname, what)
        }
        // A WRITABLE place — a `(mut T)`/`(ptr T)` place, or an explicit `(mut x)`
        // borrow — auto-borrows to a `(ptr T)`: the address-of is implicit, since
        // a reference is already a pointer. Removes the `alloc-stack` dance for
        // `call-ptr` args, parser cursors, and hashmap key spills. An IMMUTABLE
        // place does NOT auto-borrow — a `(ptr T)` is writable, so that would
        // breach const-correctness — which is exactly what keeps the metal `ptr`
        // tier distinct from the reference tier (only explicit/`(mut)` places).
        Type::Ptr(wp) => {
            if let Some(pointee) = place_pointee(&at) {
                if pointee == &**wp && is_writable(&at) {
                    return Ok(ae); // the place's pointer IS the (ptr T) value
                }
            }
            coerce(ae, at, want, is_extern, fname, what)
        }
        _ => coerce(ae, at, want, is_extern, fname, what),
    }
}

/// True for aggregate value types (a struct or array) — values that a `let`
/// binding materializes into a stack place so they can be borrowed/field-accessed.
/// Sums stay by-value (they're matched, not field-mutated).
fn is_place_value_type(t: &Type, cx: &Cx) -> bool {
    match t {
        Type::Struct(name) => cx.structs.contains_key(name),
        Type::App(name, _) => cx.structs.contains_key(name),
        Type::Array(..) => true,
        _ => false,
    }
}

/// The pointee of a reference or pointer (a "place" type), if `t` is one.
fn place_pointee(t: &Type) -> Option<&Type> {
    match t {
        Type::Ref(_, p) | Type::Ptr(p) => Some(p),
        _ => None,
    }
}

/// True if a place of this type may be written through (a `mut` ref, or a raw
/// pointer — pointers are always writable).
fn is_writable(t: &Type) -> bool {
    matches!(t, Type::Ref(true, _) | Type::Ptr(_))
}

/// Build a place type with the same kind/mutability as `place` but a new
/// pointee — so a field/element of a `(mut T)` is a `(mut field)`, of a `(ptr
/// T)` is a `(ptr field)`, etc.
fn replace_pointee(place: &Type, new: Type) -> Type {
    match place {
        Type::Ref(m, _) => Type::Ref(*m, Box::new(new)),
        _ => Type::Ptr(Box::new(new)),
    }
}

/// Substitute bound type parameters throughout a type.
fn subst_apply(t: &Type, subst: &HashMap<String, Type>) -> Type {
    match t {
        Type::Never => Type::Never,
        Type::Int(..) | Type::Float(..) | Type::Bool | Type::Void => t.clone(),
        Type::Ptr(p) => Type::Ptr(Box::new(subst_apply(p, subst))),
        Type::Ref(m, p) => Type::Ref(*m, Box::new(subst_apply(p, subst))),
        Type::Array(e, n) => Type::Array(Box::new(subst_apply(e, subst)), *n),
        Type::Slice(e) => Type::Slice(Box::new(subst_apply(e, subst))),
        Type::Vec(e, n) => Type::Vec(Box::new(subst_apply(e, subst)), *n),
        Type::Struct(name) => subst.get(name).cloned().unwrap_or_else(|| t.clone()),
        Type::App(name, args) => {
            Type::App(name.clone(), args.iter().map(|a| subst_apply(a, subst)).collect())
        }
        Type::Fn(cc, ps, r) => Type::Fn(
            cc.clone(),
            ps.iter().map(|p| subst_apply(p, subst)).collect(),
            Box::new(subst_apply(r, subst)),
        ),
    }
}

/// The nominal name of a struct-ish type (`Struct(n)` or `App(n, …)`).
fn struct_name(t: &Type) -> Option<&str> {
    match t {
        Type::Struct(n) | Type::App(n, _) => Some(n),
        _ => None,
    }
}

/// The fields of a struct type, with any generic arguments substituted in.
fn struct_fields(t: &Type, cx: &Cx) -> Result<Vec<(String, Type)>, String> {
    match t {
        Type::Struct(name) => {
            let si = cx
                .structs
                .get(name)
                .ok_or_else(|| format!("unknown struct '{name}'"))?;
            Ok(si.fields.clone())
        }
        Type::App(name, args) => {
            let si = cx
                .structs
                .get(name)
                .ok_or_else(|| format!("unknown generic struct '{name}'"))?;
            let map: HashMap<String, Type> =
                si.type_params.iter().cloned().zip(args.iter().cloned()).collect();
            Ok(si
                .fields
                .iter()
                .map(|(n, ft)| (n.clone(), subst_apply(ft, &map)))
                .collect())
        }
        _ => Err(format!("not a struct: {}", ty_str(t))),
    }
}

/// The variants of a sum type (`Struct(n)` concrete or `App(n, …)` generic),
/// with generic arguments substituted into each variant's field types.
fn sum_variants(t: &Type, cx: &Cx) -> Option<(String, Vec<SumVariant>)> {
    match t {
        Type::Struct(name) => {
            let si = cx.sums.get(name)?;
            Some((name.clone(), si.variants.clone()))
        }
        Type::App(name, args) => {
            let si = cx.sums.get(name)?;
            let map: HashMap<String, Type> =
                si.type_params.iter().cloned().zip(args.iter().cloned()).collect();
            let variants = si
                .variants
                .iter()
                .map(|v| SumVariant {
                    name: v.name.clone(),
                    fields: v
                        .fields
                        .iter()
                        .map(|(n, ft)| (n.clone(), subst_apply(ft, &map)))
                        .collect(),
                })
                .collect();
            Some((name.clone(), variants))
        }
        _ => None,
    }
}

/// Resolve a bitfield access: `ptr` must be a pointer to a `:layout bits`
/// struct; returns the elaborated pointer expr and the field's `uN` value type.
fn bit_field_type(
    ptr: &Expr,
    field: &str,
    env: &mut HashMap<String, Type>,
    cx: &Cx,
    tps: &HashSet<String>,
    fname: &str,
) -> Result<(Expr, Type), Diag> {
    let (pe, pt) = synth(ptr, None, env, cx, tps, fname)?;
    let pointee = match pt {
        Type::Ptr(p) => *p,
        other => {
            return Err(format!(
                "in '{fname}': get/set! needs a pointer, got {}",
                ty_str(&other)
            ).into())
        }
    };
    match struct_name(&pointee) {
        Some(s) if cx.structs.get(s).is_some_and(|si| si.is_bits) => {
            let fty = struct_fields(&pointee, cx)
                .ok()
                .and_then(|fs| fs.into_iter().find(|(n, _)| n == field).map(|(_, t)| t))
                .ok_or_else(|| format!("in '{fname}': bit struct '{s}' has no field '{field}'"))?;
            Ok((pe, fty))
        }
        _ => Err(format!(
            "in '{fname}': get/set! needs a pointer to a :layout bits struct, got (ptr {})",
            ty_str(&pointee)
        ).into()),
    }
}

/// True if `e` is an integer *literal* — a value whose type isn't yet pinned and
/// can adopt a concrete `iN`/`uN` from context.
fn is_literal(e: &Expr) -> bool {
    match &e.kind {
        ExprKind::Int(_) | ExprKind::Float(_) => true,
        ExprKind::Bin { lhs, rhs, .. } => is_literal(lhs) && is_literal(rhs),
        ExprKind::If { then, els, .. } => is_literal(then) && is_literal(els),
        ExprKind::Do(es) => es.last().is_some_and(is_literal),
        ExprKind::Let { body, .. } => body.last().is_some_and(is_literal),
        _ => false,
    }
}

/// An integer type, or an opaque (in-scope) type parameter — both are valid
/// operands of an arithmetic/comparison op (a type parameter is assumed numeric;
/// each monomorphic instantiation is what actually reaches codegen).
fn numeric(t: &Type, tps: &HashSet<String>) -> bool {
    matches!(t, Type::Int(..) | Type::Float(..)) || matches!(t, Type::Struct(n) if tps.contains(n))
}

/// Valid as an `if`/`static-assert` condition: a bool or any integer (nonzero
/// = true), so both `(icmp-lt …)` and a raw flag work.
fn is_cond(t: &Type) -> bool {
    matches!(t, Type::Bool | Type::Int(..))
}

/// Two scalar types that a literal may slide between (both ints, or both floats).
fn same_number_kind(a: &Type, b: &Type) -> bool {
    matches!(
        (a, b),
        (Type::Int(..), Type::Int(..)) | (Type::Float(..), Type::Float(..))
    )
}

/// Whether the integer `n` is representable in `bits` with the given signedness.
fn fits(n: i64, bits: u32, signed: bool) -> bool {
    if bits >= 64 {
        return true;
    }
    if signed {
        let lo = -(1i64 << (bits - 1));
        let hi = (1i64 << (bits - 1)) - 1;
        n >= lo && n <= hi
    } else if bits >= 63 {
        n >= 0
    } else {
        n >= 0 && n < (1i64 << bits)
    }
}

/// Wrap a literal expression in an explicit cast to `target`. A bare `Int(n)` is
/// range-checked so an out-of-range literal is rejected, not silently truncated.
/// Is `name` already taken by a function/extern signature or another const?
fn cx_sig_or_const(
    sigs: &HashMap<String, Sig>,
    consts: &HashMap<String, (Expr, Type)>,
    name: &str,
) -> bool {
    sigs.contains_key(name) || consts.contains_key(name)
}

/// Build a const's (literal expression, reported type) entry. An untyped const
/// reports the literal default (so it re-infers like an inline literal); a typed
/// const pins the type, requiring the value's kind to match and (for integers)
/// to fit.
fn const_entry(c: &Const) -> Result<(Expr, Type), Diag> {
    let bad = |ty: &Type, kind: &str| {
        format!("const '{}': {kind} value cannot have type {}", c.name, ty_str(ty))
    };
    match c.value {
        ConstLit::Int(n) => match &c.ty {
            None => Ok((Expr::dummy(ExprKind::Int(n)), Type::Int(64, true))),
            Some(Type::Int(bits, signed)) => {
                if !fits(n, *bits, *signed) {
                    return Err(format!(
                        "const '{}': literal {n} does not fit in {}",
                        c.name,
                        ty_str(c.ty.as_ref().unwrap())
                    ).into());
                }
                Ok((Expr::dummy(ExprKind::Int(n)), Type::Int(*bits, *signed)))
            }
            Some(other) => Err(bad(other, "integer").into()),
        },
        ConstLit::Float(x) => match &c.ty {
            None => Ok((Expr::dummy(ExprKind::Float(x)), Type::Float(64))),
            Some(Type::Float(b)) => Ok((Expr::dummy(ExprKind::Float(x)), Type::Float(*b))),
            Some(other) => Err(bad(other, "float").into()),
        },
        ConstLit::Bool(b) => match &c.ty {
            None | Some(Type::Bool) => Ok((Expr::dummy(ExprKind::Bool(b)), Type::Bool)),
            Some(other) => Err(bad(other, "boolean").into()),
        },
    }
}

fn coerce_lit(e: Expr, target: &Type, fname: &str) -> Result<Expr, Diag> {
    if let (ExprKind::Int(n), Type::Int(bits, signed)) = (&e.kind, target) {
        if !fits(*n, *bits, *signed) {
            return Err(format!(
                "in '{fname}': literal {n} does not fit in {}",
                ty_str(target)
            ).into());
        }
    }
    let sp = e.span;
    Ok(Expr::new(ExprKind::Cast {
        ty: target.clone(),
        expr: Box::new(e),
    }, sp))
}

/// Coerce an elaborated expression of type `et` to the expected `target` at a
/// boundary. Exact types pass through; a literal adopts the target's integer
/// type; at an `extern` boundary any pointer matches any pointer.
fn coerce(
    e: Expr,
    et: Type,
    target: &Type,
    is_extern: bool,
    fname: &str,
    what: &str,
) -> Result<Expr, Diag> {
    if &et == target {
        return Ok(e);
    }
    // A `void`-returning function's body runs for effect: any expression is
    // accepted in the return position (its value, if any, is discarded). This is
    // the ONLY place `target == Void` (params/fields can't be void).
    if target == &Type::Void {
        return Ok(e);
    }
    // Forbid USING a void value where a value is needed — a call to a
    // `(-> void)` function yields nothing (no-silent-wrong).
    if et == Type::Void {
        return Err(format!(
            "in '{fname}': {what} uses a void value (a (-> void) call yields nothing)"
        ).into());
    }
    // A divergent (Never-typed) expression — break/continue/return-from, or a
    // break-less loop — has no value and stands in for any type.
    if et == Type::Never {
        return Ok(e);
    }
    if same_number_kind(&et, target) && is_literal(&e) {
        return coerce_lit(e, target, fname);
    }
    if arg_ok(&et, target, is_extern) {
        return Ok(e);
    }
    // A reference reads AS its pointee value where a value is expected — e.g.
    // copying a by-reference struct param out via `store!`/return. Without this,
    // struct params (passed by immutable reference) couldn't be used as values,
    // while sum params (passed by value) could — an inconsistency. This is a READ
    // (an immutable ref reads fine); storing THROUGH a ref is the separate
    // `is_writable` check, so const-correctness is unaffected.
    if let Type::Ref(_, inner) = &et {
        if &**inner == target {
            let sp = e.span;
            return Ok(Expr::new(ExprKind::Load(Box::new(e)), sp));
        }
    }
    Err(format!(
        "in '{fname}': {what} has type {} but expected {}",
        ty_str(&et),
        ty_str(target)
    ).into())
}

/// Reconcile a set of sibling expressions to one common type: a literal sibling
/// adopts the lone concrete type; all-literal collapses to `i64`; two distinct
/// concrete types are an error.
fn unify_branches(
    branches: Vec<(Expr, Type)>,
    fname: &str,
    what: &str,
) -> Result<(Vec<Expr>, Type), Diag> {
    let mut concrete: Option<Type> = None;
    for (e, t) in &branches {
        // A Never-typed branch (one that diverges) is non-constraining — like a
        // literal, it adopts the other branch's concrete type. So
        // `(if c (do …T…) (break))` reconciles to T (no dummy value needed).
        if !is_literal(e) && t != &Type::Never {
            match &concrete {
                None => concrete = Some(t.clone()),
                Some(c) if c != t => return Err(mismatch_msg(c, t, fname, what).into()),
                _ => {}
            }
        }
    }
    // With no concrete sibling, the result is the literals' own default type
    // (i64 for integer literals, f64 for float literals) — skipping any Never
    // (divergent) branch, which never constrains the result.
    let target = concrete
        .or_else(|| {
            branches
                .iter()
                .find(|(_, t)| t != &Type::Never)
                .map(|(_, t)| t.clone())
        })
        .unwrap_or(Type::Int(64, true));
    let mut out = Vec::with_capacity(branches.len());
    for (e, t) in branches {
        if t == target || t == Type::Never {
            // a Never branch diverges; keep it as-is (it yields no value).
            out.push(e);
        } else if same_number_kind(&t, &target) && is_literal(&e) {
            out.push(coerce_lit(e, &target, fname)?);
        } else {
            return Err(mismatch_msg(&t, &target, fname, what).into());
        }
    }
    Ok((out, target))
}

fn mismatch_msg(a: &Type, b: &Type, fname: &str, what: &str) -> String {
    match (a, b) {
        (Type::Int(ab, _), Type::Int(bb, _)) => {
            if ab != bb {
                format!("in '{fname}': {what} on mixed widths ({ab} and {bb} bits)")
            } else {
                format!(
                    "in '{fname}': {what} on mixed signedness ({} and {})",
                    ty_str(a),
                    ty_str(b)
                )
            }
        }
        _ => format!(
            "in '{fname}': {what} on different types ({} vs {})",
            ty_str(a),
            ty_str(b)
        ),
    }
}

/// Argument-type compatibility. Normally exact; at an `extern` boundary any
/// pointer matches any pointer (like passing to `void*`).
fn arg_ok(got: &Type, want: &Type, is_extern: bool) -> bool {
    match (got, want) {
        _ if got == want => true,
        (Type::Ptr(_), Type::Ptr(_)) if is_extern => true,
        _ => false,
    }
}

fn ty_str(t: &Type) -> String {
    match t {
        Type::Never => "!".to_string(),
        Type::Void => "void".to_string(),
        Type::Int(bits, signed) => format!("{}{bits}", if *signed { "i" } else { "u" }),
        Type::Float(bits) => format!("f{bits}"),
        Type::Bool => "bool".to_string(),
        Type::Ptr(pointee) => format!("(ptr {})", ty_str(pointee)),
        Type::Ref(true, pointee) => format!("(mut {})", ty_str(pointee)),
        Type::Ref(false, pointee) => format!("(ref {})", ty_str(pointee)),
        Type::Struct(name) => name.clone(),
        Type::Array(e, n) => format!("(array {} {n})", ty_str(e)),
        Type::Slice(e) => format!("(slice {})", ty_str(e)),
        Type::Vec(e, n) => format!("(vec {} {n})", ty_str(e)),
        Type::Fn(cc, params, ret) => {
            let ps: Vec<String> = params.iter().map(ty_str).collect();
            format!("(fnptr {cc} [{}] {})", ps.join(" "), ty_str(ret))
        }
        Type::App(name, args) => {
            let a: Vec<String> = args.iter().map(ty_str).collect();
            format!("({name} {})", a.join(" "))
        }
    }
}
