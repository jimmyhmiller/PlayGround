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

struct Sig {
    /// Generic type parameters (empty for an ordinary function/extern).
    type_params: Vec<String>,
    params: Vec<Type>,
    ret: Type,
    /// Calls to externs erase pointer regions at the boundary (see `arg_ok`).
    is_extern: bool,
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
}

pub fn check(program: &Program) -> Result<Program, String> {
    // ---- type tables --------------------------------------------------------
    let mut structs: HashMap<String, StructInfo> = HashMap::new();
    for sd in &program.structs {
        let info = StructInfo {
            type_params: sd.type_params.clone(),
            fields: sd.fields.clone(),
            is_bits: matches!(sd.layout, Layout::Bits(_)),
        };
        if structs.insert(sd.name.clone(), info).is_some() {
            return Err(format!("struct '{}' defined twice", sd.name));
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
            return Err(format!("sum '{}' defined twice", sd.name));
        }
        for v in &sd.variants {
            if variant_to_sum.insert(v.name.clone(), sd.name.clone()).is_some() {
                return Err(format!("variant '{}' is declared in two sum types", v.name));
            }
        }
    }
    // ---- signatures (functions + externs) ----------------------------------
    let mut sigs: HashMap<String, Sig> = HashMap::new();
    for f in &program.funcs {
        let native = program.conventions.get(&f.cc).is_some_and(|c| !c.is_shim());
        sigs.insert(
            f.name.clone(),
            Sig {
                type_params: f.type_params.clone(),
                params: f.params.iter().map(|p| p.ty.clone()).collect(),
                ret: f.ret.clone(),
                is_extern: false,
                cc: f.cc.clone(),
                // a generic function has no single address, so no fnptr.
                fnptr_ok: native && f.type_params.is_empty(),
            },
        );
    }
    for e in &program.externs {
        if sigs.contains_key(&e.name) {
            return Err(format!("'{}' is declared more than once", e.name));
        }
        let conv = program
            .conventions
            .get(&e.cc)
            .ok_or_else(|| format!("extern '{}': unknown convention '{}'", e.name, e.cc))?;
        if conv.is_shim() {
            return Err(format!(
                "extern '{}': shim conventions for externs are not supported yet",
                e.name
            ));
        }
        sigs.insert(
            e.name.clone(),
            Sig {
                type_params: vec![],
                params: e.params.clone(),
                ret: e.ret.clone(),
                is_extern: true,
                cc: e.cc.clone(),
                fnptr_ok: true,
            },
        );
    }

    let cx = Cx {
        sigs,
        structs,
        sums,
        variant_to_sum,
    };

    // ---- validate struct/sum field types (each with its own params in scope)
    for sd in &program.structs {
        let tps: HashSet<String> = sd.type_params.iter().cloned().collect();
        let mut seen = HashSet::new();
        for (fname, fty) in &sd.fields {
            if !seen.insert(fname) {
                return Err(format!("struct '{}': duplicate field '{fname}'", sd.name));
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
                ));
            }
            if conv.params.len() < f.params.len() {
                return Err(format!(
                    "function '{}': convention '{}' provides {} param registers but the \
                     function has {} parameters",
                    f.name,
                    f.cc,
                    conv.params.len(),
                    f.params.len()
                ));
            }
        }

        for p in &f.params {
            validate_type(&p.ty, &cx, &tps)
                .map_err(|e| format!("function '{}' param '{}': {e}", f.name, p.name))?;
        }
        validate_type(&f.ret, &cx, &tps)
            .map_err(|e| format!("function '{}' return type: {e}", f.name))?;

        let mut env: HashMap<String, Type> =
            f.params.iter().map(|p| (p.name.clone(), p.ty.clone())).collect();

        let mut body: Vec<Expr> = Vec::with_capacity(f.body.len());
        let n = f.body.len();
        for (i, e) in f.body.iter().enumerate() {
            let (ee, et) = synth(e, &mut env, &cx, &tps, &f.name)?;
            if i + 1 == n {
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
                body.push(ee);
            }
        }
        if n == 0 && f.ret != Type::Int(64, true) {
            return Err(format!(
                "function '{}': body has type i64 but the declared return type is {}",
                f.name,
                ty_str(&f.ret)
            ));
        }

        funcs.push(Func {
            name: f.name.clone(),
            type_params: f.type_params.clone(),
            cc: f.cc.clone(),
            params: f.params.clone(),
            ret: f.ret.clone(),
            body,
        });
    }

    // ---- static-assert conditions ------------------------------------------
    let empty_tps = HashSet::new();
    let mut asserts: Vec<StaticAssert> = Vec::with_capacity(program.asserts.len());
    for a in &program.asserts {
        let mut env: HashMap<String, Type> = HashMap::new();
        let (cond, t) = synth(&a.cond, &mut env, &cx, &empty_tps, "static-assert")?;
        if !t.is_int() {
            return Err(format!(
                "static-assert: condition must be an integer, got {}",
                ty_str(&t)
            ));
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
    })
}

/// Validate that a type is well-formed: every named type exists (or is an
/// in-scope type parameter), and every generic application has the right arity.
fn validate_type(t: &Type, cx: &Cx, tps: &HashSet<String>) -> Result<(), String> {
    match t {
        Type::Int(..) => Ok(()),
        Type::Ptr(p) => validate_type(p, cx, tps),
        Type::Array(e, _) => validate_type(e, cx, tps),
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
                    ))
                }
            } else if let Some(si) = cx.sums.get(name) {
                if si.type_params.is_empty() {
                    Ok(())
                } else {
                    Err(format!(
                        "generic type '{name}' expects {} type arguments, got 0",
                        si.type_params.len()
                    ))
                }
            } else {
                Err(format!("unknown type '{name}'"))
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
                ));
            }
            Ok(())
        }
    }
}

/// Synthesize the type of `e` and return an elaborated copy with literal
/// coercions and inferred type arguments inserted. `tps` is the set of type
/// parameters in scope (opaque types) for the function being checked.
fn synth(
    e: &Expr,
    env: &mut HashMap<String, Type>,
    cx: &Cx,
    tps: &HashSet<String>,
    fname: &str,
) -> Result<(Expr, Type), String> {
    match e {
        Expr::Int(n) => Ok((Expr::Int(*n), Type::Int(64, true))),
        Expr::Str(s) => Ok((Expr::Str(s.clone()), Type::Ptr(Box::new(Type::Int(8, true))))),
        Expr::Var(name) => {
            let t = env
                .get(name)
                .cloned()
                .ok_or_else(|| format!("in '{fname}': unbound variable '{name}'"))?;
            Ok((Expr::Var(name.clone()), t))
        }
        Expr::Bin { op, lhs, rhs } => {
            let (le, lt) = synth(lhs, env, cx, tps, fname)?;
            let (re, rt) = synth(rhs, env, cx, tps, fname)?;
            let (mut sides, t) =
                unify_branches(vec![(le, lt), (re, rt)], fname, "arithmetic")?;
            if !numeric(&t, tps) {
                return Err(format!(
                    "in '{fname}': arithmetic requires integers, got {}",
                    ty_str(&t)
                ));
            }
            let rhs = sides.pop().unwrap();
            let lhs = sides.pop().unwrap();
            Ok((
                Expr::Bin {
                    op: *op,
                    lhs: Box::new(lhs),
                    rhs: Box::new(rhs),
                },
                t,
            ))
        }
        Expr::Cmp { op, lhs, rhs } => {
            let (le, lt) = synth(lhs, env, cx, tps, fname)?;
            let (re, rt) = synth(rhs, env, cx, tps, fname)?;
            let (mut sides, t) =
                unify_branches(vec![(le, lt), (re, rt)], fname, "comparison")?;
            if !numeric(&t, tps) {
                return Err(format!(
                    "in '{fname}': comparison requires integers, got {}",
                    ty_str(&t)
                ));
            }
            let rhs = sides.pop().unwrap();
            let lhs = sides.pop().unwrap();
            Ok((
                Expr::Cmp {
                    op: *op,
                    lhs: Box::new(lhs),
                    rhs: Box::new(rhs),
                },
                Type::Int(64, true),
            ))
        }
        Expr::If { cond, then, els } => {
            let (ce, ct) = synth(cond, env, cx, tps, fname)?;
            if !ct.is_int() {
                return Err(format!(
                    "in '{fname}': if condition must be an integer, got {}",
                    ty_str(&ct)
                ));
            }
            let (te, tt) = synth(then, env, cx, tps, fname)?;
            let (ee, et) = synth(els, env, cx, tps, fname)?;
            let (mut branches, t) =
                unify_branches(vec![(te, tt), (ee, et)], fname, "if branches")?;
            let els = branches.pop().unwrap();
            let then = branches.pop().unwrap();
            Ok((
                Expr::If {
                    cond: Box::new(ce),
                    then: Box::new(then),
                    els: Box::new(els),
                },
                t,
            ))
        }
        Expr::Do(es) => {
            let mut out = Vec::with_capacity(es.len());
            let mut last = Type::Int(64, true);
            for e in es {
                let (ee, et) = synth(e, env, cx, tps, fname)?;
                last = et;
                out.push(ee);
            }
            Ok((Expr::Do(out), last))
        }
        Expr::Let { binds, body } => {
            let saved = env.clone();
            let mut new_binds = Vec::with_capacity(binds.len());
            for (name, val) in binds {
                let (ve, vt) = synth(val, env, cx, tps, fname)?;
                env.insert(name.clone(), vt);
                new_binds.push((name.clone(), ve));
            }
            let mut out = Vec::with_capacity(body.len());
            let mut last = Type::Int(64, true);
            for e in body {
                let (ee, et) = synth(e, env, cx, tps, fname)?;
                last = et;
                out.push(ee);
            }
            *env = saved; // bindings are lexical
            Ok((
                Expr::Let {
                    binds: new_binds,
                    body: out,
                },
                last,
            ))
        }
        Expr::Call {
            func,
            type_args,
            args,
        } => {
            // Variant construction (the parser emits it as a call to the variant
            // name) routes here so we can type and infer its sum's type args.
            if cx.variant_to_sum.contains_key(func) {
                return synth_construct(func, type_args, args, env, cx, tps, fname);
            }
            let sig = cx
                .sigs
                .get(func)
                .ok_or_else(|| format!("in '{fname}': call to undefined function '{func}'"))?;
            if sig.params.len() != args.len() {
                return Err(format!(
                    "in '{fname}': '{func}' expects {} args, got {}",
                    sig.params.len(),
                    args.len()
                ));
            }
            // synthesize the arguments
            let mut arglist: Vec<(Expr, Type)> = Vec::with_capacity(args.len());
            for a in args {
                arglist.push(synth(a, env, cx, tps, fname)?);
            }

            // resolve the type-parameter substitution (explicit or inferred)
            let subst = solve_type_args(
                &sig.type_params,
                &sig.params,
                type_args,
                &arglist,
                cx,
                tps,
                fname,
                func,
            )?;

            // coerce each argument to its (substituted) parameter type
            let mut new_args = Vec::with_capacity(args.len());
            for (i, (ae, at)) in arglist.into_iter().enumerate() {
                let want = subst_apply(&sig.params[i], &subst);
                let ae = coerce(
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
                Expr::Call {
                    func: func.clone(),
                    type_args: out_type_args,
                    args: new_args,
                },
                ret,
            ))
        }
        Expr::Alloc { storage, ty } => {
            validate_type(ty, cx, tps).map_err(|e| format!("in '{fname}': alloc: {e}"))?;
            Ok((
                Expr::Alloc {
                    storage: *storage,
                    ty: ty.clone(),
                },
                Type::Ptr(Box::new(ty.clone())),
            ))
        }
        Expr::BitGet { ptr, field } => {
            let (pe, fty) = bit_field_type(ptr, field, env, cx, tps, fname)?;
            Ok((
                Expr::BitGet {
                    ptr: Box::new(pe),
                    field: field.clone(),
                },
                fty,
            ))
        }
        Expr::BitSet { ptr, field, val } => {
            let (pe, fty) = bit_field_type(ptr, field, env, cx, tps, fname)?;
            let (ve, vt) = synth(val, env, cx, tps, fname)?;
            let ve = coerce(
                ve,
                vt,
                &fty,
                false,
                fname,
                &format!("set! into bitfield '{field}'"),
            )?;
            Ok((
                Expr::BitSet {
                    ptr: Box::new(pe),
                    field: field.clone(),
                    val: Box::new(ve),
                },
                fty,
            ))
        }
        Expr::Field { ptr, field } => {
            let (pe, pt) = synth(ptr, env, cx, tps, fname)?;
            let pointee = match pt {
                Type::Ptr(p) => *p,
                other => {
                    return Err(format!(
                        "in '{fname}': field access needs a pointer, got {}",
                        ty_str(&other)
                    ))
                }
            };
            let sname = struct_name(&pointee).ok_or_else(|| {
                format!(
                    "in '{fname}': field access needs a pointer to a struct, got (ptr {})",
                    ty_str(&pointee)
                )
            })?;
            if cx.structs.get(sname).is_some_and(|s| s.is_bits) {
                return Err(format!(
                    "in '{fname}': '{sname}' is a :layout bits struct; use (get p {field}) / (set! p {field} v)"
                ));
            }
            let fields = struct_fields(&pointee, cx)
                .map_err(|e| format!("in '{fname}': {e}"))?;
            let fty = fields
                .iter()
                .find(|(n, _)| n == field)
                .map(|(_, t)| t.clone())
                .ok_or_else(|| format!("in '{fname}': struct '{sname}' has no field '{field}'"))?;
            Ok((
                Expr::Field {
                    ptr: Box::new(pe),
                    field: field.clone(),
                },
                Type::Ptr(Box::new(fty)),
            ))
        }
        Expr::Load(p) => {
            let (pe, pt) = synth(p, env, cx, tps, fname)?;
            match pt {
                Type::Ptr(pointee) => Ok((Expr::Load(Box::new(pe)), *pointee)),
                other => Err(format!(
                    "in '{fname}': load expects a pointer, got {}",
                    ty_str(&other)
                )),
            }
        }
        Expr::Store { ptr, val } => {
            let (pe, pt) = synth(ptr, env, cx, tps, fname)?;
            match pt {
                Type::Ptr(pointee) => {
                    let (ve, vt) = synth(val, env, cx, tps, fname)?;
                    let ve = coerce(ve, vt, &pointee, false, fname, "store! value")?;
                    Ok((
                        Expr::Store {
                            ptr: Box::new(pe),
                            val: Box::new(ve),
                        },
                        *pointee,
                    ))
                }
                other => Err(format!(
                    "in '{fname}': store! expects a pointer, got {}",
                    ty_str(&other)
                )),
            }
        }
        Expr::Index { ptr, idx } => {
            let (pe, pt) = synth(ptr, env, cx, tps, fname)?;
            let (ie, it) = synth(idx, env, cx, tps, fname)?;
            let ie = coerce(ie, it, &Type::Int(64, true), false, fname, "index")
                .map_err(|_| format!("in '{fname}': index must be i64"))?;
            match pt {
                Type::Ptr(pointee) => {
                    let elem_ptr = match *pointee {
                        Type::Array(elem, _) => Type::Ptr(elem),
                        p => Type::Ptr(Box::new(p)),
                    };
                    Ok((
                        Expr::Index {
                            ptr: Box::new(pe),
                            idx: Box::new(ie),
                        },
                        elem_ptr,
                    ))
                }
                other => Err(format!(
                    "in '{fname}': index expects a pointer, got {}",
                    ty_str(&other)
                )),
            }
        }
        Expr::Cast { ty, expr } => {
            validate_type(ty, cx, tps).map_err(|e| format!("in '{fname}': cast target: {e}"))?;
            let (ee, et) = synth(expr, env, cx, tps, fname)?;
            match (ty, &et) {
                // int<->int width change, ptr<->ptr reinterpret, or int<->ptr
                // (address arithmetic, null, MMIO, tagged pointers).
                (Type::Int(..), Type::Int(..))
                | (Type::Ptr(..), Type::Ptr(..))
                | (Type::Int(..), Type::Ptr(..))
                | (Type::Ptr(..), Type::Int(..)) => Ok((
                    Expr::Cast {
                        ty: ty.clone(),
                        expr: Box::new(ee),
                    },
                    ty.clone(),
                )),
                _ => Err(format!(
                    "in '{fname}': cast only converts among int and ptr (got {} to {})",
                    ty_str(&et),
                    ty_str(ty)
                )),
            }
        }
        Expr::SizeOf(ty) => {
            validate_type(ty, cx, tps).map_err(|e| format!("in '{fname}': sizeof: {e}"))?;
            Ok((Expr::SizeOf(ty.clone()), Type::Int(64, true)))
        }
        Expr::AlignOf(ty) => {
            validate_type(ty, cx, tps).map_err(|e| format!("in '{fname}': alignof: {e}"))?;
            Ok((Expr::AlignOf(ty.clone()), Type::Int(64, true)))
        }
        Expr::OffsetOf(ty, field) => {
            validate_type(ty, cx, tps).map_err(|e| format!("in '{fname}': offsetof: {e}"))?;
            let fields = struct_fields(ty, cx)
                .map_err(|_| format!("in '{fname}': offsetof needs a struct type, got {}", ty_str(ty)))?;
            if !fields.iter().any(|(n, _)| n == field) {
                let name = struct_name(ty).unwrap_or("?");
                return Err(format!(
                    "in '{fname}': offsetof: struct '{name}' has no field '{field}'"
                ));
            }
            Ok((Expr::OffsetOf(ty.clone(), field.clone()), Type::Int(64, true)))
        }
        Expr::Free(p) => {
            let (pe, pt) = synth(p, env, cx, tps, fname)?;
            match pt {
                Type::Ptr(_) => Ok((Expr::Free(Box::new(pe)), Type::Int(64, true))),
                other => Err(format!(
                    "in '{fname}': free expects a pointer, got {}",
                    ty_str(&other)
                )),
            }
        }
        Expr::Construct { sum, variant, args } => {
            // The parser never emits Construct (it uses Call); kept for safety.
            synth_construct(variant, &[], args, env, cx, tps, fname).map(|(e, _)| {
                (e, Type::Struct(sum.clone()))
            })
        }
        Expr::Match { scrut, arms } => {
            let (se, st) = synth(scrut, env, cx, tps, fname)?;
            let (sumname, variants) = sum_variants(&st, cx).ok_or_else(|| {
                format!("in '{fname}': match expects a sum value, got {}", ty_str(&st))
            })?;
            let mut covered: HashSet<&str> = HashSet::new();
            let mut bodies: Vec<(Expr, Type)> = Vec::with_capacity(arms.len());
            let mut meta: Vec<(String, Vec<String>)> = Vec::with_capacity(arms.len());
            for arm in arms {
                let v = variants.iter().find(|v| v.name == arm.variant).ok_or_else(|| {
                    format!("in '{fname}': sum '{sumname}' has no variant '{}'", arm.variant)
                })?;
                if !covered.insert(arm.variant.as_str()) {
                    return Err(format!("in '{fname}': duplicate match arm for '{}'", arm.variant));
                }
                if arm.binds.len() != v.fields.len() {
                    return Err(format!(
                        "in '{fname}': arm '{}' binds {} name(s) but the variant has {} field(s)",
                        arm.variant,
                        arm.binds.len(),
                        v.fields.len()
                    ));
                }
                let saved = env.clone();
                for (b, (_, fty)) in arm.binds.iter().zip(&v.fields) {
                    env.insert(b.clone(), fty.clone());
                }
                let (be, bt) = synth(&arm.body, env, cx, tps, fname)?;
                *env = saved;
                bodies.push((be, bt));
                meta.push((arm.variant.clone(), arm.binds.clone()));
            }
            if covered.len() != variants.len() {
                return Err(format!(
                    "in '{fname}': non-exhaustive match on '{sumname}' ({} of {} variants)",
                    covered.len(),
                    variants.len()
                ));
            }
            if bodies.is_empty() {
                return Err(format!("in '{fname}': empty match"));
            }
            let (coerced, t) = unify_branches(bodies, fname, "match arms")?;
            let new_arms = meta
                .into_iter()
                .zip(coerced)
                .map(|((variant, binds), body)| Arm {
                    variant,
                    binds,
                    body,
                })
                .collect();
            Ok((
                Expr::Match {
                    scrut: Box::new(se),
                    arms: new_arms,
                },
                t,
            ))
        }
        Expr::FnPtrOf(name) => {
            let sig = cx
                .sigs
                .get(name)
                .ok_or_else(|| format!("in '{fname}': fnptr-of unknown function '{name}'"))?;
            if !sig.type_params.is_empty() {
                return Err(format!(
                    "in '{fname}': cannot take a function pointer to generic function '{name}'"
                ));
            }
            if !sig.fnptr_ok {
                return Err(format!(
                    "in '{fname}': cannot take a function pointer to '{name}' \
                     (shim-convention functions are not supported yet)"
                ));
            }
            Ok((
                Expr::FnPtrOf(name.clone()),
                Type::Fn(sig.cc.clone(), sig.params.clone(), Box::new(sig.ret.clone())),
            ))
        }
        Expr::CallPtr { fp, args } => {
            let (fe, ft) = synth(fp, env, cx, tps, fname)?;
            match ft {
                Type::Fn(_, params, ret) => {
                    if params.len() != args.len() {
                        return Err(format!(
                            "in '{fname}': function pointer expects {} args, got {}",
                            params.len(),
                            args.len()
                        ));
                    }
                    let mut new_args = Vec::with_capacity(args.len());
                    for (i, a) in args.iter().enumerate() {
                        let (ae, at) = synth(a, env, cx, tps, fname)?;
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
                        Expr::CallPtr {
                            fp: Box::new(fe),
                            args: new_args,
                        },
                        *ret,
                    ))
                }
                other => Err(format!(
                    "in '{fname}': call-ptr expects a function pointer, got {}",
                    ty_str(&other)
                )),
            }
        }
    }
}

/// Type and elaborate a variant construction `(Variant [targs?] args…)`. The
/// sum's type arguments are read (explicit) or inferred from the field types,
/// written back onto the call so monomorphization can specialize it.
fn synth_construct(
    variant: &str,
    type_args: &[Type],
    args: &[Expr],
    env: &mut HashMap<String, Type>,
    cx: &Cx,
    tps: &HashSet<String>,
    fname: &str,
) -> Result<(Expr, Type), String> {
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
        ));
    }

    let mut arglist: Vec<(Expr, Type)> = Vec::with_capacity(args.len());
    for a in args {
        arglist.push(synth(a, env, cx, tps, fname)?);
    }

    let field_types: Vec<Type> = v.fields.iter().map(|(_, t)| t.clone()).collect();
    let subst = solve_type_args(
        &si.type_params,
        &field_types,
        type_args,
        &arglist,
        cx,
        tps,
        fname,
        variant,
    )?;

    let mut new_args = Vec::with_capacity(args.len());
    for (i, (ae, at)) in arglist.into_iter().enumerate() {
        let want = subst_apply(&field_types[i], &subst);
        let ae = coerce(
            ae,
            at,
            &want,
            false,
            fname,
            &format!("'{variant}' field {}", i + 1),
        )?;
        new_args.push(ae);
    }

    if si.type_params.is_empty() {
        // concrete sum: leave type_args empty, result is the plain sum type.
        Ok((
            Expr::Call {
                func: variant.to_string(),
                type_args: vec![],
                args: new_args,
            },
            Type::Struct(sumname),
        ))
    } else {
        let out_targs: Vec<Type> =
            si.type_params.iter().map(|p| subst[p].clone()).collect();
        Ok((
            Expr::Call {
                func: variant.to_string(),
                type_args: out_targs.clone(),
                args: new_args,
            },
            Type::App(sumname, out_targs),
        ))
    }
}

/// Resolve the substitution for a generic call/construction. With explicit
/// `type_args` it validates arity and uses them directly; otherwise it infers
/// each type parameter by unifying the declared parameter/field types against
/// the actual argument types.
#[allow(clippy::too_many_arguments)]
fn solve_type_args(
    type_params: &[String],
    declared: &[Type],
    type_args: &[Type],
    arglist: &[(Expr, Type)],
    cx: &Cx,
    tps: &HashSet<String>,
    fname: &str,
    what: &str,
) -> Result<HashMap<String, Type>, String> {
    if type_params.is_empty() {
        if !type_args.is_empty() {
            return Err(format!("in '{fname}': '{what}' is not generic but got type arguments"));
        }
        return Ok(HashMap::new());
    }
    if !type_args.is_empty() {
        if type_args.len() != type_params.len() {
            return Err(format!(
                "in '{fname}': '{what}' expects {} type arguments, got {}",
                type_params.len(),
                type_args.len()
            ));
        }
        for ta in type_args {
            validate_type(ta, cx, tps).map_err(|e| format!("in '{fname}': type argument: {e}"))?;
        }
        return Ok(type_params.iter().cloned().zip(type_args.iter().cloned()).collect());
    }
    // infer
    let tpset: HashSet<String> = type_params.iter().cloned().collect();
    let mut subst: HashMap<String, Type> = HashMap::new();
    for (decl, (_, at)) in declared.iter().zip(arglist) {
        unify(decl, at, &tpset, &mut subst, fname)?;
    }
    for p in type_params {
        if !subst.contains_key(p) {
            return Err(format!(
                "in '{fname}': cannot infer type argument '{p}' for '{what}'; \
                 provide it explicitly: ({what} [<types>] ...)"
            ));
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
) -> Result<(), String> {
    if let Type::Struct(p) = decl {
        if tpset.contains(p) {
            match subst.get(p) {
                Some(prev) if prev != actual => {
                    return Err(format!(
                        "in '{fname}': conflicting types for parameter '{p}' ({} vs {})",
                        ty_str(prev),
                        ty_str(actual)
                    ));
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
        (Type::Array(a, _), Type::Array(b, _)) => unify(a, b, tpset, subst, fname),
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

/// Substitute bound type parameters throughout a type.
fn subst_apply(t: &Type, subst: &HashMap<String, Type>) -> Type {
    match t {
        Type::Int(..) => t.clone(),
        Type::Ptr(p) => Type::Ptr(Box::new(subst_apply(p, subst))),
        Type::Array(e, n) => Type::Array(Box::new(subst_apply(e, subst)), *n),
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
) -> Result<(Expr, Type), String> {
    let (pe, pt) = synth(ptr, env, cx, tps, fname)?;
    let pointee = match pt {
        Type::Ptr(p) => *p,
        other => {
            return Err(format!(
                "in '{fname}': get/set! needs a pointer, got {}",
                ty_str(&other)
            ))
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
        )),
    }
}

/// True if `e` is an integer *literal* — a value whose type isn't yet pinned and
/// can adopt a concrete `iN`/`uN` from context.
fn is_literal(e: &Expr) -> bool {
    match e {
        Expr::Int(_) => true,
        Expr::Bin { lhs, rhs, .. } => is_literal(lhs) && is_literal(rhs),
        Expr::If { then, els, .. } => is_literal(then) && is_literal(els),
        Expr::Do(es) => es.last().is_some_and(is_literal),
        Expr::Let { body, .. } => body.last().is_some_and(is_literal),
        _ => false,
    }
}

/// An integer type, or an opaque (in-scope) type parameter — both are valid
/// operands of an arithmetic/comparison op (a type parameter is assumed numeric;
/// each monomorphic instantiation is what actually reaches codegen).
fn numeric(t: &Type, tps: &HashSet<String>) -> bool {
    t.is_int() || matches!(t, Type::Struct(n) if tps.contains(n))
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
fn coerce_lit(e: Expr, target: &Type, fname: &str) -> Result<Expr, String> {
    if let (Expr::Int(n), Type::Int(bits, signed)) = (&e, target) {
        if !fits(*n, *bits, *signed) {
            return Err(format!(
                "in '{fname}': literal {n} does not fit in {}",
                ty_str(target)
            ));
        }
    }
    Ok(Expr::Cast {
        ty: target.clone(),
        expr: Box::new(e),
    })
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
) -> Result<Expr, String> {
    if &et == target {
        return Ok(e);
    }
    if matches!((&et, target), (Type::Int(..), Type::Int(..))) && is_literal(&e) {
        return coerce_lit(e, target, fname);
    }
    if arg_ok(&et, target, is_extern) {
        return Ok(e);
    }
    Err(format!(
        "in '{fname}': {what} has type {} but expected {}",
        ty_str(&et),
        ty_str(target)
    ))
}

/// Reconcile a set of sibling expressions to one common type: a literal sibling
/// adopts the lone concrete type; all-literal collapses to `i64`; two distinct
/// concrete types are an error.
fn unify_branches(
    branches: Vec<(Expr, Type)>,
    fname: &str,
    what: &str,
) -> Result<(Vec<Expr>, Type), String> {
    let mut concrete: Option<Type> = None;
    for (e, t) in &branches {
        if !is_literal(e) {
            match &concrete {
                None => concrete = Some(t.clone()),
                Some(c) if c != t => return Err(mismatch_msg(c, t, fname, what)),
                _ => {}
            }
        }
    }
    let target = concrete.unwrap_or(Type::Int(64, true));
    let mut out = Vec::with_capacity(branches.len());
    for (e, t) in branches {
        if t == target {
            out.push(e);
        } else if matches!((&t, &target), (Type::Int(..), Type::Int(..))) && is_literal(&e) {
            out.push(coerce_lit(e, &target, fname)?);
        } else {
            return Err(mismatch_msg(&t, &target, fname, what));
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
        Type::Int(bits, signed) => format!("{}{bits}", if *signed { "i" } else { "u" }),
        Type::Ptr(pointee) => format!("(ptr {})", ty_str(pointee)),
        Type::Struct(name) => name.clone(),
        Type::Array(e, n) => format!("(array {} {n})", ty_str(e)),
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
