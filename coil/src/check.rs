//! Type + convention checks, plus **bidirectional elaboration**.
//!
//! Beyond typing it enforces convention well-formedness (M1/M2) and the
//! structural rules of the language (field/index typing, bit-struct access,
//! exhaustive `match`, every named type exists).
//!
//! The checker is also an **elaborator**: integer literals start out flexible
//! (default `i64`) and adopt a concrete `iN`/`uN` type from their context. Where
//! a literal meets a known type — the other operand of an arithmetic op, the
//! other branch of an `if`/`match`, a store/return/call/construct target — the
//! checker inserts an explicit `(cast :iN ...)` so codegen stays untouched (it
//! just sees the cast). That removes the cast-noise of writing `(cast :u8 42)`
//! everywhere: `(store! p 42)` into a `(ptr u8)` just works, and `(iadd x 1)`
//! with `x : u8` keeps `1` at `u8`. A bare literal that cannot fit its inferred
//! type is a compile error.
//!
//! `check` returns the elaborated `Program` (the same program with literals
//! coerced); `lib.rs` feeds *that* to codegen.

use std::collections::{HashMap, HashSet};

use crate::ast::*;

struct Sig {
    params: Vec<Type>,
    ret: Type,
    /// Calls to externs erase pointer regions at the boundary (see `arg_ok`).
    is_extern: bool,
    /// The calling convention's name, and whether a function pointer can be
    /// taken to it (only native conventions, for now).
    cc: String,
    fnptr_ok: bool,
}

/// Everything `synth` needs besides the local variable environment.
struct Cx {
    sigs: HashMap<String, Sig>,
    structs: HashMap<String, Vec<(String, Type)>>,
    sums: HashMap<String, Vec<SumVariant>>,
    /// Names of all known nominal types (structs + sums).
    known: HashSet<String>,
    /// Structs whose layout is `:layout bits` (accessed via get/set!, not field).
    bit_structs: HashSet<String>,
}

pub fn check(program: &Program) -> Result<Program, String> {
    // type tables
    let mut structs: HashMap<String, Vec<(String, Type)>> = HashMap::new();
    for sd in &program.structs {
        if structs.insert(sd.name.clone(), sd.fields.clone()).is_some() {
            return Err(format!("struct '{}' defined twice", sd.name));
        }
    }
    let mut sums: HashMap<String, Vec<SumVariant>> = HashMap::new();
    for sd in &program.sums {
        if sums.insert(sd.name.clone(), sd.variants.clone()).is_some() {
            return Err(format!("sum '{}' defined twice", sd.name));
        }
    }
    let known: HashSet<String> = structs.keys().chain(sums.keys()).cloned().collect();
    let bit_structs: HashSet<String> = program
        .structs
        .iter()
        .filter(|s| matches!(s.layout, Layout::Bits(_)))
        .map(|s| s.name.clone())
        .collect();

    for sd in &program.structs {
        let mut seen = HashSet::new();
        for (fname, fty) in &sd.fields {
            if !seen.insert(fname) {
                return Err(format!("struct '{}': duplicate field '{fname}'", sd.name));
            }
            validate_type(fty, &known)
                .map_err(|e| format!("struct '{}' field '{fname}': {e}", sd.name))?;
        }
    }
    for sd in &program.sums {
        for v in &sd.variants {
            for (fname, fty) in &v.fields {
                validate_type(fty, &known).map_err(|e| {
                    format!("sum '{}' variant '{}' field '{fname}': {e}", sd.name, v.name)
                })?;
            }
        }
    }

    // signatures (functions + externs)
    let mut sigs: HashMap<String, Sig> = HashMap::new();
    for f in &program.funcs {
        let native = program.conventions.get(&f.cc).is_some_and(|c| !c.is_shim());
        sigs.insert(
            f.name.clone(),
            Sig {
                params: f.params.iter().map(|p| p.ty.clone()).collect(),
                ret: f.ret.clone(),
                is_extern: false,
                cc: f.cc.clone(),
                fnptr_ok: native,
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
        known,
        bit_structs,
    };

    // Elaborate every function body, rebuilding it with literal coercions in
    // place. The non-expr parts of the program are carried through unchanged.
    let mut funcs: Vec<Func> = Vec::with_capacity(program.funcs.len());
    for f in &program.funcs {
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

        // signature types must be well-formed (pointers are region-less, so
        // there's no escape rule — a stack pointer can cross a boundary).
        for p in &f.params {
            validate_type(&p.ty, &cx.known)
                .map_err(|e| format!("function '{}' param '{}': {e}", f.name, p.name))?;
        }
        validate_type(&f.ret, &cx.known)
            .map_err(|e| format!("function '{}' return type: {e}", f.name))?;

        let mut env: HashMap<String, Type> =
            f.params.iter().map(|p| (p.name.clone(), p.ty.clone())).collect();

        // Elaborate each body expression; the last one is coerced to the
        // declared return type (so `(defn f [] (-> :u8) 42)` works).
        let mut body: Vec<Expr> = Vec::with_capacity(f.body.len());
        let n = f.body.len();
        for (i, e) in f.body.iter().enumerate() {
            let (ee, et) = synth(e, &mut env, &cx, &f.name)?;
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

    // static-assert conditions must type-check to an integer (codegen evaluates
    // them; here we just ensure they're well-formed and elaborate any literals).
    let mut asserts: Vec<StaticAssert> = Vec::with_capacity(program.asserts.len());
    for a in &program.asserts {
        let mut env: HashMap<String, Type> = HashMap::new();
        let (cond, t) = synth(&a.cond, &mut env, &cx, "static-assert")?;
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

fn validate_type(t: &Type, known: &HashSet<String>) -> Result<(), String> {
    match t {
        Type::Int(..) => Ok(()),
        Type::Ptr(p) => validate_type(p, known),
        Type::Array(e, _) => validate_type(e, known),
        Type::Struct(name) => {
            if known.contains(name) {
                Ok(())
            } else {
                Err(format!("unknown type '{name}'"))
            }
        }
        Type::Fn(_, params, ret) => {
            for p in params {
                validate_type(p, known)?;
            }
            validate_type(ret, known)
        }
        Type::App(name, _) => Err(format!("internal: unresolved generic type '{name}'")),
    }
}

/// Synthesize the type of `e` *and* return an elaborated copy of `e` with any
/// literal coercions inserted.
fn synth(
    e: &Expr,
    env: &mut HashMap<String, Type>,
    cx: &Cx,
    fname: &str,
) -> Result<(Expr, Type), String> {
    match e {
        Expr::Int(n) => Ok((Expr::Int(*n), Type::Int(64, true))),
        Expr::Var(name) => {
            let t = env
                .get(name)
                .cloned()
                .ok_or_else(|| format!("in '{fname}': unbound variable '{name}'"))?;
            Ok((Expr::Var(name.clone()), t))
        }
        Expr::Bin { op, lhs, rhs } => {
            let (le, lt) = synth(lhs, env, cx, fname)?;
            let (re, rt) = synth(rhs, env, cx, fname)?;
            let (mut sides, t) =
                unify_branches(vec![(le, lt), (re, rt)], fname, "arithmetic")?;
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
            let (le, lt) = synth(lhs, env, cx, fname)?;
            let (re, rt) = synth(rhs, env, cx, fname)?;
            let (mut sides, _) =
                unify_branches(vec![(le, lt), (re, rt)], fname, "comparison")?;
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
            let (ce, ct) = synth(cond, env, cx, fname)?;
            if !ct.is_int() {
                return Err(format!(
                    "in '{fname}': if condition must be an integer, got {}",
                    ty_str(&ct)
                ));
            }
            let (te, tt) = synth(then, env, cx, fname)?;
            let (ee, et) = synth(els, env, cx, fname)?;
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
                let (ee, et) = synth(e, env, cx, fname)?;
                last = et;
                out.push(ee);
            }
            Ok((Expr::Do(out), last))
        }
        Expr::Let { binds, body } => {
            let saved = env.clone();
            let mut new_binds = Vec::with_capacity(binds.len());
            for (name, val) in binds {
                let (ve, vt) = synth(val, env, cx, fname)?;
                env.insert(name.clone(), vt);
                new_binds.push((name.clone(), ve));
            }
            let mut out = Vec::with_capacity(body.len());
            let mut last = Type::Int(64, true);
            for e in body {
                let (ee, et) = synth(e, env, cx, fname)?;
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
            let mut new_args = Vec::with_capacity(args.len());
            for (i, a) in args.iter().enumerate() {
                let (ae, at) = synth(a, env, cx, fname)?;
                let want = sig.params[i].clone();
                let is_extern = sig.is_extern;
                let ae = coerce(
                    ae,
                    at,
                    &want,
                    is_extern,
                    fname,
                    &format!("argument {} to '{func}'", i + 1),
                )?;
                new_args.push(ae);
            }
            Ok((
                Expr::Call {
                    func: func.clone(),
                    type_args: type_args.clone(),
                    args: new_args,
                },
                sig.ret.clone(),
            ))
        }
        Expr::Alloc { storage, ty } => {
            validate_type(ty, &cx.known).map_err(|e| format!("in '{fname}': alloc: {e}"))?;
            Ok((
                Expr::Alloc {
                    storage: *storage,
                    ty: ty.clone(),
                },
                Type::Ptr(Box::new(ty.clone())),
            ))
        }
        Expr::BitGet { ptr, field } => {
            let (pe, fty) = bit_field_type(ptr, field, env, cx, fname)?;
            Ok((
                Expr::BitGet {
                    ptr: Box::new(pe),
                    field: field.clone(),
                },
                fty,
            ))
        }
        Expr::BitSet { ptr, field, val } => {
            let (pe, fty) = bit_field_type(ptr, field, env, cx, fname)?;
            let (ve, vt) = synth(val, env, cx, fname)?;
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
            let (pe, pt) = synth(ptr, env, cx, fname)?;
            let fty = match pt {
                Type::Ptr(pointee) => match *pointee {
                    Type::Struct(s) if cx.bit_structs.contains(&s) => return Err(format!(
                        "in '{fname}': '{s}' is a :layout bits struct; use (get p {field}) / (set! p {field} v)"
                    )),
                    Type::Struct(s) => {
                        let fields = cx
                            .structs
                            .get(&s)
                            .ok_or_else(|| format!("in '{fname}': unknown struct '{s}'"))?;
                        fields
                            .iter()
                            .find(|(n, _)| n == field)
                            .map(|(_, t)| t.clone())
                            .ok_or_else(|| {
                                format!("in '{fname}': struct '{s}' has no field '{field}'")
                            })?
                    }
                    other => return Err(format!(
                        "in '{fname}': field access needs a pointer to a struct, got (ptr {})",
                        ty_str(&other)
                    )),
                },
                other => {
                    return Err(format!(
                        "in '{fname}': field access needs a pointer, got {}",
                        ty_str(&other)
                    ))
                }
            };
            Ok((
                Expr::Field {
                    ptr: Box::new(pe),
                    field: field.clone(),
                },
                Type::Ptr(Box::new(fty)),
            ))
        }
        Expr::Load(p) => {
            let (pe, pt) = synth(p, env, cx, fname)?;
            match pt {
                Type::Ptr(pointee) => Ok((Expr::Load(Box::new(pe)), *pointee)),
                other => Err(format!(
                    "in '{fname}': load expects a pointer, got {}",
                    ty_str(&other)
                )),
            }
        }
        Expr::Store { ptr, val } => {
            let (pe, pt) = synth(ptr, env, cx, fname)?;
            match pt {
                Type::Ptr(pointee) => {
                    let (ve, vt) = synth(val, env, cx, fname)?;
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
            let (pe, pt) = synth(ptr, env, cx, fname)?;
            let (ie, it) = synth(idx, env, cx, fname)?;
            // The index is an i64; a bare literal already synthesizes as i64.
            let ie = coerce(ie, it, &Type::Int(64, true), false, fname, "index")
                .map_err(|_| format!("in '{fname}': index must be i64"))?;
            match pt {
                // pointer to an array: element access; otherwise pointer arithmetic.
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
            validate_type(ty, &cx.known).map_err(|e| format!("in '{fname}': cast target: {e}"))?;
            let (ee, et) = synth(expr, env, cx, fname)?;
            match (ty, &et) {
                // integer width conversion, or a pointer reinterpret.
                (Type::Int(..), Type::Int(..)) | (Type::Ptr(..), Type::Ptr(..)) => Ok((
                    Expr::Cast {
                        ty: ty.clone(),
                        expr: Box::new(ee),
                    },
                    ty.clone(),
                )),
                _ => Err(format!(
                    "in '{fname}': cast only converts int<->int or ptr<->ptr (got {} to {})",
                    ty_str(&et),
                    ty_str(ty)
                )),
            }
        }
        Expr::SizeOf(ty) => {
            validate_type(ty, &cx.known).map_err(|e| format!("in '{fname}': sizeof: {e}"))?;
            Ok((Expr::SizeOf(ty.clone()), Type::Int(64, true)))
        }
        Expr::AlignOf(ty) => {
            validate_type(ty, &cx.known).map_err(|e| format!("in '{fname}': alignof: {e}"))?;
            Ok((Expr::AlignOf(ty.clone()), Type::Int(64, true)))
        }
        Expr::OffsetOf(ty, field) => match ty {
            Type::Struct(name) => {
                let fields = cx
                    .structs
                    .get(name)
                    .ok_or_else(|| format!("in '{fname}': offsetof: '{name}' is not a struct"))?;
                if !fields.iter().any(|(n, _)| n == field) {
                    return Err(format!(
                        "in '{fname}': offsetof: struct '{name}' has no field '{field}'"
                    ));
                }
                Ok((Expr::OffsetOf(ty.clone(), field.clone()), Type::Int(64, true)))
            }
            other => Err(format!(
                "in '{fname}': offsetof needs a struct type, got {}",
                ty_str(other)
            )),
        },
        Expr::Free(p) => {
            let (pe, pt) = synth(p, env, cx, fname)?;
            match pt {
                // any pointer can be freed (it calls libc free). Freeing stack/
                // static memory is your problem — like C.
                Type::Ptr(_) => Ok((Expr::Free(Box::new(pe)), Type::Int(64, true))),
                other => Err(format!(
                    "in '{fname}': free expects a pointer, got {}",
                    ty_str(&other)
                )),
            }
        }
        Expr::Construct { sum, variant, args } => {
            let variants = cx
                .sums
                .get(sum)
                .ok_or_else(|| format!("in '{fname}': unknown sum type '{sum}'"))?;
            let v = variants
                .iter()
                .find(|v| &v.name == variant)
                .ok_or_else(|| format!("in '{fname}': sum '{sum}' has no variant '{variant}'"))?;
            if v.fields.len() != args.len() {
                return Err(format!(
                    "in '{fname}': variant '{variant}' takes {} field(s), got {}",
                    v.fields.len(),
                    args.len()
                ));
            }
            let mut new_args = Vec::with_capacity(args.len());
            for (i, a) in args.iter().enumerate() {
                let (ae, at) = synth(a, env, cx, fname)?;
                let want = v.fields[i].1.clone();
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
            Ok((
                Expr::Construct {
                    sum: sum.clone(),
                    variant: variant.clone(),
                    args: new_args,
                },
                Type::Struct(sum.clone()),
            ))
        }
        Expr::Match { scrut, arms } => {
            let (se, st) = synth(scrut, env, cx, fname)?;
            let sumname = match st {
                Type::Struct(s) if cx.sums.contains_key(&s) => s,
                other => {
                    return Err(format!(
                        "in '{fname}': match expects a sum value, got {}",
                        ty_str(&other)
                    ))
                }
            };
            let variants = &cx.sums[&sumname];
            let mut covered: HashSet<&str> = HashSet::new();
            // Elaborate each arm body first, then unify their types together so a
            // literal arm adopts a concrete sibling's type.
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
                let (be, bt) = synth(&arm.body, env, cx, fname)?;
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
            let (fe, ft) = synth(fp, env, cx, fname)?;
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
                        let (ae, at) = synth(a, env, cx, fname)?;
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

/// Resolve a bitfield access: `ptr` must be a pointer to a `:layout bits`
/// struct; returns the elaborated pointer expr and the field's `uN` value type.
fn bit_field_type(
    ptr: &Expr,
    field: &str,
    env: &mut HashMap<String, Type>,
    cx: &Cx,
    fname: &str,
) -> Result<(Expr, Type), String> {
    let (pe, pt) = synth(ptr, env, cx, fname)?;
    match pt {
        Type::Ptr(pointee) => match *pointee {
            Type::Struct(s) if cx.bit_structs.contains(&s) => {
                let fty = cx
                    .structs
                    .get(&s)
                    .and_then(|fs| fs.iter().find(|(n, _)| n == field))
                    .map(|(_, t)| t.clone())
                    .ok_or_else(|| {
                        format!("in '{fname}': bit struct '{s}' has no field '{field}'")
                    })?;
                Ok((pe, fty))
            }
            other => Err(format!(
                "in '{fname}': get/set! needs a pointer to a :layout bits struct, got (ptr {})",
                ty_str(&other)
            )),
        },
        other => Err(format!(
            "in '{fname}': get/set! needs a pointer, got {}",
            ty_str(&other)
        )),
    }
}

/// True if `e` is an integer *literal* — a value whose type isn't yet pinned and
/// can adopt a concrete `iN`/`uN` from context. Compound expressions made
/// entirely of literals (an `iadd` of two literals, an `if` whose branches are
/// literals, a `do`/`let` whose result is a literal) are themselves literals.
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

/// Wrap a literal expression in an explicit cast to `target`. For a bare
/// `Int(n)` the value is range-checked so an out-of-range literal is rejected
/// rather than silently truncated.
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
/// boundary (store/return/call argument/construct field/bitfield set). Exact
/// types pass through; a literal adopts the target's integer type; at an
/// `extern` boundary any pointer matches any pointer. Anything else is an error
/// whose text embeds `what` (so callers get e.g. "argument 1 to 'putchar'").
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

/// Reconcile a set of sibling expressions (operands of an arithmetic/comparison
/// op, the branches of an `if`, the arms of a `match`) to one common type. If
/// exactly one concrete (non-literal) type appears, every literal sibling is
/// coerced to it; if all siblings are literals the result is `i64`. Two distinct
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

/// Argument-type compatibility. Normally exact; at an `extern` boundary the
/// foreign side doesn't care about pointee types, so any pointer matches any
/// pointer (like passing to `void*`).
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
