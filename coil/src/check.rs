//! Type + convention + region checks (bidirectional, synthesis-only).
//!
//! Beyond typing it enforces convention well-formedness (M1/M2) and region
//! soundness (M3): `frame` pointers may not cross a function boundary, and
//! `free` only accepts a `heap` pointer. Structs/arrays (this milestone) add
//! `field`/`index` typing and validation that every named type exists.

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
}

pub fn check(program: &Program) -> Result<(), String> {
    // struct table
    let mut structs: HashMap<String, Vec<(String, Type)>> = HashMap::new();
    for sd in &program.structs {
        if structs.insert(sd.name.clone(), sd.fields.clone()).is_some() {
            return Err(format!("struct '{}' defined twice", sd.name));
        }
    }
    for sd in &program.structs {
        let mut seen = HashSet::new();
        for (fname, fty) in &sd.fields {
            if !seen.insert(fname) {
                return Err(format!("struct '{}': duplicate field '{fname}'", sd.name));
            }
            validate_type(fty, &structs)
                .map_err(|e| format!("struct '{}' field '{fname}': {e}", sd.name))?;
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

    let cx = Cx { sigs, structs };

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
            validate_type(&p.ty, &cx.structs)
                .map_err(|e| format!("function '{}' param '{}': {e}", f.name, p.name))?;
        }
        validate_type(&f.ret, &cx.structs)
            .map_err(|e| format!("function '{}' return type: {e}", f.name))?;

        let mut env: HashMap<String, Type> =
            f.params.iter().map(|p| (p.name.clone(), p.ty.clone())).collect();
        let mut last = Type::Int(64);
        for e in &f.body {
            last = synth(e, &mut env, &cx, &f.name)?;
        }
        if last != f.ret {
            return Err(format!(
                "function '{}': body has type {} but the declared return type is {}",
                f.name,
                ty_str(&last),
                ty_str(&f.ret)
            ));
        }
    }
    Ok(())
}

fn validate_type(t: &Type, structs: &HashMap<String, Vec<(String, Type)>>) -> Result<(), String> {
    match t {
        Type::Int(w) => {
            if matches!(w, 8 | 16 | 32 | 64) {
                Ok(())
            } else {
                Err(format!("unsupported integer width i{w}"))
            }
        }
        Type::Ptr(p) => validate_type(p, structs),
        Type::Array(e, _) => validate_type(e, structs),
        Type::Struct(name) => {
            if structs.contains_key(name) {
                Ok(())
            } else {
                Err(format!("unknown struct '{name}'"))
            }
        }
        Type::Fn(_, params, ret) => {
            for p in params {
                validate_type(p, structs)?;
            }
            validate_type(ret, structs)
        }
        Type::App(name, _) => Err(format!("internal: unresolved generic type '{name}'")),
    }
}

fn synth(
    e: &Expr,
    env: &mut HashMap<String, Type>,
    cx: &Cx,
    fname: &str,
) -> Result<Type, String> {
    match e {
        Expr::Int(_) => Ok(Type::Int(64)),
        Expr::Var(name) => env
            .get(name)
            .cloned()
            .ok_or_else(|| format!("in '{fname}': unbound variable '{name}'")),
        Expr::Bin { lhs, rhs, .. } => {
            let lw = int_width(synth(lhs, env, cx, fname)?, fname, "arithmetic operand")?;
            let rw = int_width(synth(rhs, env, cx, fname)?, fname, "arithmetic operand")?;
            if lw != rw {
                return Err(format!("in '{fname}': arithmetic on mixed widths i{lw} and i{rw}"));
            }
            Ok(Type::Int(lw))
        }
        Expr::Cmp { lhs, rhs, .. } => {
            let lw = int_width(synth(lhs, env, cx, fname)?, fname, "comparison operand")?;
            let rw = int_width(synth(rhs, env, cx, fname)?, fname, "comparison operand")?;
            if lw != rw {
                return Err(format!("in '{fname}': comparison on mixed widths i{lw} and i{rw}"));
            }
            Ok(Type::Int(64))
        }
        Expr::If { cond, then, els } => {
            int_width(synth(cond, env, cx, fname)?, fname, "if condition")?;
            let t = synth(then, env, cx, fname)?;
            let e = synth(els, env, cx, fname)?;
            if t != e {
                return Err(format!(
                    "in '{fname}': if branches have different types ({} vs {})",
                    ty_str(&t),
                    ty_str(&e)
                ));
            }
            Ok(t)
        }
        Expr::Do(es) => {
            let mut last = Type::Int(64);
            for e in es {
                last = synth(e, env, cx, fname)?;
            }
            Ok(last)
        }
        Expr::Let { binds, body } => {
            let saved = env.clone();
            for (name, val) in binds {
                let t = synth(val, env, cx, fname)?;
                env.insert(name.clone(), t);
            }
            let mut last = Type::Int(64);
            for e in body {
                last = synth(e, env, cx, fname)?;
            }
            *env = saved; // bindings are lexical
            Ok(last)
        }
        Expr::Call { func, args, .. } => {
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
            for (i, a) in args.iter().enumerate() {
                let at = synth(a, env, cx, fname)?;
                if !arg_ok(&at, &sig.params[i], sig.is_extern) {
                    return Err(format!(
                        "in '{fname}': argument {} to '{func}' has type {} but expected {}",
                        i + 1,
                        ty_str(&at),
                        ty_str(&sig.params[i])
                    ));
                }
            }
            Ok(sig.ret.clone())
        }
        Expr::Alloc { ty, .. } => {
            validate_type(ty, &cx.structs).map_err(|e| format!("in '{fname}': alloc: {e}"))?;
            Ok(Type::Ptr(Box::new(ty.clone())))
        }
        Expr::Field { ptr, field } => match synth(ptr, env, cx, fname)? {
            Type::Ptr(pointee) => match *pointee {
                Type::Struct(s) => {
                    let fields = cx
                        .structs
                        .get(&s)
                        .ok_or_else(|| format!("in '{fname}': unknown struct '{s}'"))?;
                    let fty = fields
                        .iter()
                        .find(|(n, _)| n == field)
                        .map(|(_, t)| t.clone())
                        .ok_or_else(|| format!("in '{fname}': struct '{s}' has no field '{field}'"))?;
                    Ok(Type::Ptr(Box::new(fty)))
                }
                other => Err(format!(
                    "in '{fname}': field access needs a pointer to a struct, got (ptr {})",
                    ty_str(&other)
                )),
            },
            other => Err(format!(
                "in '{fname}': field access needs a pointer, got {}",
                ty_str(&other)
            )),
        },
        Expr::Load(p) => match synth(p, env, cx, fname)? {
            Type::Ptr(pointee) => Ok(*pointee),
            other => Err(format!(
                "in '{fname}': load expects a pointer, got {}",
                ty_str(&other)
            )),
        },
        Expr::Store { ptr, val } => match synth(ptr, env, cx, fname)? {
            Type::Ptr(pointee) => {
                let vt = synth(val, env, cx, fname)?;
                if vt != *pointee {
                    return Err(format!(
                        "in '{fname}': store! of {} through a pointer to {}",
                        ty_str(&vt),
                        ty_str(&pointee)
                    ));
                }
                Ok(*pointee)
            }
            other => Err(format!(
                "in '{fname}': store! expects a pointer, got {}",
                ty_str(&other)
            )),
        },
        Expr::Index { ptr, idx } => {
            let pt = synth(ptr, env, cx, fname)?;
            let it = synth(idx, env, cx, fname)?;
            if it != Type::Int(64) {
                return Err(format!("in '{fname}': index must be i64, got {}", ty_str(&it)));
            }
            match pt {
                // pointer to an array: element access; otherwise pointer arithmetic.
                Type::Ptr(pointee) => match *pointee {
                    Type::Array(elem, _) => Ok(Type::Ptr(elem)),
                    p => Ok(Type::Ptr(Box::new(p))),
                },
                other => Err(format!(
                    "in '{fname}': index expects a pointer, got {}",
                    ty_str(&other)
                )),
            }
        }
        Expr::Cast { ty, expr } => {
            validate_type(ty, &cx.structs).map_err(|e| format!("in '{fname}': cast target: {e}"))?;
            let et = synth(expr, env, cx, fname)?;
            match (ty, &et) {
                // integer width conversion, or a pointer reinterpret.
                (Type::Int(_), Type::Int(_)) | (Type::Ptr(..), Type::Ptr(..)) => Ok(ty.clone()),
                _ => Err(format!(
                    "in '{fname}': cast only converts int<->int or ptr<->ptr (got {} to {})",
                    ty_str(&et),
                    ty_str(ty)
                )),
            }
        }
        Expr::SizeOf(ty) => {
            validate_type(ty, &cx.structs).map_err(|e| format!("in '{fname}': sizeof: {e}"))?;
            Ok(Type::Int(64))
        }
        Expr::Free(p) => match synth(p, env, cx, fname)? {
            // any pointer can be freed (it calls libc free). Freeing stack/static
            // memory is your problem — like C.
            Type::Ptr(_) => Ok(Type::Int(64)),
            other => Err(format!(
                "in '{fname}': free expects a pointer, got {}",
                ty_str(&other)
            )),
        },
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
            Ok(Type::Fn(
                sig.cc.clone(),
                sig.params.clone(),
                Box::new(sig.ret.clone()),
            ))
        }
        Expr::CallPtr { fp, args } => match synth(fp, env, cx, fname)? {
            Type::Fn(_, params, ret) => {
                if params.len() != args.len() {
                    return Err(format!(
                        "in '{fname}': function pointer expects {} args, got {}",
                        params.len(),
                        args.len()
                    ));
                }
                for (i, a) in args.iter().enumerate() {
                    let at = synth(a, env, cx, fname)?;
                    if !arg_ok(&at, &params[i], false) {
                        return Err(format!(
                            "in '{fname}': call-ptr argument {} has type {} but expected {}",
                            i + 1,
                            ty_str(&at),
                            ty_str(&params[i])
                        ));
                    }
                }
                Ok(*ret)
            }
            other => Err(format!(
                "in '{fname}': call-ptr expects a function pointer, got {}",
                ty_str(&other)
            )),
        },
    }
}

fn int_width(t: Type, fname: &str, what: &str) -> Result<u32, String> {
    match t {
        Type::Int(w) => Ok(w),
        other => Err(format!(
            "in '{fname}': {what} must be an integer, got {}",
            ty_str(&other)
        )),
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
        Type::Int(w) => format!("i{w}"),
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
