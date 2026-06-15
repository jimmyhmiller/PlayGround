//! M3 type + convention + region checks.
//!
//! The language now has two types (`i64` and `Ptr(region)`), so this is a real
//! (if small) bidirectional type checker. On top of typing it enforces:
//!
//! * **Convention well-formedness** — a shim convention must name a return
//!   register and enough argument registers for the function (M1/M2).
//! * **Region soundness (descriptive-but-checked)** — `frame` pointers may not
//!   cross a function boundary (no `frame` in any parameter or return type), and
//!   since pointees are `i64` there is no other way for one to escape. `free`
//!   only accepts a `heap` pointer. This is the design's deliberate middle path:
//!   not a full borrow checker, but enough to stop the obvious escapes.

use std::collections::HashMap;

use crate::ast::*;

struct Sig {
    params: Vec<Type>,
    ret: Type,
}

pub fn check(program: &Program) -> Result<(), String> {
    let sigs: HashMap<&str, Sig> = program
        .funcs
        .iter()
        .map(|f| {
            (
                f.name.as_str(),
                Sig {
                    params: f.params.iter().map(|p| p.ty.clone()).collect(),
                    ret: f.ret.clone(),
                },
            )
        })
        .collect();

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

        // region soundness: frame pointers can't cross function boundaries
        for p in &f.params {
            if is_frame_ptr(&p.ty) {
                return Err(format!(
                    "function '{}': parameter '{}' is a frame pointer; frame pointers \
                     cannot cross function boundaries",
                    f.name, p.name
                ));
            }
        }
        if is_frame_ptr(&f.ret) {
            return Err(format!(
                "function '{}': returns a frame pointer; frame pointers cannot escape \
                 their frame",
                f.name
            ));
        }

        // body must type-check and produce the declared return type
        let mut env: HashMap<String, Type> =
            f.params.iter().map(|p| (p.name.clone(), p.ty.clone())).collect();
        let mut last = Type::I64;
        for e in &f.body {
            last = synth(e, &mut env, &sigs, &f.name)?;
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

fn synth(
    e: &Expr,
    env: &mut HashMap<String, Type>,
    sigs: &HashMap<&str, Sig>,
    fname: &str,
) -> Result<Type, String> {
    match e {
        Expr::Int(_) => Ok(Type::I64),
        Expr::Var(name) => env
            .get(name)
            .cloned()
            .ok_or_else(|| format!("in '{fname}': unbound variable '{name}'")),
        Expr::Bin { lhs, rhs, .. } => {
            expect(synth(lhs, env, sigs, fname)?, &Type::I64, fname, "arithmetic operand")?;
            expect(synth(rhs, env, sigs, fname)?, &Type::I64, fname, "arithmetic operand")?;
            Ok(Type::I64)
        }
        Expr::Cmp { lhs, rhs, .. } => {
            expect(synth(lhs, env, sigs, fname)?, &Type::I64, fname, "comparison operand")?;
            expect(synth(rhs, env, sigs, fname)?, &Type::I64, fname, "comparison operand")?;
            Ok(Type::I64)
        }
        Expr::If { cond, then, els } => {
            expect(synth(cond, env, sigs, fname)?, &Type::I64, fname, "if condition")?;
            let t = synth(then, env, sigs, fname)?;
            let e = synth(els, env, sigs, fname)?;
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
            let mut last = Type::I64;
            for e in es {
                last = synth(e, env, sigs, fname)?;
            }
            Ok(last)
        }
        Expr::Let { binds, body } => {
            let saved = env.clone();
            for (name, val) in binds {
                let t = synth(val, env, sigs, fname)?;
                env.insert(name.clone(), t);
            }
            let mut last = Type::I64;
            for e in body {
                last = synth(e, env, sigs, fname)?;
            }
            *env = saved; // bindings are lexical
            Ok(last)
        }
        Expr::Call { func, args } => {
            let sig = sigs
                .get(func.as_str())
                .ok_or_else(|| format!("in '{fname}': call to undefined function '{func}'"))?;
            if sig.params.len() != args.len() {
                return Err(format!(
                    "in '{fname}': '{func}' expects {} args, got {}",
                    sig.params.len(),
                    args.len()
                ));
            }
            for (i, a) in args.iter().enumerate() {
                let at = synth(a, env, sigs, fname)?;
                expect(at, &sig.params[i], fname, &format!("argument {} to '{func}'", i + 1))?;
            }
            Ok(sig.ret.clone())
        }
        Expr::Alloc { region } => Ok(Type::Ptr(*region)),
        Expr::Load(p) => match synth(p, env, sigs, fname)? {
            Type::Ptr(_) => Ok(Type::I64),
            other => Err(format!(
                "in '{fname}': load expects a pointer, got {}",
                ty_str(&other)
            )),
        },
        Expr::Store { ptr, val } => {
            match synth(ptr, env, sigs, fname)? {
                Type::Ptr(_) => {}
                other => {
                    return Err(format!(
                        "in '{fname}': store! expects a pointer, got {}",
                        ty_str(&other)
                    ))
                }
            }
            expect(synth(val, env, sigs, fname)?, &Type::I64, fname, "store! value")?;
            Ok(Type::I64)
        }
        Expr::Free(p) => match synth(p, env, sigs, fname)? {
            Type::Ptr(Region::Heap) => Ok(Type::I64),
            Type::Ptr(r) => Err(format!(
                "in '{fname}': cannot free a {} pointer (only heap pointers are freed)",
                r.name()
            )),
            other => Err(format!(
                "in '{fname}': free expects a pointer, got {}",
                ty_str(&other)
            )),
        },
    }
}

fn is_frame_ptr(t: &Type) -> bool {
    matches!(t, Type::Ptr(Region::Frame))
}

fn expect(got: Type, want: &Type, fname: &str, what: &str) -> Result<(), String> {
    if &got == want {
        Ok(())
    } else {
        Err(format!(
            "in '{fname}': {what} has type {} but expected {}",
            ty_str(&got),
            ty_str(want)
        ))
    }
}

fn ty_str(t: &Type) -> String {
    match t {
        Type::I64 => "i64".to_string(),
        Type::Ptr(r) => format!("(ptr {})", r.name()),
    }
}
