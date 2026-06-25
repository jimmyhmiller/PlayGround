//! Compile-time evaluation. `(comptime E)` runs `E` in the *real* language during
//! compilation and splices the resulting literal — the same `defn`s, the same
//! `=`/arithmetic, the same `match`, executed by interpretation instead of being
//! lowered to machine code. This is the bridge toward "the whole language at
//! compile time": runtime code becomes available at compile time by running it.
//!
//! Stage 1 is the pure scalar subset: ints/bools/floats, arithmetic + comparison,
//! `if`/`let`/`do`/`match`, sum construction, and calls to (monomorphic) `defn`s
//! including recursion. Memory/aggregates/mutation/loops/generics/FFI are not
//! supported *yet* and raise a clear error rather than silently miscompiling
//! (aggregate comptime values + a comptime heap are Stage 1b). The final value of
//! a `comptime` form must be a scalar.

use crate::ast::*;
use crate::span::Span;
use std::collections::{HashMap, HashSet};

#[derive(Clone, Debug)]
enum CtVal {
    Int(i64),
    Bool(bool),
    Float(f64),
    /// A sum value — usable internally (built by construction, taken apart by
    /// `match`); a `comptime` form can't *return* one yet (scalar results only).
    Sum { variant: String, fields: Vec<CtVal> },
}

struct Ctx<'a> {
    fns: &'a HashMap<String, Func>,
    variants: &'a HashSet<String>,
    fuel: std::cell::Cell<u64>,
}

/// A generous evaluation budget — a runaway comptime loop errors instead of
/// hanging the compiler.
const FUEL: u64 = 50_000_000;

/// Replace every `(comptime …)` in the checked program with its evaluated literal.
/// Runs after checking, so all `defn` bodies are available to call.
pub fn fold_program(
    funcs: &mut Vec<Func>,
    sums: &[SumDef],
) -> Result<(), String> {
    let table: HashMap<String, Func> = funcs.iter().map(|f| (f.name.clone(), f.clone())).collect();
    let variants: HashSet<String> =
        sums.iter().flat_map(|s| s.variants.iter().map(|v| v.name.clone())).collect();
    let ctx = Ctx { fns: &table, variants: &variants, fuel: std::cell::Cell::new(FUEL) };
    for f in funcs.iter_mut() {
        let body = std::mem::take(&mut f.body);
        f.body = body.iter().map(|e| fold_expr(e, &ctx)).collect::<Result<_, _>>()?;
    }
    Ok(())
}

/// Walk an expression, replacing any `(comptime inner)` with the literal `inner`
/// evaluates to. Everything else is rebuilt structurally.
fn fold_expr(e: &Expr, ctx: &Ctx) -> Result<Expr, String> {
    let go = |e: &Expr| fold_expr(e, ctx);
    let gb = |e: &Expr| -> Result<Box<Expr>, String> { Ok(Box::new(fold_expr(e, ctx)?)) };
    let kind = match &e.kind {
        ExprKind::Comptime(inner) => {
            // Fold any nested comptime first, then evaluate.
            let inner = fold_expr(inner, ctx)?;
            let v = eval(&inner, &mut HashMap::new(), ctx)?;
            return Ok(Expr::new(literal(v, e.span)?, e.span));
        }
        // leaves
        ExprKind::Int(_)
        | ExprKind::Float(_)
        | ExprKind::Bool(_)
        | ExprKind::Str(_)
        | ExprKind::CStr(_)
        | ExprKind::Var(_)
        | ExprKind::Zeroed(_)
        | ExprKind::SizeOf(_)
        | ExprKind::AlignOf(_)
        | ExprKind::OffsetOf(_, _)
        | ExprKind::FnPtrOf(_)
        | ExprKind::Continue { .. } => e.kind.clone(),
        // recursive cases
        ExprKind::Borrow { mutable, place } => ExprKind::Borrow { mutable: *mutable, place: gb(place)? },
        ExprKind::SpillRef(x) => ExprKind::SpillRef(gb(x)?),
        ExprKind::Not(x) => ExprKind::Not(gb(x)?),
        ExprKind::Load(x) => ExprKind::Load(gb(x)?),
        ExprKind::Free(x) => ExprKind::Free(gb(x)?),
        ExprKind::Cast { ty, expr } => ExprKind::Cast { ty: ty.clone(), expr: gb(expr)? },
        ExprKind::Bin { op, lhs, rhs } => ExprKind::Bin { op: *op, lhs: gb(lhs)?, rhs: gb(rhs)? },
        ExprKind::Cmp { op, lhs, rhs } => ExprKind::Cmp { op: *op, lhs: gb(lhs)?, rhs: gb(rhs)? },
        ExprKind::If { cond, then, els } => ExprKind::If { cond: gb(cond)?, then: gb(then)?, els: gb(els)? },
        ExprKind::Do(es) => ExprKind::Do(es.iter().map(go).collect::<Result<_, _>>()?),
        ExprKind::Loop { label, body } => ExprKind::Loop {
            label: label.clone(),
            body: body.iter().map(go).collect::<Result<_, _>>()?,
        },
        ExprKind::Break { label, value } => ExprKind::Break {
            label: label.clone(),
            value: match value { Some(v) => Some(gb(v)?), None => None },
        },
        ExprKind::Let { binds, body } => ExprKind::Let {
            binds: binds.iter().map(|(n, m, e)| Ok((n.clone(), *m, fold_expr(e, ctx)?))).collect::<Result<_, String>>()?,
            body: body.iter().map(go).collect::<Result<_, _>>()?,
        },
        ExprKind::Call { func, type_args, args } => ExprKind::Call {
            func: func.clone(),
            type_args: type_args.clone(),
            args: args.iter().map(go).collect::<Result<_, _>>()?,
        },
        ExprKind::Alloc { storage, ty } => ExprKind::Alloc { storage: *storage, ty: ty.clone() },
        ExprKind::Field { ptr, field } => ExprKind::Field { ptr: gb(ptr)?, field: field.clone() },
        ExprKind::Store { ptr, val } => ExprKind::Store { ptr: gb(ptr)?, val: gb(val)? },
        ExprKind::Index { ptr, idx } => ExprKind::Index { ptr: gb(ptr)?, idx: gb(idx)? },
        ExprKind::BitGet { ptr, field } => ExprKind::BitGet { ptr: gb(ptr)?, field: field.clone() },
        ExprKind::BitSet { ptr, field, val } => ExprKind::BitSet { ptr: gb(ptr)?, field: field.clone(), val: gb(val)? },
        ExprKind::Construct { sum, variant, args } => ExprKind::Construct {
            sum: sum.clone(),
            variant: variant.clone(),
            args: args.iter().map(go).collect::<Result<_, _>>()?,
        },
        ExprKind::Match { scrut, arms } => ExprKind::Match {
            scrut: gb(scrut)?,
            arms: arms.iter().map(|a| Ok(Arm { variant: a.variant.clone(), binds: a.binds.clone(), body: fold_expr(&a.body, ctx)? })).collect::<Result<_, String>>()?,
        },
        ExprKind::CallPtr { fp, args } => ExprKind::CallPtr {
            fp: gb(fp)?,
            args: args.iter().map(go).collect::<Result<_, _>>()?,
        },
        ExprKind::LlvmIr { result, args, body } => ExprKind::LlvmIr {
            result: result.clone(),
            args: args.iter().map(go).collect::<Result<_, _>>()?,
            body: body.clone(),
        },
        ExprKind::TraitCall { .. } => e.kind.clone(),
    };
    Ok(Expr::new(kind, e.span))
}

fn literal(v: CtVal, _span: Span) -> Result<ExprKind, String> {
    match v {
        CtVal::Int(n) => Ok(ExprKind::Int(n)),
        CtVal::Bool(b) => Ok(ExprKind::Bool(b)),
        CtVal::Float(x) => Ok(ExprKind::Float(x)),
        CtVal::Sum { .. } => Err(
            "comptime: a (comptime …) form must produce a scalar (int/bool/float); \
             aggregate comptime values aren't supported yet"
                .to_string(),
        ),
    }
}

type Env = HashMap<String, CtVal>;

fn eval(e: &Expr, env: &mut Env, ctx: &Ctx) -> Result<CtVal, String> {
    let f = ctx.fuel.get();
    if f == 0 {
        return Err("comptime: evaluation budget exhausted (infinite loop/recursion?)".to_string());
    }
    ctx.fuel.set(f - 1);
    match &e.kind {
        ExprKind::Int(n) => Ok(CtVal::Int(*n)),
        ExprKind::Bool(b) => Ok(CtVal::Bool(*b)),
        ExprKind::Float(x) => Ok(CtVal::Float(*x)),
        ExprKind::Comptime(inner) => eval(inner, env, ctx),
        ExprKind::Var(name) => env
            .get(name)
            .cloned()
            .ok_or_else(|| format!("comptime: unbound variable '{name}'")),
        ExprKind::Not(x) => match eval(x, env, ctx)? {
            CtVal::Int(n) => Ok(CtVal::Int(!n)),
            _ => Err("comptime: `inot` needs an integer".to_string()),
        },
        ExprKind::Bin { op, lhs, rhs } => {
            let a = eval(lhs, env, ctx)?;
            let b = eval(rhs, env, ctx)?;
            bin(*op, a, b)
        }
        ExprKind::Cmp { op, lhs, rhs } => {
            let a = eval(lhs, env, ctx)?;
            let b = eval(rhs, env, ctx)?;
            cmp(*op, a, b)
        }
        ExprKind::If { cond, then, els } => {
            if truthy(&eval(cond, env, ctx)?)? {
                eval(then, env, ctx)
            } else {
                eval(els, env, ctx)
            }
        }
        ExprKind::Do(es) => eval_seq(es, env, ctx),
        ExprKind::Let { binds, body } => {
            let mut scope = env.clone();
            for (n, mutable, ve) in binds {
                if *mutable {
                    return Err("comptime: mutable `let` bindings aren't supported yet".to_string());
                }
                let v = eval(ve, &mut scope, ctx)?;
                scope.insert(n.clone(), v);
            }
            eval_seq(body, &mut scope, ctx)
        }
        ExprKind::Cast { ty, expr } => {
            let v = eval(expr, env, ctx)?;
            match (v, ty) {
                (CtVal::Int(n), Type::Int(bits, signed)) => Ok(CtVal::Int(trunc(n, *bits, *signed))),
                (CtVal::Int(n), _) => Ok(CtVal::Int(n)), // ptr reinterpret etc. — keep bits
                (other, _) => Ok(other),
            }
        }
        ExprKind::Construct { variant, args, .. } => {
            let fields = args.iter().map(|a| eval(a, env, ctx)).collect::<Result<_, _>>()?;
            Ok(CtVal::Sum { variant: variant.clone(), fields })
        }
        ExprKind::Match { scrut, arms } => {
            let v = eval(scrut, env, ctx)?;
            let (variant, fields) = match v {
                CtVal::Sum { variant, fields } => (variant, fields),
                _ => return Err("comptime: `match` scrutinee must be a sum value".to_string()),
            };
            let arm = arms
                .iter()
                .find(|a| a.variant == variant)
                .ok_or_else(|| format!("comptime: no match arm for variant '{variant}'"))?;
            let mut scope = env.clone();
            for (binder, fv) in arm.binds.iter().zip(fields.into_iter()) {
                scope.insert(binder.clone(), fv);
            }
            eval(&arm.body, &mut scope, ctx)
        }
        ExprKind::Call { func, type_args, args } => {
            let argv: Vec<CtVal> = args.iter().map(|a| eval(a, env, ctx)).collect::<Result<_, _>>()?;
            // Variant construction is a call to the variant name (pre-mono).
            if ctx.variants.contains(func) {
                return Ok(CtVal::Sum { variant: func.clone(), fields: argv });
            }
            if !type_args.is_empty() {
                return Err(format!("comptime: generic call to '{func}' isn't supported yet"));
            }
            let callee = ctx
                .fns
                .get(func)
                .ok_or_else(|| format!("comptime: call to '{func}' (not a comptime-evaluable function — extern/FFI?)"))?;
            let mut scope: Env = HashMap::new();
            if callee.params.len() != argv.len() {
                return Err(format!("comptime: '{func}' arity mismatch"));
            }
            for (p, v) in callee.params.iter().zip(argv) {
                scope.insert(p.name.clone(), v);
            }
            eval_seq(&callee.body, &mut scope, ctx)
        }
        other => Err(format!(
            "comptime: {} isn't supported yet (Stage 1 is the pure scalar subset: \
             arithmetic, comparison, if/let/do/match, sum construction, and calls to defns)",
            kind_name(other)
        )),
    }
}

fn eval_seq(es: &[Expr], env: &mut Env, ctx: &Ctx) -> Result<CtVal, String> {
    let mut last = CtVal::Int(0);
    for e in es {
        last = eval(e, env, ctx)?;
    }
    Ok(last)
}

fn truthy(v: &CtVal) -> Result<bool, String> {
    match v {
        CtVal::Bool(b) => Ok(*b),
        CtVal::Int(n) => Ok(*n != 0),
        _ => Err("comptime: condition must be a bool or integer".to_string()),
    }
}

fn bin(op: BinOp, a: CtVal, b: CtVal) -> Result<CtVal, String> {
    if let (CtVal::Float(x), CtVal::Float(y)) = (&a, &b) {
        return match op {
            BinOp::Add => Ok(CtVal::Float(x + y)),
            BinOp::Sub => Ok(CtVal::Float(x - y)),
            BinOp::Mul => Ok(CtVal::Float(x * y)),
            BinOp::Div => Ok(CtVal::Float(x / y)),
            _ => Err("comptime: unsupported float operation".to_string()),
        };
    }
    let (x, y) = match (a, b) {
        (CtVal::Int(x), CtVal::Int(y)) => (x, y),
        _ => return Err("comptime: arithmetic needs two integers (or two floats)".to_string()),
    };
    let r = match op {
        BinOp::Add => x.wrapping_add(y),
        BinOp::Sub => x.wrapping_sub(y),
        BinOp::Mul => x.wrapping_mul(y),
        BinOp::Div => {
            if y == 0 { return Err("comptime: division by zero".to_string()); }
            x.wrapping_div(y)
        }
        BinOp::Rem => {
            if y == 0 { return Err("comptime: remainder by zero".to_string()); }
            x.wrapping_rem(y)
        }
        BinOp::UDiv => {
            if y == 0 { return Err("comptime: division by zero".to_string()); }
            ((x as u64) / (y as u64)) as i64
        }
        BinOp::URem => {
            if y == 0 { return Err("comptime: remainder by zero".to_string()); }
            ((x as u64) % (y as u64)) as i64
        }
        BinOp::And => x & y,
        BinOp::Or => x | y,
        BinOp::Xor => x ^ y,
        BinOp::Shl => x.wrapping_shl(y as u32),
        BinOp::Shr => x.wrapping_shr(y as u32),
    };
    Ok(CtVal::Int(r))
}

fn cmp(op: CmpOp, a: CtVal, b: CtVal) -> Result<CtVal, String> {
    let ord = match (&a, &b) {
        (CtVal::Int(x), CtVal::Int(y)) => x.cmp(y),
        (CtVal::Float(x), CtVal::Float(y)) => x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Less),
        (CtVal::Bool(x), CtVal::Bool(y)) => x.cmp(y),
        _ => return Err("comptime: comparison needs two values of the same scalar kind".to_string()),
    };
    use std::cmp::Ordering::*;
    let r = match op {
        CmpOp::Lt => ord == Less,
        CmpOp::Le => ord != Greater,
        CmpOp::Gt => ord == Greater,
        CmpOp::Ge => ord != Less,
        CmpOp::Eq => ord == Equal,
        CmpOp::Ne => ord != Equal,
    };
    Ok(CtVal::Bool(r))
}

/// Truncate / sign-extend an integer to `bits` width.
fn trunc(n: i64, bits: u32, signed: bool) -> i64 {
    if bits >= 64 {
        return n;
    }
    let mask = (1i128 << bits) - 1;
    let masked = (n as i128) & mask;
    if signed && (masked & (1i128 << (bits - 1))) != 0 {
        (masked - (1i128 << bits)) as i64
    } else {
        masked as i64
    }
}

fn kind_name(k: &ExprKind) -> &'static str {
    match k {
        ExprKind::Load(_) => "load (memory access)",
        ExprKind::Store { .. } => "store! (mutation)",
        ExprKind::Field { .. } => "field (aggregate access)",
        ExprKind::Index { .. } => "index (aggregate access)",
        ExprKind::Alloc { .. } => "alloc",
        ExprKind::Zeroed(_) => "zeroed (aggregate)",
        ExprKind::Borrow { .. } => "a reference/borrow",
        ExprKind::SpillRef(_) => "a spilled reference",
        ExprKind::Loop { .. } => "loop (use recursion instead, for now)",
        ExprKind::Break { .. } | ExprKind::Continue { .. } => "break/continue",
        ExprKind::Str(_) | ExprKind::CStr(_) => "a string",
        ExprKind::SizeOf(_) | ExprKind::AlignOf(_) | ExprKind::OffsetOf(_, _) => "sizeof/alignof/offsetof",
        ExprKind::FnPtrOf(_) | ExprKind::CallPtr { .. } => "a function pointer",
        ExprKind::Free(_) => "free",
        ExprKind::LlvmIr { .. } => "raw llvm-ir",
        ExprKind::BitGet { .. } | ExprKind::BitSet { .. } => "a bitfield op",
        ExprKind::TraitCall { .. } => "an unresolved trait call",
        _ => "this expression",
    }
}
