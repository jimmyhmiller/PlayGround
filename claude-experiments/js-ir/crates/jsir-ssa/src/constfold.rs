//! Constant folding over the SSA CFG (React's `ConstantPropagation`).
//!
//! React folds compile-time-constant arithmetic before inferring reactive
//! scopes, so a value like `2 * -1` or `~3` becomes a literal and stops being a
//! dependency (often collapsing a whole scope). We do the same on the analysis
//! CFG: fold `Bin`/`Un` whose operands are constants into `Const`, to a
//! fixpoint. This only rewrites the analysis view — the reversible JSIR (and
//! therefore emitted code) is untouched, so we match React's memo *structure*
//! (`_c(N)`) without needing to re-fold the source text.

use std::collections::HashMap;

use crate::cfg::{BinOp, Cfg, Const, Op, UnOp, Value};

/// Fold constant `Bin`/`Un` instructions in place, to a fixpoint.
pub fn fold_constants(cfg: &mut Cfg) {
    let mut consts: HashMap<Value, Const> = HashMap::new();
    // Seed with literal constants already present.
    for b in &cfg.blocks {
        for ins in &b.instrs {
            if let (Some(r), Op::Const(c)) = (ins.result, &ins.op) {
                consts.insert(r, c.clone());
            }
        }
    }
    let mut changed = true;
    while changed {
        changed = false;
        for b in &mut cfg.blocks {
            for ins in &mut b.instrs {
                let r = match ins.result {
                    Some(r) if !consts.contains_key(&r) => r,
                    _ => continue,
                };
                let folded = match &ins.op {
                    Op::Bin(op, a, b) => match (consts.get(a), consts.get(b)) {
                        (Some(x), Some(y)) => eval_bin(*op, x, y),
                        _ => None,
                    },
                    Op::Un(op, a) => consts.get(a).and_then(|x| eval_un(*op, x)),
                    _ => None,
                };
                if let Some(c) = folded {
                    ins.op = Op::Const(c.clone());
                    consts.insert(r, c);
                    changed = true;
                }
            }
        }
    }
}

fn as_num(c: &Const) -> Option<f64> {
    match c {
        Const::Num(bits) => Some(f64::from_bits(*bits)),
        _ => None,
    }
}

/// JS `ToInt32`.
fn to_int32(f: f64) -> i32 {
    if !f.is_finite() {
        return 0;
    }
    let m = f.trunc().rem_euclid(4_294_967_296.0); // mod 2^32, always >= 0
    m as u32 as i32
}

/// JS `ToUint32`.
fn to_uint32(f: f64) -> u32 {
    if !f.is_finite() {
        return 0;
    }
    f.trunc().rem_euclid(4_294_967_296.0) as u32
}

fn eval_bin(op: BinOp, x: &Const, y: &Const) -> Option<Const> {
    let (a, b) = (as_num(x)?, as_num(y)?);
    Some(match op {
        BinOp::Add => Const::num(a + b),
        BinOp::Sub => Const::num(a - b),
        BinOp::Mul => Const::num(a * b),
        BinOp::Div => Const::num(a / b),
        BinOp::Mod => Const::num(a % b),
        BinOp::Pow => Const::num(a.powf(b)),
        BinOp::BitAnd => Const::num((to_int32(a) & to_int32(b)) as f64),
        BinOp::BitOr => Const::num((to_int32(a) | to_int32(b)) as f64),
        BinOp::BitXor => Const::num((to_int32(a) ^ to_int32(b)) as f64),
        BinOp::Shl => Const::num((to_int32(a).wrapping_shl(to_uint32(b) & 31)) as f64),
        BinOp::Shr => Const::num((to_int32(a) >> (to_uint32(b) & 31)) as f64),
        BinOp::UShr => Const::num((to_uint32(a) >> (to_uint32(b) & 31)) as f64),
        // Comparisons fold to a boolean (numeric operands only).
        BinOp::Eq | BinOp::StrictEq => Const::Bool(a == b),
        BinOp::Ne | BinOp::StrictNe => Const::Bool(a != b),
        BinOp::Lt => Const::Bool(a < b),
        BinOp::Le => Const::Bool(a <= b),
        BinOp::Gt => Const::Bool(a > b),
        BinOp::Ge => Const::Bool(a >= b),
    })
}

fn eval_un(op: UnOp, x: &Const) -> Option<Const> {
    match op {
        UnOp::Neg => Some(Const::num(-as_num(x)?)),
        UnOp::Pos => Some(Const::num(as_num(x)?)),
        UnOp::BitNot => Some(Const::num(!to_int32(as_num(x)?) as f64)),
        UnOp::Not => match x {
            Const::Bool(b) => Some(Const::Bool(!b)),
            Const::Num(_) => Some(Const::Bool(as_num(x)? == 0.0 || as_num(x)?.is_nan())),
            Const::Null | Const::Undef => Some(Const::Bool(true)),
            _ => None,
        },
        // `void x` is always `undefined`; `typeof` of a literal is foldable but
        // rarely worth it — leave both alone for now.
        UnOp::Void => Some(Const::Undef),
        UnOp::TypeOf => None,
    }
}
