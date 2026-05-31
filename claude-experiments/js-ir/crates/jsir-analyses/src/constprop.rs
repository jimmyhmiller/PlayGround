//! Constant propagation, expressed as one instantiation of the generic
//! [`crate::dataflow`] engine: the engine drives all control flow / fixpoints,
//! and this module only supplies the lattice (in `lib.rs`) plus the per-op
//! transfer functions and truthiness predicates below.

use jsir_ir::{Attr, Op};

use crate::dataflow::{self, Analysis, Transfer};
use crate::{Const, Lattice};

/// The constant-propagation analysis.
pub struct ConstProp;

/// Run constant propagation over a `jsir.file` op, returning the ordered value
/// facts (upstream's `// %N = <value>` lines, in document order).
pub fn analyze_constants(file: &Op) -> Vec<String> {
    dataflow::run(&ConstProp, file).facts
}

impl Analysis for ConstProp {
    type V = Lattice;

    fn transfer(&self, op: &Op, cx: &mut Transfer<Lattice>) {
        match op.name.as_str() {
            "jsir.numeric_literal" => cx.set_result(match cx.attr("value") {
                Some(Attr::F64(f)) => Lattice::Const(Const::num(*f)),
                _ => Lattice::Unknown,
            }),
            "jsir.string_literal" => cx.set_result(match cx.attr("value") {
                Some(Attr::Str(s)) => Lattice::Const(Const::Str(s.clone())),
                _ => Lattice::Unknown,
            }),
            "jsir.boolean_literal" => cx.set_result(match cx.attr("value") {
                Some(Attr::Bool(b)) => Lattice::Const(Const::Bool(*b)),
                _ => Lattice::Unknown,
            }),
            "jsir.null_literal" => cx.set_result(Lattice::Const(Const::Null)),
            "jsir.big_int_literal" => {
                let value = match cx.attr("value") {
                    Some(Attr::Str(s)) => s.clone(),
                    _ => String::new(),
                };
                let (raw, raw_value) = match cx.attr("extra") {
                    Some(Attr::BigIntLiteralExtra { raw, raw_value }) => (raw.clone(), raw_value.clone()),
                    _ => (String::new(), value.clone()),
                };
                cx.set_result(Lattice::Const(Const::BigInt { value, raw, raw_value }));
            }
            "jsir.reg_exp_literal" => {
                let s = |k| match cx.attr(k) {
                    Some(Attr::Str(s)) => s.clone(),
                    _ => String::new(),
                };
                cx.set_result(Lattice::Const(Const::RegExp { pattern: s("pattern"), flags: s("flags") }));
            }
            // An r-value identifier reads its variable's current state.
            "jsir.identifier" => {
                let v = cx.symbol().map(|k| cx.var(&k)).unwrap_or(Lattice::Unknown);
                cx.set_result(v);
            }
            // An l-value reference is not itself a value.
            "jsir.identifier_ref" => cx.set_result(Lattice::Unknown),
            // `lhs = rhs` declarator: bind the symbol to rhs's value.
            "jsir.variable_declarator" => {
                if cx.operand_count() == 2 {
                    let rhs = cx.operand(1);
                    if let Some(key) = cx.operand_symbol(0) {
                        cx.set_var(key, rhs);
                    }
                }
                cx.set_result(Lattice::Uninitialized);
            }
            "jsir.assignment_expression" => {
                // The assignment *expression's* result is unknown, but a plain
                // `lhs = rhs` updates the variable to rhs (compound -> unknown).
                let is_plain = matches!(cx.attr("operator_"), Some(Attr::Str(s)) if s == "=");
                let rhs = cx.operand(1);
                if let Some(key) = cx.operand_symbol(0) {
                    cx.set_var(key, if is_plain { rhs } else { Lattice::Unknown });
                }
                cx.set_result(Lattice::Unknown);
            }
            "jsir.update_expression" => {
                // `x++` / `--x`: read x, compute x±1, write back; result is the
                // new value (prefix) or old value (postfix).
                let key = cx.operand_symbol(0);
                let old = key.as_ref().map(|k| cx.var(k)).unwrap_or(Lattice::Unknown);
                let delta = if matches!(cx.attr("operator_"), Some(Attr::Str(s)) if s == "++") { 1.0 } else { -1.0 };
                let prefix = matches!(cx.attr("prefix"), Some(Attr::Bool(true)));
                let (new, result) = match old.as_num() {
                    Some(n) => {
                        let new = Lattice::Const(Const::num(n + delta));
                        let result = if prefix { new.clone() } else { Lattice::Const(Const::num(n)) };
                        (new, result)
                    }
                    None => (Lattice::Unknown, Lattice::Unknown),
                };
                if let Some(k) = key {
                    cx.set_var(k, new);
                }
                cx.set_result(result);
            }
            "jsir.binary_expression" => cx.set_result(fold_binary(cx)),
            "jsir.unary_expression" => cx.set_result(fold_unary(cx)),
            "jsir.member_expression" => {
                // Constant string indexed by a constant integer: `"abc"[1]` -> "b".
                let v = if cx.operand_count() == 2 {
                    let obj = cx.operand(0);
                    let idx = cx.operand(1);
                    match (&obj, idx.as_num()) {
                        (Lattice::Const(Const::Str(s)), Some(i)) if i >= 0.0 && i.fract() == 0.0 => {
                            match s.chars().nth(i as usize) {
                                Some(c) => Lattice::Const(Const::Str(c.to_string())),
                                None => Lattice::Unknown,
                            }
                        }
                        _ => Lattice::Unknown,
                    }
                } else {
                    Lattice::Unknown
                };
                cx.set_result(v);
            }
            "jsir.parenthesized_expression" => {
                let v = cx.operand(0);
                cx.set_result(v);
            }
            // Anything else with a result is over-defined (engine fills top).
            _ => {}
        }
    }

    fn is_true(&self, v: &Lattice) -> bool {
        matches!(v, Lattice::Const(c) if truthy(c))
    }
    fn is_false(&self, v: &Lattice) -> bool {
        matches!(v, Lattice::Const(c) if !truthy(c))
    }
    fn is_nullish(&self, v: &Lattice) -> bool {
        matches!(v, Lattice::Const(Const::Null))
    }
    fn is_nonnullish(&self, v: &Lattice) -> bool {
        matches!(v, Lattice::Const(c) if !matches!(c, Const::Null))
    }
}

fn fold_binary(cx: &Transfer<Lattice>) -> Lattice {
    if cx.operand_count() != 2 {
        return Lattice::Unknown;
    }
    let l = cx.operand(0);
    let r = cx.operand(1);
    let (Lattice::Const(lc), Lattice::Const(rc)) = (&l, &r) else {
        if matches!(l, Lattice::Uninitialized) || matches!(r, Lattice::Uninitialized) {
            return Lattice::Uninitialized;
        }
        return Lattice::Unknown;
    };
    let opname = match cx.attr("operator_") {
        Some(Attr::Str(s)) => s.as_str(),
        _ => return Lattice::Unknown,
    };
    fold_binary_const(opname, lc, rc).map(Lattice::Const).unwrap_or(Lattice::Unknown)
}

fn fold_unary(cx: &Transfer<Lattice>) -> Lattice {
    if cx.operand_count() != 1 {
        return Lattice::Unknown;
    }
    let a = cx.operand(0);
    let Lattice::Const(c) = &a else {
        if matches!(a, Lattice::Uninitialized) {
            return Lattice::Uninitialized;
        }
        return Lattice::Unknown;
    };
    let opname = match cx.attr("operator_") {
        Some(Attr::Str(s)) => s.as_str(),
        _ => return Lattice::Unknown,
    };
    fold_unary_const(opname, c).map(Lattice::Const).unwrap_or(Lattice::Unknown)
}

/// Fold a binary operation on two constants (JS semantics for the modeled cases).
fn fold_binary_const(opname: &str, l: &Const, r: &Const) -> Option<Const> {
    // Comparisons fold for non-numeric operands too (string/bool/null), so they
    // are handled before the numeric path.
    match opname {
        "===" => return strict_eq(l, r).map(Const::Bool),
        "!==" => return strict_eq(l, r).map(|b| Const::Bool(!b)),
        "==" => return loose_eq(l, r).map(Const::Bool),
        "!=" => return loose_eq(l, r).map(|b| Const::Bool(!b)),
        "<" | "<=" | ">" | ">=" => return relational(opname, l, r).map(Const::Bool),
        _ => {}
    }
    if opname == "+" {
        // String concatenation if either side is a string, else numeric add.
        if matches!(l, Const::Str(_)) || matches!(r, Const::Str(_)) {
            return Some(Const::Str(format!("{}{}", l.to_js_string()?, r.to_js_string()?)));
        }
        return Some(Const::num(l.to_number()? + r.to_number()?));
    }
    if let (Some(a), Some(b)) = (l.to_number(), r.to_number()) {
        let num = |x: f64| Some(Const::num(x));
        return match opname {
            "-" => num(a - b),
            "*" => num(a * b),
            "/" => num(a / b),
            "%" => num(a % b),
            "**" => num(a.powf(b)),
            "&" => num(((a as i64 as i32) & (b as i64 as i32)) as f64),
            "|" => num(((a as i64 as i32) | (b as i64 as i32)) as f64),
            "^" => num(((a as i64 as i32) ^ (b as i64 as i32)) as f64),
            "<<" => num(((a as i64 as i32).wrapping_shl(b as i64 as u32 & 31)) as f64),
            ">>" => num(((a as i64 as i32).wrapping_shr(b as i64 as u32 & 31)) as f64),
            ">>>" => num(((a as i64 as u32).wrapping_shr(b as i64 as u32 & 31)) as f64),
            _ => None,
        };
    }
    None
}

/// JS strict equality (`===`) over the constants we model. `None` when it can't
/// be decided soundly: any `RegExp` operand (two literals are distinct objects,
/// but the *same* value flowing to both sides is identical — we can't tell).
fn strict_eq(l: &Const, r: &Const) -> Option<bool> {
    use Const::*;
    if matches!(l, RegExp { .. }) || matches!(r, RegExp { .. }) {
        return None;
    }
    Some(match (l, r) {
        (Num(a), Num(b)) => f64::from_bits(*a) == f64::from_bits(*b), // NaN!=NaN, +0==-0
        (Str(a), Str(b)) => a == b,
        (Bool(a), Bool(b)) => a == b,
        (Null, Null) => true,
        (BigInt { value: a, .. }, BigInt { value: b, .. }) => a == b,
        _ => false, // different types are never strictly equal
    })
}

/// JS abstract equality (`==`) for the primitive constants. `None` (Unknown)
/// whenever a `BigInt`/`RegExp` is involved, since those need coercion rules we
/// don't fully model.
fn loose_eq(l: &Const, r: &Const) -> Option<bool> {
    use Const::*;
    if matches!(l, RegExp { .. } | BigInt { .. }) || matches!(r, RegExp { .. } | BigInt { .. }) {
        return None;
    }
    match (l, r) {
        (Null, Null) => Some(true),
        // null/undefined only loosely-equal each other; undefined isn't modeled.
        (Null, _) | (_, Null) => Some(false),
        // Same primitive type: `==` coincides with `===`.
        _ if std::mem::discriminant(l) == std::mem::discriminant(r) => strict_eq(l, r),
        // Mixed Number/String/Boolean: coerce both to number and compare.
        _ => match (l.to_number(), r.to_number()) {
            (Some(a), Some(b)) => Some(a == b),
            _ => Some(false), // e.g. "abc" == 1  ->  NaN == 1  ->  false
        },
    }
}

/// JS relational comparison (`<`/`<=`/`>`/`>=`). Both-string operands compare
/// lexicographically; otherwise both coerce to numbers (any `NaN` -> false).
fn relational(opname: &str, l: &Const, r: &Const) -> Option<bool> {
    if let (Const::Str(a), Const::Str(b)) = (l, r) {
        return Some(match opname {
            "<" => a < b,
            "<=" => a <= b,
            ">" => a > b,
            ">=" => a >= b,
            _ => return None,
        });
    }
    let (a, b) = (l.to_number()?, r.to_number()?);
    if a.is_nan() || b.is_nan() {
        return Some(false);
    }
    Some(match opname {
        "<" => a < b,
        "<=" => a <= b,
        ">" => a > b,
        ">=" => a >= b,
        _ => return None,
    })
}

fn fold_unary_const(opname: &str, c: &Const) -> Option<Const> {
    match opname {
        "-" => Some(Const::num(-c.as_num()?)),
        "+" => Some(Const::num(c.as_num()?)),
        "~" => Some(Const::num(!(c.as_num()? as i64 as i32) as f64)),
        "!" => Some(Const::Bool(!truthy(c))),
        _ => None,
    }
}

fn truthy(c: &Const) -> bool {
    match c {
        Const::Num(_) => c.as_num().map(|n| n != 0.0 && !n.is_nan()).unwrap_or(false),
        Const::Str(s) => !s.is_empty(),
        Const::Bool(b) => *b,
        Const::Null => false,
        Const::BigInt { value, .. } => value != "0",
        Const::RegExp { .. } => true,
    }
}
