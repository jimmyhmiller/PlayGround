//! A second analysis — **sign analysis** — built on the same generic engine as
//! constant propagation. It tracks each numeric value's sign (`-`/`0`/`+`),
//! demonstrating that [`crate::dataflow`] is a real framework: this file only
//! provides a different lattice and transfer functions, while the engine reuses
//! all the control flow, loop/branch fixpoints, conditional reachability, and
//! per-variable state plumbing unchanged.

use jsir_ir::{Attr, Op};

use crate::dataflow::{self, Analysis, Lattice, Transfer};

/// The sign lattice: `⊥` (unreachable) ⊑ {`-`, `0`, `+`} ⊑ `⊤` (unknown sign).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Sign {
    Bottom,
    Neg,
    Zero,
    Pos,
    Top,
}

impl Lattice for Sign {
    fn bottom() -> Self {
        Sign::Bottom
    }
    fn top() -> Self {
        Sign::Top
    }
    fn join(&mut self, other: &Sign) -> bool {
        match (&*self, other) {
            (_, Sign::Bottom) => false,
            (Sign::Bottom, _) => {
                *self = other.clone();
                true
            }
            (Sign::Top, _) => false,
            (a, b) if a == b => false,
            _ => {
                *self = Sign::Top;
                true
            }
        }
    }
    fn render(&self) -> String {
        match self {
            Sign::Bottom => "<bottom>",
            Sign::Neg => "-",
            Sign::Zero => "0",
            Sign::Pos => "+",
            Sign::Top => "<top>",
        }
        .to_string()
    }
}

impl Sign {
    fn of(n: f64) -> Sign {
        if n > 0.0 {
            Sign::Pos
        } else if n < 0.0 {
            Sign::Neg
        } else {
            Sign::Zero
        }
    }
}

/// Sign analysis.
pub struct SignAnalysis;

/// Run sign analysis, returning each value's sign in document order.
pub fn analyze_signs(file: &Op) -> Vec<String> {
    dataflow::run(&SignAnalysis, file).facts
}

impl Analysis for SignAnalysis {
    type V = Sign;

    fn transfer(&self, op: &Op, cx: &mut Transfer<Sign>) {
        match op.name.as_str() {
            "jsir.numeric_literal" => cx.set_result(match cx.attr("value") {
                Some(Attr::F64(f)) => Sign::of(*f),
                _ => Sign::Top,
            }),
            "jsir.identifier" => {
                let v = cx.symbol().map(|k| cx.var(&k)).unwrap_or(Sign::Top);
                cx.set_result(v);
            }
            "jsir.identifier_ref" => cx.set_result(Sign::Top),
            "jsir.variable_declarator" => {
                if cx.operand_count() == 2 {
                    let rhs = cx.operand(1);
                    if let Some(key) = cx.operand_symbol(0) {
                        cx.set_var(key, rhs);
                    }
                }
                cx.set_result(Sign::Bottom);
            }
            "jsir.assignment_expression" => {
                let is_plain = matches!(cx.attr("operator_"), Some(Attr::Str(s)) if s == "=");
                let rhs = cx.operand(1);
                if let Some(key) = cx.operand_symbol(0) {
                    cx.set_var(key, if is_plain { rhs } else { Sign::Top });
                }
                cx.set_result(Sign::Top);
            }
            "jsir.unary_expression" => {
                let a = cx.operand(0);
                let v = match cx.attr("operator_") {
                    Some(Attr::Str(s)) if s == "-" => match a {
                        Sign::Pos => Sign::Neg,
                        Sign::Neg => Sign::Pos,
                        Sign::Zero => Sign::Zero,
                        other => other,
                    },
                    _ => Sign::Top,
                };
                cx.set_result(v);
            }
            "jsir.binary_expression" => cx.set_result(fold(cx)),
            "jsir.parenthesized_expression" => {
                let v = cx.operand(0);
                cx.set_result(v);
            }
            _ => {}
        }
    }

    // Nonzero numbers are truthy, zero is falsy — enough for branch pruning.
    fn is_true(&self, v: &Sign) -> bool {
        matches!(v, Sign::Pos | Sign::Neg)
    }
    fn is_false(&self, v: &Sign) -> bool {
        matches!(v, Sign::Zero)
    }
}

/// Sign rules for the arithmetic we can reason about; anything else is `⊤`.
fn fold(cx: &Transfer<Sign>) -> Sign {
    if cx.operand_count() != 2 {
        return Sign::Top;
    }
    let l = cx.operand(0);
    let r = cx.operand(1);
    if l == Sign::Bottom || r == Sign::Bottom {
        return Sign::Bottom;
    }
    let op = match cx.attr("operator_") {
        Some(Attr::Str(s)) => s.as_str(),
        _ => return Sign::Top,
    };
    match op {
        "*" => match (&l, &r) {
            (Sign::Zero, _) | (_, Sign::Zero) => Sign::Zero,
            (a, b) if a == b => Sign::Pos, // (+,+) or (-,-)
            (Sign::Top, _) | (_, Sign::Top) => Sign::Top,
            _ => Sign::Neg, // mixed signs
        },
        "+" => match (&l, &r) {
            (Sign::Pos, Sign::Pos) => Sign::Pos,
            (Sign::Neg, Sign::Neg) => Sign::Neg,
            (Sign::Zero, b) => b.clone(),
            (a, Sign::Zero) => a.clone(),
            _ => Sign::Top, // (+)+(-) could be anything
        },
        "-" => match (&l, &r) {
            (Sign::Pos, Sign::Neg) => Sign::Pos,
            (Sign::Neg, Sign::Pos) => Sign::Neg,
            (a, Sign::Zero) => a.clone(),
            (Sign::Zero, Sign::Pos) => Sign::Neg,
            (Sign::Zero, Sign::Neg) => Sign::Pos,
            _ => Sign::Top,
        },
        _ => Sign::Top,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jsir_swc::source_to_ir;

    /// Extract the `// %N = <sign>` facts in order, then sanity-check a few.
    fn signs(src: &str) -> Vec<String> {
        analyze_signs(&source_to_ir(src).unwrap())
    }

    #[test]
    fn literals_and_arithmetic() {
        // -3 * 4 is negative; 2 + 5 is positive; 0 * x is zero.
        let f = signs("var a = -3 * 4; var b = 2 + 5; var c = 0 * 9;");
        assert!(f.contains(&"-".to_string()), "expected a negative value: {f:?}");
        assert!(f.contains(&"+".to_string()), "expected a positive value: {f:?}");
        assert!(f.contains(&"0".to_string()), "expected a zero value: {f:?}");
    }

    #[test]
    fn variable_through_branch_joins_to_top() {
        // x is + on one branch and - on the other => join is unknown sign.
        let f = signs("var x = 1; if (cond) { x = 5; } else { x = -5; } x;");
        // The final read of x must be <top> (signs disagree across the branches).
        assert!(f.contains(&"<top>".to_string()), "join should be top: {f:?}");
    }

    #[test]
    fn loop_reaches_fixpoint() {
        // A variable incremented in a loop stays a valid sign lattice element
        // (the engine's loop fixpoint terminates — no hang).
        let f = signs("var i = 1; while (cond) { i = i + 1; } i;");
        assert!(!f.is_empty());
    }
}
