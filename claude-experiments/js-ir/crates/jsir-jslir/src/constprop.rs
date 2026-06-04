//! `ConstantPropagation` — the SCCP-style constant analysis that runs right after
//! `eliminate_redundant_phi` in the React Compiler pipeline (upstream
//! `Optimization/ConstantPropagation.ts` / the Rust port's
//! `constant_propagation.rs`).
//!
//! Upstream this is a *transformation*: it folds instructions whose operands are
//! known constants into `Primitive` instructions and prunes branches with constant
//! tests. This increment ports the **abstract interpretation core** as a pure
//! side-table analysis over [`SsaInfo`] + the JSLIR CFG (like [`crate::ssa`]): it
//! computes, for every `ValueId`, whether it holds a known compile-time constant.
//! The IR-mutating folder and branch-pruner (which need a JSLIR rewrite path and
//! CFG re-minification) are the next increment; keeping this as analysis means the
//! oracle round-trip is untouched and the result is never *silently* wrong — a
//! value the analysis can't prove constant is simply absent from the map.
//!
//! The JS-semantics evaluators (`evaluate_binary_op`, `is_truthy`,
//! `js_strict_equal`, `js_abstract_equal`, `js_to_number`, `js_to_int32`/`uint32`)
//! are ported byte-for-byte from upstream so folded values match exactly.
//!
//! Scope of this version (everything outside it is conservatively "not constant"):
//! - Primitive constants only (`Number`/`String`/`Boolean`/`Null`/`Undefined`).
//!   `LoadGlobal` constants (e.g. `Infinity`, `undefined`) are not yet tracked.
//! - Folds: literals, `identifier` reads (via SSA `reaching`), `binary_expression`
//!   on two primitives, `unary_expression` `!`/`-` on a primitive, and phis whose
//!   operands are all the same constant (`evaluate_phi`).

use std::collections::HashMap;

use jsir_ir::{Attr, Op, Region, ValueId};

use crate::dialect;
use crate::ssa::SsaInfo;

/// A known primitive constant value. Mirrors upstream `PrimitiveValue`.
#[derive(Debug, Clone)]
pub enum Constant {
    Number(f64),
    Str(String),
    Bool(bool),
    Null,
    Undefined,
}

impl Constant {
    /// Structural equality with JS strict-equality number semantics (NaN ≠ NaN).
    /// Used to decide whether a phi's operands agree (and in tests).
    pub fn js_eq(&self, other: &Constant) -> bool {
        js_strict_equal(self, other)
    }
}

/// Compute the constant-value lattice for a function-body CFG: every `ValueId`
/// the analysis can prove holds a compile-time constant maps to that [`Constant`].
/// Absence means "not provably constant" (the conservative top/⊤). Pure: no IR
/// mutation.
///
/// Runs a fixpoint over the SSA value graph (upstream's
/// `apply_constant_propagation` loop, minus the IR rewrite): values only ever move
/// from absent → constant, and a phi resolves only once *all* its operands are
/// known and equal, so the iteration is monotonic and terminates.
pub fn constant_lattice(region: &Region, info: &SsaInfo) -> HashMap<ValueId, Constant> {
    // Every value-defining op, in block order (order is irrelevant to a fixpoint).
    let mut ops: Vec<&Op> = Vec::new();
    for b in &region.blocks {
        for op in &b.ops {
            ops.push(op);
        }
    }

    let mut lattice: HashMap<ValueId, Constant> = HashMap::new();
    loop {
        let mut changed = false;

        for op in &ops {
            let Some(&result) = op.results.first() else {
                continue;
            };
            let Some(c) = eval_op(op, info, &lattice) else {
                continue;
            };
            if insert_if_new(&mut lattice, result, c) {
                changed = true;
            }
        }

        for phi in &info.phis {
            if let Some(c) = evaluate_phi(&phi.operands, &lattice) {
                if insert_if_new(&mut lattice, phi.value, c) {
                    changed = true;
                }
            }
        }

        if !changed {
            break;
        }
    }
    lattice
}

/// Record `value → c` if absent (the lattice is monotonic — once a value is known
/// constant it never changes), reporting whether this added a new entry.
fn insert_if_new(lattice: &mut HashMap<ValueId, Constant>, value: ValueId, c: Constant) -> bool {
    if lattice.contains_key(&value) {
        return false;
    }
    lattice.insert(value, c);
    true
}

/// Apply a computed [`constant_lattice`] to the CFG: replace every *computed* op
/// (`binary_expression` / `unary_expression`) whose result is a proven constant
/// with the corresponding literal op, in place — exactly as upstream sets
/// `instruction.value = Primitive { .. }`. Returns the number of ops folded.
///
/// Deliberately narrow (matches upstream's foldable set within this version's
/// scope): literals are already literals, and variable reads (`jsir.identifier`)
/// are *not* folded — upstream leaves `LoadLocal`/`LoadGlobal` in place and lets
/// later DCE remove the now-dead reads. Constant-test branch pruning (folding a
/// `cond_br` whose test is constant + re-minifying the CFG) is a separate step and
/// is not done here.
///
/// The replacement preserves the op's `ValueId` result, `node_id`, and `trivia`
/// (the def-site source location), so def-use edges, value numbering, and the lift
/// are all unaffected beyond the literal substitution.
pub fn fold_constants(region: &mut Region, lattice: &HashMap<ValueId, Constant>) -> usize {
    let mut folded = 0;
    for block in &mut region.blocks {
        for op in &mut block.ops {
            if dialect::is_terminator(&op.name) {
                continue;
            }
            if !matches!(op.name.as_str(), "jsir.binary_expression" | "jsir.unary_expression") {
                continue;
            }
            let Some(&result) = op.results.first() else { continue };
            let Some(c) = lattice.get(&result) else { continue };
            let Some(lit) = literal_op(c) else { continue };
            // Splice the literal in, keeping this op's identity (result/node/trivia).
            op.name = lit.name;
            op.operands.clear();
            op.attrs = lit.attrs;
            op.regions.clear();
            op.successors.clear();
            folded += 1;
        }
    }
    folded
}

/// Prune `if` statements with a compile-time-constant test (upstream
/// ConstantPropagation's branch-pruning step, restricted to `if` diamonds): when a
/// `cond_br_if`'s test folds to a known truthy/falsey constant, replace it with an
/// unconditional `br` to the taken branch, then drop the blocks that became
/// unreachable. The lift follows the plain `br` straight through, so the `if`
/// dissolves into its taken branch — exactly what upstream emits.
///
/// Returns the number of branches pruned. Ternary/logical `cond_br`s (which carry
/// a merge block-argument) and loop headers are left alone here.
pub fn prune_constant_if_branches(
    region: &mut Region,
    lattice: &HashMap<ValueId, Constant>,
) -> usize {
    let mut pruned = 0;
    for block in &mut region.blocks {
        let Some(term) = block.ops.last_mut() else { continue };
        if !dialect::is_if_header(term) {
            continue;
        }
        let Some(&test) = term.operands.first() else { continue };
        let Some(c) = lattice.get(&test) else { continue };
        let truthy = is_truthy(c);
        // Successors are `[then, else]`; keep the taken one.
        let taken = term.successors.get(usize::from(!truthy)).map(|s| s.block);
        let Some(taken) = taken else { continue };
        *term = dialect::br(taken);
        pruned += 1;
    }
    if pruned > 0 {
        remove_unreachable_blocks(region);
    }
    pruned
}

/// Drop blocks not reachable from the entry by following terminator successors.
/// (Loop latch/preheader/merge blocks are all reachable via real edges, so plain
/// successor reachability is sufficient and safe.)
fn remove_unreachable_blocks(region: &mut Region) {
    let Some(entry) = region.blocks.first().map(|b| b.id) else { return };
    let by_id: HashMap<jsir_ir::BlockId, &jsir_ir::Block> =
        region.blocks.iter().map(|b| (b.id, b)).collect();
    let mut reachable = std::collections::HashSet::new();
    let mut work = vec![entry];
    while let Some(id) = work.pop() {
        if !reachable.insert(id) {
            continue;
        }
        if let Some(block) = by_id.get(&id) {
            if let Some(term) = block.ops.last() {
                for s in &term.successors {
                    work.push(s.block);
                }
            }
        }
    }
    region.blocks.retain(|b| reachable.contains(&b.id));
}

/// Build the literal op for a constant value, or `None` for a value with no literal
/// form (`Undefined` is the `undefined` global, not a literal).
fn literal_op(c: &Constant) -> Option<Op> {
    let mut op = match c {
        Constant::Number(n) => {
            let mut op = Op::new("jsir.numeric_literal");
            op.attrs.push(("extra".into(), Attr::NumericLiteralExtra {
                raw: js_number_to_string(*n),
                value: *n,
            }));
            op.attrs.push(("value".into(), Attr::F64(*n)));
            op
        }
        Constant::Str(s) => {
            let mut op = Op::new("jsir.string_literal");
            op.attrs.push(("extra".into(), Attr::StringLiteralExtra {
                raw: format!("\"{}\"", s.replace('\\', "\\\\").replace('"', "\\\"")),
                raw_value: s.clone(),
            }));
            op.attrs.push(("value".into(), Attr::Str(s.clone())));
            op
        }
        Constant::Bool(b) => {
            let mut op = Op::new("jsir.boolean_literal");
            op.attrs.push(("value".into(), Attr::Bool(*b)));
            op
        }
        Constant::Null => Op::new("jsir.null_literal"),
        Constant::Undefined => return None,
    };
    op.results.clear();
    Some(op)
}

/// Evaluate a single value-defining op against the current lattice. Returns the
/// op's constant value, or `None` if it isn't (yet) provably constant.
fn eval_op(op: &Op, info: &SsaInfo, lattice: &HashMap<ValueId, Constant>) -> Option<Constant> {
    match op.name.as_str() {
        "jsir.numeric_literal" => Some(Constant::Number(attr_f64(op, "value")?)),
        "jsir.string_literal" => Some(Constant::Str(attr_str(op, "value")?.to_string())),
        "jsir.boolean_literal" => Some(Constant::Bool(attr_bool(op, "value")?)),
        "jsir.null_literal" => Some(Constant::Null),
        // A variable read: its constant is the constant of the SSA def reaching it.
        "jsir.identifier" => {
            let result = *op.results.first()?;
            let def = *info.reaching.get(&result)?;
            lattice.get(&def).cloned()
        }
        "jsir.binary_expression" => {
            let lhs = lattice.get(op.operands.first()?)?;
            let rhs = lattice.get(op.operands.get(1)?)?;
            evaluate_binary_op(attr_str(op, "operator_")?, lhs, rhs)
        }
        "jsir.unary_expression" => {
            let operand = lattice.get(op.operands.first()?)?;
            match attr_str(op, "operator_")? {
                "!" => Some(Constant::Bool(!is_truthy(operand))),
                "-" => match operand {
                    Constant::Number(n) => Some(Constant::Number(n * -1.0)),
                    _ => None,
                },
                // `+`, `~`, `typeof`, `void` not folded (mirrors upstream).
                _ => None,
            }
        }
        _ => None,
    }
}

/// `evaluate_phi`: a phi is constant iff every operand is a known constant and they
/// are all (strictly) equal. Returns `None` if any operand is unknown or they
/// disagree. Mirrors upstream `evaluate_phi`.
fn evaluate_phi(
    operands: &[(jsir_ir::BlockId, ValueId)],
    lattice: &HashMap<ValueId, Constant>,
) -> Option<Constant> {
    let mut value: Option<Constant> = None;
    for (_pred, operand) in operands {
        let operand_value = lattice.get(operand)?;
        match &value {
            None => value = Some(operand_value.clone()),
            Some(current) => {
                if !js_strict_equal(current, operand_value) {
                    return None;
                }
            }
        }
    }
    value
}

// ---------------------------------------------------------------------------
// Attr readers
// ---------------------------------------------------------------------------

fn attr_f64(op: &Op, key: &str) -> Option<f64> {
    op.attrs.iter().find(|(k, _)| k == key).and_then(|(_, v)| match v {
        Attr::F64(n) => Some(*n),
        _ => None,
    })
}

fn attr_str<'a>(op: &'a Op, key: &str) -> Option<&'a str> {
    op.attrs.iter().find(|(k, _)| k == key).and_then(|(_, v)| match v {
        Attr::Str(s) => Some(s.as_str()),
        _ => None,
    })
}

fn attr_bool(op: &Op, key: &str) -> Option<bool> {
    op.attrs.iter().find(|(k, _)| k == key).and_then(|(_, v)| match v {
        Attr::Bool(b) => Some(*b),
        _ => None,
    })
}

// =============================================================================
// JS-semantics evaluators — ported verbatim from upstream constant_propagation.rs
// =============================================================================

fn is_truthy(value: &Constant) -> bool {
    match value {
        Constant::Null => false,
        Constant::Undefined => false,
        Constant::Bool(b) => *b,
        Constant::Number(n) => *n != 0.0 && !n.is_nan(),
        Constant::Str(s) => !s.is_empty(),
    }
}

fn evaluate_binary_op(operator: &str, lhs: &Constant, rhs: &Constant) -> Option<Constant> {
    use Constant::{Bool, Number, Str};
    match operator {
        "+" => match (lhs, rhs) {
            (Number(l), Number(r)) => Some(Number(l + r)),
            (Str(l), Str(r)) => Some(Str(format!("{l}{r}"))),
            _ => None,
        },
        "-" => match (lhs, rhs) {
            (Number(l), Number(r)) => Some(Number(l - r)),
            _ => None,
        },
        "*" => match (lhs, rhs) {
            (Number(l), Number(r)) => Some(Number(l * r)),
            _ => None,
        },
        "/" => match (lhs, rhs) {
            (Number(l), Number(r)) => Some(Number(l / r)),
            _ => None,
        },
        "%" => match (lhs, rhs) {
            (Number(l), Number(r)) => Some(Number(l % r)),
            _ => None,
        },
        "**" => match (lhs, rhs) {
            (Number(l), Number(r)) => Some(Number(l.powf(*r))),
            _ => None,
        },
        "|" => match (lhs, rhs) {
            (Number(l), Number(r)) => Some(Number((js_to_int32(*l) | js_to_int32(*r)) as f64)),
            _ => None,
        },
        "&" => match (lhs, rhs) {
            (Number(l), Number(r)) => Some(Number((js_to_int32(*l) & js_to_int32(*r)) as f64)),
            _ => None,
        },
        "^" => match (lhs, rhs) {
            (Number(l), Number(r)) => Some(Number((js_to_int32(*l) ^ js_to_int32(*r)) as f64)),
            _ => None,
        },
        "<<" => match (lhs, rhs) {
            (Number(l), Number(r)) => {
                Some(Number((js_to_int32(*l) << (js_to_uint32(*r) & 0x1f)) as f64))
            }
            _ => None,
        },
        ">>" => match (lhs, rhs) {
            (Number(l), Number(r)) => {
                Some(Number((js_to_int32(*l) >> (js_to_uint32(*r) & 0x1f)) as f64))
            }
            _ => None,
        },
        ">>>" => match (lhs, rhs) {
            (Number(l), Number(r)) => {
                Some(Number((js_to_uint32(*l) >> (js_to_uint32(*r) & 0x1f)) as f64))
            }
            _ => None,
        },
        "<" => match (lhs, rhs) {
            (Number(l), Number(r)) => Some(Bool(l < r)),
            _ => None,
        },
        "<=" => match (lhs, rhs) {
            (Number(l), Number(r)) => Some(Bool(l <= r)),
            _ => None,
        },
        ">" => match (lhs, rhs) {
            (Number(l), Number(r)) => Some(Bool(l > r)),
            _ => None,
        },
        ">=" => match (lhs, rhs) {
            (Number(l), Number(r)) => Some(Bool(l >= r)),
            _ => None,
        },
        "===" => Some(Bool(js_strict_equal(lhs, rhs))),
        "!==" => Some(Bool(!js_strict_equal(lhs, rhs))),
        "==" => Some(Bool(js_abstract_equal(lhs, rhs))),
        "!=" => Some(Bool(!js_abstract_equal(lhs, rhs))),
        // `in` / `instanceof` are not foldable.
        _ => None,
    }
}

fn js_strict_equal(lhs: &Constant, rhs: &Constant) -> bool {
    use Constant::{Bool, Null, Number, Str, Undefined};
    match (lhs, rhs) {
        (Null, Null) => true,
        (Undefined, Undefined) => true,
        (Bool(a), Bool(b)) => a == b,
        (Number(a), Number(b)) => {
            // NaN !== NaN in JS.
            if a.is_nan() || b.is_nan() {
                return false;
            }
            a == b
        }
        (Str(a), Str(b)) => a == b,
        _ => false,
    }
}

fn js_abstract_equal(lhs: &Constant, rhs: &Constant) -> bool {
    use Constant::{Bool, Null, Number, Str, Undefined};
    match (lhs, rhs) {
        (Null, Null) => true,
        (Undefined, Undefined) => true,
        (Null, Undefined) | (Undefined, Null) => true,
        (Bool(a), Bool(b)) => a == b,
        (Number(a), Number(b)) => {
            if a.is_nan() || b.is_nan() {
                return false;
            }
            a == b
        }
        (Str(a), Str(b)) => a == b,
        (Number(n), Str(s)) | (Str(s), Number(n)) => {
            let sv = js_to_number(s);
            if n.is_nan() || sv.is_nan() {
                false
            } else {
                *n == sv
            }
        }
        (Bool(b), other) => {
            let num = if *b { 1.0 } else { 0.0 };
            js_abstract_equal(&Number(num), other)
        }
        (other, Bool(b)) => {
            let num = if *b { 1.0 } else { 0.0 };
            js_abstract_equal(other, &Number(num))
        }
        _ => false,
    }
}

/// JS `ToNumber` on a string.
fn js_to_number(s: &str) -> f64 {
    let trimmed = s.trim();
    if trimmed.is_empty() {
        return 0.0;
    }
    if trimmed == "Infinity" || trimmed == "+Infinity" {
        return f64::INFINITY;
    }
    if trimmed == "-Infinity" {
        return f64::NEG_INFINITY;
    }
    if trimmed.starts_with("0x") || trimmed.starts_with("0X") {
        return u64::from_str_radix(&trimmed[2..], 16).map_or(f64::NAN, |v| v as f64);
    }
    if trimmed.starts_with("0o") || trimmed.starts_with("0O") {
        return u64::from_str_radix(&trimmed[2..], 8).map_or(f64::NAN, |v| v as f64);
    }
    if trimmed.starts_with("0b") || trimmed.starts_with("0B") {
        return u64::from_str_radix(&trimmed[2..], 2).map_or(f64::NAN, |v| v as f64);
    }
    trimmed.parse::<f64>().unwrap_or(f64::NAN)
}

/// ECMAScript `ToInt32`: f64 → i32 with modular (wrapping) semantics.
fn js_to_int32(n: f64) -> i32 {
    if n.is_nan() || n.is_infinite() || n == 0.0 {
        return 0;
    }
    let int64 = (n.trunc() as i64) & 0xFFFFFFFF;
    if int64 >= 0x80000000 {
        (int64 as u32) as i32
    } else {
        int64 as i32
    }
}

/// ECMAScript `ToUint32`: f64 → u32 with modular (wrapping) semantics.
fn js_to_uint32(n: f64) -> u32 {
    js_to_int32(n) as u32
}

/// Approximate ECMAScript `Number::toString` (for a folded literal's raw text).
/// Ported from upstream; diverges from JS only for exotic values near the
/// exponential-notation thresholds.
fn js_number_to_string(n: f64) -> String {
    if n.is_nan() {
        return "NaN".to_string();
    }
    if n.is_infinite() {
        return if n > 0.0 { "Infinity".to_string() } else { "-Infinity".to_string() };
    }
    if n == 0.0 {
        return "0".to_string();
    }
    if n.fract() == 0.0 && n.abs() < (i64::MAX as f64) {
        return format!("{}", n as i64);
    }
    format!("{n}")
}
