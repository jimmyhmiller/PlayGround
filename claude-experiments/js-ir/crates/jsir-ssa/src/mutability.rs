//! The mutable-range data model consumed by reactive-scope inference.
//!
//! For every value we carry the program-point range over which it (or anything
//! it is aliased/captured with) might still be mutated. A value whose range
//! collapses to its definition is effectively immutable afterward and can be
//! memoized / frozen freely; values with overlapping mutable ranges must live in
//! the same reactive scope.
//!
//! The ranges themselves are produced by the two-phase React-Compiler model in
//! [`crate::aliasing_ranges`] (Phase 1 effects -> Phase 2 abstract heap graph).
//! This module is now just the shared [`Ranges`] type + a debug renderer; the
//! old single-pass union-find that used to live here was replaced by that
//! faithful port.

use std::collections::HashMap;

use crate::cfg::{BlockId, Cfg, Op, Value};

/// A linear program point (instruction index in RPO order).
pub type Point = u32;

#[derive(Debug, Clone)]
pub struct Ranges {
    /// Inclusive `[start, end]` mutable range per value (in program points).
    pub range: HashMap<Value, (Point, Point)>,
    /// Whether a value is a *reference* (object/array/unknown) vs a primitive.
    pub is_ref: HashMap<Value, bool>,
    /// Definition point of each value.
    pub def: HashMap<Value, Point>,
    /// Alias-set representative per value. Retained for API compatibility; the
    /// two-phase model does not group by alias roots (scopes come from range
    /// overlap), so this is the identity map and is not consumed downstream.
    pub alias_root: HashMap<Value, Value>,
    /// Program point of each block's terminator (the point *after* the block's
    /// last instruction), in the same linear point space as instruction defs.
    pub term_point: HashMap<BlockId, Point>,
}

impl Ranges {
    /// Is this value mutated after its definition?
    pub fn is_mutable_after_def(&self, v: Value) -> bool {
        match (self.range.get(&v), self.def.get(&v)) {
            (Some((_, end)), Some(d)) => *end > *d,
            _ => false,
        }
    }
}

/// Whether an op's result is a reference (object/array/unknown) vs a primitive.
pub fn op_is_reference(op: &Op) -> bool {
    match op {
        Op::Const(_) => false,            // all our consts are primitive
        Op::Bin(_, _, _) | Op::Un(_, _) => false, // primitive results
        Op::MakeObject(_) | Op::MakeArray(_) => true,
        // Member reads, calls, globals, var reads: unknown -> assume reference.
        _ => true,
    }
}

/// A debug rendering of the ranges (for golden tests): one line per value with a
/// result, sorted by definition point.
pub fn render(cfg: &Cfg, r: &Ranges) -> String {
    let mut rows: Vec<(Point, String)> = Vec::new();
    for b in &cfg.blocks {
        for ins in &b.instrs {
            if let Some(v) = ins.result {
                let (s, e) = r.range.get(&v).copied().unwrap_or((0, 0));
                let kind = if *r.is_ref.get(&v).unwrap_or(&false) { "ref" } else { "val" };
                let mutated = if r.is_mutable_after_def(v) { " MUT" } else { "" };
                rows.push((*r.def.get(&v).unwrap_or(&0), format!("%{} [{s}..{e}] {kind}{mutated}", v.0)));
            }
        }
    }
    rows.sort_by_key(|(p, _)| *p);
    rows.into_iter().map(|(_, s)| s).collect::<Vec<_>>().join("\n")
}
