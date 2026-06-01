//! Mutable-range inference — the analysis at the heart of the React Compiler.
//!
//! For every value we compute the program-point range over which it (or anything
//! it is aliased with) might still be mutated. A value whose range collapses to
//! its definition is effectively immutable afterward and can be memoized /
//! frozen freely; values with overlapping mutable ranges must live in the same
//! reactive scope (they are produced and mutated together).
//!
//! This is intentionally **sound (over-approximate)**, mirroring how the React
//! Compiler must never under-report a mutation:
//!  - `obj.x = v` and any function call mutate their object/argument operands;
//!  - capturing a value into an object/array (`{a: v}`, `[v]`, `o.x = v`) or
//!    passing it to a call **aliases** it (union-find), so a later mutation of
//!    one alias widens the whole set's range.

use std::collections::HashMap;

use crate::cfg::{BlockId, Cfg, Op, Value};
use crate::ssa::reverse_postorder;

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
    /// Alias-set representative per value (union-find roots).
    pub alias_root: HashMap<Value, Value>,
    /// Program point of each block's terminator (the point *after* the block's
    /// last instruction). React's reactive-scope alignment compares scope ranges
    /// against `terminal.id`; this gives each terminator a slot in the same
    /// linear point space so a value's def Point precedes its terminal-use Point.
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

/// Union-find over values.
struct Uf {
    parent: HashMap<Value, Value>,
}
impl Uf {
    fn new() -> Uf {
        Uf { parent: HashMap::new() }
    }
    fn find(&mut self, v: Value) -> Value {
        let p = *self.parent.entry(v).or_insert(v);
        if p == v {
            v
        } else {
            let r = self.find(p);
            self.parent.insert(v, r);
            r
        }
    }
    fn union(&mut self, a: Value, b: Value) {
        let (ra, rb) = (self.find(a), self.find(b));
        if ra != rb {
            // Keep the smaller id as root for determinism.
            let (root, child) = if ra.0 <= rb.0 { (ra, rb) } else { (rb, ra) };
            self.parent.insert(child, root);
        }
    }
}

pub fn analyze(cfg: &Cfg) -> Ranges {
    // 1. Linearize instructions and record def points + reference-ness.
    let order = reverse_postorder(cfg);
    let mut def: HashMap<Value, Point> = HashMap::new();
    let mut is_ref: HashMap<Value, bool> = HashMap::new();
    let mut point = 0u32;

    // Parameters and block arguments are defined "before" the body; treat them
    // as references (unknown provenance) defined at point 0.
    for p in &cfg.params {
        def.insert(*p, 0);
        is_ref.insert(*p, true);
    }
    // Pre-walk to assign points and defs. Block arguments (phis) start as
    // primitive and gain ref-ness from their incoming operands below, so a loop
    // index phi isn't misclassified as a reference.
    //
    // The linear numbering interleaves terminators: each block's instructions are
    // numbered in order, then the block's terminator takes the next point (it sits
    // *after* the block's last instruction), and the counter advances into the
    // next block. This keeps a single total order consistent with RPO in which a
    // value's def Point is strictly less than its block-terminal-use Point, while
    // `block_start_point` is recorded directly rather than recomputed.
    let mut block_start: HashMap<BlockId, Point> = HashMap::new();
    let mut term_point: HashMap<BlockId, Point> = HashMap::new();
    for &b in &order {
        let block = cfg.block(b);
        block_start.insert(b, point);
        for bp in &block.params {
            def.insert(*bp, point);
            is_ref.insert(*bp, false);
        }
        for ins in &block.instrs {
            if let Some(r) = ins.result {
                def.insert(r, point);
                is_ref.insert(r, op_is_reference(&ins.op));
            }
            point += 1;
        }
        // The terminator's point is the slot immediately after the last instr.
        term_point.insert(b, point);
        point += 1;
    }
    // Fixpoint: a block argument is a reference if any operand passed to it is.
    let mut changed = true;
    while changed {
        changed = false;
        for b in &cfg.blocks {
            let edges: Vec<(crate::cfg::BlockId, Vec<Value>)> = match &b.term {
                crate::cfg::Term::Br(t, a) => vec![(*t, a.clone())],
                crate::cfg::Term::CondBr { then_block, then_args, else_block, else_args, .. } => {
                    vec![(*then_block, then_args.clone()), (*else_block, else_args.clone())]
                }
                _ => vec![],
            };
            for (succ, args) in edges {
                let params = cfg.block(succ).params.clone();
                for (p, a) in params.iter().zip(args) {
                    if *is_ref.get(&a).unwrap_or(&false) && !is_ref.get(p).copied().unwrap_or(false) {
                        is_ref.insert(*p, true);
                        changed = true;
                    }
                }
            }
        }
    }

    // Map each value to its defining op, so we can recognize pure constructor
    // calls (`React.createElement`, the JSX runtime `jsx`/`jsxs`).
    let mut def_op: HashMap<Value, &Op> = HashMap::new();
    for b in &cfg.blocks {
        for ins in &b.instrs {
            if let Some(r) = ins.result {
                def_op.insert(r, &ins.op);
            }
        }
    }
    let is_pure_call = |callee: Value| -> bool {
        match def_op.get(&callee) {
            Some(Op::Member { prop: crate::cfg::MemberKey::Static(name), .. }) => {
                matches!(name.as_str(), "createElement" | "jsx" | "jsxs" | "jsxDEV")
            }
            Some(Op::Global(name)) => matches!(name.as_str(), "jsx" | "jsxs" | "_jsx" | "_jsxs"),
            _ => false,
        }
    };
    // The container a value names, following `Member` loads to the root object:
    // `x.y.z` -> `x`. Mutating a member load (or storing through one) mutates the
    // container, so the mutation must be attributed to the base object — member
    // loads are fresh values the union-find aliasing does not connect to the base,
    // so without this a scope's mutable range stops short of `mutate(x.y.z)`.
    let base_object = |mut v: Value| -> Value {
        let mut guard = 0;
        while let Some(Op::Member { obj, .. }) = def_op.get(&v) {
            v = *obj;
            guard += 1;
            if guard > def_op.len() + 1 {
                break;
            }
        }
        v
    };

    // 2. Aliasing + mutation points.
    let mut uf = Uf::new();
    // mutation point per value (direct), later lifted to alias roots.
    let mut last_mut: HashMap<Value, Point> = HashMap::new();
    let note_mut = |v: Value, p: Point, m: &mut HashMap<Value, Point>| {
        let e = m.entry(v).or_insert(p);
        if p > *e {
            *e = p;
        }
    };

    for &b in &order {
        let block = cfg.block(b);
        let mut p = *block_start.get(&b).expect("block start point recorded in pre-walk");
        for ins in &block.instrs {
            match &ins.op {
                Op::MakeObject(props) => {
                    if let Some(r) = ins.result {
                        for (_, v) in props {
                            if *is_ref.get(v).unwrap_or(&true) {
                                uf.union(r, *v);
                            }
                        }
                    }
                }
                Op::MakeArray(elems) => {
                    if let Some(r) = ins.result {
                        for v in elems {
                            if *is_ref.get(v).unwrap_or(&true) {
                                uf.union(r, *v);
                            }
                        }
                    }
                }
                Op::StoreMember { obj, value, .. } => {
                    // Mutates `obj` (and its container if `obj` is itself a member
                    // load: `x.y.z = v` mutates `x`); captures `value` into `obj`.
                    note_mut(base_object(*obj), p, &mut last_mut);
                    if *is_ref.get(value).unwrap_or(&true) {
                        uf.union(*obj, *value);
                    }
                }
                Op::Call { callee, args, .. } => {
                    let pure = is_pure_call(*callee);
                    if let Some(r) = ins.result {
                        // The result references its args either way (the element /
                        // return value may hold them).
                        for a in args {
                            if *is_ref.get(a).unwrap_or(&true) {
                                uf.union(r, *a);
                            }
                        }
                    }
                    // A pure constructor (JSX/createElement) reads its args; an
                    // unknown call may mutate every reference argument.
                    if !pure {
                        for a in args {
                            if *is_ref.get(a).unwrap_or(&true) {
                                // A mutation of `x.y.z` passed to a call mutates the
                                // container `x`; attribute it to the base object so
                                // the scope's range covers the call.
                                note_mut(base_object(*a), p, &mut last_mut);
                            }
                        }
                    }
                }
                _ => {}
            }
            p += 1;
        }
    }

    // 3. Lift mutation points to alias-set roots, then build ranges.
    let mut set_last_mut: HashMap<Value, Point> = HashMap::new();
    let mut set_first_def: HashMap<Value, Point> = HashMap::new();
    let all_values: Vec<Value> = def.keys().copied().collect();
    for v in &all_values {
        let root = uf.find(*v);
        let d = *def.get(v).unwrap_or(&0);
        set_first_def.entry(root).and_modify(|e| *e = (*e).min(d)).or_insert(d);
    }
    for (v, m) in &last_mut {
        let root = uf.find(*v);
        set_last_mut.entry(root).and_modify(|e| *e = (*e).max(*m)).or_insert(*m);
    }

    let mut range = HashMap::new();
    let mut alias_root = HashMap::new();
    for v in &all_values {
        let root = uf.find(*v);
        alias_root.insert(*v, root);
        let start = *def.get(v).unwrap_or(&0);
        let end = set_last_mut.get(&root).copied().unwrap_or(start).max(start);
        range.insert(*v, (start, end));
    }

    Ranges { range, is_ref, def, alias_root, term_point }
}

/// Whether an op's result is a reference (object/array/unknown) vs a primitive.
fn op_is_reference(op: &Op) -> bool {
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
