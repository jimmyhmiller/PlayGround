//! Phase 2 of the React-Compiler mutation/aliasing model:
//! **`infer_mutation_aliasing_ranges`** — the abstract heap graph that consumes
//! the Phase-1 [`crate::effects::AliasingEffect`]s and computes each value's
//! mutable range.
//!
//! Faithful port of upstream
//! `react_compiler_inference::infer_mutation_aliasing_ranges` (which is itself a
//! port of `src/Inference/InferMutationAliasingRanges.ts`). Upstream keys the
//! graph off `IdentifierId`; our CFG is already SSA, so we key off [`Value`].
//! Upstream's `EvaluationOrder` (a per-instruction program point) is our
//! [`crate::mutability::Point`] (RPO instruction index).
//!
//! This module currently provides the data model + the `mutate()` worklist (the
//! heart of the pass) and is exercised by unit tests. Wiring it to consume real
//! Phase-1 effects and produce a [`crate::mutability::Ranges`]-shaped result is a
//! follow-up step (it is **not yet called** by the compile path).

use std::collections::HashMap;

use crate::cfg::Value;

/// A program point — the evaluation order of an instruction. Mirrors upstream's
/// `EvaluationOrder`; in our substrate this is the RPO instruction index used by
/// [`crate::mutability::Point`].
pub type Point = u32;

// =============================================================================
// MutationKind
// =============================================================================

/// The "strength" of a mutation reaching a node. Faithful to upstream
/// `MutationKind`. Ordering matters: a mutation only ever *raises* a node's
/// level (`None` < `Conditional` < `Definite`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum MutationKind {
    None = 0,
    Conditional = 1,
    Definite = 2,
}

// =============================================================================
// Edges and nodes
// =============================================================================

/// The three kinds of data-flow edge in the abstract heap graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EdgeKind {
    /// Information flow `from` -> `into` (e.g. `x` into `[x]`). Followed forward
    /// by any mutation, but only followed *backward* by a transitive mutation.
    Capture,
    /// Aliasing `from` -> `into`: mutating either implies mutating the other.
    /// Followed forward and backward unconditionally.
    Alias,
    /// Potential aliasing: like `Alias` but every traversal *downgrades* the
    /// mutation to `Conditional`.
    MaybeAlias,
}

/// An outgoing edge on a node, tagged with the effect-stream `index` at which it
/// was created (so a mutation only propagates through edges that existed
/// *before* the mutation in program order).
#[derive(Debug, Clone)]
pub struct Edge {
    pub index: usize,
    pub node: Value,
    pub kind: EdgeKind,
}

/// Per-node record of a mutation that reached it. Upstream also carries a source
/// location for diagnostics; our diagnostics flow through the Phase-1 error
/// effects (`validate.rs`), so we keep only the `kind`.
#[derive(Debug, Clone, Copy)]
pub struct MutationInfo {
    pub kind: MutationKind,
}

/// What a node *is*. Faithful to upstream `NodeValue` — `Function` references a
/// lowered closure body (index into `Cfg::nested`) rather than upstream's
/// `FunctionId`.
#[derive(Debug, Clone)]
pub enum NodeValue {
    Object,
    Phi,
    Function { body: usize },
}

/// A node in the abstract heap graph: one per tracked [`Value`].
#[derive(Debug, Clone)]
pub struct Node {
    pub id: Value,
    /// Backward "created-from" edges (`into` was derived from these): a
    /// transitive mutation of `into` flows back to them.
    pub created_from: HashMap<Value, usize>,
    /// Backward capture edges (values captured *into* this node).
    pub captures: HashMap<Value, usize>,
    /// Backward alias edges.
    pub aliases: HashMap<Value, usize>,
    /// Backward maybe-alias edges.
    pub maybe_aliases: HashMap<Value, usize>,
    /// Forward edges (this node flows *into* these).
    pub edges: Vec<Edge>,
    /// The strongest transitive mutation reaching this node.
    pub transitive: Option<MutationInfo>,
    /// The strongest direct (local) mutation reaching this node.
    pub local: Option<MutationInfo>,
    /// The latest effect-stream `index` at which this node was mutated.
    pub last_mutated: usize,
    pub value: NodeValue,
}

impl Node {
    fn new(id: Value, value: NodeValue) -> Self {
        Node {
            id,
            created_from: HashMap::new(),
            captures: HashMap::new(),
            aliases: HashMap::new(),
            maybe_aliases: HashMap::new(),
            edges: Vec::new(),
            transitive: None,
            local: None,
            last_mutated: 0,
            value,
        }
    }
}

// =============================================================================
// AliasingState
// =============================================================================

/// The abstract heap graph. Faithful to upstream `AliasingState`, plus an
/// auxiliary per-value mutable-range-end map (upstream writes the range end into
/// `env.identifiers[].mutable_range`; we keep it local to the graph).
pub struct AliasingState {
    pub nodes: HashMap<Value, Node>,
    /// `mutable_range.end` per value, extended by `mutate()`.
    pub range_end: HashMap<Value, Point>,
}

impl AliasingState {
    pub fn new() -> Self {
        AliasingState { nodes: HashMap::new(), range_end: HashMap::new() }
    }

    /// Create (or replace) a node for `place`.
    pub fn create(&mut self, place: Value, value: NodeValue) {
        self.nodes.insert(place, Node::new(place, value));
    }

    /// `into = create-from from` — `into` is a fresh object derived from `from`.
    /// Adds a forward `Alias` edge `from -> into` and records the backward
    /// `created_from` link on `into`. Faithful to upstream `create_from`.
    pub fn create_from(&mut self, index: usize, from: Value, into: Value) {
        self.create(into, NodeValue::Object);
        if let Some(from_node) = self.nodes.get_mut(&from) {
            from_node.edges.push(Edge { index, node: into, kind: EdgeKind::Alias });
        }
        if let Some(to_node) = self.nodes.get_mut(&into) {
            to_node.created_from.entry(from).or_insert(index);
        }
    }

    /// `into` captures `from` (information flow). Both nodes must exist.
    pub fn capture(&mut self, index: usize, from: Value, into: Value) {
        if !self.nodes.contains_key(&from) || !self.nodes.contains_key(&into) {
            return;
        }
        self.nodes.get_mut(&from).unwrap().edges.push(Edge { index, node: into, kind: EdgeKind::Capture });
        self.nodes.get_mut(&into).unwrap().captures.entry(from).or_insert(index);
    }

    /// `into` aliases `from`. Both nodes must exist.
    pub fn assign(&mut self, index: usize, from: Value, into: Value) {
        if !self.nodes.contains_key(&from) || !self.nodes.contains_key(&into) {
            return;
        }
        self.nodes.get_mut(&from).unwrap().edges.push(Edge { index, node: into, kind: EdgeKind::Alias });
        self.nodes.get_mut(&into).unwrap().aliases.entry(from).or_insert(index);
    }

    /// `into` maybe-aliases `from`. Both nodes must exist.
    pub fn maybe_alias(&mut self, index: usize, from: Value, into: Value) {
        if !self.nodes.contains_key(&from) || !self.nodes.contains_key(&into) {
            return;
        }
        self.nodes.get_mut(&from).unwrap().edges.push(Edge { index, node: into, kind: EdgeKind::MaybeAlias });
        self.nodes.get_mut(&into).unwrap().maybe_aliases.entry(from).or_insert(index);
    }

    /// Propagate a mutation through the graph. This is the heart of the pass — a
    /// faithful port of upstream `AliasingState::mutate`.
    ///
    /// - `index`: the effect-stream position of the mutation. Edges created at or
    ///   after this index are *not* followed (temporal correctness).
    /// - `start`: the mutated value.
    /// - `end`: when `Some`, every touched node's `mutable_range.end` is raised to
    ///   it. `None` for *simulated* mutations (used only to discover data-flow).
    /// - `transitive`: whether this is a transitive mutation (reaches captures).
    /// - `start_kind`: the mutation strength at the origin.
    #[allow(clippy::too_many_arguments)]
    pub fn mutate(
        &mut self,
        index: usize,
        start: Value,
        end: Option<Point>,
        transitive: bool,
        start_kind: MutationKind,
    ) {
        #[derive(Clone, Copy, PartialEq)]
        enum Direction {
            Backwards,
            Forwards,
        }
        #[derive(Clone)]
        struct QueueEntry {
            place: Value,
            transitive: bool,
            direction: Direction,
            kind: MutationKind,
        }

        let mut seen: HashMap<Value, MutationKind> = HashMap::new();
        let mut queue: Vec<QueueEntry> = vec![QueueEntry {
            place: start,
            transitive,
            direction: Direction::Backwards,
            kind: start_kind,
        }];

        while let Some(entry) = queue.pop() {
            let current = entry.place;
            if let Some(prev) = seen.get(&current).copied() {
                if prev >= entry.kind {
                    continue;
                }
            }
            seen.insert(current, entry.kind);

            let node = match self.nodes.get_mut(&current) {
                Some(n) => n,
                None => continue,
            };

            node.last_mutated = node.last_mutated.max(index);

            if let Some(end_val) = end {
                let e = self.range_end.entry(node.id).or_insert(0);
                *e = (*e).max(end_val);
            }

            if entry.transitive {
                match node.transitive {
                    None => node.transitive = Some(MutationInfo { kind: entry.kind }),
                    Some(existing) if existing.kind < entry.kind => {
                        node.transitive = Some(MutationInfo { kind: entry.kind })
                    }
                    _ => {}
                }
            } else {
                match node.local {
                    None => node.local = Some(MutationInfo { kind: entry.kind }),
                    Some(existing) if existing.kind < entry.kind => {
                        node.local = Some(MutationInfo { kind: entry.kind })
                    }
                    _ => {}
                }
            }

            // Snapshot the node's edges/links before requeuing (the borrow ends).
            let edges: Vec<Edge> = node.edges.clone();
            let is_phi = matches!(node.value, NodeValue::Phi);
            let node_aliases: Vec<(Value, usize)> = node.aliases.iter().map(|(&k, &v)| (k, v)).collect();
            let node_maybe_aliases: Vec<(Value, usize)> = node.maybe_aliases.iter().map(|(&k, &v)| (k, v)).collect();
            let node_captures: Vec<(Value, usize)> = node.captures.iter().map(|(&k, &v)| (k, v)).collect();
            let node_created_from: Vec<(Value, usize)> = node.created_from.iter().map(|(&k, &v)| (k, v)).collect();

            // Forward edges: mutate(a) where a -Capture/Alias/MaybeAlias-> b
            // propagates to b. Edges are appended in index order, so an edge at or
            // after `index` (and everything after it) post-dates this mutation.
            for edge in &edges {
                if edge.index >= index {
                    break;
                }
                queue.push(QueueEntry {
                    place: edge.node,
                    transitive: entry.transitive,
                    direction: Direction::Forwards,
                    kind: if edge.kind == EdgeKind::MaybeAlias {
                        MutationKind::Conditional
                    } else {
                        entry.kind
                    },
                });
            }

            // created_from is always followed backward, transitively.
            for (alias, when) in &node_created_from {
                if *when >= index {
                    continue;
                }
                queue.push(QueueEntry {
                    place: *alias,
                    transitive: true,
                    direction: Direction::Backwards,
                    kind: entry.kind,
                });
            }

            // Backward alias / maybe-alias edges. A forward arrival at a phi does
            // not flow back out of it (the phi's operands are not aliases of each
            // other), so only follow backward when arriving backward or at a
            // non-phi node.
            if entry.direction == Direction::Backwards || !is_phi {
                for (alias, when) in &node_aliases {
                    if *when >= index {
                        continue;
                    }
                    queue.push(QueueEntry {
                        place: *alias,
                        transitive: entry.transitive,
                        direction: Direction::Backwards,
                        kind: entry.kind,
                    });
                }
                for (alias, when) in &node_maybe_aliases {
                    if *when >= index {
                        continue;
                    }
                    queue.push(QueueEntry {
                        place: *alias,
                        transitive: entry.transitive,
                        direction: Direction::Backwards,
                        kind: MutationKind::Conditional,
                    });
                }
            }

            // Captures flow backward only for transitive mutations.
            if entry.transitive {
                for (capture, when) in &node_captures {
                    if *when >= index {
                        continue;
                    }
                    queue.push(QueueEntry {
                        place: *capture,
                        transitive: entry.transitive,
                        direction: Direction::Backwards,
                        kind: entry.kind,
                    });
                }
            }
        }
    }
}

impl Default for AliasingState {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// compute_ranges — the Part-1 driver: feed Phase-1 effects into the graph, then
// run the pending mutations, and read out a `mutability::Ranges`-shaped result.
//
// Faithful port of `infer_mutation_aliasing_ranges` Part 1 (lines 446-707),
// restricted to the intra-function mutable ranges (we do not compute the
// external function signature / Part 3, which only matters for callers).
// =============================================================================

use crate::cfg::{BlockId, Cfg, Op, Term};
use crate::effects::AliasingEffect;
use crate::infer_effects::EffectResults;
use crate::mutability::Ranges;

/// A pending mutation collected during the program-order walk, applied after the
/// whole graph is built (so every edge that pre-dates it exists).
struct PendingMutation {
    index: usize,
    end: Point,
    transitive: bool,
    kind: MutationKind,
    place: Value,
}

/// Convenience: run the full Phase-1 effect inference and produce ranges. The
/// drop-in equivalent of [`crate::mutability::analyze`] via the two-phase model.
pub fn analyze(cfg: &Cfg) -> Ranges {
    let is_component = crate::types::is_react_like_name(cfg.fn_name.as_deref().unwrap_or(""));
    let mut shapes = crate::types::build_builtin_shapes();
    let globals = crate::types::build_default_globals(&mut shapes);
    let types = crate::infer_types::infer(cfg, &shapes, &globals, is_component, Default::default());
    let effects = crate::infer_effects::infer(cfg, &types, &shapes, is_component);
    compute_ranges(cfg, &effects)
}

/// Whether the two-phase aliasing-ranges model is enabled (env `JSIR_ALIASING_RANGES`).
pub fn enabled() -> bool {
    std::env::var("JSIR_ALIASING_RANGES").map(|v| v != "0" && !v.is_empty()).unwrap_or(false)
}

/// Per-instruction mutated values, derived from the Phase-1 effect stream — the
/// single source of truth for "what does this instruction mutate" that
/// `memoize_plan`'s branch-boundary check needs (replacing its coarse
/// any-non-pure-call-mutates-all-args heuristic). A hook call (`useMemo`,
/// `useState`, …) *freezes* its args, so it contributes no mutation here, while
/// an unknown call's args appear (as transitive-conditional mutations). Keyed by
/// `(BlockId, instr_index_in_block)`.
pub fn mutations_by_instr(cfg: &Cfg) -> HashMap<(BlockId, usize), Vec<Value>> {
    let is_component = crate::types::is_react_like_name(cfg.fn_name.as_deref().unwrap_or(""));
    let mut shapes = crate::types::build_builtin_shapes();
    let globals = crate::types::build_default_globals(&mut shapes);
    let types = crate::infer_types::infer(cfg, &shapes, &globals, is_component, Default::default());
    let effects = crate::infer_effects::infer(cfg, &types, &shapes, is_component);
    let mut out: HashMap<(BlockId, usize), Vec<Value>> = HashMap::new();
    for ie in &effects.instrs {
        let mut muts = Vec::new();
        for e in &ie.effects {
            match e {
                AliasingEffect::Mutate { value, .. }
                | AliasingEffect::MutateConditionally { value }
                | AliasingEffect::MutateTransitive { value }
                | AliasingEffect::MutateTransitiveConditionally { value } => muts.push(*value),
                _ => {}
            }
        }
        if !muts.is_empty() {
            out.insert((ie.block, ie.index), muts);
        }
    }
    out
}

/// Build the abstract heap graph from `effects` and produce per-value mutable
/// ranges. Drop-in replacement for [`crate::mutability::analyze`]'s output.
pub fn compute_ranges(cfg: &Cfg, effects: &EffectResults) -> Ranges {
    // -------------------------------------------------------------------------
    // 1. Program-point numbering, identical to `mutability::analyze`, so the
    //    ranges this pass produces are directly comparable to the union-find's.
    // -------------------------------------------------------------------------
    let order = crate::ssa::reverse_postorder(cfg);
    let mut def: HashMap<Value, Point> = HashMap::new();
    let mut is_ref: HashMap<Value, bool> = HashMap::new();
    let mut block_start: HashMap<BlockId, Point> = HashMap::new();
    let mut term_point: HashMap<BlockId, Point> = HashMap::new();
    let mut point_of: HashMap<(BlockId, usize), Point> = HashMap::new();

    for p in &cfg.params {
        def.insert(*p, 0);
        is_ref.insert(*p, true);
    }
    let mut point = 0u32;
    for &b in &order {
        let block = cfg.block(b);
        block_start.insert(b, point);
        for bp in &block.params {
            def.insert(*bp, point);
            is_ref.insert(*bp, false);
        }
        for (idx, ins) in block.instrs.iter().enumerate() {
            if let Some(r) = ins.result {
                def.insert(r, point);
                is_ref.insert(r, op_is_reference(&ins.op));
            }
            point_of.insert((b, idx), point);
            point += 1;
        }
        term_point.insert(b, point);
        point += 1;
    }
    // Phi ref-ness fixpoint (a block arg is a ref if any incoming operand is).
    let mut changed = true;
    while changed {
        changed = false;
        for b in &cfg.blocks {
            let edges: Vec<(BlockId, Vec<Value>)> = match &b.term {
                Term::Br(t, a) => vec![(*t, a.clone())],
                Term::CondBr { then_block, then_args, else_block, else_args, .. } => {
                    vec![(*then_block, then_args.clone()), (*else_block, else_args.clone())]
                }
                _ => vec![],
            };
            for (succ, args) in edges {
                let params = cfg.block(succ).params.clone();
                for (pp, a) in params.iter().zip(args) {
                    if *is_ref.get(&a).unwrap_or(&false) && !is_ref.get(pp).copied().unwrap_or(false) {
                        is_ref.insert(*pp, true);
                        changed = true;
                    }
                }
            }
        }
    }

    // -------------------------------------------------------------------------
    // 2. Drive the graph in program order.
    // -------------------------------------------------------------------------
    let mut state = AliasingState::new();
    // Params are tracked nodes (upstream creates param/context nodes up front).
    for p in &cfg.params {
        state.create(*p, NodeValue::Object);
    }

    // Per-instruction effects, keyed by (block, idx) for program-order lookup.
    let mut effects_by_instr: HashMap<(BlockId, usize), &Vec<AliasingEffect>> = HashMap::new();
    for ie in &effects.instrs {
        effects_by_instr.insert((ie.block, ie.index), &ie.effects);
    }

    // Phi operands: for each (target block, param idx) the (pred, operand) pairs.
    let phi_ops = collect_phi_operands(cfg);

    // Deferred phi-operand assigns: operands from a predecessor not yet visited
    // (back-edges) are applied when that predecessor block is processed, using
    // the index captured at phi-creation time.
    struct PendingPhi {
        from: Value,
        into: Value,
        index: usize,
    }
    let mut pending_phis: HashMap<BlockId, Vec<PendingPhi>> = HashMap::new();
    let mut mutations: Vec<PendingMutation> = Vec::new();
    let mut index: usize = 0;
    let mut seen_blocks: std::collections::HashSet<BlockId> = std::collections::HashSet::new();

    for &b in &order {
        let block = cfg.block(b);

        // Phis: create Phi nodes, then wire each operand (or defer to its pred).
        for (i, &phi) in block.params.iter().enumerate() {
            state.create(phi, NodeValue::Phi);
            if let Some(ops) = phi_ops.get(&(b, i)) {
                for &(pred, operand) in ops {
                    if seen_blocks.contains(&pred) {
                        // Ensure the operand node exists (it always should, being
                        // defined before this point in a forward edge).
                        if !state.nodes.contains_key(&operand) {
                            state.create(operand, NodeValue::Object);
                        }
                        state.assign(index, operand, phi);
                    } else {
                        pending_phis.entry(pred).or_default().push(PendingPhi { from: operand, into: phi, index });
                    }
                    index += 1;
                }
            }
        }
        seen_blocks.insert(b);

        // Instruction effects.
        for idx in 0..block.instrs.len() {
            let pt = *point_of.get(&(b, idx)).expect("point recorded");
            let effs = match effects_by_instr.get(&(b, idx)) {
                Some(e) => *e,
                None => continue,
            };
            for effect in effs {
                apply_to_graph(&mut state, &mut mutations, &mut index, pt, effect);
            }
        }

        // Apply phi operands that were waiting on this block as a predecessor.
        if let Some(list) = pending_phis.remove(&b) {
            for p in list {
                if !state.nodes.contains_key(&p.from) {
                    state.create(p.from, NodeValue::Object);
                }
                state.assign(p.index, p.from, p.into);
            }
        }
    }

    // -------------------------------------------------------------------------
    // 3. Run the collected mutations through the graph.
    // -------------------------------------------------------------------------
    for m in &mutations {
        state.mutate(m.index, m.place, Some(m.end), m.transitive, m.kind);
    }

    // -------------------------------------------------------------------------
    // 4. Read out ranges. start = def point; end = max(def, propagated end).
    // -------------------------------------------------------------------------
    let mut range: HashMap<Value, (Point, Point)> = HashMap::new();
    let mut alias_root: HashMap<Value, Value> = HashMap::new();
    let all_values: Vec<Value> = def.keys().copied().collect();
    for v in &all_values {
        let d = *def.get(v).unwrap_or(&0);
        let end = state.range_end.get(v).copied().unwrap_or(d).max(d);
        range.insert(*v, (d, end));
        alias_root.insert(*v, *v); // not consumed downstream; identity for the drop-in
    }

    Ranges { range, is_ref, def, alias_root, term_point }
}

/// Apply one Phase-1 effect to the graph, building edges and collecting
/// mutations, advancing the effect-stream `index` exactly as upstream Part 1.
fn apply_to_graph(
    state: &mut AliasingState,
    mutations: &mut Vec<PendingMutation>,
    index: &mut usize,
    pt: Point,
    effect: &AliasingEffect,
) {
    match effect {
        AliasingEffect::Create { into, .. } => {
            state.create(*into, NodeValue::Object);
        }
        AliasingEffect::CreateFrom { from, into } => {
            // create_from requires `from` to exist; if it doesn't (e.g. a
            // member load off an untracked value) fall back to a bare Object.
            if state.nodes.contains_key(from) {
                state.create_from(*index, *from, *into);
            } else {
                state.create(*into, NodeValue::Object);
            }
            *index += 1;
        }
        AliasingEffect::Assign { from, into } => {
            if !state.nodes.contains_key(&into.clone()) {
                state.create(*into, NodeValue::Object);
            }
            state.assign(*index, *from, *into);
            *index += 1;
        }
        AliasingEffect::Alias { from, into } => {
            state.assign(*index, *from, *into);
            *index += 1;
        }
        AliasingEffect::MaybeAlias { from, into } => {
            state.maybe_alias(*index, *from, *into);
            *index += 1;
        }
        AliasingEffect::Capture { from, into } => {
            state.capture(*index, *from, *into);
            *index += 1;
        }
        AliasingEffect::MutateTransitive { value } => {
            mutations.push(PendingMutation { index: *index, end: pt + 1, transitive: true, kind: MutationKind::Definite, place: *value });
            *index += 1;
        }
        AliasingEffect::MutateTransitiveConditionally { value } => {
            mutations.push(PendingMutation { index: *index, end: pt + 1, transitive: true, kind: MutationKind::Conditional, place: *value });
            *index += 1;
        }
        AliasingEffect::Mutate { value, .. } => {
            mutations.push(PendingMutation { index: *index, end: pt + 1, transitive: false, kind: MutationKind::Definite, place: *value });
            *index += 1;
        }
        AliasingEffect::MutateConditionally { value } => {
            mutations.push(PendingMutation { index: *index, end: pt + 1, transitive: false, kind: MutationKind::Conditional, place: *value });
            *index += 1;
        }
        // Freeze, ImmutableCapture, Apply, Error, Render: no graph edge / no index
        // advance (Render does advance upstream, but we don't model ref-in-render
        // here; it never participates in ranges).
        _ => {}
    }
}

/// Phi operands: `(target_block, param_index) -> [(pred_block, operand_value)]`.
fn collect_phi_operands(cfg: &Cfg) -> HashMap<(BlockId, usize), Vec<(BlockId, Value)>> {
    let mut out: HashMap<(BlockId, usize), Vec<(BlockId, Value)>> = HashMap::new();
    for b in &cfg.blocks {
        let mut push = |target: BlockId, args: &[Value]| {
            for (i, a) in args.iter().enumerate() {
                out.entry((target, i)).or_default().push((b.id, *a));
            }
        };
        match &b.term {
            Term::Br(t, a) => push(*t, a),
            Term::CondBr { then_block, then_args, else_block, else_args, .. } => {
                push(*then_block, then_args);
                push(*else_block, else_args);
            }
            Term::Ret(_) | Term::Unreachable => {}
        }
    }
    out
}

/// Whether an op's result is a reference (object/array/unknown) vs a primitive.
/// Identical classification to `mutability::op_is_reference`.
fn op_is_reference(op: &Op) -> bool {
    match op {
        Op::Const(_) => false,
        Op::Bin(_, _, _) | Op::Un(_, _) => false,
        Op::MakeObject(_) | Op::MakeArray(_) => true,
        _ => true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn st_with(values: &[(Value, NodeValue)]) -> AliasingState {
        let mut st = AliasingState::new();
        for (v, kind) in values {
            st.create(*v, kind.clone());
        }
        st
    }

    const A: Value = Value(0);
    const B: Value = Value(1);

    #[test]
    fn capture_is_not_backward_alias() {
        // B captures A (A flows into B). Mutating B non-transitively must NOT
        // reach A — this is the entire globals-Boolean fix.
        let mut st = st_with(&[(A, NodeValue::Object), (B, NodeValue::Object)]);
        st.capture(0, A, B);
        st.mutate(10, B, Some(100), /*transitive*/ false, MutationKind::Definite);
        assert!(st.nodes[&A].local.is_none(), "non-transitive mutate(B) must not touch captured A");
        assert!(st.nodes[&B].local.is_some());
        assert!(!st.range_end.contains_key(&A), "A's range must not be extended");
    }

    #[test]
    fn capture_flows_forward() {
        // Mutating the captured source A DOES flow forward into the container B.
        let mut st = st_with(&[(A, NodeValue::Object), (B, NodeValue::Object)]);
        st.capture(0, A, B);
        st.mutate(10, A, Some(100), false, MutationKind::Definite);
        assert!(st.nodes[&A].local.is_some());
        assert!(st.nodes[&B].local.is_some(), "forward capture edge propagates A -> B");
    }

    #[test]
    fn capture_is_backward_when_transitive() {
        // A transitive mutation of the container B reaches the captured A.
        let mut st = st_with(&[(A, NodeValue::Object), (B, NodeValue::Object)]);
        st.capture(0, A, B);
        st.mutate(10, B, Some(100), /*transitive*/ true, MutationKind::Definite);
        assert!(st.nodes[&A].transitive.is_some(), "transitive mutate(B) reaches captured A");
    }

    #[test]
    fn alias_propagates_both_ways() {
        // B aliases A. Mutating either reaches the other.
        let mut st = st_with(&[(A, NodeValue::Object), (B, NodeValue::Object)]);
        st.assign(0, A, B);
        // backward: mutate(B) -> A
        st.mutate(10, B, Some(100), false, MutationKind::Definite);
        assert!(st.nodes[&A].local.is_some(), "mutate(B) reaches aliased A backward");

        // forward: mutate(A) -> B
        let mut st2 = st_with(&[(A, NodeValue::Object), (B, NodeValue::Object)]);
        st2.assign(0, A, B);
        st2.mutate(10, A, Some(100), false, MutationKind::Definite);
        assert!(st2.nodes[&B].local.is_some(), "mutate(A) reaches aliased B forward");
    }

    #[test]
    fn maybe_alias_downgrades_to_conditional() {
        // B maybe-aliases A. A definite mutation arrives at the other side as
        // Conditional.
        let mut st = st_with(&[(A, NodeValue::Object), (B, NodeValue::Object)]);
        st.maybe_alias(0, A, B);
        st.mutate(10, B, Some(100), false, MutationKind::Definite);
        assert_eq!(st.nodes[&A].local.map(|m| m.kind), Some(MutationKind::Conditional));
        // The origin keeps its definite strength.
        assert_eq!(st.nodes[&B].local.map(|m| m.kind), Some(MutationKind::Definite));
    }

    #[test]
    fn edge_after_mutation_index_is_not_followed() {
        // An alias edge created at index 20 is invisible to a mutation at index 10.
        let mut st = st_with(&[(A, NodeValue::Object), (B, NodeValue::Object)]);
        st.assign(20, A, B);
        st.mutate(10, B, Some(100), false, MutationKind::Definite);
        assert!(st.nodes[&A].local.is_none(), "later-created edge must not propagate to earlier mutation");
    }

    #[test]
    fn range_end_extends_to_mutation_point() {
        let mut st = st_with(&[(A, NodeValue::Object), (B, NodeValue::Object)]);
        st.assign(0, A, B);
        st.mutate(10, B, Some(42), false, MutationKind::Definite);
        assert_eq!(st.range_end.get(&B).copied(), Some(42));
        assert_eq!(st.range_end.get(&A).copied(), Some(42), "alias A's range extends with B");
    }

    #[test]
    fn level_only_raises() {
        // A conditional mutation followed by re-reaching at the same node never
        // lowers the recorded level; a stronger one raises it.
        let mut st = st_with(&[(A, NodeValue::Object)]);
        st.mutate(10, A, None, false, MutationKind::Conditional);
        assert_eq!(st.nodes[&A].local.map(|m| m.kind), Some(MutationKind::Conditional));
        st.mutate(11, A, None, false, MutationKind::Definite);
        assert_eq!(st.nodes[&A].local.map(|m| m.kind), Some(MutationKind::Definite));
    }
}
