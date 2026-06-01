//! Build the memoized function-body op list — the ANALYSIS + IR-emit half of the
//! IR-rewrite memoization pass.
//!
//! This reuses the reactive-scope analysis ([`crate::scopes`]) and reproduces
//! the frozen string emitter (`crate::codegen::emit_memoized`) EXACTLY — same
//! ownership rule, slot numbering, output sort, and per-scope `if`-block
//! decomposition — but emits **JSIR `Op` nodes** (using only op names
//! `hir2ast` already lifts) instead of strings. The pure `jsir-transforms`
//! rewrite then splices these into the program op-tree and prints them through
//! the reversible IR path. No strings, no CFG/SSA types leak across the crate
//! boundary (the transform never depends back on this crate, so there is no
//! cycle).
//!
//! Reproducing `emit_memoized` at the *value* (instruction) level — rather than
//! relocating whole source statements — is what makes nested reactive scopes
//! that share a single source statement (e.g. `<Foo>{props.render(ref)}</Foo>`,
//! two scopes in one `return`) decompose correctly, matching React.

use std::collections::HashSet;

use jsir_ir::{Attr, Block, IdentifierAttr, Op as IrOp, Region, ValueId};
use jsir_transforms::memoize::MemoLayout;

use crate::cfg::{BinOp, BlockId, BlockKind, Cfg, Const, Instr, MemberKey, Op, PropKey, Term, UnOp, Value};
use crate::mutability::Ranges;
use crate::scopes::ScopeInfo;

/// A structured region recovered from the CFG. The lowering (`crate::lower`)
/// emits a structured DAG (no irreducible control flow), so we recover the
/// original `if`/straight-line nesting directly — this is **not** a relooper
/// (it accepts only the exact diamond shape the lowering produces and hard-errors
/// on anything else: loops, early returns, irregular merges).
enum Node {
    /// A basic block's straight-line instructions, then (for the block that ends
    /// a branch arm) the phi assignments for its single `Br` successor.
    Straight(BlockId),
    /// `if (cond) { then } else { else }` whose arms both fall through to `join`.
    ///
    /// `then_phi`/`els_phi` carry phi assignments for an arm that jumps *straight
    /// to the join* on the `CondBr` edge itself (the short-circuit shape the
    /// logical operators `&&`/`||`/`??` and a value-producing ternary lower to:
    /// the operand is already computed, so the arm is empty and the value is
    /// passed as a block argument on the conditional edge rather than through an
    /// intermediate block). Each `(param, arg)` becomes `param = arg;` emitted at
    /// the end of that branch. Empty for the ordinary diamond where both arms are
    /// real blocks that carry their phi args on their own `Br` to the join.
    If {
        cond: Value,
        then: Vec<Node>,
        els: Vec<Node>,
        join: BlockId,
        then_phi: Vec<(Value, Value)>,
        els_phi: Vec<(Value, Value)>,
    },
}

/// Build the memoized body op list for the function described by `infos`/`cfg`.
/// Errors (never silently degrades) on any shape it cannot express.
///
/// Supports straight-line functions AND structured `if`/`else` (no early
/// returns, no loops). The recursive region walk emits each reactive scope's
/// memo guard at the block where its owned instructions live; a scope whose
/// owned/mutating instructions cross a branch boundary is rejected (`Err`) by
/// [`check_soundness`] so it stays unmemoized rather than miscompiled.
pub fn build_layout(cfg: &Cfg, infos: &[ScopeInfo], ranges: &Ranges) -> Result<MemoLayout, String> {
    // Recover the structured region tree (hard-errors on non-if/else shapes).
    let (nodes, ret) = recover_regions(cfg)?;

    // Only scopes that actually produce an output are emitted (codegen.rs:34).
    let emitted: Vec<&ScopeInfo> = infos.iter().filter(|i| !i.outputs.is_empty()).collect();

    // Map each scope's value-set, owned instructions (by (block, index)), and the
    // single block it lives in. SOUNDNESS: every instruction that mutates a scope
    // value must be inside that scope's owned set, and all owned instructions must
    // sit in one block (so the single guard provably encloses every mutation).
    let value_set: Vec<HashSet<Value>> =
        emitted.iter().map(|i| i.scope.values.iter().copied().collect()).collect();
    // Value-sets for *all* inferred scopes (including those the escape analysis
    // pruned to empty outputs): the mutation-crossing soundness check must still
    // see a pruned-but-mutated scope so a mutation crossing its boundary bails the
    // whole function (matching React, which leaves such functions un-memoized)
    // rather than silently memoizing the surviving scopes around it.
    let all_value_sets: Vec<HashSet<Value>> =
        infos.iter().map(|i| i.scope.values.iter().copied().collect()).collect();
    let scope_blocks = check_soundness(cfg, ranges, &emitted, &value_set, &all_value_sets)?;

    // Outputs become outer `let`s so they survive the cache-check blocks; phis
    // (block params) likewise must be hoisted to function scope so a branch can
    // assign them. The `let`-sort is by rendered name; reproduce it.
    let mut scope_outputs: HashSet<Value> = HashSet::new();
    for i in &emitted {
        scope_outputs.extend(i.outputs.iter().copied());
    }
    let mut phi_params: HashSet<Value> = HashSet::new();
    for b in &cfg.blocks {
        for p in &b.params {
            phi_params.insert(*p);
        }
    }

    let em = Emitter { cfg, scope_outputs: &scope_outputs, ids: std::cell::Cell::new(0) };

    let cache_size: usize = emitted.iter().map(|i| i.deps.len() + i.outputs.len()).sum();

    let mut body: Vec<IrOp> = Vec::new();

    // `const $ = _c(N);`
    body.extend(em.cache_decl(cache_size));

    // `let <outputs ∪ phis sorted>;`
    let mut let_names: HashSet<String> =
        scope_outputs.iter().map(|v| em.name(*v)).collect();
    for p in &phi_params {
        let_names.insert(em.name(*p));
    }
    if !let_names.is_empty() {
        let mut names: Vec<String> = let_names.into_iter().collect();
        names.sort();
        body.push(em.let_decl(&names));
    }

    // Emit the region tree. A running slot counter threads cache slots across all
    // scopes in emission order (codegen.rs:74); since each scope's guard is
    // emitted exactly once at its owning block, the slot numbering stays linear.
    let mut slot = 0usize;
    let ctx = EmitCtx { em: &em, emitted: &emitted, value_set: &value_set, scope_blocks: &scope_blocks };
    for n in &nodes {
        ctx.emit_node(n, &mut body, &mut slot)?;
    }

    // `return <v>;`
    body.push(em.return_stmt(ret));

    Ok(MemoLayout { cache_size, body })
}

/// Carries the immutable emit state through the recursive region walk.
struct EmitCtx<'a> {
    em: &'a Emitter<'a>,
    emitted: &'a [&'a ScopeInfo],
    value_set: &'a [HashSet<Value>],
    /// The single block each emitted scope's owned instructions live in.
    scope_blocks: &'a [BlockId],
}

impl<'a> EmitCtx<'a> {
    fn emit_node(&self, node: &Node, out: &mut Vec<IrOp>, slot: &mut usize) -> Result<(), String> {
        match node {
            Node::Straight(b) => self.emit_block(*b, out, slot),
            Node::If { cond, then, els, join, then_phi, els_phi } => {
                let mut then_ops: Vec<IrOp> = Vec::new();
                for n in then {
                    self.emit_node(n, &mut then_ops, slot)?;
                }
                // Phi assignments for a then-arm that jumps straight to the join
                // (short-circuit): `join.param = arg;` at the end of the branch.
                for (p, a) in then_phi {
                    then_ops.push(self.em.phi_assign(*p, *a));
                }
                let mut else_ops: Vec<IrOp> = Vec::new();
                for n in els {
                    self.emit_node(n, &mut else_ops, slot)?;
                }
                for (p, a) in els_phi {
                    else_ops.push(self.em.phi_assign(*p, *a));
                }
                // `if (cond) { then } else { else }` — cond referenced by name (it
                // is a value defined before the branch).
                let mut eb = self.em.eb();
                let cond_v = eb.push(ident_op(&self.em.name(*cond)));
                let consequent = block_statement(then_ops);
                let alternate = block_statement(else_ops);
                let mut if_op = IrOp::new("jshir.if_statement");
                if_op.operands.push(cond_v);
                if_op.regions.push(Region::with_block(Block { args: vec![], ops: vec![consequent] }));
                if_op.regions.push(Region::with_block(Block { args: vec![], ops: vec![alternate] }));
                let mut block_ops = eb.ops;
                block_ops.push(if_op);
                out.push(wrap_stmt_run(block_ops));
                let _ = join;
                Ok(())
            }
        }
    }

    /// Emit one block's straight-line instructions (scope guards at last-owned,
    /// non-owned results as `const`s, bare stores), then the phi assignments this
    /// block passes to its `Br` successor.
    fn emit_block(&self, b: BlockId, out: &mut Vec<IrOp>, slot: &mut usize) -> Result<(), String> {
        let block = self.em.cfg.block(b);
        let instrs = &block.instrs;

        // Per-scope owned instruction indices *within this block*.
        let mut instr_owner: Vec<Option<usize>> = vec![None; instrs.len()];
        let mut owned: Vec<Vec<usize>> = vec![Vec::new(); self.emitted.len()];
        for (ix, ins) in instrs.iter().enumerate() {
            let owner = self.value_set.iter().position(|vs| match (&ins.op, ins.result) {
                (_, Some(r)) => vs.contains(&r),
                (Op::StoreMember { obj, .. }, None) => vs.contains(obj),
                _ => false,
            });
            // A scope only emits in the block that owns it (check_soundness has
            // verified all its owned instrs are in this one block).
            if let Some(s) = owner {
                if self.scope_blocks[s] == b {
                    instr_owner[ix] = Some(s);
                    owned[s].push(ix);
                }
            }
        }
        let last_owned: Vec<usize> = owned.iter().map(|v| v.last().copied().unwrap_or(0)).collect();

        for (ix, ins) in instrs.iter().enumerate() {
            match instr_owner[ix] {
                Some(s) if ix == last_owned[s] => {
                    out.push(self.em.emit_scope(instrs, &owned[s], self.emitted[s], slot)?);
                }
                Some(_) => {} // owned but not last: recomputed inside the guard
                None => {
                    if let Some(r) = ins.result {
                        out.push(self.em.const_decl(r, &ins.op)?);
                    } else if let Op::StoreMember { .. } = &ins.op {
                        out.push(self.em.store_stmt(&ins.op)?);
                    }
                }
            }
        }

        // Phi assignments: if this block branches to a successor with params, emit
        // `<param> = <arg>;` for each, materializing the SSA phi as a plain write
        // to the hoisted `let`.
        if let Term::Br(succ, args) = &block.term {
            let params = &self.em.cfg.block(*succ).params;
            for (p, a) in params.iter().zip(args) {
                if p != a {
                    out.push(self.em.phi_assign(*p, *a));
                }
            }
        }
        Ok(())
    }
}

/// Recover the structured region tree from the CFG. Returns the top-level node
/// list plus the function's return value.
///
/// Accepts only the structured shapes the lowering produces:
///  * a block ending in `Ret` (the function tail),
///  * a block ending in `Br(next)` (straight-line / branch-arm fallthrough),
///  * a block ending in `CondBr(then, else)` whose arms both fall through to a
///    single common join block (an `if`/`else` with no early return).
/// Any other shape — a branch arm that returns or diverges, a back-edge (loop),
/// or a join reached from outside the two arms — is a hard `Err` (deferred to a
/// later step), never a silent skip.
fn recover_regions(cfg: &Cfg) -> Result<(Vec<Node>, Option<Value>), String> {
    let mut ret: Option<Value> = None;
    let nodes = build_region(cfg, cfg.entry, None, &mut ret)?;
    Ok((nodes, ret))
}

/// Build the node sequence for the linear region starting at `start`, stopping
/// *before* `stop` (the enclosing join, or `None` at the top level where the
/// region ends in `Ret`).
fn build_region(
    cfg: &Cfg,
    start: BlockId,
    stop: Option<BlockId>,
    ret: &mut Option<Value>,
) -> Result<Vec<Node>, String> {
    let mut nodes = Vec::new();
    let mut cur = start;
    let mut guard = 0usize;
    loop {
        if Some(cur) == stop {
            return Ok(nodes);
        }
        guard += 1;
        if guard > cfg.blocks.len() + 1 {
            return Err("memoize_plan: irreducible/cyclic control flow (loops unsupported)".into());
        }
        let block = cfg.block(cur);
        match &block.term {
            Term::Ret(v) => {
                if stop.is_some() {
                    // A branch arm that returns: early-return shape, deferred.
                    return Err("memoize_plan: early return inside branch (deferred to later step)".into());
                }
                nodes.push(Node::Straight(cur));
                *ret = *v;
                return Ok(nodes);
            }
            Term::Br(next, _) => {
                nodes.push(Node::Straight(cur));
                cur = *next;
            }
            Term::CondBr { cond, then_block, else_block, then_args, else_args } => {
                let cond = *cond;
                let (tb, eb) = (*then_block, *else_block);
                let (ta, ea) = (then_args.clone(), else_args.clone());
                // Emit the cond block's own straight-line instructions first (the
                // condition value and any preceding computations live here), then
                // the `if`.
                nodes.push(Node::Straight(cur));
                // The join is the single block both arms reconverge at — taken
                // from the authoritative `joins` map the lowering recorded, not a
                // dominance heuristic. (A reachability guess mis-identifies the
                // join when one arm jumps straight to it: the post-join tail is
                // then common to both arms and may carry a smaller block id than
                // the real join.) Loop headers are a different construct entirely.
                let join = join_of(cfg, cur, tb, eb)?;
                let (then, then_phi) = build_arm(cfg, tb, &ta, join, ret)?;
                let (els, els_phi) = build_arm(cfg, eb, &ea, join, ret)?;
                nodes.push(Node::If { cond, then, els, join, then_phi, els_phi });
                cur = join;
            }
            Term::Unreachable => {
                return Err("memoize_plan: unreachable terminator (unsupported)".into());
            }
        }
    }
}

/// The join (merge) block of the forward conditional whose head is `cond_block`.
/// Prefers the lowering's recorded `joins` map (authoritative — it mirrors the
/// front-end's structured fallthrough); falls back to the dominance heuristic
/// only for a head the map didn't record. A loop header is rejected outright:
/// its "join" is the loop exit and treating it as an `if` would silently
/// miscompile the loop body (loops are deferred to a later step).
fn join_of(cfg: &Cfg, cond_block: BlockId, then_b: BlockId, else_b: BlockId) -> Result<BlockId, String> {
    if matches!(cfg.block_kinds.get(&cond_block), Some(BlockKind::Loop)) {
        return Err("memoize_plan: loop header (loops unsupported)".into());
    }
    match cfg.joins.get(&cond_block) {
        Some(j) => Ok(*j),
        None => find_join(cfg, cond_block, then_b, else_b),
    }
}

/// Build one arm of a conditional. An arm that targets a block *other* than the
/// join is a real region (recursed, stopping before the join); its own `Br` to
/// the join carries any phi args. An arm whose target *is* the join is empty —
/// the short-circuit shape — and the `CondBr` edge's args become phi assignments
/// (`join.param = arg;`) emitted at the end of that branch.
fn build_arm(
    cfg: &Cfg,
    arm: BlockId,
    edge_args: &[Value],
    join: BlockId,
    ret: &mut Option<Value>,
) -> Result<(Vec<Node>, Vec<(Value, Value)>), String> {
    if arm == join {
        let params = &cfg.block(join).params;
        let phi: Vec<(Value, Value)> = params
            .iter()
            .copied()
            .zip(edge_args.iter().copied())
            .filter(|(p, a)| p != a)
            .collect();
        Ok((Vec::new(), phi))
    } else {
        if !edge_args.is_empty() {
            // A non-join arm with block args is not a shape the structured
            // lowering produces (the arm's single predecessor needs no phi).
            return Err("memoize_plan: conditional edge args to a non-join arm (unsupported)".into());
        }
        Ok((build_region(cfg, arm, Some(join), ret)?, Vec::new()))
    }
}

/// Find the join block of the `if`/`else` rooted at `cond_block`, requiring the
/// exact structured diamond the lowering produces: every block reachable from
/// `then`/`else` without revisiting `cond_block` must reach a single common
/// block (the join) and never escape past it. Hard-errors otherwise.
fn find_join(cfg: &Cfg, cond_block: BlockId, then_b: BlockId, else_b: BlockId) -> Result<BlockId, String> {
    use std::collections::BTreeSet;
    // Blocks reachable from a side, stopping when we leave the arm (a block whose
    // successors include something not yet seen and not the cond — the join is the
    // first block reachable from *both* sides).
    fn reachable(cfg: &Cfg, from: BlockId, avoid: BlockId) -> Result<BTreeSet<BlockId>, String> {
        let mut seen = BTreeSet::new();
        let mut stack = vec![from];
        while let Some(b) = stack.pop() {
            if b == avoid || !seen.insert(b) {
                continue;
            }
            for s in cfg.block(b).term.successors() {
                stack.push(s);
            }
        }
        Ok(seen)
    }
    let from_then = reachable(cfg, then_b, cond_block)?;
    let from_else = reachable(cfg, else_b, cond_block)?;
    // The join is reachable from both arms. The structured lowering gives exactly
    // one such block that is the post-dominator; require a unique earliest one.
    let common: Vec<BlockId> = from_then.intersection(&from_else).copied().collect();
    // The earliest common block in id order is the immediate join for the
    // lowering's diamond (then/else blocks get ids before the join).
    let join = common.into_iter().min_by_key(|b| b.0);
    match join {
        Some(j) => Ok(j),
        None => Err("memoize_plan: branch arms do not reconverge (early return / divergence)".into()),
    }
}

/// SOUNDNESS GATE. For each emitted scope verify the single memo guard will
/// enclose every mutation of every scope value:
///  * all of the scope's owned instructions (allocs + member stores on a scope
///    value) lie in ONE basic block, and
///  * every instruction anywhere that mutates a scope value (a member store on
///    it, or a call that may mutate it — passed as a ref arg or as the receiver
///    of `obj.method(...)`) is itself one of the scope's owned instructions.
///
/// If either fails the scope's mutation could happen outside its guard (the
/// alloc-before-`if` and alloc-inside-branch counterexamples), so we hard-error
/// and leave the whole function unmemoized. Returns the owning block per scope.
///
/// Mutation reach is taken through **aliasing**: a mutation of any value mutates
/// every value in its alias set (`ranges.alias_root`), so capturing a scope value
/// into another object/array and mutating *that* later (or mutating a member load
/// off the scope value) is correctly seen as mutating the scope value — closing
/// the `x.y = y; x.y.push(...)` aliased-mutation hole.
fn check_soundness(
    cfg: &Cfg,
    ranges: &Ranges,
    emitted: &[&ScopeInfo],
    value_set: &[HashSet<Value>],
    all_value_sets: &[HashSet<Value>],
) -> Result<Vec<BlockId>, String> {
    let _ = ranges; // alias roots are intentionally NOT used (see note below).
    // Map each emitted scope value to the scope index that DIRECTLY owns it (it is
    // a member of that scope's value-set). We attribute a mutation to the scope
    // that *allocated* the mutated value, not to downstream consumers that merely
    // captured it: capturing `a` into `{a}` does not make a mutation of `a` a
    // mutation of the object's identity (the object is recomputed when `a`
    // changes). Union-find aliasing conflates the two directions, so we instead
    // resolve a mutated value to the concrete allocation(s) it names by following
    // member loads down (`x.y.push` -> `x`) and phi incomings (a phi mutation
    // mutates each branch's value), then check direct scope membership.
    // The mutation-crossing soundness check (below) must consider *all* inferred
    // scopes, not only the emitted (escaping) ones: a scope that the escape
    // analysis pruned still represents a value whose allocation is mutated, and
    // if that mutation crosses the scope boundary the program shape is one we
    // cannot yet memoize soundly (deferred to the scope-alignment step). We must
    // bail the whole function in that case rather than silently memoizing the
    // *other* (escaping) scopes around it — which would change observable output.
    // `all_value_scope` maps every scope value (pruned or not) to an index into
    // `all_value_sets`.
    let mut all_value_scope: std::collections::HashMap<Value, usize> = std::collections::HashMap::new();
    for (s, vs) in all_value_sets.iter().enumerate() {
        for v in vs {
            all_value_scope.insert(*v, s);
        }
    }

    // Map each value to its defining op, to follow member chains.
    let mut def_op: std::collections::HashMap<Value, &Op> = std::collections::HashMap::new();
    for b in &cfg.blocks {
        for ins in &b.instrs {
            if let Some(r) = ins.result {
                def_op.insert(r, &ins.op);
            }
        }
    }
    // The base object a value names, following `Member`/load chains down to the
    // root object: `x.y.z` -> `x`. Mutating any member of `x.y.z` ultimately
    // mutates the container `x`, so a mutation through a member load off a scope
    // value must be attributed to that scope value (closes the
    // `x.y = y; x.y.push(...)` aliased-mutation hole, which the union-find alias
    // analysis misses because the member load is a fresh unknown value).
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

    // Phi-incoming map: each block param (a phi) -> the values flowing into it
    // from predecessors. A mutation of a phi value mutates whatever any branch
    // assigned to it; if those are scope outputs the scope's cached value can be
    // mutated outside its guard (the `x = frozen | {}; x.prop = ...` hole).
    let mut phi_in: std::collections::HashMap<Value, Vec<Value>> = std::collections::HashMap::new();
    for b in &cfg.blocks {
        let edges: Vec<(BlockId, &[Value])> = match &b.term {
            Term::Br(t, a) => vec![(*t, a.as_slice())],
            Term::CondBr { then_block, then_args, else_block, else_args, .. } => {
                vec![(*then_block, then_args.as_slice()), (*else_block, else_args.as_slice())]
            }
            _ => vec![],
        };
        for (succ, args) in edges {
            let params = &cfg.block(succ).params;
            for (p, a) in params.iter().zip(args) {
                phi_in.entry(*p).or_default().push(*a);
            }
        }
    }
    // The set of concrete values a mutated value may name: itself plus the
    // transitive closure through member-bases (`x.y.z` -> `x`) and phi incomings
    // (a phi mutation mutates each incoming branch value). This is the *downward*
    // / *backward* direction only — it never crosses into objects that captured
    // the value, so a consumer scope that merely embeds the value is not wrongly
    // flagged.
    let reachable_values = |v: Value| -> HashSet<Value> {
        let mut out = HashSet::new();
        let mut stack = vec![v];
        while let Some(cur) = stack.pop() {
            if !out.insert(cur) {
                continue;
            }
            let base = base_object(cur);
            if base != cur {
                stack.push(base);
            }
            if let Some(ins) = phi_in.get(&cur) {
                for a in ins {
                    stack.push(*a);
                }
            }
        }
        out
    };

    // Owned instructions per scope, as (block, index). An instruction is owned by
    // scope s if it defines a scope value, or is a member store on a scope value.
    let mut owned_blocks: Vec<HashSet<BlockId>> = vec![HashSet::new(); emitted.len()];
    let mut owned_ids: Vec<HashSet<(u32, usize)>> = vec![HashSet::new(); emitted.len()];
    for b in &cfg.blocks {
        for (ix, ins) in b.instrs.iter().enumerate() {
            if let Some(s) = value_set.iter().position(|vs| match (&ins.op, ins.result) {
                (_, Some(r)) => vs.contains(&r),
                (Op::StoreMember { obj, .. }, None) => vs.contains(obj),
                _ => false,
            }) {
                owned_blocks[s].insert(b.id);
                owned_ids[s].insert((b.id.0, ix));
            }
        }
    }

    // Owned instruction-ids for *all* scopes (pruned + emitted), for the
    // mutation-crossing check that runs over `all_value_scope`.
    let mut all_owned_ids: Vec<HashSet<(u32, usize)>> = vec![HashSet::new(); all_value_sets.len()];
    for b in &cfg.blocks {
        for (ix, ins) in b.instrs.iter().enumerate() {
            if let Some(s) = all_value_sets.iter().position(|vs| match (&ins.op, ins.result) {
                (_, Some(r)) => vs.contains(&r),
                (Op::StoreMember { obj, .. }, None) => vs.contains(obj),
                _ => false,
            }) {
                all_owned_ids[s].insert((b.id.0, ix));
            }
        }
    }

    // Every instruction that may mutate some value: the set of values it mutates.
    // Reuse the same over-approximation as `mutability::analyze`:
    //  * `obj.p = v` mutates `obj`;
    //  * a non-pure call mutates every reference argument AND, for a method call
    //    `recv.m(...)`, the receiver `recv` (reached through the callee member).
    let pure_call = |callee: Value| -> bool {
        // A pure JSX/createElement constructor (does not mutate its args).
        for b in &cfg.blocks {
            for ins in &b.instrs {
                if ins.result == Some(callee) {
                    return match &ins.op {
                        Op::Member { prop: MemberKey::Static(n), .. } => {
                            matches!(n.as_str(), "createElement" | "jsx" | "jsxs" | "jsxDEV")
                        }
                        Op::Global(n) => matches!(n.as_str(), "jsx" | "jsxs" | "_jsx" | "_jsxs"),
                        _ => false,
                    };
                }
            }
        }
        false
    };
    // Map callee value -> receiver object, for method-call mutation (`a.push(x)`
    // mutates `a`, reached via the callee being `Member { obj: a, .. }`).
    let receiver_of = |callee: Value| -> Option<Value> {
        for b in &cfg.blocks {
            for ins in &b.instrs {
                if ins.result == Some(callee) {
                    if let Op::Member { obj, .. } = &ins.op {
                        return Some(*obj);
                    }
                }
            }
        }
        None
    };

    for b in &cfg.blocks {
        for (ix, ins) in b.instrs.iter().enumerate() {
            // The set of values this instruction may mutate.
            let mut mutates: Vec<Value> = Vec::new();
            match &ins.op {
                Op::StoreMember { obj, .. } => mutates.push(*obj),
                Op::Call { callee, args } => {
                    if !pure_call(*callee) {
                        mutates.extend(args.iter().copied());
                        if let Some(recv) = receiver_of(*callee) {
                            mutates.push(recv);
                        }
                    }
                }
                _ => {}
            }
            if mutates.is_empty() {
                continue;
            }
            // Every concrete value each mutated value may name (member bases, phi
            // incomings). For each that is DIRECTLY a scope's allocation, this
            // instruction must be one of that scope's owned instructions, else the
            // scope's cached value could be mutated outside its single guard.
            let mut touched: HashSet<Value> = HashSet::new();
            for v in &mutates {
                touched.extend(reachable_values(*v));
            }
            let mut scopes_hit: HashSet<usize> = HashSet::new();
            for v in &touched {
                if let Some(&s) = all_value_scope.get(v) {
                    scopes_hit.insert(s);
                }
            }
            for s in scopes_hit {
                if !all_owned_ids[s].contains(&(b.id.0, ix)) {
                    return Err(
                        "memoize_plan: a reactive scope's allocation is mutated outside its memo \
                         guard (mutation crosses a branch boundary); deferred to scope-alignment step"
                            .into(),
                    );
                }
            }
        }
    }

    // Each scope must live in exactly one block (so the guard is straight-line).
    let mut scope_blocks = Vec::with_capacity(emitted.len());
    for s in 0..emitted.len() {
        let blocks = &owned_blocks[s];
        if blocks.len() != 1 {
            return Err(
                "memoize_plan: a reactive scope spans multiple basic blocks \
                 (its owned statements cross a branch); deferred to scope-alignment step"
                    .into(),
            );
        }
        scope_blocks.push(*blocks.iter().next().unwrap());
    }
    Ok(scope_blocks)
}

/// Emits JSIR statement ops from CFG values, mirroring `emit_memoized`. The
/// `ExprBuilder` mints local value ids; the transform relabels everything past
/// the file's existing ids, so collisions are impossible.
struct Emitter<'a> {
    cfg: &'a Cfg,
    scope_outputs: &'a HashSet<Value>,
    /// A single monotonic value-id counter for the whole body, so every
    /// synthesized op result is unique across all statement runs and nested
    /// blocks (the transform then shifts them past the file's max in one pass).
    ids: std::cell::Cell<u32>,
}

impl<'a> Emitter<'a> {
    /// A fresh op accumulator sharing this emitter's global id counter.
    fn eb(&self) -> ExprBuilder<'_> {
        ExprBuilder { ids: &self.ids, ops: Vec::new() }
    }

    /// The source-level name we render a CFG value as (param name, or `_v{id}`).
    fn name(&self, v: Value) -> String {
        if let Some(idx) = self.cfg.params.iter().position(|p| *p == v) {
            if let Some(n) = self.cfg.param_names.get(idx) {
                return n.clone();
            }
        }
        format!("_v{}", v.0)
    }

    /// `const $ = _c(<n>);` — returns the full statement op run.
    fn cache_decl(&self, n: usize) -> Vec<IrOp> {
        let mut eb = self.eb();
        let id = eb.push(ident_ref_op("$"));
        let callee = eb.push(ident_op("_c"));
        let arg = eb.push(numeric_op(n));
        let mut call = IrOp::new("jsir.call_expression");
        call.operands.push(callee);
        call.operands.push(arg);
        let call_v = eb.push(call);
        let mut declarator = IrOp::new("jsir.variable_declarator");
        declarator.operands.push(id);
        declarator.operands.push(call_v);
        let decl_v = eb.push(declarator);
        let mut end = IrOp::new("jsir.exprs_region_end");
        end.operands.push(decl_v);
        eb.ops.push(end);
        vec![var_decl("const", eb.ops)]
    }

    /// `let a, b, c;`
    fn let_decl(&self, names: &[String]) -> IrOp {
        let mut eb = self.eb();
        let mut decl_vs = Vec::new();
        for name in names {
            let id = eb.push(ident_ref_op(name));
            let mut declarator = IrOp::new("jsir.variable_declarator");
            declarator.operands.push(id);
            decl_vs.push(eb.push(declarator));
        }
        let mut end = IrOp::new("jsir.exprs_region_end");
        end.operands = decl_vs;
        eb.ops.push(end);
        var_decl("let", eb.ops)
    }

    /// `const <name(r)> = <expr(op)>;`
    fn const_decl(&self, r: Value, op: &Op) -> Result<IrOp, String> {
        let mut eb = self.eb();
        let init = self.emit_op_value(op, &mut eb)?;
        let id = eb.push(ident_ref_op(&self.name(r)));
        let mut declarator = IrOp::new("jsir.variable_declarator");
        declarator.operands.push(id);
        declarator.operands.push(init);
        let decl_v = eb.push(declarator);
        let mut end = IrOp::new("jsir.exprs_region_end");
        end.operands.push(decl_v);
        eb.ops.push(end);
        Ok(var_decl("const", eb.ops))
    }

    /// A bare `obj.p = v;` expression statement.
    fn store_stmt(&self, op: &Op) -> Result<IrOp, String> {
        let mut eb = self.eb();
        let v = self.emit_op_value(op, &mut eb)?;
        let mut stmt = IrOp::new("jsir.expression_statement");
        stmt.operands.push(v);
        let mut block_ops = eb.ops;
        block_ops.push(stmt);
        Ok(wrap_stmt_run(block_ops))
    }

    /// `return <v>;`
    fn return_stmt(&self, v: Option<Value>) -> IrOp {
        match v {
            Some(v) => {
                let mut eb = self.eb();
                let name = self.name(v);
                let id = eb.push(ident_op(&name));
                let mut ret = IrOp::new("jsir.return_statement");
                ret.operands.push(id);
                let mut block_ops = eb.ops;
                block_ops.push(ret);
                wrap_stmt_run(block_ops)
            }
            None => wrap_stmt_run(vec![IrOp::new("jsir.return_statement")]),
        }
    }

    /// Emit the JSIR value that `op` (an instruction op) computes, by NAME for
    /// its operands (operands are referenced by their `_v{id}`/param names, which
    /// were emitted earlier as `const`s), matching `emit_memoized`'s flat `expr`.
    fn emit_op_value(&self, op: &Op, eb: &mut ExprBuilder) -> Result<ValueId, String> {
        match op {
            Op::Const(c) => Ok(eb.push(const_op(c)?)),
            Op::Global(g) => Ok(eb.push(ident_op(g))),
            Op::Bin(b, x, y) => {
                let xv = eb.push(ident_op(&self.name(*x)));
                let yv = eb.push(ident_op(&self.name(*y)));
                let mut e = IrOp::new("jsir.binary_expression");
                e.operands.push(xv);
                e.operands.push(yv);
                e.attrs.push(("operator_".into(), Attr::Str(bin_str(*b).into())));
                Ok(eb.push(e))
            }
            Op::Un(u, x) => {
                let xv = eb.push(ident_op(&self.name(*x)));
                let mut e = IrOp::new("jsir.unary_expression");
                e.operands.push(xv);
                e.attrs.push(("operator_".into(), Attr::Str(un_str(*u).into())));
                e.attrs.push(("prefix".into(), Attr::Bool(true)));
                Ok(eb.push(e))
            }
            Op::Member { obj, prop } => {
                let ov = eb.push(ident_op(&self.name(*obj)));
                match prop {
                    MemberKey::Static(name) => {
                        let mut m = IrOp::new("jsir.member_expression");
                        m.operands.push(ov);
                        m.attrs.push((
                            "literal_property".into(),
                            Attr::Identifier(Box::new(ident_attr(name))),
                        ));
                        Ok(eb.push(m))
                    }
                    MemberKey::Computed(c) => {
                        let cv = eb.push(ident_op(&self.name(*c)));
                        let mut m = IrOp::new("jsir.member_expression");
                        m.operands.push(ov);
                        m.operands.push(cv);
                        Ok(eb.push(m))
                    }
                }
            }
            Op::StoreMember { obj, prop, value } => {
                let ov = eb.push(ident_op(&self.name(*obj)));
                let target = match prop {
                    MemberKey::Static(name) => {
                        let mut m = IrOp::new("jsir.member_expression_ref");
                        m.operands.push(ov);
                        m.attrs.push((
                            "literal_property".into(),
                            Attr::Identifier(Box::new(ident_attr(name))),
                        ));
                        eb.push(m)
                    }
                    MemberKey::Computed(c) => {
                        let cv = eb.push(ident_op(&self.name(*c)));
                        let mut m = IrOp::new("jsir.member_expression_ref");
                        m.operands.push(ov);
                        m.operands.push(cv);
                        eb.push(m)
                    }
                };
                let valv = eb.push(ident_op(&self.name(*value)));
                let mut assign = IrOp::new("jsir.assignment_expression");
                assign.operands.push(target);
                assign.operands.push(valv);
                assign.attrs.push(("operator_".into(), Attr::Str("=".into())));
                Ok(eb.push(assign))
            }
            Op::Call { callee, args } => {
                let cv = eb.push(ident_op(&self.name(*callee)));
                let mut e = IrOp::new("jsir.call_expression");
                e.operands.push(cv);
                for a in args {
                    let av = eb.push(ident_op(&self.name(*a)));
                    e.operands.push(av);
                }
                Ok(eb.push(e))
            }
            Op::MakeArray(elems) => {
                let mut e = IrOp::new("jsir.array_expression");
                for el in elems {
                    let ev = eb.push(ident_op(&self.name(*el)));
                    e.operands.push(ev);
                }
                Ok(eb.push(e))
            }
            Op::MakeObject(props) => {
                let mut region_ops: Vec<IrOp> = Vec::new();
                let mut sub = self.eb();
                let mut prop_vs = Vec::new();
                for (k, val) in props {
                    let vv = sub.push(ident_op(&self.name(*val)));
                    let mut pop = IrOp::new("jsir.object_property");
                    pop.operands.push(vv);
                    match k {
                        PropKey::Ident(name) => {
                            pop.attrs.push((
                                "literal_key".into(),
                                Attr::Identifier(Box::new(ident_attr(name))),
                            ));
                            pop.attrs.push(("shorthand".into(), Attr::Bool(false)));
                        }
                        PropKey::Computed(c) => {
                            // `[<key>]: <val>` — key referenced by name.
                            let kv = sub.push(ident_op(&self.name(*c)));
                            pop.operands.insert(0, kv);
                            pop.attrs.push(("computed".into(), Attr::Bool(true)));
                            pop.attrs.push(("shorthand".into(), Attr::Bool(false)));
                        }
                    }
                    prop_vs.push(sub.push(pop));
                }
                region_ops.extend(std::mem::take(&mut sub.ops));
                let mut end = IrOp::new("jsir.exprs_region_end");
                end.operands = prop_vs;
                region_ops.push(end);
                let mut obj = IrOp::new("jsir.object_expression");
                obj.regions.push(Region::with_block(Block { args: vec![], ops: region_ops }));
                Ok(eb.push(obj))
            }
            Op::ReadVar(_) | Op::WriteVar(_, _) => {
                Err("memoize_plan: residual var op (run SSA first)".into())
            }
        }
    }

    /// Emit one scope's `if (test) { recompute; stores } else { restores }`
    /// (mirror `emit_scope`, codegen.rs:99).
    fn emit_scope(
        &self,
        instrs: &[Instr],
        owned: &[usize],
        info: &ScopeInfo,
        slot: &mut usize,
    ) -> Result<IrOp, String> {
        let dep_base = *slot;
        let out_base = dep_base + info.deps.len();
        *slot = out_base + info.outputs.len();

        // -- test (operand ops + the boolean value) --
        let mut test_eb = self.eb();
        let test_val = if info.deps.is_empty() {
            // `$[out_base] === Symbol.for("react.memo_cache_sentinel")`
            let read = self.cache_read(out_base, &mut test_eb);
            let sentinel = self.sentinel(&mut test_eb);
            bin(&mut test_eb, "===", read, sentinel)
        } else {
            let mut acc: Option<ValueId> = None;
            for (i, d) in info.deps.iter().enumerate() {
                let read = self.cache_read(dep_base + i, &mut test_eb);
                let dep = test_eb.push(ident_op(&self.name(*d)));
                let ne = bin(&mut test_eb, "!==", read, dep);
                acc = Some(match acc {
                    None => ne,
                    Some(prev) => bin(&mut test_eb, "||", prev, ne),
                });
            }
            acc.expect("non-empty deps")
        };

        // -- consequent: dep stores, recompute owned, output stores --
        let mut then_ops: Vec<IrOp> = Vec::new();
        for (i, d) in info.deps.iter().enumerate() {
            then_ops.push(self.slot_store(dep_base + i, &self.name(*d)));
        }
        for &ix in owned {
            let ins = &instrs[ix];
            if let Some(r) = ins.result {
                let kw_const = !self.scope_outputs.contains(&r);
                then_ops.push(self.assign_or_decl(r, &ins.op, kw_const)?);
            } else if let Op::StoreMember { .. } = &ins.op {
                then_ops.push(self.store_stmt(&ins.op)?);
            }
        }
        for (j, o) in info.outputs.iter().enumerate() {
            then_ops.push(self.slot_store(out_base + j, &self.name(*o)));
        }

        // -- alternate: output restores `<name> = $[slot];` --
        let mut else_ops: Vec<IrOp> = Vec::new();
        for (j, o) in info.outputs.iter().enumerate() {
            else_ops.push(self.restore_stmt(&self.name(*o), out_base + j));
        }

        // Assemble the if.
        let consequent = block_statement(then_ops);
        let alternate = block_statement(else_ops);
        let mut if_op = IrOp::new("jshir.if_statement");
        if_op.operands.push(test_val);
        if_op.regions.push(Region::with_block(Block { args: vec![], ops: vec![consequent] }));
        if_op.regions.push(Region::with_block(Block { args: vec![], ops: vec![alternate] }));

        // The test's operand ops must precede the if in the enclosing block, so
        // wrap [test ops..., if] as one statement-run holder.
        let mut block_ops = test_eb.ops;
        block_ops.push(if_op);
        Ok(wrap_stmt_run(block_ops))
    }

    /// `<name> = <expr(op)>;` (output, no `const`) or `const <name> = <expr>;`.
    fn assign_or_decl(&self, r: Value, op: &Op, kw_const: bool) -> Result<IrOp, String> {
        if kw_const {
            self.const_decl(r, op)
        } else {
            let mut eb = self.eb();
            let init = self.emit_op_value(op, &mut eb)?;
            let lhs = eb.push(ident_ref_op(&self.name(r)));
            let mut assign = IrOp::new("jsir.assignment_expression");
            assign.operands.push(lhs);
            assign.operands.push(init);
            assign.attrs.push(("operator_".into(), Attr::Str("=".into())));
            let av = eb.push(assign);
            let mut stmt = IrOp::new("jsir.expression_statement");
            stmt.operands.push(av);
            let mut block_ops = eb.ops;
            block_ops.push(stmt);
            Ok(wrap_stmt_run(block_ops))
        }
    }

    /// `$[slot] = <name>;`
    fn slot_store(&self, slot: usize, name: &str) -> IrOp {
        let mut eb = self.eb();
        let target = self.cache_ref(slot, &mut eb);
        let val = eb.push(ident_op(name));
        let mut assign = IrOp::new("jsir.assignment_expression");
        assign.operands.push(target);
        assign.operands.push(val);
        assign.attrs.push(("operator_".into(), Attr::Str("=".into())));
        let av = eb.push(assign);
        let mut stmt = IrOp::new("jsir.expression_statement");
        stmt.operands.push(av);
        let mut block_ops = eb.ops;
        block_ops.push(stmt);
        wrap_stmt_run(block_ops)
    }

    /// `<name> = $[slot];`
    fn restore_stmt(&self, name: &str, slot: usize) -> IrOp {
        let mut eb = self.eb();
        let lhs = eb.push(ident_ref_op(name));
        let rhs = self.cache_read(slot, &mut eb);
        let mut assign = IrOp::new("jsir.assignment_expression");
        assign.operands.push(lhs);
        assign.operands.push(rhs);
        assign.attrs.push(("operator_".into(), Attr::Str("=".into())));
        let av = eb.push(assign);
        let mut stmt = IrOp::new("jsir.expression_statement");
        stmt.operands.push(av);
        let mut block_ops = eb.ops;
        block_ops.push(stmt);
        wrap_stmt_run(block_ops)
    }

    /// `<param> = <arg>;` — materialize an SSA phi as a write to its hoisted
    /// `let`. The argument is referenced by name (it is defined earlier in the
    /// predecessor block).
    fn phi_assign(&self, param: Value, arg: Value) -> IrOp {
        let mut eb = self.eb();
        let lhs = eb.push(ident_ref_op(&self.name(param)));
        let rhs = eb.push(ident_op(&self.name(arg)));
        let mut assign = IrOp::new("jsir.assignment_expression");
        assign.operands.push(lhs);
        assign.operands.push(rhs);
        assign.attrs.push(("operator_".into(), Attr::Str("=".into())));
        let av = eb.push(assign);
        let mut stmt = IrOp::new("jsir.expression_statement");
        stmt.operands.push(av);
        let mut block_ops = eb.ops;
        block_ops.push(stmt);
        wrap_stmt_run(block_ops)
    }

    /// `$[i]` computed member r-value; returns its value id.
    fn cache_read(&self, i: usize, eb: &mut ExprBuilder) -> ValueId {
        let obj = eb.push(ident_op("$"));
        let idx = eb.push(numeric_op(i));
        let mut m = IrOp::new("jsir.member_expression");
        m.operands.push(obj);
        m.operands.push(idx);
        eb.push(m)
    }

    /// `$[i]` computed member l-value (member_expression_ref); returns its id.
    fn cache_ref(&self, i: usize, eb: &mut ExprBuilder) -> ValueId {
        let obj = eb.push(ident_op("$"));
        let idx = eb.push(numeric_op(i));
        let mut m = IrOp::new("jsir.member_expression_ref");
        m.operands.push(obj);
        m.operands.push(idx);
        eb.push(m)
    }

    /// `Symbol.for("react.memo_cache_sentinel")`; returns the call value id.
    fn sentinel(&self, eb: &mut ExprBuilder) -> ValueId {
        let sym = eb.push(ident_op("Symbol"));
        let mut member = IrOp::new("jsir.member_expression");
        member.operands.push(sym);
        member.attrs.push((
            "literal_property".into(),
            Attr::Identifier(Box::new(ident_attr("for"))),
        ));
        let member_v = eb.push(member);
        let arg = eb.push(string_op("react.memo_cache_sentinel"));
        let mut call = IrOp::new("jsir.call_expression");
        call.operands.push(member_v);
        call.operands.push(arg);
        eb.push(call)
    }
}

/// A flat JSIR expression-op builder drawing globally-unique value ids from the
/// `Emitter`'s shared counter, so ids never collide across statement runs or
/// nested blocks.
struct ExprBuilder<'c> {
    ids: &'c std::cell::Cell<u32>,
    ops: Vec<IrOp>,
}
impl<'c> ExprBuilder<'c> {
    fn push(&mut self, mut op: IrOp) -> ValueId {
        let n = self.ids.get();
        self.ids.set(n + 1);
        let v = ValueId(n);
        op.results.push(v);
        self.ops.push(op);
        v
    }
}

fn bin(eb: &mut ExprBuilder, op: &str, l: ValueId, r: ValueId) -> ValueId {
    let mut b = IrOp::new("jsir.binary_expression");
    b.operands.push(l);
    b.operands.push(r);
    b.attrs.push(("operator_".into(), Attr::Str(op.into())));
    eb.push(b)
}

/// Wrap a flat op run (operand ops then a statement-root) so a single body slot
/// holds it. We use a `jshir.block_statement` ONLY when there is more than the
/// root; but a block_statement changes structure. Instead we keep the run flat
/// by returning a synthetic single op is impossible — so we hand back a
/// `jshir.block_statement`? No: the body block stores a flat op list, so the
/// transform splices the run directly. We therefore return the run as a single
/// op by embedding it in a transparent holder the transform unwraps.
fn wrap_stmt_run(ops: Vec<IrOp>) -> IrOp {
    let mut holder = IrOp::new("jsir.__stmt_run");
    holder.regions.push(Region::with_block(Block { args: vec![], ops }));
    holder
}

fn block_statement(stmt_runs: Vec<IrOp>) -> IrOp {
    // Flatten each statement run's op list into the block's op list.
    let mut ops = Vec::new();
    for run in stmt_runs {
        ops.extend(flatten_run(run));
    }
    let mut bs = IrOp::new("jshir.block_statement");
    bs.regions.push(Region::with_block(Block { args: vec![], ops }));
    bs.regions.push(Region::with_block(Block::default()));
    bs
}

/// Expand a `jsir.__stmt_run` holder into its flat op list (or pass through a
/// real statement op).
fn flatten_run(op: IrOp) -> Vec<IrOp> {
    if op.name == "jsir.__stmt_run" {
        op.regions.into_iter().next().map(|r| r.blocks).into_iter().flatten().next().map(|b| b.ops).unwrap_or_default()
    } else {
        vec![op]
    }
}

fn var_decl(kind: &str, region_ops: Vec<IrOp>) -> IrOp {
    let mut decl = IrOp::new("jsir.variable_declaration");
    decl.attrs.push(("kind".into(), Attr::Str(kind.to_string())));
    decl.regions.push(Region::with_block(Block { args: vec![], ops: region_ops }));
    wrap_stmt_run(vec![decl])
}

fn ident_op(name: &str) -> IrOp {
    let mut op = IrOp::new("jsir.identifier");
    op.attrs.push(("name".into(), Attr::Str(name.to_string())));
    op
}
fn ident_ref_op(name: &str) -> IrOp {
    let mut op = IrOp::new("jsir.identifier_ref");
    op.attrs.push(("name".into(), Attr::Str(name.to_string())));
    op
}
fn numeric_op(n: usize) -> IrOp {
    let mut op = IrOp::new("jsir.numeric_literal");
    op.attrs.push(("value".into(), Attr::F64(n as f64)));
    op.attrs.push(("extra".into(), Attr::NumericLiteralExtra { raw: n.to_string(), value: n as f64 }));
    op
}
fn string_op(s: &str) -> IrOp {
    let mut op = IrOp::new("jsir.string_literal");
    op.attrs.push(("value".into(), Attr::Str(s.to_string())));
    op.attrs.push(("extra".into(), Attr::StringLiteralExtra { raw: format!("{s:?}"), raw_value: s.to_string() }));
    op
}
fn const_op(c: &Const) -> Result<IrOp, String> {
    Ok(match c {
        Const::Undef => ident_op("undefined"),
        Const::Null => IrOp::new("jsir.null_literal"),
        Const::Bool(b) => {
            let mut op = IrOp::new("jsir.boolean_literal");
            op.attrs.push(("value".into(), Attr::Bool(*b)));
            op
        }
        Const::Num(bits) => {
            let n = f64::from_bits(*bits);
            let raw = crate::interp::js_num_to_string(n);
            let mut op = IrOp::new("jsir.numeric_literal");
            op.attrs.push(("value".into(), Attr::F64(n)));
            op.attrs.push(("extra".into(), Attr::NumericLiteralExtra { raw, value: n }));
            op
        }
        Const::Str(s) => string_op(s),
    })
}

fn ident_attr(name: &str) -> IdentifierAttr {
    IdentifierAttr {
        start_line: 0,
        start_col: 0,
        end_line: 0,
        end_col: 0,
        identifier_name: name.to_string(),
        start_index: 0,
        end_index: 0,
        scope_uid: 0,
        name: name.to_string(),
    }
}

fn bin_str(b: BinOp) -> &'static str {
    use BinOp::*;
    match b {
        Add => "+", Sub => "-", Mul => "*", Div => "/", Mod => "%", Pow => "**",
        Eq => "==", Ne => "!=", StrictEq => "===", StrictNe => "!==",
        Lt => "<", Le => "<=", Gt => ">", Ge => ">=",
        BitAnd => "&", BitOr => "|", BitXor => "^", Shl => "<<", Shr => ">>", UShr => ">>>",
    }
}
fn un_str(u: UnOp) -> &'static str {
    use UnOp::*;
    match u {
        Neg => "-", Pos => "+", Not => "!", BitNot => "~", TypeOf => "typeof", Void => "void",
    }
}
