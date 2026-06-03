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

use jsir_ir::{
    Attr, Block, IdentifierAttr, ImportSpecKind, ImportSpecifierAttr, Op as IrOp, Region,
    StringLiteralKeyAttr, ValueId,
};
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

    let em = Emitter { cfg, scope_outputs: &scope_outputs, ids: std::cell::Cell::new(0), names: std::collections::HashMap::new() };

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
    // `ranges` is consulted for the control-dependence bail below; alias roots are
    // intentionally NOT used (the precise model groups by range overlap).
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

    // When the two-phase aliasing model is enabled, derive per-instruction
    // mutations from the SAME effect stream the ranges use (the single source of
    // truth) instead of the coarse any-non-pure-call-mutates-all-args heuristic.
    // This keeps the branch-boundary check consistent with the freeze-aware
    // ranges (e.g. `useMemo`/`useCallback` freeze their args, so the hook call is
    // not a mutation of them). The legacy heuristic stays the default.
    let effect_muts = crate::aliasing_ranges::mutations_by_instr(cfg);
    let _ = (&pure_call, &receiver_of); // superseded by effect-derived mutations
    for b in &cfg.blocks {
        for (ix, _ins) in b.instrs.iter().enumerate() {
            // The set of values this instruction may mutate, from the same effect
            // stream the ranges use (single source of truth).
            let mut mutates: Vec<Value> = Vec::new();
            if let Some(vs) = effect_muts.get(&(b.id, ix)) {
                mutates.extend(vs.iter().copied());
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

    // Control-dependence soundness: a branch whose condition transitively reads a
    // *mutated* allocation makes every value selected by that branch (the join
    // phis, and any scope depending on them) control-dependent on mutable state we
    // don't surface as a reactive dependency. Memoizing such a scope as a
    // once-cache would return a stale value when the mutation's inputs change. The
    // over-approximating union-find caught this by conflating the mutated object
    // with the branch's values; the precise model does not, so we bail explicitly.
    // (Only matters when there is something to memoize; a no-scope function is a
    // pass-through regardless.)
    if !emitted.is_empty() {
        for b in &cfg.blocks {
            if let crate::cfg::Term::CondBr { cond, .. } = &b.term {
                if reachable_values(*cond).iter().any(|v| ranges.is_mutable_after_def(*v)) {
                    return Err(
                        "memoize_plan: a branch condition reads a mutated allocation \
                         (control-dependent on untracked mutable state); deferred to \
                         scope-alignment step"
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
    /// value -> source name, recovered from declarations (in-place path only;
    /// empty for the structured `build_layout` path which renders `_vN`).
    names: std::collections::HashMap<Value, String>,
}

impl<'a> Emitter<'a> {
    /// A fresh op accumulator sharing this emitter's global id counter.
    fn eb(&self) -> ExprBuilder<'_> {
        ExprBuilder { ids: &self.ids, ops: Vec::new() }
    }

    /// The source-level name we render a CFG value as (param name, or `_v{id}`).
    fn name(&self, v: Value) -> String {
        // A recovered source name (declared local or assigned temp) wins, so
        // resynthesized expressions reference real names; falls back to params
        // then a synthesized `_vN`.
        if let Some(n) = self.names.get(&v) {
            return n.clone();
        }
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
    /// The defining op of a CFG value, if it is produced by an instruction.
    fn def_op(&self, v: Value) -> Option<&'a Op> {
        for b in &self.cfg.blocks {
            for ins in &b.instrs {
                if ins.result == Some(v) {
                    return Some(&ins.op);
                }
            }
        }
        None
    }

    /// Emit a `recv.prop` / `recv[expr]` member expression value.
    fn emit_member_value(&self, obj: Value, prop: &MemberKey, eb: &mut ExprBuilder) -> Result<ValueId, String> {
        let ov = eb.push(ident_op(&self.name(obj)));
        match prop {
            MemberKey::Static(name) => {
                let mut m = IrOp::new("jsir.member_expression");
                m.operands.push(ov);
                m.attrs.push(("literal_property".into(), Attr::Identifier(Box::new(ident_attr(name)))));
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
            Op::Member { obj, prop } => self.emit_member_value(*obj, prop, eb),
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
                // If the callee is a method member load (`recv.m`), emit a bound
                // method call `recv.m(args)` rather than detaching it through a
                // temp (`const m = recv.m; m(args)` loses the `this` receiver).
                // The member's own temp (if it is also used as a dep key) is still
                // emitted separately; this only changes the call's callee.
                let cv = match self.def_op(*callee) {
                    Some(Op::Member { obj, prop }) => self.emit_member_value(*obj, prop, eb)?,
                    _ => eb.push(ident_op(&self.name(*callee))),
                };
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

    // === in-place memoization (PROTOTYPE) ================================
    // These reuse the guard machinery above but render dependencies and the
    // recomputed value from the ORIGINAL source (names + initializer op), so the
    // surrounding statements (loops, etc.) can be kept verbatim.

    /// Reconstruct a source expression for `v` from the CFG: a parameter renders
    /// as its source name, a member chain recurses, a constant/global renders
    /// directly. Errors on anything else (the prototype's bound).
    fn emit_real_expr(&self, v: Value, eb: &mut ExprBuilder) -> Result<ValueId, String> {
        // A recovered source name (a parameter or a declared local) renders as the
        // bare identifier — this is what lets a dependency like `x` or `props`
        // print correctly instead of a synthesized temp.
        if let Some(name) = self.names.get(&v) {
            return Ok(eb.push(ident_op(name)));
        }
        if let Some(idx) = self.cfg.params.iter().position(|p| *p == v) {
            let name = self.cfg.param_names.get(idx).cloned().unwrap_or_else(|| format!("_p{}", v.0));
            return Ok(eb.push(ident_op(&name)));
        }
        let op = self.def_op_of(v).ok_or_else(|| format!("inplace: dep %{} is not a param and has no def", v.0))?;
        match op {
            Op::Member { obj, prop } => {
                let ov = self.emit_real_expr(*obj, eb)?;
                match prop {
                    MemberKey::Static(name) => {
                        let mut m = IrOp::new("jsir.member_expression");
                        m.operands.push(ov);
                        m.attrs.push(("literal_property".into(), Attr::Identifier(Box::new(ident_attr(name)))));
                        Ok(eb.push(m))
                    }
                    MemberKey::Computed(c) => {
                        let cv = self.emit_real_expr(*c, eb)?;
                        let mut m = IrOp::new("jsir.member_expression");
                        m.operands.push(ov);
                        m.operands.push(cv);
                        Ok(eb.push(m))
                    }
                }
            }
            Op::Const(c) => Ok(eb.push(const_op(c)?)),
            Op::Global(g) => Ok(eb.push(ident_op(g))),
            Op::Bin(b, x, y) => {
                let xv = self.emit_real_expr(*x, eb)?;
                let yv = self.emit_real_expr(*y, eb)?;
                Ok(bin(eb, bin_str(*b), xv, yv))
            }
            Op::Un(u, x) => {
                let xv = self.emit_real_expr(*x, eb)?;
                let mut e = IrOp::new("jsir.unary_expression");
                e.operands.push(xv);
                e.attrs.push(("operator_".into(), Attr::Str(un_str(*u).into())));
                e.attrs.push(("prefix".into(), Attr::Bool(true)));
                Ok(eb.push(e))
            }
            Op::MakeArray(elems) => {
                let mut e = IrOp::new("jsir.array_expression");
                for el in elems {
                    let ev = self.emit_real_expr(*el, eb)?;
                    e.operands.push(ev);
                }
                Ok(eb.push(e))
            }
            Op::MakeObject(props) => {
                let mut sub = self.eb();
                let mut prop_vs = Vec::new();
                for (k, val) in props {
                    let vv = self.emit_real_expr(*val, &mut sub)?;
                    let mut pop = IrOp::new("jsir.object_property");
                    pop.operands.push(vv);
                    match k {
                        PropKey::Ident(s) => {
                            pop.attrs.push(("literal_key".into(), Attr::Identifier(Box::new(ident_attr(s)))));
                            pop.attrs.push(("shorthand".into(), Attr::Bool(false)));
                        }
                        PropKey::Computed(c) => {
                            let cv = self.emit_real_expr(*c, &mut sub)?;
                            pop.operands.insert(0, cv);
                            pop.attrs.push(("computed".into(), Attr::Bool(true)));
                            pop.attrs.push(("shorthand".into(), Attr::Bool(false)));
                        }
                    }
                    prop_vs.push(sub.push(pop));
                }
                let mut end = IrOp::new("jsir.exprs_region_end");
                end.operands = prop_vs;
                sub.ops.push(end);
                let mut obj = IrOp::new("jsir.object_expression");
                obj.regions.push(Region::with_block(Block { args: vec![], ops: sub.ops }));
                Ok(eb.push(obj))
            }
            Op::Call { callee, args } => {
                // Method call (`recv.m(args)`) keeps its receiver bound; a plain
                // call renders the callee inline.
                let cv = match self.def_op_of(*callee) {
                    Some(Op::Member { obj, prop }) => {
                        let ov = self.emit_real_expr(*obj, eb)?;
                        match prop {
                            MemberKey::Static(name) => {
                                let mut m = IrOp::new("jsir.member_expression");
                                m.operands.push(ov);
                                m.attrs.push(("literal_property".into(), Attr::Identifier(Box::new(ident_attr(name)))));
                                eb.push(m)
                            }
                            MemberKey::Computed(c) => {
                                let cvk = self.emit_real_expr(*c, eb)?;
                                let mut m = IrOp::new("jsir.member_expression");
                                m.operands.push(ov);
                                m.operands.push(cvk);
                                eb.push(m)
                            }
                        }
                    }
                    _ => self.emit_real_expr(*callee, eb)?,
                };
                let mut e = IrOp::new("jsir.call_expression");
                e.operands.push(cv);
                for a in args {
                    let av = self.emit_real_expr(*a, eb)?;
                    e.operands.push(av);
                }
                Ok(eb.push(e))
            }
            other => Err(format!("inplace: value %{} is {} (cannot reconstruct expression)", v.0, op_kind(other))),
        }
    }

    /// `$[slot] = <expr(v)>;` — store a reconstructed dependency expression.
    fn slot_store_real(&self, slot: usize, v: Value) -> Result<IrOp, String> {
        let mut eb = self.eb();
        let target = self.cache_ref(slot, &mut eb);
        let val = self.emit_real_expr(v, &mut eb)?;
        let mut assign = IrOp::new("jsir.assignment_expression");
        assign.operands.push(target);
        assign.operands.push(val);
        assign.attrs.push(("operator_".into(), Attr::Str("=".into())));
        let av = eb.push(assign);
        let mut stmt = IrOp::new("jsir.expression_statement");
        stmt.operands.push(av);
        let mut block_ops = eb.ops;
        block_ops.push(stmt);
        Ok(wrap_stmt_run(block_ops))
    }

    /// Turn `const X = <init>;` into `X = <init>;` (assignment to the hoisted
    /// `let`), reusing the original initializer op verbatim (so its structure,
    /// formatting, and any closure body are preserved — no resynthesis).
    fn convert_decl_to_assign(&self, decl: IrOp) -> Result<IrOp, String> {
        let name = decl_name(&decl)?;
        let init = decl_init(&decl)?;
        let init_id = *init.results.first().ok_or("inplace: initializer op has no result value")?;
        let mut eb = self.eb();
        let lhs = eb.push(ident_ref_op(&name));
        eb.ops.push(init);
        let mut assign = IrOp::new("jsir.assignment_expression");
        assign.operands.push(lhs);
        assign.operands.push(init_id);
        assign.attrs.push(("operator_".into(), Attr::Str("=".into())));
        let av = eb.push(assign);
        let mut stmt = IrOp::new("jsir.expression_statement");
        stmt.operands.push(av);
        let mut block_ops = eb.ops;
        block_ops.push(stmt);
        Ok(wrap_stmt_run(block_ops))
    }

    /// The guarded form of an in-place scope: `if (deps) { recompute } else {…}`.
    /// Recursively rewrite a statement list: each reactive scope's contiguous run
    /// of statements becomes a guard; control-flow statements are kept verbatim
    /// but their nested bodies are rewritten (so scopes inside loops/branches are
    /// guarded in place — this is what replaces the relooper).
    fn rewrite_run(
        &self,
        ops: Vec<IrOp>,
        plans: &[InplaceScope<'a>],
        scope_node: &std::collections::HashMap<u32, usize>,
        emitted: &mut HashSet<usize>,
    ) -> Result<Vec<IrOp>, String> {
        // A block is a flat op list interleaving statement-roots (0-result) with
        // their operand-defs (1-result). Group into logical statements first:
        // each group is [operand-defs…, root].
        let groups = group_into_statements(ops);
        let mut out: Vec<IrOp> = Vec::new();
        let mut iter = groups.into_iter().peekable();
        while let Some(group) = iter.next() {
            let root = group.last().cloned().unwrap_or_else(|| IrOp::new("jsir.__empty"));

            // 1. Anonymous scopes consumed by this statement.
            for (pi, p) in plans.iter().enumerate() {
                if p.anon_expr.is_some() && !emitted.contains(&pi) && stmt_owns(&root, p.stmt_node) {
                    emitted.insert(pi);
                    flatten_run_into(self.emit_anon_guard(p)?, &mut out);
                }
            }

            // 2. A named scope owned here: wrap its contiguous run of statements.
            if let Some(pi) = self.scope_of_named(&root, scope_node) {
                if !emitted.insert(pi) {
                    return Err("inplace: a scope's statements are not contiguous".into());
                }
                let mut run: Vec<IrOp> = group;
                while let Some(next) = iter.peek() {
                    let next_root = next.last().cloned().unwrap_or_else(|| IrOp::new("jsir.__empty"));
                    if self.scope_of_named(&next_root, scope_node) == Some(pi) {
                        run.extend(iter.next().unwrap());
                    } else {
                        break;
                    }
                }
                flatten_run_into(self.emit_named_guard(&plans[pi], run)?, &mut out);
            } else if is_control_flow(&root.name) {
                // Keep the operand-defs (e.g. the condition) before the construct,
                // and recurse into its nested bodies.
                let mut group = group;
                let mut root = group.pop().unwrap();
                for region in &mut root.regions {
                    for block in &mut region.blocks {
                        let inner = std::mem::take(&mut block.ops);
                        block.ops = self.rewrite_run(inner, plans, scope_node, emitted)?;
                    }
                }
                out.extend(group);
                out.push(root);
            } else {
                out.extend(group);
            }
        }
        Ok(out)
    }

    /// The named scope (plan index) a statement *root* op is owned by, or `None`.
    fn scope_of_named(&self, root: &IrOp, scope_node: &std::collections::HashMap<u32, usize>) -> Option<usize> {
        if is_control_flow(&root.name) {
            return None;
        }
        for (&node, &pi) in scope_node {
            if stmt_owns(root, node) {
                return Some(pi);
            }
        }
        None
    }

    /// Build `if (deps changed) { dep-stores; recompute; out-store } else
    /// { out-restore }` for one scope. `recompute` is the supplied ops.
    fn emit_guard(&self, plan: &InplaceScope<'a>, recompute: Vec<IrOp>) -> Result<IrOp, String> {
        let info = plan.info;
        let (dep_base, out_base) = (plan.dep_base, plan.out_base);

        let mut test_eb = self.eb();
        let test_val = if info.deps.is_empty() {
            let read = self.cache_read(out_base, &mut test_eb);
            let sentinel = self.sentinel(&mut test_eb);
            bin(&mut test_eb, "===", read, sentinel)
        } else {
            let mut acc: Option<ValueId> = None;
            for (i, d) in info.deps.iter().enumerate() {
                let read = self.cache_read(dep_base + i, &mut test_eb);
                let dep = self.emit_real_expr(*d, &mut test_eb)?;
                let ne = bin(&mut test_eb, "!==", read, dep);
                acc = Some(match acc { None => ne, Some(prev) => bin(&mut test_eb, "||", prev, ne) });
            }
            acc.expect("non-empty deps")
        };

        let mut then_ops: Vec<IrOp> = Vec::new();
        for (i, d) in info.deps.iter().enumerate() {
            then_ops.push(self.slot_store_real(dep_base + i, *d)?);
        }
        then_ops.extend(recompute);
        for (j, (_, name)) in plan.outs.iter().enumerate() {
            then_ops.push(self.slot_store(out_base + j, name));
        }

        let else_ops: Vec<IrOp> = plan
            .outs
            .iter()
            .enumerate()
            .map(|(j, (_, name))| self.restore_stmt(name, out_base + j))
            .collect();

        let consequent = block_statement(then_ops);
        let alternate = block_statement(else_ops);
        let mut if_op = IrOp::new("jshir.if_statement");
        if_op.operands.push(test_val);
        if_op.regions.push(Region::with_block(Block { args: vec![], ops: vec![consequent] }));
        if_op.regions.push(Region::with_block(Block { args: vec![], ops: vec![alternate] }));

        let mut block_ops = test_eb.ops;
        block_ops.push(if_op);
        Ok(wrap_stmt_run(block_ops))
    }

    /// A named scope: recompute = the scope's original statements verbatim, with
    /// the output `const X = E` turned into `X = E` (assignment to the hoisted
    /// `let`); intermediates and stores are kept as-is.
    fn emit_named_guard(&self, plan: &InplaceScope<'a>, run: Vec<IrOp>) -> Result<IrOp, String> {
        let mut recompute = Vec::new();
        for stmt in run {
            recompute.push(self.convert_owned_stmt(stmt)?);
        }
        self.emit_guard(plan, recompute)
    }

    /// An anonymous scope: recompute = `temp = <resynthesized expression>`. The
    /// original embedding was already patched to reference `temp`.
    fn emit_anon_guard(&self, plan: &InplaceScope<'a>) -> Result<IrOp, String> {
        let (out_value, out_name) = &plan.outs[0];
        let mut eb = self.eb();
        let rhs = self.emit_real_expr(*out_value, &mut eb)?;
        let lhs = eb.push(ident_ref_op(out_name));
        let mut assign = IrOp::new("jsir.assignment_expression");
        assign.operands.push(lhs);
        assign.operands.push(rhs);
        assign.attrs.push(("operator_".into(), Attr::Str("=".into())));
        let av = eb.push(assign);
        let mut stmt = IrOp::new("jsir.expression_statement");
        stmt.operands.push(av);
        let mut block_ops = eb.ops;
        block_ops.push(stmt);
        let recompute = vec![wrap_stmt_run(block_ops)];
        self.emit_guard(plan, recompute)
    }

    /// Inside a named scope guard, a `const X = E` declaring the scope OUTPUT
    /// becomes `X = E`; intermediate declarations and other statements stay.
    fn convert_owned_stmt(&self, stmt: IrOp) -> Result<IrOp, String> {
        if stmt.name == "jsir.variable_declaration" {
            if let Some(bound) = self.bound_value_of(&stmt) {
                if self.scope_outputs.contains(&bound) {
                    return self.convert_decl_to_assign(stmt);
                }
            }
        }
        Ok(stmt)
    }

    /// The value a single-declarator `variable_declaration` binds: the value with
    /// this statement's SrcRef and the latest def point (the initializer result).
    fn bound_value_of(&self, stmt: &IrOp) -> Option<Value> {
        let name = decl_name(stmt).ok()?;
        // Reverse the names map: the bound value is the one we recorded this name
        // for (built the same way, by latest def point).
        self.names.iter().find(|(_, n)| **n == name).map(|(v, _)| *v)
    }

    fn def_op_of(&self, v: Value) -> Option<&'a Op> {
        for b in &self.cfg.blocks {
            for ins in &b.instrs {
                if ins.result == Some(v) {
                    return Some(&ins.op);
                }
            }
        }
        None
    }
}

fn op_kind(op: &Op) -> &'static str {
    match op {
        Op::Const(_) => "Const", Op::ReadVar(_) => "ReadVar", Op::WriteVar(_, _) => "WriteVar",
        Op::Bin(_, _, _) => "Bin", Op::Un(_, _) => "Un", Op::Global(_) => "Global",
        Op::Call { .. } => "Call", Op::Member { .. } => "Member", Op::StoreMember { .. } => "StoreMember",
        Op::MakeObject(_) => "MakeObject", Op::MakeArray(_) => "MakeArray",
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

// =============================================================================
// In-place memoization (PROTOTYPE) — edit the original JSIR tree rather than
// re-synthesizing the body from the CFG. Loops / try / switch and every other
// non-scope statement are kept verbatim; only the scope-producing variable
// declarations are rewritten into cache guards. See the module-level note in the
// design doc; this exists to validate the approach end to end.
// =============================================================================

/// Memoize `file` by rewriting reactive-scope declarations in place. Supports
/// the prototype shape (each escaping scope is a single `const X = <expr>;` with
/// param/member/const dependencies) and returns a clear `Err` on anything else.
pub fn memoize_inplace(
    cfg: &Cfg,
    infos: &[ScopeInfo],
    ranges: &Ranges,
    file: &IrOp,
) -> Result<IrOp, String> {
    let emitted: Vec<&ScopeInfo> = infos.iter().filter(|i| !i.outputs.is_empty()).collect();
    if emitted.is_empty() {
        return Err("inplace: nothing escapes (cache_size 0)".into());
    }

    // SOUNDNESS: reuse the same check `build_layout` runs — a mutation crossing a
    // scope boundary, a scope whose owned statements span blocks, or a branch
    // condition reading a mutated allocation all bail here exactly as before, so
    // the in-place path never memoizes something the structured path rejects as
    // unsound. (It also rejects the multi-block scopes the single-statement
    // prototype cannot yet rewrite.)
    let value_set: Vec<HashSet<Value>> =
        emitted.iter().map(|i| i.scope.values.iter().copied().collect()).collect();
    let all_value_sets: Vec<HashSet<Value>> =
        infos.iter().map(|i| i.scope.values.iter().copied().collect()).collect();
    check_soundness(cfg, ranges, &emitted, &value_set, &all_value_sets)?;

    // value -> originating source statement node_id, via the CFG SrcRefs.
    let mut val_stmt: std::collections::HashMap<Value, u32> = std::collections::HashMap::new();
    for b in &cfg.blocks {
        for ins in &b.instrs {
            if let (Some(res), Some(sr)) = (ins.result, ins.src) {
                val_stmt.entry(res).or_insert(sr.stmt_node_id);
            }
        }
    }

    // value -> expression node_id (expression-granular provenance).
    let mut val_expr: std::collections::HashMap<Value, u32> = std::collections::HashMap::new();
    for b in &cfg.blocks {
        for ins in &b.instrs {
            if let (Some(res), Some(sr)) = (ins.result, ins.src) {
                if let Some(e) = sr.expr_node_id {
                    val_expr.entry(res).or_insert(e);
                }
            }
        }
    }

    // value -> source name: parameters, plus every declared local. A declaration
    // `const X = E` binds X to the value E evaluates to — recovered as the value
    // with this statement's SrcRef and the latest def point (the initializer's
    // result is computed last). This is what lets a dependency that is a local
    // variable render as its real name instead of a synthesized temp.
    let mut names: std::collections::HashMap<Value, String> = std::collections::HashMap::new();
    for (i, p) in cfg.params.iter().enumerate() {
        if let Some(n) = cfg.param_names.get(i) {
            names.insert(*p, n.clone());
        }
    }
    let mut decls: Vec<&IrOp> = Vec::new();
    collect_decls(function_body_stmts(file)?, &mut decls);
    for decl in decls {
        let name = match decl_name(decl) {
            Ok(n) => n,
            Err(_) => continue,
        };
        let mut best: Option<(Value, crate::mutability::Point)> = None;
        for (v, node) in &val_stmt {
            if stmt_owns(decl, *node) {
                let d = ranges.def.get(v).copied().unwrap_or(0);
                if best.map_or(true, |(_, bd)| d >= bd) {
                    best = Some((*v, d));
                }
            }
        }
        if let Some((v, _)) = best {
            names.insert(v, name);
        }
    }
    // A value that represents a source variable (a phi, or a read of a `let`
    // assigned across `&&`/`||`/`if` branches) renders as that variable's name.
    // Synthetic temps (`$t…`) are skipped — they are not real identifiers, so
    // such a value is reconstructed inline instead.
    for (&phi, &var) in &cfg.phi_var {
        if let Some(n) = cfg.var_names.get(var.0 as usize) {
            if !n.starts_with('$') {
                names.entry(phi).or_insert_with(|| n.clone());
            }
        }
    }

    // Build the per-scope plan: classify each output as a declared local (wrap
    // its declaration in place) or an anonymous sub-expression (synthesize a
    // temp, resynthesize the expression, patch the embedding to use the temp).
    let mut plans: Vec<InplaceScope> = Vec::new();
    let mut slot = 0usize;
    let mut temp_counter = 0usize;
    for info in &emitted {
        // Multi-output scopes need the scope-alignment the structured path did;
        // the in-place emitter currently gets their cache structure wrong, so
        // bail (leave un-memoized) until that lands. Single-output is the common
        // shape and the only one handled here.
        if info.outputs.len() != 1 {
            return Err("inplace: multi-output scope (deferred)".into());
        }
        let dep_base = slot;
        let out_base = dep_base + info.deps.len();
        slot = out_base + info.outputs.len();
        let stmt_node = *val_stmt
            .get(&info.outputs[0])
            .ok_or("inplace: output has no source statement")?;
        let mut outs: Vec<(Value, String)> = Vec::new();
        let mut anon_expr: Option<u32> = None;
        for o in &info.outputs {
            match names.get(o) {
                Some(n) => outs.push((*o, n.clone())),
                None => {
                    // Only a single anonymous output is supported (it is extracted
                    // into a temp); a multi-output scope must be all-named.
                    if info.outputs.len() != 1 {
                        return Err("inplace: multi-output scope with an anonymous output".into());
                    }
                    let e = *val_expr
                        .get(o)
                        .ok_or("inplace: anonymous output has no expression provenance")?;
                    let t = format!("_t{temp_counter}");
                    temp_counter += 1;
                    anon_expr = Some(e);
                    outs.push((*o, t));
                }
            }
        }
        plans.push(InplaceScope { info, dep_base, out_base, outs, anon_expr, stmt_node });
    }
    let cache_size = slot;

    let scope_outputs: HashSet<Value> =
        emitted.iter().flat_map(|i| i.outputs.iter().copied()).collect();

    let mut new_file = file.clone();
    set_module(&mut new_file);
    let base_id = file_max_value_id(&new_file);
    let em = Emitter {
        cfg,
        scope_outputs: &scope_outputs,
        ids: std::cell::Cell::new(base_id),
        names,
    };

    // Patch each anonymous output's embedding: replace the op producing it with
    // an identifier referencing the synthesized temp (keeping the result
    // value-id so consumers still resolve). The standalone guard recomputes the
    // temp; the consuming statement now reads it.
    for p in &plans {
        if let Some(expr_node) = p.anon_expr {
            if !patch_expr_to_ident(&mut new_file, expr_node, &p.outs[0].1) {
                return Err("inplace: could not locate anonymous output expression to patch".into());
            }
        }
    }

    // Map each statement-root node owned by a NAMED scope to its plan index. A
    // statement is owned if it DEFINES a scope value or MUTATES one (a member
    // store on a scope value, e.g. `y.z = …` between `let y` and `let x` when y
    // and x are aliased into one scope) — so the scope's run is the full
    // contiguous span, not just its declarations. (Anonymous scopes are handled
    // via their consuming statement in the walk.)
    let mut def_op: std::collections::HashMap<Value, &Op> = std::collections::HashMap::new();
    for b in &cfg.blocks {
        for ins in &b.instrs {
            if let Some(r) = ins.result {
                def_op.insert(r, &ins.op);
            }
        }
    }
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
    let mut scope_node: std::collections::HashMap<u32, usize> = std::collections::HashMap::new();
    for (pi, p) in plans.iter().enumerate() {
        if p.anon_expr.is_some() {
            continue;
        }
        let vals: HashSet<Value> = p.info.scope.values.iter().copied().collect();
        for b in &cfg.blocks {
            for ins in &b.instrs {
                let owns = match (&ins.result, &ins.op) {
                    (Some(r), _) if vals.contains(r) => true,
                    (_, Op::StoreMember { obj, .. }) if vals.contains(&base_object(*obj)) => true,
                    _ => false,
                };
                if owns {
                    if let Some(sr) = ins.src {
                        scope_node.insert(sr.stmt_node_id, pi);
                    }
                }
            }
        }
    }

    // Hoisted `let`s: every scope output name (recovered or temp), stable order.
    let mut let_names: Vec<String> = Vec::new();
    for p in &plans {
        for (_, name) in &p.outs {
            if !let_names.contains(name) {
                let_names.push(name.clone());
            }
        }
    }

    let body = function_body_stmts_mut(&mut new_file)?;
    let stmts = std::mem::take(body);

    // Recursively rewrite the statement tree: each scope's guard is placed in
    // position (named scopes wrap their declaration run; anonymous scopes emit a
    // standalone guard before the statement that consumes them), and control-flow
    // bodies are descended into verbatim.
    let mut emitted_scopes: HashSet<usize> = HashSet::new();
    let rewritten = em.rewrite_run(stmts, &plans, &scope_node, &mut emitted_scopes)?;

    let mut out_ops: Vec<IrOp> = Vec::new();
    for op in em.cache_decl(cache_size) {
        flatten_run_into(op, &mut out_ops);
    }
    if !let_names.is_empty() {
        flatten_run_into(em.let_decl(&let_names), &mut out_ops);
    }
    out_ops.extend(rewritten);
    *function_body_stmts_mut(&mut new_file)? = out_ops;

    prepend_import(&mut new_file)?;
    Ok(new_file)
}

/// Replace the op with `node_id == expr_node` with an `identifier(name)` that
/// keeps the original's result value-id (so consumers still resolve). Used to
/// redirect an anonymous scope output's embedding to its synthesized temp.
fn patch_expr_to_ident(op: &mut IrOp, expr_node: u32, name: &str) -> bool {
    for region in &mut op.regions {
        for block in &mut region.blocks {
            for child in &mut block.ops {
                if child.node_id == Some(expr_node) {
                    let mut id = ident_op(name);
                    id.results = child.results.clone();
                    id.node_id = child.node_id;
                    *child = id;
                    return true;
                }
                if patch_expr_to_ident(child, expr_node, name) {
                    return true;
                }
            }
        }
    }
    false
}

/// A reactive scope's cache-slot layout for the in-place emitter.
struct InplaceScope<'a> {
    info: &'a ScopeInfo,
    dep_base: usize,
    out_base: usize,
    /// Each output value with its source name (a declared local) or, for the
    /// single-output anonymous case, a synthesized temp.
    outs: Vec<(Value, String)>,
    /// `Some(expr_node)` for a single anonymous-output scope: the emitter
    /// resynthesizes it into `outs[0].1` and patches the embedding (op with this
    /// node_id) to reference the temp. `None` for named scopes (any arity).
    anon_expr: Option<u32>,
    /// The statement owning the output(s): their declaration (named) or the
    /// statement that embeds the anonymous output.
    stmt_node: u32,
}

/// Statement kinds whose nested bodies the in-place walk recurses into (rather
/// than treating the statement itself as a scope leaf).
fn is_control_flow(name: &str) -> bool {
    matches!(
        name,
        "jshir.if_statement"
            | "jshir.while_statement"
            | "jshir.do_while_statement"
            | "jshir.for_statement"
            | "jshir.for_of_statement"
            | "jshir.for_in_statement"
            | "jshir.switch_statement"
            | "jshir.try_statement"
            | "jshir.block_statement"
            | "jshir.labeled_statement"
    )
}

/// Group a block's flat op list into logical statements. The block interleaves
/// statement-root ops (0 results) with their operand-def ops (1 result, in
/// evaluation order before the root that consumes them); each returned group is
/// the run of operand-defs ending in their statement root.
fn group_into_statements(ops: Vec<IrOp>) -> Vec<Vec<IrOp>> {
    let mut groups: Vec<Vec<IrOp>> = Vec::new();
    let mut cur: Vec<IrOp> = Vec::new();
    for op in ops {
        let is_root = op.results.is_empty();
        cur.push(op);
        if is_root {
            groups.push(std::mem::take(&mut cur));
        }
    }
    if !cur.is_empty() {
        groups.push(cur);
    }
    groups
}

/// Expand a `jsir.__stmt_run` holder into its flat op list (recursively).
fn flatten_run_into(op: IrOp, out: &mut Vec<IrOp>) {
    if op.name == "jsir.__stmt_run" {
        if let Some(block) = op.regions.into_iter().next().and_then(|r| r.blocks.into_iter().next()) {
            for inner in block.ops {
                flatten_run_into(inner, out);
            }
        }
        return;
    }
    out.push(op);
}

/// The largest result/operand value id in `file`, +1 (so synthesized ids never
/// collide with the original tree's).
fn file_max_value_id(file: &IrOp) -> u32 {
    let mut max = 0u32;
    fn scan(op: &IrOp, max: &mut u32) {
        for r in &op.results {
            *max = (*max).max(r.0 + 1);
        }
        for o in &op.operands {
            *max = (*max).max(o.0 + 1);
        }
        for r in &op.regions {
            for b in &r.blocks {
                for inner in &b.ops {
                    scan(inner, max);
                }
            }
        }
    }
    scan(file, &mut max);
    max
}

fn set_module(file: &mut IrOp) {
    if let Some(prog) = first_op_mut(file, "jsir.program") {
        if let Some((_, a)) = prog.attrs.iter_mut().find(|(k, _)| k == "source_type") {
            *a = Attr::Str("module".into());
        } else {
            prog.attrs.push(("source_type".into(), Attr::Str("module".into())));
        }
    }
}

fn first_op_mut<'a>(op: &'a mut IrOp, name: &str) -> Option<&'a mut IrOp> {
    if op.name == name {
        return Some(op);
    }
    for r in &mut op.regions {
        for b in &mut r.blocks {
            for o in &mut b.ops {
                if let Some(found) = first_op_mut(o, name) {
                    return Some(found);
                }
            }
        }
    }
    None
}

fn find_function_mut(op: &mut IrOp) -> Option<&mut IrOp> {
    if op.name == "jsir.function_declaration" {
        return Some(op);
    }
    if op.name == "jsir.export_named_declaration" || op.name == "jsir.export_default_declaration" {
        return op
            .regions
            .first_mut()
            .and_then(|r| r.blocks.first_mut())
            .and_then(|b| b.ops.iter_mut().find(|o| o.name == "jsir.function_declaration"));
    }
    for r in &mut op.regions {
        for b in &mut r.blocks {
            for o in &mut b.ops {
                if let Some(f) = find_function_mut(o) {
                    return Some(f);
                }
            }
        }
    }
    None
}

fn function_body_stmts_mut(file: &mut IrOp) -> Result<&mut Vec<IrOp>, String> {
    let func = find_function_mut(file).ok_or("inplace: no function declaration")?;
    let body_block = func
        .regions
        .get_mut(1)
        .and_then(|r| r.blocks.first_mut())
        .and_then(|b| b.ops.first_mut())
        .ok_or("inplace: function has no body block_statement")?;
    if body_block.name != "jshir.block_statement" {
        return Err(format!("inplace: expected body block_statement, got {}", body_block.name));
    }
    body_block
        .regions
        .first_mut()
        .and_then(|r| r.blocks.first_mut())
        .map(|b| &mut b.ops)
        .ok_or_else(|| "inplace: body has no statements block".to_string())
}

fn prepend_import(file: &mut IrOp) -> Result<(), String> {
    let prog = first_op_mut(file, "jsir.program").ok_or("inplace: no program op")?;
    let block = prog
        .regions
        .first_mut()
        .and_then(|r| r.blocks.first_mut())
        .ok_or("inplace: program has no block")?;
    block.ops.insert(0, import_c_inplace());
    Ok(())
}

/// `import { c as _c } from "react/compiler-runtime";`
fn import_c_inplace() -> IrOp {
    let mut op = IrOp::new("jsir.import_declaration");
    op.attrs.push((
        "source".into(),
        Attr::StringLiteralKey(Box::new(StringLiteralKeyAttr {
            start_line: 0, start_col: 0, end_line: 0, end_col: 0, start_index: 0, end_index: 0,
            scope_uid: 0,
            value: "react/compiler-runtime".into(),
            raw: "\"react/compiler-runtime\"".into(),
            raw_value: "react/compiler-runtime".into(),
        })),
    ));
    op.attrs.push((
        "specifiers".into(),
        Attr::Array(vec![Attr::ImportSpecifier(Box::new(ImportSpecifierAttr {
            kind: ImportSpecKind::Named,
            start_line: 0, start_col: 0, end_line: 0, end_col: 0, start_index: 0, end_index: 0,
            scope_uid: 0,
            sym_name: "_c".into(),
            sym_scope: 0,
            imported: Some(Attr::Identifier(Box::new(ident_attr("c")))),
            local: ident_attr("_c"),
        }))]),
    ));
    op
}

/// Does statement `stmt` own SrcRef `node` (it is, or its subtree contains, the
/// statement-root node the SrcRef points at)?
fn stmt_owns(stmt: &IrOp, node: u32) -> bool {
    stmt.node_id == Some(node) || contains_node(stmt, node)
}

fn contains_node(op: &IrOp, node: u32) -> bool {
    if op.node_id == Some(node) {
        return true;
    }
    op.regions.iter().any(|r| r.blocks.iter().any(|b| b.ops.iter().any(|o| contains_node(o, node))))
}

/// The declared identifier name of a single-declarator `variable_declaration`.
fn decl_name(stmt: &IrOp) -> Result<String, String> {
    if stmt.name != "jsir.variable_declaration" {
        return Err(format!("inplace: scope statement is {}, expected variable_declaration", stmt.name));
    }
    let ops = stmt
        .regions
        .first()
        .and_then(|r| r.blocks.first())
        .map(|b| &b.ops)
        .ok_or("inplace: empty declaration")?;
    for o in ops {
        if o.name == "jsir.identifier_ref" {
            if let Some((_, Attr::Str(n))) = o.attrs.iter().find(|(k, _)| k == "name") {
                return Ok(n.clone());
            }
        }
    }
    Err("inplace: declaration has no identifier_ref".into())
}

/// The initializer expression op of a single-declarator `variable_declaration`
/// (the op between the identifier_ref and the declarator/region-end markers).
fn decl_init(stmt: &IrOp) -> Result<IrOp, String> {
    let ops = stmt
        .regions
        .first()
        .and_then(|r| r.blocks.first())
        .map(|b| &b.ops)
        .ok_or("inplace: empty declaration")?;
    let mut seen_id = false;
    for o in ops {
        if o.name == "jsir.identifier_ref" {
            seen_id = true;
            continue;
        }
        if o.name == "jsir.variable_declarator" || o.name == "jsir.exprs_region_end" {
            break;
        }
        if seen_id {
            return Ok(o.clone());
        }
    }
    Err("inplace: single-declarator initializer not found".into())
}

/// Immutable view of the function body statement op-list (mirrors
/// `function_body_stmts_mut`).
fn function_body_stmts(file: &IrOp) -> Result<&Vec<IrOp>, String> {
    fn find_function(op: &IrOp) -> Option<&IrOp> {
        if op.name == "jsir.function_declaration" {
            return Some(op);
        }
        if op.name == "jsir.export_named_declaration" || op.name == "jsir.export_default_declaration" {
            return op
                .regions
                .first()
                .and_then(|r| r.blocks.first())
                .and_then(|b| b.ops.iter().find(|o| o.name == "jsir.function_declaration"));
        }
        for r in &op.regions {
            for b in &r.blocks {
                for o in &b.ops {
                    if let Some(f) = find_function(o) {
                        return Some(f);
                    }
                }
            }
        }
        None
    }
    let func = find_function(file).ok_or("inplace: no function declaration")?;
    let body_block = func
        .regions
        .get(1)
        .and_then(|r| r.blocks.first())
        .and_then(|b| b.ops.first())
        .ok_or("inplace: function has no body block_statement")?;
    body_block
        .regions
        .first()
        .and_then(|r| r.blocks.first())
        .map(|b| &b.ops)
        .ok_or_else(|| "inplace: body has no statements block".to_string())
}

/// Collect every `variable_declaration` statement op reachable from `stmts`,
/// descending into nested control-flow bodies (if/while/for/…).
fn collect_decls<'a>(stmts: &'a [IrOp], out: &mut Vec<&'a IrOp>) {
    for op in stmts {
        if op.name == "jsir.variable_declaration" {
            out.push(op);
        }
        for r in &op.regions {
            for b in &r.blocks {
                collect_decls(&b.ops, out);
            }
        }
    }
}
