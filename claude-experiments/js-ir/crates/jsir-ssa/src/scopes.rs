//! Reactive-scope inference (a first cut of the React Compiler's
//! `InferReactiveScopes`).
//!
//! A reactive scope is a contiguous range of program points that must be
//! memoized as a unit: you cannot cache a value separately from another value
//! whose mutable range overlaps it (they are produced/mutated together). We
//! compute scope boundaries by **merging overlapping mutable ranges** — the one
//! place the analysis is forced, since a mutation in the middle of a value's
//! range means everything between its creation and that mutation is one unit.
//!
//! Immutable reference values (pure construction) each form their own singleton
//! scope: they are independently memoizable. Dependency-based merging of such
//! scopes (`MergeScopesThatInvalidateTogether`) is left for a later pass.

use crate::cfg::{Cfg, Value};
use crate::mutability::{Point, Ranges};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Scope {
    pub start: Point,
    pub end: Point,
    /// Reference values produced inside this scope.
    pub values: Vec<Value>,
    /// Whether any value in the scope is mutated within it.
    pub mutable: bool,
}

/// Infer reactive scopes — a faithful port of React's
/// `InferReactiveScopeVariables` / `findDisjointMutableValues`: union the
/// identifiers that *co-mutate* into disjoint sets, each of which becomes one
/// reactive scope.
///
/// Per instruction, build a group and union it:
///  - the **result** joins iff it is mutated (its range extends past its
///    definition) or its op `may_allocate` a non-primitive;
///  - an **operand** joins iff this instruction lies within the operand's
///    mutable range (i.e. the instruction mutates/captures it) and the operand
///    is not a function input (param / block-arg).
///
/// This replaces the old interval-containment grouping (which over-merged
/// everything fitting inside a merged mutable interval, and under-merged via a
/// singleton fallback). Operands that are unmutated never have their range
/// extend to the using instruction, so e.g. `const y = Boolean(x)` does not pull
/// `x` into the array scope of `[x, y]` — matching React's separate scopes.
///
/// React keys "not an input" off `range.start > 0` because upstream instruction
/// ids start at 1 and only params sit at 0. OUR point space starts at 0 for the
/// FIRST instruction, so `start > 0` would wrongly exclude a value defined at
/// point 0 (e.g. a leading `const foo = () => …`). We test param/block-arg
/// membership directly instead.
pub fn infer(cfg: &Cfg, r: &Ranges) -> Vec<Scope> {
    let mut inputs: std::collections::HashSet<Value> = cfg.params.iter().copied().collect();
    for b in &cfg.blocks {
        for p in &b.params {
            inputs.insert(*p);
        }
    }
    let def_op = def_op_map(cfg);
    let mut uf = UnionFind::new();
    // Walk in RPO with a per-instruction point counter IDENTICAL to
    // `aliasing_ranges::compute_ranges` so every instruction — including
    // no-result ones like `store x.y <- v` (a mutation that must co-group its
    // operands) — has the correct point.
    let order = crate::ssa::reverse_postorder(cfg);
    let mut point = 0u32;
    for &b in &order {
        for ins in &cfg.block(b).instrs {
            let pt = point;
            point += 1;
            let mut group: Vec<Value> = Vec::new();
            if let Some(lv) = ins.result {
                let (s, e) = r.range.get(&lv).copied().unwrap_or((0, 0));
                let is_primitive = !r.is_ref.get(&lv).copied().unwrap_or(false);
                // ours `end > start` (inclusive ranges) ≡ React `end > start + 1`.
                if e > s || may_allocate(&ins.op, is_primitive) {
                    uf.ensure(lv);
                    group.push(lv);
                }
            }
            // Candidate operands (processed for EVERY instruction, incl. no-result
            // mutations). A method call `y.push(z)` lowers to `Call(Member(y,push),
            // [z])`: the *receiver* y is the base of the callee member, not a direct
            // operand. React's HIR MethodCall lists the receiver directly, so we
            // resolve the callee member chain to its root and include it — else y
            // and z never co-mutate (the push that captures z into y is invisible).
            let mut cands: Vec<Value> = ins.op.operands();
            if let crate::cfg::Op::Call { callee, .. } = &ins.op {
                let base = member_root(&def_op, *callee);
                if base != *callee {
                    cands.push(base);
                }
            }
            for o in cands {
                let (os, oe) = r.range.get(&o).copied().unwrap_or((0, 0));
                // operand mutated through this instruction (its range contains the
                // instruction point) and not a function input.
                if !inputs.contains(&o) && os <= pt && pt <= oe {
                    group.push(o);
                }
            }
            if group.len() > 1 {
                uf.union_all(&group);
            }
        }
        point += 1; // terminator slot
    }

    // Each disjoint set is one scope; its range spans the union of member ranges.
    let mut sets: std::collections::HashMap<Value, Vec<Value>> = std::collections::HashMap::new();
    for v in uf.members() {
        let root = uf.find(v);
        sets.entry(root).or_default().push(v);
    }
    let mut scopes: Vec<Scope> = Vec::new();
    for (_root, mut values) in sets {
        values.sort_by_key(|v| (r.def.get(v).copied().unwrap_or(0), v.0));
        let start = values
            .iter()
            .filter_map(|v| r.range.get(v).map(|x| x.0))
            .min()
            .unwrap_or(0);
        let end = values
            .iter()
            .filter_map(|v| r.range.get(v).map(|x| x.1))
            .max()
            .unwrap_or(0);
        let mutable = values.iter().any(|v| {
            let (s, e) = r.range.get(v).copied().unwrap_or((0, 0));
            e > s
        });
        scopes.push(Scope {
            start,
            end,
            values,
            mutable,
        });
    }
    scopes.sort_by_key(scope_sort_key);
    scopes
}

/// React's `MergeOverlappingReactiveScopes`: scopes whose ranges *cross*
/// (partially overlap, neither fully contains the other) or are *identical*
/// invalidate together and must merge; a scope fully *nested* inside another is
/// kept separate (React emits nested memo blocks). The union-find baked into the
/// old `infer` merged ALL overlapping ranges including nested ones — over-merge.
/// This pass restores the cross-merge that aliasing cases need (a value mutated
/// through an alias shares a scope) while preserving genuine nesting.
fn merge_overlapping_scopes(scopes: Vec<Scope>) -> Vec<Scope> {
    let n = scopes.len();
    if n < 2 {
        return scopes;
    }
    let mut parent: Vec<usize> = (0..n).collect();
    fn find(parent: &mut [usize], i: usize) -> usize {
        let mut root = i;
        while parent[root] != root {
            root = parent[root];
        }
        let mut cur = i;
        while parent[cur] != root {
            let next = parent[cur];
            parent[cur] = root;
            cur = next;
        }
        root
    }
    for i in 0..n {
        for j in (i + 1)..n {
            let (si, ei) = (scopes[i].start, scopes[i].end);
            let (sj, ej) = (scopes[j].start, scopes[j].end);
            let overlap = si <= ej && sj <= ei;
            if !overlap {
                continue;
            }
            let identical = si == sj && ei == ej;
            let i_contains_j = si <= sj && ej <= ei;
            let j_contains_i = sj <= si && ei <= ej;
            // proper nesting (one strictly inside the other) is preserved.
            if (i_contains_j || j_contains_i) && !identical {
                continue;
            }
            let (a, b) = (find(&mut parent, i), find(&mut parent, j));
            if a != b {
                parent[a] = b;
            }
        }
    }
    let mut groups: std::collections::HashMap<usize, Vec<usize>> = std::collections::HashMap::new();
    for i in 0..n {
        let r = find(&mut parent, i);
        groups.entry(r).or_default().push(i);
    }
    let mut out: Vec<Scope> = Vec::new();
    for (_root, idxs) in groups {
        let mut values: Vec<Value> = Vec::new();
        let mut start = u32::MAX;
        let mut end = 0u32;
        let mut mutable = false;
        for &i in &idxs {
            values.extend(scopes[i].values.iter().copied());
            start = start.min(scopes[i].start);
            end = end.max(scopes[i].end);
            mutable |= scopes[i].mutable;
        }
        values.sort_by_key(|v| v.0);
        values.dedup();
        out.push(Scope {
            start,
            end,
            values,
            mutable,
        });
    }
    out.sort_by_key(scope_sort_key);
    out
}

fn scope_sort_key(s: &Scope) -> (Point, Point, Vec<u32>) {
    let mut values: Vec<u32> = s.values.iter().map(|v| v.0).collect();
    values.sort();
    (s.start, s.end, values)
}

/// Whether an op may allocate a fresh non-primitive value (so its result is a
/// memoization candidate even when never subsequently mutated). Mirrors React's
/// `mayAllocate`: object/array/JSX/closure/regexp literals and member stores
/// always allocate; a call allocates iff its result is not a primitive; loads,
/// arithmetic, and member reads never do.
fn may_allocate(op: &crate::cfg::Op, lvalue_is_primitive: bool) -> bool {
    use crate::cfg::Op;
    match op {
        Op::MakeObject(_) | Op::MakeArray(_) | Op::StoreMember { .. } => true,
        Op::Call { .. } => !lvalue_is_primitive,
        _ => false,
    }
}

/// Map each value to its defining op (for member-chain resolution in grouping).
fn def_op_map(cfg: &Cfg) -> std::collections::HashMap<Value, &crate::cfg::Op> {
    let mut m = std::collections::HashMap::new();
    for b in &cfg.blocks {
        for ins in &b.instrs {
            if let Some(res) = ins.result {
                m.insert(res, &ins.op);
            }
        }
    }
    m
}

/// Resolve a member-load chain to its root object: `member_root(y.a.b) == y`.
/// A non-member value is its own root.
fn member_root(def_op: &std::collections::HashMap<Value, &crate::cfg::Op>, v: Value) -> Value {
    let mut cur = v;
    let mut guard = 0;
    while let Some(crate::cfg::Op::Member { obj, .. }) = def_op.get(&cur).copied() {
        cur = *obj;
        guard += 1;
        if guard > def_op.len() + 1 {
            break;
        }
    }
    cur
}

/// Minimal union-find over [`Value`]s for reactive-scope grouping.
struct UnionFind {
    parent: std::collections::HashMap<Value, Value>,
}
impl UnionFind {
    fn new() -> Self {
        Self {
            parent: std::collections::HashMap::new(),
        }
    }
    fn ensure(&mut self, v: Value) {
        self.parent.entry(v).or_insert(v);
    }
    fn find(&mut self, v: Value) -> Value {
        let p = self.parent[&v];
        if p == v {
            v
        } else {
            let root = self.find(p);
            self.parent.insert(v, root);
            root
        }
    }
    fn union_all(&mut self, vs: &[Value]) {
        for &v in vs {
            self.ensure(v);
        }
        let r0 = self.find(vs[0]);
        for &v in &vs[1..] {
            let rv = self.find(v);
            if rv != r0 {
                self.parent.insert(rv, r0);
            }
        }
    }
    fn members(&self) -> Vec<Value> {
        self.parent.keys().copied().collect()
    }
}

/// Values that derive (transitively) from a parameter — the *reactive* values.
/// Globals and constants are non-reactive (stable across renders) and never
/// become dependencies.
fn reactive_values(
    cfg: &Cfg,
    stable: &std::collections::HashSet<Value>,
) -> std::collections::HashSet<Value> {
    let mut reactive: std::collections::HashSet<Value> = cfg.params.iter().copied().collect();
    // Hook results are reactive roots, like props/state — a `useState` value, a
    // `useContext` value, a custom hook's return all change across renders. The
    // stable hooks (`useRef`) and the setter element are excluded via `stable`.
    for b in &cfg.blocks {
        for ins in &b.instrs {
            if let (Some(res), crate::cfg::Op::Call { callee, .. }) = (ins.result, &ins.op) {
                if is_hook_callee(cfg, *callee) && !stable.contains(&res) {
                    reactive.insert(res);
                }
            }
        }
    }
    // Block params (phis) and any op whose operand is reactive become reactive —
    // but a stable value never becomes reactive.
    let mut changed = true;
    while changed {
        changed = false;
        for b in &cfg.blocks {
            for p in &b.params {
                // a block arg is reactive if reachable from reactive (approx: leave
                // to operand propagation via terminators below)
                let _ = p;
            }
            for ins in &b.instrs {
                if let Some(res) = ins.result {
                    if !reactive.contains(&res)
                        && !stable.contains(&res)
                        && ins.op.operands().iter().any(|o| reactive.contains(o))
                    {
                        reactive.insert(res);
                        changed = true;
                    }
                }
            }
            // propagate through terminator block-args
            let edges: Vec<(crate::cfg::BlockId, Vec<Value>)> = match &b.term {
                crate::cfg::Term::Br(t, a) => vec![(*t, a.clone())],
                crate::cfg::Term::CondBr {
                    then_block,
                    then_args,
                    else_block,
                    else_args,
                    ..
                } => {
                    vec![
                        (*then_block, then_args.clone()),
                        (*else_block, else_args.clone()),
                    ]
                }
                _ => vec![],
            };
            for (succ, args) in edges {
                let params = cfg.block(succ).params.clone();
                for (param, a) in params.iter().zip(args) {
                    if reactive.contains(&a) && !reactive.contains(param) && !stable.contains(param)
                    {
                        reactive.insert(*param);
                        changed = true;
                    }
                }
            }
        }
    }
    reactive
}

/// A reactive scope plus its inferred memoization interface: the values it reads
/// from outside (cache key) and the values it produces that are used later
/// (cache outputs).
#[derive(Debug, Clone)]
pub struct ScopeInfo {
    pub scope: Scope,
    /// Reactive inputs: operands used inside the scope but defined before it.
    pub deps: Vec<Value>,
    /// Outputs: values produced in the scope and used after it (or returned).
    pub outputs: Vec<Value>,
}

// =============================================================================
// PruneNonEscapingScopes (port of React's prune_non_escaping_scopes.rs)
// =============================================================================
//
// React's escape analysis: a reactive scope is only worth memoizing if one of
// its values transitively escapes — i.e. flows into a `return` value or a hook
// argument. Our previous predicate ("used by any later instruction or *any*
// terminator", where terminator operands include the CondBr condition and every
// branch argument) over-approximated escape massively and kept ~86 scopes React
// prunes. This port computes React's actual memoized set and uses it to (a)
// decide which scopes survive and (b) restrict each surviving scope's outputs to
// the values that genuinely escape (escaping ∩ memoized).
//
// Port notes / divergences (documented, sound):
//  - We operate over SSA `Value`s rather than React's `DeclarationId`s. SSA has
//    no `LoadLocal` indirection (every value is its own definition), so the
//    `definitions`/`resolve` map collapses to identity here.
//  - We have no function-signature DB, so we never apply React's `noAlias`
//    optimization — every call/allocation is assumed to alias its operands
//    (over-memoize, never under-memoize), matching the spec's mandate.
//  - Block arguments (SSA phis) are treated as `Conditional` values whose
//    dependencies are the incoming branch arguments from every predecessor.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MemoLevel {
    Memoized,
    Conditional,
    Unmemoized,
    Never,
}

fn join_levels(a: MemoLevel, b: MemoLevel) -> MemoLevel {
    use MemoLevel::*;
    if a == Memoized || b == Memoized {
        Memoized
    } else if a == Conditional || b == Conditional {
        Conditional
    } else if a == Unmemoized || b == Unmemoized {
        Unmemoized
    } else {
        Never
    }
}

/// The base memoization level a defining `Op` assigns to its result value.
///
/// `is_jsx`/`is_hook` distinguish JSX-element and hook calls, but in React's
/// `getMemoizationLevel` *every* allocating instruction — object/array literals,
/// `new`, and any call (JSX, hook, or plain) — gets the unconditional `Memoized`
/// level: an escaping allocation is always cached so its identity is stable
/// across renders. Verified against the official compiler: `return foo(p.a)`,
/// `return [p.a]`, `return {x: p.a}`, `return new Foo(p.a)`, and even `return
/// [1, 2]` / `return foo()` (no deps, sentinel-guarded) all memoize. Only
/// member/computed loads are `Conditional` (memoize iff their base is memoized),
/// and scalars (`const`/global/binary/unary) are `Never`.
fn op_level(op: &crate::cfg::Op, is_jsx: bool, is_value_hook: bool) -> MemoLevel {
    use crate::cfg::Op;
    let _ = is_jsx; // jsx calls memoize like any allocation; kept for callers
    match op {
        // A "value hook" call (`useState`/`useReducer`/`useContext`/`useRef`/a
        // custom hook) is NOT itself a memoized scope — React executes it inline
        // each render and caches the values *derived* from its result, not the
        // call. (`useMemo`/`useCallback` are the exception: their result IS a
        // cached value, so they fall through to `Memoized`.)
        Op::Call { .. } if is_value_hook => MemoLevel::Never,
        // Allocations (literals, `new`, calls) escaping a function are always
        // memoized — this is the core of what the React Compiler caches.
        Op::MakeObject(_) | Op::MakeArray(_) | Op::Call { .. } => MemoLevel::Memoized,
        // Member reads / loads are conditional: memoize only if a dep is memoized.
        Op::Member { .. } => MemoLevel::Conditional,
        // Cheaply comparable scalars never need memoization on their own.
        Op::Const(_) | Op::Global(_) | Op::Bin(..) | Op::Un(..) => MemoLevel::Never,
        // Stores have no escaping result identity; treat as Never (no result
        // value is consumed downstream as a memoized identity).
        Op::StoreMember { .. } => MemoLevel::Never,
        // Pre-SSA forms must have been eliminated before scope analysis.
        Op::ReadVar(_) | Op::WriteVar(..) => MemoLevel::Unmemoized,
    }
}

/// Is the call value `callee` a JSX-element constructor
/// (`createElement`/`jsx`/`jsxs`/`jsxDEV`)? Mirrors `mutability.rs`'s
/// `is_pure_call`.
fn is_jsx_callee(cfg: &Cfg, callee: Value) -> bool {
    use crate::cfg::{MemberKey, Op};
    for b in &cfg.blocks {
        for ins in &b.instrs {
            if ins.result == Some(callee) {
                return match &ins.op {
                    Op::Member {
                        prop: MemberKey::Static(name),
                        ..
                    } => {
                        matches!(name.as_str(), "createElement" | "jsx" | "jsxs" | "jsxDEV")
                    }
                    Op::Global(name) => {
                        matches!(name.as_str(), "jsx" | "jsxs" | "_jsx" | "_jsxs")
                    }
                    _ => false,
                };
            }
        }
    }
    false
}

/// Is the call value `callee` a React hook? The callee resolves to a global or
/// member named `use[A-Z]…` (`useState`, `useMemo`, `useFoo`, `React.useFoo`).
fn is_hook_callee(cfg: &Cfg, callee: Value) -> bool {
    use crate::cfg::{MemberKey, Op};
    for b in &cfg.blocks {
        for ins in &b.instrs {
            if ins.result == Some(callee) {
                return match &ins.op {
                    Op::Global(name) => is_hook_name(name),
                    Op::Member {
                        prop: MemberKey::Static(name),
                        ..
                    } => is_hook_name(name),
                    _ => false,
                };
            }
        }
    }
    false
}

/// Is the call value `callee` a hook that returns a STABLE value React never
/// memoizes or treats as a dependency? `useRef` (the `{current}` box) is the
/// canonical one; its identity is guaranteed stable across renders.
fn is_stable_hook_callee(cfg: &Cfg, callee: Value) -> bool {
    use crate::cfg::{MemberKey, Op};
    for b in &cfg.blocks {
        for ins in &b.instrs {
            if ins.result == Some(callee) {
                return matches!(
                    &ins.op,
                    Op::Global(name) | Op::Member { prop: MemberKey::Static(name), .. }
                        if name == "useRef"
                );
            }
        }
    }
    false
}

/// Is `callee` a `useMemo`/`useCallback` call? These hooks produce a *cached*
/// value (React keeps the memoization), unlike other hooks whose call result is
/// not itself a scope.
fn is_memo_hook_callee(cfg: &Cfg, callee: Value) -> bool {
    use crate::cfg::{MemberKey, Op};
    for b in &cfg.blocks {
        for ins in &b.instrs {
            if ins.result == Some(callee) {
                return matches!(
                    &ins.op,
                    Op::Global(name) | Op::Member { prop: MemberKey::Static(name), .. }
                        if name == "useMemo" || name == "useCallback"
                );
            }
        }
    }
    false
}

/// The set of STABLE values React never treats as dependencies: `useRef`
/// results, and the setter/dispatch element (`[1]`) of a `useState`/`useReducer`
/// destructure (`const [x, setX] = useState()` — `setX` is stable). The setter
/// is recognised structurally as the computed `[1]` member of a state-hook call
/// result (how our array-pattern lowering binds it).
fn stable_values(cfg: &Cfg) -> std::collections::HashSet<Value> {
    use crate::cfg::{Const, MemberKey, Op};
    use std::collections::{HashMap, HashSet};
    let mut def_op: HashMap<Value, &Op> = HashMap::new();
    let mut one_vals: HashSet<Value> = HashSet::new();
    for b in &cfg.blocks {
        for ins in &b.instrs {
            if let Some(r) = ins.result {
                def_op.insert(r, &ins.op);
                if let Op::Const(Const::Num(bits)) = &ins.op {
                    if f64::from_bits(*bits) == 1.0 {
                        one_vals.insert(r);
                    }
                }
            }
        }
    }
    let is_state_hook = |callee: Value| -> bool {
        matches!(def_op.get(&callee),
            Some(Op::Global(n)) | Some(Op::Member { prop: MemberKey::Static(n), .. })
                if n == "useState" || n == "useReducer")
    };
    let mut stable = HashSet::new();
    for b in &cfg.blocks {
        for ins in &b.instrs {
            let Some(r) = ins.result else { continue };
            match &ins.op {
                Op::Call { callee, .. } if is_stable_hook_callee(cfg, *callee) => {
                    stable.insert(r);
                }
                Op::Member {
                    obj,
                    prop: MemberKey::Computed(c),
                } if one_vals.contains(c) => {
                    if matches!(def_op.get(obj), Some(Op::Call { callee, .. }) if is_state_hook(*callee))
                    {
                        stable.insert(r);
                    }
                }
                _ => {}
            }
        }
    }
    stable
}

/// Is `name` a React hook name? React keys on `use[A-Z]`-style naming for
/// non-builtin hooks; we have no hook registry, so we mirror that lexical rule.
fn is_hook_name(name: &str) -> bool {
    let bytes = name.as_bytes();
    if bytes.len() >= 4 && &name[..3] == "use" {
        // `use` followed by an uppercase letter (useState, useMemo, useFoo…).
        bytes[3].is_ascii_uppercase()
    } else {
        false
    }
}

/// A node in the escape/memoization graph for one SSA value.
struct EscapeNode {
    level: MemoLevel,
    deps: std::collections::HashSet<Value>,
    scopes: std::collections::HashSet<usize>,
    seen: bool,
    memoized: bool,
}

/// Result of the escape analysis: the set of values that React would memoize,
/// and the set of values that escape (flow into a return or a hook arg).
pub struct EscapeResult {
    pub memoized: std::collections::HashSet<Value>,
    pub escaping: std::collections::HashSet<Value>,
}

/// Port of `PruneNonEscapingScopes`: build the value-level memoization graph,
/// seed escape roots from returns + hook args, run the memoized-set DFS
/// (`compute_memoized_identifiers` + `force_memoize_scope_dependencies`), and
/// return the memoized + escaping sets. `scope_values` / `scope_deps` describe
/// the raw inferred scopes (value-set + reactive dependency-set) so that
/// `force_memoize_scope_dependencies` can walk a memoized value's scopes' deps.
pub fn prune_non_escaping(
    cfg: &Cfg,
    scope_values: &[Vec<Value>],
    scope_deps: &[std::collections::HashSet<Value>],
) -> EscapeResult {
    use std::collections::{HashMap, HashSet};

    let mut nodes: HashMap<Value, EscapeNode> = HashMap::new();
    let ensure = |nodes: &mut HashMap<Value, EscapeNode>, v: Value| {
        nodes.entry(v).or_insert_with(|| EscapeNode {
            level: MemoLevel::Never,
            deps: HashSet::new(),
            scopes: HashSet::new(),
            seen: false,
            memoized: false,
        });
    };

    // Params are inputs (Never, no deps).
    for p in &cfg.params {
        ensure(&mut nodes, *p);
    }

    // Locally-allocated reference values (object/array literal or call result):
    // the values a callee can capture/freeze. Used to decide which ordinary-call
    // arguments to promote to `Memoized` (see the aliasing note below).
    let mut is_alloc: HashSet<Value> = HashSet::new();
    for b in &cfg.blocks {
        for ins in &b.instrs {
            if let Some(res) = ins.result {
                if matches!(
                    ins.op,
                    crate::cfg::Op::MakeObject(_)
                        | crate::cfg::Op::MakeArray(_)
                        | crate::cfg::Op::Call { .. }
                ) {
                    is_alloc.insert(res);
                }
            }
        }
    }

    // Build nodes from instructions: result value gets the op's level + its
    // operands as dependencies. Mirror React's aliasing rule: operands captured
    // into an allocation/call are themselves bumped to `Memoized` (their identity
    // is captured into the escaping value).
    for b in &cfg.blocks {
        for ins in &b.instrs {
            let operands = ins.op.operands();
            if let Some(res) = ins.result {
                ensure(&mut nodes, res);
                let is_jsx = matches!(&ins.op, crate::cfg::Op::Call { callee, .. } if is_jsx_callee(cfg, *callee));
                // A "value hook" call is a hook other than useMemo/useCallback
                // (which produce cached values): its result is not its own scope.
                let is_value_hook = matches!(&ins.op, crate::cfg::Op::Call { callee, .. }
                    if is_hook_callee(cfg, *callee) && !is_memo_hook_callee(cfg, *callee));
                let lvl = op_level(&ins.op, is_jsx, is_value_hook);
                {
                    let n = nodes.get_mut(&res).unwrap();
                    n.level = join_levels(n.level, lvl);
                    for o in &operands {
                        if *o != res {
                            n.deps.insert(*o);
                        }
                    }
                }
                // Aliasing (React: bump operands to `Memoized` when *mutably*
                // captured, `op.effect.is_mutable()`). We have no per-operand
                // effect info, so we approximate by op kind:
                //  - A *call* may mutate/capture its arguments (and a JSX call's
                //    props object must propagate its memoization to the element),
                //    so we bump call arguments. This is React's behaviour for the
                //    mutable-effect operands of `CallExpression`.
                //  - An array/object *literal* merely reads its elements (a frozen
                //    capture, e.g. `[x[0]]`), so its operands are NOT bumped —
                //    bumping them over-memoizes returned bare literals that React
                //    leaves alone.
                if let crate::cfg::Op::Call { callee, args } = &ins.op {
                    // React bumps a call/JSX operand to `Memoized` exactly when it
                    // is *mutably* captured (`op.effect.is_mutable()`): the callee
                    // freezes / holds onto the argument. We approximate which calls
                    // freeze their args:
                    //  - JSX / hook calls always capture their operands.
                    //  - A call to a *locally-defined* function — whose callee
                    //    lowers to `Const(Undef)` (the hoisted local binding,
                    //    distinct from an imported/undeclared `Global(name)`) — is
                    //    treated as possibly-freezing: per the escape-analysis
                    //    mandate we assume aliasing where unknown and never
                    //    under-memoize. We bump only its *locally-allocated* args
                    //    (object/array literal or call result), the values a callee
                    //    can capture and that React memoizes when subsequently
                    //    returned (the `useFreeze(a)` / `frozen-after-alias` shape).
                    //    A call to an imported/global function (`bar(a)`) is NOT
                    //    bumped: React leaves `bar(a); return a` un-memoized.
                    let callee_is_local = matches!(
                        def_op_of(cfg, *callee),
                        Some(crate::cfg::Op::Const(crate::cfg::Const::Undef))
                    );
                    for o in args {
                        if *o == res || *o == *callee {
                            continue;
                        }
                        let bump = is_jsx
                            || is_hook_callee(cfg, *callee)
                            || (callee_is_local && is_alloc.contains(o));
                        if bump {
                            ensure(&mut nodes, *o);
                            let n = nodes.get_mut(o).unwrap();
                            n.level = join_levels(n.level, MemoLevel::Memoized);
                        }
                    }
                }
            } else {
                // No result (e.g. StoreMember used for effect): still register
                // operands so the graph is total.
                for o in &operands {
                    ensure(&mut nodes, *o);
                }
            }
        }
    }

    // Block-args (phis): Conditional, with dependencies = incoming branch args.
    for b in &cfg.blocks {
        for (i, param) in b.params.iter().enumerate() {
            ensure(&mut nodes, *param);
            let n = nodes.get_mut(param).unwrap();
            n.level = join_levels(n.level, MemoLevel::Conditional);
            // collect after releasing borrow
            let _ = i;
        }
    }
    // Wire phi dependencies from each predecessor's branch args.
    for b in &cfg.blocks {
        let edges: Vec<(crate::cfg::BlockId, Vec<Value>)> = match &b.term {
            crate::cfg::Term::Br(t, a) => vec![(*t, a.clone())],
            crate::cfg::Term::CondBr {
                then_block,
                then_args,
                else_block,
                else_args,
                ..
            } => vec![
                (*then_block, then_args.clone()),
                (*else_block, else_args.clone()),
            ],
            _ => vec![],
        };
        for (succ, args) in edges {
            let params = cfg.block(succ).params.clone();
            for (param, a) in params.iter().zip(args) {
                if *param != a {
                    ensure(&mut nodes, *param);
                    nodes.get_mut(param).unwrap().deps.insert(a);
                }
            }
        }
    }

    // Associate each value with the scope(s) it belongs to.
    for (si, vals) in scope_values.iter().enumerate() {
        for v in vals {
            ensure(&mut nodes, *v);
            nodes.get_mut(v).unwrap().scopes.insert(si);
        }
    }

    // Seed escape roots.
    let mut escaping: HashSet<Value> = HashSet::new();
    for b in &cfg.blocks {
        // (a) returned values.
        if let crate::cfg::Term::Ret(Some(v)) = &b.term {
            escaping.insert(*v);
        }
        // (b) hook-call arguments: callee is a global named `use[A-Z]`.
        for ins in &b.instrs {
            if let crate::cfg::Op::Call { callee, args } = &ins.op {
                let is_hook = matches!(
                    instr_global_name(cfg, *callee),
                    Some(name) if is_hook_name(&name)
                );
                if is_hook {
                    for a in args {
                        escaping.insert(*a);
                    }
                }
            }
        }
    }
    for v in &escaping {
        ensure(&mut nodes, *v);
    }

    // ---- compute_memoized_identifiers DFS ----
    let mut memoized: HashSet<Value> = HashSet::new();
    // scope_seen mirrors React's ScopeNode.seen flag.
    let mut scope_seen = vec![false; scope_values.len()];

    fn visit(
        v: Value,
        force: bool,
        nodes: &mut HashMap<Value, EscapeNode>,
        scope_seen: &mut [bool],
        scope_deps: &[HashSet<Value>],
        memoized: &mut HashSet<Value>,
    ) -> bool {
        let (level, seen) = match nodes.get(&v) {
            Some(n) => (n.level, n.seen),
            None => return false,
        };
        if seen {
            return nodes.get(&v).unwrap().memoized;
        }
        nodes.get_mut(&v).unwrap().seen = true;
        nodes.get_mut(&v).unwrap().memoized = false;

        let mut deps: Vec<Value> = nodes.get(&v).unwrap().deps.iter().copied().collect();
        deps.sort();
        let mut has_memoized_dep = false;
        for d in deps {
            let m = visit(d, false, nodes, scope_seen, scope_deps, memoized);
            has_memoized_dep |= m;
        }

        let should = level == MemoLevel::Memoized
            || (level == MemoLevel::Conditional && (has_memoized_dep || force))
            || (level == MemoLevel::Unmemoized && force);
        if should {
            nodes.get_mut(&v).unwrap().memoized = true;
            memoized.insert(v);
            let mut scopes: Vec<usize> = nodes.get(&v).unwrap().scopes.iter().copied().collect();
            scopes.sort();
            for s in scopes {
                force_memoize_scope(s, nodes, scope_seen, scope_deps, memoized);
            }
        }
        nodes.get(&v).unwrap().memoized
    }

    fn force_memoize_scope(
        s: usize,
        nodes: &mut HashMap<Value, EscapeNode>,
        scope_seen: &mut [bool],
        scope_deps: &[HashSet<Value>],
        memoized: &mut HashSet<Value>,
    ) {
        if scope_seen[s] {
            return;
        }
        scope_seen[s] = true;
        let mut deps: Vec<Value> = scope_deps[s].iter().copied().collect();
        deps.sort();
        for d in deps {
            visit(d, true, nodes, scope_seen, scope_deps, memoized);
        }
    }

    let mut roots: Vec<Value> = escaping.iter().copied().collect();
    roots.sort();
    for v in roots {
        visit(
            v,
            false,
            &mut nodes,
            &mut scope_seen,
            scope_deps,
            &mut memoized,
        );
    }

    EscapeResult { memoized, escaping }
}

/// The defining `Op` of value `v`, if any (None for params/block-args).
fn def_op_of(cfg: &Cfg, v: Value) -> Option<&crate::cfg::Op> {
    for b in &cfg.blocks {
        for ins in &b.instrs {
            if ins.result == Some(v) {
                return Some(&ins.op);
            }
        }
    }
    None
}

/// Resolve the global/free-identifier name a callee value refers to, if its
/// defining op is `Op::Global`. Used to recognize hook calls.
fn instr_global_name(cfg: &Cfg, v: Value) -> Option<String> {
    for b in &cfg.blocks {
        for ins in &b.instrs {
            if ins.result == Some(v) {
                if let crate::cfg::Op::Global(name) = &ins.op {
                    return Some(name.clone());
                }
                return None;
            }
        }
    }
    None
}

/// Infer each scope's dependencies and outputs — the shape of the memo cache
/// block React would emit (`const [outputs] = useMemo(() => {…}, [deps])`).
pub fn analyze(cfg: &Cfg, r: &Ranges) -> Vec<ScopeInfo> {
    use std::collections::{HashMap, HashSet};
    let scopes = merge_overlapping_scopes(infer(cfg, r));
    let stable = stable_values(cfg);
    let reactive = reactive_values(cfg, &stable);

    // Per-value: defining-op operands, and use sites (Some(user) or None=external
    // i.e. terminator/return). Also the const-defined values.
    let mut operands: HashMap<Value, Vec<Value>> = HashMap::new();
    let mut users: HashMap<Value, Vec<Option<Value>>> = HashMap::new();
    let mut const_vals: HashSet<Value> = HashSet::new();
    let mut def_op: HashMap<Value, &crate::cfg::Op> = HashMap::new();
    for b in &cfg.blocks {
        for ins in &b.instrs {
            if let Some(res) = ins.result {
                operands.insert(res, ins.op.operands());
                def_op.insert(res, &ins.op);
                if matches!(ins.op, crate::cfg::Op::Const(_)) {
                    const_vals.insert(res);
                }
                for o in ins.op.operands() {
                    users.entry(o).or_default().push(Some(res));
                }
            }
        }
        for o in term_operands(cfg, b.id) {
            users.entry(o).or_default().push(None);
        }
    }

    // --- PruneNonEscapingScopes (run BEFORE the merge fold) ---------------
    // Compute React's memoized + escaping sets over the raw inferred scopes.
    // A raw scope's reactive dependency-set = reactive, non-const operands of
    // its values that are defined outside the scope.
    let raw_values: Vec<Vec<Value>> = scopes.iter().map(|sc| sc.values.clone()).collect();
    let raw_deps: Vec<HashSet<Value>> = scopes
        .iter()
        .map(|sc| {
            let set: HashSet<Value> = sc.values.iter().copied().collect();
            let mut d = HashSet::new();
            for v in &sc.values {
                for o in operands.get(v).map(|x| x.as_slice()).unwrap_or(&[]) {
                    if !set.contains(o) && reactive.contains(o) && !const_vals.contains(o) {
                        d.insert(*o);
                    }
                }
            }
            d
        })
        .collect();
    let escape = prune_non_escaping(cfg, &raw_values, &raw_deps);

    // Transient values: the props object of a `createElement`/`jsx` call. Its
    // identity is never observed except through the element, which rebuilds
    // whenever any attribute changes — so it can always merge into the element
    // scope, even when its deps aren't a subset (React does the same).
    let mut create_member: HashSet<Value> = HashSet::new();
    for b in &cfg.blocks {
        for ins in &b.instrs {
            if let (
                Some(res),
                crate::cfg::Op::Member {
                    prop: crate::cfg::MemberKey::Static(n),
                    ..
                },
            ) = (ins.result, &ins.op)
            {
                if matches!(n.as_str(), "createElement" | "jsx" | "jsxs" | "jsxDEV") {
                    create_member.insert(res);
                }
            }
        }
    }
    let mut transient: HashSet<Value> = HashSet::new();
    // Structural props->element edges: (props operand P, element call result E)
    // for each `createElement`/`jsx`/`jsxs`/`jsxDEV` call. The props object is
    // args[1]; it has no observable identity apart from the element it feeds, so
    // it always belongs in the element's scope — even when neither has reactive
    // deps (constant JSX). This is the structural counterpart to the reactive
    // dependency-based fold below, which never fires for constant inputs.
    let mut props_to_element: Vec<(Value, Value)> = Vec::new();
    for b in &cfg.blocks {
        for ins in &b.instrs {
            if let crate::cfg::Op::Call { callee, args } = &ins.op {
                if create_member.contains(callee) {
                    if let Some(props) = args.get(1) {
                        transient.insert(*props);
                        if let Some(elem) = ins.result {
                            props_to_element.push((*props, elem));
                        }
                    }
                }
            }
        }
    }

    // Working scopes as value-sets.
    struct Work {
        values: HashSet<Value>,
        mutable: bool,
        start: Point,
        end: Point,
    }
    let mut work: Vec<Option<Work>> = scopes
        .iter()
        .map(|sc| {
            Some(Work {
                values: sc.values.iter().copied().collect(),
                mutable: sc.mutable,
                start: sc.start,
                end: sc.end,
            })
        })
        .collect();
    let mut owner: HashMap<Value, usize> = HashMap::new();
    for (i, w) in work.iter().enumerate() {
        if let Some(w) = w {
            for v in &w.values {
                owner.insert(*v, i);
            }
        }
    }

    // Structural transient fold: fold a JSX props object into the element that
    // consumes it, keyed only on the `call.args[1] == props` edge — independent
    // of any reactive dependency. React always memoizes a constant element as a
    // single sentinel-guarded scope (cache=1), so the props transient and the
    // element must share one scope even when both have empty deps.
    for &(props, elem) in &props_to_element {
        // The props value must be a transient (a props object literal we own),
        // and observed *only* through this element call. If it is read anywhere
        // else, its identity escapes and we cannot soundly merge it away.
        if !transient.contains(&props) {
            continue;
        }
        let escapes = users
            .get(&props)
            .map(|us| us.iter().any(|u| *u != Some(elem)))
            .unwrap_or(false);
        if escapes {
            continue;
        }
        let cp = match owner.get(&props) {
            Some(&cp) => cp,
            None => continue,
        };
        let ce = match owner.get(&elem) {
            Some(&ce) if ce != cp => ce,
            _ => continue,
        };
        // Never merge into a mutable scope (that is a forced boundary), and never
        // fold a transient into its own producer (the producer is `cp` itself).
        if work[ce].as_ref().map(|w| w.mutable).unwrap_or(true) {
            continue;
        }
        // Merge producer scope `cp` (the props object) into consumer `ce` (the
        // element), reusing the standard merge body.
        let from = match work[cp].take() {
            Some(f) => f,
            None => continue,
        };
        let to = work[ce].as_mut().unwrap();
        for v in &from.values {
            owner.insert(*v, ce);
        }
        to.values.extend(from.values);
        to.start = to.start.min(from.start);
        to.end = to.end.max(from.end);
    }

    // MergeScopesThatInvalidateTogether (matching the React Compiler): fold a
    // consumer scope `C` into its producer `P` when `C` has **exactly one**
    // reactive dependency — `P`'s output. Then `C` invalidates exactly when `P`
    // does. (React is deliberately conservative: a consumer with two deps stays
    // separate even if one is a subset, to keep both producers' identities.) A
    // JSX props object (a transient with no independent identity) also folds into
    // its element.
    let scope_deps = |values: &HashSet<Value>| -> HashSet<Value> {
        let mut d = HashSet::new();
        for v in values {
            for o in operands.get(v).map(|x| x.as_slice()).unwrap_or(&[]) {
                if !values.contains(o) && reactive.contains(o) && !const_vals.contains(o) {
                    d.insert(*o);
                }
            }
        }
        d
    };
    let scope_owned_inputs =
        |values: &HashSet<Value>, owner: &HashMap<Value, usize>| -> HashSet<Value> {
            let mut d = HashSet::new();
            for v in values {
                for o in operands.get(v).map(|x| x.as_slice()).unwrap_or(&[]) {
                    if !values.contains(o) && !const_vals.contains(o) && owner.contains_key(o) {
                        d.insert(*o);
                    }
                }
            }
            d
        };
    let mut changed = true;
    while changed {
        changed = false;
        'scan: for c in 0..work.len() {
            // A lone transient (JSX props object) must only be absorbed by its
            // consumer (the element); it must never merge into its producer.
            let (cdeps, owned_inputs) = match &work[c] {
                Some(w)
                    if !w.mutable
                        && !(w.values.len() == 1
                            && transient.contains(w.values.iter().next().unwrap())) =>
                {
                    (scope_deps(&w.values), scope_owned_inputs(&w.values, &owner))
                }
                _ => continue,
            };
            let mut producer: Option<usize> = None;
            let mut all_inputs_from_one_producer = !cdeps.is_empty() || !owned_inputs.is_empty();
            let mut sources: HashSet<Value> = cdeps.iter().copied().collect();
            sources.extend(owned_inputs.iter().copied());
            for &d in &sources {
                match owner.get(&d).copied() {
                    Some(p) if p != c => match producer {
                        Some(prev) if prev != p => {
                            all_inputs_from_one_producer = false;
                            break;
                        }
                        Some(_) => {}
                        None => producer = Some(p),
                    },
                    _ => {
                        all_inputs_from_one_producer = false;
                        break;
                    }
                }
            }
            if let Some(p) = producer {
                let producer_is_mutable = work[p].as_ref().map(|w| w.mutable).unwrap_or(true);
                let can_merge = if sources.iter().any(|d| transient.contains(d)) {
                    true
                } else if producer_is_mutable {
                    // React folds a pure consumer into a mutable producer when all
                    // of the consumer's inputs are produced by that same scope
                    // and there are no independent reactive deps. This covers
                    // local captures (`a`/`b` allocated, then returned only
                    // through JSX) without collapsing dep-bearing chains such as
                    // array mutation followed by property reads.
                    cdeps.is_empty() && all_inputs_from_one_producer
                } else {
                    // Pure producer: retain the conservative existing rule.
                    cdeps.len() == 1
                };
                if can_merge {
                    let from = work[c].take().unwrap();
                    let to = work[p].as_mut().unwrap();
                    for v in &from.values {
                        owner.insert(*v, p);
                    }
                    to.values.extend(from.values);
                    to.start = to.start.min(from.start);
                    to.end = to.end.max(from.end);
                    changed = true;
                    break 'scan;
                }
            }
        }
    }

    // MergeConsecutiveScopes (React's `mergeReactiveScopesThatInvalidateTogether`,
    // the equal-dependency case): consecutive reactive scopes in the same block
    // whose dependency sets are EQUAL invalidate together, so React merges them
    // into one. The classic case is two adjacent no-dependency allocations
    // (`return {session_id: getNumber()}` — the call scope and the object scope
    // both have empty deps → one cache slot, not two). We walk surviving scopes in
    // program order, folding a pure scope into the immediately preceding pure
    // scope of the same block with the same deps; a mutable scope is a barrier
    // that resets the run (its guard must stay its own).
    let value_block: HashMap<Value, crate::cfg::BlockId> = {
        let mut m = HashMap::new();
        for b in &cfg.blocks {
            for ins in &b.instrs {
                if let Some(res) = ins.result {
                    m.insert(res, b.id);
                }
            }
        }
        m
    };
    let scope_block = |w: &Work| -> Option<crate::cfg::BlockId> {
        w.values
            .iter()
            .min_by_key(|v| r.def.get(v).copied().unwrap_or(0))
            .and_then(|v| value_block.get(v).copied())
    };
    let mut order: Vec<usize> = (0..work.len()).filter(|&i| work[i].is_some()).collect();
    order.sort_by_key(|&i| work[i].as_ref().map(|w| (w.start, w.end)).unwrap_or((0, 0)));
    let mut prev: Option<usize> = None;
    for idx in order {
        let (mutable, block, deps) = match &work[idx] {
            Some(w) => (w.mutable, scope_block(w), scope_deps(&w.values)),
            None => continue,
        };
        if mutable {
            prev = None; // barrier: a forced scope keeps its own guard
            continue;
        }
        match prev {
            Some(p) => {
                let (pblock, pdeps) = {
                    let w = work[p].as_ref().unwrap();
                    (scope_block(w), scope_deps(&w.values))
                };
                if pblock == block && pdeps == deps {
                    // Merge `idx` into `p`; `p` remains the run head.
                    let from = work[idx].take().unwrap();
                    let to = work[p].as_mut().unwrap();
                    for v in &from.values {
                        owner.insert(*v, p);
                    }
                    to.values.extend(from.values);
                    to.start = to.start.min(from.start);
                    to.end = to.end.max(from.end);
                } else {
                    prev = Some(idx);
                }
            }
            None => prev = Some(idx),
        }
    }

    // Compute deps/outputs per surviving scope from its value-set.
    let mut out = Vec::new();
    for w in work.into_iter().flatten() {
        let values = &w.values;
        let mut deps: Vec<Value> = Vec::new();
        // Dedup dependencies by reactive ACCESS PATH, not SSA value: two reads of
        // `props.router.location` are one dependency (React's rule), so the cache
        // key count matches React's.
        let mut seen: HashSet<(Value, Vec<String>)> = HashSet::new();
        let mut sorted_vals: Vec<Value> = values.iter().copied().collect();
        sorted_vals.sort_by_key(|v| r.def.get(v).copied().unwrap_or(0));
        for v in &sorted_vals {
            for o in operands.get(v).map(|x| x.as_slice()).unwrap_or(&[]) {
                if !values.contains(o)
                    && reactive.contains(o)
                    && !const_vals.contains(o)
                    && seen.insert(access_path_key(&def_op, *o))
                {
                    deps.push(*o);
                }
            }
        }
        // Output = a scope value that escapes the scope (used outside it) AND is
        // in React's `memoized` set (transitively reachable from a return value
        // or a hook argument). The structural "used outside" predicate alone is
        // the over-approximation that kept the ~86 spurious scopes (it counted a
        // value flowing as a CondBr condition or branch argument as escaping);
        // intersecting with `memoized` restricts outputs to the values React
        // actually caches. A scope whose values are none of these produces empty
        // outputs and is therefore pruned by the `!outputs.is_empty()` emission
        // filter — matching React's "keep iff declares a memoized value" rule.
        let outputs: Vec<Value> = sorted_vals
            .iter()
            .copied()
            .filter(|v| {
                let escapes_scope = users
                    .get(v)
                    .map(|us| {
                        us.iter().any(|u| match u {
                            None => true,
                            Some(uv) => !values.contains(uv),
                        })
                    })
                    .unwrap_or(false);
                escapes_scope && escape.memoized.contains(v)
            })
            .collect();
        deps.sort();
        let scope = Scope {
            start: w.start,
            end: w.end,
            values: sorted_vals,
            mutable: w.mutable,
        };
        out.push(ScopeInfo {
            scope,
            deps,
            outputs,
        });
    }
    out.sort_by_key(scope_info_sort_key);
    out
}

fn scope_info_sort_key(i: &ScopeInfo) -> (Point, Point, Vec<u32>, Vec<u32>, Vec<u32>) {
    let mut values: Vec<u32> = i.scope.values.iter().map(|v| v.0).collect();
    let mut deps: Vec<u32> = i.deps.iter().map(|v| v.0).collect();
    let mut outputs: Vec<u32> = i.outputs.iter().map(|v| v.0).collect();
    values.sort();
    deps.sort();
    outputs.sort();
    (i.scope.start, i.scope.end, values, deps, outputs)
}

/// The reactive **access path** of a value: its root value plus the chain of
/// static member names it is read through. `props.router.location` (a chain of
/// `Member` loads from the `props` param) has key `(props, ["router","location"])`.
/// Two different SSA values that read the same path share a key, so a scope that
/// depends on the same path twice (e.g. once in a closure body, once in a
/// `useCallback` dep array) counts it as ONE dependency — React keys cache
/// dependencies by access path, not by SSA value. A value that is not a static
/// member chain is its own atomic path (`(v, [])`).
fn access_path_key(
    def_op: &std::collections::HashMap<Value, &crate::cfg::Op>,
    v: Value,
) -> (Value, Vec<String>) {
    use crate::cfg::{MemberKey, Op};
    let mut props: Vec<String> = Vec::new();
    let mut cur = v;
    let mut guard = 0;
    while let Some(Op::Member {
        obj,
        prop: MemberKey::Static(name),
    }) = def_op.get(&cur).copied()
    {
        props.push(name.clone());
        cur = *obj;
        guard += 1;
        if guard > def_op.len() + 1 {
            break;
        }
    }
    props.reverse();
    (cur, props)
}

fn term_operands(cfg: &Cfg, b: crate::cfg::BlockId) -> Vec<Value> {
    match &cfg.block(b).term {
        crate::cfg::Term::Br(_, a) => a.clone(),
        crate::cfg::Term::CondBr {
            cond,
            then_args,
            else_args,
            ..
        } => {
            let mut v = vec![*cond];
            v.extend(then_args.iter().copied());
            v.extend(else_args.iter().copied());
            v
        }
        crate::cfg::Term::Ret(Some(v)) => vec![*v],
        crate::cfg::Term::Ret(None) | crate::cfg::Term::Unreachable => vec![],
    }
}

pub fn render_info(infos: &[ScopeInfo]) -> String {
    infos
        .iter()
        .map(|i| {
            let deps: Vec<String> = i.deps.iter().map(|v| format!("%{}", v.0)).collect();
            let outs: Vec<String> = i.outputs.iter().map(|v| format!("%{}", v.0)).collect();
            format!(
                "scope [{}..{}] {} out=[{}] deps=[{}]",
                i.scope.start,
                i.scope.end,
                if i.scope.mutable { "mut" } else { "pure" },
                outs.join(", "),
                deps.join(", ")
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

pub fn render(scopes: &[Scope]) -> String {
    scopes
        .iter()
        .map(|s| {
            let vs: Vec<String> = s.values.iter().map(|v| format!("%{}", v.0)).collect();
            format!(
                "scope [{}..{}] {} {{{}}}",
                s.start,
                s.end,
                if s.mutable { "mut" } else { "pure" },
                vs.join(", ")
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}
