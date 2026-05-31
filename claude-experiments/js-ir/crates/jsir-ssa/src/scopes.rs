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

/// Infer reactive scopes from mutable ranges.
pub fn infer(cfg: &Cfg, r: &Ranges) -> Vec<Scope> {
    // Only *allocations* are memoization candidates: object/array/JSX-call
    // results. Member reads, arithmetic, and identifiers are cheap and recomputed
    // (they serve as dependencies), exactly as the React Compiler does.
    let alloc = allocations(cfg);

    // Collect the mutable intervals (end > start), the forced scope boundaries.
    let mut intervals: Vec<(Point, Point)> = Vec::new();
    for (&v, &(s, e)) in &r.range {
        if e > s && alloc.contains(&v) {
            intervals.push((s, e));
        }
    }
    intervals.sort();
    let mut merged: Vec<(Point, Point)> = Vec::new();
    for (s, e) in intervals {
        match merged.last_mut() {
            Some((_, pe)) if s <= *pe => *pe = (*pe).max(e),
            _ => merged.push((s, e)),
        }
    }

    // Assign each allocation to a merged interval (mutable scope) if its range
    // fits inside one; otherwise it is an independent singleton scope.
    let mut scopes: Vec<Scope> = merged.iter().map(|&(s, e)| Scope { start: s, end: e, values: Vec::new(), mutable: true }).collect();

    // Stable iteration over allocations by def point.
    let mut vals: Vec<(Value, (Point, Point))> = r
        .range
        .iter()
        .filter(|(v, _)| alloc.contains(v))
        .map(|(v, rg)| (*v, *rg))
        .collect();
    vals.sort_by_key(|(v, rg)| (rg.0, v.0));

    for (v, (s, e)) in vals {
        // Skip params/block-args (defined "before" the body at point 0 with no
        // body instruction) — they are inputs, not produced values.
        if !is_produced(cfg, v) {
            continue;
        }
        if let Some(scope) = scopes.iter_mut().find(|sc| sc.start <= s && e <= sc.end) {
            scope.values.push(v);
        } else {
            scopes.push(Scope { start: s, end: e, values: vec![v], mutable: false });
        }
    }

    scopes.sort_by_key(|s| (s.start, s.end));
    scopes
}

/// The set of "allocation" values (memoization candidates): object/array
/// literals and call results (which may allocate, e.g. `React.createElement`).
fn allocations(cfg: &Cfg) -> std::collections::HashSet<Value> {
    let mut s = std::collections::HashSet::new();
    for b in &cfg.blocks {
        for ins in &b.instrs {
            if let Some(r) = ins.result {
                if matches!(ins.op, crate::cfg::Op::MakeObject(_) | crate::cfg::Op::MakeArray(_) | crate::cfg::Op::Call { .. }) {
                    s.insert(r);
                }
            }
        }
    }
    s
}

/// Values that derive (transitively) from a parameter — the *reactive* values.
/// Globals and constants are non-reactive (stable across renders) and never
/// become dependencies.
fn reactive_values(cfg: &Cfg) -> std::collections::HashSet<Value> {
    let mut reactive: std::collections::HashSet<Value> = cfg.params.iter().copied().collect();
    // Block params (phis) and any op whose operand is reactive become reactive.
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
                    if !reactive.contains(&res) && ins.op.operands().iter().any(|o| reactive.contains(o)) {
                        reactive.insert(res);
                        changed = true;
                    }
                }
            }
            // propagate through terminator block-args
            let edges: Vec<(crate::cfg::BlockId, Vec<Value>)> = match &b.term {
                crate::cfg::Term::Br(t, a) => vec![(*t, a.clone())],
                crate::cfg::Term::CondBr { then_block, then_args, else_block, else_args, .. } => {
                    vec![(*then_block, then_args.clone()), (*else_block, else_args.clone())]
                }
                _ => vec![],
            };
            for (succ, args) in edges {
                let params = cfg.block(succ).params.clone();
                for (param, a) in params.iter().zip(args) {
                    if reactive.contains(&a) && !reactive.contains(param) {
                        reactive.insert(*param);
                        changed = true;
                    }
                }
            }
        }
    }
    reactive
}

/// Is `v` produced by an instruction (vs a param / block argument input)?
fn is_produced(cfg: &Cfg, v: Value) -> bool {
    if cfg.params.contains(&v) {
        return false;
    }
    for b in &cfg.blocks {
        if b.params.contains(&v) {
            return false;
        }
        for ins in &b.instrs {
            if ins.result == Some(v) {
                return true;
            }
        }
    }
    false
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

/// Infer each scope's dependencies and outputs — the shape of the memo cache
/// block React would emit (`const [outputs] = useMemo(() => {…}, [deps])`).
pub fn analyze(cfg: &Cfg, r: &Ranges) -> Vec<ScopeInfo> {
    use std::collections::{HashMap, HashSet};
    let scopes = infer(cfg, r);
    let reactive = reactive_values(cfg);

    // Per-value: defining-op operands, and use sites (Some(user) or None=external
    // i.e. terminator/return). Also the const-defined values.
    let mut operands: HashMap<Value, Vec<Value>> = HashMap::new();
    let mut users: HashMap<Value, Vec<Option<Value>>> = HashMap::new();
    let mut const_vals: HashSet<Value> = HashSet::new();
    for b in &cfg.blocks {
        for ins in &b.instrs {
            if let Some(res) = ins.result {
                operands.insert(res, ins.op.operands());
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

    // Transient values: the props object of a `createElement`/`jsx` call. Its
    // identity is never observed except through the element, which rebuilds
    // whenever any attribute changes — so it can always merge into the element
    // scope, even when its deps aren't a subset (React does the same).
    let mut create_member: HashSet<Value> = HashSet::new();
    for b in &cfg.blocks {
        for ins in &b.instrs {
            if let (Some(res), crate::cfg::Op::Member { prop: crate::cfg::MemberKey::Static(n), .. }) = (ins.result, &ins.op) {
                if matches!(n.as_str(), "createElement" | "jsx" | "jsxs" | "jsxDEV") {
                    create_member.insert(res);
                }
            }
        }
    }
    let mut transient: HashSet<Value> = HashSet::new();
    for b in &cfg.blocks {
        for ins in &b.instrs {
            if let crate::cfg::Op::Call { callee, args } = &ins.op {
                if create_member.contains(callee) {
                    if let Some(props) = args.get(1) {
                        transient.insert(*props);
                    }
                }
            }
        }
    }

    // Working scopes as value-sets.
    struct Work { values: HashSet<Value>, mutable: bool, start: Point, end: Point }
    let mut work: Vec<Option<Work>> = scopes
        .iter()
        .map(|sc| Some(Work { values: sc.values.iter().copied().collect(), mutable: sc.mutable, start: sc.start, end: sc.end }))
        .collect();
    let mut owner: HashMap<Value, usize> = HashMap::new();
    for (i, w) in work.iter().enumerate() {
        if let Some(w) = w {
            for v in &w.values {
                owner.insert(*v, i);
            }
        }
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
    let mut changed = true;
    while changed {
        changed = false;
        'scan: for c in 0..work.len() {
            // A lone transient (JSX props object) must only be absorbed by its
            // consumer (the element); it must never merge into its producer.
            let cdeps = match &work[c] {
                Some(w)
                    if !w.mutable
                        && !(w.values.len() == 1 && transient.contains(w.values.iter().next().unwrap())) =>
                {
                    scope_deps(&w.values)
                }
                _ => continue,
            };
            for &d in &cdeps {
                // `d` must be the output of another pure scope `p`.
                let p = match owner.get(&d) {
                    Some(&p) if p != c => p,
                    _ => continue,
                };
                if work[p].as_ref().map(|w| w.mutable).unwrap_or(true) {
                    continue;
                }
                // Merge a single-dependency consumer, or a transient props object.
                if !transient.contains(&d) && cdeps.len() != 1 {
                    continue;
                }
                // Merge consumer `c` into producer `p`.
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

    // Compute deps/outputs per surviving scope from its value-set.
    let mut out = Vec::new();
    for w in work.into_iter().flatten() {
        let values = &w.values;
        let mut deps: Vec<Value> = Vec::new();
        let mut seen: HashSet<Value> = HashSet::new();
        let mut sorted_vals: Vec<Value> = values.iter().copied().collect();
        sorted_vals.sort_by_key(|v| r.def.get(v).copied().unwrap_or(0));
        for v in &sorted_vals {
            for o in operands.get(v).map(|x| x.as_slice()).unwrap_or(&[]) {
                if !values.contains(o) && reactive.contains(o) && !const_vals.contains(o) && seen.insert(*o) {
                    deps.push(*o);
                }
            }
        }
        // Output = a scope value used by something outside the scope.
        let outputs: Vec<Value> = sorted_vals
            .iter()
            .copied()
            .filter(|v| {
                users.get(v).map(|us| us.iter().any(|u| match u {
                    None => true,
                    Some(uv) => !values.contains(uv),
                })).unwrap_or(false)
            })
            .collect();
        deps.sort();
        let scope = Scope { start: w.start, end: w.end, values: sorted_vals, mutable: w.mutable };
        out.push(ScopeInfo { scope, deps, outputs });
    }
    out.sort_by_key(|i| (i.scope.start, i.scope.end));
    out
}

fn term_operands(cfg: &Cfg, b: crate::cfg::BlockId) -> Vec<Value> {
    match &cfg.block(b).term {
        crate::cfg::Term::Br(_, a) => a.clone(),
        crate::cfg::Term::CondBr { cond, then_args, else_args, .. } => {
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
            format!("scope [{}..{}] {} {{{}}}", s.start, s.end, if s.mutable { "mut" } else { "pure" }, vs.join(", "))
        })
        .collect::<Vec<_>>()
        .join("\n")
}
