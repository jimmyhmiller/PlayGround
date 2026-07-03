//! PHASE E1 — the termination half of the totality checker.
//!
//! This module is an ELABORATION-SIDE analysis. It does NOT grow the trusted
//! base: a function the checker certifies `Total` is one whose recursion is
//! structural and therefore lowers to a kernel ELIMINATOR (which the kernel then
//! re-checks total-by-construction); a function it cannot certify lowers to an
//! opaque `Fix` (partial) that the kernel never unfolds. So a bug here can only
//! REJECT a program or downgrade it to partial — never make an unsound term pass.
//!
//! The verdict is purely about a function's OWN recursion:
//!   * a recursive call must pass, in the matched-scrutinee position, an argument
//!     that is a STRICT SUBTERM of the matched value (a recursive-field pattern
//!     binder) — positively verified, not assumed;
//!   * for the result to LOWER to an eliminator (so `%total` is kernel-backed),
//!     every other argument must be passed verbatim — accumulator-varying
//!     recursion is terminating but not yet lowerable, so it is honestly
//!     reported `Partial` (Phase E2/E3), never silently called total;
//!   * mutual recursion (a call-graph cycle through ≥2 functions) is not yet
//!     lowerable to a single eliminator, so it is `Partial` (Phase E2);
//!   * a `Total` verdict also requires every function it CALLS to be `Total`
//!     (partiality is contagious) — so a `%total` certificate is airtight.

use crate::rust_surface::Tm;
use std::collections::{HashMap, HashSet};

/// The verdict for one function.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Totality {
    /// Structurally terminating AND lowerable to a kernel eliminator.
    Total,
    /// Terminating by RUNTIME-WITNESSED well-founded descent (Phase E3/B1):
    /// every self-call's measure argument is guarded by a `dlt`-decided
    /// `Lt new old` fact — dlt's `DYes` arm IS the machine compare, so when
    /// the recursive branch runs, the measure has strictly decreased, and
    /// `<` on the packed machine Nat is well-founded (a `u64` cannot
    /// decrease forever). CERTIFIED total, but lowered as `Fix` (there is no
    /// eliminator for this shape) — so unlike `Total`, this certificate
    /// rests on the dlt lowering + this analyzer, not on a kernel re-check
    /// (documented in docs/TRUSTED_BASE.md).
    TotalWf,
    /// Not certifiable as total (with a clear, actionable reason).
    Partial(String),
}

impl Totality {
    pub(crate) fn is_total(&self) -> bool {
        matches!(self, Totality::Total | Totality::TotalWf)
    }
    pub(crate) fn reason(&self) -> Option<&str> {
        match self {
            Totality::Partial(r) => Some(r),
            Totality::Total | Totality::TotalWf => None,
        }
    }
}

/// One call occurrence: the callee name and the surface argument terms.
#[derive(Clone, Debug)]
pub(crate) struct Call {
    pub callee: String,
    pub args: Vec<Tm>,
    /// the RUNTIME-WITNESSED `Lt` facts in scope at this call site: `(e1, e2)`
    /// means the call sits inside the `DYes` arm of a `match dlt(e1, e2)`,
    /// so `e1 < e2` holds whenever this call runs. Facts whose variables are
    /// rebound between the match and the call are dropped by the collector.
    pub lt_facts: Vec<(Tm, Tm)>,
}

/// One `match` arm, distilled for the analysis.
#[derive(Clone, Debug)]
pub(crate) struct ArmInfo {
    /// pattern binders that are STRICT SUBTERMS of the scrutinee (DIRECT recursive
    /// fields of the matched constructor) — the legal structural-decrease targets.
    pub smaller: Vec<String>,
    /// pattern binders that are HIGHER-ORDER recursive fields (`(z…) → data idxs`,
    /// e.g. a W-type's children-function or `Acc`'s accessibility function). A
    /// recursive call whose matched-position argument is `f(args…)` for such an `f`
    /// descends structurally: each `f(args…)` is a strict subterm (Phase 1b /
    /// well-founded recursion). The kernel's IH for such a field is itself a
    /// function (`λz…. elim (f z…)`), so this is genuinely the sub-derivation.
    pub ho_smaller: Vec<String>,
    /// every call (to any function) occurring in this arm's body.
    pub calls: Vec<Call>,
}

/// One function, distilled.
#[derive(Clone, Debug)]
pub(crate) struct FnClauses {
    pub name: String,
    /// full positional parameter names.
    pub params: Vec<String>,
    /// position (in `params`) of the matched scrutinee, if the body is a `match`.
    pub scrut_pos: Option<usize>,
    /// the match arms (empty if the body is not a `match`).
    pub arms: Vec<ArmInfo>,
    /// calls in a non-`match` body (empty if the body is a `match`).
    pub body_calls: Vec<Call>,
    /// is the matched scrutinee a `%builtin Nat`? If so, an accumulator-style fold
    /// (a recursive call that descends on the scrutinee but VARIES other arguments)
    /// is certifiably total: it lowers to a function-typed-motive `NatElim` (Phase
    /// 1a′). For a boxed datatype that lowering isn't available yet, so a varying
    /// argument is still declined (`Partial`).
    pub scrut_is_nat: bool,
}

impl FnClauses {
    fn all_calls(&self) -> impl Iterator<Item = &Call> {
        self.body_calls.iter().chain(self.arms.iter().flat_map(|a| a.calls.iter()))
    }
}

/// The two verdicts for one function.
#[derive(Clone, Debug)]
pub(crate) struct TotInfo {
    /// verdict about the function's OWN recursion only — drives LOWERING (a
    /// structurally-recursive fn lowers to an eliminator; a non-structural one
    /// to an opaque `Fix`). A non-recursive fn is always `Total` here, so it
    /// lowers normally even when it calls a partial helper.
    pub structural: Totality,
    /// the END-TO-END verdict (own recursion AND every callee total) — drives the
    /// `%total` certificate and the reported status. Calling a partial function
    /// makes you non-total, but does not change how you lower.
    pub full: Totality,
}

/// Analyze a whole program's functions. `fn_names` is the set of names that are
/// user functions (so calls to postulates / built-ins are treated as total
/// leaves). Returns the structural and propagated verdict per function.
pub(crate) fn analyze(fns: &[FnClauses]) -> HashMap<String, TotInfo> {
    let fn_names: HashSet<&str> = fns.iter().map(|f| f.name.as_str()).collect();

    // --- call graph over user functions only ---
    let mut edges: HashMap<&str, HashSet<&str>> = HashMap::new();
    for f in fns {
        let e = edges.entry(f.name.as_str()).or_default();
        for c in f.all_calls() {
            if fn_names.contains(c.callee.as_str()) {
                e.insert(fns.iter().find(|g| g.name == c.callee).unwrap().name.as_str());
            }
        }
    }
    // reachability closure (who can each function reach, transitively).
    let reach = transitive_closure(&edges);

    // --- step 1: each function's STRUCTURAL verdict (own recursion only) ---
    let structural: HashMap<String, Totality> =
        fns.iter().map(|f| (f.name.clone(), structural_verdict(f, &reach))).collect();

    // --- step 2: the END-TO-END verdict — partiality is contagious. A function
    // whose own recursion is fine but which CALLS a non-total function is itself
    // not total (so a `%total` certificate is airtight). This does NOT affect
    // lowering — only the certificate/status. Monotone fixpoint. ---
    let mut full = structural.clone();
    loop {
        let mut changed = false;
        for f in fns {
            if !full[&f.name].is_total() {
                continue;
            }
            for c in f.all_calls() {
                if let Some(v) = full.get(&c.callee) {
                    if !v.is_total() {
                        full.insert(
                            f.name.clone(),
                            Totality::Partial(format!("calls `{}`, which is not total", c.callee)),
                        );
                        changed = true;
                        break;
                    }
                }
            }
        }
        if !changed {
            break;
        }
    }

    fns.iter()
        .map(|f| {
            (
                f.name.clone(),
                TotInfo { structural: structural[&f.name].clone(), full: full[&f.name].clone() },
            )
        })
        .collect()
}

/// The structural verdict for one function, considering only its OWN recursion.
fn structural_verdict(f: &FnClauses, reach: &HashMap<String, HashSet<String>>) -> Totality {
    let reaches_self = reach.get(&f.name).map(|s| s.contains(&f.name)).unwrap_or(false);
    if !reaches_self {
        // not part of any recursive cycle ⇒ terminates (no self-dependence).
        return Totality::Total;
    }

    // Is the cycle MUTUAL (goes through another function) or a direct self-loop?
    // This is a CONSERVATIVE reachability-symmetry test, not a precise Tarjan SCC:
    // if `f` reaches some other `g` that reaches `f` back, treat `f` as mutually
    // recursive ⇒ `Partial`. It can over-approximate (a `g` reachable from `f`
    // that is only coincidentally in a cycle with `f` still trips it), which only
    // ever errs toward REJECTING — sound for the certificate, never accepting a
    // non-terminating set. A real SCC would be more precise; not needed yet.
    let empty = HashSet::new();
    let r = reach.get(&f.name).unwrap_or(&empty);
    let mutual = r
        .iter()
        .any(|g| g != &f.name && reach.get(g).map(|s| s.contains(&f.name)).unwrap_or(false));
    if mutual {
        return Totality::Partial(format!(
            "`{}` is part of a mutual-recursion cycle, which is not yet certifiable \
             as total (only direct structural self-recursion is — Phase E2)",
            f.name
        ));
    }

    // direct self-recursion: every self-call must structurally descend AND pass
    // every non-scrutinee argument verbatim (so it lowers to an eliminator).
    let sp = match f.scrut_pos {
        Some(p) => p,
        None => {
            return Totality::Partial(format!(
                "`{}` recurses but does not pattern-match an argument, so there is \
                 no structural measure to decrease on",
                f.name
            ))
        }
    };

    // self-calls live inside arms (each with its own `smaller` set) and possibly
    // in a non-match body (no smaller set — can never descend).
    for c in &f.body_calls {
        if c.callee == f.name {
            return Totality::Partial(format!(
                "`{}` calls itself outside of any pattern-match arm, so the call \
                 cannot decrease a structural measure",
                f.name
            ));
        }
    }
    // did any self-call descend by a RUNTIME-WITNESSED `dlt` decrease (Phase
    // E3/B1)? Such a fn is certified total but must lower as `Fix`.
    let mut wf_used = false;
    for arm in &f.arms {
        for c in &arm.calls {
            if c.callee != f.name {
                continue;
            }
            if c.args.len() != f.params.len() {
                return Totality::Partial(format!(
                    "recursive call to `{}` has the wrong number of arguments",
                    f.name
                ));
            }
            // scrutinee-position argument must be a strict subterm: either a DIRECT
            // recursive-field binder (`smaller`), or — for a HIGHER-ORDER recursive
            // field `f` (a W-type child-function / `Acc`'s accessibility fn) — an
            // application `f(args…)` of it, which is the genuine sub-derivation the
            // kernel's functional IH eliminates (Phase 1b / well-founded recursion).
            // is this call's descent HIGHER-ORDER (`f(args…)` of a higher-order
            // recursive field — well-founded recursion) rather than first-order?
            let ho_descent =
                matches!(&c.args[sp], Tm::Call(g, _) if arm.ho_smaller.contains(g));
            // WELL-FOUNDED descent by runtime witness (Phase E3/B1): the call's
            // measure argument is exactly the `e1` of an in-scope `dlt(e1, e2)`
            // `DYes` fact whose `e2` is the matched parameter itself — so when
            // this branch runs, the measure strictly decreased in `<`, which is
            // well-founded on the packed machine Nat.
            let wf_descent = c.lt_facts.iter().any(|(e1, e2)| {
                c.args[sp] == *e1 && matches!(e2, Tm::Var(v) if v == &f.params[sp])
            });
            if wf_descent {
                wf_used = true;
            }
            match &c.args[sp] {
                _ if wf_descent => {}
                Tm::Var(v) if arm.smaller.contains(v) => {}
                Tm::Call(g, _) if arm.ho_smaller.contains(g) => {}
                other => {
                    return Totality::Partial(format!(
                        "recursive call to `{}` does not decrease: the argument in the \
                         matched position is {}, not a sub-structure of `{}` (and no \
                         in-scope `dlt` fact witnesses it smaller)",
                        f.name,
                        describe(other),
                        f.params[sp]
                    ))
                }
            }
            // For a `%builtin Nat` scrutinee, varying a non-scrutinee argument is
            // FINE: the call descends on the scrutinee, and the fold lowers to a
            // function-typed-motive `NatElim` (Phase 1a′ accumulator fold) that the
            // kernel re-checks total-by-construction. Likewise, under HIGHER-ORDER
            // descent (well-founded recursion — `f(y, h(y, prf))`), the non-scrutinee
            // args legitimately VARY (the new `y`): they are part of the descent and
            // are certified by the functional IH; the elaborator's VALUE-CORRECTNESS
            // guard (in `ih_for`) ensures the varying arg matches the field-application
            // argument, so a wrong-value lowering is rejected. For a first-order BOXED
            // descent the accumulator lowering isn't available, so verbatim is still
            // required (honest decline).
            if !f.scrut_is_nat && !ho_descent && !wf_descent {
                for (i, a) in c.args.iter().enumerate() {
                    if i == sp {
                        continue;
                    }
                    match a {
                        Tm::Var(v) if v == &f.params[i] => {}
                        _ => {
                            return Totality::Partial(format!(
                                "`{}` is accumulator-style recursion (argument `{}` changes \
                                 across the recursive call) over a BOXED datatype: this is \
                                 terminating but not yet lowerable to an eliminator (the \
                                 accumulator fold is implemented only for a `%builtin Nat` \
                                 scrutinee so far — Phase E2/E3). Mark it `%partial`, or \
                                 restructure as a plain fold.",
                                f.name, f.params[i]
                            ))
                        }
                    }
                }
            }
        }
    }
    if wf_used {
        Totality::TotalWf
    } else {
        Totality::Total
    }
}

fn describe(t: &Tm) -> String {
    match t {
        Tm::Var(v) => format!("the variable `{v}`"),
        Tm::Call(c, _) => format!("a constructor/call `{c}(…)`"),
        Tm::Lit(n) => format!("the literal `{n}`"),
        Tm::Str(_) => "a string literal".into(),
        Tm::Add(_, _) => "an addition".into(),
        Tm::Match(_, _) | Tm::MatchN(_, _) => "a match".into(),
        Tm::LetPair(_, _, _) | Tm::Let(_, _, _) => "a let".into(),
        Tm::Ann(_, _) => "a type ascription".into(),
    }
}

/// Transitive closure of a directed graph (Floyd-Warshall-style over reachable
/// sets). Small graphs (one entry per user fn), so the naive fixpoint is fine.
fn transitive_closure(edges: &HashMap<&str, HashSet<&str>>) -> HashMap<String, HashSet<String>> {
    let mut reach: HashMap<String, HashSet<String>> = edges
        .iter()
        .map(|(k, vs)| (k.to_string(), vs.iter().map(|s| s.to_string()).collect()))
        .collect();
    loop {
        let mut changed = false;
        let keys: Vec<String> = reach.keys().cloned().collect();
        for k in &keys {
            let succs: Vec<String> = reach[k].iter().cloned().collect();
            for s in succs {
                let add: Vec<String> = reach.get(&s).map(|x| x.iter().cloned().collect()).unwrap_or_default();
                let entry = reach.get_mut(k).unwrap();
                for a in add {
                    if entry.insert(a) {
                        changed = true;
                    }
                }
            }
        }
        if !changed {
            break;
        }
    }
    reach
}
