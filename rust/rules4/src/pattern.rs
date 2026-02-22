use crate::term::*;

// ── Pattern, Env, Clause, Rule ──

#[derive(Clone, Debug)]
pub enum Pattern {
    Num(i64),
    Float(u64), // f64 stored as bits
    Sym(SymId),
    Var(VarId),
    Wildcard,
    Call(Box<Pattern>, Vec<Pattern>),
    Map(Vec<(Pattern, Pattern)>),  // partial map matching: entries are (key_pat, val_pat)
    Spread(VarId),    // ?rest... — binds remaining elements as a vec/set
    WildSpread,       // _... — matches remaining elements, discards
    Set(Vec<Pattern>), // #{pat, pat} — subset matching
}

pub const MAX_VARS: usize = 16;

#[derive(Clone, Copy)]
pub struct Env {
    bindings: [TermId; MAX_VARS],
    bound: u16,
}

impl Env {
    #[inline]
    pub fn new() -> Self {
        Env {
            bindings: [TermId(0); MAX_VARS],
            bound: 0,
        }
    }

    #[inline]
    pub fn get(&self, v: VarId) -> Option<TermId> {
        let i = v.0 as u16;
        if self.bound & (1 << i) != 0 {
            Some(self.bindings[i as usize])
        } else {
            None
        }
    }

    #[inline]
    pub fn set(&mut self, v: VarId, t: TermId) {
        let i = v.0 as usize;
        self.bindings[i] = t;
        self.bound |= 1 << (i as u16);
    }
}

pub struct Clause {
    pub lhs: Pattern,
    pub rhs: Pattern,
    pub guards: Vec<Pattern>,  // where-conditions: each must eval to true
}

pub struct Rule {
    pub name: String,
    pub clauses: Vec<Clause>,
}

// ── Pattern matching context ──
// Bundles symbol IDs needed for computed pattern matching.

pub struct PatternContext {
    pub quote_sym: SymId,
    pub add_sym: SymId,
    pub sub_sym: SymId,
    pub mul_sym: SymId,
    pub div_sym: SymId,
    pub mod_sym: SymId,
}

// ── Computed pattern support ──

/// Check if a pattern is "ground" given the current env — i.e., all variables
/// are bound and there are no wildcards. Such a pattern can be computed into
/// a concrete term (via substitute + arithmetic evaluation) and compared directly.
fn pattern_is_ground(pat: &Pattern, env: &Env) -> bool {
    match pat {
        Pattern::Num(_) | Pattern::Float(_) | Pattern::Sym(_) => true,
        Pattern::Wildcard | Pattern::WildSpread => false,
        Pattern::Var(v) => env.get(*v).is_some(),
        Pattern::Spread(v) => env.get(*v).is_some(),
        Pattern::Call(head, args) => {
            pattern_is_ground(head, env) && args.iter().all(|a| pattern_is_ground(a, env))
        }
        Pattern::Set(elems) => elems.iter().all(|e| pattern_is_ground(e, env)),
        Pattern::Map(entries) => entries.iter().all(|(k, v)| pattern_is_ground(k, env) && pattern_is_ground(v, env)),
    }
}

/// Check if a pattern contains any Call sub-patterns that look like arithmetic
/// (add, sub, mul, div, mod). Used to avoid the overhead of compute_bound_pattern
/// on simple patterns that don't need arithmetic evaluation.
fn pattern_has_arithmetic(pat: &Pattern, ctx: &PatternContext) -> bool {
    match pat {
        Pattern::Call(head, args) => {
            if let Pattern::Sym(s) = **head {
                if s == ctx.add_sym || s == ctx.sub_sym || s == ctx.mul_sym
                    || s == ctx.div_sym || s == ctx.mod_sym
                {
                    return true;
                }
            }
            args.iter().any(|a| pattern_has_arithmetic(a, ctx))
        }
        Pattern::Set(elems) => elems.iter().any(|e| pattern_has_arithmetic(e, ctx)),
        Pattern::Map(entries) => entries.iter().any(|(k, v)| pattern_has_arithmetic(k, ctx) || pattern_has_arithmetic(v, ctx)),
        _ => false,
    }
}

/// Evaluate arithmetic operations in a term tree. Recursively reduces args first,
/// then attempts arithmetic reduction (add, sub, mul, div, mod) on binary calls.
/// Non-arithmetic calls are left as-is with their args evaluated.
fn eval_pattern_arithmetic(store: &mut TermStore, term: TermId, ctx: &PatternContext) -> TermId {
    match store.get(term) {
        TermData::Num(_) | TermData::Float(_) | TermData::Sym(_) => term,
        TermData::Call { head, args_start, args_len } => {
            // Recursively evaluate args
            let len = args_len as usize;
            let mut new_args: Vec<TermId> = Vec::with_capacity(len);
            let mut changed = false;
            for i in 0..len {
                let arg = store.args_pool[args_start as usize + i];
                let evaled = eval_pattern_arithmetic(store, arg, ctx);
                changed |= evaled != arg;
                new_args.push(evaled);
            }

            // Try arithmetic reduction for binary ops
            if args_len == 2 {
                if let TermData::Sym(s) = store.get(head) {
                    let a = new_args[0];
                    let b = new_args[1];
                    let a_val = match store.get(a) {
                        TermData::Num(n) => Some((n as f64, true)),
                        TermData::Float(bits) => Some((f64::from_bits(bits), false)),
                        _ => None,
                    };
                    let b_val = match store.get(b) {
                        TermData::Num(n) => Some((n as f64, true)),
                        TermData::Float(bits) => Some((f64::from_bits(bits), false)),
                        _ => None,
                    };
                    if let (Some((av, a_int)), Some((bv, b_int))) = (a_val, b_val) {
                        let both_int = a_int && b_int;
                        let result = if s == ctx.add_sym { Some(av + bv) }
                            else if s == ctx.sub_sym { Some(av - bv) }
                            else if s == ctx.mul_sym { Some(av * bv) }
                            else if s == ctx.div_sym { Some(av / bv) }
                            else if s == ctx.mod_sym {
                                if both_int { Some((av as i64 % bv as i64) as f64) }
                                else { Some(av.rem_euclid(bv)) }
                            }
                            else { None };
                        if let Some(r) = result {
                            return if both_int { store.num(r as i64) } else { store.float(r) };
                        }
                    }
                }
            }

            if changed {
                store.call(head, &new_args)
            } else {
                term
            }
        }
    }
}

/// Compute a fully-ground pattern into a concrete term by substituting bound
/// variables and evaluating arithmetic. Used for computed pattern matching:
/// `at(?x + 1, ?y, crate)` with ?x=3, ?y=4 → `at(4, 4, crate)`.
fn compute_bound_pattern(store: &mut TermStore, pat: &Pattern, env: &Env, ctx: &PatternContext) -> TermId {
    let raw = substitute(store, pat, env);
    eval_pattern_arithmetic(store, raw, ctx)
}

// ── Pattern matching ──

#[inline]
fn unwrap_quote(store: &TermStore, term: TermId, quote_sym: SymId) -> TermId {
    if let TermData::Call { head, args_start, args_len: 1 } = store.get(term) {
        if let TermData::Sym(s) = store.get(head) {
            if s == quote_sym {
                return store.args_pool[args_start as usize];
            }
        }
    }
    term
}

pub fn match_pattern(store: &mut TermStore, pat: &Pattern, term: TermId, env: &mut Env, ctx: &PatternContext) -> bool {
    match pat {
        Pattern::Num(n) => {
            let term = unwrap_quote(store, term, ctx.quote_sym);
            matches!(store.get(term), TermData::Num(m) if m == *n)
        }
        Pattern::Float(bits) => {
            let term = unwrap_quote(store, term, ctx.quote_sym);
            matches!(store.get(term), TermData::Float(b) if b == *bits)
        }
        Pattern::Sym(s) => {
            let term = unwrap_quote(store, term, ctx.quote_sym);
            matches!(store.get(term), TermData::Sym(t) if t == *s)
        }
        Pattern::Wildcard | Pattern::WildSpread => true,
        Pattern::Var(v) => {
            // Don't unwrap quote — preserve protection for variables
            if let Some(bound) = env.get(*v) {
                bound == term
            } else {
                env.set(*v, term);
                true
            }
        }
        Pattern::Spread(v) => {
            // Spread outside of Call/Set context — treat like Var
            if let Some(bound) = env.get(*v) {
                bound == term
            } else {
                env.set(*v, term);
                true
            }
        }
        Pattern::Call(head_pat, arg_pats) => {
            let term = unwrap_quote(store, term, ctx.quote_sym);

            // Computed pattern matching: if all variables in this Call pattern
            // are already bound AND the pattern contains arithmetic, we can
            // compute the expected concrete term and compare via TermId equality.
            // This enables patterns like `at(?x + 1, ?y, crate)` where ?x, ?y
            // are bound from earlier pattern matching.
            if pattern_is_ground(pat, env) && pattern_has_arithmetic(pat, ctx) {
                let expected = compute_bound_pattern(store, pat, env, ctx);
                return expected == term;
            }

            if let TermData::Call { head, args_start, args_len } = store.get(term) {
                if !match_pattern(store, head_pat, head, env, ctx) {
                    return false;
                }
                // Check if last arg pattern is a spread
                let has_spread = arg_pats.last().map_or(false, |p| matches!(p, Pattern::Spread(_) | Pattern::WildSpread));
                if has_spread {
                    let fixed_count = arg_pats.len() - 1;
                    if (args_len as usize) < fixed_count {
                        return false;
                    }
                    // Match fixed patterns
                    for (i, ap) in arg_pats[..fixed_count].iter().enumerate() {
                        let arg = store.args_pool[args_start as usize + i];
                        if !match_pattern(store, ap, arg, env, ctx) {
                            return false;
                        }
                    }
                    // Handle spread
                    match arg_pats.last().unwrap() {
                        Pattern::Spread(v) => {
                            // Collect remaining args into a vec(...) term
                            let remaining: Vec<TermId> = (fixed_count..args_len as usize)
                                .map(|i| store.args_pool[args_start as usize + i])
                                .collect();
                            // Determine the collection head from the call head
                            // If the call head is "set", collect into set(); otherwise vec()
                            let head_name = if let TermData::Sym(s) = store.get(head) {
                                store.sym_name(s).to_string()
                            } else {
                                String::new()
                            };
                            let coll_sym = if head_name == "set" {
                                store.sym("set")
                            } else {
                                store.sym("vec")
                            };
                            let coll_head = store.sym_term(coll_sym);
                            let rest_term = store.call(coll_head, &remaining);
                            env.set(*v, rest_term);
                        }
                        Pattern::WildSpread => { /* discard */ }
                        _ => unreachable!(),
                    }
                    true
                } else {
                    // Exact match — no spread
                    if args_len as usize != arg_pats.len() {
                        return false;
                    }
                    for (i, ap) in arg_pats.iter().enumerate() {
                        let arg = store.args_pool[args_start as usize + i];
                        if !match_pattern(store, ap, arg, env, ctx) {
                            return false;
                        }
                    }
                    true
                }
            } else {
                false
            }
        }
        Pattern::Set(elem_pats) => {
            // Subset matching with join ordering:
            // 1. Patterns whose vars are all bound → compute term, direct membership test (O(1))
            // 2. Patterns with free vars → linear scan through set elements
            // This turns O(n^k) backtracking into O(n) scan + O(1) lookups for spatial patterns.
            let term = unwrap_quote(store, term, ctx.quote_sym);
            if let TermData::Call { head, args_start, args_len } = store.get(term) {
                if let TermData::Sym(s) = store.get(head) {
                    if store.sym_name(s) != "set" {
                        return false;
                    }
                } else {
                    return false;
                }

                // Check for spread as last element
                let has_spread = elem_pats.last().map_or(false, |p| matches!(p, Pattern::Spread(_) | Pattern::WildSpread));
                let fixed_pats = if has_spread { &elem_pats[..elem_pats.len() - 1] } else { &elem_pats[..] };

                if (args_len as usize) < fixed_pats.len() {
                    return false;
                }

                let saved_env = *env;
                let mut used = vec![false; args_len as usize];
                let mut remaining: Vec<usize> = (0..fixed_pats.len()).collect();

                while !remaining.is_empty() {
                    // First, look for a computable pattern (all vars bound).
                    // This is the "join ordering" optimization: after binding vars from
                    // free patterns, we can directly compute and look up remaining patterns.
                    let computable_idx = remaining.iter().position(|&pi| pattern_is_ground(&fixed_pats[pi], env));

                    if let Some(ri) = computable_idx {
                        let pi = remaining[ri];
                        let expected = compute_bound_pattern(store, &fixed_pats[pi], env, ctx);
                        // Direct membership test: scan for matching TermId
                        let mut found = false;
                        for i in 0..args_len as usize {
                            if used[i] { continue; }
                            let arg = store.args_pool[args_start as usize + i];
                            if arg == expected {
                                used[i] = true;
                                found = true;
                                break;
                            }
                        }
                        if !found {
                            *env = saved_env;
                            return false;
                        }
                        remaining.remove(ri);
                    } else {
                        // No computable patterns — pick first free pattern and search
                        let ri = 0;
                        let pi = remaining[ri];
                        let mut found = false;
                        for i in 0..args_len as usize {
                            if used[i] { continue; }
                            let arg = store.args_pool[args_start as usize + i];
                            let mut trial_env = *env;
                            if match_pattern(store, &fixed_pats[pi], arg, &mut trial_env, ctx) {
                                *env = trial_env;
                                used[i] = true;
                                found = true;
                                break;
                            }
                        }
                        if !found {
                            *env = saved_env;
                            return false;
                        }
                        remaining.remove(ri);
                    }
                }

                // Handle spread
                if has_spread {
                    match elem_pats.last().unwrap() {
                        Pattern::Spread(v) => {
                            let remaining: Vec<TermId> = (0..args_len as usize)
                                .filter(|i| !used[*i])
                                .map(|i| store.args_pool[args_start as usize + i])
                                .collect();
                            let set_sym = store.sym("set");
                            let set_head = store.sym_term(set_sym);
                            let rest_term = store.call(set_head, &remaining);
                            env.set(*v, rest_term);
                        }
                        Pattern::WildSpread => { /* discard */ }
                        _ => unreachable!(),
                    }
                }

                true
            } else {
                false
            }
        }
        Pattern::Map(entries) => {
            // Partial map matching: the term must be map(entry(k,v), ...).
            // For each (key_pat, val_pat) in the pattern, find a matching entry.
            let term = unwrap_quote(store, term, ctx.quote_sym);
            if let TermData::Call { head, args_start, args_len } = store.get(term) {
                if let TermData::Sym(s) = store.get(head) {
                    if store.sym_name(s) != "map" {
                        return false;
                    }
                } else {
                    return false;
                }
                // Save env state for backtracking on failure
                let saved_env = *env;
                for (key_pat, val_pat) in entries {
                    let mut found = false;
                    for i in 0..args_len as usize {
                        let entry_term = store.args_pool[args_start as usize + i];
                        if let TermData::Call { head: eh, args_start: eas, args_len: 2 } = store.get(entry_term) {
                            if let TermData::Sym(es) = store.get(eh) {
                                if store.sym_name(es) == "entry" {
                                    let k = store.args_pool[eas as usize];
                                    let v = store.args_pool[eas as usize + 1];
                                    let mut trial_env = *env;
                                    if match_pattern(store, key_pat, k, &mut trial_env, ctx)
                                        && match_pattern(store, val_pat, v, &mut trial_env, ctx)
                                    {
                                        *env = trial_env;
                                        found = true;
                                        break;
                                    }
                                }
                            }
                        }
                    }
                    if !found {
                        *env = saved_env;
                        return false;
                    }
                }
                true
            } else {
                false
            }
        }
    }
}

// ── Substitution ──

pub fn substitute(store: &mut TermStore, pat: &Pattern, env: &Env) -> TermId {
    match pat {
        Pattern::Num(n) => store.num(*n),
        Pattern::Float(bits) => store.float(f64::from_bits(*bits)),
        Pattern::Sym(s) => store.sym_term(*s),
        Pattern::Wildcard => panic!("Wildcard _ should never appear in RHS"),
        Pattern::WildSpread => panic!("WildSpread _... should never appear in RHS"),
        Pattern::Var(v) => env.get(*v).unwrap(),
        Pattern::Spread(v) => {
            // In RHS substitution, Spread is treated like Var (used for splice in Call/Set below)
            env.get(*v).unwrap()
        }
        Pattern::Call(head, args) => {
            let h = substitute(store, head, env);
            // Check if any arg is a Spread — if so, splice its children inline
            let mut a: Vec<TermId> = Vec::new();
            for ap in args {
                if let Pattern::Spread(v) = ap {
                    // The bound value should be a vec(...) or set(...) — splice its children
                    let spread_term = env.get(*v).unwrap();
                    if let TermData::Call { args_start, args_len, .. } = store.get(spread_term) {
                        for i in 0..args_len as usize {
                            a.push(store.args_pool[args_start as usize + i]);
                        }
                    } else {
                        // Not a collection — just include the term directly
                        a.push(spread_term);
                    }
                } else {
                    a.push(substitute(store, ap, env));
                }
            }
            store.call(h, &a)
        }
        Pattern::Set(elems) => {
            let set_sym = store.sym("set");
            let set_head = store.sym_term(set_sym);
            let mut a: Vec<TermId> = Vec::new();
            for ep in elems {
                if let Pattern::Spread(v) = ep {
                    let spread_term = env.get(*v).unwrap();
                    if let TermData::Call { args_start, args_len, .. } = store.get(spread_term) {
                        for i in 0..args_len as usize {
                            a.push(store.args_pool[args_start as usize + i]);
                        }
                    } else {
                        a.push(spread_term);
                    }
                } else {
                    a.push(substitute(store, ep, env));
                }
            }
            store.call(set_head, &a)
        }
        Pattern::Map(entries) => {
            let map_sym = store.sym("map");
            let map_head = store.sym_term(map_sym);
            let entry_sym = store.sym("entry");
            let entry_head = store.sym_term(entry_sym);
            let a: Vec<TermId> = entries.iter().map(|(kp, vp)| {
                let k = substitute(store, kp, env);
                let v = substitute(store, vp, env);
                store.call(entry_head, &[k, v])
            }).collect();
            store.call(map_head, &a)
        }
    }
}

pub fn pattern_to_term(store: &mut TermStore, pat: &Pattern) -> TermId {
    match pat {
        Pattern::Num(n) => store.num(*n),
        Pattern::Float(bits) => store.float(f64::from_bits(*bits)),
        Pattern::Sym(s) => store.sym_term(*s),
        Pattern::Wildcard => panic!("Wildcard _ should never appear in top-level expression"),
        Pattern::WildSpread => panic!("WildSpread _... should never appear in top-level expression"),
        Pattern::Var(v) => panic!("Unbound variable ?{} in top-level expression", v.0),
        Pattern::Spread(v) => panic!("Spread ?{}... should never appear in top-level expression", v.0),
        Pattern::Call(head, args) => {
            let h = pattern_to_term(store, head);
            let a: Vec<TermId> = args.iter().map(|a| pattern_to_term(store, a)).collect();
            store.call(h, &a)
        }
        Pattern::Set(elems) => {
            let set_sym = store.sym("set");
            let set_head = store.sym_term(set_sym);
            let a: Vec<TermId> = elems.iter().map(|e| pattern_to_term(store, e)).collect();
            store.call(set_head, &a)
        }
        Pattern::Map(entries) => {
            let map_sym = store.sym("map");
            let map_head = store.sym_term(map_sym);
            let entry_sym = store.sym("entry");
            let entry_head = store.sym_term(entry_sym);
            let a: Vec<TermId> = entries.iter().map(|(kp, vp)| {
                let k = pattern_to_term(store, kp);
                let v = pattern_to_term(store, vp);
                store.call(entry_head, &[k, v])
            }).collect();
            store.call(map_head, &a)
        }
    }
}
