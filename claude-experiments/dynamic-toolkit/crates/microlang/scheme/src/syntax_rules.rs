//! `syntax-rules`: a pattern → template macro engine, ENTIRELY in the frontend.
//!
//! This is the answer to a design question the operating procedure poses: does
//! the core's macro machinery need to know about hygienic macros? No. A
//! `syntax-rules` macro is a pure form-to-form rewrite (no code runs during
//! expansion), so it is a frontend compile-time pass — the core never sees it.
//! The core's re-entrant procedural-macro driver stays for a different job.
//!
//! Supported: multiple rules, literals, pattern variables, `_` wildcard, and
//! single-level `...` ellipsis (in both patterns and templates). NOT YET:
//! HYGIENE (template identifiers are inserted verbatim, so a template that
//! introduces a binding can capture a user identifier) and nested ellipsis.
//! Hygiene is the genuinely hard remaining piece; a hygiene-requiring case is
//! kept pending in the conformance suite to mark it.

use std::collections::HashMap;

use microlang::{Runtime, Val, ValueModel};

pub struct SyntaxRules {
    literals: Vec<u32>,
    rules: Vec<(u64, u64)>, // (pattern, template)
}

#[derive(Clone)]
enum PatVar {
    One(u64),
    Many(Vec<u64>),
}

/// Parse `(syntax-rules (lit ...) (pattern template) ...)`.
pub fn parse<M: ValueModel>(rt: &Runtime<M>, form: u64) -> SyntaxRules {
    let items = rt.list_to_vec(form);
    let literals = rt
        .list_to_vec(items[1])
        .iter()
        .filter_map(|&l| match rt.decode(l) {
            Val::Sym(s) => Some(s),
            _ => None,
        })
        .collect();
    let rules = items[2..]
        .iter()
        .map(|&r| {
            let rv = rt.list_to_vec(r);
            (rv[0], rv[1])
        })
        .collect();
    SyntaxRules { literals, rules }
}

/// Apply the macro to a use `(name args...)`, returning the expansion.
pub fn apply<M: ValueModel>(rt: &mut Runtime<M>, sr: &SyntaxRules, form: u64) -> Option<u64> {
    let input = rt.list_to_vec(form);
    for (pat, tmpl) in &sr.rules {
        let pv = rt.list_to_vec(*pat);
        let mut binds = HashMap::new();
        // The pattern's first element is the macro keyword — ignore it.
        if match_seq(rt, &pv[1..], &input[1..], &sr.literals, &mut binds) {
            return Some(instantiate(rt, *tmpl, &binds));
        }
    }
    None
}

fn is_named<M: ValueModel>(rt: &Runtime<M>, form: u64, name: &str) -> bool {
    matches!(rt.decode(form), Val::Sym(s) if rt.sym_name(s) == name)
}

fn match_one<M: ValueModel>(
    rt: &Runtime<M>,
    pat: u64,
    inp: u64,
    literals: &[u32],
    binds: &mut HashMap<u32, PatVar>,
) -> bool {
    match rt.decode(pat) {
        Val::Sym(s) => {
            let n = rt.sym_name(s);
            if literals.contains(&s) {
                matches!(rt.decode(inp), Val::Sym(i) if i == s)
            } else if n == "_" {
                true
            } else {
                binds.insert(s, PatVar::One(inp));
                true
            }
        }
        _ => {
            if rt.as_cons(pat).is_some() {
                let pats = rt.list_to_vec(pat);
                let inps = rt.list_to_vec(inp);
                // reject if inp is not a list (list_to_vec of a non-list is empty)
                if inps.is_empty() && rt.as_cons(inp).is_none() && !pats.is_empty() {
                    return false;
                }
                match_seq(rt, &pats, &inps, literals, binds)
            } else {
                rt.equal(pat, inp) // literal datum (number, bool, ...)
            }
        }
    }
}

fn match_seq<M: ValueModel>(
    rt: &Runtime<M>,
    pats: &[u64],
    inps: &[u64],
    literals: &[u32],
    binds: &mut HashMap<u32, PatVar>,
) -> bool {
    let ell = pats.iter().position(|&p| is_named(rt, p, "..."));
    match ell {
        None => {
            if pats.len() != inps.len() {
                return false;
            }
            pats.iter()
                .zip(inps)
                .all(|(&p, &i)| match_one(rt, p, i, literals, binds))
        }
        Some(k) => {
            // pats: [before... , sub, "...", after...]
            let sub = pats[k - 1];
            let before = &pats[..k - 1];
            let after = &pats[k + 1..];
            if inps.len() < before.len() + after.len() {
                return false;
            }
            for (&p, &i) in before.iter().zip(inps) {
                if !match_one(rt, p, i, literals, binds) {
                    return false;
                }
            }
            let split = inps.len() - after.len();
            for (&p, &i) in after.iter().zip(&inps[split..]) {
                if !match_one(rt, p, i, literals, binds) {
                    return false;
                }
            }
            // ellipsis matches the middle; init each sub-var to an empty Many
            let mut evars = Vec::new();
            pattern_vars(rt, sub, literals, &mut evars);
            for &v in &evars {
                binds.insert(v, PatVar::Many(Vec::new()));
            }
            for &m in &inps[before.len()..split] {
                let mut sb = HashMap::new();
                if !match_one(rt, sub, m, literals, &mut sb) {
                    return false;
                }
                for (v, pv) in sb {
                    if let (PatVar::One(val), Some(PatVar::Many(vec))) = (pv, binds.get_mut(&v)) {
                        vec.push(val);
                    }
                }
            }
            true
        }
    }
}

fn pattern_vars<M: ValueModel>(rt: &Runtime<M>, pat: u64, literals: &[u32], out: &mut Vec<u32>) {
    match rt.decode(pat) {
        Val::Sym(s) => {
            let n = rt.sym_name(s);
            if !literals.contains(&s) && n != "_" && n != "..." && !out.contains(&s) {
                out.push(s);
            }
        }
        _ => {
            if rt.as_cons(pat).is_some() {
                for e in rt.list_to_vec(pat) {
                    pattern_vars(rt, e, literals, out);
                }
            }
        }
    }
}

fn ellipsis_vars<M: ValueModel>(
    rt: &Runtime<M>,
    tmpl: u64,
    binds: &HashMap<u32, PatVar>,
    out: &mut Vec<u32>,
) {
    match rt.decode(tmpl) {
        Val::Sym(s) => {
            if matches!(binds.get(&s), Some(PatVar::Many(_))) && !out.contains(&s) {
                out.push(s);
            }
        }
        _ => {
            if rt.as_cons(tmpl).is_some() {
                for e in rt.list_to_vec(tmpl) {
                    ellipsis_vars(rt, e, binds, out);
                }
            }
        }
    }
}

fn instantiate<M: ValueModel>(
    rt: &mut Runtime<M>,
    tmpl: u64,
    binds: &HashMap<u32, PatVar>,
) -> u64 {
    match rt.decode(tmpl) {
        Val::Sym(s) => match binds.get(&s) {
            Some(PatVar::One(v)) => *v,
            _ => tmpl, // not a pattern var -> inserted verbatim (UNHYGIENIC)
        },
        _ => {
            if rt.as_cons(tmpl).is_none() {
                return tmpl; // literal datum
            }
            let elems = rt.list_to_vec(tmpl);
            let mut out = Vec::new();
            let mut i = 0;
            while i < elems.len() {
                if i + 1 < elems.len() && is_named(rt, elems[i + 1], "...") {
                    let sub = elems[i];
                    let mut evars = Vec::new();
                    ellipsis_vars(rt, sub, binds, &mut evars);
                    let n = evars
                        .iter()
                        .filter_map(|v| match binds.get(v) {
                            Some(PatVar::Many(vec)) => Some(vec.len()),
                            _ => None,
                        })
                        .min()
                        .unwrap_or(0);
                    for j in 0..n {
                        let mut sb: HashMap<u32, PatVar> = HashMap::new();
                        for (k, pv) in binds {
                            match pv {
                                PatVar::Many(vec) if evars.contains(k) => {
                                    sb.insert(*k, PatVar::One(vec[j]));
                                }
                                other => {
                                    sb.insert(*k, other.clone());
                                }
                            }
                        }
                        let e = instantiate(rt, sub, &sb);
                        out.push(e);
                    }
                    i += 2;
                } else {
                    let e = instantiate(rt, elems[i], binds);
                    out.push(e);
                    i += 1;
                }
            }
            rt.vec_to_list(&out)
        }
    }
}
