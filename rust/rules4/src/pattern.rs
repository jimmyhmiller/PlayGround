use crate::term::*;

// ── Pattern, Env, Clause, Rule ──

#[derive(Clone, Debug)]
pub enum Pattern {
    Num(i64),
    Float(u64), // f64 stored as bits
    Sym(SymId),
    Var(VarId),
    Call(Box<Pattern>, Vec<Pattern>),
}

pub const MAX_VARS: usize = 8;

#[derive(Clone, Copy)]
pub struct Env {
    bindings: [TermId; MAX_VARS],
    bound: u8,
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
        let i = v.0 as u8;
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
        self.bound |= 1 << (i as u8);
    }
}

pub struct Clause {
    pub lhs: Pattern,
    pub rhs: Pattern,
}

pub struct Rule {
    pub name: String,
    pub clauses: Vec<Clause>,
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

pub fn match_pattern(store: &TermStore, pat: &Pattern, term: TermId, env: &mut Env, quote_sym: SymId) -> bool {
    match pat {
        Pattern::Num(n) => {
            let term = unwrap_quote(store, term, quote_sym);
            matches!(store.get(term), TermData::Num(m) if m == *n)
        }
        Pattern::Float(bits) => {
            let term = unwrap_quote(store, term, quote_sym);
            matches!(store.get(term), TermData::Float(b) if b == *bits)
        }
        Pattern::Sym(s) => {
            let term = unwrap_quote(store, term, quote_sym);
            matches!(store.get(term), TermData::Sym(t) if t == *s)
        }
        Pattern::Var(v) => {
            // Don't unwrap quote — preserve protection for variables
            if let Some(bound) = env.get(*v) {
                bound == term
            } else {
                env.set(*v, term);
                true
            }
        }
        Pattern::Call(head_pat, arg_pats) => {
            let term = unwrap_quote(store, term, quote_sym);
            if let TermData::Call { head, args_start, args_len } = store.get(term) {
                if args_len as usize != arg_pats.len() {
                    return false;
                }
                if !match_pattern(store, head_pat, head, env, quote_sym) {
                    return false;
                }
                for (i, ap) in arg_pats.iter().enumerate() {
                    let arg = store.args_pool[args_start as usize + i];
                    if !match_pattern(store, ap, arg, env, quote_sym) {
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
        Pattern::Var(v) => env.get(*v).unwrap(),
        Pattern::Call(head, args) => {
            let h = substitute(store, head, env);
            let mut buf = [TermId(0); 16];
            let len = args.len();
            for (i, ap) in args.iter().enumerate() {
                buf[i] = substitute(store, ap, env);
            }
            store.call(h, &buf[..len])
        }
    }
}

pub fn pattern_to_term(store: &mut TermStore, pat: &Pattern) -> TermId {
    match pat {
        Pattern::Num(n) => store.num(*n),
        Pattern::Float(bits) => store.float(f64::from_bits(*bits)),
        Pattern::Sym(s) => store.sym_term(*s),
        Pattern::Var(v) => panic!("Unbound variable ?{} in top-level expression", v.0),
        Pattern::Call(head, args) => {
            let h = pattern_to_term(store, head);
            let a: Vec<TermId> = args.iter().map(|a| pattern_to_term(store, a)).collect();
            store.call(h, &a)
        }
    }
}
