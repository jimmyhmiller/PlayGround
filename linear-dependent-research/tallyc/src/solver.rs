//! Stratum (A) of the programmable checker (`docs/03`): a small, DECIDABLE
//! constraint domain that sits *underneath* the trusted kernel. This first slice
//! decides definitional equality over the **linear-Nat arithmetic** fragment of
//! the index language, so that `n + m` and `m + n`, `n + 0` and `n`, `(a+b)+c`
//! and `a+(b+c)`, and `Suc n` and `n + 1` are all judged equal by the kernel.
//!
//! It is in the TCB (like an SMT core): it may only ever equate terms that are
//! *genuinely* equal. Every rewrite below is a true identity of `Nat` `+`
//! (commutativity, associativity, `+0`, `Suc n = n + 1`), and it fires ONLY on
//! syntactically-arithmetic nodes — so folding it into `conv` makes definitional
//! equality coarser strictly by valid equalities, never by unsound ones. It is
//! also complete for the linear fragment: equal linear combinations produce an
//! identical canonical term.
//!
//! It is a pure *decision* procedure — no proof search, no hole-filling. That is
//! the deliberate "explicit-first" scope (see `docs/PHASE_C_SOLVER_PLAN.md`).

use crate::dep::Term;

/// A linear combination over `Nat`: a constant plus a multiset of `coeff · atom`,
/// where an `atom` is an already-canonicalized non-arithmetic `Term`.
struct LinNat {
    konst: u64,
    /// `(atom, coefficient)`, atoms pairwise distinct.
    terms: Vec<(Term, u64)>,
}

impl LinNat {
    fn zero() -> LinNat {
        LinNat { konst: 0, terms: Vec::new() }
    }
    /// Add a single canonicalized atom with coefficient 1.
    fn atom(t: Term) -> LinNat {
        LinNat { konst: 0, terms: vec![(t, 1)] }
    }
    /// Merge two linear combinations (constants add; coefficients of equal atoms
    /// add). O(n·m), fine for index-sized expressions.
    fn plus(mut self, other: LinNat) -> LinNat {
        self.konst += other.konst;
        for (atom, c) in other.terms {
            match self.terms.iter_mut().find(|(a, _)| *a == atom) {
                Some((_, existing)) => *existing += c,
                None => self.terms.push((atom, c)),
            }
        }
        self
    }
}

/// Decompose a term into its linear-Nat combination. Arithmetic constructors are
/// broken apart; anything else is an opaque atom (canonicalized so that e.g.
/// `f (n + 0)` and `f n` collapse to the same atom).
fn to_lin(t: &Term) -> LinNat {
    match t {
        Term::Zero => LinNat::zero(),
        Term::NatLit(k) => LinNat { konst: *k, terms: Vec::new() },
        Term::Suc(x) => to_lin(x).plus(LinNat { konst: 1, terms: Vec::new() }),
        Term::Add(a, b) => to_lin(a).plus(to_lin(b)),
        other => LinNat::atom(canon(other)),
    }
}

/// Emit the canonical `Term` for a linear combination: atoms in a deterministic
/// order (each repeated by its coefficient), then the constant. A lone part is
/// emitted bare (so an atom `n` stays `n`, not `n + 0`); the empty combination is
/// `NatLit(0)`. Everything routes through here, so two equal combinations — no
/// matter how they were written — produce byte-identical terms.
fn emit(mut lin: LinNat) -> Term {
    // Deterministic atom order. Distinct terms have distinct `Debug`, so this is a
    // total order on the atoms present; the sort is stable, so it is well-defined.
    lin.terms.sort_by(|(a, _), (b, _)| format!("{a:?}").cmp(&format!("{b:?}")));
    let mut parts: Vec<Term> = Vec::new();
    for (atom, c) in &lin.terms {
        for _ in 0..*c {
            parts.push(atom.clone());
        }
    }
    if lin.konst > 0 || parts.is_empty() {
        parts.push(Term::NatLit(lin.konst));
    }
    let mut it = parts.into_iter();
    let mut acc = it.next().expect("emit always produces at least one part");
    for p in it {
        acc = Term::Add(Box::new(acc), Box::new(p));
    }
    acc
}

/// Canonicalize a term: maximal Nat-arithmetic subterms are rewritten to their
/// canonical linear-combination form; all other structure is preserved, with
/// children canonicalized recursively (so arithmetic nested anywhere is normal).
pub fn canon(t: &Term) -> Term {
    match t {
        // The arithmetic fragment — normalize the whole maximal expression.
        Term::Zero | Term::NatLit(_) | Term::Suc(_) | Term::Add(_, _) => emit(to_lin(t)),

        // Leaves with no subterms.
        Term::Var(_) | Term::Type(_) | Term::Nat | Term::Const(_) => t.clone(),

        // Everything else: rebuild with canonicalized children.
        Term::Pi(m, a, b) => Term::Pi(*m, Box::new(canon(a)), Box::new(canon(b))),
        Term::Sigma(m, a, b) => Term::Sigma(*m, Box::new(canon(a)), Box::new(canon(b))),
        Term::Lam(b) => Term::Lam(Box::new(canon(b))),
        Term::App(f, a) => Term::App(Box::new(canon(f)), Box::new(canon(a))),
        Term::Pair(a, b) => Term::Pair(Box::new(canon(a)), Box::new(canon(b))),
        Term::Fst(p) => Term::Fst(Box::new(canon(p))),
        Term::Snd(p) => Term::Snd(Box::new(canon(p))),
        Term::NatElim(p, z, s, sc) => Term::NatElim(
            Box::new(canon(p)),
            Box::new(canon(z)),
            Box::new(canon(s)),
            Box::new(canon(sc)),
        ),
        Term::NatCase(p, z, s, sc) => Term::NatCase(
            Box::new(canon(p)),
            Box::new(canon(z)),
            Box::new(canon(s)),
            Box::new(canon(sc)),
        ),
        Term::Fix(ty, body) => Term::Fix(Box::new(canon(ty)), Box::new(canon(body))),
        Term::J(pm, b, e) => {
            Term::J(Box::new(canon(pm)), Box::new(canon(b)), Box::new(canon(e)))
        }
        Term::Eq(a, x, y) => {
            Term::Eq(Box::new(canon(a)), Box::new(canon(x)), Box::new(canon(y)))
        }
        Term::Refl(a) => Term::Refl(Box::new(canon(a))),
        Term::Data(n, args) => Term::Data(n.clone(), args.iter().map(canon).collect()),
        Term::Constr(n, args) => Term::Constr(n.clone(), args.iter().map(canon).collect()),
        Term::Elim(d, m, methods, sc) => Term::Elim(
            d.clone(),
            Box::new(canon(m)),
            methods.iter().map(canon).collect(),
            Box::new(canon(sc)),
        ),
        Term::Case(d, m, methods, sc) => Term::Case(
            d.clone(),
            Box::new(canon(m)),
            methods.iter().map(canon).collect(),
            Box::new(canon(sc)),
        ),
        Term::Ann(e, ty) => Term::Ann(Box::new(canon(e)), Box::new(canon(ty))),
        Term::Let(m, ty, e, body) => Term::Let(
            *m,
            Box::new(canon(ty)),
            Box::new(canon(e)),
            Box::new(canon(body)),
        ),
    }
}

/// Two terms are equal in the linear-Nat domain iff their canonical forms match.
/// (Exposed for direct testing; `conv` uses `canon` on quoted normal forms.)
pub fn lin_eq(a: &Term, b: &Term) -> bool {
    canon(a) == canon(b)
}

/// Coefficient of `atom` in a linear combination (0 if absent).
fn coeff_of(lin: &LinNat, atom: &Term) -> u64 {
    lin.terms.iter().find(|(a, _)| a == atom).map(|(_, c)| *c).unwrap_or(0)
}

/// DECISION for `subtrahend ≤ minuend` over `Nat` (all atoms range over ℕ,
/// independently). Returns `Some(d)` — a canonical `Term` witnessing the
/// difference, i.e. `subtrahend + d ≡ minuend` — when the inequality holds for
/// *every* valuation, and `None` when it does not.
///
/// Soundness/completeness: over ℕ with independent atoms, `s ≤ m` holds for all
/// valuations iff `m − s` is nonnegative componentwise — the constant and *every*
/// atom coefficient of `m` are ≥ those of `s`. (If some atom's coefficient were
/// larger in `s`, sending that atom to ∞ breaks the inequality; if all are ≥, the
/// difference is a genuine ℕ combination.) So this is an exact decision, and the
/// returned `d` makes `subtrahend + d = minuend` a true identity — which is why
/// the emitted `refl` re-checks in the kernel. Pure decision, no search.
pub fn diff_witness(minuend: &Term, subtrahend: &Term) -> Option<Term> {
    let m = to_lin(minuend);
    let s = to_lin(subtrahend);
    if m.konst < s.konst {
        return None;
    }
    let mut diff = LinNat { konst: m.konst - s.konst, terms: Vec::new() };
    // every atom of `m`: coefficient must not drop below `s`'s.
    for (atom, mc) in &m.terms {
        let sc = coeff_of(&s, atom);
        if *mc < sc {
            return None;
        }
        if *mc - sc > 0 {
            diff.terms.push((atom.clone(), *mc - sc));
        }
    }
    // any atom present in `s` but not `m` has minuend-coefficient 0 < subtrahend's.
    for (atom, sc) in &s.terms {
        if *sc > 0 && coeff_of(&m, atom) == 0 {
            return None;
        }
    }
    Some(emit(diff))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(i: usize) -> Term {
        Term::Var(i)
    }
    fn add(a: Term, b: Term) -> Term {
        Term::Add(Box::new(a), Box::new(b))
    }
    fn suc(a: Term) -> Term {
        Term::Suc(Box::new(a))
    }

    // ---- equalities the domain MUST decide (soundness of coarsening: all true) ----

    #[test]
    fn commutativity() {
        assert!(lin_eq(&add(v(0), v(1)), &add(v(1), v(0))));
    }

    #[test]
    fn associativity() {
        let l = add(add(v(0), v(1)), v(2));
        let r = add(v(0), add(v(1), v(2)));
        assert!(lin_eq(&l, &r));
    }

    #[test]
    fn right_identity() {
        assert!(lin_eq(&add(v(0), Term::NatLit(0)), &v(0)));
        assert!(lin_eq(&add(v(0), Term::Zero), &v(0)));
    }

    #[test]
    fn left_identity() {
        assert!(lin_eq(&add(Term::NatLit(0), v(0)), &v(0)));
    }

    #[test]
    fn suc_is_plus_one() {
        assert!(lin_eq(&suc(v(0)), &add(v(0), Term::NatLit(1))));
        assert!(lin_eq(&add(Term::NatLit(1), v(0)), &suc(v(0))));
    }

    #[test]
    fn literal_folding_and_suc_stack() {
        assert!(lin_eq(&add(Term::NatLit(2), Term::NatLit(3)), &Term::NatLit(5)));
        assert!(lin_eq(&suc(suc(v(0))), &add(v(0), Term::NatLit(2))));
    }

    #[test]
    fn coefficients_accumulate() {
        // n + n  ≡  2·n ;  and  (n + m) + n  ≡  (m + n) + n
        let l = add(add(v(0), v(1)), v(0));
        let r = add(add(v(1), v(0)), v(0));
        assert!(lin_eq(&l, &r));
    }

    #[test]
    fn arithmetic_normalized_under_atoms() {
        // f (n + 0)  ≡  f n   — the atom's internals are canonicalized too.
        let app = |x| Term::App(Box::new(v(9)), Box::new(x));
        assert!(lin_eq(&app(add(v(0), Term::NatLit(0))), &app(v(0))));
    }

    // ---- NON-equalities the domain MUST keep distinct (no over-eager equating) ----

    #[test]
    fn distinct_variables_stay_distinct() {
        assert!(!lin_eq(&v(0), &v(1)));
        assert!(!lin_eq(&add(v(0), v(1)), &v(0)));
    }

    #[test]
    fn different_coefficients_stay_distinct() {
        // n + n  ≢  n
        assert!(!lin_eq(&add(v(0), v(0)), &v(0)));
    }

    #[test]
    fn distinct_atoms_stay_distinct() {
        // f a  ≢  f b  (opaque atoms compared structurally)
        let fa = Term::App(Box::new(v(9)), Box::new(v(0)));
        let fb = Term::App(Box::new(v(9)), Box::new(v(1)));
        assert!(!lin_eq(&fa, &fb));
    }

    #[test]
    fn constants_differ() {
        assert!(!lin_eq(&add(v(0), Term::NatLit(1)), &add(v(0), Term::NatLit(2))));
    }

    #[test]
    fn idempotent() {
        let t = add(suc(v(0)), add(v(1), Term::NatLit(2)));
        let c = canon(&t);
        assert_eq!(c, canon(&c));
    }

    // ---- the inequality decision (diff_witness) ----

    /// `subtrahend + witness ≡ minuend` must hold whenever a witness is returned.
    fn witness_is_valid(minuend: &Term, subtrahend: &Term) {
        let d = diff_witness(minuend, subtrahend).expect("expected the bound to hold");
        assert!(
            lin_eq(&add(subtrahend.clone(), d), minuend),
            "witness must satisfy subtrahend + d = minuend"
        );
    }

    #[test]
    fn closed_bounds_decided() {
        witness_is_valid(&Term::NatLit(3), &Term::NatLit(1)); // 1 ≤ 3
        assert!(diff_witness(&Term::NatLit(3), &Term::NatLit(3)).is_some()); // 3 ≤ 3
        assert!(diff_witness(&Term::NatLit(2), &Term::NatLit(3)).is_none()); // 3 ≤ 2 ✗
    }

    #[test]
    fn open_bounds_decided() {
        // n ≤ n + m  (the case the inductive `LT` cannot build over variables)
        witness_is_valid(&add(v(0), v(1)), &v(0));
        // n + 1 ≤ n + m is NOT valid for all m (m could be 0)
        assert!(diff_witness(&add(v(0), v(1)), &add(v(0), Term::NatLit(1))).is_none());
        // n < n is false: n + 1 ≤ n ✗
        assert!(diff_witness(&v(0), &add(v(0), Term::NatLit(1))).is_none());
    }

    #[test]
    fn coefficient_bounds_decided() {
        // n ≤ 2·n  (i.e. n ≤ n + n)
        witness_is_valid(&add(v(0), v(0)), &v(0));
        // 2·n ≤ n is false
        assert!(diff_witness(&v(0), &add(v(0), v(0))).is_none());
        // an atom only in the subtrahend ⇒ invalid (m ≤ n fails when m huge)
        assert!(diff_witness(&v(0), &v(1)).is_none());
    }
}
