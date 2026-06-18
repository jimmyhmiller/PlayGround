//! The DEPENDENT + LINEAR core (Route B), via normalization by evaluation: a
//! Quantitative Type Theory (QTT) checker. A Rust port of the engine in
//! `../../prototype/qtt_checker.py`; the metatheory is `../../agda/Dependent.agda`.
//!
//! One syntax for terms AND types. Every `Π` carries a MULTIPLICITY `{0,1,ω}`
//! (`Π[π](x:A). B`), and the bidirectional checker threads a USAGE CONTEXT — one
//! multiplicity per variable, combined with the rig's `+` and `·`. So linearity
//! is enforced UNDER dependency:
//!
//!   * `Π[0]` — erased: the argument exists only in types (the 0-fragment);
//!   * `Π[1]` — linear: the argument is used exactly once;
//!   * `Π[ω]` — unrestricted.
//!
//! The headline is the dependent+linear+erasure unification, the polymorphic
//! LINEAR identity, machine-checked here:
//!     λA. λx. x   :   Π[0](A:Type). Π[1](x:A). A
//! `A` is used 0 times at runtime (erased, appears only in the type); `x` is used
//! exactly once. And proofs are still terms checked by computation
//! (`refl : Eq Nat (2+2) 4`), now living in the 0-fragment.
//!
//! `Type : Type` for now (fine for a language; for a logic, a universe hierarchy
//! is needed — ../../docs/07-implementation-guide.md §7).
//!
//! Inductive families with DEPENDENT ELIMINATORS (the gate to Idris-like data):
//! `Nat` (zero/suc + `natElim`) and the length-indexed `Vec A n` (nil/cons +
//! `vecElim`). The eliminator is the only recursion, so functions written with
//! it are TOTAL BY CONSTRUCTION (docs/09 §1.3). `add`/`append` are then
//! *definitions*; `append`'s result index `add m n` type-checks because the
//! eliminator makes `+` reduce definitionally. (These two families are
//! hand-written; generalising to user-declared strictly-positive families is
//! the next step — see the README.)

use crate::mult::Mult;

#[derive(Clone, PartialEq, Debug)]
pub enum Term {
    Var(usize),                                   // de Bruijn index (0 = innermost)
    Type,
    Pi(Mult, Box<Term>, Box<Term>),               // Π[π](_:A). B
    Lam(Box<Term>),
    App(Box<Term>, Box<Term>),
    Sigma(Mult, Box<Term>, Box<Term>),            // Σ[π](_:A). B   (dependent pair type)
    Pair(Box<Term>, Box<Term>),
    Fst(Box<Term>),
    Snd(Box<Term>),
    Nat,
    NatLit(u64),
    Zero,                                         // the constructor 0
    Suc(Box<Term>),                               // the constructor suc
    /// the dependent eliminator (induction):  natElim P z s n : P n
    ///   P : Nat → Type,  z : P 0,  s : Π(k:Nat). P k → P (suc k)
    NatElim(Box<Term>, Box<Term>, Box<Term>, Box<Term>),
    Add(Box<Term>, Box<Term>),
    // ---- Vec: a LENGTH-INDEXED inductive family (the headline dependent data) ----
    VecTy(Box<Term>, Box<Term>),                  // Vec A n           : Type
    Nil(Box<Term>),                               // nil A             : Vec A 0
    Cons(Box<Term>, Box<Term>, Box<Term>, Box<Term>), // cons A n h t  : Vec A (suc n)
    /// the dependent eliminator:  vecElim A P pnil pcons n xs : P n xs
    ///   P     : Π(k:Nat). Vec A k → Type
    ///   pnil  : P 0 (nil A)
    ///   pcons : Π(k:Nat). Π(h:A). Π(t:Vec A k). P k t → P (suc k) (cons A k h t)
    VecElim(
        Box<Term>,
        Box<Term>,
        Box<Term>,
        Box<Term>,
        Box<Term>,
        Box<Term>,
    ),
    Eq(Box<Term>, Box<Term>, Box<Term>),
    Refl(Box<Term>),
    Ann(Box<Term>, Box<Term>),
}

#[derive(Clone)]
pub enum Value {
    VType,
    VPi(Mult, Box<Value>, Closure),
    VLam(Closure),
    VSigma(Mult, Box<Value>, Closure),
    VPair(Box<Value>, Box<Value>),
    VNat,
    VNatLit(u64),
    VSuc(Box<Value>),                             // suc of a *neutral* Nat (open term)
    VVecTy(Box<Value>, Box<Value>),               // Vec A n
    VNil(Box<Value>),                             // nil A
    VCons(Box<Value>, Box<Value>, Box<Value>, Box<Value>), // cons A n h t
    VEq(Box<Value>, Box<Value>, Box<Value>),
    VRefl(Box<Value>),
    VNeu(Neutral),
}

#[derive(Clone)]
pub enum Neutral {
    NVar(usize), // de Bruijn LEVEL
    NApp(Box<Neutral>, Box<Value>),
    NAdd(Box<Value>, Box<Value>),
    NFst(Box<Neutral>),
    NSnd(Box<Neutral>),
    /// natElim stuck on a neutral scrutinee: P, z, s, and the stuck Nat.
    NNatElim(Box<Value>, Box<Value>, Box<Value>, Box<Neutral>),
    /// vecElim stuck on a neutral scrutinee: A, P, pnil, pcons, n, and the stuck Vec.
    NVecElim(
        Box<Value>,
        Box<Value>,
        Box<Value>,
        Box<Value>,
        Box<Value>,
        Box<Neutral>,
    ),
}

#[derive(Clone)]
pub struct Closure {
    env: Vec<Value>,
    body: Term,
}

impl Closure {
    fn apply(&self, arg: Value) -> Value {
        let mut env = self.env.clone();
        env.push(arg);
        eval(&env, &self.body)
    }
}

// ---------------------------------------------------------------------------
// evaluation / quotation / conversion (NbE) -- unchanged except Π carries π
// ---------------------------------------------------------------------------

fn eval(env: &[Value], t: &Term) -> Value {
    match t {
        Term::Var(i) => env[env.len() - 1 - i].clone(),
        Term::Type => Value::VType,
        Term::Pi(pi, a, b) => Value::VPi(
            *pi,
            Box::new(eval(env, a)),
            Closure { env: env.to_vec(), body: (**b).clone() },
        ),
        Term::Lam(b) => Value::VLam(Closure { env: env.to_vec(), body: (**b).clone() }),
        Term::App(f, a) => vapp(eval(env, f), eval(env, a)),
        Term::Sigma(pi, a, b) => Value::VSigma(
            *pi,
            Box::new(eval(env, a)),
            Closure { env: env.to_vec(), body: (**b).clone() },
        ),
        Term::Pair(a, b) => Value::VPair(Box::new(eval(env, a)), Box::new(eval(env, b))),
        Term::Fst(p) => vfst(eval(env, p)),
        Term::Snd(p) => vsnd(eval(env, p)),
        Term::Nat => Value::VNat,
        Term::NatLit(n) => Value::VNatLit(*n),
        Term::Zero => Value::VNatLit(0),
        Term::Suc(t) => vsuc(eval(env, t)),
        Term::NatElim(p, z, s, scrut) => {
            vnatelim(eval(env, p), eval(env, z), eval(env, s), eval(env, scrut))
        }
        Term::VecTy(a, n) => Value::VVecTy(Box::new(eval(env, a)), Box::new(eval(env, n))),
        Term::Nil(a) => Value::VNil(Box::new(eval(env, a))),
        Term::Cons(a, n, h, t) => Value::VCons(
            Box::new(eval(env, a)),
            Box::new(eval(env, n)),
            Box::new(eval(env, h)),
            Box::new(eval(env, t)),
        ),
        Term::VecElim(a, p, pn, pc, n, xs) => vvecelim(
            eval(env, a),
            eval(env, p),
            eval(env, pn),
            eval(env, pc),
            eval(env, n),
            eval(env, xs),
        ),
        Term::Add(a, b) => match (eval(env, a), eval(env, b)) {
            (Value::VNatLit(x), Value::VNatLit(y)) => Value::VNatLit(x + y),
            (va, vb) => Value::VNeu(Neutral::NAdd(Box::new(va), Box::new(vb))),
        },
        Term::Eq(a, x, y) => Value::VEq(
            Box::new(eval(env, a)),
            Box::new(eval(env, x)),
            Box::new(eval(env, y)),
        ),
        Term::Refl(a) => Value::VRefl(Box::new(eval(env, a))),
        Term::Ann(e, _) => eval(env, e),
    }
}

fn vapp(f: Value, a: Value) -> Value {
    match f {
        Value::VLam(clo) => clo.apply(a),
        Value::VNeu(n) => Value::VNeu(Neutral::NApp(Box::new(n), Box::new(a))),
        _ => unreachable!("vapp on a non-function (ill-typed term reached eval)"),
    }
}

fn vsuc(n: Value) -> Value {
    match n {
        Value::VNatLit(k) => Value::VNatLit(k + 1),
        other => Value::VSuc(Box::new(other)), // suc of a stuck Nat
    }
}

/// The ι-rule for `natElim P z s n`:
///   natElim P z s 0       = z
///   natElim P z s (suc k) = s k (natElim P z s k)
/// stuck when `n` is neutral.
fn vnatelim(p: Value, z: Value, s: Value, scrut: Value) -> Value {
    match scrut {
        Value::VNatLit(0) => z,
        Value::VNatLit(k) => {
            let pred = Value::VNatLit(k - 1);
            let rec = vnatelim(p.clone(), z, s.clone(), pred.clone());
            vapp(vapp(s, pred), rec)
        }
        Value::VSuc(pred) => {
            let rec = vnatelim(p.clone(), z, s.clone(), (*pred).clone());
            vapp(vapp(s, *pred), rec)
        }
        Value::VNeu(nu) => Value::VNeu(Neutral::NNatElim(
            Box::new(p),
            Box::new(z),
            Box::new(s),
            Box::new(nu),
        )),
        _ => unreachable!("natElim on a non-Nat (ill-typed term reached eval)"),
    }
}

/// The ι-rule for `vecElim A P pnil pcons n xs`:
///   vecElim _ _ pnil _     0       (nil _)        = pnil
///   vecElim A P pnil pcons (suc m) (cons _ m h t) = pcons m h t (vecElim A P pnil pcons m t)
/// stuck when `xs` is neutral.
fn vvecelim(va: Value, vp: Value, vpn: Value, vpc: Value, vn: Value, vxs: Value) -> Value {
    match vxs {
        Value::VNil(_) => vpn,
        Value::VCons(_, m, h, t) => {
            let rec = vvecelim(va, vp, vpn.clone(), vpc.clone(), (*m).clone(), (*t).clone());
            vapp(vapp(vapp(vapp(vpc, *m), *h), *t), rec)
        }
        Value::VNeu(nu) => Value::VNeu(Neutral::NVecElim(
            Box::new(va),
            Box::new(vp),
            Box::new(vpn),
            Box::new(vpc),
            Box::new(vn),
            Box::new(nu),
        )),
        _ => unreachable!("vecElim on a non-Vec (ill-typed term reached eval)"),
    }
}

fn vfst(p: Value) -> Value {
    match p {
        Value::VPair(a, _) => *a,
        Value::VNeu(n) => Value::VNeu(Neutral::NFst(Box::new(n))),
        _ => unreachable!("fst of a non-pair"),
    }
}

fn vsnd(p: Value) -> Value {
    match p {
        Value::VPair(_, b) => *b,
        Value::VNeu(n) => Value::VNeu(Neutral::NSnd(Box::new(n))),
        _ => unreachable!("snd of a non-pair"),
    }
}

fn quote(lvl: usize, v: &Value) -> Term {
    match v {
        Value::VType => Term::Type,
        Value::VPi(pi, a, clo) => Term::Pi(
            *pi,
            Box::new(quote(lvl, a)),
            Box::new(quote(lvl + 1, &clo.apply(Value::VNeu(Neutral::NVar(lvl))))),
        ),
        Value::VLam(clo) => {
            Term::Lam(Box::new(quote(lvl + 1, &clo.apply(Value::VNeu(Neutral::NVar(lvl))))))
        }
        Value::VSigma(pi, a, clo) => Term::Sigma(
            *pi,
            Box::new(quote(lvl, a)),
            Box::new(quote(lvl + 1, &clo.apply(Value::VNeu(Neutral::NVar(lvl))))),
        ),
        Value::VPair(a, b) => Term::Pair(Box::new(quote(lvl, a)), Box::new(quote(lvl, b))),
        Value::VNat => Term::Nat,
        Value::VNatLit(n) => Term::NatLit(*n),
        Value::VSuc(k) => Term::Suc(Box::new(quote(lvl, k))),
        Value::VVecTy(a, n) => Term::VecTy(Box::new(quote(lvl, a)), Box::new(quote(lvl, n))),
        Value::VNil(a) => Term::Nil(Box::new(quote(lvl, a))),
        Value::VCons(a, n, h, t) => Term::Cons(
            Box::new(quote(lvl, a)),
            Box::new(quote(lvl, n)),
            Box::new(quote(lvl, h)),
            Box::new(quote(lvl, t)),
        ),
        Value::VEq(a, x, y) => Term::Eq(
            Box::new(quote(lvl, a)),
            Box::new(quote(lvl, x)),
            Box::new(quote(lvl, y)),
        ),
        Value::VRefl(a) => Term::Refl(Box::new(quote(lvl, a))),
        Value::VNeu(n) => quote_neu(lvl, n),
    }
}

fn quote_neu(lvl: usize, n: &Neutral) -> Term {
    match n {
        Neutral::NVar(k) => Term::Var(lvl - 1 - k),
        Neutral::NApp(f, a) => Term::App(Box::new(quote_neu(lvl, f)), Box::new(quote(lvl, a))),
        Neutral::NAdd(a, b) => Term::Add(Box::new(quote(lvl, a)), Box::new(quote(lvl, b))),
        Neutral::NFst(p) => Term::Fst(Box::new(quote_neu(lvl, p))),
        Neutral::NSnd(p) => Term::Snd(Box::new(quote_neu(lvl, p))),
        Neutral::NNatElim(p, z, s, scrut) => Term::NatElim(
            Box::new(quote(lvl, p)),
            Box::new(quote(lvl, z)),
            Box::new(quote(lvl, s)),
            Box::new(quote_neu(lvl, scrut)),
        ),
        Neutral::NVecElim(a, p, pn, pc, n, scrut) => Term::VecElim(
            Box::new(quote(lvl, a)),
            Box::new(quote(lvl, p)),
            Box::new(quote(lvl, pn)),
            Box::new(quote(lvl, pc)),
            Box::new(quote(lvl, n)),
            Box::new(quote_neu(lvl, scrut)),
        ),
    }
}

fn conv(lvl: usize, a: &Value, b: &Value) -> bool {
    quote(lvl, a) == quote(lvl, b)
}

/// de Bruijn shift: add `d` to every free `Var(i)` with `i >= cutoff`. Used to
/// lift a quoted term under newly-introduced binders when building the expected
/// type of `natElim`'s step argument.
fn shift(d: usize, cutoff: usize, t: &Term) -> Term {
    match t {
        Term::Var(i) => Term::Var(if *i >= cutoff { *i + d } else { *i }),
        Term::Type | Term::Nat | Term::NatLit(_) | Term::Zero => t.clone(),
        Term::Pi(pi, a, b) => Term::Pi(
            *pi,
            Box::new(shift(d, cutoff, a)),
            Box::new(shift(d, cutoff + 1, b)),
        ),
        Term::Sigma(pi, a, b) => Term::Sigma(
            *pi,
            Box::new(shift(d, cutoff, a)),
            Box::new(shift(d, cutoff + 1, b)),
        ),
        Term::Lam(b) => Term::Lam(Box::new(shift(d, cutoff + 1, b))),
        Term::App(f, a) => Term::App(Box::new(shift(d, cutoff, f)), Box::new(shift(d, cutoff, a))),
        Term::Pair(a, b) => {
            Term::Pair(Box::new(shift(d, cutoff, a)), Box::new(shift(d, cutoff, b)))
        }
        Term::Fst(p) => Term::Fst(Box::new(shift(d, cutoff, p))),
        Term::Snd(p) => Term::Snd(Box::new(shift(d, cutoff, p))),
        Term::Suc(k) => Term::Suc(Box::new(shift(d, cutoff, k))),
        Term::NatElim(p, z, s, scrut) => Term::NatElim(
            Box::new(shift(d, cutoff, p)),
            Box::new(shift(d, cutoff, z)),
            Box::new(shift(d, cutoff, s)),
            Box::new(shift(d, cutoff, scrut)),
        ),
        Term::VecTy(a, n) => {
            Term::VecTy(Box::new(shift(d, cutoff, a)), Box::new(shift(d, cutoff, n)))
        }
        Term::Nil(a) => Term::Nil(Box::new(shift(d, cutoff, a))),
        Term::Cons(a, n, h, t) => Term::Cons(
            Box::new(shift(d, cutoff, a)),
            Box::new(shift(d, cutoff, n)),
            Box::new(shift(d, cutoff, h)),
            Box::new(shift(d, cutoff, t)),
        ),
        Term::VecElim(a, p, pn, pc, n, xs) => Term::VecElim(
            Box::new(shift(d, cutoff, a)),
            Box::new(shift(d, cutoff, p)),
            Box::new(shift(d, cutoff, pn)),
            Box::new(shift(d, cutoff, pc)),
            Box::new(shift(d, cutoff, n)),
            Box::new(shift(d, cutoff, xs)),
        ),
        Term::Add(a, b) => Term::Add(Box::new(shift(d, cutoff, a)), Box::new(shift(d, cutoff, b))),
        Term::Eq(a, x, y) => Term::Eq(
            Box::new(shift(d, cutoff, a)),
            Box::new(shift(d, cutoff, x)),
            Box::new(shift(d, cutoff, y)),
        ),
        Term::Refl(a) => Term::Refl(Box::new(shift(d, cutoff, a))),
        Term::Ann(e, ty) => {
            Term::Ann(Box::new(shift(d, cutoff, e)), Box::new(shift(d, cutoff, ty)))
        }
    }
}

// ---------------------------------------------------------------------------
// usage contexts: one multiplicity per variable (indexed by level)
// ---------------------------------------------------------------------------

type Usage = Vec<Mult>;

fn uzero(n: usize) -> Usage {
    vec![Mult::Zero; n]
}
fn uunit(n: usize, lvl: usize) -> Usage {
    let mut u = uzero(n);
    u[lvl] = Mult::One;
    u
}
fn uadd(a: &Usage, b: &Usage) -> Usage {
    a.iter().zip(b).map(|(x, y)| x.add(*y)).collect()
}
fn uscale(p: Mult, a: &Usage) -> Usage {
    a.iter().map(|x| p.mul(*x)).collect()
}

// ---------------------------------------------------------------------------
// resourced bidirectional checking: check/infer compute a usage context
// ---------------------------------------------------------------------------

pub struct Ctx {
    types: Vec<Value>,
}

impl Ctx {
    pub fn new() -> Ctx {
        Ctx { types: Vec::new() }
    }
    fn level(&self) -> usize {
        self.types.len()
    }
    fn env(&self) -> Vec<Value> {
        (0..self.level()).map(|k| Value::VNeu(Neutral::NVar(k))).collect()
    }
    fn extend(&self, ty: Value) -> Ctx {
        let mut types = self.types.clone();
        types.push(ty);
        Ctx { types }
    }
}

/// Check that `t` is a well-formed TYPE. Type formation is the 0-fragment, so
/// the usage it incurs is discarded (everything in a type is used 0 times).
fn check_type(ctx: &Ctx, t: &Term) -> Result<(), String> {
    check(ctx, t, &Value::VType).map(|_| ())
}

fn check(ctx: &Ctx, t: &Term, ty: &Value) -> Result<Usage, String> {
    let n = ctx.level();
    match (t, ty) {
        (Term::Lam(body), Value::VPi(pi, dom, cod)) => {
            let ctx2 = ctx.extend((**dom).clone());
            let codty = cod.apply(Value::VNeu(Neutral::NVar(n)));
            let mut ub = check(&ctx2, body, &codty)?; // length n+1
            let sigma = ub.pop().unwrap(); // the bound variable's usage
            if !sigma.leq(*pi) {
                return Err(format!(
                    "the variable bound at multiplicity {pi} is used {sigma} time(s) \
                     ({sigma} ⋢ {pi})"
                ));
            }
            Ok(ub)
        }
        (Term::Pair(a, b), Value::VSigma(pi, dom, cod)) => {
            let ua = check(ctx, a, dom)?;
            let va = eval(&ctx.env(), a);
            let ub = check(ctx, b, &cod.apply(va))?; // B[a] — the dependency
            Ok(uadd(&uscale(*pi, &ua), &ub)) // first component used at multiplicity π
        }
        (Term::Refl(a), Value::VEq(aty, x, y)) => {
            check(ctx, a, aty)?; // a proof lives in the 0-fragment; discard usage
            let va = eval(&ctx.env(), a);
            if conv(n, x, &va) && conv(n, y, &va) {
                Ok(uzero(n))
            } else {
                Err(format!(
                    "refl: the equation does not hold by computation:\n  {:?}  =/=  {:?}",
                    quote(n, x),
                    quote(n, y)
                ))
            }
        }
        _ => {
            let (ty2, u) = infer(ctx, t)?;
            if conv(n, ty, &ty2) {
                Ok(u)
            } else {
                Err(format!(
                    "type mismatch:\n  expected {:?}\n  found    {:?}",
                    quote(n, ty),
                    quote(n, &ty2)
                ))
            }
        }
    }
}

fn infer(ctx: &Ctx, t: &Term) -> Result<(Value, Usage), String> {
    let n = ctx.level();
    match t {
        Term::Var(i) => {
            if *i >= n {
                return Err(format!("unbound variable (de Bruijn {i})"));
            }
            let lvl = n - 1 - i;
            Ok((ctx.types[lvl].clone(), uunit(n, lvl)))
        }
        Term::Type | Term::Nat => Ok((Value::VType, uzero(n))),
        Term::NatLit(_) | Term::Zero => Ok((Value::VNat, uzero(n))),
        Term::Suc(k) => {
            let u = check(ctx, k, &Value::VNat)?;
            Ok((Value::VNat, u))
        }
        Term::NatElim(p, z, s, scrut) => {
            // P : Nat → Type  (a type-former; its usage is in the 0-fragment)
            let motive_ty = Value::VPi(
                Mult::Omega,
                Box::new(Value::VNat),
                Closure { env: vec![], body: Term::Type },
            );
            check(ctx, p, &motive_ty)?;
            let vp = eval(&ctx.env(), p);

            // z : P 0
            let pzero = vapp(vp.clone(), Value::VNatLit(0));
            let uz = check(ctx, z, &pzero)?;

            // s : Π[ω](k:Nat). Π[ω](_:P k). P (suc k)
            let p_tm = quote(n, &vp);
            let sty_term = Term::Pi(
                Mult::Omega,
                Box::new(Term::Nat),
                Box::new(Term::Pi(
                    Mult::Omega,
                    // P k          (k is Var(0) under the outer binder)
                    Box::new(Term::App(Box::new(shift(1, 0, &p_tm)), Box::new(Term::Var(0)))),
                    // P (suc k)    (k is Var(1) under both binders)
                    Box::new(Term::App(
                        Box::new(shift(2, 0, &p_tm)),
                        Box::new(Term::Suc(Box::new(Term::Var(1)))),
                    )),
                )),
            );
            let vsty = eval(&ctx.env(), &sty_term);
            let us = check(ctx, s, &vsty)?;

            // n : Nat
            let uscr = check(ctx, scrut, &Value::VNat)?;

            // result type: P n
            let result = vapp(vp, eval(&ctx.env(), scrut));
            // z is used 0/1×, s is used n×, the scrutinee is inspected — all
            // bounded above by ω, so scale conservatively (sound; see docs/09).
            let u = uadd(
                &uscale(Mult::Omega, &uz),
                &uadd(&uscale(Mult::Omega, &us), &uscale(Mult::Omega, &uscr)),
            );
            Ok((result, u))
        }
        Term::VecTy(a, len) => {
            check_type(ctx, a)?; // A : Type
            check(ctx, len, &Value::VNat)?; // n : Nat   (the index, 0-fragment)
            Ok((Value::VType, uzero(n)))
        }
        Term::Nil(a) => {
            check_type(ctx, a)?; // A : Type (the element type is a parameter)
            let va = eval(&ctx.env(), a);
            // nil A : Vec A 0
            Ok((
                Value::VVecTy(Box::new(va), Box::new(Value::VNatLit(0))),
                uzero(n),
            ))
        }
        Term::Cons(a, len, h, t) => {
            check_type(ctx, a)?;
            check(ctx, len, &Value::VNat)?; // the index is erased
            let va = eval(&ctx.env(), a);
            let vlen = eval(&ctx.env(), len);
            let uh = check(ctx, h, &va)?; // head : A
            let ut = check(ctx, t, &Value::VVecTy(Box::new(va.clone()), Box::new(vlen.clone())))?;
            // cons A n h t : Vec A (suc n)
            Ok((
                Value::VVecTy(Box::new(va), Box::new(vsuc(vlen))),
                uadd(&uh, &ut),
            ))
        }
        Term::VecElim(a, p, pn, pc, len, xs) => {
            check_type(ctx, a)?; // A : Type (a parameter, 0-fragment)
            let va = eval(&ctx.env(), a);
            let a_tm = quote(n, &va);

            // P : Π[ω](k:Nat). Π[ω](_:Vec A k). Type
            let motive_ty_tm = Term::Pi(
                Mult::Omega,
                Box::new(Term::Nat),
                Box::new(Term::Pi(
                    Mult::Omega,
                    Box::new(Term::VecTy(Box::new(shift(1, 0, &a_tm)), Box::new(Term::Var(0)))),
                    Box::new(Term::Type),
                )),
            );
            let motive_ty = eval(&ctx.env(), &motive_ty_tm);
            check(ctx, p, &motive_ty)?; // motive: 0-fragment, usage discarded
            let vp = eval(&ctx.env(), p);
            let p_tm = quote(n, &vp);

            // pnil : P 0 (nil A)
            let pnil_ty = vapp(
                vapp(vp.clone(), Value::VNatLit(0)),
                Value::VNil(Box::new(va.clone())),
            );
            let upn = check(ctx, pn, &pnil_ty)?;

            // pcons : Π[ω](k:Nat). Π[ω](h:A). Π[ω](t:Vec A k). Π[ω](_:P k t).
            //         P (suc k) (cons A k h t)
            // binders, outside→in: k, h, t, ih   (so k=Var(3), h=Var(2), t=Var(1), ih=Var(0)
            // at the deepest point).
            let pcons_ty_tm = Term::Pi(
                Mult::Omega,
                Box::new(Term::Nat), // k
                Box::new(Term::Pi(
                    Mult::Omega,
                    Box::new(shift(1, 0, &a_tm)), // h : A
                    Box::new(Term::Pi(
                        Mult::Omega,
                        // t : Vec A k        (k = Var(1) under {k,h})
                        Box::new(Term::VecTy(
                            Box::new(shift(2, 0, &a_tm)),
                            Box::new(Term::Var(1)),
                        )),
                        Box::new(Term::Pi(
                            Mult::Omega,
                            // ih : P k t      (k = Var(2), t = Var(0) under {k,h,t})
                            Box::new(Term::App(
                                Box::new(Term::App(
                                    Box::new(shift(3, 0, &p_tm)),
                                    Box::new(Term::Var(2)),
                                )),
                                Box::new(Term::Var(0)),
                            )),
                            // result: P (suc k) (cons A k h t)
                            // under {k,h,t,ih}: k=Var(3), h=Var(2), t=Var(1)
                            Box::new(Term::App(
                                Box::new(Term::App(
                                    Box::new(shift(4, 0, &p_tm)),
                                    Box::new(Term::Suc(Box::new(Term::Var(3)))),
                                )),
                                Box::new(Term::Cons(
                                    Box::new(shift(4, 0, &a_tm)),
                                    Box::new(Term::Var(3)),
                                    Box::new(Term::Var(2)),
                                    Box::new(Term::Var(1)),
                                )),
                            )),
                        )),
                    )),
                )),
            );
            let pcons_ty = eval(&ctx.env(), &pcons_ty_tm);
            let upc = check(ctx, pc, &pcons_ty)?;

            // n : Nat (the length index — erased, 0-fragment)
            check(ctx, len, &Value::VNat)?;
            let vlen = eval(&ctx.env(), len);

            // xs : Vec A n
            let uxs = check(ctx, xs, &Value::VVecTy(Box::new(va), Box::new(vlen.clone())))?;
            let vxs = eval(&ctx.env(), xs);

            // result type: P n xs
            let result = vapp(vapp(vp, vlen), vxs);
            let u = uadd(
                &uscale(Mult::Omega, &upn),
                &uadd(&uscale(Mult::Omega, &upc), &uscale(Mult::Omega, &uxs)),
            );
            Ok((result, u))
        }
        Term::Add(a, b) => {
            let ua = check(ctx, a, &Value::VNat)?;
            let ub = check(ctx, b, &Value::VNat)?;
            Ok((Value::VNat, uadd(&ua, &ub)))
        }
        Term::Pi(_, a, b) | Term::Sigma(_, a, b) => {
            check_type(ctx, a)?;
            let va = eval(&ctx.env(), a);
            check_type(&ctx.extend(va), b)?;
            Ok((Value::VType, uzero(n)))
        }
        Term::Fst(p) => match infer(ctx, p)? {
            (Value::VSigma(_, dom, _), up) => Ok((*dom, up)),
            (other, _) => Err(format!("fst of a non-pair (type {:?})", quote(n, &other))),
        },
        Term::Snd(p) => match infer(ctx, p)? {
            (Value::VSigma(_, _, cod), up) => {
                let v1 = vfst(eval(&ctx.env(), p)); // snd's type depends on fst
                Ok((cod.apply(v1), up))
            }
            (other, _) => Err(format!("snd of a non-pair (type {:?})", quote(n, &other))),
        },
        Term::Pair(_, _) => Err("cannot infer a bare pair; annotate it `(e : T)`".to_string()),
        Term::Eq(a, x, y) => {
            check_type(ctx, a)?;
            let va = eval(&ctx.env(), a);
            check(ctx, x, &va)?; // type-level, usage discarded
            check(ctx, y, &va)?;
            Ok((Value::VType, uzero(n)))
        }
        Term::Refl(a) => {
            let (ta, _) = infer(ctx, a)?;
            let va = eval(&ctx.env(), a);
            Ok((
                Value::VEq(Box::new(ta), Box::new(va.clone()), Box::new(va)),
                uzero(n),
            ))
        }
        Term::App(f, x) => match infer(ctx, f)? {
            (Value::VPi(pi, dom, cod), uf) => {
                let ux = check(ctx, x, &dom)?;
                let vx = eval(&ctx.env(), x);
                Ok((cod.apply(vx), uadd(&uf, &uscale(pi, &ux)))) // u_f + π · u_x
            }
            (other, _) => Err(format!(
                "application of a non-function (type {:?})",
                quote(n, &other)
            )),
        },
        Term::Ann(e, ty) => {
            check_type(ctx, ty)?;
            let vty = eval(&ctx.env(), ty);
            let u = check(ctx, e, &vty)?;
            Ok((vty, u))
        }
        Term::Lam(_) => Err("cannot infer a bare lambda; annotate it `(e : T)`".to_string()),
    }
}

/// Check a closed term against a closed type (usage context is empty).
pub fn check_closed(t: &Term, ty: &Term) -> Result<(), String> {
    let ctx = Ctx::new();
    check_type(&ctx, ty)?;
    let vty = eval(&[], ty);
    check(&ctx, t, &vty).map(|_| ())
}

/// Infer the (β-normal) type of a closed term.
pub fn infer_closed(t: &Term) -> Result<Term, String> {
    Ok(quote(0, &infer(&Ctx::new(), t)?.0))
}

/// Fully normalize a closed term (NbE: eval then quote). This is the partial
/// evaluator — see docs/09 §2: type-level computation, applied known functions,
/// and ι-reductions of `natElim` are all carried out here.
pub fn normalize_closed(t: &Term) -> Term {
    quote(0, &eval(&[], t))
}

#[cfg(test)]
mod tests {
    use super::Term::*;
    use super::*;
    use crate::mult::Mult::{Omega, One, Zero};

    fn b(t: Term) -> Box<Term> {
        Box::new(t)
    }

    #[test]
    fn polymorphic_linear_identity() {
        // λA. λx. x  :  Π[0](A:Type). Π[1](x:A). A
        // A is erased (used 0× at runtime, appears only in the type); x is linear.
        let ty = Pi(Zero, b(Type), b(Pi(One, b(Var(0)), b(Var(1)))));
        let tm = Lam(b(Lam(b(Var(0)))));
        assert!(check_closed(&tm, &ty).is_ok(), "{:?}", check_closed(&tm, &ty));

        // applying it:  id Nat 3  :  Nat   (A erased, the 3 used once)
        let app = App(b(App(b(Ann(b(tm), b(ty))), b(Nat))), b(NatLit(3)));
        assert_eq!(infer_closed(&app), Ok(Nat));
    }

    #[test]
    fn linearity_is_enforced_under_dependency() {
        // Π[1](x:Nat). Nat
        let lin = Pi(One, b(Nat), b(Nat));
        // using x exactly once: ok
        assert!(check_closed(&Lam(b(Var(0))), &lin).is_ok());
        // using x twice: ω ⋢ 1  -> rejected
        assert!(check_closed(&Lam(b(Add(b(Var(0)), b(Var(0))))), &lin).is_err());
        // dropping x: 0 ⋢ 1  -> rejected
        assert!(check_closed(&Lam(b(NatLit(5))), &lin).is_err());

        // Π[ω](x:Nat). Nat  -- now using x twice is fine
        let unr = Pi(Omega, b(Nat), b(Nat));
        assert!(check_closed(&Lam(b(Add(b(Var(0)), b(Var(0))))), &unr).is_ok());
    }

    #[test]
    fn dependent_pairs() {
        // Σ[ω](b:Nat). Eq Nat b 4   -- "a Nat together with a proof it equals 4"
        let ty = Sigma(Omega, b(Nat), b(Eq(b(Nat), b(Var(0)), b(NatLit(4)))));
        // (2+2, refl) inhabits it: the second component's type is Eq Nat (2+2) 4
        let good = Pair(b(Add(b(NatLit(2)), b(NatLit(2)))), b(Refl(b(NatLit(4)))));
        assert!(check_closed(&good, &ty).is_ok(), "{:?}", check_closed(&good, &ty));

        // (5, refl) does NOT: its proof obligation Eq Nat 5 4 is false
        let bad = Pair(b(NatLit(5)), b(Refl(b(NatLit(5)))));
        assert!(check_closed(&bad, &ty).is_err());

        // projection: fst (2+2, _) reduces to 4 : Nat
        let proj = Fst(b(Ann(b(good), b(ty))));
        assert_eq!(infer_closed(&proj), Ok(Nat));
    }

    // `add` defined by the dependent eliminator (induction on the first arg):
    //   add = λm. λn. natElim (λ_.Nat) n (λk. λrec. suc rec) m
    // recursing structurally on m, so it is TOTAL BY CONSTRUCTION (docs/09 §1.3).
    fn add_def() -> Term {
        Lam(b(Lam(b(NatElim(
            b(Lam(b(Nat))),            // motive P = λ_. Nat
            b(Var(0)),                 // base:  add 0 n = n
            b(Lam(b(Lam(b(Suc(b(Var(0)))))))), // step: λk. λrec. suc rec
            b(Var(1)),                 // scrutinee = m
        )))))
    }
    fn add_ty() -> Term {
        Pi(Omega, b(Nat), b(Pi(Omega, b(Nat), b(Nat))))
    }

    #[test]
    fn natelim_is_a_total_recursor() {
        // add type-checks at Π[ω](m:Nat). Π[ω](n:Nat). Nat
        assert!(
            check_closed(&add_def(), &add_ty()).is_ok(),
            "{:?}",
            check_closed(&add_def(), &add_ty())
        );
        // and it COMPUTES: add 2 3  ↝  5   (ι-reduction of natElim by NbE)
        let add = Ann(b(add_def()), b(add_ty()));
        let app = App(b(App(b(add), b(NatLit(2)))), b(NatLit(3)));
        assert_eq!(normalize_closed(&app), NatLit(5));
    }

    #[test]
    fn proof_by_user_defined_computation() {
        // refl : Eq Nat (add 2 3) 5   — a proof discharged by the eliminator's
        // own computation, not a built-in `+`. This is the Idris-like payoff.
        let add = Ann(b(add_def()), b(add_ty()));
        let app = App(b(App(b(add), b(NatLit(2)))), b(NatLit(3)));
        let prop = Eq(b(Nat), b(app.clone()), b(NatLit(5)));
        assert!(check_closed(&Refl(b(app)), &prop).is_ok());

        // the false equation add 2 3 = 6 is rejected
        let add2 = Ann(b(add_def()), b(add_ty()));
        let app2 = App(b(App(b(add2), b(NatLit(2)))), b(NatLit(3)));
        let bad = Eq(b(Nat), b(app2.clone()), b(NatLit(6)));
        assert!(check_closed(&Refl(b(app2)), &bad).is_err());
    }

    #[test]
    fn natelim_is_type_checked() {
        // an ill-typed eliminator is rejected: motive P = λ_. Eq Nat 0 0, so the
        // base case z must be a PROOF (Eq Nat 0 0), but we hand it a Nat.
        let motive = Lam(b(Eq(b(Nat), b(NatLit(0)), b(NatLit(0)))));
        let bad = NatElim(
            b(motive),
            b(NatLit(3)),                       // wrong: should be `refl`
            b(Lam(b(Lam(b(Var(0)))))),
            b(NatLit(2)),
        );
        assert!(infer_closed(&bad).is_err());
    }

    // `add m n` INLINED as an eliminator (recursing on m), so 0+n≡n and
    // (suc k)+n≡suc(k+n) hold DEFINITIONALLY — which is what lets `append`
    // below type-check against the index `add m n`.
    fn add_tm(m: Term, n: Term) -> Term {
        NatElim(b(Lam(b(Nat))), b(n), b(Lam(b(Lam(b(Suc(b(Var(0)))))))), b(m))
    }

    // a concrete vector  cons … : Vec Nat len
    fn cons(n: u64, h: u64, t: Term) -> Term {
        Cons(b(Nat), b(NatLit(n)), b(NatLit(h)), b(t))
    }

    #[test]
    fn vectors_are_length_indexed() {
        // [10,20,30] : Vec Nat 3
        let v3 = cons(2, 10, cons(1, 20, cons(0, 30, Nil(b(Nat)))));
        assert!(check_closed(&v3, &VecTy(b(Nat), b(NatLit(3)))).is_ok());
        // the SAME term is rejected at Vec Nat 2 — the length is in the type
        assert!(check_closed(&v3, &VecTy(b(Nat), b(NatLit(2)))).is_err());
        // a cons whose tail length disagrees with the index is rejected
        let bad = Cons(b(Nat), b(NatLit(5)), b(NatLit(0)), b(Nil(b(Nat)))); // claims tail len 5, is 0
        assert!(check_closed(&bad, &VecTy(b(Nat), b(NatLit(6)))).is_err());
    }

    #[test]
    fn append_tracks_length_in_the_type_and_computes() {
        // append : Π[0](A:Type). Π[0](m:Nat). Π[0](n:Nat).
        //          Π[ω](xs:Vec A m). Π[ω](ys:Vec A n). Vec A (add m n)
        let result = VecTy(b(Var(4)), b(add_tm(Var(3), Var(2)))); // Vec A (add m n)
        let p5 = Pi(Omega, b(VecTy(b(Var(3)), b(Var(1)))), b(result)); // ys : Vec A n
        let p4 = Pi(Omega, b(VecTy(b(Var(2)), b(Var(1)))), b(p5)); // xs : Vec A m
        let p3 = Pi(Zero, b(Nat), b(p4)); // n
        let p2 = Pi(Zero, b(Nat), b(p3)); // m
        let append_ty = Pi(Zero, b(Type), b(p2)); // A
        // append = λA.λm.λn.λxs.λys. vecElim A P ys pcons m xs
        //   P     = λk.λ_. Vec A (add k n)
        //   pcons = λk.λh.λt.λrec. cons A (add k n) h rec
        let motive = Lam(b(Lam(b(VecTy(b(Var(6)), b(add_tm(Var(1), Var(4))))))));
        let pcons = Lam(b(Lam(b(Lam(b(Lam(b(Cons(
            b(Var(8)),
            b(add_tm(Var(3), Var(6))),
            b(Var(2)),
            b(Var(0)),
        )))))))));
        let append = Lam(b(Lam(b(Lam(b(Lam(b(Lam(b(VecElim(
            b(Var(4)),   // A
            b(motive),
            b(Var(0)),   // pnil = ys
            b(pcons),
            b(Var(3)),   // length = m
            b(Var(1)),   // scrutinee = xs
        )))))))))));

        assert!(
            check_closed(&append, &append_ty).is_ok(),
            "{:?}",
            check_closed(&append, &append_ty)
        );

        // append Nat 2 1 [10,20] [30]  ↝  [10,20,30]  : Vec Nat 3
        let xs = cons(1, 10, cons(0, 20, Nil(b(Nat))));
        let ys = cons(0, 30, Nil(b(Nat)));
        let app = App(
            b(App(
                b(App(
                    b(App(b(App(b(Ann(b(append), b(append_ty))), b(Nat))), b(NatLit(2)))),
                    b(NatLit(1)),
                )),
                b(xs),
            )),
            b(ys),
        );
        // the type carries the summed length: Vec Nat (add 2 1) = Vec Nat 3
        assert_eq!(infer_closed(&app), Ok(VecTy(b(Nat), b(NatLit(3)))));
        // and it actually concatenates, by the eliminator's ι-rules
        let expected = cons(2, 10, cons(1, 20, cons(0, 30, Nil(b(Nat)))));
        assert_eq!(normalize_closed(&app), expected);
    }

    #[test]
    fn proofs_by_computation_still_work() {
        // refl : Eq Nat (2 + 2) 4   -- a proof, in the 0-fragment
        let good = Refl(b(Add(b(NatLit(2)), b(NatLit(2)))));
        let prop = Eq(b(Nat), b(Add(b(NatLit(2)), b(NatLit(2)))), b(NatLit(4)));
        assert!(check_closed(&good, &prop).is_ok());
        // a false equation is rejected
        let falseprop = Eq(b(Nat), b(Add(b(NatLit(2)), b(NatLit(2)))), b(NatLit(5)));
        assert!(check_closed(&good, &falseprop).is_err());
    }
}
