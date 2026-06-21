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
//!
//! GENERAL INDUCTIVE FAMILIES (the gate to Idris-like data). A `Signature` holds
//! strictly-positive datatype declarations: parameters, indices, and
//! constructors (each a telescope ending in `D params idxs`). Each family gets a
//! DEPENDENT ELIMINATOR whose type is *computed from its constructors* (with an
//! induction hypothesis for every recursive argument). The eliminator is the
//! only recursion, so every function written with it is TOTAL BY CONSTRUCTION
//! (docs/09 §1.3). `Nat`, `Vec n`, `Fin n`, `List` are then *declarations*, not
//! built-ins — see the tests. (`Nat` is also kept as a primitive, with literals
//! and `+`, because it is a convenient index type.)
//!
//! UNIVERSES (Phase F): a predicative, cumulative hierarchy `Type i : Type
//! (i+1)` (NOT the old, inconsistent `Type : Type`). `Π`/`Σ` live at the `max`
//! of their parts; cumulativity is one-directional (`Type i` is accepted where
//! `Type j` is wanted iff `i ≤ j`) and conversion stays strict, so the hierarchy
//! never collapses. A datatype's universe must be ≥ every constructor argument's
//! level (`check_signature`), which blocks the Girard/Hurkens paradox — a total
//! program can no longer inhabit `False`. Universe POLYMORPHISM in definitions is
//! not yet implemented; the surface defaults every `Type` to `Type 0` (a sound
//! sublanguage). See FUTURE_WORK §3.1, §4.3, §13.

use crate::mult::Mult;
use std::rc::Rc;

#[derive(Clone, PartialEq, Debug)]
pub enum Term {
    Var(usize), // de Bruijn index (0 = innermost)
    /// `Type i` — the `i`-th universe. `Type i : Type (i+1)`, predicative and
    /// cumulative (a `Type i` is also a `Type j` for `i ≤ j`). The hierarchy is
    /// what makes the total fragment a consistent logic: with the old `Type :
    /// Type`, Girard's paradox inhabits `False` (FUTURE_WORK §3.1, §13).
    Type(usize),
    Pi(Mult, Box<Term>, Box<Term>), // Π[π](_:A). B
    Lam(Box<Term>),
    App(Box<Term>, Box<Term>),
    Sigma(Mult, Box<Term>, Box<Term>), // Σ[π](_:A). B
    Pair(Box<Term>, Box<Term>),
    Fst(Box<Term>),
    Snd(Box<Term>),
    // ---- built-in Nat (a convenient primitive index type) ----
    Nat,
    NatLit(u64),
    Zero,
    Suc(Box<Term>),
    NatElim(Box<Term>, Box<Term>, Box<Term>, Box<Term>),
    /// `NatCase(motive, z, s, scrut)` — a NON-recursive case split on a `Nat`:
    /// `z : motive 0`, `s : (k:Nat) → motive (Suc k)` (NO induction hypothesis),
    /// result `motive scrut`. Used to express the case-split inside a general
    /// recursive function (`Fix`), where recursion is by explicit self-call.
    NatCase(Box<Term>, Box<Term>, Box<Term>, Box<Term>),
    Add(Box<Term>, Box<Term>),
    /// `Fix(ty, body)` — GENERAL recursion: `body : ty` may reference itself via
    /// de Bruijn 0 (bound by the `Fix`). This is NOT total — it is the program-level
    /// recursion for value computations (not for type-level indices). The kernel
    /// treats it OPAQUELY (it never unfolds during type-checking, so normalization
    /// stays terminating); only the backend unfolds it, as a real native function.
    Fix(Box<Term>, Box<Term>),
    // ---- the identity type ----
    Eq(Box<Term>, Box<Term>, Box<Term>),
    Refl(Box<Term>),
    // ---- GENERAL inductive families (looked up in the Signature) ----
    /// `D` applied to its parameters then indices: a fully-applied type.
    Data(String, Vec<Term>),
    /// constructor `c` applied to the family's parameters then the ctor's args.
    Constr(String, Vec<Term>),
    /// the dependent eliminator:  Elim(D, motive, methods, scrutinee) : P idxs scrut
    /// `methods` is one per constructor, in declaration order. The parameters
    /// and indices are recovered from the scrutinee's type.
    Elim(String, Box<Term>, Vec<Term>, Box<Term>),
    /// `Case(D, motive, methods, scrutinee)` — a NON-RECURSIVE general case-split
    /// (the general-datatype analog of `NatCase`): dispatch on the scrutinee's
    /// constructor and bind its fields into the matched method, with NO induction
    /// hypothesis and NO recursion. A method's telescope is the constructor's ARGS
    /// only (`λargs. body`), never the args-then-IHs of `Elim`. Used as the body of
    /// a `Fix` (general recursion on a boxed/heap structure), where recursion is
    /// explicit via the `Fix` self-binder — so dispatch (`Case`) and recursion
    /// (`Fix`) are cleanly separated, avoiding `Elim`'s implicit-IH exponential
    /// blow-up. `Case` REDUCES on a concrete constructor (terminating); a `%partial`
    /// `Fix` wrapping it stays opaque to the checker.
    Case(String, Box<Term>, Vec<Term>, Box<Term>),
    /// an opaque POSTULATE (a typed constant with no reduction rule), looked up
    /// in the Signature. Used to embed the memory primitives (`Own`, `alloc`, …)
    /// in the calculus so they are checked by the QTT core.
    Const(String),
    Ann(Box<Term>, Box<Term>),
    /// `Let(σ, ty, e, body)` — a CALL-BY-VALUE let: bind `e : ty` to a variable
    /// usable at multiplicity `σ` in `body`. UNLIKE the β-redex `(λ[σ]u.body) e`
    /// (which SCALES `e`'s usage by `σ`), the let counts `e` EXACTLY ONCE: its usage
    /// is `U_e ⊕ U_body`, with the bound variable's usage limited to `σ` separately.
    /// This is the correct rule for sequencing an effectful / linear-consuming `e`
    /// whose (copyable) result is then used `σ` times — e.g. `let _ = free(x); free(y)`
    /// frees x and y exactly once (the β-redex wrongly scaled `free(x)` by ω).
    Let(Mult, Box<Term>, Box<Term>, Box<Term>),
}

#[derive(Clone)]
pub enum Value {
    VType(usize),
    VPi(Mult, Box<Value>, Closure),
    VLam(Closure),
    /// A NATIVE (semantic / HOAS) function value: a Rust closure over the PURE
    /// evaluator. Used ONLY to build the induction hypothesis of a HIGHER-ORDER
    /// recursive constructor argument (e.g. `Acc`'s `acc` field, a W-type's `sup`)
    /// in the eliminator's ι-rule (`velim`), where the IH is `λ z…. elim (f z…)`
    /// — a function whose body is meta-computed and so has no `Term` body for an
    /// ordinary `Closure`. It is treated EXACTLY like a `VLam` everywhere: `vapp`
    /// applies it, and `quote` ETA-EXPANDS it (`λ x. quote (f x)`), so a native
    /// closure is definitionally equal to its eta-`VLam` form and two closures
    /// with different bodies quote to different terms — conversion (which is
    /// quote-based) is therefore correct with no special-casing. The closure is
    /// pure (no mutable capture); equality is decided by read-back, never by
    /// pointer. (Phase E3 — general strictly-positive eliminators.)
    VLamNative(std::rc::Rc<dyn Fn(Value) -> Value>),
    VSigma(Mult, Box<Value>, Closure),
    VPair(Box<Value>, Box<Value>),
    VNat,
    VNatLit(u64),
    VSuc(Box<Value>), // suc of a neutral Nat
    VEq(Box<Value>, Box<Value>, Box<Value>),
    VRefl(Box<Value>),
    VData(String, Vec<Value>),
    VConstr(String, Vec<Value>),
    VNeu(Neutral),
}

#[derive(Clone)]
pub enum Neutral {
    NVar(usize), // de Bruijn LEVEL
    NApp(Box<Neutral>, Box<Value>),
    NAdd(Box<Value>, Box<Value>),
    NFst(Box<Neutral>),
    NSnd(Box<Neutral>),
    NNatElim(Box<Value>, Box<Value>, Box<Value>, Box<Neutral>),
    /// a `NatCase` stuck on a neutral scrutinee.
    NNatCase(Box<Value>, Box<Value>, Box<Value>, Box<Neutral>),
    /// an OPAQUE recursive function (`Fix`): never unfolds in the kernel. Holds
    /// the (closed) type and body terms; applications accumulate as `NApp`.
    NFix(Box<Term>, Box<Term>),
    /// elim stuck on a neutral scrutinee: data name, motive, methods, scrutinee.
    NElim(String, Box<Value>, Vec<Value>, Box<Neutral>),
    /// a `Case` (non-recursive general case-split) stuck on a neutral scrutinee.
    NCase(String, Box<Value>, Vec<Value>, Box<Neutral>),
    /// an opaque postulate constant.
    NConst(String),
}

#[derive(Clone)]
pub struct Closure {
    sig: Rc<Signature>,
    env: Vec<Value>,
    body: Term,
}

impl Closure {
    fn apply(&self, arg: Value) -> Value {
        let mut env = self.env.clone();
        env.push(arg);
        eval(&self.sig, &env, &self.body)
    }
}

// ---------------------------------------------------------------------------
// signatures: strictly-positive inductive families
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct Constructor {
    pub name: String,
    /// argument telescope, in the context [params].
    pub args: Vec<(Mult, Term)>,
    /// the index arguments of the result `D params idxs`, in context [params++args].
    pub idxs: Vec<Term>,
}

#[derive(Clone)]
pub struct DataDecl {
    pub name: String,
    /// parameter telescope, in the empty context.
    pub params: Vec<(Mult, Term)>,
    /// index telescope, in the context [params].
    pub indices: Vec<(Mult, Term)>,
    pub ctors: Vec<Constructor>,
    /// The universe the fully-applied family `D params idxs` inhabits: `D … :
    /// Type universe`. `check_signature` REQUIRES `universe ≥ i` for every
    /// constructor argument of type `Type i` — a constructor that stores a
    /// `Type j` forces the family up to `Type (j+1)`, so a datatype cannot sit
    /// in a universe small enough to quantify over itself. This is the
    /// predicativity side-condition that blocks Girard's/Hurkens' paradox; with
    /// the old `Type : Type` it was vacuous (FUTURE_WORK §3.1, §4.3, §13).
    pub universe: usize,
}

#[derive(Clone, Default)]
pub struct Signature {
    pub datas: Vec<DataDecl>,
    /// opaque postulates: a name and its (closed) type.
    pub postulates: Vec<(String, Term)>,
}

impl Signature {
    pub(crate) fn data(&self, name: &str) -> Option<&DataDecl> {
        self.datas.iter().find(|d| d.name == name)
    }
    pub(crate) fn postulate(&self, name: &str) -> Option<&Term> {
        self.postulates.iter().find(|(n, _)| n == name).map(|(_, t)| t)
    }
    pub(crate) fn ctor(&self, name: &str) -> Option<(&DataDecl, &Constructor)> {
        for d in &self.datas {
            for c in &d.ctors {
                if c.name == name {
                    return Some((d, c));
                }
            }
        }
        None
    }
}

// ---------------------------------------------------------------------------
// evaluation (NbE)
// ---------------------------------------------------------------------------

fn eval(sig: &Rc<Signature>, env: &[Value], t: &Term) -> Value {
    match t {
        Term::Var(i) => env[env.len() - 1 - i].clone(),
        Term::Type(i) => Value::VType(*i),
        // CALL-BY-VALUE let: evaluate `e` ONCE, bind it, evaluate the body.
        Term::Let(_sigma, _ty, e, body) => {
            let ve = eval(sig, env, e);
            let mut env2 = env.to_vec();
            env2.push(ve);
            eval(sig, &env2, body)
        }
        Term::Pi(pi, a, b) => Value::VPi(
            *pi,
            Box::new(eval(sig, env, a)),
            Closure { sig: sig.clone(), env: env.to_vec(), body: (**b).clone() },
        ),
        Term::Lam(b) => Value::VLam(Closure {
            sig: sig.clone(),
            env: env.to_vec(),
            body: (**b).clone(),
        }),
        Term::App(f, a) => vapp(eval(sig, env, f), eval(sig, env, a)),
        Term::Sigma(pi, a, b) => Value::VSigma(
            *pi,
            Box::new(eval(sig, env, a)),
            Closure { sig: sig.clone(), env: env.to_vec(), body: (**b).clone() },
        ),
        Term::Pair(a, b) => {
            Value::VPair(Box::new(eval(sig, env, a)), Box::new(eval(sig, env, b)))
        }
        Term::Fst(p) => vfst(eval(sig, env, p)),
        Term::Snd(p) => vsnd(eval(sig, env, p)),
        Term::Nat => Value::VNat,
        Term::NatLit(n) => Value::VNatLit(*n),
        Term::Zero => Value::VNatLit(0),
        Term::Suc(t) => vsuc(eval(sig, env, t)),
        Term::NatElim(p, z, s, scrut) => vnatelim(
            eval(sig, env, p),
            eval(sig, env, z),
            eval(sig, env, s),
            eval(sig, env, scrut),
        ),
        Term::NatCase(p, z, s, scrut) => vnatcase(
            eval(sig, env, p),
            eval(sig, env, z),
            eval(sig, env, s),
            eval(sig, env, scrut),
        ),
        // Fix is OPAQUE: it evaluates to a neutral and never unfolds in the kernel.
        Term::Fix(ty, body) => Value::VNeu(Neutral::NFix(ty.clone(), body.clone())),
        Term::Add(a, b) => match (eval(sig, env, a), eval(sig, env, b)) {
            (Value::VNatLit(x), Value::VNatLit(y)) => Value::VNatLit(x + y),
            (va, vb) => Value::VNeu(Neutral::NAdd(Box::new(va), Box::new(vb))),
        },
        Term::Eq(a, x, y) => Value::VEq(
            Box::new(eval(sig, env, a)),
            Box::new(eval(sig, env, x)),
            Box::new(eval(sig, env, y)),
        ),
        Term::Refl(a) => Value::VRefl(Box::new(eval(sig, env, a))),
        Term::Data(name, args) => {
            Value::VData(name.clone(), args.iter().map(|a| eval(sig, env, a)).collect())
        }
        Term::Constr(name, args) => {
            Value::VConstr(name.clone(), args.iter().map(|a| eval(sig, env, a)).collect())
        }
        Term::Elim(data, motive, methods, scrut) => {
            let vm = eval(sig, env, motive);
            let vmeth: Vec<Value> = methods.iter().map(|m| eval(sig, env, m)).collect();
            velim(sig, data, &vm, &vmeth, eval(sig, env, scrut))
        }
        Term::Case(data, motive, methods, scrut) => {
            let vm = eval(sig, env, motive);
            let vmeth: Vec<Value> = methods.iter().map(|m| eval(sig, env, m)).collect();
            vcase(sig, data, &vm, &vmeth, eval(sig, env, scrut))
        }
        Term::Const(c) => Value::VNeu(Neutral::NConst(c.clone())),
        Term::Ann(e, _) => eval(sig, env, e),
    }
}

fn vapp(f: Value, a: Value) -> Value {
    match f {
        Value::VLam(clo) => clo.apply(a),
        Value::VLamNative(f) => f(a),
        Value::VNeu(n) => Value::VNeu(Neutral::NApp(Box::new(n), Box::new(a))),
        _ => unreachable!("vapp on a non-function (ill-typed term reached eval)"),
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

fn vsuc(n: Value) -> Value {
    match n {
        Value::VNatLit(k) => Value::VNatLit(k + 1),
        other => Value::VSuc(Box::new(other)),
    }
}

fn vnatelim(p: Value, z: Value, s: Value, scrut: Value) -> Value {
    match scrut {
        // A CLOSED literal `n` folds ITERATIVELY: acc = z; for k in 0..n { acc =
        // s k acc }. This is the `%builtin Nat` representation optimization at the
        // evaluator (à la Idris 2): a `Nat` is a machine integer, so normalizing an
        // eliminator over a big `n` is a counting loop — never a recursion `n` deep
        // that would blow the host stack during type-checking.
        Value::VNatLit(n) => {
            let mut acc = z;
            for k in 0..n {
                acc = vapp(vapp(s.clone(), Value::VNatLit(k)), acc);
            }
            acc
        }
        // `suc pred` for a NEUTRAL pred: one unfolding, then stuck on `pred`.
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

/// `natCase z s n` — a single case split (no recursion): `n = 0 ⇒ z`,
/// `n = suc k ⇒ s k`. Stuck on a neutral.
fn vnatcase(p: Value, z: Value, s: Value, scrut: Value) -> Value {
    match scrut {
        Value::VNatLit(0) => z,
        Value::VNatLit(n) => vapp(s, Value::VNatLit(n - 1)),
        Value::VSuc(pred) => vapp(s, *pred),
        Value::VNeu(nu) => Value::VNeu(Neutral::NNatCase(
            Box::new(p),
            Box::new(z),
            Box::new(s),
            Box::new(nu),
        )),
        _ => unreachable!("natCase on a non-Nat (ill-typed term reached eval)"),
    }
}

/// The RECURSIVE SPINE of a constructor-argument type, for the general dependent
/// eliminator (Phase E3). A strictly-positive recursive argument has the shape
/// `(z₁:C₁) → … → (zₘ:Cₘ) → D self-params idxs` (the `Cᵢ` do not mention `D` —
/// strict positivity guarantees this). Returns `Some((doms, idxs))` — the domain
/// telescope and the final index spine — when the type ends in the family `data`,
/// else `None`. `m = 0` is the ordinary direct-recursion case `D … idxs`.
/// Is `t` a recursive occurrence of `data` — either DIRECT (`data idxs`) or
/// HIGHER-ORDER (`(z₁…zₘ) → data idxs`, a strictly-positive function field)? If so,
/// returns the telescope arity `m` (0 for a direct field). Used by the surface
/// match-compiler to detect recursive constructor fields (mirroring `method_ty_tm`
/// /`velim`, which read recursion off the declared type via `rec_spine`).
pub(crate) fn rec_field_arity(data: &str, t: &Term) -> Option<usize> {
    rec_spine(data, t).map(|(doms, _)| doms.len())
}

fn rec_spine<'a>(data: &str, mut t: &'a Term) -> Option<(Vec<&'a Term>, &'a [Term])> {
    let mut doms: Vec<&Term> = Vec::new();
    loop {
        match t {
            Term::Pi(_, dom, cod) => {
                doms.push(dom);
                t = cod;
            }
            Term::Data(name, idxs) if name == data => return Some((doms, &idxs[..])),
            _ => return None,
        }
    }
}

/// Build the induction hypothesis VALUE for a (possibly higher-order) recursive
/// argument whose type has `m` telescope domains: `λ z₁…zₘ. elim (g z₁…zₘ)`. For
/// `m = 0` this is just `elim g`. For `m > 0` the IH is a NATIVE closure (no
/// `Term` body exists for "apply then recurse"); applying it feeds the arguments
/// to `g` and eliminates the result — the genuine sub-derivation. Pure and total.
fn build_ih(
    sig: Rc<Signature>,
    data: String,
    motive: Value,
    methods: Vec<Value>,
    g: Value,
    m: usize,
) -> Value {
    if m == 0 {
        velim(&sig, &data, &motive, &methods, g)
    } else {
        Value::VLamNative(Rc::new(move |z: Value| {
            build_ih(
                sig.clone(),
                data.clone(),
                motive.clone(),
                methods.clone(),
                vapp(g.clone(), z),
                m - 1,
            )
        }))
    }
}

/// The generic ι-rule for a dependent eliminator. On a constructor, it applies
/// the corresponding method to the constructor's arguments, inserting — after
/// each RECURSIVE argument — the induction hypothesis (the result of eliminating
/// that argument, under its telescope for a higher-order recursive position).
/// Recursion + the recursive indices are read off the constructor's argument
/// types via `rec_spine`.
fn velim(
    sig: &Rc<Signature>,
    data: &str,
    motive: &Value,
    methods: &[Value],
    scrut: Value,
) -> Value {
    match scrut {
        Value::VConstr(cname, vargs) => {
            let decl = sig.data(data).expect("eliminating an undeclared family");
            let (cidx, ctor) = decl
                .ctors
                .iter()
                .enumerate()
                .find(|(_, c)| c.name == cname)
                .expect("constructor not in this family");
            let p = decl.params.len();
            let mut result = methods[cidx].clone();
            // Apply ALL the constructor's arguments first…
            for (i, _) in ctor.args.iter().enumerate() {
                result = vapp(result, vargs[p + i].clone());
            }
            // …then one induction hypothesis per recursive argument, in order
            // (args-then-IHs — the standard method telescope; see `method_ty_tm`).
            // A recursive argument may be HIGHER-ORDER (`(z…)→ D … idxs`); its IH
            // is `λ z…. elim (arg z…)`, built by `build_ih`. The recursive shape is
            // read off the DECLARED argument type (`rec_spine`), independent of the
            // value, so it cannot be spoofed by the runtime value.
            for (i, (_, aty)) in ctor.args.iter().enumerate() {
                if let Some((doms, _)) = rec_spine(data, aty) {
                    let ih = build_ih(
                        sig.clone(),
                        data.to_string(),
                        motive.clone(),
                        methods.to_vec(),
                        vargs[p + i].clone(),
                        doms.len(),
                    );
                    result = vapp(result, ih);
                }
            }
            result
        }
        Value::VNeu(nu) => Value::VNeu(Neutral::NElim(
            data.to_string(),
            Box::new(motive.clone()),
            methods.to_vec(),
            Box::new(nu),
        )),
        _ => unreachable!("elim on a non-data value (ill-typed term reached eval)"),
    }
}

/// Reduce a NON-RECURSIVE case-split (`Case`): dispatch on the scrutinee's
/// constructor and apply the matched method to the constructor's ARGS only — NO
/// induction hypotheses, NO recursion (the `Elim`/`velim` IH loop is absent). This is
/// what makes `Case` the clean dispatch primitive inside a `Fix`: recursion is
/// explicit via the `Fix` self-binder, never the implicit structural IH. Reduces on a
/// concrete constructor (terminating); stays stuck (`NCase`) on a neutral scrutinee.
fn vcase(
    sig: &Rc<Signature>,
    data: &str,
    motive: &Value,
    methods: &[Value],
    scrut: Value,
) -> Value {
    match scrut {
        Value::VConstr(cname, vargs) => {
            let decl = sig.data(data).expect("case on an undeclared family");
            let (cidx, ctor) = decl
                .ctors
                .iter()
                .enumerate()
                .find(|(_, c)| c.name == cname)
                .expect("constructor not in this family");
            let p = decl.params.len();
            let mut result = methods[cidx].clone();
            // apply the constructor's arguments — and NOTHING else (no IHs).
            for (i, _) in ctor.args.iter().enumerate() {
                result = vapp(result, vargs[p + i].clone());
            }
            result
        }
        Value::VNeu(nu) => Value::VNeu(Neutral::NCase(
            data.to_string(),
            Box::new(motive.clone()),
            methods.to_vec(),
            Box::new(nu),
        )),
        _ => unreachable!("case on a non-data value (ill-typed term reached eval)"),
    }
}

fn quote(lvl: usize, v: &Value) -> Term {
    match v {
        Value::VType(i) => Term::Type(*i),
        Value::VPi(pi, a, clo) => Term::Pi(
            *pi,
            Box::new(quote(lvl, a)),
            Box::new(quote(lvl + 1, &clo.apply(Value::VNeu(Neutral::NVar(lvl))))),
        ),
        Value::VLam(clo) => {
            Term::Lam(Box::new(quote(lvl + 1, &clo.apply(Value::VNeu(Neutral::NVar(lvl))))))
        }
        // ETA read-back of a native closure: `λ x. quote (f x)` — identical in form
        // to a `VLam`, so a native IH and its eta-`VLam` form quote equal, and two
        // closures with different behaviour quote to different terms. This is what
        // makes conversion (quote-based) correct on native closures.
        Value::VLamNative(f) => {
            Term::Lam(Box::new(quote(lvl + 1, &f(Value::VNeu(Neutral::NVar(lvl))))))
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
        Value::VEq(a, x, y) => Term::Eq(
            Box::new(quote(lvl, a)),
            Box::new(quote(lvl, x)),
            Box::new(quote(lvl, y)),
        ),
        Value::VRefl(a) => Term::Refl(Box::new(quote(lvl, a))),
        Value::VData(name, args) => {
            Term::Data(name.clone(), args.iter().map(|a| quote(lvl, a)).collect())
        }
        Value::VConstr(name, args) => {
            Term::Constr(name.clone(), args.iter().map(|a| quote(lvl, a)).collect())
        }
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
        Neutral::NNatCase(p, z, s, scrut) => Term::NatCase(
            Box::new(quote(lvl, p)),
            Box::new(quote(lvl, z)),
            Box::new(quote(lvl, s)),
            Box::new(quote_neu(lvl, scrut)),
        ),
        // a `Fix` is closed and opaque — re-emit it verbatim.
        Neutral::NFix(ty, body) => Term::Fix(ty.clone(), body.clone()),
        Neutral::NElim(data, m, methods, scrut) => Term::Elim(
            data.clone(),
            Box::new(quote(lvl, m)),
            methods.iter().map(|x| quote(lvl, x)).collect(),
            Box::new(quote_neu(lvl, scrut)),
        ),
        Neutral::NCase(data, m, methods, scrut) => Term::Case(
            data.clone(),
            Box::new(quote(lvl, m)),
            methods.iter().map(|x| quote(lvl, x)).collect(),
            Box::new(quote_neu(lvl, scrut)),
        ),
        Neutral::NConst(c) => Term::Const(c.clone()),
    }
}

fn conv(lvl: usize, a: &Value, b: &Value) -> bool {
    quote(lvl, a) == quote(lvl, b)
}

// ---------------------------------------------------------------------------
// de Bruijn shift + simultaneous substitution (used to build eliminator types)
// ---------------------------------------------------------------------------

/// Add `d` to every free `Var(i)` with `i >= cutoff`.
fn shift(d: usize, cutoff: usize, t: &Term) -> Term {
    map_vars(t, cutoff, &|i, c| {
        Term::Var(if i >= c { i + d } else { i })
    })
}

/// Simultaneous substitution: in `t` (valid in a context of size `sub.len()`),
/// replace the variable with de Bruijn index `i` by `sub[i]` (lifted under any
/// binders crossed). Variables are assumed in range.
fn subst(t: &Term, sub: &[Term]) -> Term {
    map_vars(t, 0, &|i, depth| {
        if i < depth {
            Term::Var(i) // bound locally inside `t`
        } else {
            shift(depth, 0, &sub[i - depth])
        }
    })
}

/// Generic variable traversal: `f(index, binder_depth)` is called for each
/// `Var`, with `binder_depth` = number of binders crossed so far.
fn map_vars(t: &Term, depth: usize, f: &dyn Fn(usize, usize) -> Term) -> Term {
    let go = |t: &Term| map_vars(t, depth, f);
    let go1 = |t: &Term| map_vars(t, depth + 1, f);
    match t {
        Term::Var(i) => f(*i, depth),
        Term::Type(_) | Term::Nat | Term::NatLit(_) | Term::Zero | Term::Const(_) => t.clone(),
        Term::Pi(m, a, b) => Term::Pi(*m, Box::new(go(a)), Box::new(go1(b))),
        Term::Sigma(m, a, b) => Term::Sigma(*m, Box::new(go(a)), Box::new(go1(b))),
        Term::Lam(b) => Term::Lam(Box::new(go1(b))),
        // `Let(σ, ty, e, body)` binds the let variable (de Bruijn 0) in `body` only.
        Term::Let(m, ty, e, body) => {
            Term::Let(*m, Box::new(go(ty)), Box::new(go(e)), Box::new(go1(body)))
        }
        Term::App(f0, a) => Term::App(Box::new(go(f0)), Box::new(go(a))),
        Term::Pair(a, b) => Term::Pair(Box::new(go(a)), Box::new(go(b))),
        Term::Fst(p) => Term::Fst(Box::new(go(p))),
        Term::Snd(p) => Term::Snd(Box::new(go(p))),
        Term::Suc(k) => Term::Suc(Box::new(go(k))),
        Term::NatCase(p, z, s, sc) => Term::NatCase(
            Box::new(go(p)),
            Box::new(go(z)),
            Box::new(go(s)),
            Box::new(go(sc)),
        ),
        // `Fix(ty, body)` binds `self` (de Bruijn 0) in `body`; `ty` does not see it.
        Term::Fix(ty, body) => Term::Fix(Box::new(go(ty)), Box::new(go1(body))),
        Term::NatElim(p, z, s, sc) => Term::NatElim(
            Box::new(go(p)),
            Box::new(go(z)),
            Box::new(go(s)),
            Box::new(go(sc)),
        ),
        Term::Add(a, b) => Term::Add(Box::new(go(a)), Box::new(go(b))),
        Term::Eq(a, x, y) => Term::Eq(Box::new(go(a)), Box::new(go(x)), Box::new(go(y))),
        Term::Refl(a) => Term::Refl(Box::new(go(a))),
        Term::Data(n, args) => Term::Data(n.clone(), args.iter().map(|a| go(a)).collect()),
        Term::Constr(n, args) => Term::Constr(n.clone(), args.iter().map(|a| go(a)).collect()),
        Term::Elim(data, m, methods, sc) => Term::Elim(
            data.clone(),
            Box::new(go(m)),
            methods.iter().map(|x| go(x)).collect(),
            Box::new(go(sc)),
        ),
        // `Case` mirrors `Elim`: motive/methods/scrutinee at the same depth (each
        // method is a `λargs.…`, whose binders are handled by the `Lam` case).
        Term::Case(data, m, methods, sc) => Term::Case(
            data.clone(),
            Box::new(go(m)),
            methods.iter().map(|x| go(x)).collect(),
            Box::new(go(sc)),
        ),
        Term::Ann(e, ty) => Term::Ann(Box::new(go(e)), Box::new(go(ty))),
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
/// per-variable LEAST UPPER BOUND — the BRANCH combinator for a `match`/eliminator:
/// only ONE arm runs, so a variable's usage across the arms is their `lub`, NOT their
/// sum. `lub(1,1)=1` (used once in each mutually-exclusive arm = once), while
/// `lub(0,1)=ω` fails a linear budget (used inconsistently ⇒ some arm leaks/dups it).
fn ujoin(a: &Usage, b: &Usage) -> Usage {
    a.iter().zip(b).map(|(x, y)| x.lub(*y)).collect()
}

// ---------------------------------------------------------------------------
// building eliminator types (motive + per-constructor methods) from a decl
// ---------------------------------------------------------------------------

/// Map a declaration term `t`, living in context `[params, prior(j)]` (params
/// instantiated to the concrete `sparam_tms`, the `j` prior binders recorded by
/// their bind-depth in `prior_bd`), into the current context at depth `cur`.
fn map_decl(t: &Term, p: usize, sparam_tms: &[Term], prior_bd: &[usize], cur: usize) -> Term {
    let j = prior_bd.len();
    let m = p + j;
    let mut sub: Vec<Term> = Vec::with_capacity(m);
    // decl indices 0..j are the prior binders, innermost first.
    for d in 0..j {
        let arg = j - 1 - d;
        sub.push(Term::Var(cur - 1 - prior_bd[arg]));
    }
    // decl indices j..m are the parameters, p_{p-1} first.
    for d in j..m {
        let k = m - 1 - d;
        sub.push(shift(cur, 0, &sparam_tms[k]));
    }
    subst(t, &sub)
}

/// Validate an eliminator MOTIVE and return the universe `ℓ` it targets. The
/// motive must denote `Π[ω](dom₀)…(domₖ). Type ℓ` for the given domain telescope
/// `doms` (innermost last). We extend the context by each domain, apply the
/// (evaluated) motive to a fresh variable per domain, and `check_type` the
/// resulting body — so a bad motive is rejected by a real type error, and `ℓ` is
/// whatever universe the body lives in (large elimination ⇒ any `ℓ`). This
/// replaces the old fixed `… → Type` motive type, which baked in `Type : Type`.
fn motive_level(ctx: &Ctx, motive: &Term, doms: &[Value]) -> Result<usize, String> {
    let base = ctx.level();
    let mut mctx = ctx.extend_clone();
    let mut applied = eval(&ctx.sig, &ctx.env(), motive);
    for (k, dom) in doms.iter().enumerate() {
        mctx = mctx.extend(dom.clone());
        match applied {
            Value::VLam(_) | Value::VNeu(_) => {
                applied = vapp(applied, Value::VNeu(Neutral::NVar(base + k)));
            }
            _ => {
                return Err(
                    "eliminator motive is not a function of its indices and scrutinee".into(),
                )
            }
        }
    }
    check_type(&mctx, &quote(mctx.level(), &applied))
}

/// The induction-hypothesis TYPE for recursive argument `j`, built in the method
/// context at depth `cur` (where the constructor's arguments are bound at the
/// depths recorded in `bd`). For a HIGHER-ORDER recursive position
/// `a_j : (z₁:C₁)…(zₘ:Cₘ) → D self idxs` the IH is
///     `(z₁:C₁) → … → (zₘ:Cₘ) → motive idxs (a_j z₁ … zₘ)`
/// — the same telescope, with the family's result replaced by `motive idxs`
/// applied to the field applied to the bound `zᵢ`. For a direct recursive
/// argument (`m = 0`) this collapses to `motive idxs a_j`, the original rule.
fn ih_type(
    decl: &DataDecl,
    ctor: &Constructor,
    sparam_tms: &[Term],
    motive_tm: &Term,
    j: usize,
    bd: &[usize],
    cur: usize,
) -> Term {
    let p = decl.params.len();
    let (doms, dargs) =
        rec_spine(&decl.name, &ctor.args[j].1).expect("ih_type on a non-recursive argument");
    let m = doms.len();
    // the `zᵢ` binders sit at depths cur, cur+1, …, cur+m-1; the body at cur+m.
    let z_depths: Vec<usize> = (0..m).map(|i| cur + i).collect();
    let cur_m = cur + m;

    // body: `motive idxs (a_j z₁ … zₘ)`, all in the context [params, a₀..a_{k-1},
    // z₁..zₘ]. The index terms `idxs` live in the decl context [params, a₀..a_{j-1},
    // z₁..zₘ]; map them with those j prior args + the m `z` binders.
    let prior_full: Vec<usize> = bd[..j].iter().copied().chain(z_depths.iter().copied()).collect();
    let mut ih = shift(cur_m, 0, motive_tm);
    for it in &dargs[p..] {
        let mi = map_decl(it, p, sparam_tms, &prior_full, cur_m);
        ih = Term::App(Box::new(ih), Box::new(mi));
    }
    // `a_j z₁ … zₘ`
    let mut applied = Term::Var(cur_m - 1 - bd[j]);
    for zd in &z_depths {
        applied = Term::App(Box::new(applied), Box::new(Term::Var(cur_m - 1 - zd)));
    }
    ih = Term::App(Box::new(ih), Box::new(applied));

    // wrap the domain telescope, innermost (zₘ) first. `Cᵢ` lives in the decl
    // context [params, a₀..a_{j-1}, z₁..z_{i-1}] (i prior `z`s = z_depths[..i]).
    for i in (0..m).rev() {
        let prior_i: Vec<usize> =
            bd[..j].iter().copied().chain(z_depths[..i].iter().copied()).collect();
        let dom = map_decl(doms[i], p, sparam_tms, &prior_i, cur + i);
        ih = Term::Pi(Mult::Omega, Box::new(dom), Box::new(ih));
    }
    ih
}

/// The method type for one constructor:
///   Π(args). Π(IH for each recursive arg). motive result-idxs (c params args)
/// The method type for one constructor. With `with_ih = true` it is the ELIM method
/// telescope `Π(args). Π(IHs). motive idxs (c args)` (an induction hypothesis per
/// recursive arg). With `with_ih = false` it is the CASE method telescope
/// `Π(args). motive idxs (c args)` — the constructor's args only, NO IHs (for the
/// non-recursive `Case`).
fn method_ty_tm(
    decl: &DataDecl,
    ctor: &Constructor,
    sparam_tms: &[Term],
    motive_tm: &Term,
    with_ih: bool,
) -> Term {
    let p = decl.params.len();
    let data = &decl.name;
    let rec_args: Vec<usize> = ctor
        .args
        .iter()
        .enumerate()
        .filter(|(_, (_, aty))| rec_spine(data, aty).is_some())
        .map(|(i, _)| i)
        .collect();

    // result: motive (result idxs) (c params args)   [all `args` bound, depth=cur]
    fn result(
        decl: &DataDecl,
        ctor: &Constructor,
        sparam_tms: &[Term],
        motive_tm: &Term,
        bd: &[usize],
        cur: usize,
    ) -> Term {
        let p = decl.params.len();
        let k = ctor.args.len();
        let mut res = shift(cur, 0, motive_tm);
        for it in &ctor.idxs {
            let mi = map_decl(it, p, sparam_tms, &bd[..k], cur);
            res = Term::App(Box::new(res), Box::new(mi));
        }
        let mut cargs: Vec<Term> = (0..p).map(|kk| shift(cur, 0, &sparam_tms[kk])).collect();
        for jj in 0..k {
            cargs.push(Term::Var(cur - 1 - bd[jj]));
        }
        res = Term::App(
            Box::new(res),
            Box::new(Term::Constr(ctor.name.clone(), cargs)),
        );
        res
    }

    // bind the induction hypotheses (one per recursive arg), then the result.
    fn ihs(
        decl: &DataDecl,
        ctor: &Constructor,
        sparam_tms: &[Term],
        motive_tm: &Term,
        rec_args: &[usize],
        w: usize,
        bd: &[usize],
        cur: usize,
    ) -> Term {
        if w < rec_args.len() {
            let j = rec_args[w];
            // the IH type for recursive arg `j` — `(z…)→ motive idxs (a_j z…)` for a
            // higher-order recursive position, `motive idxs a_j` for a direct one.
            let ih = ih_type(decl, ctor, sparam_tms, motive_tm, j, bd, cur);
            let body = ihs(decl, ctor, sparam_tms, motive_tm, rec_args, w + 1, bd, cur + 1);
            Term::Pi(Mult::Omega, Box::new(ih), Box::new(body))
        } else {
            result(decl, ctor, sparam_tms, motive_tm, bd, cur)
        }
    }

    // bind the constructor's arguments.
    fn args(
        decl: &DataDecl,
        ctor: &Constructor,
        sparam_tms: &[Term],
        motive_tm: &Term,
        rec_args: &[usize],
        i: usize,
        bd: &mut Vec<usize>,
        cur: usize,
        with_ih: bool,
    ) -> Term {
        let p = decl.params.len();
        if i < ctor.args.len() {
            let (mult, aty) = &ctor.args[i];
            let dom = map_decl(aty, p, sparam_tms, bd, cur);
            bd.push(cur);
            let body = args(decl, ctor, sparam_tms, motive_tm, rec_args, i + 1, bd, cur + 1, with_ih);
            bd.pop();
            Term::Pi(*mult, Box::new(dom), Box::new(body))
        } else if with_ih {
            ihs(decl, ctor, sparam_tms, motive_tm, rec_args, 0, bd, cur)
        } else {
            // CASE method: no induction hypotheses — straight to the result.
            result(decl, ctor, sparam_tms, motive_tm, bd, cur)
        }
    }

    let _ = (p, data);
    args(decl, ctor, sparam_tms, motive_tm, &rec_args, 0, &mut Vec::new(), 0, with_ih)
}

// ---------------------------------------------------------------------------
// resourced bidirectional checking
// ---------------------------------------------------------------------------

pub struct Ctx {
    sig: Rc<Signature>,
    types: Vec<Value>,
}

impl Ctx {
    pub fn new() -> Ctx {
        Ctx { sig: Rc::new(Signature::default()), types: Vec::new() }
    }
    fn with_sig(sig: Rc<Signature>) -> Ctx {
        Ctx { sig, types: Vec::new() }
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
        Ctx { sig: self.sig.clone(), types }
    }
}

/// Check that `t` denotes a TYPE, and return the universe level `i` it lives in
/// (`t : Type i`). A type is always inferrable (Π/Σ/Data/Eq/Var/Const/Type/Nat),
/// so this infers `t`'s type and reads off the universe; anything whose type is
/// not a `Type i` is rejected. The returned level feeds the `max` rules for
/// Π/Σ and the datatype-universe constraint that blocks Girard's paradox.
fn check_type(ctx: &Ctx, t: &Term) -> Result<usize, String> {
    let n = ctx.level();
    let (ty, _u) = infer(ctx, t)?;
    match ty {
        Value::VType(i) => Ok(i),
        other => Err(format!(
            "expected a type, but this term has type {:?}",
            quote(n, &other)
        )),
    }
}

/// Check a spine of arguments against a telescope (each entry in the context of
/// the preceding ones). Returns the argument VALUES and the combined usage
/// (each scaled by its declared multiplicity).
fn check_spine(
    ctx: &Ctx,
    args: &[Term],
    tele: &[(Mult, Term)],
) -> Result<(Vec<Value>, Usage), String> {
    if args.len() != tele.len() {
        return Err(format!(
            "wrong number of arguments: expected {}, got {}",
            tele.len(),
            args.len()
        ));
    }
    let mut vals: Vec<Value> = Vec::new();
    let mut usage = uzero(ctx.level());
    for (i, (mult, ty_tm)) in tele.iter().enumerate() {
        let tyv = eval(&ctx.sig, &vals, ty_tm); // telescope is in the context of prior args
        let u = check(ctx, &args[i], &tyv)?;
        usage = uadd(&usage, &uscale(*mult, &u));
        vals.push(eval(&ctx.sig, &ctx.env(), &args[i]));
    }
    Ok((vals, usage))
}

fn check(ctx: &Ctx, t: &Term, ty: &Value) -> Result<Usage, String> {
    let n = ctx.level();
    match (t, ty) {
        (Term::Lam(body), Value::VPi(pi, dom, cod)) => {
            let ctx2 = ctx.extend((**dom).clone());
            let codty = cod.apply(Value::VNeu(Neutral::NVar(n)));
            let mut ub = check(&ctx2, body, &codty)?;
            let sigma = ub.pop().unwrap();
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
            let va = eval(&ctx.sig, &ctx.env(), a);
            let ub = check(ctx, b, &cod.apply(va))?;
            Ok(uadd(&uscale(*pi, &ua), &ub))
        }
        (Term::Refl(a), Value::VEq(aty, x, y)) => {
            check(ctx, a, aty)?;
            let va = eval(&ctx.sig, &ctx.env(), a);
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
            // CUMULATIVITY (one-directional): if `t` is itself a type living in
            // `Type i`, it is accepted where a `Type j` is expected as long as
            // `i ≤ j`. This is the ONLY subtyping — conversion (`conv`) stays
            // strict, so `Type i ≢ Type j` for `i ≠ j` and the hierarchy never
            // collapses. (FUTURE_WORK §3.1.)
            if let (Value::VType(i), Value::VType(j)) = (&ty2, ty) {
                if i <= j {
                    return Ok(u);
                }
                return Err(format!(
                    "universe mismatch: this is a `Type {i}` but a `Type {j}` was \
                     expected ({i} ⋠ {j}; cumulativity only lifts upward)"
                ));
            }
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
        // `Type i : Type (i+1)` — the universe hierarchy (NOT `Type : Type`).
        // Guard the successor: an unchecked `i + 1` would wrap `Type usize::MAX`
        // to `VType(0)` (release) — accepting `Type MAX : Type 0`, i.e. reinstating
        // `Type : Type` at the apex — or panic (debug). Levels never legitimately
        // approach this; overflow is a hard error.
        Term::Type(i) => match i.checked_add(1) {
            Some(j) => Ok((Value::VType(j), uzero(n))),
            None => Err("universe level overflow (`Type` level too large)".to_string()),
        },
        Term::Nat => Ok((Value::VType(0), uzero(n))), // Nat : Type 0
        Term::NatLit(_) | Term::Zero => Ok((Value::VNat, uzero(n))),
        Term::Suc(k) => {
            let u = check(ctx, k, &Value::VNat)?;
            Ok((Value::VNat, u))
        }
        Term::NatElim(p, z, s, scrut) => {
            // motive `p : Nat → Type ℓ` for any ℓ (large elimination permitted).
            motive_level(ctx, p, &[Value::VNat])?;
            let vp = eval(&ctx.sig, &ctx.env(), p);

            let pzero = vapp(vp.clone(), Value::VNatLit(0));
            let uz = check(ctx, z, &pzero)?;

            let p_tm = quote(n, &vp);
            let sty_term = Term::Pi(
                Mult::Omega,
                Box::new(Term::Nat),
                Box::new(Term::Pi(
                    Mult::Omega,
                    Box::new(Term::App(Box::new(shift(1, 0, &p_tm)), Box::new(Term::Var(0)))),
                    Box::new(Term::App(
                        Box::new(shift(2, 0, &p_tm)),
                        Box::new(Term::Suc(Box::new(Term::Var(1)))),
                    )),
                )),
            );
            let vsty = eval(&ctx.sig, &ctx.env(), &sty_term);
            let us = check(ctx, s, &vsty)?;

            let uscr = check(ctx, scrut, &Value::VNat)?;
            let result = vapp(vp, eval(&ctx.sig, &ctx.env(), scrut));
            let u = uadd(
                &uscale(Mult::Omega, &uz),
                &uadd(&uscale(Mult::Omega, &us), &uscale(Mult::Omega, &uscr)),
            );
            Ok((result, u))
        }
        // `NatCase` is `NatElim` without the induction hypothesis: the successor
        // method has type `(k:Nat) → motive (Suc k)`, no `motive k →` premise.
        Term::NatCase(p, z, s, scrut) => {
            // motive `p : Nat → Type ℓ` for any ℓ (large elimination permitted).
            motive_level(ctx, p, &[Value::VNat])?;
            let vp = eval(&ctx.sig, &ctx.env(), p);

            let pzero = vapp(vp.clone(), Value::VNatLit(0));
            let uz = check(ctx, z, &pzero)?;

            let p_tm = quote(n, &vp);
            let sty_term = Term::Pi(
                Mult::Omega,
                Box::new(Term::Nat),
                Box::new(Term::App(
                    Box::new(shift(1, 0, &p_tm)),
                    Box::new(Term::Suc(Box::new(Term::Var(0)))),
                )),
            );
            let vsty = eval(&ctx.sig, &ctx.env(), &sty_term);
            let us = check(ctx, s, &vsty)?;

            let uscr = check(ctx, scrut, &Value::VNat)?;
            let result = vapp(vp, eval(&ctx.sig, &ctx.env(), scrut));
            let u = uadd(
                &uscale(Mult::Omega, &uz),
                &uadd(&uscale(Mult::Omega, &us), &uscale(Mult::Omega, &uscr)),
            );
            Ok((result, u))
        }
        // `Fix(ty, body)` — general recursion. Check `body : ty` with `self : ty`
        // in scope (the recursion is unrestricted: ω). NOT total; the kernel never
        // unfolds it, so this does not affect normalization/termination of checking.
        Term::Fix(ty_tm, body) => {
            check_type(ctx, ty_tm)?;
            let vty = eval(&ctx.sig, &ctx.env(), ty_tm);
            let ctx2 = ctx.extend(vty.clone());
            let mut ub = check(&ctx2, body, &vty)?;
            ub.pop(); // discard the self-binder's usage (recursion is unrestricted)
            Ok((vty, ub))
        }
        Term::Add(a, b) => {
            let ua = check(ctx, a, &Value::VNat)?;
            let ub = check(ctx, b, &Value::VNat)?;
            Ok((Value::VNat, uadd(&ua, &ub)))
        }
        // PREDICATIVE Π/Σ: `(x:A)→B` lives in `Type (max i j)` where `A : Type i`
        // and `B : Type j`. Predicativity (no impredicative `Prop`) is what keeps
        // the hierarchy consistent (FUTURE_WORK §3.1).
        Term::Pi(_, a, b) | Term::Sigma(_, a, b) => {
            let la = check_type(ctx, a)?;
            let va = eval(&ctx.sig, &ctx.env(), a);
            let lb = check_type(&ctx.extend(va), b)?;
            Ok((Value::VType(la.max(lb)), uzero(n)))
        }
        Term::Fst(p) => match infer(ctx, p)? {
            (Value::VSigma(_, dom, _), up) => Ok((*dom, up)),
            (other, _) => Err(format!("fst of a non-pair (type {:?})", quote(n, &other))),
        },
        Term::Snd(p) => match infer(ctx, p)? {
            (Value::VSigma(_, _, cod), up) => {
                let v1 = vfst(eval(&ctx.sig, &ctx.env(), p));
                Ok((cod.apply(v1), up))
            }
            (other, _) => Err(format!("snd of a non-pair (type {:?})", quote(n, &other))),
        },
        Term::Pair(_, _) => Err("cannot infer a bare pair; annotate it `(e : T)`".to_string()),
        // `Id A x y : Type i` when `A : Type i` (equality lives at its type's level).
        Term::Eq(a, x, y) => {
            let la = check_type(ctx, a)?;
            let va = eval(&ctx.sig, &ctx.env(), a);
            check(ctx, x, &va)?;
            check(ctx, y, &va)?;
            Ok((Value::VType(la), uzero(n)))
        }
        Term::Refl(a) => {
            let (ta, _) = infer(ctx, a)?;
            let va = eval(&ctx.sig, &ctx.env(), a);
            Ok((
                Value::VEq(Box::new(ta), Box::new(va.clone()), Box::new(va)),
                uzero(n),
            ))
        }
        Term::App(f, x) => match infer(ctx, f)? {
            (Value::VPi(pi, dom, cod), uf) => {
                let ux = check(ctx, x, &dom)?;
                let vx = eval(&ctx.sig, &ctx.env(), x);
                Ok((cod.apply(vx), uadd(&uf, &uscale(pi, &ux))))
            }
            (other, _) => Err(format!(
                "application of a non-function (type {:?})",
                quote(n, &other)
            )),
        },
        Term::Data(name, args) => {
            let decl = ctx
                .sig
                .data(name)
                .ok_or_else(|| format!("unknown datatype `{name}`"))?
                .clone();
            let tele: Vec<(Mult, Term)> =
                decl.params.iter().chain(decl.indices.iter()).cloned().collect();
            check_spine(ctx, args, &tele)?; // a type former: usage is 0-fragment
            // the fully-applied family lives in its declared universe.
            Ok((Value::VType(decl.universe), uzero(n)))
        }
        Term::Constr(name, args) => {
            let (decl, ctor) = ctx
                .sig
                .ctor(name)
                .map(|(d, c)| (d.clone(), c.clone()))
                .ok_or_else(|| format!("unknown constructor `{name}`"))?;
            let tele: Vec<(Mult, Term)> =
                decl.params.iter().cloned().chain(ctor.args.iter().cloned()).collect();
            let (vals, usage) = check_spine(ctx, args, &tele)?;
            // result type: D params (idxs evaluated at the full arg environment)
            let p = decl.params.len();
            let mut dargs: Vec<Value> = vals[0..p].to_vec();
            for it in &ctor.idxs {
                dargs.push(eval(&ctx.sig, &vals, it));
            }
            Ok((Value::VData(decl.name.clone(), dargs), usage))
        }
        Term::Elim(data, motive, methods, scrut) => {
            let decl = ctx
                .sig
                .data(data)
                .ok_or_else(|| format!("unknown datatype `{data}`"))?
                .clone();
            let p = decl.params.len();
            let ni = decl.indices.len();

            let (scrut_ty, u_scrut) = infer(ctx, scrut)?;
            let sargs = match scrut_ty {
                Value::VData(ref sn, ref a) if *sn == *data => a.clone(),
                other => {
                    return Err(format!(
                        "elim[{data}] on a scrutinee of type {:?} (not `{data}`)",
                        quote(n, &other)
                    ))
                }
            };
            let sparams: Vec<Value> = sargs[0..p].to_vec();
            let sindices: Vec<Value> = sargs[p..p + ni].to_vec();
            let sparam_tms: Vec<Term> = sparams.iter().map(|v| quote(n, v)).collect();

            // motive : Π[ω](indices). D params indices → Type ℓ. We validate it
            // against this telescope and read off ℓ (ANY ℓ — large elimination is
            // allowed; the result universe is whatever the motive targets). The
            // motive's body is genuinely type-checked, so this is not a no-op.
            let mut idx_doms: Vec<Value> = Vec::new();
            let mut idx_neus: Vec<Value> = Vec::new();
            for i in 0..ni {
                let mut env: Vec<Value> = sparams.clone();
                env.extend(idx_neus.iter().cloned());
                idx_doms.push(eval(&ctx.sig, &env, &decl.indices[i].1));
                idx_neus.push(Value::VNeu(Neutral::NVar(n + i)));
            }
            let mut d_args: Vec<Value> = sparams.clone();
            d_args.extend(idx_neus.iter().cloned());
            let mut doms = idx_doms;
            doms.push(Value::VData(data.clone(), d_args));
            motive_level(ctx, motive, &doms)?; // 0-fragment: usage discarded
            let vmotive = eval(&ctx.sig, &ctx.env(), motive);
            let motive_tm = quote(n, &vmotive);

            // methods
            if methods.len() != decl.ctors.len() {
                return Err(format!(
                    "elim[{data}]: expected {} method(s), got {}",
                    decl.ctors.len(),
                    methods.len()
                ));
            }
            // JOIN (not SUM) the per-method usages: a `match`/eliminator runs exactly
            // ONE arm, so a captured variable used the same amount in every arm is used
            // that amount (e.g. freed once-per-arm = freed once), while an inconsistent
            // use (`0⊔1=ω`) fails a linear budget. (Summing branches over-rejected the
            // freed-once-per-arm pattern — the dual of the CBV-let over-counting fix.)
            // `lub`'s identity is not `0`, so fold from the FIRST method (empty family ⇒
            // no usage).
            let mut umeth: Option<Usage> = None;
            for (ci, ctor) in decl.ctors.iter().enumerate() {
                let meth_ty_tm = method_ty_tm(&decl, ctor, &sparam_tms, &motive_tm, true);
                let meth_ty = eval(&ctx.sig, &ctx.env(), &meth_ty_tm);
                let um = check(ctx, &methods[ci], &meth_ty)?;
                umeth = Some(match umeth {
                    None => um,
                    Some(acc) => ujoin(&acc, &um),
                });
            }
            let umeth = umeth.unwrap_or_else(|| uzero(n));

            // result: motive indices scrutinee
            let vscrut = eval(&ctx.sig, &ctx.env(), scrut);
            let mut res = vmotive;
            for si in sindices {
                res = vapp(res, si);
            }
            res = vapp(res, vscrut);
            // usage: the scrutinee is consumed once; a method fires ω times for a
            // RECURSIVE family (once per node) but exactly once for a non-recursive
            // one — so a linear value can be destructured (a linear pair) while a
            // recursive eliminator stays conservative.
            let recursive = decl.ctors.iter().any(|c| {
                c.args.iter().any(|(_, a)| rec_spine(data, a).is_some())
            });
            let mscale = if recursive { Mult::Omega } else { Mult::One };
            let u = uadd(&uscale(mscale, &umeth), &u_scrut);
            Ok((res, u))
        }
        Term::Case(data, motive, methods, scrut) => {
            // a NON-RECURSIVE case-split: same type discipline as `Elim` EXCEPT (1) the
            // method telescope has NO induction hypotheses (`method_ty_tm(.., false)`),
            // and (2) exactly ONE method fires ONCE — so the usage scale is `1`, never
            // `ω`. Recursion, if any, is the enclosing `Fix`'s self-call, not `Case`.
            let decl = ctx
                .sig
                .data(data)
                .ok_or_else(|| format!("unknown datatype `{data}`"))?
                .clone();
            let p = decl.params.len();
            let ni = decl.indices.len();

            let (scrut_ty, u_scrut) = infer(ctx, scrut)?;
            let sargs = match scrut_ty {
                Value::VData(ref sn, ref a) if *sn == *data => a.clone(),
                other => {
                    return Err(format!(
                        "case[{data}] on a scrutinee of type {:?} (not `{data}`)",
                        quote(n, &other)
                    ))
                }
            };
            let sparams: Vec<Value> = sargs[0..p].to_vec();
            let sindices: Vec<Value> = sargs[p..p + ni].to_vec();
            let sparam_tms: Vec<Term> = sparams.iter().map(|v| quote(n, v)).collect();

            // motive : Π[ω](indices). D params indices → Type ℓ (validated; large
            // elimination allowed — identical to `Elim`).
            let mut idx_doms: Vec<Value> = Vec::new();
            let mut idx_neus: Vec<Value> = Vec::new();
            for i in 0..ni {
                let mut env: Vec<Value> = sparams.clone();
                env.extend(idx_neus.iter().cloned());
                idx_doms.push(eval(&ctx.sig, &env, &decl.indices[i].1));
                idx_neus.push(Value::VNeu(Neutral::NVar(n + i)));
            }
            let mut d_args: Vec<Value> = sparams.clone();
            d_args.extend(idx_neus.iter().cloned());
            let mut doms = idx_doms;
            doms.push(Value::VData(data.clone(), d_args));
            motive_level(ctx, motive, &doms)?;
            let vmotive = eval(&ctx.sig, &ctx.env(), motive);
            let motive_tm = quote(n, &vmotive);

            if methods.len() != decl.ctors.len() {
                return Err(format!(
                    "case[{data}]: expected {} method(s), got {}",
                    decl.ctors.len(),
                    methods.len()
                ));
            }
            // JOIN (not SUM) the per-method usages — one arm runs (same branch rule as
            // `Elim`/`NatCase`): freed-once-per-arm = once; inconsistent use ⇒ ω⋢1.
            let mut umeth: Option<Usage> = None;
            for (ci, ctor) in decl.ctors.iter().enumerate() {
                let meth_ty_tm = method_ty_tm(&decl, ctor, &sparam_tms, &motive_tm, false);
                let meth_ty = eval(&ctx.sig, &ctx.env(), &meth_ty_tm);
                let um = check(ctx, &methods[ci], &meth_ty)?;
                umeth = Some(match umeth {
                    None => um,
                    Some(acc) => ujoin(&acc, &um),
                });
            }
            let umeth = umeth.unwrap_or_else(|| uzero(n));

            // result: motive indices scrutinee
            let vscrut = eval(&ctx.sig, &ctx.env(), scrut);
            let mut res = vmotive;
            for si in sindices {
                res = vapp(res, si);
            }
            res = vapp(res, vscrut);
            // a `Case` fires exactly ONE method ONCE — scale `1` (NEVER `ω`); the
            // scrutinee is consumed once.
            let u = uadd(&umeth, &u_scrut);
            Ok((res, u))
        }
        Term::Const(c) => {
            let ty = ctx
                .sig
                .postulate(c)
                .ok_or_else(|| format!("unknown postulate `{c}`"))?
                .clone();
            Ok((eval(&ctx.sig, &[], &ty), uzero(n))) // a constant is used 0 times itself
        }
        Term::Ann(e, ty) => {
            check_type(ctx, ty)?;
            let vty = eval(&ctx.sig, &ctx.env(), ty);
            let u = check(ctx, e, &vty)?;
            Ok((vty, u))
        }
        Term::Let(sigma, ty, e, body) => {
            // CALL-BY-VALUE let. `ty` is a type; `e : ty` is counted EXACTLY ONCE; the
            // bound variable is usable at multiplicity `σ` in `body`. The usage is
            // `U_e ⊕ U_body` — `e` is NOT scaled by `σ` (the correction over the β-redex
            // `(λ[σ]u.body) e`, which would `uscale(σ, U_e)` and wrongly multiply the
            // linear resources `e` consumes). The bound variable's own usage is checked
            // against `σ` separately, exactly as a Π/λ binder is.
            check_type(ctx, ty)?;
            let vty = eval(&ctx.sig, &ctx.env(), ty);
            let u_e = check(ctx, e, &vty)?;
            let ctx2 = ctx.extend(vty.clone());
            let (rty, mut u_body) = infer(&ctx2, body)?;
            let sigma_u = u_body.pop().expect("let body sees the bound variable");
            if !sigma_u.leq(*sigma) {
                return Err(format!(
                    "the let-bound variable at multiplicity {sigma} is used {sigma_u} \
                     time(s) ({sigma_u} ⋢ {sigma})"
                ));
            }
            // result type = the body's type with the bound variable SUBSTITUTED by `e`
            // (CBV let `let u = e; body : body_ty[u := e]`). `rty` lives in `ctx2` (with
            // the bound var at level `n`); quoting it at `n+1` and re-evaluating in
            // `ctx.env()` extended by `eval(e)` substitutes `u := e` and re-grounds the
            // type in the OUTER context. This SUPPORTS a dependent body type — and, for
            // the common non-dependent case, simply returns the body's type unchanged
            // (no `eval(e)` trace). (Previously this returned `rty` directly, which
            // de-Bruijn-underflowed in `quote` on a dependent body — a cryptic panic;
            // this hardens it per the hard-error-clearly discipline by handling the
            // case correctly instead.)
            let ve = eval(&ctx.sig, &ctx.env(), e);
            let rty_tm = quote(n + 1, &rty);
            let mut env2 = ctx.env();
            env2.push(ve);
            let result_ty = eval(&ctx.sig, &env2, &rty_tm);
            Ok((result_ty, uadd(&u_e, &u_body)))
        }
        Term::Lam(_) => Err("cannot infer a bare lambda; annotate it `(e : T)`".to_string()),
    }
}

// ---------------------------------------------------------------------------
// signature well-formedness: telescopes type-check + strict positivity
// ---------------------------------------------------------------------------

/// Does the datatype `data` occur anywhere in `t`?
fn occurs(data: &str, t: &Term) -> bool {
    let mut found = false;
    fn go(data: &str, t: &Term, found: &mut bool) {
        if let Term::Data(n, _) = t {
            if n == data {
                *found = true;
            }
        }
        // structural recursion over children
        match t {
            Term::Pi(_, a, b) | Term::Sigma(_, a, b) | Term::App(a, b) | Term::Add(a, b) => {
                go(data, a, found);
                go(data, b, found);
            }
            Term::Lam(b) | Term::Fst(b) | Term::Snd(b) | Term::Suc(b) | Term::Refl(b) => {
                go(data, b, found)
            }
            Term::Pair(a, b) => {
                go(data, a, found);
                go(data, b, found);
            }
            Term::Eq(a, x, y) => {
                go(data, a, found);
                go(data, x, found);
                go(data, y, found);
            }
            Term::NatElim(a, b, c, d) | Term::NatCase(a, b, c, d) => {
                // BOTH the recursive eliminator AND the non-recursive case split
                // must be traversed: `NatCase` does large elimination and REDUCES
                // (`vnatcase`), so a family hidden in one of its branches can
                // surface as a genuine (possibly NEGATIVE) occurrence after
                // reduction. Missing this is a positivity bypass ⇒ unsoundness.
                for s in [a, b, c, d] {
                    go(data, s, found);
                }
            }
            // a `Let` can mention the family in its type, bound expr, or body — a
            // positivity search must see through all three (like `Fix`).
            Term::Let(_, ty, e, body) => {
                go(data, ty, found);
                go(data, e, found);
                go(data, body, found);
            }
            // `Fix` is opaque to the checker, but it can still mention the family
            // in its type or body; a positivity search must see through it.
            Term::Fix(ty, body) => {
                go(data, ty, found);
                go(data, body, found);
            }
            Term::Data(_, args) | Term::Constr(_, args) => {
                for a in args {
                    go(data, a, found);
                }
            }
            Term::Elim(_, m, methods, sc) => {
                go(data, m, found);
                for s in methods {
                    go(data, s, found);
                }
                go(data, sc, found);
            }
            // `Case` (non-recursive case-split) carries subterms (motive/methods/
            // scrutinee) and REDUCES on a concrete constructor (`vcase`), so — exactly
            // like `NatCase`/`Elim` — a family hidden in a method can surface as a
            // (possibly NEGATIVE) occurrence after reduction. It MUST be traversed; a
            // missed subterm-bearing variant here is a positivity bypass ⇒ Curry's
            // paradox (the load-bearing E3 invariant).
            Term::Case(_, m, methods, sc) => {
                go(data, m, found);
                for s in methods {
                    go(data, s, found);
                }
                go(data, sc, found);
            }
            Term::Ann(e, ty) => {
                go(data, e, found);
                go(data, ty, found);
            }
            // EXHAUSTIVE on purpose — NO catch-all. This is a soundness-critical
            // traversal (it is the only barrier against non-positive datatypes); a
            // `_ => {}` once silently swallowed `NatCase`/`Fix` and let a hidden
            // negative occurrence reduce past positivity (Curry's paradox). Listing
            // every LEAF explicitly means a future `Term` variant that carries
            // subterms is a COMPILE ERROR here until it is handled.
            Term::Var(_)
            | Term::Type(_)
            | Term::Nat
            | Term::NatLit(_)
            | Term::Zero
            | Term::Const(_) => {}
        }
    }
    go(data, t, &mut found);
    found
}

/// A constructor argument is STRICTLY POSITIVE in the family `data` iff `data`
/// occurs only in strictly-positive position: peeling the argument's function
/// telescope `(z₁:C₁)→…→(zₘ:Cₘ)→ Head`, the family must NOT occur in ANY domain
/// `Cᵢ` (an occurrence left of an arrow is NEGATIVE — the source of
/// non-termination / `False`), and the `Head` is either the family applied
/// `Data(self, idxs)` with the family not recurring in the `idxs`, or a term not
/// mentioning the family at all. This is the standard rule (Agda/Coq/Idris); it
/// admits higher-order recursive fields like `Acc`'s `acc`/a W-type's `sup`
/// (family only in the codomain head) while still rejecting `(Bad→A)→Bad` and the
/// double-negative `((Bad→A)→A)→Bad`. (Phase E3.)
/// If `t` is the owned-pointer wrapper `Own T` (the postulate `Own` applied to one
/// argument), return the pointee `T`. A pointer is a POSITIVE, polarity-PRESERVING
/// occurrence of its pointee: `data` may recurse through `Own data` (like a ctor
/// field), but `Own (data → B)` still has `data` in a NEGATIVE position (the arrow
/// domain) — so positivity recurses into `T` via `strictly_positive`, which re-applies
/// the Pi-domain discipline. (`↦` will be added the same way in Phase C.)
fn own_spine(t: &Term) -> Option<&Term> {
    if let Term::App(f, arg) = t {
        let mut head: &Term = f;
        loop {
            match head {
                Term::Ann(e, _) => head = e,
                Term::Const(n) if n == "Own" => return Some(arg),
                _ => return None,
            }
        }
    }
    None
}

/// The probe datatype name standing in for a parameter when computing its variance.
/// It contains SPACES, so it is UN-LEXABLE as a single identifier — no user/library
/// datatype can ever be named this, so the variance check is soundness-by-construction
/// (not relying on a user-unreachable-by-convention `__`-sentinel that a user could in
/// fact declare). The name appears only inside `param_covariant`'s probe substitution +
/// the `strictly_positive` it drives; it is never looked up in the signature.
const SP_PROBE: &str = "<positivity probe>";

/// Substitute the datatype-parameter at de Bruijn `base` (at depth 0) with a fresh
/// probe datatype `SP_PROBE`, tracking binder depth. Used to compute a parameter's
/// variance by reusing `strictly_positive`: a datatype is strictly-positive in a
/// parameter iff the probe (standing in for that parameter) occurs only positively.
fn subst_param_with_probe(t: &Term, base: usize) -> Term {
    map_vars(t, 0, &|i, depth| {
        if i == base + depth {
            Term::Data(SP_PROBE.to_string(), vec![])
        } else {
            Term::Var(i)
        }
    })
}

/// Is datatype `dname` STRICTLY-POSITIVE in its `i`-th parameter — i.e. may a recursive
/// occurrence nest through `D …Aᵢ…` at that argument? Computed by substituting a probe
/// for the parameter in every constructor field and checking the probe occurs only
/// positively (`strictly_positive`). `seen` breaks cycles CONSERVATIVELY: a parameter
/// reached through a (mutual-)recursion cycle returns `false` (not provably covariant ⇒
/// reject nesting) — SOUND (never wrongly accepts a negative occurrence hidden through a
/// cycle), at the cost of over-rejecting nesting through a recursive covariant container
/// (e.g. `Vec`/`List` of the family) until a full variance fixpoint lands. (The memory
/// model's `Opt` is non-recursive, so `Opt (Own Node)` is accepted.)
fn param_covariant(
    sig: &Rc<Signature>,
    dname: &str,
    i: usize,
    seen: &mut std::collections::HashSet<(String, usize)>,
) -> bool {
    let key = (dname.to_string(), i);
    if seen.contains(&key) {
        return false; // cycle ⇒ conservatively NOT provably covariant
    }
    let decl = match sig.data(dname) {
        Some(d) => d,
        None => return false, // unknown / postulate datatype ⇒ conservative reject
    };
    let np = decl.params.len();
    if i >= np {
        return false; // an INDEX argument (not a covariant parameter) ⇒ reject nesting
    }
    seen.insert(key.clone());
    let mut covariant = true;
    'ctors: for ctor in &decl.ctors {
        for (k, (_, fty)) in ctor.args.iter().enumerate() {
            // field k is in scope [params, args[0..k]]: param i sits at de Bruijn
            // `k + (np-1-i)` from this field's base.
            let base = k + (np - 1 - i);
            let probed = subst_param_with_probe(fty, base);
            if !strictly_positive_seen(SP_PROBE, &probed, sig, seen) {
                covariant = false;
                break 'ctors;
            }
        }
    }
    seen.remove(&key);
    covariant
}

fn strictly_positive(data: &str, t: &Term, sig: &Rc<Signature>) -> bool {
    strictly_positive_seen(data, t, sig, &mut std::collections::HashSet::new())
}

/// Strict positivity with NESTED-positivity (variance-aware) and the `Own` pointer
/// wrapper. `data` may occur: in a Pi CODOMAIN (positive) but never a Pi DOMAIN
/// (negative); as the head `Data(data, idxs)` (direct recursion, not in its own
/// indices); inside `D …Aᵢ…` at argument `i` IFF `D` is strictly-positive in parameter
/// `i` (and then recursively positive within `Aᵢ`); and inside `Own T` (a pointer —
/// recurse into `T`, polarity-preserving). `seen` threads the variance cycle-guard.
fn strictly_positive_seen(
    data: &str,
    t: &Term,
    sig: &Rc<Signature>,
    seen: &mut std::collections::HashSet<(String, usize)>,
) -> bool {
    if let Term::Pi(_, dom, cod) = t {
        // a Pi DOMAIN is a negative position — `data` must not occur there at all.
        return !occurs(data, dom) && strictly_positive_seen(data, cod, sig, seen);
    }
    if !occurs(data, t) {
        return true; // `data` absent ⇒ trivially positive
    }
    // a pointer wrapper is transparent + polarity-preserving: recurse into the pointee.
    if let Some(pointee) = own_spine(t) {
        return strictly_positive_seen(data, pointee, sig, seen);
    }
    match t {
        Term::Ann(e, _) => strictly_positive_seen(data, e, sig, seen),
        // direct recursive occurrence: the family applied — but not recurring in its
        // own indices (a non-strictly-positive index is the canonical unsoundness).
        Term::Data(name, args) if name == data => args.iter().all(|a| !occurs(data, a)),
        // NESTED occurrence: `data` may appear in `D`'s argument `i` only if `D` is
        // strictly-positive in parameter `i`, and then must itself be positive there.
        Term::Data(name, args) => args.iter().enumerate().all(|(i, a)| {
            !occurs(data, a) || (param_covariant(sig, name, i, seen) && strictly_positive_seen(data, a, sig, seen))
        }),
        // `data` occurs in some other (non-positive) position ⇒ reject.
        _ => false,
    }
}

/// A parameter or index binder `x : T` that RANGES OVER a universe (`T = Type i`)
/// forces the datatype into `Type i` or higher: a `Type 0`-sized family must not
/// quantify (even phantomly) over `Type 0` or above. Reject when `decl.universe`
/// is too small. This complements the constructor-argument check — together they
/// stop a universe from being smuggled through ANY telescope (the params/indices
/// case the constructor-arg check alone misses). Non-universe binders (`n : Nat`,
/// a value parameter) contribute nothing here; if a constructor STORES such a
/// value it is caught by the argument-level check instead. (FUTURE_WORK §4.3.)
fn enforce_binder_universe(
    decl_name: &str,
    decl_universe: usize,
    kind: &str,
    v: &Value,
) -> Result<(), String> {
    if let Value::VType(i) = v {
        if *i > decl_universe {
            return Err(format!(
                "{decl_name}: a {kind} ranging over `Type {i}` forces `{decl_name}` into \
                 `Type {i}` or higher, but it is declared in `Type {decl_universe}`. A \
                 datatype cannot quantify over a universe at or above its own \
                 (predicativity; this is part of what blocks Girard's paradox)."
            ));
        }
    }
    Ok(())
}

/// Check a signature: every telescope type-checks, constructors return the right
/// family with the right index arity, and recursion is strictly positive.
pub fn check_signature(sig: &Signature) -> Result<(), String> {
    let rc = Rc::new(sig.clone());
    for decl in &sig.datas {
        // params telescope well-formed
        let mut ctx = Ctx::with_sig(rc.clone());
        for (_, ty) in &decl.params {
            check_type(&ctx, ty).map_err(|e| format!("in {} params: {e}", decl.name))?;
            let v = eval(&rc, &ctx.env(), ty);
            enforce_binder_universe(&decl.name, decl.universe, "parameter", &v)?;
            ctx = ctx.extend(v);
        }
        let params_ctx = ctx; // context [params]
        // index telescope well-formed (in context [params])
        let mut ictx = params_ctx.extend_clone();
        for (_, ty) in &decl.indices {
            check_type(&ictx, ty).map_err(|e| format!("in {} indices: {e}", decl.name))?;
            let v = eval(&rc, &ictx.env(), ty);
            enforce_binder_universe(&decl.name, decl.universe, "index", &v)?;
            ictx = ictx.extend(v);
        }
        // each constructor
        for ctor in &decl.ctors {
            // a struct's single constructor is named after the type, so collapse
            // `Foo.Foo` to `Foo` in diagnostics; enums keep the `Type.ctor` form.
            let owner = if ctor.name == decl.name {
                decl.name.clone()
            } else {
                format!("{}.{}", decl.name, ctor.name)
            };
            let mut cctx = params_ctx.extend_clone();
            for (_, aty) in &ctor.args {
                if !strictly_positive(&decl.name, aty, &rc) {
                    return Err(format!(
                        "{owner}: argument is not strictly positive in `{}`",
                        decl.name
                    ));
                }
                let la = check_type(&cctx, aty)
                    .map_err(|e| format!("in {owner} args: {e}"))?;
                // THE PARADOX BLOCKER (predicativity side-condition): a field of
                // type `Type la` forces the family into a universe ≥ la. If the
                // declared `universe` is too small, the datatype would be able to
                // quantify over (a universe containing) itself — exactly the
                // Girard/Hurkens loop — so reject it. With the old `Type : Type`
                // every `la` was 0 and this was vacuous (FUTURE_WORK §4.3, §13).
                //
                // NOTE the bound is `la > universe`, NOT `la ≥ universe`: a field
                // (or parameter/index) AT the family's own level is fine — a
                // `Type 0` family may store `Type 0` *values* and quantify over a
                // `Type 0` *parameter* (e.g. `Vec (A:Type 0) : Type 0`). Only
                // storing a *universe itself* (`la = level+1 > level`) is the
                // Girard retract, and that is what this rejects.
                if la > decl.universe {
                    return Err(format!(
                        "{owner}: this argument is a `Type {la}`, so `{}` must live in \
                         `Type {la}` or higher, but it is declared in `Type {}`. A \
                         datatype cannot store a type from its own universe \
                         (predicativity; this is what blocks Girard's paradox).",
                        decl.name, decl.universe
                    ));
                }
                let v = eval(&rc, &cctx.env(), aty);
                cctx = cctx.extend(v);
            }
            if ctor.idxs.len() != decl.indices.len() {
                return Err(format!(
                    "{owner}: returns {} index(es), the family has {}",
                    ctor.idxs.len(),
                    decl.indices.len()
                ));
            }
            // index terms type-check against the (instantiated) index telescope
            let mut env_for_idx: Vec<Value> = cctx.env(); // [params, args] as neutrals
            // index i's expected type is decl.indices[i] in env [params, idx0..i-1]
            let mut idx_vals: Vec<Value> = Vec::new();
            for (i, it) in ctor.idxs.iter().enumerate() {
                // expected type: eval indices[i].1 in [params(neutral) ++ idx_vals]
                let mut idx_env: Vec<Value> =
                    (0..decl.params.len()).map(|k| Value::VNeu(Neutral::NVar(k))).collect();
                idx_env.extend(idx_vals.iter().cloned());
                let exp = eval(&rc, &idx_env, &decl.indices[i].1);
                check(&cctx, it, &exp)
                    .map_err(|e| format!("in {owner} index {i}: {e}"))?;
                idx_vals.push(eval(&rc, &cctx.env(), it));
            }
            let _ = &mut env_for_idx;
        }
    }
    // postulate types must be well-formed types
    let pctx = Ctx::with_sig(rc.clone());
    for (name, ty) in &sig.postulates {
        check_type(&pctx, ty).map_err(|e| format!("postulate {name}: {e}"))?;
    }
    Ok(())
}

impl Ctx {
    fn extend_clone(&self) -> Ctx {
        Ctx { sig: self.sig.clone(), types: self.types.clone() }
    }
}

// ---------------------------------------------------------------------------
// public API
// ---------------------------------------------------------------------------

pub fn check_closed_in(sig: Signature, t: &Term, ty: &Term) -> Result<(), String> {
    let ctx = Ctx::with_sig(Rc::new(sig));
    check_type(&ctx, ty)?;
    let vty = eval(&ctx.sig, &[], ty);
    check(&ctx, t, &vty).map(|_| ())
}

pub fn infer_closed_in(sig: Signature, t: &Term) -> Result<Term, String> {
    let ctx = Ctx::with_sig(Rc::new(sig));
    Ok(quote(0, &infer(&ctx, t)?.0))
}

pub fn normalize_closed_in(sig: Signature, t: &Term) -> Term {
    let rc = Rc::new(sig);
    quote(0, &eval(&rc, &[], t))
}

/// Check a closed term against a closed type (no datatypes).
pub fn check_closed(t: &Term, ty: &Term) -> Result<(), String> {
    check_closed_in(Signature::default(), t, ty)
}

/// Infer the (β-normal) type of a closed term (no datatypes).
pub fn infer_closed(t: &Term) -> Result<Term, String> {
    infer_closed_in(Signature::default(), t)
}

/// Fully normalize a closed term (no datatypes). This is the partial evaluator.
pub fn normalize_closed(t: &Term) -> Term {
    normalize_closed_in(Signature::default(), t)
}

// ---------------------------------------------------------------------------
// elaboration support (used by the surface elaborator for implicit solving)
// ---------------------------------------------------------------------------

/// Evaluate `t` under `env` against `sig` (NbE).
pub(crate) fn eval_rc(sig: &Rc<Signature>, env: &[Value], t: &Term) -> Value {
    eval(sig, env, t)
}

/// Quote a value back to a β-normal term at de Bruijn level `lvl`.
pub(crate) fn quote_at(lvl: usize, v: &Value) -> Term {
    quote(lvl, v)
}

/// The semantic free variable at level `lvl`.
pub(crate) fn nvar(lvl: usize) -> Value {
    Value::VNeu(Neutral::NVar(lvl))
}

/// Lift a term by `d` (shift all free variables up). Used by the surface
/// elaborator to build a constant motive for a non-dependent `match`.
pub(crate) fn shift_term(d: usize, t: &Term) -> Term {
    shift(d, 0, t)
}

/// The eliminator method telescope for `ctor`: the types of its binders (the
/// constructor's arguments, then one induction hypothesis per recursive
/// argument) and the method's return type, all in the eliminator's context
/// extended left-to-right by the binders. Used by the surface elaborator to set
/// up a `match` arm's typing context.
pub(crate) fn elim_method_telescope(
    sig: &Signature,
    data: &str,
    sparam_tms: &[Term],
    motive_tm: &Term,
    ctor_name: &str,
) -> Result<(Vec<(Mult, Term)>, Term), String> {
    let decl = sig.data(data).ok_or_else(|| format!("unknown datatype `{data}`"))?;
    let ctor = decl
        .ctors
        .iter()
        .find(|c| c.name == ctor_name)
        .ok_or_else(|| format!("unknown constructor `{ctor_name}`"))?;
    let nrec = ctor
        .args
        .iter()
        .filter(|(_, a)| rec_spine(data, a).is_some())
        .count();
    let mut t = method_ty_tm(decl, ctor, sparam_tms, motive_tm, true);
    let mut binders = Vec::new();
    for _ in 0..(ctor.args.len() + nrec) {
        match t {
            Term::Pi(m, a, b) => {
                binders.push((m, *a));
                t = *b;
            }
            _ => return Err("internal: malformed method type".into()),
        }
    }
    Ok((binders, t))
}

#[cfg(test)]
mod tests;
