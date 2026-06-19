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
//! `Type : Type` for now (fine for a language; for a logic, a universe hierarchy
//! is needed — ../../docs/07-implementation-guide.md §7).

use crate::mult::Mult;
use std::rc::Rc;

#[derive(Clone, PartialEq, Debug)]
pub enum Term {
    Var(usize), // de Bruijn index (0 = innermost)
    Type,
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
    Add(Box<Term>, Box<Term>),
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
    /// an opaque POSTULATE (a typed constant with no reduction rule), looked up
    /// in the Signature. Used to embed the memory primitives (`Own`, `alloc`, …)
    /// in the calculus so they are checked by the QTT core.
    Const(String),
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
    /// elim stuck on a neutral scrutinee: data name, motive, methods, scrutinee.
    NElim(String, Box<Value>, Vec<Value>, Box<Neutral>),
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
        Term::Type => Value::VType,
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
        Term::Const(c) => Value::VNeu(Neutral::NConst(c.clone())),
        Term::Ann(e, _) => eval(sig, env, e),
    }
}

fn vapp(f: Value, a: Value) -> Value {
    match f {
        Value::VLam(clo) => clo.apply(a),
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

/// The generic ι-rule for a dependent eliminator. On a constructor, it applies
/// the corresponding method to the constructor's arguments, inserting — after
/// each RECURSIVE argument — the result of eliminating that argument (the
/// induction hypothesis). Recursion + the recursive indices are read off the
/// constructor's declared argument types.
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
            for (i, (_, aty)) in ctor.args.iter().enumerate() {
                let arg_val = vargs[p + i].clone();
                result = vapp(result, arg_val.clone());
                // is this argument a recursive occurrence of `data`?
                let env: Vec<Value> = vargs[0..p + i].to_vec();
                if let Value::VData(dn, _) = eval(sig, &env, aty) {
                    if dn == data {
                        let rec = velim(sig, data, motive, methods, arg_val);
                        result = vapp(result, rec);
                    }
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
        Neutral::NElim(data, m, methods, scrut) => Term::Elim(
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
        Term::Type | Term::Nat | Term::NatLit(_) | Term::Zero | Term::Const(_) => t.clone(),
        Term::Pi(m, a, b) => Term::Pi(*m, Box::new(go(a)), Box::new(go1(b))),
        Term::Sigma(m, a, b) => Term::Sigma(*m, Box::new(go(a)), Box::new(go1(b))),
        Term::Lam(b) => Term::Lam(Box::new(go1(b))),
        Term::App(f0, a) => Term::App(Box::new(go(f0)), Box::new(go(a))),
        Term::Pair(a, b) => Term::Pair(Box::new(go(a)), Box::new(go(b))),
        Term::Fst(p) => Term::Fst(Box::new(go(p))),
        Term::Snd(p) => Term::Snd(Box::new(go(p))),
        Term::Suc(k) => Term::Suc(Box::new(go(k))),
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

/// `P : Π(indices). D params indices → Type`, instantiated at `sparam_tms`.
fn motive_ty_tm(decl: &DataDecl, sparam_tms: &[Term]) -> Term {
    fn go(decl: &DataDecl, sparam_tms: &[Term], i: usize, bd: &mut Vec<usize>, cur: usize) -> Term {
        let p = decl.params.len();
        if i < decl.indices.len() {
            let dom = map_decl(&decl.indices[i].1, p, sparam_tms, bd, cur);
            bd.push(cur);
            let body = go(decl, sparam_tms, i + 1, bd, cur + 1);
            bd.pop();
            Term::Pi(Mult::Omega, Box::new(dom), Box::new(body))
        } else {
            // Π(_ : D params index-vars). Type
            let mut dargs: Vec<Term> =
                (0..p).map(|k| shift(cur, 0, &sparam_tms[k])).collect();
            for ii in 0..decl.indices.len() {
                dargs.push(Term::Var(cur - 1 - bd[ii]));
            }
            Term::Pi(
                Mult::Omega,
                Box::new(Term::Data(decl.name.clone(), dargs)),
                Box::new(Term::Type),
            )
        }
    }
    go(decl, sparam_tms, 0, &mut Vec::new(), 0)
}

/// The method type for one constructor:
///   Π(args). Π(IH for each recursive arg). motive result-idxs (c params args)
fn method_ty_tm(decl: &DataDecl, ctor: &Constructor, sparam_tms: &[Term], motive_tm: &Term) -> Term {
    let p = decl.params.len();
    let data = &decl.name;
    let rec_args: Vec<usize> = ctor
        .args
        .iter()
        .enumerate()
        .filter(|(_, (_, aty))| matches!(aty, Term::Data(dn, _) if dn == data))
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
        let p = decl.params.len();
        if w < rec_args.len() {
            let j = rec_args[w];
            let idx_terms = match &ctor.args[j].1 {
                Term::Data(_, dargs) => dargs[p..].to_vec(),
                _ => unreachable!(),
            };
            let mut ih = shift(cur, 0, motive_tm);
            for it in &idx_terms {
                // a_j's type lives in context [params, args 0..j-1]
                let mi = map_decl(it, p, sparam_tms, &bd[..j], cur);
                ih = Term::App(Box::new(ih), Box::new(mi));
            }
            ih = Term::App(Box::new(ih), Box::new(Term::Var(cur - 1 - bd[j])));
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
    ) -> Term {
        let p = decl.params.len();
        if i < ctor.args.len() {
            let (mult, aty) = &ctor.args[i];
            let dom = map_decl(aty, p, sparam_tms, bd, cur);
            bd.push(cur);
            let body = args(decl, ctor, sparam_tms, motive_tm, rec_args, i + 1, bd, cur + 1);
            bd.pop();
            Term::Pi(*mult, Box::new(dom), Box::new(body))
        } else {
            ihs(decl, ctor, sparam_tms, motive_tm, rec_args, 0, bd, cur)
        }
    }

    let _ = (p, data);
    args(decl, ctor, sparam_tms, motive_tm, &rec_args, 0, &mut Vec::new(), 0)
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

fn check_type(ctx: &Ctx, t: &Term) -> Result<(), String> {
    check(ctx, t, &Value::VType).map(|_| ())
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
            let motive_ty = Value::VPi(
                Mult::Omega,
                Box::new(Value::VNat),
                Closure { sig: ctx.sig.clone(), env: vec![], body: Term::Type },
            );
            check(ctx, p, &motive_ty)?;
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
        Term::Add(a, b) => {
            let ua = check(ctx, a, &Value::VNat)?;
            let ub = check(ctx, b, &Value::VNat)?;
            Ok((Value::VNat, uadd(&ua, &ub)))
        }
        Term::Pi(_, a, b) | Term::Sigma(_, a, b) => {
            check_type(ctx, a)?;
            let va = eval(&ctx.sig, &ctx.env(), a);
            check_type(&ctx.extend(va), b)?;
            Ok((Value::VType, uzero(n)))
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
        Term::Eq(a, x, y) => {
            check_type(ctx, a)?;
            let va = eval(&ctx.sig, &ctx.env(), a);
            check(ctx, x, &va)?;
            check(ctx, y, &va)?;
            Ok((Value::VType, uzero(n)))
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
            Ok((Value::VType, uzero(n)))
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

            // motive
            let mty_tm = motive_ty_tm(&decl, &sparam_tms);
            let mty = eval(&ctx.sig, &ctx.env(), &mty_tm);
            check(ctx, motive, &mty)?; // 0-fragment: usage discarded
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
            let mut umeth = uzero(n);
            for (ci, ctor) in decl.ctors.iter().enumerate() {
                let meth_ty_tm = method_ty_tm(&decl, ctor, &sparam_tms, &motive_tm);
                let meth_ty = eval(&ctx.sig, &ctx.env(), &meth_ty_tm);
                let um = check(ctx, &methods[ci], &meth_ty)?;
                umeth = uadd(&umeth, &um);
            }

            // result: motive indices scrutinee
            let vscrut = eval(&ctx.sig, &ctx.env(), scrut);
            let mut res = vmotive;
            for si in sindices {
                res = vapp(res, si);
            }
            res = vapp(res, vscrut);
            let u = uadd(&uscale(Mult::Omega, &umeth), &uscale(Mult::Omega, &u_scrut));
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
            Term::NatElim(a, b, c, d) => {
                for s in [a, b, c, d] {
                    go(data, s, found);
                }
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
            Term::Ann(e, ty) => {
                go(data, e, found);
                go(data, ty, found);
            }
            _ => {}
        }
    }
    go(data, t, &mut found);
    found
}

/// A constructor argument is strictly positive if the family occurs only as a
/// direct head `Data(self, idxs)` (with no further occurrence in the indices),
/// or not at all.
fn strictly_positive(data: &str, t: &Term) -> bool {
    match t {
        Term::Data(name, args) if name == data => args.iter().all(|a| !occurs(data, a)),
        _ => !occurs(data, t),
    }
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
            ctx = ctx.extend(v);
        }
        let params_ctx = ctx; // context [params]
        // index telescope well-formed (in context [params])
        let mut ictx = params_ctx.extend_clone();
        for (_, ty) in &decl.indices {
            check_type(&ictx, ty).map_err(|e| format!("in {} indices: {e}", decl.name))?;
            let v = eval(&rc, &ictx.env(), ty);
            ictx = ictx.extend(v);
        }
        // each constructor
        for ctor in &decl.ctors {
            let mut cctx = params_ctx.extend_clone();
            for (_, aty) in &ctor.args {
                if !strictly_positive(&decl.name, aty) {
                    return Err(format!(
                        "{}.{}: argument is not strictly positive in `{}`",
                        decl.name, ctor.name, decl.name
                    ));
                }
                check_type(&cctx, aty)
                    .map_err(|e| format!("in {}.{} args: {e}", decl.name, ctor.name))?;
                let v = eval(&rc, &cctx.env(), aty);
                cctx = cctx.extend(v);
            }
            if ctor.idxs.len() != decl.indices.len() {
                return Err(format!(
                    "{}.{}: returns {} index(es), the family has {}",
                    decl.name,
                    ctor.name,
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
                    .map_err(|e| format!("in {}.{} index {i}: {e}", decl.name, ctor.name))?;
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
        .filter(|(_, a)| matches!(a, Term::Data(dn, _) if dn == data))
        .count();
    let mut t = method_ty_tm(decl, ctor, sparam_tms, motive_tm);
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
