//! The DEPENDENT core (Route B: proofs-as-terms), via normalization by
//! evaluation. A Rust port of the engine in `../../prototype/qtt_checker.py`.
//!
//! One syntax for terms AND types (types are terms). A universe `Type`, Π, λ,
//! application, a base `Nat` with `+` (to demonstrate computation), and an
//! identity type `Eq A a b` with `refl` — so a PROOF is a term whose type is a
//! proposition, checked by NORMALISATION:  `refl : Eq Nat (2+2) 4` type-checks
//! because `2+2` and `4` have the same normal form.
//!
//! Definitional equality is decided by NbE: `eval` to semantic values (with
//! closures for binders), `quote` back to a β-normal term, and compare. This is
//! the kernel every dependently-typed proof assistant is built on.
//!
//! `Type : Type` for now (fine for a programming language; for a *logic* it
//! needs a universe hierarchy — see ../../docs/07-implementation-guide.md §7).
//! Multiplicities (the linear/quantitative layer) and inductive families layer
//! on top of this engine next.

#[derive(Clone, PartialEq, Debug)]
pub enum Term {
    Var(usize),                              // de Bruijn index (0 = innermost)
    Type,                                    // the universe
    Pi(Box<Term>, Box<Term>),                // Π(_:A). B   (B binds the variable)
    Lam(Box<Term>),                          // λ. body
    App(Box<Term>, Box<Term>),
    Nat,
    NatLit(u64),
    Add(Box<Term>, Box<Term>),
    Eq(Box<Term>, Box<Term>, Box<Term>),     // Eq A a b   (the identity type)
    Refl(Box<Term>),                         // refl a : Eq _ a a
    Ann(Box<Term>, Box<Term>),               // (e : T)   — type annotation
}

#[derive(Clone)]
pub enum Value {
    VType,
    VPi(Box<Value>, Closure),
    VLam(Closure),
    VNat,                                    // the type Nat
    VNatLit(u64),
    VEq(Box<Value>, Box<Value>, Box<Value>),
    VRefl(Box<Value>),
    VNeu(Neutral),
}

#[derive(Clone)]
pub enum Neutral {
    NVar(usize),                             // de Bruijn LEVEL
    NApp(Box<Neutral>, Box<Value>),
    NAdd(Box<Value>, Box<Value>),
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
// evaluation  (term -> value)
// ---------------------------------------------------------------------------

fn eval(env: &[Value], t: &Term) -> Value {
    match t {
        Term::Var(i) => env[env.len() - 1 - i].clone(),
        Term::Type => Value::VType,
        Term::Pi(a, b) => Value::VPi(
            Box::new(eval(env, a)),
            Closure { env: env.to_vec(), body: (**b).clone() },
        ),
        Term::Lam(b) => Value::VLam(Closure { env: env.to_vec(), body: (**b).clone() }),
        Term::App(f, a) => vapp(eval(env, f), eval(env, a)),
        Term::Nat => Value::VNat,
        Term::NatLit(n) => Value::VNatLit(*n),
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

// ---------------------------------------------------------------------------
// quotation  (value -> β-normal term), and definitional equality
// ---------------------------------------------------------------------------

fn quote(lvl: usize, v: &Value) -> Term {
    match v {
        Value::VType => Term::Type,
        Value::VPi(a, clo) => Term::Pi(
            Box::new(quote(lvl, a)),
            Box::new(quote(lvl + 1, &clo.apply(Value::VNeu(Neutral::NVar(lvl))))),
        ),
        Value::VLam(clo) => {
            Term::Lam(Box::new(quote(lvl + 1, &clo.apply(Value::VNeu(Neutral::NVar(lvl))))))
        }
        Value::VNat => Term::Nat,
        Value::VNatLit(n) => Term::NatLit(*n),
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
    }
}

fn conv(lvl: usize, a: &Value, b: &Value) -> bool {
    quote(lvl, a) == quote(lvl, b)
}

// ---------------------------------------------------------------------------
// bidirectional type checking with conversion
// ---------------------------------------------------------------------------

pub struct Ctx {
    types: Vec<Value>, // types[level] = the type of the variable at that level
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

fn check(ctx: &Ctx, t: &Term, ty: &Value) -> Result<(), String> {
    match (t, ty) {
        (Term::Lam(body), Value::VPi(dom, cod)) => {
            let ctx2 = ctx.extend((**dom).clone());
            let codty = cod.apply(Value::VNeu(Neutral::NVar(ctx.level())));
            check(&ctx2, body, &codty)
        }
        (Term::Refl(a), Value::VEq(aty, x, y)) => {
            check(ctx, a, aty)?;
            let va = eval(&ctx.env(), a);
            if conv(ctx.level(), x, &va) && conv(ctx.level(), y, &va) {
                Ok(())
            } else {
                Err(format!(
                    "refl: the equation does not hold by computation:\n  {:?}  =/=  {:?}",
                    quote(ctx.level(), x),
                    quote(ctx.level(), y)
                ))
            }
        }
        _ => {
            let ty2 = infer(ctx, t)?;
            if conv(ctx.level(), ty, &ty2) {
                Ok(())
            } else {
                Err(format!(
                    "type mismatch:\n  expected {:?}\n  found    {:?}",
                    quote(ctx.level(), ty),
                    quote(ctx.level(), &ty2)
                ))
            }
        }
    }
}

fn infer(ctx: &Ctx, t: &Term) -> Result<Value, String> {
    match t {
        Term::Var(i) => {
            if *i >= ctx.level() {
                return Err(format!("unbound variable (de Bruijn {i})"));
            }
            Ok(ctx.types[ctx.level() - 1 - i].clone())
        }
        Term::Type => Ok(Value::VType), // Type : Type
        Term::Nat => Ok(Value::VType),
        Term::NatLit(_) => Ok(Value::VNat),
        Term::Add(a, b) => {
            check(ctx, a, &Value::VNat)?;
            check(ctx, b, &Value::VNat)?;
            Ok(Value::VNat)
        }
        Term::Pi(a, b) => {
            check(ctx, a, &Value::VType)?;
            let va = eval(&ctx.env(), a);
            check(&ctx.extend(va), b, &Value::VType)?;
            Ok(Value::VType)
        }
        Term::Eq(a, x, y) => {
            check(ctx, a, &Value::VType)?;
            let va = eval(&ctx.env(), a);
            check(ctx, x, &va)?;
            check(ctx, y, &va)?;
            Ok(Value::VType)
        }
        Term::Refl(a) => {
            let ta = infer(ctx, a)?;
            let va = eval(&ctx.env(), a);
            Ok(Value::VEq(Box::new(ta), Box::new(va.clone()), Box::new(va)))
        }
        Term::App(f, x) => match infer(ctx, f)? {
            Value::VPi(dom, cod) => {
                check(ctx, x, &dom)?;
                Ok(cod.apply(eval(&ctx.env(), x)))
            }
            other => Err(format!(
                "application of a non-function (type {:?})",
                quote(ctx.level(), &other)
            )),
        },
        Term::Ann(e, ty) => {
            check(ctx, ty, &Value::VType)?;
            let vty = eval(&ctx.env(), ty);
            check(ctx, e, &vty)?;
            Ok(vty)
        }
        Term::Lam(_) => Err("cannot infer a bare lambda; annotate it `(e : T)`".to_string()),
    }
}

/// Check a closed term against a closed type.
pub fn check_closed(t: &Term, ty: &Term) -> Result<(), String> {
    let ctx = Ctx::new();
    check(&ctx, ty, &Value::VType)?;
    let vty = eval(&[], ty);
    check(&ctx, t, &vty)
}

/// Infer the (β-normal) type of a closed term.
pub fn infer_closed(t: &Term) -> Result<Term, String> {
    Ok(quote(0, &infer(&Ctx::new(), t)?))
}

#[cfg(test)]
mod tests {
    use super::Term::*;
    use super::*;

    fn b(t: Term) -> Box<Term> {
        Box::new(t)
    }

    // Π(A:Type). Π(x:A). A
    fn id_ty() -> Term {
        Pi(b(Type), b(Pi(b(Var(0)), b(Var(1)))))
    }
    // λA. λx. x
    fn id_tm() -> Term {
        Lam(b(Lam(b(Var(0)))))
    }

    #[test]
    fn dependent_function() {
        // the polymorphic identity checks at its dependent type
        assert!(check_closed(&id_tm(), &id_ty()).is_ok());
        // apply it to a type and a value:  id Nat 3  :  Nat
        let app = App(b(App(b(Ann(b(id_tm()), b(id_ty()))), b(Nat))), b(NatLit(3)));
        assert_eq!(infer_closed(&app), Ok(Nat));
    }

    #[test]
    fn proofs_by_computation() {
        // refl : Eq Nat (2 + 2) 4   -- a PROOF, checked by normalisation
        let good = Refl(b(Add(b(NatLit(2)), b(NatLit(2)))));
        let prop = Eq(b(Nat), b(Add(b(NatLit(2)), b(NatLit(2)))), b(NatLit(4)));
        assert!(check_closed(&good, &prop).is_ok());

        // a dependent application reduces inside the proposition, too:
        //   refl : Eq Nat ((λx.x) 7) 7
        let idapp = App(b(Ann(b(Lam(b(Var(0)))), b(Pi(b(Nat), b(Nat))))), b(NatLit(7)));
        let prop2 = Eq(b(Nat), b(idapp), b(NatLit(7)));
        assert!(check_closed(&Refl(b(NatLit(7))), &prop2).is_ok());
    }

    #[test]
    fn bad_proofs_and_mismatches_rejected() {
        // refl cannot prove a false equation
        let bad = Refl(b(Add(b(NatLit(2)), b(NatLit(2)))));
        let falseprop = Eq(b(Nat), b(Add(b(NatLit(2)), b(NatLit(2)))), b(NatLit(5)));
        assert!(check_closed(&bad, &falseprop).is_err());

        // plain type mismatch
        assert!(check_closed(&NatLit(3), &Type).is_err());
        // ill-typed application
        assert!(infer_closed(&App(b(NatLit(1)), b(NatLit(2)))).is_err());
    }
}
