use super::Term::*;
use super::*;
use crate::mult::Mult::{Omega, One, Zero};

fn b(t: Term) -> Box<Term> {
    Box::new(t)
}

// ===========================================================================
// the QTT core (built-in Nat / Π / Σ / Eq)
// ===========================================================================

#[test]
fn polymorphic_linear_identity() {
    // λA. λx. x  :  Π[0](A:Type). Π[1](x:A). A
    let ty = Pi(Zero, b(Type), b(Pi(One, b(Var(0)), b(Var(1)))));
    let tm = Lam(b(Lam(b(Var(0)))));
    assert!(check_closed(&tm, &ty).is_ok(), "{:?}", check_closed(&tm, &ty));

    let app = App(b(App(b(Ann(b(tm), b(ty))), b(Nat))), b(NatLit(3)));
    assert_eq!(infer_closed(&app), Ok(Nat));
}

#[test]
fn linearity_is_enforced_under_dependency() {
    let lin = Pi(One, b(Nat), b(Nat));
    assert!(check_closed(&Lam(b(Var(0))), &lin).is_ok());
    assert!(check_closed(&Lam(b(Add(b(Var(0)), b(Var(0))))), &lin).is_err());
    assert!(check_closed(&Lam(b(NatLit(5))), &lin).is_err());

    let unr = Pi(Omega, b(Nat), b(Nat));
    assert!(check_closed(&Lam(b(Add(b(Var(0)), b(Var(0))))), &unr).is_ok());
}

#[test]
fn dependent_pairs() {
    let ty = Sigma(Omega, b(Nat), b(Eq(b(Nat), b(Var(0)), b(NatLit(4)))));
    let good = Pair(b(Add(b(NatLit(2)), b(NatLit(2)))), b(Refl(b(NatLit(4)))));
    assert!(check_closed(&good, &ty).is_ok(), "{:?}", check_closed(&good, &ty));

    let bad = Pair(b(NatLit(5)), b(Refl(b(NatLit(5)))));
    assert!(check_closed(&bad, &ty).is_err());

    let proj = Fst(b(Ann(b(good), b(ty))));
    assert_eq!(infer_closed(&proj), Ok(Nat));
}

#[test]
fn proofs_by_computation_still_work() {
    let good = Refl(b(Add(b(NatLit(2)), b(NatLit(2)))));
    let prop = Eq(b(Nat), b(Add(b(NatLit(2)), b(NatLit(2)))), b(NatLit(4)));
    assert!(check_closed(&good, &prop).is_ok());
    let falseprop = Eq(b(Nat), b(Add(b(NatLit(2)), b(NatLit(2)))), b(NatLit(5)));
    assert!(check_closed(&good, &falseprop).is_err());
}

// the built-in Nat eliminator (kept as a primitive)
fn add_builtin() -> Term {
    Lam(b(Lam(b(NatElim(
        b(Lam(b(Nat))),
        b(Var(0)),
        b(Lam(b(Lam(b(Suc(b(Var(0)))))))),
        b(Var(1)),
    )))))
}
fn add_builtin_ty() -> Term {
    Pi(Omega, b(Nat), b(Pi(Omega, b(Nat), b(Nat))))
}

#[test]
fn natelim_is_a_total_recursor() {
    assert!(check_closed(&add_builtin(), &add_builtin_ty()).is_ok());
    let add = Ann(b(add_builtin()), b(add_builtin_ty()));
    let app = App(b(App(b(add), b(NatLit(2)))), b(NatLit(3)));
    assert_eq!(normalize_closed(&app), NatLit(5));
}

#[test]
fn proof_by_user_defined_computation() {
    let add = Ann(b(add_builtin()), b(add_builtin_ty()));
    let app = App(b(App(b(add), b(NatLit(2)))), b(NatLit(3)));
    let prop = Eq(b(Nat), b(app.clone()), b(NatLit(5)));
    assert!(check_closed(&Refl(b(app)), &prop).is_ok());
}

// ===========================================================================
// GENERAL inductive families (the headline: Nat / Vec / Fin as DECLARATIONS)
// ===========================================================================

fn ctor(name: &str, args: Vec<(crate::mult::Mult, Term)>, idxs: Vec<Term>) -> Constructor {
    Constructor { name: name.to_string(), args, idxs }
}

// ---- Nat as a user datatype "N" ------------------------------------------

fn nat_sig() -> Signature {
    Signature {
        datas: vec![DataDecl {
            name: "N".into(),
            params: vec![],
            indices: vec![],
            ctors: vec![
                ctor("z", vec![], vec![]),
                ctor("s", vec![(Omega, Data("N".into(), vec![]))], vec![]),
            ],
        }],
    }
}
fn n_data() -> Term {
    Data("N".into(), vec![])
}
fn numeral(k: u64) -> Term {
    let mut t = Constr("z".into(), vec![]);
    for _ in 0..k {
        t = Constr("s".into(), vec![t]);
    }
    t
}

#[test]
fn nat_as_a_user_datatype_with_generic_elim() {
    assert!(check_signature(&nat_sig()).is_ok(), "{:?}", check_signature(&nat_sig()));

    // add = λm.λn. elim[N] (λ_.N) n (λk.λih. s ih) m
    let motive = Lam(b(n_data())); // λ_. N
    let z_method = Var(0); // = n
    let s_method = Lam(b(Lam(b(Constr("s".into(), vec![Var(0)]))))); // λk.λih. s ih
    let add = Lam(b(Lam(b(Elim(
        "N".into(),
        b(motive),
        vec![z_method, s_method],
        b(Var(1)), // scrutinee = m
    )))));
    let add_ty = Pi(Omega, b(n_data()), b(Pi(Omega, b(n_data()), b(n_data()))));
    assert!(
        check_closed_in(nat_sig(), &add, &add_ty).is_ok(),
        "{:?}",
        check_closed_in(nat_sig(), &add, &add_ty)
    );

    // add 2 3  ↝  5   (= s(s(s(s(s z)))))
    let app = App(
        b(App(b(Ann(b(add), b(add_ty))), b(numeral(2)))),
        b(numeral(3)),
    );
    assert_eq!(normalize_closed_in(nat_sig(), &app), numeral(5));
    assert_eq!(infer_closed_in(nat_sig(), &app), Ok(n_data()));
}

// ---- length-indexed vectors as a user datatype ---------------------------

fn vec_sig() -> Signature {
    Signature {
        datas: vec![DataDecl {
            name: "Vec".into(),
            params: vec![(Zero, Type)],  // A : Type   (erased)
            indices: vec![(Zero, Nat)],  // n : Nat    (built-in Nat, erased index)
            ctors: vec![
                // vnil : Vec A 0
                ctor("vnil", vec![], vec![NatLit(0)]),
                // vcons : Π(n:Nat). Π(h:A). Π(t:Vec A n). Vec A (suc n)
                ctor(
                    "vcons",
                    vec![
                        (Zero, Nat),                                 // n        (ctx [A])
                        (Omega, Var(1)),                             // h : A    (ctx [A,n])
                        (Omega, Data("Vec".into(), vec![Var(2), Var(1)])), // t : Vec A n
                    ],
                    vec![Suc(b(Var(2)))], // Vec A (suc n)   (ctx [A,n,h,t])
                ),
            ],
        }],
    }
}

// built-in `add` inlined as an eliminator, for the length index `m + n`.
fn add_tm(m: Term, n: Term) -> Term {
    NatElim(b(Lam(b(Nat))), b(n), b(Lam(b(Lam(b(Suc(b(Var(0)))))))), b(m))
}
fn vcons(n: u64, h: u64, t: Term) -> Term {
    Constr("vcons".into(), vec![Nat, NatLit(n), NatLit(h), t])
}
fn vnil() -> Term {
    Constr("vnil".into(), vec![Nat])
}

#[test]
fn vectors_as_a_user_datatype_are_length_indexed() {
    assert!(check_signature(&vec_sig()).is_ok(), "{:?}", check_signature(&vec_sig()));

    // [10,20,30] : Vec Nat 3
    let v3 = vcons(2, 10, vcons(1, 20, vcons(0, 30, vnil())));
    let ty3 = Data("Vec".into(), vec![Nat, NatLit(3)]);
    assert!(check_closed_in(vec_sig(), &v3, &ty3).is_ok(), "{:?}", check_closed_in(vec_sig(), &v3, &ty3));
    // the SAME term is rejected at Vec Nat 2 — the length is in the type
    let ty2 = Data("Vec".into(), vec![Nat, NatLit(2)]);
    assert!(check_closed_in(vec_sig(), &v3, &ty2).is_err());
}

#[test]
fn append_via_generic_elim_tracks_length_and_computes() {
    // append : Π[0](A).Π[0](m).Π[0](n). Vec A m → Vec A n → Vec A (add m n)
    let result = Data("Vec".into(), vec![Var(4), add_tm(Var(3), Var(2))]);
    let p5 = Pi(Omega, b(Data("Vec".into(), vec![Var(3), Var(1)])), b(result)); // ys
    let p4 = Pi(Omega, b(Data("Vec".into(), vec![Var(2), Var(1)])), b(p5)); // xs
    let p3 = Pi(Zero, b(Nat), b(p4)); // n
    let p2 = Pi(Zero, b(Nat), b(p3)); // m
    let append_ty = Pi(Zero, b(Type), b(p2)); // A

    // motive = λk.λ_. Vec A (add k n)
    let motive = Lam(b(Lam(b(Data("Vec".into(), vec![Var(6), add_tm(Var(1), Var(4))])))));
    // vcons-method = λk.λh.λt.λih. vcons A (add k n) h ih
    let vcons_method = Lam(b(Lam(b(Lam(b(Lam(b(Constr(
        "vcons".into(),
        vec![Var(8), add_tm(Var(3), Var(6)), Var(2), Var(0)],
    )))))))));
    let append = Lam(b(Lam(b(Lam(b(Lam(b(Lam(b(Elim(
        "Vec".into(),
        b(motive),
        vec![Var(0) /* vnil → ys */, vcons_method],
        b(Var(1)), // scrutinee = xs
    )))))))))));

    assert!(
        check_closed_in(vec_sig(), &append, &append_ty).is_ok(),
        "{:?}",
        check_closed_in(vec_sig(), &append, &append_ty)
    );

    // append Nat 2 1 [10,20] [30]  :  Vec Nat 3   ↝  [10,20,30]
    let xs = vcons(1, 10, vcons(0, 20, vnil()));
    let ys = vcons(0, 30, vnil());
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
    assert_eq!(
        infer_closed_in(vec_sig(), &app),
        Ok(Data("Vec".into(), vec![Nat, NatLit(3)]))
    );
    let expected = vcons(2, 10, vcons(1, 20, vcons(0, 30, vnil())));
    assert_eq!(normalize_closed_in(vec_sig(), &app), expected);
}

// ---- Fin: a two-constructor indexed family with recursion ----------------

fn fin_sig() -> Signature {
    Signature {
        datas: vec![DataDecl {
            name: "Fin".into(),
            params: vec![],
            indices: vec![(Zero, Nat)],
            ctors: vec![
                // fz : Π(n:Nat). Fin (suc n)
                ctor("fz", vec![(Zero, Nat)], vec![Suc(b(Var(0)))]),
                // fs : Π(n:Nat). Fin n → Fin (suc n)
                ctor(
                    "fs",
                    vec![(Zero, Nat), (Omega, Data("Fin".into(), vec![Var(0)]))],
                    vec![Suc(b(Var(1)))],
                ),
            ],
        }],
    }
}

#[test]
fn fin_is_a_recursive_indexed_family() {
    assert!(check_signature(&fin_sig()).is_ok(), "{:?}", check_signature(&fin_sig()));

    // fin2nat : Π[0](n:Nat). Fin n → Nat
    //   = λn.λi. elim[Fin] (λm.λ_. Nat) (λm. zero) (λm.λprev.λih. suc ih) i
    let motive = Lam(b(Lam(b(Nat))));
    let fz_method = Lam(b(NatLit(0))); // λm. 0
    let fs_method = Lam(b(Lam(b(Lam(b(Suc(b(Var(0))))))))); // λm.λprev.λih. suc ih
    let fin2nat = Lam(b(Lam(b(Elim(
        "Fin".into(),
        b(motive),
        vec![fz_method, fs_method],
        b(Var(0)), // scrutinee = i
    )))));
    let fin2nat_ty = Pi(Zero, b(Nat), b(Pi(Omega, b(Data("Fin".into(), vec![Var(0)])), b(Nat))));
    assert!(
        check_closed_in(fin_sig(), &fin2nat, &fin2nat_ty).is_ok(),
        "{:?}",
        check_closed_in(fin_sig(), &fin2nat, &fin2nat_ty)
    );

    // the element "1" of Fin 2:  fs 1 (fz 0) : Fin 2     fin2nat 2 (…) ↝ 1
    let elem = Constr("fs".into(), vec![NatLit(1), Constr("fz".into(), vec![NatLit(0)])]);
    let app = App(b(App(b(Ann(b(fin2nat), b(fin2nat_ty))), b(NatLit(2)))), b(elem));
    assert_eq!(infer_closed_in(fin_sig(), &app), Ok(Nat));
    assert_eq!(normalize_closed_in(fin_sig(), &app), NatLit(1));
}

#[test]
fn strict_positivity_is_enforced() {
    // a non-strictly-positive "Bad" with ctor  mk : (Bad → Bad) → Bad  is rejected
    let bad = Signature {
        datas: vec![DataDecl {
            name: "Bad".into(),
            params: vec![],
            indices: vec![],
            ctors: vec![ctor(
                "mk",
                vec![(Omega, Pi(Omega, b(Data("Bad".into(), vec![])), b(Data("Bad".into(), vec![]))))],
                vec![],
            )],
        }],
    };
    assert!(check_signature(&bad).is_err());
}
