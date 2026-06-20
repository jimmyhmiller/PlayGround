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
    let ty = Pi(Zero, b(Type(0)), b(Pi(One, b(Var(0)), b(Var(1)))));
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

// ===========================================================================
// PHASE E3 — GENERAL strictly-positive eliminators (higher-order recursive args)
// ===========================================================================

fn datadecl(name: &str, params: Vec<(Mult, Term)>, indices: Vec<(Mult, Term)>, ctors: Vec<Constructor>) -> DataDecl {
    DataDecl { name: name.into(), universe: 0, params, indices, ctors }
}

/// A W-type-shaped family: `Tree` with a finitely-branching node whose children
/// are a FUNCTION `Bool → Tree` — a HIGHER-ORDER recursive constructor argument.
fn tree_sig() -> Signature {
    Signature {
        postulates: vec![],
        datas: vec![
            datadecl("Bool", vec![], vec![], vec![ctor("btrue", vec![], vec![]), ctor("bfalse", vec![], vec![])]),
            datadecl(
                "Tree",
                vec![],
                vec![],
                vec![
                    ctor("leaf", vec![], vec![]),
                    // node2 : (Bool → Tree) → Tree   (the higher-order recursive field)
                    ctor(
                        "node2",
                        vec![(Omega, Pi(Omega, b(Data("Bool".into(), vec![])), b(Data("Tree".into(), vec![]))))],
                        vec![],
                    ),
                ],
            ),
        ],
    }
}

#[test]
fn higher_order_recursive_datatype_is_well_formed_and_eliminates() {
    // strict positivity ACCEPTS the higher-order field (Tree only in the codomain).
    assert!(check_signature(&tree_sig()).is_ok(), "{:?}", check_signature(&tree_sig()));

    // `size t = ` leaves count, via the eliminator. The node method receives the
    // induction hypothesis `ih : (b:Bool) → Nat` (a NATIVE closure `λb. elim (f b)`)
    // and sums it over both children: size(node2 f) = ih btrue + ih bfalse.
    let btrue = Constr("btrue".into(), vec![]);
    let bfalse = Constr("bfalse".into(), vec![]);
    let motive = Lam(b(Nat)); // λ_. Nat
    let leaf_m = NatLit(1);
    // λ f. λ ih. (ih btrue) + (ih bfalse)
    let node2_m = Lam(b(Lam(b(Add(
        b(App(b(Var(0)), b(btrue.clone()))),
        b(App(b(Var(0)), b(bfalse.clone()))),
    )))));
    // a concrete tree: node2 (λb. leaf) — two leaves ⇒ size 2.
    let tree = Constr("node2".into(), vec![Lam(b(Constr("leaf".into(), vec![])))]);
    let elim = Elim(
        "Tree".into(),
        b(motive),
        vec![leaf_m, node2_m],
        b(tree),
    );
    // it has type Nat and the higher-order IH genuinely fires: ↝ 2.
    assert_eq!(infer_closed_in(tree_sig(), &elim), Ok(Nat));
    assert_eq!(normalize_closed_in(tree_sig(), &elim), NatLit(2));
}

#[test]
fn deeper_higher_order_recursion_computes() {
    // node2 (λb. node2 (λc. leaf)) — four leaves ⇒ size 4. Exercises the native
    // IH recursing through a nested node (the closure applied, then re-eliminated).
    let btrue = Constr("btrue".into(), vec![]);
    let bfalse = Constr("bfalse".into(), vec![]);
    let motive = Lam(b(Nat));
    let leaf_m = NatLit(1);
    let node2_m = Lam(b(Lam(b(Add(
        b(App(b(Var(0)), b(btrue))),
        b(App(b(Var(0)), b(bfalse))),
    )))));
    let inner = Constr("node2".into(), vec![Lam(b(Constr("leaf".into(), vec![])))]);
    let tree = Constr("node2".into(), vec![Lam(b(inner))]);
    let elim = Elim("Tree".into(), b(motive), vec![leaf_m, node2_m], b(tree));
    assert_eq!(normalize_closed_in(tree_sig(), &elim), NatLit(4));
}

#[test]
fn strict_positivity_rejects_negative_and_double_negative_occurrences() {
    // single NEGATIVE: `mk : (Bad → Bad) → Bad` — Bad left of an arrow. REJECTED.
    let neg = Signature {
        postulates: vec![],
        datas: vec![datadecl(
            "Bad",
            vec![],
            vec![],
            vec![ctor("mk", vec![(Omega, Pi(Omega, b(Data("Bad".into(), vec![])), b(Data("Bad".into(), vec![]))))], vec![])],
        )],
    };
    assert!(check_signature(&neg).is_err(), "single-negative occurrence must be rejected");

    // DOUBLE NEGATIVE: `mk : ((Bad → Nat) → Nat) → Bad` — Bad two arrows deep (a
    // strictly-positive checker conservatively REJECTS this; it is not strictly
    // positive even though it is "positive").
    let dneg = Signature {
        postulates: vec![],
        datas: vec![datadecl(
            "Bad",
            vec![],
            vec![],
            vec![ctor(
                "mk",
                vec![(
                    Omega,
                    Pi(
                        Omega,
                        b(Pi(Omega, b(Data("Bad".into(), vec![])), b(Nat))),
                        b(Nat),
                    ),
                )],
                vec![],
            )],
        )],
    };
    assert!(check_signature(&dneg).is_err(), "double-negative occurrence must be rejected");

    // POSITIVE higher-order (Tree's node2) is ACCEPTED — the discriminating test.
    assert!(check_signature(&tree_sig()).is_ok());
}

#[test]
fn acc_accessibility_family_is_well_formed() {
    // `Acc (A:Type) (R:A→A→Type) : A → Type` with
    //   acc : (x:A) → ((y:A) → R y x → Acc A R y) → Acc A R x
    // — the INDEXED higher-order recursive family at the heart of well-founded
    // recursion. Its `acc` field is strictly positive (Acc only in the codomain
    // head `Acc A R y`), and its eliminator (= the well-founded recursor) is
    // generated by the general machinery. check_signature exercises the indexed
    // strict-positivity + telescope type-checking of the higher-order field.
    let r_ty = Pi(Omega, b(Var(0)), b(Pi(Omega, b(Var(1)), b(Type(0))))); // A→A→Type, ctx [A]
    let acc_fn = Pi(
        Omega,
        b(Var(2)), // y : A           (ctx [A,R,x])
        b(Pi(
            Omega,
            b(App(b(App(b(Var(2)), b(Var(0)))), b(Var(1)))), // R y x   (ctx [A,R,x,y])
            b(Data("Acc".into(), vec![Var(4), Var(3), Var(1)])), // Acc A R y (ctx […,proof])
        )),
    );
    let sig = Signature {
        postulates: vec![],
        datas: vec![datadecl(
            "Acc",
            vec![(Zero, Type(0)), (Zero, r_ty)],     // params A, R
            vec![(Zero, Var(1))],                     // index x : A   (ctx [A,R])
            vec![ctor(
                "acc",
                vec![(Zero, Var(1)), (Omega, acc_fn)], // x:A, then the accessibility fn
                vec![Var(1)],                          // result index x  (ctx [A,R,x,fn])
            )],
        )],
    };
    assert!(check_signature(&sig).is_ok(), "{:?}", check_signature(&sig));

    // and a NON-strictly-positive variant — `acc`'s field puts `Acc` to the LEFT
    // of an arrow (`(Acc A R y → R y x) → …`) — must be REJECTED.
    let bad_fn = Pi(
        Omega,
        b(Var(2)),
        b(Pi(
            Omega,
            b(Pi(Omega, b(Data("Acc".into(), vec![Var(3), Var(2), Var(0)])), b(App(b(App(b(Var(2)), b(Var(0)))), b(Var(1)))))),
            b(Data("Acc".into(), vec![Var(4), Var(3), Var(1)])),
        )),
    );
    let bad = Signature {
        postulates: vec![],
        datas: vec![datadecl(
            "Acc",
            vec![(Zero, Pi(Omega, b(Var(0)), b(Pi(Omega, b(Var(1)), b(Type(0))))))],
            vec![(Zero, Var(0))],
            vec![ctor("acc", vec![(Zero, Var(0)), (Omega, bad_fn)], vec![Var(1)])],
        )],
    };
    assert!(check_signature(&bad).is_err(), "Acc left of an arrow must be rejected");
}

// ---- Nat as a user datatype "N" ------------------------------------------

fn nat_sig() -> Signature {
    Signature {
        postulates: vec![],
        datas: vec![DataDecl {
            name: "N".into(),
            universe: 0,
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
        postulates: vec![],
        datas: vec![DataDecl {
            name: "Vec".into(),
            universe: 0,
            params: vec![(Zero, Type(0))],  // A : Type   (erased)
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
    let append_ty = Pi(Zero, b(Type(0)), b(p2)); // A

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
        postulates: vec![],
        datas: vec![DataDecl {
            name: "Fin".into(),
            universe: 0,
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
        postulates: vec![],
        datas: vec![DataDecl {
            name: "Bad".into(),
            universe: 0,
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

// ===========================================================================
// PHASE F — the universe hierarchy (replaces `Type : Type`). These are the
// ADVERSARIAL/soundness tests: programs that SHOULD be rejected, each shown to
// be rejected for a genuine UNIVERSE reason (not an incidental one), plus the
// invariants that prove the hierarchy did not silently collapse.
// ===========================================================================

#[test]
fn universe_hierarchy_is_stratified_not_type_in_type() {
    // `Type i : Type (i+1)` — and crucially NOT `Type i : Type i` (which was the
    // old `Type : Type`, inconsistent by Girard's paradox).
    assert_eq!(infer_closed(&Type(0)), Ok(Type(1)));
    assert_eq!(infer_closed(&Type(1)), Ok(Type(2)));
    assert_eq!(infer_closed(&Type(7)), Ok(Type(8)));
    // `Type 0 : Type 0` must FAIL (the heart of the matter).
    assert!(check_closed(&Type(0), &Type(0)).is_err());
}

#[test]
fn cumulativity_is_one_directional_and_does_not_collapse_the_hierarchy() {
    // CUMULATIVITY (upward only): a `Type 0` is accepted where `Type 2` is wanted.
    assert!(check_closed(&Type(0), &Type(2)).is_ok());
    assert!(check_closed(&Nat, &Type(0)).is_ok());
    assert!(check_closed(&Nat, &Type(5)).is_ok()); // Nat : Type 0 ⊑ Type 5
    // …but DOWNWARD subsumption is rejected — `Type 2` is NOT a `Type 1`. If the
    // hierarchy had collapsed (`Type i ≡ Type j`), this would wrongly succeed and
    // everything would look green. This is the decisive non-collapse check.
    let err = check_closed(&Type(2), &Type(1)).unwrap_err();
    assert!(
        err.contains("universe") && err.contains("Type"),
        "downward subsumption must fail with a universe error, got: {err}"
    );
    // and `Type 1 : Type 1` is likewise rejected (only `Type 1 : Type 2`).
    assert!(check_closed(&Type(1), &Type(1)).is_err());
    assert!(check_closed(&Type(1), &Type(2)).is_ok());
}

#[test]
fn pi_lives_in_the_max_of_its_parts() {
    // `(0 _ : Type 0) → Type 0`  is a `Type 1`  (max(1,1) — Type 0 itself is a
    // Type 1), so it inhabits `Type 1` but NOT `Type 0`.
    let arrow = Pi(Zero, b(Type(0)), b(Type(0)));
    assert_eq!(infer_closed(&arrow), Ok(Type(1)));
    assert!(check_closed(&arrow, &Type(1)).is_ok());
    assert!(check_closed(&arrow, &Type(0)).is_err());
    // `(0 _ : Type 3) → Nat`  is a `Type 4`  (max(4, 1)).
    let arrow2 = Pi(Zero, b(Type(3)), b(Nat));
    assert_eq!(infer_closed(&arrow2), Ok(Type(4)));
}

// --- THE PARADOX BLOCKER: a datatype may not quantify over its own universe ---

/// `data U where mk : (_ : Type 0) → U` at a chosen universe. Storing a `Type 0`
/// inside `U` is the type-in-type retract that Girard's/Hurkens' paradox needs:
/// if `U : Type 0`, then `U` is a `Type 0` that contains a code for every `Type
/// 0` — including itself — and `False` becomes inhabited in the total fragment.
fn universe_storing_sig(universe: usize) -> Signature {
    Signature {
        postulates: vec![],
        datas: vec![DataDecl {
            name: "U".into(),
            universe,
            params: vec![],
            indices: vec![],
            ctors: vec![ctor("mk", vec![(Omega, Type(0))], vec![])],
        }],
    }
}

#[test]
fn datatype_cannot_sit_in_its_own_universe_girard_blocker() {
    // (a) Declared at `Type 0`: REJECTED, and for a GENUINE universe reason — the
    //     error must talk about the universe restriction, not arity/positivity/a
    //     name lookup. (A faked checker would error for an unrelated cause.)
    let err = check_signature(&universe_storing_sig(0)).unwrap_err();
    assert!(
        err.contains("Type 0") && err.contains("Type 1"),
        "must be rejected with a universe-level diagnostic, got: {err}"
    );
    assert!(
        err.contains("predicativity") || err.contains("Girard") || err.contains("own universe"),
        "rejection must cite the predicativity/Girard restriction, got: {err}"
    );

    // (b) RELAX ONLY THE UNIVERSE BOUND — the very same datatype, declared one
    //     universe up (`U : Type 1`), is ACCEPTED. This proves the restriction is
    //     LOAD-BEARING: it is the universe check doing the rejecting in (a), not
    //     something incidental about the term.
    assert!(
        check_signature(&universe_storing_sig(1)).is_ok(),
        "{:?}",
        check_signature(&universe_storing_sig(1))
    );
    // …and declaring it even higher is fine too (cumulative headroom).
    assert!(check_signature(&universe_storing_sig(3)).is_ok());
}

#[test]
fn strict_positivity_and_universe_checks_are_independent() {
    // A datatype storing a *higher* universe is caught by the UNIVERSE rule even
    // though it is perfectly strictly-positive — the two guards are distinct.
    let sig = universe_storing_sig(0);
    // it IS strictly positive (no self-occurrence to the left of an arrow):
    assert!(strictly_positive("U", &Type(0)));
    // yet check_signature still rejects it — on universe grounds.
    assert!(check_signature(&sig).is_err());
}

// --- large elimination still works, at universes above 0 ---

#[test]
fn large_elimination_into_a_higher_universe() {
    // A type computed by recursion: `natElim (λ_. Type 0) Nat (λk.λih. Nat) 3`.
    // The motive `λ_. Type 0 : Nat → Type 1` targets universe 1 (large elim), so
    // this exercises `motive_level` at ℓ = 1 — a path that the old fixed `Nat →
    // Type` motive (built on Type:Type) could not type honestly.
    let motive = Lam(b(Type(0))); // λ_. Type 0
    let elim = NatElim(
        b(motive),
        b(Nat),                                  // base:  Nat : Type 0
        b(Lam(b(Lam(b(Nat))))),                  // step:  λk.λih. Nat
        b(NatLit(3)),
    );
    // it has type `Type 0` and normalizes to `Nat`.
    assert_eq!(infer_closed(&elim), Ok(Type(0)));
    assert_eq!(normalize_closed(&elim), Nat);
}

// --- the linearity invariants are UNTOUCHED by the universe work ---

#[test]
fn linearity_still_enforced_after_universes() {
    // ω ⋢ 1 : a linearly-bound variable used twice is still rejected.
    let lin = Pi(One, b(Nat), b(Nat));
    assert!(check_closed(&Lam(b(Add(b(Var(0)), b(Var(0))))), &lin).is_err());
    // 0 ⋢ 1 : dropping a linear variable (a leak) is still rejected.
    assert!(check_closed(&Lam(b(NatLit(5))), &lin).is_err());
    // the polymorphic LINEAR identity still checks at the new `Type 0`.
    let id_ty = Pi(Zero, b(Type(0)), b(Pi(One, b(Var(0)), b(Var(1)))));
    assert!(check_closed(&Lam(b(Lam(b(Var(0))))), &id_ty).is_ok());
}

// --- regressions for holes found by the adversarial red-team ---

#[test]
fn universe_level_successor_does_not_overflow() {
    // `Type usize::MAX` must NOT wrap to `Type 0` (which would re-accept
    // `Type MAX : Type 0` — Type:Type at the apex) nor panic. It is a hard error.
    assert!(infer_closed(&Type(usize::MAX)).is_err());
    // and the wrap-to-Type-0 acceptance is gone:
    assert!(check_closed(&Type(usize::MAX), &Type(0)).is_err());
    // a normal large-but-finite level is still fine.
    assert_eq!(infer_closed(&Type(1000)), Ok(Type(1001)));
}

#[test]
fn a_parameter_ranging_over_a_universe_lifts_the_datatype_universe() {
    // `data Box (A : Type 1) where mk : Box`  — a PHANTOM parameter over `Type 1`.
    // The constructor stores nothing, so the ctor-arg check never fires; the
    // PARAMETER-telescope check must still force `Box` up to `Type 1`.
    let mk_box = |universe: usize| Signature {
        postulates: vec![],
        datas: vec![DataDecl {
            name: "Box".into(),
            universe,
            params: vec![(Zero, Type(1))], // A : Type 1
            indices: vec![],
            ctors: vec![ctor("mk", vec![], vec![])],
        }],
    };
    // declared at Type 0: REJECTED for a genuine universe/predicativity reason.
    let err = check_signature(&mk_box(0)).unwrap_err();
    assert!(
        err.contains("Type 1") && err.contains("parameter") && err.contains("predicativity"),
        "must be a predicativity rejection citing the parameter, got: {err}"
    );
    // relax ONLY the universe bound → accepted (load-bearing).
    assert!(check_signature(&mk_box(1)).is_ok(), "{:?}", check_signature(&mk_box(1)));
}

#[test]
fn an_index_ranging_over_a_universe_lifts_the_datatype_universe() {
    // `data Tag : Type 1 → Type`  with  `mk : Tag (Type 0)` — the index ranges
    // over `Type 1` and `mk` pins a genuine universe value into its type. The
    // INDEX-telescope check must force `Tag` up to `Type 1`.
    let mk_tag = |universe: usize| Signature {
        postulates: vec![],
        datas: vec![DataDecl {
            name: "Tag".into(),
            universe,
            params: vec![],
            indices: vec![(Zero, Type(1))], // index : Type 1
            ctors: vec![ctor("mk", vec![], vec![Type(0)])], // mk : Tag (Type 0)
        }],
    };
    let err = check_signature(&mk_tag(0)).unwrap_err();
    assert!(
        err.contains("Type 1") && err.contains("index") && err.contains("predicativity"),
        "must be a predicativity rejection citing the index, got: {err}"
    );
    assert!(check_signature(&mk_tag(1)).is_ok(), "{:?}", check_signature(&mk_tag(1)));
}

#[test]
fn vec_style_value_parameter_does_not_over_restrict() {
    // GUARD AGAINST OVER-CORRECTION: a `Type 0` parameter (Vec-style) and a `Nat`
    // index must NOT be forced above `Type 0` — only universes ≥ the family's
    // level lift it. This keeps real length-indexed data at `Type 0`.
    assert!(check_signature(&vec_sig()).is_ok(), "{:?}", check_signature(&vec_sig()));
    assert!(check_signature(&fin_sig()).is_ok(), "{:?}", check_signature(&fin_sig()));
    // and a `Type 0` parameter explicitly stays at universe 0:
    let box0 = Signature {
        postulates: vec![],
        datas: vec![DataDecl {
            name: "Phantom".into(),
            universe: 0,
            params: vec![(Zero, Type(0))], // A : Type 0  ⇒ contributes level 0
            indices: vec![],
            ctors: vec![ctor("mk", vec![], vec![])],
        }],
    };
    assert!(check_signature(&box0).is_ok(), "{:?}", check_signature(&box0));
}

#[test]
fn struct_predicativity_diagnostic_does_not_double_name() {
    // a struct's single constructor shares the type's name; the predicativity
    // error must read `SBox`, not `SBox.SBox`.
    let sbox = Signature {
        postulates: vec![],
        datas: vec![DataDecl {
            name: "SBox".into(),
            universe: 0,
            params: vec![],
            indices: vec![],
            ctors: vec![ctor("SBox", vec![(Omega, Type(0))], vec![])], // field : Type
        }],
    };
    let err = check_signature(&sbox).unwrap_err();
    assert!(err.contains("SBox:"), "expected `SBox:` prefix, got: {err}");
    assert!(!err.contains("SBox.SBox"), "must not double-name the struct, got: {err}");
}
