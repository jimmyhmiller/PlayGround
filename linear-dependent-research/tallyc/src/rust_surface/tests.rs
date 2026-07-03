use super::*;
use crate::dep::Term;

fn ndata() -> Term {
    Term::Data("Nat".into(), vec![])
}
fn zero() -> Term {
    Term::Constr("Zero".into(), vec![])
}
fn succ(t: Term) -> Term {
    Term::Constr("Succ".into(), vec![t])
}
fn num(k: u64) -> Term {
    let mut t = zero();
    for _ in 0..k {
        t = succ(t);
    }
    t
}

const NAT: &str = r#"
enum Nat {
    Zero : Nat,
    Succ : Nat -> Nat,
}

add : Nat -> Nat -> Nat
fn add(m, n) {
    match m {
        Zero    => n,
        Succ(k) => Succ(add(k, n)),
    }
}
"#;

#[test]
fn nat_add_compiles_and_runs() {
    let src = format!(
        "{NAT}\nmain : Nat\nfn main() {{ add(Succ(Succ(Zero)), Succ(Succ(Succ(Zero)))) }}\n"
    );
    let prog = check_program(&src).unwrap_or_else(|e| panic!("{e:?}"));
    assert_eq!(prog.normalize("main"), Some(num(5)));
}

#[test]
fn a_proof_by_computation() {
    let src = format!(
        "{NAT}\np : Eq Nat (add (Succ Zero) (Succ Zero)) (Succ (Succ Zero))\nfn p() {{ refl(add(Succ(Zero), Succ(Zero))) }}\n"
    );
    assert!(check_program(&src).is_ok(), "{:?}", check_program(&src).err());

    let bad = format!(
        "{NAT}\np : Eq Nat (add (Succ Zero) (Succ Zero)) (Succ Zero)\nfn p() {{ refl(add(Succ(Zero), Succ(Zero))) }}\n"
    );
    assert!(check_program(&bad).is_err());
}

// length-indexed vectors with IMPLICIT element type and indices — the `a` and
// `k` arguments are written nowhere in the constructor uses; they are inferred.
const VEC: &str = r#"
enum Nat {
    Zero : Nat,
    Succ : Nat -> Nat,
}

add : Nat -> Nat -> Nat
fn add(m, n) {
    match m {
        Zero    => n,
        Succ(k) => Succ(add(k, n)),
    }
}

boxed enum Vec (a : Type) : Nat -> Type {
    Nil  : Vec a Zero,
    Cons : {0 k : Nat} -> a -> Vec a k -> Vec a (Succ k),
}

append : {0 a : Type} -> {0 m : Nat} -> {0 n : Nat} -> Vec a m -> Vec a n -> Vec a (add m n)
fn append(xs, ys) {
    match xs {
        Nil        => ys,
        Cons(h, t) => Cons(h, append(t, ys)),
    }
}
"#;

#[test]
fn dependent_vectors_with_implicits() {
    // a vector literal built with implicit `a` and `k`, checked against its type
    let src = format!(
        "{VEC}\nmain : Vec Nat (Succ (Succ Zero))\n\
         fn main() {{ Cons(Succ(Zero), Cons(Zero, Nil)) }}\n"
    );
    let prog = check_program(&src).unwrap_or_else(|e| panic!("{e:?}"));
    let nil = Term::Constr("Nil".into(), vec![ndata()]);
    let cons = |k: Term, h: Term, t: Term| Term::Constr("Cons".into(), vec![ndata(), k, h, t]);
    // [1, 0] : Vec Nat 2   (k indices: 1 then 0)
    let expected = cons(succ(zero()), succ(zero()), cons(zero(), zero(), nil));
    assert_eq!(prog.normalize("main"), Some(expected));
}

#[test]
fn append_with_implicits_type_checks() {
    // the headline: `Cons(h, append(t, ys))` with no element type or length args
    assert!(check_program(VEC).is_ok(), "{:?}", check_program(VEC).err());
}

const FIN: &str = r#"
enum Nat {
    Zero : Nat,
    Succ : Nat -> Nat,
}

boxed enum Fin : Nat -> Type {
    FZ : {0 k : Nat} -> Fin (Succ k),
    FS : {0 k : Nat} -> Fin k -> Fin (Succ k),
}

fin_to_nat : {0 n : Nat} -> Fin n -> Nat
fn fin_to_nat(i) {
    match i {
        FZ       => Zero,
        FS(prev) => Succ(fin_to_nat(prev)),
    }
}
"#;

#[test]
fn fin_indexed_implicits() {
    // the `fin_to_nat` definition type-checks with FZ / FS(prev) patterns — the
    // index `k` is written nowhere (implicit in both constructors).
    assert!(check_program(FIN).is_ok(), "{:?}", check_program(FIN).err());

    // building a Fin element: the indices of FS/FZ are inferred from the
    // declared type `Fin 2`.  FS(FZ) : Fin 2  ↝  FS (Succ Zero) (FZ Zero)
    let src = format!("{FIN}\nmain : Fin (Succ (Succ Zero))\nfn main() {{ FS(FZ) }}\n");
    let prog = check_program(&src).unwrap_or_else(|e| panic!("{e:?}"));
    let expected = Term::Constr(
        "FS".into(),
        vec![succ(zero()), Term::Constr("FZ".into(), vec![zero()])],
    );
    assert_eq!(prog.normalize("main"), Some(expected));
}

#[test]
fn function_call_implicits_are_solved_from_argument_types() {
    // `append(xs, ys)` — append's implicit a, m, n are solved from the (def)
    // argument types; no type or length is written at the call site.
    let src = format!(
        "{VEC}\n\
         xs : Vec Nat (Succ Zero)\nfn xs() {{ Cons(Succ(Zero), Nil) }}\n\
         ys : Vec Nat (Succ Zero)\nfn ys() {{ Cons(Zero, Nil) }}\n\
         main : Vec Nat (Succ (Succ Zero))\nfn main() {{ append(xs, ys) }}\n"
    );
    let prog = check_program(&src).unwrap_or_else(|e| panic!("{e:?}"));
    let nil = Term::Constr("Nil".into(), vec![ndata()]);
    let cons = |k: Term, h: Term, t: Term| Term::Constr("Cons".into(), vec![ndata(), k, h, t]);
    let expected = cons(succ(zero()), succ(zero()), cons(zero(), zero(), nil)); // [1, 0]
    assert_eq!(prog.normalize("main"), Some(expected));
}

#[test]
fn fin_to_nat_called_with_inferred_implicit() {
    // fin_to_nat(one) — the implicit `n` is solved from `one`'s type (Fin 2)
    let src = format!(
        "{FIN}\n\
         one : Fin (Succ (Succ Zero))\nfn one() {{ FS(FZ) }}\n\
         main : Nat\nfn main() {{ fin_to_nat(one) }}\n"
    );
    let prog = check_program(&src).unwrap_or_else(|e| panic!("{e:?}"));
    assert_eq!(prog.normalize("main"), Some(num(1)));
}

#[test]
fn phase_1b_inductive_lt_constructs_explicit_and_implicit() {
    // PHASE 1b (Acc step 1): an INDUCTIVE `Lt` whose proofs can be constructed and
    // ELIMINATED is the natWf prerequisite (a postulated `Lt` can't be analyzed).
    // EXPLICIT indices: `ltS(0,1,ltZ(0))` builds `Lt 1 2`.
    let explicit = "%builtin Nat Nat\nenum Nat { Zero : Nat, Succ : Nat -> Nat }\n\
        boxed enum Lt : Nat -> Nat -> Type {\n\
          ltZ : (n : Nat) -> Lt Zero (Succ n),\n\
          ltS : (m : Nat) -> (n : Nat) -> Lt m n -> Lt (Succ m) (Succ n),\n\
        }\n\
        p12 : Lt (Succ Zero) (Succ (Succ Zero))\nfn p12() { ltS(Zero, Succ(Zero), ltZ(Zero)) }\n";
    assert!(check_program(explicit).is_ok(), "explicit Lt construction must work: {:?}", check_program(explicit).err());

    // gap (i) NOW FIXED — IMPLICIT inference solves a relation constructor's implicits
    // from the EXPECTED type's index spine (reconciling the packed `VNatLit` literal
    // with the `Succ`-spine in `solve`). `ltZ : {0 n} -> Lt Zero (Succ n)` against
    // `Lt Zero (Succ Zero)` solves `n = Zero`; `ltS(ltZ) : Lt 1 2` solves both. So
    // proofs read clean (no explicit indices) — natWf no longer needs them everywhere.
    let implicit = "%builtin Nat Nat\nenum Nat { Zero : Nat, Succ : Nat -> Nat }\n\
        boxed enum Lt : Nat -> Nat -> Type {\n\
          ltZ : {0 n : Nat} -> Lt Zero (Succ n),\n\
          ltS : {0 m : Nat} -> {0 n : Nat} -> Lt m n -> Lt (Succ m) (Succ n),\n\
        }\n\
        p01 : Lt Zero (Succ Zero)\nfn p01() { ltZ }\n\
        p12 : Lt (Succ Zero) (Succ (Succ Zero))\nfn p12() { ltS(ltZ) }\n";
    assert!(check_program(implicit).is_ok(), "implicit-from-result-index inference must now work: {:?}", check_program(implicit).err());
}

#[test]
fn phase_1b_value_correctness_guard_rejects_dropped_arg() {
    // VALUE-CORRECTNESS (the 1a′-class subtlety for higher-order / well-founded
    // recursion): the IH for a higher-order recursive field is a function OF the
    // field-application arguments, so a recursive call lowers to `ih(callargs…)` and
    // does NOT thread other arguments. `g(f(btrue), Succ(acc))` would lower to
    // `ih(btrue)`, silently DROPPING `Succ(acc)` ⇒ a well-typed WRONG value the
    // (non-dependent) kernel re-check can't catch (`g(node2(kids), 0)` would compute 0,
    // not 1). The guard REJECTS the mismatch. (For the valid `Acc` shape
    // `f(y, h(y, prf))` the new `y` matches `h`'s `y`, so it is accepted.)
    let src = "%builtin Nat Nat\nenum Nat { Zero : Nat, Succ : Nat -> Nat }\n\
        enum Bool { btrue : Bool, bfalse : Bool }\n\
        boxed enum Tree { leaf : Tree, node2 : (Bool -> Tree) -> Tree }\n\
        g : Tree -> Nat -> Nat\n\
        fn g(t, acc) { match t { leaf => acc, node2(f) => g(f(btrue), Succ(acc)) } }\n";
    let err = match check_program(src) {
        Err(d) => format!("{d:?}"),
        Ok(_) => panic!("a recursive call that drops a varying arg through a higher-order IH must be REJECTED"),
    };
    assert!(
        err.contains("well-founded") && err.contains("silently dropped"),
        "must reject for value-correctness (silently-dropped arg), got: {err}"
    );
}

#[test]
fn phase_1b_acc_indexed_higher_order_family_eliminates() {
    // PHASE 1b (Acc foundation): the well-founded `Acc` family is an INDEXED
    // higher-order recursive family — `accN x` carries an accessibility FUNCTION
    // `(y:Nat) -> Lt y x -> AccN y`. This confirms the 1b foundation handles the
    // indexed higher-order shape (beyond the non-indexed W-type): a monomorphized
    // `AccN : Nat -> Type` over a relation `Lt` declares (strict positivity ✓), and
    // a function eliminating an `Acc` proof type-checks and is certified total.
    //
    // What this does NOT yet do — and is the documented next chunk (PHASE_1B_PLAN.md):
    // a wf-RECURSIVE body needs a proof `Lt y x` to justify the recursive call
    // `f(y, h(y, prf))`, which requires an accessibility lemma `natWf` (and an
    // inductive `Lt` whose proofs can be eliminated). Hence the recursion + the
    // value-correctness guard land with that infrastructure.
    let src = "%builtin Nat Nat\nenum Nat { Zero : Nat, Succ : Nat -> Nat }\n\
               postulate Lt : Nat -> Nat -> Type\n\
               boxed enum AccN : Nat -> Type {\n\
                 accN : (x : Nat) -> ((y : Nat) -> Lt y x -> AccN y) -> AccN x,\n\
               }\n\
               depth : (x : Nat) -> AccN x -> Nat\n\
               fn depth(x, a) { match a { accN(xx, h) => Zero } }\n";
    let prog = check_program(src).unwrap_or_else(|e| panic!("Acc family/elimination must type-check: {e:?}"));
    assert!(is_total(&prog, "depth"), "the Acc eliminator must be certified total");
}

#[test]
fn phase_1b_wtype_higher_order_recursive_fold() {
    // PHASE 1b: a W-type `Tree` with a HIGHER-ORDER recursive field
    // `node2 : (Bool -> Tree) -> Tree`. The surface match-compiler now recognizes
    // the higher-order recursive field, generates the functional induction
    // hypothesis, and certifies a fold over it `%total`. A recursive call
    // `size(f(btrue))` (the field `f` applied to an argument) maps to the IH applied
    // to that argument. Construction passes a NAMED helper (the surface has no
    // lambdas). size(node2(kids)) = 2 (one leaf per Bool child). Kernel evaluator.
    let src = format!(
        "{NATB}\n\
         enum Bool {{ btrue : Bool, bfalse : Bool }}\n\
         boxed enum Tree {{ leaf : Tree, node2 : (Bool -> Tree) -> Tree }}\n\
         add : Nat -> Nat -> Nat\n\
         fn add(m, n) {{ match m {{ Zero => n, Succ(k) => Succ(add(k, n)) }} }}\n\
         kids : Bool -> Tree\nfn kids(b) {{ leaf }}\n\
         t1 : Tree\nfn t1() {{ node2(kids) }}\n\
         size : Tree -> Nat\n\
         fn size(t) {{ match t {{ leaf => Succ(Zero), node2(f) => add(size(f(btrue)), size(f(bfalse))) }} }}\n\
         main : Nat\nfn main() {{ size(t1) }}\n"
    );
    let prog = check_program(&src).unwrap_or_else(|e| panic!("{e:?}"));
    assert!(is_total(&prog, "size"), "higher-order recursive fold must be certified total");
    assert_eq!(prog.normalize("main"), Some(Term::NatLit(2)), "size(node2(kids)) must run to 2");
}

#[test]
fn let_linearity_rejects_double_free_and_leak() {
    // SOUNDNESS FIX (memory-model §5 finding): a `let`-bound LINEAR value binds at 1,
    // so the rig catches misuse. Before the fix the `let` ω-binder LAUNDERED linearity
    // and both of these were wrongly ACCEPTED (a live double-free / leak).
    const NB: &str = "%builtin Nat Nat\nenum Nat { Zero : Nat, Succ : Nat -> Nat }\n";

    // double-free: the let-bound Own `o` is consumed twice ⇒ ω ⋢ 1.
    let dbl = format!(
        "{NB}\ndbl : Unit\nfn dbl() {{ let o = alloc(Zero); let u = free(o); free(o) }}\n\
         main : Unit\nfn main() {{ dbl }}\n"
    );
    let err = match check_program(&dbl) {
        Err(d) => format!("{d:?}"),
        Ok(_) => panic!("double-free through `let` must be REJECTED"),
    };
    assert!(err.contains("⋢") || err.contains("multiplicity"), "must be a linearity error, got: {err}");

    // leak: the let-bound Own is never consumed ⇒ 0 ⋢ 1.
    let leak = format!(
        "{NB}\nlk : Unit\nfn lk() {{ let o = alloc(Zero); U }}\n\
         main : Unit\nfn main() {{ lk }}\n"
    );
    assert!(check_program(&leak).is_err(), "dropping a let-bound linear Own must be REJECTED (leak)");

    // single use: the linear value is consumed exactly once ⇒ ACCEPTED.
    let ok = format!(
        "{NB}\none : Unit\nfn one() {{ let o = alloc(Zero); free(o) }}\n\
         main : Unit\nfn main() {{ one }}\n"
    );
    assert!(check_program(&ok).is_ok(), "single free must be ACCEPTED: {:?}", check_program(&ok).err());
}

#[test]
fn linear_param_defaults_to_one_no_double_free() {
    // SOUNDNESS (found by RED-TEAMING the let-fix boundary, not asserted): a function
    // PARAMETER of a linear type defaults to multiplicity 1, not ω — otherwise
    // `fn f(x : Own Nat) { free(x); free(x) }` (param ω, used twice ⇒ ω ≤ ω) is a
    // double-free, ACCEPTED. Same fail-toward-linearity rule as the `let` binder; an
    // explicit `(1 x : …)` was already 1, and an abstract `{0 a}`-typed param stays ω
    // (the §13 polymorphism case — a leak, not a double-free; needs real surface linear
    // params in Phase A).
    const NB: &str = "%builtin Nat Nat\nenum Nat { Zero : Nat, Succ : Nat -> Nat }\n";
    let dbl = format!(
        "{NB}f : Own Nat -> Unit\nfn f(x) {{ let u = free(x); free(x) }}\n\
         main : Unit\nfn main() {{ U }}\n"
    );
    let err = match check_program(&dbl) {
        Err(d) => format!("{d:?}"),
        Ok(_) => panic!("a linear (Own) parameter freed twice must be REJECTED"),
    };
    assert!(err.contains("⋢") || err.contains("multiplicity"), "must be a linearity error, got: {err}");
    // single free still accepted (no over-rejection).
    let one = format!("{NB}f : Own Nat -> Unit\nfn f(x) {{ free(x) }}\nmain : Unit\nfn main() {{ U }}\n");
    assert!(check_program(&one).is_ok(), "single free must be ACCEPTED: {:?}", check_program(&one).err());
}

#[test]
fn phase_a_eliminator_joins_branch_usages() {
    // PHASE A (b): a `match`/eliminator runs exactly ONE arm, so a captured linear
    // value's usage across the arms is their JOIN (lub), not their SUM. Freeing an
    // owned value once-per-arm is now ACCEPTED (lub(1,1)=1, was sum=ω over-rejected);
    // the dual of the CBV-let over-counting fix. Soundness preserved: a per-path leak
    // (freed in one arm only ⇒ lub(0,1)=ω) and a within-arm double-free are REJECTED.
    const P: &str = "%builtin Nat Nat\nenum Nat { Zero : Nat, Succ : Nat -> Nat }\n\
        enum Bool { btrue : Bool, bfalse : Bool }\n";
    // ACCEPT: linear `o` freed once in EVERY arm.
    let ok = format!("{P}g : Own Nat -> Bool -> Unit\nfn g(o, b) {{ match b {{ btrue => free(o), bfalse => free(o) }} }}\nmain : Unit\nfn main() {{ U }}\n");
    assert!(check_program(&ok).is_ok(), "freed once per arm must be ACCEPTED (join): {:?}", check_program(&ok).err());
    // REJECT: freed in one arm only → the other arm leaks it (lub(0,1)=ω).
    let leak = format!("{P}g : Own Nat -> Bool -> Unit\nfn g(o, b) {{ match b {{ btrue => free(o), bfalse => U }} }}\nmain : Unit\nfn main() {{ U }}\n");
    assert!(check_program(&leak).is_err(), "freeing in only one arm leaks on the other path — must be REJECTED");
    // REJECT: double-free WITHIN one arm (the branch's own usage is ω).
    let dbl = format!("{P}g : Own Nat -> Bool -> Unit\nfn g(o, b) {{ match b {{ btrue => let u = free(o); free(o), bfalse => free(o) }} }}\nmain : Unit\nfn main() {{ U }}\n");
    assert!(check_program(&dbl).is_err(), "double-free within an arm must be REJECTED");
}

#[test]
fn differentiator_demo_linked_list_is_type_safe_and_red_teamed() {
    // THE DIFFERENTIATOR DEMO at the TYPE level: an owned linked list built with
    // `alloc`, traversed + freed with `unbox`. The type system enforces that every
    // `Own` is consumed EXACTLY ONCE on EVERY path — so the safe program is ACCEPTED,
    // while a LEAK (an owned binder dropped) and a DOUBLE-FREE (an owned binder used
    // twice) are both REJECTED. (Runs natively to 3 — see dep_codegen's
    // differentiator_demo_owned_linked_list_runs_natively.)
    const HDR: &str = "%builtin Nat Nat\nenum Nat { Zero : Nat, Succ : Nat -> Nat }\n\
        enum Opt (a : Type) { none : Opt a, some : a -> Opt a }\n\
        struct Node { head : Nat, tail : Opt (Own Node) }\n\
        add : Nat -> Nat -> Nat\nfn add(m, n) { match m { Zero => n, Succ(k) => Succ(add(k, n)) } }\n\
        freeNode : (1 o : Own Node) -> Unit\n\
        %partial\n\
        fn freeNode(o) { match unbox(o) { Node(h, t) => match t { none => U, some(o2) => freeNode(o2) } } }\n\
        main : Nat\n";
    // SAFE: every Own consumed once (live path unboxes both; dead arms free their owns).
    let safe = format!("{HDR}fn main() {{ match unbox(alloc(Node(Succ(Zero), some(alloc(Node(Succ(Succ(Zero)), none)))))) {{ \
        Node(h1, t1) => match t1 {{ none => Zero, \
          some(o2) => match unbox(o2) {{ Node(h2, t2) => match t2 {{ \
            none => add(h1, h2), some(o3) => match freeNode(o3) {{ U => Zero }} }} }} }} }} }}\n");
    assert!(check_program(&safe).is_ok(), "the safe owned-list demo must type-check: {:?}", check_program(&safe).err());
    // LEAK: the dead `some(o3)` arm DROPS o3 (an Own) instead of freeing it ⇒ 0⋢1.
    let leak = format!("{HDR}fn main() {{ match unbox(alloc(Node(Succ(Zero), some(alloc(Node(Succ(Succ(Zero)), none)))))) {{ \
        Node(h1, t1) => match t1 {{ none => Zero, \
          some(o2) => match unbox(o2) {{ Node(h2, t2) => match t2 {{ \
            none => add(h1, h2), some(o3) => Zero }} }} }} }} }}\n");
    assert!(check_program(&leak).is_err(), "dropping an owned binder (leak) must be REJECTED");
    // DOUBLE-FREE: o2 is consumed by unbox AND again by free ⇒ ω⋢1.
    let dbl = format!("{HDR}fn main() {{ match unbox(alloc(Node(Succ(Zero), some(alloc(Node(Succ(Succ(Zero)), none)))))) {{ \
        Node(h1, t1) => match t1 {{ none => Zero, \
          some(o2) => match unbox(o2) {{ Node(h2, t2) => match t2 {{ \
            none => match free(o2) {{ U => add(h1, h2) }}, some(o3) => match freeNode(o3) {{ U => Zero }} }} }} }} }} }}\n");
    assert!(check_program(&dbl).is_err(), "using an owned binder twice (double-free) must be REJECTED");
}

#[test]
fn phase_a_positivity_probe_is_unforgeable() {
    // DEFENSE-IN-DEPTH (soundness-by-construction): the variance check's probe datatype
    // is the UN-LEXABLE name "<positivity probe>" (contains spaces), so no user/library
    // datatype can ever collide with it. A user even declaring `__sp_probe__` (the old
    // convention-only sentinel) has NO effect on the variance check: recursive Own still
    // accepts, and a contravariant nesting is still rejected.
    const P: &str = "%builtin Nat Nat\nenum Nat { Zero : Nat, Succ : Nat -> Nat }\n\
        enum Opt (a : Type) { none : Opt a, some : a -> Opt a }\n\
        enum __sp_probe__ { mk : __sp_probe__ }\n";
    let ok = format!("{P}struct Node {{ next : Opt (Own Node) }}\nmain : Unit\nfn main() {{ U }}\n");
    assert!(check_program(&ok).is_ok(), "a user `__sp_probe__` must not break recursive-Own: {:?}", check_program(&ok).err());
    let bad = format!("{P}enum Weird (a:Type) {{ w : (a -> Nat) -> Weird a }}\nstruct Node {{ x : Weird Node }}\nmain : Unit\nfn main() {{ U }}\n");
    assert!(check_program(&bad).is_err(), "a user `__sp_probe__` must not let contravariant nesting through");
}

#[test]
fn phase_a_variance_aware_nested_positivity() {
    // PHASE A (a): strict positivity is now VARIANCE-AWARE + NESTED, with `Own` as a
    // positivity-transparent pointer wrapper — so recursive `Own` structures (linked
    // lists, trees via `Opt (Own T)`) compile, while every negative/contravariant
    // occurrence is still REJECTED (no Curry/non-termination). A recursive occurrence
    // may nest through datatype D's argument iff D is COVARIANT in that parameter; a
    // pointer `Own T` recurses into T PRESERVING polarity (so `Own (A→B)` keeps A
    // negative).
    const P: &str = "%builtin Nat Nat\nenum Nat { Zero : Nat, Succ : Nat -> Nat }\n\
        enum Opt (a : Type) { none : Opt a, some : a -> Opt a }\n";
    let ok = |s: &str| assert!(check_program(s).is_ok(), "must be ACCEPTED: {:?}", check_program(s).err());
    let no = |s: &str| assert!(check_program(s).is_err(), "must be REJECTED (positivity): {s}");

    // ACCEPT — covariant nesting / pointer recursion (the memory model's shapes):
    ok(&format!("{P}struct Node {{ next : Own Node }}\nmain : Unit\nfn main() {{ U }}\n"));
    ok(&format!("{P}struct Node {{ next : Opt (Own Node) }}\nmain : Unit\nfn main() {{ U }}\n"));
    ok(&format!("{P}struct Tree {{ l : Opt (Own Tree), v : Nat, r : Opt (Own Tree) }}\nmain : Unit\nfn main() {{ U }}\n"));

    // `Opt Node` WITHOUT indirection is positivity-fine but LAYOUT-rejected
    // under zero-implicit-allocation: as a VALUE it would be infinitely sized
    // (it only ever "worked" because cells were allocated behind your back).
    let err = check_program(&format!(
        "{P}struct Node {{ next : Opt Node }}\nmain : Unit\nfn main() {{ U }}\n"
    ))
    .err()
    .expect("value-recursion without indirection must be layout-rejected");
    assert!(
        format!("{err:?}").contains("RECURSIVE without indirection"),
        "expected the layout guidance, got {err:?}"
    );

    // REJECT — negative / contravariant occurrences (soundness):
    no(&format!("{P}struct Node {{ f : Own (Node -> Nat) }}\nmain : Unit\nfn main() {{ U }}\n"));      // negative inside Own
    no(&format!("{P}struct Node {{ f : (Node -> Nat) -> Nat }}\nmain : Unit\nfn main() {{ U }}\n"));   // direct negative (unchanged)
    no(&format!("{P}enum Weird (a:Type) {{ mk : (a -> Nat) -> Weird a }}\nstruct Node {{ w : Weird Node }}\nmain : Unit\nfn main() {{ U }}\n")); // contravariant param
    // mutual-recursion cycle whose contravariance is only visible THROUGH the cycle:
    no(&format!("{P}enum D2 (a:Type) {{ d2 : (a -> Nat) -> D2 a }}\nenum D1 (a:Type) {{ d1 : D2 a -> D1 a }}\nstruct Node {{ n : D1 Node }}\nmain : Unit\nfn main() {{ U }}\n"));
}

#[test]
fn phase_a_use_site_linearity_closes_the_whole_double_free_class() {
    // PHASE A gate 2 — the CONVERGENT, whack-a-mole-proof fix (replaces the per-
    // hiding-spot forbids). Linearity is checked at the USE SITE on the field's ACTUAL
    // (instantiated, field-aware) type: at a `match`, each field whose instantiated type
    // `is_linear` is re-bound at 1, so the kernel enforces exactly-once use through it —
    // catching a hidden `Own` HOWEVER it is hidden. Struct/enum `Own` fields are now
    // ALLOWED to declare (the forbid is lifted); misuse is caught where the value is USED.
    // Each instance: double-free REJECTED, leak REJECTED, single-use ACCEPTED.
    const P: &str = "%builtin Nat Nat\nenum Nat { Zero : Nat, Succ : Nat -> Nat }\n\
        enum Opt (a : Type) { none : Opt a, some : a -> Opt a }\n\
        enum Pair (a : Type) (b : Type) { MkPair : a -> b -> Pair a b }\n\
        struct Box { p : Own Nat }\n";
    let dbl = |s: &str| assert!(check_program(s).is_err(), "double-free/leak must be REJECTED: {s}");
    let ok = |s: &str| assert!(check_program(s).is_ok(), "single-use must be ACCEPTED: {:?}", check_program(s).err());

    // 1. MONOMORPHIC struct field (Own hidden behind the type name — `struct Box` now
    //    DECLARES; the forbid is gone).
    ok(&format!("{P}mk : Box\nfn mk() {{ Box(alloc(Zero)) }}\nf : Unit\nfn f() {{ let b = mk; match b {{ Box(q) => free(q) }} }}\nmain : Unit\nfn main() {{ f }}\n"));
    dbl(&format!("{P}mk : Box\nfn mk() {{ Box(alloc(Zero)) }}\nf : Unit\nfn f() {{ let b = mk; match b {{ Box(q) => let u = free(q); free(q) }} }}\nmain : Unit\nfn main() {{ f }}\n"));
    dbl(&format!("{P}mk : Box\nfn mk() {{ Box(alloc(Zero)) }}\nf : Unit\nfn f() {{ let b = mk; match b {{ Box(q) => U }} }}\nmain : Unit\nfn main() {{ f }}\n")); // leak

    // 2. PARAMETRIC container instantiated at Own (the reviewer's `Pair (Own Nat)` case).
    ok(&format!("{P}mk : Pair (Own Nat) Unit\nfn mk() {{ MkPair(alloc(Zero), U) }}\nf : Unit\nfn f() {{ let p = mk; match p {{ MkPair(x, y) => free(x) }} }}\nmain : Unit\nfn main() {{ f }}\n"));
    dbl(&format!("{P}mk : Pair (Own Nat) Unit\nfn mk() {{ MkPair(alloc(Zero), U) }}\nf : Unit\nfn f() {{ let p = mk; match p {{ MkPair(x, y) => let u = free(x); free(x) }} }}\nmain : Unit\nfn main() {{ f }}\n"));

    // 3. Opt(Own) via the eliminator (the 4th instance I found by red-teaming).
    ok(&format!("{P}mk : Opt (Own Nat)\nfn mk() {{ some(alloc(Zero)) }}\nf : Unit\nfn f() {{ let m = mk; match m {{ none => U, some(o) => free(o) }} }}\nmain : Unit\nfn main() {{ f }}\n"));
    dbl(&format!("{P}mk : Opt (Own Nat)\nfn mk() {{ some(alloc(Zero)) }}\nf : Unit\nfn f() {{ let m = mk; match m {{ none => U, some(o) => let u = free(o); free(o) }} }}\nmain : Unit\nfn main() {{ f }}\n"));
    dbl(&format!("{P}mk : Opt (Own Nat)\nfn mk() {{ some(alloc(Zero)) }}\nf : Unit\nfn f() {{ let m = mk; match m {{ none => U, some(o) => U }} }}\nmain : Unit\nfn main() {{ f }}\n")); // leak (some-arm drops o)

    // 4. DEEPLY-NESTED generic: `Pair (Pair (Own Nat) Unit) Unit` — the inner linear
    //    pair, once bound, can't be used twice.
    dbl(&format!("{P}mk : Pair (Pair (Own Nat) Unit) Unit\nfn mk() {{ MkPair(MkPair(alloc(Zero), U), U) }}\nf : Unit\nfn f() {{ let p = mk; match p {{ MkPair(inner, y) => let u = inner; inner }} }}\nmain : Unit\nfn main() {{ f }}\n"));

    // 5. NO OVER-REJECTION: a non-linear struct/enum still declares and is freely usable.
    ok(&format!("{P}struct PP {{ a : Nat, b : Nat }}\nmain : Unit\nfn main() {{ U }}\n"));
}

#[test]
fn cbv_let_sequences_linear_ops_without_over_counting() {
    // NIT 1 FIXED via the CALL-BY-VALUE `let` (replaces the β-redex `(λx.body) e`,
    // which SCALED `e`'s usage by the binder mult — over-counting the linear resources
    // an effectful `e` consumes). The CBV let counts `e` exactly ONCE, so SEQUENCING
    // effectful linear-consuming ops works: you can free TWO owned values. Soundness is
    // preserved — double-free and leak are still rejected.
    const NB: &str = "%builtin Nat Nat\nenum Nat { Zero : Nat, Succ : Nat -> Nat }\n";
    // two owners, each freed exactly once via let-sequencing → ACCEPTED (was REJECTED).
    let two = format!(
        "{NB}g : Own Nat -> Own Nat -> Unit\nfn g(x, y) {{ let u = free(x); free(y) }}\n\
         main : Unit\nfn main() {{ U }}\n"
    );
    assert!(check_program(&two).is_ok(), "freeing two owners (sequenced) must be ACCEPTED: {:?}", check_program(&two).err());
    // double-free of the SAME owner via let → still REJECTED (ω ⋢ 1).
    let dbl = format!("{NB}g : Own Nat -> Unit\nfn g(x) {{ let u = free(x); free(x) }}\nmain : Unit\nfn main() {{ U }}\n");
    assert!(check_program(&dbl).is_err(), "double-free via let must still be REJECTED");
    // dropping an owner → still REJECTED (0 ⋢ 1, leak).
    let leak = format!("{NB}g : Own Nat -> Unit\nfn g(x) {{ U }}\nmain : Unit\nfn main() {{ U }}\n");
    assert!(check_program(&leak).is_err(), "dropping an owner must still be REJECTED (leak)");
}

#[test]
fn let_copyable_value_still_usable_many_times() {
    // NO REGRESSION: a COPYABLE let-bound value (a `Nat` — no linear component) still
    // binds at ω and may be used multiple times. `let n = 2; add(n, n) = 4`.
    let src = format!(
        "{NATB}\nadd : Nat -> Nat -> Nat\n\
         fn add(m, n) {{ match m {{ Zero => n, Succ(k) => Succ(add(k, n)) }} }}\n\
         main : Nat\nfn main() {{ let n = Succ(Succ(Zero)); add(n, n) }}\n"
    );
    let prog = check_program(&src).unwrap_or_else(|e| panic!("{e:?}"));
    assert_eq!(prog.normalize("main"), Some(Term::NatLit(4)), "let n=2; add(n,n) must run to 4");
}

#[test]
fn structs_with_implicit_param() {
    let src = r#"
enum Nat { Zero : Nat, Succ : Nat -> Nat }
struct Box (a : Type) { val : a }
mk : Box Nat
fn mk() { Box(Succ(Zero)) }
"#;
    let prog = check_program(src).unwrap_or_else(|e| panic!("{e:?}"));
    assert_eq!(
        prog.normalize("mk"),
        Some(Term::Constr("Box".into(), vec![ndata(), succ(zero())]))
    );
}

#[test]
fn non_exhaustive_match_is_rejected() {
    let src = format!(
        "{}\nbad : Nat -> Nat\nfn bad(m) {{ match m {{ Zero => Zero }} }}\n",
        r#"enum Nat { Zero : Nat, Succ : Nat -> Nat }"#
    );
    assert!(check_program(&src).is_err());
}

#[test]
fn void_and_exfalso_are_the_for_all_t_sentinel() {
    // The for-all-T sentinel foundation for the absurd-case discharge: a genuinely
    // uninhabited `enum Void {}` (no constructors) + `exfalso : {0 a} -> Void -> a` (the
    // empty match on Void). Type-checks out of the box — the empty match on a no-constructor
    // type IS the all-absurd discharge. `exfalso` yields ANY result type T (T-INDEPENDENT —
    // no Nat/Unit sentinel backstop hole), which is what makes a PARTIAL absurd discharge
    // (real cases beside impossible ones) sound: an impossible case discharges via
    // `exfalso(<genuine contradiction>)`, type-checking ONLY if the contradiction is real.
    let src = "%builtin Nat Nat\nenum Nat { Zero : Nat, Succ : Nat -> Nat }\n\
        enum Void { }\n\
        exfalso : {0 a : Type} -> Void -> a\nfn exfalso(v) { match v { } }\n\
        main : Nat\nfn main() { Zero }\n";
    assert!(check_program(src).is_ok(), "Void + exfalso (the for-all-T sentinel) must type-check: {:?}", check_program(src).err());
}

#[test]
fn dependent_ast_makes_out_of_scope_variables_impossible_by_typing() {
    // THE DEPENDENT HALF on the dogfood (the "powers of Idris" cardinal): index the AST by
    // SCOPE DEPTH so an out-of-scope variable is IMPOSSIBLE BY TYPING — REJECTED AT
    // TYPE-CHECK, not at runtime. `var : Fin d -> Expr d` makes a variable a BOUNDED index
    // into the scope: in a scope of 1 (`Expr (Succ Zero)`), `var FZ` (index 0) type-checks,
    // but `var (FS FZ)` (index 1) does NOT — `FS FZ : Fin (Succ (Succ _))` cannot be
    // `Fin (Succ Zero)`. A non-dependent language must catch the out-of-scope var at
    // RUNTIME; the dependent type ELIMINATES it by construction. (The `Own (Expr d)`
    // children declare via the (a)-positivity — `Own` of the indexed recursive family.)
    const HDR: &str = "%builtin Nat Nat\nenum Nat { Zero : Nat, Succ : Nat -> Nat }\n\
        boxed enum Fin : Nat -> Type { FZ : {0 n : Nat} -> Fin (Succ n), FS : {0 n : Nat} -> Fin n -> Fin (Succ n) }\n\
        boxed enum Expr : Nat -> Type { lit : {0 d : Nat} -> Nat -> Expr d, var : {0 d : Nat} -> Fin d -> Expr d, add : {0 d : Nat} -> Own (Expr d) -> Own (Expr d) -> Expr d }\n\
        main : Nat\nfn main() { Zero }\n";
    // IN scope (depth 1, variable 0) — type-checks.
    let in_scope = format!("{HDR}e0 : Expr (Succ Zero)\nfn e0() {{ var(FZ) }}\n");
    assert!(check_program(&in_scope).is_ok(), "an in-scope variable must type-check: {:?}", check_program(&in_scope).err());
    // OUT of scope (depth 1, variable 1) — REJECTED at type-check (impossible by typing).
    let out_scope = format!("{HDR}eb : Expr (Succ Zero)\nfn eb() {{ var(FS(FZ)) }}\n");
    assert!(check_program(&out_scope).is_err(), "an OUT-OF-SCOPE variable must be REJECTED at type-check");
}

#[test]
fn partial_heap_recursion_still_enforces_linearity() {
    // CARDINAL for (A): `%partial` relaxes TERMINATION, NOT LINEARITY. A heap-recursive
    // `%partial` `Fix` body is STILL fully linearity-checked — a leak or double-free is
    // REJECTED. `%partial` is "may diverge", never "may leak / double-free".
    const H: &str = "%builtin Nat Nat\nenum Nat { Zero : Nat, Succ : Nat -> Nat }\n\
        enum Opt (a : Type) { none : Opt a, some : a -> Opt a }\n\
        struct Node { head : Nat, tail : Opt (Own Node) }\n\
        add : Nat -> Nat -> Nat\nfn add(m, n) { match m { Zero => n, Succ(k) => Succ(add(k, n)) } }\n\
        main : Nat\nfn main() { Zero }\n";
    // OK: the correct traversal (recursive result let-sequenced) is accepted.
    let ok = format!("{H}sumFree : Opt (Own Node) -> Nat\nfn sumFree(l) {{ match l {{ none => Zero, some(o) => match unbox(o) {{ Node(h, t) => let s = sumFree(t); add(h, s) }} }} }}\n");
    assert!(check_program(&ok).is_ok(), "{:?}", check_program(&ok).err());
    // LEAK: the some-arm drops the tail `t` instead of recursing/freeing ⇒ 0⋢1.
    let leak = format!("{H}sumFree : Opt (Own Node) -> Nat\nfn sumFree(l) {{ match l {{ none => Zero, some(o) => match unbox(o) {{ Node(h, t) => h }} }} }}\n");
    assert!(check_program(&leak).is_err(), "a %partial heap-recursive fn that LEAKS must be REJECTED");
    // DOUBLE-FREE: `o` used twice (unbox AND free) ⇒ ω⋢1.
    let dbl = format!("{H}sumFree : Opt (Own Node) -> Nat\nfn sumFree(l) {{ match l {{ none => Zero, some(o) => match unbox(o) {{ Node(h, t) => let s = sumFree(t); match free(o) {{ U => add(h, s) }} }} }} }}\n");
    assert!(check_program(&dbl).is_err(), "a %partial heap-recursive fn that DOUBLE-FREES must be REJECTED");
}

#[test]
fn non_structural_recursion_lowers_to_partial_fix() {
    // A non-structural recursion (`loop(Succ(k))` — the measure INCREASES) is NO LONGER
    // rejected: with general/heap recursion now supported (a `%partial` `Fix` over a
    // boxed/heap scrutinee, dispatched by `Term::Case`), an UNANNOTATED fn lowers to an
    // opaque `Fix`. `%partial` relaxes TERMINATION (it may diverge — fine), NOT soundness:
    // the kernel treats the `Fix` opaquely so it never reduces during type-checking, and
    // it is reported PARTIAL (not total). A `%total` annotation on it is STILL a hard
    // error (see the `%total`-on-`loop` test in the totality section). Previously this was
    // rejected only because general recursion on a non-`%builtin Nat` scrutinee had no
    // lowering — that limitation is gone.
    let src = format!(
        "{}\nloop : Nat -> Nat\nfn loop(m) {{ match m {{ Zero => Zero, Succ(k) => loop(Succ(k)) }} }}\nmain : Nat\nfn main() {{ Zero }}\n",
        r#"enum Nat { Zero : Nat, Succ : Nat -> Nat }"#
    );
    let prog = check_program(&src).expect("non-structural recursion now lowers as %partial");
    assert!(!is_total(&prog, "loop"), "loop must be reported PARTIAL, not total");
}

// ---- Phase 3: the memory layer as postulates in the dependent+linear core ----

const MEM: &str = r#"
enum Unit { U : Unit }
enum Nat  { Zero : Nat, Succ : Nat -> Nat }
enum Pair (a : Type) (b : Type) { MkPair : a -> b -> Pair a b }

postulate Own   : Type -> Type
postulate alloc : {0 a : Type} -> a -> Own a
postulate free  : {0 a : Type} -> (1 o : Own a) -> Unit
"#;

#[test]
fn linear_memory_capability_accepts_alloc_then_free() {
    // alloc a cell, then free it once — the Own capability is consumed exactly once
    let src = format!("{MEM}\nroundtrip : Unit\nfn roundtrip() {{ free(alloc(Zero)) }}\n");
    assert!(check_program(&src).is_ok(), "{:?}", check_program(&src).err());

    // a function that consumes its owned argument exactly once
    let src2 = format!("{MEM}\nuse_once : (1 o : Own Nat) -> Unit\nfn use_once(o) {{ free(o) }}\n");
    assert!(check_program(&src2).is_ok(), "{:?}", check_program(&src2).err());
}

#[test]
fn double_free_is_a_linearity_error() {
    // using the owned `o` twice violates linearity (ω ⋢ 1) — rejected by the kernel
    let src = format!(
        "{MEM}\ndbl : (1 o : Own Nat) -> Pair Unit Unit\nfn dbl(o) {{ MkPair(free(o), free(o)) }}\n"
    );
    assert!(check_program(&src).is_err());
}

#[test]
fn leaking_an_owned_value_is_a_linearity_error() {
    // dropping `o` (using it 0 times) violates linearity (0 ⋢ 1) — rejected
    let src = format!("{MEM}\nleak : (1 o : Own Nat) -> Unit\nfn leak(o) {{ U }}\n");
    assert!(check_program(&src).is_err());
}

// ---- Phase 3b: capabilities INDEXED BY PROPOSITIONS (a proof gates the op) ----

const PROOFS: &str = r#"
enum Nat { Zero : Nat, Succ : Nat -> Nat }

-- `LT m n` : a proof that m < n
boxed enum LT : Nat -> Nat -> Type {
    LTZ : {0 n : Nat} -> LT Zero (Succ n),
    LTS : {0 m : Nat} -> {0 n : Nat} -> LT m n -> LT (Succ m) (Succ n),
}

postulate Arr : Type -> Nat -> Type
-- reading index `i` REQUIRES a proof that i < n (erased at runtime)
postulate get : {0 a : Type} -> {0 n : Nat} -> {0 i : Nat} -> (0 _ : LT i n) -> Arr a n -> a

-- a proof 1 < 3, built by the dependent core
p13 : LT (Succ Zero) (Succ (Succ (Succ Zero)))
fn p13() { LTS(LTZ) }
"#;

#[test]
fn a_proof_gates_a_memory_read() {
    // read index 1 of a length-3 array — the proof `p13 : LT 1 3` is required
    let src = format!(
        "{PROOFS}\nread1 : {{0 a : Type}} -> Arr a (Succ (Succ (Succ Zero))) -> a\n\
         fn read1(arr) {{ get(p13, arr) }}\n"
    );
    assert!(check_program(&src).is_ok(), "{:?}", check_program(&src).err());
}

#[test]
fn an_impossible_bound_has_no_proof() {
    // there is no proof of 3 < 3, so the out-of-bounds witness cannot be built
    let src = format!(
        "{PROOFS}\np33 : LT (Succ (Succ (Succ Zero))) (Succ (Succ (Succ Zero)))\n\
         fn p33() {{ LTS(LTS(LTS(LTZ))) }}\n"
    );
    assert!(check_program(&src).is_err());
}

// ---- Phase 3c: regions + the intrusive doubly-linked list (O(1) remove) ----

const DLL: &str = r#"
enum Nat  { Zero : Nat, Succ : Nat -> Nat }

postulate Region : Type
postulate List   : Region -> Type
postulate Cursor : Region -> Type

enum CL (r : Region) { MkCL : (1 c : Cursor r) -> (1 l : List r) -> CL r }
enum VL (r : Region) { MkVL : Nat -> (1 l : List r) -> VL r }

postulate insert : {0 r : Region} -> (1 l : List r) -> Nat -> CL r
postulate remove : {0 r : Region} -> (1 c : Cursor r) -> (1 l : List r) -> VL r
postulate free   : {0 r : Region} -> (1 l : List r) -> Unit
enum Unit { U : Unit }
"#;

#[test]
fn intrusive_list_insert_remove_is_accepted() {
    // insert a node, remove it by its cursor (O(1)), return the list
    let src = format!(
        "{DLL}\nclient : {{0 r : Region}} -> (1 l0 : List r) -> List r\n\
         fn client(l0) {{ let (c, l1) = insert(l0, Succ(Zero)); let (v, l2) = remove(c, l1); l2 }}\n"
    );
    assert!(check_program(&src).is_ok(), "{:?}", check_program(&src).err());
}

#[test]
fn double_remove_by_cursor_is_rejected() {
    // using the cursor `c` twice is a use-after-remove (ω ⋢ 1)
    let src = format!(
        "{DLL}\nbad : {{0 r : Region}} -> (1 l0 : List r) -> List r\n\
         fn bad(l0) {{ let (c, l1) = insert(l0, Succ(Zero)); \
         let (v, l2) = remove(c, l1); let (w, l3) = remove(c, l2); l3 }}\n"
    );
    assert!(check_program(&src).is_err());
}

#[test]
fn leaking_the_list_is_rejected() {
    // returning `v` drops the linear list `l2` — a leak (0 ⋢ 1)
    let src = format!(
        "{DLL}\nbad : {{0 r : Region}} -> (1 l0 : List r) -> Nat\n\
         fn bad(l0) {{ let (c, l1) = insert(l0, Succ(Zero)); let (v, l2) = remove(c, l1); v }}\n"
    );
    assert!(check_program(&src).is_err());
}

#[test]
fn cross_region_remove_is_rejected() {
    // a cursor from region `s` cannot remove from a list in region `r`
    let src = format!(
        "{DLL}\ncross : {{0 r : Region}} -> {{0 s : Region}} -> (1 cs : Cursor s) -> (1 lr : List r) -> VL r\n\
         fn cross(cs, lr) {{ remove(cs, lr) }}\n"
    );
    assert!(check_program(&src).is_err());
}

#[test]
fn parse_and_name_errors() {
    assert!(check_program("fn f( {").is_err());
    assert!(check_program("enum Nat { Zero : Nat }\nfn g() { Bogus(Zero) }").is_err());
}

#[test]
fn builtin_nat_pragma_requires_nat_shape() {
    // a `%builtin Nat` on a non-Nat-shaped enum is rejected with a clear error.
    assert!(check_program(
        "%builtin Nat Foo\nenum Foo { A : Foo, B : Foo, C : Foo }\nmain : Foo\nfn main() { A }\n"
    )
    .is_err());
    // naming an undeclared type is rejected.
    assert!(check_program("%builtin Nat Nope\nmain : Nat\nfn main() { 0 }\n").is_err());
}

#[test]
fn builtin_nat_unifies_literals_with_constructors() {
    // with the pragma, `0`/`5` and `Zero`/`Succ` are the SAME packed type.
    let src = "%builtin Nat Nat\nenum Nat { Zero : Nat, Succ : Nat -> Nat }\n\
               add : Nat -> Nat -> Nat\n\
               fn add(m, n) { match m { Zero => n, Succ(k) => Succ(add(k, n)) } }\n\
               main : Nat\nfn main() { add(2, Succ(Succ(Zero))) }\n";
    assert!(check_program(src).is_ok(), "{:?}", check_program(src).err());
}

#[test]
fn implicit_solved_through_nested_constructor_arg() {
    // regression: `alloc(Succ(Zero))` must let the enclosing call infer its erased
    // implicit `a` THROUGH the constructor-application argument. This used to make
    // the elaborator fall into check-mode and underflow when quoting an unsolved
    // hole; now `infer_arg` infers `Succ(Zero) : Nat`, pinning `a = Nat`. (Uses the
    // built-in memory prelude — no `postulate Own/alloc/free` boilerplate.)
    let src = "enum Nat { Zero : Nat, Succ : Nat -> Nat }\n\
               main : Unit\n\
               fn main() { free(alloc(Succ(Zero))) }\n";
    assert!(check_program(src).is_ok(), "should type-check: {:?}", check_program(src).err());
}

// ===========================================================================
// PHASE E1 — totality (termination) checker + `%total` certificates
// ===========================================================================

const NATB: &str = "%builtin Nat Nat\nenum Nat { Zero : Nat, Succ : Nat -> Nat }\n";

/// Is `fnname` reported total in the program's totality status?
fn is_total(prog: &Program, fnname: &str) -> bool {
    prog.totality.iter().any(|(n, t, _)| n == fnname && *t)
}

#[test]
fn total_certificate_rejects_non_terminating() {
    // `%total` on a genuinely non-terminating fn is a HARD ERROR, and for the
    // RIGHT reason (a termination/decrease failure, not something incidental).
    let src = format!(
        "{NATB}\n%total fn loop(m) {{ match m {{ Zero => Zero, Succ(k) => loop(Succ(k)) }} }}\n\
         loop : Nat -> Nat\nmain : Nat\nfn main() {{ Zero }}\n"
    );
    let err = match check_program(&src) { Err(d) => format!("{d:?}"), Ok(_) => panic!("expected rejection") };
    assert!(err.contains("loop") && err.contains("not total"), "got: {err}");
    assert!(
        err.contains("does not decrease") && err.contains("Succ"),
        "must cite the non-decreasing recursive argument, got: {err}"
    );
}

#[test]
fn total_certificate_accepts_structural_fold_and_does_not_dodge() {
    // DUAL-FAILURE GUARD (other direction): a genuinely-total structural fold
    // annotated `%total` PASSES, and is reported `total` — the checker does NOT
    // dodge by making everything partial/Fix.
    let src = format!(
        "{NATB}\n%total fn add(m, n) {{ match m {{ Zero => n, Succ(k) => Succ(add(k, n)) }} }}\n\
         add : Nat -> Nat -> Nat\nmain : Nat\nfn main() {{ add(2, 3) }}\n"
    );
    let prog = check_program(&src).unwrap_or_else(|e| panic!("{e:?}"));
    assert!(is_total(&prog, "add"), "add must be certified total");
}

#[test]
fn phase_1a_prime_accumulator_nat_fold_certified_total_and_computes() {
    // PHASE 1a′: an accumulator-style fold on a `%builtin Nat` (a recursive call
    // that DESCENDS on the scrutinee but VARIES another argument) is now certifiable
    // `%total` — it lowers to a function-typed-motive `NatElim` (the IH is itself a
    // function of the accumulator). `addacc(m, n) = m + n`.
    let src = format!(
        "{NATB}\n%total fn addacc(m, n) {{ match m {{ Zero => n, Succ(k) => addacc(k, Succ(n)) }} }}\n\
         addacc : Nat -> Nat -> Nat\nmain : Nat\nfn main() {{ addacc(2, 3) }}\n"
    );
    let prog = check_program(&src).unwrap_or_else(|e| panic!("{e:?}"));
    assert!(is_total(&prog, "addacc"), "addacc must be certified total (Phase 1a′)");
    assert_eq!(prog.normalize("main"), Some(Term::NatLit(5)), "addacc(2,3) must run to 5");

    // `sumacc(m, acc) = acc + m` — the feasibility fold, now written in surface
    // syntax and certified total. (These assertions check the KERNEL EVALUATOR via
    // `normalize`; the LLVM-backend native run is proven in `dep_codegen` tests.)
    let src = format!(
        "{NATB}\n%total fn sumacc(m, acc) {{ match m {{ Zero => acc, Succ(k) => sumacc(k, Succ(acc)) }} }}\n\
         sumacc : Nat -> Nat -> Nat\nmain : Nat\nfn main() {{ sumacc(3, 0) }}\n"
    );
    let prog = check_program(&src).unwrap_or_else(|e| panic!("{e:?}"));
    assert!(is_total(&prog, "sumacc"), "sumacc must be certified total");
    assert_eq!(prog.normalize("main"), Some(Term::NatLit(3)), "sumacc(3,0) must run to 3");
}

#[test]
fn phase_1a_prime_multi_accumulator_fold() {
    // K=2 accumulators, BOTH varying across the recursive call (the IH is applied
    // to two new accumulators). `twoacc(m, a, b) = match m { 0 => a; Succ k =>
    // twoacc(k, b, Succ a) }`. twoacc(2,0,10) = 1 (see by hand: → (1,10,1) →
    // (0,1,11) → 1).
    let src = format!(
        "{NATB}\n%total fn twoacc(m, a, b) {{ match m {{ Zero => a, Succ(k) => twoacc(k, b, Succ(a)) }} }}\n\
         twoacc : Nat -> Nat -> Nat -> Nat\nmain : Nat\nfn main() {{ twoacc(2, 0, 10) }}\n"
    );
    let prog = check_program(&src).unwrap_or_else(|e| panic!("{e:?}"));
    assert!(is_total(&prog, "twoacc"), "twoacc (K=2) must be certified total");
    assert_eq!(prog.normalize("main"), Some(Term::NatLit(1)), "twoacc(2,0,10) must run to 1");
}

#[test]
fn phase_1a_prime_sub_via_nested_match_accumulator() {
    // truncated subtraction `sub(m, n) = m - n`, scrutinee `m`, accumulator `n`
    // VARYING (n's predecessor). The recursive call sits inside a NESTED match on
    // the accumulator — `rec` (the IH) is threaded through, so it still lowers to
    // the function-typed-motive `NatElim`. sub(5,2) = 3, sub(2,5) = 0.
    let src = format!(
        "{NATB}\n\
         %total fn sub(m, n) {{ match m {{ \
             Zero => Zero, \
             Succ(j) => match n {{ Zero => Succ(j), Succ(k) => sub(j, k) }} \
         }} }}\n\
         sub : Nat -> Nat -> Nat\nmain : Nat\nfn main() {{ sub(5, 2) }}\n"
    );
    let prog = check_program(&src).unwrap_or_else(|e| panic!("{e:?}"));
    assert!(is_total(&prog, "sub"), "sub (nested-match accumulator) must be certified total");
    assert_eq!(prog.normalize("main"), Some(Term::NatLit(3)), "sub(5,2) must run to 3");

    let src2 = format!(
        "{NATB}\n\
         %total fn sub(m, n) {{ match m {{ \
             Zero => Zero, \
             Succ(j) => match n {{ Zero => Succ(j), Succ(k) => sub(j, k) }} \
         }} }}\n\
         sub : Nat -> Nat -> Nat\nmain : Nat\nfn main() {{ sub(2, 5) }}\n"
    );
    let prog2 = check_program(&src2).unwrap_or_else(|e| panic!("{e:?}"));
    assert_eq!(prog2.normalize("main"), Some(Term::NatLit(0)), "sub(2,5) must run to 0 (truncated)");
}

#[test]
fn phase_1a_prime_accumulator_with_non_nat_return_type() {
    // the result type R need not be `Nat`: `lt(m, n) : Bool` returns a BOXED
    // datatype value. The accumulator lowering uses a CLOSED `R` (here `Bool`), so
    // the motive is `λ_. (Nat → Bool)`. lt(2,3) = True, lt(3,3) = False.
    const BOOLNATB: &str = "%builtin Nat Nat\nenum Nat { Zero : Nat, Succ : Nat -> Nat }\n\
                            enum Bool { False : Bool, True : Bool }\n";
    let lt = "%total fn lt(m, n) { match m { \
                 Zero => match n { Zero => False, Succ(x) => True }, \
                 Succ(j) => match n { Zero => False, Succ(k) => lt(j, k) } \
              } }\nlt : Nat -> Nat -> Bool\n";
    let src = format!("{BOOLNATB}\n{lt}main : Bool\nfn main() {{ lt(2, 3) }}\n");
    let prog = check_program(&src).unwrap_or_else(|e| panic!("{e:?}"));
    assert!(is_total(&prog, "lt"), "lt (Bool return) must be certified total");
    assert_eq!(prog.normalize("main"), Some(Term::Constr("True".into(), vec![])), "lt(2,3) = True");

    let src2 = format!("{BOOLNATB}\n{lt}main : Bool\nfn main() {{ lt(3, 3) }}\n");
    let prog2 = check_program(&src2).unwrap_or_else(|e| panic!("{e:?}"));
    assert_eq!(prog2.normalize("main"), Some(Term::Constr("False".into(), vec![])), "lt(3,3) = False");
}

#[test]
fn phase_1a_prime_fuel_div_proof_target() {
    // THE HEADLINE PROOF TARGET of Phase 1a′: `%total fuel-div`, written in 1a
    // surface syntax (nested/expression `match`), composing two accumulator folds
    // (`lt : Nat→Nat→Bool` and `sub : Nat→Nat→Nat`) and a fuel-driven divider
    // (`div`, itself an accumulator fold on `fuel` with `n` varying). All three are
    // certified `%total`. This checks the KERNEL EVALUATOR (`normalize`); the actual
    // LLVM-backend native run div(10,7,2)=3 is `dep_codegen::tests::fuel_div_runs_natively`.
    const PRE: &str = "%builtin Nat Nat\nenum Nat { Zero : Nat, Succ : Nat -> Nat }\n\
                       enum Bool { False : Bool, True : Bool }\n";
    let prog_src = format!(
        "{PRE}\n\
         %total fn lt(m, n) {{ match m {{ \
             Zero => match n {{ Zero => False, Succ(x) => True }}, \
             Succ(j) => match n {{ Zero => False, Succ(k) => lt(j, k) }} \
         }} }}\nlt : Nat -> Nat -> Bool\n\
         %total fn sub(m, n) {{ match m {{ \
             Zero => Zero, \
             Succ(j) => match n {{ Zero => Succ(j), Succ(k) => sub(j, k) }} \
         }} }}\nsub : Nat -> Nat -> Nat\n\
         %total fn div(fuel, n, d) {{ match fuel {{ \
             Zero => Zero, \
             Succ(f) => match lt(n, d) {{ True => Zero, False => Succ(div(f, sub(n, d), d)) }} \
         }} }}\ndiv : Nat -> Nat -> Nat -> Nat\n\
         main : Nat\nfn main() {{ div(10, 7, 2) }}\n"
    );
    let prog = check_program(&prog_src).unwrap_or_else(|e| panic!("{e:?}"));
    assert!(is_total(&prog, "lt"), "lt must be total");
    assert!(is_total(&prog, "sub"), "sub must be total");
    assert!(is_total(&prog, "div"), "fuel-div must be certified total (the 1a′ proof target)");
    assert_eq!(prog.normalize("main"), Some(Term::NatLit(3)), "div(10,7,2) must run to 3");
}

#[test]
fn phase_1a_prime_red_team_non_descending_accumulator_still_rejected() {
    // DUAL-FAILURE GUARD: the accumulator gate loosens ONLY the "other args
    // verbatim" rule — the SCRUTINEE-descent requirement is UNCONDITIONAL. A fold
    // that varies an accumulator but does NOT decrease the scrutinee (here it
    // recurses on `Succ(k)`, GROWING it) is non-terminating and must still be
    // REJECTED — never mis-certified via the accumulator path.
    let src = format!(
        "{NATB}\n%total fn bad(m, a) {{ match m {{ Zero => a, Succ(k) => bad(Succ(k), Succ(a)) }} }}\n\
         bad : Nat -> Nat -> Nat\nmain : Nat\nfn main() {{ bad(1, 0) }}\n"
    );
    let err = match check_program(&src) { Err(d) => format!("{d:?}"), Ok(_) => panic!("expected rejection") };
    assert!(
        err.contains("not total") && err.contains("decrease"),
        "must reject for a scrutinee non-decrease, not accept via the accumulator path, got: {err}"
    );
}

#[test]
fn phase_1a_prime_red_team_boxed_accumulator_still_partial() {
    // the accumulator fold is implemented ONLY for a `%builtin Nat` scrutinee. The
    // SAME accumulator recursion over a BOXED datatype is terminating but not yet
    // lowerable, so `%total` still declines it — with a message that says BOXED and
    // points at the later phase. (No regression: boxed verbatim folds stay total.)
    let src = format!(
        "{NAT}\n%total fn addacc(m, n) {{ match m {{ Zero => n, Succ(k) => addacc(k, Succ(n)) }} }}\n\
         addacc : Nat -> Nat -> Nat\nmain : Nat\nfn main() {{ addacc(2, 3) }}\n"
    );
    let err = match check_program(&src) { Err(d) => format!("{d:?}"), Ok(_) => panic!("expected rejection") };
    assert!(
        err.contains("accumulator") && err.contains("BOXED") && (err.contains("E2") || err.contains("E3")),
        "boxed accumulator must be declined as BOXED + point at the later phase, got: {err}"
    );
}

#[test]
fn phase_1a_prime_verbatim_fold_unchanged_no_regression() {
    // a verbatim-arg structural fold on a `%builtin Nat` still lowers via the plain
    // `NatElim` path (NOT the accumulator path) and stays total + reducible in types.
    // `add(m, n) = match m { 0 => n; Succ k => Succ(add(k, n)) }` (n passed verbatim).
    let src = format!(
        "{NATB}\n%total fn add(m, n) {{ match m {{ Zero => n, Succ(k) => Succ(add(k, n)) }} }}\n\
         add : Nat -> Nat -> Nat\nmain : Nat\nfn main() {{ add(2, 3) }}\n"
    );
    let prog = check_program(&src).unwrap_or_else(|e| panic!("{e:?}"));
    assert!(is_total(&prog, "add"), "verbatim fold must stay total");
    assert_eq!(prog.normalize("main"), Some(Term::NatLit(5)), "add(2,3) must run to 5");
}

#[test]
fn total_certificate_rejects_mutual_recursion() {
    // a mutual-recursion cycle is not yet certifiable as total; `%total` on a
    // member is a hard error citing the cycle (never silently accepted).
    let src = format!(
        "{NATB}\n\
         %total fn iseven(m) {{ match m {{ Zero => Zero, Succ(k) => isodd(k) }} }}\niseven : Nat -> Nat\n\
         fn isodd(m) {{ match m {{ Zero => Succ(Zero), Succ(k) => iseven(k) }} }}\nisodd : Nat -> Nat\n\
         main : Nat\nfn main() {{ iseven(2) }}\n"
    );
    let err = match check_program(&src) { Err(d) => format!("{d:?}"), Ok(_) => panic!("expected rejection") };
    assert!(err.contains("mutual"), "must cite the mutual-recursion cycle, got: {err}");
}

#[test]
fn unannotated_partial_recursion_lowers_to_fix_and_is_reported_partial() {
    // WITHOUT `%total`, non-structural recursion on a `%builtin Nat` is honestly
    // accepted as PARTIAL (an opaque `Fix` the kernel never unfolds) — and the
    // status reports it partial, so the default is not "silently assumed total".
    let src = format!(
        "{NATB}\nfn loop(m) {{ match m {{ Zero => Zero, Succ(k) => loop(Succ(k)) }} }}\n\
         loop : Nat -> Nat\nmain : Nat\nfn main() {{ loop(Zero) }}\n"
    );
    let prog = check_program(&src).unwrap_or_else(|e| panic!("{e:?}"));
    assert!(!is_total(&prog, "loop"), "loop must be reported partial, not total");
}

#[test]
fn calling_a_partial_fn_taints_totality_but_still_compiles() {
    // `main` calls a partial `loop`: main is reported NON-total (partiality is
    // contagious for the certificate), but it still COMPILES — calling a partial
    // helper does not change how a non-recursive fn lowers.
    let src = format!(
        "{NATB}\nfn loop(m) {{ match m {{ Zero => Zero, Succ(k) => loop(Succ(k)) }} }}\n\
         loop : Nat -> Nat\nmain : Nat\nfn main() {{ loop(Zero) }}\n"
    );
    let prog = check_program(&src).unwrap_or_else(|e| panic!("{e:?}"));
    assert!(!is_total(&prog, "main"), "main calls a partial fn ⇒ not total");
    // but a `%total main` here WOULD be rejected:
    let src2 = src.replace("fn main()", "%total fn main()");
    assert!(check_program(&src2).is_err(), "%total main calling a partial fn must be rejected");
}

#[test]
fn structural_fold_is_total_without_annotation() {
    // the default verdict still RECOGNISES totality (status), even un-annotated.
    let src = format!(
        "{NATB}\nfn add(m, n) {{ match m {{ Zero => n, Succ(k) => Succ(add(k, n)) }} }}\n\
         add : Nat -> Nat -> Nat\nmain : Nat\nfn main() {{ add(2, 3) }}\n"
    );
    let prog = check_program(&src).unwrap_or_else(|e| panic!("{e:?}"));
    assert!(is_total(&prog, "add") && is_total(&prog, "main"));
}

#[test]
fn non_nat_enum_reusing_succ_name_is_not_treated_as_nat_descent() {
    // Red-team hardening: a non-Nat enum `Bad` reuses the constructor name `Succ`.
    // `Bad.Succ`'s field is a `Nat`, NOT a sub-structure of `Bad`, so recursing on
    // it does NOT decrease. The structural check must NOT short-circuit on the
    // name-keyed Nat role table; `%total` must reject (and it must never be
    // certified total via the name collision).
    let src = "%builtin Nat Nat\nenum Nat { Zero : Nat, Succ : Nat -> Nat }\n\
               enum Bad { Stop : Bad, Succ : Nat -> Bad }\n\
               %total fn loop(b) { match b { Stop => Zero, Succ(x) => loop(x) } }\n\
               loop : Bad -> Nat\nmain : Nat\nfn main() { Zero }\n";
    // must be rejected — either by the totality gate (preferred) or a hard error,
    // but NEVER accepted as a certified-total non-terminating fn.
    assert!(check_program(src).is_err(), "name-collision Succ must not certify total");
}

// ===========================================================================
// PHASE E2 — coverage / pattern-match hygiene (redundant + unknown arms)
// ===========================================================================

#[test]
fn redundant_duplicate_arm_is_rejected() {
    // boxed datatype: a second arm for the same constructor is rejected.
    let src = "enum Bool { T : Bool, F : Bool }\n\
               neg : Bool -> Bool\nfn neg(b) { match b { T => F, F => T, T => T } }\n\
               main : Bool\nfn main() { neg(T) }\n";
    assert!(check_program(src).is_err(), "duplicate arm must be rejected");
    // %builtin Nat path too:
    let nsrc = "%builtin Nat Nat\nenum Nat { Zero : Nat, Succ : Nat -> Nat }\n\
                f : Nat -> Nat\nfn f(m) { match m { Zero => Zero, Succ(k) => k, Zero => m } }\n\
                main : Nat\nfn main() { f(Zero) }\n";
    assert!(check_program(nsrc).is_err(), "duplicate Nat arm must be rejected");
}

#[test]
fn arm_for_a_non_constructor_is_rejected() {
    let src = "enum Bool { T : Bool, F : Bool }\n\
               neg : Bool -> Bool\nfn neg(b) { match b { T => F, F => T, Bogus => T } }\n\
               main : Bool\nfn main() { neg(T) }\n";
    assert!(check_program(src).is_err(), "unknown-constructor arm must be rejected");
}

#[test]
fn exhaustive_match_still_accepted_after_hygiene() {
    // the hygiene checks must not break a legitimate exhaustive match.
    let src = "enum Bool { T : Bool, F : Bool }\n\
               neg : Bool -> Bool\nfn neg(b) { match b { T => F, F => T } }\n\
               main : Bool\nfn main() { neg(T) }\n";
    assert!(check_program(src).is_ok(), "{:?}", check_program(src).err());
}

// ---- E2: absurd-case discharge (Fin 0 ⇒ zero clauses) ----

const FIN2: &str = "%builtin Nat Nat\nenum Nat { Zero : Nat, Succ : Nat -> Nat }\n\
    enum Bool { T : Bool, F : Bool }\n\
    boxed enum Fin : Nat -> Type {\n\
      FZ : {0 n : Nat} -> Fin (Succ n),\n\
      FS : {0 n : Nat} -> Fin n -> Fin (Succ n),\n\
    }\n";

#[test]
fn absurd_match_on_fin_zero_is_discharged_with_zero_clauses() {
    // `Fin Zero` is empty (both ctors need a `Succ` index) ⇒ a match with NO arms
    // is accepted, for ANY result type. The derived term is kernel-rechecked.
    let src = format!(
        "{FIN2}absurdN : Fin Zero -> Nat\nfn absurdN(x) {{ match x {{ }} }}\n\
         absurdB : Fin Zero -> Bool\nfn absurdB(x) {{ match x {{ }} }}\n\
         main : Nat\nfn main() {{ Zero }}\n"
    );
    assert!(check_program(&src).is_ok(), "{:?}", check_program(&src).err());
}

#[test]
fn an_arm_on_an_absurd_match_is_rejected() {
    // you cannot WRITE a clause for an impossible constructor.
    let src = format!(
        "{FIN2}bad : Fin Zero -> Nat\nfn bad(x) {{ match x {{ FZ => Zero }} }}\n\
         main : Nat\nfn main() {{ Zero }}\n"
    );
    let err = match check_program(&src) { Err(d) => format!("{d:?}"), Ok(_) => panic!("expected rejection") };
    assert!(err.contains("absurd") || err.contains("NO arms"), "got: {err}");
}

#[test]
fn missing_reachable_case_still_rejected_with_variable_index() {
    // BOTH-DIRECTIONS guard: a reachable constructor may NOT be dropped — matching
    // `Fin n` (n a variable) must still cover both FZ and FS.
    let src = format!(
        "{FIN2}f2n : {{0 n : Nat}} -> Fin n -> Nat\nfn f2n(x) {{ match x {{ FZ => Zero }} }}\n\
         main : Nat\nfn main() {{ Zero }}\n"
    );
    assert!(check_program(&src).is_err(), "missing reachable FS must be rejected");
}

// ===========================================================================
// PHASE 1a — surface expressiveness (let, nested/expression match)
// ===========================================================================

const ONEA: &str = "%builtin Nat Nat\nenum Nat { Zero : Nat, Succ : Nat -> Nat }\n\
    enum Bool { T : Bool, F : Bool }\n\
    boxed enum Tree { leaf : Tree, node : Tree -> Tree -> Tree }\n\
    add : Nat -> Nat -> Nat\n\
    %total fn add(a, b) { match a { Zero => b, Succ(k) => Succ(add(k, b)) } }\n";

#[test]
fn simple_let_binding_works() {
    let src = format!(
        "{ONEA}f : Nat -> Nat\nfn f(n) {{ let x = add(n, n); add(x, n) }}\n\
         main : Nat\nfn main() {{ f(3) }}\n"
    );
    assert!(check_program(&src).is_ok(), "{:?}", check_program(&src).err());
}

#[test]
fn match_on_a_let_bound_var_and_on_an_expression() {
    // match on a let-bound var:
    let s1 = format!(
        "{ONEA}g : Bool -> Nat\nfn g(b) {{ let c = b; match c {{ T => Succ(Zero), F => Zero }} }}\n\
         main : Nat\nfn main() {{ g(T) }}\n"
    );
    assert!(check_program(&s1).is_ok(), "{:?}", check_program(&s1).err());
    // match directly on a CALL expression (desugars to a let):
    let s2 = format!(
        "{ONEA}not : Bool -> Bool\nfn not(b) {{ match b {{ T => F, F => T }} }}\n\
         h : Bool -> Nat\nfn h(b) {{ match not(b) {{ T => Succ(Zero), F => Zero }} }}\n\
         main : Nat\nfn main() {{ h(F) }}\n"
    );
    assert!(check_program(&s2).is_ok(), "{:?}", check_program(&s2).err());
}

#[test]
fn nested_match_bool_and_nat() {
    // a nested Bool match AND a nested Nat case (NatCase), non-recursive.
    let src = format!(
        "{ONEA}pick : Bool -> Nat -> Nat\n\
         fn pick(b, n) {{ match b {{ T => match n {{ Zero => Zero, Succ(k) => k }}, F => n }} }}\n\
         main : Nat\nfn main() {{ pick(T, Succ(Succ(Zero))) }}\n"
    );
    assert!(check_program(&src).is_ok(), "{:?}", check_program(&src).err());
}

#[test]
fn recursive_tree_fold_with_let_is_total() {
    // a structural fold over a boxed binary tree (two induction hypotheses),
    // using a `let` in the recursive arm — stays `%total`.
    let src = format!(
        "{ONEA}size : Tree -> Nat\n\
         %total fn size(t) {{ match t {{ leaf => Succ(Zero), node(l, r) => let s = add(size(l), size(r)); s }} }}\n\
         main : Nat\nfn main() {{ size(node(node(leaf, leaf), leaf)) }}\n"
    );
    let prog = check_program(&src).unwrap_or_else(|e| panic!("{e:?}"));
    assert!(prog.totality.iter().any(|(n, t, _)| n == "size" && *t), "size must be total");
}

// ---------------------------------------------------------------------------
// stratum (A): the linear-Nat index decision procedure (src/solver.rs). Index
// `+` is now decided up to commutativity / associativity / `+0` / `Suc n = n+1`,
// so these equalities hold BY THE KERNEL — no rewrite proofs — while false ones
// are still rejected. See docs/PHASE_C_SOLVER_PLAN.md.
// ---------------------------------------------------------------------------

const BNAT: &str = r#"
%builtin Nat Nat
enum Nat { Zero : Nat, Succ : Nat -> Nat }
"#;

#[test]
fn index_plus_is_commutative() {
    let src = format!(
        "{BNAT}\ncomm : (n : Nat) -> (m : Nat) -> Eq Nat (n + m) (m + n)\nfn comm(n, m) {{ refl(n + m) }}\n"
    );
    assert!(check_program(&src).is_ok(), "{:?}", check_program(&src).err());
}

#[test]
fn index_plus_zero_is_identity() {
    let src = format!(
        "{BNAT}\nrid : (n : Nat) -> Eq Nat (n + Zero) n\nfn rid(n) {{ refl(n) }}\n"
    );
    assert!(check_program(&src).is_ok(), "{:?}", check_program(&src).err());
}

#[test]
fn index_plus_is_associative() {
    let src = format!(
        "{BNAT}\nas : (a : Nat) -> (b : Nat) -> (c : Nat) -> Eq Nat ((a + b) + c) (a + (b + c))\nfn as(a, b, c) {{ refl(a + b + c) }}\n"
    );
    assert!(check_program(&src).is_ok(), "{:?}", check_program(&src).err());
}

#[test]
fn index_succ_is_plus_one() {
    let src = format!(
        "{BNAT}\nsp : (n : Nat) -> Eq Nat (Succ n) (n + Succ Zero)\nfn sp(n) {{ refl(Succ(n)) }}\n"
    );
    assert!(check_program(&src).is_ok(), "{:?}", check_program(&src).err());
}

#[test]
fn index_false_equation_is_still_rejected() {
    // n + m = n is NOT a linear-Nat identity — the solver must keep it distinct,
    // or the kernel would be unsound.
    let src = format!(
        "{BNAT}\nbad : (n : Nat) -> (m : Nat) -> Eq Nat (n + m) n\nfn bad(n, m) {{ refl(n + m) }}\n"
    );
    assert!(check_program(&src).is_err());
}

// ---------------------------------------------------------------------------
// stratum (A): the inequality decision — `Le`/`Lt` propositions discharged by
// the explicit, proof-producing `le`/`lt` (src/solver.rs::diff_witness). The
// solver emits `(d, refl)`; the kernel re-checks it, so soundness does not rest
// on the solver. See docs/PHASE_C_SOLVER_PLAN.md.
// ---------------------------------------------------------------------------

#[test]
fn closed_bound_is_discharged() {
    // 1 < 3
    let src = format!(
        "{BNAT}\np : Lt (Succ Zero) (Succ (Succ (Succ Zero)))\nfn p() {{ lt(Succ(Zero), Succ(Succ(Succ(Zero)))) }}\n"
    );
    assert!(check_program(&src).is_ok(), "{:?}", check_program(&src).err());
}

#[test]
fn open_bound_over_variables_is_discharged() {
    // n <= n + m  — impossible to build with an inductive LT over variables.
    let src = format!(
        "{BNAT}\nw : (n : Nat) -> (m : Nat) -> Le n (n + m)\nfn w(n, m) {{ le(n, n + m) }}\n"
    );
    assert!(check_program(&src).is_ok(), "{:?}", check_program(&src).err());
}

#[test]
fn false_strict_bound_is_a_hard_error() {
    // n < n does not hold.
    let src = format!("{BNAT}\nb : (n : Nat) -> Lt n n\nfn b(n) {{ lt(n, n) }}\n");
    assert!(check_program(&src).is_err());
}

#[test]
fn false_open_bound_is_a_hard_error() {
    // n + m <= n does not hold for all m.
    let src = format!(
        "{BNAT}\nb : (n : Nat) -> (m : Nat) -> Le (n + m) n\nfn b(n, m) {{ le(n + m, n) }}\n"
    );
    assert!(check_program(&src).is_err());
}

#[test]
fn decided_bound_gates_an_array_read() {
    // the headline: a bounds-checked read whose `i < n` obligation is discharged
    // by the solver, and whose out-of-bounds variant is rejected.
    let prelude = format!(
        "{BNAT}\n\
         postulate Arr : Type -> Nat -> Type\n\
         postulate get : {{0 a : Type}} -> (n : Nat) -> (i : Nat) -> (0 _ : Lt i n) -> Arr a n -> a\n"
    );
    let ok = format!(
        "{prelude}\
         read1 : {{0 a : Type}} -> Arr a (Succ (Succ (Succ Zero))) -> a\n\
         fn read1(arr) {{ get(Succ(Succ(Succ(Zero))), Succ(Zero), lt(Succ(Zero), Succ(Succ(Succ(Zero)))), arr) }}\n"
    );
    assert!(check_program(&ok).is_ok(), "{:?}", check_program(&ok).err());

    let oob = format!(
        "{prelude}\
         readbad : {{0 a : Type}} -> Arr a (Succ (Succ (Succ Zero))) -> a\n\
         fn readbad(arr) {{ get(Succ(Succ(Succ(Zero))), Succ(Succ(Succ(Zero))), lt(Succ(Succ(Succ(Zero))), Succ(Succ(Succ(Zero)))), arr) }}\n"
    );
    assert!(check_program(&oob).is_err(), "out-of-bounds read must be rejected");
}

// ---------------------------------------------------------------------------
// the VIEW LAYER (L3 address/permission split), slice 1 — TYPE-LEVEL. The
// docs/02 memory model expressed in the QTT core as postulates: a copyable
// `Ptr l` (the alias) separated from a LINEAR `PtsTo l a` (the permission). All
// of use-after-free / double-free / leak fall out of linearity on the view, and
// STRONG UPDATE (type-changing `vwrite`) type-checks. See examples/views.tal.
// ---------------------------------------------------------------------------

// `Loc`/`Ptr`/`PtsTo`/`Cell`/`valloc`/`vwrite`/`vread`/`vfree`/`Unit` all come
// from the BUILT-IN prelude (these tests exercise the real primitives); only the
// program-specific datatypes are declared here.
const VIEWS: &str = r#"
enum Nat  { Zero : Nat, Succ : Nat -> Nat }
enum Bool { False : Bool, True : Bool }
enum Two  { MkTwo : Unit -> Unit -> Two }
"#;

#[test]
fn view_strong_update_then_free_typechecks() {
    // allocate a Nat cell, strong-update it to a Bool in place, then free.
    let src = format!(
        "{VIEWS}\ndemo : Unit\nfn demo() {{ match valloc(Zero) {{ MkCell(p, v) => vfree(p, vwrite(p, v, True)), }} }}\n"
    );
    assert!(check_program(&src).is_ok(), "{:?}", check_program(&src).err());
}

#[test]
fn view_pointer_is_copyable_view_is_linear() {
    // `p` used many times (ω address), the view threaded exactly once — accepted.
    let src = format!(
        "{VIEWS}\nok : Unit\nfn ok() {{ match valloc(Zero) {{ MkCell(p, v) => vfree(p, vwrite(p, vwrite(p, v, True), Zero)), }} }}\n"
    );
    assert!(check_program(&src).is_ok(), "{:?}", check_program(&src).err());
}

#[test]
fn view_leak_is_rejected() {
    // dropping the linear view is a leak (0 ⋢ 1).
    let src = format!(
        "{VIEWS}\nbad : Unit\nfn bad() {{ match valloc(Zero) {{ MkCell(p, v) => U, }} }}\n"
    );
    assert!(check_program(&src).is_err());
}

#[test]
fn view_double_free_is_rejected() {
    // using the view twice is a double-free (ω ⋢ 1).
    let src = format!(
        "{VIEWS}\nbad : Two\nfn bad() {{ match valloc(Zero) {{ MkCell(p, v) => MkTwo(vfree(p, v), vfree(p, v)), }} }}\n"
    );
    assert!(check_program(&src).is_err());
}

#[test]
fn view_use_after_free_is_rejected() {
    // reusing the view after a strong update consumed it (use-after-free).
    let src = format!(
        "{VIEWS}\nbad : Two\nfn bad() {{ match valloc(Zero) {{ MkCell(p, v) => MkTwo(vfree(p, vwrite(p, v, True)), vfree(p, v)), }} }}\n"
    );
    assert!(check_program(&src).is_err());
}

#[test]
fn view_take_then_leaking_the_hole_is_rejected() {
    // `vtake` MOVES the value out and hands back a `PtsTo l Hole`; forgetting it
    // is a leak (the cell is never refilled or freed). Sound: 0 ⋢ 1.
    let src = format!(
        "{VIEWS}\nbad : Nat\nfn bad() {{ match valloc(Zero) {{ MkCell(p, v) => let (x, vh) = vtake(p, v); x, }} }}\n"
    );
    assert!(check_program(&src).is_err());
}

// ---------------------------------------------------------------------------
// PROPER LINEAR TYPES: the `linear` declaration marks a type whose values are
// resources (no drop, no dup). Views (`PtsTo`) and `Own` are `linear` in the
// prelude; users can declare their own (file handles, sockets, …). An
// un-annotated binder of a linear type defaults to multiplicity 1, closing the
// silent-leak hole. See examples/linear_resource.tal, docs/VIEW_LAYER_PLAN.md.
// ---------------------------------------------------------------------------

const FILE: &str = r#"
enum Nat { Zero : Nat, Succ : Nat -> Nat }
enum Two { MkTwo : Unit -> Unit -> Two }
linear postulate File : Type
postulate openf  : Nat -> File
postulate readf  : File -> File
postulate closef : File -> Unit
"#;

#[test]
fn linear_resource_used_once_is_accepted() {
    let src = format!("{FILE}\np : Nat -> Unit\nfn p(n) {{ closef(readf(openf(n))) }}\n");
    assert!(check_program(&src).is_ok(), "{:?}", check_program(&src).err());
}

#[test]
fn linear_resource_dropped_is_rejected() {
    // an un-annotated `File` binder defaults to multiplicity 1, so dropping it leaks.
    let src = format!("{FILE}\nbad : File -> Unit\nfn bad(f) {{ U }}\n");
    assert!(check_program(&src).is_err());
}

#[test]
fn linear_resource_used_twice_is_rejected() {
    let src = format!("{FILE}\nbad : File -> Two\nfn bad(f) {{ MkTwo(closef(f), closef(f)) }}\n");
    assert!(check_program(&src).is_err());
}

#[test]
fn bare_view_binder_defaults_linear() {
    // regression guard: a bare (un-annotated) view binder must NOT silently leak.
    // `PtsTo` is `linear` in the prelude, so dropping `v` is rejected.
    let src = "enum Nat { Zero : Nat, Succ : Nat -> Nat }\n\
               bad : {0 l : Loc} -> {0 a : Type} -> Ptr l -> PtsTo l a -> Unit\n\
               fn bad(p, v) { U }\n";
    assert!(check_program(src).is_err());
}

#[test]
fn generic_higher_order_over_linear_data() {
    // ONE generic `lmap` transports a whole list of LINEAR resources: it maps
    // `free` over a list of `Own Nat`, releasing each exactly once. Proves generic
    // higher-order code over linear data is expressible with explicit
    // multiplicities. See examples/linear_generic.tal. The `f(h)` consumption is
    // CBV-`let`-sequenced (the sanctioned form for feeding a linear consumption
    // into an unrestricted constructor position) — that is also what the
    // LINEAR-CAPABILITY check verifies, so `lmap` may be instantiated at `Own Nat`.
    let src = "enum Nat { Zero : Nat, Succ : Nat -> Nat }\n\
               boxed enum LList (a : Type) { LNil : LList a, LCons : a -> LList a -> LList a }\n\
               lmap : {0 a : Type} -> {0 b : Type} -> (w f : (1 x : a) -> b) -> (1 xs : LList a) -> LList b\n\
               fn lmap(f, xs) { match xs { LNil => LNil, LCons(h, t) => let y = f(h); let r = lmap(f, t); LCons(y, r), } }\n\
               freeNat : (1 o : Own Nat) -> Unit\n\
               fn freeNat(o) { free(o) }\n\
               freeall : (1 xs : LList (Own Nat)) -> LList Unit\n\
               fn freeall(xs) { lmap(freeNat, xs) }\n";
    assert!(check_program(src).is_ok(), "{:?}", check_program(src).err());
}

#[test]
fn generic_code_cannot_leak_linear_elements() {
    // SOUNDNESS (closes the FUTURE_WORK §13 parametricity hole): a generic
    // function whose body DROPS values of its abstract type parameter must not
    // be instantiable at a LINEAR type — before the linear-capability check,
    // `drophead` silently leaked every `Own Nat` in the list (the use-site
    // rebind saw only the abstract `a`, never the linear instantiation).
    let src = "enum Nat { Zero : Nat, Succ : Nat -> Nat }\n\
               boxed enum LList (a : Type) { LNil : LList a, LCons : a -> LList a -> LList a }\n\
               drophead : {0 a : Type} -> (1 xs : LList a) -> Nat\n\
               fn drophead(xs) { match xs { LNil => Zero, LCons(h, t) => drophead(t) } }\n\
               leak : (1 xs : LList (Own Nat)) -> Nat\n\
               fn leak(xs) { drophead(xs) }\n";
    let err = check_program(src).err().expect("generic linear leak must be rejected");
    assert!(err.iter().any(|e| e.contains("LINEAR type")), "got: {err:?}");
    // ...while the same generic function at a COPYABLE type stays fine.
    let ok = "enum Nat { Zero : Nat, Succ : Nat -> Nat }\n\
              boxed enum LList (a : Type) { LNil : LList a, LCons : a -> LList a -> LList a }\n\
              drophead : {0 a : Type} -> (1 xs : LList a) -> Nat\n\
              fn drophead(xs) { match xs { LNil => Zero, LCons(h, t) => drophead(t) } }\n\
              use2 : (1 xs : LList Nat) -> Nat\n\
              fn use2(xs) { drophead(xs) }\n";
    assert!(check_program(ok).is_ok(), "{:?}", check_program(ok).err());
}

#[test]
fn mult_variable_binder_parses_and_errors_cleanly() {
    // Multiplicity-polymorphism foundation: a `(m x : a)` binder (mult variable)
    // parses, and — until the monomorphization layer lands — resolution fails with
    // a CLEAN error, never a panic. (Locks in the SMult representation + parsing.)
    let src = "enum Nat { Zero : Nat, Succ : Nat -> Nat }\n\
               id : {0 a : Type} -> (m x : a) -> a\n\
               fn id(x) { x }\n";
    let err = check_program(src).err().expect("mult-var must not yet check");
    assert!(err.iter().any(|e| e.contains("multiplicity variable")), "got: {err:?}");
}

// ---------------------------------------------------------------------------
// THE CONVOY (docs/CONVOY_HANDOFF.md) — index refinement in dependent match
// ---------------------------------------------------------------------------

/// The shared header for the convoy tests: Nat, Vec, Fin, Void + exfalso + fzv.
const CONVOY_HDR: &str = "%builtin Nat Nat\nenum Nat { Zero : Nat, Succ : Nat -> Nat }\n\
    boxed enum Vec (a : Type) : Nat -> Type { Nil : Vec a Zero, Cons : {0 k : Nat} -> a -> Vec a k -> Vec a (Succ k) }\n\
    boxed enum Fin : Nat -> Type { FZ : {0 n : Nat} -> Fin (Succ n), FS : {0 n : Nat} -> Fin n -> Fin (Succ n) }\n\
    enum Void { }\n\
    exfalso : {0 a : Type} -> Void -> a\nfn exfalso(v) { match v { } }\n\
    fzv : Fin Zero -> Void\nfn fzv(f) { match f { } }\n";

#[test]
fn convoy_dependent_lookup_type_checks() {
    // THE dependent-types benchmark function: total-coverage, bounds-check-free
    // vector lookup. The OUTER match refines `i : Fin n` per arm (`Fin Zero` in
    // `Nil` — discharged as absurd; `Fin (Succ k)` in `Cons`); the INNER match
    // on `i` re-binds `rest` through the motive's Succ-INVERSION (`NatCase`
    // large elimination — J-free), so the recursive call's implicit is inferred
    // consistently. The kernel re-checks everything (the convoy is untrusted).
    let src = format!(
        "{CONVOY_HDR}lookup : {{0 n : Nat}} -> Fin n -> Vec Nat n -> Nat\n\
         fn lookup(i, env) {{ match env {{ Nil => exfalso(fzv(i)), Cons(v, rest) => match i {{ FZ => v, FS(j) => lookup(j, rest) }} }} }}\n\
         main : Nat\nfn main() {{ Zero }}\n"
    );
    assert!(check_program(&src).is_ok(), "{:?}", check_program(&src).err());
}

#[test]
fn convoy_impossible_arm_must_be_omitted_and_reachable_arm_must_be_present() {
    // vhead over `Vec Nat (Succ k)`: the `Nil` arm is REFUTED by the index —
    // it may (must) be omitted...
    let ok = format!(
        "{CONVOY_HDR}vhead : {{0 k : Nat}} -> Vec Nat (Succ k) -> Nat\n\
         fn vhead(v) {{ match v {{ Cons(h, t) => h }} }}\nmain : Nat\nfn main() {{ Zero }}\n"
    );
    assert!(check_program(&ok).is_ok(), "{:?}", check_program(&ok).err());
    // ...WRITING the impossible arm is rejected (it cannot be given a body)...
    let written = format!(
        "{CONVOY_HDR}vhead : {{0 k : Nat}} -> Vec Nat (Succ k) -> Nat\n\
         fn vhead(v) {{ match v {{ Nil => 0, Cons(h, t) => h }} }}\nmain : Nat\nfn main() {{ Zero }}\n"
    );
    let err = check_program(&written).err().expect("impossible arm written must reject");
    assert!(err.iter().any(|e| e.contains("impossible")), "got: {err:?}");
    // ...and omitting a REACHABLE arm (unconstrained index `n`) is still a
    // missing case — the refuted-arm relaxation never weakens coverage.
    let missing = format!(
        "{CONVOY_HDR}vhead : {{0 n : Nat}} -> Vec Nat n -> Nat\n\
         fn vhead(v) {{ match v {{ Cons(h, t) => h }} }}\nmain : Nat\nfn main() {{ Zero }}\n"
    );
    let err = check_program(&missing).err().expect("omitting a reachable arm must reject");
    assert!(err.iter().any(|e| e.contains("missing a case")), "got: {err:?}");
}

#[test]
fn convoy_refinement_is_real_not_a_loophole() {
    // RED-TEAM: the refinement must be the CONSTRUCTOR'S index, not a blanket
    // coercion — using `i` at `Fin Zero` inside the `Cons` arm (where it is
    // `Fin (Succ k)`) must fail the kernel re-check.
    let src = format!(
        "{CONVOY_HDR}bad : {{0 n : Nat}} -> Fin n -> Vec Nat n -> Nat\n\
         fn bad(i, env) {{ match env {{ Nil => exfalso(fzv(i)), Cons(v, rest) => exfalso(fzv(i)) }} }}\n\
         main : Nat\nfn main() {{ Zero }}\n"
    );
    let err = check_program(&src).err().expect("misused refinement must reject");
    assert!(err.iter().any(|e| e.contains("mismatch")), "got: {err:?}");
}

#[test]
fn convoy_vtail_index_projection_type_checks() {
    // The Succ-INVERSION in the motive types the classic index projections:
    // `vtail : Vec Nat (Succ k) -> Vec Nat k` — the RESULT type mentions the
    // predecessor of the scrutinee's index, computed back by the motive's
    // NatCase. No equality eliminator anywhere.
    let src = format!(
        "{CONVOY_HDR}vtail : {{0 k : Nat}} -> Vec Nat (Succ k) -> Vec Nat k\n\
         fn vtail(v) {{ match v {{ Cons(h, t) => t }} }}\nmain : Nat\nfn main() {{ Zero }}\n"
    );
    assert!(check_program(&src).is_ok(), "{:?}", check_program(&src).err());
}

// ---------------------------------------------------------------------------
// Multiplicity polymorphism, slice 2 (docs/MULT_POLY_PLAN.md) — monomorphization
// ---------------------------------------------------------------------------

#[test]
fn mult_poly_one_lmap_serves_linear_and_unrestricted_callbacks() {
    // ONE `lmap` with an explicit `(m : Mult)` parameter is monomorphized per
    // call site: `lmap(1, freeNat, xs)` frees a list of linear Owns (m := 1);
    // `lmap(w, inc, xs)` maps a copyable list (m := w). The kernel only ever
    // sees the concrete instances (`lmap$1`, `lmap$w`).
    let src = "enum Nat { Zero : Nat, Succ : Nat -> Nat }\n\
        boxed enum LList (a : Type) { LNil : LList a, LCons : a -> LList a -> LList a }\n\
        lmap : {0 a : Type} -> {0 b : Type} -> (m : Mult) -> (w f : (m x : a) -> b) -> (1 xs : LList a) -> LList b\n\
        fn lmap(m, f, xs) { match xs { LNil => LNil, LCons(h, t) => let y = f(h); let r = lmap(m, f, t); LCons(y, r), } }\n\
        freeNat : (1 o : Own Nat) -> Unit\nfn freeNat(o) { free(o) }\n\
        inc : Nat -> Nat\nfn inc(n) { Succ(n) }\n\
        freeall : (1 xs : LList (Own Nat)) -> LList Unit\nfn freeall(xs) { lmap(1, freeNat, xs) }\n\
        incall : LList Nat -> LList Nat\nfn incall(xs) { lmap(w, inc, xs) }\n";
    assert!(check_program(src).is_ok(), "{:?}", check_program(src).err());
}

#[test]
fn mult_poly_unsound_instantiations_are_rejected() {
    // m := 0 over LINEAR elements: the 0-callback never consumes them, so
    // `lmap$0` fails the linear-capability check and the instantiation at
    // `Own Nat` is rejected (the leak is a compile error).
    let zero = "enum Nat { Zero : Nat, Succ : Nat -> Nat }\n\
        boxed enum LList (a : Type) { LNil : LList a, LCons : a -> LList a -> LList a }\n\
        lmap : {0 a : Type} -> {0 b : Type} -> (m : Mult) -> (w f : (m x : a) -> b) -> (1 xs : LList a) -> LList b\n\
        fn lmap(m, f, xs) { match xs { LNil => LNil, LCons(h, t) => let y = f(h); let r = lmap(m, f, t); LCons(y, r), } }\n\
        dropNat : (0 o : Own Nat) -> Unit\nfn dropNat(o) { U }\n\
        leakall : (1 xs : LList (Own Nat)) -> LList Unit\nfn leakall(xs) { lmap(0, dropNat, xs) }\n";
    let err = check_program(zero).err().expect("m := 0 over linear elements must reject");
    assert!(err.iter().any(|e| e.contains("LINEAR type")), "got: {err:?}");
    // m := w with a LINEAR-consuming callback: `(1 x) -> b` is not `(w x) -> b`
    // (the ω instance could call it many times on one value) — plain type error.
    let omega = "enum Nat { Zero : Nat, Succ : Nat -> Nat }\n\
        boxed enum LList (a : Type) { LNil : LList a, LCons : a -> LList a -> LList a }\n\
        lmap : {0 a : Type} -> {0 b : Type} -> (m : Mult) -> (w f : (m x : a) -> b) -> (1 xs : LList a) -> LList b\n\
        fn lmap(m, f, xs) { match xs { LNil => LNil, LCons(h, t) => let y = f(h); let r = lmap(m, f, t); LCons(y, r), } }\n\
        freeNat : (1 o : Own Nat) -> Unit\nfn freeNat(o) { free(o) }\n\
        badall : (1 xs : LList (Own Nat)) -> LList Unit\nfn badall(xs) { lmap(w, freeNat, xs) }\n";
    assert!(check_program(omega).is_err(), "m := w with a linear-consuming callback must reject");
}

// ---------------------------------------------------------------------------
// NESTED PATTERNS — the pattern-matrix desugar
// ---------------------------------------------------------------------------

const NPAT_HDR: &str = "%builtin Nat Nat\nenum Nat { Zero : Nat, Succ : Nat -> Nat }\n\
    boxed enum LList (a : Type) { LNil : LList a, LCons : a -> LList a -> LList a }\n";

#[test]
fn nested_patterns_desugar_and_check() {
    // merged outer arms + inner matches; nested `%builtin Nat` patterns;
    // constructor patterns in the first position; `Nil`-style nullary
    // constructors inside a pattern (NOT silently a binder anymore).
    let src = format!(
        "{NPAT_HDR}second : LList Nat -> Nat\n\
         fn second(xs) {{ match xs {{ LCons(h, LCons(h2, t)) => h2, LCons(h, LNil) => h, LNil => 0 }} }}\n\
         pred2 : Nat -> Nat\n\
         fn pred2(n) {{ match n {{ Succ(Succ(k)) => k, Succ(Zero) => 0, Zero => 0 }} }}\n\
         swaps : LList Nat -> Nat\n\
         fn swaps(xs) {{ match xs {{ LCons(Zero, r) => 100, LCons(Succ(k), r) => k, LNil => 7 }} }}\n"
    );
    assert!(check_program(&src).is_ok(), "{:?}", check_program(&src).err());
}

#[test]
fn nested_pattern_coverage_and_reachability() {
    // a missing INNER case is a coverage error…
    let missing = format!(
        "{NPAT_HDR}bad : LList Nat -> Nat\n\
         fn bad(xs) {{ match xs {{ LCons(h, LCons(h2, t)) => h2, LNil => 0 }} }}\n"
    );
    let err = check_program(&missing).err().expect("missing inner case must reject");
    assert!(err.iter().any(|e| e.contains("missing a case")), "got: {err:?}");
    // …an arm no input can reach is rejected…
    let dead = format!(
        "{NPAT_HDR}red : LList Nat -> Nat\n\
         fn red(xs) {{ match xs {{ LCons(a, r) => a, LCons(b, s) => b, LNil => 0 }} }}\n"
    );
    let err = check_program(&dead).err().expect("dead arm must reject");
    assert!(err.iter().any(|e| e.contains("unreachable")), "got: {err:?}");
    // …but a catch-all row below specific rows is fine (first match wins).
    let mixed = format!(
        "{NPAT_HDR}mixed : LList Nat -> Nat\n\
         fn mixed(xs) {{ match xs {{ LCons(Zero, r) => 100, LCons(other, r) => other, LNil => 7 }} }}\n"
    );
    assert!(check_program(&mixed).is_ok(), "{:?}", check_program(&mixed).err());
}

#[test]
fn nested_patterns_preserve_linearity() {
    // destructuring two levels deep MOVES both Owns out — the correct body
    // consumes each exactly once…
    const H: &str = "%builtin Nat Nat\nenum Nat { Zero : Nat, Succ : Nat -> Nat }\n\
        boxed enum OL { ONil : OL, OCons : Own Nat -> OL -> OL }\n\
        freeRest : (1 r : OL) -> Nat\n\
        fn freeRest(r) { match r { ONil => Zero, OCons(a, t) => let u = free(a); freeRest(t) } }\n";
    let ok = format!(
        "{H}sum2 : (1 xs : OL) -> Nat\n\
         fn sum2(xs) {{ match xs {{ OCons(a, OCons(b, ONil)) => let x = unbox(a); let y = unbox(b); x + y, OCons(a, r) => let u = free(a); freeRest(r), ONil => Zero }} }}\n"
    );
    assert!(check_program(&ok).is_ok(), "{:?}", check_program(&ok).err());
    // …and DROPPING the tail bound through a nested pattern is still a leak.
    let leak = format!(
        "{H}bad : (1 xs : OL) -> Nat\n\
         fn bad(xs) {{ match xs {{ OCons(a, OCons(b, ONil)) => let x = unbox(a); Zero, OCons(a, r) => let u = free(a); Zero, ONil => Zero }} }}\n"
    );
    assert!(check_program(&leak).is_err(), "a leak through a nested pattern must reject");
}

// ---------------------------------------------------------------------------
// Universe levels at the surface (Phase F: `Type i` annotations)
// ---------------------------------------------------------------------------

#[test]
fn surface_universe_levels() {
    // a Type-storing container is now DECLARABLE — at `Type 1`, where it lives;
    // matching on it (large elimination) works; a `{0 a : Type 1}` binder
    // accepts a level-0 type by CUMULATIVITY (practical level polymorphism for
    // the common direction).
    let ok = "enum Nat { Zero : Nat, Succ : Nat -> Nat }\n\
        enum TyBox : Type 1 { mkTyBox : Type -> TyBox }\n\
        unbox1 : TyBox -> Type\nfn unbox1(b) { match b { mkTyBox(t) => t } }\n\
        pick : {0 a : Type 1} -> a -> a\nfn pick(x) { x }\n\
        main : Nat\nfn main() { pick(Zero) }\n";
    assert!(check_program(ok).is_ok(), "{:?}", check_program(ok).err());
    // …and WITHOUT the annotation the Girard/predicativity guard still fires:
    // a `Type 0` datatype cannot store a type from its own universe.
    let bad = "enum Nat { Zero : Nat, Succ : Nat -> Nat }\n\
        enum TyBox { mkTyBox : Type -> TyBox }\n\
        main : Nat\nfn main() { Zero }\n";
    let err = check_program(bad).err().expect("a Type-storing Type-0 enum must reject");
    assert!(err.iter().any(|e| e.contains("predicativity") || e.contains("universe") || e.contains("Type 1")), "got: {err:?}");
}

#[test]
fn convoy_fold_certifies_lookup_total() {
    // THE CONVOY FOLD: `lookup`'s recursion varies `i` — the analyzer declines
    // it, but the varying argument is exactly the scrutinee's index-dependent
    // value, so the dependent motive abstracts it and the recursion is a
    // KERNEL-CHECKED eliminator: certified `%total` (and its callers with it).
    let src = format!(
        "{CONVOY_HDR}lookup : {{0 n : Nat}} -> Fin n -> Vec Nat n -> Nat\n\
         %total fn lookup(i, env) {{ match env {{ Nil => exfalso(fzv(i)), Cons(v, rest) => match i {{ FZ => v, FS(j) => lookup(j, rest) }} }} }}\n\
         main : Nat\nfn main() {{ Zero }}\n"
    );
    let prog = check_program(&src).unwrap_or_else(|e| panic!("total lookup must certify: {e:?}"));
    assert!(is_total(&prog, "lookup"), "lookup must be certified total via the convoy fold");
    // red-team: varying an argument that is NOT index-dependent still falls back
    // (the fold's verbatim guard rejects it; the fn stays honest %partial-style)
    // — and `%total` on it is a hard error, not a silent acceptance.
    let bad = format!(
        "{CONVOY_HDR}wander : {{0 n : Nat}} -> Nat -> Vec Nat n -> Nat\n\
         %total fn wander(acc, env) {{ match env {{ Nil => acc, Cons(v, rest) => wander(v, rest) }} }}\n\
         main : Nat\nfn main() {{ Zero }}\n"
    );
    assert!(check_program(&bad).is_err(), "a non-dep varying arg must not certify %total");
}

#[test]
fn forward_references_between_fns() {
    // definitions no longer need to precede their callers: the fn items are
    // topologically reordered by the call graph (callees first, source order
    // preserved among independent fns). `main` can come first.
    let src = "%builtin Nat Nat\nenum Nat { Zero : Nat, Succ : Nat -> Nat }\n\
        main : Nat\nfn main() { let a = double(10); a + tag }\n\
        double : Nat -> Nat\nfn double(n) { n + n }\n\
        tag : Nat\nfn tag() { 22 }\n";
    let prog = check_program(src).unwrap_or_else(|e| panic!("forward refs must check: {e:?}"));
    assert_eq!(prog.normalize("main"), Some(Term::NatLit(42)));
    // a genuine CYCLE (mutual recursion) still errors as before — the reorder
    // must not silently accept what the language does not yet support.
    let cyc = "%builtin Nat Nat\nenum Nat { Zero : Nat, Succ : Nat -> Nat }\n\
        isEven : Nat -> Nat\nfn isEven(n) { match n { Zero => 1, Succ(k) => isOdd(k) } }\n\
        isOdd : Nat -> Nat\nfn isOdd(n) { match n { Zero => 0, Succ(k) => isEven(k) } }\n\
        main : Nat\nfn main() { isEven(4) }\n";
    assert!(check_program(cyc).is_err(), "mutual recursion is still an (honest) error");
}

// ---- CONTIGUOUS ARRAYS: the safety half (rejections at compile time) ----

const BNAT_ARR: &str = "%builtin Nat Nat\nenum Nat { Zero : Nat, Succ : Nat -> Nat }\n";

#[test]
fn array_out_of_bounds_is_rejected_at_compile_time() {
    // reading a[3] of a length-3 array: `lt(3, 3)` has no proof — the read
    // cannot be written. No runtime check exists to fall back on.
    let src = format!(
        "{BNAT_ARR}\
         main : Nat\n\
         fn main() {{\n\
             let a0 = anew(3, 0);\n\
             let (x, a1) = aget(3, lt(3, 3), a0);\n\
             let u = afree(a1);\n\
             x\n\
         }}\n"
    );
    let err = check_program(&src).err().expect("out-of-bounds must be a type error");
    assert!(
        format!("{err:?}").contains("cannot prove"),
        "expected the solver's bound failure, got {err:?}"
    );
}

#[test]
fn array_leak_and_double_free_are_rejected() {
    // dropping the array (no afree) is a leak: 0 ⋢ 1.
    let leak = format!(
        "{BNAT_ARR}\
         main : Nat\n\
         fn main() {{\n\
             let a0 = anew(3, 0);\n\
             let (x, a1) = aget(1, lt(1, 3), a0);\n\
             x\n\
         }}\n"
    );
    let err = check_program(&leak).err().expect("leaking an Arr must be rejected");
    assert!(format!("{err:?}").contains("0 ⋢ 1"), "expected a leak error, got {err:?}");

    // freeing it twice is a double-free: ω ⋢ 1.
    let dfree = format!(
        "{BNAT_ARR}\
         main : Nat\n\
         fn main() {{\n\
             let a0 = anew(3, 0);\n\
             let u = afree(a0);\n\
             let v = afree(a0);\n\
             0\n\
         }}\n"
    );
    let err = check_program(&dfree).err().expect("double-freeing an Arr must be rejected");
    assert!(format!("{err:?}").contains("ω ⋢ 1"), "expected a double-free error, got {err:?}");
}

#[test]
fn array_linear_element_cannot_be_stored() {
    // `anew(n, init)` conceptually copies `init` into every slot, and `aget`
    // copies an element out — sound only for unrestricted elements. The element
    // positions in the array API are ω, so an `Own` cannot be smuggled in: the
    // duplication is rejected at the call, not discovered at runtime.
    let src = format!(
        "{BNAT_ARR}\
         main : Nat\n\
         fn main() {{\n\
             let o = alloc(Zero);\n\
             let a0 = anew(3, o);\n\
             let u = afree(a0);\n\
             0\n\
         }}\n"
    );
    let err = check_program(&src).err().expect("a linear element must be rejected");
    assert!(format!("{err:?}").contains("ω ⋢ 1"), "expected ω ⋢ 1, got {err:?}");
}

#[test]
fn linear_field_in_by_value_record_still_accounted() {
    // Phase B2 changes the REPRESENTATION of a multi-field struct (registers,
    // not a cell) but must not change the ACCOUNTING: an `Own` field inside a
    // by-value record is still consumed exactly once.
    let base = "%builtin Nat Nat\nenum Nat { Zero : Nat, Succ : Nat -> Nat }\n\
        struct Handle { o : Own Nat, tag : Nat }\n";
    let dbl = format!(
        "{base}\
         use2 : Handle -> Nat\n\
         fn use2(h) {{ match h {{ Handle(o, t) => let u = free(o); let v = free(o); t }} }}\n\
         main : Nat\n\
         fn main() {{ use2(Handle(alloc(0), 9)) }}\n"
    );
    let err = check_program(&dbl).err().expect("double-free through a record must be rejected");
    assert!(format!("{err:?}").contains("ω ⋢ 1"), "expected ω ⋢ 1, got {err:?}");

    let ok = format!(
        "{base}\
         use1 : Handle -> Nat\n\
         fn use1(h) {{ match h {{ Handle(o, t) => let u = free(o); t }} }}\n\
         main : Nat\n\
         fn main() {{ use1(Handle(alloc(0), 9)) }}\n"
    );
    assert!(check_program(&ok).is_ok(), "{:?}", check_program(&ok).err());
}

#[test]
fn constructor_intro_consumes_linear_args_once() {
    // the INTRO-side rule: a constructor stores each runtime argument exactly
    // once, so a linear VARIABLE can be routed into a field (building linked
    // structures node by node) — while duplication through a constructor is
    // still impossible (usages add across arguments).
    const H: &str = "%builtin Nat Nat\nenum Nat { Zero : Nat, Succ : Nat -> Nat }\n\
        enum Opt (a : Type) { None : Opt a, Some : a -> Opt a }\n\
        struct Node { v : Nat, next : Opt (Own Node) }\n";
    let ok = format!(
        "{H}cons : (n : Nat) -> (1 t : Opt (Own Node)) -> Opt (Own Node)\n\
         fn cons(n, t) {{ Some(alloc(Node(n, t))) }}\n\
         eat : (1 o : Opt (Own Node)) -> Nat\n\
         %partial fn eat(o) {{ match o {{ None => 0, Some(p) => let x = unbox(p); match x {{ Node(v, t) => v + eat(t) }} }} }}\n\
         main : Nat\n\
         fn main() {{ let l = cons(7, None); eat(l) }}\n"
    );
    assert!(check_program(&ok).is_ok(), "{:?}", check_program(&ok).err());

    // same linear variable into TWO fields: still ω ⋢ 1.
    let dup = format!(
        "{H}enum P {{ MkP : Own Nat -> Own Nat -> P }}\n\
         main : Nat\n\
         fn main() {{ let o = alloc(5); let p = MkP(o, o); 0 }}\n"
    );
    assert!(check_program(&dup).is_err(), "double-store must be rejected");
}

#[test]
fn borrow_discipline_is_the_existing_accounting() {
    // Phase C red-team: every misuse of a borrow is caught by the ordinary
    // QTT rules — no bespoke borrow checker.
    const H: &str = "%builtin Nat Nat\nenum Nat { Zero : Nat, Succ : Nat -> Nat }\n";
    let no = |body: &str, what: &str| {
        let src = format!("{H}main : Nat\nfn main() {{\n    let o = alloc(40);\n{body}\n}}\n");
        assert!(check_program(&src).is_err(), "{what} must be rejected:\n{src}");
    };
    // forget to restore (drop view + loan): leak.
    no("    match borrow(o) { MkBorrowed(p, v, ln) => let (x, vh) = vtake(p, v); x }",
       "dropping a borrow without restore");
    // free the cell while borrowed: the loan is stranded.
    no("    match borrow(o) { MkBorrowed(p, v, ln) => let u = vfree(p, v); 0 }",
       "vfree under a borrow");
    // use the view twice.
    no("    match borrow(o) { MkBorrowed(p, v, ln) => let v1 = vwrite(p, v, 1); let v2 = vwrite(p, v, 2); let o2 = restore(p, v2, ln); unbox(o2) }",
       "double use of a view");
    // write through the view after restoring.
    no("    match borrow(o) { MkBorrowed(p, v, ln) => let o2 = restore(p, v, ln); let v9 = vwrite(p, v, 99); unbox(o2) }",
       "write after restore");
    // the GOOD program checks.
    let ok = format!(
        "{H}bump : (1 o : Own Nat) -> Own Nat\n\
         fn bump(o) {{ match borrow(o) {{ MkBorrowed(p, v, ln) => let (x, vh) = vtake(p, v); let v2 = vwrite(p, vh, x + 1); restore(p, v2, ln), }} }}\n\
         main : Nat\nfn main() {{ let o = alloc(41); let o1 = bump(o); unbox(o1) }}\n"
    );
    assert!(check_program(&ok).is_ok(), "{:?}", check_program(&ok).err());
}

#[test]
fn pool_discipline_red_team() {
    // Phase D: the pool token is the single linear authority — misuse is
    // caught by scope/accounting, never at runtime.
    const H: &str = "%builtin Nat Nat\nenum Nat { Zero : Nat, Succ : Nat -> Nat }\n\
        mk : {0 r : Region} -> (1 cap : RegionCap r) -> Pool r Nat\n\
        fn mk(cap) { pnew(cap) }\n";
    // use-after-release: the token was consumed by prelease.
    let uar = format!(
        "{H}main : Nat\nfn main() {{ match rnew(U) {{ MkRegionPack(cap) => \
         let P = mk(cap); let (c, p1) = palloc(P, 7); let u = prelease(p1); \
         let (x, p2) = pget(p1, c); x, }} }}\n"
    );
    assert!(check_program(&uar).is_err(), "use-after-release must reject");
    // leaking the pool.
    let leak = format!(
        "{H}main : Nat\nfn main() {{ match rnew(U) {{ MkRegionPack(cap) => \
         let P = mk(cap); let (c, p1) = palloc(P, 7); 0, }} }}\n"
    );
    assert!(check_program(&leak).is_err(), "a leaked pool must reject");
    // cross-region dereference: r indices don't unify.
    let cross = format!(
        "{H}main : Nat\nfn main() {{ match rnew(U) {{ MkRegionPack(cap) => \
         match rnew(U) {{ MkRegionPack(cap2) => \
         let pa = mk(cap); let pb = mk(cap2); \
         let (c, pa1) = palloc(pa, 7); let (x, pb1) = pget(pb, c); \
         let ua = prelease(pa1); let ub = prelease(pb1); x, }}, }} }}\n"
    );
    assert!(check_program(&cross).is_err(), "cross-region deref must reject");
    // a LINEAR element type is rejected by the copying-container gate.
    let lin = "%builtin Nat Nat\nenum Nat { Zero : Nat, Succ : Nat -> Nat }\n\
        mk : {0 r : Region} -> (1 cap : RegionCap r) -> Pool r (Own Nat)\n\
        fn mk(cap) { pnew(cap) }\n\
        main : Nat\nfn main() { match rnew(U) { MkRegionPack(cap) => \
        let P = mk(cap); let (c, p1) = palloc(P, alloc(0)); \
        let (o, p2) = pget(p1, c); let a = unbox(o); \
        let u = prelease(p2); a, } }\n";
    let err = check_program(lin).err().expect("linear pool elements must reject");
    assert!(
        format!("{err:?}").contains("LINEAR"),
        "expected the copying-container gate, got {err:?}"
    );
}

#[test]
fn arr_anonymous_linear_element_is_rejected() {
    // the hole the copying-container gate closed: an ANONYMOUS linear value
    // has no binder to over-count, so the ω-parameter gate alone missed it —
    // double-aget would then double-free. Now the element TYPE is checked.
    let src = "%builtin Nat Nat\nenum Nat { Zero : Nat, Succ : Nat -> Nat }\n\
        main : Nat\nfn main() {\n\
            let a0 = anew(3, alloc(0));\n\
            let (o1, a1) = aget(0, lt(0, 3), a0);\n\
            let (o2, a2) = aget(0, lt(0, 3), a1);\n\
            let x = unbox(o1);\n\
            let y = unbox(o2);\n\
            let u = afree(a2);\n\
            x + y\n\
        }\n";
    let err = check_program(src).err().expect("anonymous linear element must reject");
    assert!(format!("{err:?}").contains("LINEAR"), "got {err:?}");
}

// ═══════════════════════════════════════════════════════════════════════════
// PHASE 0 (docs/PHASE_SAFETY_PLAN.md §0.1) — the adversarial coverage corpus.
// THE P0 INVARIANT: any non-exhaustive match that COMPILES is a memory-safety
// bug (an unhandled tag would reach the backend's switch-default
// `unreachable` = UB). Every bad program here must be REJECTED at
// elaboration; every good one must CHECK. The ledger of the guarded
// `unreachable` sites lives in docs/TRUSTED_BASE.md §7.
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn phase0_adversarial_coverage_corpus() {
    let rejects = |src: &str, why: &str, needle: &str| {
        let err = check_program(src).err().unwrap_or_else(|| {
            panic!("P0 BUG — a non-exhaustive/ill-formed match COMPILED ({why}):\n{src}")
        });
        let msg = format!("{err:?}");
        assert!(msg.contains(needle), "{why}: expected error containing `{needle}`, got {msg}");
    };
    let accepts = |src: &str, why: &str| {
        check_program(src).unwrap_or_else(|e| panic!("{why} must CHECK, got {e:?}\n{src}"));
    };

    const COLOR: &str = "enum Color { Red : Color, Green : Color, Blue : Color }\n\
                         enum Nat { Zero : Nat, Succ : Nat -> Nat }\n";
    const LIST: &str = "enum Nat { Zero : Nat, Succ : Nat -> Nat }\n\
                        boxed enum List { Nil : List, Cons : Nat -> List -> List }\n";
    const FIN: &str = "%builtin Nat Nat\nenum Nat { Zero : Nat, Succ : Nat -> Nat }\n\
                       boxed enum Fin : Nat -> Type {\n\
                           FZ : {0 n : Nat} -> Fin (Succ n),\n\
                           FS : {0 n : Nat} -> Fin n -> Fin (Succ n),\n\
                       }\n";

    // ---- flat matches ----
    rejects(
        &format!("{COLOR}f : Color -> Nat\nfn f(c) {{ match c {{ Red => Zero, Green => Zero }} }}\nmain : Nat\nfn main() {{ f(Blue) }}\n"),
        "flat missing arm",
        "missing a case for `Blue`",
    );
    // %partial must NOT skip coverage — partiality is about termination only.
    rejects(
        &format!("{COLOR}f : Color -> Nat\n%partial\nfn f(c) {{ match c {{ Red => Zero, Green => Zero }} }}\nmain : Nat\nfn main() {{ f(Blue) }}\n"),
        "flat missing arm under %partial",
        "missing a case for `Blue`",
    );
    rejects(
        &format!("{COLOR}f : Color -> Nat\nfn f(c) {{ match c {{ Red => Zero, Green => Zero, Zero => Zero }} }}\nmain : Nat\nfn main() {{ Zero }}\n"),
        "cross-family constructor arm",
        "not a constructor of `Color`",
    );
    // a VALUE enum (flat tagged union) — same discipline, different lowering.
    rejects(
        "%builtin Nat Nat\nenum Nat { Zero : Nat, Succ : Nat -> Nat }\n\
         enum Opt { None : Opt, Some : Nat -> Opt }\n\
         f : Opt -> Nat\nfn f(o) { match o { Some(x) => x } }\nmain : Nat\nfn main() { f(Some(3)) }\n",
        "value-enum missing arm",
        "missing a case for `None`",
    );
    // %builtin Nat matches lower to native compare-and-branch — still gated.
    rejects(
        "%builtin Nat Nat\nenum Nat { Zero : Nat, Succ : Nat -> Nat }\n\
         f : Nat -> Nat\nfn f(n) { match n { Zero => Zero } }\nmain : Nat\nfn main() { f(Zero) }\n",
        "%builtin Nat missing successor",
        "missing the successor case",
    );

    // ---- nested patterns (the pattern-matrix path) ----
    rejects(
        &format!("{LIST}f : List -> Nat\nfn f(l) {{ match l {{ Cons(x, Cons(y, t)) => x, Nil => Zero }} }}\nmain : Nat\nfn main() {{ Zero }}\n"),
        "nested missing Cons(_, Nil)",
        "missing a case",
    );
    rejects(
        "enum Nat { Zero : Nat, Succ : Nat -> Nat }\n\
         f : Nat -> Nat\nfn f(n) { match n { Succ(Succ(Succ(k))) => k, Zero => Zero, Succ(Zero) => Zero } }\nmain : Nat\nfn main() { Zero }\n",
        "3-deep nested missing Succ(Succ(Zero))",
        "missing a case",
    );
    rejects(
        &format!("{LIST}f : List -> Nat\nfn f(l) {{ match l {{ Nil => Zero, Cons(x, t) => match t {{ Cons(y, Cons(z, r)) => z, Nil => x }} }} }}\nmain : Nat\nfn main() {{ Zero }}\n"),
        "nested match inside an arm, inner missing",
        "missing a case",
    );
    rejects(
        &format!("{LIST}f : List -> Nat\nfn f(l) {{ match l {{ Cons(x) => x, Nil => Zero }} }}\nmain : Nat\nfn main() {{ Zero }}\n"),
        "wrong-arity pattern",
        "expected 2 binder(s)",
    );
    rejects(
        &format!("{LIST}f : List -> Nat\nfn f(l) {{ match l {{ Cons(x, t) => x, Cons(x, Nil) => Zero, Nil => Zero }} }}\nmain : Nat\nfn main() {{ Zero }}\n"),
        "shadowed (unreachable) arm",
        "unreachable `match` arm",
    );

    // ---- absurd discharge (indexed families) ----
    // mixed absurd+reachable: at index Succ n BOTH ctors are live — FS required.
    rejects(
        &format!("{FIN}f : {{0 n : Nat}} -> Fin (Succ n) -> Nat\nfn f(x) {{ match x {{ FZ => Zero }} }}\nmain : Nat\nfn main() {{ Zero }}\n"),
        "mixed absurd+reachable missing the reachable arm",
        "missing a case for `FS`",
    );
    // claiming absurdity at a NON-empty index must be rejected, not trusted.
    rejects(
        &format!("{FIN}f : {{0 n : Nat}} -> Fin n -> Nat\nfn f(x) {{ match x {{ }} }}\nmain : Nat\nfn main() {{ Zero }}\n"),
        "zero-arm match at a possibly-inhabited index",
        "missing a case",
    );
    // genuinely absurd: Fin Zero has no inhabitants — zero arms CHECK.
    accepts(
        &format!("{FIN}f : Fin Zero -> Nat\nfn f(x) {{ match x {{ }} }}\nmain : Nat\nfn main() {{ Zero }}\n"),
        "the absurd Fin Zero match",
    );

    // ---- valid programs stay valid (rejection is not the cheap way out) ----
    accepts(
        &format!("{LIST}f : List -> Nat\nfn f(l) {{ match l {{ Cons(x, Cons(y, t)) => y, Cons(x, Nil) => x, Nil => Zero }} }}\nmain : Nat\nfn main() {{ f(Cons(Succ(Zero), Cons(Zero, Nil))) }}\n"),
        "the exhaustive nested match",
    );
    accepts(
        "enum Nat { Zero : Nat, Succ : Nat -> Nat }\n\
         f : Nat -> Nat\nfn f(n) { match n { Succ(Succ(Succ(k))) => k, Succ(Succ(Zero)) => Zero, Succ(Zero) => Zero, Zero => Zero } }\nmain : Nat\nfn main() { f(Zero) }\n",
        "the exhaustive 3-deep nested match",
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// PHASE A1 (docs/PHASE_SAFETY_PLAN.md) — the real coverage checker: multi-
// scrutinee matches, top-level wildcards/catch-alls, generalized absurd
// discharge. Everything lowers through the pattern matrix to kernel
// Case/Elim, which RE-CHECKS coverage (one method per constructor) — the
// elaborator stays untrusted.
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn a1_multi_scrutinee_coverage_and_dead_arms() {
    const BN: &str = "%builtin Nat Nat\nenum Nat { Zero : Nat, Succ : Nat -> Nat }\n";
    // a complete two-scrutinee match CHECKS
    let ok = format!(
        "{BN}f : Nat -> Nat -> Nat\nfn f(a, b) {{\n\
             match a, b {{ Zero, Zero => 1, Zero, Succ(y) => 2, Succ(x), Zero => 3, Succ(x), Succ(y) => x + y }}\n\
         }}\nmain : Nat\nfn main() {{ f(5, 7) }}\n"
    );
    check_program(&ok).unwrap_or_else(|e| panic!("complete multi-scrutinee match must check: {e:?}"));

    // a missing COMBINATION is a coverage error, not UB
    let missing = format!(
        "{BN}f : Nat -> Nat -> Nat\nfn f(a, b) {{\n\
             match a, b {{ Zero, Zero => 1, Succ(x), Zero => 2, Succ(x), Succ(y) => 3 }}\n\
         }}\nmain : Nat\nfn main() {{ f(0, 1) }}\n"
    );
    let err = check_program(&missing).err().expect("P0 BUG: missing combination compiled");
    assert!(format!("{err:?}").contains("missing"), "got {err:?}");

    // an arm shadowed by an earlier catch-all row is DEAD — rejected
    let dead = format!(
        "{BN}f : Nat -> Nat -> Nat\nfn f(a, b) {{\n\
             match a, b {{ x, y => 1, Zero, Zero => 2 }}\n\
         }}\nmain : Nat\nfn main() {{ f(0, 1) }}\n"
    );
    let err = check_program(&dead).err().expect("dead multi-scrutinee arm must reject");
    assert!(format!("{err:?}").contains("unreachable"), "got {err:?}");

    // wrong pattern count in a row
    let arity = format!(
        "{BN}f : Nat -> Nat -> Nat\nfn f(a, b) {{ match a, b {{ Zero => 1, Succ(x), y => 2 }} }}\n\
         main : Nat\nfn main() {{ f(0, 1) }}\n"
    );
    assert!(check_program(&arity).is_err(), "row with 1 pattern over 2 scrutinees must reject");
}

#[test]
fn a1_top_level_wildcards_and_catch_all() {
    const SETUP: &str = "enum Color { Red : Color, Green : Color, Blue : Color }\n\
                         enum Nat { Zero : Nat, Succ : Nat -> Nat }\n";
    // `_` and named catch-alls cover the rest of the family
    for arm in ["_ => Zero", "other => Zero"] {
        let src = format!(
            "{SETUP}f : Color -> Nat\nfn f(c) {{ match c {{ Red => Succ(Zero), {arm} }} }}\n\
             main : Nat\nfn main() {{ f(Blue) }}\n"
        );
        check_program(&src).unwrap_or_else(|e| panic!("catch-all `{arm}` must check: {e:?}"));
    }
    // the catch-all BINDS the scrutinee (usable in the body)
    let bind = format!(
        "{SETUP}g : Color -> Color\nfn g(c) {{ match c {{ Red => Green, other => other }} }}\n\
         main : Nat\nfn main() {{ match g(Blue) {{ Blue => Zero, Red => Zero, Green => Zero }} }}\n"
    );
    check_program(&bind).unwrap_or_else(|e| panic!("catch-all binder must be usable: {e:?}"));
    // a catch-all FIRST makes later arms dead — rejected
    let dead = format!(
        "{SETUP}f : Color -> Nat\nfn f(c) {{ match c {{ x => Zero, Red => Zero }} }}\n\
         main : Nat\nfn main() {{ Zero }}\n"
    );
    let err = check_program(&dead).err().expect("arm after catch-all must reject");
    assert!(format!("{err:?}").contains("unreachable"), "got {err:?}");
    // an ARGFUL unknown name is a typo, never a binder
    let typo = format!(
        "{SETUP}f : Nat -> Nat\nfn f(n) {{ match n {{ Zero => Zero, Sucx(k) => k }} }}\n\
         main : Nat\nfn main() {{ Zero }}\n"
    );
    let err = check_program(&typo).err().expect("argful unknown ctor must reject");
    assert!(format!("{err:?}").contains("not a declared constructor"), "got {err:?}");
}

#[test]
fn a1_absurd_discharge_generalized() {
    // (a) boxed-Nat-indexed Fin Zero: no %builtin required anymore
    let boxed_nat = "enum Nat { Zero : Nat, Succ : Nat -> Nat }\n\
        boxed enum Fin : Nat -> Type {\n\
            FZ : {0 n : Nat} -> Fin (Succ n),\n\
            FS : {0 n : Nat} -> Fin n -> Fin (Succ n),\n\
        }\n\
        f : Fin Zero -> Nat\nfn f(x) { match x { } }\n\
        main : Nat\nfn main() { Zero }\n";
    check_program(boxed_nat)
        .unwrap_or_else(|e| panic!("boxed-Nat-indexed absurd match must check: {e:?}"));

    // (b) a simple-enum index (Color) via the large-eliminating Case sentinel
    let color = "enum Nat { Zero : Nat, Succ : Nat -> Nat }\n\
        enum Color { Red : Color, Green : Color, Blue : Color }\n\
        boxed enum OnlyRed : Color -> Type { MkRed : OnlyRed Red }\n\
        f : OnlyRed Green -> Nat\nfn f(x) { match x { } }\n\
        main : Nat\nfn main() { Zero }\n";
    check_program(color).unwrap_or_else(|e| panic!("Color-indexed absurd match must check: {e:?}"));

    // (c) a TWO-index family where the SECOND index refutes
    let two = "%builtin Nat Nat\nenum Nat { Zero : Nat, Succ : Nat -> Nat }\n\
        boxed enum Pairy : Nat -> Nat -> Type { MkP : {0 n : Nat} -> Pairy n (Succ n) }\n\
        f : {0 n : Nat} -> Pairy n Zero -> Nat\nfn f(x) { match x { } }\n\
        main : Nat\nfn main() { Zero }\n";
    check_program(two).unwrap_or_else(|e| panic!("second-index refutation must check: {e:?}"));

    // (d) the INHABITED index with zero arms is REJECTED (missing case), and
    // (e) a variable index can never discharge
    let inhabited = "enum Nat { Zero : Nat, Succ : Nat -> Nat }\n\
        enum Color { Red : Color, Green : Color, Blue : Color }\n\
        boxed enum OnlyRed : Color -> Type { MkRed : OnlyRed Red }\n\
        f : OnlyRed Red -> Nat\nfn f(x) { match x { } }\n\
        main : Nat\nfn main() { Zero }\n";
    let err = check_program(inhabited).err().expect("P0 BUG: inhabited zero-arm match compiled");
    assert!(format!("{err:?}").contains("missing"), "got {err:?}");
    let varidx = "%builtin Nat Nat\nenum Nat { Zero : Nat, Succ : Nat -> Nat }\n\
        boxed enum Fin : Nat -> Type {\n\
            FZ : {0 n : Nat} -> Fin (Succ n),\n\
            FS : {0 n : Nat} -> Fin n -> Fin (Succ n),\n\
        }\n\
        f : {0 n : Nat} -> Fin n -> Nat\nfn f(x) { match x { } }\n\
        main : Nat\nfn main() { Zero }\n";
    let err = check_program(varidx).err().expect("P0 BUG: variable-index zero-arm match compiled");
    assert!(format!("{err:?}").contains("missing"), "got {err:?}");
}

// ═══════════════════════════════════════════════════════════════════════════
// PHASE A3 (docs/PHASE_SAFETY_PLAN.md) — the Raw/Init initialization
// typestate, and the dropping-destructor gate that closes the deep-leak
// hole the A3 audit found.
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn a3_raw_init_typestate() {
    const BN: &str = "%builtin Nat Nat\nenum Nat { Zero : Nat, Succ : Nat -> Nat }\n";
    // the safe path: ralloc (no store) -> winit (first write) -> vread
    let ok = format!(
        "{BN}main : Nat\nfn main() {{\n\
            let c : RawCell Nat = ralloc(U);\n\
            match c {{ MkRawCell(p, v) => let v2 = winit(p, v, 41); vread(p, v2) + 1 }}\n\
        }}\n"
    );
    check_program(&ok).unwrap_or_else(|e| panic!("ralloc/winit/vread must check: {e:?}"));
    // a never-initialized cell can be reclaimed
    let free_raw = format!(
        "{BN}main : Nat\nfn main() {{\n\
            let c : RawCell Nat = ralloc(U);\n\
            match c {{ MkRawCell(p, v) => let u = rfree(p, v); 7 }}\n\
        }}\n"
    );
    check_program(&free_raw).unwrap_or_else(|e| panic!("rfree on a raw cell must check: {e:?}"));

    // READING RAW IS A TYPE ERROR: no read/write op accepts `RawTo`.
    for bad_body in [
        "match c { MkRawCell(p, v) => vread(p, v) }",                    // read raw
        "match c { MkRawCell(p, v) => let (x, vh) = vtake(p, v); x }",   // take raw
        "match c { MkRawCell(p, v) => let u = vfree(p, v); 0 }",         // vfree wants PtsTo
        "match c { MkRawCell(p, v) => let v2 = vwrite(p, v, 3); vread(p, v2) }", // vwrite wants PtsTo
    ] {
        let bad = format!(
            "{BN}main : Nat\nfn main() {{ let c : RawCell Nat = ralloc(U);\n {bad_body} }}\n"
        );
        let err = check_program(&bad).err().unwrap_or_else(|| {
            panic!("P0 BUG: an op consumed a RAW (uninitialized) cell:\n{bad_body}")
        });
        assert!(format!("{err:?}").contains("RawTo"), "expected a RawTo type clash, got {err:?}");
    }

    // the raw permission is LINEAR: dropping it leaks, double-init double-uses.
    let leak = format!(
        "{BN}main : Nat\nfn main() {{ let c : RawCell Nat = ralloc(U); match c {{ MkRawCell(p, v) => 7 }} }}\n"
    );
    assert!(check_program(&leak).is_err(), "dropping a RawTo must be rejected (leak)");
    let dbl = format!(
        "{BN}main : Nat\nfn main() {{ let c : RawCell Nat = ralloc(U);\n\
            match c {{ MkRawCell(p, v) => let a = winit(p, v, 1); let b = winit(p, v, 2); vread(p, a) + vread(p, b) }} }}\n"
    );
    assert!(check_program(&dbl).is_err(), "initializing twice through one RawTo must be rejected");
}

#[test]
fn a3_dropping_destructor_gate_closes_deep_leaks() {
    const BN: &str = "enum Nat { Zero : Nat, Succ : Nat -> Nat }\n";
    // `free` on a cell whose PAYLOAD is linear would silently leak the inner
    // resource — the A3 audit found this compiled. Now rejected.
    let nested = format!(
        "{BN}main : Nat\nfn main() {{ let o = alloc(alloc(Zero)); let u = free(o); Zero }}\n"
    );
    let err = check_program(&nested).err().expect("free of Own (Own _) must be rejected");
    assert!(format!("{err:?}").contains("LEAK"), "got {err:?}");
    let vnested = format!(
        "{BN}main : Nat\nfn main() {{ match valloc(alloc(Zero)) {{ MkCell(p, v) => let u = vfree(p, v); Zero }} }}\n"
    );
    let err = check_program(&vnested).err().expect("vfree of a linear payload must be rejected");
    assert!(format!("{err:?}").contains("LEAK"), "got {err:?}");
    // the CORRECT pattern stays accepted: consume the payload first.
    let ok = format!(
        "{BN}main : Nat\nfn main() {{ let o = alloc(alloc(Zero)); let inner = unbox(o); let x = unbox(inner); x }}\n"
    );
    check_program(&ok).unwrap_or_else(|e| panic!("unbox-then-consume must check: {e:?}"));
}

// ═══════════════════════════════════════════════════════════════════════════
// PHASE A2 (docs/PHASE_SAFETY_PLAN.md) — shared read-only borrows. The
// counting discipline: `share` splits an Own into ω address + linear SRead
// token + linear SLoan; `sdup`/`sjoin` split/merge tokens; `sread` reads and
// hands the token back; `unshare` needs the SINGLE remaining token + loan.
// So no reader survives reunification, and &mut/free are unrepresentable
// while any token is outstanding — by the ordinary QTT accounting.
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn a2_shared_borrows_coexist_and_reunify() {
    const BN: &str = "%builtin Nat Nat\nenum Nat { Zero : Nat, Succ : Nat -> Nat }\n";
    // many & coexist: two split tokens read the same cell, rejoin, reunify.
    let ok = format!(
        "{BN}main : Nat\nfn main() {{\n\
            let o = alloc(21);\n\
            match share(o) {{ MkShared(p, s, ln) =>\n\
                match sdup(p, s) {{ MkSPair(p2, s1, s2) =>\n\
                    let (x, s1b) = sread(p2, s1);\n\
                    let (y, s2b) = sread(p2, s2);\n\
                    let s = sjoin(s1b, s2b);\n\
                    let o2 = unshare(p2, s, ln);\n\
                    let z = unbox(o2);\n\
                    x + y + z, }}, }}\n\
        }}\n"
    );
    check_program(&ok).unwrap_or_else(|e| panic!("shared reads + reunify must check: {e:?}"));

    // free (or any &mut path needing the Own) cannot coexist with a live share
    let free_under = format!(
        "{BN}main : Nat\nfn main() {{\n\
            let o = alloc(21);\n\
            match share(o) {{ MkShared(p, s, ln) =>\n\
                let u = free(o);\n\
                let o2 = unshare(p, s, ln); unbox(o2), }}\n\
        }}\n"
    );
    let err = check_program(&free_under).err().expect("free under a live share must reject");
    assert!(format!("{err:?}").contains("ω"), "expected a double-use error, got {err:?}");

    // unshare with an OUTSTANDING token strands it: 0 ⋢ 1
    let outstanding = format!(
        "{BN}main : Nat\nfn main() {{\n\
            let o = alloc(21);\n\
            match share(o) {{ MkShared(p, s, ln) =>\n\
                match sdup(p, s) {{ MkSPair(p2, s1, s2) =>\n\
                    let o2 = unshare(p2, s1, ln); unbox(o2), }}, }}\n\
        }}\n"
    );
    let err = check_program(&outstanding)
        .err()
        .expect("unshare while a second token is outstanding must reject");
    assert!(format!("{err:?}").contains("0 ⋢ 1") || format!("{err:?}").contains("0 time"),
        "expected a stranded-token error, got {err:?}");

    // a LINEAR payload cannot be shared (sread would duplicate it)
    let linear = format!(
        "{BN}main : Nat\nfn main() {{\n\
            let o = alloc(alloc(3));\n\
            match share(o) {{ MkShared(p, s, ln) =>\n\
                let (x, s2) = sread(p, s);\n\
                let o2 = unshare(p, s2, ln);\n\
                let inner = unbox(o2); let v = unbox(inner); let w = unbox(x); v + w, }}\n\
        }}\n"
    );
    let err = check_program(&linear).err().expect("sharing a linear payload must reject");
    assert!(format!("{err:?}").contains("LINEAR"), "got {err:?}");
}

// ═══════════════════════════════════════════════════════════════════════════
// PHASE B1 / E3 (docs/PHASE_SAFETY_PLAN.md) — well-founded recursion by
// RUNTIME WITNESS: a self-call whose measure argument is guarded by the
// `DYes` arm of `match dlt(new, m)` (m the matched measure parameter) is
// certified `%total` — dlt IS the machine compare, so the measure strictly
// decreased whenever the branch runs, and `<` on the packed machine Nat is
// well-founded. Lowered as `Fix` (verdict `TotalWf`).
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn b1_wf_recursion_gcd_certified_total() {
    const BN: &str = "%builtin Nat Nat\nenum Nat { Zero : Nat, Succ : Nat -> Nat }\n";
    const GCD: &str = "gcd : Nat -> Nat -> Nat\n%total\nfn gcd(a, b) {\n\
        match b {\n\
            Zero => a,\n\
            Succ(k) =>\n\
                match dlt(mod(a, b), b) {\n\
                    DYes(p) => gcd(b, mod(a, b)),\n\
                    DNo => Succ(k),\n\
                },\n\
        }\n\
    }\n";
    let src = format!("{BN}{GCD}main : Nat\nfn main() {{ gcd(48, 18) }}\n");
    let prog = check_program(&src).unwrap_or_else(|e| panic!("wf gcd must check as %total: {e:?}"));
    let (_, total, reason) = prog
        .totality
        .iter()
        .find(|(n, _, _)| n == "gcd")
        .expect("gcd in totality report");
    assert!(*total, "gcd must be CERTIFIED total, got: {reason:?}");

    // annotation is NOT proof: a non-decreasing call under the same dlt guard
    // (measure argument is `b` itself, not the dlt-witnessed smaller value).
    let bad = format!(
        "{BN}bad : Nat -> Nat -> Nat\n%total\nfn bad(a, b) {{\n\
            match b {{ Zero => a, Succ(k) => match dlt(mod(a, b), b) {{ DYes(p) => bad(b, b), DNo => Succ(k) }} }}\n\
        }}\nmain : Nat\nfn main() {{ bad(1, 2) }}\n"
    );
    let err = check_program(&bad).err().expect("non-decreasing wf call must be rejected");
    assert!(format!("{err:?}").contains("does not decrease"), "got {err:?}");

    // SHADOWING kills the fact: rebinding a variable the witnessed expression
    // mentions between the guard and the call must drop the certificate.
    let shadow = format!(
        "{BN}bad : Nat -> Nat -> Nat\n%total\nfn bad(a, b) {{\n\
            match b {{ Zero => a, Succ(k) =>\n\
                match dlt(mod(a, b), b) {{\n\
                    DYes(p) => let a = b + b; bad(b, mod(a, b)),\n\
                    DNo => Succ(k),\n\
                }} }}\n\
        }}\nmain : Nat\nfn main() {{ bad(1, 2) }}\n"
    );
    let err = check_program(&shadow).err().expect("a shadowed witness must not certify");
    assert!(format!("{err:?}").contains("does not decrease"), "got {err:?}");

    // the guard must witness the MATCHED measure parameter — dlt against some
    // other variable proves nothing about the measure.
    let wrong_bound = format!(
        "{BN}bad : Nat -> Nat -> Nat\n%total\nfn bad(a, b) {{\n\
            match b {{ Zero => a, Succ(k) => match dlt(mod(a, b), a) {{ DYes(p) => bad(a, mod(a, b)), DNo => Succ(k) }} }}\n\
        }}\nmain : Nat\nfn main() {{ bad(1, 2) }}\n"
    );
    let err = check_program(&wrong_bound).err().expect("a wrong-bound dlt must not certify");
    assert!(format!("{err:?}").contains("does not decrease"), "got {err:?}");
}

#[test]
fn b1_qsort_total_example_fully_certified() {
    // THE B1 GATE: the full in-place quicksort — fill, Lomuto partition,
    // divide-and-conquer driver, sorted-check — every function CERTIFIED
    // %total via dlt-witnessed descent. (The native 30k run + answer parity
    // with the %partial twin is dep_codegen::tests::b1_qsort_total_runs.)
    // the 1M literals make the kernel recurse deeply — use a CLI-sized stack.
    let prog = std::thread::Builder::new()
        .stack_size(1 << 28)
        .spawn(|| {
            let src = std::fs::read_to_string("examples/qsort_total.tal").unwrap();
            check_program(&src).unwrap_or_else(|e| panic!("qsort_total must check: {e:?}"))
        })
        .unwrap()
        .join()
        .unwrap();
    for (name, total, reason) in &prog.totality {
        assert!(total, "`{name}` must be certified total, got: {reason:?}");
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// PHASE A4 (docs/PHASE_SAFETY_PLAN.md) — concurrency as a linear-typed
// library. `spawn` MOVES a one-slot linear env into a fresh OS thread (the
// work fn is a closed top-level fn — the surface has no lambdas, so nothing
// can be captured); `join` recovers the linear result. Data-race freedom is
// the ordinary QTT accounting: no new checker, no new kernel rule.
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn a4_spawn_linearity_makes_races_unwritable() {
    const BN: &str = "%builtin Nat Nat\nenum Nat { Zero : Nat, Succ : Nat -> Nat }\n";
    const WORKER: &str = "worker : (1 o : Own Nat) -> Own Nat\n\
                          fn worker(o) { let x = unbox(o); alloc(x + x) }\n";
    // the safe move-in/join-out program CHECKS
    let ok = format!(
        "{BN}{WORKER}main : Nat\nfn main() {{\n\
            let env = alloc(21);\n\
            let h : JoinHandle (Own Nat) = spawn(worker, env);\n\
            let r = join(h);\n\
            unbox(r)\n\
        }}\n"
    );
    check_program(&ok).unwrap_or_else(|e| panic!("spawn/join must check: {e:?}"));

    // touching the env after it was MOVED into the thread: ω ⋢ 1
    let use_after_move = format!(
        "{BN}{WORKER}main : Nat\nfn main() {{\n\
            let env = alloc(21);\n\
            let h : JoinHandle (Own Nat) = spawn(worker, env);\n\
            let x = unbox(env);\n\
            let r = join(h);\n\
            unbox(r) + x\n\
        }}\n"
    );
    let err = check_program(&use_after_move).err().expect("env use-after-move must reject");
    assert!(format!("{err:?}").contains("ω"), "got {err:?}");

    // giving the SAME moved state to two threads: ω ⋢ 1
    let two_threads_one_env = format!(
        "{BN}{WORKER}main : Nat\nfn main() {{\n\
            let env = alloc(21);\n\
            let h1 : JoinHandle (Own Nat) = spawn(worker, env);\n\
            let h2 : JoinHandle (Own Nat) = spawn(worker, env);\n\
            let r1 = join(h1); let r2 = join(h2);\n\
            unbox(r1) + unbox(r2)\n\
        }}\n"
    );
    let err = check_program(&two_threads_one_env)
        .err()
        .expect("one env into two threads must reject");
    assert!(format!("{err:?}").contains("ω"), "got {err:?}");

    // a JoinHandle is linear: never joining (losing the thread's resource) leaks
    let dropped = format!(
        "{BN}{WORKER}main : Nat\nfn main() {{\n\
            let env = alloc(21);\n\
            let h : JoinHandle (Own Nat) = spawn(worker, env);\n\
            7\n\
        }}\n"
    );
    let err = check_program(&dropped).err().expect("a dropped JoinHandle must reject");
    assert!(format!("{err:?}").contains("0 time"), "got {err:?}");
}

#[test]
fn a4_slices_split_disjoint_and_cannot_free() {
    const BN: &str = "%builtin Nat Nat\nenum Nat { Zero : Nat, Succ : Nat -> Nat }\n";
    // afree on a HALF is a type error (Slice has no free — freeing a
    // non-base pointer would corrupt the heap; the only way out is ajoin).
    let free_half = format!(
        "{BN}main : Nat\nfn main() {{\n\
            let a0 = anew(8, 5);\n\
            match asplit(4, 4, a0) {{ MkASplit(lo, hi, rj) => let u = afree(hi); 0 }}\n\
        }}\n"
    );
    let err = check_program(&free_half).err().expect("afree of a slice must reject");
    assert!(format!("{err:?}").contains("Slice"), "got {err:?}");
    // dropping the Rejoin obligation (never reuniting) leaks: 0 ⋢ 1 — the
    // whole allocation would be lost (slices cannot free it).
    let no_rejoin = format!(
        "{BN}sink : (1 s : Slice Nat 4) -> Nat\n%partial fn sink(s) {{ match sget(0, lt(0, 4), s) {{ MkSliceRead(x, s2) => x + sink(s2) }} }}\n\
         main : Nat\nfn main() {{\n\
            let a0 = anew(8, 5);\n\
            match asplit(4, 4, a0) {{ MkASplit(lo, hi, rj) => sink(lo) + sink(hi) }}\n\
        }}\n"
    );
    assert!(check_program(&no_rejoin).is_err(), "stranding the Rejoin obligation must reject");
}
