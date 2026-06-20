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

enum Vec (a : Type) : Nat -> Type {
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

enum Fin : Nat -> Type {
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
fn explore_wtype_construction_and_recursion() {
    // EXPLORATORY (1b scoping): a W-type `Tree` with a higher-order recursive field
    // `node2 : (Bool -> Tree) -> Tree`. Construct a value by passing a NAMED helper
    // (no surface lambdas needed), then recurse over it.
    let src = format!(
        "{NATB}\n\
         enum Bool {{ btrue : Bool, bfalse : Bool }}\n\
         enum Tree {{ leaf : Tree, node2 : (Bool -> Tree) -> Tree }}\n\
         add : Nat -> Nat -> Nat\n\
         fn add(m, n) {{ match m {{ Zero => n, Succ(k) => Succ(add(k, n)) }} }}\n\
         kids : Bool -> Tree\nfn kids(b) {{ leaf }}\n\
         t1 : Tree\nfn t1() {{ node2(kids) }}\n\
         size : Tree -> Nat\n\
         fn size(t) {{ match t {{ leaf => Succ(Zero), node2(f) => add(size(f(btrue)), size(f(bfalse))) }} }}\n\
         main : Nat\nfn main() {{ size(t1) }}\n"
    );
    match check_program(&src) {
        Ok(prog) => {
            eprintln!("OK; is_total(size)={}", is_total(&prog, "size"));
            eprintln!("main = {:?}", prog.normalize("main"));
        }
        Err(e) => eprintln!("ERR: {e:?}"),
    }
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
fn non_structural_recursion_is_rejected() {
    let src = format!(
        "{}\nloop : Nat -> Nat\nfn loop(m) {{ match m {{ Zero => Zero, Succ(k) => loop(Succ(k)) }} }}\n",
        r#"enum Nat { Zero : Nat, Succ : Nat -> Nat }"#
    );
    assert!(check_program(&src).is_err());
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
enum LT : Nat -> Nat -> Type {
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
    // syntax and certified total + run natively.
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
    // certified `%total` and run natively. div(10, 7, 2) = 3.
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
    enum Fin : Nat -> Type {\n\
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
    enum Tree { leaf : Tree, node : Tree -> Tree -> Tree }\n\
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
