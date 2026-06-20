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
