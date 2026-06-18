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
    Cons : (0 k : Nat) -> a -> Vec a k -> Vec a (Succ k),
}

append : (0 a : Type) -> (0 m : Nat) -> (0 n : Nat) -> Vec a m -> Vec a n -> Vec a (add m n)
fn append(a, m, n, xs, ys) {
    match xs {
        Nil           => ys,
        Cons(k, h, t) => Cons(a, add(k, n), h, append(a, k, n, t, ys)),
    }
}
"#;

#[test]
fn dependent_vectors_and_append() {
    // append [1] [2] : Vec Nat 2  ↝  [1,2]
    let src = format!(
        "{VEC}\nmain : Vec Nat (Succ (Succ Zero))\n\
         fn main() {{ append(Nat, Succ(Zero), Succ(Zero), \
         Cons(Nat, Zero, Succ(Zero), Nil(Nat)), \
         Cons(Nat, Zero, Succ(Succ(Zero)), Nil(Nat))) }}\n"
    );
    let prog = check_program(&src).unwrap_or_else(|e| panic!("{e:?}"));

    let nil = Term::Constr("Nil".into(), vec![ndata()]);
    let cons = |k: Term, h: Term, t: Term| Term::Constr("Cons".into(), vec![ndata(), k, h, t]);
    let expected = cons(succ(zero()), succ(zero()), cons(zero(), succ(succ(zero())), nil));
    assert_eq!(prog.normalize("main"), Some(expected));

    // wrong declared length is rejected
    let bad = format!(
        "{VEC}\nmain : Vec Nat (Succ Zero)\n\
         fn main() {{ append(Nat, Succ(Zero), Succ(Zero), \
         Cons(Nat, Zero, Succ(Zero), Nil(Nat)), \
         Cons(Nat, Zero, Succ(Succ(Zero)), Nil(Nat))) }}\n"
    );
    assert!(check_program(&bad).is_err());
}

const FIN: &str = r#"
enum Nat {
    Zero : Nat,
    Succ : Nat -> Nat,
}

enum Fin : Nat -> Type {
    FZ : (0 k : Nat) -> Fin (Succ k),
    FS : (0 k : Nat) -> Fin k -> Fin (Succ k),
}

fin_to_nat : (0 n : Nat) -> Fin n -> Nat
fn fin_to_nat(n, i) {
    match i {
        FZ(k)       => Zero,
        FS(k, prev) => Succ(fin_to_nat(k, prev)),
    }
}
"#;

#[test]
fn fin_and_fin_to_nat() {
    // the element "1" of Fin 2  ↝  1
    let src = format!(
        "{FIN}\nmain : Nat\nfn main() {{ fin_to_nat(Succ(Succ(Zero)), FS(Succ(Zero), FZ(Zero))) }}\n"
    );
    let prog = check_program(&src).unwrap_or_else(|e| panic!("{e:?}"));
    assert_eq!(prog.normalize("main"), Some(num(1)));
}

#[test]
fn structs_elaborate_and_construct() {
    let src = r#"
enum Nat { Zero : Nat, Succ : Nat -> Nat }
struct Box (a : Type) { val : a }
mk : Box Nat
fn mk() { Box(Nat, Succ(Zero)) }
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
    let err = check_program(&src).err();
    assert!(err.is_some(), "expected rejection");
}

#[test]
fn parse_and_name_errors() {
    assert!(check_program("fn f( {").is_err());
    assert!(check_program("enum Nat { Zero : Nat }\nfn g() { Bogus(Zero) }").is_err());
}
