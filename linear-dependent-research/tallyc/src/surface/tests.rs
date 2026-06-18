use super::*;
use crate::dep::Term;

// helpers to build expected normal forms (kernel Terms)
fn ndata() -> Term {
    Term::Data("N".into(), vec![])
}
fn z() -> Term {
    Term::Constr("z".into(), vec![])
}
fn s(t: Term) -> Term {
    Term::Constr("s".into(), vec![t])
}
fn num(k: u64) -> Term {
    let mut t = z();
    for _ in 0..k {
        t = s(t);
    }
    t
}

const NAT: &str = r#"
data N where
  | z : N
  | s : N -> N

def add : (m : N) -> (n : N) -> N
  = fun m n => elim m to (fun _ => N) {
      | z      => n ;
      | s k ih => s ih
    }
"#;

#[test]
fn nat_and_add_in_surface_syntax() {
    let src = format!("{NAT}\ndef main : N = add (s (s z)) (s (s (s z)))\n");
    let prog = check_program(&src).unwrap_or_else(|e| panic!("{e:?}"));
    assert_eq!(prog.normalize("main"), Some(num(5)));
}

#[test]
fn a_proof_in_surface_syntax() {
    // refl : Eq N (add 1 1) 2   — discharged by add's own computation
    let src = format!(
        "{NAT}\ndef proof : Eq N (add (s z) (s z)) (s (s z)) = refl (add (s z) (s z))\n"
    );
    assert!(check_program(&src).is_ok(), "{:?}", check_program(&src).err());

    // the false equation is rejected
    let bad =
        format!("{NAT}\ndef proof : Eq N (add (s z) (s z)) (s (s (s z))) = refl (add (s z) (s z))\n");
    assert!(check_program(&bad).is_err());
}

const VEC: &str = r#"
data N where
  | z : N
  | s : N -> N

def add : (m : N) -> (n : N) -> N
  = fun m n => elim m to (fun _ => N) {
      | z      => n ;
      | s k ih => s ih
    }

data Vec (A : Type) : N -> Type where
  | nil  : Vec A z
  | cons : (0 k : N) -> A -> Vec A k -> Vec A (s k)

def append : (0 A : Type) -> (0 m : N) -> (0 n : N)
           -> Vec A m -> Vec A n -> Vec A (add m n)
  = fun A m n xs ys =>
      elim xs to (fun k _ => Vec A (add k n)) {
        | nil            => ys ;
        | cons k h t ih  => cons A (add k n) h ih
      }
"#;

#[test]
fn dependent_vectors_and_append_in_surface_syntax() {
    // append [1] [2] : Vec N 2  ↝  [1,2]   (length tracked in the type)
    let src = format!(
        "{VEC}\ndef vmain : Vec N (s (s z)) \
         = append N (s z) (s z) (cons N z (s z) (nil N)) (cons N z (s (s z)) (nil N))\n"
    );
    let prog = check_program(&src).unwrap_or_else(|e| panic!("{e:?}"));

    let nil = Term::Constr("nil".into(), vec![ndata()]);
    let cons = |k: Term, h: Term, t: Term| Term::Constr("cons".into(), vec![ndata(), k, h, t]);
    let expected = cons(s(z()), s(z()), cons(z(), s(s(z())), nil));
    assert_eq!(prog.normalize("vmain"), Some(expected));

    // a length mismatch in the declared type is rejected
    let bad = format!(
        "{VEC}\ndef vmain : Vec N (s (s (s z))) \
         = append N (s z) (s z) (cons N z (s z) (nil N)) (cons N z (s (s z)) (nil N))\n"
    );
    assert!(check_program(&bad).is_err());
}

const FIN: &str = r#"
data N where
  | z : N
  | s : N -> N

data Fin : N -> Type where
  | fz : (n : N) -> Fin (s n)
  | fs : (n : N) -> Fin n -> Fin (s n)

def fin2nat : (0 n : N) -> Fin n -> N
  = fun n i => elim i to (fun m _ => N) {
      | fz m         => z ;
      | fs m prev ih => s ih
    }
"#;

#[test]
fn fin_and_fin2nat_in_surface_syntax() {
    // the element "1" of Fin 2:  fs 1 (fz 0)  ;  fin2nat ↝ 1
    let src = format!("{FIN}\ndef fmain : N = fin2nat (s (s z)) (fs (s z) (fz z))\n");
    let prog = check_program(&src).unwrap_or_else(|e| panic!("{e:?}"));
    assert_eq!(prog.normalize("fmain"), Some(num(1)));
}

#[test]
fn strict_positivity_is_rejected_in_surface_syntax() {
    let src = r#"
data Bad where
  | mk : (Bad -> Bad) -> Bad
"#;
    assert!(check_program(src).is_err());
}

#[test]
fn ill_typed_definitions_are_rejected() {
    // a lambda where an N is expected
    let src = format!("{NAT}\ndef bad : N = fun x => x\n");
    assert!(check_program(&src).is_err());
    // wrong arity for a constructor
    let src2 = format!("{NAT}\ndef bad : N = s\n");
    assert!(check_program(&src2).is_err());
}

#[test]
fn parse_errors_are_reported() {
    assert!(check_program("def x : = z").is_err());
    assert!(check_program("data").is_err());
}
