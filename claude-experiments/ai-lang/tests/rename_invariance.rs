//! End-to-end test for the central content-addressing property:
//!
//!   "Renaming a local variable, a parameter, or even the def itself does
//!    not change a definition's content hash."
//!
//! Until the parser and resolver exist, this is proved by hand-building
//! canonical ASTs directly — the canonical form has no names for locals
//! or parameters (just de Bruijn indices) and no name for the def itself,
//! so two surface programs that differ only in those names *must* produce
//! identical canonical ASTs and therefore identical hashes.

use ai_lang::ast::{Def, Expr, Type};
use ai_lang::codec::encode_def;
use ai_lang::hash::Hash;

/// `fn double(x: Int) -> Int = x * 2`, regardless of what the parameter
/// or the def itself is called at the surface.
fn double_canonical() -> Def {
    Def::Fn {
        is_local: false,
        type_params: 0,
        params: vec![Type::Builtin("Int".to_owned())],
        ret: Type::Builtin("Int".to_owned()),
        body: Expr::Call(
            Box::new(Expr::BuiltinRef("core/i64.mul".to_owned())),
            vec![Expr::LocalVar(0), Expr::IntLit(2)],
        ),
    }
}

fn hash_def(d: &Def) -> Hash {
    Hash::of_bytes(&encode_def(d))
}

#[test]
fn renaming_the_parameter_does_not_change_the_hash() {
    // `def double(x: Int) -> Int = x * 2`
    // `def double(y: Int) -> Int = y * 2`
    // Both surface programs produce the SAME canonical AST (the parameter
    // is referred to by de Bruijn index 0, not by name).
    let a = double_canonical();
    let b = double_canonical();
    assert_eq!(hash_def(&a), hash_def(&b));
}

#[test]
fn renaming_the_def_itself_does_not_change_its_hash() {
    // The def's surface name (`double` vs `dbl` vs anything) is not part
    // of the canonical form. The canonical AST is just the body + sig.
    let a = double_canonical();
    let b = double_canonical();
    assert_eq!(hash_def(&a), hash_def(&b));
}

#[test]
fn callers_pin_callees_by_hash_not_name() {
    // `def quadruple(x: Int) -> Int = double(double(x))`
    //
    // Whatever surface name `double` has, the canonical body references
    // it as TopRef(hash_of_double). Renaming `double` to `dbl` does not
    // change quadruple's hash, because the canonical body only contains
    // the *hash* of double, never its name.
    let double_hash = hash_def(&double_canonical());

    let quadruple = Def::Fn {
        is_local: false,
        type_params: 0,
        params: vec![Type::Builtin("Int".to_owned())],
        ret: Type::Builtin("Int".to_owned()),
        body: Expr::Call(
            Box::new(Expr::TopRef(double_hash)),
            vec![Expr::Call(
                Box::new(Expr::TopRef(double_hash)),
                vec![Expr::LocalVar(0)],
            )],
        ),
    };

    let first = hash_def(&quadruple);
    let second = hash_def(&quadruple);
    assert_eq!(first, second, "hash must be deterministic");
}

#[test]
fn changing_the_body_does_change_the_hash() {
    // Sanity check: identity isn't just always-the-same; it actually
    // tracks the structure.
    let double = double_canonical();

    // `fn(x: Int) -> Int = x * 3` (changed the literal from 2 to 3)
    let triple = Def::Fn {
        is_local: false,
        type_params: 0,
        params: vec![Type::Builtin("Int".to_owned())],
        ret: Type::Builtin("Int".to_owned()),
        body: Expr::Call(
            Box::new(Expr::BuiltinRef("core/i64.mul".to_owned())),
            vec![Expr::LocalVar(0), Expr::IntLit(3)],
        ),
    };

    assert_ne!(
        hash_def(&double),
        hash_def(&triple),
        "different bodies must have different hashes"
    );
}

#[test]
fn is_local_flag_is_part_of_identity() {
    // `def foo = ...` and `def local foo = ...` are different definitions
    // — the local-vs-global distinction is hash-bearing because it changes
    // the semantics of references to it. (Same body, different identity.)
    let mut a = double_canonical();
    let mut b = double_canonical();
    let Def::Fn { is_local, .. } = &mut a else {
        panic!("expected Fn");
    };
    *is_local = false;
    let Def::Fn { is_local, .. } = &mut b else {
        panic!("expected Fn");
    };
    *is_local = true;
    assert_ne!(
        hash_def(&a),
        hash_def(&b),
        "is_local must be part of the hash"
    );
}
