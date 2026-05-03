//! End-to-end proof: a real moving GC (SemiSpace) traces the JitModule's
//! `LiteralPool`, relocates quote-literal cons cells to to-space, rewrites
//! the pool slots in place, and the next call to a JIT-compiled function
//! using those literals reads the *new* relocated pointers — without any
//! code patching or recompilation.

use microlisp::Engine;
use microlisp::value::*;

#[test]
fn quote_literal_survives_moving_collection() {
    let mut e = Engine::new();
    e.run_source("(define (lit) '(1 2 3))");

    // Sanity: the function works before any collection.
    let r = e.run_source("(lit)");
    assert!(is_cons(r));
    let pre_ptr = as_cons_ptr(r);
    assert_eq!(as_number(car(r)), 1.0);
    assert_eq!(as_number(car(cdr(r))), 2.0);
    assert_eq!(as_number(car(cdr(cdr(r)))), 3.0);

    // Force a SemiSpace collection. Live cons cells get copied to to-space;
    // the LiteralPool entry is rewritten in place to point at the new
    // address. No code is patched.
    e.collect();

    // Read the same literal again. The emitted load reads the (updated)
    // pool slot and returns the relocated cons cell.
    let r2 = e.run_source("(lit)");
    assert!(is_cons(r2));
    let post_ptr = as_cons_ptr(r2);

    // The pointer should have moved (different from-space → to-space addr).
    assert_ne!(
        pre_ptr, post_ptr,
        "expected GC to relocate the literal — got same pointer 0x{:x}",
        pre_ptr as usize
    );

    // Values are still correct after relocation.
    assert_eq!(as_number(car(r2)), 1.0);
    assert_eq!(as_number(car(cdr(r2))), 2.0);
    assert_eq!(as_number(car(cdr(cdr(r2)))), 3.0);
}

#[test]
fn many_literals_relocate_correctly() {
    // Several quote literals + several collections. Every literal still
    // resolves to its expected value after each collection.
    let mut e = Engine::new();
    e.run_source("(define (a) '(1 2))");
    e.run_source("(define (b) '(3 4 5))");
    e.run_source("(define (c) '(7))");

    for _ in 0..3 {
        e.collect();
        let ra = e.run_source("(a)");
        assert_eq!(as_number(car(ra)), 1.0);
        assert_eq!(as_number(car(cdr(ra))), 2.0);

        let rb = e.run_source("(b)");
        assert_eq!(as_number(car(rb)), 3.0);
        assert_eq!(as_number(car(cdr(rb))), 4.0);
        assert_eq!(as_number(car(cdr(cdr(rb)))), 5.0);

        let rc = e.run_source("(c)");
        assert_eq!(as_number(car(rc)), 7.0);
    }
}

#[test]
fn nested_cons_literal_traced_recursively() {
    // The literal '((1 2) (3 4)) is a cons whose car is itself a cons.
    // The GC must trace the spine AND each car/cdr of the inner cells.
    let mut e = Engine::new();
    e.run_source("(define (nested) '((1 2) (3 4)))");
    e.collect();

    let r = e.run_source("(nested)");
    let first = car(r);
    assert!(is_cons(first));
    assert_eq!(as_number(car(first)), 1.0);
    assert_eq!(as_number(car(cdr(first))), 2.0);

    let second = car(cdr(r));
    assert!(is_cons(second));
    assert_eq!(as_number(car(second)), 3.0);
    assert_eq!(as_number(car(cdr(second))), 4.0);
}

#[test]
fn macro_emitted_quote_literal_survives_collection() {
    // Macro produces a quote-literal in its expansion. After the macro is
    // compiled and used, the resulting code holds a GcLiteral reference.
    // GC across compile/run boundaries should preserve the literal.
    let mut e = Engine::new();
    e.run_source(
        "(defmacro list3 () `(quote (10 20 30)))
         (define (m) (list3))",
    );

    let r1 = e.run_source("(m)");
    let p1 = as_cons_ptr(r1);
    assert_eq!(as_number(car(r1)), 10.0);

    e.collect();

    let r2 = e.run_source("(m)");
    let p2 = as_cons_ptr(r2);
    assert_ne!(p1, p2, "expected literal to relocate");
    assert_eq!(as_number(car(r2)), 10.0);
    assert_eq!(as_number(car(cdr(r2))), 20.0);
    assert_eq!(as_number(car(cdr(cdr(r2)))), 30.0);
}
