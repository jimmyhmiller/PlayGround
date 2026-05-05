//! End-to-end proof that microlisp's quote-literals go through the
//! GC-traced LiteralPool — and that emitted code reads the *current*
//! slot value on every access, so a moving GC could rewrite the pool
//! without patching code.

use dynobj::RootSource;
use microlisp::Engine;
use microlisp::value::*;

#[test]
fn cons_quote_literal_routes_through_pool() {
    let mut e = Engine::new();
    // Define a function whose body returns the cons quote-literal '(1 2 3).
    e.run_source("(define (lit) '(1 2 3))");
    // Pool should now have at least one entry — the cons cell.
    assert!(e.literal_pool().len() >= 1, "expected at least one literal in pool");

    // Calling the function should return a cons whose car/cdr unwrap to 1, 2, 3.
    let r = e.run_source("(lit)");
    assert!(is_cons(r));
    assert_eq!(as_number(car(r)), 1.0);
    assert_eq!(as_number(car(cdr(r))), 2.0);
    assert_eq!(as_number(car(cdr(cdr(r)))), 3.0);
}

#[test]
fn quote_load_reads_through_pool_not_baked() {
    // Compile a function that returns the literal `'foo` first... wait, foo
    // is a symbol, which is non-pointer (interned in our scheme). Use a
    // cons literal instead and prove that mutating the pool slot changes
    // what subsequent calls return.
    let mut e = Engine::new();
    e.run_source("(define (lit) '(1 2 3))");

    let r1 = e.run_source("(lit)");
    let original_car = car(r1);
    assert_eq!(as_number(original_car), 1.0);

    // Find the pool slot holding our cons literal.
    let pool = e.literal_pool();
    // Snapshot the live slots and locate the one matching the cons of '(1 2 3).
    // We saved that cons during compilation; calling `(lit)` returns it.
    let target_bits = r1; // the live cons
    let mut slot_idx: Option<usize> = None;
    for i in 0..pool.len() {
        if pool.get(i) == target_bits {
            slot_idx = Some(i);
            break;
        }
    }
    let idx = slot_idx.expect("expected cons literal in pool");

    // Build a different cons cell '(99 100) and overwrite the pool slot
    // (simulating a GC having relocated the original to a new address).
    let new_cell = e.with_thread_state(|_h| {
        cons_compile_time(encode_num(99.0), cons_compile_time(encode_num(100.0), NIL))
    });
    pool.set(idx, new_cell);

    // The next call must read the new value through the pool, not the
    // baked-in original.
    let r2 = e.run_source("(lit)");
    assert!(is_cons(r2));
    assert_eq!(as_number(car(r2)), 99.0);
    assert_eq!(as_number(car(cdr(r2))), 100.0);
}

#[test]
fn scan_roots_visits_every_quote_literal() {
    // Multiple cons literals; scan_roots should visit each pool slot.
    let mut e = Engine::new();
    e.run_source("(define (a) '(1 2))");
    e.run_source("(define (b) '(3 4))");
    e.run_source("(define (c) '(5 6))");

    let pool = e.literal_pool();
    assert!(pool.len() >= 3);

    let mut visit_count = 0usize;
    pool.scan_roots(&mut |slot_ptr| {
        let _ = unsafe { *slot_ptr };
        visit_count += 1;
    });
    assert_eq!(visit_count, pool.len());
}

#[test]
fn macro_expansion_quote_literals_also_use_pool() {
    // A macro returns `(quote ,(some-cons-tree)) — the resulting quote-literal
    // is a fresh cons-tree from macro expansion. It should still go through
    // the pool when the surrounding form gets compiled.
    let mut e = Engine::new();
    let pool_before = e.literal_pool().len();
    e.run_source(
        "(defmacro one-two-three () `(quote (1 2 3)))
         (define (m) (one-two-three))",
    );
    let pool_after = e.literal_pool().len();
    assert!(
        pool_after > pool_before,
        "expected macro-emitted quote literal to go through the pool"
    );

    let r = e.run_source("(m)");
    assert!(is_cons(r));
    assert_eq!(as_number(car(r)), 1.0);
}

#[test]
fn non_pointer_literals_stay_inline() {
    // Numbers, nil/true/false, and symbols are immortal/non-pointer values.
    // They should NOT consume pool slots — they're baked as immediates.
    let mut e = Engine::new();
    let pool_before = e.literal_pool().len();
    e.run_source(
        "(define (n) 42)
         (define (s) 'foo)
         (define (b) #t)
         (define (z) nil)",
    );
    let pool_after = e.literal_pool().len();
    assert_eq!(pool_after, pool_before, "non-pointer quote literals should not enter the pool");

    assert_eq!(as_number(e.run_source("(n)")), 42.0);
    assert!(is_symbol(e.run_source("(s)")));
    assert_eq!(e.run_source("(b)"), TRUE);
    assert_eq!(e.run_source("(z)"), NIL);
}
