//! End-to-end tests that exercise read → expand → compile → extend → run.

use microlisp::Engine;
use microlisp::value::*;

fn run(src: &str) -> (Engine, u64) {
    let mut e = Engine::new();
    let r = e.run_source(src);
    (e, r)
}

fn run_num(src: &str) -> f64 {
    let (_e, r) = run(src);
    assert!(is_number(r), "expected number, got 0x{:016x}", r);
    as_number(r)
}

#[test]
fn arithmetic() {
    assert_eq!(run_num("(+ 1 2)"), 3.0);
    assert_eq!(run_num("(* (+ 1 2) (- 5 1))"), 12.0);
    assert_eq!(run_num("(/ 10 4)"), 2.5);
}

#[test]
fn quote_and_cons() {
    let (e, r) = run("(car '(1 2 3))");
    assert_eq!(as_number(r), 1.0);
    drop(e);

    let (e, r) = run("(car (cdr '(1 2 3)))");
    assert_eq!(as_number(r), 2.0);
    drop(e);

    let (_e, r) = run("(cons 1 (cons 2 nil))");
    assert!(is_cons(r));
    assert_eq!(as_number(car(r)), 1.0);
}

#[test]
fn if_form() {
    assert_eq!(run_num("(if #t 1 2)"), 1.0);
    assert_eq!(run_num("(if #f 1 2)"), 2.0);
    assert_eq!(run_num("(if nil 1 2)"), 2.0);
    assert_eq!(run_num("(if 0 1 2)"), 1.0); // 0 is truthy in our scheme
}

#[test]
fn user_define_and_call() {
    let (_e, r) = run("(define (double x) (* x 2)) (double 21)");
    assert_eq!(as_number(r), 42.0);
}

#[test]
fn recursive_factorial() {
    let (_e, r) = run(
        "(define (fact n) (if (<= n 1) 1 (* n (fact (- n 1)))))
         (fact 10)",
    );
    assert_eq!(as_number(r), 3628800.0);
}

#[test]
fn let_and_set() {
    let (_e, r) = run("(let ((x 1) (y 2)) (+ x y))");
    assert_eq!(as_number(r), 3.0);

    let (_e, r) = run("(let ((x 10)) (set! x 99) x)");
    assert_eq!(as_number(r), 99.0);
}

#[test]
fn equal_predicate() {
    let (_e, r) = run("(equal? '(1 2 3) '(1 2 3))");
    assert_eq!(r, TRUE);
    let (_e, r) = run("(equal? '(1 2) '(1 2 3))");
    assert_eq!(r, FALSE);
    let (_e, r) = run("(eq? 'foo 'foo)");
    assert_eq!(r, TRUE);
}

// ─── Macros ────────────────────────────────────────────────────────

#[test]
fn macro_when() {
    let (_e, r) = run(
        "(defmacro when (cond . body)
           `(if ,cond (begin ,@body) nil))
         (when #t 42)",
    );
    assert_eq!(as_number(r), 42.0);
}

#[test]
fn macro_when_skips_when_false() {
    let (_e, r) = run(
        "(defmacro when (cond . body)
           `(if ,cond (begin ,@body) nil))
         (when #f 42)",
    );
    assert_eq!(r, NIL);
}

#[test]
fn macro_unless() {
    let (_e, r) = run(
        "(defmacro unless (cond . body)
           `(if ,cond nil (begin ,@body)))
         (unless #f 7)",
    );
    assert_eq!(as_number(r), 7.0);
}

#[test]
fn macro_calls_macro_when_uses_if_directly() {
    // Here `when` expands to (if ...). After expansion the `when` macro
    // sees `(when (...))` already gone. This proves the recursive walk.
    let (_e, r) = run(
        "(defmacro when (cond . body)
           `(if ,cond (begin ,@body) nil))
         (define (positive? x) (when (> x 0) #t))
         (if (positive? 5) 1 0)",
    );
    assert_eq!(as_number(r), 1.0);
}

#[test]
fn macro_recursive_let_star() {
    // let* expands to nested let. Recursive macro that re-emits a use of itself.
    let (_e, r) = run(
        "(defmacro let* (bindings . body)
           (if (null? bindings)
               `(begin ,@body)
               `(let (,(car bindings))
                  (let* ,(cdr bindings) ,@body))))
         (let* ((a 1) (b (+ a 1)) (c (* b 3))) c)",
    );
    assert_eq!(as_number(r), 6.0);
}

#[test]
fn macro_twice_doubles_evaluation() {
    // (twice e) expands to (begin e e). Calling it with a side-effecting
    // expression should evaluate it twice; we test that the *value* of the
    // begin is the second occurrence by using a let-bound counter.
    let (_e, r) = run(
        "(defmacro twice (e)
           `(begin ,e ,e))
         (let ((n 0))
           (twice (set! n (+ n 1)))
           n)",
    );
    assert_eq!(as_number(r), 2.0);
}

#[test]
fn macro_helper_compute_at_expansion_time() {
    // Macro calls a helper function during expansion — the helper computes
    // a literal value baked into the result. Genuine compile-time execution
    // of user code through the JIT.
    let (_e, r) = run(
        "(define (square n) (* n n))
         (defmacro static-square (n)
           `(quote ,(square n)))
         (static-square 7)",
    );
    assert_eq!(as_number(r), 49.0);
}
