//! A non-trivial microlisp program: symbolic differentiation.
//!
//! Exercises every layer at once:
//!   - User-defined `case` macro that calls a user helper (`case-expand`)
//!     at expansion time. Same-image macros: helper lives in the same JIT.
//!   - `cond` macro built recursively from the basics.
//!   - Recursive top-level functions (`deriv`, `member?`, `case-expand`)
//!     each producing fresh cons trees.
//!   - Many quote-literals (`'+`, `'*`, `'(+ x x)`, …) of which only the
//!     cons-shaped ones land in the GC-traced LiteralPool.
//!   - Several thousand cons-cell allocations followed by a forced moving
//!     collection, then more allocations and a verified result.

use microlisp::Engine;
use microlisp::value::*;

const PROGRAM: &str = r#"
;; ── Macros ──────────────────────────────────────────────────────

(defmacro when (cond . body)
  `(if ,cond (begin ,@body) nil))

(defmacro cond clauses
  (if (null? clauses)
      'nil
      (let ((first (car clauses))
            (rest (cdr clauses)))
        (if (eq? (car first) 'else)
            `(begin ,@(cdr first))
            `(if ,(car first)
                 (begin ,@(cdr first))
                 (cond ,@rest))))))

;; ── Helpers used by the case macro at expansion time ────────────
;; Defined BEFORE the `case` macro so the macro body can resolve
;; `case-expand` to a known FuncRef when its body is compiled.

(define (member? x lst)
  (if (null? lst)
      #f
      (if (eq? x (car lst))
          #t
          (member? x (cdr lst)))))

(define (case-expand var clauses)
  (if (null? clauses)
      'nil
      (let ((c (car clauses))
            (rest (cdr clauses)))
        (if (eq? (car c) 'else)
            (cons 'begin (cdr c))
            (list 'if
                  (list 'member? var (list 'quote (car c)))
                  (cons 'begin (cdr c))
                  (case-expand var rest))))))

;; (case scrutinee
;;   ((sym1 sym2) result1)
;;   ((sym3)      result2)
;;   (else        default))
;;
;; The expansion uses `case-expand` (a regular function defined above)
;; at compile time — i.e. the macro body calls into already-JIT'd code.
(defmacro case (scrut . clauses)
  (let ((g (gensym 'case)))
    `(let ((,g ,scrut))
       ,(case-expand g clauses))))

;; ── List helpers ────────────────────────────────────────────────

(define (cadr x)  (car (cdr x)))
(define (caddr x) (car (cdr (cdr x))))

;; ── The actual program: symbolic differentiation ────────────────
;;
;; (deriv expr var)  — returns the (unsimplified) derivative of `expr`
;; with respect to `var`. Supported operators: + and *.
;;
;; Rules:
;;   d(c)/dx       = 0           (c a number)
;;   d(x)/dx       = 1
;;   d(y)/dx       = 0           (y any other symbol)
;;   d(u + v)/dx   = du/dx + dv/dx
;;   d(u * v)/dx   = u*(dv/dx) + (du/dx)*v

(define (deriv expr var)
  (cond
    ((number? expr) 0)
    ((symbol? expr) (if (eq? expr var) 1 0))
    (else
      (case (car expr)
        ((+) (list '+
                   (deriv (cadr expr) var)
                   (deriv (caddr expr) var)))
        ((*) (list '+
                   (list '* (cadr expr)         (deriv (caddr expr) var))
                   (list '* (deriv (cadr expr) var) (caddr expr))))
        (else 'unknown-op)))))
"#;

fn check_deriv_x_x(r: u64) {
    // d(x)/dx = 1
    assert!(is_number(r), "deriv 'x 'x should return number 1");
    assert_eq!(as_number(r), 1.0);
}

fn check_deriv_plus_x_x(r: u64) {
    // d(x + x)/dx = (+ 1 1)
    assert!(is_cons(r));
    assert!(is_symbol(car(r)));
    assert_eq!(as_number(car(cdr(r))), 1.0);
    assert_eq!(as_number(car(cdr(cdr(r)))), 1.0);
}

fn check_deriv_x_times_x(r: u64) {
    // d(x*x)/dx unsimplified = (+ (* x 1) (* 1 x))
    assert!(is_cons(r));
    assert!(is_symbol(car(r))); // '+
    let term1 = car(cdr(r));    // (* x 1)
    let term2 = car(cdr(cdr(r))); // (* 1 x)
    assert!(is_cons(term1));
    assert!(is_cons(term2));
    // (* x 1)
    assert!(is_symbol(car(term1)));         // '*
    assert!(is_symbol(car(cdr(term1))));    // 'x
    assert_eq!(as_number(car(cdr(cdr(term1)))), 1.0);
    // (* 1 x)
    assert!(is_symbol(car(term2)));         // '*
    assert_eq!(as_number(car(cdr(term2))), 1.0);
    assert!(is_symbol(car(cdr(cdr(term2))))); // 'x
}

#[test]
fn symbolic_deriv_with_moving_gc() {
    let mut e = Engine::new();
    e.run_source(PROGRAM);

    // ── Pre-GC: each shape works ────────────────────────────────
    check_deriv_x_x(e.run_source("(deriv 'x 'x)"));
    check_deriv_plus_x_x(e.run_source("(deriv '(+ x x) 'x)"));
    check_deriv_x_times_x(e.run_source("(deriv '(* x x) 'x)"));

    // ── Heavy allocation phase ─────────────────────────────────
    // Each deriv on '(+ (* x x) (* 3 x)) builds ~10 fresh cons cells.
    // 500 iterations of an in-program loop avoids paying the JIT
    // compile cost per iteration; we want allocation pressure, not
    // compile pressure. The fact that the JIT compile itself also
    // allocates (quote literals + macroexpansion temps) is exercised
    // by the surrounding form processing.
    e.run_source(
        "(define (loop-deriv n)
           (if (eq? n 0)
               nil
               (begin
                 (deriv '(+ (* x x) (* 3 x)) 'x)
                 (loop-deriv (- n 1)))))",
    );
    e.run_source("(loop-deriv 500)");

    // ── Force a SemiSpace collection ───────────────────────────
    //
    // What survives:
    //   - LiteralPool entries (the cons-shaped quote literals embedded
    //     in the compiled functions: '(+ x x), '(* x x), '(+) and '(*)
    //     case-clause lists, '(+ (* x x) (* 3 x)) etc.)
    //   - The symbol table (immortal, not on the GC heap).
    //   - The JIT module's compiled code (in PagedCodeMemory, not GC).
    //
    // The pool slots get rewritten in place to point at to-space; emitted
    // `gc_literal` loads pick up the new addresses on the next call.
    e.collect();

    // ── Post-GC: every shape still computes the right answer ──
    check_deriv_x_x(e.run_source("(deriv 'x 'x)"));
    check_deriv_plus_x_x(e.run_source("(deriv '(+ x x) 'x)"));
    check_deriv_x_times_x(e.run_source("(deriv '(* x x) 'x)"));

    // Nested case — the case macro fires on '+ at the outer level and on
    // '* at the inner level. Verifies the macro expansion's inner literal
    // lists ('(+) and '(*)) survived the collection.
    let r = e.run_source("(deriv '(+ (* x x) x) 'x)");
    assert!(is_cons(r));
    // Result is (+ (+ (* x 1) (* 1 x)) 1)
    assert!(is_symbol(car(r))); // '+

    // ── Another heavy phase + collection ──────────────────────
    e.run_source(
        "(define (loop2 n)
           (if (eq? n 0)
               nil
               (begin
                 (deriv '(* (+ x x) (* x x)) 'x)
                 (loop2 (- n 1)))))",
    );
    e.run_source("(loop2 500)");
    e.collect();

    // Final shape check after second GC.
    check_deriv_x_times_x(e.run_source("(deriv '(* x x) 'x)"));
}

#[test]
fn case_macro_expansion_round_trips_through_gc() {
    // Define everything, force a GC IMMEDIATELY (before running anything),
    // then call into the case-using function. The pool literals must survive
    // even when nothing has executed yet.
    let mut e = Engine::new();
    e.run_source(PROGRAM);
    e.collect();
    e.collect(); // double-collect: spaces flip back, prove idempotency

    check_deriv_plus_x_x(e.run_source("(deriv '(+ x x) 'x)"));
    check_deriv_x_times_x(e.run_source("(deriv '(* x x) 'x)"));
}

#[test]
fn helpers_called_from_macros_survive_gc() {
    // The case macro's body calls case-expand (a regular function) and
    // gensym (a primitive extern). After a GC, defining a NEW use of
    // `case` triggers a fresh expansion — meaning the macro body, which
    // traffics in newly-allocated cons cells, runs against a freshly-
    // collected heap.
    let mut e = Engine::new();
    e.run_source(PROGRAM);

    // Allocate to fill some of from-space with garbage.
    e.run_source(
        "(define (warm n)
           (if (eq? n 0)
               nil
               (begin
                 (deriv '(+ x (* x x)) 'x)
                 (warm (- n 1)))))",
    );
    e.run_source("(warm 200)");
    e.collect();

    // Define a NEW function that uses `case`. Macro expansion runs now,
    // post-GC. case-expand is called via the JIT call table; the resulting
    // expansion contains fresh cons cells; the compiler embeds the
    // expansion in the IR; new pool slots are added.
    e.run_source(
        "(define (classify x)
           (case x
             ((zero) 0)
             ((one two three) 1)
             (else 99)))",
    );

    assert_eq!(as_number(e.run_source("(classify 'zero)")), 0.0);
    assert_eq!(as_number(e.run_source("(classify 'two)")), 1.0);
    assert_eq!(as_number(e.run_source("(classify 'banana)")), 99.0);

    // Force another GC; classify must keep working.
    e.collect();
    assert_eq!(as_number(e.run_source("(classify 'three)")), 1.0);
    assert_eq!(as_number(e.run_source("(classify 'orange)")), 99.0);
}
