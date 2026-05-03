//! Verify that gc-stress mode genuinely runs a moving collection between
//! every pair of top-level forms — and that the resulting program produces
//! the same output as the non-stressed run.

use microlisp::Engine;
use microlisp::value::*;

const PROGRAM: &str = r#"
(defmacro cond clauses
  (if (null? clauses) 'nil
      (let ((first (car clauses)) (rest (cdr clauses)))
        (if (eq? (car first) 'else)
            `(begin ,@(cdr first))
            `(if ,(car first) (begin ,@(cdr first)) (cond ,@rest))))))

(define (member? x lst)
  (if (null? lst) #f
      (if (eq? x (car lst)) #t (member? x (cdr lst)))))

(define (case-expand var clauses)
  (if (null? clauses) 'nil
      (let ((c (car clauses)) (rest (cdr clauses)))
        (if (eq? (car c) 'else)
            (cons 'begin (cdr c))
            (list 'if
                  (list 'member? var (list 'quote (car c)))
                  (cons 'begin (cdr c))
                  (case-expand var rest))))))

(defmacro case (scrut . clauses)
  (let ((g (gensym 'case)))
    `(let ((,g ,scrut))
       ,(case-expand g clauses))))

(define (cadr x)  (car (cdr x)))
(define (caddr x) (car (cdr (cdr x))))

(define (deriv expr var)
  (cond
    ((number? expr) 0)
    ((symbol? expr) (if (eq? expr var) 1 0))
    (else
      (case (car expr)
        ((+) (list '+ (deriv (cadr expr) var) (deriv (caddr expr) var)))
        ((*) (list '+
                   (list '* (cadr expr) (deriv (caddr expr) var))
                   (list '* (deriv (cadr expr) var) (caddr expr))))
        (else 'unknown-op)))))

(deriv 'x 'x)
(deriv '(+ x x) 'x)
(deriv '(* x x) 'x)
(deriv '(* x (* x x)) 'x)
(deriv '(+ (* x x) (* 3 x)) 'x)
"#;

#[test]
fn stress_mode_collects_between_every_form() {
    let mut e = Engine::new();
    e.set_gc_stress(true);

    let count_before = e.gc().collection_count();
    let result = e.run_source(PROGRAM);

    // The last form returned a derivative — verify it has the right shape.
    assert!(is_cons(result));

    // Count the top-level forms in PROGRAM. read_forms allocates cons
    // cells, so the GC thread state must be installed.
    let n_forms = e.with_thread_state(|_h| e.read_forms(PROGRAM).len());

    let count_after = e.gc().collection_count();
    let actual_collections = count_after - count_before;

    // Stress mode collects after every form except the last (since we
    // can't safely drop the final result on the floor). So we expect
    // exactly `n_forms - 1` collections from this run_source call.
    assert_eq!(
        actual_collections,
        n_forms - 1,
        "expected {} collections (n_forms={}, all but the last), got {}",
        n_forms - 1,
        n_forms,
        actual_collections,
    );

    // Sanity: program produced the right derivative for the last form,
    // (deriv '(+ (* x x) (* 3 x)) 'x) = (+ (+ (* x 1) (* 1 x)) (+ (* 3 1) (* 0 x)))
    let pretty = e.print(result);
    assert_eq!(
        pretty,
        "(+ (+ (* x 1) (* 1 x)) (+ (* 3 1) (* 0 x)))"
    );
}

#[test]
fn stress_and_normal_produce_identical_output() {
    // Same program, two runs. The output of the last form must match
    // byte-for-byte regardless of how aggressively we collected.
    let mut e_normal = Engine::new();
    let r_normal = e_normal.run_source(PROGRAM);
    let pretty_normal = e_normal.print(r_normal);

    let mut e_stress = Engine::new();
    e_stress.set_gc_stress(true);
    let r_stress = e_stress.run_source(PROGRAM);
    let pretty_stress = e_stress.print(r_stress);

    assert_eq!(pretty_normal, pretty_stress);
    // And the stress run should have done many collections, the normal
    // run zero (we never explicitly call collect, and the heap is large
    // enough that it doesn't auto-collect during this small program).
    assert_eq!(e_normal.gc().collection_count(), 0);
    assert!(e_stress.gc().collection_count() > 0);
}
