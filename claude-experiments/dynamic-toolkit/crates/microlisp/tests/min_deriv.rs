use microlisp::Engine;
use microlisp::value::*;

#[test]
fn min_repro() {
    let mut e = Engine::new();
    e.run_source(
        "(defmacro cond clauses
           (if (null? clauses) 'nil
               (let ((first (car clauses)) (rest (cdr clauses)))
                 (if (eq? (car first) 'else)
                     `(begin ,@(cdr first))
                     `(if ,(car first) (begin ,@(cdr first)) (cond ,@rest))))))
         (define (cadr x) (car (cdr x)))
         (define (caddr x) (car (cdr (cdr x))))
         (define (deriv expr var)
           (cond
             ((number? expr) 0)
             ((symbol? expr) (if (eq? expr var) 1 0))
             (else (list '+ (deriv (cadr expr) var) (deriv (caddr expr) var)))))",
    );
    let r = e.run_source("(deriv '(+ x x) 'x)");
    eprintln!("got: {}", e.print(r));
    assert!(is_cons(r), "expected cons, got 0x{:016x}", r);
}
