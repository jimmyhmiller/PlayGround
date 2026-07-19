//! The R7RS conformance case table — shared source of truth for both the
//! backend-agnostic `conformance` suite (run on the CEK machine) and the
//! `jit_conformance` suite (run on the native JIT with a CEK fallback).
//!
//! Lives under `tests/common/` so Cargo does not compile it as its own test
//! binary; each suite pulls it in with `#[path = "common/suite.rs"] mod suite;`.

#![allow(dead_code)]

pub struct Case {
    pub area: &'static str,
    pub setup: &'static str,
    pub expr: &'static str,
    pub expect: &'static str,
    pub live: bool,
}

pub const fn c(
    area: &'static str,
    setup: &'static str,
    expr: &'static str,
    expect: &'static str,
    live: bool,
) -> Case {
    Case { area, setup, expr, expect, live }
}

/// Filinski's derivation of `shift*`/`reset*` from multi-shot `call/cc` + a cell.
pub const DELIM: &str = "
(define meta-k (lambda (t) (error \"missing top-level reset\")))
(define (reset* t)
  (let ((saved meta-k))
    (call/cc (lambda (k)
      (set! meta-k (lambda (r) (set! meta-k saved) (k r)))
      (let ((v (t))) (meta-k v))))))
(define (shift* h)
  (call/cc (lambda (k)
    (meta-k (h (lambda (x) (reset* (lambda () (k x)))))))))
";

/// Does this program use FIRST-CLASS continuations? Such programs must run
/// WHOLLY on the CEK machine — a host-stack JIT cannot capture native frames, so
/// no JIT frame may sit inside a `call/cc` capture. (`apply`/`values` do NOT
/// capture, so they are fine to mix and are not flagged here.) This is the
/// all-or-nothing tier rule the JIT harness applies per program.
pub fn uses_first_class_continuations(setup: &str, expr: &str) -> bool {
    let hit = |s: &str| s.contains("call/cc") || s.contains("call-with-current-continuation");
    hit(setup) || hit(expr)
}

#[rustfmt::skip]
pub const CASES: &[Case] = &[
    // ── implemented (live) ──────────────────────────────────
    c("numbers",      "", "(+ 1 2)",                       "3",       true),
    c("numbers",      "", "(- 10 3)",                      "7",       true),
    c("numbers",      "", "(* 3 4)",                       "12",      true),
    c("comparison",   "", "(< 2 3)",                       "#t",      true),
    c("comparison",   "", "(< 3 2)",                       "#f",      true),
    c("equality",     "", "(= 2 2)",                       "#t",      true),
    c("conditional",  "", "(if (< 1 2) 'a 'b)",            "a",       true),
    c("conditional",  "", "(if #f 'a 'b)",                 "b",       true),
    c("cond",         "", "(cond ((< 3 2) 'x) (else 'y))", "y",       true),
    c("cond",         "", "(cond ((< 2 3) 'x) (else 'y))", "x",       true),
    c("let",          "", "(let ((a 2) (b 3)) (+ a b))",   "5",       true),
    c("let-parallel", "", "(let ((a 1)) (let ((a 2) (b a)) b))", "1", true),
    c("let*",         "", "(let* ((a 1) (b (+ a 1))) b)",  "2",       true),
    c("lambda",       "", "((lambda (x) (* x x)) 5)",      "25",      true),
    c("begin",        "", "(begin 1 2 3)",                 "3",       true),
    c("quote",        "", "(quote hello)",                 "hello",   true),
    c("quote",        "", "(quote (1 2 3))",               "(1 2 3)", true),
    c("define/call",  "(define (sq n) (* n n))", "(sq 6)", "36",      true),
    c("recursion",    "(define (fact n) (if (< n 2) 1 (* n (fact (- n 1)))))", "(fact 5)", "120", true),
    c("lists",        "", "(list 1 2 3)",                  "(1 2 3)", true),
    c("lists",        "", "(cons 1 (list 2 3))",           "(1 2 3)", true),
    c("lists",        "", "(car (list 4 5 6))",            "4",       true),
    c("lists",        "", "(cdr (list 4 5 6))",            "(5 6)",   true),
    c("lists",        "", "(null? (list))",                "#t",      true),
    c("lists",        "", "(null? (list 1))",              "#f",      true),

    // arithmetic/comparison/boolean sugar
    c("numbers-var",  "", "(+ 1 2 3)",                     "6",       true),
    c("comparison",   "", "(> 5 3)",                       "#t",      true),
    c("comparison",   "", "(<= 2 2)",                      "#t",      true),
    c("comparison",   "", "(>= 3 5)",                      "#f",      true),
    c("boolean-ops",  "", "(and #t #f)",                   "#f",      true),
    c("boolean-ops",  "", "(or #f 3)",                     "3",       true),
    c("when-unless",  "", "(when (< 1 2) 'yes)",           "yes",     true),
    c("case",         "", "(case 2 ((1) 'one) ((2) 'two) (else 'other))", "two", true),
    c("equality",     "", "(equal? (list 1 2) (list 1 2))", "#t",     true),
    // set!, letrec, named let — via a real core mechanism (local mutation)
    c("set!",         "(define c 0)", "(begin (set! c (+ c 1)) (set! c (+ c 1)) c)", "2", true),
    c("letrec",       "", "(letrec ((e? (lambda (n) (if (= n 0) #t (o? (- n 1)))))\n                              (o? (lambda (n) (if (= n 0) #f (e? (- n 1)))))) (e? 10))", "#t", true),
    c("named-let",    "", "(let loop ((i 0) (acc 0)) (if (< i 5) (loop (+ i 1) (+ acc i)) acc))", "10", true),

    // ── pending (roadmap) ───────────────────────────────────
    c("higher-order", "", "(map (lambda (x) (* x x)) (list 1 2 3))", "(1 4 9)", true),
    c("lists",        "", "(append (list 1 2) (list 3 4))", "(1 2 3 4)", true),
    c("equality",     "", "(eq? 'a 'a)",                   "#t",      true),
    c("quasiquote",   "", "`(1 ,(+ 1 1) 3)",               "(1 2 3)", true),
    c("strings",      "", "(string-length \"hello\")",     "5",       true),
    c("chars",        "", "(char->integer #\\A)",          "65",      true),
    c("vectors",      "", "(vector-ref (vector 10 20 30) 1)", "20",    true),
    c("tail-calls",   "(define (loop n) (if (= n 0) 'done (loop (- n 1))))", "(loop 1000000)", "done", true),
    c("numeric-tower","", "(* 100000000000 100000000000)", "10000000000000000000000", true),
    c("bignum-add",   "", "(+ 4611686018427387904 4611686018427387904)", "9223372036854775808", true),
    // beyond i128 — true arbitrary precision (hand-rolled BigInt, no deps).
    c("bignum-huge",  "", "(* 100000000000000000000 100000000000000000000)", "10000000000000000000000000000000000000000", true),
    c("call/cc",      "", "(call/cc (lambda (k) (+ 1 (k 10))))", "10", true),
    c("call/cc-search", "(define (find-neg lst) (call/cc (lambda (return) (letrec ((loop (lambda (l) (if (null? l) 'none (if (< (car l) 0) (return (car l)) (loop (cdr l))))))) (loop lst)))))", "(find-neg (list 1 2 -3 4))", "-3", true),
    // MULTI-SHOT: the continuation is invoked repeatedly to RE-ENTER a
    // computation that already returned — impossible for an escape continuation.
    c("call/cc-multishot", "", "(let ((n 0) (k #f)) (call/cc (lambda (c) (set! k c))) (set! n (+ n 1)) (if (< n 5) (k #f) n))", "5", true),
    // DELIMITED continuations, derived from multi-shot call/cc (see `DELIM`).
    // `reset*` bounds the continuation; `shift*` captures it up to that bound.
    c("shift/reset",     DELIM, "(+ 1 (reset* (lambda () (+ 10 (shift* (lambda (k) (k (k 5))))))))", "26", true),
    c("shift/reset-mul", DELIM, "(* 2 (reset* (lambda () (+ 1 (shift* (lambda (k) (k 5)))))))", "12", true),
    c("shift/reset-abort", DELIM, "(reset* (lambda () (+ 1 (shift* (lambda (k) 5)))))", "5", true),
    // MULTI-SHOT delimited: the delimited continuation `k` is invoked TWICE.
    c("shift/reset-multishot", DELIM, "(reset* (lambda () (+ 1 (shift* (lambda (k) (+ 3 (k 5) (k 5)))))))", "15", true),
    c("dynamic-wind", "", "(let ((r '())) (dynamic-wind (lambda () (set! r (cons 'in r))) (lambda () 'body) (lambda () (set! r (cons 'out r)))) (reverse r))", "(in out)", true),
    c("values",       "", "(call-with-values (lambda () (values 1 2)) +)", "3", true),
    c("syntax-rules", "(define-syntax swap (syntax-rules () ((_ a b) (list b a))))", "(swap 1 2)", "(2 1)", true),
    c("syntax-rules-ellipsis", "(define-syntax my-list (syntax-rules () ((_ x ...) (list x ...))))", "(my-list 1 2 3)", "(1 2 3)", true),
    c("syntax-rules-nested", "(define-syntax unless2 (syntax-rules () ((_ c body) (if c (quote skip) body)))) (define-syntax my-list (syntax-rules () ((_ x ...) (list x ...))))", "(unless2 #f (my-list 1 2 3))", "(1 2 3)", true),
    // HYGIENE (pending): the macro's `t` must not capture the user's `t`. A
    // hygienic Scheme returns 5; our unhygienic engine returns #f. This is the
    // marker for the remaining hard piece.
    c("hygiene", "(define-syntax my-or (syntax-rules () ((_ a b) (let ((t a)) (if t t b)))))", "(let ((t 5)) (my-or #f t))", "5", true),
];
