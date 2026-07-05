//! R7RS conformance suite for the Scheme frontend.
//!
//! Doubles as the roadmap. Each case is `(area, setup, expr, expect, live)`:
//!   * `expect` is the R7RS `display` of `expr` (given `setup` at top level).
//!   * `live` = we claim to support it now; these MUST pass (regression guard).
//!   * `!live` = pending; run but not asserted, reported as the frontier. When a
//!     pending case starts passing, promote it to `live`.
//!
//! "Knowing we got it right" comes from an objective oracle: if Chicken Scheme
//! (`csi`) is installed, EVERY `expect` is validated against it. So a wrong
//! expected value fails loudly, and a live case is correct iff our frontend's
//! output equals the oracle's.

use std::io::Write;
use std::process::{Command, Stdio};

use microlang::{LowBitModel, Runtime, TreeWalk};

struct Case {
    area: &'static str,
    setup: &'static str,
    expr: &'static str,
    expect: &'static str,
    live: bool,
}

const fn c(area: &'static str, setup: &'static str, expr: &'static str, expect: &'static str, live: bool) -> Case {
    Case { area, setup, expr, expect, live }
}

#[rustfmt::skip]
const CASES: &[Case] = &[
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
    c("higher-order", "", "(map (lambda (x) (* x x)) (list 1 2 3))", "(1 4 9)", false),
    c("lists",        "", "(append (list 1 2) (list 3 4))", "(1 2 3 4)", false),
    c("equality",     "", "(eq? 'a 'a)",                   "#t",      false),
    c("quasiquote",   "", "`(1 ,(+ 1 1) 3)",               "(1 2 3)", false),
    c("strings",      "", "(string-length \"hello\")",     "5",       false),
    c("chars",        "", "(char->integer #\\A)",          "65",      false),
    c("vectors",      "", "(vector-ref (vector 10 20 30) 1)", "20",    false),
    c("tail-calls",   "(define (loop n) (if (= n 0) 'done (loop (- n 1))))", "(loop 1000000)", "done", true),
    c("numeric-tower","", "(* 100000000000 100000000000)", "10000000000000000000000", false),
    c("call/cc",      "", "(call/cc (lambda (k) (+ 1 (k 10))))", "10", true),
    c("call/cc-search", "(define (find-neg lst) (call/cc (lambda (return) (letrec ((loop (lambda (l) (if (null? l) 'none (if (< (car l) 0) (return (car l)) (loop (cdr l))))))) (loop lst)))))", "(find-neg (list 1 2 -3 4))", "-3", true),
    c("dynamic-wind", "", "(let ((r '())) (dynamic-wind (lambda () (set! r (cons 'in r))) (lambda () 'body) (lambda () (set! r (cons 'out r)))) (reverse r))", "(in out)", false),
    c("values",       "", "(call-with-values (lambda () (values 1 2)) +)", "3", false),
    c("syntax-rules", "(define-syntax swap (syntax-rules () ((_ a b) (list b a))))", "(swap 1 2)", "(2 1)", true),
    c("syntax-rules-ellipsis", "(define-syntax my-list (syntax-rules () ((_ x ...) (list x ...))))", "(my-list 1 2 3)", "(1 2 3)", true),
    c("syntax-rules-nested", "(define-syntax unless2 (syntax-rules () ((_ c body) (if c (quote skip) body)))) (define-syntax my-list (syntax-rules () ((_ x ...) (list x ...))))", "(unless2 #f (my-list 1 2 3))", "(1 2 3)", true),
    // HYGIENE (pending): the macro's `t` must not capture the user's `t`. A
    // hygienic Scheme returns 5; our unhygienic engine returns #f. This is the
    // marker for the remaining hard piece.
    c("hygiene", "(define-syntax my-or (syntax-rules () ((_ a b) (let ((t a)) (if t t b)))))", "(let ((t 5)) (my-or #f t))", "5", false),
];

fn our_output(setup: &str, expr: &str) -> Option<String> {
    let prog = format!("{setup} {expr}");
    let prev = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let out = std::panic::catch_unwind(|| {
        let mut rt = Runtime::<LowBitModel>::new();
        let v = scheme::run(&mut rt, &TreeWalk, &prog);
        scheme::write_value(&rt, v)
    })
    .ok();
    std::panic::set_hook(prev);
    out
}

fn csi_path() -> Option<String> {
    Command::new("csi").arg("-version").stdout(Stdio::null()).stderr(Stdio::null()).status().ok()?;
    Some("csi".to_string())
}

fn oracle_output(csi: &str, setup: &str, expr: &str) -> Option<String> {
    let program = format!("{setup}\n(display {expr})(newline)");
    let mut child = Command::new(csi)
        .arg("-q")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .ok()?;
    child.stdin.take()?.write_all(program.as_bytes()).ok()?;
    let out = child.wait_with_output().ok()?;
    Some(String::from_utf8_lossy(&out.stdout).trim().to_string())
}

#[test]
fn r7rs_conformance() {
    let csi = csi_path();
    if csi.is_none() {
        eprintln!("note: `csi` (Chicken Scheme) not found — expected values not oracle-validated");
    }

    let (mut live_ok, mut live_total) = (0, 0);
    let (mut pending_ok, mut pending_total) = (0, 0);
    let mut live_failures = Vec::new();
    let mut oracle_failures = Vec::new();
    let mut promote = Vec::new();

    for case in CASES {
        // Validate the expected value against the oracle (all cases).
        if let Some(csi) = &csi {
            if let Some(oracle) = oracle_output(csi, case.setup, case.expr) {
                if oracle != case.expect {
                    oracle_failures.push(format!(
                        "[{}] {}  oracle={:?} but expect={:?}",
                        case.area, case.expr, oracle, case.expect
                    ));
                }
            }
        }

        let ours = our_output(case.setup, case.expr);
        let ok = ours.as_deref() == Some(case.expect);
        if case.live {
            live_total += 1;
            if ok {
                live_ok += 1;
            } else {
                live_failures.push(format!(
                    "[{}] {}  got={:?} want={:?}",
                    case.area, case.expr, ours, case.expect
                ));
            }
        } else {
            pending_total += 1;
            if ok {
                pending_ok += 1;
                promote.push(format!("[{}] {}", case.area, case.expr));
            }
        }
    }

    println!("\nR7RS conformance:");
    println!("  live:    {live_ok}/{live_total} passing");
    println!("  pending: {pending_ok}/{pending_total} now pass (promote these to live)");
    if !promote.is_empty() {
        println!("  ready to promote:\n    {}", promote.join("\n    "));
    }

    assert!(
        oracle_failures.is_empty(),
        "expected values disagree with the Chicken Scheme oracle:\n{}",
        oracle_failures.join("\n")
    );
    assert!(
        live_failures.is_empty(),
        "LIVE conformance regressions:\n{}",
        live_failures.join("\n")
    );
}
