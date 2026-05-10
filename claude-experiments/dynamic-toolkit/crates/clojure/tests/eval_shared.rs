//! Shared-Engine multi-threaded execution: a single `Engine` lives
//! behind an `Arc<Mutex<…>>`, and multiple threads each take the
//! lock, run macros / fn calls, release. Demonstrates goal #3:
//! "macros that work multi-threaded".

use clojure::Engine;
use std::sync::{Arc, Mutex};
use std::thread;

#[test]
fn shared_engine_two_threads() {
    let e = Arc::new(Mutex::new(Engine::new()));

    // Pre-define a macro and a fn in the shared engine.
    {
        let mut g = e.lock().unwrap();
        g.eval(
            "(defmacro twice [x] (cons (quote +) (cons x (cons x nil)))) \
             (def square (fn [x] (* x x)))",
        );
    }

    // Two threads each call into the shared engine.
    let e1 = e.clone();
    let h1 = thread::spawn(move || {
        let mut g = e1.lock().unwrap();
        let v = g.eval("(twice 7)");
        g.print(v)
    });
    let e2 = e.clone();
    let h2 = thread::spawn(move || {
        let mut g = e2.lock().unwrap();
        let v = g.eval("(square 9)");
        g.print(v)
    });

    let r1 = h1.join().unwrap();
    let r2 = h2.join().unwrap();

    assert!(r1 == "14" || r1 == "81");
    assert!(r2 == "14" || r2 == "81");
    assert_ne!(r1, r2, "the two threads ran different programs");
}

#[test]
fn shared_engine_macro_defined_then_called_from_another_thread() {
    let e = Arc::new(Mutex::new(Engine::new()));

    let e_def = e.clone();
    let h_def = thread::spawn(move || {
        let mut g = e_def.lock().unwrap();
        g.eval(
            "(defmacro unless [c body] \
               (cons (quote if) (cons c (cons nil (cons body nil)))))",
        );
    });
    h_def.join().unwrap();

    let e_use = e.clone();
    let h_use = thread::spawn(move || {
        let mut g = e_use.lock().unwrap();
        let v = g.eval("(unless false 42)");
        g.print(v)
    });
    let result = h_use.join().unwrap();
    assert_eq!(result, "42");
}

#[test]
fn shared_engine_many_concurrent_calls() {
    let e = Arc::new(Mutex::new(Engine::new()));
    {
        let mut g = e.lock().unwrap();
        g.eval("(def fib (fn [n] (if (< n 2) n (+ (fib (- n 1)) (fib (- n 2))))))");
    }

    let mut handles = Vec::new();
    let inputs: Vec<(i64, &str)> = vec![
        (0, "0"),
        (1, "1"),
        (2, "1"),
        (3, "2"),
        (5, "5"),
        (8, "21"),
        (10, "55"),
        (12, "144"),
    ];
    for (n, expected) in inputs {
        let e = e.clone();
        let expected = expected.to_string();
        handles.push(thread::spawn(move || {
            let mut g = e.lock().unwrap();
            let v = g.eval(&format!("(fib {})", n));
            assert_eq!(g.print(v), expected, "fib({}) result", n);
        }));
    }
    for h in handles {
        h.join().unwrap();
    }
}
