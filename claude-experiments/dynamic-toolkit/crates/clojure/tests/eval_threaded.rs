//! Multi-threaded execution: each thread has its own Engine,
//! each runs its own program (including macros), all in parallel.

use clojure::Engine;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;

#[test]
fn two_threads_each_with_own_engine() {
    let h1 = thread::spawn(|| {
        let mut e = Engine::new();
        let v = e.eval(
            "(def fib (fn [n] (if (< n 2) n (+ (fib (- n 1)) (fib (- n 2)))))) \
             (fib 10)",
        );
        assert_eq!(e.print(v), "55");
    });
    let h2 = thread::spawn(|| {
        let mut e = Engine::new();
        let v = e.eval(
            "(defmacro unless [c body] \
               (cons (quote if) (cons c (cons nil (cons body nil))))) \
             (unless false 99)",
        );
        assert_eq!(e.print(v), "99");
    });
    h1.join().unwrap();
    h2.join().unwrap();
}

#[test]
fn many_threads_with_macros() {
    // Stress: 8 threads, each evaluates a macro-driven program.
    // All run truly in parallel — each has its own GC, its own
    // JitModule, its own namespace registry.
    let count = Arc::new(AtomicUsize::new(0));
    let mut handles = Vec::new();
    for i in 0..8 {
        let count = count.clone();
        handles.push(thread::spawn(move || {
            let mut e = Engine::new();
            let src = format!(
                "(defmacro twice [x] (cons (quote +) (cons x (cons x nil)))) \
                 (twice {})",
                i
            );
            let v = e.eval(&src);
            assert_eq!(e.print(v), format!("{}", i * 2));
            count.fetch_add(1, Ordering::Relaxed);
        }));
    }
    for h in handles {
        h.join().unwrap();
    }
    assert_eq!(count.load(Ordering::Relaxed), 8);
}
