//! Heap-pressure stress test: deep recursion with many list
//! allocations per call (one per recursive site), single-threaded.

use clojure::Engine;

#[test]
fn fib_10_single_thread() {
    let e = Engine::new();
    e.eval("(def fib (fn [n] (if (< n 2) n (+ (fib (- n 1)) (fib (- n 2))))))");
    let v = e.call_compiled("fib", &[f64::to_bits(10.0)]);
    let n = f64::from_bits(v) as i64;
    assert_eq!(n, 55);
}

#[test]
fn fib_15_single_thread() {
    let e = Engine::new();
    e.eval("(def fib (fn [n] (if (< n 2) n (+ (fib (- n 1)) (fib (- n 2))))))");
    let v = e.call_compiled("fib", &[f64::to_bits(15.0)]);
    let n = f64::from_bits(v) as i64;
    assert_eq!(n, 610);
}

#[test]
fn fib_20_single_thread() {
    let e = Engine::new();
    e.eval("(def fib (fn [n] (if (< n 2) n (+ (fib (- n 1)) (fib (- n 2))))))");
    let v = e.call_compiled("fib", &[f64::to_bits(20.0)]);
    let n = f64::from_bits(v) as i64;
    assert_eq!(n, 6765);
}

#[test]
fn fib_24_single_thread() {
    let e = Engine::new();
    e.eval("(def fib (fn [n] (if (< n 2) n (+ (fib (- n 1)) (fib (- n 2))))))");
    let v = e.call_compiled("fib", &[f64::to_bits(24.0)]);
    let n = f64::from_bits(v) as i64;
    assert_eq!(n, 46368);
}
