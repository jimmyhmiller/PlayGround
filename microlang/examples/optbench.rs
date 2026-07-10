//! Plain JIT vs Optimized (nanopass) JIT, same programs as the scheme bench.
#![cfg(feature = "jit")]
use std::time::Instant;
use microlang::{CodeSpace, JitCranelift, LowBitModel, Optimized, Runtime};

fn time(make: impl Fn() -> Box<dyn CodeSpace<LowBitModel>>, src: &str) -> (String, f64) {
    let mut rt = Runtime::<LowBitModel>::new();
    let cs = make();
    let t = Instant::now();
    let v = microlang::sexpr::eval_str(&mut rt, cs.as_ref(), src);
    let dt = t.elapsed().as_secs_f64();
    (rt.print(v), dt)
}

fn main() {
    let progs = [
        ("fib(30)", "(def fib (fn (n) (if (< n 2) n (+ (fib (- n 1)) (fib (- n 2)))))) (fib 30)"),
        ("tak(24,16,8)", "(def tak (fn (x y z) (if (< y x) (tak (tak (- x 1) y z) (tak (- y 1) z x) (tak (- z 1) x y)) z))) (tak 24 16 8)"),
        ("ack(3,7)", "(def ack (fn (m n) (if (= m 0) (+ n 1) (if (= n 0) (ack (- m 1) 1) (ack (- m 1) (ack m (- n 1))))))) (ack 3 7)"),
        ("count-loop(50M)", "(def go (fn (n acc) (if (= n 0) acc (go (- n 1) (+ acc 1))))) (go 50000000 0)"),
        ("sum-tail(10M)", "(def go (fn (n acc) (if (= n 0) acc (go (- n 1) (+ acc n))))) (go 10000000 0)"),
        // inlining: a lambda applied every iteration (no closure alloc / call after beta).
        ("inline-loop(10M)", "(def go (fn (n acc) (if (= n 0) acc (go (- n 1) ((fn (x) (+ acc (* x x))) n))))) (go 10000000 0)"),
        // constant folding: a constant subexpression recomputed every iteration.
        ("fold-loop(10M)", "(def go (fn (n acc) (if (= n 0) acc (go (- n 1) (+ acc (+ (* 2 3) (* 4 5))))))) (go 10000000 0)"),
    ];
    println!("{:<16} {:>10} {:>10} {:>8}   result-match", "program", "plain", "optimized", "speedup");
    println!("{}", "-".repeat(64));
    for (name, src) in progs {
        // best of 3
        let mut pj = f64::MAX; let mut oj = f64::MAX; let mut pv = String::new(); let mut ov = String::new();
        for _ in 0..3 {
            let (v, t) = time(|| Box::new(JitCranelift::<LowBitModel>::new()), src); pj = pj.min(t); pv = v;
            let (v, t) = time(|| Box::new(Optimized::new(JitCranelift::<LowBitModel>::new())), src); oj = oj.min(t); ov = v;
        }
        println!("{:<16} {:>10.4} {:>10.4} {:>7.2}x   {}", name, pj, oj, pj/oj, if pv==ov {"ok"} else {"MISMATCH!"});
    }
}
