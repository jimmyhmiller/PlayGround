//! Engine performance benchmark: the same programs on every configuration of
//! the one engine, with a native Rust baseline for context.
//!
//! Run with: `cargo run --release --bin livetype-bench`

use livetype::*;
use std::time::Instant;

/// One benchmark program: source, entry, argument, expected result, and the
/// "work units" the timing is normalized by (iterations, calls, allocations).
struct Bench {
    name: &'static str,
    src: &'static str,
    arg: i64,
    units: u64,
    expected: i64,
}

fn benches() -> Vec<Bench> {
    vec![
        Bench {
            name: "loop_sum   (1M iters, arithmetic)",
            src: "fn main(n: i64) -> i64 {
                    let i = 0;
                    let s = 0;
                    while i < n {
                        s = s + i;
                        i = i + 1;
                    }
                    s
                  }",
            arg: 1_000_000,
            units: 1_000_000,
            expected: 499_999_500_000,
        },
        Bench {
            name: "call_add   (300k calls)",
            src: "fn add(a: i64, b: i64) -> i64 { a + b }
                  fn main(n: i64) -> i64 {
                    let i = 0;
                    let s = 0;
                    while i < n {
                        s = add(s, i);
                        i = i + 1;
                    }
                    s
                  }",
            arg: 300_000,
            units: 300_000,
            expected: 44_999_850_000,
        },
        Bench {
            name: "fib_25     (243k rec calls)",
            src: "fn fib(n: i64) -> i64 {
                    if n < 2 { return n; }
                    return fib(n - 1) + fib(n - 2);
                  }
                  fn main(n: i64) -> i64 { fib(n) }",
            arg: 25,
            units: 242_785,
            expected: 75_025,
        },
        Bench {
            name: "alloc_read (200k alloc + field)",
            src: "struct Box { v: i64 }
                  fn main(n: i64) -> i64 {
                    let i = 0;
                    let s = 0;
                    while i < n {
                        let b = Box { v: i };
                        s = s + b.v;
                        i = i + 1;
                    }
                    s
                  }",
            arg: 200_000,
            units: 200_000,
            expected: 19_999_900_000,
        },
        Bench {
            name: "yield_loop (200k iters w/ yield)",
            src: "fn main(n: i64) -> i64 {
                    let i = 0;
                    let s = 0;
                    while i < n {
                        s = s + i;
                        i = i + 1;
                        yield;
                    }
                    s
                  }",
            arg: 200_000,
            units: 200_000,
            expected: 19_999_900_000,
        },
    ]
}

fn run_on(engine: std::sync::Arc<Engine>, b: &Bench) -> (f64, f64) {
    let compiled = livetype_core::compile_on(b.src, engine).expect("compile");
    let main = compiled.functions["main"];
    // Warm once (compilation, promotion counters) on a small input where
    // possible; for tiering this is part of the story, so warm with the real
    // arg once and time the second run.
    let warm = compiled.engine.run_call(main, vec![Value::I64(b.arg)]);
    assert_eq!(
        warm,
        Outcome::Complete(Value::I64(b.expected)),
        "{}: wrong result on warmup",
        b.name
    );
    compiled.engine.collect(&[]);
    let start = Instant::now();
    let out = compiled.engine.run_call(main, vec![Value::I64(b.arg)]);
    let dt = start.elapsed();
    assert_eq!(out, Outcome::Complete(Value::I64(b.expected)), "{}: wrong result", b.name);
    compiled.engine.collect(&[]);
    (dt.as_secs_f64() * 1e3, dt.as_secs_f64() * 1e9 / b.units as f64)
}

/// Native Rust equivalents, for context only (black_box-ed).
fn native_baseline(b: &Bench) -> Option<f64> {
    use std::hint::black_box;
    fn fib(n: i64) -> i64 {
        if n < 2 { n } else { fib(n - 1) + fib(n - 2) }
    }
    let start = Instant::now();
    let result = match b.name.split_whitespace().next().unwrap() {
        "loop_sum" | "yield_loop" => {
            let n = black_box(b.arg);
            let mut s = 0i64;
            let mut i = 0i64;
            while i < n {
                s += black_box(i);
                i += 1;
            }
            s
        }
        "call_add" => {
            #[inline(never)]
            fn add(a: i64, b: i64) -> i64 {
                black_box(a + b)
            }
            let n = black_box(b.arg);
            let mut s = 0i64;
            let mut i = 0i64;
            while i < n {
                s = add(s, i);
                i += 1;
            }
            s
        }
        "fib_25" => fib(black_box(b.arg)),
        _ => return None,
    };
    let dt = start.elapsed();
    assert_eq!(result, b.expected);
    Some(dt.as_secs_f64() * 1e9 / b.units as f64)
}

fn main() {
    println!("livetype engine benchmark (release)\n");
    println!(
        "{:<36} {:>14} {:>14} {:>14} {:>12}",
        "", "interp", "jit(0)", "tiered(10)", "native rust"
    );
    for b in benches() {
        let (interp_ms, interp_ns) = run_on(Engine::interp(), &b);
        let (_, jit_ns) = run_on(jit_engine(0), &b);
        let (_, tiered_ns) = run_on(jit_engine(10), &b);
        let native = native_baseline(&b);
        println!(
            "{:<36} {:>10.0} ns {:>10.0} ns {:>10.0} ns {:>10}",
            b.name,
            interp_ns,
            jit_ns,
            tiered_ns,
            native.map_or("—".to_string(), |n| format!("{n:.1} ns")),
        );
        let _ = interp_ms;
    }
    println!("\n(per work unit: iteration / call / allocation)");
}
