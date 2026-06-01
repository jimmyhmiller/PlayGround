//! Throughput benchmark for our compiler: compile every fixture in-process,
//! warm, and report ms/compile. Pair with `oracle/bench.js` (the official React
//! compiler) for a head-to-head. Set REACT_FIXTURES / ITERS to override.
use std::time::Instant;

fn main() {
    let dir = std::env::var("REACT_FIXTURES")
        .unwrap_or_else(|_| "crates/jsir-ssa/oracle/fixtures".into());
    let mut srcs = Vec::new();
    for e in std::fs::read_dir(&dir).unwrap() {
        let p = e.unwrap().path();
        if p.extension().map(|x| x == "js").unwrap_or(false) {
            if let Ok(s) = std::fs::read_to_string(&p) {
                srcs.push(s);
            }
        }
    }
    let iters: usize = std::env::var("ITERS").ok().and_then(|s| s.parse().ok()).unwrap_or(10);

    // Warmup; keep only the fixtures we actually compile (exclude fast bails so
    // the per-compile figure reflects real work, not error paths).
    let mut okset: Vec<&String> = Vec::new();
    let mut err = 0usize;
    for s in &srcs {
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| jsir_ssa::codegen::compile(s))) {
            Ok(Ok(_)) => okset.push(s),
            _ => err += 1,
        }
    }

    for (label, set) in [("ALL", srcs.iter().collect::<Vec<_>>()), ("OK-only", okset.clone())] {
        let start = Instant::now();
        for _ in 0..iters {
            for s in &set {
                let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    jsir_ssa::codegen::compile(s)
                }));
            }
        }
        let el = start.elapsed();
        let total = set.len() * iters;
        println!(
            "OURS[{label}]: n={} {:.1}ms => {:.4} ms/compile, {:.0} compiles/sec",
            set.len(),
            el.as_secs_f64() * 1000.0,
            el.as_secs_f64() * 1000.0 / total as f64,
            total as f64 / el.as_secs_f64()
        );
    }
    println!("OURS: fixtures={} ok={} err={}", srcs.len(), okset.len(), err);
}
