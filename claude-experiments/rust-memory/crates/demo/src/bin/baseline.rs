//! The SAME churn workload as `mtbench`, but with NO memscope at all — the plain
//! System allocator. This is the "original program" baseline; compare its
//! Mops/s to `mtbench off|sampled|full` to get the true slowdown factor.
//!
//!   baseline <threads> <ops_per_thread>

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

// No #[global_allocator]: this binary uses the default System allocator.

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let threads: usize = args.first().and_then(|s| s.parse().ok()).unwrap_or(4);
    let ops: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(1_000_000);

    let start = Arc::new(AtomicBool::new(false));
    let mut handles = Vec::new();
    for t in 0..threads {
        let start = start.clone();
        handles.push(std::thread::spawn(move || {
            while !start.load(Ordering::Acquire) {
                std::hint::spin_loop();
            }
            // Identical to mtbench's loop.
            let mut sink: Vec<u8> = Vec::new();
            for i in 0..ops {
                let sz = 16 + ((i + t) & 127);
                let v: Vec<u8> = Vec::with_capacity(sz);
                sink = v;
                std::hint::black_box(&sink);
            }
            std::hint::black_box(&sink);
        }));
    }

    let t0 = Instant::now();
    start.store(true, Ordering::Release);
    for h in handles {
        h.join().unwrap();
    }
    let elapsed = t0.elapsed();

    let total_ops = (threads * ops) as f64;
    let mops = total_ops / elapsed.as_secs_f64() / 1e6;
    let ns_per_op = elapsed.as_nanos() as f64 / total_ops;
    println!(
        "baseline (no memscope) threads={threads} ops/thread={ops}  wall={elapsed:?}  {mops:.1} Mops/s  {ns_per_op:.1} ns/op",
    );
}
