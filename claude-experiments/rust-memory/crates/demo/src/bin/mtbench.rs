//! Multi-threaded allocation benchmark — measures how the tracking allocator
//! scales across threads (where the hot-path locks matter).
//!
//!   mtbench <off|full|sampled> <threads> <ops_per_thread> [rate]
//!
//! Each thread runs a tight alloc+free churn loop. Reports aggregate throughput
//! (Mops/s) and ns/op so you can see lock contention as threads increase.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

use memscope::{MemScope, Mode};

#[global_allocator]
static GLOBAL: MemScope = MemScope::system();

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mode_s = args.first().map(String::as_str).unwrap_or("full");
    let threads: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(4);
    let ops: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(1_000_000);
    let rate: u32 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(100);

    let mode = match mode_s {
        "off" => Mode::Off,
        "sampled" => Mode::Sampled,
        _ => Mode::Full,
    };
    // Decomposition knobs: isolate the cost of each hot-path layer.
    if std::env::var("MEMSCOPE_NOSITES").is_ok() {
        memscope::set_capture_sites(false); // skip stack capture + interning
    }
    if let Ok(d) = std::env::var("MEMSCOPE_DEPTH") {
        if let Ok(d) = d.parse::<usize>() {
            memscope::set_backtrace_depth(d); // cap captured stack depth
        }
    }
    memscope::set_mode(mode);
    if matches!(mode, Mode::Sampled) {
        memscope::set_sample_rate(rate);
    }

    let start = Arc::new(AtomicBool::new(false));
    let mut handles = Vec::new();
    for t in 0..threads {
        let start = start.clone();
        handles.push(std::thread::spawn(move || {
            while !start.load(Ordering::Acquire) {
                std::hint::spin_loop();
            }
            // Churn: allocate and free a small, varied object each iteration.
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
        "mode={mode_s} threads={threads} ops/thread={ops}  wall={elapsed:?}  {mops:.1} Mops/s  {ns_per_op:.1} ns/op",
    );
}
