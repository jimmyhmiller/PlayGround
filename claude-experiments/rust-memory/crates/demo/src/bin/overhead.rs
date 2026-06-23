//! Measures memscope's space + time overhead.
//!
//!   overhead <off|full|sampled> <N> [rate]
//!
//! Allocates N retained small boxes and N transient allocations, then reports:
//!   - tracked payload bytes (what the program actually asked for)
//!   - process RSS (via `ps`) — real memory footprint
//!   - per-live-allocation bookkeeping overhead
//!   - allocation throughput (ns per alloc)
//!
//! Run each mode in its own process and compare.

use std::time::Instant;

use memscope::{MemScope, Mode};

#[global_allocator]
static GLOBAL: MemScope = MemScope::system();

fn rss_kb() -> u64 {
    let out = std::process::Command::new("ps")
        .args(["-o", "rss=", "-p"])
        .arg(std::process::id().to_string())
        .output()
        .ok();
    out.and_then(|o| String::from_utf8(o.stdout).ok())
        .and_then(|s| s.trim().parse::<u64>().ok())
        .unwrap_or(0)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mode_s = args.first().map(String::as_str).unwrap_or("full");
    let n: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(1_000_000);
    let rate: u32 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(100);

    let mode = match mode_s {
        "off" => Mode::Off,
        "sampled" => Mode::Sampled,
        _ => Mode::Full,
    };
    memscope::set_mode(mode);
    if matches!(mode, Mode::Sampled) {
        memscope::set_sample_rate(rate);
    }

    let rss_before = rss_kb();

    // Retained small allocations — these stay in the live table.
    let t0 = Instant::now();
    let mut retained: Vec<Box<u64>> = Vec::with_capacity(n);
    for i in 0..n {
        retained.push(Box::new(i as u64));
    }
    let alloc_elapsed = t0.elapsed();

    // Transient allocations — exercise alloc+dealloc churn (event ring, table
    // insert/remove) without growing the live set.
    let t1 = Instant::now();
    let churn = n;
    for i in 0..churn {
        let v: Vec<u8> = Vec::with_capacity(32 + (i & 63));
        std::hint::black_box(&v);
    }
    let churn_elapsed = t1.elapsed();

    let rss_after = rss_kb();
    let stats = memscope::stats();
    std::hint::black_box(&retained);

    let payload_kb = (n * std::mem::size_of::<u64>()) as u64 / 1024 // box payloads
        + (n * std::mem::size_of::<Box<u64>>()) as u64 / 1024; // the Vec of ptrs
    let rss_delta = rss_after.saturating_sub(rss_before);
    let overhead_kb = rss_delta.saturating_sub(payload_kb);

    println!("mode={mode_s} n={n}");
    println!("  tracked live allocations : {}", stats_live_count());
    println!("  tracked live bytes       : {} KiB", stats.live_bytes / 1024);
    println!("  RSS before / after       : {rss_before} / {rss_after} KiB  (Δ {rss_delta} KiB)");
    println!("  payload (boxes + vec)    : ~{payload_kb} KiB");
    println!("  bookkeeping overhead     : ~{overhead_kb} KiB");
    if n > 0 {
        println!(
            "  overhead per live alloc  : ~{:.1} bytes",
            (overhead_kb * 1024) as f64 / n as f64
        );
    }
    println!(
        "  retained-alloc time      : {:?}  ({:.1} ns/alloc)",
        alloc_elapsed,
        alloc_elapsed.as_nanos() as f64 / n as f64
    );
    println!(
        "  churn alloc+free time    : {:?}  ({:.1} ns/op)",
        churn_elapsed,
        churn_elapsed.as_nanos() as f64 / churn as f64
    );
}

fn stats_live_count() -> usize {
    // snapshot() is the only way to count entries; do it with tracking paused so
    // it doesn't perturb the number it's reporting.
    memscope::set_mode(Mode::Off);
    let snap = memscope::snapshot();
    snap.live.len()
}
