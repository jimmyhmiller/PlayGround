//! Standalone benchmark of the *statically compiled* (AOT) SIMD JS stage-1.
//!
//! No JIT, no MLIR, no LLVM at runtime — this binary links the native-NEON
//! object that `simd-lang compile` emitted and calls it directly. It:
//!   1. runs the AOT kernel and the pure-Rust reference (`simd_lang::stage1`)
//!      and asserts their two bitmaps are byte-identical (correctness gate), and
//!   2. reports stage-1 throughput (MB/s) for both, in the same shape the JIT
//!      path prints via `simd-lang jit-stage1`, so the numbers line up.
//!
//! Usage:  aot-bench <file.js> [iters]

// The auto-generated bindings: an `extern "C"` block (with
// `#[link(name = "js_stage1", kind = "static")]`) plus a safe `js_stage1`
// wrapper over the memref ABI.
mod aot {
    #![allow(dead_code)]
    include!("../generated/js_stage1.rs");
}

use std::hint::black_box;
use std::time::Instant;

/// Pad to a multiple of 64 so the kernel reads whole chunks (extra bytes NUL).
fn pad64(src: &[u8]) -> Vec<u8> {
    let mut v = src.to_vec();
    while v.len() % 64 != 0 {
        v.push(0);
    }
    v
}

/// Run the AOT kernel, returning (token-start count, start_masks, word_masks).
fn run_aot(raw: &[u8]) -> (i32, Vec<u64>, Vec<u64>) {
    let mut padded = pad64(raw);
    let nchunks = padded.len() / 64 + 1;
    let mut start_masks = vec![0u64; nchunks];
    let mut word_masks = vec![0u64; nchunks];
    let n = aot::js_stage1(&mut padded, &mut start_masks, &mut word_masks);
    (n, start_masks, word_masks)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = args.get(1).map(String::as_str).unwrap_or_else(|| {
        eprintln!("Usage: aot-bench <file.js> [iters]");
        std::process::exit(1);
    });
    let iters: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(200);

    let raw = std::fs::read(path).unwrap_or_else(|e| {
        eprintln!("cannot read {path}: {e}");
        std::process::exit(1);
    });
    let bytes = raw.len();
    let mb = bytes as f64 / 1_000_000.0;

    // ── Correctness: AOT bitmaps must equal the pure-Rust reference ──────────
    let (n_aot, aot_start, aot_word) = run_aot(&raw);
    let (ref_start, ref_word) = simd_lang::stage1::stage1(&raw);
    assert_eq!(
        aot_start, ref_start,
        "AOT start_masks diverge from the pure-Rust reference"
    );
    assert_eq!(
        aot_word, ref_word,
        "AOT word_masks diverge from the pure-Rust reference"
    );
    let n_ref: u32 = ref_start.iter().map(|w| w.count_ones()).sum();
    assert_eq!(n_aot as u32, n_ref, "AOT token-start count mismatch");

    // Reusable buffers for the timing loops (alloc out of the hot path).
    let mut padded = pad64(&raw);
    let nchunks = padded.len() / 64 + 1;
    let mut sm = vec![0u64; nchunks];
    let mut wm = vec![0u64; nchunks];

    // ── Time the AOT kernel ─────────────────────────────────────────────────
    for _ in 0..5 {
        black_box(aot::js_stage1(&mut padded, &mut sm, &mut wm));
    }
    let (aot_med, aot_best) = measure(iters, || {
        black_box(aot::js_stage1(&mut padded, &mut sm, &mut wm));
    }, mb);

    // ── Time the pure-Rust reference (NEON intrinsics, no codegen layer) ─────
    for _ in 0..5 {
        black_box(simd_lang::stage1::stage1(&raw));
    }
    let (ref_med, ref_best) = measure(iters, || {
        black_box(simd_lang::stage1::stage1(&raw));
    }, mb);

    println!("=== AOT (statically compiled libjs_stage1.a — no JIT, no MLIR) ===");
    println!("file            : {path}  ({bytes} bytes, {:.2} MB)", mb);
    println!("token starts    : {n_aot}");
    println!("correctness     : bitmaps byte-identical to pure-Rust reference ✓");
    println!("JIT compile time: 0 ms  (compiled ahead of time, linked once)");
    println!("stage-1 MB/s    : {aot_med:.0} median   {aot_best:.0} best   ({iters} iters)");
    println!();
    println!("[ref] pure-Rust stage1 (hand-written NEON): {ref_med:.0} median   {ref_best:.0} best MB/s");
}

/// Run `f` `iters` times, returning (median, best) MB/s given `mb` bytes/call.
fn measure(iters: usize, mut f: impl FnMut(), mb: f64) -> (f64, f64) {
    let mut best = 0.0f64;
    let mut samples = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t = Instant::now();
        f();
        let mbps = mb / t.elapsed().as_secs_f64();
        best = best.max(mbps);
        samples.push(mbps);
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (samples[samples.len() / 2], best)
}
