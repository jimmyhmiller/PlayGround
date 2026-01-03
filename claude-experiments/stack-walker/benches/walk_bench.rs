use std::time::Instant;

use stack_walker::{
    walk_current_stack, WalkConfig,
    UnsafeDirectReader, NativeStackWalker, StackWalker, capture_current_native,
};

// Force frame pointer preservation with dynamic dispatch to avoid monomorphization limit
#[inline(never)]
fn deep_call(depth: usize, f: &dyn Fn()) {
    std::hint::black_box(&depth);
    if depth == 0 {
        f();
    } else {
        deep_call(depth - 1, f);
    }
    std::hint::black_box(());
}

fn main() {
    let config = WalkConfig {
        validate_return_addresses: false,
        ..Default::default()
    };

    // Warm up
    for _ in 0..100 {
        let _ = walk_current_stack();
    }

    // Benchmark shallow stack
    let iterations = 100_000;
    let start = Instant::now();
    for _ in 0..iterations {
        let trace = walk_current_stack();
        std::hint::black_box(trace);
    }
    let elapsed = start.elapsed();
    let shallow_depth = walk_current_stack().len();
    println!("Shallow stack ({} frames):", shallow_depth);
    println!("  {} iterations in {:?}", iterations, elapsed);
    println!("  {:.0} ns/walk", elapsed.as_nanos() as f64 / iterations as f64);
    println!("  {:.0} ns/frame", elapsed.as_nanos() as f64 / (iterations * shallow_depth) as f64);

    // Benchmark deeper stack
    let iterations = 10_000;
    deep_call(50, &|| {
        let start = Instant::now();
        for _ in 0..iterations {
            let trace = walk_current_stack();
            std::hint::black_box(trace);
        }
        let elapsed = start.elapsed();
        let deep_depth = walk_current_stack().len();
        println!("\nDeep stack ({} frames):", deep_depth);
        println!("  {} iterations in {:?}", iterations, elapsed);
        println!("  {:.0} ns/walk", elapsed.as_nanos() as f64 / iterations as f64);
        println!("  {:.0} ns/frame", elapsed.as_nanos() as f64 / (iterations * deep_depth) as f64);
    });

    // Benchmark just frame pointer reading (no allocation)
    let iterations = 100_000;
    let walker = NativeStackWalker::new();
    let mut reader = UnsafeDirectReader::new();

    let start = Instant::now();
    for _ in 0..iterations {
        let regs = capture_current_native();
        let mut count = 0usize;
        walker.walk_with(&regs, &mut reader, &config, |_frame| {
            count += 1;
            true
        });
        std::hint::black_box(count);
    }
    let elapsed = start.elapsed();
    let shallow_depth = walk_current_stack().len();
    println!("\nCallback-only ({} frames, no Vec allocation):", shallow_depth);
    println!("  {} iterations in {:?}", iterations, elapsed);
    println!("  {:.0} ns/walk", elapsed.as_nanos() as f64 / iterations as f64);
    println!("  {:.0} ns/frame", elapsed.as_nanos() as f64 / (iterations * shallow_depth) as f64);

    // Benchmark very deep stack with callback
    println!("\n--- Deep stack callback benchmark ---");
    deep_call(100, &|| {
        let config = WalkConfig {
            validate_return_addresses: false,
            ..Default::default()
        };
        let iterations = 10_000;
        let walker = NativeStackWalker::new();
        let mut reader = UnsafeDirectReader::new();

        let start = Instant::now();
        for _ in 0..iterations {
            let regs = capture_current_native();
            let mut count = 0usize;
            walker.walk_with(&regs, &mut reader, &config, |_frame| {
                count += 1;
                true
            });
            std::hint::black_box(count);
        }
        let elapsed = start.elapsed();
        let depth = walk_current_stack().len();
        println!("Deep callback ({} frames):", depth);
        println!("  {} iterations in {:?}", iterations, elapsed);
        println!("  {:.0} ns/walk", elapsed.as_nanos() as f64 / iterations as f64);
        println!("  {:.0} ns/frame", elapsed.as_nanos() as f64 / (iterations * depth) as f64);
    });
}
