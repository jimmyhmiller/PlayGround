/*!
 * Real-World JavaScript Library Benchmarks (Rust)
 */

use std::env;
use std::fs;
use std::time::Instant;
use std::sync::Arc;

// OXC imports
use oxc_allocator::Allocator;
use oxc_parser::Parser as OxcParser;
use oxc_span::SourceType;

// SWC imports
use swc_common::SourceMap;
use swc_ecma_parser::{Parser as SwcParser, StringInput, Syntax, EsSyntax};

const LIBS_DIR: &str = "../real-world-libs/";

struct BenchmarkResult {
    name: String,
    library: String,
    avg_millis: f64,
    throughput_kb_per_ms: f64,
}

const DEFAULT_WARMUP_ITERATIONS: usize = 5;
const DEFAULT_MEASUREMENT_ITERATIONS: usize = 10;

fn benchmark<F>(name: &str, library: &str, code: &str, mut parse_fn: F, warmup_iterations: usize, iterations: usize) -> BenchmarkResult
where
    F: FnMut(&str),
{
    // Warmup
    for _ in 0..warmup_iterations {
        parse_fn(code);
    }

    // Measure
    let start = Instant::now();
    for _ in 0..iterations {
        parse_fn(code);
    }
    let elapsed = start.elapsed();

    // Use as_secs_f64() for precise sub-millisecond measurements
    let avg_millis = (elapsed.as_secs_f64() * 1000.0) / iterations as f64;
    let size_kb = code.len() as f64 / 1024.0;
    let throughput_kb_per_ms = size_kb / avg_millis;

    BenchmarkResult {
        name: name.to_string(),
        library: library.to_string(),
        avg_millis,
        throughput_kb_per_ms,
    }
}

fn parse_with_oxc(code: &str) {
    let allocator = Allocator::default();
    let source_type = SourceType::default();
    let _ = OxcParser::new(&allocator, code, source_type).parse();
}

fn parse_with_swc(code: &str) {
    let cm = Arc::new(SourceMap::default());
    let fm = cm.new_source_file(swc_common::FileName::Anon.into(), code.to_string());
    let input = StringInput::from(&*fm);

    let syntax = Syntax::Es(EsSyntax {
        jsx: false,
        fn_bind: false,
        decorators: false,
        decorators_before_export: false,
        export_default_from: false,
        import_attributes: true,
        allow_super_outside_method: false,
        allow_return_outside_function: false,
        auto_accessors: false,
        explicit_resource_management: false,
    });

    let mut parser = SwcParser::new(syntax, input, None);
    let _ = parser.parse_module();
}

fn run_benchmark_suite(library_name: &str, filename: &str, warmup_iterations: usize, iterations: usize) {
    let path = format!("{}{}", LIBS_DIR, filename);
    let code = match fs::read_to_string(&path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Failed to read {}: {}", path, e);
            return;
        }
    };

    let size_kb = code.len() as f64 / 1024.0;

    println!("\n{}", "=".repeat(80));
    println!("Library: {}", library_name);
    println!("Size: {:.1} KB ({} bytes)", size_kb, code.len());
    println!("{}\n", "=".repeat(80));

    let mut results = vec![
        benchmark("OXC (Rust)", library_name, &code, |c| parse_with_oxc(c), warmup_iterations, iterations),
        benchmark("SWC (Rust)", library_name, &code, |c| parse_with_swc(c), warmup_iterations, iterations),
    ];

    results.sort_by(|a, b| a.avg_millis.partial_cmp(&b.avg_millis).unwrap());

    println!("Results:");
    println!("{}", "-".repeat(80));
    println!("{:<20} | {:>15} | {:>15} | {:>20}", "Parser", "Avg Time (ms)", "vs Fastest", "Throughput (KB/ms)");
    println!("{}", "-".repeat(80));

    let fastest = results[0].avg_millis;

    for (i, result) in results.iter().enumerate() {
        let ratio = result.avg_millis / fastest;
        let indicator = match i {
            0 => "🥇",
            1 => "🥈",
            _ => "  ",
        };

        println!(
            "{} {:<18} | {:>15.3} | {:>15.2}x | {:>20.1}",
            indicator,
            result.name,
            result.avg_millis,
            ratio,
            result.throughput_kb_per_ms
        );
    }
}

fn main() {
    // Parse command line arguments: benchmark-real-world [warmup] [measurement]
    let args: Vec<String> = env::args().collect();
    let warmup_iterations = args.get(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_WARMUP_ITERATIONS);
    let measurement_iterations = args.get(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_MEASUREMENT_ITERATIONS);

    println!("\n{}", "═".repeat(80));
    println!("Real-World JavaScript Library Benchmarks (Rust)");
    println!("{}", "═".repeat(80));
    println!("Warmup iterations: {}", warmup_iterations);
    println!("Measurement iterations: {}", measurement_iterations);
    println!("{}", "═".repeat(80));

    run_benchmark_suite("React", "react.production.min.js", warmup_iterations, measurement_iterations);
    run_benchmark_suite("Vue 3", "vue.global.prod.js", warmup_iterations, measurement_iterations);
    run_benchmark_suite("React DOM", "react-dom.production.min.js", warmup_iterations, measurement_iterations);
    run_benchmark_suite("Lodash", "lodash.js", warmup_iterations, measurement_iterations);
    run_benchmark_suite("Three.js", "three.js", warmup_iterations, measurement_iterations);
    run_benchmark_suite("TypeScript Compiler", "typescript.js", warmup_iterations, measurement_iterations);

    println!("\n{}", "═".repeat(80));
    println!("Benchmark complete!");
    println!("{}", "═".repeat(80));
}
