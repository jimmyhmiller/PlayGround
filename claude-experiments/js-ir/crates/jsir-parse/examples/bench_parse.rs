//! Front-end throughput: **our no-AST `text → SIMD tokens → js-ir`** vs **oxc's
//! `text → AST`**, on the same subset program.
//!
//!   cargo run --release -p jsir-parse --example bench_parse
//!
//! Both produce a full in-memory structure from source text. Ours emits the
//! columnar JSIR `Module` directly (no AST); oxc emits its arena AST. We also
//! time the SIMD lexer alone, for reference.

use std::hint::black_box;
use std::time::{Duration, Instant};

use oxc_allocator::Allocator;
use oxc_parser::Parser;
use oxc_span::SourceType;

/// A program using only the supported subset (var decls, assignments, binary /
/// update expressions, expression statements), so both front ends accept it.
fn generate(stmts: usize) -> String {
    let mut s = String::with_capacity(stmts * 40);
    for i in 0..stmts {
        match i % 3 {
            0 => s.push_str(&format!("var v{i} = {i} + {a} * 3 - 1;\n", a = i * 2)),
            1 => s.push_str(&format!("v{j} = v{j} + {i} * 2;\n", j = i.saturating_sub(1))),
            _ => s.push_str(&format!("var w{i} = {a} & 7 | {i};\n", a = i + 3)),
        }
    }
    s
}

fn bench(name: &str, bytes: usize, iters: u32, mut f: impl FnMut()) -> Duration {
    f(); // warm up
    let t = Instant::now();
    for _ in 0..iters {
        f();
    }
    let per = t.elapsed() / iters;
    let mbps = (bytes as f64 / (1 << 20) as f64) / per.as_secs_f64();
    println!("  {name:<28} {per:>10.2?}/iter   {mbps:>8.0} MiB/s");
    per
}

fn main() {
    let stmts: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(20_000);
    let src = generate(stmts);
    let bytes = src.len();
    let iters = 50;
    println!("source: {} statements, {:.2} MiB\n", stmts, bytes as f64 / (1 << 20) as f64);

    // sanity: our parser accepts it
    let m = jsir_parse::parse_to_module(&src).expect("our parser");
    println!("our IR: {} ops\n", m.op_count());

    println!("text → JSIR (apples-to-apples: both produce the same IR):");
    let ours = bench("ours: SIMD → js-ir (no AST)", bytes, iters, || {
        let m = jsir_parse::parse_to_module(&src).unwrap();
        black_box(m.op_count());
    });
    // The swc AST→jsir converter is ~O(n²); only run it for small inputs.
    let ast = if stmts <= 3000 {
        bench("swc: parse→AST→ast2hir", bytes, 5, || {
            let op = jsir_swc::source_to_ir(&src).unwrap();
            black_box(op); // build only (no print)
        })
    } else {
        println!("  swc: parse→AST→ast2hir       (skipped; O(n²) converter)");
        Duration::ZERO
    };

    println!("\ntext → richer structure (oxc passes, closer to our IR):");
    let oxc = bench("oxc:  lex+parse → AST", bytes, iters, || {
        let alloc = Allocator::default();
        let ret = Parser::new(&alloc, &src, SourceType::default()).parse();
        black_box(ret.program.body.len());
    });
    let oxc_sem = bench("oxc:  parse + semantic", bytes, iters, || {
        let alloc = Allocator::default();
        let ret = Parser::new(&alloc, &src, SourceType::default()).parse();
        let sem = oxc_semantic::SemanticBuilder::new().build(&ret.program);
        black_box(sem.semantic.nodes().len());
    });

    println!("\nlexing only (reference):");
    bench("ours: SIMD tokenize", bytes, iters, || {
        black_box(jsir_parse::tokenize_simd(&src).unwrap().len());
    });

    println!(
        "\ntext→JSIR speedup vs AST-based (swc parse+lower): {:.2}x",
        ast.as_secs_f64() / ours.as_secs_f64()
    );
    println!(
        "ours vs oxc-to-AST (oxc does less — only an AST): {:.2}x",
        oxc.as_secs_f64() / ours.as_secs_f64()
    );
    println!(
        "ours vs oxc parse+semantic (a richer, analyzable repr): {:.2}x",
        oxc_sem.as_secs_f64() / ours.as_secs_f64()
    );
}
