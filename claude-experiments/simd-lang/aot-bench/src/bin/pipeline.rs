//! Full **text → JSIR** pipeline driven by the *real* AOT-compiled `.simd`
//! stage-1 kernel (no JIT, no MLIR at runtime), vs oxc.
//!
//! This is the honest end-to-end comparison: stage-1 here is the MLIR-compiled
//! native-NEON kernel (`libjs_stage1.a`), not jsir-parse's bundled Rust-NEON
//! `stage1`. We feed its bitmaps straight into `jsir_parse::parse_with_masks`.
//!
//! Usage:  pipeline [statements]

mod aot {
    #![allow(dead_code)]
    include!("../../generated/js_stage1.rs");
}

use std::hint::black_box;
use std::time::{Duration, Instant};

use oxc_allocator::Allocator;
use oxc_parser::Parser;
use oxc_span::SourceType;

fn pad64(src: &[u8]) -> Vec<u8> {
    let mut v = src.to_vec();
    while v.len() % 64 != 0 {
        v.push(0);
    }
    v
}

/// Stage-1 bitmaps via the AOT `.simd` kernel.
fn aot_masks(raw: &[u8]) -> (Vec<u64>, Vec<u64>) {
    let mut padded = pad64(raw);
    let nchunks = padded.len() / 64 + 1;
    let mut sm = vec![0u64; nchunks];
    let mut wm = vec![0u64; nchunks];
    aot::js_stage1(&mut padded, &mut sm, &mut wm);
    (sm, wm)
}

/// A subset program both front ends accept.
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
    f();
    let t = Instant::now();
    for _ in 0..iters {
        f();
    }
    let per = t.elapsed() / iters;
    let mbps = (bytes as f64 / (1 << 20) as f64) / per.as_secs_f64();
    println!("  {name:<34} {per:>10.2?}/iter   {mbps:>8.0} MiB/s");
    per
}

fn main() {
    let stmts: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(20_000);
    let src = generate(stmts);
    let bytes = src.len();
    let iters = 50;
    println!("source: {} statements, {:.2} MiB\n", stmts, bytes as f64 / (1 << 20) as f64);

    // Correctness: AOT bitmaps must equal the bundled Rust-NEON reference, and
    // the resulting IR must match the default (Rust-NEON-fed) parse byte-for-byte.
    let (sm, wm) = aot_masks(src.as_bytes());
    let (rsm, rwm) = simd_lang::stage1::stage1(src.as_bytes());
    assert_eq!((sm.as_slice(), wm.as_slice()), (rsm.as_slice(), rwm.as_slice()), "AOT≠ref bitmaps");
    let aot_ir = jsir_parse::parse_with_masks(&src, &sm, &wm).unwrap().print();
    let def_ir = jsir_parse::parse_to_module(&src).unwrap().print();
    assert_eq!(aot_ir, def_ir, "AOT-fed IR diverged from default");
    println!("correctness: AOT stage-1 → identical IR ✓\n");

    println!("text → JSIR (real AOT .simd stage-1 + our parser):");
    let ours = bench("ours: AOT .simd → js-ir", bytes, iters, || {
        let (sm, wm) = aot_masks(src.as_bytes());
        let m = jsir_parse::parse_with_masks(&src, &sm, &wm).unwrap();
        black_box(m.op_count());
    });

    println!("\noxc (for reference):");
    let oxc = bench("oxc: lex+parse → AST", bytes, iters, || {
        let alloc = Allocator::default();
        let ret = Parser::new(&alloc, &src, SourceType::default()).parse();
        black_box(ret.program.body.len());
    });
    let sem = bench("oxc: parse + semantic", bytes, iters, || {
        let alloc = Allocator::default();
        let ret = Parser::new(&alloc, &src, SourceType::default()).parse();
        let s = oxc_semantic::SemanticBuilder::new().build(&ret.program);
        black_box(s.semantic.nodes().len());
    });

    println!("\nstage-1 only (AOT .simd kernel):");
    bench("aot .simd stage-1", bytes, 200, || {
        black_box(aot_masks(src.as_bytes()));
    });

    println!("\nours vs oxc bare AST:       {:.2}x", oxc.as_secs_f64() / ours.as_secs_f64());
    println!("ours vs oxc parse+semantic: {:.2}x", sem.as_secs_f64() / ours.as_secs_f64());
}
