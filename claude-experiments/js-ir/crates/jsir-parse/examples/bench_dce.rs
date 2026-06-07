//! DCE comparison: **text → dead-code-eliminated** across the three Rust JS
//! ecosystems. Each runs its *DCE-only* path (not full minify):
//!   - ours: parse → JSIR → `jsir_ir::build::dce`
//!   - oxc:  parse → AST  → `Minifier::dce`
//!   - swc:  parse → AST  → resolver + `simplify::dce`
//!
//! Caveat: the three don't do *identical* DCE (oxc's is the most aggressive,
//! swc's tree-shakes unused bindings, ours drops unread `var`s) — so we report
//! both time and how much each removed on the same dead-code-heavy input.
//!
//! Usage:  cargo run --release -p jsir-parse --example bench_dce [statements]

use std::hint::black_box;
use std::time::{Duration, Instant};

use oxc_allocator::Allocator;
use oxc_minifier::{CompressOptions, Minifier, MinifierOptions};
use oxc_parser::Parser;
use oxc_span::SourceType;

use swc_common::{Globals, Mark, GLOBALS};
use swc_ecma_ast::Program;
use swc_ecma_transforms_base::resolver;
use swc_ecma_transforms_optimization::simplify::dce::{dce, Config};
use swc_ecma_visit::VisitMutWith;

fn swc_stmt_count(p: &Program) -> usize {
    match p {
        Program::Module(m) => m.body.len(),
        Program::Script(s) => s.body.len(),
    }
}

/// Half the declarations are dead (declared, never read); half are live (read
/// via a following assignment). All three DCEs should drop the dead ones.
fn generate(stmts: usize) -> String {
    let mut s = String::with_capacity(stmts * 40);
    for i in 0..stmts {
        if i % 2 == 0 {
            s.push_str(&format!("var dead{i} = {i} + 1;\n")); // never referenced
        } else {
            s.push_str(&format!("var live{i} = {i};\n"));
            s.push_str(&format!("live{i} = live{i} + 2;\n")); // references live{i}
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
    println!("  {name:<30} {per:>10.2?}/iter   {mbps:>7.0} MiB/s");
    per
}

fn main() {
    let stmts: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(10_000);
    let src = generate(stmts);
    let bytes = src.len();
    let iters = 30;
    println!("source: {} stmts, {:.2} MiB ({} dead vars)\n", stmts, bytes as f64 / (1 << 20) as f64, stmts / 2);

    // Sanity: copying DCE and in-place DCE remove the dead vars and agree.
    let m = jsir_parse::parse_to_module(&src).unwrap();
    let before = m.op_count();
    let (after_m, removed) = jsir_ir::build::dce(&m);
    let mut m2 = jsir_parse::parse_to_module(&src).unwrap();
    let removed2 = jsir_ir::build::dce_in_place(&mut m2);
    assert_eq!(removed, removed2, "in-place and copying DCE remove different counts");
    assert_eq!(after_m.print(), m2.print(), "in-place DCE output differs from copying DCE");
    println!("ours: removed {removed} dead var-decls ({before} ops); in-place == copying ✓\n");

    println!("text → DCE'd (parse + DCE-only):");
    bench("ours: JSIR + dce (copying)", bytes, iters, || {
        let m = jsir_parse::parse_to_module(&src).unwrap();
        let (out, r) = jsir_ir::build::dce(&m);
        black_box((out.op_count(), r));
    });
    bench("ours: JSIR + dce (in-place)", bytes, iters, || {
        let mut m = jsir_parse::parse_to_module(&src).unwrap();
        let r = jsir_ir::build::dce_in_place(&mut m);
        black_box((m.op_count(), r));
    });
    bench("oxc: AST + Minifier::dce", bytes, iters, || {
        let alloc = Allocator::default();
        let mut ret = Parser::new(&alloc, &src, SourceType::default()).parse();
        let opts = MinifierOptions { mangle: None, compress: Some(CompressOptions::default()) };
        Minifier::new(opts).dce(&alloc, &mut ret.program);
        black_box(ret.program.body.len());
    });

    // swc: resolver (assign scopes/marks) + simplify::dce TreeShaker, in GLOBALS.
    GLOBALS.set(&Globals::default(), || {
        // correctness: confirm swc drops the dead vars
        let mut p = jsir_swc::parse(&src).unwrap();
        let before = swc_stmt_count(&p);
        let u = Mark::new();
        let t = Mark::new();
        p.visit_mut_with(&mut resolver(u, t, false));
        p.visit_mut_with(&mut dce(Config { top_level: true, ..Default::default() }, u));
        println!("\nswc: {before} top-level stmts → {} after dce", swc_stmt_count(&p));

        bench("swc: AST + simplify::dce", bytes, iters, || {
            let mut p = jsir_swc::parse(&src).unwrap();
            let u = Mark::new();
            let t = Mark::new();
            p.visit_mut_with(&mut resolver(u, t, false));
            p.visit_mut_with(&mut dce(Config { top_level: true, ..Default::default() }, u));
            black_box(swc_stmt_count(&p));
        });
    });
}
