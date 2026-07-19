//! Demo + benchmark. First asserts the generic transform and both hand-written
//! direct transforms emit identical code; then times all four combos (generic
//! vs direct, on each backend) on a large workload — parsing fresh each
//! iteration (untimed) and timing only the transform.
//!
//! Run: `cargo run --release` (optional args: `<statements> <iters>`).

use std::hint::black_box;
use std::time::{Duration, Instant};

use minimal_optchain::generic::desugar;
use minimal_optchain::import_map::ImportMap;
use minimal_optchain::{Backend, Oxc, Swc};

// ---------------------------------------------------------------------------
// The four end-to-end paths (parse + transform + codegen), for correctness.
// ---------------------------------------------------------------------------

fn oxc_generic(src: &str) -> String {
    let arena = Default::default();
    let mut prog = Oxc::parse(&arena, src);
    desugar::<Oxc>(Oxc::ctx(&arena), &mut prog);
    Oxc::codegen(&prog)
}

fn oxc_direct(src: &str) -> String {
    let arena = Default::default();
    let mut prog = Oxc::parse(&arena, src);
    minimal_optchain::oxc::direct(&mut prog, Oxc::ctx(&arena));
    Oxc::codegen(&prog)
}

fn swc_generic(src: &str) -> String {
    let mut prog = Swc::parse(&(), src);
    desugar::<Swc>((), &mut prog);
    Swc::codegen(&prog)
}

fn swc_direct(src: &str) -> String {
    let mut prog = Swc::parse(&(), src);
    minimal_optchain::swc::direct(&mut prog);
    Swc::codegen(&prog)
}

fn norm(s: &str) -> String {
    s.chars().filter(|c| !c.is_whitespace()).map(|c| if c == '\'' { '"' } else { c }).collect()
}

fn correctness_check() {
    let cases = ["a?.b;", "a?.b.c;", "a.b?.c;", "x?.y?.z;", "a?.[k];", "a.b.c;"];
    println!("{:<14} -> {}", "input", "desugared (all four paths agree)");
    for src in cases {
        let og = oxc_generic(src);
        let od = oxc_direct(src);
        let sg = swc_generic(src);
        let sd = swc_direct(src);
        assert_eq!(norm(&og), norm(&od), "oxc generic vs direct differ for {src}");
        assert_eq!(norm(&sg), norm(&sd), "swc generic vs direct differ for {src}");
        assert_eq!(norm(&og), norm(&sg), "oxc vs swc differ for {src}");
        println!("{:<14} -> {}", src, og.trim());
    }
    println!();
}

// ---------------------------------------------------------------------------
// ImportMap: the generic binding-resolution analysis, proven identical across
// backends. For each expression we record how the ImportMap resolves it; oxc
// (SymbolId side-table) and swc (SyntaxContext on the ident) must agree.
// ---------------------------------------------------------------------------

/// One resolution line per interesting expression: `<free>` for a global,
/// `source#name` for an import reference, `source#*` for a namespace binding.
fn collect_resolution<'a, B: Backend>(
    map: &ImportMap<B>,
    sem: &B::Semantics<'a>,
    e: &B::Expr<'a>,
    out: &mut Vec<String>,
) {
    if let Some(name) = B::ident_name(e) {
        if B::is_free_ident(sem, e) {
            out.push(format!("{name} = <free>"));
        } else if let Some(desc) = map.describe(sem, e) {
            out.push(format!("{name} = {desc}"));
        }
    } else if let Some((obj, prop)) = B::as_static_member(e) {
        if let (Some(oname), Some(desc)) = (B::ident_name(obj), map.describe(sem, e)) {
            out.push(format!("{oname}.{prop} = {desc}"));
        }
    }
}

fn import_report_oxc(src: &str) -> Vec<String> {
    let arena = Default::default();
    let prog = Oxc::parse(&arena, src);
    let sem = Oxc::build_semantics(&prog);
    let map = ImportMap::<Oxc>::analyze(&prog, &sem);
    let mut out = Vec::new();
    Oxc::visit_exprs(&prog, |e| collect_resolution::<Oxc>(&map, &sem, e, &mut out));
    out.sort();
    out
}

fn import_report_swc(src: &str) -> Vec<String> {
    let prog = Swc::parse(&(), src);
    let sem = Swc::build_semantics(&prog);
    let map = ImportMap::<Swc>::analyze(&prog, &sem);
    let mut out = Vec::new();
    Swc::visit_exprs(&prog, |e| collect_resolution::<Swc>(&map, &sem, e, &mut out));
    out.sort();
    out
}

fn import_correctness_check() {
    // Every import shape, plus a global, a re-used import, and a plain local.
    let src = concat!(
        "import { useState } from \"react\";\n",
        "import { foo as bar } from \"lib\";\n",
        "import def from \"mod\";\n",
        "import * as ns from \"ns\";\n",
        "useState(unknownGlobal);\n", // callee is an import; arg is a free global
        "bar;\n",                     // renamed named import
        "def;\n",                     // default import
        "ns.thing;\n",                // namespace member access
        "ns.other;\n",
        "useState;\n",                // same import, second reference
        "let local = 1;\n",
        "local;\n",                   // a resolved local: not free, not an import
    );
    let o = import_report_oxc(src);
    let s = import_report_swc(src);
    assert_eq!(o, s, "oxc and swc ImportMap disagree:\n oxc={o:#?}\n swc={s:#?}");
    println!("ImportMap resolution — oxc and swc agree on all {} sites:", o.len());
    for line in &o {
        println!("  {line}");
    }
    println!();
}

// ---------------------------------------------------------------------------
// Benchmark: parse fresh each iteration (untimed); time only the transform.
// ---------------------------------------------------------------------------

/// A workload of member-only optional chains (calls would hit the hard error),
/// in a few shapes so the transform touches every projection arm.
fn workload(n: usize) -> String {
    let mut s = String::with_capacity(n * 32);
    for i in 0..n {
        let stmt = match i % 4 {
            0 => format!("obj{i}?.alpha?.beta.gamma;\n"),
            1 => format!("obj{i}.alpha?.beta?.gamma;\n"),
            2 => format!("obj{i}?.alpha.beta?.[key{i}];\n"),
            _ => format!("obj{i}?.alpha?.beta?.gamma.delta;\n"),
        };
        s.push_str(&stmt);
    }
    s
}

/// Warm up, then average `iters` runs; each `run` parses fresh (untimed) and
/// returns just the transform's duration.
fn bench(iters: usize, mut run: impl FnMut() -> Duration) -> Duration {
    for _ in 0..(iters / 8).max(1) {
        black_box(run());
    }
    let mut total = Duration::ZERO;
    for _ in 0..iters {
        total += run();
    }
    total / iters as u32
}

fn time_oxc_generic(src: &str) -> Duration {
    let arena = Default::default();
    let mut prog = Oxc::parse(&arena, src);
    let cx = Oxc::ctx(&arena);
    let t = Instant::now();
    desugar::<Oxc>(cx, &mut prog);
    let d = t.elapsed();
    black_box(&prog);
    d
}

fn time_oxc_direct(src: &str) -> Duration {
    let arena = Default::default();
    let mut prog = Oxc::parse(&arena, src);
    let cx = Oxc::ctx(&arena);
    let t = Instant::now();
    minimal_optchain::oxc::direct(&mut prog, cx);
    let d = t.elapsed();
    black_box(&prog);
    d
}

fn time_swc_generic(src: &str) -> Duration {
    let mut prog = Swc::parse(&(), src);
    let t = Instant::now();
    desugar::<Swc>((), &mut prog);
    let d = t.elapsed();
    black_box(&prog);
    d
}

fn time_swc_direct(src: &str) -> Duration {
    let mut prog = Swc::parse(&(), src);
    let t = Instant::now();
    minimal_optchain::swc::direct(&mut prog);
    let d = t.elapsed();
    black_box(&prog);
    d
}

fn main() {
    correctness_check();
    import_correctness_check();

    let mut args = std::env::args().skip(1);
    // Large workload so each timed transform runs for milliseconds — the fast
    // oxc path is otherwise swamped by timer/cache jitter.
    let n: usize = args.next().and_then(|a| a.parse().ok()).unwrap_or(20_000);
    let iters: usize = args.next().and_then(|a| a.parse().ok()).unwrap_or(1000);
    let src = workload(n);
    println!("workload: {n} optional-chain statements, {iters} iters (transform only)\n");

    let og = bench(iters, || time_oxc_generic(&src));
    let od = bench(iters, || time_oxc_direct(&src));
    let sg = bench(iters, || time_swc_generic(&src));
    let sd = bench(iters, || time_swc_direct(&src));

    let us = |d: Duration| d.as_secs_f64() * 1e6;
    let overhead = |g: Duration, d: Duration| (us(g) / us(d) - 1.0) * 100.0;
    println!("{:<8} {:>12} {:>12} {:>10}", "backend", "generic (us)", "direct (us)", "overhead");
    println!("{:<8} {:>12.1} {:>12.1} {:>9.1}%", "swc", us(sg), us(sd), overhead(sg, sd));
    println!("{:<8} {:>12.1} {:>12.1} {:>9.1}%", "oxc", us(og), us(od), overhead(og, od));
    println!("\noxc vs swc (generic): {:.2}x faster", us(sg) / us(og));
}
