//! How much of a JS parse is *structural skeleton* (matched brackets + nesting
//! depth + statement boundaries) — the part SIMD can extract in parallel — vs
//! the leaf-expression work a sequential parser must still do?
//!
//! We lex real JS with the SIMD lexer (correct for strings/comments/regex/
//! templates), then do a single linear pass that recovers the skeleton with a
//! bracket stack. We measure how cheap that is relative to oxc's full parse, and
//! report what fraction of tokens are structural. This sizes the prize for a
//! simdjson-style structural-index front end.
//!
//! Usage:  structure <file.js> [iters]

use std::hint::black_box;
use std::time::Instant;

use oxc_allocator::Allocator;
use oxc_parser::Parser;
use oxc_span::SourceType;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = args.get(1).map(String::as_str).unwrap_or_else(|| {
        eprintln!("Usage: structure <file.js> [iters]");
        std::process::exit(1);
    });
    let iters: u32 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(200);
    let raw = std::fs::read(path).expect("read file");
    let src = String::from_utf8(raw).expect("utf8");
    let bytes = src.len();
    let mib = bytes as f64 / (1 << 20) as f64;

    // ── One structural pass: lex (SIMD) → bracket stack → skeleton stats ──────
    let (sm, wm) = simd_lang::stage1::lex(src.as_bytes());
    let mut tokens = 0usize;
    let mut structural = 0usize; // brackets + ; , . :
    let mut pairs = 0usize; // matched () [] {}
    let mut max_depth = 0i32;
    let mut depth = 0i32;
    let mut top_stmts = 0usize; // ; or } at depth 0
    let b = src.as_bytes();
    simd_lang::js::drive(b, &sm, &wm, |t| {
        tokens += 1;
        use simd_lang::js::TokKind::*;
        if matches!(t.kind, Punct) {
            match b[t.start] {
                b'(' | b'[' | b'{' => {
                    structural += 1;
                    depth += 1;
                    max_depth = max_depth.max(depth);
                }
                b')' | b']' | b'}' => {
                    structural += 1;
                    pairs += 1;
                    depth -= 1;
                    if depth == 0 {
                        top_stmts += 1;
                    }
                }
                b';' => {
                    structural += 1;
                    if depth == 0 {
                        top_stmts += 1;
                    }
                }
                b',' | b'.' | b':' => structural += 1,
                _ => {}
            }
        }
    });

    println!("file: {path}  ({bytes} bytes, {mib:.2} MiB)\n");
    println!("tokens            : {tokens}");
    println!(
        "structural tokens : {structural}  ({:.1}% of tokens — the SIMD-able skeleton)",
        100.0 * structural as f64 / tokens as f64
    );
    println!("matched bracket pairs : {pairs}");
    println!("max nesting depth     : {max_depth}");
    println!("top-level statements  : {top_stmts}\n");

    // ── Throughput: lex only ─────────────────────────────────────────────────
    let lex_only = bench(iters, mib, || {
        let (sm, wm) = simd_lang::stage1::lex(src.as_bytes());
        let mut n = 0usize;
        simd_lang::js::drive(src.as_bytes(), &sm, &wm, |_| n += 1);
        black_box(n);
    });

    // ── Throughput: lex + skeleton extraction (bracket stack) ────────────────
    let skel = bench(iters, mib, || {
        let (sm, wm) = simd_lang::stage1::lex(src.as_bytes());
        let mut depth = 0i32;
        let mut pairs = 0usize;
        let b = src.as_bytes();
        simd_lang::js::drive(b, &sm, &wm, |t| {
            if matches!(t.kind, simd_lang::js::TokKind::Punct) {
                match b[t.start] {
                    b'(' | b'[' | b'{' => depth += 1,
                    b')' | b']' | b'}' => {
                        depth -= 1;
                        pairs += 1;
                    }
                    _ => {}
                }
            }
        });
        black_box((depth, pairs));
    });

    // ── Throughput: oxc full parse → AST ─────────────────────────────────────
    let oxc = bench(iters, mib, || {
        let alloc = Allocator::default();
        let ret = Parser::new(&alloc, &src, SourceType::default()).parse();
        black_box(ret.program.body.len());
    });

    println!("─────────────────────────────────────────────");
    println!("{:<34}{:>10}", "", "MiB/s");
    println!("{:<34}{:>10.0}", "lex only (SIMD)", lex_only);
    println!("{:<34}{:>10.0}", "lex + skeleton (bracket stack)", skel);
    println!("{:<34}{:>10.0}", "oxc full parse → AST", oxc);
    println!("─────────────────────────────────────────────");
    println!(
        "skeleton extraction is {:.1}x oxc's full parse — that's the headroom\n\
         a structural-index front end has before it has to touch leaf precedence.",
        skel / oxc
    );
}

fn bench(iters: u32, mib: f64, mut f: impl FnMut()) -> f64 {
    for _ in 0..3 {
        f();
    }
    let mut best = 0.0f64;
    for _ in 0..iters {
        let t = Instant::now();
        f();
        best = best.max(mib / t.elapsed().as_secs_f64());
    }
    best
}
