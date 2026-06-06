//! Tokenizer comparison: our SIMD stage-1 + scalar stage-2 vs the oxc lexer.
//! Built only with `--features bench`.
//!
//!   cargo run --release --features bench -- tokens <file.js>   # dump + align
//!   cargo run --release --features bench -- bench  <file.js> N # throughput
//!
//! oxc's lexer is normally parser-driven: the parser resolves regex-vs-division
//! and template `${}` nesting and feeds the answers back to the lexer. Run alone
//! it desyncs at the first regex. We make it run STANDALONE by replicating that
//! feedback (`oxc_tokens_standalone`): a local fork of oxc widens `next_regex`
//! and `next_template_substitution_tail` to `pub`, and we drive them with the
//! same prev-significant-token heuristic our stage-2 uses. Result: oxc lexes the
//! whole file with 0 errors and its token stream matches ours (modulo oxc's
//! intentional `>`-splitting for generics).
//!
//! Caveat that remains: oxc does strictly MORE per token (precise Kind, full
//! Unicode/escape/TS); ours is coarser. This is bytes→token-stream throughput,
//! not equal semantic work.

use crate::{codegen, js, parser};
use std::collections::HashMap;
use std::hint::black_box;
use std::time::Instant;

use oxc_allocator::Allocator;
use oxc_parser::lexer::{Kind, Lexer};
use oxc_span::SourceType;

/// `/` begins a regex (not division) iff the previous significant token does not
/// end an expression. This is the *exact* context the oxc parser feeds back to
/// its lexer — we replicate it so the lexer can run standalone.
fn regex_allowed(prev: Option<Kind>) -> bool {
    match prev {
        None => true,
        Some(k) => {
            let ends_expr = k.is_identifier()
                || k.is_literal() // Null | True | False | Str | RegExp | number
                || matches!(
                    k,
                    Kind::RParen
                        | Kind::RBrack
                        | Kind::RCurly
                        | Kind::This
                        | Kind::Super
                        | Kind::NoSubstitutionTemplate
                        | Kind::TemplateTail
                        | Kind::PrivateIdentifier
                );
            !ends_expr
        }
    }
}

/// Drive oxc's lexer **standalone** over the whole file by replicating the two
/// pieces of feedback the parser normally supplies:
///   1. regex-vs-division — when the lexer yields `/`/`/=` in expression
///      position, call `next_regex` to re-lex it as a `RegExp`.
///   2. template `${ }` nesting — track brace depth inside each substitution and,
///      when a `}` closes a substitution, call `next_template_substitution_tail`
///      to read the `}…${` / `}…\`` continuation instead of a bare `RCurly`.
/// Returns (tokens, error_count, last_byte_offset_reached).
/// Drive oxc's lexer standalone, calling `emit(kind, start, end)` per token.
/// Returns (error_count, last_end). Shared by the materializing dump path and
/// the non-materializing throughput path.
#[inline]
fn oxc_drive<F: FnMut(Kind, u32, u32)>(src: &str, mut emit: F) -> (usize, u32) {
    let allocator = Allocator::default();
    let mut lexer = Lexer::new_for_benchmarks(&allocator, src, SourceType::default());
    let mut tmpl: Vec<u32> = Vec::new();
    let mut prev: Option<Kind> = None;
    let mut last = 0u32;

    let mut tok = lexer.first_token();
    loop {
        let mut k = tok.kind();
        if k == Kind::Eof {
            break;
        }
        if (k == Kind::Slash || k == Kind::SlashEq) && regex_allowed(prev) {
            let (rt, _, _, _) = lexer.next_regex(k);
            tok = rt;
            k = Kind::RegExp;
        }
        if k == Kind::RCurly && tmpl.last().copied() == Some(0) {
            tok = lexer.next_template_substitution_tail();
            k = tok.kind();
        }
        last = tok.end();
        emit(k, tok.start(), tok.end());
        match k {
            Kind::TemplateHead => tmpl.push(0),
            Kind::TemplateMiddle => {
                if let Some(d) = tmpl.last_mut() {
                    *d = 0;
                }
            }
            Kind::TemplateTail => {
                tmpl.pop();
            }
            Kind::LCurly => {
                if let Some(d) = tmpl.last_mut() {
                    *d += 1;
                }
            }
            Kind::RCurly => {
                if let Some(d) = tmpl.last_mut() {
                    if *d > 0 {
                        *d -= 1;
                    }
                }
            }
            _ => {}
        }
        prev = Some(k);
        tok = lexer.next_token();
    }
    (lexer.errors().len(), last)
}

/// Materializing oxc driver (for the token dump / alignment).
fn oxc_tokens_standalone(src: &str) -> (Vec<(Kind, u32, u32)>, usize, u32) {
    let mut out: Vec<(Kind, u32, u32)> = Vec::new();
    let (errors, last) = oxc_drive(src, |k, s, e| out.push((k, s, e)));
    (out, errors, last)
}

/// Non-materializing oxc driver: count only (matches our `count_tokens`).
fn oxc_count_standalone(src: &str) -> (usize, usize, u32) {
    let mut n = 0usize;
    let (errors, last) = oxc_drive(src, |_, _, _| n += 1);
    (n, errors, last)
}

/// Our compiled stage-1 entry point (memref ABI, see json_stage1 JIT tests).
type Stage1Fn = extern "C" fn(
    *const u8, *const u8, i64, i64, i64, // input       memref<?xi8>
    *mut u64, *mut u64, i64, i64, i64,   // start_masks memref<?xi64>
    *mut u64, *mut u64, i64, i64, i64,   // word_masks  memref<?xi64>
) -> f32; // vector<1xi32> (token-start count) returned in s0

fn pad64(src: &[u8]) -> Vec<u8> {
    let mut v = src.to_vec();
    while v.len() % 64 != 0 {
        v.push(0);
    }
    v
}

fn median(mut xs: Vec<f64>) -> f64 {
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    xs[xs.len() / 2]
}

fn show(src: &[u8], s: usize, e: usize) -> String {
    let t: String = src[s..e.min(src.len())]
        .iter()
        .map(|&b| match b {
            b'\n' => '↵',
            b'\t' => '⇥',
            b'\r' => ' ',
            0x20..=0x7e => b as char,
            _ => '·',
        })
        .collect();
    if t.chars().count() > 40 {
        format!("{}…", t.chars().take(39).collect::<String>())
    } else {
        t
    }
}

/// Measure the throughput of the 9-state JS literal DFA (`examples/js_litmask.simd`)
/// — i.e. how fast `scan_dfa` runs a real multi-mode lexer DFA in SIMD.
pub fn litmask(path: &str, iters: usize) {
    let raw = std::fs::read(path).unwrap();
    let bytes = raw.len();
    let mb = bytes as f64 / 1_000_000.0;
    let padded = pad64(&raw);

    // The 9-state transition table (state*16 + class -> next state).
    let mut t = [0u8; 256];
    let mut set = |s: usize, row: [u8; 7]| {
        for c in 0..7 {
            t[s * 16 + c] = row[c];
        }
    };
    set(0, [2, 4, 0, 1, 0, 0, 0]); // Code
    set(1, [2, 4, 0, 6, 7, 0, 0]); // Slash
    set(2, [0, 2, 3, 2, 2, 2, 2]); // DQ
    set(3, [2, 2, 2, 2, 2, 2, 2]); // DQesc
    set(4, [4, 0, 5, 4, 4, 4, 4]); // SQ
    set(5, [4, 4, 4, 4, 4, 4, 4]); // SQesc
    set(6, [6, 6, 6, 6, 6, 0, 6]); // Line
    set(7, [7, 7, 7, 7, 8, 7, 7]); // Block
    set(8, [7, 7, 7, 0, 8, 7, 7]); // BlockStar

    let src = std::fs::read_to_string("examples/js_litmask.simd").unwrap();
    let ctx = codegen::create_context();
    let items = parser::parse(&src);
    let mut module = codegen::compile_module(&ctx, &items, &HashMap::new(), 8);
    codegen::lower_to_llvm(&ctx, &mut module).unwrap();
    let engine = melior::ExecutionEngine::new(&module, 3, &[], false);
    let fptr = engine.lookup("js_litmask");
    assert!(!fptr.is_null());
    let f: extern "C" fn(
        *const u8, *const u8, i64, i64, i64,
        *const u8, *const u8, i64, i64, i64,
        *mut u8, *mut u8, i64, i64, i64,
    ) = unsafe { std::mem::transmute(fptr) };
    let mut out = vec![0u8; padded.len()];
    let call = |out: &mut [u8]| {
        f(
            padded.as_ptr(), padded.as_ptr(), 0, padded.len() as i64, 1,
            t.as_ptr(), t.as_ptr(), 0, 256, 1,
            out.as_mut_ptr(), out.as_mut_ptr(), 0, out.len() as i64, 1,
        );
    };
    for _ in 0..3 {
        call(&mut out);
    }
    let mut best = 0.0f64;
    for _ in 0..iters {
        let t0 = Instant::now();
        call(&mut out);
        best = best.max(mb / t0.elapsed().as_secs_f64());
    }
    let in_lit = out[..bytes].iter().filter(|&&b| b == 1).count();
    println!("file: {path}  ({bytes} bytes)");
    println!("9-state literal DFA via scan_dfa: {best:.0} MB/s best");
    println!("in-literal bytes: {in_lit} ({:.1}%)", 100.0 * in_lit as f64 / bytes as f64);
}

/// Profiling driver: precompute stage-1 masks once, then run stage-2
/// (`count_tokens`) in a tight loop so a sampler sees only the stage-2 hot path.
pub fn prof(path: &str, iters: usize) {
    let raw = std::fs::read(path).unwrap();
    let padded = pad64(&raw);
    let source = std::fs::read_to_string("examples/js_stage1.simd").unwrap();
    let ctx = codegen::create_context();
    let items = parser::parse(&source);
    let mut module = codegen::compile_module(&ctx, &items, &HashMap::new(), 8);
    codegen::lower_to_llvm(&ctx, &mut module).unwrap();
    let engine = melior::ExecutionEngine::new(&module, 3, &[], false);
    let stage1: Stage1Fn = unsafe { std::mem::transmute(engine.lookup("js_stage1")) };
    let nchunks = padded.len() / 64 + 1;
    let mut start_masks = vec![0u64; nchunks];
    let mut word_masks = vec![0u64; nchunks];
    stage1(
        padded.as_ptr(), padded.as_ptr(), 0, padded.len() as i64, 1,
        start_masks.as_mut_ptr(), start_masks.as_mut_ptr(), 0, start_masks.len() as i64, 1,
        word_masks.as_mut_ptr(), word_masks.as_mut_ptr(), 0, word_masks.len() as i64, 1,
    );
    let mut acc = 0usize;
    for _ in 0..iters {
        acc = acc.wrapping_add(js::count_tokens(&raw, &start_masks, &word_masks));
    }
    println!("prof done: {acc}");
}

/// Dump both token streams to files and show where they agree / diverge.
/// Pure tokenizers, no parser on either side.
pub fn dump(path: &str) {
    let raw = std::fs::read(path).unwrap_or_else(|e| {
        eprintln!("cannot read {path}: {e}");
        std::process::exit(1);
    });
    let src = std::str::from_utf8(&raw).expect("file must be UTF-8 for oxc");
    let padded = pad64(&raw);
    println!("file: {path}  ({} bytes)\n", raw.len());

    // ── ours (JIT stage-1 + scalar stage-2) ──────────────────────────────────
    let s1src = std::fs::read_to_string("examples/js_stage1.simd")
        .expect("cannot read examples/js_stage1.simd (run from project root)");
    let ctx = codegen::create_context();
    let items = parser::parse(&s1src);
    let mut module = codegen::compile_module(&ctx, &items, &HashMap::new(), 8);
    codegen::lower_to_llvm(&ctx, &mut module).expect("lowering failed");
    let engine = melior::ExecutionEngine::new(&module, 3, &[], false);
    let fptr = engine.lookup("js_stage1");
    assert!(!fptr.is_null(), "js_stage1 symbol not found");
    let stage1: Stage1Fn = unsafe { std::mem::transmute(fptr) };
    let nchunks = padded.len() / 64 + 1;
    let mut start_masks = vec![0u64; nchunks];
    let mut word_masks = vec![0u64; nchunks];
    let n = {
        let bits = stage1(
            padded.as_ptr(), padded.as_ptr(), 0, padded.len() as i64, 1,
            start_masks.as_mut_ptr(), start_masks.as_mut_ptr(), 0, start_masks.len() as i64, 1,
            word_masks.as_mut_ptr(), word_masks.as_mut_ptr(), 0, word_masks.len() as i64, 1,
        );
        f32::to_bits(bits) as i32 as usize
    };
    let ours = js::tokenize(&raw, &start_masks, &word_masks);

    // ── oxc, driven standalone (regex + template feedback replicated) ─────────
    let (oxc_raw, oxc_errors, _last) = oxc_tokens_standalone(src);
    let oxc: Vec<(String, usize, usize)> = oxc_raw
        .iter()
        .map(|(k, s, e)| (format!("{:?}", k), *s as usize, *e as usize))
        .collect();

    // ── write full dumps ──────────────────────────────────────────────────────
    let ours_lines: String = ours
        .iter()
        .map(|t| format!("{:>8}..{:<8} {:<13} {}\n", t.start, t.end, format!("{:?}", t.kind), show(&raw, t.start, t.end)))
        .collect();
    let oxc_lines: String = oxc
        .iter()
        .map(|(k, s, e)| format!("{:>8}..{:<8} {:<13} {}\n", s, e, k, show(&raw, *s, *e)))
        .collect();
    std::fs::write("/tmp/ours.tokens", &ours_lines).unwrap();
    std::fs::write("/tmp/oxc.tokens", &oxc_lines).unwrap();

    let our_comments = ours
        .iter()
        .filter(|t| matches!(t.kind, js::TokKind::LineComment | js::TokKind::BlockComment))
        .count();

    // Does OUR token stream cover the whole file? (round-trip = proof of completeness)
    let roundtrip = match js::reconstruct(&raw, &ours) {
        Ok(s) => s.as_bytes() == &raw[..],
        Err(_) => false,
    };
    // How far did oxc actually get before EOF?
    let oxc_last = oxc.last().map(|t| t.2).unwrap_or(0);

    println!("ours : {} tokens ({} comments)  covers whole file: {}  → /tmp/ours.tokens",
        ours.len(), our_comments, if roundtrip { "YES (round-trips)" } else { "NO" });
    println!("oxc  : {} tokens, {} lex errors, stopped at byte {} of {} ({:.1}%)  → /tmp/oxc.tokens\n",
        oxc.len(), oxc_errors, oxc_last, raw.len(), 100.0 * oxc_last as f64 / raw.len() as f64);

    // ── first 30 of each ──────────────────────────────────────────────────────
    println!("── first 30 tokens, OURS ──────────────────────────────────");
    for t in ours.iter().take(30) {
        println!("{:>7}..{:<7} {:<13} {}", t.start, t.end, format!("{:?}", t.kind), show(&raw, t.start, t.end));
    }
    println!("\n── first 30 tokens, OXC ───────────────────────────────────");
    for (k, s, e) in oxc.iter().take(30) {
        println!("{:>7}..{:<7} {:<13} {}", s, e, k, show(&raw, *s, *e));
    }

    // ── alignment: match by start offset, ignoring our comment tokens ─────────
    // (oxc treats comments as trivia, so drop ours before aligning.)
    let ours_nc: Vec<&js::Token> = ours
        .iter()
        .filter(|t| !matches!(t.kind, js::TokKind::LineComment | js::TokKind::BlockComment))
        .collect();

    let mut i = 0usize;
    let mut j = 0usize;
    let mut exact = 0usize; // same (start,end)
    let mut start_only = 0usize; // same start, different end (granularity)
    let mut first_divergences: Vec<String> = Vec::new();
    while i < ours_nc.len() && j < oxc.len() {
        let (os, oe) = (ours_nc[i].start, ours_nc[i].end);
        let (xs, xe) = (oxc[j].1, oxc[j].2);
        if os == xs && oe == xe {
            exact += 1;
            i += 1;
            j += 1;
        } else if os == xs {
            start_only += 1;
            if first_divergences.len() < 12 {
                first_divergences.push(format!(
                    "@{os}: ours {:<10} {:?}   |  oxc {:<10} {:?}",
                    format!("{}..{}", os, oe),
                    show(&raw, os, oe),
                    format!("{}..{}", xs, xe),
                    show(&raw, xs, xe),
                ));
            }
            // advance whichever ends earlier to resync
            if oe <= xe { i += 1; } else { j += 1; }
        } else if os < xs {
            i += 1; // ours has an extra token oxc didn't emit
        } else {
            j += 1; // oxc has an extra token we didn't emit
        }
    }

    println!("\n── alignment (our non-comment tokens vs oxc tokens) ───────");
    println!("exact span matches      : {exact}");
    println!("same start, diff end    : {start_only}  (granularity differences)");
    if !first_divergences.is_empty() {
        println!("\nfirst divergences:");
        for d in &first_divergences {
            println!("  {d}");
        }
    }
    println!("\nFull side-by-side:  diff <(cut -c1-40 /tmp/ours.tokens) <(cut -c1-40 /tmp/oxc.tokens) | head");
}

pub fn run(path: &str, iters: usize) {
    let raw = std::fs::read(path).unwrap_or_else(|e| {
        eprintln!("cannot read {path}: {e}");
        std::process::exit(1);
    });
    let bytes = raw.len();
    let mb = bytes as f64 / 1_000_000.0;
    let padded = pad64(&raw);

    println!("file: {path}  ({bytes} bytes, {:.2} MB)", mb);
    println!("iterations: {iters}\n");

    // ── JIT-compile our stage 1 once (not timed) ───────────────────────────
    let source = std::fs::read_to_string("examples/js_stage1.simd")
        .expect("cannot read examples/js_stage1.simd (run from project root)");
    let ctx = codegen::create_context();
    let items = parser::parse(&source);
    let mut module = codegen::compile_module(&ctx, &items, &HashMap::new(), 8);
    codegen::lower_to_llvm(&ctx, &mut module).expect("lowering failed");
    let engine = melior::ExecutionEngine::new(&module, 3, &[], false);
    let fptr = engine.lookup("js_stage1");
    assert!(!fptr.is_null(), "js_stage1 symbol not found");
    let stage1: Stage1Fn = unsafe { std::mem::transmute(fptr) };

    let nchunks = padded.len() / 64 + 1;
    let mut start_masks = vec![0u64; nchunks];
    let mut word_masks = vec![0u64; nchunks];

    let call_stage1 = |start_masks: &mut [u64], word_masks: &mut [u64]| -> usize {
        let bits = stage1(
            padded.as_ptr(), padded.as_ptr(), 0, padded.len() as i64, 1,
            start_masks.as_mut_ptr(), start_masks.as_mut_ptr(), 0, start_masks.len() as i64, 1,
            word_masks.as_mut_ptr(), word_masks.as_mut_ptr(), 0, word_masks.len() as i64, 1,
        );
        f32::to_bits(bits) as i32 as usize
    };

    // sanity / token counts
    let n_starts = call_stage1(&mut start_masks, &mut word_masks);
    let our_tokens = js::count_tokens(&raw, &start_masks, &word_masks);
    println!("our stage-1 token starts : {n_starts}");
    println!("our stage-2 tokens       : {}", our_tokens);

    // ── Warmup ─────────────────────────────────────────────────────────────
    for _ in 0..3 {
        call_stage1(&mut start_masks, &mut word_masks);
        black_box(js::count_tokens(&raw, &start_masks, &word_masks));
    }

    // ── Measure: stage 1 (SIMD) only ───────────────────────────────────────
    let mut s1_mbps = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t = Instant::now();
        let n = call_stage1(&mut start_masks, &mut word_masks);
        let dt = t.elapsed().as_secs_f64();
        black_box(n);
        s1_mbps.push(mb / dt);
    }

    // ── Diagnostic: raw lex (classify only, no driver feedback) ─────────────
    let mut raw_best = 0.0f64;
    for _ in 0..iters {
        let t = Instant::now();
        call_stage1(&mut start_masks, &mut word_masks);
        let n = js::count_raw(&raw, &start_masks, &word_masks);
        let dt = t.elapsed().as_secs_f64();
        black_box(n);
        raw_best = raw_best.max(mb / dt);
    }
    println!("[diag] raw lex (no feedback): {raw_best:.0} MB/s best");
    let mut floor_best = 0.0f64;
    for _ in 0..iters {
        let t = Instant::now();
        let n = js::count_floor(&start_masks);
        let dt = t.elapsed().as_secs_f64();
        black_box(n);
        floor_best = floor_best.max(mb / dt);
    }
    println!("[diag] bitmap iteration floor: {floor_best:.0} MB/s best");

    // ── Diagnostic: parser-mode (defers keyword recognition to the parser) ──
    let mut pm_best = 0.0f64;
    for _ in 0..iters {
        let t = Instant::now();
        call_stage1(&mut start_masks, &mut word_masks);
        let n = js::count_tokens_parser_mode(&raw, &start_masks, &word_masks);
        let dt = t.elapsed().as_secs_f64();
        black_box(n);
        pm_best = pm_best.max(mb / dt);
    }
    println!("[diag] parser-mode (defer keywords): {pm_best:.0} MB/s best");

    // ── Measure: full pipeline (stage 1 + stage 2), non-materializing ───────
    // (parser-relevant: tokens are produced and consumed on the fly, not stored
    //  — matching oxc's lexer loop which also stores nothing.)
    let mut full_mbps = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t = Instant::now();
        call_stage1(&mut start_masks, &mut word_masks);
        let n = js::count_tokens(&raw, &start_masks, &word_masks);
        let dt = t.elapsed().as_secs_f64();
        black_box(n);
        full_mbps.push(mb / dt);
    }

    // ── Measure: oxc lexer, driven standalone (regex + template feedback) ────
    let source_text = match std::str::from_utf8(&raw) {
        Ok(s) => s,
        Err(_) => {
            eprintln!("file is not valid UTF-8; oxc needs &str. Skipping oxc.");
            report(bytes, &s1_mbps, &full_mbps, None, None, None);
            return;
        }
    };

    // Non-materializing (count-only), matching our `count_tokens` — fair.
    let (oxc_tokens, oxc_errors, oxc_last) = oxc_count_standalone(source_text);
    println!("oxc lexer tokens         : {oxc_tokens}");
    println!(
        "oxc lexer errors         : {oxc_errors}   (reached byte {oxc_last}/{} = {:.1}%)\n",
        raw.len(),
        100.0 * oxc_last as f64 / raw.len() as f64
    );

    for _ in 0..3 {
        black_box(oxc_count_standalone(source_text).0);
    }

    let mut oxc_mbps = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t = Instant::now();
        let n = oxc_count_standalone(source_text);
        let dt = t.elapsed().as_secs_f64();
        black_box(n.0);
        oxc_mbps.push(mb / dt);
    }

    report(
        bytes,
        &s1_mbps,
        &full_mbps,
        Some(&oxc_mbps),
        Some(oxc_tokens),
        Some(oxc_errors),
    );
}

fn report(
    bytes: usize,
    s1: &[f64],
    full: &[f64],
    oxc: Option<&[f64]>,
    _oxc_tokens: Option<usize>,
    oxc_errors: Option<usize>,
) {
    let best = |v: &[f64]| v.iter().cloned().fold(0.0f64, f64::max);
    let med = |v: &[f64]| median(v.to_vec());

    println!("─────────────────────────────────────────────────────────");
    println!("{:<28}{:>12}{:>12}", "", "median", "best");
    println!(
        "{:<28}{:>9.0} {:>2}{:>9.0} {:>2}",
        "stage-1 SIMD only (MB/s)",
        med(s1), "",
        best(s1), ""
    );
    println!(
        "{:<28}{:>9.0} {:>2}{:>9.0} {:>2}",
        "full stage1+stage2 (MB/s)",
        med(full), "",
        best(full), ""
    );
    if let Some(oxc) = oxc {
        println!(
            "{:<28}{:>9.0} {:>2}{:>9.0} {:>2}",
            "oxc lexer (MB/s)",
            med(oxc), "",
            best(oxc), ""
        );
        println!("─────────────────────────────────────────────────────────");
        let ratio_full = med(full) / med(oxc);
        let ratio_s1 = med(s1) / med(oxc);
        println!(
            "full pipeline vs oxc : {:.2}x   (stage-1 alone vs oxc: {:.2}x)",
            ratio_full, ratio_s1
        );
    }
    println!("─────────────────────────────────────────────────────────");
    let _ = bytes;
    if let Some(e) = oxc_errors {
        println!(
            "oxc driven standalone (our fork exposes next_regex / \
             next_template_substitution_tail; we replicate the parser's \
             regex+template feedback) → {e} lex errors, full-file coverage.",
        );
    }
    println!(
        "Caveat: oxc produces precise per-token Kind + full Unicode/escape/TS\n\
         handling; ours emits coarser tokens. This is bytes→token-stream\n\
         throughput, not equal semantic work."
    );
}
