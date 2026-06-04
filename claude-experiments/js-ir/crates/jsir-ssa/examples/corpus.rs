//! React-Compiler parity harness over the full fixture corpus.
//!
//! For every `.js` fixture in React's `__tests__/fixtures/compiler` we compile
//! with **our** pipeline (`jsir_ssa::codegen::compile`) and with the **real**
//! `react-compiler-e2e` CLI (the cloned Rust port), then bucket the pair by the
//! *memoization decision* each side made and, when both memoize, whether the
//! memo *structure* (`_c(N)` cache size + number of memo blocks) agrees.
//!
//! This is the trusted gate for parity work: the React side is ground truth
//! (we never edit it), and the only number that counts is **agreement on the
//! fixtures React memoizes**. Coverage (how many of those we even attempt) and
//! fidelity (how many we match) are both surfaced.
//!
//! Buckets (mirrors PARITY.md):
//!   comparable   both memoize           -> agree | mismatch
//!   react_only   React memoizes, we don't (bail / panic / pass-through)
//!   ours_only    we memoize, React doesn't (over-memoization)
//!   neither      neither memoizes (non-components, error fixtures, ...)
//!   panic        our compile panicked (a real lowering/codegen bug; overlay)
//!
//! Usage:
//!   cargo run --release -p jsir-ssa --example corpus                 # summary
//!   cargo run --release -p jsir-ssa --example corpus -- --list react_only
//!   cargo run --release -p jsir-ssa --example corpus -- --list mismatch
//!   cargo run --release -p jsir-ssa --example corpus -- --list panic
//!   cargo run --release -p jsir-ssa --example corpus -- --show <fixture-basename>
//!   cargo run --release -p jsir-ssa --example corpus -- --json       # machine summary
//!
//! Env:
//!   REACT_FIXTURES  fixtures dir (default: cloned react-rust path under /tmp)
//!   REACT_CC        react-compiler-e2e binary (default: cloned build under /tmp)
//!   REACT_CACHE_DIR optional dir for cached React oracle stdout/failures

use std::collections::BTreeMap;
use std::fs;
use std::io::Write;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

fn fixtures_dir() -> PathBuf {
    if let Ok(p) = std::env::var("REACT_FIXTURES") {
        return p.into();
    }
    "/tmp/react-rust/compiler/packages/babel-plugin-react-compiler/src/__tests__/fixtures/compiler"
        .into()
}

fn react_cli() -> PathBuf {
    if let Ok(p) = std::env::var("REACT_CC") {
        return p.into();
    }
    "/tmp/react-rust/compiler/target/release/react-compiler-e2e".into()
}

fn react_cache_dir() -> Option<PathBuf> {
    std::env::var("REACT_CACHE_DIR").ok().map(PathBuf::from)
}

/// `(cache_size, memo_block_count)` extracted from compiled output, or None if
/// the output is not memoized (no `_c(` cache declaration).
fn structure(code: &str) -> Option<(usize, usize)> {
    let n = code
        .split("_c(")
        .nth(1)?
        .split(')')
        .next()?
        .trim()
        .parse::<usize>()
        .ok()?;
    // A memo block is `if (` then any number of `(` then a cache check `$[`.
    let block_count = code
        .match_indices("if (")
        .filter(|(i, _)| {
            code[i + 4..]
                .trim_start_matches(['(', ' '])
                .starts_with("$[")
        })
        .count();
    Some((n, block_count))
}

/// What one side decided for a fixture.
#[derive(Clone)]
enum Side {
    /// Memoized: (cache_size, block_count).
    Memo(usize, usize),
    /// Compiled/ran but produced no memoization (pass-through / bail).
    NoMemo,
    /// Failed: error string (converter error, CLI nonzero exit, or panic).
    Fail(String),
}

fn run_react(cli: &Path, src: &str) -> Side {
    let mut child = match Command::new(cli)
        .args(["--frontend", "swc", "--filename", "t.jsx"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
    {
        Ok(c) => c,
        Err(e) => return Side::Fail(format!("spawn: {e}")),
    };
    if let Some(mut stdin) = child.stdin.take() {
        let _ = stdin.write_all(src.as_bytes());
    }
    let out = match child.wait_with_output() {
        Ok(o) => o,
        Err(e) => return Side::Fail(format!("wait: {e}")),
    };
    if !out.status.success() {
        return Side::Fail("react-cli-error".into());
    }
    let code = String::from_utf8_lossy(&out.stdout);
    match structure(&code) {
        Some((n, b)) => Side::Memo(n, b),
        None => Side::NoMemo,
    }
}

fn fixture_cache_stem(path: &Path) -> String {
    path.file_name()
        .unwrap()
        .to_string_lossy()
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-') {
                c
            } else {
                '_'
            }
        })
        .collect()
}

fn side_from_react_stdout(code: &str) -> Side {
    match structure(code) {
        Some((n, b)) => Side::Memo(n, b),
        None => Side::NoMemo,
    }
}

fn run_react_cached(cli: &Path, path: &Path, src: &str, cache_dir: Option<&Path>) -> Side {
    let Some(cache_dir) = cache_dir else {
        return run_react(cli, src);
    };
    let stem = fixture_cache_stem(path);
    let ok_path = cache_dir.join(format!("{stem}.stdout.js"));
    let fail_path = cache_dir.join(format!("{stem}.fail"));
    if let Ok(code) = fs::read_to_string(&ok_path) {
        return side_from_react_stdout(&code);
    }
    if fail_path.exists() {
        return Side::Fail("react-cli-error".into());
    }

    let _ = fs::create_dir_all(cache_dir);
    let mut child = match Command::new(cli)
        .args(["--frontend", "swc", "--filename", "t.jsx"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
    {
        Ok(c) => c,
        Err(e) => return Side::Fail(format!("spawn: {e}")),
    };
    if let Some(mut stdin) = child.stdin.take() {
        let _ = stdin.write_all(src.as_bytes());
    }
    let out = match child.wait_with_output() {
        Ok(o) => o,
        Err(e) => return Side::Fail(format!("wait: {e}")),
    };
    if !out.status.success() {
        let _ = fs::write(&fail_path, b"react-cli-error\n");
        return Side::Fail("react-cli-error".into());
    }
    let code = String::from_utf8_lossy(&out.stdout).to_string();
    let _ = fs::write(&ok_path, &code);
    side_from_react_stdout(&code)
}

fn react_output_cached(cli: &Path, path: &Path, src: &str, cache_dir: Option<&Path>) -> String {
    let Some(cache_dir) = cache_dir else {
        return react_output_uncached(cli, src);
    };
    let stem = fixture_cache_stem(path);
    let ok_path = cache_dir.join(format!("{stem}.stdout.js"));
    if let Ok(code) = fs::read_to_string(&ok_path) {
        return code;
    }
    let side = run_react_cached(cli, path, src, Some(cache_dir));
    match side {
        Side::Memo(_, _) | Side::NoMemo => fs::read_to_string(&ok_path).unwrap_or_default(),
        Side::Fail(e) => format!("<react failed: {e}>"),
    }
}

fn react_output_uncached(cli: &Path, src: &str) -> String {
    Command::new(cli)
        .args(["--frontend", "swc", "--filename", "t.jsx"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .and_then(|mut c| {
            c.stdin.take().unwrap().write_all(src.as_bytes())?;
            c.wait_with_output()
        })
        .map(|o| String::from_utf8_lossy(&o.stdout).to_string())
        .unwrap_or_else(|e| format!("<react failed: {e}>"))
}

fn run_ours(src: &str) -> Side {
    match catch_unwind(AssertUnwindSafe(|| jsir_ssa::codegen::compile(src))) {
        Ok(Ok(code)) => match structure(&code) {
            Some((n, b)) => Side::Memo(n, b),
            None => Side::NoMemo,
        },
        Ok(Err(e)) => Side::Fail(e),
        Err(panic) => {
            let msg = panic
                .downcast_ref::<&str>()
                .map(|s| s.to_string())
                .or_else(|| panic.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "<non-string panic>".into());
            Side::Fail(format!("PANIC: {msg}"))
        }
    }
}

/// Collapse an error/panic message into a coarse bucket key so similar failures
/// aggregate (drop locations, quoted literals, digits).
fn reason_key(e: &str) -> String {
    let mut k = e.to_string();
    if let Some(i) = k.find(" at ") {
        k.truncate(i);
    }
    // crude: drop digits and quoted strings
    let mut out = String::new();
    let mut in_q = false;
    let mut prev_digit = false;
    for c in k.chars() {
        match c {
            '"' | '\'' => {
                in_q = !in_q;
                if !in_q {
                    out.push_str("_");
                }
            }
            _ if in_q => {}
            c if c.is_ascii_digit() => {
                if !prev_digit {
                    out.push('N');
                }
                prev_digit = true;
            }
            c => {
                out.push(c);
                prev_digit = false;
            }
        }
    }
    if out.len() > 90 {
        out.truncate(90);
    }
    out.trim().to_string()
}

/// Did our failure come from the SWC parser (a gap the user authorized skipping)?
fn is_parser_skip(reason: &str) -> bool {
    let r = reason.to_ascii_lowercase();
    r.contains("expectedident")
        || r.contains("unexpected")
        || r.contains("expected ")
        || r.contains("parse")
        || r.contains("ts1")
        || r.contains("asyncconstructor")
}

struct Entry {
    name: String,
    react: Side,
    ours: Side,
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut limit: Option<usize> = None;
    let mut list: Option<String> = None;
    let mut show: Option<String> = None;
    let mut json = false;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--list" => {
                list = args.get(i + 1).cloned();
                i += 2;
            }
            "--show" => {
                show = args.get(i + 1).cloned();
                i += 2;
            }
            "--json" => {
                json = true;
                i += 1;
            }
            n => {
                if let Ok(v) = n.parse::<usize>() {
                    limit = Some(v);
                }
                i += 1;
            }
        }
    }

    let dir = fixtures_dir();
    let cli = react_cli();
    let cache_dir = react_cache_dir();
    if !cli.exists() {
        eprintln!(
            "react-compiler-e2e not found at {} (set REACT_CC)",
            cli.display()
        );
        std::process::exit(2);
    }
    let mut files: Vec<PathBuf> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("read fixtures {}: {e}", dir.display()))
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().map(|x| x == "js").unwrap_or(false))
        .collect();
    files.sort();

    // --show: diff one fixture (find by basename substring).
    if let Some(target) = &show {
        for p in &files {
            let name = p.file_name().unwrap().to_string_lossy().to_string();
            if name.contains(target.as_str()) {
                let src = std::fs::read_to_string(p).unwrap();
                let react_out = react_output_cached(&cli, p, &src, cache_dir.as_deref());
                let ours_out =
                    match catch_unwind(AssertUnwindSafe(|| jsir_ssa::codegen::compile(&src))) {
                        Ok(Ok(c)) => c,
                        Ok(Err(e)) => format!("<ours error: {e}>"),
                        Err(_) => "<ours PANIC>".into(),
                    };
                println!("== {name} ==\n--- source ---\n{src}\n--- react ---\n{react_out}\n--- ours ---\n{ours_out}");
                return;
            }
        }
        eprintln!("fixture matching '{target}' not found");
        return;
    }

    let mut entries: Vec<Entry> = Vec::new();
    for (idx, p) in files.iter().enumerate() {
        if let Some(l) = limit {
            if idx >= l {
                break;
            }
        }
        let src = match std::fs::read_to_string(p) {
            Ok(s) => s,
            Err(_) => continue,
        };
        let react = run_react_cached(&cli, p, &src, cache_dir.as_deref());
        let ours = run_ours(&src);
        entries.push(Entry {
            name: p.file_name().unwrap().to_string_lossy().to_string(),
            react,
            ours,
        });
    }

    // Bucketing.
    let mut agree: Vec<String> = Vec::new();
    let mut mismatch: Vec<(String, (usize, usize), (usize, usize))> = Vec::new();
    let mut react_only: Vec<String> = Vec::new();
    let mut react_only_parser_skip: Vec<String> = Vec::new();
    let mut ours_only: Vec<String> = Vec::new();
    let mut neither = 0u32;
    let mut panic: Vec<String> = Vec::new();
    // why we miss the react_only fixtures (ours side reason)
    let mut react_only_reasons: BTreeMap<String, (u32, String)> = BTreeMap::new();

    for e in &entries {
        if let Side::Fail(m) = &e.ours {
            if m.starts_with("PANIC") {
                panic.push(e.name.clone());
            }
        }
        match (&e.react, &e.ours) {
            (Side::Memo(rn, rb), Side::Memo(on, ob)) => {
                if (rn, rb) == (on, ob) {
                    agree.push(e.name.clone());
                } else {
                    mismatch.push((e.name.clone(), (*rn, *rb), (*on, *ob)));
                }
            }
            (Side::Memo(_, _), ours) => {
                let reason = match ours {
                    Side::NoMemo => "no-memo (we pass through / bail)".to_string(),
                    Side::Fail(m) => m.clone(),
                    Side::Memo(_, _) => unreachable!(),
                };
                if matches!(ours, Side::Fail(m) if is_parser_skip(m)) {
                    react_only_parser_skip.push(e.name.clone());
                } else {
                    react_only.push(e.name.clone());
                    let key = reason_key(&reason);
                    let ent = react_only_reasons.entry(key).or_insert((0, e.name.clone()));
                    ent.0 += 1;
                }
            }
            (_, Side::Memo(_, _)) => ours_only.push(e.name.clone()),
            _ => neither += 1,
        }
    }

    // The parity universe = fixtures React memoizes that aren't parser-skipped.
    let universe = agree.len() + mismatch.len() + react_only.len();
    let agreement_pct = if universe > 0 {
        100.0 * agree.len() as f64 / universe as f64
    } else {
        0.0
    };

    if let Some(kind) = &list {
        let v: Vec<String> = match kind.as_str() {
            "agree" => agree.clone(),
            "react_only" => react_only.clone(),
            "parser_skip" => react_only_parser_skip.clone(),
            "ours_only" => ours_only.clone(),
            "panic" => panic.clone(),
            "mismatch" => mismatch
                .iter()
                .map(|(n, r, o)| format!("{n}  react={r:?} ours={o:?}"))
                .collect(),
            _ => {
                eprintln!("--list expects agree|react_only|parser_skip|ours_only|panic|mismatch");
                return;
            }
        };
        for s in v {
            println!("{s}");
        }
        return;
    }

    if json {
        println!(
            "{{\"total\":{},\"react_memoized_universe\":{},\"agree\":{},\"mismatch\":{},\"react_only\":{},\"parser_skip\":{},\"ours_only\":{},\"neither\":{},\"panic\":{},\"agreement_pct\":{:.2}}}",
            entries.len(),
            universe,
            agree.len(),
            mismatch.len(),
            react_only.len(),
            react_only_parser_skip.len(),
            ours_only.len(),
            neither,
            panic.len(),
            agreement_pct,
        );
        return;
    }

    println!(
        "== React-Compiler corpus parity ({} .js fixtures) ==",
        entries.len()
    );
    println!("  React-memoized universe (excl. parser-skips): {universe}");
    println!(
        "    agree          {:5}  ({:.1}% of universe)",
        agree.len(),
        agreement_pct
    );
    println!(
        "    mismatch       {:5}  (both memoize, structure differs)",
        mismatch.len()
    );
    println!(
        "    react_only     {:5}  (React memoizes, we don't — coverage gap)",
        react_only.len()
    );
    println!(
        "  parser_skip      {:5}  (React memoizes, SWC can't parse — authorized skip)",
        react_only_parser_skip.len()
    );
    println!(
        "  ours_only        {:5}  (we memoize, React doesn't — over-memoization)",
        ours_only.len()
    );
    println!("  neither          {:5}", neither);
    println!(
        "  panic (overlay)  {:5}  (our compile panicked — real bug)",
        panic.len()
    );

    if !mismatch.is_empty() {
        println!(
            "\n-- structure mismatches (first 25 of {}) --",
            mismatch.len()
        );
        for (n, r, o) in mismatch.iter().take(25) {
            println!("  react(cache,blocks)={r:?}  ours={o:?}   {n}");
        }
    }

    if !react_only_reasons.is_empty() {
        let mut v: Vec<(&String, &(u32, String))> = react_only_reasons.iter().collect();
        v.sort_by(|a, b| b.1 .0.cmp(&a.1 .0));
        println!("\n-- react_only: why we miss (top 25 of {}) --", v.len());
        for (key, (n, ex)) in v.into_iter().take(25) {
            println!("  {n:4}  {key}   e.g. {ex}");
        }
    }
}
