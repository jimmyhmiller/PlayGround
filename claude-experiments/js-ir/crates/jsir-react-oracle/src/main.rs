//! The oracle gate.
//!
//! For every fixture: run our current pipeline → normalize (tools/normalize-batch.js,
//! the exact snap formatter) → diff against the committed `cache/` extracted from the
//! pinned upstream (UPSTREAM.lock). Reports buckets + a `frontier` of remaining work.
//!
//! The cache is the in-tree TS compiler's `## Code` output, already snap-normalized, so
//! the oracle side needs NO normalization at gate time — only OUR output does.
//!
//! Today the "pipeline" is an identity round-trip (parse → JSHIR IR → print). That is a
//! placeholder seam: as the JSLIR port lands (BuildJSLIR → passes → LiftJSLIR), replace
//! `compile_fixture` and the frontier shrinks. Even the identity baseline exercises the
//! full harness (parse, emit, normalize, diff) against all 1725 real fixtures.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::io::Write;

use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
struct ManifestEntry {
    name: String,
    kind: String,    // "code" | "error"
    dialect: String, // "typescript" | "flow"
}

#[derive(Serialize)]
struct NormIn<'a> {
    name: &'a str,
    code: String,
    dialect: &'a str,
}

#[derive(Deserialize)]
struct NormOut {
    name: String,
    ok: bool,
    #[serde(default)]
    code: String,
    #[serde(default)]
    #[allow(dead_code)] // surfaced via --list normalize_error later; kept for parity with JS
    error: String,
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Bucket {
    Match,          // normalize(ours) == cached code  ✅
    Mismatch,       // code fixture, output differs (the frontier)
    OurError,       // IN SCOPE: parsed as ES+JSX but lowering/emit failed (a real gap)
    NormalizeError, // IN SCOPE: our emitted JS didn't parse
    ErrorFixture,   // oracle expects a compiler bail (## Error); not modeled yet
    SkipFlow,       // OUT OF SCOPE: Flow fixture (@flow / .flow) — explicitly skipped
    SkipUnparsed,   // OUT OF SCOPE: not ES+JSX-parseable (TypeScript type syntax, etc.)
}

impl Bucket {
    fn label(self) -> &'static str {
        match self {
            Bucket::Match => "match",
            Bucket::Mismatch => "mismatch",
            Bucket::OurError => "our_error",
            Bucket::NormalizeError => "normalize_error",
            Bucket::ErrorFixture => "error_fixture",
            Bucket::SkipFlow => "skip_flow",
            Bucket::SkipUnparsed => "skip_unparsed",
        }
    }
}

/// THE PIPELINE SEAM. Currently: parse → JSHIR → build_jslir (CFG) → lift_jslir →
/// JS. No passes yet, so the round-trip is faithful; as passes land they slot in
/// between build and lift. Also returns JSLIR coverage stats for this fixture.
fn compile_fixture(input: &str) -> Result<(String, jsir_jslir::Stats), String> {
    let op = jsir_swc::source_to_ir(input)?;
    let (lifted, stats) = jsir_jslir::compile(&op);
    Ok((jsir_swc::ir_to_source(&lifted)?, stats))
}

fn crate_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn batch_normalize(dir: &Path, items: &[NormIn]) -> Result<Vec<NormOut>, String> {
    let script = dir.join("tools/normalize-batch.js");
    let payload = serde_json::to_string(items).map_err(|e| e.to_string())?;
    let mut child = Command::new("node")
        .arg(&script)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .map_err(|e| format!("spawn node: {e} (is node on PATH?)"))?;
    child
        .stdin
        .take()
        .unwrap()
        .write_all(payload.as_bytes())
        .map_err(|e| e.to_string())?;
    let out = child.wait_with_output().map_err(|e| e.to_string())?;
    if !out.status.success() {
        return Err(format!("normalize-batch exited {}", out.status));
    }
    serde_json::from_slice(&out.stdout).map_err(|e| format!("normalize-batch bad json: {e}"))
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let json_out = args.iter().any(|a| a == "--json");
    let show_errors = args.iter().any(|a| a == "--errors");
    let list_bucket = arg_value(&args, "--list");
    let filter = arg_value(&args, "--filter");
    let limit: Option<usize> = arg_value(&args, "--limit").and_then(|s| s.parse().ok());

    let dir = crate_dir();

    // --ir <path>: dump the JSHIR (source_to_ir) textual IR for a source file.
    // A scratch window for designing the JSLIR conversion.
    if let Some(path) = arg_value(&args, "--ir") {
        let src = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {path}: {e}"));
        match jsir_swc::source_to_ir(&src) {
            Ok(op) => println!("{}", op.print()),
            Err(e) => println!("source_to_ir ERROR: {e}"),
        }
        return;
    }

    // --emit <name>: dump our raw ir_to_source output (pre-normalize) for one fixture,
    // or the source_to_ir error. A debugging window into the pipeline seam.
    if let Some(name) = arg_value(&args, "--emit") {
        let input = std::fs::read_to_string(dir.join("fixtures").join(format!("{name}.js")))
            .unwrap_or_else(|e| panic!("read fixture {name}: {e}"));
        println!("===== INPUT =====\n{input}");
        match jsir_swc::source_to_ir(&input) {
            Err(e) => {
                println!("===== source_to_ir ERROR =====\n{e}");
                if let Ok(p) = jsir_swc::parse(&input) {
                    println!("===== swc parse tree (lowering failed on this) =====\n{p:#?}");
                }
            }
            Ok(op) => match jsir_swc::ir_to_source(&op) {
                Err(e) => println!("===== ir_to_source ERROR =====\n{e}"),
                Ok(js) => println!("===== ir_to_source (raw, pre-normalize) =====\n{js}"),
            },
        }
        return;
    }

    let manifest: Vec<ManifestEntry> = {
        let p = dir.join("fixtures.manifest.json");
        let txt = std::fs::read_to_string(&p)
            .unwrap_or_else(|e| panic!("read {}: {e}", p.display()));
        serde_json::from_str(&txt).expect("parse manifest")
    };

    // Stage 1: run our pipeline over every fixture; collect outputs to normalize.
    // TS/Flow are explicitly out of scope: Flow fixtures are skipped outright, and any
    // fixture that does not parse as ES+JSX (TypeScript type syntax) is skipped too.
    let mut our_errors: Vec<(String, String)> = Vec::new();
    let mut skip_flow: Vec<String> = Vec::new();
    let mut skip_unparsed: Vec<(String, String)> = Vec::new();
    let mut to_norm: Vec<NormIn> = Vec::new();
    // JSLIR coverage across the (in-scope) corpus.
    let mut jslir_functions = 0usize;
    let mut jslir_lowered = 0usize;
    let mut dialect_of: BTreeMap<String, String> = BTreeMap::new();
    let mut kind_of: BTreeMap<String, String> = BTreeMap::new();

    let mut considered = 0usize;
    for m in &manifest {
        if let Some(f) = &filter {
            if !m.name.contains(f) {
                continue;
            }
        }
        if let Some(l) = limit {
            if considered >= l {
                break;
            }
        }
        considered += 1;
        dialect_of.insert(m.name.clone(), m.dialect.clone());
        kind_of.insert(m.name.clone(), m.kind.clone());

        // Flow is out of scope — skip without compiling.
        if m.dialect == "flow" {
            skip_flow.push(m.name.clone());
            continue;
        }

        let input = std::fs::read_to_string(dir.join("fixtures").join(format!("{}.js", m.name)))
            .unwrap_or_default();
        match compile_fixture(&input) {
            Ok((code, stats)) => {
                jslir_functions += stats.functions;
                jslir_lowered += stats.lowered;
                to_norm.push(NormIn {
                    name: &m.name,
                    code,
                    dialect: "typescript", // prettier parser; babel-ts is a superset of JS
                })
            }
            // A parse failure means the input isn't ES+JSX (TypeScript type syntax) → skip.
            // A non-parse failure (lowering/emit) is a real in-scope gap → our_error.
            Err(e) if e.starts_with("swc parse error") => skip_unparsed.push((m.name.clone(), e)),
            Err(e) => our_errors.push((m.name.clone(), e)),
        }
    }
    // The NormIn borrows m.name; re-key normalized results by name below.
    let normed = if to_norm.is_empty() {
        Vec::new()
    } else {
        batch_normalize(&dir, &to_norm).unwrap_or_else(|e| {
            eprintln!("FATAL: {e}");
            std::process::exit(2);
        })
    };

    // Stage 2: bucket.
    let mut buckets: BTreeMap<&'static str, Vec<String>> = BTreeMap::new();
    let put = |b: Bucket, name: &str, m: &mut BTreeMap<&'static str, Vec<String>>| {
        m.entry(b.label()).or_default().push(name.to_string());
    };
    for (name, _e) in &our_errors {
        put(Bucket::OurError, name, &mut buckets);
    }
    for name in &skip_flow {
        put(Bucket::SkipFlow, name, &mut buckets);
    }
    for (name, _e) in &skip_unparsed {
        put(Bucket::SkipUnparsed, name, &mut buckets);
    }
    for n in &normed {
        let kind = kind_of.get(&n.name).map(String::as_str).unwrap_or("code");
        if !n.ok {
            put(Bucket::NormalizeError, &n.name, &mut buckets);
            continue;
        }
        if kind == "error" {
            // Oracle expects a compiler bail; identity pipeline can't model it yet.
            put(Bucket::ErrorFixture, &n.name, &mut buckets);
            continue;
        }
        let cache = dir.join("cache").join(format!("{}.code.js", n.name));
        let expected = std::fs::read_to_string(&cache).unwrap_or_default();
        if n.code == expected {
            put(Bucket::Match, &n.name, &mut buckets);
        } else {
            put(Bucket::Mismatch, &n.name, &mut buckets);
        }
    }

    if show_errors {
        // our_error: pipeline (source_to_ir / ir_to_source) failed.
        for (name, e) in &our_errors {
            println!("our_error\t{name}\t{}", e.replace('\n', " "));
        }
        // normalize_error: our emitted JS didn't parse under prettier.
        for n in &normed {
            if !n.ok {
                println!("normalize_error\t{}\t{}", n.name, n.error.replace('\n', " "));
            }
        }
        // skip_unparsed: not ES+JSX (TypeScript) — shown for auditability.
        for (name, e) in &skip_unparsed {
            println!("skip_unparsed\t{name}\t{}", e.replace('\n', " "));
        }
        return;
    }

    if let Some(b) = &list_bucket {
        if let Some(names) = buckets.get(b.as_str()) {
            for n in names {
                println!("{n}");
            }
        }
        return;
    }

    let count = |b: Bucket| buckets.get(b.label()).map(Vec::len).unwrap_or(0);
    let total = considered;
    let skipped = count(Bucket::SkipFlow) + count(Bucket::SkipUnparsed);
    let matched = count(Bucket::Match);
    let frontier = count(Bucket::Mismatch) + count(Bucket::OurError) + count(Bucket::NormalizeError);
    // In-scope = ES+JSX, non-Flow code fixtures that reached the diff stage.
    let in_scope_code = matched + frontier;

    if json_out {
        let obj = serde_json::json!({
            "total": total,
            "in_scope_code": in_scope_code,
            "matched": matched,
            "frontier": frontier,
            "skipped": skipped,
            "error_fixtures": count(Bucket::ErrorFixture),
            "jslir_functions": jslir_functions,
            "jslir_lowered": jslir_lowered,
            "buckets": {
                "match": count(Bucket::Match),
                "mismatch": count(Bucket::Mismatch),
                "our_error": count(Bucket::OurError),
                "normalize_error": count(Bucket::NormalizeError),
                "error_fixture": count(Bucket::ErrorFixture),
                "skip_flow": count(Bucket::SkipFlow),
                "skip_unparsed": count(Bucket::SkipUnparsed),
            },
        });
        println!("{}", serde_json::to_string_pretty(&obj).unwrap());
    } else {
        println!(
            "React-on-JSLIR oracle  (pipeline: JSHIR -> build_jslir -> {} -> lift_jslir -> JS)",
            jsir_jslir::pipeline::coverage_summary()
        );
        println!("  fixtures considered : {total}   (TS/Flow out of scope)");
        println!();
        println!("  IN SCOPE (ES+JSX):");
        println!("    match           : {:>5}   (normalize(ours) == upstream `## Code`)", count(Bucket::Match));
        println!("    mismatch        : {:>5}   (frontier: needs JSLIR compilation)", count(Bucket::Mismatch));
        println!("    our_error       : {:>5}   (parsed, but lowering/emit gap)", count(Bucket::OurError));
        println!("    normalize_error : {:>5}   (our emitted JS didn't parse)", count(Bucket::NormalizeError));
        println!("    error_fixture   : {:>5}   (oracle expects a bail; not modeled yet)", count(Bucket::ErrorFixture));
        println!();
        println!("  OUT OF SCOPE (skipped):");
        println!("    skip_flow       : {:>5}   (@flow / .flow)", count(Bucket::SkipFlow));
        println!("    skip_unparsed   : {:>5}   (TypeScript type syntax — not ES+JSX)", count(Bucket::SkipUnparsed));
        println!();
        println!("  matched {matched}/{in_scope_code} in-scope code fixtures; frontier = {frontier}");
        let pct = if jslir_functions > 0 {
            100.0 * jslir_lowered as f64 / jslir_functions as f64
        } else {
            0.0
        };
        println!("  JSLIR coverage  : {jslir_lowered}/{jslir_functions} functions lowered to a CFG ({pct:.1}%)");
        println!("  (use --list <bucket> to inspect, --errors, --filter <substr>, --json)");
    }
}

fn arg_value(args: &[String], flag: &str) -> Option<String> {
    args.iter().position(|a| a == flag).and_then(|i| args.get(i + 1)).cloned()
}
