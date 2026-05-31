//! Local test262 parity harness against a pre-serialized upstream oracle.
//!
//! The oracle is a gzipped JSONL where each line is
//! `{"p": relpath, "src": <js source>, "ir": <upstream ir text>|null}`
//! produced on the Linux box by running upstream `jsir_gen` over every
//! test262 file (see `gen_oracle.py`). Having it local means we compare
//! our `source_to_ir` against upstream byte-for-byte with no SSH round-trip.
//!
//! Usage:
//!   cargo run --release --example parity                  # full corpus summary
//!   cargo run --release --example parity -- 2000          # first 2000 entries
//!   cargo run --release --example parity -- --show <relpath>   # diff one file
//!   cargo run --release --example parity -- --list struct      # print all differ_struct paths
//!
//! Corpus path: env ORACLE, else <workspace>/corpus/test262_oracle.jsonl.gz

use std::io::BufRead;
use std::panic::{catch_unwind, AssertUnwindSafe};

use flate2::read::GzDecoder;
use regex::Regex;

fn norm(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Structural normalization: drop source locations and integer literals so
/// only the op structure remains (matches parity2.py `deep`).
fn deep(s: &str, loc: &Regex, num: &Regex) -> String {
    let s = loc.replace_all(s, "<L>");
    let s = num.replace_all(&s, "N");
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Run our lowering, catching panics so one bad file does not abort the run.
/// Ok(ir) on success; Err(reason) on a converter error or a panic.
fn ours(src: &str) -> Result<String, String> {
    match catch_unwind(AssertUnwindSafe(|| jsir_swc::source_to_ir(src))) {
        Ok(Ok(op)) => Ok(op.print()),
        Ok(Err(e)) => Err(e),
        Err(panic) => {
            let msg = panic
                .downcast_ref::<&str>()
                .map(|s| s.to_string())
                .or_else(|| panic.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "<non-string panic>".to_string());
            Err(format!("PANIC: {msg}"))
        }
    }
}

/// Signature of a structural diff: the first differing token between the two
/// deep-normalized forms, with a small window of context from each side. This
/// buckets "same kind of divergence" together regardless of which file.
fn struct_sig(up_deep: &str, ours_deep: &str) -> String {
    let u: Vec<&str> = up_deep.split(' ').collect();
    let o: Vec<&str> = ours_deep.split(' ').collect();
    let mut i = 0;
    while i < u.len() && i < o.len() && u[i] == o[i] {
        i += 1;
    }
    let win = |v: &[&str], at: usize| {
        let lo = at.saturating_sub(3);
        let hi = (at + 4).min(v.len());
        v[lo..hi].join(" ")
    };
    let mut s = format!("UP[{}] | OURS[{}]", win(&u, i), win(&o, i));
    if s.len() > 160 {
        s.truncate(160);
    }
    s
}

/// Collapse a concrete error message into a coarse bucket key so similar
/// failures (differing only by an identifier/number/location) aggregate.
fn reason_key(e: &str) -> String {
    // strip trailing source-location detail and per-instance tokens
    let mut k = e.to_string();
    if let Some(i) = k.find(" at ") {
        k.truncate(i);
    }
    // drop quoted literals and digits which vary per file
    let k = Regex::new(r#""[^"]*""#).unwrap().replace_all(&k, "\"_\"");
    let k = Regex::new(r"\b\d+\b").unwrap().replace_all(&k, "N");
    let mut k = k.into_owned();
    if k.len() > 80 {
        k.truncate(80);
    }
    k
}

fn corpus_path() -> std::path::PathBuf {
    if let Ok(p) = std::env::var("ORACLE") {
        return p.into();
    }
    // crate is <ws>/crates/jsir-swc
    let crate_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    crate_dir
        .parent()
        .and_then(|p| p.parent())
        .unwrap()
        .join("corpus/test262_oracle.jsonl.gz")
}

/// Outcome of a full round-trip stability check on one file.
enum Rt {
    /// `source_to_ir` itself failed; the file can't round-trip (not our fault here).
    NoLower,
    /// `ir_to_source` (hir2ast + ast2source) failed.
    GenFail(String),
    /// Re-lowering the regenerated JS failed.
    RelowerFail(String),
    /// Round-trip succeeded; IR identical before and after (information-preserving).
    Stable,
    /// Round-trip succeeded but the IR changed (lossy lift or codegen).
    Drifted,
}

/// `source_to_ir -> ir_to_source -> source_to_ir` and check the IR is the same
/// up to source positions. The round-trip regenerates JS with different
/// formatting, so byte offsets and `<L C>` locations necessarily change; the
/// meaningful invariant is that the *structure* is identical (the `deep` form),
/// proving hir2ast + ast2source lose no structural information.
fn roundtrip_one(src: &str, loc: &Regex, num: &Regex, comment_attr: &Regex, comment_array: &Regex, raw_extra: &Regex) -> Rt {
    // Comments and literal `extra.raw` now round-trip (codegen comment store +
    // Str/Number raw), so compare the IR structurally with no special stripping.
    let _ = (comment_attr, comment_array, raw_extra);
    let strip = |s: &str| s.to_string();
    let ir1 = match catch_unwind(AssertUnwindSafe(|| jsir_swc::source_to_ir(src))) {
        Ok(Ok(op)) => op,
        _ => return Rt::NoLower,
    };
    let printed1 = ir1.print();
    let js = match catch_unwind(AssertUnwindSafe(|| jsir_swc::ir_to_source(&ir1))) {
        Ok(Ok(s)) => s,
        Ok(Err(e)) => return Rt::GenFail(e),
        Err(_) => return Rt::GenFail("PANIC".into()),
    };
    let printed2 = match catch_unwind(AssertUnwindSafe(|| jsir_swc::source_to_ir(&js))) {
        Ok(Ok(op)) => op.print(),
        Ok(Err(e)) => return Rt::RelowerFail(e),
        Err(_) => return Rt::RelowerFail("PANIC".into()),
    };
    let n1 = deep(&strip(&printed1), loc, num);
    let n2 = deep(&strip(&printed2), loc, num);
    if n1 == n2 {
        Rt::Stable
    } else {
        Rt::Drifted
    }
}

fn run_roundtrip(reader: impl BufRead, limit: Option<usize>) {
    // Comments aren't re-emitted by ir_to_source yet (swc codegen needs
    // BytePos-keyed comments; our lifted AST has dummy spans), so strip comment
    // attrs and empty the comments array before the structural comparison to
    // measure round-trip fidelity of everything *except* comment re-emission.
    let comment_attr = Regex::new(r#"#jsir<comment_(line|block)[^"]*"(\\.|[^"\\])*">"#).unwrap();
    // After attrs are stripped the file's array is left as `[, , ]`; flatten to `[]`.
    let comment_array = Regex::new(r"comments = \[[\s,]*\]").unwrap();
    // `extra.raw` is the verbatim source token (quote style etc.) which swc's
    // code generator does not reproduce; normalize it to compare structure+value.
    let raw_extra = Regex::new(r#"(_extra )"(\\.|[^"\\])*""#).unwrap();
    let loc = Regex::new(r"<L \d+ C \d+>").unwrap();
    let num = Regex::new(r"\b\d+\b").unwrap();
    let (mut stable, mut drift, mut genfail, mut relfail, mut nolower) = (0u32, 0u32, 0u32, 0u32, 0u32);
    let mut drift_paths: Vec<String> = Vec::new();
    let mut fail_reasons: std::collections::HashMap<String, (u32, String)> =
        std::collections::HashMap::new();
    let mut count = 0usize;
    for line in reader.lines() {
        let line = line.unwrap();
        if line.is_empty() {
            continue;
        }
        let v: serde_json::Value = serde_json::from_str(&line).unwrap();
        let p = v["p"].as_str().unwrap_or("");
        let src = v["src"].as_str().unwrap_or("");
        if let Some(l) = limit {
            if count >= l {
                break;
            }
        }
        count += 1;
        match roundtrip_one(src, &loc, &num, &comment_attr, &comment_array, &raw_extra) {
            Rt::NoLower => nolower += 1,
            Rt::Stable => stable += 1,
            Rt::Drifted => {
                drift += 1;
                drift_paths.push(p.to_string());
            }
            Rt::GenFail(e) => {
                genfail += 1;
                fail_reasons.entry(format!("GEN: {}", reason_key(&e))).or_insert((0, p.to_string())).0 += 1;
            }
            Rt::RelowerFail(e) => {
                relfail += 1;
                fail_reasons.entry(format!("RELOWER: {}", reason_key(&e))).or_insert((0, p.to_string())).0 += 1;
            }
        }
    }
    // Denominator = files we can lower (round-trip only applies to those).
    let total = stable + drift + genfail + relfail;
    let pct = |x: u32| if total > 0 { 100.0 * x as f64 / total as f64 } else { 0.0 };
    println!("== round-trip stability (lowerable files: {total}; not-lowerable skipped: {nolower}) ==");
    println!("  stable        {stable:6}  ({:.1}%)", pct(stable));
    println!("  drifted       {drift:6}  ({:.1}%)", pct(drift));
    println!("  gen_fail      {genfail:6}  ({:.1}%)  (ir_to_source)", pct(genfail));
    println!("  relower_fail  {relfail:6}  ({:.1}%)  (re-parse of our JS)", pct(relfail));
    let mut reasons: Vec<(&String, &(u32, String))> = fail_reasons.iter().collect();
    reasons.sort_by(|a, b| b.1 .0.cmp(&a.1 .0));
    if !reasons.is_empty() {
        println!("-- round-trip failure reasons (top 25 of {}) --", reasons.len());
        for (key, (n, example)) in reasons.into_iter().take(25) {
            println!("  {n:6}  {key}   e.g. {example}");
        }
    }
    if !drift_paths.is_empty() {
        println!("-- drifted (first 15 of {}) --", drift_paths.len());
        for p in drift_paths.iter().take(15) {
            println!("   {p}");
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut limit: Option<usize> = None;
    let mut show: Option<String> = None;
    let mut list: Option<String> = None;
    let mut roundtrip = false;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--show" => {
                show = args.get(i + 1).cloned();
                i += 2;
            }
            "--list" => {
                list = args.get(i + 1).cloned();
                i += 2;
            }
            "--roundtrip" => {
                roundtrip = true;
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

    let path = corpus_path();
    let file = std::fs::File::open(&path)
        .unwrap_or_else(|e| panic!("open oracle {}: {e}\nRun gen_oracle.py + pull it local first.", path.display()));
    let reader = std::io::BufReader::new(GzDecoder::new(file));

    if roundtrip {
        run_roundtrip(reader, limit);
        return;
    }

    let loc = Regex::new(r"<L \d+ C \d+>").unwrap();
    let num = Regex::new(r"\b\d+\b").unwrap();

    let (mut m, mut doff, mut dstruct, mut ofail, mut ufail) = (0u32, 0u32, 0u32, 0u32, 0u32);
    let mut struct_paths: Vec<String> = Vec::new();
    let mut offset_paths: Vec<String> = Vec::new();
    let mut ours_fail_paths: Vec<String> = Vec::new();
    let mut fail_reasons: std::collections::HashMap<String, (u32, String)> =
        std::collections::HashMap::new();
    let mut struct_sigs: std::collections::HashMap<String, (u32, String)> =
        std::collections::HashMap::new();
    let mut offset_sigs: std::collections::HashMap<String, (u32, String)> =
        std::collections::HashMap::new();
    let mut count = 0usize;

    for line in reader.lines() {
        let line = line.unwrap();
        if line.is_empty() {
            continue;
        }
        let v: serde_json::Value = serde_json::from_str(&line).unwrap();
        let p = v["p"].as_str().unwrap_or("");
        let src = v["src"].as_str().unwrap_or("");
        let up = v["ir"].as_str(); // null => upstream failed

        if let Some(target) = &show {
            if p == target {
                let o = ours(src);
                let ours_str = match &o {
                    Ok(s) => s.clone(),
                    Err(e) => format!("<ours failed: {e}>"),
                };
                println!("== {p} ==\n--- upstream ---\n{}\n--- ours ---\n{}",
                    up.unwrap_or("<upstream failed>"), ours_str);
                return;
            }
            continue;
        }

        // Skip upstream failures from the meaningful denominator but count them.
        let up = match up {
            Some(u) => u,
            None => {
                ufail += 1;
                continue;
            }
        };

        if let Some(l) = limit {
            if count >= l {
                break;
            }
        }
        count += 1;

        match ours(src) {
            Err(e) => {
                ofail += 1;
                ours_fail_paths.push(p.to_string());
                let key = reason_key(&e);
                let entry = fail_reasons.entry(key).or_insert((0, p.to_string()));
                entry.0 += 1;
            }
            Ok(o) => {
                if norm(up) == norm(&o) {
                    m += 1;
                } else if deep(up, &loc, &num) == deep(&o, &loc, &num) {
                    doff += 1;
                    offset_paths.push(p.to_string());
                    let sig = struct_sig(&norm(up), &norm(&o));
                    let entry = offset_sigs.entry(sig).or_insert((0, p.to_string()));
                    entry.0 += 1;
                } else {
                    dstruct += 1;
                    struct_paths.push(p.to_string());
                    let sig = struct_sig(&deep(up, &loc, &num), &deep(&o, &loc, &num));
                    let entry = struct_sigs.entry(sig).or_insert((0, p.to_string()));
                    entry.0 += 1;
                }
            }
        }
    }

    if let Some(kind) = list {
        let v = match kind.as_str() {
            "struct" => &struct_paths,
            "offset" => &offset_paths,
            "ours_fail" | "ofail" => &ours_fail_paths,
            _ => {
                eprintln!("--list expects struct|offset|ours_fail");
                return;
            }
        };
        for p in v {
            println!("{p}");
        }
        return;
    }

    if show.is_some() {
        eprintln!("relpath not found in corpus");
        return;
    }

    let total = m + doff + dstruct + ofail;
    println!("== local parity (upstream-ok entries: {total}; upstream_fail skipped: {ufail}) ==");
    let pct = |x: u32| if total > 0 { 100.0 * x as f64 / total as f64 } else { 0.0 };
    println!("  match          {m:6}  ({:.1}%)", pct(m));
    println!("  differ_offset  {doff:6}  ({:.1}%)", pct(doff));
    println!("  differ_struct  {dstruct:6}  ({:.1}%)", pct(dstruct));
    println!("  ours_fail      {ofail:6}  ({:.1}%)", pct(ofail));
    // Top structural-diff signatures (where our IR diverges in shape/attrs).
    let mut sigs: Vec<(&String, &(u32, String))> = struct_sigs.iter().collect();
    sigs.sort_by(|a, b| b.1 .0.cmp(&a.1 .0));
    println!("-- differ_struct signatures (top 25 of {}) --", sigs.len());
    for (sig, (n, example)) in sigs.into_iter().take(25) {
        println!("  {n:5}  {sig}\n          e.g. {example}");
    }

    // Top offset-diff signatures.
    let mut osigs: Vec<(&String, &(u32, String))> = offset_sigs.iter().collect();
    osigs.sort_by(|a, b| b.1 .0.cmp(&a.1 .0));
    println!("-- differ_offset signatures (top 15 of {}) --", osigs.len());
    for (sig, (n, example)) in osigs.into_iter().take(15) {
        println!("  {n:5}  {sig}\n          e.g. {example}");
    }

    // Top failure reasons (the highest-value signal for what to implement next).
    let mut reasons: Vec<(&String, &(u32, String))> = fail_reasons.iter().collect();
    reasons.sort_by(|a, b| b.1 .0.cmp(&a.1 .0));
    println!("-- ours_fail reasons (top 25 of {}) --", reasons.len());
    for (key, (n, example)) in reasons.into_iter().take(25) {
        println!("  {n:6}  {key}   e.g. {example}");
    }
}
