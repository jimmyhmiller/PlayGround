//! Fast in-process batch probe: compile every `.js` under a directory and print
//! one line per fixture — `MEMO <name>` if our output memoizes (`_c(`), else
//! `BAIL <name>`. No Node oracle, no per-process startup, so it runs in ~1s over
//! the whole corpus. Used to detect memoization regressions between changes:
//!   cargo run --release -p jsir-ssa --example cc_batch -- <fixtures-dir> > a.txt
//!   (make a change) ... > b.txt ; diff a.txt b.txt
use std::fs;
use std::path::Path;

/// `(cache_size, memo_block_count)` from compiled output, mirroring the gate harness.
fn structure(code: &str) -> Option<(usize, usize)> {
    let n = code.split("_c(").nth(1)?.split(')').next()?.trim().parse::<usize>().ok()?;
    let block_count = code
        .match_indices("if (")
        .filter(|(i, _)| code[i + 4..].trim_start_matches(['(', ' ']).starts_with("$["))
        .count();
    Some((n, block_count))
}

fn main() {
    let dir = std::env::args().nth(1).unwrap_or_else(|| {
        "crates/jsir-ssa/oracle/fixtures".to_string()
    });
    let mut entries: Vec<_> = fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("read_dir {dir}: {e}"))
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().map(|x| x == "js").unwrap_or(false))
        .collect();
    entries.sort();
    for path in entries {
        let name = Path::new(&path).file_name().unwrap().to_string_lossy().to_string();
        let src = match fs::read_to_string(&path) {
            Ok(s) => s,
            Err(_) => continue,
        };
        // Emit `<name> <struct>` where struct is `cache,blocks` (e.g. `3,1`) when
        // memoized, else `none` (pass-through) or `err`. Stable, parseable, fast.
        let s = match jsir_ssa::codegen::compile(&src) {
            Ok(out) => structure(&out).map(|(c, b)| format!("{c},{b}")).unwrap_or_else(|| "none".into()),
            Err(_) => "err".into(),
        };
        println!("{name}\t{s}");
    }
}
