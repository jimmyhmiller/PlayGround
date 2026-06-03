//! Categorize why we miss a list of fixtures: run codegen::compile and bucket by
//! bail reason (Err string head) / outcome, plus note source constructs present.
//! Usage: why_miss <list-file> [fixtures-dir]
use std::collections::BTreeMap;
use std::fs;

fn main() {
    let list = std::env::args().nth(1).expect("usage: why_miss <list> [dir]");
    let dir = std::env::args().nth(2).unwrap_or_else(|| "oracle/fixtures".into());
    let names = fs::read_to_string(&list).expect("read list");
    let mut buckets: BTreeMap<String, usize> = BTreeMap::new();
    let mut examples: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let mut total = 0;
    for name in names.lines().map(|s| s.trim()).filter(|s| !s.is_empty()) {
        let path = format!("{dir}/{name}");
        let src = match fs::read_to_string(&path) {
            Ok(s) => s,
            Err(_) => continue,
        };
        total += 1;
        let key = match jsir_ssa::codegen::compile(&src) {
            Ok(out) => {
                if out.contains("_c(") {
                    "MEMOIZED (structure mismatch)".to_string()
                } else {
                    "compiled, NO memo (pass-through / cache_size 0)".to_string()
                }
            }
            Err(e) => {
                // Head of the error message (before first ':' detail or '(').
                let head = e.split([':', '(']).next().unwrap_or(&e).trim().to_string();
                format!("BAIL: {head}")
            }
        };
        *buckets.entry(key.clone()).or_default() += 1;
        let ex = examples.entry(key).or_default();
        if ex.len() < 4 {
            ex.push(name.to_string());
        }
    }
    println!("total {total}\n");
    let mut rows: Vec<(&String, &usize)> = buckets.iter().collect();
    rows.sort_by(|a, b| b.1.cmp(a.1));
    for (k, n) in rows {
        println!("{n:4}  {k}");
        if let Some(ex) = examples.get(k) {
            println!("        e.g. {}", ex.join(", "));
        }
    }
}
