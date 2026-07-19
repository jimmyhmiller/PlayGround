// Burn-down harness. Arg1: backend "tw" (default) or "jit". Arg2: start index.
// Categorizes each corpus expr vs the real-Clojure oracle. Runs on TreeWalk for
// measurement (panics unwind, so failures are catchable); the JIT pass verifies
// only the exprs that PASS on TreeWalk (they don't panic, so no shim-abort).
use microlang::{LowBitModel, Runtime};
use microlang::code::TreeWalk;
use microlang::jit_cranelift::JitCranelift;
use std::io::{BufRead, Write};

fn run(src: &str, jit: bool) -> Result<String, String> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut rt = Runtime::<LowBitModel>::new();
        let r = if jit {
            let b = JitCranelift::<LowBitModel>::new();
            clojure_stub::run(&mut rt, &b, src)
        } else {
            clojure_stub::run(&mut rt, &TreeWalk, src)
        };
        clojure_stub::clj_str(&rt, r)
    })).map_err(|e| e.downcast_ref::<&str>().map(|s| s.to_string())
        .or_else(|| e.downcast_ref::<String>().cloned()).unwrap_or_else(|| "panic".into())
        .lines().next().unwrap_or("").to_string())
}

fn main() {
    std::panic::set_hook(Box::new(|_| {}));
    let jit = std::env::args().nth(1).as_deref() == Some("jit");
    let start: usize = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(0);
    let f = std::fs::File::open("clojure-stub/tests/core_suite/expected.txt").unwrap();
    let entries: Vec<(String, String)> = std::io::BufReader::new(f).lines().filter_map(|l| l.ok())
        .filter_map(|l| l.split_once('\t').map(|(a, b)| (a.to_string(), b.to_string())))
        .filter(|(_, b)| b != "ERROR").collect();
    let mut out = std::io::stdout();
    for i in start..entries.len() {
        let (expr, expected) = &entries[i];
        let verdict = match run(expr, jit) {
            Ok(got) if &got == expected => "PASS".to_string(),
            Ok(got) => format!("WRONG\twant {expected}\tgot {got}"),
            Err(e) => {
                let cat = if e.contains("Unable to resolve") || e.contains("not lowerable") || e.contains("not callable") { "MISSING" } else { "PANIC" };
                format!("{cat}\twant {expected}\t{e}")
            }
        };
        writeln!(out, "{i}\t{expr}\t{verdict}").ok();
        out.flush().ok();
    }
}
