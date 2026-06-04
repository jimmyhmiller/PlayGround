//! Differential conformance harness: evaluate a corpus of expressions in
//! BOTH clojure-jvm and real Clojure, diff the `pr-str` of each result, and
//! report a coverage percentage plus every mismatch. Turns ad-hoc bug-hunting
//! into a measured gap against the reference implementation.
//!
//! Oracle = the `clojure` CLI (one JVM launch). Our side loads full upstream
//! core.clj once. Because some expressions trigger NON-UNWINDING aborts that
//! `catch_unwind` can't trap, our evaluation runs in a HELPER SUBPROCESS that
//! writes results incrementally; if it crashes, the driver records the
//! offending expression as `ABORT`, adds it to a skip set, and reruns — so one
//! crash doesn't lose the whole run.
//!
//! Run: `cargo test -p clojure-jvm --test conformance -- --ignored --nocapture`

use clojure_jvm::lang::compiler::Session;
use clojure_jvm::lang::lisp_reader::Reader;
use std::cell::RefCell;
use std::io::Write;

const UPSTREAM_CORE: &str =
    "/Users/jimmyhmiller/Documents/Code/open-source/clojure/src/clj/clojure/core.clj";
const CORPUS: &str = include_str!("conformance_corpus.txt");

thread_local! {
    static LAST_PANIC: RefCell<Option<String>> = const { RefCell::new(None) };
}

fn corpus_exprs() -> Vec<String> {
    CORPUS
        .lines()
        .map(|l| l.trim())
        .filter(|l| !l.is_empty() && !l.starts_with(';'))
        .map(|l| l.to_string())
        .collect()
}

fn esc(s: &str) -> String {
    s.replace('\\', "\\\\").replace('\n', "\\n").replace('\t', "\\t")
}
fn unesc(s: &str) -> String {
    let mut out = String::new();
    let mut it = s.chars();
    while let Some(c) = it.next() {
        if c == '\\' {
            match it.next() {
                Some('n') => out.push('\n'),
                Some('t') => out.push('\t'),
                Some('\\') => out.push('\\'),
                Some(o) => out.push(o),
                None => {}
            }
        } else {
            out.push(c);
        }
    }
    out
}

fn load_full_core() -> Session {
    std::panic::set_hook(Box::new(|info| {
        LAST_PANIC.with(|p| *p.borrow_mut() = Some(info.to_string()));
    }));
    let src = std::fs::read_to_string(UPSTREAM_CORE).expect("upstream core.clj");
    let mut sess = Session::new();
    let mut byte_pos = 0usize;
    loop {
        let slice = &src[byte_pos..];
        let read = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut r = Reader::new(slice);
            let f = r.read();
            (f, r.byte_pos())
        }));
        let (form, after) = match read {
            Ok((Ok(Some(f)), a)) => (f, a),
            _ => break,
        };
        byte_pos += after;
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            sess.eval_form(form);
        }));
    }
    // Make clojure.string available (a require would do this in a real REPL).
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        sess.eval_str("(clojure.lang.RT/load \"clojure/string\")");
    }));
    sess
}

fn ours_eval(sess: &mut Session, src: &str) -> String {
    LAST_PANIC.with(|p| *p.borrow_mut() = None);
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        clojure_jvm::runtime::pr_str_bits(sess.eval_str(src))
    }));
    match r {
        Ok(s) => s,
        Err(_) => {
            let m = LAST_PANIC
                .with(|p| p.borrow().clone())
                .unwrap_or_else(|| "panic".into());
            format!("ERROR:{}", m.chars().take(60).collect::<String>())
        }
    }
}

/// HELPER MODE: load core once, eval each corpus expr whose index isn't in
/// CONF_SKIP, append `idx<TAB>escaped-result` to CONF_OUT (flushing each), so
/// a later abort still leaves a record of everything completed so far.
fn run_helper() {
    let skip: std::collections::HashSet<usize> = std::env::var("CONF_SKIP")
        .unwrap_or_default()
        .split(',')
        .filter_map(|s| s.parse().ok())
        .collect();
    let out_path = std::env::var("CONF_OUT").unwrap();
    let mut out = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&out_path)
        .unwrap();
    let exprs = corpus_exprs();
    let mut sess = load_full_core();
    for (i, expr) in exprs.iter().enumerate() {
        if skip.contains(&i) {
            continue;
        }
        // Mark in-progress BEFORE eval, flushed, so a crash is attributable.
        writeln!(out, "{i}\tINPROGRESS").unwrap();
        out.flush().unwrap();
        let got = ours_eval(&mut sess, expr);
        writeln!(out, "{i}\t{}", esc(&got)).unwrap();
        out.flush().unwrap();
    }
    writeln!(out, "DONE").unwrap();
    out.flush().unwrap();
}

/// Drive the helper subprocess, recovering from crashes by skipping the
/// offending index. Returns one result per corpus index (`ABORT` for crashers).
fn ours_results(n: usize) -> Vec<String> {
    let exe = std::env::current_exe().unwrap();
    let out_path = std::env::temp_dir().join("cljvm_conf_ours.txt");
    let mut results: Vec<Option<String>> = vec![None; n];
    let mut skip: Vec<usize> = Vec::new();
    loop {
        let _ = std::fs::remove_file(&out_path);
        let status = std::process::Command::new(&exe)
            .args(["--ignored", "--exact", "conformance_vs_clojure", "--nocapture"])
            .env("CONF_HELPER", "1")
            .env("CONF_OUT", &out_path)
            .env("CONF_SKIP", skip.iter().map(|i| i.to_string()).collect::<Vec<_>>().join(","))
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .unwrap();
        let text = std::fs::read_to_string(&out_path).unwrap_or_default();
        let mut last_inprogress: Option<usize> = None;
        let mut done = false;
        for line in text.lines() {
            if line == "DONE" {
                done = true;
                break;
            }
            if let Some((idx, val)) = line.split_once('\t') {
                let idx: usize = idx.parse().unwrap();
                if val == "INPROGRESS" {
                    last_inprogress = Some(idx);
                } else {
                    results[idx] = Some(unesc(val));
                    last_inprogress = None;
                }
            }
        }
        if done || status.success() {
            break;
        }
        // Crashed: the index marked INPROGRESS but never completed is the
        // offender. Record it as ABORT and skip on the next pass.
        match last_inprogress {
            Some(k) => {
                results[k] = Some("ABORT".to_string());
                skip.push(k);
            }
            None => break, // crashed without progress marker — give up
        }
    }
    results
        .into_iter()
        .map(|o| o.unwrap_or_else(|| "ABORT".to_string()))
        .collect()
}

fn clojure_oracle(exprs: &[String]) -> Vec<String> {
    let program = r#"
(let [lines (clojure.string/split-lines (slurp (first *command-line-args*)))]
  (doseq [line lines]
    (let [r (try (pr-str (eval (read-string line)))
                 (catch Throwable e (str "ERROR:" (.getSimpleName (class e)))))]
      (print (char 1)) (print r) (print (char 2)))))
"#;
    let dir = std::env::temp_dir();
    let corpus_path = dir.join("cljvm_conf_corpus.txt");
    let prog_path = dir.join("cljvm_conf_oracle.clj");
    std::fs::write(&corpus_path, exprs.join("\n")).unwrap();
    std::fs::write(&prog_path, program).unwrap();
    let out = std::process::Command::new("clojure")
        .arg("-M")
        .arg(prog_path.to_str().unwrap())
        .arg(corpus_path.to_str().unwrap())
        .current_dir(&dir)
        .output()
        .expect("failed to run `clojure`");
    if !out.status.success() {
        panic!("clojure oracle failed:\n{}", String::from_utf8_lossy(&out.stderr));
    }
    let stdout = String::from_utf8_lossy(&out.stdout);
    let mut results = Vec::new();
    for chunk in stdout.split('\u{0001}').skip(1) {
        if let Some(end) = chunk.find('\u{0002}') {
            results.push(chunk[..end].to_string());
        }
    }
    assert_eq!(results.len(), exprs.len(), "oracle result count mismatch");
    results
}

#[test]
#[ignore = "differential conformance vs real Clojure — long-running, needs `clojure`"]
fn conformance_vs_clojure() {
    if std::env::var("CONF_HELPER").is_ok() {
        run_helper();
        return;
    }
    let exprs = corpus_exprs();
    eprintln!("=== corpus: {} expressions ===", exprs.len());
    eprintln!("=== clojure oracle ===");
    let oracle = clojure_oracle(&exprs);
    eprintln!("=== our impl (subprocess, crash-recovering) ===");
    let ours = ours_results(exprs.len());

    let mut matches = 0usize;
    let mut both_err = 0usize;
    let mut mismatches: Vec<(String, String, String)> = Vec::new();
    for ((expr, want), got) in exprs.iter().zip(oracle.iter()).zip(ours.iter()) {
        let want_err = want.starts_with("ERROR:");
        let got_err = got.starts_with("ERROR:") || got == "ABORT";
        if got == want {
            matches += 1;
        } else if got_err && want_err {
            both_err += 1;
        } else {
            mismatches.push((expr.clone(), got.clone(), want.clone()));
        }
    }
    let total = exprs.len();
    eprintln!("\n========== CONFORMANCE ==========");
    eprintln!("exact match : {matches}/{total}");
    eprintln!("both error  : {both_err}/{total}  (we agree it's unsupported)");
    eprintln!("MISMATCH    : {}/{total}", mismatches.len());
    eprintln!(
        "agreement   : {:.1}%  (matches / (total - both-error))",
        100.0 * matches as f64 / (total - both_err).max(1) as f64
    );
    eprintln!("---------- mismatches ----------");
    for (expr, got, want) in &mismatches {
        eprintln!("  {expr}");
        eprintln!("      ours: {got}");
        eprintln!("      clj : {want}");
    }
    eprintln!("=================================");
}
