//! Conformance runner: loads partial core, evals the corpus, writes our
//! pr-str output per expression for comparison against real Clojure.
//! Driven by env vars so an external harness can restart it past crashes:
//!   CORPUS=path  RESULTS=path  START=idx
use clojure_jvm::lang::compiler::Session;
use clojure_jvm::lang::lisp_reader::{Reader, read_str};
use clojure_jvm::runtime::pr_str_bits;
use std::cell::RefCell;
use std::io::Write;
use std::panic::AssertUnwindSafe;

thread_local! { static MSG: RefCell<Option<String>> = const { RefCell::new(None) }; }
const TAG_NIL: u64 = 0x7FFC_0000_0000_0000;

fn append(path: &str, line: &str) {
    let mut f = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .unwrap();
    writeln!(f, "{line}").ok();
    f.flush().ok();
}

#[test]
fn run() {
    let corpus = std::env::var("CORPUS").unwrap_or_else(|_| "conformance/corpus.edn".into());
    let results = std::env::var("RESULTS").unwrap_or_else(|_| "/tmp/ours.tsv".into());
    let start: usize = std::env::var("START")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    std::panic::set_hook(Box::new(|info| {
        MSG.with(|m| {
            *m.borrow_mut() = Some(
                info.to_string()
                    .replace('\n', " ")
                    .chars()
                    .take(120)
                    .collect(),
            )
        });
    }));
    let results2 = results.clone();
    std::thread::Builder::new()
        .stack_size(512 * 1024 * 1024)
        .spawn(move || {
            let exprs: Vec<String> = std::fs::read_to_string(&corpus)
                .unwrap()
                .lines()
                .map(|s| s.to_string())
                .collect();
            // Load partial core (stop before known hard-panic form 619).
            let core =
                std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/clojure/core.clj"))
                    .unwrap();
            let mut sess = Session::new();
            let mut bp = 0usize;
            let mut idx = 0;
            loop {
                if idx == 619 {
                    break;
                }
                let slice = &core[bp..];
                let mut r = Reader::new(slice);
                let before = r.byte_pos();
                let form = match r.read() {
                    Ok(Some(f)) => f,
                    _ => break,
                };
                bp += r.byte_pos() - before;
                if std::panic::catch_unwind(AssertUnwindSafe(|| sess.eval_form(form))).is_err() {
                    break;
                }
                idx += 1;
            }
            // Eval corpus from START.
            for i in start..exprs.len() {
                let src = exprs[i].trim();
                if src.is_empty() {
                    append(&results2, &format!("{i}\tEMPTY\t"));
                    continue;
                }
                MSG.with(|m| *m.borrow_mut() = None);
                let r = std::panic::catch_unwind(AssertUnwindSafe(|| {
                    let f = read_str(src).ok()?;
                    Some(sess.eval_form(f))
                }));
                match r {
                    Ok(Some(b)) if b != TAG_NIL => {
                        append(&results2, &format!("{i}\tOK\t{}", pr_str_bits(b)))
                    }
                    Ok(Some(_)) => append(&results2, &format!("{i}\tOK\tnil")),
                    Ok(None) => append(&results2, &format!("{i}\tREADFAIL\t")),
                    Err(_) => {
                        let m = MSG.with(|m| m.borrow().clone()).unwrap_or_default();
                        append(&results2, &format!("{i}\tPANIC\t{m}"));
                        std::process::exit(0); // session may be poisoned; harness restarts
                    }
                }
            }
            append(&results2, "DONE\t\t");
        })
        .unwrap()
        .join()
        .ok();
}
