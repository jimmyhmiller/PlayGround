//! Probe loading `cljs/core.cljc` — the ClojureScript analyzer-side
//! macro file. 3504 lines, 166 `defmacro`s, runs on the JVM during
//! ClojureScript compilation. Loading this in isolation tells us how
//! much of the bootstrap macro surface our reader + analyzer handle.

use clojure_jvm::lang::compiler::Session;
use clojure_jvm::lang::lisp_reader::Reader;
use std::cell::RefCell;

thread_local! {
    static LAST_PANIC: RefCell<Option<String>> = const { RefCell::new(None) };
}

fn install_capturing_hook() {
    std::panic::set_hook(Box::new(|info| {
        let msg = info.to_string();
        eprintln!("[panic-hook] {}", msg);
        LAST_PANIC.with(|p| *p.borrow_mut() = Some(msg));
    }));
}

const CLJS_CORE_CLJC: &str = "/tmp/cljs_core_cljc.cljc";

#[test]
#[ignore = "tracks cljs.core.cljc loading progress — long-running"]
fn load_cljs_core_cljc_progressively() {
    install_capturing_hook();
    let src = std::fs::read_to_string(CLJS_CORE_CLJC).unwrap_or_else(|e| {
        panic!(
            "cljs core cljc must be at {CLJS_CORE_CLJC} \
             (download from https://raw.githubusercontent.com/clojure/clojurescript/master/src/main/clojure/cljs/core.cljc): {e}"
        )
    });

    let mut sess = Session::new();
    let mut i: usize = 0;
    let mut byte_pos: usize = 0;
    loop {
        eprintln!("[cljc] iter {i} starting at byte {byte_pos}");
        let slice = &src[byte_pos..];
        let read_outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut r = Reader::new(slice);
            let before = r.byte_pos();
            let form = r.read();
            let after = r.byte_pos();
            (form, before, after)
        }));
        let (form_opt, _before, after) = match read_outcome {
            Ok(t) => match t.0 {
                Ok(Some(form)) => (Some(form), t.1, t.2),
                Ok(None) => break,
                Err(e) => {
                    eprintln!("[cljc] form {i} READ ERR: {e}");
                    panic!("read err at form {i}: {e}");
                }
            },
            Err(err) => {
                let msg = LAST_PANIC
                    .with(|p| p.borrow().clone())
                    .unwrap_or_else(|| panic_msg(&*err));
                eprintln!("[cljc] form {i} READ PANIC: {msg}");
                panic!("read panic at form {i}");
            }
        };
        byte_pos += after;
        eprintln!("[cljc] form {i} read OK, byte_pos={byte_pos}");
        let form = form_opt.unwrap();

        eprintln!("[cljc] form {i} starting eval…");
        let eval_outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            sess.eval_form(form);
        }));
        eprintln!(
            "[cljc] form {i} eval completed (outcome: {})",
            if eval_outcome.is_ok() { "Ok" } else { "Err" }
        );
        if let Err(err) = eval_outcome {
            let msg = LAST_PANIC.with(|p| p.borrow().clone())
                .unwrap_or_else(|| panic_msg(&*err));
            eprintln!("[cljc] form {i} EVAL PANIC: {msg}");
            if std::env::var("CLJVM_LOADER_SKIP_PANICS").is_ok() {
                eprintln!("[cljc] form {i} SKIPPED (CLJVM_LOADER_SKIP_PANICS=1)");
                LAST_PANIC.with(|p| *p.borrow_mut() = None);
                i += 1;
                continue;
            }
            panic!("eval panic at form {i} — see preceding line for message");
        }
        i += 1;
    }
    eprintln!("[cljc] processed {i} forms successfully");
}

fn panic_msg(err: &(dyn std::any::Any + Send)) -> String {
    if let Some(s) = err.downcast_ref::<&'static str>() {
        (*s).to_string()
    } else if let Some(s) = err.downcast_ref::<String>() {
        s.clone()
    } else if let Some(s) = err.downcast_ref::<Box<str>>() {
        s.to_string()
    } else {
        format!("<panic payload type {}>", std::any::type_name_of_val(err))
    }
}
