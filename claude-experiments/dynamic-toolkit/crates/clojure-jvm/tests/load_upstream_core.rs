//! Attempt to load upstream `clojure/core.clj` directly through our
//! `Session::eval_str`, one form at a time. Reports the first form that
//! breaks so we know exactly what's still missing.
//!
//! The user goal: load every fn in upstream core.clj and run them
//! through our JIT. This is the driver test that proves we're there.

use clojure_jvm::lang::compiler::Session;
use clojure_jvm::lang::lisp_reader::Reader;
use std::cell::RefCell;

thread_local! {
    static LAST_PANIC: RefCell<Option<String>> = const { RefCell::new(None) };
}

fn install_capturing_hook() {
    std::panic::set_hook(Box::new(|info| {
        let msg = info.to_string();
        LAST_PANIC.with(|p| *p.borrow_mut() = Some(msg));
    }));
}

const UPSTREAM_CORE: &str =
    "/Users/jimmyhmiller/Documents/Code/open-source/clojure/src/clj/clojure/core.clj";

/// Read+eval upstream core.clj one form at a time. Report the first form
/// (read or eval) that panics, so the next thing to implement is obvious.
#[test]
#[ignore = "tracks progress against upstream core.clj — long-running"]
fn load_upstream_core_progressively() {
    install_capturing_hook();
    let src = std::fs::read_to_string(UPSTREAM_CORE)
        .expect("upstream clojure/core.clj must be reachable at the hard-coded path");

    let mut sess = Session::new();
    let mut i: usize = 0;
    let mut byte_pos: usize = 0;
    loop {
        eprintln!("[upstream] iter {i} starting at byte {byte_pos}");
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
                Err(e) => panic!("[upstream] form {i} READ ERR: {e}"),
            },
            Err(err) => {
                let msg = LAST_PANIC
                    .with(|p| p.borrow().clone())
                    .unwrap_or_else(|| panic_msg(&*err));
                panic!("[upstream] form {i} READ PANIC: {msg}");
            }
        };
        byte_pos += after;
        eprintln!("[upstream] form {i} read OK, byte_pos={byte_pos}");
        let form = form_opt.unwrap();

        eprintln!("[upstream] form {i} starting eval…");
        let eval_outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            sess.eval_form(form);
        }));
        eprintln!("[upstream] form {i} eval completed (outcome: {})", if eval_outcome.is_ok() { "Ok" } else { "Err" });
        if let Err(err) = eval_outcome {
            let msg = LAST_PANIC.with(|p| p.borrow().clone())
                .unwrap_or_else(|| panic_msg(&*err));
            eprintln!("[upstream] form {i} EVAL PANIC: {msg}");
            panic!("eval panic at form {i} — see preceding line for message");
        }
        i += 1;
    }
    eprintln!("[upstream] processed {i} forms successfully");
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
