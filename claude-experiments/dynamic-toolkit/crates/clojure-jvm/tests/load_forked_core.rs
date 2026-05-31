//! Probe loading our forked `clojure/core.clj` via `Session::
//! new_with_clojure_core`. Reports which form it stalls on so we know
//! the next patch target.
//!
//! This is the "we own this file" version of `load_upstream_core`.
//! Forms can be edited / commented out in place to advance the loader.

use clojure_jvm::lang::compiler::Session;
use std::cell::RefCell;

thread_local! {
    static LAST_PANIC: RefCell<Option<String>> = const { RefCell::new(None) };
}

fn install_capturing_hook() {
    std::panic::set_hook(Box::new(|info| {
        let msg = info.to_string();
        eprintln!("[forked-core] {}", msg);
        LAST_PANIC.with(|p| *p.borrow_mut() = Some(msg));
    }));
}

#[test]
#[ignore = "drives forked clojure.core loading; long-running"]
fn forked_core_loads_fully() {
    install_capturing_hook();
    let outcome = std::panic::catch_unwind(|| {
        let _sess = Session::new_with_clojure_core();
    });
    if let Err(_e) = outcome {
        let msg = LAST_PANIC
            .with(|p| p.borrow().clone())
            .unwrap_or_else(|| "<no panic message captured>".into());
        panic!("forked core load failed: {msg}");
    }
}
