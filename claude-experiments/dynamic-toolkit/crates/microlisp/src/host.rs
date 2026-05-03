//! Host context: thread-local pointer to the live `Engine` so externs can
//! reach the symbol table and macro env. Mirrors the lighter `dynlang::host`
//! pattern but specialized — we always run single-threaded.

use std::cell::{Cell, RefCell};
use std::collections::HashMap;

use crate::symbols::SymbolTable;
use dynir::ir::FuncRef;
use dynlang::gc::DynGcRuntime;

pub struct Host {
    pub sym: RefCell<SymbolTable>,
    /// macro_env: symbol id → FuncRef of the JIT-compiled macro body.
    pub macro_env: RefCell<HashMap<u32, FuncRef>>,
    /// Raw pointer to the engine's `DynGcRuntime`. Valid for as long as the
    /// owning `Engine` lives. Read by `alloc_cons` and friends.
    pub gc: *const DynGcRuntime,
}

thread_local! {
    static ACTIVE: Cell<*const Host> = const { Cell::new(std::ptr::null()) };
}

pub fn with_host<R>(f: impl FnOnce(&Host) -> R) -> R {
    let p = ACTIVE.with(|c| c.get());
    assert!(!p.is_null(), "microlisp: no Host installed on this thread");
    let host = unsafe { &*p };
    f(host)
}

pub struct HostGuard {
    prev: *const Host,
}

impl Drop for HostGuard {
    fn drop(&mut self) {
        ACTIVE.with(|c| c.set(self.prev));
    }
}

pub fn install(host: &Host) -> HostGuard {
    let prev = ACTIVE.with(|c| {
        let p = c.get();
        c.set(host as *const Host);
        p
    });
    HostGuard { prev }
}
