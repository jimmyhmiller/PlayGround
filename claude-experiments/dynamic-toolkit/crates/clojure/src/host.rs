//! Host context: thread-local pointer to the live `Engine` so externs
//! can reach the symbol table, the namespace registry, and the GC
//! runtime. Mirrors `microlisp::host`.

use std::cell::Cell;
use std::sync::Mutex;

use dynlang::gc::DynGcRuntime;

use crate::symbols::SymbolTable;
use crate::types::Types;

pub struct Host {
    /// Symbol intern table. Behind a `Mutex` so multiple threads can
    /// share an `Engine` and intern concurrently.
    pub sym: Mutex<SymbolTable>,
    /// Raw pointer to the engine's `DynGcRuntime`. Valid for the
    /// lifetime of the engine.
    pub gc: *const DynGcRuntime,
    /// Type ID handles. Externs that allocate need to know these.
    pub types: Types,
}

thread_local! {
    static ACTIVE: Cell<*const Host> = const { Cell::new(std::ptr::null()) };
}

pub fn with_host<R>(f: impl FnOnce(&Host) -> R) -> R {
    let p = ACTIVE.with(|c| c.get());
    assert!(!p.is_null(), "clojure: no Host installed on this thread");
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
