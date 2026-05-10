//! Host context: thread-local pointer to the live `Engine` so externs
//! can reach the symbol table, the namespace registry, and the GC
//! runtime. Mirrors `microlisp::host`.

use std::cell::Cell;
use std::collections::HashMap;
use std::sync::Mutex;

use dynlang::gc::DynGcRuntime;
use dynlower::JitModule;
use dynobj::roots::AtomicRootSet;

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
    /// Raw pointer to the engine's `JitModule`. Set after
    /// `JitModule::new_empty` in `Engine::new`; valid for the lifetime
    /// of the engine.
    pub jit: *const JitModule,
    /// Keyword intern table. Maps a symbol-id (the keyword's name as
    /// interned in the `SymbolTable`) to an index into `kw_roots`,
    /// which holds the actual `Keyword` heap pointer. The roots are
    /// registered with the GC as an extra root source so a moving
    /// collection rewrites them in place.
    pub kw_index: Mutex<HashMap<u32, usize>>,
    /// Pinned, GC-traced storage for interned `Keyword` heap
    /// pointers. Indexed by the values stored in `kw_index`.
    pub kw_roots: AtomicRootSet,
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

impl Host {
    /// Intern a keyword by its symbol-id (the keyword's name in the
    /// shared `SymbolTable`). The first call for a given sym_id
    /// allocates a fresh `Keyword` heap object and pins its tagged
    /// pointer in `kw_roots`; every subsequent call returns the
    /// SAME tagged pointer, so `clj_eq` (bitwise) treats `:foo` and
    /// `:foo` as equal.
    ///
    /// Concurrency: a fast-path check under a brief `kw_index` lock,
    /// then alloc OUTSIDE the lock (so a GC during alloc can't
    /// deadlock with another thread's intern), then a re-check under
    /// the lock to handle the race where two threads tried to intern
    /// the same name simultaneously.
    pub fn intern_keyword(&self, sym_id: u32) -> u64 {
        // Fast path.
        {
            let map = self.kw_index.lock().unwrap();
            if let Some(&idx) = map.get(&sym_id) {
                return self.kw_roots.get(idx);
            }
        }
        // Slow path: allocate a fresh Keyword. Held in a RootScope
        // until we publish it into `kw_roots` so an intervening GC
        // doesn't strand it.
        dynobj::roots::with_scope(2, |scope| {
            let kw = crate::collections::alloc_keyword(scope, sym_id);
            let bits = kw.get();
            let mut map = self.kw_index.lock().unwrap();
            // Re-check: another thread may have raced us.
            if let Some(&idx) = map.get(&sym_id) {
                return self.kw_roots.get(idx);
            }
            let idx = self.kw_roots.add(bits);
            map.insert(sym_id, idx);
            bits
        })
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
