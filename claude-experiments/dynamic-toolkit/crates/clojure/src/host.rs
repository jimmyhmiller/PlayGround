//! Host context: thread-local pointer to the live `Engine` so externs
//! can reach the symbol table, the namespace registry, and the GC
//! runtime. Mirrors `microlisp::host`.

use std::cell::Cell;
use std::collections::HashMap;
use std::collections::HashSet;
use std::sync::Mutex;

use dynlang::gc::DynGcRuntime;
use dynlower::JitModule;
use dynobj::roots::AtomicRootSet;

use crate::symbols::SymbolTable;
use crate::types::{Layouts, Types};

pub struct Host {
    /// Symbol intern table. `SymbolTable` is internally synchronized
    /// (RwLock around the HashMap+Vec); callers just hold a shared
    /// reference. Mirrors JVM Clojure's `Symbol/intern` model — no
    /// outer Mutex held across compile/expand, so JIT-called macro
    /// bodies can intern freely without deadlocking on the lock the
    /// driver was holding.
    pub sym: SymbolTable,
    /// Raw pointer to the engine's `DynGcRuntime`. Valid for the
    /// lifetime of the engine.
    pub gc: *const DynGcRuntime,
    /// Type ID handles. Externs that allocate need to know these.
    pub types: Types,
    /// Heap-object field offsets resolved once at engine init from
    /// `dm.obj_types`. Single source of truth for `unsafe { p.add(N) }`
    /// sites in `value.rs` / `collections.rs` / `namespace.rs` —
    /// previously each file maintained its own parallel `const`s.
    pub layouts: Layouts,
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
    /// pointers. Indexed by the values stored in `kw_index`. Held
    /// behind a `Box` so its address is stable when the surrounding
    /// `Engine` is moved out of `Engine::new()`'s stack frame — the
    /// raw `*const dyn RootSource` we register with the GC must
    /// outlive the engine, and a non-boxed inline field would be
    /// invalidated by that move.
    pub kw_roots: Box<AtomicRootSet>,
    /// Field-list metadata for every `(deftype* Name [f0 f1 …])`.
    /// Maps the type's name (symbol-id) to the field names in
    /// declaration order. The compiler/runtime use this to convert
    /// `(.-field instance)` into a varlen-values index lookup.
    pub deftype_fields: Mutex<HashMap<u32, Vec<u32>>>,
    /// `(method_sym_id, type_name_sym_id) → Fn heap pointer`. Each
    /// `extend-type` invocation populates entries here; each
    /// `(.method obj …)` dispatch reads `obj`'s type_name and looks
    /// up the matching `Fn`. Behind a Mutex (multi-thread safe);
    /// the Fn pointers are GC-traced via `method_roots`.
    pub method_table: Mutex<HashMap<(u32, u32), usize>>,
    /// GC-traced storage for the `Fn` pointers in `method_table`.
    /// `method_table` stores indices into this set rather than the
    /// pointers directly, so a moving collection can rewrite the
    /// entries in place. Same `Box`-pinned reasoning as `kw_roots`.
    pub method_roots: Box<AtomicRootSet>,
    /// Raw pointer to the engine's `globals` `RootSet`, plus the
    /// slot index where the `clojure.core` namespace pointer lives.
    /// Externs read the namespace fresh from this slot every time
    /// — that guarantees they pick up post-GC relocations (the GC
    /// rewrites the slot when the namespace moves).
    pub globals_ptr: *const dynobj::roots::RootSet,
    pub core_ns_slot: usize,
    /// Built-in heap-type-id → user-facing type-name symbol. Lets
    /// `extend-type __ReaderList …` register methods against the
    /// built-in `list` type, and lets dispatch on a list-typed
    /// receiver find them. Records carry their type-name in the
    /// instance itself (`record_type_name`); built-ins don't, so we
    /// look the name up here. Key is the raw `ObjTypeId.0` (a usize
    /// — `ObjTypeId` itself doesn't impl Hash).
    pub builtin_type_names: Mutex<HashMap<usize, u32>>,
    /// Pre-interned sym-ids for the protocol methods that the Rust
    /// side dispatches on (seq_iter, printer, apply). Computed once
    /// at engine init so the lock-free helpers in `protocol::` and
    /// `collections::seq_iter` don't need `host.sym` — the expander
    /// and compiler hold that lock when they call into them.
    pub seq_method_sym: u32,
    pub first_method_sym: u32,
    pub next_method_sym: u32,
    pub rest_method_sym: u32,
    pub count_method_sym: u32,
    /// Sym IDs for the protocols themselves, so the printer can ask
    /// `(satisfies? ISeq x)` / `(satisfies? IVector x)` / `(satisfies?
    /// IMap x)` to decide which surface syntax to emit.
    pub iseq_sym: u32,
    pub ivector_sym: u32,
    pub imap_sym: u32,
    /// Marker protocol claimed by code-as-data list types
    /// (PList/Cons/EmptyList). The expander uses this to decide
    /// whether a non-list seqable returned by a macro should be
    /// normalized into a built-in __ReaderList — vectors and maps
    /// implement ISeq too but are values, not code, so they pass
    /// through unchanged.
    pub ilist_sym: u32,
    /// `(type-name-sym, protocol-name-sym) → ()` registry of
    /// protocol-membership claims made via `extend-type`. Powers
    /// `satisfies?`. We record membership at extend time rather than
    /// inferring it from method-table entries because marker
    /// protocols (`IList`, `IRecord`, …) declare zero methods but
    /// still need to report `true` from `satisfies?`.
    pub protocol_membership: Mutex<HashSet<(u32, u32)>>,
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

/// Cheap accessor for the field-layout table installed on this thread.
/// Equivalent to `with_host(|h| h.layouts)` but reads nicer at every
/// `unsafe { p.add(layouts().list_first) }` site.
#[inline]
pub fn layouts() -> Layouts {
    with_host(|h| h.layouts)
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
    /// Read the current `clojure.core` namespace pointer. Reads
    /// from the engine's `globals` `RootSet` so a moving GC's
    /// in-place slot rewrite is observed.
    pub fn core_ns(&self) -> u64 {
        unsafe { (*self.globals_ptr).get(self.core_ns_slot) }
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
