//! Port of `clojure.lang.Var`.
//!
//! Source: `~/Documents/Code/open-source/clojure/src/jvm/clojure/lang/Var.java`
//!
//! Java declaration:
//! ```text
//! public final class Var extends ARef implements IFn, IRef, Settable, Serializable
//! ```
//!
//! This is a foundation skeleton. We model:
//!   * the per-Var `root` value (`Object` slot)
//!   * the `dynamic` flag, `threadBound` flag
//!   * inner `TBox`, `Unbound`, `Frame` types
//!   * the `dvals` ThreadLocal frame stack (as a Rust `thread_local!`)
//!   * static factory methods Compiler.java actually uses:
//!     `create()`, `create(Object root)`, `intern(Namespace, Symbol)`,
//!     `intern(Namespace, Symbol, Object)`,
//!     `intern(Namespace, Symbol, Object, boolean)`
//!
//! Concurrency-control parts (pushThreadBindings, alter root, watches, etc.)
//! are stubbed until needed by Compiler.java's main paths.

use std::cell::RefCell;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicI64, Ordering};
use std::sync::{Mutex, RwLock};

use super::namespace::Namespace;
use super::object::Object;
use super::symbol::Symbol;

// ---------- inner types -------------------------------------------------

/// Java `Var.TBox`: per-thread mutable cell holding the dynamic binding value.
#[derive(Debug)]
pub struct TBox {
    pub val: Mutex<Object>,
    pub thread_id: std::thread::ThreadId,
}

impl TBox {
    pub fn new(val: Object) -> Self {
        TBox {
            val: Mutex::new(val),
            thread_id: std::thread::current().id(),
        }
    }
}

/// Java `Var.Unbound`: the sentinel value sitting in `root` when a var has been
/// declared but has no value yet. Calling it as a fn throws ArityException.
#[derive(Debug)]
pub struct Unbound {
    pub v: Arc<Var>,
}

impl Unbound {
    pub fn to_string(&self) -> String {
        format!("Unbound: {}", self.v)
    }

    pub fn throw_arity(&self, _n: i32) -> ! {
        panic!(
            "clojure-jvm: IllegalStateException — Attempting to call unbound fn: {}",
            self.v
        );
    }
}

/// Java `Var.Frame`: one node of the per-thread dynamic-binding chain.
/// `bindings` is `Var → TBox`. The chain is single-linked; pushing a binding
/// produces a new `Frame` whose `prev` is the current top.
#[derive(Debug, Clone)]
pub struct Frame {
    // Var → TBox. Will become an `Associative` once that's ported; for now
    // a Vec of pairs keeps it explicit + small.
    pub bindings: Vec<(Arc<Var>, Arc<TBox>)>,
    pub prev: Option<Arc<Frame>>,
}

impl Frame {
    pub fn top() -> Arc<Frame> {
        Arc::new(Frame {
            bindings: Vec::new(),
            prev: None,
        })
    }
}

thread_local! {
    /// Java: `static final ThreadLocal<Frame> dvals`. One dynamic-binding
    /// stack per thread, initialized to TOP.
    static DVALS: RefCell<Arc<Frame>> = RefCell::new(Frame::top());
}

// ---------- the Var itself ---------------------------------------------

/// Where a Var's root value currently lives.
///
/// A Var is an indirection cell, and its root value must be a genuine GC
/// root whenever it is a heap pointer. We achieve that by storing
/// NanBox-representable values (which includes every GC-heap pointer) in the
/// Var's slot in the global [`VAR_ROOTS`](crate::lang::var_roots::VAR_ROOTS)
/// table, which is registered with the collector as a `RootSource`. Values
/// with no NanBox representation (e.g. `Object::Namespace`) are never heap
/// pointers and so cannot dangle; those stay Rust-side.
#[derive(Debug)]
enum VarRoot {
    /// Value is the NanBox in `VAR_ROOTS[slot_index]`. GC-rooted: the
    /// collector forwards it in place when the object moves.
    Slot,
    /// A Rust-side value with no NanBox representation. Not a heap pointer.
    Object(Object),
}

/// `clojure.lang.Var`. Variable holding a root value, optional dynamic
/// per-thread bindings, and metadata.
#[derive(Debug)]
pub struct Var {
    /// Java: `volatile Object root;`. See [`VarRoot`]: heap values live in
    /// the GC-rooted slot table, Rust-only values live inline here.
    root: RwLock<VarRoot>,
    /// Stable index into the global `VAR_ROOTS` table, allocated once at
    /// creation and never reused. Holds this Var's root value as a NanBox
    /// whenever the root is [`VarRoot::Slot`]-backed.
    slot_index: usize,
    /// Java: `volatile boolean dynamic = false;`
    pub(crate) dynamic: AtomicBool,
    /// Java: `transient final AtomicBoolean threadBound;`
    pub(crate) thread_bound: AtomicBool,
    /// Java: `public final Symbol sym;`
    pub sym: Option<Arc<Symbol>>,
    /// Java: `public final Namespace ns;`
    pub ns: Option<Arc<Namespace>>,
    /// Java: stored as `:macro true` in the Var's metadata map (Var.meta).
    /// We model it as a dedicated boolean since the compiler reads it on
    /// every analyze and we don't yet have IPersistentMap on Vars.
    /// Set via `setMacro()`; checked by `analyze_seq` to decide whether
    /// to macroexpand a `(name args...)` call.
    pub(crate) is_macro: AtomicBool,
}

/// Java: `static public volatile int rev = 0;` — bumped on Var changes; readers
/// recheck their caches.
pub static REV: AtomicI64 = AtomicI64::new(0);

impl Var {
    // ---- factories ------------------------------------------------------

    /// `Var.create()` — `create((Object) null)` for a bare unbound Var with no
    /// namespace.
    pub fn create() -> Arc<Self> {
        Self::create_with_root(Object::Nil)
    }

    /// `Var.create(Object root)` — bare Var with a root value.
    pub fn create_with_root(root: Object) -> Arc<Self> {
        let v = Arc::new(Var {
            root: RwLock::new(VarRoot::Object(Object::Nil)),
            slot_index: crate::lang::var_roots::VAR_ROOTS.alloc_slot(),
            dynamic: AtomicBool::new(false),
            thread_bound: AtomicBool::new(false),
            sym: None,
            ns: None,
            is_macro: AtomicBool::new(false),
        });
        v.bind_root(root);
        v
    }

    /// Construct a Var tagged with its owning namespace + symbol but without
    /// a root value. Called from `Namespace::intern`.
    pub fn create_in_ns(ns: Arc<Namespace>, sym: Arc<Symbol>) -> Arc<Self> {
        Arc::new(Var {
            root: RwLock::new(VarRoot::Object(Object::Nil)),
            slot_index: crate::lang::var_roots::VAR_ROOTS.alloc_slot(),
            dynamic: AtomicBool::new(false),
            thread_bound: AtomicBool::new(false),
            sym: Some(sym),
            ns: Some(ns),
            is_macro: AtomicBool::new(false),
        })
    }

    /// Java: `Var.setMacro()` — flags this var as a macro. Called from
    /// `def` analysis when the symbol carries `:macro` metadata.
    pub fn set_macro(&self) {
        self.is_macro.store(true, Ordering::Release);
    }

    /// Java: `Var.isMacro()`.
    pub fn is_macro(&self) -> bool {
        self.is_macro.load(Ordering::Acquire)
    }

    /// `Var.intern(Namespace, Symbol)` — interns (or finds) a Var in the
    /// given namespace.
    pub fn intern_sym(ns: Arc<Namespace>, sym: Arc<Symbol>) -> Arc<Self> {
        ns.intern(sym)
    }

    /// `Var.intern(Namespace, Symbol, Object)` — interns and sets the root.
    pub fn intern_sym_root(ns: Arc<Namespace>, sym: Arc<Symbol>, root: Object) -> Arc<Self> {
        let v = ns.intern(sym);
        v.bind_root(root);
        v
    }

    /// `Var.intern(Namespace, Symbol, Object, boolean replaceRoot)`. If
    /// `replace_root` is `false`, only sets the root when the Var has no
    /// existing root binding.
    pub fn intern_sym_root_replace(
        ns: Arc<Namespace>,
        sym: Arc<Symbol>,
        root: Object,
        replace_root: bool,
    ) -> Arc<Self> {
        let v = ns.intern(sym);
        if replace_root || v.is_unbound() {
            v.bind_root(root);
        }
        v
    }

    /// True if this Var has never been given a root value (still nil).
    fn is_unbound(&self) -> bool {
        match &*self.root.read().unwrap() {
            VarRoot::Object(Object::Nil) => true,
            VarRoot::Slot => {
                crate::lang::var_roots::VAR_ROOTS.get(self.slot_index)
                    == crate::runtime::nanbox_nil()
            }
            _ => false,
        }
    }

    /// Java: `Var.bindRoot(Object root)`. Sets the root value
    /// (`alterRoot`/`bindRoot` collapse here — we don't track watches).
    ///
    /// NanBox-representable values (all GC-heap pointers, plus immediates)
    /// are stored in the Var's GC-rooted slot so the collector forwards
    /// them. Values with no NanBox form (e.g. a Namespace) are kept
    /// Rust-side; those are never heap pointers and cannot dangle.
    pub fn bind_root(&self, root: Object) {
        match crate::runtime::try_object_to_nanbox(&root) {
            Some(bits) => {
                crate::lang::var_roots::VAR_ROOTS.set(self.slot_index, bits);
                *self.root.write().unwrap() = VarRoot::Slot;
            }
            None => {
                // Clear the slot so the GC never sees a stale pointer there.
                crate::lang::var_roots::VAR_ROOTS
                    .set(self.slot_index, crate::runtime::nanbox_nil());
                *self.root.write().unwrap() = VarRoot::Object(root);
            }
        }
        REV.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    /// Bind the root directly from NanBox bits — the JIT path
    /// (`cljvm_var_bind_root`). The bits go straight into the GC-rooted
    /// slot, so a heap pointer is rooted with no `Object` round-trip.
    pub fn bind_root_bits(&self, bits: u64) {
        crate::lang::var_roots::VAR_ROOTS.set(self.slot_index, bits);
        *self.root.write().unwrap() = VarRoot::Slot;
        REV.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    /// VAR_ROOTS slot index (diagnostics).
    pub fn slot_index(&self) -> usize {
        self.slot_index
    }

    /// Read the current root value as NanBox bits — the JIT path
    /// (`cljvm_var_deref`). Reads the live (forwarded) slot bits directly
    /// for Slot-backed roots; falls back to encoding the Rust-side value
    /// otherwise. Honors a dynamic thread binding if one is in scope.
    pub fn deref_bits(&self) -> u64 {
        if let Some(b) = self.get_thread_binding() {
            return crate::runtime::object_to_nanbox(&b.val.lock().unwrap());
        }
        match &*self.root.read().unwrap() {
            VarRoot::Slot => crate::lang::var_roots::VAR_ROOTS.get(self.slot_index),
            VarRoot::Object(o) => crate::runtime::object_to_nanbox(o),
        }
    }

    // ---- dynamic flag ---------------------------------------------------

    /// `setDynamic()` — fluent setter, returns `this` (we return the `Rc`).
    pub fn set_dynamic(self: Arc<Self>) -> Arc<Self> {
        self.dynamic
            .store(true, std::sync::atomic::Ordering::Relaxed);
        self
    }

    pub fn set_dynamic_bool(self: Arc<Self>, b: bool) -> Arc<Self> {
        self.dynamic.store(b, std::sync::atomic::Ordering::Relaxed);
        self
    }

    pub fn is_dynamic(&self) -> bool {
        self.dynamic.load(std::sync::atomic::Ordering::Relaxed)
    }

    // ---- deref / set ----------------------------------------------------

    /// `deref()` — read the current value (thread binding if present, else root).
    pub fn deref(&self) -> Object {
        if let Some(b) = self.get_thread_binding() {
            return b.val.lock().unwrap().clone();
        }
        match &*self.root.read().unwrap() {
            VarRoot::Slot => crate::runtime::nanbox_to_object(
                crate::lang::var_roots::VAR_ROOTS.get(self.slot_index),
            ),
            VarRoot::Object(o) => o.clone(),
        }
    }

    /// Java: `getThreadBinding()`. Walks the current thread's `Frame` chain
    /// looking for a TBox bound to this Var.
    pub fn get_thread_binding(self: &Self) -> Option<Arc<TBox>> {
        DVALS.with(|cell| {
            let mut frame = Some(cell.borrow().clone());
            while let Some(f) = frame {
                for (var, tbox) in &f.bindings {
                    if std::ptr::eq(var.as_ref(), self) {
                        return Some(tbox.clone());
                    }
                }
                frame = f.prev.clone();
            }
            None
        })
    }

    // ---- thread-binding frame helpers (static) --------------------------

    pub fn get_thread_binding_frame() -> Arc<Frame> {
        DVALS.with(|cell| cell.borrow().clone())
    }

    pub fn clone_thread_binding_frame() -> Arc<Frame> {
        DVALS.with(|cell| Arc::new((**cell.borrow()).clone()))
    }

    pub fn reset_thread_binding_frame(frame: Arc<Frame>) {
        DVALS.with(|cell| *cell.borrow_mut() = frame);
    }

    /// Java: `Var.pushThreadBindings(Associative bindings)`. We pass a Vec of
    /// `(Var, Object)` pairs rather than an `Associative` since IPersistentMap
    /// isn't ported yet — semantically equivalent for the compiler's use.
    ///
    /// Each pair becomes a TBox on a fresh Frame whose `prev` is the current
    /// top of `DVALS`. Bindings are visible only to this thread.
    pub fn push_thread_bindings(bindings: Vec<(Arc<Var>, Object)>) {
        DVALS.with(|cell| {
            let prev = cell.borrow().clone();
            let tboxes: Vec<(Arc<Var>, Arc<TBox>)> = bindings
                .into_iter()
                .map(|(v, val)| {
                    // Mirror Java: setting threadBound flag so readers know
                    // to consult the frame chain.
                    v.thread_bound
                        .store(true, std::sync::atomic::Ordering::Relaxed);
                    (v, Arc::new(TBox::new(val)))
                })
                .collect();
            let new_frame = Arc::new(Frame {
                bindings: tboxes,
                prev: Some(prev),
            });
            *cell.borrow_mut() = new_frame;
        });
    }

    /// Java: `Var.popThreadBindings()`. Pops back to `Frame.prev`. Panics if
    /// called when the chain is already at TOP.
    pub fn pop_thread_bindings() {
        DVALS.with(|cell| {
            let cur = cell.borrow().clone();
            match cur.prev.clone() {
                Some(prev) => *cell.borrow_mut() = prev,
                None => panic!(
                    "clojure-jvm: IllegalStateException — popThreadBindings without matching push"
                ),
            }
        });
    }

    /// Java: `Var.set(Object val)`. If this var has a thread binding, mutate
    /// the bound TBox; otherwise panic (the compiler relies on the in-binding
    /// path — `LOCAL_ENV.set(...)`, `NEXT_LOCAL_NUM.set(...)` — and never
    /// mutates a root from within analyze).
    pub fn set_value(&self, val: Object) {
        if let Some(b) = self.get_thread_binding() {
            *b.val.lock().unwrap() = val;
            return;
        }
        panic!(
            "clojure-jvm: IllegalStateException — Can't change/establish root binding of: {} \
             with set (no thread binding)",
            self
        );
    }
}

impl std::fmt::Display for Var {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match (&self.ns, &self.sym) {
            (Some(ns), Some(s)) => write!(f, "#'{}/{}", ns, s.get_name()),
            (None, Some(s)) => write!(f, "#'{}", s.get_name()),
            _ => write!(f, "#<Var: --unnamed-->"),
        }
    }
}
