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
use std::sync::atomic::{AtomicBool, AtomicI64};
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
        TBox { val: Mutex::new(val), thread_id: std::thread::current().id() }
    }
}

/// Java `Var.Unbound`: the sentinel value sitting in `root` when a var has been
/// declared but has no value yet. Calling it as a fn throws ArityException.
#[derive(Debug)]
pub struct Unbound {
    pub v: Arc<Var>,
}

impl Unbound {
    pub fn to_string(&self) -> String { format!("Unbound: {}", self.v) }

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
        Arc::new(Frame { bindings: Vec::new(), prev: None })
    }
}

thread_local! {
    /// Java: `static final ThreadLocal<Frame> dvals`. One dynamic-binding
    /// stack per thread, initialized to TOP.
    static DVALS: RefCell<Arc<Frame>> = RefCell::new(Frame::top());
}

// ---------- the Var itself ---------------------------------------------

/// `clojure.lang.Var`. Variable holding a root value, optional dynamic
/// per-thread bindings, and metadata.
#[derive(Debug)]
pub struct Var {
    /// Java: `volatile Object root;`
    pub(crate) root: RwLock<Object>,
    /// Java: `volatile boolean dynamic = false;`
    pub(crate) dynamic: AtomicBool,
    /// Java: `transient final AtomicBoolean threadBound;`
    pub(crate) thread_bound: AtomicBool,
    /// Java: `public final Symbol sym;`
    pub sym: Option<Arc<Symbol>>,
    /// Java: `public final Namespace ns;`
    pub ns: Option<Arc<Namespace>>,
}

/// Java: `static public volatile int rev = 0;` — bumped on Var changes; readers
/// recheck their caches.
pub static REV: AtomicI64 = AtomicI64::new(0);

impl Var {
    // ---- factories ------------------------------------------------------

    /// `Var.create()` — `create((Object) null)` for a bare unbound Var with no
    /// namespace.
    pub fn create() -> Arc<Self> { Self::create_with_root(Object::Nil) }

    /// `Var.create(Object root)` — bare Var with a root value.
    pub fn create_with_root(root: Object) -> Arc<Self> {
        let v = Arc::new(Var {
            root: RwLock::new(root.clone()),
            dynamic: AtomicBool::new(false),
            thread_bound: AtomicBool::new(false),
            sym: None,
            ns: None,
        });
        // Java sets root via `setRoot` after construction so it can install
        // `Unbound` when root is `null`. We mirror that — but until we have an
        // `Unbound` install path wired up (needs IFn), we just keep the value
        // (or Nil sentinel) we were handed.
        let _ = root;
        v
    }

    /// Construct a Var tagged with its owning namespace + symbol but without
    /// a root value. Called from `Namespace::intern`.
    pub fn create_in_ns(ns: Arc<Namespace>, sym: Arc<Symbol>) -> Arc<Self> {
        Arc::new(Var {
            root: RwLock::new(Object::Nil),
            dynamic: AtomicBool::new(false),
            thread_bound: AtomicBool::new(false),
            sym: Some(sym),
            ns: Some(ns),
        })
    }

    /// `Var.intern(Namespace, Symbol)` — interns (or finds) a Var in the
    /// given namespace.
    pub fn intern_sym(ns: Arc<Namespace>, sym: Arc<Symbol>) -> Arc<Self> {
        ns.intern(sym)
    }

    /// `Var.intern(Namespace, Symbol, Object)` — interns and sets the root.
    pub fn intern_sym_root(
        ns: Arc<Namespace>,
        sym: Arc<Symbol>,
        root: Object,
    ) -> Arc<Self> {
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
        if replace_root || matches!(*v.root.read().unwrap(), Object::Nil) {
            v.bind_root(root);
        }
        v
    }

    /// Java: `Var.bindRoot(Object root)`. Sets the root value
    /// (`alterRoot`/`bindRoot` collapse here — we don't track watches).
    pub fn bind_root(&self, root: Object) {
        *self.root.write().unwrap() = root;
        REV.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    // ---- dynamic flag ---------------------------------------------------

    /// `setDynamic()` — fluent setter, returns `this` (we return the `Rc`).
    pub fn set_dynamic(self: Arc<Self>) -> Arc<Self> {
        self.dynamic.store(true, std::sync::atomic::Ordering::Relaxed);
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
        self.root.read().unwrap().clone()
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
                    v.thread_bound.store(true, std::sync::atomic::Ordering::Relaxed);
                    (v, Arc::new(TBox::new(val)))
                })
                .collect();
            let new_frame = Arc::new(Frame { bindings: tboxes, prev: Some(prev) });
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
