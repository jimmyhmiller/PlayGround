//! Port of `clojure.lang.Keyword`.
//!
//! Source: `~/Documents/Code/open-source/clojure/src/jvm/clojure/lang/Keyword.java`
//!
//! Java declaration:
//! ```text
//! public class Keyword implements IFn, Comparable, Named, Serializable, IHashEq
//! ```
//!
//! Java keeps a `ConcurrentHashMap<Symbol, WeakReference<Keyword>>` so keywords
//! are interned process-wide and GC'd when no longer reachable. We model the
//! same shape with a process-global `Mutex<HashMap<...>>` holding `Weak<Keyword>`.
//! Weak refs are pruned on `intern`.

use std::collections::HashMap;
use std::sync::atomic::{AtomicI32, Ordering};
use std::sync::{Arc, Mutex, Weak};


use super::object::Object;
use super::symbol::Symbol;

/// Process-wide intern table.
///
/// Java uses `ConcurrentHashMap` for lock-free reads. We use a `Mutex` for now;
/// can switch to a sharded / lock-free table later if it shows up in profiles.
///
/// Keyed by the `(ns, name)` pair the underlying `Symbol` holds. Using
/// `(Option<String>, String)` rather than the `Symbol` itself avoids a chicken-
/// and-egg with `Symbol`'s own potential interning.
static KEYWORD_TABLE: Mutex<
    Option<HashMap<(Option<String>, String), Weak<Keyword>>>,
> = Mutex::new(None);

fn with_table<R>(
    f: impl FnOnce(&mut HashMap<(Option<String>, String), Weak<Keyword>>) -> R,
) -> R {
    let mut guard = KEYWORD_TABLE.lock().unwrap();
    let t = guard.get_or_insert_with(HashMap::new);
    f(t)
}

/// `clojure.lang.Keyword`.
#[derive(Debug)]
pub struct Keyword {
    /// The underlying `Symbol`. Java: `public final Symbol sym;`
    pub sym: Arc<Symbol>,

    /// Cached hash. Java: `final int hasheq;` initialized as
    /// `sym.hasheq() + 0x9e3779b9`. Filled lazily once `Symbol::hasheq` works.
    /// `AtomicI32` (rather than `Cell`) so `Keyword` is `Sync` and can live in
    /// the process-global intern table.
    pub(crate) hasheq: AtomicI32,
}

impl Keyword {
    // ---- intern / find ---------------------------------------------------

    /// `Keyword.intern(Symbol sym)`.
    ///
    /// Returns the canonical keyword for `sym`, creating + interning one if
    /// needed.
    pub fn intern(sym: Arc<Symbol>) -> Arc<Self> {
        let key = (
            sym.get_namespace().map(|s| s.to_string()),
            sym.get_name().to_string(),
        );
        with_table(|t| {
            // Prune dead weak refs opportunistically (Java does this via
            // `Util.clearCache(rq, table)`).
            t.retain(|_, w| w.strong_count() > 0);

            if let Some(w) = t.get(&key) {
                if let Some(k) = w.upgrade() {
                    return k;
                }
            }
            let k = Arc::new(Keyword { sym, hasheq: AtomicI32::new(0) });
            t.insert(key, Arc::downgrade(&k));
            k
        })
    }

    /// `Keyword.intern(String ns, String name)`.
    pub fn intern_ns_name(ns: Option<&str>, name: &str) -> Arc<Self> {
        Self::intern(Symbol::intern_ns_name(ns, name))
    }

    /// `Keyword.intern(String nsname)`.
    pub fn intern_nsname(nsname: &str) -> Arc<Self> {
        Self::intern(Symbol::intern(nsname))
    }

    /// `Keyword.find(Symbol sym)`.
    pub fn find(sym: &Symbol) -> Option<Arc<Self>> {
        let key = (
            sym.get_namespace().map(|s| s.to_string()),
            sym.get_name().to_string(),
        );
        with_table(|t| t.get(&key).and_then(|w| w.upgrade()))
    }

    // ---- Named impl ------------------------------------------------------

    pub fn get_namespace(&self) -> Option<&str> { self.sym.get_namespace() }
    pub fn get_name(&self) -> &str { self.sym.get_name() }

    // ---- IHashEq / hashCode ---------------------------------------------

    pub fn hasheq(&self) -> i32 {
        let cur = self.hasheq.load(Ordering::Relaxed);
        if cur == 0 {
            crate::unimplemented_port!(
                "Keyword.hasheq",
                "depends on Symbol.hasheq + 0x9e3779b9"
            )
        }
        cur
    }

    // ---- IFn invoke ------------------------------------------------------

    pub fn invoke0(&self) -> Object { self.throw_arity(0) }

    pub fn invoke1(&self, obj: Object) -> Object {
        // Java: if obj instanceof ILookup → ((ILookup)obj).valAt(this);
        //       else RT.get(obj, this)
        let _ = obj;
        crate::unimplemented_port!("Keyword.invoke(Object)", "needs ILookup + RT.get")
    }

    pub fn invoke2(&self, obj: Object, not_found: Object) -> Object {
        let _ = (obj, not_found);
        crate::unimplemented_port!(
            "Keyword.invoke(Object,Object)",
            "needs ILookup + RT.get"
        )
    }

    fn throw_arity(&self, n: i32) -> ! {
        // Java throws ArityException(n, toString()).
        panic!(
            "clojure-jvm: ArityException({}) — wrong number of args for keyword {}",
            n,
            self
        );
    }
}

// `Keyword.toString` is `:" + sym`.
impl std::fmt::Display for Keyword {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, ":{}", self.sym.to_string())
    }
}

// Identity-based equality and ordering since keywords are interned.
impl PartialEq for Keyword {
    fn eq(&self, other: &Self) -> bool { Arc::ptr_eq(&self.sym, &other.sym) }
}
impl Eq for Keyword {}

impl Ord for Keyword {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering { self.sym.cmp(&other.sym) }
}
impl PartialOrd for Keyword {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl std::hash::Hash for Keyword {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.sym.hash(state);
    }
}
