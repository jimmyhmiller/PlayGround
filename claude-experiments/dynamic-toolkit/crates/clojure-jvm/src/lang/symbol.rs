//! Port of `clojure.lang.Symbol`.
//!
//! Source: `~/Documents/Code/open-source/clojure/src/jvm/clojure/lang/Symbol.java`
//!
//! Java declaration:
//! ```text
//! public class Symbol extends AFn implements IObj, Comparable, Named,
//!                                              Serializable, IHashEq
//! ```
//!
//! We keep the same field layout (`ns`, `name`, `_meta`, cached `_hasheq`) and
//! the static `intern` factories. Method bodies that reach into `Util`,
//! `Murmur3`, `RT.get`, or `IPersistentMap` will stub-panic until those are
//! ported.

use std::sync::Arc;
use std::sync::atomic::{AtomicI32, Ordering};


/// `clojure.lang.Symbol`.
#[derive(Debug)]
pub struct Symbol {
    /// Namespace part. `None` matches Java `null`.
    pub(crate) ns: Option<Arc<String>>,

    /// Bare name. Always present.
    pub(crate) name: Arc<String>,

    /// Cached `hasheq`. Java has `private int _hasheq` mutated without sync;
    /// `AtomicI32` keeps the cache lazy while satisfying Rust's `Sync` rules
    /// (Symbols live in process-global tables in our port).
    pub(crate) _hasheq: AtomicI32,

    /// Symbol metadata (`IPersistentMap` in Java). Stubbed until
    /// `IPersistentMap` is ported.
    pub(crate) _meta: Option<()>,
}

impl Symbol {
    // ---- factory thunks (Java `create` / `intern`) -----------------------

    /// `Symbol.create(String ns, String name)`.
    pub fn create_ns_name(ns: Option<&str>, name: &str) -> Arc<Self> {
        Self::intern_ns_name(ns, name)
    }

    /// `Symbol.create(String nsname)`.
    pub fn create(nsname: &str) -> Arc<Self> { Self::intern(nsname) }

    /// `Symbol.intern(String ns, String name)`.
    pub fn intern_ns_name(ns: Option<&str>, name: &str) -> Arc<Self> {
        Arc::new(Symbol {
            ns: ns.map(|s| Arc::new(s.to_string())),
            name: Arc::new(name.to_string()),
            _hasheq: AtomicI32::new(0),
            _meta: None,
        })
    }

    /// `Symbol.intern(String nsname)`. Splits on the first `'/'`, except for
    /// the bare `/` symbol.
    pub fn intern(nsname: &str) -> Arc<Self> {
        match nsname.find('/') {
            None => Self::intern_ns_name(None, nsname),
            Some(_) if nsname == "/" => Self::intern_ns_name(None, nsname),
            Some(i) => {
                Self::intern_ns_name(Some(&nsname[..i]), &nsname[i + 1..])
            }
        }
    }

    // ---- Named impl ------------------------------------------------------

    pub fn get_namespace(&self) -> Option<&str> {
        self.ns.as_deref().map(|s| s.as_str())
    }

    pub fn get_name(&self) -> &str { &self.name }

    // ---- toString --------------------------------------------------------

    pub fn to_string(&self) -> String {
        match &self.ns {
            Some(ns) => format!("{}/{}", ns, self.name),
            None => (*self.name).clone(),
        }
    }

    // ---- IHashEq impl (stub — needs Murmur3 + Util.hashCombine) ----------

    pub fn hasheq(&self) -> i32 {
        let cur = self._hasheq.load(Ordering::Relaxed);
        if cur == 0 {
            crate::unimplemented_port!("Symbol.hasheq", "needs Murmur3 + Util.hashCombine")
        }
        cur
    }

    // ---- IObj / IMeta (stub — needs IPersistentMap) ----------------------

    pub fn with_meta(self: &Arc<Self>, _meta: ()) -> Arc<Self> {
        crate::unimplemented_port!("Symbol.withMeta", "needs IPersistentMap")
    }

    pub fn meta(&self) -> Option<()> { self._meta }

    // ---- AFn invoke arity-1 / arity-2 (stub — needs RT.get) --------------

    pub fn invoke1(&self, _obj: super::object::Object) -> super::object::Object {
        crate::unimplemented_port!("Symbol.invoke(Object)", "needs RT.get")
    }

    pub fn invoke2(
        &self,
        _obj: super::object::Object,
        _not_found: super::object::Object,
    ) -> super::object::Object {
        crate::unimplemented_port!("Symbol.invoke(Object,Object)", "needs RT.get")
    }
}

// Java `equals`: structural over (ns, name).
impl PartialEq for Symbol {
    fn eq(&self, other: &Self) -> bool {
        self.ns.as_deref().map(|s| s.as_str())
            == other.ns.as_deref().map(|s| s.as_str())
            && *self.name == *other.name
    }
}
impl Eq for Symbol {}

// Java `compareTo` ordering.
impl Ord for Symbol {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        use std::cmp::Ordering::*;
        if self == other {
            return Equal;
        }
        match (&self.ns, &other.ns) {
            (None, Some(_)) => Less,
            (Some(_), None) => Greater,
            (Some(a), Some(b)) => match a.cmp(b) {
                Equal => self.name.cmp(&other.name),
                ord => ord,
            },
            (None, None) => self.name.cmp(&other.name),
        }
    }
}
impl PartialOrd for Symbol {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl std::hash::Hash for Symbol {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        // Java: Util.hashCombine(name.hashCode(), Util.hash(ns))
        self.name.hash(state);
        if let Some(ns) = &self.ns {
            ns.hash(state);
        }
    }
}
