//! Port of `clojure.lang.Namespace`.
//!
//! Source: `~/Documents/Code/open-source/clojure/src/jvm/clojure/lang/Namespace.java`
//!
//! Java has `mappings`, `aliases`, and `imports` as `AtomicReference<IPersistentMap>`s.
//! We model `mappings` (Symbol → Var) as `Mutex<HashMap<…>>`; aliases / imports
//! are stubbed until needed.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use super::object::Object;
use super::symbol::Symbol;
use super::var::Var;

/// `clojure.lang.Namespace`. Owns the symbol → Var binding table for one
/// Clojure namespace.
#[derive(Debug)]
pub struct Namespace {
    pub name: Arc<Symbol>,
    /// Java: `AtomicReference<IPersistentMap> mappings`. Maps interned syms
    /// to either Vars (for `def`s/`refer`s) or Classes (for `import`s).
    pub mappings: Mutex<HashMap<Arc<Symbol>, Object>>,
    /// Java: `AtomicReference<IPersistentMap> aliases`. Maps short ns names
    /// to fully-qualified Namespaces. Stubbed.
    pub aliases: Mutex<HashMap<Arc<Symbol>, Arc<Namespace>>>,
}

/// Process-wide registry. Java: `static final ConcurrentHashMap<Symbol, Namespace> namespaces`.
static NAMESPACES: Mutex<Option<HashMap<(Option<String>, String), Arc<Namespace>>>> =
    Mutex::new(None);

fn with_registry<R>(
    f: impl FnOnce(&mut HashMap<(Option<String>, String), Arc<Namespace>>) -> R,
) -> R {
    let mut g = NAMESPACES.lock().unwrap();
    let t = g.get_or_insert_with(HashMap::new);
    f(t)
}

impl Namespace {
    /// `Namespace.findOrCreate(Symbol name)`.
    pub fn find_or_create(name: Arc<Symbol>) -> Arc<Namespace> {
        let key = (
            name.get_namespace().map(|s| s.to_string()),
            name.get_name().to_string(),
        );
        with_registry(|t| {
            if let Some(ns) = t.get(&key) {
                return ns.clone();
            }
            let ns = Arc::new(Namespace {
                name: name.clone(),
                mappings: Mutex::new(HashMap::new()),
                aliases: Mutex::new(HashMap::new()),
            });
            t.insert(key, ns.clone());
            ns
        })
    }

    /// `Namespace.find(Symbol name)`.
    pub fn find(name: &Symbol) -> Option<Arc<Namespace>> {
        let key = (
            name.get_namespace().map(|s| s.to_string()),
            name.get_name().to_string(),
        );
        with_registry(|t| t.get(&key).cloned())
    }

    /// `Namespace.intern(Symbol sym)`. Looks up the var by `sym` in this ns's
    /// `mappings`; if absent, creates a fresh `Var(ns, sym)` and registers it.
    ///
    /// Java: rejects qualified syms (`sym.ns != null`) with an exception.
    pub fn intern(self: &Arc<Self>, sym: Arc<Symbol>) -> Arc<Var> {
        if sym.get_namespace().is_some() {
            panic!(
                "clojure-jvm: IllegalStateException — Can't intern namespace-qualified symbol: {}",
                sym.get_name()
            );
        }
        let mut map = self.mappings.lock().unwrap();
        if let Some(existing) = map.get(&sym).cloned() {
            if let Object::Var(v) = existing {
                if Arc::ptr_eq(&v.ns.as_ref().unwrap(), self) {
                    return v;
                }
            }
            // Java: if the existing mapping is a Var from another ns OR a
            // Class, throw / replace. For our minimal port we just replace.
        }
        let v = Var::create_in_ns(self.clone(), sym.clone());
        map.insert(sym, Object::Var(v.clone()));
        v
    }

    /// `Namespace.findInternedVar(Symbol sym)`. Returns the Var for `sym` if
    /// it was `intern`'d here (not just referred from another ns).
    pub fn find_interned_var(&self, sym: &Symbol) -> Option<Arc<Var>> {
        let map = self.mappings.lock().unwrap();
        for (k, v) in map.iter() {
            if **k == *sym {
                if let Object::Var(var) = v {
                    return Some(var.clone());
                }
            }
        }
        None
    }

    /// `Namespace.getMapping(Symbol name)`. Returns whatever the symbol is
    /// mapped to (`Var`, `Class`, or absent).
    pub fn get_mapping(&self, sym: &Symbol) -> Option<Object> {
        let map = self.mappings.lock().unwrap();
        for (k, v) in map.iter() {
            if **k == *sym {
                return Some(v.clone());
            }
        }
        None
    }

    /// `Namespace.refer(Symbol sym, Var var)`. Adds a mapping `sym -> var`
    /// into this namespace. Backs the `(. *ns* (refer sym v))` call at the
    /// bottom of the `refer` function (which the `ns` macro emits as
    /// `(refer 'clojure.core)`). A genuine clash with a different existing
    /// var is replaced (Java warns/throws depending on context; we keep it
    /// simple), but a re-refer of the identical var is idempotent.
    pub fn refer(self: &Arc<Self>, sym: Arc<Symbol>, var: Arc<Var>) {
        let mut map = self.mappings.lock().unwrap();
        // Replace any structurally-equal existing key (the HashMap hashes on
        // Arc identity, so locate the slot by structural symbol equality).
        let existing_key = map
            .iter()
            .find(|(k, _)| ***k == *sym)
            .map(|(k, _)| k.clone());
        if let Some(k) = existing_key {
            map.insert(k, Object::Var(var));
        } else {
            map.insert(sym, Object::Var(var));
        }
    }

    /// Snapshot of this namespace's mappings as `(Symbol, Object)` pairs.
    /// Backs the `Namespace.getMappings` host method (used by `ns-map` /
    /// `ns-publics`).
    pub fn mappings_snapshot(&self) -> Vec<(Arc<Symbol>, Object)> {
        let map = self.mappings.lock().unwrap();
        map.iter().map(|(k, v)| (k.clone(), v.clone())).collect()
    }

    /// `Namespace.addAlias(Symbol alias, Namespace ns)`. Registers `alias`
    /// → `ns` in this namespace's aliases map. Used by `(ns foo (:require
    /// [bar :as b]))` and top-level `(alias 'b 'bar)`.
    pub fn add_alias(&self, alias: Arc<Symbol>, ns: Arc<Namespace>) {
        let mut a = self.aliases.lock().unwrap();
        a.insert(alias, ns);
    }

    /// `Namespace.lookupAlias(Symbol alias)`. Returns the namespace that
    /// `alias` was registered to point at, or `None`.
    pub fn lookup_alias(&self, alias: &Symbol) -> Option<Arc<Namespace>> {
        let a = self.aliases.lock().unwrap();
        for (k, v) in a.iter() {
            if **k == *alias {
                return Some(v.clone());
            }
        }
        None
    }
}

impl std::fmt::Display for Namespace {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name.get_name())
    }
}
