//! Host-side stand-in for `clojure.lang.PersistentHashMap`.
//!
//! Original Java: `~/Documents/Code/open-source/clojure/src/jvm/clojure/lang/PersistentHashMap.java`
//!
//! Java's `PersistentHashMap` is a Bagwell HAMT — a 32-way branching trie with
//! `O(log32 n)` `assoc`/`get`/`without`. Compiler.java and the bootstrap loader
//! only need the public surface (`count`, `assoc`, `without`, `entry_at`,
//! `val_at`, iteration), and bootstrap-sized maps fit comfortably in a flat
//! pair vector, so we mirror the trick `PersistentVector` already uses: back
//! the type with `Arc<Vec<(Object, Object)>>` and do linear-scan lookups.
//!
//! Equality of keys uses [`super::object::object_equiv`] — the same
//! structural rule Clojure's `Util.equiv` implements, ported to Rust-side
//! `Object`s.
//!
//! When (if) we hit a workload where the constant factors matter, this gets
//! swapped for a real HAMT under the same public API.

use std::sync::Arc;

use super::object::{Object, object_equiv};

/// `clojure.lang.PersistentHashMap`. Insertion-order-preserving for now —
/// real Clojure doesn't preserve order for `PersistentHashMap`, but it does
/// for `PersistentArrayMap` (the small-map representation). Map literals
/// `{:a 1 :b 2}` flow through `PersistentArrayMap` until they exceed 16
/// entries, so preserving insertion order matches what the reader produces
/// today. We can switch to deterministic-but-not-insertion order once the
/// HAMT implementation lands.
#[derive(Debug)]
pub struct PersistentHashMap {
    /// Key/value pairs. Linear scan for now (bootstrap-sized).
    pairs: Arc<Vec<(Object, Object)>>,
}

impl PersistentHashMap {
    /// Java: `PersistentHashMap.EMPTY`.
    pub fn empty() -> Arc<Self> {
        Arc::new(PersistentHashMap {
            pairs: Arc::new(Vec::new()),
        })
    }

    /// Java: `PersistentHashMap.create(Object... init)` — build a map from
    /// an even-length sequence of `[k1, v1, k2, v2, ...]`. Later duplicate
    /// keys overwrite earlier ones (matches Clojure's reader behavior).
    pub fn create_pairs(init: Vec<(Object, Object)>) -> Arc<Self> {
        let mut out: Vec<(Object, Object)> = Vec::with_capacity(init.len());
        for (k, v) in init {
            // Replace existing if the key already exists; otherwise push.
            if let Some(slot) = out.iter_mut().find(|(ek, _)| object_equiv(ek, &k)) {
                slot.1 = v;
            } else {
                out.push((k, v));
            }
        }
        Arc::new(PersistentHashMap {
            pairs: Arc::new(out),
        })
    }

    /// Java: `PersistentHashMap.create(Object... init)` flat form.
    /// Panics on an odd-length `init` (Java throws IllegalArgumentException).
    pub fn create_flat(init: Vec<Object>) -> Arc<Self> {
        if init.len() % 2 != 0 {
            panic!(
                "clojure-jvm: IllegalArgumentException — \
                 No value supplied for key (PersistentHashMap.create of {} items)",
                init.len()
            );
        }
        let mut pairs: Vec<(Object, Object)> = Vec::with_capacity(init.len() / 2);
        let mut it = init.into_iter();
        while let (Some(k), Some(v)) = (it.next(), it.next()) {
            pairs.push((k, v));
        }
        Self::create_pairs(pairs)
    }

    /// Java: `int count()`.
    pub fn count(&self) -> i32 {
        self.pairs.len() as i32
    }

    /// Java: `IMapEntry entryAt(Object key)`. Returns the matching `(k, v)`
    /// pair, or `None` if the key isn't present.
    pub fn entry_at(&self, key: &Object) -> Option<(Object, Object)> {
        self.pairs
            .iter()
            .find(|(k, _)| object_equiv(k, key))
            .map(|(k, v)| (k.clone(), v.clone()))
    }

    /// Java: `Object valAt(Object key)`.
    pub fn val_at(&self, key: &Object) -> Object {
        self.entry_at(key).map(|(_, v)| v).unwrap_or(Object::Nil)
    }

    /// Java: `Object valAt(Object key, Object notFound)`.
    pub fn val_at_or(&self, key: &Object, not_found: Object) -> Object {
        self.entry_at(key).map(|(_, v)| v).unwrap_or(not_found)
    }

    /// Java: `IPersistentMap assoc(Object key, Object val)`. Returns a new
    /// map with `key -> val`; replaces an existing entry in place
    /// (preserving its position) or appends.
    pub fn assoc(self: &Arc<Self>, key: Object, val: Object) -> Arc<Self> {
        let mut new_pairs: Vec<(Object, Object)> = (*self.pairs).clone();
        if let Some(slot) = new_pairs.iter_mut().find(|(k, _)| object_equiv(k, &key)) {
            slot.1 = val;
        } else {
            new_pairs.push((key, val));
        }
        Arc::new(PersistentHashMap {
            pairs: Arc::new(new_pairs),
        })
    }

    /// Java: `IPersistentMap without(Object key)`.
    pub fn without(self: &Arc<Self>, key: &Object) -> Arc<Self> {
        let mut new_pairs: Vec<(Object, Object)> = (*self.pairs).clone();
        new_pairs.retain(|(k, _)| !object_equiv(k, key));
        Arc::new(PersistentHashMap {
            pairs: Arc::new(new_pairs),
        })
    }

    /// Java: `boolean containsKey(Object key)`.
    pub fn contains_key(&self, key: &Object) -> bool {
        self.pairs.iter().any(|(k, _)| object_equiv(k, key))
    }

    /// Iterator over `(k, v)` pairs in insertion order.
    pub fn iter(&self) -> impl Iterator<Item = (Object, Object)> + '_ {
        self.pairs.iter().map(|(k, v)| (k.clone(), v.clone()))
    }

    /// Structural equality (Java `equiv`). Two maps are equal iff they have
    /// the same count and every key in `self` maps to an equiv value in
    /// `other`.
    pub fn equiv(&self, other: &PersistentHashMap) -> bool {
        if self.count() != other.count() {
            return false;
        }
        for (k, v) in self.pairs.iter() {
            match other.entry_at(k) {
                Some((_, ov)) if object_equiv(v, &ov) => continue,
                _ => return false,
            }
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lang::keyword::Keyword;

    fn kw(name: &str) -> Object {
        Object::Keyword(Keyword::intern_ns_name(None, name))
    }

    #[test]
    fn empty_map_has_count_zero() {
        let m = PersistentHashMap::empty();
        assert_eq!(m.count(), 0);
        assert!(matches!(m.val_at(&kw("a")), Object::Nil));
    }

    #[test]
    fn create_flat_pairs_round_trip() {
        let m = PersistentHashMap::create_flat(vec![
            kw("a"),
            Object::Long(1),
            kw("b"),
            Object::Long(2),
        ]);
        assert_eq!(m.count(), 2);
        assert!(matches!(m.val_at(&kw("a")), Object::Long(1)));
        assert!(matches!(m.val_at(&kw("b")), Object::Long(2)));
        assert!(matches!(m.val_at(&kw("c")), Object::Nil));
    }

    #[test]
    fn duplicate_keys_keep_last_value() {
        let m = PersistentHashMap::create_flat(vec![
            kw("a"),
            Object::Long(1),
            kw("a"),
            Object::Long(99),
        ]);
        assert_eq!(m.count(), 1);
        assert!(matches!(m.val_at(&kw("a")), Object::Long(99)));
    }

    #[test]
    fn assoc_and_without_are_persistent() {
        let m0 = PersistentHashMap::empty();
        let m1 = m0.assoc(kw("a"), Object::Long(1));
        let m2 = m1.assoc(kw("b"), Object::Long(2));
        let m3 = m2.without(&kw("a"));
        assert_eq!(m0.count(), 0);
        assert_eq!(m1.count(), 1);
        assert_eq!(m2.count(), 2);
        assert_eq!(m3.count(), 1);
        assert!(matches!(m3.val_at(&kw("a")), Object::Nil));
        assert!(matches!(m3.val_at(&kw("b")), Object::Long(2)));
        // m2 is still intact.
        assert!(matches!(m2.val_at(&kw("a")), Object::Long(1)));
    }

    #[test]
    fn create_flat_odd_length_panics() {
        let result = std::panic::catch_unwind(|| {
            PersistentHashMap::create_flat(vec![kw("a")]);
        });
        assert!(result.is_err());
    }

    #[test]
    fn equiv_matches_regardless_of_insertion_order() {
        let a = PersistentHashMap::create_flat(vec![
            kw("a"),
            Object::Long(1),
            kw("b"),
            Object::Long(2),
        ]);
        let b = PersistentHashMap::create_flat(vec![
            kw("b"),
            Object::Long(2),
            kw("a"),
            Object::Long(1),
        ]);
        assert!(a.equiv(&b));
    }
}
