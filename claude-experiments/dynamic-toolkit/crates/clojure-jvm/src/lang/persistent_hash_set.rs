//! Host-side stand-in for `clojure.lang.PersistentHashSet`.
//!
//! Original Java: `~/Documents/Code/open-source/clojure/src/jvm/clojure/lang/PersistentHashSet.java`
//!
//! Java's `PersistentHashSet` is a thin wrapper over `PersistentHashMap`
//! using the set's elements as keys and a dummy value (`null` / itself).
//! We mirror that: the set is an `Arc<PersistentHashMap>` keyed by the
//! element, with `Object::Nil` as the value.

use std::sync::Arc;

use super::object::Object;
use super::persistent_hash_map::PersistentHashMap;

#[derive(Debug)]
pub struct PersistentHashSet {
    /// Underlying map; keys are the set elements, values are always Nil.
    map: Arc<PersistentHashMap>,
}

impl PersistentHashSet {
    /// Java: `PersistentHashSet.EMPTY`.
    pub fn empty() -> Arc<Self> {
        Arc::new(PersistentHashSet {
            map: PersistentHashMap::empty(),
        })
    }

    /// Java: `PersistentHashSet.create(Object... init)`.
    pub fn create(items: Vec<Object>) -> Arc<Self> {
        let pairs: Vec<(Object, Object)> = items.into_iter().map(|x| (x, Object::Nil)).collect();
        Arc::new(PersistentHashSet {
            map: PersistentHashMap::create_pairs(pairs),
        })
    }

    /// Java: `int count()`.
    pub fn count(&self) -> i32 {
        self.map.count()
    }

    /// Java: `boolean contains(Object x)`.
    pub fn contains(&self, x: &Object) -> bool {
        self.map.contains_key(x)
    }

    /// Java: `IPersistentSet cons(Object x)`. Adding an element already
    /// present is a no-op (returns an equivalent set).
    pub fn cons(self: &Arc<Self>, x: Object) -> Arc<Self> {
        if self.contains(&x) {
            return self.clone();
        }
        Arc::new(PersistentHashSet {
            map: self.map.assoc(x, Object::Nil),
        })
    }

    /// Java: `IPersistentSet disjoin(Object x)`.
    pub fn disjoin(self: &Arc<Self>, x: &Object) -> Arc<Self> {
        if !self.contains(x) {
            return self.clone();
        }
        Arc::new(PersistentHashSet {
            map: self.map.without(x),
        })
    }

    /// Iterator over elements in insertion order (delegates to the
    /// underlying map's key iteration).
    pub fn iter(&self) -> impl Iterator<Item = Object> + '_ {
        self.map.iter().map(|(k, _)| k)
    }

    /// Structural equality. Two sets are equal iff they contain the same
    /// elements (`equiv` semantics from Clojure's `Util.equiv`).
    pub fn equiv(&self, other: &PersistentHashSet) -> bool {
        if self.count() != other.count() {
            return false;
        }
        for (k, _) in self.map.iter() {
            if !other.contains(&k) {
                return false;
            }
        }
        true
    }

    /// Access the underlying map. Useful for heap-encoding paths that
    /// want to walk the (k, nil) pairs.
    pub fn as_map(&self) -> &Arc<PersistentHashMap> {
        &self.map
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
    fn empty_set_has_count_zero() {
        let s = PersistentHashSet::empty();
        assert_eq!(s.count(), 0);
        assert!(!s.contains(&kw("a")));
    }

    #[test]
    fn create_then_contains() {
        let s = PersistentHashSet::create(vec![kw("a"), kw("b"), kw("a")]);
        // Duplicates collapse — set has 2 elements.
        assert_eq!(s.count(), 2);
        assert!(s.contains(&kw("a")));
        assert!(s.contains(&kw("b")));
        assert!(!s.contains(&kw("c")));
    }

    #[test]
    fn cons_and_disjoin_are_persistent() {
        let s0 = PersistentHashSet::empty();
        let s1 = s0.cons(kw("a"));
        let s2 = s1.cons(kw("b"));
        let s3 = s2.disjoin(&kw("a"));
        assert_eq!(s0.count(), 0);
        assert_eq!(s1.count(), 1);
        assert_eq!(s2.count(), 2);
        assert_eq!(s3.count(), 1);
        assert!(!s3.contains(&kw("a")));
        assert!(s3.contains(&kw("b")));
    }

    #[test]
    fn equiv_matches_regardless_of_insertion_order() {
        let a = PersistentHashSet::create(vec![kw("a"), kw("b")]);
        let b = PersistentHashSet::create(vec![kw("b"), kw("a")]);
        assert!(a.equiv(&b));
    }
}
