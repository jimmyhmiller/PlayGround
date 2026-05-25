//! Host-side stand-in for `clojure.lang.PersistentTreeSet`.
//!
//! Original Java: `~/Documents/Code/open-source/clojure/src/jvm/clojure/lang/PersistentTreeSet.java`
//!
//! Java's `PersistentTreeSet` is a thin wrapper over `PersistentTreeMap`
//! using the set's elements as keys and a dummy value (`null`). We mirror
//! that: the set is an `Arc<PersistentTreeMap>` keyed by the element with
//! `Object::Nil` as the value.

use std::cmp::Ordering;
use std::sync::Arc;

use super::object::Object;
use super::persistent_tree_map::PersistentTreeMap;

#[derive(Debug)]
pub struct PersistentTreeSet {
    /// Underlying sorted map; keys are the set elements, values are always Nil.
    map: Arc<PersistentTreeMap>,
}

impl PersistentTreeSet {
    /// Java: `PersistentTreeSet.EMPTY`.
    pub fn empty() -> Arc<Self> {
        Arc::new(PersistentTreeSet { map: PersistentTreeMap::empty() })
    }

    /// Java: `PersistentTreeSet.create(ISeq init)` — natural ordering.
    pub fn create(items: Vec<Object>) -> Arc<Self> {
        let mut flat: Vec<Object> = Vec::with_capacity(items.len() * 2);
        for x in items {
            flat.push(x);
            flat.push(Object::Nil);
        }
        Arc::new(PersistentTreeSet {
            map: PersistentTreeMap::create_flat(flat),
        })
    }

    /// Java: `PersistentTreeSet.create(Comparator c, ISeq init)`.
    pub fn create_cmp<F>(items: Vec<Object>, cmp: F) -> Arc<Self>
    where
        F: FnMut(&Object, &Object) -> Ordering,
    {
        let mut flat: Vec<Object> = Vec::with_capacity(items.len() * 2);
        for x in items {
            flat.push(x);
            flat.push(Object::Nil);
        }
        Arc::new(PersistentTreeSet {
            map: PersistentTreeMap::create_flat_cmp(flat, cmp),
        })
    }

    pub fn count(&self) -> i32 { self.map.count() }
    pub fn contains(&self, x: &Object) -> bool { self.map.contains_key(x) }

    pub fn cons(self: &Arc<Self>, x: Object) -> Arc<Self> {
        if self.contains(&x) {
            return self.clone();
        }
        Arc::new(PersistentTreeSet { map: self.map.assoc(x, Object::Nil) })
    }

    pub fn disjoin(self: &Arc<Self>, x: &Object) -> Arc<Self> {
        if !self.contains(x) {
            return self.clone();
        }
        Arc::new(PersistentTreeSet { map: self.map.without(x) })
    }

    pub fn iter(&self) -> impl Iterator<Item = Object> + '_ {
        self.map.iter().map(|(k, _)| k)
    }
}
