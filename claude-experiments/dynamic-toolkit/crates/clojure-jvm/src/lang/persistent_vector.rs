//! Port of `clojure.lang.PersistentVector` (minimal slice).
//!
//! Source: `~/Documents/Code/open-source/clojure/src/jvm/clojure/lang/PersistentVector.java`
//!
//! Java's real PersistentVector is a Bagwell HAMT-style tree: 32-way branching
//! over a `root` node plus a tail buffer, with `O(log32 n)` `nth`/`assoc` and
//! `O(1)` `count`/`peek`. Compiler.java touches this through a small surface:
//! `nth(i)`, `count()`, `cons(x)`, `assoc(i, x)`. For analyze-time data the
//! constant factors don't matter, so we back the type with `Arc<Vec<Object>>`
//! and preserve the public methods. When (if) we need the real persistent
//! semantics for runtime collections, this gets swapped for a HAMT.

use std::sync::Arc;

use super::object::Object;

/// `clojure.lang.PersistentVector`. Immutable list of `Object`s; `Arc` makes
/// the structural-sharing-by-cloning cheap. Mutation methods return a new
/// vector via `Arc::make_mut` (clone-on-write).
#[derive(Debug)]
pub struct PersistentVector {
    items: Arc<Vec<Object>>,
}

impl PersistentVector {
    /// Java: `PersistentVector.EMPTY`.
    pub fn empty() -> Arc<Self> {
        Arc::new(PersistentVector {
            items: Arc::new(Vec::new()),
        })
    }

    /// Java: `PersistentVector.create(Object... items)`.
    pub fn create(items: Vec<Object>) -> Arc<Self> {
        Arc::new(PersistentVector {
            items: Arc::new(items),
        })
    }

    /// Java: `int count()`. O(1).
    pub fn count(&self) -> i32 {
        self.items.len() as i32
    }

    /// Java: `Object nth(int i)`. Panics on out-of-bounds (Java throws
    /// `IndexOutOfBoundsException`).
    pub fn nth(&self, i: i32) -> Object {
        let idx = i as usize;
        if idx >= self.items.len() {
            panic!(
                "clojure-jvm: IndexOutOfBoundsException — PersistentVector.nth({}) on vector of count {}",
                i,
                self.items.len()
            );
        }
        self.items[idx].clone()
    }

    /// Java: `IPersistentVector cons(Object o)`. Returns a new vector with
    /// `o` appended.
    pub fn cons(self: &Arc<Self>, o: Object) -> Arc<Self> {
        let mut new_items: Vec<Object> = (*self.items).clone();
        new_items.push(o);
        Arc::new(PersistentVector {
            items: Arc::new(new_items),
        })
    }

    /// Java: `IPersistentVector assocN(int i, Object val)`.
    pub fn assoc_n(self: &Arc<Self>, i: i32, val: Object) -> Arc<Self> {
        let idx = i as usize;
        let mut new_items: Vec<Object> = (*self.items).clone();
        if idx == new_items.len() {
            // Java treats `assocN(count, v)` as cons.
            new_items.push(val);
        } else if idx < new_items.len() {
            new_items[idx] = val;
        } else {
            panic!(
                "clojure-jvm: IndexOutOfBoundsException — PersistentVector.assocN({}) on vector of count {}",
                i,
                new_items.len()
            );
        }
        Arc::new(PersistentVector {
            items: Arc::new(new_items),
        })
    }

    /// Iterator over the vector's items. Compiler.java iterates with
    /// `for (Object o : exprs)` (Java `Iterable`); this is the equivalent.
    pub fn iter(&self) -> impl Iterator<Item = Object> + '_ {
        self.items.iter().cloned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_vector_has_count_zero() {
        let v = PersistentVector::empty();
        assert_eq!(v.count(), 0);
    }

    #[test]
    fn nth_returns_element_at_index() {
        let v =
            PersistentVector::create(vec![Object::Long(10), Object::Long(20), Object::Long(30)]);
        assert!(matches!(v.nth(0), Object::Long(10)));
        assert!(matches!(v.nth(2), Object::Long(30)));
        assert_eq!(v.count(), 3);
    }

    #[test]
    fn cons_appends_and_preserves_original() {
        let v0 = PersistentVector::create(vec![Object::Long(1)]);
        let v1 = v0.cons(Object::Long(2));
        assert_eq!(v0.count(), 1);
        assert_eq!(v1.count(), 2);
        assert!(matches!(v1.nth(1), Object::Long(2)));
    }

    #[test]
    fn assoc_n_replaces_at_index() {
        let v0 = PersistentVector::create(vec![Object::Long(1), Object::Long(2)]);
        let v1 = v0.assoc_n(1, Object::Long(99));
        assert!(matches!(v0.nth(1), Object::Long(2)));
        assert!(matches!(v1.nth(1), Object::Long(99)));
    }

    #[test]
    #[should_panic(expected = "IndexOutOfBoundsException")]
    fn nth_panics_on_out_of_bounds() {
        let v = PersistentVector::empty();
        let _ = v.nth(0);
    }
}
