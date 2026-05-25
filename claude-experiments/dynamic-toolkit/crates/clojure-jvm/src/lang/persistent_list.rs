//! Port of `clojure.lang.PersistentList` (minimal slice).
//!
//! Source: `~/Documents/Code/open-source/clojure/src/jvm/clojure/lang/PersistentList.java`
//!
//! Java declaration:
//! ```text
//! public class PersistentList extends ASeq implements IPersistentList,
//!                                                       IReduce, List, Counted
//! ```
//!
//! What's ported:
//!   * The cons/empty data shape (singly-linked nodes + EMPTY sentinel).
//!   * `first`, `next`, `more`, `cons`, `count`.
//!   * The `create(Vec<Object>)` constructor used by the reader / `analyze`.
//!
//! What's stubbed:
//!   * Metadata (`IObj.withMeta` / `meta()`)
//!   * `IReduce` / `List` Java collection ops
//!   * `Primordial` (the `list` fn) — it lives on `IFn` and depends on more.
//!
//! The Java `EmptyList` inner class is collapsed into a single `Empty` variant
//! of our enum — saves a whole class while preserving observable behavior
//! (first → throws? no, Java returns null; count → 0; next → null; cons →
//! single-element PersistentList).

use std::sync::Arc;

use super::i_seq::ISeq;
use super::object::Object;

/// `clojure.lang.PersistentList`. Two-state enum representing the empty list
/// and a cons cell. The Java class is two classes (`EmptyList` + `PersistentList`)
/// but Rust enums make the dispatch local and cheap.
#[derive(Debug)]
pub enum PersistentList {
    /// Java: `final public static EmptyList EMPTY = new EmptyList(null);`
    Empty,
    /// Java: `_first`, `_rest`, `_count`. We hold `rest` as another
    /// `Arc<PersistentList>` (always non-null in Java's PersistentList; the
    /// EMPTY sentinel terminates the chain).
    Cons {
        first: Object,
        rest: Arc<PersistentList>,
        count: i32,
    },
}

impl PersistentList {
    /// Java: `PersistentList.EMPTY`.
    pub fn empty() -> Arc<Self> {
        // A fresh Arc each call is fine here — equality of empties is by
        // variant, not identity. If this shows up as hot we can cache it in a
        // `OnceLock`.
        Arc::new(PersistentList::Empty)
    }

    /// Single-element list. Java: `public PersistentList(Object first)`.
    pub fn single(first: Object) -> Arc<Self> {
        Arc::new(PersistentList::Cons {
            first,
            rest: PersistentList::empty(),
            count: 1,
        })
    }

    /// Java: `public static IPersistentList create(List init)`. Builds a list
    /// from a Vec in O(n), preserving order. Java conses front-to-back via a
    /// reverse iterator; here we walk the Vec backwards.
    pub fn create(items: Vec<Object>) -> Arc<Self> {
        let mut acc = PersistentList::empty();
        for item in items.into_iter().rev() {
            acc = acc.cons_arc(item);
        }
        acc
    }

    /// `cons` returning a typed `Arc<PersistentList>` (not the `dyn ISeq`
    /// trait object). Convenience for callers that statically know they hold a
    /// PersistentList.
    pub fn cons_arc(self: Arc<Self>, o: Object) -> Arc<Self> {
        let count = match &*self {
            PersistentList::Empty => 1,
            PersistentList::Cons { count, .. } => count + 1,
        };
        Arc::new(PersistentList::Cons { first: o, rest: self, count })
    }

    /// Iterator over the list's `Object` elements. Convenience helper —
    /// matches Clojure's "seq is iterable" but skips going through ISeq.
    pub fn iter(self: &Arc<Self>) -> ListIter {
        ListIter { node: Some(self.clone()) }
    }

    /// Java: `int count()`. O(1).
    pub fn count(&self) -> i32 {
        match self {
            PersistentList::Empty => 0,
            PersistentList::Cons { count, .. } => *count,
        }
    }
}

impl ISeq for PersistentList {
    fn first(&self) -> Object {
        match self {
            PersistentList::Empty => Object::Nil,
            PersistentList::Cons { first, .. } => first.clone(),
        }
    }

    fn next(&self) -> Option<Arc<dyn ISeq>> {
        match self {
            PersistentList::Empty => None,
            PersistentList::Cons { count: 1, .. } => None,
            PersistentList::Cons { rest, .. } => {
                Some(rest.clone() as Arc<dyn ISeq>)
            }
        }
    }

    fn more(&self) -> Arc<dyn ISeq> {
        match self {
            PersistentList::Empty => PersistentList::empty() as Arc<dyn ISeq>,
            PersistentList::Cons { count: 1, .. } => {
                PersistentList::empty() as Arc<dyn ISeq>
            }
            PersistentList::Cons { rest, .. } => rest.clone() as Arc<dyn ISeq>,
        }
    }

    fn cons(&self, o: Object) -> Arc<dyn ISeq> {
        // We can't call cons_arc(self) because we only have `&Self`.
        // Rebuild a fresh head with this node as the rest.
        let rest: Arc<PersistentList> = match self {
            PersistentList::Empty => PersistentList::empty(),
            // Clone of `Cons { … }` via the variant fields is awkward; instead
            // we wrap the borrowed shape into a fresh Arc holding a structural
            // copy. (Persistent lists are immutable, so two Arcs over the same
            // structural contents are observationally identical.)
            PersistentList::Cons { first, rest, count } => {
                Arc::new(PersistentList::Cons {
                    first: first.clone(),
                    rest: rest.clone(),
                    count: *count,
                })
            }
        };
        let count = match &*rest {
            PersistentList::Empty => 1,
            PersistentList::Cons { count, .. } => count + 1,
        };
        Arc::new(PersistentList::Cons { first: o, rest, count }) as Arc<dyn ISeq>
    }

    fn count(&self) -> i32 { PersistentList::count(self) }
}

/// Iterator over a `PersistentList`. `Cons` values stay live via the Arc chain
/// while iteration is in progress.
pub struct ListIter {
    node: Option<Arc<PersistentList>>,
}

impl Iterator for ListIter {
    type Item = Object;

    fn next(&mut self) -> Option<Object> {
        let node = self.node.take()?;
        match &*node {
            PersistentList::Empty => None,
            PersistentList::Cons { first, rest, .. } => {
                let v = first.clone();
                self.node = Some(rest.clone());
                Some(v)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_list_has_count_zero_and_no_next() {
        let e = PersistentList::empty();
        assert_eq!(ISeq::count(&*e), 0);
        assert!(matches!(ISeq::first(&*e), Object::Nil));
        assert!(ISeq::next(&*e).is_none());
    }

    #[test]
    fn create_preserves_order() {
        let l = PersistentList::create(vec![
            Object::Long(1),
            Object::Long(2),
            Object::Long(3),
        ]);
        assert_eq!(ISeq::count(&*l), 3);
        let got: Vec<i64> = l
            .iter()
            .map(|o| match o {
                Object::Long(n) => n,
                _ => panic!("unexpected"),
            })
            .collect();
        assert_eq!(got, vec![1, 2, 3]);
    }

    #[test]
    fn cons_increments_count_and_first() {
        let l = PersistentList::single(Object::Long(2)).cons_arc(Object::Long(1));
        assert_eq!(l.count(), 2);
        assert!(matches!(ISeq::first(&*l), Object::Long(1)));
        let rest = ISeq::next(&*l).unwrap();
        assert!(matches!(rest.first(), Object::Long(2)));
    }

    #[test]
    fn iter_walks_three_element_list() {
        let l = PersistentList::create(vec![
            Object::Long(10),
            Object::Long(20),
            Object::Long(30),
        ]);
        let collected: Vec<_> = l.iter().collect();
        assert_eq!(collected.len(), 3);
        assert!(matches!(collected[0], Object::Long(10)));
        assert!(matches!(collected[1], Object::Long(20)));
        assert!(matches!(collected[2], Object::Long(30)));
    }

    #[test]
    fn more_returns_empty_at_singleton() {
        let l = PersistentList::single(Object::Long(42));
        let m = ISeq::more(&*l);
        assert_eq!(m.count(), 0);
    }
}
