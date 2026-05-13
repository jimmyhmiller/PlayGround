//! Host-side stand-in for `clojure.lang.PersistentTreeMap`.
//!
//! Original Java: `~/Documents/Code/open-source/clojure/src/jvm/clojure/lang/PersistentTreeMap.java`
//!
//! Java's `PersistentTreeMap` is a persistent red/black tree with `O(log n)`
//! ops. Our bootstrap loader only needs the public surface (`create`, `count`,
//! `assoc`, `entry_at`, `val_at`, ordered iteration), so we mirror the trick
//! [`PersistentHashMap`] uses: back the type with `Arc<Vec<(Object, Object)>>`
//! kept sorted by [`compare_objects`]. Operations are `O(n)` instead of
//! `O(log n)` — fine for bootstrap-sized maps; gets a real RB-tree once
//! something demands it.
//!
//! Ordering uses Clojure's natural `compare` for the supported `Object`
//! variants. Comparing values of incompatible types panics, matching
//! Java's `ClassCastException`.

use std::cmp::Ordering;
use std::sync::Arc;

use super::object::{object_equiv, Object};

/// `clojure.lang.PersistentTreeMap` — sorted by [`compare_objects`].
#[derive(Debug)]
pub struct PersistentTreeMap {
    /// Key/value pairs, kept sorted ascending by key.
    pairs: Arc<Vec<(Object, Object)>>,
}

impl PersistentTreeMap {
    /// Java: `PersistentTreeMap.EMPTY`.
    pub fn empty() -> Arc<Self> {
        Arc::new(PersistentTreeMap { pairs: Arc::new(Vec::new()) })
    }

    /// Java: `PersistentTreeMap create(Comparator c, ISeq items)` —
    /// flat alternating `[k1, v1, k2, v2, …]` ordered by `cmp`. Equal
    /// keys (cmp returns 0) overwrite earlier values.
    pub fn create_flat_cmp<F>(init: Vec<Object>, mut cmp: F) -> Arc<Self>
    where
        F: FnMut(&Object, &Object) -> Ordering,
    {
        if init.len() % 2 != 0 {
            panic!(
                "clojure-jvm: IllegalArgumentException — \
                 No value supplied for key (PersistentTreeMap.create of {} items)",
                init.len()
            );
        }
        let mut out: Vec<(Object, Object)> = Vec::with_capacity(init.len() / 2);
        let mut it = init.into_iter();
        while let (Some(k), Some(v)) = (it.next(), it.next()) {
            insert_sorted_with(&mut out, k, v, &mut cmp);
        }
        Arc::new(PersistentTreeMap { pairs: Arc::new(out) })
    }

    /// Java: `PersistentTreeMap create(ISeq items)` — flat alternating
    /// `[k1, v1, k2, v2, …]`. Later duplicate keys overwrite earlier ones.
    pub fn create_flat(init: Vec<Object>) -> Arc<Self> {
        if init.len() % 2 != 0 {
            panic!(
                "clojure-jvm: IllegalArgumentException — \
                 No value supplied for key (PersistentTreeMap.create of {} items)",
                init.len()
            );
        }
        let mut out: Vec<(Object, Object)> = Vec::with_capacity(init.len() / 2);
        let mut it = init.into_iter();
        while let (Some(k), Some(v)) = (it.next(), it.next()) {
            insert_sorted(&mut out, k, v);
        }
        Arc::new(PersistentTreeMap { pairs: Arc::new(out) })
    }

    pub fn count(&self) -> i32 { self.pairs.len() as i32 }

    pub fn entry_at(&self, key: &Object) -> Option<(Object, Object)> {
        self.pairs
            .iter()
            .find(|(k, _)| object_equiv(k, key))
            .map(|(k, v)| (k.clone(), v.clone()))
    }

    pub fn val_at(&self, key: &Object) -> Object {
        self.entry_at(key).map(|(_, v)| v).unwrap_or(Object::Nil)
    }

    pub fn val_at_or(&self, key: &Object, not_found: Object) -> Object {
        self.entry_at(key).map(|(_, v)| v).unwrap_or(not_found)
    }

    pub fn assoc(self: &Arc<Self>, key: Object, val: Object) -> Arc<Self> {
        let mut new_pairs: Vec<(Object, Object)> = (*self.pairs).clone();
        insert_sorted(&mut new_pairs, key, val);
        Arc::new(PersistentTreeMap { pairs: Arc::new(new_pairs) })
    }

    pub fn without(self: &Arc<Self>, key: &Object) -> Arc<Self> {
        let mut new_pairs: Vec<(Object, Object)> = (*self.pairs).clone();
        new_pairs.retain(|(k, _)| !object_equiv(k, key));
        Arc::new(PersistentTreeMap { pairs: Arc::new(new_pairs) })
    }

    pub fn contains_key(&self, key: &Object) -> bool {
        self.pairs.iter().any(|(k, _)| object_equiv(k, key))
    }

    /// Iterator in sorted-key order.
    pub fn iter(&self) -> impl Iterator<Item = (Object, Object)> + '_ {
        self.pairs.iter().map(|(k, v)| (k.clone(), v.clone()))
    }
}

fn insert_sorted(pairs: &mut Vec<(Object, Object)>, key: Object, val: Object) {
    insert_sorted_with(pairs, key, val, &mut compare_objects)
}

fn insert_sorted_with<F>(
    pairs: &mut Vec<(Object, Object)>,
    key: Object,
    val: Object,
    cmp: &mut F,
) where
    F: FnMut(&Object, &Object) -> Ordering,
{
    let mut insert_idx = pairs.len();
    for (i, (ek, _)) in pairs.iter().enumerate() {
        match cmp(&key, ek) {
            Ordering::Equal => {
                pairs[i].1 = val;
                return;
            }
            Ordering::Less => {
                insert_idx = i;
                break;
            }
            Ordering::Greater => {}
        }
    }
    pairs.insert(insert_idx, (key, val));
}

/// Clojure's `compare` semantics for the variants we currently produce.
/// Panics (Java `ClassCastException`) on incompatible types.
pub fn compare_objects(a: &Object, b: &Object) -> Ordering {
    use Object::*;
    match (a, b) {
        (Nil, Nil) => Ordering::Equal,
        (Nil, _) => Ordering::Less,
        (_, Nil) => Ordering::Greater,

        (Bool(x), Bool(y)) => x.cmp(y),

        (Long(x), Long(y)) => x.cmp(y),
        (Long(x), Double(y)) => (*x as f64).partial_cmp(y).unwrap_or(Ordering::Equal),
        (Double(x), Long(y)) => x.partial_cmp(&(*y as f64)).unwrap_or(Ordering::Equal),
        (Double(x), Double(y)) => x.partial_cmp(y).unwrap_or(Ordering::Equal),

        (String(x), String(y)) => x.cmp(y),

        (Keyword(x), Keyword(y)) => {
            let xs = format!(
                "{}/{}",
                x.get_namespace().unwrap_or(""),
                x.get_name()
            );
            let ys = format!(
                "{}/{}",
                y.get_namespace().unwrap_or(""),
                y.get_name()
            );
            xs.cmp(&ys)
        }
        (Symbol(x), Symbol(y)) => {
            let xs = format!(
                "{}/{}",
                x.get_namespace().unwrap_or(""),
                x.get_name()
            );
            let ys = format!(
                "{}/{}",
                y.get_namespace().unwrap_or(""),
                y.get_name()
            );
            xs.cmp(&ys)
        }

        (WithMeta(inner, _), _) => compare_objects(inner, b),
        (_, WithMeta(inner, _)) => compare_objects(a, inner),

        _ => panic!(
            "clojure-jvm: ClassCastException — cannot compare \
             incompatible Objects: {a:?} vs {b:?}"
        ),
    }
}
