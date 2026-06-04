//! Stand-in for `java.lang.Object`.
//!
//! Clojure JVM pervasively uses raw `Object` references and downcasts to the
//! concrete Clojure type. To stay 1-for-1 we keep that same shape: a single
//! `Object` enum carrying any Clojure value. Code that's "polymorphic over
//! `Object`" in Java becomes a `match` over `Object` here.
//!
//! Reference identity (Java `==`) is modeled by `Arc::ptr_eq` on the inner
//! `Arc<T>` for heap variants. Value equality (Clojure `=`) is implemented in
//! [`super::util::equiv`] (eventually).
//!
//! Heap variants use `Arc` rather than `Rc` so Clojure values can cross thread
//! boundaries the way Java references do.

use std::sync::Arc;

use std::any::Any;

use super::keyword::Keyword;
use super::namespace::Namespace;
use super::persistent_hash_map::PersistentHashMap;
use super::persistent_hash_set::PersistentHashSet;
use super::persistent_list::PersistentList;
use super::persistent_tree_map::PersistentTreeMap;
use super::persistent_tree_set::PersistentTreeSet;
use super::persistent_vector::PersistentVector;
use super::symbol::Symbol;
use super::var::Var;

/// A Clojure / JVM `Object` reference.
///
/// New variants are added as ported code needs them. Everything else stays
/// behind `Object::Unported` so it's loud rather than silent.
#[derive(Clone)]
pub enum Object {
    /// Java `null`.
    Nil,

    /// `java.lang.Boolean`.
    Bool(bool),

    /// `java.lang.Long` / boxed long. Clojure's default integer type.
    Long(i64),

    /// `java.lang.Double` / boxed double.
    Double(f64),

    /// `java.lang.Character`. A Unicode codepoint. Distinct from `Long`:
    /// `(str \a)` is `"a"` (not `"97"`) and `(= \a 97)` is `false`, while
    /// `(int \a)` is still `97`. Read from `\c` / `\newline` / `\uHHHH`.
    Char(u32),

    /// `java.lang.String`. Interned by `Rc` so cheap to clone.
    String(Arc<String>),

    /// `clojure.lang.Symbol`.
    Symbol(Arc<Symbol>),

    /// `clojure.lang.Keyword`.
    Keyword(Arc<Keyword>),

    /// `clojure.lang.Var`.
    Var(Arc<Var>),

    /// `clojure.lang.Namespace`.
    Namespace(Arc<Namespace>),

    /// `clojure.lang.PersistentList` (and seq views thereof). The first piece
    /// of structural Clojure data the compiler analyzes — `(a b c)` reads as
    /// an `Object::List`.
    List(Arc<PersistentList>),

    /// `clojure.lang.PersistentVector`. `[a b c]` reads as this.
    Vector(Arc<PersistentVector>),

    /// `clojure.lang.PersistentHashMap`. `{a 1 b 2}` reads as this.
    Map(Arc<PersistentHashMap>),

    /// `clojure.lang.PersistentHashSet`. `#{a b c}` reads as this.
    Set(Arc<PersistentHashSet>),

    /// `clojure.lang.PersistentTreeMap` (sorted map). `(sorted-map ...)`.
    TreeMap(Arc<PersistentTreeMap>),

    /// `clojure.lang.PersistentTreeSet` (sorted set). `(sorted-set ...)`.
    TreeSet(Arc<PersistentTreeSet>),

    /// Escape hatch for arbitrary host-side state Java threads through `Var`s
    /// during compilation (`LOCAL_ENV` map, `METHOD` (ObjMethod), `PathNode`
    /// chain, etc.). Java has all of these as plain `Object` references; in
    /// Rust we erase the concrete type via `Arc<dyn Any + Send + Sync>` and
    /// the reader uses `Arc::downcast` to recover it.
    ///
    /// Use sparingly: this is for compile-time scaffolding, not for values
    /// users can observe. Anything with a real Clojure type should get its
    /// own variant.
    Host(Arc<dyn Any + Send + Sync>),

    /// A value whose Clojure type isn't ported yet. Carries the original
    /// Java class name so panic messages are actionable.
    Unported { java_class: &'static str },

    /// A value with reader-attached metadata. Wraps any `Object` that
    /// would otherwise satisfy `IMeta`/`IObj` (Symbols, collections,
    /// fns). Equality, structural traversal, and type predicates all
    /// peek through this wrapper transparently — only code that
    /// explicitly asks for the metadata (`meta_of`) sees it.
    ///
    /// The inner is `Box`'d to keep the enum's discriminant small;
    /// nested `WithMeta` is collapsed when constructed via
    /// `with_meta` so the wrapper is always exactly one layer deep.
    WithMeta(Box<Object>, Arc<PersistentHashMap>),
}

impl Object {
    /// Returns `true` if this is `Object::Nil` (Java `null`).
    #[inline]
    pub fn is_nil(&self) -> bool {
        matches!(self.peel_meta_ref(), Object::Nil)
    }

    /// Java-style `instanceof Symbol`.
    #[inline]
    pub fn as_symbol(&self) -> Option<&Arc<Symbol>> {
        if let Object::Symbol(s) = self.peel_meta_ref() {
            Some(s)
        } else {
            None
        }
    }

    #[inline]
    pub fn as_keyword(&self) -> Option<&Arc<Keyword>> {
        if let Object::Keyword(k) = self.peel_meta_ref() {
            Some(k)
        } else {
            None
        }
    }

    #[inline]
    pub fn as_var(&self) -> Option<&Arc<Var>> {
        if let Object::Var(v) = self.peel_meta_ref() {
            Some(v)
        } else {
            None
        }
    }

    /// View through any `WithMeta` wrapper to the underlying value.
    /// Most type-dispatch code wants this — equality, `is_nil`, type
    /// predicates all ignore metadata in Clojure semantics.
    #[inline]
    pub fn peel_meta_ref(&self) -> &Object {
        match self {
            Object::WithMeta(inner, _) => inner.peel_meta_ref(),
            other => other,
        }
    }

    /// Owned variant of `peel_meta_ref`: clone out the inner value,
    /// dropping any metadata. Use when you need an owned `Object` for
    /// dispatch and the metadata is irrelevant.
    #[inline]
    pub fn peel_meta(self) -> Object {
        match self {
            Object::WithMeta(inner, _) => (*inner).peel_meta(),
            other => other,
        }
    }

    /// Return the metadata map attached to this value, or `None` if
    /// there is no metadata. Walks through nested `WithMeta` wrappers
    /// (which `with_meta` normally collapses).
    #[inline]
    pub fn meta_of(&self) -> Option<&Arc<PersistentHashMap>> {
        if let Object::WithMeta(_, m) = self {
            Some(m)
        } else {
            None
        }
    }

    /// Attach `meta` to `inner`, collapsing nested `WithMeta` wrappers
    /// so the resulting `Object` is at most one layer deep. Mirrors
    /// Java's `IObj.withMeta`: identity-comparable to the receiver only
    /// if the receiver already carried the same metadata.
    pub fn with_meta_map(inner: Object, meta: Arc<PersistentHashMap>) -> Object {
        let bare = inner.peel_meta();
        Object::WithMeta(Box::new(bare), meta)
    }

    /// Wrap any host-side state in `Object::Host`.
    pub fn host<T: Any + Send + Sync + 'static>(value: T) -> Self {
        Object::Host(Arc::new(value))
    }

    /// Try to recover a host-side value from `Object::Host`. Returns `None`
    /// if `self` isn't a Host of the requested type.
    pub fn host_as<T: Any + Send + Sync + 'static>(&self) -> Option<Arc<T>> {
        if let Object::Host(a) = self {
            a.clone().downcast::<T>().ok()
        } else {
            None
        }
    }

    /// Convert a numeric `Object` (`Long` / `Double`) to `f64`. Used by IR
    /// lowering when handing a value off to dynlang's NanBox `number()`
    /// (which is float-only at the moment). Returns `None` for non-numeric
    /// variants.
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Object::Long(n) => Some(*n as f64),
            Object::Double(x) => Some(*x),
            _ => None,
        }
    }
}

/// `clojure.lang.Util.equiv(a, b)` — Clojure's structural equality, ported
/// to host-side `Object`s. The runtime extern [`crate::runtime::equiv_impl`]
/// is the sibling on NanBox-encoded heap values; both should agree on every
/// case they have in common.
///
/// Rules:
/// * `nil` equals only `nil`.
/// * `true`/`false` equal themselves; `false` is NOT equiv to `nil`
///   (Clojure's `=`, unlike its truthiness rules, distinguishes them).
/// * `Long` and `Double` cross-compare numerically (Clojure's
///   `Util.equiv(Number, Number)` defers to `Numbers.equal` which treats
///   `1` and `1.0` as `=`).
/// * `String`, `Symbol`, `Keyword`: structural over their fields. Keywords
///   are globally interned so pointer equality short-circuits.
/// * `List`, `Vector`: element-wise structural recursion.
/// * `Map`: delegates to [`PersistentHashMap::equiv`].
/// * `Var`, `Namespace`, `Host`, `Unported`: identity (`Arc::ptr_eq` for the
///   first two; otherwise `false`).
pub fn object_equiv(a: &Object, b: &Object) -> bool {
    // Clojure equality ignores metadata: `(= ^{:m 1} '(a) '(a))` ⇒ true.
    // Peel any `WithMeta` wrapper before structural compare.
    let a = a.peel_meta_ref();
    let b = b.peel_meta_ref();
    use Object::*;
    match (a, b) {
        (Nil, Nil) => true,
        (Bool(x), Bool(y)) => x == y,
        (Long(x), Long(y)) => x == y,
        (Double(x), Double(y)) => x == y,
        (Long(x), Double(y)) | (Double(y), Long(x)) => (*x as f64) == *y,
        // Characters compare by codepoint. `(= \a \a)` ⇒ true, and they must
        // dedup as map/set keys (e.g. `(frequencies "aabbbc")`). A Character
        // is NOT equal to its int codepoint: `(= \a 97)` ⇒ false (no Char/Long
        // cross-arm), matching Clojure.
        (Char(x), Char(y)) => x == y,
        (String(x), String(y)) => **x == **y,
        (Symbol(x), Symbol(y)) => x == y || **x == **y,
        (Keyword(x), Keyword(y)) => Arc::ptr_eq(x, y) || **x == **y,
        (Var(x), Var(y)) => Arc::ptr_eq(x, y),
        (Namespace(x), Namespace(y)) => Arc::ptr_eq(x, y),
        (List(x), List(y)) => list_equiv(x, y),
        (Vector(x), Vector(y)) => vector_equiv(x, y),
        (Map(x), Map(y)) => x.equiv(y),
        (Set(x), Set(y)) => x.equiv(y),
        (TreeMap(x), TreeMap(y)) => {
            x.count() == y.count()
                && x.iter().all(|(k, v)| match y.entry_at(&k) {
                    Some((_, ov)) => object_equiv(&v, &ov),
                    None => false,
                })
        }
        (TreeSet(x), TreeSet(y)) => {
            x.count() == y.count() && x.iter().all(|e| y.contains(&e))
        }
        _ => false,
    }
}

fn list_equiv(a: &Arc<PersistentList>, b: &Arc<PersistentList>) -> bool {
    if Arc::ptr_eq(a, b) {
        return true;
    }
    let mut ca: &PersistentList = a;
    let mut cb: &PersistentList = b;
    loop {
        match (ca, cb) {
            (PersistentList::Empty, PersistentList::Empty) => return true,
            (
                PersistentList::Cons {
                    first: fa,
                    rest: ra,
                    ..
                },
                PersistentList::Cons {
                    first: fb,
                    rest: rb,
                    ..
                },
            ) => {
                if !object_equiv(fa, fb) {
                    return false;
                }
                ca = ra;
                cb = rb;
            }
            _ => return false,
        }
    }
}

fn vector_equiv(a: &Arc<PersistentVector>, b: &Arc<PersistentVector>) -> bool {
    if Arc::ptr_eq(a, b) {
        return true;
    }
    if a.count() != b.count() {
        return false;
    }
    for i in 0..a.count() {
        if !object_equiv(&a.nth(i), &b.nth(i)) {
            return false;
        }
    }
    true
}

impl std::fmt::Debug for Object {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Object::Nil => write!(f, "nil"),
            Object::Bool(b) => write!(f, "{b}"),
            Object::Long(n) => write!(f, "{n}"),
            Object::Double(x) => write!(f, "{x}"),
            Object::Char(c) => match char::from_u32(*c) {
                Some(ch) => write!(f, "\\{ch}"),
                None => write!(f, "\\u{c:04x}"),
            },
            Object::String(s) => write!(f, "{s:?}"),
            Object::Symbol(s) => write!(f, "{s:?}"),
            Object::Keyword(k) => write!(f, "{k:?}"),
            Object::Var(v) => write!(f, "{v:?}"),
            Object::Namespace(n) => write!(f, "{n:?}"),
            Object::List(l) => write!(f, "{l:?}"),
            Object::Vector(v) => write!(f, "{v:?}"),
            Object::Map(m) => write!(f, "{m:?}"),
            Object::Set(s) => write!(f, "{s:?}"),
            Object::TreeMap(m) => {
                let items: Vec<String> =
                    m.iter().map(|(k, v)| format!("{k:?} {v:?}")).collect();
                write!(f, "{{{}}}", items.join(", "))
            }
            Object::TreeSet(s) => {
                let items: Vec<String> = s.iter().map(|x| format!("{x:?}")).collect();
                write!(f, "#{{{}}}", items.join(" "))
            }
            Object::Host(_) => write!(f, "#<host>"),
            Object::Unported { java_class } => {
                write!(f, "#<unported {java_class}>")
            }
            Object::WithMeta(inner, meta) => {
                // Mirror Java's IObj print contract: the metadata is
                // typically invisible to `pr`, but for Debug we want it
                // visible so test failures show what got attached.
                write!(f, "^{:?} {:?}", meta, inner)
            }
        }
    }
}
