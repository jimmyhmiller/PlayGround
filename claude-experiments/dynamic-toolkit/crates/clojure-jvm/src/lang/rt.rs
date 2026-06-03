//! Port of `clojure.lang.RT` — runtime helpers used pervasively by
//! Compiler.java.
//!
//! Source: `~/Documents/Code/open-source/clojure/src/jvm/clojure/lang/RT.java`
//!
//! RT.java is 2400+ lines of static helpers. We port them lazily as
//! Compiler.java actually calls them. Anything not yet ported stub-panics.

use std::sync::{Arc, LazyLock};

use super::keyword::Keyword;
use super::namespace::Namespace;
use super::object::Object;
use super::symbol::Symbol;
use super::var::Var;

/// `clojure.lang.RT.CLOJURE_NS` — the `clojure.core` namespace. Created
/// lazily on first reference.
pub static CLOJURE_NS: LazyLock<Arc<Namespace>> =
    LazyLock::new(|| Namespace::find_or_create(Symbol::intern("clojure.core")));

/// `clojure.lang.RT.CURRENT_NS` — Java: `Var.intern(CLOJURE_NS, *ns*, CLOJURE_NS).setDynamic()`.
/// Dynamic Var holding the current namespace. Default: `clojure.core`.
pub static CURRENT_NS: LazyLock<Arc<Var>> = LazyLock::new(|| {
    let ns = CLOJURE_NS.clone();
    let v = Var::intern_sym_root(ns.clone(), Symbol::intern("*ns*"), Object::Namespace(ns));
    v.clone().set_dynamic()
});

/// `Compiler.currentNS()` — Java line ~8106. `(Namespace) RT.CURRENT_NS.deref()`.
pub fn current_ns() -> Arc<Namespace> {
    match CURRENT_NS.deref() {
        Object::Namespace(ns) => ns,
        other => panic!("clojure-jvm: *ns* must hold a Namespace, found {other:?}"),
    }
}

/// Booleans `true` / `false` are used as plain `Object::Bool` everywhere in
/// Clojure. These constants match Java's `RT.T` / `RT.F`.
pub fn T() -> Object {
    Object::Bool(true)
}
pub fn F() -> Object {
    Object::Bool(false)
}

/// `RT.map(Object... init)` — build a small map. Compiler.java uses this to
/// attach :once meta to `FNONCE`, among other things. Stubbed until
/// PersistentHashMap is real.
pub fn map(_init: &[Object]) -> Object {
    crate::unimplemented_port!("RT.map(Object...)", "needs PersistentArrayMap / HashMap")
}

/// `RT.keyword(String ns, String name)`.
pub fn keyword(ns: Option<&str>, name: &str) -> Arc<Keyword> {
    Keyword::intern_ns_name(ns, name)
}

/// `RT.get(Object coll, Object key)`.
pub fn get(_coll: Object, _key: Object) -> Object {
    crate::unimplemented_port!("RT.get(Object,Object)", "needs ILookup dispatch")
}

/// `RT.get(Object coll, Object key, Object notFound)`.
pub fn get_or(_coll: Object, _key: Object, _not_found: Object) -> Object {
    crate::unimplemented_port!("RT.get(Object,Object,Object)", "needs ILookup dispatch")
}

/// `RT.assoc(Object coll, Object key, Object val)`.
pub fn assoc(_coll: Object, _key: Object, _val: Object) -> Object {
    crate::unimplemented_port!("RT.assoc", "needs Associative")
}

/// `RT.dissoc(Object coll, Object key)`.
pub fn dissoc(_coll: Object, _key: Object) -> Object {
    crate::unimplemented_port!("RT.dissoc", "needs IPersistentMap")
}

/// `RT.first(Object x)`. Returns the head of a seq-able value, or Nil.
pub fn first(x: &Object) -> Object {
    use super::i_seq::ISeq;
    match x {
        Object::Nil => Object::Nil,
        Object::List(l) => l.first(),
        _ => crate::unimplemented_port!(
            "RT.first",
            "non-list seq sources not ported yet (string, vector, map, set, …)"
        ),
    }
}

/// `RT.next(Object x)`. Returns the rest as an `Object` (List or Nil if
/// exhausted), matching Java's "null when empty" contract.
pub fn next(x: &Object) -> Object {
    use super::persistent_list::PersistentList;
    match x {
        Object::Nil => Object::Nil,
        Object::List(l) => match &**l {
            PersistentList::Empty => Object::Nil,
            PersistentList::Cons { count: 1, .. } => Object::Nil,
            PersistentList::Cons { rest, .. } => Object::List(rest.clone()),
        },
        _ => crate::unimplemented_port!("RT.next", "non-list seq sources not ported yet"),
    }
}

/// `RT.second(Object x)`. Equivalent to `first(next(x))`.
pub fn second(x: &Object) -> Object {
    let r = next(x);
    first(&r)
}

/// `RT.third(Object x)`. `first(next(next(x)))`.
pub fn third(x: &Object) -> Object {
    let r1 = next(x);
    let r2 = next(&r1);
    first(&r2)
}

/// `RT.fourth(Object x)`. `first(next(next(next(x))))`.
pub fn fourth(x: &Object) -> Object {
    let r1 = next(x);
    let r2 = next(&r1);
    let r3 = next(&r2);
    first(&r3)
}

/// `RT.seq(Object coll)`. Returns the seq-view (`Object::List`) or Nil for
/// empty/nil. Mirrors Java's "seq-or-null" idiom by way of the `Nil` variant.
pub fn seq(coll: &Object) -> Object {
    use super::i_seq::ISeq;
    match coll {
        Object::Nil => Object::Nil,
        Object::List(l) if l.count() == 0 => Object::Nil,
        Object::List(_) => coll.clone(),
        _ => crate::unimplemented_port!("RT.seq", "non-list seq sources not ported yet"),
    }
}

/// `RT.count(Object coll)`.
pub fn count(coll: &Object) -> i32 {
    use super::i_seq::ISeq;
    match coll {
        Object::Nil => 0,
        Object::List(l) => l.count(),
        _ => crate::unimplemented_port!("RT.count", "non-list seq sources not ported yet"),
    }
}

/// `RT.readString(String s)`.
pub fn read_string(_s: &str) -> Object {
    crate::unimplemented_port!("RT.readString", "needs LispReader")
}

/// `RT.var(String ns, String name)`.
pub fn var(_ns: &str, _name: &str) -> Object {
    crate::unimplemented_port!("RT.var(ns, name)", "needs Var.intern")
}

/// Sentinel used by `intern` factories: namespace `"clojure.core"`.
pub fn clojure_core_sym() -> Arc<Symbol> {
    Symbol::intern("clojure.core")
}
