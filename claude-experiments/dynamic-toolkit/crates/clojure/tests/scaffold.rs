//! Smoke tests for the crate scaffold.
//!
//! These don't exercise any real behavior — they just confirm the
//! crate compiles, the engine constructs, and every heap type was
//! declared with a distinct ID.

use clojure::Engine;

#[test]
fn engine_constructs() {
    let _e = Engine::new();
}

#[test]
fn types_have_distinct_ids() {
    let e = Engine::new();
    let t = &e.host.types;
    let ids = [
        t.symbol.0,
        t.keyword.0,
        t.string.0,
        t.list.0,
        t.vector.0,
        t.map.0,
        t.set.0,
        t.fn_obj.0,
        t.var.0,
        t.namespace.0,
        t.registry.0,
    ];
    let mut sorted = ids.to_vec();
    sorted.sort();
    sorted.dedup();
    assert_eq!(
        sorted.len(),
        ids.len(),
        "duplicate ObjTypeId across heap types"
    );
}
