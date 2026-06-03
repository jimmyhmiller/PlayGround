//! Checkpoint 4: `def` interns a Var in the `clojure.core`
//! namespace heap object.

use clojure::Engine;
use clojure::namespace::{
    fn_func_ref, ns_lookup, ns_name, registry_find_ns, registry_namespaces, var_root, var_sym,
};
use clojure::value::{self as v};

#[test]
fn engine_has_clojure_core() {
    let e = Engine::new();
    let reg = e.registry();
    assert!(v::is_ptr(reg), "registry should be a heap pointer");
    let core = e.core_ns();
    assert!(v::is_ptr(core), "clojure.core should be a heap pointer");
    // Registry should also resolve clojure.core by name.
    let name_id = e.host.sym.intern("clojure.core");
    let found = registry_find_ns(reg, v::encode_sym_id(name_id));
    assert_eq!(found, core, "registry should map 'clojure.core to core_ns");
}

#[test]
fn def_creates_var_in_core() {
    let mut e = Engine::new();
    let _ = e.eval("(def square (fn [x] (* x x)))");

    let core = e.core_ns();
    let sym_id = e.host.sym.intern("square");
    let var = ns_lookup(core, v::encode_sym_id(sym_id));
    assert!(v::is_ptr(var), "square Var should be a heap object");
    // Var.sym should match the symbol we asked about.
    assert_eq!(var_sym(var), v::encode_sym_id(sym_id));
}

#[test]
fn var_root_is_fn_with_correct_funcref() {
    let mut e = Engine::new();
    let _ = e.eval("(def add2 (fn [a b] (+ a b)))");

    let core = e.core_ns();
    let sym_id = e.host.sym.intern("add2");
    let var = ns_lookup(core, v::encode_sym_id(sym_id));
    let root = var_root(var);
    assert!(v::is_ptr(root), "var.root should be a Fn heap object");
    let fr = fn_func_ref(root);
    // The FuncRef should not be 0 (which is reserved for the first
    // declared func). Beyond that, exact value depends on declaration
    // order, so we only check it's > 0.
    assert!(fr > 0, "Fn.func_ref should be > 0 (got {fr})");
}

#[test]
fn redefining_a_var_preserves_identity() {
    let mut e = Engine::new();
    e.eval("(def f (fn [x] x))");
    let core = e.core_ns();
    let sym_id = e.host.sym.intern("f");
    let var1 = ns_lookup(core, v::encode_sym_id(sym_id));

    // Redefine. Var identity must persist.
    e.eval("(def f (fn [x] (* x 2)))");
    let var2 = ns_lookup(e.core_ns(), v::encode_sym_id(sym_id));
    assert_eq!(
        var1, var2,
        "redef must update Var.root in place, not allocate a new Var"
    );
}

#[test]
fn ns_name_round_trips() {
    let e = Engine::new();
    let core = e.core_ns();
    let name = ns_name(core);
    assert!(v::is_sym_id(name), "ns name should be a symbol id");
}

#[test]
fn original_fib_program_still_works_after_var_intern() {
    let mut e = Engine::new();
    let v = e.eval(
        "(def fib (fn [n] (if (< n 2) n (+ (fib (- n 1)) (fib (- n 2)))))) \
         (fib 10)",
    );
    assert_eq!(e.print(v), "55");
    // And the fib var is in the namespace.
    let core = e.core_ns();
    let sym_id = e.host.sym.intern("fib");
    let var = ns_lookup(core, v::encode_sym_id(sym_id));
    assert!(v::is_ptr(var), "fib Var should be in clojure.core");
}

#[test]
fn registry_mappings_grow_with_namespaces() {
    let e = Engine::new();
    let reg = e.registry();
    let m = registry_namespaces(reg);
    assert!(v::is_ptr(m), "registry.namespaces should be a Map");
    // After bootstrap, exactly clojure.core should be present.
    assert_eq!(
        clojure::namespace::map_count(m),
        1,
        "should have 1 namespace at startup"
    );
}
