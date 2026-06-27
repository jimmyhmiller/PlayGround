//! Runtime-dispatched trait objects ("dyn traits") as a pure library (lib/dyn.coil).
//! `(defdyn Trait)` reflects the trait's method signatures (the new
//! `code-trait-*` comptime ops) and generates a vtable struct + a fat-pointer
//! struct + a forwarding impl; `(dyn-impl Trait Type)` emits the erased thunks and
//! Erasure is the compiler's `make-dyn` coercion: a `(ptr T)` auto-erases to a
//! `(dyn Trait)` argument (or explicitly via `(make-dyn Trait ptr)`), building a
//! constant vtable of the impl methods' addresses + a caller-frame fat pointer.
//! A function taking `(dyn Trait)` dispatches at runtime to whichever concrete
//! impl was erased — no hand-written function pointers. See lib/dyn.coil.

mod common;
use common::build_and_run;

// A trait with one method, two implementers, and a `total` function written ONCE
// against `(dyn Area)` that dispatches at runtime.
const H: &str = concat!(
    "(module app)\n",
    "(import \"lib/dyn.coil\" :use *)\n",
    "(deftrait Area [Self] (area [(self (ptr Self))] (-> i64)))\n",
    "(defdyn Area)\n",
    "(defstruct Square [(side i64)])\n",
    "(impl Area Square\n",
    "  (area [(self (ptr Square))] (-> i64)\n",
    "    (let [s (load (field self side))] (imul s s))))\n",
    "(defstruct Rect [(w i64) (h i64)])\n",
    "(impl Area Rect\n",
    "  (area [(self (ptr Rect))] (-> i64)\n",
    "    (imul (load (field self w)) (load (field self h)))))\n",
    // one function, any Area — dynamic dispatch through the vtable
    "(defn area-of [(a (dyn Area))] (-> i64) (area a))\n",
);

#[test]
fn dyn_dispatch_picks_the_right_impl_at_runtime() {
    // Square(5)->25 via dyn, Rect(3,4)->12 via dyn; `area-of` is written once and
    // the concrete (ptr Square)/(ptr Rect) auto-erase to (dyn Area) at the call.
    let code = build_and_run(&format!(
        "{H}(defn main [] (-> i64)\n\
           (let [sq (alloc-stack Square) rc (alloc-stack Rect)]\n\
             (store! (field sq side) 5)\n\
             (store! (field rc w) 3) (store! (field rc h) 4)\n\
             (iadd (area-of sq) (area-of rc))))" // 25 + 12
    ));
    assert_eq!(code, 37);
}

#[test]
fn dyn_object_is_first_class_can_be_stored_and_reselected() {
    // Pick one of two erased objects at runtime, then dispatch — proves the dyn
    // value carries its own vtable (not resolved at the call site).
    let code = build_and_run(&format!(
        "{H}(defn pick [(which i64) (a (dyn Area)) (b (dyn Area))] (-> i64)\n\
           (area (if (icmp-eq which 0) a b)))\n\
         (defn main [] (-> i64)\n\
           (let [sq (alloc-stack Square) rc (alloc-stack Rect)]\n\
             (store! (field sq side) 6)\n\
             (store! (field rc w) 2) (store! (field rc h) 10)\n\
             (let [da (make-dyn Area sq) db (make-dyn Area rc)]\n\
               (iadd (pick 0 da db) (pick 1 da db)))))" // 36 + 20
    ));
    assert_eq!(code, 56);
}

#[test]
fn non_object_safe_method_is_a_clear_error_not_silent() {
    // A method returning Self can't go in a vtable (the receiver is type-erased).
    // defdyn must hard-error, never silently drop it.
    let src = concat!(
        "(module app)\n",
        "(import \"lib/dyn.coil\" :use *)\n",
        "(deftrait Bad [Self] (clone [(self (ptr Self))] (-> Self)))\n",
        "(defdyn Bad)\n",
        "(defn main [] (-> i64) 0)\n",
    );
    let err = coil::emit_ir(src).expect_err("non-object-safe trait must be rejected");
    assert!(
        err.contains("object-safe") || err.contains("Self"),
        "expected an object-safety error, got: {err}"
    );
}
