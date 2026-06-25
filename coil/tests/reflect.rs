//! Stage 2: compile-time type reflection. `field-count`/`variant-count` (i64) and
//! `struct?`/`sum?`/`int?`/`float?`/`ptr?`/`array?` (bool) are evaluated by the
//! comptime interpreter and fold to literals — usable in `const`, `static-assert`,
//! `(comptime …)`, and ordinary code.

mod common;
use common::build_and_run;

const PRE: &str = "(module a)\n\
    (defstruct Point [(x i64) (y i64) (z i64)])\n\
    (defsum Shape (Circle [(r i64)]) (Rect [(w i64) (h i64)]) (Tri []))\n";

#[test]
fn field_and_variant_counts() {
    let code = build_and_run(&format!(
        "{PRE}(defn main [] (-> i64) (+ (field-count Point) (variant-count Shape)))" // 3 + 3 = 6
    ));
    assert_eq!(code, 6);
}

#[test]
fn kind_predicates() {
    let code = build_and_run(&format!(
        "{PRE}(defn main [] (-> i64)\n\
           (+ (if (struct? Point) 1 0)\n\
              (+ (if (sum? Shape) 2 0)\n\
                 (+ (if (int? i64) 4 0)\n\
                    (+ (if (array? (array i64 4)) 8 0)\n\
                       (if (sum? Point) 16 0))))))" // 1+2+4+8+0 = 15
    ));
    assert_eq!(code, 15);
}

#[test]
fn reflection_in_a_const() {
    let code = build_and_run(&format!(
        "{PRE}(const NF (field-count Point))\n(defn main [] (-> i64) (* NF 14))" // 3*14 = 42
    ));
    assert_eq!(code, 42);
}

#[test]
fn reflection_in_comptime_computation() {
    // a comptime expression that uses a count
    let code = build_and_run(&format!(
        "{PRE}(defn main [] (-> i64) (comptime (* (field-count Point) (variant-count Shape))))" // 9
    ));
    assert_eq!(code, 9);
}

#[test]
fn reflection_in_static_assert() {
    // A true reflection assert compiles; a false one fails the build.
    let ok = build_and_run(&format!(
        "{PRE}(static-assert (struct? Point) \"Point is a struct\")\n\
         (defn main [] (-> i64) 5)"
    ));
    assert_eq!(ok, 5);
    let err = coil::emit_ir(&format!(
        "{PRE}(static-assert (struct? Shape) \"Shape is not a struct\")\n\
         (defn main [] (-> i64) 0)"
    ))
    .unwrap_err();
    assert!(err.contains("static assertion failed"), "got:\n{err}");
}

#[test]
fn field_count_of_non_struct_errors() {
    let err = coil::emit_ir(&format!(
        "{PRE}(defn main [] (-> i64) (field-count Shape))" // Shape is a sum
    ))
    .unwrap_err();
    assert!(err.contains("not a struct"), "got:\n{err}");
}
