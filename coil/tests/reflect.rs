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

// ---- per-field reflection: field-name (comptime string) + field-type-kind ------

#[test]
fn field_type_kind_counts_int_fields() {
    // Count integer fields of a mixed struct at compile time (kind 0 = int).
    let code = build_and_run(
        "(module a)\n\
         (defstruct Mix [(x i64) (y f64) (z i64) (b bool)])\n\
         (const N (comptime (let [(mut i) 0 (mut n) 0]\n\
           (loop (if (>= (load i) (field-count Mix)) (break)\n\
                   (do (if (= (field-type-kind Mix (load i)) 0) (store! n (+ (load n) 1)) 0)\n\
                       (store! i (+ (load i) 1)))))\n\
           (load n))))\n\
         (defn main [] (-> i64) N)", // x,z are i64 -> 2
    );
    assert_eq!(code, 2);
}

#[test]
fn field_name_is_a_comptime_string() {
    let code = build_and_run(
        "(module a)\n\
         (import \"lib/slice.coil\" :use *)\n\
         (defstruct Point [(xx i64) (yyy i64)])\n\
         (defn main [] (-> i64)\n\
           (+ (slice-len (comptime (field-name Point 0)))\n\
              (slice-len (comptime (field-name Point 1)))))", // len("xx")=2 + len("yyy")=3 = 5
    );
    assert_eq!(code, 5);
}

#[test]
fn compile_time_field_metadata_table() {
    // The payoff: a runtime field-metadata table (names + kinds) generated at
    // compile time and emitted as a constant global.
    let code = build_and_run(
        "(module a)\n\
         (import \"lib/slice.coil\" :use *)\n\
         (defstruct Mix [(a i64) (b f64) (c i64)])\n\
         (defstruct FieldDesc [(name (slice u8)) (kind i64)])\n\
         (const FIELDS\n\
           (comptime (let [(mut t) (zeroed (array FieldDesc 3)) (mut i) 0]\n\
             (loop (if (>= (load i) (field-count Mix)) (break)\n\
                     (do (store! (field (index (mut t) (load i)) name) (field-name Mix (load i)))\n\
                         (store! (field (index (mut t) (load i)) kind) (field-type-kind Mix (load i)))\n\
                         (store! i (+ (load i) 1)))))\n\
             (load t))))\n\
         (defn main [] (-> i64)\n\
           (+ (* (load (field (index FIELDS 1) kind)) 10)\n\
              (slice-len (load (field (index FIELDS 1) name)))))", // b: kind 1, len 1 => 11
    );
    assert_eq!(code, 11);
}
