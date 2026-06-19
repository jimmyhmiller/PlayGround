//! C variadic externs (`...`): calling printf-style functions.

mod common;
use common::{build_and_capture, build_and_run};

#[test]
fn calls_printf_with_mixed_args() {
    let src = r#"
        (extern printf :cc c [(ptr i8) ...] (-> i32))
        (defn main [] (-> i64)
          (printf "int=%d str=%s char=%c float=%.1f\n" 42 "hi" 33 2.5)
          0)
    "#;
    let (code, out) = build_and_capture(src);
    assert_eq!(code, 0);
    assert_eq!(out, "int=42 str=hi char=! float=2.5\n");
}

#[test]
fn snprintf_returns_length() {
    // a variadic with a non-trivial fixed prefix, returning a useful value.
    let src = r#"
        (extern snprintf :cc c [(ptr i8) i64 (ptr i8) ...] (-> i32))
        (defn main [] (-> i64)
          (let [buf (alloc-stack (array i8 32))]
            (cast i64 (snprintf (cast (ptr i8) buf) 32 "%d-%d" 4 2))))  ; writes "4-2", len 3
    "#;
    assert_eq!(build_and_run(src), 3);
}

#[test]
fn rejects_too_few_fixed_args() {
    let src = r#"
        (extern printf :cc c [(ptr i8) ...] (-> i32))
        (defn main [] (-> i64) (printf))
    "#;
    assert!(coil::check_source(src).unwrap_err().contains("at least 1"));
}
