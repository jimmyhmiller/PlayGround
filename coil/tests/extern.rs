//! `extern` + C interop: foreign declarations are type-checked and linked.

mod common;
use common::{build_and_capture, build_and_capture_args};

#[test]
fn putchar_prints_to_stdout() {
    let src = r#"
        (extern putchar :cc c [:i64] (-> :i64))
        (defn main [] (-> :i64)
          (do (putchar 72) (putchar 105) (putchar 33) (putchar 10) 0))
    "#;
    let (code, out) = build_and_capture(src);
    assert_eq!(code, 0);
    assert_eq!(out, "Hi!\n");
}

#[test]
fn prints_a_decimal_number() {
    let src = include_str!("../examples/extern.coil");
    let (code, out) = build_and_capture(src);
    assert_eq!(code, 0);
    assert_eq!(out, "12345\n");
}

#[test]
fn write_with_a_pointer_argument() {
    // Pack "Hi\n" into one i64 slot (little-endian) and hand a pointer to
    // write(2). Proves a pointer crossing the extern boundary works.
    // 'H'=72, 'i'=105, '\n'=10  ->  72 + 105*256 + 10*65536 = 682312
    let src = r#"
        (extern write :cc c [:i64 (ptr i64) :i64] (-> :i64))
        (defn main [] (-> :i64)
          (let [buf (alloc-static i64)]
            (store! buf 682312)
            (write 1 buf 3)
            0))
    "#;
    let (code, out) = build_and_capture(src);
    assert_eq!(code, 0);
    assert_eq!(out, "Hi\n");
}

#[test]
fn extern_erases_pointer_region() {
    // write's buf is declared (ptr i64), but a heap pointer is accepted at
    // the extern boundary (the foreign side doesn't track regions).
    let src = r#"
        (extern write :cc c [:i64 (ptr i64) :i64] (-> :i64))
        (defn main [] (-> :i64)
          (let [buf (alloc-heap i64)]
            (store! buf 682312)
            (write 1 buf 3)
            (free buf)
            0))
    "#;
    let (code, out) = build_and_capture(src);
    assert_eq!(code, 0);
    assert_eq!(out, "Hi\n");
}

#[test]
fn reads_argc_and_argv() {
    // main(argc: i32, argv: char**); print each arg with puts.
    let src = include_str!("../examples/args.coil");
    let (code, out) = build_and_capture_args(src, &["one", "two", "three"]);
    assert_eq!(code, 0);
    // argv[0] is the program path (varies); the rest are our args.
    assert!(out.ends_with("one\ntwo\nthree\n"), "got: {out:?}");
    assert_eq!(out.lines().count(), 4); // argv[0] + 3 args
}

#[test]
fn integer_widths_and_casts() {
    // Load an i32 through a typed pointer, widen to i64, use it.
    let src = r#"
        (defn main [] (-> :i64)
          (let [p (alloc-heap i64)]
            (store! p 42)
            (let [v (load p)] (free p) v)))
    "#;
    assert_eq!(build_and_capture(src).0, 42);

    // A truncate/extend round-trip: 300 -> i8 (300 & 0xff = 44) -> i64.
    let src2 = r#"
        (defn main [] (-> :i64)
          (cast :i64 (cast :i8 300)))
    "#;
    assert_eq!(build_and_capture(src2).0, 44);
}

#[test]
fn rejects_mixed_width_arithmetic() {
    let src = r#"
        (defn main [] (-> :i64)
          (iadd (cast :i32 1) 2))
    "#;
    assert!(coil::check_source(src).unwrap_err().contains("mixed widths"));
}

#[test]
fn rejects_wrong_argument_type_to_extern() {
    // putchar wants an i64; passing a pointer is a type error.
    let src = r#"
        (extern putchar :cc c [:i64] (-> :i64))
        (defn main [] (-> :i64)
          (let [p (alloc-heap i64)] (putchar p)))
    "#;
    let err = coil::check_source(src).unwrap_err();
    assert!(err.contains("argument 1 to 'putchar'"), "got: {err}");
}
