//! lib/fmt.coil — formatting over the IO Writer capability (signed/hex/bool/char/
//! line). Output is captured from a real `(stdout)` writer; IO stays explicit.

mod common;
use common::build_and_capture;

const IMPORT: &str = concat!(
    "(module app)\n",
    "(import \"lib/fmt.coil\" :use *)\n",
    "(import \"lib/io.coil\" :use *)\n",
    "(import \"lib/result.coil\" :use *)\n",
);

fn out(body: &str) -> String {
    let (_code, s) = build_and_capture(&format!("{IMPORT}{body}"));
    s
}

#[test]
fn signed_decimal() {
    let s = out(
        r#"(defn main [] (-> :i64)
             (let [w (stdout)]
               (print-i w -42) (print-char w 32) (print-i w 0) (print-char w 32) (print-i w 7) 0))"#,
    );
    assert_eq!(s, "-42 0 7");
}

#[test]
fn hex_lowercase() {
    let s = out(
        r#"(defn main [] (-> :i64)
             (let [w (stdout)] (print-hex w 255) (print-char w 32) (print-hex w 16) 0))"#,
    );
    assert_eq!(s, "ff 10");
}

#[test]
fn booleans() {
    let s = out(
        r#"(defn main [] (-> :i64)
             (let [w (stdout)] (print-bool w true) (print-char w 32) (print-bool w false) 0))"#,
    );
    assert_eq!(s, "true false");
}

#[test]
fn format_string_macro() {
    // The positional (fmt …) macro expands to a sequence of print-* calls; specs
    // d/s/x/c/b and {{ }} escapes. Pure library macro over str-bytes/bytes->str.
    let s = out(
        r#"(defn main [] (-> :i64)
             (let [w (stdout)]
               (fmt w "n={d} hex={x} s={s} ok={b} {{lit}}" -42 255 "hi" true) 0))"#,
    );
    assert_eq!(s, "n=-42 hex=ff s=hi ok=true {lit}");
}

#[test]
fn format_string_too_few_args_is_compile_error() {
    let err = coil::check_source(&format!(
        "{IMPORT}(defn main [] (-> :i64) (let [w (stdout)] (fmt w \"{{d}} {{d}}\" 1) 0))"
    ))
    .unwrap_err();
    assert!(err.contains("not enough arguments"), "got: {err}");
}

#[test]
fn format_string_unknown_spec_is_compile_error() {
    let err = coil::check_source(&format!(
        "{IMPORT}(defn main [] (-> :i64) (let [w (stdout)] (fmt w \"{{q}}\" 1) 0))"
    ))
    .unwrap_err();
    assert!(err.contains("unknown spec"), "got: {err}");
}

#[test]
fn lines_and_newline() {
    let s = out(
        r#"(defn main [] (-> :i64)
             (let [w (stdout)] (print-line w "a") (print-str w "b") (print-nl w) 0))"#,
    );
    assert_eq!(s, "a\nb\n");
}

#[test]
fn print_uhex_covers_the_full_unsigned_range() {
    // print-uhex (over udiv/urem) prints the unsigned hex of any i64, including
    // values with the high bit set — unlike print-hex, which assumes non-negative.
    let (_, out) = build_and_capture(&format!(
        "{IMPORT}(defn main [] (-> :i64)\n\
           (do (print-uhex (stdout) -1) (print-char (stdout) 32)\n\
               (print-uhex (stdout) 0) (print-char (stdout) 32)\n\
               (print-uhex (stdout) 255) 0))"
    ));
    assert_eq!(out, "ffffffffffffffff 0 ff");
}
