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
fn lines_and_newline() {
    let s = out(
        r#"(defn main [] (-> :i64)
             (let [w (stdout)] (print-line w "a") (print-str w "b") (print-nl w) 0))"#,
    );
    assert_eq!(s, "a\nb\n");
}
