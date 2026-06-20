//! Strings as `(slice u8)` views (lib/str.coil over the core slice type): content
//! equality/hashing/search, allocator-aware concat, string-keyed HashMap via
//! `str-keyops`, and an owned/growable `StrBuf` (a library over ArrayList<u8>).
//! `"…"` literals are slices (no allocator); `c"…"` is the distinct FFI cstring.

mod common;
use common::{build_and_capture, build_and_run};

const IMPORT: &str = "(module app)\n\
    (import \"lib/slice.coil\" :use *)\n\
    (import \"lib/str.coil\" :use *)\n\
    (import \"lib/alloc.coil\" :use *)\n\
    (import \"lib/result.coil\" :use *)\n\
    (import \"lib/hashmap.coil\" :use *)\n";

fn run_with(body: &str) -> i32 {
    build_and_run(&format!("{IMPORT}{body}"))
}

#[test]
fn string_literal_is_a_slice_with_compile_time_length() {
    // "…" is a (slice u8) view: its length is carried, no allocator, no strlen.
    assert_eq!(run_with("(defn main [] (-> :i64) (str-len \"hello\"))"), 5);
}

#[test]
fn content_equality_and_starts_with() {
    let code = run_with(
        r#"(defn main [] (-> :i64)
             (iadd (if (str-eq "abc" "abc") 1 0)
               (iadd (if (str-eq "abc" "abd") 0 2)
                 (iadd (if (starts-with "hello world" "hello") 4 0)
                       (if (starts-with "hi" "hello") 0 8)))))"#,
    );
    assert_eq!(code, 15); // 1 + 2 + 4 + 8
}

#[test]
fn find_returns_index_or_none() {
    let code = run_with(
        r#"(defn main [] (-> :i64)
             (iadd (match (str-find "hello world" "world") (None [] -1) (Some [i] i))   ; 6
                   (match (str-find "abc" "zzz") (None [] 100) (Some [i] i))))"#,        // 100
    );
    assert_eq!(code, 106);
}

#[test]
fn concat_allocates_and_joins() {
    let (code, out) = build_and_capture(&format!(
        "{IMPORT}(import \"lib/io.coil\" :use *)\n\
         (defn main [] (-> :i64)\n\
           (let [a (malloc-allocator)]\n\
             (match (str-concat a \"foo\" \"bar\")\n\
               (None [] 1)\n\
               (Some [s] (do (print-str (stdout) s) 0)))))",
    ));
    assert_eq!(code, 0);
    assert_eq!(out, "foobar");
}

#[test]
fn string_keyed_hashmap_via_str_keyops() {
    // Content keys: an update by an equal-content key lands on the same entry.
    let code = run_with(
        r#"(defn main [] (-> :i64)
             (let [a (malloc-allocator) (mut m) (hm-new [(slice u8) i64] a (str-keyops))]
               (hm-put! (mut m) "alpha" 10)
               (hm-put! (mut m) "alpha" 100)
               (hm-put! (mut m) "beta" 20)
               (iadd (match (hm-get [(slice u8) i64] m "alpha") (None [] -1) (Some [v] v))
                     (match (hm-get [(slice u8) i64] m "beta")  (None [] -1) (Some [v] v)))))"#,
    );
    assert_eq!(code, 120); // 100 + 20 (alpha updated by content, not duplicated)
}

#[test]
fn owned_strbuf_over_arraylist() {
    let (code, out) = build_and_capture(&format!(
        "{IMPORT}(import \"lib/io.coil\" :use *)\n\
         (defn main [] (-> :i64)\n\
           (let [a (malloc-allocator) (mut sb) (sb-new a)]\n\
             (sb-push-str! (mut sb) \"hi\")\n\
             (sb-push-byte! (mut sb) (cast u8 33))\n\
             (sb-push-str! (mut sb) \"!\")\n\
             (print-str (stdout) (sb-str sb))\n\
             (let [n (str-len (sb-str sb))] (sb-free! (mut sb)) n)))",
    ));
    assert_eq!(out, "hi!!");
    assert_eq!(code, 4);
}

#[test]
fn cstring_is_distinct_ptr_i8_for_ffi() {
    // c"…" is a NUL-terminated (ptr i8); strlen sees the C length.
    let code = run_with(
        "(extern strlen :cc c [(ptr i8)] (-> :i64))\n\
         (defn main [] (-> :i64) (strlen c\"abcdef\"))",
    );
    assert_eq!(code, 6);
}
