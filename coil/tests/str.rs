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
fn string_literal_backing_global_is_nul_terminated_for_c_apis() {
    // Regression: a "…" literal's backing global must carry a trailing NUL so its
    // `data` pointer is safe to hand to a C string API. Before the fix the global
    // was a bare [N x i8]; adjacent literals with no alignment padding ran into
    // one another, so `printf` on a 7-byte "g %lld\n" printed the NEXT literal too
    // (and ate a garbage vararg). `slice-len` still reports N — the NUL is not part
    // of the slice's extent.
    let src = r#"
        (module app)
        (extern printf :cc c [(ptr i8) ...] (-> :i32))
        (defn cstr [(s (slice u8))] (-> (ptr u8))
          (llvm-ir (ptr u8) [s] "%d = extractvalue $t0 $0, 0
ret $ret %d"))
        (defn go [(n :i64) (i :i64)] (-> :i64)
          (if (icmp-ge i n)
              (cast :i64 (printf (cstr "done\n")))
              (do (printf (cstr "g %lld\n") i)
                  (go n (iadd i 1)))))
        (defn main [] (-> :i64) (do (go 3 0) 0))
    "#;
    let (code, out) = build_and_capture(src);
    assert_eq!(code, 0);
    assert_eq!(
        out, "g 0\ng 1\ng 2\ndone\n",
        "a string literal bled into its neighbor — missing NUL terminator:\n{out}"
    );
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
