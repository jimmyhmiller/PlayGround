//! Runtime s-expression reader (lib/sexp.coil): `read-all` parses `(slice u8)`
//! text into a tree of `Sexp` (numbers, symbols, lists), with accessors so a
//! consumer never matches the representation. The thing every Lisp/config/format
//! parser used to rewrite by hand — now a library (see examples/lisp.coil).

mod common;
use common::build_and_run;

const IMPORT: &str = "(module app)\n\
    (import \"lib/alloc.coil\" :use *)\n\
    (import \"lib/result.coil\" :use *)\n\
    (import \"lib/slice.coil\" :use *)\n\
    (import \"lib/str.coil\" :use *)\n\
    (import \"lib/arraylist.coil\" :use *)\n\
    (import \"lib/sexp.coil\" :use *)\n";

fn run_with(body: &str) -> i32 {
    build_and_run(&format!("{IMPORT}{body}"))
}

#[test]
fn reads_a_number_atom() {
    assert_eq!(
        run_with("(defn main [] (-> i64)\n  (let [a (malloc-allocator) fs (read-all a \"42\")]\n    (sexp-num (al-get [(ptr Sexp)] (load fs) 0))))"),
        42,
    );
}

#[test]
fn reads_a_nested_list_and_counts_it() {
    // (+ 1 (* 2 3)) has 3 top-level items; element 2 is itself a 3-item list.
    let body = "(defn main [] (-> i64)\n\
                  (let [a (malloc-allocator) fs (read-all a \"(+ 1 (* 2 3))\")]\n\
                    (let [top (al-get [(ptr Sexp)] (load fs) 0)]\n\
                      (iadd (imul (sexp-count top) 10) (sexp-count (sexp-nth top 2))))))"; // 3*10+3 = 33
    assert_eq!(run_with(body), 33);
}

#[test]
fn multiple_top_level_forms() {
    // two forms: "10" and "(a b)"  -> count = 2
    assert_eq!(
        run_with("(defn main [] (-> i64)\n  (let [a (malloc-allocator) fs (read-all a \"10 (a b)\")]\n    (al-len [(ptr Sexp)] (load fs))))"),
        2,
    );
}

#[test]
fn symbol_dispatch_and_negative_numbers() {
    // head symbol test + a negative numeric atom.
    let body = "(defn main [] (-> i64)\n\
                  (let [a (malloc-allocator) fs (read-all a \"(neg -7)\")]\n\
                    (let [top (al-get [(ptr Sexp)] (load fs) 0)]\n\
                      (if (sexp-sym-is (sexp-nth top 0) \"neg\")\n\
                          (sexp-num (sexp-nth top 1)) 0))))"; // -7
    assert_eq!(run_with(body), -7i32 as u8 as i32 & 0xff); // exit code is the low byte of -7
}

#[test]
fn atom_kind_predicates() {
    // read "(x 5)": elem 0 is a symbol, elem 1 is a number.
    let body = "(defn main [] (-> i64)\n\
                  (let [a (malloc-allocator) fs (read-all a \"(x 5)\")]\n\
                    (let [top (al-get [(ptr Sexp)] (load fs) 0)]\n\
                      (iadd (if (sexp-sym? (sexp-nth top 0)) 1 0)\n\
                            (if (sexp-num? (sexp-nth top 1)) 10 0)))))"; // 11
    assert_eq!(run_with(body), 11);
}
