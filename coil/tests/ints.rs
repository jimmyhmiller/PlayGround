//! Arbitrary-width integers (Zig-style uN/iN) and signed/unsigned semantics.

mod common;
use common::build_and_run;

#[test]
fn arbitrary_widths_truncate() {
    assert_eq!(build_and_run("(defn main [] (-> :i64) (cast :i64 (cast :u2 7)))"), 3); // 7 mod 4
    assert_eq!(build_and_run("(defn main [] (-> :i64) (cast :i64 (cast :u7 200)))"), 72); // 200 mod 128
    assert_eq!(build_and_run("(defn main [] (-> :i64) (cast :i64 (cast :u23 42)))"), 42);
}

#[test]
fn unsigned_vs_signed_compare() {
    // 200 is > 100 as u8, but (= -56) < 100 as i8.
    let u = "(defn main [] (-> :i64) (if (icmp-gt (cast :u8 200) (cast :u8 100)) 42 0))";
    let s = "(defn main [] (-> :i64) (if (icmp-gt (cast :i8 200) (cast :i8 100)) 42 0))";
    assert_eq!(build_and_run(u), 42);
    assert_eq!(build_and_run(s), 0);
}

#[test]
fn unsigned_vs_signed_divide() {
    // 200/50 = 4 unsigned; signed (200 = -56) gives a different result.
    let u = "(defn main [] (-> :i64) (cast :i64 (idiv (cast :u8 200) (cast :u8 50))))";
    assert_eq!(build_and_run(u), 4);
}

#[test]
fn zero_extend_for_unsigned() {
    // u8 100 widened to i64 zero-extends to 100.
    assert_eq!(build_and_run("(defn main [] (-> :i64) (cast :i64 (cast :u8 100)))"), 100);
}

#[test]
fn arbitrary_width_struct_field() {
    let src = r#"
        (defstruct Bits [(small :u12) (big :i64)])
        (defn main [] (-> :i64)
          (let [p (alloc-stack Bits)]
            (store! (field p small) (cast :u12 40))
            (store! (field p big) 2)
            (iadd (cast :i64 (load (field p small))) (load (field p big)))))
    "#;
    assert_eq!(build_and_run(src), 42);
}

#[test]
fn rejects_mixed_signedness() {
    let src = "(defn main [] (-> :i64) (iadd (cast :u8 1) (cast :i8 1)))";
    assert!(coil::check_source(src).unwrap_err().contains("mixed signedness"));
}

#[test]
fn rejects_mixed_width() {
    let src = "(defn main [] (-> :i64) (iadd (cast :i32 1) 2))";
    assert!(coil::check_source(src).unwrap_err().contains("mixed widths"));
}
