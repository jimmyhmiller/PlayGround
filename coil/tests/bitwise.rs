//! Bitwise and shift operators on integers.

mod common;
use common::build_and_run;

#[test]
fn and_or_xor() {
    let src = "(defn main [] (-> i64) (iadd (iand 255 (ior (ishl 1 5) 8)) (ixor 3 1)))";
    assert_eq!(build_and_run(src), 42); // (255 & (32|8)) + (3^1) = 40 + 2
}

#[test]
fn shifts_respect_signedness() {
    // logical shift right for unsigned, arithmetic for signed.
    let u = "(defn main [] (-> i64) (cast i64 (ishr (cast u8 200) 2)))"; // 200>>2 = 50
    let s = "(defn main [] (-> i64) (cast i64 (ishr (cast i8 200) 2)))"; // (-56)>>2 = -14
    assert_eq!(build_and_run(u), 50);
    assert_eq!(build_and_run(s), 256 - 14); // exit code is low 8 bits of -14
}

#[test]
fn complement_masks() {
    let src = "(defn main [] (-> i64) (cast i64 (iand (cast u8 255) (inot (cast u8 7)))))";
    assert_eq!(build_and_run(src), 248); // 255 & ~7
}

#[test]
fn bitwise_in_static_assert() {
    let src = r#"
        (static-assert (icmp-eq (iand 12 10) 8) "12 & 10 == 8")
        (static-assert (icmp-eq (ishl 1 4) 16) "1 << 4 == 16")
        (defn main [] (-> i64) (ior 40 2))
    "#;
    assert_eq!(build_and_run(src), 42);
}
