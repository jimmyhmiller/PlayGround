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
    // Two *concrete* widths: a literal would just adopt the other's type, so
    // this uses casts on both sides to force a genuine mismatch.
    let src = "(defn main [] (-> :i64) (iadd (cast :i32 1) (cast :i64 2)))";
    assert!(coil::check_source(src).unwrap_err().contains("mixed widths"));
}

// --- bidirectional literal inference ----------------------------------------

#[test]
fn literal_adopts_operand_width() {
    // `x : u8`; the bare `1` is inferred as u8 (no cast needed). Result 41 fits.
    let src = r#"
        (defn main [] (-> :i64)
          (let [x (cast :u8 41)]
            (cast :i64 (iadd x 1))))
    "#;
    assert_eq!(build_and_run(src), 42);
}

#[test]
fn literal_stored_into_typed_pointer() {
    // (store! p 42) into a (ptr u8) — the literal adopts u8, no (cast :u8 ...).
    let src = r#"
        (defn main [] (-> :i64)
          (let [p (alloc-stack :u8)]
            (store! p 42)
            (cast :i64 (load p))))
    "#;
    assert_eq!(build_and_run(src), 42);
}

#[test]
fn literal_inferred_from_return_type() {
    // The function returns u8; the body literal is coerced to u8 at the boundary.
    let src = r#"
        (defn answer [] (-> :u8) 42)
        (defn main [] (-> :i64) (cast :i64 (answer)))
    "#;
    assert_eq!(build_and_run(src), 42);
}

#[test]
fn literal_inferred_through_if_branches() {
    // One branch is `(cast :u8 ...)`, the other a bare literal that adopts u8.
    let src = r#"
        (defn main [] (-> :i64)
          (cast :i64 (if 1 (cast :u8 42) 0)))
    "#;
    assert_eq!(build_and_run(src), 42);
}

#[test]
fn literal_inferred_as_call_argument() {
    let src = r#"
        (defn take :cc c [(x :u8)] (-> :i64) (cast :i64 x))
        (defn main [] (-> :i64) (take 42))
    "#;
    assert_eq!(build_and_run(src), 42);
}

#[test]
fn out_of_range_literal_is_rejected() {
    // 300 doesn't fit in u8 — even with inference, that's a compile error.
    let src = r#"
        (defn main [] (-> :i64)
          (let [p (alloc-stack :u8)] (store! p 300) (cast :i64 (load p))))
    "#;
    let err = coil::check_source(src).unwrap_err();
    assert!(err.contains("does not fit in u8"), "got: {err}");
}

#[test]
fn udiv_urem_force_unsigned_on_signed_operands() {
    // idiv/irem dispatch by operand type; udiv/urem ALWAYS interpret the bits as
    // unsigned, so a signed i64 with the high bit set divides as the large
    // positive value. -1 (0xFFFFFFFFFFFFFFFF) / 2^56 = 255 unsigned (idiv = 0).
    assert_eq!(build_and_run("(defn main [] (-> :i64) (udiv -1 72057594037927936))"), 255);
    assert_eq!(build_and_run("(defn main [] (-> :i64) (idiv -1 72057594037927936))"), 0);
    assert_eq!(build_and_run("(defn main [] (-> :i64) (urem -1 100))"), 15); // 0xFFFF…FFFF % 100 (signed: -1)
}

#[test]
fn idiv_on_unsigned_operands_is_unsigned() {
    // The existing type-dispatch path: idiv on u64 operands is unsigned.
    let src = "(defn main [] (-> :i64) (cast :i64 (idiv (cast u64 -1) (cast u64 72057594037927936))))";
    assert_eq!(build_and_run(src), 255);
}

#[test]
fn udiv_urem_static_assert_folds_unsigned() {
    // const-eval (static-assert) folds udiv/urem over the unsigned interpretation.
    let src = "(static-assert (icmp-eq (udiv -1 72057594037927936) 255) \"udiv folds unsigned\")\n\
               (defn main [] (-> :i64) 0)";
    assert_eq!(build_and_run(src), 0);
}

#[test]
fn const_eval_dispatches_div_and_cmp_by_operand_signedness() {
    // const_eval (static-assert) must fold idiv/irem/comparison with the SAME
    // signedness the runtime uses (by operand type), not signed-only — else a
    // const context silently diverges from runtime. (cast u64 -1) is the large
    // unsigned value: /2^56 = 255 unsigned (0 signed), and > 5 unsigned (not signed).
    let src = "(static-assert (icmp-eq (idiv (cast u64 -1) (cast u64 72057594037927936)) 255) \"udiv\")\n\
               (static-assert (icmp-gt (cast u64 -1) (cast u64 5)) \"ucmp\")\n\
               (defn main [] (-> :i64) 0)";
    assert_eq!(build_and_run(src), 0); // both asserts hold -> compiles + runs
}

#[test]
fn const_eval_false_unsigned_assert_still_fails() {
    // The unsigned fold must not silently pass a FALSE assertion. (static-asserts
    // are evaluated in codegen, so this needs emit_ir, not check_source.)
    let err = coil::emit_ir(
        "(static-assert (icmp-eq (idiv (cast u64 -1) (cast u64 72057594037927936)) 0) \"no\")\n\
         (defn main [] (-> :i64) 0)",
    )
    .unwrap_err();
    assert!(err.contains("static assertion failed"), "got: {err}");
}
