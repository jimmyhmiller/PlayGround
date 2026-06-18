//! Layout control: :layout c/packed/(align N), sizeof/alignof/offsetof, static-assert.

mod common;
use common::build_and_run;

#[test]
fn packed_vs_c_layout() {
    let src = r#"
        (defstruct C [(a :i8) (b :i64)])                   ; C ABI: size 16, b @ 8
        (defstruct P :layout packed [(a :i8) (b :i64)])    ; packed:  size 9,  b @ 1
        (static-assert (icmp-eq (sizeof C) 16) "C is 16")
        (static-assert (icmp-eq (sizeof P) 9)  "P is 9")
        (static-assert (icmp-eq (offsetof C b) 8) "C.b at 8")
        (static-assert (icmp-eq (offsetof P b) 1) "P.b at 1")
        (defn main [] (-> :i64) (iadd (sizeof C) (sizeof P)))   ; 25
    "#;
    assert_eq!(build_and_run(src), 25);
}

#[test]
fn alignof_and_sizeof() {
    let src = r#"
        (static-assert (icmp-eq (alignof :i64) 8) "i64 aligns to 8")
        (static-assert (icmp-eq (sizeof (array i32 4)) 16) "[4 x i32] is 16")
        (defn main [] (-> :i64) (sizeof :u8))   ; 1
    "#;
    assert_eq!(build_and_run(src), 1);
}

#[test]
fn packed_struct_round_trips() {
    // A packed struct still loads/stores correctly through its fields.
    let src = r#"
        (defstruct P :layout packed [(a :i8) (b :i64)])
        (defn main [] (-> :i64)
          (let [p (alloc-stack P)]
            (store! (field p a) (cast :i8 2))
            (store! (field p b) 40)
            (iadd (cast :i64 (load (field p a))) (load (field p b)))))
    "#;
    assert_eq!(build_and_run(src), 42);
}

#[test]
fn aligned_struct_allocates_aligned() {
    let src = r#"
        (defstruct Cache :layout (align 64) [(x :i64)])
        (defn main [] (-> :i64)
          (let [c (alloc-stack Cache)] (store! (field c x) 42) (load (field c x))))
    "#;
    assert_eq!(build_and_run(src), 42);
    assert!(coil::emit_ir(src).unwrap().contains("align 64"));
}

#[test]
fn static_assert_failure_is_a_compile_error() {
    // static-assert is evaluated during codegen, so emit_ir surfaces it.
    let src = r#"
        (defstruct P :layout packed [(a :i8) (b :i64)])
        (static-assert (icmp-eq (sizeof P) 10) "P should be 9 not 10")
        (defn main [] (-> :i64) 0)
    "#;
    let err = coil::emit_ir(src).unwrap_err();
    assert!(err.contains("P should be 9 not 10"), "got: {err}");
}

#[test]
fn explicit_offsets_with_padding() {
    let src = r#"
        (defstruct Reg :layout explicit :size 16
          [(status :u32 :at 0)
           (data   :i64 :at 8)])          ; bytes 4-7 are explicit padding
        (static-assert (icmp-eq (sizeof Reg) 16) "Reg is 16")
        (static-assert (icmp-eq (offsetof Reg data) 8) "data at 8")
        (defn main [] (-> :i64)
          (let [r (alloc-stack Reg)]
            (store! (field r data) 40)
            (store! (field r status) (cast :u32 2))
            (iadd (load (field r data)) (cast :i64 (load (field r status))))))
    "#;
    assert_eq!(build_and_run(src), 42);
}

#[test]
fn explicit_union_overlap() {
    // two fields at the same :at overlap — write a u32, read the low byte back.
    let src = r#"
        (defstruct U :layout explicit :size 4
          [(word  :u32         :at 0)
           (bytes (array u8 4) :at 0)])
        (defn main [] (-> :i64)
          (let [u (alloc-stack U)]
            (store! (field u word) (cast :u32 42))
            (cast :i64 (load (index (field u bytes) 0)))))
    "#;
    assert_eq!(build_and_run(src), 42);
}

#[test]
fn explicit_requires_at_on_every_field() {
    let src = r#"
        (defstruct Bad :layout explicit [(a :i64 :at 0) (b :i64)])
        (defn main [] (-> :i64) 0)
    "#;
    assert!(coil::check_source(src).unwrap_err().contains(":at"));
}

#[test]
fn bitfields_pack_and_mask() {
    // four fields packed into an i32; setting one must not corrupt the others,
    // and each field reads back as its uN value.
    let src = r#"
        (defstruct Reg :layout bits :backing :i32
          [(enable :bits 1) (mode :bits 3) (channel :bits 4) (rest :bits 24)])
        (static-assert (icmp-eq (sizeof Reg) 4) "Reg is 4 bytes")
        (defn main [] (-> :i64)
          (let [r (alloc-stack Reg)]
            (set! r channel (cast :u4 10))
            (set! r mode    (cast :u3 2))
            (set! r enable  (cast :u1 1))
            (iadd (imul (cast :i64 (get r channel)) 4)
                  (iadd (cast :i64 (get r mode)) (cast :i64 (get r enable))))))
    "#;
    assert_eq!(build_and_run(src), 43); // 40 + 2 + 1
}

#[test]
fn bitfield_default_backing() {
    // 12 bits with no :backing -> i16.
    let src = r#"
        (defstruct Flags :layout bits [(a :bits 4) (b :bits 8)])
        (static-assert (icmp-eq (sizeof Flags) 2) "12 bits -> 2 bytes")
        (defn main [] (-> :i64)
          (let [f (alloc-stack Flags)]
            (set! f b (cast :u8 200))
            (set! f a (cast :u4 2))
            (cast :i64 (get f b))))   ; 200 (a doesn't disturb b)
    "#;
    assert_eq!(build_and_run(src), 200);
}

#[test]
fn rejects_field_on_bit_struct() {
    let src = r#"
        (defstruct R :layout bits [(a :bits 4)])
        (defn main [] (-> :i64) (let [r (alloc-stack R)] (load (field r a))))
    "#;
    assert!(coil::check_source(src).unwrap_err().contains(":layout bits"));
}

#[test]
fn offsetof_on_generic_struct() {
    let src = r#"
        (defstruct Pair [A B] [(fst A) (snd B)])
        (static-assert (icmp-eq (offsetof (Pair i32 i64) snd) 8) "snd at 8")
        (defn main [] (-> :i64) (sizeof (Pair i32 i64)))   ; i32 + pad + i64 = 16
    "#;
    assert_eq!(build_and_run(src), 16);
}
