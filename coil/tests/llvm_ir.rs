//! The `(llvm-ir ...)` raw-IR escape hatch + `(vec T N)` SIMD vectors. One
//! general primitive: any LLVM instruction/intrinsic is reachable without
//! per-opcode compiler support, type-checked at the boundary and verified by
//! LLVM. SIMD is then a macro library (`lib/simd.coil`).

mod common;
use common::build_and_run;

/// A scalar `add` straight through the raw-IR primitive: $ret/$N placeholders,
/// inlined helper, constant-folded by -O3.
#[test]
fn scalar_llvm_ir_add() {
    let src = r#"
        (defn main [] (-> i64)
          (llvm-ir i64 [40 2] "%r = add $ret $0, $1
        ret $ret %r"))
    "#;
    assert_eq!(build_and_run(src), 42);
}

/// Build a `<2 x i64>` with insertelement, read the lanes back, sum them —
/// exercises `Type::Vec` through a value, an operand, and the result type.
#[test]
fn vector_build_extract_sum() {
    let src = r#"
        (defn main [] (-> i64)
          (let [v (llvm-ir (vec i64 2) [40 2]
                    "%a = insertelement <2 x i64> undef, $t0 $0, i32 0
        %b = insertelement <2 x i64> %a, $t1 $1, i32 1
        ret <2 x i64> %b")]
            (llvm-ir i64 [v]
              "%x = extractelement $t0 $0, i32 0
        %y = extractelement $t0 $0, i32 1
        %r = add i64 %x, %y
        ret i64 %r")))
    "#;
    assert_eq!(build_and_run(src), 42);
}

/// A vector value round-trips through stack memory: `alloc-stack`, `store!`,
/// `load` of a `(vec i64 2)` (these come for free from `Type::Vec` in codegen).
#[test]
fn vector_load_store() {
    let src = r#"
        (defn main [] (-> i64)
          (let [p (alloc-stack (vec i64 2))]
            (store! p (llvm-ir (vec i64 2) [30 12]
              "%a = insertelement <2 x i64> undef, $t0 $0, i32 0
        %b = insertelement <2 x i64> %a, $t1 $1, i32 1
        ret <2 x i64> %b"))
            (llvm-ir i64 [(load p)]
              "%x = extractelement $t0 $0, i32 0
        %y = extractelement $t0 $0, i32 1
        %r = add i64 %x, %y
        ret i64 %r")))
    "#;
    assert_eq!(build_and_run(src), 42);
}

/// An intrinsic call (`@llvm.vector.reduce.add`) — the `declare` line is hoisted
/// to module scope by the primitive.
#[test]
fn vector_reduce_intrinsic() {
    let src = r#"
        (defn main [] (-> i64)
          (let [v (llvm-ir (vec i64 4) []
                    "ret <4 x i64> <i64 10, i64 20, i64 3, i64 9>")]
            (llvm-ir i64 [v]
              "declare i64 @llvm.vector.reduce.add.v4i64(<4 x i64>)
        %r = call i64 @llvm.vector.reduce.add.v4i64($t0 $0)
        ret i64 %r")))
    "#;
    assert_eq!(build_and_run(src), 42); // 10+20+3+9
}

/// SIMD via the macro library: a 4-wide dot product over `lib/simd.coil`. The
/// whole vocabulary is macros over `(llvm-ir ...)` — nothing in the compiler.
#[test]
fn simd_dot_product_via_library() {
    // dot([1,2,3,4],[2,2,2,2]) = 2+4+6+8 = 20
    let src = r#"
        (module test)
        (import "lib/simd.coil" :use *)
        (defn main [] (-> i64)
          (cast :i64 (dot4f (vec4f 1 2 3 4) (vec4f 2 2 2 2))))
    "#;
    assert_eq!(build_and_run(src), 20);
}

/// A C function (the murmur3 finalizer) lifted from `clang -O1 -emit-llvm` and
/// pasted verbatim into `(llvm-ir ...)`, producing the same value as C. clang's
/// body uses %0 (param) and %2.. (temps), which line up with the helper wrapper.
#[test]
fn c_function_embedded_via_raw_ir() {
    let src = r#"
        (defn mix [(x :i64)] (-> :i64)
          (llvm-ir i64 [x]
            "%2 = lshr i64 $0, 33
        %3 = xor i64 %2, $0
        %4 = mul i64 %3, -49064778989728563
        %5 = lshr i64 %4, 33
        %6 = xor i64 %5, %4
        ret i64 %6"))
        (defn main [] (-> i64) (mix 42))
    "#;
    // mix(42) in C is 16386023354290304276; the process exit code is its low 8 bits.
    assert_eq!(build_and_run(src), (16386023354290304276u64 & 0xFF) as i32);
}
