//! Struct-by-value across the C boundary, ABI-correct for both x86-64 SysV and
//! AArch64 AAPCS64.
//!
//! The end-to-end tests actually *call* C functions that take/return structs by
//! value — libc's `div`, plus a small C helper we compile and link — and check
//! the numeric result, so a wrong coercion (passing a pointer where the ABI wants
//! registers, mis-sized eightbytes, wrong sret/byval) would corrupt the answer.
//!
//! The host (arm64) path runs natively. The x86-64 SysV path is cross-compiled
//! and run under Rosetta when available; either way the emitted IR is also
//! diffed against the shape clang produces (see `emits_abi_*`).

mod common;
use common::{build_and_run, build_link_run, rosetta_available, CrossArch};

const HELPER: &str = "tests/fixtures/struct_abi_helper.c";

// --- real libc: div() returns div_t { int quot; int rem; } by value ---------

#[test]
fn libc_div_returns_struct_by_value() {
    // div(42, 5) -> { quot: 8, rem: 2 }; encode as quot*10 + rem = 82.
    let src = r#"
        (defstruct DivT [(quot :i32) (rem :i32)])
        (extern div [:i32 :i32] (-> DivT))
        (defn main [] (-> :i64)
          (let [d (div 42 5)]
            (iadd (imul (cast :i64 (load (field d quot))) 10)
                  (cast :i64 (load (field d rem))))))
    "#;
    assert_eq!(build_and_run(src), 82);
}

// --- linked C helper: small (<=16B) and large (>16B), arg and return ---------

/// One Coil program exercising every position: returning a small struct,
/// passing it back by value, returning a large struct (sret), passing it back
/// (byval), a small struct that is both an argument *and* a return, an HFA, and
/// a large-in/large-out call.
const ROUNDTRIP_SRC: &str = r#"
    (defstruct Small [(a :i32) (b :i32)])
    (defstruct Big   [(a :i64) (b :i64) (c :i64)])
    (defstruct Hfa2  [(x :f32) (y :f32)])

    (extern make_small  [:i32 :i32]    (-> Small))
    (extern sum_small   [Small]        (-> :i32))
    (extern scale_small [Small :i32]   (-> Small))
    (extern make_big    [:i64 :i64 :i64] (-> Big))
    (extern sum_big     [Big]          (-> :i64))
    (extern add_big     [Big Big]      (-> Big))
    (extern make_hfa2   [:f32 :f32]    (-> Hfa2))
    (extern sum_hfa2    [Hfa2]         (-> :f32))

    (defn main [] (-> :i64)
      (let [s  (make_small 30 12)          ; {30,12}
            ss (sum_small s)               ; 42
            s2 (scale_small s 2)           ; {60,24}  (small in AND out)
            ss2 (sum_small s2)             ; 84
            g  (make_big 10 20 12)         ; {10,20,12}  (sret return)
            gs (sum_big g)                 ; 42         (byval arg)
            g2 (add_big g g)               ; {20,40,24} (byval x2 -> sret)
            gs2 (sum_big g2)               ; 84
            h  (make_hfa2 1.5 2.5)         ; HFA return
            hs (cast :i64 (sum_hfa2 h))]   ; 4
        ; ss(42) + (ss2-42 = 42) + (gs2-gs = 42) + hs(4) = 130
        (iadd (cast :i64 ss)
          (iadd (cast :i64 (isub ss2 42))
            (iadd (isub gs2 gs) hs)))))
"#;

#[test]
fn roundtrip_structs_native_host() {
    assert_eq!(build_link_run(ROUNDTRIP_SRC, HELPER, CrossArch::Host), 130);
}

#[test]
fn roundtrip_structs_x86_64_sysv_under_rosetta() {
    if !rosetta_available() {
        eprintln!("skipping: Rosetta 2 not available to run the x86-64 SysV binary");
        return;
    }
    assert_eq!(build_link_run(ROUNDTRIP_SRC, HELPER, CrossArch::X86_64), 130);
}

// --- the other direction: a C `main` calls Coil functions returning structs ---

/// Coil library (no `main`) whose functions return structs by value; the C
/// fixture's `main` calls them and checks the result, so a C *caller* depends on
/// Coil's emitted code honoring the platform struct ABI (both small/[2 x i64] and
/// large/sret returns).
const COIL_LIB_SRC: &str = r#"
    (defstruct Point  [(x :i64) (y :i64)])
    (defstruct Triple [(a :i64) (b :i64) (c :i64)])
    (defn coil_make_point [(x :i64) (y :i64)] (-> Point)
      (let [p (alloc-stack Point)]
        (store! (field p x) x) (store! (field p y) y) (load p)))
    (defn coil_make_triple [(a :i64) (b :i64) (c :i64)] (-> Triple)
      (let [t (alloc-stack Triple)]
        (store! (field t a) a) (store! (field t b) b) (store! (field t c) c)
        (load t)))
"#;

const CALLS_COIL: &str = "tests/fixtures/calls_coil_main.c";

#[test]
fn c_calls_coil_returning_structs_native_host() {
    assert_eq!(build_link_run(COIL_LIB_SRC, CALLS_COIL, CrossArch::Host), 42);
}

#[test]
fn c_calls_coil_returning_structs_x86_64_under_rosetta() {
    if !rosetta_available() {
        eprintln!("skipping: Rosetta 2 not available to run the x86-64 SysV binary");
        return;
    }
    assert_eq!(build_link_run(COIL_LIB_SRC, CALLS_COIL, CrossArch::X86_64), 42);
}

/// `(export-c …)` end to end: C calls Coil functions by their exported C symbols (one
/// renamed via `:as`), while a private helper internalizes. The library has no `main`
/// but the exports are the anchor (docs/SYMBOL_EXPORT.md).
const EXPORTED_LIB_SRC: &str = r#"
    (module shapes)
    (defstruct Point [(x :i64) (y :i64)])
    (defn clamp0 [(n :i64)] (-> :i64) (if (icmp-lt n 0) 0 n))   ; private -> internal
    (defn make-point [(x :i64) (y :i64)] (-> Point)
      (let [p (alloc-stack Point)]
        (store! (field p x) (clamp0 x)) (store! (field p y) (clamp0 y)) (load p)))
    (defn add [(a :i64) (b :i64)] (-> :i64) (iadd a b))
    (defn dist2 [(p Point)] (-> :i64)                          ; struct BY VALUE -> thunk
      (iadd (imul (load (field p x)) (load (field p x)))
            (imul (load (field p y)) (load (field p y)))))
    (export-c
      [make-point :as "shapes_make_point"]
      [add :as "shapes_add"]
      [dist2 :as "shapes_dist2"])
"#;
const CALLS_EXPORTED: &str = "tests/fixtures/calls_exported_coil.c";

#[test]
fn c_calls_exported_coil_functions() {
    assert_eq!(build_link_run(EXPORTED_LIB_SRC, CALLS_EXPORTED, CrossArch::Host), 0);
}

/// The same, cross-compiled to x86-64 (SysV ABI) and run under Rosetta — verifies the
/// export thunk's struct-by-value marshalling on a *different* ABI than the host's.
#[test]
fn c_calls_exported_coil_functions_x86_64_under_rosetta() {
    if !rosetta_available() {
        eprintln!("skipping: Rosetta 2 not available to run the x86-64 SysV binary");
        return;
    }
    assert_eq!(build_link_run(EXPORTED_LIB_SRC, CALLS_EXPORTED, CrossArch::X86_64), 0);
}

// --- IR shape vs clang (no execution needed; pure ABI verification) ----------

/// Emit IR for `src` targeting `triple` and return the `declare`/`define` lines.
fn decls(src: &str, triple: &str) -> Vec<String> {
    coil::emit_ir_for(src, triple)
        .expect("emit_ir_for")
        .lines()
        .filter(|l| l.starts_with("declare") || l.starts_with("define"))
        .filter(|l| !l.contains("llvm.memcpy"))
        .map(|l| l.to_string())
        .collect()
}

#[test]
fn emits_abi_x86_64_sysv() {
    let src = r#"
        (defstruct Small [(a :i32) (b :i32)])
        (defstruct Big   [(a :i64) (b :i64) (c :i64)])
        (defstruct Hfa2  [(x :f32) (y :f32)])
        (defstruct Pair16 [(a :i64) (b :i64)])
        (extern ret_small  [] (-> Small))
        (extern take_small [Small] (-> :i32))
        (extern ret_big    [] (-> Big))
        (extern take_big   [Big] (-> :i64))
        (extern ret_hfa2   [] (-> Hfa2))
        (extern take_hfa2  [Hfa2] (-> :f32))
        (extern ret_pair16  [] (-> Pair16))
        (extern take_pair16 [Pair16] (-> :i64))
        (defn main [] (-> :i64) 0)
    "#;
    let d = decls(src, "x86_64-apple-macosx11.0.0");
    let has = |needle: &str| d.iter().any(|l| l.contains(needle));
    // Small (8B, one INTEGER eightbyte): returned i64, passed i64.
    assert!(has("i64 @ret_small()"), "{d:?}");
    assert!(has("@take_small(i64)"), "{d:?}");
    // Big (24B): sret return, byval argument.
    assert!(has("void @ret_big(ptr sret(%Big) align 8)"), "{d:?}");
    assert!(has("@take_big(ptr byval(%Big) align 8)"), "{d:?}");
    // Hfa2 (two floats, one SSE eightbyte): <2 x float> both ways.
    assert!(has("<2 x float> @ret_hfa2()"), "{d:?}");
    assert!(has("@take_hfa2(<2 x float>)"), "{d:?}");
    // Pair16 (exactly 16B = two INTEGER eightbytes, the boundary case):
    // returned in RAX:RDX as { i64, i64 }, passed in two integer registers.
    // Matches `clang -arch x86_64` byte-for-byte.
    assert!(has("{ i64, i64 } @ret_pair16()"), "{d:?}");
    assert!(has("@take_pair16(i64, i64)"), "{d:?}");
}

#[test]
fn emits_abi_aarch64_aapcs64() {
    let src = r#"
        (defstruct Small [(a :i32) (b :i32)])
        (defstruct Big   [(a :i64) (b :i64) (c :i64)])
        (defstruct Hfa2  [(x :f32) (y :f32)])
        (defstruct Pair16 [(a :i64) (b :i64)])
        (extern ret_small  [] (-> Small))
        (extern take_small [Small] (-> :i32))
        (extern ret_big    [] (-> Big))
        (extern take_big   [Big] (-> :i64))
        (extern ret_hfa2   [] (-> Hfa2))
        (extern take_hfa2  [Hfa2] (-> :f32))
        (extern ret_pair16  [] (-> Pair16))
        (extern take_pair16 [Pair16] (-> :i64))
        (defn main [] (-> :i64) 0)
    "#;
    let d = decls(src, "arm64-apple-macosx11.0.0");
    let has = |needle: &str| d.iter().any(|l| l.contains(needle));
    // Small (8B int composite): returned i64, passed i64.
    assert!(has("i64 @ret_small()"), "{d:?}");
    assert!(has("@take_small(i64)"), "{d:?}");
    // Big (24B): indirect — sret return, plain pointer argument (no byval).
    assert!(has("void @ret_big(ptr sret(%Big) align 8)"), "{d:?}");
    assert!(has("@take_big(ptr)") && !has("@take_big(ptr byval"), "{d:?}");
    // Hfa2 (HFA): returned as the struct type, passed as [2 x float].
    assert!(has("%Hfa2 @ret_hfa2()"), "{d:?}");
    assert!(has("@take_hfa2([2 x float])"), "{d:?}");
    // Pair16 (exactly 16B composite, the boundary case): packed into two X
    // registers as [2 x i64] both ways. Matches `clang -arch arm64`.
    assert!(has("[2 x i64] @ret_pair16()"), "{d:?}");
    assert!(has("@take_pair16([2 x i64])"), "{d:?}");
}

// --- a struct shape that is NOT classifiable raises a clear error ------------

#[test]
fn unsupported_shape_is_a_clear_error_not_a_silent_pointer() {
    // A by-value struct returned across a *non-C* native convention has no
    // defined struct ABI; codegen must reject it loudly rather than silently
    // passing a pointer (which would miscompile).
    let src = r#"
        (defcc weird :native fast)
        (defstruct S [(a :i64) (b :i64) (c :i64)])
        (extern thing :cc weird [] (-> S))
        (defn main [] (-> :i64) 0)
    "#;
    let err = coil::emit_ir_for(src, "x86_64-apple-macosx11.0.0")
        .expect_err("a by-value struct on a non-C convention must be rejected");
    assert!(
        err.contains("C convention") || err.contains("C ABI"),
        "error should explain the missing C ABI, got: {err}"
    );
}

#[test]
fn rejects_by_value_slice_at_c_extern_on_all_targets() {
    // A slice is a Coil view type with no C representation: passing one BY VALUE
    // to/from a C extern must be REJECTED, not silently crossed as {ptr,len}. The
    // guard is target-independent — verify the host (arm64) AND x86 SysV.
    let param = r#"(module app)
        (extern takes_slice :cc c [(slice u8)] (-> :i64))
        (defn main [] (-> :i64) 0)"#;
    let ret = r#"(module app)
        (extern makes_slice :cc c [] (-> (slice u8)))
        (defn main [] (-> :i64) 0)"#;
    for src in [param, ret] {
        let host = coil::emit_ir(src).expect_err("host must reject a by-value slice at a C extern");
        assert!(host.contains("slice cannot cross the C ABI"), "host: {host}");
        let x86 = coil::emit_ir_for(src, "x86_64-apple-macosx11.0.0")
            .expect_err("x86 must reject a by-value slice at a C extern");
        assert!(x86.contains("slice cannot cross the C ABI"), "x86: {x86}");
    }
}

#[test]
fn rejects_slice_field_of_byvalue_struct_at_c_extern_on_aarch64() {
    // A slice nested in a by-value struct field would be waved through by AArch64's
    // size-based classification; the pre-classification guard catches it on the host.
    let src = r#"(module app)
        (defstruct Wrap [(s (slice u8)) (n i64)])
        (extern takes_wrap :cc c [Wrap] (-> :i64))
        (defn main [] (-> :i64) 0)"#;
    let err = coil::emit_ir(src).expect_err("a slice field must be rejected at the C boundary");
    assert!(err.contains("slice cannot cross the C ABI"), "got: {err}");
}

#[test]
fn rejects_by_value_vec_at_c_extern() {
    let src = r#"(module app)
        (extern takes_vec :cc c [(vec f32 4)] (-> :i64))
        (defn main [] (-> :i64) 0)"#;
    let err = coil::emit_ir(src).expect_err("a by-value vec must be rejected at the C boundary");
    assert!(err.contains("vec cannot cross the C ABI"), "got: {err}");
}

#[test]
fn pointer_to_slice_crosses_c_extern_fine() {
    // Only BY-VALUE views are rejected; a pointer to a slice is a plain pointer.
    let src = r#"(module app)
        (extern takes_slice_ptr :cc c [(ptr (slice u8)) i64] (-> :i64))
        (defn main [] (-> :i64) 0)"#;
    assert!(coil::emit_ir(src).is_ok(), "a pointer to a slice must be allowed across C");
}
