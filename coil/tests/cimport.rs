//! `coil cimport` (C-interop §6): generate Coil FFI bindings from a C header via clang's
//! AST — a REAL header import, ABI-faithful, REFUSING unmappable constructs (never a
//! silent-wrong binding). Gated on clang being installed (skips otherwise).

mod common;
use common::build_and_run;
use std::process::Command;

fn have_clang() -> bool {
    Command::new("clang").arg("--version").output().map(|o| o.status.success()).unwrap_or(false)
}

fn write_header(name: &str, contents: &str) -> String {
    let p = std::env::temp_dir().join(name);
    std::fs::write(&p, contents).expect("write header");
    p.to_str().unwrap().to_string()
}

#[test]
fn cimport_maps_scalars_and_struct_layout_abi_faithfully() {
    if !have_clang() {
        eprintln!("SKIP: clang not found");
        return;
    }
    let h = write_header("coil_ci_types.h", "struct Pt { int x; long y; };\nint add(int a, long b);\n");
    let b = coil::cimport::cimport_header(&h).expect("cimport");
    // clang's reported widths: int->i32, long->i64 (not assumptions).
    assert!(b.contains("(defstruct Pt [(x i32) (y i64)])"), "struct layout:\n{b}");
    assert!(b.contains("(extern add :cc c [i32 i64] (-> i32))"), "fn mapping:\n{b}");
}

#[test]
fn cimport_bindings_call_real_libc_abi_correct() {
    if !have_clang() {
        eprintln!("SKIP: clang not found");
        return;
    }
    // declare real libc functions, cimport them, then CALL them end-to-end.
    let h = write_header("coil_ci_libc.h", "double sqrt(double x);\nlong labs(long n);\n");
    let b = coil::cimport::cimport_header(&h).expect("cimport");
    assert!(b.contains("(extern sqrt :cc c [f64] (-> f64))"), "{b}");
    assert!(b.contains("(extern labs :cc c [i64] (-> i64))"), "{b}");
    // de-dup: clang lists builtin libc fns twice — we emit ONE.
    assert_eq!(b.matches("(extern sqrt").count(), 1, "sqrt must be emitted once:\n{b}");

    let bpath = std::env::temp_dir().join("coil_ci_libc_bindings.coil");
    std::fs::write(&bpath, &b).unwrap();
    let prog = format!(
        "(module app)\n(import \"{}\" :use *)\n\
         (defn main [] (-> i64) (iadd (cast i64 (sqrt 1764.0)) (isub (labs -7) 7)))",
        bpath.display()
    );
    // sqrt(1764)=42, labs(-7)-7=0 → 42 : the cimported bindings are ABI-correct.
    assert_eq!(build_and_run(&prog), 42);
}

#[test]
fn cimport_red_team_refuses_silent_wrong_bindings() {
    if !have_clang() {
        eprintln!("SKIP: clang not found");
        return;
    }
    // Red-team the exact ABI-corruption failure modes (the reviewer's concern):
    let h = write_header(
        "coil_ci_redteam.h",
        "struct Flags { int a : 3; int b : 5; int c; };\n\
         enum Color { RED, GREEN };\n\
         int paint(enum Color c);\n\
         long double precise(long double x);\n\
         _Bool is_ready(int x);\n",
    );
    let b = coil::cimport::cimport_header(&h).expect("cimport");
    // bitfields pack into bit ranges — emitting full-width fields would corrupt the
    // layout. The struct must be REFUSED, not mis-bound.
    assert!(!b.contains("(defstruct Flags"), "bitfield struct must be refused:\n{b}");
    // long double has no Coil type → refused (no silent f64 guess).
    assert!(!b.contains("(extern precise"), "long-double fn must be refused:\n{b}");
    // C `_Bool` is a 1-byte value at the ABI → u8, not Coil `bool` (i1, ambiguous width).
    assert!(b.contains("(extern is_ready :cc c [i32] (-> u8))"), "_Bool must map to u8:\n{b}");
    // Phase 2: an enum is no longer refused — it lowers to its integer width (`int`
    // → i32, the real C ABI), and its constants become `const` defs.
    assert!(b.contains("(extern paint :cc c [i32] (-> i32))"), "enum param → i32:\n{b}");
    assert!(b.contains("(const RED 0)"), "enum constant RED:\n{b}");
    assert!(b.contains("(const GREEN 1)"), "enum constant GREEN:\n{b}");
}

#[test]
fn cimport_resolves_typedefs_abi_faithfully() {
    if !have_clang() {
        eprintln!("SKIP: clang not found");
        return;
    }
    // typedefs (incl. typedef-of-typedef and typedef'd struct) resolve through
    // clang's desugaring to their real ABI width — not refused, not guessed.
    let h = write_header(
        "coil_ci_typedef.h",
        "typedef unsigned long size_t;\n\
         typedef size_t mysize;\n\
         typedef struct Point Point;\n\
         struct Point { int x; int y; };\n\
         mysize span(Point *p, size_t n);\n",
    );
    let b = coil::cimport::cimport_header(&h).expect("cimport");
    // struct Point still emitted once (the forward decl from the typedef must not
    // dedup it away).
    assert!(b.contains("(defstruct Point [(x i32) (y i32)])"), "struct:\n{b}");
    // size_t→u64, mysize→size_t→u64, Point*→(ptr Point).
    assert!(
        b.contains("(extern span :cc c [(ptr Point) u64] (-> u64))"),
        "typedef resolution:\n{b}"
    );
}

#[test]
fn cimport_enum_constants_and_defines_are_usable() {
    if !have_clang() {
        eprintln!("SKIP: clang not found");
        return;
    }
    // enum constants and object-like #define constants become `const` defs that a
    // Coil program can USE as bare names. Function-like / string macros are skipped.
    let h = write_header(
        "coil_ci_consts.h",
        "enum E { A, B = 10, C };\n\
         #define WIDTH 4\n\
         #define NAME \"coil\"\n\
         #define DOUBLE(x) ((x)+(x))\n",
    );
    let b = coil::cimport::cimport_header(&h).expect("cimport");
    assert!(b.contains("(const A 0)") && b.contains("(const B 10)") && b.contains("(const C 11)"), "enum:\n{b}");
    assert!(b.contains("(const WIDTH 4)"), "object-like #define:\n{b}");
    assert!(!b.contains("NAME"), "string macro must be skipped:\n{b}");
    assert!(!b.contains("DOUBLE"), "function-like macro must be skipped:\n{b}");

    // Use them end-to-end: C + WIDTH = 11 + 4 = 15, *? keep to exit code. B+C+WIDTH=10+11+4=25.
    let bpath = std::env::temp_dir().join("coil_ci_consts_bindings.coil");
    std::fs::write(&bpath, &b).unwrap();
    let prog = format!(
        "(module app)\n(import \"{}\" :use *)\n\
         (defn main [] (-> i64) (iadd (iadd B C) WIDTH))",
        bpath.display()
    );
    assert_eq!(build_and_run(&prog), 25, "cimported consts usable as values");
}

#[test]
fn cimport_plus_link_flag_calls_a_custom_c_library() {
    if !have_clang() {
        eprintln!("SKIP: clang not found");
        return;
    }
    // the full "drop-in for C" loop: a CUSTOM C library (not libc) — compile it, cimport
    // its header, then link + call it from Coil via the link-flag passthrough.
    let dir = std::env::temp_dir();
    let cfile = dir.join("coil_ci_foo.c");
    let hfile = dir.join("coil_ci_foo.h");
    let ofile = dir.join("coil_ci_foo.o");
    std::fs::write(&cfile, "int triple_it(int x){return x*3;}\n").unwrap();
    std::fs::write(&hfile, "int triple_it(int x);\n").unwrap();
    assert!(Command::new("clang")
        .arg("-c").arg(&cfile).arg("-o").arg(&ofile)
        .status().unwrap().success());

    let b = coil::cimport::cimport_header(hfile.to_str().unwrap()).expect("cimport");
    let bpath = dir.join("coil_ci_foo.coil");
    std::fs::write(&bpath, &b).unwrap();

    let prog = format!(
        "(module app)\n(import \"{}\" :use *)\n\
         (defn main [] (-> i64) (cast i64 (triple_it 14)))", // 14*3 = 42
        bpath.display()
    );
    let exe = dir.join("coil_ci_foo_exe");
    let triple = coil::codegen::target_triple();
    coil::build_executable_linked(&prog, &exe, triple, &[ofile.to_str().unwrap().to_string()])
        .expect("link against the C object");
    let code = Command::new(&exe).status().unwrap().code().unwrap();
    assert_eq!(code, 42, "cimported + linked custom C function must return 42");
}

#[test]
fn cimport_refuses_unmappable_never_mis_binds() {
    if !have_clang() {
        eprintln!("SKIP: clang not found");
        return;
    }
    // a union (overlapping layout) must NOT be emitted as a sequential defstruct — that
    // would be a silent-wrong binding (the cardinal failure). It's refused; mappable
    // decls in the same header still come through.
    let h = write_header(
        "coil_ci_refuse.h",
        "union U { int i; float f; };\nstruct Bad { union U u; };\nint ok(int x);\n",
    );
    let b = coil::cimport::cimport_header(&h).expect("cimport");
    assert!(b.contains("(extern ok :cc c [i32] (-> i32))"), "mappable decl emitted:\n{b}");
    assert!(!b.contains("(defstruct U"), "union must NOT be mis-bound as a struct:\n{b}");
    assert!(!b.contains("(defstruct Bad"), "struct containing a union must be refused:\n{b}");
    assert!(b.contains("SKIPPED"), "refusals must be noted in the bindings:\n{b}");
}
