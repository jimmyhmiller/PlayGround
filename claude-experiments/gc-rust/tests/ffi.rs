//! FFI tests (Phase A): `extern "C" fn` declarations calling real C symbols.
//!
//! Phase A permits only scalar types across the boundary (no managed GC
//! pointers), so these are correct without pinning, roots, or a thread
//! transition. We verify: (1) the JIT resolves libm/libc symbols against the
//! host process and calls them correctly; (2) the AOT linker resolves them; and
//! (3) the blittable-only rule rejects a non-scalar parameter at compile time.
//! See `docs/ffi.md`.

use std::path::{Path, PathBuf};
use std::process::Command;

use gcrust::codegen::{build_executable, jit_run_i64};
use gcrust::lexer::lex;
use gcrust::lower::lower_program;
use gcrust::parser::parse_module;
use gcrust::resolve::resolve_module;

fn jit(src: &str) -> i64 {
    let module = parse_module(&lex(src).unwrap()).unwrap();
    let resolved = resolve_module(module).unwrap();
    let prog = lower_program(&resolved.globals).unwrap();
    jit_run_i64(&prog).expect("jit_run_i64 failed")
}

/// Lower `src` and return the lowering error message, if any (None = compiled).
fn lower_err(src: &str) -> Option<String> {
    let module = parse_module(&lex(src).unwrap()).unwrap();
    let resolved = resolve_module(module).unwrap();
    match lower_program(&resolved.globals) {
        Ok(_) => None,
        Err(e) => Some(e.msg),
    }
}

#[test]
fn jit_calls_libm_scalar_externs() {
    // sqrt(144.0)=12, pow(2,10)=1024, abs(-30)=30  ->  1066
    let src = r#"
        extern "C" fn sqrt(x: f64) -> f64;
        extern "C" fn pow(base: f64, exp: f64) -> f64;
        extern "C" fn abs(x: i32) -> i32;
        fn main() -> i64 {
            let r = sqrt(144.0);
            let p = pow(2.0, 10.0);
            let a = abs((0 - 30) as i32);
            (r as i64) + (p as i64) + (a as i64)
        }
    "#;
    assert_eq!(jit(src), 1066);
}

#[test]
fn extern_rejects_non_scalar_param() {
    let src = r#"
        extern "C" fn strlen(s: String) -> i64;
        fn main() -> i64 { strlen("hi") }
    "#;
    let err = lower_err(src).expect("expected a blittable-only error");
    assert!(
        err.contains("only scalar types") && err.contains("FFI boundary"),
        "unexpected error: {err}"
    );
}

#[test]
fn jit_passes_value_struct_by_pointer_out_param() {
    // gettimeofday fills our value struct in place through the pointer. tv_sec
    // ends up far past 2023, proving C wrote the caller's stack-local struct.
    let src = r#"
        #[value] struct TimeVal { tv_sec: i64, tv_usec: i64 }
        extern "C" fn gettimeofday(mut tv: TimeVal, tz: i64) -> i32;
        fn main() -> i64 {
            let mut tv = TimeVal { tv_sec: 0, tv_usec: 0 };
            let rc = gettimeofday(tv, 0);
            if rc != 0 { return 0 - 1; }
            if tv.tv_sec > 1700000000 { 1 } else { 0 }
        }
    "#;
    assert_eq!(jit(src), 1);
}

#[test]
fn extern_accepts_value_struct_of_scalars() {
    // A value struct of scalar fields is blittable and may cross by pointer.
    let src = r#"
        #[value] struct Pair { a: i64, b: i64 }
        extern "C" fn takes_pair(p: Pair) -> i64;
        fn main() -> i64 { 0 }
    "#;
    assert!(lower_err(src).is_none(), "#[value] struct of scalars should be accepted");
}

#[test]
fn extern_rejects_heap_struct_param() {
    // A non-value (heap `Ref`) struct must NOT cross — its address would move
    // under GC. Only stack-resident value structs are allowed.
    let src = r#"
        struct Heapy { a: i64 }
        extern "C" fn bad(h: Heapy) -> i64;
        fn main() -> i64 { let h = Heapy { a: 1 }; bad(h) }
    "#;
    let err = lower_err(src).expect("expected a blittable-only error");
    assert!(
        err.contains("FFI boundary") && err.contains("heap"),
        "unexpected error: {err}"
    );
}

#[test]
fn jit_passes_string_bytes_via_as_c_bytes() {
    // as_c_bytes copies the String's UTF-8 (+NUL) to a stack buffer and passes a
    // RawPtr. strlen("hello, ffi")=10, atoi("32")=32 -> 42.
    let src = r#"
        extern "C" fn strlen(s: RawPtr) -> i64;
        extern "C" fn atoi(s: RawPtr) -> i32;
        fn main() -> i64 {
            let n = strlen(as_c_bytes("hello, ffi"));
            let v = atoi(as_c_bytes("32"));
            n + (v as i64)
        }
    "#;
    assert_eq!(jit(src), 42);
}

#[test]
fn jit_array_copy_out_buffer() {
    // memset fills a mut Array<u8> in place; the copy-out write-back makes the
    // result visible. memcmp on two equal arrays returns 0. -> 7+7+0 = 14.
    let src = r#"
        extern "C" fn memset(mut buf: RawPtr, c: i32, n: i64) -> RawPtr;
        extern "C" fn memcmp(a: RawPtr, b: RawPtr, n: i64) -> i32;
        fn main() -> i64 {
            let mut buf: Array<u8> = array_new(5);
            memset(as_c_bytes(buf), 7, 5);
            let sum = (array_get_unchecked(buf, 0) as i64) + (array_get_unchecked(buf, 4) as i64);
            let mut a: Array<u8> = array_new(2);
            array_set(a, 0, 9); array_set(a, 1, 9);
            let mut b: Array<u8> = array_new(2);
            array_set(b, 0, 9); array_set(b, 1, 9);
            sum + (memcmp(as_c_bytes(a), as_c_bytes(b), 2) as i64)
        }
    "#;
    assert_eq!(jit(src), 14);
}

#[test]
fn jit_c_callback_into_gcrust() {
    // libc qsort calls a gc-rust comparator back through a synthesized C-ABI
    // trampoline. Sorts [30,10,50,20,40] -> [10,20,30,40,50]; 10*100+50 = 1050.
    let src = r#"
        extern "C" fn qsort(mut base: RawPtr, n: i64, size: i64, compar: extern fn(RawPtr, RawPtr) -> i32);
        fn cmp(a: RawPtr, b: RawPtr) -> i32 {
            let x = ptr_read_i64(a); let y = ptr_read_i64(b);
            if x < y { (0 - 1) as i32 } else if x > y { 1i32 } else { 0i32 }
        }
        fn main() -> i64 {
            let mut arr: Array<i64> = array_new(5);
            array_set(arr, 0, 30); array_set(arr, 1, 10); array_set(arr, 2, 50);
            array_set(arr, 3, 20); array_set(arr, 4, 40);
            qsort(as_c_bytes(arr), 5, 8, cmp);
            array_get_unchecked(arr, 0) * 100 + array_get_unchecked(arr, 4)
        }
    "#;
    assert_eq!(jit(src), 1050);
}

#[test]
fn callback_signature_mismatch_rejected() {
    // cmp returns i64 but the callback type expects i32 -> compile error.
    let src = r#"
        extern "C" fn takes_cb(f: extern fn(i64) -> i32);
        fn cb(x: i64) -> i64 { x }
        fn main() -> i64 { takes_cb(cb); 0 }
    "#;
    assert!(lower_err(src).is_some(), "expected a callback signature error");
}

#[test]
fn as_c_bytes_rejects_array_of_non_scalars() {
    let src = r#"
        extern "C" fn f(p: RawPtr) -> i64;
        fn main() -> i64 {
            let mut a: Array<String> = array_new(1);
            f(as_c_bytes(a))
        }
    "#;
    let err = lower_err(src).expect("expected a non-scalar-array error");
    assert!(err.contains("array of scalars"), "unexpected error: {err}");
}

#[test]
fn as_c_bytes_rejected_outside_extern_arg() {
    let src = r#"
        fn main() -> i64 { let _p = as_c_bytes("x"); 0 }
    "#;
    let err = lower_err(src).expect("expected a misuse error");
    assert!(
        err.contains("direct argument") && err.contains("extern"),
        "unexpected error: {err}"
    );
}

#[test]
fn raw_string_to_rawptr_param_rejected() {
    // Passing a String directly to a RawPtr param (without as_c_bytes) must fail
    // — there is no way to hand raw heap memory to C.
    let src = r#"
        extern "C" fn g(s: RawPtr) -> i64;
        fn main() -> i64 { g("x") }
    "#;
    let err = lower_err(src).expect("expected a type mismatch");
    assert!(err.contains("RawPtr") && err.contains("String"), "unexpected error: {err}");
}

#[test]
fn extern_rejects_non_scalar_return() {
    // The extern is only checked when reached, so main must call it.
    let src = r#"
        extern "C" fn make_str() -> String;
        fn main() -> i64 { let _s = make_str(); 0 }
    "#;
    let err = lower_err(src).expect("expected a blittable-only error");
    assert!(
        err.contains("only scalar types") && err.contains("FFI boundary"),
        "unexpected error: {err}"
    );
}

// --- AOT: the system linker resolves the C symbols against libc/libm ---

fn ensure_runtime_lib() -> PathBuf {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let status = Command::new(env!("CARGO"))
        .args(["build", "-p", "gcrust-rt"])
        .current_dir(&manifest)
        .status()
        .expect("failed to run cargo build -p gcrust-rt");
    assert!(status.success(), "building gcrust-rt staticlib failed");
    let lib = manifest.join("target").join("debug").join("libgcrust_rt.a");
    assert!(lib.exists(), "libgcrust_rt.a not found at {}", lib.display());
    lib
}

fn run_exit_code(bin: &Path) -> i32 {
    let status = Command::new(bin).status().expect("failed to run AOT binary");
    status.code().expect("AOT binary terminated by signal")
}

#[test]
fn aot_calls_libm_scalar_externs() {
    let lib = ensure_runtime_lib();
    unsafe { std::env::set_var("GCRUST_RUNTIME_LIB", &lib); }
    let mut out = std::env::temp_dir();
    out.push(format!("gcrust_ffi_test_{}", std::process::id()));

    let src = include_str!("../examples/ffi.gcr");
    let module = parse_module(&lex(src).unwrap()).unwrap();
    let resolved = resolve_module(module).unwrap();
    let prog = lower_program(&resolved.globals).unwrap();
    build_executable(&prog, &out, &[]).expect("build_executable failed");

    // 1066 & 0xFF = 42
    assert_eq!(run_exit_code(&out), 1066 & 0xFF);
    let _ = std::fs::remove_file(&out);
}

#[test]
fn aot_passes_value_struct_by_pointer() {
    let lib = ensure_runtime_lib();
    unsafe { std::env::set_var("GCRUST_RUNTIME_LIB", &lib); }
    let mut out = std::env::temp_dir();
    out.push(format!("gcrust_ffi_struct_test_{}", std::process::id()));

    let src = include_str!("../examples/ffi_struct.gcr");
    let module = parse_module(&lex(src).unwrap()).unwrap();
    let resolved = resolve_module(module).unwrap();
    let prog = lower_program(&resolved.globals).unwrap();
    build_executable(&prog, &out, &[]).expect("build_executable failed");

    // Returns 1 once C has written tv_sec (current Unix time > 2023).
    assert_eq!(run_exit_code(&out), 1);
    let _ = std::fs::remove_file(&out);
}

#[test]
fn aot_passes_value_struct_by_value() {
    // Pass value structs BY VALUE per the C ABI (AAPCS64), with NO shim: a 4-byte
    // `Color` coerced into a GPR, an HFA `Vector2 {f32,f32}` into SIMD registers,
    // both directions (args + returns). Links a small C helper compiled by `cc`,
    // so the system ABI is the oracle.
    let lib = ensure_runtime_lib();
    unsafe { std::env::set_var("GCRUST_RUNTIME_LIB", &lib); }
    let dir = std::env::temp_dir().join(format!("gcrust_ffi_byval_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();

    let cpath = dir.join("abi.c");
    std::fs::write(
        &cpath,
        "#include <stdint.h>\n\
         typedef struct { uint8_t r,g,b,a; } Color;\n\
         typedef struct { float x,y; } Vector2;\n\
         int color_sum(Color c){ return c.r+c.g+c.b+c.a; }\n\
         int vec_sum(Vector2 v){ return (int)(v.x+v.y); }\n\
         Vector2 mk_vec(float x,float y){ Vector2 v; v.x=x; v.y=y; return v; }\n\
         Color mk_color(int r,int g,int b){ Color c; c.r=r;c.g=g;c.b=b;c.a=255; return c; }\n",
    )
    .unwrap();
    let obj = dir.join("abi.o");
    let ok = Command::new("cc")
        .args(["-c"])
        .arg(&cpath)
        .arg("-o")
        .arg(&obj)
        .status()
        .expect("run cc")
        .success();
    assert!(ok, "compiling the C ABI helper failed");

    let src = "\
#[value] struct Color { r: u8, g: u8, b: u8, a: u8 }\n\
#[value] struct Vector2 { x: f32, y: f32 }\n\
extern \"C\" fn color_sum(c: Color) -> i32;\n\
extern \"C\" fn vec_sum(v: Vector2) -> i32;\n\
extern \"C\" fn mk_vec(x: f32, y: f32) -> Vector2;\n\
extern \"C\" fn mk_color(r: i32, g: i32, b: i32) -> Color;\n\
fn main() -> i64 {\n\
  let s1 = color_sum(Color { r: 10, g: 20, b: 30, a: 255 });\n\
  let s2 = vec_sum(Vector2 { x: 3.0, y: 4.0 });\n\
  let v = mk_vec(1.5, 2.5);\n\
  let s3 = (v.x + v.y) as i32;\n\
  let c = mk_color(1, 2, 3);\n\
  let s4 = (c.r as i32) + (c.g as i32) + (c.b as i32) + (c.a as i32);\n\
  (s1 as i64) + (s2 as i64) + (s3 as i64) + (s4 as i64)\n\
}\n";
    let module = parse_module(&lex(src).unwrap()).unwrap();
    let resolved = resolve_module(module).unwrap();
    let prog = lower_program(&resolved.globals).unwrap();
    let out = dir.join("byval");
    build_executable(&prog, &out, &[obj.to_string_lossy().into_owned()])
        .expect("build_executable failed");

    // 315 (color_sum) + 7 (vec_sum) + 4 (mk_vec) + 261 (mk_color) = 587; &0xFF = 75.
    assert_eq!(run_exit_code(&out), 587 & 0xFF);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn aot_passes_string_bytes() {
    let lib = ensure_runtime_lib();
    unsafe { std::env::set_var("GCRUST_RUNTIME_LIB", &lib); }
    let mut out = std::env::temp_dir();
    out.push(format!("gcrust_ffi_bytes_test_{}", std::process::id()));

    let src = include_str!("../examples/ffi_bytes.gcr");
    let module = parse_module(&lex(src).unwrap()).unwrap();
    let resolved = resolve_module(module).unwrap();
    let prog = lower_program(&resolved.globals).unwrap();
    build_executable(&prog, &out, &[]).expect("build_executable failed");

    // strlen("hello, ffi")=10 + atoi("32")=32 -> 42
    assert_eq!(run_exit_code(&out), 42);
    let _ = std::fs::remove_file(&out);
}

#[test]
fn aot_array_copy_out_buffer() {
    let lib = ensure_runtime_lib();
    unsafe { std::env::set_var("GCRUST_RUNTIME_LIB", &lib); }
    let mut out = std::env::temp_dir();
    out.push(format!("gcrust_ffi_buffer_test_{}", std::process::id()));

    let src = include_str!("../examples/ffi_buffer.gcr");
    let module = parse_module(&lex(src).unwrap()).unwrap();
    let resolved = resolve_module(module).unwrap();
    let prog = lower_program(&resolved.globals).unwrap();
    build_executable(&prog, &out, &[]).expect("build_executable failed");

    // memset copy-out (7+7) + memcmp equal (0) -> 14
    assert_eq!(run_exit_code(&out), 14);
    let _ = std::fs::remove_file(&out);
}

#[test]
fn aot_c_callback_into_gcrust() {
    let lib = ensure_runtime_lib();
    unsafe { std::env::set_var("GCRUST_RUNTIME_LIB", &lib); }
    let mut out = std::env::temp_dir();
    out.push(format!("gcrust_ffi_callback_test_{}", std::process::id()));

    let src = include_str!("../examples/ffi_callback.gcr");
    let module = parse_module(&lex(src).unwrap()).unwrap();
    let resolved = resolve_module(module).unwrap();
    let prog = lower_program(&resolved.globals).unwrap();
    build_executable(&prog, &out, &[]).expect("build_executable failed");

    // 1050 & 0xFF = 26
    assert_eq!(run_exit_code(&out), 1050 & 0xFF);
    let _ = std::fs::remove_file(&out);
}
