//! Allocation-site profiling (Target-1b) — end-to-end coverage.
//!
//! Two layers:
//!   1. The compiler threads a `site_id` constant into every `ai_gc_alloc_*`
//!      call (the ABI change), verified at the LLVM-IR level.
//!   2. An AOT binary, run with `GCR_ALLOC_PROFILE=1`, prints a per-site
//!      count+bytes profile that names each site's `(function, type)`.

use std::path::{Path, PathBuf};
use std::process::Command;

use gcrust::codegen::{build_executable, emit_llvm_ir};
use gcrust::lexer::lex;
use gcrust::lower::lower_program;
use gcrust::parser::parse_module;
use gcrust::resolve::resolve_module;

fn lower(src: &str) -> gcrust::core::CoreProgram {
    let module = parse_module(&lex(src).unwrap()).unwrap();
    let resolved = resolve_module(module).unwrap();
    lower_program(&resolved.globals).unwrap()
}

/// The allocation ABI must carry a site id: `ai_gc_alloc_fixed(ptr, i32, i32)`
/// and `ai_gc_alloc_varlen(ptr, i32, i64, i32)`. Checked on the unoptimized IR
/// so the declarations are present verbatim.
#[test]
fn alloc_calls_carry_site_id_in_ir() {
    // A struct alloc (fixed) and a string literal (varlen).
    let src = r#"
        struct Point { x: i64, y: i64 }
        fn main() -> i64 {
            let p = Point { x: 1, y: 2 };
            let s = "hi";
            p.x + str_len(s)
        }
    "#;
    let prog = lower(src);
    let ir = emit_llvm_ir(&prog, false).expect("emit IR");
    assert!(
        ir.contains("declare ptr @ai_gc_alloc_fixed(ptr, i32, i32)"),
        "ai_gc_alloc_fixed must take (thread, type_id, site_id); IR:\n{ir}"
    );
    assert!(
        ir.contains("declare ptr @ai_gc_alloc_varlen(ptr, i32, i64, i32)"),
        "ai_gc_alloc_varlen must take (thread, type_id, n, site_id); IR:\n{ir}"
    );
}

// ─── End-to-end AOT profile ──────────────────────────────────────────

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

// Serialize AOT builds: `GCRUST_RUNTIME_LIB` is a process-global env var and the
// staticlib rebuild is shared, so concurrent build() calls (tests run in threads)
// race. The feature is fine; this keeps the suite green in parallel.
static AOT_BUILD_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn build(src: &str, out: &Path, runtime_lib: &Path) {
    let _guard = AOT_BUILD_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    unsafe {
        std::env::set_var("GCRUST_RUNTIME_LIB", runtime_lib);
    }
    let prog = lower(src);
    build_executable(&prog, out, &[]).expect("build_executable failed");
}

fn tmp(name: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!("gcrust_allocprof_{}_{}", std::process::id(), name));
    p
}

#[test]
fn aot_alloc_profile_reports_sites() {
    let lib = ensure_runtime_lib();
    let out = tmp("shapes");
    build(include_str!("../examples/shapes.gcr"), &out, &lib);

    let output = Command::new(&out)
        .env("GCR_ALLOC_PROFILE", "1")
        .output()
        .expect("failed to run AOT binary");
    assert_eq!(output.status.code(), Some(47), "shapes.gcr should exit 47");
    let stderr = String::from_utf8_lossy(&output.stderr);

    assert!(
        stderr.contains("allocation-site profile"),
        "profile header missing; stderr:\n{stderr}"
    );
    // shapes.gcr allocates a Point struct and Shape enum values in compiled code.
    assert!(
        stderr.contains("Point"),
        "profile should name the Point type; stderr:\n{stderr}"
    );
    assert!(
        stderr.contains("Shape"),
        "profile should name the Shape type; stderr:\n{stderr}"
    );
    // Sites are labelled by their containing function.
    assert!(
        stderr.contains("main"),
        "profile should name the allocating function; stderr:\n{stderr}"
    );
    // And the honesty caveat is stated, not hidden.
    assert!(
        stderr.contains("runtime-internal allocations"),
        "profile must state its coverage caveat; stderr:\n{stderr}"
    );

    let _ = std::fs::remove_file(&out);
}

#[test]
fn alloc_profile_line_col_distinguishes_same_type_same_function() {
    // Debugger P1 span-threading proof: two allocations of the SAME type in the
    // SAME function at DIFFERENT lines must be DISTINCT sites with file:line:col
    // locations (before span-threading they collapsed to one (function,type) site).
    let dir = std::env::temp_dir().join(format!("gcr_loc_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let srcp = dir.join("loc.gcr");
    std::fs::write(
        &srcp,
        "struct P { a: i64, b: i64 }\nfn main() -> i64 {\n  let x = P { a: 1, b: 2 };\n  let y = P { a: 3, b: 4 };\n  x.a + y.b\n}\n",
    )
    .unwrap();

    let out = Command::new(env!("CARGO_BIN_EXE_gcr"))
        .args(["run", srcp.to_str().unwrap()])
        .env("GCR_ALLOC_PROFILE", "1")
        .output()
        .expect("run gcr");
    assert!(out.status.success(), "run failed: {}", String::from_utf8_lossy(&out.stderr));
    let stderr = String::from_utf8_lossy(&out.stderr);

    // The two P allocations are on source lines 3 and 4.
    let p = srcp.to_string_lossy();
    let line3 = format!("{p}:3:");
    let line4 = format!("{p}:4:");
    assert!(stderr.contains(&line3), "expected a site at line 3; stderr:\n{stderr}");
    assert!(stderr.contains(&line4), "expected a site at line 4; stderr:\n{stderr}");
    // Two distinct P sites (not one collapsed site): two profile rows whose TYPE
    // column (3rd whitespace field: bytes, count, type, ...) is exactly "P".
    let p_rows = stderr
        .lines()
        .filter(|l| l.split_whitespace().nth(2) == Some("P"))
        .count();
    assert!(p_rows >= 2, "expected >=2 distinct P sites; stderr:\n{stderr}");

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn alloc_profile_prelude_allocs_are_not_fabricated_locations() {
    // Regression guard for the multi-source span misattribution: allocations that
    // happen INSIDE prelude functions (Vec/array_new) must NOT be labeled with a
    // fabricated user-file line (the bug clamped an out-of-range prelude offset to
    // the end of the small user file → e.g. user.gcr:11:1 in a 10-line file).
    // They must resolve to the real prelude (<std>) location or honest empty (-).
    let dir = std::env::temp_dir().join(format!("gcr_pre_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let srcp = dir.join("pre.gcr");
    let src = "fn main() -> i64 {\n  let v: Vec<i64> = vec_new();\n  let v2 = vec_push(v, 7);\n  vec_len(v2)\n}\n";
    std::fs::write(&srcp, src).unwrap();
    let user_lines = src.lines().count(); // 5

    let out = Command::new(env!("CARGO_BIN_EXE_gcr"))
        .args(["run", srcp.to_str().unwrap()])
        .env("GCR_ALLOC_PROFILE", "1")
        .output()
        .expect("run gcr");
    assert!(out.status.success(), "run failed: {}", String::from_utf8_lossy(&out.stderr));
    let stderr = String::from_utf8_lossy(&out.stderr);

    // The user file must NEVER be cited at a line beyond its length (the bug).
    let p = srcp.to_string_lossy();
    for line in stderr.lines() {
        if let Some(idx) = line.find(&*p) {
            let rest = &line[idx + p.len()..]; // ":<line>:<col>..."
            if let Some(num) = rest.strip_prefix(':') {
                let lineno: usize = num.split(':').next().unwrap_or("0").parse().unwrap_or(0);
                assert!(
                    lineno <= user_lines,
                    "FABRICATED location: {p}:{lineno} but the file has {user_lines} lines; stderr:\n{stderr}"
                );
            }
        }
    }
    // And there IS at least one prelude (<std>) location — proving prelude allocs
    // resolve to the real prelude source, not the user file or a bogus line.
    assert!(
        stderr.contains("<std>:"),
        "expected prelude allocs labeled <std>:line:col; stderr:\n{stderr}"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn aot_no_profile_without_env() {
    let lib = ensure_runtime_lib();
    let out = tmp("shapes_noenv");
    build(include_str!("../examples/shapes.gcr"), &out, &lib);

    // Without GCR_ALLOC_PROFILE the binary must not print a profile (zero cost,
    // no behavioural change).
    let output = Command::new(&out).output().expect("failed to run AOT binary");
    assert_eq!(output.status.code(), Some(47));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        !stderr.contains("allocation-site profile"),
        "profile must be opt-in; stderr:\n{stderr}"
    );

    let _ = std::fs::remove_file(&out);
}
