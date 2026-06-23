//! Debugger P3 end-to-end: a `gcr build --debug` executable, driven by LLDB,
//! must show gc-rust locals/params by their SOURCE names with CORRECT VALUES via
//! `frame variable`. This is the gate the dwarfdump test (in `dwarf.rs`) can't
//! prove — that the DWARF locations actually resolve to the right stack slots at
//! a live stop. macOS-only (uses `xcrun lldb`); skips if lldb is unavailable.

#![cfg(target_os = "macos")]

use gcrust::codegen::{build_executable_level, DebugLevel};
use gcrust::compile::parse_with_prelude;
use gcrust::lower::lower_program;
use gcrust::resolve::resolve_module;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Build the runtime staticlib and point the linker at it (mirrors `aot.rs`).
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

/// Is `xcrun lldb` runnable here? (CI without the toolchain → skip.)
fn lldb_available() -> bool {
    Command::new("xcrun")
        .args(["lldb", "--version"])
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Build `src` as a full-debug executable named `prog.gcr` at `out`.
fn build_debug(src: &str, out: &Path, runtime_lib: &Path) {
    // Safety: this test binary is the only writer of this var (single test).
    unsafe {
        std::env::set_var("GCRUST_RUNTIME_LIB", runtime_lib);
    }
    let (module, sources) = parse_with_prelude(src).expect("parse");
    let resolved = resolve_module(module).expect("resolve");
    let mut prog = lower_program(&resolved.globals).expect("lower");
    let mut sources = sources;
    if let Some(s) = sources.get_mut(0) {
        s.path = "prog.gcr".to_string();
    }
    prog.sources = sources;
    build_executable_level(&prog, out, &[], DebugLevel::Full).expect("build --debug");
}

/// Run `xcrun lldb -b` with the given command script against `bin`, return stdout.
fn lldb_run(bin: &Path, script: &str) -> String {
    let dir = bin.parent().unwrap();
    let script_path = dir.join("cmds.lldb");
    std::fs::write(&script_path, script).unwrap();
    let out = Command::new("xcrun")
        .args(["lldb", "-b", "-s"])
        .arg(&script_path)
        .arg(bin)
        .output()
        .expect("failed to run lldb");
    String::from_utf8_lossy(&out.stdout).into_owned()
}

/// Absolute path to the reflection pretty-printer script.
fn pretty_printer_path() -> String {
    format!("{}/tools/gcr_lldb.py", env!("CARGO_MANIFEST_DIR"))
}

#[test]
fn lldb_reflection_printer_renders_enum_payload_and_nested() {
    // The reflection pretty-printer (design §3) decodes the baked `gcrust_type_meta`
    // blob and renders enum PAYLOADS + nested refs inline — what native DWARF
    // can't express. Importing `gcr_lldb.py` registers type summaries.
    if !lldb_available() {
        eprintln!("skipping: xcrun lldb unavailable");
        return;
    }
    let lib = ensure_runtime_lib();
    let dir = std::env::temp_dir().join(format!("gcr_lldb_refl_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let bin = dir.join("refl");

    let src = "struct Point { x: i64, y: i64 }\n\
struct Line { a: Point, b: Point }\n\
enum Shape { Circle(i64), Rect(i64, i64), Empty }\n\
fn main() -> i64 {\n  let s = Shape::Rect(3, 4);\n  let p = Point { x: 1, y: 2 };\n  let q = Point { x: 5, y: 6 };\n  let ln = Line { a: p, b: q };\n  let probe = 0;\n  probe\n}\n";
    build_debug(src, &bin, &lib);

    let script = format!(
        "command script import {}\n\
breakpoint set --file prog.gcr --line 9\nrun\nframe variable s ln\nquit\n",
        pretty_printer_path()
    );
    let out = lldb_run(&bin, &script);

    // Enum payload rendered, plus nested structs inline.
    for needle in ["Shape::Rect(3, 4)", "Line { a: Point { x: 1, y: 2 }"] {
        assert!(
            out.contains(needle),
            "reflection printer missing `{needle}`:\n{out}"
        );
    }

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn lldb_frame_variable_shows_correct_scalar_locals() {
    if !lldb_available() {
        eprintln!("skipping: xcrun lldb unavailable");
        return;
    }
    let lib = ensure_runtime_lib();
    let dir = std::env::temp_dir().join(format!("gcr_lldb_p3_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let bin = dir.join("dbgprog");

    // add(7, 35): at line 4 (`doubled`), x=7 y=35 sum=42 doubled=84.
    let src = "fn add(x: i64, y: i64) -> i64 {\n  let sum = x + y;\n  let doubled = sum + sum;\n  doubled\n}\nfn main() -> i64 {\n  let a = 7;\n  add(a, 35)\n}\n";
    build_debug(src, &bin, &lib);

    // Break on line 4 (after `sum` and `doubled` are assigned), dump locals.
    let out = lldb_run(
        &bin,
        "breakpoint set --file prog.gcr --line 4\nrun\nframe variable\nquit\n",
    );

    // The frame header decodes the params; `frame variable` lists all four with
    // their computed values. Assert each name=value pair appears.
    for needle in ["x = 7", "y = 35", "sum = 42", "doubled = 84"] {
        assert!(
            out.contains(needle),
            "lldb `frame variable` missing `{needle}`:\n{out}"
        );
    }

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn lldb_frame_variable_renders_heap_struct_by_field() {
    if !lldb_available() {
        eprintln!("skipping: xcrun lldb unavailable");
        return;
    }
    let lib = ensure_runtime_lib();
    let dir = std::env::temp_dir().join(format!("gcr_lldb_struct_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let bin = dir.join("structprog");

    // A heap struct local must render by field name (gc-rust reference semantics:
    // `p` IS the Point) — the native-DWARF-struct + slot-deref path.
    let src = "struct Point { x: i64, y: i64 }\nfn main() -> i64 {\n  let p = Point { x: 3, y: 4 };\n  let probe = p.x + p.y;\n  probe\n}\n";
    build_debug(src, &bin, &lib);

    let out = lldb_run(
        &bin,
        "breakpoint set --file prog.gcr --line 5\nrun\nframe variable p\nquit\n",
    );

    // `(Point) p = { x = 3, y = 4 }` — type name + both fields with values.
    for needle in ["(Point)", "x = 3", "y = 4"] {
        assert!(
            out.contains(needle),
            "lldb `frame variable p` missing `{needle}`:\n{out}"
        );
    }

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn lldb_frame_variable_shows_enum_variant() {
    if !lldb_available() {
        eprintln!("skipping: xcrun lldb unavailable");
        return;
    }
    let lib = ensure_runtime_lib();
    let dir = std::env::temp_dir().join(format!("gcr_lldb_enum_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let bin = dir.join("enumprog");

    // A heap enum local shows its active variant via the `tag` enumeration member.
    let src = "enum Shape { Circle(i64), Rect(i64, i64), Empty }\n\
fn main() -> i64 {\n  let s = Shape::Rect(3, 4);\n  let t = Shape::Empty;\n  let probe = 0;\n  probe\n}\n";
    build_debug(src, &bin, &lib);

    let out = lldb_run(
        &bin,
        "breakpoint set --file prog.gcr --line 5\nrun\nframe variable s t\nquit\n",
    );
    // `s` is the Rect variant, `t` is Empty — the tag names must appear.
    for needle in ["(Shape)", "tag = Rect", "tag = Empty"] {
        assert!(
            out.contains(needle),
            "lldb `frame variable` missing `{needle}`:\n{out}"
        );
    }

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn lldb_heap_struct_survives_moving_gc() {
    // The §5 moving-GC check: a heap struct local's DWARF location derefs its GC
    // frame ROOT slot (the collector keeps it current on relocation), NOT a
    // cached object address. So after a relocating GC moves `p`, `frame variable`
    // still reads the right fields. We force many minor GCs (build/discard large
    // trees, ~millions of allocations through the ~1MB nursery) while `p` is a
    // live root, then inspect it.
    if !lldb_available() {
        eprintln!("skipping: xcrun lldb unavailable");
        return;
    }
    let lib = ensure_runtime_lib();
    let dir = std::env::temp_dir().join(format!("gcr_lldb_movegc_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let bin = dir.join("movegc");

    let src = "struct Point { x: i64, y: i64 }\n\
enum Tree { Leaf, Node(Tree, Tree) }\n\
fn make(depth: i64) -> Tree {\n  if depth == 0 { Tree::Leaf } else { Tree::Node(make(depth - 1), make(depth - 1)) }\n}\n\
fn check(t: Tree) -> i64 {\n  match t { Tree::Leaf => 1, Tree::Node(l, r) => 1 + check(l) + check(r) }\n}\n\
fn main() -> i64 {\n  let p = Point { x: 42, y: 7 };\n  let mut total = 0;\n  let mut iter = 0;\n  while iter < 60 {\n    let t = make(15);\n    total = total + check(t);\n    iter = iter + 1;\n  }\n  let result = p.x + p.y;\n  result\n}\n";
    build_debug(src, &bin, &lib);

    // Line 18 is `let result = p.x + p.y;` — after all the GC churn.
    let out = lldb_run(
        &bin,
        "breakpoint set --file prog.gcr --line 18\nrun\nframe variable p\nquit\n",
    );
    for needle in ["(Point)", "x = 42", "y = 7"] {
        assert!(
            out.contains(needle),
            "after moving GC, `frame variable p` missing `{needle}`:\n{out}"
        );
    }

    let _ = std::fs::remove_dir_all(&dir);
}
