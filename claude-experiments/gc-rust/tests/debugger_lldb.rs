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
