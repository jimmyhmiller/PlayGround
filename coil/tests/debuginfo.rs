//! DWARF debug info (`-g`): function-granularity line tables — a compile unit
//! plus a `DISubprogram` per function carrying its source line, so `lldb`/`gdb`
//! can map functions to source and show file:line in backtraces. Verified by
//! emitting an object, linking a debug executable, and reading the DWARF back
//! with `dwarfdump` (gated on the tool). Per-statement stepping and local-var
//! info are a later increment (they need spans on every AST node).

mod common;
use common::unique_path;
use std::path::Path;
use std::process::Command;

fn have(tool: &str) -> bool {
    Command::new(tool).arg("--version").output().map(|o| o.status.success()).unwrap_or(false)
}

const PROG: &str = "(module app)\n\n\
                    (defn square [(x i64)] (-> i64)\n  (imul x x))\n\n\
                    (defn main [] (-> i64)\n  (let [a 6] (square a)))\n";

/// A `-g` build emits a DWARF compile unit with a subprogram per function at its
/// real source line; a debugger reads function ↔ source from it.
#[test]
fn debug_build_emits_dwarf_subprograms_at_source_lines() {
    if !have("dwarfdump") {
        eprintln!("SKIP: dwarfdump not found");
        return;
    }
    let src_path = unique_path("dbg").with_extension("coil");
    std::fs::write(&src_path, PROG).unwrap();
    let exe = unique_path("dbgexe");
    coil::build_executable_linked_dbg(
        PROG,
        &exe,
        coil::codegen::target_triple(),
        &[],
        Some(&src_path),
    )
    .expect("debug build");

    // The binary still runs correctly (debug info doesn't change behavior).
    let code = Command::new(&exe).status().unwrap().code().unwrap();
    assert_eq!(code, 36, "debug build must still compute square(6) = 36");

    // Read DWARF from the .dSYM (macOS) or the executable itself (ELF).
    let dsym = exe.with_extension("dSYM");
    let target: &Path = if dsym.exists() { &dsym } else { &exe };
    let dump = Command::new("dwarfdump").arg("--debug-info").arg(target).output().unwrap();
    let out = String::from_utf8_lossy(&dump.stdout);

    assert!(out.contains("DW_TAG_subprogram"), "no subprograms in DWARF:\n{out}");
    assert!(out.contains("\"main\""), "main subprogram missing:\n{out}");
    assert!(out.contains("\"app.square\""), "square subprogram missing:\n{out}");
    // square is defined on line 3, main on line 6 of PROG.
    assert!(out.contains("DW_AT_decl_line\t(3)") || out.contains("DW_AT_decl_line (3)"), "square line:\n{out}");
    assert!(out.contains("DW_AT_decl_line\t(6)") || out.contains("DW_AT_decl_line (6)"), "main line:\n{out}");

    let _ = std::fs::remove_file(&exe);
    let _ = std::fs::remove_dir_all(&dsym);
    let _ = std::fs::remove_file(&src_path);
}

/// `-g` is opt-in: an ordinary (release) build emits NO debug info, so the
/// default `cc -O3`-parity output is unchanged.
#[test]
fn release_build_has_no_debug_info() {
    if !have("dwarfdump") {
        eprintln!("SKIP: dwarfdump not found");
        return;
    }
    let exe = unique_path("reldbg");
    coil::build_executable(PROG, &exe).expect("release build");
    let dump = Command::new("dwarfdump").arg("--debug-info").arg(&exe).output().unwrap();
    let out = String::from_utf8_lossy(&dump.stdout);
    assert!(!out.contains("DW_TAG_subprogram"), "release build must have no DWARF:\n{out}");
    let _ = std::fs::remove_file(&exe);
}

/// lldb resolves a breakpoint on a Coil function to its source file:line and
/// stops there. (Gated on lldb; uses `noinline` under `-g` so the function
/// survives the optimizer.)
#[test]
fn lldb_resolves_breakpoint_to_source() {
    if !have("lldb") {
        eprintln!("SKIP: lldb not found");
        return;
    }
    let src_path = unique_path("lldbg").with_extension("coil");
    std::fs::write(&src_path, PROG).unwrap();
    let exe = unique_path("lldbexe");
    coil::build_executable_linked_dbg(PROG, &exe, coil::codegen::target_triple(), &[], Some(&src_path))
        .expect("debug build");

    let out = Command::new("lldb")
        .arg(&exe)
        .args(["-b", "-o", "breakpoint set --name app.square", "-o", "run", "-o", "bt", "-o", "quit"])
        .output()
        .unwrap();
    let text = format!("{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
    // The breakpoint must resolve to our source file (proves lldb read the DWARF).
    assert!(text.contains(".coil:3"), "breakpoint did not resolve to source:\n{text}");
    // And it must actually stop there (proves the function survived optimization).
    assert!(text.contains("stop reason = breakpoint"), "breakpoint never hit:\n{text}");

    let _ = std::fs::remove_file(&exe);
    let _ = std::fs::remove_dir_all(exe.with_extension("dSYM"));
    let _ = std::fs::remove_file(&src_path);
}
