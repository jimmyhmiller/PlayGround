//! DWARF debug info (`-g`): a compile unit + a `DISubprogram` per function, a
//! per-statement line table (line-by-line stepping), and scalar-parameter
//! locals (`frame variable`). Verified by reading the DWARF back with `dwarfdump`
//! and by driving `lldb` (both gated on the tools).

mod common;
use common::unique_path;
use std::path::Path;
use std::process::Command;
use std::sync::Mutex;

fn have(tool: &str) -> bool {
    Command::new(tool).arg("--version").output().map(|o| o.status.success()).unwrap_or(false)
}

/// Serializes the lldb-driven tests: several `lldb` + `dsymutil` processes
/// running concurrently contend and intermittently mis-resolve breakpoints.
static LLDB: Mutex<()> = Mutex::new(());

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
    let _g = LLDB.lock().unwrap_or_else(|e| e.into_inner());
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
    // With per-statement line info it lands on the first body statement —
    // `(imul x x)` on line 4 — not the `defn` line. (lldb may print an address
    // offset, e.g. `app.square + 8 at …`.)
    assert!(text.contains("app.square") && text.contains(".coil:4"), "breakpoint did not resolve to source:\n{text}");
    // And it must actually stop there (proves the function survived optimization).
    assert!(text.contains("stop reason = breakpoint"), "breakpoint never hit:\n{text}");

    let _ = std::fs::remove_file(&exe);
    let _ = std::fs::remove_dir_all(exe.with_extension("dSYM"));
    let _ = std::fs::remove_file(&src_path);
}

/// Local-variable DI: at a line breakpoint inside a function, `frame variable`
/// shows the scalar parameters with their correct names and values (via
/// `DILocalVariable` + `llvm.dbg.declare` over `-g`-spilled debug slots).
#[test]
fn lldb_frame_variable_shows_params() {
    if !have("lldb") {
        eprintln!("SKIP: lldb not found");
        return;
    }
    let _g = LLDB.lock().unwrap_or_else(|e| e.into_inner());
    let prog = "(extern putchar :cc c [i32] (-> i32))\n\
                (defn show [(a i64) (b i64)] (-> i64)\n  (putchar 65)\n  (iadd a b))\n\
                (defn main [] (-> i64)\n  (show 30 12))\n";
    let src_path = unique_path("varsg").with_extension("coil");
    std::fs::write(&src_path, prog).unwrap();
    let exe = unique_path("varsexe");
    coil::build_executable_linked_dbg(prog, &exe, coil::codegen::target_triple(), &[], Some(&src_path))
        .expect("debug build");

    // Break on the `(iadd a b)` line (line 4) — inside the body, past the
    // prologue, so the spilled params hold their real values.
    let file = src_path.file_name().unwrap().to_str().unwrap();
    let out = Command::new("lldb")
        .arg(&exe)
        .args([
            "-b", "-o", &format!("breakpoint set --file {file} --line 4"),
            "-o", "run", "-o", "frame variable", "-o", "quit",
        ])
        .output()
        .unwrap();
    let text = format!("{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
    assert!(text.contains("a = 30"), "param a not shown with value 30:\n{text}");
    assert!(text.contains("b = 12"), "param b not shown with value 12:\n{text}");

    let _ = std::fs::remove_file(&exe);
    let _ = std::fs::remove_dir_all(exe.with_extension("dSYM"));
    let _ = std::fs::remove_file(&src_path);
}

/// Per-statement line tables: `next` in lldb walks the body line by line (each
/// statement maps to its own source line via the per-Expr span). Uses calls
/// with side effects so each statement is a real, steppable instruction.
#[test]
fn lldb_steps_line_by_line() {
    if !have("lldb") {
        eprintln!("SKIP: lldb not found");
        return;
    }
    let _g = LLDB.lock().unwrap_or_else(|e| e.into_inner());
    // putchar so each statement is a real instruction (arithmetic on constants
    // would fold away with nothing to step through).
    let prog = "(extern putchar :cc c [i32] (-> i32))\n\
                (defn main [] (-> i64)\n  (putchar 65)\n  (putchar 66)\n  (putchar 10)\n  0)\n";
    let src_path = unique_path("stepg").with_extension("coil");
    std::fs::write(&src_path, prog).unwrap();
    let exe = unique_path("stepexe");
    coil::build_executable_linked_dbg(prog, &exe, coil::codegen::target_triple(), &[], Some(&src_path))
        .expect("debug build");

    let out = Command::new("lldb")
        .arg(&exe)
        .args([
            "-b", "-o", "breakpoint set --name main", "-o", "run",
            "-o", "next", "-o", "next", "-o", "next", "-o", "quit",
        ])
        .output()
        .unwrap();
    let text = format!("{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
    // Stepping must visit successive source lines (the three putchar statements
    // are on lines 3, 4, 5). Proves per-statement line info, not just function-level.
    for line in [":3", ":4", ":5"] {
        assert!(text.contains(line), "stepping did not reach line {line}:\n{text}");
    }

    let _ = std::fs::remove_file(&exe);
    let _ = std::fs::remove_dir_all(exe.with_extension("dSYM"));
    let _ = std::fs::remove_file(&src_path);
}
