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

/// Run `lldb` in batch mode over `exe` with the given commands, returning its
/// combined stdout+stderr. Loads the `.dSYM` *explicitly* (`add-dsym`) rather
/// than via Spotlight, which doesn't reliably index a freshly-built temp dSYM.
fn run_lldb(exe: &Path, cmds: &[&str]) -> String {
    let dsym = exe.with_extension("dSYM");
    let mut args: Vec<String> = vec!["-b".into()];
    if dsym.exists() {
        args.push("-o".into());
        args.push(format!("add-dsym {}", dsym.display()));
    }
    for c in cmds {
        args.push("-o".into());
        args.push((*c).into());
    }
    let out = Command::new("lldb").arg(exe).args(&args).output().unwrap();
    format!("{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr))
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
    // `main` is qualified like every function now (`app.main`); its DWARF name
    // matches the source module, consistent with `app.square`. The LINKER symbol
    // is still `main`, so `break main` in lldb/gdb continues to work.
    assert!(out.contains("\"app.main\""), "main subprogram missing:\n{out}");
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

    let text = run_lldb(&exe, &["breakpoint set --name app.square", "run", "bt", "quit"]);
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
    let text = run_lldb(
        &exe,
        &[&format!("breakpoint set --file {file} --line 4"), "run", "frame variable", "quit"],
    );
    assert!(text.contains("a = 30"), "param a not shown with value 30:\n{text}");
    assert!(text.contains("b = 12"), "param b not shown with value 12:\n{text}");

    let _ = std::fs::remove_file(&exe);
    let _ = std::fs::remove_dir_all(exe.with_extension("dSYM"));
    let _ = std::fs::remove_file(&src_path);
}

/// Local-variable DI for `let` bindings: at a line breakpoint, `frame variable`
/// shows the in-scope locals with their correct values (via `create_auto_variable`
/// + a debug spill; the `-g` backend runs at -O0 so the spill stores aren't
/// reordered/merged away from their statement).
#[test]
fn lldb_frame_variable_shows_let_locals() {
    if !have("lldb") {
        eprintln!("SKIP: lldb not found");
        return;
    }
    let _g = LLDB.lock().unwrap_or_else(|e| e.into_inner());
    let prog = "(extern putchar :cc c [i32] (-> i32))\n\
                (defn main [] (-> i64)\n  (let [x 10]\n    (let [y 20]\n      \
                (let [z (iadd x y)]\n        (putchar 65)\n        (cast i64 z)))))\n";
    let src_path = unique_path("letg").with_extension("coil");
    std::fs::write(&src_path, prog).unwrap();
    let exe = unique_path("letexe");
    coil::build_executable_linked_dbg(prog, &exe, coil::codegen::target_triple(), &[], Some(&src_path))
        .expect("debug build");

    let file = src_path.file_name().unwrap().to_str().unwrap();
    let text = run_lldb(
        &exe,
        &[&format!("breakpoint set --file {file} --line 6"), "run", "frame variable", "quit"],
    );
    assert!(text.contains("x = 10"), "x not shown as 10:\n{text}");
    assert!(text.contains("y = 20"), "y not shown as 20:\n{text}");
    assert!(text.contains("z = 30"), "z not shown as 30:\n{text}");

    let _ = std::fs::remove_file(&exe);
    let _ = std::fs::remove_dir_all(exe.with_extension("dSYM"));
    let _ = std::fs::remove_file(&src_path);
}

/// Aggregate DI: a struct parameter prints with its fields (a `DICompositeType`
/// with members at their byte offsets; the by-ref param shows as `Struct *`,
/// dereferencing to the field values).
#[test]
fn lldb_frame_variable_shows_struct_fields() {
    if !have("lldb") {
        eprintln!("SKIP: lldb not found");
        return;
    }
    let _g = LLDB.lock().unwrap_or_else(|e| e.into_inner());
    let prog = "(defstruct Point [(x i64) (y i64)])\n\
                (defn sumpt [(p Point)] (-> i64)\n  (iadd (load (field p x)) (load (field p y))))\n\
                (defn main [] (-> i64)\n  (let [(mut p) (zeroed Point)]\n    \
                (store! (field (mut p) x) 30)\n    (store! (field (mut p) y) 12)\n    (sumpt (mut p))))\n";
    let src_path = unique_path("stg").with_extension("coil");
    std::fs::write(&src_path, prog).unwrap();
    let exe = unique_path("stexe");
    coil::build_executable_linked_dbg(prog, &exe, coil::codegen::target_triple(), &[], Some(&src_path))
        .expect("debug build");

    // Break inside sumpt (line 3); `p` is a `Point *`, deref shows the fields.
    let file = src_path.file_name().unwrap().to_str().unwrap();
    let text = run_lldb(
        &exe,
        &[&format!("breakpoint set --file {file} --line 3"), "run", "p *p", "quit"],
    );
    assert!(text.contains("(Point)"), "deref should show a Point:\n{text}");
    assert!(text.contains("x = 30") && text.contains("y = 12"), "struct fields not shown:\n{text}");

    let _ = std::fs::remove_file(&exe);
    let _ = std::fs::remove_dir_all(exe.with_extension("dSYM"));
    let _ = std::fs::remove_file(&src_path);
}

/// Aggregate DI for a slice: a `(slice u8)` string local prints as a fat-pointer
/// record `{ ptr, len }` (len = 2 for "Hi").
#[test]
fn lldb_frame_variable_shows_slice() {
    if !have("lldb") {
        eprintln!("SKIP: lldb not found");
        return;
    }
    let _g = LLDB.lock().unwrap_or_else(|e| e.into_inner());
    let prog = "(extern putchar :cc c [i32] (-> i32))\n\
                (defn main [] (-> i64)\n  (let [msg \"Hi\"]\n    (putchar 65)\n    0))\n";
    let src_path = unique_path("slg").with_extension("coil");
    std::fs::write(&src_path, prog).unwrap();
    let exe = unique_path("slexe");
    coil::build_executable_linked_dbg(prog, &exe, coil::codegen::target_triple(), &[], Some(&src_path))
        .expect("debug build");

    let file = src_path.file_name().unwrap().to_str().unwrap();
    let text = run_lldb(
        &exe,
        &[&format!("breakpoint set --file {file} --line 4"), "run", "frame variable msg", "quit"],
    );
    assert!(text.contains("len = 2"), "slice length not shown:\n{text}");

    let _ = std::fs::remove_file(&exe);
    let _ = std::fs::remove_dir_all(exe.with_extension("dSYM"));
    let _ = std::fs::remove_file(&src_path);
}

/// A self-referential struct (`next: (ptr Node)`) must build without the DIType
/// builder recursing forever — the pointer-to-self breaks the cycle.
#[test]
fn self_referential_struct_debug_builds() {
    let prog = "(defstruct Node [(val i64) (next (ptr Node))])\n\
                (defn main [] (-> i64)\n  (let [(mut n) (zeroed Node)]\n    \
                (store! (field (mut n) val) 7)\n    (load (field (mut n) val))))\n";
    let src_path = unique_path("cycg").with_extension("coil");
    std::fs::write(&src_path, prog).unwrap();
    let exe = unique_path("cycexe");
    coil::build_executable_linked_dbg(prog, &exe, coil::codegen::target_triple(), &[], Some(&src_path))
        .expect("debug build must terminate (cycle guard)");
    assert_eq!(Command::new(&exe).status().unwrap().code().unwrap(), 7);
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

    let text = run_lldb(
        &exe,
        &["breakpoint set --name main", "run", "next", "next", "next", "quit"],
    );
    // Stepping must visit successive source lines (the three putchar statements
    // are on lines 3, 4, 5). Proves per-statement line info, not just function-level.
    for line in [":3", ":4", ":5"] {
        assert!(text.contains(line), "stepping did not reach line {line}:\n{text}");
    }

    let _ = std::fs::remove_file(&exe);
    let _ = std::fs::remove_dir_all(exe.with_extension("dSYM"));
    let _ = std::fs::remove_file(&src_path);
}

/// Multi-file DWARF: a function defined in an *imported* file maps to THAT file in
/// the DWARF (its own `DIFile` + line table), not the importing main file. Before
/// per-source `DIFile`s this misattributed every imported function to `main.coil`.
#[test]
fn imported_function_maps_to_its_own_file() {
    if !have("dwarfdump") {
        eprintln!("SKIP: dwarfdump not found");
        return;
    }
    // Two files in one temp dir; main imports the helper by ABSOLUTE path (so the
    // import resolves regardless of the test's working directory).
    let helper = unique_path("helper").with_extension("coil");
    std::fs::write(&helper, "(module helper)\n(export triple)\n(defn triple [(n :i64)] (-> :i64)\n  (imul n 3))\n").unwrap();
    let main = unique_path("mainm").with_extension("coil");
    let main_src = format!(
        "(module app)\n(import \"{}\" :use *)\n(defn main [] (-> :i64)\n  (triple 7))\n",
        helper.display()
    );
    std::fs::write(&main, &main_src).unwrap();
    let exe = unique_path("mfexe");
    coil::build_executable_linked_dbg(&main_src, &exe, coil::codegen::target_triple(), &[], Some(&main))
        .expect("debug build");

    // It still computes triple(7) = 21.
    assert_eq!(Command::new(&exe).status().unwrap().code().unwrap(), 21);

    let dsym = exe.with_extension("dSYM");
    let target: &Path = if dsym.exists() { &dsym } else { &exe };
    let dump = Command::new("dwarfdump").arg("--debug-info").arg(target).output().unwrap();
    let out = String::from_utf8_lossy(&dump.stdout);

    // The imported function's subprogram must name the HELPER file at its real line
    // (3), NOT the importing main file. Look at the window after the first
    // `helper.triple` mention (the subprogram DIE), which carries decl_file/decl_line.
    let helper_name = helper.file_name().unwrap().to_str().unwrap();
    let main_name = main.file_name().unwrap().to_str().unwrap();
    let sp = out.split_once("helper.triple").map(|(_, rest)| rest).unwrap_or("");
    let frame = &sp[..sp.len().min(400)];
    assert!(frame.contains(helper_name), "imported fn should map to {helper_name}:\n{frame}");
    assert!(!frame.contains(main_name), "imported fn must NOT map to the main file {main_name}:\n{frame}");
    assert!(
        frame.contains("decl_line\t(3)") || frame.contains("decl_line (3)"),
        "imported fn should be at line 3 of its own file:\n{frame}"
    );

    let _ = std::fs::remove_file(&exe);
    let _ = std::fs::remove_dir_all(&dsym);
    let _ = std::fs::remove_file(&helper);
    let _ = std::fs::remove_file(&main);
}
