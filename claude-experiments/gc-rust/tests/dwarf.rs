//! Debugger P2: DWARF line-table emission. `gcr build` AOT objects must carry a
//! `.debug_line` table mapping instructions back to source lines, so `lldb`
//! steps through gc-rust source. We emit the object directly (the executable on
//! macOS keeps DWARF in the `.o` via a debug map) and read it with
//! `llvm-dwarfdump --debug-line`.

use gcrust::codegen::{codegen_aot_object, codegen_aot_object_level, DebugLevel};
use gcrust::compile::parse_with_prelude;
use gcrust::lower::lower_program;
use gcrust::resolve::resolve_module;
use std::path::PathBuf;
use std::process::Command;

fn lower_named(src: &str, path: &str) -> gcrust::core::CoreProgram {
    let (module, sources) = parse_with_prelude(src).expect("parse");
    let resolved = resolve_module(module).expect("resolve");
    let mut prog = lower_program(&resolved.globals).expect("lower");
    // Name source 0 (the user source) so the DIFile/line table is identifiable.
    let mut sources = sources;
    if let Some(s) = sources.get_mut(0) {
        s.path = path.to_string();
    }
    prog.sources = sources;
    prog
}

/// Resolve `llvm-dwarfdump` (prefer xcrun on macOS).
fn dwarfdump(obj: &PathBuf, section: &str) -> String {
    // Try `xcrun llvm-dwarfdump` then a bare `llvm-dwarfdump`.
    for (cmd, pre) in [("xcrun", vec!["llvm-dwarfdump"]), ("llvm-dwarfdump", vec![])] {
        let mut c = Command::new(cmd);
        c.args(&pre).arg(section).arg(obj);
        if let Ok(out) = c.output() {
            if out.status.success() {
                return String::from_utf8_lossy(&out.stdout).into_owned();
            }
        }
    }
    panic!("llvm-dwarfdump not available");
}

#[test]
fn aot_object_has_line_table_for_user_source() {
    let dir = std::env::temp_dir().join(format!("gcr_dwarf_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let obj = dir.join("prog.o");
    // Two struct literals on lines 3 and 4 → two spanned construction nodes.
    let prog = lower_named(
        "struct P { a: i64, b: i64 }\nfn main() -> i64 {\n  let p = P { a: 1, b: 2 };\n  let q = P { a: 3, b: 4 };\n  p.a + q.b\n}\n",
        "prog.gcr",
    );
    codegen_aot_object(&prog, &obj).expect("aot object");

    let line = dwarfdump(&obj, "--debug-line");
    // The line program must name the user file and emit rows for the construction
    // lines (3 and 4). dwarfdump prints a file table + address/line rows.
    assert!(line.contains("prog.gcr"), "no user file in .debug_line:\n{line}");
    assert!(
        line.contains(" 3 ") || line.contains("\t3\t") || line.contains("line 3"),
        "expected a line-3 row in the line table:\n{line}"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn full_debug_emits_named_local_and_param_dies() {
    // Debugger P3: a `--debug` (full) build must emit DWARF local-variable and
    // formal-parameter DIEs with their SOURCE names + a base type, so lldb's
    // `frame variable` shows `x`, `sum`, … (not `l7`). We dwarfdump the object's
    // `.debug_info` and check the variable DIEs are present and named. (The
    // line-tables-only default emits NONE of these — asserted below.)
    let dir = std::env::temp_dir().join(format!("gcr_dwarf_p3_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let src = "fn add(x: i64, y: i64) -> i64 {\n  let sum = x + y;\n  let doubled = sum + sum;\n  doubled\n}\nfn main() -> i64 {\n  let a = 7;\n  add(a, 35)\n}\n";

    // Full-debug object: variable + parameter DIEs present.
    let full = dir.join("full.o");
    let prog = lower_named(src, "prog.gcr");
    codegen_aot_object_level(&prog, &full, DebugLevel::Full).expect("full aot object");
    let info = dwarfdump(&full, "--debug-info");
    for needle in [
        "DW_TAG_formal_parameter",
        "DW_TAG_variable",
        "DW_AT_location",
        "(\"x\")",
        "(\"y\")",
        "(\"sum\")",
        "(\"doubled\")",
        "(\"a\")",
        "(\"i64\")", // the base type
    ] {
        assert!(
            info.contains(needle),
            "full-debug .debug_info missing `{needle}`:\n{info}"
        );
    }

    // Line-tables-only default: NONE of the gc-rust source variable DIEs. (The
    // runtime staticlib's own Rust DWARF is not in this object — only gc-rust
    // code is — so a clean object has zero variable DIEs.)
    let lines_only = dir.join("lines.o");
    let prog2 = lower_named(src, "prog.gcr");
    codegen_aot_object(&prog2, &lines_only).expect("line-table aot object");
    let info2 = dwarfdump(&lines_only, "--debug-info");
    assert!(
        !info2.contains("DW_TAG_variable") && !info2.contains("DW_TAG_formal_parameter"),
        "line-tables-only build must not emit variable DIEs:\n{info2}"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn aot_object_line_table_is_multi_source() {
    // A program that allocates via the prelude (vec_new) must produce line-table
    // rows for BOTH the user file AND the prelude (<std>) — proving per-source
    // DIFiles (the source-id foundation carried into DWARF).
    let dir = std::env::temp_dir().join(format!("gcr_dwarf2_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let obj = dir.join("prog.o");
    let prog = lower_named(
        "fn main() -> i64 {\n  let v: Vec<i64> = vec_new();\n  vec_len(vec_push(v, 7))\n}\n",
        "prog.gcr",
    );
    codegen_aot_object(&prog, &obj).expect("aot object");

    let line = dwarfdump(&obj, "--debug-line");
    assert!(line.contains("prog.gcr"), "no user file in .debug_line:\n{line}");
    assert!(
        line.contains("std") || line.contains("prelude"),
        "expected the prelude (<std>) source in the multi-source line table:\n{line}"
    );

    let _ = std::fs::remove_dir_all(&dir);
}
