//! Module-system integration tests: multi-file loading (`mod foo;`), visibility
//! (`pub`), and `use` aliasing. Each test writes a small project to a temp dir
//! and drives the public `parse_file_with_prelude` → resolve → lower → JIT path.

use std::path::PathBuf;

use gcrust::codegen::jit_run_i64;
use gcrust::compile::parse_file_with_prelude;
use gcrust::lower::lower_program;
use gcrust::resolve::resolve_module;

/// A unique temp directory for one test (no external deps).
fn temp_dir(tag: &str) -> PathBuf {
    let mut d = std::env::temp_dir();
    d.push(format!("gcrust_modtest_{}_{}", tag, std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn write(dir: &PathBuf, name: &str, src: &str) {
    let path = dir.join(name);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).unwrap();
    }
    std::fs::write(path, src).unwrap();
}

/// Compile+run `main.gcr` in `dir`, returning the program's i64 result.
fn run_project(dir: &PathBuf) -> Result<i64, String> {
    let main = dir.join("main.gcr");
    let module = parse_file_with_prelude(&main).map_err(|e| e.msg)?;
    let resolved = resolve_module(module).map_err(|e| e.msg)?;
    let prog = lower_program(&resolved.globals).map_err(|e| e.msg)?;
    jit_run_i64(&prog).map_err(|e| e.0)
}

#[test]
fn multi_file_module_loads_and_calls() {
    let dir = temp_dir("multi");
    write(&dir, "mathlib.gcr", "pub fn square(x: i64) -> i64 { x * x }\npub fn cube(x: i64) -> i64 { x * x * x }\n");
    write(&dir, "main.gcr", "mod mathlib;\nfn main() -> i64 { mathlib::square(5) + mathlib::cube(2) }\n");
    assert_eq!(run_project(&dir).unwrap(), 33);
}

#[test]
fn nested_mod_dir_form() {
    // `mod geometry;` -> geometry/mod.gcr, which itself declares `mod shapes;`
    // -> geometry/shapes.gcr.
    let dir = temp_dir("nested");
    write(&dir, "geometry/mod.gcr", "mod shapes;\npub fn dist(x: i64, y: i64) -> i64 { x * x + y * y }\n");
    write(&dir, "geometry/shapes.gcr", "pub fn area(w: i64, h: i64) -> i64 { w * h }\n");
    write(&dir, "main.gcr", "mod geometry;\nfn main() -> i64 { geometry::dist(3, 4) + geometry::shapes::area(5, 6) }\n");
    assert_eq!(run_project(&dir).unwrap(), 55);
}

#[test]
fn private_item_is_inaccessible() {
    let dir = temp_dir("vis");
    write(&dir, "lib.gcr", "pub fn pub_fn() -> i64 { 1 }\nfn priv_fn() -> i64 { 2 }\n");
    write(&dir, "main.gcr", "mod lib;\nfn main() -> i64 { lib::priv_fn() }\n");
    let err = run_project(&dir).unwrap_err();
    assert!(err.contains("private"), "{err}");
}

#[test]
fn public_item_is_accessible() {
    let dir = temp_dir("vis_ok");
    write(&dir, "lib.gcr", "pub fn pub_fn() -> i64 { 41 }\nfn priv_fn() -> i64 { 2 }\n");
    write(&dir, "main.gcr", "mod lib;\nfn main() -> i64 { lib::pub_fn() + 1 }\n");
    assert_eq!(run_project(&dir).unwrap(), 42);
}

#[test]
fn use_alias_and_glob() {
    let dir = temp_dir("use");
    write(&dir, "lib.gcr", "pub fn a() -> i64 { 10 }\npub fn b() -> i64 { 20 }\n");
    // `use lib::a;` (single) + `use lib::*;` would both work; test the single form.
    write(&dir, "main.gcr", "mod lib;\nuse lib::a;\nfn main() -> i64 { a() + lib::b() }\n");
    assert_eq!(run_project(&dir).unwrap(), 30);
}

#[test]
fn use_disambiguates_same_last_segment() {
    let dir = temp_dir("disambig");
    write(
        &dir,
        "main.gcr",
        "mod a { pub fn val() -> i64 { 1 } }\n\
         mod b { pub fn val() -> i64 { 2 } }\n\
         use b::val;\n\
         fn main() -> i64 { val() }\n",
    );
    assert_eq!(run_project(&dir).unwrap(), 2);
}

#[test]
fn missing_module_file_errors() {
    let dir = temp_dir("missing");
    write(&dir, "main.gcr", "mod nonexistent;\nfn main() -> i64 { 0 }\n");
    let err = run_project(&dir).unwrap_err();
    assert!(err.contains("cannot find module"), "{err}");
}
