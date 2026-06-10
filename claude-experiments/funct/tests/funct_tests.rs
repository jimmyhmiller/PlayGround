//! The funct-native test framework: `#[test]` annotations, assertions,
//! per-test isolation, and the runner.

use funct::testing::{discover_files, run_test_file};
use funct::Funct;
use std::path::PathBuf;

fn tmpdir(files: &[(&str, &str)]) -> PathBuf {
    use std::sync::atomic::{AtomicU64, Ordering};
    static N: AtomicU64 = AtomicU64::new(0);
    let dir = std::env::temp_dir().join(format!(
        "funct_runner_test_{}_{}",
        std::process::id(),
        N.fetch_add(1, Ordering::SeqCst)
    ));
    for (path, src) in files {
        let full = dir.join(path);
        std::fs::create_dir_all(full.parent().unwrap()).unwrap();
        std::fs::write(full, src).unwrap();
    }
    dir
}

#[test]
fn annotation_marks_tests_not_names() {
    let dir = tmpdir(&[(
        "m.ft",
        r#"
fn helper() = 1

#[test]
fn anything_at_all() {
    assert_eq(helper(), 1)
}

// looks like a test by name, but has no annotation -> never runs
fn test_named_but_not_annotated() {
    fail("must not run")
}
"#,
    )]);
    let report = run_test_file(&dir.join("m.ft"));
    assert!(report.load_error.is_none());
    assert_eq!(report.outcomes.len(), 1);
    assert_eq!(report.outcomes[0].name, "anything_at_all");
    assert!(report.ok());
}

#[test]
fn failures_report_function_line_and_values() {
    let dir = tmpdir(&[(
        "m.ft",
        r#"
fn double(x) = x * 3 // bug

#[test]
fn doubles() {
    assert_eq(double(2), 4)
}

#[test]
fn still_passes() {
    assert(true)
}
"#,
    )]);
    let report = run_test_file(&dir.join("m.ft"));
    assert_eq!(report.passed(), 1);
    assert_eq!(report.failed(), 1);
    let err = report.outcomes[0].error.as_deref().unwrap();
    assert!(err.contains("assert_eq failed"), "{}", err);
    assert!(err.contains("left:  6") && err.contains("right: 4"), "{}", err);
    assert!(err.contains("doubles:"), "should point at the failing fn: {}", err);
}

#[test]
fn tests_are_isolated_per_run() {
    // each test re-evaluates the file: shared top-level atoms reset
    let dir = tmpdir(&[(
        "m.ft",
        r#"
let counter = atom(0)

#[test]
fn first_bump() {
    swap!(counter, n => n + 1)
    assert_eq(@counter, 1)
}

#[test]
fn second_bump_sees_fresh_state() {
    swap!(counter, n => n + 1)
    assert_eq(@counter, 1)
}
"#,
    )]);
    let report = run_test_file(&dir.join("m.ft"));
    assert!(report.ok(), "{:?}", report.outcomes.iter().filter_map(|o| o.error.clone()).collect::<Vec<_>>());
}

#[test]
fn importing_a_module_does_not_run_its_tests() {
    let dir = tmpdir(&[
        (
            "lib.ft",
            r#"
export fn one() = 1

#[test]
fn lib_test_runs_only_when_lib_is_the_target() {
    fail("must not run from the importer")
}
"#,
        ),
        (
            "main.ft",
            r#"
import { one } from "lib"

#[test]
fn uses_the_module() {
    assert_eq(one(), 1)
}
"#,
        ),
    ]);
    let report = run_test_file(&dir.join("main.ft"));
    assert_eq!(report.outcomes.len(), 1, "only main.ft's own test runs");
    assert!(report.ok());

    // running the module file directly DOES run its (failing) test
    let lib_report = run_test_file(&dir.join("lib.ft"));
    assert_eq!(lib_report.failed(), 1);
}

#[test]
fn directory_scan_finds_tests_in_any_ft_file() {
    let dir = tmpdir(&[
        ("a.ft", "#[test]\nfn a_works() { assert(true) }"),
        ("sub/b.ft", "#[test]\nfn b_works() { assert(true) }"),
        ("no_tests.ft", "fn plain() = 1"),
    ]);
    let files = discover_files(&dir);
    assert_eq!(files.len(), 3);
    let total: usize = files.iter().map(|f| run_test_file(f).outcomes.len()).sum();
    assert_eq!(total, 2);
}

#[test]
fn load_errors_are_reported() {
    let dir = tmpdir(&[("bad.ft", "fn oops( = 1")]);
    let report = run_test_file(&dir.join("bad.ft"));
    assert!(report.load_error.is_some());
    assert!(!report.ok());
}

#[test]
fn attribute_validation() {
    let mut vm = Funct::new();
    // unknown attribute is a loud parse error
    let err = vm.eval("#[bench]\nfn f() { 1 }").unwrap_err().to_string();
    assert!(err.contains("unknown attribute"), "{}", err);
    // attributes only apply to fns
    let err = vm.eval("#[test]\nlet x = 1").unwrap_err().to_string();
    assert!(err.contains("only be applied to `fn`"), "{}", err);
    // test fns take no arguments
    let err = vm.eval("#[test]\nfn f(x) { x }").unwrap_err().to_string();
    assert!(err.contains("no arguments"), "{}", err);
}

#[test]
fn annotated_fns_are_ordinary_fns() {
    let mut vm = Funct::new();
    vm.eval("#[test]\nfn check() { assert_eq(1, 1) }\nfn plain() = 2").unwrap();
    // callable like any fn; evaluating the file never auto-runs tests
    vm.call("check", vec![]).unwrap();
    assert_eq!(vm.test_names(), vec!["check".to_string()]);
}

#[test]
fn assertion_builtins() {
    let mut vm = Funct::new();
    vm.eval("assert(1 < 2)").unwrap();
    vm.eval("assert_eq([1, 2], [1, 2])").unwrap();
    vm.eval("assert_ne(1, 2)").unwrap();
    let e = vm.eval("assert(1 > 2, \"one is not bigger\")").unwrap_err().to_string();
    assert!(e.contains("one is not bigger"), "{}", e);
    let e = vm.eval("assert_eq(\"a\", \"b\", \"labels differ\")").unwrap_err().to_string();
    assert!(e.contains("labels differ") && e.contains("\"a\"") && e.contains("\"b\""), "{}", e);
    let e = vm.eval("fail(\"nope\")").unwrap_err().to_string();
    assert!(e.contains("nope"), "{}", e);
}

#[test]
fn example_files_pass() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("examples");
    for f in ["geometry.ft", "geometry_test.ft"] {
        let report = run_test_file(&root.join(f));
        assert!(report.load_error.is_none(), "{:?}", report.load_error);
        assert!(report.outcomes.len() >= 2);
        assert!(
            report.ok(),
            "{}: {:?}",
            f,
            report.outcomes.iter().filter_map(|o| o.error.clone()).collect::<Vec<_>>()
        );
    }
}
