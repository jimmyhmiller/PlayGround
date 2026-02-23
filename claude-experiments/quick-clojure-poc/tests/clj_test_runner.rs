/// Runs Clojure test files as single-process integration tests.
///
/// Each test function runs ONE Clojure file that contains many assertions.
/// The Clojure test framework (test_helper.clj) prints a summary and throws
/// on failure, causing a non-zero exit code.

use std::path::PathBuf;
use std::process::Command;
use std::sync::OnceLock;

static BINARY_PATH: OnceLock<PathBuf> = OnceLock::new();

fn get_binary_path() -> &'static PathBuf {
    BINARY_PATH.get_or_init(|| {
        let status = Command::new("cargo")
            .args(&["build", "--release", "--quiet"])
            .status()
            .expect("Failed to build release binary");
        assert!(status.success(), "Failed to build release binary");
        let manifest_dir =
            std::env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".to_string());
        PathBuf::from(manifest_dir).join("target/release/quick-clojure-poc")
    })
}

fn run_clj_test(filename: &str) {
    let binary_path = get_binary_path();
    let output = Command::new(binary_path.as_os_str())
        .arg(filename)
        .output()
        .expect("Failed to execute");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    // Always print stdout so test results are visible
    if !stdout.is_empty() {
        println!("{}", stdout);
    }

    assert!(
        output.status.success(),
        "\n{} failed:\nStdout:\n{}\nStderr:\n{}",
        filename, stdout, stderr
    );
}

#[test]
fn clj_ported() {
    run_clj_test("tests/clj/clj_ported_test.clj");
}

#[test]
fn cljs_ported_core() {
    run_clj_test("tests/clj/cljs_ported_core_test.clj");
}

#[test]
fn sci_ported_core() {
    run_clj_test("tests/clj/sci_ported_core_test.clj");
}

#[test]
fn sci_ported_extra() {
    run_clj_test("tests/clj/sci_ported_extra_test.clj");
}
