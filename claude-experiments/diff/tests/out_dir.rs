// `diffpack build --out-dir` resolution.
//
// Vite resolves `build.outDir` against the project root; diffpack used to resolve an
// explicit `--out-dir` against the process CWD instead, so building a project from a
// different directory scattered the output next to the caller and left the project
// with no build at all. These tests drive the real binary from an UNRELATED CWD,
// which is the only place the bug is observable.
//
// Soft-skips when `target/release/diffpack` has not been built (same convention as
// the gated sub-block in tests/next_corpus.rs).

use std::path::{Path, PathBuf};
use std::process::Command;

use tempfile::tempdir;

fn release_binary() -> Option<PathBuf> {
    let binary = Path::new(env!("CARGO_MANIFEST_DIR")).join("target/release/diffpack");
    binary.is_file().then_some(binary)
}

/// A minimal HTML-rooted web app: one page, one module script, no dependencies.
fn write_project(root: &Path) {
    std::fs::write(
        root.join("index.html"),
        "<!doctype html><html><head><title>out-dir</title></head>\
         <body><div id=\"app\"></div>\
         <script type=\"module\" src=\"/main.js\"></script></body></html>",
    )
    .unwrap();
    std::fs::write(
        root.join("main.js"),
        "document.getElementById('app').textContent = 'ok';\n",
    )
    .unwrap();
}

fn build(binary: &Path, root: &Path, cwd: &Path, out_dir: Option<&str>) {
    let mut command = Command::new(binary);
    command.arg("build").arg(root).current_dir(cwd);
    if let Some(out_dir) = out_dir {
        command.arg("--out-dir").arg(out_dir);
    }
    let output = command.output().unwrap();
    assert!(
        output.status.success(),
        "build failed: {}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn relative_out_dir_resolves_against_the_project_root_not_the_cwd() {
    let Some(binary) = release_binary() else {
        eprintln!("skipping: target/release/diffpack not built");
        return;
    };
    let project = tempdir().unwrap();
    let elsewhere = tempdir().unwrap();
    write_project(project.path());

    build(
        &binary,
        project.path(),
        elsewhere.path(),
        Some("dist-custom"),
    );

    assert!(
        project.path().join("dist-custom/index.html").is_file(),
        "a relative --out-dir must land under the project root"
    );
    assert!(
        !elsewhere.path().join("dist-custom").exists(),
        "a relative --out-dir must NOT be resolved against the process CWD"
    );
}

#[test]
fn absolute_out_dir_is_used_as_given() {
    let Some(binary) = release_binary() else {
        eprintln!("skipping: target/release/diffpack not built");
        return;
    };
    let project = tempdir().unwrap();
    let elsewhere = tempdir().unwrap();
    let out_dir = elsewhere.path().join("somewhere/else");
    write_project(project.path());

    build(
        &binary,
        project.path(),
        elsewhere.path(),
        Some(out_dir.to_str().unwrap()),
    );

    assert!(out_dir.join("index.html").is_file());
    assert!(!project.path().join("dist").exists());
}

#[test]
fn default_out_dir_is_the_project_roots_dist() {
    let Some(binary) = release_binary() else {
        eprintln!("skipping: target/release/diffpack not built");
        return;
    };
    let project = tempdir().unwrap();
    let elsewhere = tempdir().unwrap();
    write_project(project.path());

    build(&binary, project.path(), elsewhere.path(), None);

    assert!(project.path().join("dist/index.html").is_file());
    assert!(!elsewhere.path().join("dist").exists());
}
