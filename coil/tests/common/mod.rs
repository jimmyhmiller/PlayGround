//! Shared test helpers. Coil has no JIT/eval, so the only way to check a
//! program's result is to AOT-compile it to a native executable and run it.
#![allow(dead_code)]

use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicU32, Ordering};

static N: AtomicU32 = AtomicU32::new(0);

pub fn unique_path(tag: &str) -> PathBuf {
    let n = N.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!("coil_test_{}_{}_{}", tag, std::process::id(), n))
}

/// AOT-compile `src` to a native executable, run it, and return its exit code
/// (the low 8 bits of the i64 `main` returns). No JIT anywhere in the pipeline.
pub fn build_and_run(src: &str) -> i32 {
    let exe = unique_path("exe");
    coil::build_executable(src, &exe).expect("build_executable");
    let code = Command::new(&exe)
        .status()
        .expect("run executable")
        .code()
        .expect("exit code");
    let _ = std::fs::remove_file(&exe);
    code
}

/// Like `build_and_run`, but also captures stdout (for testing C interop).
pub fn build_and_capture(src: &str) -> (i32, String) {
    build_and_capture_args(src, &[])
}

/// Build, run with command-line `args`, capture (exit code, stdout).
pub fn build_and_capture_args(src: &str, args: &[&str]) -> (i32, String) {
    let exe = unique_path("exe");
    coil::build_executable(src, &exe).expect("build_executable");
    let out = Command::new(&exe).args(args).output().expect("run executable");
    let _ = std::fs::remove_file(&exe);
    (
        out.status.code().expect("exit code"),
        String::from_utf8_lossy(&out.stdout).into_owned(),
    )
}
