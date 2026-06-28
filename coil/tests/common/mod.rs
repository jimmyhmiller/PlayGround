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

/// Popup-diagnosis timeline: append `<epoch_ms> | <event>` to a shared log that
/// the click recorder also writes to. Set COIL_DIAG=1 to enable. Lets us correlate
/// exactly which compile/link/run step was happening when a "Verifying…" popup
/// appears. No-op unless COIL_DIAG is set, so normal test runs are unaffected.
pub fn diag(event: &str) {
    use std::io::Write;
    if std::env::var_os("COIL_DIAG").is_none() {
        return;
    }
    let t = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);
    if let Ok(mut f) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open("/tmp/diag/timeline.log")
    {
        let _ = writeln!(f, "{t} | {event}");
    }
}

/// Run a compile on a thread with a large stack. The compiler is recursive
/// descent; a deep program (e.g. the macro-heavy arraylist tests) can exceed the
/// 2 MiB default *test-thread* stack, while it's fine on the 8 MiB main thread
/// `coil run` uses. Matches the `examples` test, which does the same.
fn on_big_stack<T: Send + 'static>(f: impl FnOnce() -> T + Send + 'static) -> T {
    std::thread::Builder::new()
        .stack_size(64 * 1024 * 1024)
        .spawn(f)
        .expect("spawn compile thread")
        .join()
        .expect("compile thread")
}

/// AOT-compile `src` to a native executable, run it, and return its exit code
/// (the low 8 bits of the i64 `main` returns). No JIT anywhere in the pipeline.
pub fn build_and_run(src: &str) -> i32 {
    let exe = unique_path("exe");
    let tag = exe.file_name().and_then(|s| s.to_str()).unwrap_or("?").to_string();
    let src = src.to_string();
    let exe2 = exe.clone();
    diag(&format!("COMPILE+LINK start  {tag}"));
    on_big_stack(move || coil::build_executable(&src, &exe2).expect("build_executable"));
    diag(&format!("COMPILE+LINK end    {tag}"));
    diag(&format!("RUN exe start       {tag}"));
    let code = Command::new(&exe)
        .status()
        .expect("run executable")
        .code()
        .expect("exit code");
    diag(&format!("RUN exe end         {tag}  (exit {code})"));
    let _ = std::fs::remove_file(&exe);
    code
}

/// Like `build_and_run`, but also captures stdout (for testing C interop).
pub fn build_and_capture(src: &str) -> (i32, String) {
    build_and_capture_args(src, &[])
}

/// A native architecture name for `cc -arch` / `arch -<a>`.
#[derive(Clone, Copy)]
pub enum CrossArch {
    /// The host arch (no cross-compilation; runs directly).
    Host,
    /// x86-64 (object cross-compiled, executable run under Rosetta).
    X86_64,
}

impl CrossArch {
    /// The `COIL_TARGET` triple, or `None` for the host.
    fn coil_triple(self) -> Option<&'static str> {
        match self {
            CrossArch::Host => None,
            CrossArch::X86_64 => Some("x86_64-apple-macosx11.0.0"),
        }
    }
    fn cc_arch(self) -> Option<&'static str> {
        match self {
            CrossArch::Host => None,
            CrossArch::X86_64 => Some("x86_64"),
        }
    }
}

/// True if `arch -x86_64 true` succeeds — i.e. Rosetta 2 is installed so we can
/// actually *run* a cross-compiled x86-64 binary (vs. only emitting it).
pub fn rosetta_available() -> bool {
    Command::new("arch")
        .args(["-x86_64", "true"])
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

/// Compile `src` to a Coil object for `arch`, compile the C source at `c_path`
/// for the same arch, link them, run the result (under Rosetta for a cross arch),
/// and return its exit code. The Coil object is emitted for `arch`'s triple so
/// its struct-ABI lowering matches the C compiler's.
pub fn build_link_run(src: &str, c_path: &str, arch: CrossArch) -> i32 {
    let coil_obj = unique_path("coilobj").with_extension("o");
    let c_obj = unique_path("cobj").with_extension("o");
    let exe = unique_path("linkexe");

    match arch.coil_triple() {
        Some(t) => coil::compile_to_object_for(
            src,
            &coil_obj,
            inkwell::targets::TargetTriple::create(t),
        )
        .expect("compile_to_object_for"),
        None => coil::compile_to_object(src, &coil_obj).expect("compile_to_object"),
    }

    let mut cc = Command::new("cc");
    if let Some(a) = arch.cc_arch() {
        cc.arg("-arch").arg(a);
    }
    let ok = cc
        .arg("-c")
        .arg(c_path)
        .arg("-o")
        .arg(&c_obj)
        .status()
        .expect("cc -c (helper)")
        .success();
    assert!(ok, "failed to compile C helper {c_path}");

    let mut link = Command::new("cc");
    if let Some(a) = arch.cc_arch() {
        link.arg("-arch").arg(a);
    }
    let ok = link
        .arg(&coil_obj)
        .arg(&c_obj)
        .arg("-o")
        .arg(&exe)
        .status()
        .expect("cc (link)")
        .success();
    assert!(ok, "failed to link");

    let code = match arch.cc_arch() {
        Some(a) => Command::new("arch")
            .arg(format!("-{a}"))
            .arg(&exe)
            .status()
            .expect("run cross executable")
            .code()
            .expect("exit code"),
        None => Command::new(&exe)
            .status()
            .expect("run executable")
            .code()
            .expect("exit code"),
    };

    let _ = std::fs::remove_file(&coil_obj);
    let _ = std::fs::remove_file(&c_obj);
    let _ = std::fs::remove_file(&exe);
    code
}

/// Build, run with command-line `args`, capture (exit code, stdout).
pub fn build_and_capture_args(src: &str, args: &[&str]) -> (i32, String) {
    let exe = unique_path("exe");
    let tag = exe.file_name().and_then(|s| s.to_str()).unwrap_or("?").to_string();
    let src = src.to_string();
    let exe2 = exe.clone();
    diag(&format!("COMPILE+LINK start  {tag}"));
    on_big_stack(move || coil::build_executable(&src, &exe2).expect("build_executable"));
    diag(&format!("COMPILE+LINK end    {tag}"));
    diag(&format!("RUN exe start       {tag}"));
    let out = Command::new(&exe).args(args).output().expect("run executable");
    diag(&format!("RUN exe end         {tag}"));
    let _ = std::fs::remove_file(&exe);
    (
        out.status.code().expect("exit code"),
        String::from_utf8_lossy(&out.stdout).into_owned(),
    )
}
