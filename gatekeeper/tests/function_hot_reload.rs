//! Verifies that a function dylib is hot-reloaded when the file on disk changes
//! (atomic rename), so shipping a new build of a function takes effect on the
//! next request — no gate restart, no config reload.
//!
//! Uses the two example dylibs as "two versions" of a function at one path:
//! `libhello_fn.so` (v1, serves an HTML greeting) and `libanalytics_fn.so` (v2,
//! returns 404 for `/hello` since it has no such endpoint). We invoke through a
//! single `FunctionRegistry` at a fixed path, then atomically rename v2 over the
//! path and invoke again; the registry must serve v2 without being recreated.
//!
//! IMPORTANT: deploy is via rename(2), never an in-place overwrite of a loaded
//! `.so` (that corrupts the live mapping and crashes the process). The registry
//! keys its staleness check on (mtime, size, inode); rename changes the inode.

use std::path::{Path, PathBuf};

use gatekeeper::function::FunctionRegistry;

fn target_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("target")
        .join(if cfg!(debug_assertions) { "debug" } else { "release" })
}

fn require(p: &Path) -> PathBuf {
    assert!(
        p.exists(),
        "missing {} — run `cargo build -p hello-fn -p analytics-fn` first",
        p.display()
    );
    p.to_path_buf()
}

/// Atomically place `src`'s contents at `dst` via a temp file + rename, the
/// only safe way to replace a possibly-loaded dylib.
fn deploy(src: &Path, dst: &Path) {
    let tmp = dst.with_extension("so.new");
    std::fs::copy(src, &tmp).expect("copy to temp");
    std::fs::rename(&tmp, dst).expect("atomic rename into place");
}

#[test]
fn rebuilt_dylib_is_picked_up_without_restart() {
    let hello = require(&target_dir().join("libhello_fn.so"));
    let analytics = require(&target_dir().join("libanalytics_fn.so"));

    // A unique per-test path under the target dir (avoids cross-test races).
    let live = target_dir().join("libhotswap_test.so");
    let _ = std::fs::remove_file(&live);

    // v1 = hello.
    deploy(&hello, &live);
    let reg = FunctionRegistry::new();
    let r1 = reg.invoke(&live, "GET", "/hello", "", &[], b"");
    assert_eq!(r1.status, 200, "v1 hello should serve 200");
    let body1 = String::from_utf8_lossy(&r1.body);
    assert!(
        body1.contains("hello from a gatekeeper function"),
        "v1 should be the hello handler, got: {body1}"
    );

    // Deploy v2 = analytics over the same path (atomic rename). The analytics
    // handler has no "/hello" endpoint, so it returns 404 — a behavior change
    // that proves the NEW code is running.
    deploy(&analytics, &live);
    let r2 = reg.invoke(&live, "GET", "/hello", "", &[], b"");
    assert_eq!(
        r2.status, 404,
        "after hot-swap, the analytics handler should 404 on /hello (proving v2 is live); \
         got {} with body {:?}",
        r2.status,
        String::from_utf8_lossy(&r2.body)
    );

    // And swapping BACK to v1 is picked up too (not a one-way latch).
    deploy(&hello, &live);
    let r3 = reg.invoke(&live, "GET", "/hello", "", &[], b"");
    assert_eq!(r3.status, 200, "swapping back to hello should serve 200 again");
    assert!(String::from_utf8_lossy(&r3.body).contains("hello from a gatekeeper function"));

    let _ = std::fs::remove_file(&live);
}

#[test]
fn unchanged_dylib_is_served_from_cache() {
    // Sanity: repeated invokes of an unchanged file must keep working (the fast
    // path) — the stat-per-call must not break the warm cache.
    let hello = require(&target_dir().join("libhello_fn.so"));
    let live = target_dir().join("libhotswap_cache_test.so");
    let _ = std::fs::remove_file(&live);
    deploy(&hello, &live);

    let reg = FunctionRegistry::new();
    for _ in 0..5 {
        let r = reg.invoke(&live, "GET", "/x", "", &[], b"");
        assert_eq!(r.status, 200);
    }
    let _ = std::fs::remove_file(&live);
}
