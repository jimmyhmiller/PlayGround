//! End-to-end test of the function (serverless dylib) backend through the real
//! C ABI. Builds nothing itself — it loads the already-built `hello-fn` dylib
//! and drives it the same way the gate does, exercising the full
//! marshal → dispatch → copy → free cycle across the boundary.
//!
//! Run under valgrind to validate the unsafe memory handling:
//!   cargo test --test function_abi --no-run
//!   valgrind --leak-check=full --error-exitcode=1 \
//!     ./target/debug/deps/function_abi-*  (the test binary)
//!
//! The assertions here cover behaviour; valgrind covers soundness (no leak, no
//! invalid read/write, no double free across the gate/dylib allocator split).

use std::path::PathBuf;

use gatekeeper::function::FunctionRegistry;

/// Path to the example dylib, built by `cargo build -p hello-fn`. We locate it
/// relative to the test binary's target dir so it works in debug and release.
fn dylib_path() -> PathBuf {
    let ext = if cfg!(target_os = "macos") {
        "dylib"
    } else if cfg!(target_os = "windows") {
        "dll"
    } else {
        "so"
    };
    let name = if cfg!(target_os = "windows") {
        format!("hello_fn.{ext}")
    } else {
        format!("libhello_fn.{ext}")
    };
    // tests run from the crate root; the workspace target dir is ./target.
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("target")
        .join(if cfg!(debug_assertions) { "debug" } else { "release" })
        .join(&name);
    assert!(
        p.exists(),
        "dylib not built at {} — run `cargo build -p hello-fn` first",
        p.display()
    );
    p
}

fn hdr(name: &str, value: &str) -> tiny_http::Header {
    tiny_http::Header::from_bytes(name.as_bytes(), value.as_bytes()).unwrap()
}

#[test]
fn invoke_covers_request_shapes() {
    let reg = FunctionRegistry::new();
    let lib = dylib_path();

    // Plain GET, no headers, no body -> the catch-all HTML response.
    let r = reg.invoke(&lib, "GET", "/anything", "", &[], &[]);
    assert_eq!(r.status, 200);
    assert!(String::from_utf8_lossy(&r.body).contains("/anything"));
    assert!(r
        .headers
        .iter()
        .any(|(k, v)| k == "Content-Type" && v.contains("text/html")));

    // Health route.
    let r = reg.invoke(&lib, "GET", "/health", "", &[], &[]);
    assert_eq!(r.status, 200);
    assert_eq!(r.body, b"ok");

    // Echo with headers, query and a body — exercises every borrowed field.
    let headers = [hdr("X-Test", "abc"), hdr("Accept", "application/json")];
    let r = reg.invoke(&lib, "POST", "/echo", "a=1&b=2", &headers, b"payload");
    assert_eq!(r.status, 200);
    let body = String::from_utf8_lossy(&r.body);
    assert!(body.contains("\"method\":\"POST\""), "{body}");
    assert!(body.contains("\"query\":\"a=1&b=2\""), "{body}");
    assert!(body.contains("\"body\":\"payload\""), "{body}");
}

#[test]
fn describe_returns_self_description() {
    // The hello dylib exports gk_describe (ABI v2). The registry should fetch it
    // and return valid JSON naming the function and its endpoints.
    let reg = FunctionRegistry::new();
    let lib = dylib_path();
    let desc = reg
        .describe(&lib)
        .expect("hello exports gk_describe (required) -> json");
    assert!(desc.contains("\"name\":\"hello\""), "got: {desc}");
    assert!(desc.contains("\"/health\""), "should list the /health endpoint: {desc}");
    // It must be valid JSON.
    let v: serde_json::Value = serde_json::from_str(&desc).expect("describe is valid JSON");
    assert!(v.get("endpoints").and_then(|e| e.as_array()).is_some());
}

#[test]
fn function_without_describe_is_rejected() {
    // #[describe] is REQUIRED. The nodescribe-fn fixture has a #[handler] but no
    // #[describe], so the gate must refuse to load it (invoke -> 502, describe ->
    // Err), guaranteeing the catalog can never have an undocumented function.
    let lib = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("target")
        .join(if cfg!(debug_assertions) { "debug" } else { "release" })
        .join("libnodescribe_fn.so");
    if !lib.exists() {
        // Built by `cargo build -p nodescribe-fn`; skip cleanly if absent.
        eprintln!("skipping: {} not built", lib.display());
        return;
    }
    let reg = FunctionRegistry::new();
    let r = reg.invoke(&lib, "GET", "/x", "", &[], &[]);
    assert_eq!(r.status, 502, "a function without #[describe] must fail to load");
    assert!(
        reg.describe(&lib).is_err(),
        "describe() on a no-describe dylib must error, not silently succeed"
    );
}

#[test]
fn handler_panic_becomes_500_not_abort() {
    let reg = FunctionRegistry::new();
    let lib = dylib_path();
    // The /panic route panics; the SDK catches it and returns 500. If the panic
    // unwound across the ABI this test would abort the whole process instead.
    let r = reg.invoke(&lib, "GET", "/panic", "", &[], &[]);
    assert_eq!(r.status, 500);
}

#[test]
fn repeated_invocations_reuse_cached_library() {
    // Many calls through one registry: loads once, then reuses. Under valgrind
    // this also checks we don't leak per-call (every response is freed).
    let reg = FunctionRegistry::new();
    let lib = dylib_path();
    for i in 0..50 {
        let q = format!("n={i}");
        let r = reg.invoke(&lib, "GET", "/echo", &q, &[hdr("X-N", &q)], q.as_bytes());
        assert_eq!(r.status, 200);
    }
}

#[test]
fn missing_dylib_fails_closed() {
    let reg = FunctionRegistry::new();
    let r = reg.invoke(
        &PathBuf::from("definitely/not/a/real.so"),
        "GET",
        "/x",
        "",
        &[],
        &[],
    );
    // Load failure -> 502, never a panic.
    assert_eq!(r.status, 502);
}
