// Probe the Vercel Functions (Fluid) runtime sandbox to settle the one
// open question for running ai-lang there: can the process `dlopen`
// libcurl / libcrypto / libssl at runtime (ai-lang's FFI model), and what
// architecture / libc / system libs are actually present? Everything here
// is reported as JSON so a single `curl <deployment>/api/probe` answers it.

use serde_json::{json, Value};
use vercel_runtime::{run, service_fn, Error, Request};

/// Try to `dlopen` a library by name; if it opens, also try to resolve a
/// known symbol so we know it's the real, usable library (not just a stub).
fn try_open(name: &str, symbol: &[u8]) -> Value {
    match unsafe { libloading::Library::new(name) } {
        Ok(lib) => {
            let sym_ok = unsafe {
                lib.get::<unsafe extern "C" fn()>(symbol).is_ok()
            };
            json!({ "ok": true, "symbol_resolved": sym_ok })
        }
        Err(e) => json!({ "ok": false, "error": e.to_string() }),
    }
}

/// List .so files in `dir` whose name mentions curl/crypto/ssl.
fn list_libs(dir: &str) -> Vec<String> {
    std::fs::read_dir(dir)
        .map(|rd| {
            let mut v: Vec<String> = rd
                .filter_map(|e| e.ok())
                .map(|e| e.file_name().to_string_lossy().into_owned())
                .filter(|n| n.contains("curl") || n.contains("crypto") || n.contains("ssl"))
                .collect();
            v.sort();
            v
        })
        .unwrap_or_default()
}

fn read_file(path: &str) -> String {
    std::fs::read_to_string(path).unwrap_or_default()
}

#[tokio::main]
async fn main() -> Result<(), Error> {
    run(service_fn(handler)).await
}

async fn handler(_req: Request) -> Result<Value, Error> {
    // Candidate sonames (Amazon Linux 2023 ships .so.N variants).
    let dlopen = json!({
        "libcurl.so.4":      try_open("libcurl.so.4", b"curl_easy_init"),
        "libcurl.so":        try_open("libcurl.so", b"curl_easy_init"),
        "libcrypto.so.3":    try_open("libcrypto.so.3", b"SHA256"),
        "libcrypto.so.1.1":  try_open("libcrypto.so.1.1", b"SHA256"),
        "libcrypto.so":      try_open("libcrypto.so", b"SHA256"),
        "libssl.so.3":       try_open("libssl.so.3", b"SSL_new"),
        "libssl.so":         try_open("libssl.so", b"SSL_new"),
    });

    let dirs = json!({
        "/lib64":        list_libs("/lib64"),
        "/usr/lib64":    list_libs("/usr/lib64"),
        "/lib":          list_libs("/lib"),
        "/usr/lib":      list_libs("/usr/lib"),
        "/var/task":     list_libs("/var/task"),
        "/opt/lib":      list_libs("/opt/lib"),
    });

    Ok(json!({
        "arch": std::env::consts::ARCH,
        "os": std::env::consts::OS,
        "os_release": read_file("/etc/os-release"),
        "kernel": read_file("/proc/version"),
        "ld_library_path": std::env::var("LD_LIBRARY_PATH").unwrap_or_default(),
        "cwd": std::env::current_dir().map(|p| p.display().to_string()).unwrap_or_default(),
        "dlopen": dlopen,
        "lib_dirs": dirs,
    }))
}
