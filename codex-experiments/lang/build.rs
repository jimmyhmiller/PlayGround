use std::process::Command;

fn main() {
    let llvm_config = find_llvm_config();
    let cflags = Command::new(&llvm_config)
        .arg("--cflags")
        .output()
        .expect("failed to run llvm-config --cflags");
    let cflags_str = String::from_utf8_lossy(&cflags.stdout);

    // Compile LLVM shims (needs LLVM cflags)
    let mut cc = cc::Build::new();
    cc.file("runtime/llvm_shims.c");
    for flag in cflags_str.split_whitespace() {
        if !flag.is_empty() {
            cc.flag(flag);
        }
    }
    cc.compile("llvm_shims");

    // Compile runtime.c
    cc::Build::new()
        .file("runtime/runtime.c")
        .compile("lang_runtime_c");

    // Compile gc_bridge.c (needs gc-library include path)
    let gc_include = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../claude-experiments/gc-library/include");
    cc::Build::new()
        .file("runtime/gc_bridge.c")
        .include(&gc_include)
        .compile("gc_bridge");

    // Link gc-library static lib
    let gc_lib_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../claude-experiments/gc-library/target/release");
    println!("cargo:rustc-link-search=native={}", gc_lib_dir.display());
    println!("cargo:rustc-link-lib=static=gc_library");
}

fn find_llvm_config() -> String {
    let candidates = ["llvm-config", "/opt/homebrew/opt/llvm/bin/llvm-config", "/usr/local/opt/llvm/bin/llvm-config"];
    for c in &candidates {
        if let Ok(output) = Command::new(c).arg("--version").output() {
            if output.status.success() {
                return c.to_string();
            }
        }
    }
    "llvm-config".to_string()
}
