//! Point the linker at the AOT-compiled `libjs_stage1.a`.
//!
//! The static lib + its Rust bindings (`generated/js_stage1.rs`) are produced by
//!   simd-lang compile examples/js_stage1.simd -o aot-bench/generated
//! Re-run that whenever `js_stage1.simd` changes. (Run from the simd-lang root.)

use std::path::PathBuf;

fn main() {
    let generated = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("generated");
    let lib = generated.join("libjs_stage1.a");

    if !lib.exists() {
        panic!(
            "missing {}\n\nGenerate it first (from the simd-lang root):\n  \
             simd-lang compile examples/js_stage1.simd -o aot-bench/generated",
            lib.display()
        );
    }

    println!("cargo:rustc-link-search=native={}", generated.display());
    println!("cargo:rerun-if-changed={}", lib.display());
    println!(
        "cargo:rerun-if-changed={}",
        generated.join("js_stage1.rs").display()
    );
}
