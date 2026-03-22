fn main() {
    println!("cargo:rustc-link-search=native=/tmp/hprof-simd-output");
    println!("cargo:rustc-link-lib=static=hprof_parser");

    // MLIR/LLVM runtime libs needed for lowered code
    let llvm_ldflags = std::process::Command::new("llvm-config")
        .arg("--ldflags")
        .output();

    // Link against system libs the MLIR-generated code may need
    println!("cargo:rustc-link-lib=dylib=c++");
}
