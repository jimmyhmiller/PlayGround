fn main() {
    // Link the native engine C++ wrapper
    println!("cargo:rustc-link-search=native=/Users/jimmyhmiller/Documents/Code/open-source/melior/melior/cpp");
    println!("cargo:rustc-link-lib=static=native_engine");
    // MLIR/LLVM symbols are already provided by mlir-sys, no need to link again
}
