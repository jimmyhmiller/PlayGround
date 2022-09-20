extern crate cpp_build;

fn main() {
    let include_path = "/opt/homebrew/opt/llvm/include";
    let lib_path = "/opt/homebrew/opt/llvm/lib";

    let mut build_config = cpp_build::Config::new();
    build_config.include(include_path);

    build_config.build("src/lib.rs");
    println!("cargo:rustc-link-search={}", lib_path);
    println!("cargo:rustc-link-lib=dylib=lldb");

}
