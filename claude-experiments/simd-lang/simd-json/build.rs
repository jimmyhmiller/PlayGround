fn main() {
    println!("cargo:rustc-link-search=native={}/lib", env!("CARGO_MANIFEST_DIR"));
    println!("cargo:rustc-link-lib=static=json_stage1");
}
