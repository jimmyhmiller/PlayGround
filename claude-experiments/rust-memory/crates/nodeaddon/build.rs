//! A Node addon links against N-API symbols that only exist inside the host
//! process (`node` itself exports them), so the `napi_*` calls must stay
//! undefined at link time and be resolved by dyld when node loads the module.
//! This is the same flag every Rust-Node-addon toolchain sets; without it the
//! link fails with "undefined symbols: _napi_create_function".

fn main() {
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
        println!("cargo:rustc-link-arg-cdylib=-undefined");
        println!("cargo:rustc-link-arg-cdylib=dynamic_lookup");
    }
    // ELF leaves undefined symbols in a shared object alone by default.
}
