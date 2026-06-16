//! Build script for `gc-rust`. It bakes the profile-matched path of the
//! `gcrust-rt` runtime staticlib (`libgcrust_rt.a`) into the binary, so the AOT
//! linker in `codegen.rs` finds a runtime built with the SAME cargo profile as
//! this `gcr` (a release `gcr` links the release runtime, a debug `gcr` the
//! debug one — avoiding the silent ~4x slowdown of linking a debug runtime into
//! a release build).
//!
//! `gcrust-rt` is `crate-type = ["rlib", "staticlib"]`. Cargo builds the
//! staticlib whenever the `gcrust-rt` package is built for this profile (e.g.
//! `cargo build -p gcrust-rt`, or `cargo build --workspace`, both of which the
//! project's `cargo build` alias does). We derive that staticlib's path from
//! `OUT_DIR` and export it as `GCRUST_RT_STATICLIB`. Forcing the staticlib build
//! from inside this script is avoided on purpose: a nested `cargo` under the
//! parent build's target lock is unreliable. `$GCRUST_RUNTIME_LIB` overrides the
//! baked path for unusual setups.

use std::path::PathBuf;

fn main() {
    // OUT_DIR is  <target>/<profile>/build/gc-rust-<hash>/out  → the profile dir
    // (where cargo places the dependency's staticlib) is four levels up.
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let profile_dir = out_dir
        .ancestors()
        .nth(3)
        .expect("OUT_DIR has the expected cargo layout (.../<profile>/build/<pkg>/out)")
        .to_path_buf();
    let lib = profile_dir.join("libgcrust_rt.a");

    println!("cargo:rustc-env=GCRUST_RT_STATICLIB={}", lib.display());
    println!("cargo:rerun-if-changed=build.rs");
}
