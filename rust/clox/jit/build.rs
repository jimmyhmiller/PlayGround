extern crate bindgen;

use std::env;
use std::path::{Path};

fn main() {

    let dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let dir = Path::new(&dir);
    let clox_src = dir.parent().unwrap().join("src");
    println!("cargo:rustc-link-search={}", clox_src.display());

    // Tell cargo to invalidate the built crate whenever the wrapper changes
    println!("cargo:rerun-if-changed={}", clox_src.join("vm.h").display());

    // The bindgen::Builder is the main entry point
    // to bindgen, and lets you build up options for
    // the resulting bindings.
    let bindings = bindgen::Builder::default()
        // The input header we would like to generate
        // bindings for.
        .header(clox_src.join("vm.h").to_str().unwrap())
        // Tell cargo to invalidate the built crate whenever any of the
        // included header files changed.
        .parse_callbacks(Box::new(bindgen::CargoCallbacks))
        .whitelist_type("Chunk")
        .whitelist_type("Table")
        .whitelist_type("ObjType")
        .whitelist_type("ObjFunction")
        .whitelist_type("ObjNative")
        .whitelist_type("ObjString")
        .whitelist_type("ObjUpvalue")
        .whitelist_type("ObjClosure")
        .whitelist_type("ObjClass")
        .whitelist_type("ObjInstance")
        .whitelist_type("ObjBoundMethod")
        .whitelist_type("CallFrame")
        .whitelist_type("VM")
        .whitelist_type("InterpretResult")
        // Finish the builder and generate the bindings.
        .generate()
        // Unwrap the Result and panic on failure.
        .expect("Unable to generate bindings");

    // Write the bindings to the $OUT_DIR/bindings.rs file.
    let out_path = dir.join("src");
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}
