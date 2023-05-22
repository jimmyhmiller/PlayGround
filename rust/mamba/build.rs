extern crate bindgen;

use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo:include=/Users/jimmyhmiller/Documents/Code/open-source/cpython/Include/");


    // Tell cargo to invalidate the built crate whenever the wrapper changes
    println!("cargo:rerun-if-changed=cpython/wrapper.h");

    // The bindgen::Builder is the main entry point
    // to bindgen, and lets you build up options for
    // the resulting bindings.
    let bindings = bindgen::Builder::default()
        // The input header we would like to generate
        // bindings for.
        .header("cpython/wrapper.h")
        // Tell cargo to invalidate the built crate whenever any of the
        // included header files changed.
        .parse_callbacks(Box::new(bindgen::CargoCallbacks))
        .clang_arg("-I/Users/jimmyhmiller/Documents/Code/open-source/cpython/Include/")
        .clang_arg("-I/Users/jimmyhmiller/Documents/Code/open-source/cpython/")
        .clang_arg("-I/Users/jimmyhmiller/Documents/Code/open-source/cpython/Modules/_decimal/libmpdec/")
        .layout_tests(false)
        // Finish the builder and generate the bindings.
        .generate()
        // Unwrap the Result and panic on failure.
        .expect("Unable to generate bindings");

    // Write the bindings to the $OUT_DIR/bindings.rs file.
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}
