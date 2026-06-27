//! The CHIP-8 app (apps/chip8) compiles: both front-ends, the vendored VM core
//! (chip8.coil), and the vendored macOS objc glue (objc.coil). Driven through the
//! real `coil` binary with CWD set to the app dir, so its app-local imports
//! `(import "chip8.coil")` / `(import "objc.coil")` resolve. `emit-ir` compiles the
//! whole front end + codegen without linking, so no frameworks are needed here.

use std::process::Command;

#[test]
fn chip8_app_compiles() {
    let coil = env!("CARGO_BIN_EXE_coil");
    for entry in ["main.coil", "terminal.coil"] {
        let out = Command::new(coil)
            .current_dir("apps/chip8")
            .args(["emit-ir", entry])
            .output()
            .expect("run coil");
        assert!(
            out.status.success(),
            "apps/chip8/{entry} failed to compile:\n{}",
            String::from_utf8_lossy(&out.stderr)
        );
    }
}
