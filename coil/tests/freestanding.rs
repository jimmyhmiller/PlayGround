//! Freestanding bare-metal dogfood (§7): Coil running with NO runtime — no libc, no
//! OS, no crt0 — on aarch64 under qemu-system. Builds freestanding/hello.coil for a
//! bare-metal target, links it with ld.lld (no libc) + a linker script, and runs it
//! under qemu-system-aarch64, checking the UART output. The strongest "as low as
//! Zig/C" test: Coil runs bare-metal, exactly where C/Zig run.
//!
//! GATED on `ld.lld` + `qemu-system-aarch64` being present (they are the bare-metal
//! toolchain); the test SKIPS (does not fail) when they're absent, so CI on a machine
//! without the cross toolchain still passes.

use std::path::{Path, PathBuf};
use std::process::Command;

fn find(names: &[&str], extra_dirs: &[&str]) -> Option<PathBuf> {
    // PATH first, then known Homebrew LLVM/qemu locations.
    for n in names {
        if let Ok(o) = Command::new("sh").arg("-c").arg(format!("command -v {n}")).output() {
            if o.status.success() {
                let p = String::from_utf8_lossy(&o.stdout).trim().to_string();
                if !p.is_empty() {
                    return Some(PathBuf::from(p));
                }
            }
        }
        for d in extra_dirs {
            let p = Path::new(d).join(n);
            if p.exists() {
                return Some(p);
            }
        }
    }
    None
}

#[test]
fn bare_metal_aarch64_runs_under_qemu() {
    let lld = find(
        &["ld.lld"],
        &["/opt/homebrew/opt/llvm@18/bin", "/opt/homebrew/opt/llvm@17/bin"],
    );
    let qemu = find(&["qemu-system-aarch64"], &["/opt/homebrew/bin"]);
    let (lld, qemu) = match (lld, qemu) {
        (Some(l), Some(q)) => (l, q),
        _ => {
            eprintln!("SKIP: ld.lld and/or qemu-system-aarch64 not found (bare-metal toolchain)");
            return;
        }
    };

    let src = std::fs::read_to_string("freestanding/hello.coil").expect("read hello.coil");
    let obj = std::env::temp_dir().join("coil-bare-test.o");
    let elf = std::env::temp_dir().join("coil-bare-test.elf");

    // 1. compiler: emit a bare-metal aarch64 object (no link).
    let triple = inkwell::targets::TargetTriple::create("aarch64-unknown-none");
    coil::compile_to_object_for(&src, &obj, triple).expect("emit bare-metal object");

    // 2. link freestanding with ld.lld: no crt0, no libc, our linker script, entry =
    //    the Coil `start` function (`bare.start`).
    let link = Command::new(&lld)
        .args(["-T", "freestanding/virt.ld", "-e", "bare.start"])
        .arg(&obj)
        .arg("-o")
        .arg(&elf)
        .status()
        .expect("invoke ld.lld");
    assert!(link.success(), "ld.lld failed");

    // a freestanding image must have NO undefined symbols (no libc dependency).
    let nm = Command::new("nm").arg("-u").arg(&elf).output().expect("nm");
    let undef = String::from_utf8_lossy(&nm.stdout);
    assert!(
        undef.trim().is_empty(),
        "freestanding image has undefined symbols (libc leak):\n{undef}"
    );

    // 3. run under full-system qemu (the ELF is the kernel). The program PSCI-poweroffs.
    let out = Command::new(&qemu)
        .args(["-M", "virt", "-cpu", "cortex-a72", "-nographic", "-kernel"])
        .arg(&elf)
        .output()
        .expect("run qemu");
    let text = String::from_utf8_lossy(&out.stdout);
    assert!(
        text.contains("hi") && text.contains("from coil (bare metal)"),
        "unexpected UART output: {text:?}"
    );
}
