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
        .args(["--gc-sections", "-T", "freestanding/virt.ld", "-e", "bare.start"])
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

#[test]
fn importing_stdlib_does_not_drag_libc_with_gc_sections() {
    // The DCE friction the dogfood surfaced: Coil emits all defns, so importing
    // lib/alloc pulled malloc/abort/free/realloc into the link even unused. Fix =
    // per-function sections (compiler) + ld.lld --gc-sections (recipe): the unused
    // stdlib functions are GC'd, so a freestanding program can IMPORT the stdlib
    // without dragging libc. This locks that in (and documents the regression: link
    // WITHOUT --gc-sections fails on `malloc`, WITH it is clean).
    let lld = match find(
        &["ld.lld"],
        &["/opt/homebrew/opt/llvm@18/bin", "/opt/homebrew/opt/llvm@17/bin"],
    ) {
        Some(l) => l,
        None => {
            eprintln!("SKIP: ld.lld not found");
            return;
        }
    };

    // a freestanding program that IMPORTS lib/alloc but uses nothing from it.
    let src = "(module bare)\n\
        (import \"lib/alloc.coil\" :use *)\n\
        (defn uart-byte [(c i64)] (-> i64)\n\
          (llvm-ir i64 [c] \"%p = inttoptr i64 150994944 to ptr\\n\
           %b = trunc i64 $0 to i8\\nstore volatile i8 %b, ptr %p\\nret i64 0\"))\n\
        (defn start [] (-> i64) (uart-byte 75) (loop 0))";
    let obj = std::env::temp_dir().join("coil-dce.o");
    let triple = inkwell::targets::TargetTriple::create("aarch64-unknown-none");
    coil::compile_to_object_for(src, &obj, triple).expect("emit object");

    let link = |gc: bool| {
        let elf = std::env::temp_dir().join(if gc { "coil-dce-gc.elf" } else { "coil-dce-nogc.elf" });
        let mut c = Command::new(&lld);
        if gc {
            c.arg("--gc-sections");
        }
        c.args(["-T", "freestanding/virt.ld", "-e", "bare.start"])
            .arg(&obj)
            .arg("-o")
            .arg(&elf)
            .output()
            .expect("invoke ld.lld")
    };

    // WITHOUT --gc-sections: the unused stdlib drags malloc → link fails.
    assert!(
        !link(false).status.success(),
        "expected the link WITHOUT --gc-sections to fail on a dragged libc symbol"
    );
    // WITH --gc-sections: the unused stdlib is GC'd → clean link.
    assert!(
        link(true).status.success(),
        "expected the link WITH --gc-sections to succeed (stdlib GC'd)"
    );
}
