//! Freestanding bare-metal dogfood (§7): Coil running with NO runtime — no libc, no
//! OS — on aarch64 under qemu-system. The whole pipeline: compile a bare-metal object
//! (`+strict-align` for the MMU-off Device-memory target), assemble the crt0 boot stub
//! (set SP, enable FP/SIMD, zero .bss, call the entry), link with ld.lld (no libc) +
//! `--gc-sections` (so importing the stdlib drops its unused libc calls), and run under
//! qemu-system-aarch64. The strongest "as low as Zig/C" test — Coil runs where C/Zig do.
//!
//! GATED on the bare-metal toolchain (ld.lld + clang-as-assembler + qemu-system-aarch64);
//! the tests SKIP (don't fail) when it's absent, so CI without it still passes.

use std::path::{Path, PathBuf};
use std::process::Command;

const LLVM_DIRS: &[&str] = &["/opt/homebrew/opt/llvm@18/bin", "/opt/homebrew/opt/llvm@17/bin"];

fn find(names: &[&str], extra_dirs: &[&str]) -> Option<PathBuf> {
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

/// (ld.lld, clang, qemu-system-aarch64) or None if the bare-metal toolchain is absent.
fn tools() -> Option<(PathBuf, PathBuf, PathBuf)> {
    Some((
        find(&["ld.lld"], LLVM_DIRS)?,
        find(&["clang", "cc"], LLVM_DIRS)?,
        find(&["qemu-system-aarch64"], &["/opt/homebrew/bin"])?,
    ))
}

/// Compile a `.coil` source string to a bare-metal aarch64 object.
fn emit_obj(src: &str, obj: &Path) {
    let triple = inkwell::targets::TargetTriple::create("aarch64-unknown-none");
    coil::compile_to_object_for(src, obj, triple).expect("emit bare-metal object");
}

/// Assemble the crt0 stub + link it with the Coil object. Returns the linker's success.
fn link(clang: &Path, lld: &Path, coil_obj: &Path, elf: &Path, gc: bool) -> bool {
    let boot = std::env::temp_dir().join("coil-boot.o");
    assert!(
        Command::new(clang)
            .args(["-target", "aarch64-unknown-none", "-c", "freestanding/start.s", "-o"])
            .arg(&boot)
            .status()
            .expect("assemble start.s")
            .success(),
        "crt0 assembly failed"
    );
    let mut c = Command::new(lld);
    if gc {
        c.arg("--gc-sections");
    }
    c.args(["-T", "freestanding/virt.ld"])
        .arg(&boot)
        .arg(coil_obj)
        .arg("-o")
        .arg(elf)
        .status()
        .expect("invoke ld.lld")
        .success()
}

fn no_undefined(elf: &Path) -> bool {
    let nm = Command::new("nm").arg("-u").arg(elf).output().expect("nm");
    String::from_utf8_lossy(&nm.stdout).trim().is_empty()
}

fn run_qemu(qemu: &Path, elf: &Path) -> String {
    let out = Command::new(qemu)
        .args(["-M", "virt", "-cpu", "cortex-a72", "-nographic", "-kernel"])
        .arg(elf)
        .output()
        .expect("run qemu");
    String::from_utf8_lossy(&out.stdout).into_owned()
}

/// Build `freestanding/<name>.coil`, link it freestanding, run it, return qemu's stdout
/// — and assert the image has zero undefined symbols (the no-runtime moat).
fn build_run(name: &str, want: &str) {
    let (lld, clang, qemu) = match tools() {
        Some(t) => t,
        None => {
            eprintln!("SKIP: bare-metal toolchain (ld.lld + clang + qemu-system-aarch64) not found");
            return;
        }
    };
    let src = std::fs::read_to_string(format!("freestanding/{name}.coil")).expect("read .coil");
    let obj = std::env::temp_dir().join(format!("coil-{name}.o"));
    let elf = std::env::temp_dir().join(format!("coil-{name}.elf"));
    emit_obj(&src, &obj);
    assert!(link(&clang, &lld, &obj, &elf, true), "ld.lld failed for {name}");
    assert!(no_undefined(&elf), "{name}: freestanding image has undefined symbols (libc leak)");
    let out = run_qemu(&qemu, &elf);
    assert!(out.contains(want), "{name}: unexpected UART output: {out:?}");
}

#[test]
fn bare_metal_hello_runs_under_qemu() {
    build_run("hello", "from coil (bare metal)");
}

#[test]
fn bare_metal_typed_register_uart_driver_runs_under_qemu() {
    // The :bits capstone: device registers as typed bitfields (defmmio-reg pure macro);
    // a polled PL011 driver (poll TXFF, then write the data register).
    build_run("uart", "PL011 via typed registers");
}

#[test]
fn bare_metal_stdlib_arena_runs_under_qemu() {
    // The cardinal: a program that IMPORTS lib/alloc and RUNS its arena (arena-over-
    // buffer + create) genuinely executes bare-metal — not just links. Proves the
    // stdlib's capability-as-value design works with NO runtime (crt0 sets SP +
    // FP/SIMD + zeroes .bss; +strict-align suits the MMU-off Device-memory target).
    build_run("arena", "stdlib arena on bare metal: 42");
}

#[test]
fn importing_stdlib_does_not_drag_libc_with_gc_sections() {
    // The DCE friction the dogfood surfaced: Coil emits all defns, so importing
    // lib/alloc pulled malloc/abort/free/realloc into the link even unused. Fix =
    // per-function sections (compiler) + ld.lld --gc-sections: unused stdlib functions
    // are GC'd. Documents BOTH directions (without --gc-sections the link fails on a
    // dragged libc symbol; with it, clean).
    let (lld, clang, _qemu) = match tools() {
        Some(t) => t,
        None => {
            eprintln!("SKIP: bare-metal toolchain not found");
            return;
        }
    };
    let src = "(module bare)\n\
        (import \"lib/alloc.coil\" :use *)\n\
        (defn uart-byte [(c i64)] (-> i64)\n\
          (llvm-ir i64 [c] \"%p = inttoptr i64 150994944 to ptr\\n\
           %b = trunc i64 $0 to i8\\nstore volatile i8 %b, ptr %p\\nret i64 0\"))\n\
        (defn start [] (-> i64) (uart-byte 75) (loop 0))";
    let obj = std::env::temp_dir().join("coil-dce.o");
    emit_obj(src, &obj);
    let elf = std::env::temp_dir().join("coil-dce.elf");
    assert!(!link(&clang, &lld, &obj, &elf, false), "expected the no-gc link to fail on a dragged libc symbol");
    assert!(link(&clang, &lld, &obj, &elf, true), "expected --gc-sections to drop the unused stdlib (clean link)");
}
