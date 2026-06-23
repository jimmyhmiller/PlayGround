//! Coil — a low-level Lisp where calling convention (and, later, allocation) is
//! part of the type system. This crate is the M0/M1 skeleton: reader → core →
//! convention-aware checks → LLVM codegen that JIT-runs `main`.
//!
//! See `docs/DESIGN.md` for the full design.

pub mod abi;
pub mod ast;
pub mod check;
pub mod cimport;
pub mod codegen;
pub mod convention;
pub mod macros;
pub mod mono;
pub mod parse;
pub mod reader;
pub mod resolve;
pub mod span;

use inkwell::context::Context;
use inkwell::module::Module;
use inkwell::passes::PassBuilderOptions;
use inkwell::targets::{
    CodeModel, FileType, InitializationConfig, RelocMode, Target, TargetMachine,
};
use inkwell::OptimizationLevel;

/// The LLVM new-pass-manager pipeline Coil runs over every module before it
/// emits an object. `emit_ir` deliberately skips this (it shows the raw,
/// readable IR we generate, which the struct-ABI tests diff against clang's
/// unoptimized output); the *compiled* output is fully optimized.
const OPT_PIPELINE: &str = "default<O3>";
use std::path::Path;
use std::process::Command;

use span::Diag;

/// Render any pipeline error against `src` into a finished diagnostic string
/// (`file:line:col` + caret when the error carries a span; a bare `error: msg`
/// otherwise). Accepts both `String` errors (from passes that don't yet carry
/// spans) and `Diag` errors (reader/parser), so every public entry point reports
/// uniformly. The CLI substitutes the real path for the `<source>` placeholder.
fn reported<T, E: Into<Diag>>(r: Result<T, E>, src: &str) -> Result<T, String> {
    r.map_err(|e| span::render(&e.into(), src, "<source>"))
}

/// Read + macro-expand + load the module graph, then resolve names (the qualify
/// pass) into one whole-program `Program`.
fn read_expand_resolve(src: &str) -> Result<ast::Program, Diag> {
    let forms = reader::read_all(src)?;
    let (tagged, imports, exports) = macros::expand_program(&forms, &host_target())?;
    resolve::resolve_program(tagged, &imports, &exports)
}

/// The compile-time target description handed to the macro evaluator. Derived
/// from the host triple (Coil currently AOT-targets the host).
fn host_target() -> macros::TargetInfo {
    let triple = TargetMachine::get_default_triple()
        .as_str()
        .to_string_lossy()
        .into_owned();
    let mut parts = triple.split('-');
    let arch = parts.next().unwrap_or("unknown").to_string();
    let _vendor = parts.next();
    let os = parts.next().unwrap_or("unknown").to_string();
    let pointer_width = match arch.as_str() {
        "i386" | "i686" | "arm" | "armv7" | "thumbv7" | "wasm32" | "riscv32" | "mips" | "mipsel" => {
            32
        }
        _ => 64,
    };
    macros::TargetInfo {
        arch,
        os,
        triple,
        pointer_width,
    }
}

/// The shared front end: read → expand → parse → check (elaborate + infer) →
/// monomorphize → module. The checker runs *before* monomorphization now: it
/// types polymorphic code and infers/fills the generic type arguments, so the
/// monomorphizer is a pure specializer over fully-explicit type args.
fn build_module<'ctx>(ctx: &'ctx Context, src: &str) -> Result<Module<'ctx>, Diag> {
    build_module_for(ctx, src, codegen::target_triple())
}

/// `build_module` for an explicitly chosen target triple (cross-targeting).
fn build_module_for<'ctx>(
    ctx: &'ctx Context,
    src: &str,
    triple: inkwell::targets::TargetTriple,
) -> Result<Module<'ctx>, Diag> {
    build_module_dbg(ctx, src, triple, None)
}

/// `build_module_for`, optionally emitting DWARF debug info (`-g`).
fn build_module_dbg<'ctx>(
    ctx: &'ctx Context,
    src: &str,
    triple: inkwell::targets::TargetTriple,
    dbg: Option<codegen::DebugInput>,
) -> Result<Module<'ctx>, Diag> {
    let program = read_expand_resolve(src)?;
    let program = check::check(&program)?;
    let program = mono::monomorphize(program)?;
    Ok(codegen::compile_for_dbg(ctx, &program, triple, dbg)?)
}

/// Build the `DebugInput` (source text + file identity for the `DIFile`) from a
/// source path. `<source>` stands in when there is no real file (the library/test
/// entry points compile from a string).
fn debug_input<'a>(src: &'a str, src_path: Option<&Path>) -> codegen::DebugInput<'a> {
    let (file_name, directory) = match src_path {
        Some(p) => {
            let dir = p
                .parent()
                .filter(|d| !d.as_os_str().is_empty())
                .map(|d| d.to_string_lossy().into_owned())
                .unwrap_or_else(|| ".".to_string());
            let name = p.file_name().map_or_else(
                || p.to_string_lossy().into_owned(),
                |n| n.to_string_lossy().into_owned(),
            );
            (name, dir)
        }
        None => ("<source>".to_string(), ".".to_string()),
    };
    codegen::DebugInput { source: src, file_name, directory }
}

/// Macro-expand and pretty-print the resulting forms (for `--expand`). Shows the
/// post-expansion forms (before name resolution).
pub fn expand_to_string(src: &str) -> Result<String, String> {
    let forms = reported(reader::read_all(src), src)?;
    let (tagged, _, _) = reported(macros::expand_program(&forms, &host_target()), src)?;
    Ok(tagged
        .iter()
        .map(|(f, _)| f.to_string())
        .collect::<Vec<_>>()
        .join("\n"))
}

/// Parse + check + emit textual LLVM IR (no JIT). Useful in tests and for
/// inspecting how conventions lower.
pub fn emit_ir(src: &str) -> Result<String, String> {
    let ctx = Context::create();
    let module = reported(build_module(&ctx, src), src)?;
    Ok(module.print_to_string().to_string())
}

/// `emit_ir` for an explicitly chosen target triple — used to inspect a
/// program's ABI lowering for a non-host target (e.g. verifying the x86-64 SysV
/// struct coercion from an arm64 host).
pub fn emit_ir_for(src: &str, triple: &str) -> Result<String, String> {
    let ctx = Context::create();
    let module = reported(
        build_module_for(&ctx, src, inkwell::targets::TargetTriple::create(triple)),
        src,
    )?;
    Ok(module.print_to_string().to_string())
}

/// AOT: compile to a native object file. This is the language's primary output
/// — no runtime dependency on LLVM, links with a real linker, and the `:shim`
/// trampolines become ordinary relocations the system toolchain resolves.
pub fn compile_to_object(src: &str, obj_path: &Path) -> Result<(), String> {
    compile_to_object_for(src, obj_path, codegen::target_triple())
}

/// Place each DEFINED function in its own `.text.<name>` section
/// (`-ffunction-sections`), so a linker invoked with `--gc-sections` (the
/// freestanding recipe) can garbage-collect UNREFERENCED functions. Without this,
/// importing the stdlib drags every defn's libc calls (`malloc`/`abort`/…) into the
/// link even when unused — a freestanding program would link libc just for an
/// `import`. Done in the OBJECT path only (not `emit_ir`), so the IR text the tests
/// diff is unchanged; harmless for normal `cc` links (no `--gc-sections` → every
/// section kept), so it needs no flag — a generic mechanism, not a "freestanding mode".
/// Whether `triple` uses the ELF object format (so `.text.<name>` per-function
/// sections are valid). Mach-O (apple/darwin) and COFF (windows/msvc) have a
/// different section grammar and dead-strip differently, so they're excluded — and
/// emitting an ELF section name on Mach-O is a hard LLVM error.
fn target_uses_elf(triple: &str) -> bool {
    !(triple.contains("apple")
        || triple.contains("darwin")
        || triple.contains("macho")
        || triple.contains("windows")
        || triple.contains("msvc")
        || triple.contains("wasm"))
}

fn set_function_sections(module: &inkwell::module::Module) {
    use inkwell::values::AsValueRef;
    for f in module.get_functions() {
        if f.count_basic_blocks() == 0 {
            continue; // a declaration (extern) — no body to place in a section
        }
        if let Ok(name) = f.get_name().to_str() {
            if let Ok(sec) = std::ffi::CString::new(format!(".text.{name}")) {
                unsafe { inkwell::llvm_sys::core::LLVMSetSection(f.as_value_ref(), sec.as_ptr()) };
            }
        }
    }
}

/// `compile_to_object` for an explicitly chosen target triple. Used to
/// cross-compile (e.g. an x86-64 object on an arm64 host to exercise the SysV
/// struct ABI under Rosetta); the IR and the emitted machine code share `triple`.
pub fn compile_to_object_for(
    src: &str,
    obj_path: &Path,
    triple: inkwell::targets::TargetTriple,
) -> Result<(), String> {
    compile_to_object_dbg(src, obj_path, triple, None)
}

/// `compile_to_object_for`, optionally emitting DWARF (`src_path` = `Some` ⇒ `-g`,
/// using that path for the `DIFile`).
pub fn compile_to_object_dbg(
    src: &str,
    obj_path: &Path,
    triple: inkwell::targets::TargetTriple,
    src_path: Option<&Path>,
) -> Result<(), String> {
    Target::initialize_all(&InitializationConfig::default());
    let ctx = Context::create();
    let target = Target::from_triple(&triple).map_err(|e| e.to_string())?;
    let dbg = src_path.map(|_| debug_input(src, src_path));
    let module = reported(build_module_dbg(&ctx, src, triple, dbg), src)?;
    let triple = module.get_triple();
    // Bare-metal aarch64 (the freestanding `-none` target) runs with the MMU off, so
    // RAM is Device memory — UNALIGNED accesses fault. `+strict-align` stops the backend
    // emitting unaligned wide accesses (the optimizer otherwise uses an unaligned
    // 16-byte SIMD store for struct/array init, e.g. `stur q0,[x8,#8]`, which faults on
    // Device memory); aligned accesses (incl. an aligned vector copy) are fine. The
    // standard bare-metal approach (matches the Linux arm64 kernel). Hosted targets
    // (MMU on → Normal memory) don't need it and aren't changed.
    let triple_s = triple.as_str().to_string_lossy();
    let features = if triple_s.contains("aarch64") && triple_s.contains("none") {
        "+strict-align"
    } else {
        ""
    };
    // A `-g` build also needs the *backend* (instruction selection, scheduling,
    // store-merging) at -O0: otherwise it reorders/merges the debug-spill stores
    // away from the statement they belong to, so a local reads as stale at a line
    // breakpoint. A release build keeps Aggressive for `cc -O3` parity.
    let backend_opt = if src_path.is_some() {
        OptimizationLevel::None
    } else {
        OptimizationLevel::Aggressive
    };
    let tm = target
        .create_target_machine(
            &triple,
            "generic",
            features,
            backend_opt,
            RelocMode::PIC,
            CodeModel::Default,
        )
        .ok_or("could not create target machine")?;

    module.set_triple(&triple);
    module.set_data_layout(&tm.get_target_data().get_data_layout());
    // Per-function sections are an ELF concept (`.text.<name>`); Mach-O and COFF use a
    // different section grammar (and dead-strip by symbol atom, not by section), so
    // only emit them for ELF targets — the freestanding/bare-metal + Linux case where
    // `--gc-sections` is used. Emitting `.text.foo` on Mach-O is a hard LLVM error.
    if target_uses_elf(&triple.as_str().to_string_lossy()) {
        set_function_sections(&module);
    }
    // Run the full LLVM optimization pipeline (mem2reg, inlining, GVN, loop
    // opts, tail-call elimination, …) before lowering to machine code. Without
    // this the emitted object would be ~`-O0`: every `let`/field stays an
    // `alloca`, nothing inlines, and self-tail-recursion (Coil's only loop) is
    // never turned into a loop (it would overflow the stack).
    //
    // A `-g` (debug) build runs an almost-empty pipeline so the code stays
    // faithful to the source for line-by-line stepping and variable inspection:
    // `alwaysinline` (the `(llvm-ir …)` zero-overhead helpers MUST still inline)
    // and `tailcallelim` (Coil's only loop is self-tail-recursion — without TRE a
    // recursive program overflows the stack). Everything else is left OFF —
    // notably mem2reg/instcombine/GVN, which would fold statements away (e.g.
    // `(iadd 1 2)` → a constant) and leave nothing to step through; keeping the
    // `alloca`s also means locals live in memory where the debugger can read them.
    let pipeline = if src_path.is_some() {
        "function(tailcallelim),always-inline"
    } else {
        OPT_PIPELINE
    };
    module
        .run_passes(pipeline, &tm, PassBuilderOptions::create())
        .map_err(|e| format!("optimization passes failed: {e}"))?;
    tm.write_to_file(&module, FileType::Object, obj_path)
        .map_err(|e| e.to_string())
}

/// AOT: compile to an object and link a native executable with `cc`. The Coil
/// `main` (i64, no args) becomes the process entry; its return value is the
/// exit code (low 8 bits).
pub fn build_executable(src: &str, out_path: &Path) -> Result<(), String> {
    build_executable_for(src, out_path, codegen::target_triple())
}

/// `build_executable` for an explicitly chosen target triple (cross-compile).
/// The object is emitted for `triple` *and* the linker is told to produce an
/// executable for that triple's architecture (`cc -arch <arch>`), so the result
/// is a real binary for the chosen target (runnable under Rosetta on macOS for a
/// cross arch). The triple's arch must be one codegen supports.
pub fn build_executable_for(
    src: &str,
    out_path: &Path,
    triple: inkwell::targets::TargetTriple,
) -> Result<(), String> {
    build_executable_linked(src, out_path, triple, &[])
}

/// `build_executable_for` plus extra arguments passed through to the `cc` link line —
/// e.g. `-lm`, `-lfoo`, or a C object file path — so a Coil program can link against C
/// libraries / objects (the C-interop §6 linking half). A generic passthrough, NOT a
/// baked-in "C-lib mode": the compiler emits the object; how it's linked is the caller's.
pub fn build_executable_linked(
    src: &str,
    out_path: &Path,
    triple: inkwell::targets::TargetTriple,
    link_flags: &[String],
) -> Result<(), String> {
    build_executable_linked_dbg(src, out_path, triple, link_flags, None)
}

/// `build_executable_linked`, optionally emitting DWARF. `src_path = Some` turns
/// on debug info (`-g`) and names the `DIFile`. On macOS the DWARF stays in the
/// `.o` (the linker records a debug map that points at it), so a debug build
/// keeps the object file and runs `dsymutil` to gather a `.dSYM` next to the
/// executable; a release build deletes the `.o` as before.
pub fn build_executable_linked_dbg(
    src: &str,
    out_path: &Path,
    triple: inkwell::targets::TargetTriple,
    link_flags: &[String],
    src_path: Option<&Path>,
) -> Result<(), String> {
    let triple_str = triple.as_str().to_string_lossy().into_owned();
    let obj_path = out_path.with_extension("o");
    compile_to_object_dbg(src, &obj_path, triple, src_path)?;
    let mut cc = Command::new("cc");
    if let Some(arch) = link_arch_flag(&triple_str) {
        cc.arg("-arch").arg(arch);
    }
    cc.arg(&obj_path).arg("-o").arg(out_path);
    for f in link_flags {
        cc.arg(f);
    }
    let status = cc
        .status()
        .map_err(|e| format!("failed to invoke linker (cc): {e}"))?;
    if !status.success() {
        let _ = std::fs::remove_file(&obj_path);
        return Err(format!("linker (cc) failed with {status}"));
    }
    if src_path.is_some() {
        // macOS keeps DWARF in the `.o`; collect it into a `.dSYM` so the debugger
        // finds it even after the `.o` is gone. Only remove the `.o` if dsymutil
        // actually succeeded — otherwise keep it as the debug-map fallback (losing
        // both would silently strip all debug info).
        let dsym_ok = Command::new("dsymutil")
            .arg(out_path)
            .status()
            .map(|s| s.success())
            .unwrap_or(false);
        if dsym_ok {
            let _ = std::fs::remove_file(&obj_path);
        }
    } else {
        let _ = std::fs::remove_file(&obj_path);
    }
    Ok(())
}

/// The `cc -arch` value for a target triple's architecture, or `None` if it is
/// the host arch (no `-arch` needed). An unsupported arch is a hard error so a
/// `--target` typo doesn't silently link for the host.
fn link_arch_flag(triple: &str) -> Option<&'static str> {
    let arch = triple.split('-').next().unwrap_or("");
    let host = TargetMachine::get_default_triple()
        .as_str()
        .to_string_lossy()
        .split('-')
        .next()
        .unwrap_or("")
        .to_string();
    match arch {
        "x86_64" | "amd64" => (host != "x86_64").then_some("x86_64"),
        "aarch64" | "arm64" | "arm64e" => (host != "aarch64" && host != "arm64").then_some("arm64"),
        _ => None,
    }
}

/// Front end only: read → expand → parse → check. No codegen, no LLVM
/// execution — just diagnostics. (Coil has no `eval`/JIT: the only way to run a
/// program is to AOT-compile it.)
pub fn check_source(src: &str) -> Result<(), String> {
    let program = reported(read_expand_resolve(src), src)?;
    let program = reported(check::check(&program), src)?;
    // Run monomorphization too, so specialization-time errors (if any) surface
    // in a check-only pass as well.
    reported(mono::monomorphize(program), src)?;
    Ok(())
}
