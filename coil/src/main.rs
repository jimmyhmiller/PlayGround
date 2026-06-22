//! CLI — AOT only (no JIT, no `eval`).
//!
//!   coil build <file> [-o out]   compile + link a native executable (default: ./<stem>)
//!   coil run   <file>            build to a temp executable and run it (exit code = result)
//!   coil emit-obj <file> [-o o]  emit a native object file (default: ./<stem>.o)
//!   coil emit-ir  <file>         print the generated LLVM IR
//!   coil expand   <file>         print the program after macro expansion
//!   coil cimport  <header.h>     generate Coil FFI bindings from a C header (via clang)
//!
//! `build`/`run` accept `--link-flag <arg>` (repeatable) or `-l<lib>` to pass arguments
//! to the `cc` link line — link C libraries / objects (C-interop §6).
//!
//! Any command accepts `--target <triple>` to cross-compile for a non-host
//! target (e.g. `--target x86_64-apple-macosx11.0.0` to produce the System V
//! AMD64 ABI from an arm64 host). The same triple drives both the IR/ABI
//! lowering and the linker's `-arch`.

use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};

use inkwell::targets::TargetTriple;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        return usage();
    }
    let cmd = args[1].as_str();
    let file = &args[2];
    let opts = match Opts::parse(&args[3..]) {
        Ok(o) => o,
        Err(e) => {
            eprintln!("error: {e}");
            return usage();
        }
    };
    let triple = opts.target.as_deref().map(TargetTriple::create);

    let src = match std::fs::read_to_string(file) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error reading {file}: {e}");
            return ExitCode::FAILURE;
        }
    };

    match cmd {
        "build" => {
            let out = opts.out.unwrap_or_else(|| default_out(file, ""));
            let t = triple.unwrap_or_else(coil::codegen::target_triple);
            let src_path = opts.debug.then(|| Path::new(file));
            let r = coil::build_executable_linked_dbg(&src, &out, t, &opts.link_flags, src_path);
            report(r.map(|_| format!("wrote {}", out.display())), file)
        }
        "emit-obj" => {
            let out = opts.out.unwrap_or_else(|| default_out(file, "o"));
            let r = match triple {
                Some(t) => coil::compile_to_object_for(&src, &out, t),
                None => coil::compile_to_object(&src, &out),
            };
            report(r.map(|_| format!("wrote {}", out.display())), file)
        }
        "emit-ir" => report(
            match &opts.target {
                Some(t) => coil::emit_ir_for(&src, t),
                None => coil::emit_ir(&src),
            },
            file,
        ),
        "expand" => report(coil::expand_to_string(&src), file),
        "run" => run_aot(&src, file, opts.target.as_deref(), &opts.link_flags, opts.debug),
        // cimport <header.h> [-o out.coil]: generate Coil FFI bindings from a C header
        // via clang's AST. (`file` is the header path; `src` above just read it.)
        "cimport" => {
            let out = opts.out.unwrap_or_else(|| default_out(file, "coil"));
            let r = coil::cimport::cimport_header(file).and_then(|b| {
                std::fs::write(&out, b).map_err(|e| format!("writing {}: {e}", out.display()))?;
                Ok(format!("wrote {}", out.display()))
            });
            report(r, file)
        }
        _ => usage(),
    }
}

/// Print a finished diagnostic body under a single `error:` prefix, substituting
/// the real source path for the `<source>` placeholder the library renders (it
/// has the text but not the file name). A spanless message has no placeholder,
/// so the substitution is a harmless no-op there.
fn print_error(body: &str, file: &str) {
    eprintln!("error: {}", body.replacen("<source>", file, 1));
}

/// Build to a temp executable, run it, and propagate its exit code. With a cross
/// `--target` the binary is run via `arch -<arch>` (Rosetta on macOS).
fn run_aot(src: &str, file: &str, triple: Option<&str>, link_flags: &[String], debug: bool) -> ExitCode {
    let exe = std::env::temp_dir().join(format!("coil_run_{}", std::process::id()));
    let t = triple.map(TargetTriple::create).unwrap_or_else(coil::codegen::target_triple);
    let src_path = debug.then(|| Path::new(file));
    let build = coil::build_executable_linked_dbg(src, &exe, t, link_flags, src_path);
    if let Err(e) = build {
        print_error(&e, file);
        return ExitCode::FAILURE;
    }
    let status = match run_arch(triple) {
        Some(arch) => Command::new("arch").arg(format!("-{arch}")).arg(&exe).status(),
        None => Command::new(&exe).status(),
    };
    let _ = std::fs::remove_file(&exe);
    match status {
        Ok(s) => ExitCode::from(s.code().unwrap_or(1) as u8),
        Err(e) => {
            eprintln!("error running executable: {e}");
            ExitCode::FAILURE
        }
    }
}

/// The `arch -<a>` value needed to *run* an executable for `triple` on this host,
/// or `None` if it is the host arch and runs directly.
fn run_arch(triple: Option<&str>) -> Option<&'static str> {
    let t = triple?;
    let arch = t.split('-').next().unwrap_or("");
    let host = inkwell::targets::TargetMachine::get_default_triple()
        .as_str()
        .to_string_lossy()
        .split('-')
        .next()
        .unwrap_or("")
        .to_string();
    match arch {
        "x86_64" | "amd64" if host != "x86_64" => Some("x86_64"),
        "aarch64" | "arm64" | "arm64e" if host != "aarch64" && host != "arm64" => Some("arm64"),
        _ => None,
    }
}

/// Parsed CLI options shared by every command.
struct Opts {
    out: Option<PathBuf>,
    target: Option<String>,
    link_flags: Vec<String>,
    /// Emit DWARF debug info (`-g`) — function-level source mapping for lldb/gdb.
    debug: bool,
}

impl Opts {
    fn parse(rest: &[String]) -> Result<Opts, String> {
        let mut out = None;
        let mut target = None;
        let mut link_flags = Vec::new();
        let mut debug = false;
        let mut i = 0;
        while i < rest.len() {
            match rest[i].as_str() {
                "-o" => {
                    let path = rest.get(i + 1).ok_or("-o needs a path")?;
                    out = Some(PathBuf::from(path));
                    i += 2;
                }
                "--target" => {
                    let t = rest.get(i + 1).ok_or("--target needs a triple")?;
                    target = Some(t.clone());
                    i += 2;
                }
                // pass an argument through to the `cc` link line (e.g. `-lm`, a C object
                // path) — the C-interop §6 linking half.
                "--link-flag" => {
                    let f = rest.get(i + 1).ok_or("--link-flag needs an argument")?;
                    link_flags.push(f.clone());
                    i += 2;
                }
                "-g" | "--debug" => {
                    debug = true;
                    i += 1;
                }
                // shorthand: `-lfoo` is passed straight through.
                other if other.starts_with("-l") && other.len() > 2 => {
                    link_flags.push(other.to_string());
                    i += 1;
                }
                other => return Err(format!("unknown argument '{other}'")),
            }
        }
        Ok(Opts { out, target, link_flags, debug })
    }
}

fn report(result: Result<String, String>, file: &str) -> ExitCode {
    match result {
        Ok(out) => {
            println!("{out}");
            ExitCode::SUCCESS
        }
        Err(e) => {
            print_error(&e, file);
            ExitCode::FAILURE
        }
    }
}

fn default_out(file: &str, ext: &str) -> PathBuf {
    let stem = Path::new(file)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("a");
    PathBuf::from(if ext.is_empty() {
        stem.to_string()
    } else {
        format!("{stem}.{ext}")
    })
}

fn usage() -> ExitCode {
    eprintln!(
        "usage:\n  \
         coil <build|run|emit-obj|emit-ir|expand> <file.coil> [-o out] [--target <triple>] \
         [--link-flag <arg> | -l<lib>]… [-g]\n  \
         coil cimport <header.h> [-o out.coil]   generate C FFI bindings via clang\n  \
         -g / --debug   emit DWARF debug info (build/run) for lldb/gdb"
    );
    ExitCode::from(2)
}
