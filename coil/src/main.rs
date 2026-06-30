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
    let all: Vec<String> = std::env::args().collect();
    // Split coil's own args from the program's args at `--` (cargo-style):
    //   coil run -- <prog args>        coil run file.coil -- <prog args>
    let (args, prog_args): (Vec<String>, Vec<String>) = match all.iter().position(|a| a == "--") {
        Some(i) => (all[..i].to_vec(), all[i + 1..].to_vec()),
        None => (all.clone(), Vec::new()),
    };
    // Manifest mode: `coil build` / `coil run` with no source file (or only flags)
    // reads ./Coil.toml and drives the build from it.
    if matches!(args.get(1).map(String::as_str), Some("build") | Some("run"))
        && args.get(2).is_none_or(|a| a.starts_with('-'))
    {
        return run_manifest(args[1].as_str(), &prog_args);
    }
    // `coil new <name>` — scaffold a manifest project.
    if args.get(1).map(String::as_str) == Some("new") {
        return match args.get(2) {
            Some(name) => cmd_new(name),
            None => usage(),
        };
    }
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
            let t = triple.unwrap_or_else(coil::codegen::target_triple);
            let src_path = opts.debug.then(|| Path::new(file));
            // Optionally also write a C header for the exports (any build mode).
            if let Some(hpath) = &opts.emit_header {
                let r = coil::emit_header(&src).and_then(|h| {
                    std::fs::write(hpath, h).map_err(|e| format!("writing {}: {e}", hpath.display()))
                });
                if let Err(e) = r {
                    print_error(&e, file);
                    return ExitCode::FAILURE;
                }
                println!("wrote {}", hpath.display());
            }
            let stem = Path::new(file).file_stem().and_then(|s| s.to_str()).unwrap_or("a").to_string();
            let r = if opts.lib {
                let out = opts.out.unwrap_or_else(|| PathBuf::from(format!("lib{stem}.a")));
                coil::build_static_lib(&src, &out, t, src_path).map(|_| format!("wrote {}", out.display()))
            } else if opts.shared {
                let apple = t.as_str().to_string_lossy().contains("apple")
                    || t.as_str().to_string_lossy().contains("darwin")
                    || t.as_str().to_string_lossy().contains("macos");
                let ext = if apple { "dylib" } else { "so" };
                let out = opts.out.unwrap_or_else(|| PathBuf::from(format!("lib{stem}.{ext}")));
                coil::build_shared_lib(&src, &out, t, &opts.link_flags, src_path)
                    .map(|_| format!("wrote {}", out.display()))
            } else {
                let out = opts.out.unwrap_or_else(|| default_out(file, ""));
                coil::build_executable_linked_dbg(&src, &out, t, &opts.link_flags, src_path)
                    .map(|_| format!("wrote {}", out.display()))
            };
            report(r, file)
        }
        // cheader <file.coil> [-o out.h]: generate a C header for the file's
        // `(export-c …)` set (the inverse of `cimport`).
        "cheader" => {
            let out = opts.out.unwrap_or_else(|| default_out(file, "h"));
            let r = coil::emit_header(&src).and_then(|h| {
                std::fs::write(&out, h).map_err(|e| format!("writing {}: {e}", out.display()))?;
                Ok(format!("wrote {}", out.display()))
            });
            report(r, file)
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
        "dump-read" => report(coil::dump_read(&src), file),
        "run" => run_aot(&src, file, opts.target.as_deref(), &opts.link_flags, opts.debug, &prog_args),
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
    eprintln!("error: {}", body.replace("<source>", file));
}

/// Build to a temp executable, run it, and propagate its exit code. With a cross
/// `--target` the binary is run via `arch -<arch>` (Rosetta on macOS).
fn run_aot(src: &str, file: &str, triple: Option<&str>, link_flags: &[String], debug: bool, prog_args: &[String]) -> ExitCode {
    let exe = std::env::temp_dir().join(format!("coil_run_{}", std::process::id()));
    let t = triple.map(TargetTriple::create).unwrap_or_else(coil::codegen::target_triple);
    let src_path = debug.then(|| Path::new(file));
    let build = coil::build_executable_linked_dbg(src, &exe, t, link_flags, src_path);
    if let Err(e) = build {
        print_error(&e, file);
        return ExitCode::FAILURE;
    }
    let status = match run_arch(triple) {
        Some(arch) => Command::new("arch").arg(format!("-{arch}")).arg(&exe).args(prog_args).status(),
        None => Command::new(&exe).args(prog_args).status(),
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

/// `coil build` / `coil run` driven by `./Coil.toml` instead of CLI arguments.
fn run_manifest(cmd: &str, prog_args: &[String]) -> ExitCode {
    let path = Path::new(coil::manifest::MANIFEST);
    if !path.exists() {
        eprintln!(
            "error: no {} in the current directory (pass a <file> to build directly)",
            coil::manifest::MANIFEST
        );
        return ExitCode::FAILURE;
    }
    let m = match coil::manifest::Manifest::load(path) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::FAILURE;
        }
    };
    let entry = m.entry();
    let src = match std::fs::read_to_string(&entry) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error reading entry {entry}: {e}");
            return ExitCode::FAILURE;
        }
    };
    let link_flags = m.link_flags();
    let triple = m.build.target.as_deref().map(TargetTriple::create);
    match cmd {
        "build" => {
            let out = PathBuf::from(m.out());
            let t = triple.unwrap_or_else(coil::codegen::target_triple);
            let src_path = m.build.debug.then(|| Path::new(&entry));
            let r = coil::build_executable_linked_dbg(&src, &out, t, &link_flags, src_path);
            report(r.map(|_| format!("wrote {}", out.display())), &entry)
        }
        "run" => {
            // Program args: those after `--`, else the manifest's [run] args default.
            let args = if prog_args.is_empty() { m.run.args.clone() } else { prog_args.to_vec() };
            run_aot(&src, &entry, m.build.target.as_deref(), &link_flags, m.build.debug, &args)
        }
        _ => usage(),
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
    /// `--lib`: build a static archive (`.a`) instead of an executable.
    lib: bool,
    /// `--shared`: build a shared library (`.dylib`/`.so`).
    shared: bool,
    /// `--emit-header <path>`: also write a C header for the `(export-c …)` set.
    emit_header: Option<PathBuf>,
}

impl Opts {
    fn parse(rest: &[String]) -> Result<Opts, String> {
        let mut out = None;
        let mut target = None;
        let mut link_flags = Vec::new();
        let mut debug = false;
        let mut lib = false;
        let mut shared = false;
        let mut emit_header = None;
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
                "--lib" => {
                    lib = true;
                    i += 1;
                }
                "--shared" => {
                    shared = true;
                    i += 1;
                }
                "--emit-header" => {
                    let p = rest.get(i + 1).ok_or("--emit-header needs a path")?;
                    emit_header = Some(PathBuf::from(p));
                    i += 2;
                }
                // shorthand: `-lfoo` is passed straight through.
                other if other.starts_with("-l") && other.len() > 2 => {
                    link_flags.push(other.to_string());
                    i += 1;
                }
                other => return Err(format!("unknown argument '{other}'")),
            }
        }
        if lib && shared {
            return Err("--lib and --shared are mutually exclusive".into());
        }
        Ok(Opts { out, target, link_flags, debug, lib, shared, emit_header })
    }
}

/// `coil new <name>` — scaffold `<name>/Coil.toml` + `<name>/src/main.coil`, ready
/// to `cd <name> && coil run`.
fn cmd_new(name: &str) -> ExitCode {
    let dir = Path::new(name);
    if dir.exists() {
        eprintln!("error: '{name}' already exists");
        return ExitCode::FAILURE;
    }
    let manifest = format!("[package]\nname  = \"{name}\"\nentry = \"src/main.coil\"\n");
    let main = "(module app)\n\n(defn main [] (-> i64)\n  42)\n";
    let r = std::fs::create_dir_all(dir.join("src"))
        .and_then(|_| std::fs::write(dir.join(coil::manifest::MANIFEST), manifest))
        .and_then(|_| std::fs::write(dir.join("src/main.coil"), main));
    match r {
        Ok(()) => {
            println!("created {name}/ — `cd {name} && coil run`");
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("error scaffolding '{name}': {e}");
            ExitCode::FAILURE
        }
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
         coil new <name>                          scaffold a project (Coil.toml + src/main.coil)\n  \
         coil <build|run>                         build/run the ./Coil.toml project\n  \
         coil run -- <args…>                      … forwarding args to the program (else [run] args)\n  \
         coil <build|run|emit-obj|emit-ir|expand> <file.coil> [-o out] [--target <triple>] \
         [--link-flag <arg> | -l<lib>]… [-g] [-- <args…>]\n  \
         coil build <file.coil> --lib|--shared    build a C library (.a / .dylib) from (export-c …)\n  \
         coil build <file.coil> --emit-header <h>  also write a C header for the exports\n  \
         coil cheader <file.coil> [-o out.h]      generate the C header for (export-c …) (inverse of cimport)\n  \
         coil cimport <header.h> [-o out.coil]    generate C FFI bindings via clang\n  \
         -g / --debug   emit DWARF debug info (build/run) for lldb/gdb"
    );
    ExitCode::from(2)
}
