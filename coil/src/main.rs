//! CLI — AOT only (no JIT, no `eval`).
//!
//!   coil build <file> [-o out]   compile + link a native executable (default: ./<stem>)
//!   coil run   <file>            build to a temp executable and run it (exit code = result)
//!   coil emit-obj <file> [-o o]  emit a native object file (default: ./<stem>.o)
//!   coil emit-ir  <file>         print the generated LLVM IR
//!   coil expand   <file>         print the program after macro expansion
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
            let r = match triple {
                Some(t) => coil::build_executable_for(&src, &out, t),
                None => coil::build_executable(&src, &out),
            };
            report(r.map(|_| format!("wrote {}", out.display())))
        }
        "emit-obj" => {
            let out = opts.out.unwrap_or_else(|| default_out(file, "o"));
            let r = match triple {
                Some(t) => coil::compile_to_object_for(&src, &out, t),
                None => coil::compile_to_object(&src, &out),
            };
            report(r.map(|_| format!("wrote {}", out.display())))
        }
        "emit-ir" => report(match &opts.target {
            Some(t) => coil::emit_ir_for(&src, t),
            None => coil::emit_ir(&src),
        }),
        "expand" => report(coil::expand_to_string(&src)),
        "run" => run_aot(&src, opts.target.as_deref()),
        _ => usage(),
    }
}

/// Build to a temp executable, run it, and propagate its exit code. With a cross
/// `--target` the binary is run via `arch -<arch>` (Rosetta on macOS).
fn run_aot(src: &str, triple: Option<&str>) -> ExitCode {
    let exe = std::env::temp_dir().join(format!("coil_run_{}", std::process::id()));
    let build = match triple {
        Some(t) => coil::build_executable_for(src, &exe, TargetTriple::create(t)),
        None => coil::build_executable(src, &exe),
    };
    if let Err(e) = build {
        eprintln!("error: {e}");
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
}

impl Opts {
    fn parse(rest: &[String]) -> Result<Opts, String> {
        let mut out = None;
        let mut target = None;
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
                other => return Err(format!("unknown argument '{other}'")),
            }
        }
        Ok(Opts { out, target })
    }
}

fn report(result: Result<String, String>) -> ExitCode {
    match result {
        Ok(out) => {
            println!("{out}");
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("error: {e}");
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
        "usage: coil <build|run|emit-obj|emit-ir|expand> <file.coil> [-o out] [--target <triple>]"
    );
    ExitCode::from(2)
}
