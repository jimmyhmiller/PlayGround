//! CLI — AOT only (no JIT, no `eval`).
//!
//!   coil build <file> [-o out]   compile + link a native executable (default: ./<stem>)
//!   coil run   <file>            build to a temp executable and run it (exit code = result)
//!   coil emit-obj <file> [-o o]  emit a native object file (default: ./<stem>.o)
//!   coil emit-ir  <file>         print the generated LLVM IR
//!   coil expand   <file>         print the program after macro expansion

use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        return usage();
    }
    let cmd = args[1].as_str();
    let file = &args[2];
    let out_flag = parse_out_flag(&args[3..]);

    let src = match std::fs::read_to_string(file) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error reading {file}: {e}");
            return ExitCode::FAILURE;
        }
    };

    match cmd {
        "build" => {
            let out = out_flag.unwrap_or_else(|| default_out(file, ""));
            report(coil::build_executable(&src, &out).map(|_| format!("wrote {}", out.display())))
        }
        "emit-obj" => {
            let out = out_flag.unwrap_or_else(|| default_out(file, "o"));
            report(coil::compile_to_object(&src, &out).map(|_| format!("wrote {}", out.display())))
        }
        "emit-ir" => report(coil::emit_ir(&src)),
        "expand" => report(coil::expand_to_string(&src)),
        "run" => run_aot(&src),
        _ => usage(),
    }
}

/// Build to a temp executable, run it, and propagate its exit code.
fn run_aot(src: &str) -> ExitCode {
    let exe = std::env::temp_dir().join(format!("coil_run_{}", std::process::id()));
    if let Err(e) = coil::build_executable(src, &exe) {
        eprintln!("error: {e}");
        return ExitCode::FAILURE;
    }
    let status = Command::new(&exe).status();
    let _ = std::fs::remove_file(&exe);
    match status {
        Ok(s) => ExitCode::from(s.code().unwrap_or(1) as u8),
        Err(e) => {
            eprintln!("error running executable: {e}");
            ExitCode::FAILURE
        }
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

fn parse_out_flag(rest: &[String]) -> Option<PathBuf> {
    match rest {
        [flag, path] if flag == "-o" => Some(PathBuf::from(path)),
        _ => None,
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
    eprintln!("usage: coil <build|run|emit-obj|emit-ir|expand> <file.coil> [-o out]");
    ExitCode::from(2)
}
