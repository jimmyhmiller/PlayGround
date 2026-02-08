mod ast;
mod codegen;
mod lexer;
mod interp;
mod llvm_wrappers;
mod parser;
mod qualify;
mod resolve;
mod runtime;
mod token;
mod typecheck;

use std::env;
use std::fs;
use std::path::Path;
use std::process::Command;

use crate::lexer::Lexer;
use crate::parser::Parser;

fn main() {
    let mut args = env::args().skip(1);
    let first = match args.next() {
        Some(p) => p,
        None => {
            eprintln!("usage: langc [check|run|build|bootstrap] <file.lang>...");
            std::process::exit(2);
        }
    };
    if first == "bootstrap" {
        let mut targets: Vec<String> = Vec::new();
        for arg in args {
            if arg == "--" {
                break;
            }
            targets.push(arg);
        }
        run_bootstrap(targets);
    }
    let (mode, paths, run_args) = if first == "check" || first == "run" || first == "build" {
        let mut paths: Vec<String> = Vec::new();
        let mut run_args: Vec<String> = Vec::new();
        let mut after_sep = false;
        for arg in args {
            if !after_sep && arg == "--" {
                after_sep = true;
                continue;
            }
            if after_sep {
                run_args.push(arg);
            } else {
                paths.push(arg);
            }
        }
        if paths.is_empty() {
            eprintln!("usage: langc {} <file.lang>...", first);
            std::process::exit(2);
        }
        (first, paths, run_args)
    } else {
        ("check".to_string(), vec![first], Vec::new())
    };

    let mut all_paths: Vec<String> = Vec::new();
    for p in &paths {
        let path = Path::new(p);
        if path.is_dir() {
            collect_lang_files(path, &mut all_paths);
        } else {
            all_paths.push(p.clone());
        }
    }
    if all_paths.is_empty() {
        eprintln!("no .lang files found");
        std::process::exit(2);
    }

    let roots = build_roots(&paths);
    let root_file = if paths.len() == 1 && Path::new(&paths[0]).is_file() {
        Some(Path::new(&paths[0]).to_path_buf())
    } else {
        None
    };
    let mut modules: Vec<ast::Module> = Vec::new();
    for path in &all_paths {
        let src = match fs::read_to_string(path) {
            Ok(s) => s,
            Err(err) => {
                eprintln!("failed to read {}: {}", path, err);
                std::process::exit(2);
            }
        };

        let tokens = match Lexer::new(&src).lex_all() {
            Ok(toks) => toks,
            Err(errors) => {
                for err in errors {
                    eprintln!("lex error: {} at {}..{}", err.message, err.span.start, err.span.end);
                }
                std::process::exit(1);
            }
        };

        let mut module = match Parser::new(tokens).parse_module() {
            Ok(module) => module,
            Err(errors) => {
                for err in errors {
                    eprintln!("parse error: {} at {}..{}", err.message, err.span.start, err.span.end);
                }
                std::process::exit(1);
            }
        };
        if module.path.is_none() {
            if let Some(p) = derive_module_path(
                Path::new(path),
                &roots,
                root_file.as_deref(),
            ) {
                if !p.is_empty() {
                    module.path = Some(p);
                }
            }
        }
        modules.push(module);
    }

    if let Err(errors) = qualify::qualify_modules(&mut modules) {
        for err in errors {
            eprintln!("qualify error: {} at {}..{}", err.message, err.span.start, err.span.end);
        }
        std::process::exit(1);
    }
    if let Err(errors) = resolve::resolve_modules(&modules) {
        for err in errors {
            eprintln!("resolve error: {} at {}..{}", err.message, err.span.start, err.span.end);
        }
        std::process::exit(1);
    }
    if let Err(errors) = typecheck::typecheck_modules(&modules) {
        for err in errors {
            eprintln!("type error: {} at {}..{}", err.message, err.span.start, err.span.end);
        }
        std::process::exit(1);
    }
    if mode == "run" {
        // Use AOT compile→link→execute to avoid LLVM FastISel crash in JIT
        // with complex enum match IR (SIGBUS in handlePHINodesInSuccessorBlocks).
        let obj_path = std::env::temp_dir().join(format!("langc_run_{}.o", std::process::id()));
        let exe_path = std::env::temp_dir().join(format!("langc_run_{}", std::process::id()));

        if let Err(err) = codegen::compile_to_object(&modules, &obj_path) {
            eprintln!("codegen error: {}", err.message);
            std::process::exit(1);
        }

        let status = Command::new("cargo")
            .arg("build")
            .arg("--quiet")
            .arg("--lib")
            .status();
        match status {
            Ok(s) if s.success() => {}
            Ok(s) => {
                eprintln!("failed to build runtime lib: {}", s);
                let _ = fs::remove_file(&obj_path);
                std::process::exit(1);
            }
            Err(err) => {
                eprintln!("failed to invoke cargo: {err}");
                let _ = fs::remove_file(&obj_path);
                std::process::exit(1);
            }
        }

        let target_dir = env::var("CARGO_TARGET_DIR")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|_| Path::new(env!("CARGO_MANIFEST_DIR")).join("target"));
        let profile = env::var("PROFILE").unwrap_or_else(|_| "debug".to_string());
        let lib_path = target_dir.join(&profile).join("liblang_runtime.a");

        let mut link_cmd = Command::new("cc");
        link_cmd.arg(&obj_path).arg(&lib_path).arg("-O2").arg("-o").arg(&exe_path);
        add_llvm_link_flags(&mut link_cmd);
        let status = link_cmd.status();
        let _ = fs::remove_file(&obj_path);
        match status {
            Ok(s) if s.success() => {}
            Ok(s) => {
                eprintln!("link failed: {}", s);
                std::process::exit(1);
            }
            Err(err) => {
                eprintln!("failed to invoke cc: {err}");
                std::process::exit(1);
            }
        }

        let result = Command::new(&exe_path)
            .args(&run_args)
            .status();
        let _ = fs::remove_file(&exe_path);
        match result {
            Ok(s) => std::process::exit(s.code().unwrap_or(1)),
            Err(err) => {
                eprintln!("failed to run: {err}");
                std::process::exit(1);
            }
        }
    } else if mode == "build" {
        let input = Path::new(&all_paths[0]);
        let obj_path = input.with_extension("o");
        let mut exe_path = input.with_extension("");
        if input.extension().is_none() {
            exe_path = input.with_extension("out");
        }

        if let Err(err) = codegen::compile_to_object(&modules, &obj_path) {
            eprintln!("codegen error: {}", err.message);
            std::process::exit(1);
        }

        let status = Command::new("cargo")
            .arg("build")
            .arg("--quiet")
            .arg("--lib")
            .status();
        match status {
            Ok(s) if s.success() => {}
            Ok(s) => {
                eprintln!("failed to build runtime lib: {}", s);
                std::process::exit(1);
            }
            Err(err) => {
                eprintln!("failed to invoke cargo: {err}");
                std::process::exit(1);
            }
        }

        let target_dir = env::var("CARGO_TARGET_DIR")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|_| Path::new(env!("CARGO_MANIFEST_DIR")).join("target"));
        let profile = env::var("PROFILE").unwrap_or_else(|_| "debug".to_string());
        let lib_path = target_dir.join(profile).join("liblang_runtime.a");

        let mut link_cmd = Command::new("cc");
        link_cmd.arg(&obj_path).arg(&lib_path).arg("-O2").arg("-o").arg(&exe_path);
        add_llvm_link_flags(&mut link_cmd);
        let status = link_cmd.status();
        match status {
            Ok(s) if s.success() => {
                println!("built {}", exe_path.display());
            }
            Ok(s) => {
                eprintln!("link failed: {}", s);
                std::process::exit(1);
            }
            Err(err) => {
                eprintln!("failed to invoke cc: {err}");
                std::process::exit(1);
            }
        }
    } else {
        println!("ok");
    }
}

fn run_bootstrap(mut targets: Vec<String>) -> ! {
    let compiler_dir = Path::new("compiler");
    let mut compiler_files: Vec<String> = Vec::new();
    collect_lang_files(compiler_dir, &mut compiler_files);
    if compiler_files.is_empty() {
        eprintln!("no bootstrap compiler files found in ./compiler");
        std::process::exit(2);
    }
    if targets.is_empty() {
        targets = compiler_files.clone();
    }
    let exe = match env::current_exe() {
        Ok(p) => p,
        Err(err) => {
            eprintln!("failed to locate current executable: {}", err);
            std::process::exit(2);
        }
    };
    let mut cmd = Command::new(exe);
    cmd.arg("run");
    for file in compiler_files {
        cmd.arg(file);
    }
    cmd.arg("--");
    for target in targets {
        cmd.arg(target);
    }
    match cmd.status() {
        Ok(status) => std::process::exit(status.code().unwrap_or(1)),
        Err(err) => {
            eprintln!("failed to run bootstrap compiler: {}", err);
            std::process::exit(1);
        }
    }
}

fn collect_lang_files(dir: &Path, out: &mut Vec<String>) {
    let entries = match fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return,
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_lang_files(&path, out);
        } else if path.extension().and_then(|s| s.to_str()) == Some("lang") {
            if let Some(p) = path.to_str() {
                out.push(p.to_string());
            }
        }
    }
}

fn build_roots(paths: &[String]) -> Vec<std::path::PathBuf> {
    let mut roots = Vec::new();
    for p in paths {
        let path = Path::new(p);
        if path.is_dir() {
            roots.push(path.to_path_buf());
        } else if let Some(parent) = path.parent() {
            roots.push(parent.to_path_buf());
        }
    }
    if roots.is_empty() {
        roots.push(Path::new(".").to_path_buf());
    }
    roots
}

fn derive_module_path(
    path: &Path,
    roots: &[std::path::PathBuf],
    root_file: Option<&Path>,
) -> Option<Vec<String>> {
    let mut best_root: Option<&std::path::PathBuf> = None;
    let mut best_len = 0usize;
    for root in roots {
        if path.starts_with(root) {
            let len = root.components().count();
            if len >= best_len {
                best_len = len;
                best_root = Some(root);
            }
        }
    }
    let root = best_root?;
    let rel = path.strip_prefix(root).ok()?;
    let rel_dir = rel.parent().unwrap_or_else(|| Path::new(""));
    let mut parts = Vec::new();
    for comp in rel_dir.components() {
        if let std::path::Component::Normal(s) = comp {
            if let Some(text) = s.to_str() {
                parts.push(text.to_string());
            }
        }
    }
    let skip_stem = root_file.map_or(false, |root| root == path);
    if !skip_stem {
        let stem = path.file_stem().and_then(|s| s.to_str());
        if let Some(stem) = stem {
            let is_root = matches!(stem, "main" | "lib" | "mod");
            if !is_root {
                parts.push(stem.to_string());
            }
        }
    }
    Some(parts)
}

fn add_llvm_link_flags(cmd: &mut Command) {
    // Try multiple llvm-config paths
    let candidates = ["llvm-config", "/opt/homebrew/opt/llvm/bin/llvm-config", "/usr/local/opt/llvm/bin/llvm-config"];
    for llvm_config in &candidates {
        if let Ok(output) = Command::new(llvm_config).args(["--ldflags", "--libs", "--system-libs"]).output() {
            if output.status.success() {
                let flags = String::from_utf8_lossy(&output.stdout);
                for flag in flags.split_whitespace() {
                    if !flag.is_empty() {
                        cmd.arg(flag);
                    }
                }
                // Also add system libs on macOS
                cmd.arg("-lc++");
                cmd.arg("-lz");
                cmd.arg("-lzstd");
                return;
            }
        }
    }
    // Fallback
    cmd.arg("-L/opt/homebrew/lib");
    cmd.arg("-lLLVM");
    cmd.arg("-lc++");
    cmd.arg("-lz");
}
