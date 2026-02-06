mod ast;
mod codegen;
mod lexer;
mod interp;
mod parser;
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
            eprintln!("usage: langc [check|run|build] <file.lang>...");
            std::process::exit(2);
        }
    };
    let (mode, paths) = if first == "check" || first == "run" || first == "build" {
        let mut paths: Vec<String> = Vec::new();
        for arg in args {
            if arg == "--" {
                break;
            }
            paths.push(arg);
        }
        if paths.is_empty() {
            eprintln!("usage: langc {} <file.lang>...", first);
            std::process::exit(2);
        }
        (first, paths)
    } else {
        ("check".to_string(), vec![first])
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

    let mut combined = ast::Module { path: None, items: Vec::new() };
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

        match Parser::new(tokens).parse_module() {
            Ok(module) => {
                combined.items.extend(module.items);
            }
            Err(errors) => {
                for err in errors {
                    eprintln!("parse error: {} at {}..{}", err.message, err.span.start, err.span.end);
                }
                std::process::exit(1);
            }
        }
    }

    let module = combined;
    if let Err(errors) = resolve::resolve_module(&module) {
        for err in errors {
            eprintln!("resolve error: {} at {}..{}", err.message, err.span.start, err.span.end);
        }
        std::process::exit(1);
    }
    if let Err(errors) = typecheck::typecheck_module(&module) {
        for err in errors {
            eprintln!("type error: {} at {}..{}", err.message, err.span.start, err.span.end);
        }
        std::process::exit(1);
    }
    if mode == "run" {
        match codegen::compile_and_run(&module) {
            Ok(val) => println!("{val}"),
            Err(err) => {
                eprintln!("codegen error: {}", err.message);
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

        if let Err(err) = codegen::compile_to_object(&module, &obj_path) {
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

        let status = Command::new("cc")
            .arg(&obj_path)
            .arg(&lib_path)
            .arg("-O2")
            .arg("-o")
            .arg(&exe_path)
            .status();
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
