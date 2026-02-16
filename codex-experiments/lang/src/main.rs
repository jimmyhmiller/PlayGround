mod ast;
mod codegen;
mod lexer;
mod parser;
mod qualify;
mod resolve;
mod runtime;
mod token;
mod typecheck;

use std::collections::HashSet;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
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

    let stdlib_path = format!("{}/stdlib", env!("CARGO_MANIFEST_DIR"));
    let search_paths = vec![stdlib_path];
    let (mut modules, link_libs) = discover_modules(&paths, &search_paths);

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
        let exe_path = std::env::temp_dir().join(format!("langc_run_{}", std::process::id()));

        if let Err(err) = codegen::compile_to_executable(&modules, &exe_path, &link_libs) {
            eprintln!("compilation error: {}", err.message);
            std::process::exit(1);
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
        let input = Path::new(&paths[0]);
        let build_dir = Path::new("build");
        fs::create_dir_all(build_dir).expect("failed to create build/ directory");
        let stem = input.file_stem().unwrap_or_else(|| std::ffi::OsStr::new("output"));
        let exe_path = build_dir.join(stem);

        if let Err(err) = codegen::compile_to_executable(&modules, &exe_path, &link_libs) {
            eprintln!("compilation error: {}", err.message);
            std::process::exit(1);
        }

        println!("built {}", exe_path.display());
    } else {
        println!("ok");
    }
}

/// Parse a .lang file, derive its module path, and collect link libs.
fn parse_and_derive(
    path: &str,
    roots: &[PathBuf],
    root_file: Option<&Path>,
) -> (ast::Module, Vec<String>) {
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

    let mut link_libs = Vec::new();
    for item in &module.items {
        if let ast::Item::Link(l) = item {
            link_libs.push(l.lib.clone());
        }
    }

    if module.path.is_none() {
        if let Some(p) = derive_module_path(Path::new(path), roots, root_file) {
            if !p.is_empty() {
                module.path = Some(p);
            }
        }
    }
    (module, link_libs)
}

/// Resolve a module name to a file path by searching the importing file's
/// directory and then each search path.
fn resolve_module_file(
    module_name: &str,
    importing_file: &Path,
    search_paths: &[String],
) -> Option<PathBuf> {
    let filename = format!("{}.lang", module_name);

    // Check directory of importing file
    if let Some(dir) = importing_file.parent() {
        let candidate = dir.join(&filename);
        if candidate.is_file() {
            return Some(candidate);
        }
    }

    // Check each search path
    for sp in search_paths {
        let candidate = Path::new(sp).join(&filename);
        if candidate.is_file() {
            return Some(candidate);
        }
    }

    None
}

/// Discover all modules starting from entry files, following `use` declarations.
fn discover_modules(
    entry_paths: &[String],
    search_paths: &[String],
) -> (Vec<ast::Module>, Vec<String>) {
    let mut loaded_files: HashSet<PathBuf> = HashSet::new();
    let mut queue: Vec<PathBuf> = Vec::new();
    let mut modules: Vec<ast::Module> = Vec::new();
    let mut link_libs: Vec<String> = Vec::new();

    // Build roots from entry files + search paths
    let mut roots: Vec<PathBuf> = Vec::new();
    for p in entry_paths {
        let path = Path::new(p);
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                roots.push(parent.to_path_buf());
            }
        }
    }
    for sp in search_paths {
        roots.push(Path::new(sp).to_path_buf());
    }
    if roots.is_empty() {
        roots.push(Path::new(".").to_path_buf());
    }

    // Determine which file is the "root" file (gets empty module path)
    let root_file = if entry_paths.len() == 1 && Path::new(&entry_paths[0]).is_file() {
        Some(fs::canonicalize(&entry_paths[0]).unwrap_or_else(|_| PathBuf::from(&entry_paths[0])))
    } else {
        None
    };

    // Seed queue with all entry paths
    for p in entry_paths {
        let path = PathBuf::from(p);
        let canonical = fs::canonicalize(&path).unwrap_or_else(|_| path.clone());
        if !loaded_files.contains(&canonical) {
            queue.push(path);
        }
    }

    while let Some(file_path) = queue.pop() {
        let canonical = fs::canonicalize(&file_path).unwrap_or_else(|_| file_path.clone());
        if loaded_files.contains(&canonical) {
            continue;
        }
        loaded_files.insert(canonical.clone());

        let path_str = file_path.to_str().unwrap_or("").to_string();
        let is_root = root_file.as_ref().map_or(false, |rf| &canonical == rf);
        let rf = if is_root { Some(file_path.as_path()) } else { None };

        let (module, libs) = parse_and_derive(&path_str, &roots, rf);
        link_libs.extend(libs);

        // Extract module names from `use` declarations
        for item in &module.items {
            if let ast::Item::Use(u) = item {
                if u.path.len() >= 2 {
                    let module_name = &u.path[0];
                    if let Some(resolved) = resolve_module_file(module_name, &file_path, search_paths) {
                        let resolved_canonical = fs::canonicalize(&resolved).unwrap_or_else(|_| resolved.clone());
                        if !loaded_files.contains(&resolved_canonical) {
                            queue.push(resolved);
                        }
                    }
                }
            }
        }

        modules.push(module);
    }

    // Dedup link libs
    let mut seen = HashSet::new();
    link_libs.retain(|lib| seen.insert(lib.clone()));

    (modules, link_libs)
}

/// Add `-l` flags for link libs collected from `link` declarations.
fn add_link_libs(cmd: &mut Command, link_libs: &[String]) {
    for lib in link_libs {
        cmd.arg(format!("-l{}", lib));
    }
}

fn run_bootstrap(targets: Vec<String>) -> ! {
    let exe = match env::current_exe() {
        Ok(p) => p,
        Err(err) => {
            eprintln!("failed to locate current executable: {}", err);
            std::process::exit(2);
        }
    };
    let mut cmd = Command::new(exe);
    cmd.arg("run");
    cmd.arg("compiler/main.lang");
    cmd.arg("--");
    if targets.is_empty() {
        // Default: bootstrap the compiler itself
        cmd.arg("compiler/main.lang");
    } else {
        for target in targets {
            cmd.arg(target);
        }
    }
    match cmd.status() {
        Ok(status) => std::process::exit(status.code().unwrap_or(1)),
        Err(err) => {
            eprintln!("failed to run bootstrap compiler: {}", err);
            std::process::exit(1);
        }
    }
}


fn derive_module_path(
    path: &Path,
    roots: &[PathBuf],
    root_file: Option<&Path>,
) -> Option<Vec<String>> {
    let mut best_root: Option<&PathBuf> = None;
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

pub(crate) fn find_llvm_config() -> String {
    let candidates = ["llvm-config", "/opt/homebrew/opt/llvm/bin/llvm-config", "/usr/local/opt/llvm/bin/llvm-config"];
    for c in &candidates {
        if let Ok(output) = Command::new(c).arg("--version").output() {
            if output.status.success() {
                return c.to_string();
            }
        }
    }
    "llvm-config".to_string()
}

fn compile_llvm_shims() -> PathBuf {
    let shims_c = Path::new(env!("CARGO_MANIFEST_DIR")).join("runtime/llvm_shims.c");
    let shims_o = Path::new(env!("CARGO_MANIFEST_DIR")).join("runtime/llvm_shims.o");
    let llvm_config = find_llvm_config();
    let cflags = Command::new(&llvm_config)
        .arg("--cflags")
        .output()
        .expect("failed to run llvm-config --cflags");
    let cflags_str = String::from_utf8_lossy(&cflags.stdout);
    let mut cc = Command::new("cc");
    for flag in cflags_str.split_whitespace() {
        if !flag.is_empty() {
            cc.arg(flag);
        }
    }
    cc.arg("-c").arg(&shims_c).arg("-o").arg(&shims_o);
    let status = cc.status().expect("failed to invoke cc for llvm_shims.c");
    if !status.success() {
        eprintln!("failed to compile runtime/llvm_shims.c");
        std::process::exit(1);
    }
    shims_o
}

fn compile_c_runtime() -> (PathBuf, PathBuf) {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let runtime_dir = manifest_dir.join("runtime");

    // Compile runtime.c with baked-in stdlib path
    let stdlib_define = format!("-DLANG_STDLIB_PATH=\"{}/stdlib\"", env!("CARGO_MANIFEST_DIR"));
    let runtime_o = runtime_dir.join("runtime.o");
    let status = Command::new("cc")
        .arg("-c")
        .arg(runtime_dir.join("runtime.c"))
        .arg(&stdlib_define)
        .arg("-O2")
        .arg("-o")
        .arg(&runtime_o)
        .status()
        .expect("failed to invoke cc for runtime.c");
    if !status.success() {
        eprintln!("failed to compile runtime/runtime.c");
        std::process::exit(1);
    }

    // Compile gc_bridge.c
    let gc_bridge_o = runtime_dir.join("gc_bridge.o");
    let gc_include = manifest_dir.join("../../claude-experiments/gc-library/include");
    let status = Command::new("cc")
        .arg("-c")
        .arg(runtime_dir.join("gc_bridge.c"))
        .arg("-I").arg(&gc_include)
        .arg("-O2")
        .arg("-o")
        .arg(&gc_bridge_o)
        .status()
        .expect("failed to invoke cc for gc_bridge.c");
    if !status.success() {
        eprintln!("failed to compile runtime/gc_bridge.c");
        std::process::exit(1);
    }

    (runtime_o, gc_bridge_o)
}

fn gc_library_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../claude-experiments/gc-library/target/release/libgc_library.a")
}

fn add_llvm_link_flags(cmd: &mut Command) {
    let llvm_config = find_llvm_config();
    if let Ok(output) = Command::new(&llvm_config).args(["--ldflags", "--libs", "--system-libs"]).output() {
        if output.status.success() {
            let flags = String::from_utf8_lossy(&output.stdout);
            for flag in flags.split_whitespace() {
                if !flag.is_empty() {
                    cmd.arg(flag);
                }
            }
            cmd.arg("-lc++");
            cmd.arg("-lz");
            cmd.arg("-lzstd");
            return;
        }
    }
    // Fallback
    cmd.arg("-L/opt/homebrew/lib");
    cmd.arg("-lLLVM");
    cmd.arg("-lc++");
    cmd.arg("-lz");
}
