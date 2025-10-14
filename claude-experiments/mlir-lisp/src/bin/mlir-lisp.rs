/// MLIR-Lisp CLI
///
/// Usage: mlir-lisp <file.lisp>
///
/// The file can define dialects, transforms, and programs.
/// Everything is self-contained in the Lisp file.

use mlir_lisp::self_contained::SelfContainedCompiler;
use melior::{Context, dialect::DialectRegistry};
use std::env;

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: mlir-lisp <file.lisp>");
        eprintln!();
        eprintln!("Example file:");
        eprintln!("  ;; Define dialect");
        eprintln!("  (import lisp-core)");
        eprintln!("  ");
        eprintln!("  ;; Write program");
        eprintln!("  (defn main [] i32");
        eprintln!("    (+ (* 10 20) 30))");
        std::process::exit(1);
    }

    let filename = &args[1];

    // Initialize MLIR
    let registry = DialectRegistry::new();
    let context = Context::new();
    context.append_dialect_registry(&registry);
    context.load_all_available_dialects();

    // Create compiler (it creates its own context internally)
    let mut compiler = SelfContainedCompiler::new();

    println!("Loading: {}", filename);

    // Load the file
    match compiler.load_file(filename) {
        Ok(_) => {
            println!("✅ File loaded successfully");
        }
        Err(e) => {
            eprintln!("❌ Error loading file: {}", e);
            std::process::exit(1);
        }
    }

    // Show what was loaded
    println!("\nRegistry Status:");
    println!("  Dialects: {:?}", compiler.registry().list_dialects());
    println!("  Transforms: {:?}", compiler.registry().list_transforms());
    println!("  Patterns: {:?}", compiler.registry().list_patterns());
}
