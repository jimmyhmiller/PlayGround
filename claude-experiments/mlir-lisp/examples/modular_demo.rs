/// Modular Import System Demo
///
/// This demonstrates the import system where dialects, transforms, and patterns
/// are split across multiple files and imported as needed.
///
/// Run: cargo run --example modular_demo

use mlir_lisp::self_contained::SelfContainedCompiler;
use melior::Context;

fn main() {
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║             Modular Import System Demo                       ║");
    println!("║       Split Your Compiler Across Multiple Files!            ║");
    println!("╚═══════════════════════════════════════════════════════════════╝\n");

    let context = Context::new();
    let mut compiler = SelfContainedCompiler::new(&context);

    println!("═══════════════════════════════════════════════════════════════");
    println!("File Structure");
    println!("═══════════════════════════════════════════════════════════════\n");

    println!("bootstrap-modular.lisp");
    println!("  ├─ (import lisp-core)       → dialects/lisp-core.lisp");
    println!("  ├─ (import optimizations)   → dialects/optimizations.lisp");
    println!("  └─ (import lowering)        → dialects/lowering.lisp");
    println!();

    println!("dialects/");
    println!("  ├─ lisp-core.lisp           (defirdl-dialect lisp ...)");
    println!("  ├─ optimizations.lisp       (defpdl-pattern constant-fold-add ...)");
    println!("  └─ lowering.lisp            (deftransform lower-to-arith ...)");
    println!();

    println!("═══════════════════════════════════════════════════════════════");
    println!("Loading Modular Bootstrap");
    println!("═══════════════════════════════════════════════════════════════\n");

    match compiler.load_file("bootstrap-modular.lisp") {
        Ok(_) => {
            println!("✅ Bootstrap loaded successfully!");
            println!();
        }
        Err(e) => {
            println!("❌ Bootstrap failed: {}", e);
            println!();
            println!("Note: Make sure you're running from the project root:");
            println!("  cd /path/to/mlir-lisp");
            println!("  cargo run --example modular_demo");
            return;
        }
    }

    println!("═══════════════════════════════════════════════════════════════");
    println!("What Got Loaded");
    println!("═══════════════════════════════════════════════════════════════\n");

    // List dialects
    match compiler.eval_string("(list-dialects)") {
        Ok(result) => {
            println!("Loaded Dialects:");
            println!("{:#?}\n", result);
        }
        Err(e) => {
            println!("Error: {}", e);
        }
    }

    // List transforms
    match compiler.eval_string("(list-transforms)") {
        Ok(result) => {
            println!("Loaded Transforms:");
            println!("{:#?}\n", result);
        }
        Err(e) => {
            println!("Error: {}", e);
        }
    }

    // List patterns
    match compiler.eval_string("(list-patterns)") {
        Ok(result) => {
            println!("Loaded Patterns:");
            println!("{:#?}\n", result);
        }
        Err(e) => {
            println!("Error: {}", e);
        }
    }

    println!("═══════════════════════════════════════════════════════════════");
    println!("Registry Details");
    println!("═══════════════════════════════════════════════════════════════\n");

    let registry = compiler.registry();

    for dialect_name in registry.list_dialects() {
        if let Some(dialect) = registry.get_dialect(dialect_name) {
            println!("📦 Dialect: {}", dialect.name);
            println!("   Namespace: {}", dialect.namespace);
            println!("   Description: {}", dialect.description);
            println!("   Operations ({}):", dialect.operations.len());
            for op in &dialect.operations {
                println!("     • {} - {}", op.name, op.summary);
            }
            println!();
        }
    }

    for transform_name in registry.list_transforms() {
        if let Some(transform) = registry.get_transform(transform_name) {
            println!("🔄 Transform: {}", transform.name);
            println!("   Description: {}", transform.description);
            println!();
        }
    }

    println!("🎨 Patterns: {}", registry.list_patterns().len());
    for pattern_name in registry.list_patterns() {
        if let Some(pattern) = registry.get_pattern(pattern_name) {
            println!("   • {} (benefit: {})", pattern.name, pattern.benefit);
        }
    }
    println!();

    println!("═══════════════════════════════════════════════════════════════");
    println!("Benefits of Modular System");
    println!("═══════════════════════════════════════════════════════════════\n");

    println!("✅ SEPARATION OF CONCERNS");
    println!("   • Dialect definitions in separate files");
    println!("   • Optimizations in separate files");
    println!("   • Lowering in separate files");
    println!();

    println!("✅ REUSABILITY");
    println!("   • Import only what you need");
    println!("   • Share modules across projects");
    println!("   • Build libraries of dialects");
    println!();

    println!("✅ MAINTAINABILITY");
    println!("   • Easier to understand small files");
    println!("   • Easier to test individual modules");
    println!("   • Easier to collaborate");
    println!();

    println!("✅ NO DUPLICATION");
    println!("   • Files loaded only once");
    println!("   • Circular imports handled");
    println!("   • Clean dependency graph");
    println!();

    println!("═══════════════════════════════════════════════════════════════");
    println!("Example: Custom Bootstrap");
    println!("═══════════════════════════════════════════════════════════════\n");

    println!("Create your own bootstrap with only what you need:");
    println!();
    println!("  ;; my-bootstrap.lisp");
    println!("  (import lisp-core)      ; Core dialect");
    println!("  (import optimizations)  ; Just optimizations");
    println!("  ; Skip lowering - we don't need it!");
    println!();
    println!("  ;; Add custom patterns");
    println!("  (defpdl-pattern my-pattern ...)");
    println!();

    println!("═══════════════════════════════════════════════════════════════");
    println!("Try It Yourself!");
    println!("═══════════════════════════════════════════════════════════════\n");

    println!("1. Create a new file in dialects/");
    println!("   (defirdl-dialect my-lang ...)");
    println!();

    println!("2. Import it in your bootstrap:");
    println!("   (import my-lang)");
    println!();

    println!("3. Run your program:");
    println!("   cargo run --example modular_demo");
    println!();

    println!("The import system handles everything! 🎉");
}
