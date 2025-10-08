/// Complete IRDL + Transform System Demo
///
/// This demonstrates the full workflow of:
/// 1. Parsing IRDL dialect definitions in Lisp
/// 2. Expanding them with macros
/// 3. Registering them in the dialect registry
/// 4. Parsing Transform patterns
/// 5. Registering transforms and PDL patterns

use mlir_lisp::{
    parser,
    macro_expander::MacroExpander,
    dialect_registry::DialectRegistry,
};

fn main() {
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║        Complete IRDL + Transform System Demo                 ║");
    println!("║          Meta-Circular Compiler Infrastructure               ║");
    println!("╚═══════════════════════════════════════════════════════════════╝\n");

    // Initialize the system
    let expander = MacroExpander::new();
    let mut registry = DialectRegistry::new();

    println!("═══════════════════════════════════════════════════════════════");
    println!("STEP 1: Define a Dialect with IRDL");
    println!("═══════════════════════════════════════════════════════════════\n");

    let dialect_source = r#"
(defirdl-dialect lisp
  :namespace "lisp"
  :description "High-level Lisp semantic operations"

  (defirdl-op constant
    :summary "Immutable constant value"
    :attributes [(value IntegerAttr)]
    :results [(result AnyInteger)]
    :traits [Pure NoMemoryEffect])

  (defirdl-op add
    :summary "Pure functional addition"
    :operands [(lhs AnyInteger) (rhs AnyInteger)]
    :results [(result AnyInteger)]
    :traits [Pure Commutative NoMemoryEffect]
    :constraints [(same-type lhs rhs result)]))
"#;

    println!("Dialect Definition (Lisp):");
    println!("{}", dialect_source);

    match parser::parse(dialect_source) {
        Ok((_, values)) => {
            for value in values {
                match expander.expand(&value) {
                    Ok(expanded) => {
                        println!("\n✓ Macro expanded successfully");

                        // Register with the dialect registry
                        match registry.process_expanded_form(&expanded) {
                            Ok(_) => {
                                println!("✓ Dialect registered successfully");
                            }
                            Err(e) => {
                                println!("❌ Registration error: {}", e);
                            }
                        }
                    }
                    Err(e) => {
                        println!("❌ Expansion error: {}", e);
                    }
                }
            }
        }
        Err(e) => {
            println!("❌ Parse error: {:?}", e);
        }
    }

    // Verify dialect was registered
    if let Some(dialect) = registry.get_dialect("lisp") {
        println!("\n✓ Dialect '{}' is registered", dialect.name);
        println!("  Namespace: {}", dialect.namespace);
        println!("  Description: {}", dialect.description);
        println!("  Operations:");
        for op in &dialect.operations {
            println!("    • {} - {}", op.name, op.summary);
        }
    }

    println!("\n═══════════════════════════════════════════════════════════════");
    println!("STEP 2: Define Transform Patterns");
    println!("═══════════════════════════════════════════════════════════════\n");

    let transform_source = r#"
(deftransform lower-lisp-to-arith
  :description "Lower lisp dialect to arith dialect"
  (transform.sequence
    (let [adds (transform.match :ops ["lisp.add"])]
      (transform.apply-patterns :to adds
        (use-pattern add-lowering)))))
"#;

    println!("Transform Definition (Lisp):");
    println!("{}", transform_source);

    match parser::parse(transform_source) {
        Ok((_, values)) => {
            for value in values {
                match expander.expand(&value) {
                    Ok(expanded) => {
                        println!("\n✓ Transform macro expanded successfully");

                        match registry.process_expanded_form(&expanded) {
                            Ok(_) => {
                                println!("✓ Transform registered successfully");
                            }
                            Err(e) => {
                                println!("❌ Registration error: {}", e);
                            }
                        }
                    }
                    Err(e) => {
                        println!("❌ Expansion error: {}", e);
                    }
                }
            }
        }
        Err(e) => {
            println!("❌ Parse error: {:?}", e);
        }
    }

    println!("\n═══════════════════════════════════════════════════════════════");
    println!("STEP 3: Define PDL Patterns");
    println!("═══════════════════════════════════════════════════════════════\n");

    let pattern_source = r#"
(defpdl-pattern add-lowering
  :benefit 1
  :description "Lower lisp.add to arith.addi"
  :match
  (let [lhs (pdl.operand)
        rhs (pdl.operand)
        type (pdl.type)
        op (pdl.operation "lisp.add" :operands [lhs rhs] :results [type])]
    op)
  :rewrite
  (let [new-op (pdl.operation "arith.addi" :operands [lhs rhs] :results [type])]
    (pdl.replace op :with new-op)))
"#;

    println!("PDL Pattern Definition (Lisp):");
    println!("{}", pattern_source);

    match parser::parse(pattern_source) {
        Ok((_, values)) => {
            for value in values {
                match expander.expand(&value) {
                    Ok(expanded) => {
                        println!("\n✓ PDL pattern macro expanded successfully");

                        match registry.process_expanded_form(&expanded) {
                            Ok(_) => {
                                println!("✓ Pattern registered successfully");
                            }
                            Err(e) => {
                                println!("❌ Registration error: {}", e);
                            }
                        }
                    }
                    Err(e) => {
                        println!("❌ Expansion error: {}", e);
                    }
                }
            }
        }
        Err(e) => {
            println!("❌ Parse error: {:?}", e);
        }
    }

    // Verify pattern was registered
    if let Some(pattern) = registry.get_pattern("add-lowering") {
        println!("\n✓ Pattern '{}' is registered", pattern.name);
        println!("  Benefit: {}", pattern.benefit);
        println!("  Description: {}", pattern.description);
    }

    println!("\n═══════════════════════════════════════════════════════════════");
    println!("STEP 4: Registry Summary");
    println!("═══════════════════════════════════════════════════════════════\n");

    println!("Registered Dialects:");
    for dialect_name in registry.list_dialects() {
        println!("  • {}", dialect_name);
    }

    println!("\nRegistered Transforms:");
    for transform_name in registry.list_transforms() {
        println!("  • {}", transform_name);
    }

    println!("\nRegistered Patterns:");
    for pattern_name in registry.list_patterns() {
        println!("  • {}", pattern_name);
    }

    println!("\n═══════════════════════════════════════════════════════════════");
    println!("What We've Accomplished");
    println!("═══════════════════════════════════════════════════════════════\n");

    println!("✅ PHASE 1: Macro System");
    println!("   • defirdl-dialect macro ✓");
    println!("   • defirdl-op macro ✓");
    println!("   • deftransform macro ✓");
    println!("   • defpdl-pattern macro ✓");

    println!("\n✅ PHASE 2: Dialect Registry");
    println!("   • Dialect registration ✓");
    println!("   • Transform registration ✓");
    println!("   • Pattern registration ✓");
    println!("   • Query by name ✓");

    println!("\n🔄 PHASE 3: Code Generation (Next Steps)");
    println!("   • Generate actual MLIR IRDL operations");
    println!("   • Generate Transform dialect IR");
    println!("   • Generate PDL pattern IR");
    println!("   • Integrate with MLIR context");

    println!("\n🔄 PHASE 4: Import System (Next Steps)");
    println!("   • #lang style imports");
    println!("   • import-dialect functionality");
    println!("   • Namespace resolution");
    println!("   • Module system");

    println!("\n═══════════════════════════════════════════════════════════════");
    println!("The Meta-Circular Vision");
    println!("═══════════════════════════════════════════════════════════════\n");

    println!("This system demonstrates:");
    println!("  1. DEFINING DIALECTS IN THE LANGUAGE ITSELF");
    println!("     • No C++ code needed");
    println!("     • Dialects are first-class data");
    println!("     • Runtime loading possible");
    println!();
    println!("  2. TRANSFORMATIONS AS DATA");
    println!("     • Pattern matching is declarative");
    println!("     • Rewrite rules are composable");
    println!("     • Transform IR is inspectable");
    println!();
    println!("  3. META-CIRCULAR COMPILATION");
    println!("     • Compiler defined in the language");
    println!("     • Self-describing system");
    println!("     • Can modify itself at runtime");
    println!();
    println!("This is the power of Lisp + MLIR!");
}
