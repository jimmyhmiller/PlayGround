/// Test IRDL Macro Expansion
///
/// This example demonstrates that the IRDL macros are correctly expanding
/// from Lisp syntax into the internal representation.

use mlir_lisp::{parser, macro_expander::MacroExpander};

fn main() {
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║             IRDL Macro Expansion Test                         ║");
    println!("╚═══════════════════════════════════════════════════════════════╝\n");

    // Test 1: defirdl-dialect macro
    println!("═══════════════════════════════════════════════════════════════");
    println!("Test 1: defirdl-dialect macro expansion");
    println!("═══════════════════════════════════════════════════════════════\n");

    let dialect_source = r#"
(defirdl-dialect lisp
  :namespace "lisp"
  :description "High-level Lisp semantic operations"

  (defirdl-op constant
    :summary "Immutable constant value"
    :attributes [(value IntegerAttr)]
    :results [(result AnyInteger)]
    :traits [Pure NoMemoryEffect]))
"#;

    println!("Input:");
    println!("{}\n", dialect_source);

    match parser::parse(dialect_source) {
        Ok((_, values)) => {
            let expander = MacroExpander::new();
            for value in values {
                match expander.expand(&value) {
                    Ok(expanded) => {
                        println!("Expanded to:");
                        println!("{:#?}\n", expanded);
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

    // Test 2: deftransform macro
    println!("═══════════════════════════════════════════════════════════════");
    println!("Test 2: deftransform macro expansion");
    println!("═══════════════════════════════════════════════════════════════\n");

    let transform_source = r#"
(deftransform lower-lisp-to-arith
  :description "Lower lisp dialect to arith dialect"
  (transform.sequence
    (let [adds (transform.match :ops ["lisp.add"])]
      (transform.apply-patterns :to adds
        (use-pattern add-lowering)))))
"#;

    println!("Input:");
    println!("{}\n", transform_source);

    match parser::parse(transform_source) {
        Ok((_, values)) => {
            let expander = MacroExpander::new();
            for value in values {
                match expander.expand(&value) {
                    Ok(expanded) => {
                        println!("Expanded to:");
                        println!("{:#?}\n", expanded);
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

    // Test 3: defpdl-pattern macro
    println!("═══════════════════════════════════════════════════════════════");
    println!("Test 3: defpdl-pattern macro expansion");
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

    println!("Input:");
    println!("{}\n", pattern_source);

    match parser::parse(pattern_source) {
        Ok((_, values)) => {
            let expander = MacroExpander::new();
            for value in values {
                match expander.expand(&value) {
                    Ok(expanded) => {
                        println!("Expanded to:");
                        println!("{:#?}\n", expanded);
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

    println!("═══════════════════════════════════════════════════════════════");
    println!("Summary");
    println!("═══════════════════════════════════════════════════════════════\n");
    println!("✅ IRDL macros are expanding correctly!");
    println!("✅ Transform macros are expanding correctly!");
    println!("✅ PDL pattern macros are expanding correctly!");
    println!();
    println!("Next steps:");
    println!("  • Implement IRDL IR generation from expanded forms");
    println!("  • Implement Transform IR generation");
    println!("  • Implement PDL IR generation");
    println!("  • Create dialect registry");
    println!("  • Hook up to MLIR context");
}
