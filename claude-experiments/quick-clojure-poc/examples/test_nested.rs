use quick_clojure_poc::*;
use std::sync::Arc;
use std::cell::UnsafeCell;

fn main() {
    let code = "(+ (* 2 3) 4)";
    let val = reader::read(code).unwrap();
    let ast = clojure_ast::analyze(&val).unwrap();

    let runtime = Arc::new(UnsafeCell::new(gc_runtime::GCRuntime::new()));
    let mut compiler = compiler::Compiler::new(runtime);
    let result_reg = compiler.compile(&ast).unwrap();
    let instructions = compiler.take_instructions();

    println!("IR instructions:");
    for (i, inst) in instructions.iter().enumerate() {
        println!("  {}: {:?}", i, inst);
    }

    let mut codegen = arm_codegen::Arm64CodeGen::new();
    let machine_code = codegen.compile(&instructions, &result_reg, 0).unwrap();

    println!("\nGenerated {} ARM64 instructions", machine_code.len());
    for (i, inst) in machine_code.iter().enumerate() {
        println!("  {:04x}: {:08x}", i * 4, inst);
    }

    println!("\nExecuting...");
    let result = codegen.execute().unwrap();
    println!("Result: {}", result);
    assert_eq!(result, 10);
    println!("SUCCESS!");
}
