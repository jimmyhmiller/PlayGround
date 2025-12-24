use quick_clojure_poc::*;
use std::cell::UnsafeCell;
use std::sync::Arc;

fn main() {
    let code = "(+ (* 2 3) 4)";
    let val = reader::read(code).unwrap();
    let ast = clojure_ast::analyze(&val).unwrap();

    let runtime = Arc::new(UnsafeCell::new(gc_runtime::GCRuntime::new()));
    let mut compiler = compiler::Compiler::new(runtime);
    compiler.compile(&ast).unwrap();
    let instructions = compiler.take_instructions();
    let num_locals = compiler.builder.num_locals;

    println!("IR instructions:");
    for (i, inst) in instructions.iter().enumerate() {
        println!("  {}: {:?}", i, inst);
    }

    let compiled =
        arm_codegen::Arm64CodeGen::compile_function(&instructions, num_locals, 0).unwrap();

    println!("\nGenerated {} ARM64 instructions", compiled.code_len);

    println!("\nExecuting...");
    let trampoline = trampoline::Trampoline::new(64 * 1024);
    let result = unsafe { trampoline.execute(compiled.code_ptr as *const u8) };
    // Result is tagged: 10 << 3 = 80
    println!("Result (tagged): {}", result);
    let untagged = result >> 3;
    println!("Result (untagged): {}", untagged);
    assert_eq!(untagged, 10);
    println!("SUCCESS!");
}
