// Generate example LLVM MIR JSON for testing
use iongraph_rust_redux::compilers::llvm::LLVMModule;
use std::env;
use std::fs;
use std::process;

fn main() {
    let args: Vec<String> = env::args().collect();

    let output_file = args.get(1).map(|s| s.as_str()).unwrap_or("examples/llvm/simple-loop.json");

    println!("Generating example LLVM MIR...");

    let module = LLVMModule::example_loop();

    let json = serde_json::to_string_pretty(&module).unwrap_or_else(|err| {
        eprintln!("Error serializing to JSON: {}", err);
        process::exit(1);
    });

    fs::write(output_file, json).unwrap_or_else(|err| {
        eprintln!("Error writing to {}: {}", output_file, err);
        process::exit(1);
    });

    println!("✓ LLVM MIR example written to: {}", output_file);
    println!("  Function: {}", module.functions[0].name);
    println!("  Blocks: {}", module.functions[0].blocks.len());
    println!("  Target: {}", module.target);
}
