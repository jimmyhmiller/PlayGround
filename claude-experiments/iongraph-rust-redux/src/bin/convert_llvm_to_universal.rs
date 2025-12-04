use iongraph_rust_redux::compilers::llvm::LLVMModule;
use iongraph_rust_redux::compilers::universal::llvm_to_universal;
use std::env;
use std::fs;
use std::process;

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: {} <llvm-json-file> [output-file]", args[0]);
        eprintln!();
        eprintln!("Example:");
        eprintln!("  {} examples/llvm/simple-loop.json output.json", args[0]);
        process::exit(1);
    }

    let input_file = &args[1];
    let json_str = fs::read_to_string(input_file).unwrap_or_else(|err| {
        eprintln!("Error reading {}: {}", input_file, err);
        process::exit(1);
    });

    let llvm: LLVMModule = serde_json::from_str(&json_str).unwrap_or_else(|err| {
        eprintln!("Error parsing LLVM JSON: {}", err);
        process::exit(1);
    });

    eprintln!("Converting LLVM MIR to Universal format...");
    eprintln!("  Functions: {}", llvm.functions.len());
    eprintln!("  Target: {}", llvm.target);

    let universal = llvm_to_universal(&llvm);

    let output_file = args.get(2).map(|s| s.as_str()).unwrap_or("output.json");

    let output_json = serde_json::to_string_pretty(&universal).unwrap_or_else(|err| {
        eprintln!("Error serializing to JSON: {}", err);
        process::exit(1);
    });

    fs::write(output_file, output_json).unwrap_or_else(|err| {
        eprintln!("Error writing to {}: {}", output_file, err);
        process::exit(1);
    });

    eprintln!("✓ Universal JSON written to: {}", output_file);
    eprintln!("  Blocks: {}", universal.blocks.len());
}
