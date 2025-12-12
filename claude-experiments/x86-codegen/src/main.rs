//! x86-codegen CLI tool

use x86_codegen::{InstructionFilter, X86CodeGen};
use x86_codegen::rust_generator::RustGenerator;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();

    let xml_path = if args.len() > 1 {
        &args[1]
    } else {
        "x86reference.xml"
    };

    println!("// Loading x86reference.xml...");
    let codegen = X86CodeGen::new(xml_path)?;

    println!("// Found {} instruction entries", codegen.instructions.len());

    // Print available mnemonics for reference
    let mnemonics = codegen.available_mnemonics();
    eprintln!("// Available mnemonics ({}):", mnemonics.len());
    for mnem in &mnemonics[..20.min(mnemonics.len())] {
        eprintln!("//   {}", mnem);
    }
    eprintln!("//   ... and {} more", mnemonics.len().saturating_sub(20));

    // Generate code for commonly needed instructions
    let needed_mnemonics = vec![
        // Arithmetic
        "ADD", "SUB", "IMUL", "IDIV", "NEG", "CQO",
        // Bitwise
        "AND", "OR", "XOR", "NOT",
        // Shifts
        "SHL", "SHR", "SAR",
        // Move
        "MOV", "MOVSX", "MOVZX", "LEA",
        // Stack
        "PUSH", "POP",
        // Control flow
        "CALL", "RETN", "JMP",
        "JE", "JNE", "JG", "JGE", "JL", "JLE", "JA", "JAE", "JB", "JBE",
        // Comparison
        "CMP", "TEST",
        // Set on condition
        "SETE", "SETNE", "SETG", "SETGE", "SETL", "SETLE",
        // Misc
        "NOP", "INT",
    ];

    let filter = InstructionFilter::new()
        .allow(needed_mnemonics.iter().map(|s| s.to_string()).collect());

    let rust_code = codegen.generate_filtered(&RustGenerator, filter);
    println!("{}", rust_code);

    Ok(())
}
