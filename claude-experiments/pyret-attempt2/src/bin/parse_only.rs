use pyret_attempt2::tokenizer::Tokenizer;
use pyret_attempt2::{FileRegistry, Parser};
use std::env;
use std::fs;
use std::process;

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() != 2 {
        eprintln!("Usage: {} <input-file>", args[0]);
        process::exit(1);
    }

    let input_file = &args[1];

    // Read the input file
    let source = match fs::read_to_string(input_file) {
        Ok(content) => content,
        Err(e) => {
            eprintln!("Error reading file '{}': {}", input_file, e);
            process::exit(1);
        }
    };

    // Create file registry and register the file
    let mut registry = FileRegistry::new();
    let file_id = registry.register(input_file.to_string());

    // Tokenize
    let mut tokenizer = Tokenizer::new(&source, file_id);
    let tokens = tokenizer.tokenize();

    // Parse the source code
    let mut parser = Parser::new(tokens, file_id);
    match parser.parse_program() {
        Ok(_program) => {
            // Success! Just exit with 0
            process::exit(0);
        }
        Err(e) => {
            eprintln!("Parse error: {:?}", e);
            process::exit(1);
        }
    }
}
