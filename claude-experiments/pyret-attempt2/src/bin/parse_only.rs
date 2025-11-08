use std::env;
use std::fs;
use std::process;
use pyret_attempt2::Parser;
use pyret_attempt2::tokenizer::Tokenizer;

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

    // Tokenize
    let mut tokenizer = Tokenizer::new(&source);
    let tokens = tokenizer.tokenize();

    // Parse the source code
    let mut parser = Parser::new(tokens, input_file.to_string());
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
