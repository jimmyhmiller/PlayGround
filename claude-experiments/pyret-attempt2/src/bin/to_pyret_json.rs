use pyret_attempt2::{Parser, FileRegistry};
use pyret_attempt2::tokenizer::Tokenizer;
use pyret_attempt2::pyret_json::program_to_pyret_json;
use std::env;
use std::fs;
use std::io::{self, Read};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();

    let input = if args.len() > 1 {
        // Read from file
        fs::read_to_string(&args[1])?
    } else {
        // Read from stdin
        let mut buffer = String::new();
        io::stdin().read_to_string(&mut buffer)?;
        buffer
    };

    // Create file registry and register the input file
    let mut registry = FileRegistry::new();
    let file_id = registry.register("input.arr".to_string());

    let mut tokenizer = Tokenizer::new(&input, file_id);
    let tokens = tokenizer.tokenize();
    let mut parser = Parser::new(tokens, file_id);

    // Parse as full program
    let program = parser.parse_program()?;

    let json = program_to_pyret_json(&program, &registry);
    println!("{}", serde_json::to_string_pretty(&json)?);

    Ok(())
}
