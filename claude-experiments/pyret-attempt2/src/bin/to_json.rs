use pyret_attempt2::{Parser, Expr};
use pyret_attempt2::tokenizer::Tokenizer;
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

    let mut tokenizer = Tokenizer::new(&input);
    let tokens = tokenizer.tokenize();
    let mut parser = Parser::new(tokens, "input.arr".to_string());
    let expr = parser.parse_expr()?;

    let json = serde_json::to_string_pretty(&expr)?;
    println!("{}", json);

    Ok(())
}
