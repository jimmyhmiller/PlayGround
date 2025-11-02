use pyret_attempt2::{Parser, Expr};
use pyret_attempt2::tokenizer::Tokenizer;

fn parse_expr(input: &str) -> Result<Expr, Box<dyn std::error::Error>> {
    let mut tokenizer = Tokenizer::new(input);
    let tokens = tokenizer.tokenize();
    let mut parser = Parser::new(tokens, "test.arr".to_string());
    Ok(parser.parse_expr_complete()?)
}

fn main() {
    println!("Testing: f(x, y,)");
    match parse_expr("f(x, y,)") {
        Ok(expr) => println!("  SUCCESS (unexpected!): {:?}", expr),
        Err(e) => println!("  FAILED (expected): {}", e),
    }

    println!("\nTesting: [1, 2, 3,]");
    match parse_expr("[1, 2, 3,]") {
        Ok(expr) => println!("  SUCCESS (unexpected!): {:?}", expr),
        Err(e) => println!("  FAILED (expected): {}", e),
    }

    println!("\nTesting: [list: 1, 2, 3,]");
    match parse_expr("[list: 1, 2, 3,]") {
        Ok(expr) => println!("  SUCCESS (unexpected!): {:?}", expr),
        Err(e) => println!("  FAILED (expected): {}", e),
    }
}
