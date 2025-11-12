use clap::{Parser as ClapParser, ValueEnum};
use pyret_attempt2::{ast::*, parser::Parser, tokenizer::Tokenizer, SchemeCompiler};
use std::fs;
use std::path::PathBuf;

#[derive(Debug, Clone, Copy, ValueEnum)]
enum Mode {
    /// Tokenize the input and print tokens
    Tokenize,
    /// Parse the input and print the raw AST (Debug format)
    Parse,
    /// Parse the input and output Pyret-compatible JSON
    Json,
    /// Compile to R4RS Scheme
    Scheme,
}

#[derive(ClapParser, Debug)]
#[command(name = "pyret-parser")]
#[command(about = "A Pyret parser with multiple output modes", long_about = None)]
struct Args {
    /// Input file to process
    #[arg(value_name = "FILE")]
    input: Option<PathBuf>,

    /// Output mode
    #[arg(short, long, value_enum, default_value = "json")]
    mode: Mode,

    /// Print tokens/AST with pretty formatting
    #[arg(short, long)]
    pretty: bool,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    // Read input from file or run demo
    let (code, filename) = if let Some(input_path) = args.input {
        let code = fs::read_to_string(&input_path)?;
        let filename = input_path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("input.arr")
            .to_string();
        (code, filename)
    } else {
        // No file provided, run demo
        println!("No input file provided. Run with --help for usage.");
        println!("\nExample usage:");
        println!("  cargo run -- --mode tokenize myfile.arr");
        println!("  cargo run -- --mode parse myfile.arr");
        println!("  cargo run -- --mode json myfile.arr");
        println!("  cargo run -- --mode scheme myfile.arr");
        println!("  cargo run -- myfile.arr  (defaults to json mode)");
        return Ok(());
    };

    // Create file registry
    let mut registry = FileRegistry::new();
    let file_id = registry.register(filename.clone());

    // Tokenize the input
    let mut tokenizer = Tokenizer::new(&code, file_id);
    let tokens = tokenizer.tokenize();

    match args.mode {
        Mode::Tokenize => {
            if args.pretty {
                println!("File: {}", filename);
                println!("Total tokens: {}\n", tokens.len());
                for (i, token) in tokens.iter().enumerate() {
                    println!(
                        "{:4}. {:25} | {:20} | {}:{}",
                        i + 1,
                        format!("{:?}", token.token_type),
                        if token.value.len() > 20 {
                            format!("{}...", &token.value[..17])
                        } else {
                            token.value.clone()
                        },
                        token.location.start_line,
                        token.location.start_column
                    );
                }
            } else {
                // Output debug format for tokens
                for token in &tokens {
                    println!("{:?}", token);
                }
            }
        }
        Mode::Parse => {
            let mut parser = Parser::new(tokens, file_id);
            match parser.parse_program() {
                Ok(program) => {
                    if args.pretty {
                        println!("File: {}", filename);
                        println!("Parse successful!\n");
                        println!("{:#?}", program);
                    } else {
                        println!("{:?}", program);
                    }
                }
                Err(e) => {
                    eprintln!("Parse error: {:?}", e);
                    std::process::exit(1);
                }
            }
        }
        Mode::Json => {
            let mut parser = Parser::new(tokens, file_id);
            match parser.parse_program() {
                Ok(program) => {
                    // Use the to_pyret_json conversion
                    let json_str = if args.pretty {
                        serde_json::to_string_pretty(&program)?
                    } else {
                        serde_json::to_string(&program)?
                    };
                    println!("{}", json_str);
                }
                Err(e) => {
                    eprintln!("Parse error: {:?}", e);
                    std::process::exit(1);
                }
            }
        }
        Mode::Scheme => {
            let mut parser = Parser::new(tokens, file_id);
            match parser.parse_program() {
                Ok(program) => {
                    let mut compiler = SchemeCompiler::new();
                    match compiler.compile_program(&program) {
                        Ok(scheme_code) => {
                            println!("{}", scheme_code);
                        }
                        Err(e) => {
                            eprintln!("Compilation error: {}", e);
                            std::process::exit(1);
                        }
                    }
                }
                Err(e) => {
                    eprintln!("Parse error: {:?}", e);
                    std::process::exit(1);
                }
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use pyret_attempt2::tokenizer::TokenType;

    #[test]
    fn test_tokenizer() {
        let code = "fun factorial(n): n end";
        let mut registry = FileRegistry::new();
        let file_id = registry.register("test.arr".to_string());
        let mut tokenizer = Tokenizer::new(code, file_id);
        let tokens = tokenizer.tokenize();

        assert!(!tokens.is_empty());
        assert!(tokens.iter().any(|t| t.token_type == TokenType::Fun));
        assert!(tokens
            .iter()
            .any(|t| t.token_type == TokenType::Name && t.value == "factorial"));
    }

    #[test]
    fn test_ast_serialization() {
        let mut registry = FileRegistry::new();
        let file_id = registry.register("test.arr".to_string());
        let loc = Loc::new(file_id, 1, 0, 0, 1, 1, 1);
        let expr = Expr::SNum {
            l: loc,
            value: "123".to_string(),
        };

        let json = serde_json::to_string(&expr).unwrap();
        // Our AST uses "value" field internally (as string for arbitrary precision)
        // The to_pyret_json binary transforms it to "n" for Pyret-compatible output
        assert!(json.contains("\"type\":\"s-num\""));
        assert!(json.contains("\"value\":\"123\""));
    }
}
