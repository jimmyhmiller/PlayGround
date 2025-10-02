use std::sync::Arc;
use bumpalo::Bump;
use rustyline::Editor;
use honu_rust::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Honu-Rust REPL");
    println!("Type expressions to evaluate them, or 'quit' to exit.");
    println!("Examples:");
    println!("  (+ 1 2)");
    println!("  (and true false)");
    println!("  (or false true)");
    println!("  (while condition body)");
    println!("  (cond [true 1] [false 2])");
    println!();

    let mut rl = Editor::<()>::new()?;
    let arena = Bump::new();
    let mut parser = Parser::new(&arena);
    
    // Register built-in operators
    register_builtin_operators(&mut parser);
    
    // Register built-in macros
    register_builtin_macros_to_parser(&mut parser);

    loop {
        let readline = rl.readline("honu> ");
        match readline {
            Ok(line) => {
                rl.add_history_entry(line.as_str());
                
                if line.trim() == "quit" || line.trim() == "exit" {
                    println!("Goodbye!");
                    break;
                }
                
                if line.trim().is_empty() {
                    continue;
                }
                
                match evaluate_expression(&arena, &mut parser, &line) {
                    Ok(result) => println!("=> {:?}", result),
                    Err(e) => println!("Error: {}", e),
                }
            }
            Err(rustyline::error::ReadlineError::Interrupted) => {
                println!("CTRL-C");
                break;
            }
            Err(rustyline::error::ReadlineError::Eof) => {
                println!("CTRL-D");
                break;
            }
            Err(err) => {
                println!("Error: {:?}", err);
                break;
            }
        }
    }

    Ok(())
}

fn register_builtin_operators(parser: &mut Parser) {
    let scopes = parser.enforester.env.current_scopes();
    
    // Arithmetic operators
    parser.enforester.env.bind(
        "+".to_string(),
        scopes.clone(),
        BindingInfo::BinaryOp { precedence: 10, assoc: Associativity::Left },
    );
    
    parser.enforester.env.bind(
        "-".to_string(),
        scopes.clone(),
        BindingInfo::BinaryOp { precedence: 10, assoc: Associativity::Left },
    );
    
    parser.enforester.env.bind(
        "*".to_string(),
        scopes.clone(),
        BindingInfo::BinaryOp { precedence: 20, assoc: Associativity::Left },
    );
    
    parser.enforester.env.bind(
        "/".to_string(),
        scopes.clone(),
        BindingInfo::BinaryOp { precedence: 20, assoc: Associativity::Left },
    );
    
    // Comparison operators
    parser.enforester.env.bind(
        "=".to_string(),
        scopes.clone(),
        BindingInfo::BinaryOp { precedence: 5, assoc: Associativity::Right },
    );
    
    parser.enforester.env.bind(
        "<".to_string(),
        scopes.clone(),
        BindingInfo::BinaryOp { precedence: 8, assoc: Associativity::Left },
    );
    
    parser.enforester.env.bind(
        ">".to_string(),
        scopes.clone(),
        BindingInfo::BinaryOp { precedence: 8, assoc: Associativity::Left },
    );
    
    parser.enforester.env.bind(
        "<=".to_string(),
        scopes.clone(),
        BindingInfo::BinaryOp { precedence: 8, assoc: Associativity::Left },
    );
    
    parser.enforester.env.bind(
        ">=".to_string(),
        scopes,
        BindingInfo::BinaryOp { precedence: 8, assoc: Associativity::Left },
    );
}

fn register_builtin_macros_to_parser(parser: &mut Parser) {
    let scopes = parser.enforester.env.current_scopes();
    let macros = register_builtin_macros();
    
    for (name, transformer) in macros {
        parser.enforester.env.bind(
            name.to_string(),
            scopes.clone(),
            BindingInfo::Macro(transformer),
        );
    }
}

fn evaluate_expression(
    arena: &Bump,
    parser: &mut Parser,
    input: &str,
) -> Result<String, ParseError> {
    // Lexical analysis
    let mut lexer = Lexer::new(input.to_string());
    let terms = lexer.tokenize()?;
    
    if terms.is_empty() {
        return Ok("()".to_string());
    }
    
    // Parse and evaluate
    let ast_nodes = parser.parse(terms)?;
    
    if ast_nodes.is_empty() {
        Ok("()".to_string())
    } else {
        // For simplicity, just show the AST structure
        // In a real interpreter, we would evaluate the AST
        Ok(format!("{:?}", ast_nodes))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_arithmetic_expression() {
        let arena = Bump::new();
        let mut parser = Parser::new(&arena);
        register_builtin_operators(&mut parser);
        
        let result = evaluate_expression(&arena, &mut parser, "(+ 1 2)");
        assert!(result.is_ok());
    }

    #[test]
    fn test_and_macro() {
        let arena = Bump::new();
        let mut parser = Parser::new(&arena);
        register_builtin_operators(&mut parser);
        register_builtin_macros_to_parser(&mut parser);
        
        let result = evaluate_expression(&arena, &mut parser, "(and true false)");
        assert!(result.is_ok());
    }

    #[test]
    fn test_or_macro() {
        let arena = Bump::new();
        let mut parser = Parser::new(&arena);
        register_builtin_operators(&mut parser);
        register_builtin_macros_to_parser(&mut parser);
        
        let result = evaluate_expression(&arena, &mut parser, "(or false true)");
        assert!(result.is_ok());
    }

    #[test]
    fn test_while_macro() {
        let arena = Bump::new();
        let mut parser = Parser::new(&arena);
        register_builtin_operators(&mut parser);
        register_builtin_macros_to_parser(&mut parser);
        
        let result = evaluate_expression(&arena, &mut parser, "(while condition body)");
        assert!(result.is_ok());
    }

    #[test]
    fn test_cond_macro() {
        let arena = Bump::new();
        let mut parser = Parser::new(&arena);
        register_builtin_operators(&mut parser);
        register_builtin_macros_to_parser(&mut parser);
        
        let result = evaluate_expression(&arena, &mut parser, "(cond [true 1] [false 2])");
        assert!(result.is_ok());
    }
}