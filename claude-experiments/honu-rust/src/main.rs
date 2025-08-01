use bumpalo::Bump;
use rustyline::Editor;
use honu_rust::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Honu-Rust REPL");
    println!("Type expressions to evaluate them, or 'quit' to exit.");
    println!("Examples:");
    println!("  1 + 2");
    println!("  true and false");
    println!("  false or true");
    println!("  while condition {{ body }}");
    println!("  cond [true 1] [false 2]");
    println!();

    let mut rl: Editor<(), _> = Editor::new()?;
    let arena = Bump::new();
    let mut parser = Parser::new(&arena);
    
    // Parser::new already sets up basic operators and macros
    // Only register additional macros if needed
    register_additional_macros(&mut parser);

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
        scopes.clone(),
        BindingInfo::BinaryOp { precedence: 8, assoc: Associativity::Left },
    );
    
    parser.enforester.env.bind(
        "==".to_string(),
        scopes.clone(),
        BindingInfo::BinaryOp { precedence: 8, assoc: Associativity::Left },
    );
    
    parser.enforester.env.bind(
        "!=".to_string(),
        scopes.clone(),
        BindingInfo::BinaryOp { precedence: 8, assoc: Associativity::Left },
    );
    
    // Register built-in functions
    let binding_id = parser.enforester.env.new_binding();
    parser.enforester.env.bind(
        "if".to_string(),
        scopes.clone(),
        BindingInfo::Variable(binding_id),
    );
    
    let binding_id = parser.enforester.env.new_binding();
    parser.enforester.env.bind(
        "loop".to_string(),
        scopes.clone(),
        BindingInfo::Variable(binding_id),
    );
    
    let binding_id = parser.enforester.env.new_binding();
    parser.enforester.env.bind(
        "break".to_string(),
        scopes.clone(),
        BindingInfo::Variable(binding_id),
    );
    
    let binding_id = parser.enforester.env.new_binding();
    parser.enforester.env.bind(
        "do".to_string(),
        scopes.clone(),
        BindingInfo::Variable(binding_id),
    );
}

fn register_additional_macros(parser: &mut Parser) {
    let scopes = parser.enforester.env.current_scopes();
    
    // Register additional macros beyond what Parser::new provides
    // (Parser::new already registers and, or - let's add others)
    use std::sync::Arc;
    use crate::macros::{WhileMacro, CondMacro};
    
    parser.enforester.env.bind(
        "while".to_string(),
        scopes.clone(),
        BindingInfo::Macro(Arc::new(WhileMacro)),
    );
    
    parser.enforester.env.bind(
        "cond".to_string(),
        scopes.clone(),
        BindingInfo::Macro(Arc::new(CondMacro)),
    );
}

fn evaluate_expression(
    _arena: &Bump,
    parser: &mut Parser,
    input: &str,
) -> Result<String, ParseError> {
    // Lexical analysis
    let mut lexer = Lexer::new(input.to_string());
    let terms = lexer.tokenize()?;
    println!("Tokens: {:?}", terms);
    
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
        
        let result = evaluate_expression(&arena, &mut parser, "1 + 2");
        assert!(result.is_ok());
    }

    #[test]
    fn test_and_macro() {
        let arena = Bump::new();
        let mut parser = Parser::new(&arena);
        
        let result = evaluate_expression(&arena, &mut parser, "true and false");
        assert!(result.is_ok());
    }

    #[test]
    fn test_or_macro() {
        let arena = Bump::new();
        let mut parser = Parser::new(&arena);
        
        let result = evaluate_expression(&arena, &mut parser, "false or true");
        assert!(result.is_ok());
    }

    #[test]
    fn test_while_macro() {
        let arena = Bump::new();
        let mut parser = Parser::new(&arena);
        register_additional_macros(&mut parser);
        
        let result = evaluate_expression(&arena, &mut parser, "(while true 1)");
        assert!(result.is_ok());
    }

    #[test]
    fn test_cond_macro() {
        let arena = Bump::new();
        let mut parser = Parser::new(&arena);
        register_additional_macros(&mut parser);
        
        let result = evaluate_expression(&arena, &mut parser, "(cond [true 1] [false 2])");
        assert!(result.is_ok());
    }
}