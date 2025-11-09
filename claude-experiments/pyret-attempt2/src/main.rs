use pyret_attempt2::{ast::*, tokenizer::Tokenizer};

fn main() {
    println!("Pyret Parser - Phase 1: AST & Tokenizer");
    println!("{}", "=".repeat(70));

    // Demonstrate tokenizer
    demo_tokenizer();

    println!("\n{}", "=".repeat(70));

    // Demonstrate AST JSON serialization
    demo_ast_json();
}

fn demo_tokenizer() {
    println!("\n📝 Tokenizer Example:\n");

    let code = r#"
fun factorial(n):
  if n == 0:
    1
  else:
    n * factorial(n - 1)
  end
end
"#;

    let mut tokenizer = Tokenizer::new(code);
    let tokens = tokenizer.tokenize();

    println!("Code:\n{}", code);
    println!("Tokens found: {}\n", tokens.len());

    for (i, token) in tokens.iter().take(15).enumerate() {
        println!(
            "  {:2}. {:20} | {:15} | {}:{}",
            i + 1,
            format!("{:?}", token.token_type),
            if token.value.len() > 15 {
                format!("{}...", &token.value[..12])
            } else {
                token.value.clone()
            },
            token.location.start_line,
            token.location.start_col
        );
    }
}

fn demo_ast_json() {
    println!("\n🌳 AST JSON Serialization Example:\n");

    // Create a simple program AST manually
    let loc = Loc::new("example.arr".to_string(), 1, 0, 0, 1, 10, 10);

    // Simple number expression: 42
    let num_expr = Expr::SNum {
        l: loc.clone(),
        value: "42".to_string(),
    };

    // String expression: "Hello Pyret"
    let str_expr = Expr::SStr {
        l: loc.clone(),
        s: "Hello Pyret".to_string(),
    };

    // Block with two statements
    let block = Expr::SBlock {
        l: loc.clone(),
        stmts: vec![Box::new(num_expr), Box::new(str_expr)],
    };

    // Create a minimal program
    let program = Program::new(
        loc.clone(),
        None,                                     // no use statement
        Provide::SProvideNone { l: loc.clone() }, // provide-none
        ProvideTypes::SProvideTypesNone { l: loc.clone() }, // provide-types-none
        vec![],                                   // no provide blocks
        vec![],                                   // no imports
        block,
    );

    // Serialize to JSON
    let json = serde_json::to_string_pretty(&program).unwrap();

    println!("Generated AST as JSON:\n");
    println!("{}", json);

    println!("\n✅ JSON serialization working correctly!");
}

#[cfg(test)]
mod tests {
    use super::*;
    use pyret_attempt2::tokenizer::TokenType;

    #[test]
    fn test_tokenizer() {
        let code = "fun factorial(n): n end";
        let mut tokenizer = Tokenizer::new(code);
        let tokens = tokenizer.tokenize();

        assert!(!tokens.is_empty());
        assert!(tokens.iter().any(|t| t.token_type == TokenType::Fun));
        assert!(tokens.iter().any(|t| t.token_type == TokenType::Name && t.value == "factorial"));
    }

    #[test]
    fn test_ast_serialization() {
        let loc = Loc::new("test.arr".to_string(), 1, 0, 0, 1, 1, 1);
        let expr = Expr::SNum { l: loc, value: "123".to_string() };

        let json = serde_json::to_string(&expr).unwrap();
        assert!(json.contains("\"type\":\"s-num\""));
        assert!(json.contains("\"n\":123"));
    }
}
