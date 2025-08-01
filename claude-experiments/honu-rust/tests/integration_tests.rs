use bumpalo::Bump;
use honu_rust::*;

fn setup_parser(arena: &Bump) -> Parser {
    // Parser::new already sets up basic operators and macros
    Parser::new(arena)
}

fn parse_expression<'a>(arena: &'a Bump, parser: &mut Parser<'a>, input: &str) -> Result<Vec<&'a AST<'a>>, ParseError> {
    let mut lexer = Lexer::new(input.to_string());
    let terms = lexer.tokenize()?;
    println!("Tokens: {:?}", terms);
    parser.parse(terms)
}

#[test]
fn test_basic_arithmetic() {
    let arena = Bump::new();
    let mut parser = setup_parser(&arena);
    
    let result = parse_expression(&arena, &mut parser, "(+ 1 2)");
    if let Err(e) = &result {
        println!("Error: {:?}", e);
    }
    assert!(result.is_ok());
    
    let ast = result.unwrap();
    assert_eq!(ast.len(), 1);
}

#[test]
fn test_operator_precedence() {
    let arena = Bump::new();
    let mut parser = setup_parser(&arena);
    
    // Test that * has higher precedence than +
    let result = parse_expression(&arena, &mut parser, "(+ 1 (* 2 3))");
    assert!(result.is_ok());
}

#[test]
fn test_and_macro_expansion() {
    let arena = Bump::new();
    let mut parser = setup_parser(&arena);
    
    // Test empty and
    let result = parse_expression(&arena, &mut parser, "(and)");
    assert!(result.is_ok());
    
    // Test single argument and
    let result = parse_expression(&arena, &mut parser, "(and true)");
    assert!(result.is_ok());
    
    // Test multiple argument and
    let result = parse_expression(&arena, &mut parser, "(and true false true)");
    assert!(result.is_ok());
}

#[test]
fn test_or_macro_expansion() {
    let arena = Bump::new();
    let mut parser = setup_parser(&arena);
    
    // Test empty or
    let result = parse_expression(&arena, &mut parser, "(or)");
    assert!(result.is_ok());
    
    // Test single argument or
    let result = parse_expression(&arena, &mut parser, "(or false)");
    assert!(result.is_ok());
    
    // Test multiple argument or
    let result = parse_expression(&arena, &mut parser, "(or false true false)");
    assert!(result.is_ok());
}

#[test]
fn test_while_macro_expansion() {
    let arena = Bump::new();
    let mut parser = setup_parser(&arena);
    
    let result = parse_expression(&arena, &mut parser, "(while (< x 10) (+ x 1))");
    assert!(result.is_ok());
}

#[test]
fn test_cond_macro_expansion() {
    let arena = Bump::new();
    let mut parser = setup_parser(&arena);
    
    // Test empty cond
    let result = parse_expression(&arena, &mut parser, "(cond)");
    assert!(result.is_ok());
    
    // Test single clause cond
    let result = parse_expression(&arena, &mut parser, "(cond [true 1])");
    assert!(result.is_ok());
    
    // Test multiple clause cond
    let result = parse_expression(&arena, &mut parser, "(cond [false 1] [true 2] [false 3])");
    assert!(result.is_ok());
}

#[test]
fn test_nested_macro_expansion() {
    let arena = Bump::new();
    let mut parser = setup_parser(&arena);
    
    // Test nested and/or
    let result = parse_expression(&arena, &mut parser, "(and (or false true) (or true false))");
    assert!(result.is_ok());
    
    // Test while with and condition
    let result = parse_expression(&arena, &mut parser, "(while (and (< x 10) (> x 0)) (+ x 1))");
    assert!(result.is_ok());
}

#[test]
fn test_complex_expressions() {
    let arena = Bump::new();
    let mut parser = setup_parser(&arena);
    
    // Test complex nested expression
    let result = parse_expression(&arena, &mut parser, 
        "(cond 
           [(and (< x 0) (> y 0)) (* x y)]
           [(or (= x 0) (= y 0)) 0]
           [true (+ x y)])");
    assert!(result.is_ok());
}

#[test]
fn test_multiple_expressions() {
    let arena = Bump::new();
    let mut parser = setup_parser(&arena);
    
    let input = "
        (+ 1 2)
        (and true false)
        (or false true)
        (while condition body)
        (cond [true 1] [false 2])
    ";
    
    let mut lexer = Lexer::new(input.to_string());
    let terms = lexer.tokenize().unwrap();
    
    let result = parser.parse(terms);
    assert!(result.is_ok());
    
    let ast = result.unwrap();
    assert_eq!(ast.len(), 5); // Should parse 5 expressions
}

#[test]
fn test_lexer_comprehensive() {
    let mut lexer = Lexer::new("(+ 1.5 \"hello\" true false nil)".to_string());
    let terms = lexer.tokenize().unwrap();
    
    assert_eq!(terms.len(), 1);
    match &terms[0] {
        Term::Bracket(BracketType::Paren, inner) => {
            assert_eq!(inner.len(), 6);
            
            // Check operator
            match &inner[0] {
                Term::Identifier(op, _) => assert_eq!(op, "+"),
                _ => panic!("Expected + operator"),
            }
            
            // Check number
            match &inner[1] {
                Term::Literal(LiteralValue::Number(n)) => assert_eq!(*n, 1.5),
                _ => panic!("Expected number 1.5"),
            }
            
            // Check string
            match &inner[2] {
                Term::Literal(LiteralValue::String(s)) => assert_eq!(s, "hello"),
                _ => panic!("Expected string hello"),
            }
            
            // Check booleans
            match &inner[3] {
                Term::Literal(LiteralValue::Boolean(b)) => assert_eq!(*b, true),
                _ => panic!("Expected boolean true"),
            }
            
            match &inner[4] {
                Term::Literal(LiteralValue::Boolean(b)) => assert_eq!(*b, false),
                _ => panic!("Expected boolean false"),
            }
            
            // Check nil
            match &inner[5] {
                Term::Literal(LiteralValue::Nil) => {},
                _ => panic!("Expected nil"),
            }
        }
        _ => panic!("Expected parenthesized expression"),
    }
}

#[test]
fn test_error_handling() {
    let arena = Bump::new();
    let mut parser = setup_parser(&arena);
    
    // Test invalid syntax
    let result = parse_expression(&arena, &mut parser, "(+ 1");
    assert!(result.is_err());
    
    // Test wrong number of arguments to while
    let result = parse_expression(&arena, &mut parser, "(while condition)");
    assert!(result.is_err());
    
    // Test invalid cond clause
    let result = parse_expression(&arena, &mut parser, "(cond [condition])");
    assert!(result.is_err());
}

#[test]
fn test_string_escapes() {
    let mut lexer = Lexer::new(r#""hello\nworld\t!""#.to_string());
    let terms = lexer.tokenize().unwrap();
    
    match &terms[0] {
        Term::Literal(LiteralValue::String(s)) => {
            assert_eq!(s, "hello\nworld\t!");
        }
        _ => panic!("Expected string literal"),
    }
}

#[test]
fn test_comments() {
    let mut lexer = Lexer::new("; This is a comment\n(+ 1 2) ; Another comment".to_string());
    let terms = lexer.tokenize().unwrap();
    
    assert_eq!(terms.len(), 1);
    match &terms[0] {
        Term::Bracket(BracketType::Paren, inner) => {
            assert_eq!(inner.len(), 3);
        }
        _ => panic!("Expected parenthesized expression"),
    }
}

#[test]
fn test_multiline_expressions() {
    let arena = Bump::new();
    let mut parser = setup_parser(&arena);
    
    let input = "
        (cond 
          [(< x 0) 
           (- x)]
          [(= x 0) 
           0]
          [true 
           x])
    ";
    
    let result = parse_expression(&arena, &mut parser, input);
    assert!(result.is_ok());
}