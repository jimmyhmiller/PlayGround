use pyret_attempt2::{Parser, Expr, Member, Name, ConstructModifier, LetBind, Bind, ForBind, Program, Provide};
use pyret_attempt2::tokenizer::Tokenizer;

/// Helper to parse a string into an expression
/// This expects a complete expression with no trailing tokens (enforces EOF)
fn parse_expr(input: &str) -> Result<Expr, Box<dyn std::error::Error>> {
    let mut tokenizer = Tokenizer::new(input);
    let tokens = tokenizer.tokenize();

    // Debug: print tokens for troubleshooting
    if std::env::var("DEBUG_TOKENS").is_ok() {
        eprintln!("Tokens for '{}': {:?}", input, tokens.iter().map(|t| format!("{:?}", t.token_type)).collect::<Vec<_>>());
    }

    let mut parser = Parser::new(tokens, "test.arr".to_string());
    Ok(parser.parse_expr_complete()?)
}

/// Helper to parse a string into a program
fn parse_program(input: &str) -> Result<Program, Box<dyn std::error::Error>> {
    let mut tokenizer = Tokenizer::new(input);
    let tokens = tokenizer.tokenize();

    // Debug: print tokens for troubleshooting
    if std::env::var("DEBUG_TOKENS").is_ok() {
        eprintln!("Tokens for '{}': {:?}", input, tokens.iter().map(|t| format!("{:?}", t.token_type)).collect::<Vec<_>>());
    }

    let mut parser = Parser::new(tokens, "test.arr".to_string());
    Ok(parser.parse_program()?)
}

#[test]
fn test_parse_number() {
    let expr = parse_expr("42").expect("Failed to parse");

    match expr {
        Expr::SNum { n, .. } => {
            assert_eq!(n, 42.0);
        }
        _ => panic!("Expected SNum, got {:?}", expr),
    }
}

#[test]
fn test_parse_decimal_as_rational() {
    // Test that decimals are parsed as SNum
    // (Conversion to fractions happens during JSON serialization)
    let expr = parse_expr("3.14").expect("Failed to parse");

    match expr {
        Expr::SNum { n, .. } => {
            assert!((n - 3.14).abs() < 1e-10);
        }
        _ => panic!("Expected SNum for decimal, got {:?}", expr),
    }
}

#[test]
fn test_parse_simple_decimal() {
    // Test 2.5 is parsed as SNum
    // (Conversion to fractions happens during JSON serialization)
    let expr = parse_expr("2.5").expect("Failed to parse");

    match expr {
        Expr::SNum { n, .. } => {
            assert!((n - 2.5).abs() < 1e-10);
        }
        _ => panic!("Expected SNum, got {:?}", expr),
    }
}

#[test]
fn test_parse_explicit_rational() {
    // Test explicit rational like "3/4"
    let expr = parse_expr("3/4").expect("Failed to parse");

    match expr {
        Expr::SFrac { num, den, .. } => {
            assert_eq!(num, "3");
            assert_eq!(den, "4");
        }
        _ => panic!("Expected SFrac, got {:?}", expr),
    }
}

#[test]
fn test_parse_rational_simplification() {
    // Test that rationals are simplified: 6/8 = 3/4
    let expr = parse_expr("6/8").expect("Failed to parse");

    match expr {
        Expr::SFrac { num, den, .. } => {
            assert_eq!(num, "3");
            assert_eq!(den, "4");
        }
        _ => panic!("Expected SFrac, got {:?}", expr),
    }
}

#[test]
fn test_invalid_decimal_without_leading_digit() {
    // Pyret requires at least one digit before the decimal point
    // .5 is invalid (must be 0.5)
    // .0 is invalid (must be 0.0)
    // This matches official Pyret parser behavior: "BAD-NUMBER"

    let result = parse_expr(".5");
    assert!(result.is_err(), "Expected .5 to be invalid, but it parsed successfully");

    let result = parse_expr(".0");
    assert!(result.is_err(), "Expected .0 to be invalid, but it parsed successfully");

    let result = parse_expr(".123");
    assert!(result.is_err(), "Expected .123 to be invalid, but it parsed successfully");

    // Valid version with leading digit should work
    let expr = parse_expr("0.5").expect("Failed to parse");
    match expr {
        Expr::SNum { .. } | Expr::SFrac { .. } => {}, // Success - either is valid
        _ => panic!("Expected SNum or SFrac for 0.5, got {:?}", expr),
    }
}

#[test]
fn test_parse_string() {
    let expr = parse_expr("\"hello\"").expect("Failed to parse");

    match expr {
        Expr::SStr { s, .. } => {
            // The tokenizer strips the quotes
            assert_eq!(s, "hello");
        }
        _ => panic!("Expected SStr, got {:?}", expr),
    }
}

#[test]
fn test_parse_true() {
    let expr = parse_expr("true").expect("Failed to parse");

    match expr {
        Expr::SBool { b, .. } => {
            assert!(b);
        }
        _ => panic!("Expected SBool(true), got {:?}", expr),
    }
}

#[test]
fn test_parse_false() {
    let expr = parse_expr("false").expect("Failed to parse");

    match expr {
        Expr::SBool { b, .. } => {
            assert!(!b);
        }
        _ => panic!("Expected SBool(false), got {:?}", expr),
    }
}

#[test]
fn test_parse_identifier() {
    let expr = parse_expr("x").expect("Failed to parse");

    match expr {
        Expr::SId { id, .. } => {
            match id {
                Name::SName { s, .. } => {
                    assert_eq!(s, "x");
                }
                _ => panic!("Expected SName, got {:?}", id),
            }
        }
        _ => panic!("Expected SId, got {:?}", expr),
    }
}

#[test]
fn test_parse_simple_addition() {
    let expr = parse_expr("1 + 2").expect("Failed to parse");

    match expr {
        Expr::SOp { op, left, right, .. } => {
            assert_eq!(op, "op+");

            match *left {
                Expr::SNum { n, .. } => assert_eq!(n, 1.0),
                _ => panic!("Expected left to be SNum(1)"),
            }

            match *right {
                Expr::SNum { n, .. } => assert_eq!(n, 2.0),
                _ => panic!("Expected right to be SNum(2)"),
            }
        }
        _ => panic!("Expected SOp, got {:?}", expr),
    }
}

#[test]
fn test_parse_left_associative() {
    // 1 + 2 + 3 should parse as (1 + 2) + 3
    let expr = parse_expr("1 + 2 + 3").expect("Failed to parse");

    match expr {
        Expr::SOp { op, left, right, .. } => {
            assert_eq!(op, "op+");

            // Right should be just 3
            match *right {
                Expr::SNum { n, .. } => assert_eq!(n, 3.0),
                _ => panic!("Expected right to be SNum(3)"),
            }

            // Left should be (1 + 2)
            match *left {
                Expr::SOp { op, left, right, .. } => {
                    assert_eq!(op, "op+");
                    match *left {
                        Expr::SNum { n, .. } => assert_eq!(n, 1.0),
                        _ => panic!("Expected inner left to be SNum(1)"),
                    }
                    match *right {
                        Expr::SNum { n, .. } => assert_eq!(n, 2.0),
                        _ => panic!("Expected inner right to be SNum(2)"),
                    }
                }
                _ => panic!("Expected left to be SOp"),
            }
        }
        _ => panic!("Expected SOp, got {:?}", expr),
    }
}

#[test]
fn test_parse_multiple_operators() {
    let expr = parse_expr("x + y * z").expect("Failed to parse");

    // With flat precedence and left-associativity, this should be (x + y) * z
    match expr {
        Expr::SOp { op, .. } => {
            assert_eq!(op, "op*");
        }
        _ => panic!("Expected SOp, got {:?}", expr),
    }
}

#[test]
fn test_parse_subtraction() {
    let expr = parse_expr("10 - 5").expect("Failed to parse");

    match expr {
        Expr::SOp { op, left, right, .. } => {
            assert_eq!(op, "op-");

            match *left {
                Expr::SNum { n, .. } => assert_eq!(n, 10.0),
                _ => panic!("Expected left to be SNum(10)"),
            }

            match *right {
                Expr::SNum { n, .. } => assert_eq!(n, 5.0),
                _ => panic!("Expected right to be SNum(5)"),
            }
        }
        _ => panic!("Expected SOp, got {:?}", expr),
    }
}

#[test]
fn test_parse_comparison() {
    let expr = parse_expr("x < 10").expect("Failed to parse");

    match expr {
        Expr::SOp { op, .. } => {
            assert_eq!(op, "op<");
        }
        _ => panic!("Expected SOp, got {:?}", expr),
    }
}

#[test]
fn test_parse_logical_and() {
    let expr = parse_expr("true and false").expect("Failed to parse");

    match expr {
        Expr::SOp { op, .. } => {
            assert_eq!(op, "opand");
        }
        _ => panic!("Expected SOp, got {:?}", expr),
    }
}

#[test]
fn test_parse_serialization() {
    // Test that we can serialize to JSON
    let expr = parse_expr("42").expect("Failed to parse");
    let json = serde_json::to_string(&expr).expect("Failed to serialize");
    assert!(json.contains("\"type\":\"s-num\""));
    assert!(json.contains("\"n\":42"));
}

// ============================================================================
// Parenthesized Expression Tests
// ============================================================================

#[test]
fn test_parse_simple_paren_expr() {
    let expr = parse_expr("(42)").expect("Failed to parse");

    match expr {
        Expr::SParen { expr, .. } => {
            match *expr {
                Expr::SNum { n, .. } => assert_eq!(n, 42.0),
                _ => panic!("Expected SNum inside paren, got {:?}", expr),
            }
        }
        _ => panic!("Expected SParen, got {:?}", expr),
    }
}

#[test]
fn test_parse_paren_with_binop() {
    // (1 + 2) should parse as SParen containing SOp
    let expr = parse_expr("(1 + 2)").expect("Failed to parse");

    match expr {
        Expr::SParen { expr, .. } => {
            match *expr {
                Expr::SOp { op, left, right, .. } => {
                    assert_eq!(op, "op+");
                    match (*left, *right) {
                        (Expr::SNum { value: v1, .. }, Expr::SNum { value: v2, .. }) => {
                            assert_eq!(v1, "1");
                            assert_eq!(v2, "2");
                        }
                        _ => panic!("Expected SNum operands"),
                    }
                }
                _ => panic!("Expected SOp inside paren, got {:?}", expr),
            }
        }
        _ => panic!("Expected SParen, got {:?}", expr),
    }
}

#[test]
fn test_parse_nested_parens() {
    // ((5)) should parse as nested SParen nodes
    let expr = parse_expr("((5))").expect("Failed to parse");

    match expr {
        Expr::SParen { expr: outer, .. } => {
            match *outer {
                Expr::SParen { expr: inner, .. } => {
                    match *inner {
                        Expr::SNum { n, .. } => assert_eq!(n, 5.0),
                        _ => panic!("Expected SNum in innermost paren"),
                    }
                }
                _ => panic!("Expected inner SParen"),
            }
        }
        _ => panic!("Expected outer SParen, got {:?}", expr),
    }
}

// ============================================================================
// Function Application Tests
// ============================================================================

#[test]
fn test_parse_simple_function_call() {
    // f(x) should parse as SApp
    let expr = parse_expr("f(x)").expect("Failed to parse");

    match expr {
        Expr::SApp { _fun, args, .. } => {
            // Function should be an identifier 'f'
            match *_fun {
                Expr::SId { id, .. } => {
                    match id {
                        Name::SName { s, .. } => assert_eq!(s, "f"),
                        _ => panic!("Expected SName for function"),
                    }
                }
                _ => panic!("Expected SId for function, got {:?}", _fun),
            }

            // Should have one argument
            assert_eq!(args.len(), 1);

            // Argument should be identifier 'x'
            match &*args[0] {
                Expr::SId { id, .. } => {
                    match id {
                        Name::SName { s, .. } => assert_eq!(s, "x"),
                        _ => panic!("Expected SName for argument"),
                    }
                }
                _ => panic!("Expected SId for argument"),
            }
        }
        _ => panic!("Expected SApp, got {:?}", expr),
    }
}

#[test]
fn test_parse_function_call_multiple_args() {
    // f(x, y, z) should parse with three arguments
    let expr = parse_expr("f(x, y, z)").expect("Failed to parse");

    match expr {
        Expr::SApp { args, .. } => {
            assert_eq!(args.len(), 3);
        }
        _ => panic!("Expected SApp, got {:?}", expr),
    }
}

#[test]
fn test_parse_function_call_no_args() {
    // f() should parse with zero arguments
    let expr = parse_expr("f()").expect("Failed to parse");

    match expr {
        Expr::SApp { args, .. } => {
            assert_eq!(args.len(), 0);
        }
        _ => panic!("Expected SApp, got {:?}", expr),
    }
}

#[test]
fn test_parse_chained_function_calls() {
    // f(x)(y) should parse as nested SApp: (f(x))(y)
    let expr = parse_expr("f(x)(y)").expect("Failed to parse");

    match expr {
        Expr::SApp { _fun: outer_fun, args: outer_args, .. } => {
            // Outer function should be f(x)
            match *outer_fun {
                Expr::SApp { _fun: inner_fun, args: inner_args, .. } => {
                    // Inner function should be 'f'
                    match *inner_fun {
                        Expr::SId { id, .. } => {
                            match id {
                                Name::SName { s, .. } => assert_eq!(s, "f"),
                                _ => panic!("Expected SName"),
                            }
                        }
                        _ => panic!("Expected SId for inner function"),
                    }

                    // Inner args should be [x]
                    assert_eq!(inner_args.len(), 1);
                }
                _ => panic!("Expected inner SApp"),
            }

            // Outer args should be [y]
            assert_eq!(outer_args.len(), 1);
        }
        _ => panic!("Expected outer SApp, got {:?}", expr),
    }
}

#[test]
fn test_parse_function_call_with_expr_args() {
    // f(1 + 2, 3 * 4) should parse with expression arguments
    let expr = parse_expr("f(1 + 2, 3 * 4)").expect("Failed to parse");

    match expr {
        Expr::SApp { args, .. } => {
            assert_eq!(args.len(), 2);

            // First arg should be (1 + 2)
            match &*args[0] {
                Expr::SOp { op, .. } => assert_eq!(op, "op+"),
                _ => panic!("Expected SOp for first arg"),
            }

            // Second arg should be (3 * 4)
            match &*args[1] {
                Expr::SOp { op, .. } => assert_eq!(op, "op*"),
                _ => panic!("Expected SOp for second arg"),
            }
        }
        _ => panic!("Expected SApp, got {:?}", expr),
    }
}

// ============================================================================
// Whitespace Distinction Tests
// ============================================================================

#[test]
fn test_whitespace_paren_space() {
    // CORRECTED: "f (x)" with space should NOT parse as function application
    // Pyret treats this as TWO separate statements/expressions
    // So parse_expr_complete() should fail because there are leftover tokens
    // This test now correctly expects an error
    let result = parse_expr("f (x)");
    assert!(result.is_err(), "Should fail: 'f (x)' has leftover tokens after 'f'")
}

#[test]
fn test_whitespace_no_space() {
    // "f(x)" without space should parse as direct function application
    // This is: SApp { fun: f, args: [x] }
    let expr = parse_expr("f(x)").expect("Failed to parse");

    match expr {
        Expr::SApp { args, .. } => {
            assert_eq!(args.len(), 1);

            // Argument should be directly 'x', not wrapped in SParen
            match &*args[0] {
                Expr::SId { .. } => {}, // Good, direct identifier
                Expr::SParen { .. } => panic!("Should not be wrapped in SParen for f(x)"),
                _ => panic!("Expected SId for argument"),
            }
        }
        _ => panic!("Expected SApp, got {:?}", expr),
    }
}

// ============================================================================
// Mixed Expression Tests
// ============================================================================

#[test]
fn test_parse_function_in_binop() {
    // f(x) + g(y) should parse as SOp with two SApp children
    let expr = parse_expr("f(x) + g(y)").expect("Failed to parse");

    match expr {
        Expr::SOp { op, left, right, .. } => {
            assert_eq!(op, "op+");

            // Left should be f(x)
            match *left {
                Expr::SApp { .. } => {},
                _ => panic!("Expected SApp on left"),
            }

            // Right should be g(y)
            match *right {
                Expr::SApp { .. } => {},
                _ => panic!("Expected SApp on right"),
            }
        }
        _ => panic!("Expected SOp, got {:?}", expr),
    }
}

#[test]
fn test_parse_paren_changes_associativity() {
    // Without parens: 1 + 2 * 3 = (1 + 2) * 3 (left-associative)
    // With parens: 1 + (2 * 3) keeps the grouping
    let expr = parse_expr("1 + (2 * 3)").expect("Failed to parse");

    match expr {
        Expr::SOp { op, right, .. } => {
            assert_eq!(op, "op+");

            // Right should be parenthesized (2 * 3)
            match *right {
                Expr::SParen { expr, .. } => {
                    match *expr {
                        Expr::SOp { op, .. } => assert_eq!(op, "op*"),
                        _ => panic!("Expected SOp inside paren"),
                    }
                }
                _ => panic!("Expected SParen on right"),
            }
        }
        _ => panic!("Expected SOp, got {:?}", expr),
    }
}

// ===== Construct Expression Tests =====

#[test]
fn test_parse_empty_construct() {
    let expr = parse_expr("[list: ]").expect("Failed to parse");

    match expr {
        Expr::SConstruct { modifier, constructor, values, .. } => {
            assert_eq!(values.len(), 0);
            match modifier {
                ConstructModifier::SConstructNormal => {},
                _ => panic!("Expected SConstructNormal modifier"),
            }
            match *constructor {
                Expr::SId { id: Name::SName { s, .. }, .. } => {
                    assert_eq!(s, "list");
                }
                _ => panic!("Expected SId constructor, got {:?}", constructor),
            }
        }
        _ => panic!("Expected SConstruct, got {:?}", expr),
    }
}

#[test]
fn test_parse_construct_with_numbers() {
    let expr = parse_expr("[list: 1, 2, 3]").expect("Failed to parse");

    match expr {
        Expr::SConstruct { values, constructor, .. } => {
            assert_eq!(values.len(), 3);

            // Check constructor
            match *constructor {
                Expr::SId { id: Name::SName { s, .. }, .. } => {
                    assert_eq!(s, "list");
                }
                _ => panic!("Expected SId constructor"),
            }

            // Check values
            match &*values[0] {
                Expr::SNum { n, .. } => assert_eq!(*n, 1.0),
                _ => panic!("Expected SNum"),
            }
            match &*values[1] {
                Expr::SNum { n, .. } => assert_eq!(*n, 2.0),
                _ => panic!("Expected SNum"),
            }
            match &*values[2] {
                Expr::SNum { n, .. } => assert_eq!(*n, 3.0),
                _ => panic!("Expected SNum"),
            }
        }
        _ => panic!("Expected SConstruct, got {:?}", expr),
    }
}

#[test]
fn test_parse_construct_lazy_modifier() {
    let expr = parse_expr("[lazy list: 1, 2]").expect("Failed to parse");

    match expr {
        Expr::SConstruct { modifier, constructor, values, .. } => {
            match modifier {
                ConstructModifier::SConstructLazy => {},
                _ => panic!("Expected SConstructLazy modifier"),
            }
            match *constructor {
                Expr::SId { id: Name::SName { s, .. }, .. } => {
                    assert_eq!(s, "list");
                }
                _ => panic!("Expected SId constructor"),
            }
            assert_eq!(values.len(), 2);
        }
        _ => panic!("Expected SConstruct, got {:?}", expr),
    }
}

#[test]
fn test_parse_construct_with_identifiers() {
    let expr = parse_expr("[set: x, y, z]").expect("Failed to parse");

    match expr {
        Expr::SConstruct { values, constructor, .. } => {
            assert_eq!(values.len(), 3);

            // Check constructor
            match *constructor {
                Expr::SId { id: Name::SName { s, .. }, .. } => {
                    assert_eq!(s, "set");
                }
                _ => panic!("Expected SId constructor"),
            }

            // Check values
            match &*values[0] {
                Expr::SId { id, .. } => match id {
                    Name::SName { s, .. } => assert_eq!(s, "x"),
                    _ => panic!("Expected SName"),
                },
                _ => panic!("Expected SId"),
            }
        }
        _ => panic!("Expected SConstruct, got {:?}", expr),
    }
}

#[test]
fn test_parse_nested_constructs() {
    let expr = parse_expr("[list: [list: 1, 2], [list: 3, 4]]").expect("Failed to parse");

    match expr {
        Expr::SConstruct { values, .. } => {
            assert_eq!(values.len(), 2);

            // First nested construct
            match &*values[0] {
                Expr::SConstruct { values: inner1, .. } => {
                    assert_eq!(inner1.len(), 2);
                }
                _ => panic!("Expected nested SConstruct"),
            }

            // Second nested construct
            match &*values[1] {
                Expr::SConstruct { values: inner2, .. } => {
                    assert_eq!(inner2.len(), 2);
                }
                _ => panic!("Expected nested SConstruct"),
            }
        }
        _ => panic!("Expected SConstruct, got {:?}", expr),
    }
}

#[test]
fn test_parse_construct_with_expressions() {
    let expr = parse_expr("[list: 1 + 2, 3 * 4]").expect("Failed to parse");

    match expr {
        Expr::SConstruct { values, .. } => {
            assert_eq!(values.len(), 2);

            // First element should be 1 + 2
            match &*values[0] {
                Expr::SOp { op, .. } => assert_eq!(op, "op+"),
                _ => panic!("Expected SOp"),
            }

            // Second element should be 3 * 4
            match &*values[1] {
                Expr::SOp { op, .. } => assert_eq!(op, "op*"),
                _ => panic!("Expected SOp"),
            }
        }
        _ => panic!("Expected SConstruct, got {:?}", expr),
    }
}

// ===== Dot Access Tests =====

#[test]
fn test_parse_simple_dot_access() {
    let expr = parse_expr("obj.field").expect("Failed to parse");

    match expr {
        Expr::SDot { obj, field, .. } => {
            assert_eq!(field, "field");
            match &*obj {
                Expr::SId { id, .. } => match id {
                    Name::SName { s, .. } => assert_eq!(s, "obj"),
                    _ => panic!("Expected SName"),
                },
                _ => panic!("Expected SId"),
            }
        }
        _ => panic!("Expected SDot, got {:?}", expr),
    }
}

#[test]
fn test_parse_chained_dot_access() {
    let expr = parse_expr("obj.field1.field2").expect("Failed to parse");

    match expr {
        Expr::SDot { obj, field, .. } => {
            assert_eq!(field, "field2");

            // obj should be another dot access
            match &*obj {
                Expr::SDot { obj: inner_obj, field: inner_field, .. } => {
                    assert_eq!(inner_field, "field1");
                    match &**inner_obj {
                        Expr::SId { id, .. } => match id {
                            Name::SName { s, .. } => assert_eq!(s, "obj"),
                            _ => panic!("Expected SName"),
                        },
                        _ => panic!("Expected SId"),
                    }
                }
                _ => panic!("Expected inner SDot"),
            }
        }
        _ => panic!("Expected SDot, got {:?}", expr),
    }
}

#[test]
fn test_parse_dot_access_on_function_call() {
    let expr = parse_expr("f(x).field").expect("Failed to parse");

    match expr {
        Expr::SDot { obj, field, .. } => {
            assert_eq!(field, "field");
            match &*obj {
                Expr::SApp { .. } => {}, // Good, function call
                _ => panic!("Expected SApp for obj"),
            }
        }
        _ => panic!("Expected SDot, got {:?}", expr),
    }
}

#[test]
fn test_parse_function_call_on_dot_access() {
    let expr = parse_expr("obj.foo()").expect("Failed to parse");

    match expr {
        Expr::SApp { _fun, args, .. } => {
            assert_eq!(args.len(), 0);
            match &*_fun {
                Expr::SDot { obj, field, .. } => {
                    assert_eq!(field, "foo");
                    match &**obj {
                        Expr::SId { id, .. } => match id {
                            Name::SName { s, .. } => assert_eq!(s, "obj"),
                            _ => panic!("Expected SName"),
                        },
                        _ => panic!("Expected SId"),
                    }
                }
                _ => panic!("Expected SDot for function"),
            }
        }
        _ => panic!("Expected SApp, got {:?}", expr),
    }
}

#[test]
fn test_parse_dot_access_in_binop() {
    let expr = parse_expr("obj.x + obj.y").expect("Failed to parse");

    match expr {
        Expr::SOp { op, left, right, .. } => {
            assert_eq!(op, "op+");

            // Left should be obj.x
            match &*left {
                Expr::SDot { field, .. } => assert_eq!(field, "x"),
                _ => panic!("Expected SDot on left"),
            }

            // Right should be obj.y
            match &*right {
                Expr::SDot { field, .. } => assert_eq!(field, "y"),
                _ => panic!("Expected SDot on right"),
            }
        }
        _ => panic!("Expected SOp, got {:?}", expr),
    }
}

// ===== Bracket Access Tests =====

#[test]
fn test_parse_simple_bracket_access() {
    let expr = parse_expr("arr[0]").expect("Failed to parse");

    match expr {
        Expr::SBracket { obj, field, .. } => {
            // Check obj is an identifier
            match &*obj {
                Expr::SId { id, .. } => match id {
                    Name::SName { s, .. } => assert_eq!(s, "arr"),
                    _ => panic!("Expected SName"),
                },
                _ => panic!("Expected SId"),
            }

            // Check field is a number
            match &*field {
                Expr::SNum { n, .. } => assert_eq!(*n, 0.0),
                _ => panic!("Expected SNum"),
            }
        }
        _ => panic!("Expected SBracket, got {:?}", expr),
    }
}

#[test]
fn test_parse_bracket_access_with_string() {
    let expr = parse_expr(r#"dict["key"]"#).expect("Failed to parse");

    match expr {
        Expr::SBracket { obj, field, .. } => {
            match &*obj {
                Expr::SId { id, .. } => match id {
                    Name::SName { s, .. } => assert_eq!(s, "dict"),
                    _ => panic!("Expected SName"),
                },
                _ => panic!("Expected SId"),
            }

            match &*field {
                Expr::SStr { s, .. } => assert_eq!(s, "key"),
                _ => panic!("Expected SStr"),
            }
        }
        _ => panic!("Expected SBracket, got {:?}", expr),
    }
}

#[test]
fn test_parse_chained_bracket_access() {
    let expr = parse_expr("matrix[i][j]").expect("Failed to parse");

    match expr {
        Expr::SBracket { obj, field, .. } => {
            // Outer bracket should access [j]
            match &*field {
                Expr::SId { id, .. } => match id {
                    Name::SName { s, .. } => assert_eq!(s, "j"),
                    _ => panic!("Expected SName for j"),
                },
                _ => panic!("Expected SId for j"),
            }

            // Inner should be matrix[i]
            match &*obj {
                Expr::SBracket { obj: inner_obj, field: inner_field, .. } => {
                    match &**inner_obj {
                        Expr::SId { id, .. } => match id {
                            Name::SName { s, .. } => assert_eq!(s, "matrix"),
                            _ => panic!("Expected SName for matrix"),
                        },
                        _ => panic!("Expected SId for matrix"),
                    }
                    match &**inner_field {
                        Expr::SId { id, .. } => match id {
                            Name::SName { s, .. } => assert_eq!(s, "i"),
                            _ => panic!("Expected SName for i"),
                        },
                        _ => panic!("Expected SId for i"),
                    }
                }
                _ => panic!("Expected inner SBracket"),
            }
        }
        _ => panic!("Expected SBracket, got {:?}", expr),
    }
}

#[test]
fn test_parse_mixed_dot_and_bracket() {
    let expr = parse_expr("obj.foo[0].bar").expect("Failed to parse");

    match expr {
        Expr::SDot { obj, field, .. } => {
            // Outer should be .bar
            assert_eq!(field, "bar");

            // Middle should be bracket access [0]
            match &*obj {
                Expr::SBracket { obj: bracket_obj, field: bracket_field, .. } => {
                    match &**bracket_field {
                        Expr::SNum { n, .. } => assert_eq!(*n, 0.0),
                        _ => panic!("Expected SNum for [0]"),
                    }

                    // Inner should be obj.foo
                    match &**bracket_obj {
                        Expr::SDot { obj: dot_obj, field: dot_field, .. } => {
                            assert_eq!(dot_field, "foo");
                            match &**dot_obj {
                                Expr::SId { id, .. } => match id {
                                    Name::SName { s, .. } => assert_eq!(s, "obj"),
                                    _ => panic!("Expected SName for obj"),
                                },
                                _ => panic!("Expected SId for obj"),
                            }
                        }
                        _ => panic!("Expected SDot for obj.foo"),
                    }
                }
                _ => panic!("Expected SBracket for [0]"),
            }
        }
        _ => panic!("Expected SDot, got {:?}", expr),
    }
}

#[test]
fn test_parse_bracket_with_expression() {
    let expr = parse_expr("arr[i + 1]").expect("Failed to parse");

    match expr {
        Expr::SBracket { field, .. } => {
            // Field should be i + 1
            match &*field {
                Expr::SOp { op, .. } => assert_eq!(op, "op+"),
                _ => panic!("Expected SOp"),
            }
        }
        _ => panic!("Expected SBracket, got {:?}", expr),
    }
}

// ============================================================================
// Object Expression Tests
// ============================================================================

#[test]
fn test_parse_empty_object() {
    let expr = parse_expr("{}").expect("Failed to parse empty object");

    match expr {
        Expr::SObj { fields, .. } => {
            assert_eq!(fields.len(), 0, "Empty object should have no fields");
        }
        _ => panic!("Expected SObj, got {:?}", expr),
    }
}

#[test]
fn test_parse_simple_object() {
    let expr = parse_expr("{ x: 1, y: 2 }").expect("Failed to parse simple object");

    match expr {
        Expr::SObj { fields, .. } => {
            assert_eq!(fields.len(), 2, "Object should have 2 fields");

            // Check first field
            match &fields[0] {
                Member::SDataField { name, value, .. } => {
                    assert_eq!(name, "x");
                    match value.as_ref() {
                        Expr::SNum { n, .. } => assert_eq!(*n, 1.0f64),
                        _ => panic!("Expected SNum for x value"),
                    }
                }
                _ => panic!("Expected SDataField"),
            }

            // Check second field
            match &fields[1] {
                Member::SDataField { name, value, .. } => {
                    assert_eq!(name, "y");
                    match value.as_ref() {
                        Expr::SNum { n, .. } => assert_eq!(*n, 2.0f64),
                        _ => panic!("Expected SNum for y value"),
                    }
                }
                _ => panic!("Expected SDataField"),
            }
        }
        _ => panic!("Expected SObj, got {:?}", expr),
    }
}

#[test]
fn test_parse_nested_object() {
    let expr = parse_expr("{ point: { x: 0, y: 0 } }").expect("Failed to parse nested object");

    match expr {
        Expr::SObj { fields, .. } => {
            assert_eq!(fields.len(), 1);

            match &fields[0] {
                Member::SDataField { name, value, .. } => {
                    assert_eq!(name, "point");

                    // Check nested object
                    match value.as_ref() {
                        Expr::SObj { fields: inner_fields, .. } => {
                            assert_eq!(inner_fields.len(), 2);
                        }
                        _ => panic!("Expected nested SObj"),
                    }
                }
                _ => panic!("Expected SDataField"),
            }
        }
        _ => panic!("Expected SObj, got {:?}", expr),
    }
}

#[test]
fn test_parse_object_with_expressions() {
    let expr = parse_expr("{ sum: 1 + 2, product: 3 * 4 }").expect("Failed to parse object with expressions");

    match expr {
        Expr::SObj { fields, .. } => {
            assert_eq!(fields.len(), 2);

            // First field should have an addition operation
            match &fields[0] {
                Member::SDataField { name, value, .. } => {
                    assert_eq!(name, "sum");
                    match value.as_ref() {
                        Expr::SOp { op, .. } => assert_eq!(op, "op+"),
                        _ => panic!("Expected SOp"),
                    }
                }
                _ => panic!("Expected SDataField"),
            }
        }
        _ => panic!("Expected SObj, got {:?}", expr),
    }
}

#[test]
fn test_parse_object_trailing_comma() {
    let expr = parse_expr("{ x: 1, y: 2, }").expect("Failed to parse object with trailing comma");

    match expr {
        Expr::SObj { fields, .. } => {
            assert_eq!(fields.len(), 2, "Trailing comma should not affect parsing");
        }
        _ => panic!("Expected SObj, got {:?}", expr),
    }
}

#[test]
fn test_parse_object_with_method() {
    let expr = parse_expr("{ method _plus(self, other): self.arr end }").expect("Failed to parse object with method");

    match expr {
        Expr::SObj { fields, .. } => {
            assert_eq!(fields.len(), 1, "Object should have 1 method field");

            match &fields[0] {
                Member::SMethodField { name, args, body, .. } => {
                    assert_eq!(name, "_plus", "Method name should be _plus");
                    assert_eq!(args.len(), 2, "Method should have 2 parameters");

                    // Check parameter names
                    match &args[0] {
                        Bind::SBind { id, .. } => match id {
                            Name::SName { s, .. } => assert_eq!(s, "self"),
                            _ => panic!("Expected SName for first parameter"),
                        },
                        _ => panic!("Expected SBind for first parameter"),
                    }

                    match &args[1] {
                        Bind::SBind { id, .. } => match id {
                            Name::SName { s, .. } => assert_eq!(s, "other"),
                            _ => panic!("Expected SName for second parameter"),
                        },
                        _ => panic!("Expected SBind for second parameter"),
                    }

                    // Check body is a block with one statement
                    match body.as_ref() {
                        Expr::SBlock { stmts, .. } => {
                            assert_eq!(stmts.len(), 1, "Method body should have 1 statement");
                            // The statement should be a dot access (self.arr)
                            match stmts[0].as_ref() {
                                Expr::SDot { obj, field, .. } => {
                                    match obj.as_ref() {
                                        Expr::SId { id, .. } => match id {
                                            Name::SName { s, .. } => assert_eq!(s, "self"),
                                            _ => panic!("Expected SName for object"),
                                        },
                                        _ => panic!("Expected SId for object"),
                                    }
                                    assert_eq!(field, "arr");
                                }
                                _ => panic!("Expected SDot for method body statement"),
                            }
                        }
                        _ => panic!("Expected SBlock for method body"),
                    }
                }
                _ => panic!("Expected SMethodField, got {:?}", fields[0]),
            }
        }
        _ => panic!("Expected SObj, got {:?}", expr),
    }
}

// ============================================================================
// Bug Fix Tests - Trailing Token Detection
// ============================================================================

#[test]
fn test_reject_trailing_tokens() {
    // Bug: ideal-thorny-kingfisher
    // Parser should reject "42 unexpected"
    let result = parse_expr("42 unexpected");
    assert!(result.is_err(), "Should reject trailing tokens after number");
}

#[test]
fn test_reject_multiple_expressions() {
    // Bug: oily-awkward-hedgehog
    // Parser should reject "1 + 2 3 + 4" (multiple expressions without separator)
    let result = parse_expr("1 + 2 3 + 4");
    assert!(result.is_err(), "Should reject multiple expressions without separator");
}

#[test]
fn test_reject_unmatched_closing_paren() {
    // Bug: comfortable-pink-hedgehog
    // Parser should reject "42)"
    let result = parse_expr("42)");
    assert!(result.is_err(), "Should reject unmatched closing parenthesis");
}

#[test]
fn test_reject_unmatched_closing_bracket() {
    // Bug: palatable-edible-crayfish
    // Parser should reject "1]"
    let result = parse_expr("1]");
    assert!(result.is_err(), "Should reject unmatched closing bracket");
}

#[test]
fn test_reject_invalid_character_after_valid_expr() {
    // Bug: reflecting-enchanting-caribou
    // Parser should reject "f(x) @"
    let result = parse_expr("f(x) @");
    assert!(result.is_err(), "Should reject invalid character after valid expression");
}

#[test]
fn test_reject_trailing_operators() {
    // Additional test: trailing operators should also fail
    let result = parse_expr("1 + 2 +");
    assert!(result.is_err(), "Should reject trailing operator");
}

#[test]
fn test_reject_trailing_comma() {
    // Additional test: trailing comma in function call
    let result = parse_expr("f(1, 2,) extra");
    assert!(result.is_err(), "Should reject trailing tokens after function call");
}

#[test]
fn test_whitespace_before_paren_stops_parsing() {
    // Bug: slow-thankful-krill
    // When there is whitespace before parentheses, like "f (x)",
    // Pyret treats this as TWO separate statements:
    // 1. The identifier "f"
    // 2. A parenthesized expression "(x)"
    //
    // This means parse_expr_complete() should fail because there are
    // leftover tokens after parsing "f".
    let result = parse_expr("f (x)");
    assert!(result.is_err(), "Should fail: 'f (x)' has leftover tokens after 'f'");
}

#[test]
fn test_no_whitespace_before_paren_is_function_call() {
    // Contrast: "f(x)" (no space) IS a function call
    let expr = parse_expr("f(x)").expect("Failed to parse");
    match expr {
        Expr::SApp { _fun, args, .. } => {
            assert_eq!(args.len(), 1);
            match &*_fun {
                Expr::SId { id, .. } => match id {
                    Name::SName { s, .. } => assert_eq!(s, "f"),
                    _ => panic!("Expected SName"),
                },
                _ => panic!("Expected SId for fun"),
            }
        }
        _ => panic!("Expected SApp, got {:?}", expr),
    }
}

#[test]
fn test_function_call_then_whitespace_before_paren() {
    // Bug: slow-thankful-krill
    // "f() (x)" should be TWO statements: f() and (x)
    // So parse_expr_complete() should fail
    let result = parse_expr("f() (x)");
    assert!(result.is_err(), "Should fail: 'f() (x)' has leftover tokens after 'f()'");
}

#[test]
fn test_parse_chained_call_with_call_args() {
    // Bug fix: interesting-artistic-shark
    // f()(g()) should parse as: SApp(fun: SApp(f, []), args: [SApp(g, [])])
    let expr = parse_expr("f()(g())").expect("Failed to parse");

    match expr {
        Expr::SApp { _fun, args, .. } => {
            // Outer call should have 1 argument: g()
            assert_eq!(args.len(), 1);

            // The function should be f()
            match *_fun {
                Expr::SApp { _fun: inner_fun, args: inner_args, .. } => {
                    // Inner function should be f
                    match *inner_fun {
                        Expr::SId { id, .. } => match id {
                            Name::SName { s, .. } => assert_eq!(s, "f"),
                            _ => panic!("Expected SName"),
                        },
                        _ => panic!("Expected SId for inner function"),
                    }

                    // f() has no arguments
                    assert_eq!(inner_args.len(), 0);
                }
                _ => panic!("Expected inner SApp"),
            }

            // The argument should be g()
            match &*args[0] {
                Expr::SApp { _fun: arg_fun, args: arg_args, .. } => {
                    // Argument function should be g
                    match &**arg_fun {
                        Expr::SId { id, .. } => match id {
                            Name::SName { s, .. } => assert_eq!(s, "g"),
                            _ => panic!("Expected SName"),
                        },
                        _ => panic!("Expected SId for arg function"),
                    }

                    // g() has no arguments
                    assert_eq!(arg_args.len(), 0);
                }
                _ => panic!("Expected SApp for argument"),
            }
        }
        _ => panic!("Expected outer SApp, got {:?}", expr),
    }
}

// ============================================================================
// Block Expression Tests
// ============================================================================

#[test]
fn test_parse_simple_block() {
    // block: 5 end
    let expr = parse_expr("block: 5 end").expect("Failed to parse");

    match expr {
        Expr::SUserBlock { body, .. } => {
            // Body should be an SBlock
            match *body {
                Expr::SBlock { stmts, .. } => {
                    assert_eq!(stmts.len(), 1);

                    // First statement should be a number
                    match *stmts[0] {
                        Expr::SNum { n, .. } => assert_eq!(n, 5.0),
                        _ => panic!("Expected SNum in block"),
                    }
                }
                _ => panic!("Expected SBlock as body"),
            }
        }
        _ => panic!("Expected SUserBlock, got {:?}", expr),
    }
}

#[test]
fn test_parse_block_multiple_expressions() {
    // block: 1 + 2 3 * 4 end
    let expr = parse_expr("block: 1 + 2 3 * 4 end").expect("Failed to parse");

    match expr {
        Expr::SUserBlock { body, .. } => {
            match *body {
                Expr::SBlock { stmts, .. } => {
                    assert_eq!(stmts.len(), 2);

                    // First statement: 1 + 2
                    match stmts[0].as_ref() {
                        Expr::SOp { op, .. } => assert_eq!(op, "op+"),
                        _ => panic!("Expected SOp for first statement"),
                    }

                    // Second statement: 3 * 4
                    match stmts[1].as_ref() {
                        Expr::SOp { op, .. } => assert_eq!(op, "op*"),
                        _ => panic!("Expected SOp for second statement"),
                    }
                }
                _ => panic!("Expected SBlock as body"),
            }
        }
        _ => panic!("Expected SUserBlock, got {:?}", expr),
    }
}

#[test]
fn test_parse_empty_block() {
    // block: end
    let expr = parse_expr("block: end").expect("Failed to parse");

    match expr {
        Expr::SUserBlock { body, .. } => {
            match *body {
                Expr::SBlock { stmts, .. } => {
                    assert_eq!(stmts.len(), 0);
                }
                _ => panic!("Expected SBlock as body"),
            }
        }
        _ => panic!("Expected SUserBlock, got {:?}", expr),
    }
}

#[test]
fn test_parse_nested_blocks() {
    // block: block: 1 end end
    let expr = parse_expr("block: block: 1 end end").expect("Failed to parse");

    match expr {
        Expr::SUserBlock { body, .. } => {
            match *body {
                Expr::SBlock { stmts, .. } => {
                    assert_eq!(stmts.len(), 1);

                    // Inner block
                    match *stmts[0] {
                        Expr::SUserBlock { body: ref inner_body, .. } => {
                            match **inner_body {
                                Expr::SBlock { stmts: ref inner_stmts, .. } => {
                                    assert_eq!(inner_stmts.len(), 1);

                                    match *inner_stmts[0] {
                                        Expr::SNum { n, .. } => assert_eq!(n, 1.0),
                                        _ => panic!("Expected SNum in inner block"),
                                    }
                                }
                                _ => panic!("Expected inner SBlock"),
                            }
                        }
                        _ => panic!("Expected inner SUserBlock"),
                    }
                }
                _ => panic!("Expected SBlock as body"),
            }
        }
        _ => panic!("Expected SUserBlock, got {:?}", expr),
    }
}

// ============================================================================
// ===== Let Expression Tests =====
// ============================================================================
// NOTE: Let bindings only work in blocks, not as standalone expressions

#[test]
fn test_parse_simple_let() {
    // x = 5
    let expr = parse_expr("x = 5").expect("Failed to parse");

    match expr {
        Expr::SLetExpr { binds, body, blocky, .. } => {
            assert_eq!(binds.len(), 1);
            assert_eq!(blocky, false);

            // Check bind
            match &binds[0] {
                LetBind::SLetBind { b, value, .. } => {
                    match b {
                        Bind::SBind { id, .. } => {
                            match id {
                                Name::SName { s, .. } => assert_eq!(s, "x"),
                                _ => panic!("Expected SName"),
                            }
                        }
                        _ => panic!("Expected SBind"),
                    }

                    // Check value
                    match **value {
                        Expr::SNum { n, .. } => assert_eq!(n, 5.0),
                        _ => panic!("Expected SNum"),
                    }
                }
                _ => panic!("Expected SLetBind"),
            }

            // Check body (should be the value)
            match *body {
                Expr::SNum { n, .. } => assert_eq!(n, 5.0),
                _ => panic!("Expected SNum as body"),
            }
        }
        _ => panic!("Expected SLetExpr, got {:?}", expr),
    }
}

#[test]
fn test_parse_var_binding() {
    // var x := 5
    let expr = parse_expr("var x := 5").expect("Failed to parse");

    match expr {
        Expr::SLetExpr { binds, .. } => {
            assert_eq!(binds.len(), 1);

            match &binds[0] {
                LetBind::SVarBind { b, value, .. } => {
                    match b {
                        Bind::SBind { id, .. } => {
                            match id {
                                Name::SName { s, .. } => assert_eq!(s, "x"),
                                _ => panic!("Expected SName"),
                            }
                        }
                        _ => panic!("Expected SBind"),
                    }

                    match **value {
                        Expr::SNum { n, .. } => assert_eq!(n, 5.0),
                        _ => panic!("Expected SNum"),
                    }
                }
                _ => panic!("Expected SVarBind"),
            }
        }
        _ => panic!("Expected SLetExpr with SVarBind"),
    }
}

#[test]
fn test_parse_let_with_expression() {
    // y = 2 + 3
    let expr = parse_expr("y = 2 + 3").expect("Failed to parse");

    match expr {
        Expr::SLetExpr { binds, .. } => {
            assert_eq!(binds.len(), 1);

            match &binds[0] {
                LetBind::SLetBind { b, value, .. } => {
                    match b {
                        Bind::SBind { id, .. } => {
                            match id {
                                Name::SName { s, .. } => assert_eq!(s, "y"),
                                _ => panic!("Expected SName"),
                            }
                        }
                        _ => panic!("Expected SBind"),
                    }

                    // Value should be SOp (2 + 3)
                    match **value {
                        Expr::SOp { ref op, .. } => assert_eq!(op, "op+"),
                        _ => panic!("Expected SOp"),
                    }
                }
                _ => panic!("Expected SLetBind"),
            }
        }
        _ => panic!("Expected SLetExpr"),
    }
}

// ============================================================================
// ===== For Expression Tests =====
// ============================================================================

#[test]
fn test_parse_simple_for() {
    // for map(x from lst): x + 1 end
    let expr = parse_expr("for map(x from lst): x + 1 end").expect("Failed to parse");

    match expr {
        Expr::SFor { iterator, bindings, body, blocky, .. } => {
            assert_eq!(blocky, false);

            // Check iterator is SId(map)
            match *iterator {
                Expr::SId { ref id, .. } => {
                    match id {
                        Name::SName { s, .. } => assert_eq!(s, "map"),
                        _ => panic!("Expected SName for iterator"),
                    }
                }
                _ => panic!("Expected SId for iterator"),
            }

            // Check bindings
            assert_eq!(bindings.len(), 1);
            match &bindings[0].bind {
                Bind::SBind { id, .. } => {
                    match id {
                        Name::SName { s, .. } => assert_eq!(s, "x"),
                        _ => panic!("Expected SName for binding"),
                    }
                }
                _ => panic!("Expected SBind"),
            }

            // Check body is SBlock with one SOp statement
            match *body {
                Expr::SBlock { ref stmts, .. } => {
                    assert_eq!(stmts.len(), 1);
                    match stmts[0].as_ref() {
                        Expr::SOp { op, .. } => assert_eq!(op, "op+"),
                        _ => panic!("Expected SOp in body"),
                    }
                }
                _ => panic!("Expected SBlock for body"),
            }
        }
        _ => panic!("Expected SFor, got {:?}", expr),
    }
}

#[test]
fn test_parse_for_with_dot_access() {
    // for lists.map2(a1 from arr1, a2 from arr2): a1 + a2 end
    let expr = parse_expr("for lists.map2(a1 from arr1, a2 from arr2): a1 + a2 end").expect("Failed to parse");

    match expr {
        Expr::SFor { iterator, bindings, .. } => {
            // Check iterator is SDot(lists, map2)
            match *iterator {
                Expr::SDot { ref obj, ref field, .. } => {
                    match **obj {
                        Expr::SId { ref id, .. } => {
                            match id {
                                Name::SName { s, .. } => assert_eq!(s, "lists"),
                                _ => panic!("Expected SName for object"),
                            }
                        }
                        _ => panic!("Expected SId for object"),
                    }
                    assert_eq!(field, "map2");
                }
                _ => panic!("Expected SDot for iterator"),
            }

            // Check bindings
            assert_eq!(bindings.len(), 2);
        }
        _ => panic!("Expected SFor"),
    }
}

#[test]
fn test_parse_provide_all() {
    // Test: provide *
    let program = parse_program("provide *").expect("Failed to parse");

    match program._provide {
        Provide::SProvideAll { .. } => {
            // Success!
        }
        _ => panic!("Expected SProvideAll, got {:?}", program._provide),
    }
}

#[test]
fn test_parse_provide_block() {
    // Test: provide: x end
    let program = parse_program("provide: x end").expect("Failed to parse");

    match program._provide {
        Provide::SProvide { block, .. } => {
            match *block {
                Expr::SBlock { ref stmts, .. } => {
                    assert_eq!(stmts.len(), 1);
                    match stmts[0].as_ref() {
                        Expr::SId { id, .. } => {
                            match id {
                                Name::SName { s, .. } => assert_eq!(s, "x"),
                                _ => panic!("Expected SName"),
                            }
                        }
                        _ => panic!("Expected SId in provide block"),
                    }
                }
                _ => panic!("Expected SBlock for provide block"),
            }
        }
        _ => panic!("Expected SProvide, got {:?}", program._provide),
    }
}
