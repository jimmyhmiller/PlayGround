use pyret_attempt2::{Parser, Expr, Name, ConstructModifier};
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
    // Test that decimals are converted to rationals (Pyret behavior)
    let expr = parse_expr("3.14").expect("Failed to parse");

    match expr {
        Expr::SFrac { num, den, .. } => {
            // 3.14 = 314/100 = 157/50 (simplified)
            assert_eq!(num, 157);
            assert_eq!(den, 50);
        }
        _ => panic!("Expected SFrac for decimal, got {:?}", expr),
    }
}

#[test]
fn test_parse_simple_decimal() {
    // Test 2.5 = 5/2
    let expr = parse_expr("2.5").expect("Failed to parse");

    match expr {
        Expr::SFrac { num, den, .. } => {
            assert_eq!(num, 5);
            assert_eq!(den, 2);
        }
        _ => panic!("Expected SFrac, got {:?}", expr),
    }
}

#[test]
fn test_parse_explicit_rational() {
    // Test explicit rational like "3/4"
    let expr = parse_expr("3/4").expect("Failed to parse");

    match expr {
        Expr::SFrac { num, den, .. } => {
            assert_eq!(num, 3);
            assert_eq!(den, 4);
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
            assert_eq!(num, 3);
            assert_eq!(den, 4);
        }
        _ => panic!("Expected SFrac, got {:?}", expr),
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
                        (Expr::SNum { n: n1, .. }, Expr::SNum { n: n2, .. }) => {
                            assert_eq!(n1, 1.0);
                            assert_eq!(n2, 2.0);
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
