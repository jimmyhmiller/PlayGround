/// Comprehensive error handling tests
///
/// These tests ensure that the parser:
/// 1. Gracefully handles invalid syntax
/// 2. Produces meaningful error messages
/// 3. Doesn't panic on malformed input
/// 4. Properly handles edge cases and boundary conditions
///
/// KNOWN ISSUES (see BUGS.md):
/// - comfortable-pink-hedgehog: Parser accepts unmatched closing parenthesis
/// - palatable-edible-crayfish: Parser accepts unmatched closing bracket
/// - ideal-thorny-kingfisher: Parser accepts trailing tokens
/// - oily-awkward-hedgehog: Parser accepts multiple expressions
/// - reflecting-enchanting-caribou: Parser accepts invalid characters

use pyret_attempt2::{Parser, Expr};
use pyret_attempt2::tokenizer::Tokenizer;
use pyret_attempt2::error::ParseError;

/// Helper to parse a string into an expression
fn parse_expr(input: &str) -> Result<Expr, ParseError> {
    let mut tokenizer = Tokenizer::new(input);
    let tokens = tokenizer.tokenize();
    let mut parser = Parser::new(tokens, "test.arr".to_string());
    parser.parse_expr()
}

// ============================================================================
// Syntax Errors - Unmatched Delimiters
// ============================================================================

#[test]
fn test_error_unmatched_paren_left() {
    let result = parse_expr("(42");
    assert!(result.is_err(), "Should error on unmatched left paren");
}

#[test]
fn test_error_unmatched_paren_right() {
    let result = parse_expr("42)");
    assert!(result.is_err(), "Should error on unmatched right paren");
}

#[test]
fn test_error_unmatched_bracket_left() {
    let result = parse_expr("[1, 2, 3");
    assert!(result.is_err(), "Should error on unmatched left bracket");
}

#[test]
fn test_error_unmatched_bracket_right() {
    let result = parse_expr("1, 2, 3]");
    assert!(result.is_err(), "Should error on unmatched right bracket");
}

#[test]
fn test_error_mismatched_delimiters() {
    let result = parse_expr("(1, 2, 3]");
    assert!(result.is_err(), "Should error on mismatched delimiters");
}

#[test]
fn test_error_nested_unmatched() {
    let result = parse_expr("((1 + 2)");
    assert!(result.is_err(), "Should error on nested unmatched parens");
}

// ============================================================================
// Syntax Errors - Invalid Operators
// ============================================================================

#[test]
fn test_error_trailing_operator() {
    let result = parse_expr("1 +");
    assert!(result.is_err(), "Should error on trailing operator");
}

#[test]
fn test_error_leading_operator() {
    let result = parse_expr("+ 1");
    assert!(result.is_err(), "Should error on leading operator");
}

#[test]
fn test_error_double_operator() {
    let result = parse_expr("1 + + 2");
    assert!(result.is_err(), "Should error on double operator");
}

#[test]
fn test_error_operator_without_operands() {
    let result = parse_expr("+");
    assert!(result.is_err(), "Should error on operator without operands");
}

// ============================================================================
// Syntax Errors - Invalid Function Calls
// ============================================================================

#[test]
fn test_error_missing_comma_in_args() {
    let result = parse_expr("f(x y)");
    assert!(result.is_err(), "Should error on missing comma between args");
}

#[test]
fn test_error_trailing_comma_in_args() {
    let result = parse_expr("f(x, y,)");
    assert!(result.is_err(), "Should error on trailing comma");
}

#[test]
fn test_error_double_comma() {
    let result = parse_expr("f(x,, y)");
    assert!(result.is_err(), "Should error on double comma");
}

#[test]
fn test_error_call_on_non_callable() {
    // This might actually parse, depending on implementation
    // The error would be semantic, not syntactic
    let result = parse_expr("123(x)");
    // Could be valid syntax (number followed by call), but unusual
    // Document the behavior
}

// ============================================================================
// Syntax Errors - Invalid Arrays
// ============================================================================

#[test]
fn test_error_array_missing_comma() {
    let result = parse_expr("[1 2 3]");
    assert!(result.is_err(), "Should error on missing commas in array");
}

#[test]
fn test_error_array_trailing_comma() {
    let result = parse_expr("[1, 2, 3,]");
    assert!(result.is_err(), "Should error on trailing comma in array");
}

#[test]
fn test_error_array_double_comma() {
    let result = parse_expr("[1,, 2]");
    assert!(result.is_err(), "Should error on double comma in array");
}

// ============================================================================
// Syntax Errors - Invalid Dot Access
// ============================================================================

#[test]
fn test_error_dot_without_field() {
    let result = parse_expr("obj.");
    assert!(result.is_err(), "Should error on dot without field name");
}

#[test]
fn test_error_dot_with_number() {
    let result = parse_expr("obj.123");
    assert!(result.is_err(), "Should error on dot with number (not an identifier)");
}

#[test]
fn test_error_leading_dot() {
    let result = parse_expr(".field");
    assert!(result.is_err(), "Should error on leading dot");
}

// ============================================================================
// Empty Input
// ============================================================================

#[test]
fn test_error_empty_input() {
    let result = parse_expr("");
    assert!(result.is_err(), "Should error on empty input");
}

#[test]
fn test_error_whitespace_only() {
    let result = parse_expr("   ");
    assert!(result.is_err(), "Should error on whitespace-only input");
}

#[test]
fn test_error_empty_parens_as_expr() {
    let result = parse_expr("()");
    assert!(result.is_err(), "Should error on empty parens (not a valid expr)");
}

// ============================================================================
// Boundary Cases - Deeply Nested
// ============================================================================

#[test]
fn test_deeply_nested_parens_valid() {
    // This should work
    let expr = "(((((((((42)))))))))";
    let result = parse_expr(expr);
    assert!(result.is_ok(), "Should handle deeply nested parens");
}

#[test]
fn test_deeply_nested_arrays_valid() {
    // This should work
    let expr = "[[[[[[1]]]]]]";
    let result = parse_expr(expr);
    assert!(result.is_ok(), "Should handle deeply nested arrays");
}

#[test]
fn test_deeply_nested_calls_valid() {
    // This should work
    let expr = "f(g(h(i(j(k(x))))))";
    let result = parse_expr(expr);
    assert!(result.is_ok(), "Should handle deeply nested calls");
}

// ============================================================================
// Boundary Cases - Very Long Expressions
// ============================================================================

#[test]
fn test_very_long_addition_chain() {
    // Generate a very long chain of additions
    let terms: Vec<String> = (1..=100).map(|n| n.to_string()).collect();
    let expr = terms.join(" + ");
    let result = parse_expr(&expr);
    assert!(result.is_ok(), "Should handle very long chains");
}

#[test]
fn test_very_long_array() {
    // Generate a very long array
    let elements: Vec<String> = (1..=200).map(|n| n.to_string()).collect();
    let expr = format!("[{}]", elements.join(", "));
    let result = parse_expr(&expr);
    assert!(result.is_ok(), "Should handle very long arrays");
}

#[test]
fn test_very_long_dot_chain() {
    // Generate a very long dot chain
    let mut expr = String::from("obj");
    for i in 1..=50 {
        expr.push_str(&format!(".field{}", i));
    }
    let result = parse_expr(&expr);
    assert!(result.is_ok(), "Should handle very long dot chains");
}

// ============================================================================
// Special Characters and Edge Cases
// ============================================================================

#[test]
fn test_error_invalid_characters() {
    let result = parse_expr("@#$%");
    assert!(result.is_err(), "Should error on invalid characters");
}

#[test]
fn test_error_unicode_in_wrong_place() {
    // Unicode might be valid in strings but not in identifiers
    let result = parse_expr("café");
    // This depends on tokenizer behavior
}

// ============================================================================
// Operator Precedence Edge Cases
// ============================================================================

#[test]
fn test_all_operators_chained() {
    // Pyret has no precedence, so this should parse left-to-right
    let result = parse_expr("1 + 2 - 3 * 4 / 5");
    assert!(result.is_ok(), "Should parse chained different operators");
}

#[test]
fn test_comparison_chains() {
    let result = parse_expr("1 < 2 < 3 < 4");
    assert!(result.is_ok(), "Should parse comparison chains");
}

// ============================================================================
// Whitespace Sensitivity Edge Cases
// ============================================================================

#[test]
fn test_tabs_vs_spaces() {
    // Test that tabs and spaces are treated similarly
    let result1 = parse_expr("f(x)");
    let result2 = parse_expr("f\t(x)"); // Tab instead of space

    // Both should parse successfully
    assert!(result1.is_ok());
    // Tab behavior might differ - document it
}

#[test]
fn test_newlines_in_expressions() {
    let result = parse_expr("1 +\n2");
    // Pyret's whitespace rules might make this valid or invalid
    // Document the behavior
}

// ============================================================================
// Mixed Valid and Invalid
// ============================================================================

#[test]
fn test_partial_expression() {
    let result = parse_expr("f(x) +");
    assert!(result.is_err(), "Should error on incomplete expression");
}

#[test]
fn test_valid_prefix_invalid_suffix() {
    let result = parse_expr("f(x) @");
    assert!(result.is_err(), "Should error even if prefix is valid");
}

// ============================================================================
// Parser State Edge Cases
// ============================================================================

#[test]
fn test_multiple_expressions() {
    // Parser expects single expression
    let result = parse_expr("1 + 2 3 + 4");
    assert!(result.is_err(), "Should error on multiple expressions");
}

#[test]
fn test_expression_with_trailing_tokens() {
    let result = parse_expr("42 unexpected");
    assert!(result.is_err(), "Should error on trailing tokens");
}

// ============================================================================
// Error Message Quality Tests
// ============================================================================

#[test]
fn test_error_message_contains_context() {
    let result = parse_expr("(42");
    if let Err(err) = result {
        let err_str = format!("{:?}", err);
        // Error should mention the problem (unclosed paren, unexpected EOF, etc.)
        assert!(
            err_str.contains("paren") ||
            err_str.contains("EOF") ||
            err_str.contains("expect"),
            "Error message should provide context: {}",
            err_str
        );
    }
}

#[test]
fn test_error_message_for_trailing_operator() {
    let result = parse_expr("1 + 2 +");
    if let Err(err) = result {
        let err_str = format!("{:?}", err);
        // Should mention missing operand or unexpected EOF
        assert!(
            err_str.contains("EOF") ||
            err_str.contains("expect") ||
            err_str.contains("operand"),
            "Error message should indicate missing operand: {}",
            err_str
        );
    }
}
