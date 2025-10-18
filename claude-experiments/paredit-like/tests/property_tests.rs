use proptest::prelude::*;
use paredit_like::parinfer::Parinfer;
use paredit_like::parser::ClojureParser;

/// Test that parinfer always produces balanced output
#[test]
fn test_parinfer_always_balances() {
    proptest!(|(input in "\\PC{0,200}")| {
        let parinfer = Parinfer::new(&input);
        if let Ok(result) = parinfer.balance() {
            // Count opening and closing parens
            let open_parens = result.chars().filter(|&c| c == '(').count();
            let close_parens = result.chars().filter(|&c| c == ')').count();
            let open_brackets = result.chars().filter(|&c| c == '[').count();
            let close_brackets = result.chars().filter(|&c| c == ']').count();
            let open_braces = result.chars().filter(|&c| c == '{').count();
            let close_braces = result.chars().filter(|&c| c == '}').count();
            
            prop_assert_eq!(open_parens, close_parens);
            prop_assert_eq!(open_brackets, close_brackets);
            prop_assert_eq!(open_braces, close_braces);
        }
    });
}

/// Test that already balanced input is preserved
#[test]
fn test_parinfer_preserves_balanced() {
    proptest!(|(
        depth in 1..10usize,
        content in prop::collection::vec("\\w+", 0..5)
    )| {
        let mut input = String::new();
        
        // Create nested balanced structure
        for _ in 0..depth {
            input.push('(');
        }
        
        for (i, word) in content.iter().enumerate() {
            if i > 0 {
                input.push(' ');
            }
            input.push_str(word);
        }
        
        for _ in 0..depth {
            input.push(')');
        }
        
        let parinfer = Parinfer::new(&input);
        if let Ok(result) = parinfer.balance() {
            prop_assert_eq!(result, input);
        }
    });
}

/// Test that parser doesn't crash on arbitrary input
#[test]
fn test_parser_robustness() {
    proptest!(|(input in "\\PC{0,500}")| {
        let mut parser = ClojureParser::new().unwrap();
        // Should not panic, even on malformed input
        let _ = parser.parse_to_sexpr(&input);
    });
}

/// Generate valid Clojure-like s-expressions
fn arb_sexpr() -> impl Strategy<Value = String> {
    let leaf = prop_oneof![
        "\\w+",
        ":\\w+",
        "[0-9]+",
        "\"[^\"]*\"",
        "true|false|nil"
    ];
    
    leaf.prop_recursive(
        3, // max depth
        10, // max size
        3, // items per collection
        |inner| {
            prop_oneof![
                prop::collection::vec(inner.clone(), 0..5)
                    .prop_map(|items| format!("({})", items.join(" "))),
                prop::collection::vec(inner.clone(), 0..5)
                    .prop_map(|items| format!("[{}]", items.join(" "))),
                prop::collection::vec(inner, 0..5)
                    .prop_map(|items| format!("{{{}}}", items.join(" ")))
            ]
        }
    )
}

/// Test that valid s-expressions parse successfully
#[test]
fn test_parser_on_valid_input() {
    proptest!(|(input in arb_sexpr())| {
        let mut parser = ClojureParser::new().unwrap();
        let result = parser.parse_to_sexpr(&input);
        prop_assert!(result.is_ok());
    });
}

/// Test parinfer with valid s-expressions
#[test]
fn test_parinfer_on_valid_input() {
    proptest!(|(input in arb_sexpr())| {
        let parinfer = Parinfer::new(&input);
        let result = parinfer.balance();
        prop_assert!(result.is_ok());
        
        if let Ok(output) = result {
            // Should remain valid after balancing
            let mut parser = ClojureParser::new().unwrap();
            prop_assert!(parser.parse_to_sexpr(&output).is_ok());
        }
    });
}

/// Test that strings with parentheses inside are preserved
#[test]
fn test_strings_preserved() {
    proptest!(|(content in "[^\"\\\\]*")| {
        let input = format!("\"{}\"", content);
        let parinfer = Parinfer::new(&input);
        if let Ok(result) = parinfer.balance() {
            prop_assert_eq!(result, input);
        }
    });
}

/// Test that comments with parentheses are preserved
#[test]
fn test_comments_preserved() {
    proptest!(|(content in "[^\\n]*")| {
        let input = format!("; {}", content);
        let parinfer = Parinfer::new(&input);
        if let Ok(result) = parinfer.balance() {
            prop_assert_eq!(result, input);
        }
    });
}

/// Test that parinfer is idempotent (running twice gives same result)
#[test]
fn test_parinfer_idempotent() {
    proptest!(|(input in "\\PC{0,100}")| {
        let parinfer1 = Parinfer::new(&input);
        if let Ok(result1) = parinfer1.balance() {
            let parinfer2 = Parinfer::new(&result1);
            if let Ok(result2) = parinfer2.balance() {
                prop_assert_eq!(result1, result2);
            }
        }
    });
}

/// Test character encoding preservation
#[test]
fn test_unicode_preservation() {
    proptest!(|(input in "[\u{0080}-\u{FFFF}]{0,50}")| {
        if !input.is_empty() {
            let wrapped = format!("\"{}\"", input);
            let parinfer = Parinfer::new(&wrapped);
            if let Ok(result) = parinfer.balance() {
                prop_assert!(result.contains(&input));
            }
        }
    });
}

/// Generate unbalanced parentheses patterns
fn arb_unbalanced() -> impl Strategy<Value = String> {
    prop_oneof![
        // Missing closing parens
        "\\([^)]*",
        "\\[[^\\]]*",
        "\\{[^}]*",
        // Extra closing parens
        "[^(]*\\)",
        "[^\\[]*\\]",
        "[^{]*\\}",
        // Mismatched parens
        "\\([^)]*\\]",
        "\\[[^\\]]*\\)",
        "\\{[^}]*\\)",
    ]
}

/// Test that unbalanced input gets fixed
#[test]
fn test_unbalanced_gets_fixed() {
    proptest!(|(input in arb_unbalanced())| {
        let parinfer = Parinfer::new(&input);
        if let Ok(result) = parinfer.balance() {
            // Result should be balanced
            let open_parens = result.chars().filter(|&c| c == '(').count();
            let close_parens = result.chars().filter(|&c| c == ')').count();
            let open_brackets = result.chars().filter(|&c| c == '[').count();
            let close_brackets = result.chars().filter(|&c| c == ']').count();
            let open_braces = result.chars().filter(|&c| c == '{').count();
            let close_braces = result.chars().filter(|&c| c == '}').count();
            
            prop_assert_eq!(open_parens, close_parens);
            prop_assert_eq!(open_brackets, close_brackets);
            prop_assert_eq!(open_braces, close_braces);
        }
    });
}

/// Test that whitespace is handled correctly
#[test]
fn test_whitespace_handling() {
    proptest!(|(
        spaces_before in 0..10usize,
        spaces_after in 0..10usize,
        spaces_inside in 0..5usize
    )| {
        let mut input = String::new();
        
        // Add leading whitespace
        for _ in 0..spaces_before {
            input.push(' ');
        }
        
        input.push('(');
        
        // Add internal whitespace
        for _ in 0..spaces_inside {
            input.push(' ');
        }
        
        input.push_str("foo");
        
        for _ in 0..spaces_inside {
            input.push(' ');
        }
        
        input.push(')');
        
        // Add trailing whitespace
        for _ in 0..spaces_after {
            input.push(' ');
        }
        
        let parinfer = Parinfer::new(&input);
        if let Ok(result) = parinfer.balance() {
            prop_assert!(result.contains("foo"));
            prop_assert_eq!(result.chars().filter(|&c| c == '(').count(), 1);
            prop_assert_eq!(result.chars().filter(|&c| c == ')').count(), 1);
        }
    });
}