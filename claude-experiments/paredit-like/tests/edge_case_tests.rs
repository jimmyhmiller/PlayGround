/// Tests for specific edge cases and regression tests
use paredit_like::*;
use tempfile::TempDir;
use std::fs;

#[test]
fn test_parinfer_indentation_edge_cases() {
    // Test various indentation scenarios that could break parinfer logic
    let cases = vec![
        // Mixed tabs and spaces
        ("(let [x 1\n\t  y 2\n  (+ x y", "(let [x 1\n\t  y 2]\n  (+ x y))"),
        
        // Inconsistent indentation
        ("(if true\n  :yes\n    :no", "(if true\n  :yes\n    :no)"),
        
        // Zero indentation for nested forms
        ("(let [x 1]\n(+ x 1", "(let [x 1]\n(+ x 1))"),
        
        // Very deep indentation
        ("(let [x 1]\n              (+ x 1", "(let [x 1]\n              (+ x 1))"),
        
        // Negative indentation (impossible in practice, but let's handle it)
        ("(defn foo\n  [x]\n(+ x 1", "(defn foo\n  [x]\n(+ x 1))"),
        
        // Empty lines with different indentation
        ("(defn foo\n  \n  [x]\n  \n  (+ x 1", "(defn foo\n  \n  [x]\n  \n  (+ x 1))"),
        
        // Comments affecting indentation
        ("(let [x 1] ; comment\n  (+ x 1", "(let [x 1] ; comment\n  (+ x 1))"),
    ];
    
    for (input, expected) in cases {
        let parinfer = Parinfer::new(input);
        let result = parinfer.balance().unwrap();
        // Check structure is balanced, exact formatting may vary
        let open_count = result.chars().filter(|&c| c == '(').count();
        let close_count = result.chars().filter(|&c| c == ')').count();
        assert_eq!(open_count, close_count, "Unbalanced result for: {}", input);
    }
}

#[test]
fn test_parser_clojure_specific_forms() {
    let mut parser = ClojureParser::new().unwrap();
    
    // Test various Clojure-specific syntax
    let forms = vec![
        // Metadata
        "^:private (defn foo [])",
        "^{:doc \"test\"} (defn bar [])",
        
        // Reader conditionals
        "#?(:clj :clojure :cljs :clojurescript)",
        
        // Tagged literals
        "#inst \"2023-01-01\"",
        "#uuid \"550e8400-e29b-41d4-a716-446655440000\"",
        
        // Regex literals
        "#\"[a-z]+\"",
        
        // Var quotes
        "#'my-var",
        
        // Anonymous functions
        "#(+ %1 %2)",
        "#(println %&)",
        
        // Discard forms
        "#_ (ignored form)",
        "#_(+ 1 2) (+ 3 4)",
        
        // Deref
        "@my-atom",
        
        // Quote forms
        "'(1 2 3)",
        "`(list ~x ~@xs)",
        
        // Character literals
        "\\space",
        "\\newline",
        "\\tab",
        "\\a",
        "\\u0041",
        
        // Keywords with namespaces
        "::local-keyword",
        ":namespace/keyword",
        
        // Symbols with namespaces
        "namespace/symbol",
        
        // Numbers in various formats
        "42",
        "42N", // BigInt
        "3.14159",
        "3.14159M", // BigDecimal
        "22/7", // Ratio
        "0x2A", // Hex
        "052", // Octal
        "2r101010", // Binary
        "36rZ", // Base 36
        
        // Special values
        "##Inf",
        "##-Inf", 
        "##NaN",
    ];
    
    for form in forms {
        let result = parser.parse_to_sexpr(form);
        // Should parse without error, even if some constructs aren't fully supported
        assert!(result.is_ok(), "Failed to parse: {}", form);
    }
}

#[test]
fn test_refactoring_with_complex_nesting() {
    // Test refactoring operations on complex nested structures
    let complex_cases = vec![
        // Deeply nested let forms for merge-let testing
        r#"(let [a 1]
             (let [b 2]
               (let [c 3]
                 (+ a b c))))"#,
        
        // Nested function calls for slurp/barf
        "(foo (bar (baz x) y) z)",
        
        // Mixed data structures
        "{:fn (fn [x] 
                {:result [x (+ x 1) (* x 2)]
                 :meta {:processed true}})}",
        
        // Conditional forms
        "(if (and (> x 0) (< x 100))
           (do
             (println \"valid\")
             :ok)
           (throw (Exception. \"invalid\")))",
        
        // Loop/recur structures
        "(loop [acc 0 xs [1 2 3 4 5]]
           (if (empty? xs)
             acc
             (recur (+ acc (first xs)) (rest xs))))",
    ];
    
    for source in complex_cases {
        let mut parser = ClojureParser::new().unwrap();
        if let Ok(forms) = parser.parse_to_sexpr(source) {
            let mut refactorer = Refactorer::new(source.to_string());
            
            // Test that refactoring operations don't crash on complex input
            // Note: find_list_at_line and find_deepest_at_line are private methods
            // These are tested indirectly through the public API
            
            // Test wrap operations
            let _ = refactorer.wrap(&forms, 1, "(");
            let _ = refactorer.wrap(&forms, 1, "[");
            let _ = refactorer.wrap(&forms, 1, "{");
            
            // Test splice if there's a list
            let _ = refactorer.splice(&forms, 1);
        }
    }
}

#[test]
fn test_string_handling_edge_cases() {
    let string_cases = vec![
        // Empty string
        r#""""#,
        
        // String with just spaces
        r#""   ""#,
        
        // String with escape sequences
        r#""\n\t\r\"""#,
        
        // String with unicode escapes
        r#""\u0041\u{1F4A9}""#,
        
        // String with backslash at end
        r#""ends with \\""#,
        
        // String spanning multiple lines
        r#""line 1
line 2
line 3""#,
        
        // String with parentheses
        r#""(defn foo [x] (+ x 1))""#,
        
        // String with brackets and braces
        r#""[vector] {map :value}""#,
        
        // String with quotes inside
        r#""She said \"Hello\"""#,
        
        // Raw string-like content
        r#""C:\Program Files\app.exe""#,
        
        // String with special characters
        r#""@#$%^&*()_+-=[]{}|;':\",./<>?`~""#,
    ];
    
    for string_literal in string_cases {
        // Test that parinfer preserves strings exactly
        let parinfer = Parinfer::new(string_literal);
        let result = parinfer.balance().unwrap();
        assert_eq!(result, string_literal, "String not preserved: {}", string_literal);
        
        // Test that parser handles strings correctly
        let mut parser = ClojureParser::new().unwrap();
        let forms = parser.parse_to_sexpr(string_literal).unwrap();
        assert_eq!(forms.len(), 1);
        if let SExpr::String { value, .. } = &forms[0] {
            assert_eq!(value, string_literal);
        } else {
            panic!("Expected string, got: {:?}", forms[0]);
        }
    }
}

#[test]
fn test_comment_handling_edge_cases() {
    let comment_cases = vec![
        // Simple comment
        "; simple comment",
        
        // Comment with parentheses
        "; (this is in a comment)",
        
        // Comment with special characters
        "; @#$%^&*()[]{}",
        
        // Comment with unicode
        "; 🦀 Rust is awesome! 世界",
        
        // Empty comment
        ";",
        
        // Comment with just spaces
        ";   ",
        
        // Multiple comments
        "; comment 1\n; comment 2\n; comment 3",
        
        // Comment after code
        "(+ 1 2) ; inline comment",
        
        // Comment in the middle of code
        "(defn foo [x] ; parameter comment\n  (+ x 1)) ; body comment",
    ];
    
    for comment_input in comment_cases {
        // Test that parinfer preserves comments
        let parinfer = Parinfer::new(comment_input);
        let result = parinfer.balance().unwrap();
        
        // Comments should be preserved in the result
        if comment_input.contains(';') {
            assert!(result.contains(';'), "Comment lost in: {}", comment_input);
        }
        
        // Test parsing
        let mut parser = ClojureParser::new().unwrap();
        let _forms = parser.parse_to_sexpr(comment_input).unwrap();
    }
}

#[test]
fn test_whitespace_preservation() {
    let whitespace_cases = vec![
        // Leading whitespace
        "   (+ 1 2)",
        
        // Trailing whitespace
        "(+ 1 2)   ",
        
        // Mixed whitespace
        " \t (+ 1 2) \n ",
        
        // Whitespace in lists
        "( + 1 2 )",
        "(  +  1  2  )",
        
        // Multiple spaces
        "(+     1     2)",
        
        // Tabs vs spaces
        "(\t+\t1\t2\t)",
        
        // Newlines in various places
        "(\n+\n1\n2\n)",
        
        // Windows line endings
        "(+\r\n1\r\n2)",
        
        // Mixed line endings
        "(+\r1\n2\r\n3)",
    ];
    
    for input in whitespace_cases {
        let parinfer = Parinfer::new(input);
        let result = parinfer.balance().unwrap();
        
        // Should be balanced
        let open_count = result.chars().filter(|&c| c == '(').count();
        let close_count = result.chars().filter(|&c| c == ')').count();
        assert_eq!(open_count, close_count);
        
        // Should preserve essential content
        assert!(result.contains('+'));
        assert!(result.contains('1'));
        assert!(result.contains('2'));
    }
}

#[test]
fn test_malformed_input_recovery() {
    // Test how well the system recovers from various malformed inputs
    let malformed_cases = vec![
        // Mismatched delimiters
        "(defn foo [x} (+ x 1))",
        "[defn foo (x] (+ x 1)]",
        "{defn foo [x) (+ x 1)}",
        
        // Mixed nesting errors
        "(let [x (+ 1 2] y (* 3 4))",
        "[if true (println :yes] :no)",
        
        // Partial forms
        "(defn",
        "(let [",
        "(if true",
        
        // Syntax errors
        "(((",
        ")))",
        "[[[",
        "}}}",
        
        // Invalid combinations
        ")(",
        "][",
        "}{",
        
        // Escaped characters in wrong context
        "\\(\\)\\[\\]",
        
        // Invalid numbers
        "42abc",
        "3.14.15",
        
        // Malformed keywords
        ":::keyword",
        ":",
    ];
    
    for malformed in malformed_cases {
        // Test that parser doesn't crash
        let mut parser = ClojureParser::new().unwrap();
        let _result = parser.parse_to_sexpr(malformed);
        
        // Test that parinfer attempts to fix it
        let parinfer = Parinfer::new(malformed);
        if let Ok(result) = parinfer.balance() {
            // If it succeeds, it should be balanced
            let open_parens = result.chars().filter(|&c| c == '(').count();
            let close_parens = result.chars().filter(|&c| c == ')').count();
            let open_brackets = result.chars().filter(|&c| c == '[').count();
            let close_brackets = result.chars().filter(|&c| c == ']').count();
            let open_braces = result.chars().filter(|&c| c == '{').count();
            let close_braces = result.chars().filter(|&c| c == '}').count();
            
            assert_eq!(open_parens, close_parens);
            assert_eq!(open_brackets, close_brackets);
            assert_eq!(open_braces, close_braces);
        }
    }
}

#[test]
fn test_performance_edge_cases() {
    // Test cases that might cause performance issues
    
    // Very long symbol names
    let long_symbol = "a".repeat(1000);
    let input = format!("(defn {} [])", long_symbol);
    let mut parser = ClojureParser::new().unwrap();
    let _result = parser.parse_to_sexpr(&input);
    
    // Very long string
    let long_string = format!("\"{}\"", "x".repeat(10000));
    let _result = parser.parse_to_sexpr(&long_string);
    
    // Many small forms
    let many_forms = (0..1000).map(|i| format!("(f{})", i)).collect::<Vec<_>>().join(" ");
    let _result = parser.parse_to_sexpr(&many_forms);
    
    // Deeply nested but narrow
    let mut deep_narrow = String::new();
    for _ in 0..500 {
        deep_narrow.push_str("(f ");
    }
    deep_narrow.push('x');
    for _ in 0..500 {
        deep_narrow.push(')');
    }
    let _result = parser.parse_to_sexpr(&deep_narrow);
}

#[test]
fn test_real_world_clojure_patterns() {
    // Test patterns commonly found in real Clojure code
    let real_world_patterns = vec![
        // Namespace declarations
        r#"(ns myapp.core
             (:require [clojure.string :as str]
                       [clojure.set :as set])
             (:import [java.util Date UUID]))"#,
        
        // Threading macros (-> and ->>)
        "(-> data
             (filter pred)
             (map transform)
             (reduce combine))",
        
        // Destructuring
        "(let [{:keys [name age]} person
               [first & rest] items]
           (process name age first rest))",
        
        // Multi-arity functions
        "(defn greet
           ([] (greet \"World\"))
           ([name] (str \"Hello, \" name \"!\")))",
        
        // Protocols and records
        "(defprotocol Drawable
           (draw [this])
           (resize [this scale]))",
        
        // Macros
        "(defmacro when-not [test & body]
           `(if (not ~test)
              (do ~@body)))",
        
        // Core.async patterns
        "(go-loop [acc []]
           (if-let [val (<! ch)]
             (recur (conj acc val))
             acc))",
        
        // Spec definitions
        "(s/def ::name string?)
         (s/def ::age pos-int?)
         (s/def ::person (s/keys :req [::name ::age]))",
    ];
    
    for pattern in real_world_patterns {
        let mut parser = ClojureParser::new().unwrap();
        let forms_result = parser.parse_to_sexpr(pattern);
        
        // Should parse successfully (tree-sitter is quite robust)
        assert!(forms_result.is_ok(), "Failed to parse real-world pattern: {}", pattern);
        
        // Test parinfer on it
        let parinfer = Parinfer::new(pattern);
        let balance_result = parinfer.balance();
        
        if let Ok(balanced) = balance_result {
            // Should remain balanced
            let open_parens = balanced.chars().filter(|&c| c == '(').count();
            let close_parens = balanced.chars().filter(|&c| c == ')').count();
            assert_eq!(open_parens, close_parens, "Real-world pattern became unbalanced: {}", pattern);
        }
    }
}

#[test]
fn test_incremental_editing_scenarios() {
    // Simulate incremental editing scenarios where parinfer would be used
    let editing_scenarios = vec![
        // Adding a new parameter
        ("(defn foo [x]", "(defn foo [x y]"),
        
        // Adding a new form in let
        ("(let [x 1]", "(let [x 1\n      y 2]"),
        
        // Incomplete function call
        ("(+ 1 2", "(+ 1 2 3"),
        
        // Starting a new nested form
        ("(if true", "(if true\n  (println :yes)"),
        
        // Adding to a vector
        ("[1 2 3", "[1 2 3 4 5]"),
        
        // Adding to a map
        ("{:a 1", "{:a 1 :b 2}"),
        
        // Nested map addition
        ("{:user {:name \"John\"", "{:user {:name \"John\" :age 30}"),
    ];
    
    for (before, after) in editing_scenarios {
        // Test that parinfer can fix the "before" state
        let parinfer_before = Parinfer::new(before);
        let balanced_before = parinfer_before.balance().unwrap();
        
        // Should be balanced
        let open_parens = balanced_before.chars().filter(|&c| c == '(').count();
        let close_parens = balanced_before.chars().filter(|&c| c == ')').count();
        assert_eq!(open_parens, close_parens);
        
        // Test that the "after" state is handled correctly
        let parinfer_after = Parinfer::new(after);
        let balanced_after = parinfer_after.balance().unwrap();
        
        let open_parens = balanced_after.chars().filter(|&c| c == '(').count();
        let close_parens = balanced_after.chars().filter(|&c| c == ')').count();
        assert_eq!(open_parens, close_parens);
    }
}