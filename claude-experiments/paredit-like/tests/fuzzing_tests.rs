/// Fuzzing-inspired tests for robustness
use paredit_like::*;
use std::fs;
use tempfile::TempDir;

/// Generate various malformed inputs to test robustness
fn generate_malformed_inputs() -> Vec<String> {
    vec![
        // Structural issues
        "(".repeat(100),
        ")".repeat(100),
        "[".repeat(50) + &"]".repeat(30),
        "{".repeat(30) + &"}".repeat(50),
        
        // Mixed delimiters
        "([)]".to_string(),
        "[{]}".to_string(),
        "{[})".to_string(),
        "((])".to_string(),
        
        // Incomplete strings
        "\"incomplete".to_string(),
        "\"has\nnewline".to_string(),
        "\"has\\".to_string(),
        "\"\\u{".to_string(),
        
        // Control characters
        "\x00\x01\x02\x03".to_string(),
        "\x7F\u{0080}\u{0081}\u{0082}".to_string(),
        
        // Large whitespace
        " ".repeat(10000),
        "\t".repeat(1000),
        "\n".repeat(1000),
        
        // Unicode edge cases
        "\u{FEFF}(test)".to_string(), // BOM
        "🦀".repeat(1000),
        "\u{200B}".repeat(100), // Zero-width space
        
        // Pathological nesting
        "(".repeat(1000) + "atom" + &")".repeat(500),
        "[".repeat(500) + "atom" + &"]".repeat(1000),
        
        // Very long tokens
        "very_long_symbol_".to_string() + &"a".repeat(10000),
        ":very-long-keyword-".to_string() + &"x".repeat(5000),
        
        // Numbers that look malformed
        "123abc".to_string(),
        "3.14.15.92".to_string(),
        "0x".to_string(),
        "22/".to_string(),
        
        // Comment variations
        ";".repeat(1000),
        "; ".to_string() + &"x".repeat(10000),
        
        // Mixed valid/invalid
        "(valid-form) (".to_string(),
        "[1 2 3] }".to_string(),
        "{:a 1} )".to_string(),
    ]
}

#[test]
fn test_parser_robustness_fuzzing() {
    let malformed_inputs = generate_malformed_inputs();
    
    for input in malformed_inputs {
        let mut parser = ClojureParser::new().unwrap();
        
        // Should never panic, regardless of input
        let result = parser.parse_to_sexpr(&input);
        
        // We don't care if it succeeds or fails, just that it doesn't crash
        match result {
            Ok(forms) => {
                println!("Successfully parsed {} forms from malformed input", forms.len());
            }
            Err(_) => {
                println!("Parsing failed gracefully");
            }
        }
    }
}

#[test]
fn test_parinfer_robustness_fuzzing() {
    let malformed_inputs = generate_malformed_inputs();
    
    for input in malformed_inputs {
        let parinfer = Parinfer::new(&input);
        
        // Should never panic, regardless of input
        let result = parinfer.balance();
        
        match result {
            Ok(balanced) => {
                // If it succeeds, it should produce balanced output
                let open_parens = balanced.chars().filter(|&c| c == '(').count();
                let close_parens = balanced.chars().filter(|&c| c == ')').count();
                let open_brackets = balanced.chars().filter(|&c| c == '[').count();
                let close_brackets = balanced.chars().filter(|&c| c == ']').count();
                let open_braces = balanced.chars().filter(|&c| c == '{').count();
                let close_braces = balanced.chars().filter(|&c| c == '}').count();
                
                assert_eq!(open_parens, close_parens, "Unbalanced parens in output");
                assert_eq!(open_brackets, close_brackets, "Unbalanced brackets in output");
                assert_eq!(open_braces, close_braces, "Unbalanced braces in output");
                
                println!("Successfully balanced malformed input");
            }
            Err(_) => {
                println!("Balancing failed gracefully");
            }
        }
    }
}

#[test]
fn test_refactoring_robustness_fuzzing() {
    let malformed_inputs = generate_malformed_inputs();
    
    for input in malformed_inputs {
        let mut parser = ClojureParser::new().unwrap();
        
        if let Ok(forms) = parser.parse_to_sexpr(&input) {
            let mut refactorer = Refactorer::new(input.clone());
            
            // Test all refactoring operations on potentially malformed input
            for line in [1, 2, 5, 10] {
                // Should never panic
                let _ = refactorer.slurp_forward(&forms, line);
                let _ = refactorer.slurp_backward(&forms, line);
                let _ = refactorer.barf_forward(&forms, line);
                let _ = refactorer.barf_backward(&forms, line);
                let _ = refactorer.splice(&forms, line);
                let _ = refactorer.raise(&forms, line);
                let _ = refactorer.wrap(&forms, line, "(");
                let _ = refactorer.wrap(&forms, line, "[");
                let _ = refactorer.wrap(&forms, line, "{");
                let _ = refactorer.merge_let(&forms, line);
            }
        }
    }
}

/// Test with randomly generated character sequences
#[test]
fn test_random_character_sequences() {
    // Use a simple deterministic "random" generator for reproducible tests
    let mut seed = 42u32;
    
    for _ in 0..100 {
        let mut random_input = String::new();
        let length = (seed % 100) + 1; // 1-100 characters
        
        for _ in 0..length {
            seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
            let char_code = seed % 128; // ASCII range
            
            if let Some(ch) = char::from_u32(char_code) {
                if ch.is_control() && ch != '\n' && ch != '\t' {
                    // Skip most control characters except newline and tab
                    continue;
                }
                random_input.push(ch);
            }
        }
        
        // Test parsing
        let mut parser = ClojureParser::new().unwrap();
        let _ = parser.parse_to_sexpr(&random_input);
        
        // Test balancing
        let parinfer = Parinfer::new(&random_input);
        let _ = parinfer.balance();
    }
}

/// Test with inputs that have specific patterns that might break things
#[test]
fn test_pathological_patterns() {
    let patterns = vec![
        // Alternating delimiters
        "()()()()()".to_string(),
        "[][][][]".to_string(),
        "{}{}{}{}".to_string(),
        
        // Deeply nested single type
        "(".repeat(100) + &")".repeat(100),
        "[".repeat(100) + &"]".repeat(100),
        "{".repeat(100) + &"}".repeat(100),
        
        // Very wide flat structures
        format!("({} )", "a ".repeat(1000)),
        format!("[{} ]", "1 ".repeat(1000)),
        format!("{{{} }}", ":a 1 ".repeat(500)),
        
        // Strings with lots of escapes
        format!("\"{}\"", "\\n".repeat(1000)),
        format!("\"{}\"", "\\\"".repeat(500)),
        format!("\"{}\"", "\\\\".repeat(1000)),
        
        // Comments with various content
        format!("; {}", "test ".repeat(1000)),
        format!("; {}", "()[]{}".repeat(200)),
        
        // Mixed everything
        format!("(let [x {}] (+ x 1))", "\"string\" ".repeat(100)),
    ];
    
    for pattern in patterns {
        // Test parsing
        let mut parser = ClojureParser::new().unwrap();
        let parse_result = parser.parse_to_sexpr(&pattern);
        
        // Test balancing
        let parinfer = Parinfer::new(&pattern);
        let balance_result = parinfer.balance();
        
        // If balancing succeeds, verify it's actually balanced
        if let Ok(balanced) = balance_result {
            let open_parens = balanced.chars().filter(|&c| c == '(').count();
            let close_parens = balanced.chars().filter(|&c| c == ')').count();
            assert_eq!(open_parens, close_parens);
        }
        
        // Test refactoring if parsing succeeded
        if let Ok(forms) = parse_result {
            let mut refactorer = Refactorer::new(pattern.clone());
            let _ = refactorer.wrap(&forms, 1, "(");
            let _ = refactorer.splice(&forms, 1);
        }
    }
}

/// Test memory usage with large inputs
#[test]
fn test_memory_stress() {
    // Test various large input patterns
    let large_inputs = vec![
        // Many small forms
        (0..10000).map(|i| format!("(f{})", i)).collect::<Vec<_>>().join(" "),
        
        // One huge form
        format!("({})", (0..10000).map(|i| format!("item{}", i)).collect::<Vec<_>>().join(" ")),
        
        // Deep nesting
        "(".repeat(1000) + "deep" + &")".repeat(1000),
        
        // Long strings
        format!("\"{}\"", "content ".repeat(10000)),
        
        // Many comments
        (0..1000).map(|i| format!("; comment {}", i)).collect::<Vec<_>>().join("\n"),
    ];
    
    for large_input in large_inputs {
        // Test parsing (should not run out of memory)
        let mut parser = ClojureParser::new().unwrap();
        let _result = parser.parse_to_sexpr(&large_input);
        
        // Test balancing (should not run out of memory)
        let parinfer = Parinfer::new(&large_input);
        let _result = parinfer.balance();
        
        // Basic smoke test passed if we reach here without OOM
    }
}

/// Test with files containing various encodings and line endings
#[test]
fn test_file_encoding_robustness() {
    let temp_dir = TempDir::new().unwrap();
    
    // Test different line ending styles
    let line_ending_tests = vec![
        ("unix.clj", "(defn test [])\n(+ 1 2)"),
        ("windows.clj", "(defn test [])\r\n(+ 1 2)"),
        ("mac.clj", "(defn test [])\r(+ 1 2)"),
        ("mixed.clj", "(defn test [])\n(+ 1 2)\r\n(* 3 4)"),
    ];
    
    for (filename, content) in line_ending_tests {
        let file_path = temp_dir.path().join(filename);
        fs::write(&file_path, content).unwrap();
        
        // Test reading and processing
        let read_content = paredit_like::cli::read_file(&file_path).unwrap();
        
        let mut parser = ClojureParser::new().unwrap();
        let _forms = parser.parse_to_sexpr(&read_content).unwrap();
        
        let parinfer = Parinfer::new(&read_content);
        let _balanced = parinfer.balance().unwrap();
    }
    
    // Test with Unicode content
    let unicode_tests = vec![
        ("unicode.clj", "(str \"Hello 世界\")"),
        ("emoji.clj", "(println \"🦀 Rust! 🎉\")"),
        ("greek.clj", "(def π 3.14159)"),
        ("mixed.clj", "(defn café [] \"résumé 数学\")"),
    ];
    
    for (filename, content) in unicode_tests {
        let file_path = temp_dir.path().join(filename);
        fs::write(&file_path, content).unwrap();
        
        let read_content = paredit_like::cli::read_file(&file_path).unwrap();
        
        let mut parser = ClojureParser::new().unwrap();
        let _forms = parser.parse_to_sexpr(&read_content).unwrap();
        
        let parinfer = Parinfer::new(&read_content);
        let _balanced = parinfer.balance().unwrap();
    }
}

/// Test error propagation and recovery
#[test]
fn test_error_recovery() {
    let problematic_inputs = vec![
        // Parsing might fail but shouldn't panic
        "\x00\x01\x02",
        "incomplete\"string",
        "([{})]",
        
        // Balancing might fail on certain inputs
        "\"never ending string...",
        
        // Refactoring should fail gracefully on non-lists
        "42",
        ":keyword",
        "symbol",
    ];
    
    for input in problematic_inputs {
        // Test full pipeline resilience
        let mut parser = ClojureParser::new().unwrap();
        
        match parser.parse_to_sexpr(input) {
            Ok(forms) => {
                // If parsing succeeded, test refactoring error handling
                let mut refactorer = Refactorer::new(input.to_string());
                
                // These should either succeed or fail gracefully
                let _slurp_result = refactorer.slurp_forward(&forms, 1);
                let _barf_result = refactorer.barf_forward(&forms, 1);
                let _splice_result = refactorer.splice(&forms, 1);
                let _wrap_result = refactorer.wrap(&forms, 1, "(");
                let _raise_result = refactorer.raise(&forms, 1);
                let _merge_result = refactorer.merge_let(&forms, 1);
            }
            Err(_) => {
                // Parsing failed, which is acceptable for malformed input
            }
        }
        
        // Test parinfer error handling
        let parinfer = Parinfer::new(input);
        match parinfer.balance() {
            Ok(balanced) => {
                // If it succeeded, verify it's balanced
                let open_count = balanced.chars().filter(|&c| c == '(').count();
                let close_count = balanced.chars().filter(|&c| c == ')').count();
                assert_eq!(open_count, close_count);
            }
            Err(_) => {
                // Failure is acceptable for some inputs
            }
        }
    }
}