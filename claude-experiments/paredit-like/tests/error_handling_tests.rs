use paredit_like::*;
use std::fs;
use tempfile::TempDir;

#[test]
fn test_parser_error_handling() {
    // Test various edge cases that might cause parser issues
    let edge_cases = vec![
        "",                           // Empty input
        "   \n  \t  ",               // Whitespace only
        ")",                         // Lone closing paren
        "]",                         // Lone closing bracket
        "}",                         // Lone closing brace
        "(((",                       // Only opening parens
        ")))",                       // Only closing parens
        "([)]",                      // Mismatched parens
        "\"unclosed string",         // Unclosed string
        "\"string with \\",          // String ending with backslash
        "; just a comment",          // Comment only
        "\\",                        // Lone backslash
        "\0",                        // Null character
        "🦀",                        // Unicode emoji
        "\"\\u{1F980}\"",           // Unicode in string
        r#""\n\t\r""#,              // Escape sequences
        "(defn",                     // Incomplete form
        "(defn foo",                 // Incomplete with partial name
        "(defn foo [",               // Incomplete with partial params
        "(let [x",                   // Incomplete let binding
        "#",                         // Hash without following form
        "^",                         // Metadata marker without form
        "'",                         // Quote without form
        "`",                         // Syntax quote without form
        "~",                         // Unquote without form
        "@",                         // Deref without form
        "#_",                        // Discard without form
        "#{",                        // Incomplete set
        "#{}",                       // Empty set
        "#'",                        // Var quote without form
        "#\"",                       // Incomplete regex
        "(+ 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15)", // Long list
    ];
    
    for input in edge_cases {
        let mut parser = ClojureParser::new().unwrap();
        // Should not panic, even if parsing fails or returns partial results
        let result = parser.parse_to_sexpr(input);
        // We don't assert success because some inputs are intentionally malformed
        // But we do verify no panic occurred by reaching this point
        println!("Parsed '{}': {:?}", input, result.is_ok());
    }
}

#[test]
fn test_parinfer_error_handling() {
    let edge_cases = vec![
        "",                           // Empty input
        "   \n  \t  ",               // Whitespace only
        "\"unclosed string",          // Unclosed string
        "\"string with\nnewlines\"",  // String with newlines
        "\"string with \\\"quotes\\\"\"", // String with escaped quotes
        "; comment only",             // Comment only
        "\u{0000}",                   // Null character
        "\u{FEFF}",                   // BOM character
        "🦀 (+ 1 2)",                // Unicode before form
        "(str \"🦀\")",              // Unicode in string
        "((((((((((deep))))))))))))", // Very deep nesting
    ];
    
    let many_opens = "(".repeat(1000);
    let many_closes = ")".repeat(1000);
    let wide_form_content = "a ".repeat(1000);
    let wide_form = format!("({})", wide_form_content);
    
    let additional_cases = vec![
        many_opens.as_str(),         // Very many opening parens
        many_closes.as_str(),        // Very many closing parens
        wide_form.as_str(),          // Very wide form
    ];
    
    for input in edge_cases {
        let parinfer = Parinfer::new(input);
        match parinfer.balance() {
            Ok(result) => {
                // If it succeeds, verify the result is balanced
                let open_parens = result.chars().filter(|&c| c == '(').count();
                let close_parens = result.chars().filter(|&c| c == ')').count();
                assert_eq!(open_parens, close_parens, "Unbalanced result for input: {}", input);
            }
            Err(e) => {
                // Some inputs might legitimately fail, that's ok
                println!("Failed to balance '{}': {}", input, e);
            }
        }
    }
}

#[test]
fn test_refactoring_error_handling() {
    let problematic_inputs = vec![
        ("", 1),                      // Empty input
        ("atom", 1),                  // Not a list
        ("()", 1),                    // Empty list
        ("( )", 1),                   // Empty list with spaces
        ("(a)", 1),                   // Single element list
        ("(a b c)", 10),              // Line number too high
        ("(a b c)", 0),               // Line number too low
        ("\"string\"", 1),            // String, not list
        (":keyword", 1),              // Keyword, not list
        ("42", 1),                    // Number, not list
        ("true", 1),                  // Boolean, not list
        ("nil", 1),                   // Nil, not list
        ("; comment", 1),             // Comment, not list
    ];
    
    for (input, line) in problematic_inputs {
        let mut parser = ClojureParser::new().unwrap();
        if let Ok(forms) = parser.parse_to_sexpr(input) {
            let mut refactorer = Refactorer::new(input.to_string());
            
            // Test all refactoring operations
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
            
            // Should not panic, even if operations fail
        }
    }
}

#[test]
fn test_invalid_wrap_parameters() {
    let source = "foo".to_string();
    let mut parser = ClojureParser::new().unwrap();
    let forms = parser.parse_to_sexpr(&source).unwrap();
    let mut refactorer = Refactorer::new(source);
    
    let invalid_wrappers = vec!["<", ">", ")", "]", "}", "invalid", "", "(("];
    
    for wrapper in invalid_wrappers {
        let result = refactorer.wrap(&forms, 1, wrapper);
        match wrapper {
            "(" | "[" | "{" => {
                // These should succeed (if we have a valid form)
                assert!(result.is_ok() || result.unwrap_err().to_string().contains("No form found"));
            }
            _ => {
                // These should fail
                assert!(result.is_err());
            }
        }
    }
}

#[test]
fn test_file_operation_errors() {
    let temp_dir = TempDir::new().unwrap();
    
    // Test reading non-existent file
    let non_existent = temp_dir.path().join("does_not_exist.clj");
    let result = paredit_like::cli::read_file(&non_existent);
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("Failed to read file"));
    
    // Test reading directory as file
    let dir_path = temp_dir.path().join("directory");
    fs::create_dir(&dir_path).unwrap();
    let result = paredit_like::cli::read_file(&dir_path);
    assert!(result.is_err());
    
    // Test with invalid UTF-8 (this is hard to test portably)
    // Most filesystems and Rust's fs::read_to_string handle this gracefully
    
    // Test very large file (create a file that's large but not too large for CI)
    let large_file = temp_dir.path().join("large.clj");
    let large_content = "(defn test [])\n".repeat(10000);
    fs::write(&large_file, &large_content).unwrap();
    let result = paredit_like::cli::read_file(&large_file);
    assert!(result.is_ok());
    
    // Test file with special characters in name
    let special_file = temp_dir.path().join("special-file[1].clj");
    fs::write(&special_file, "(test)").unwrap();
    let result = paredit_like::cli::read_file(&special_file);
    assert!(result.is_ok());
}

#[test]
fn test_output_display_errors() {
    use paredit_like::cli::Output;
    use std::path::PathBuf;
    
    let output = Output {
        original: "test".to_string(),
        modified: "modified".to_string(),
    };
    
    // Test writing to non-existent directory
    let bad_path = PathBuf::from("/this/path/does/not/exist/file.clj");
    let result = output.display(true, false, false, &bad_path);
    assert!(result.is_err());

    // Test writing to read-only location (platform specific)
    #[cfg(unix)]
    {
        let readonly_path = PathBuf::from("/dev/null/file.clj");
        let result = output.display(true, false, false, &readonly_path);
        assert!(result.is_err());
    }
}

#[test]
fn test_ast_edge_cases() {
    use paredit_like::ast::{Position, Span, SExpr};
    
    // Test position with extreme values
    let pos = Position::new(usize::MAX, usize::MAX, usize::MAX);
    assert_eq!(pos.line, usize::MAX);
    
    // Test span with zero length
    let start = Position::new(1, 0, 10);
    let end = Position::new(1, 0, 10);
    let span = Span::new(start, end);
    assert_eq!(span.len(), 0);
    
    // Test span operations
    assert!(span.contains_line(1));
    assert!(!span.contains_line(2));
    assert!(span.contains_offset(10));
    assert!(!span.contains_offset(11));
    
    // Test with inverted span (end before start)
    let start = Position::new(1, 0, 20);
    let end = Position::new(1, 0, 10);
    let span = Span::new(start, end);
    // This creates an invalid span, but shouldn't panic
    let _len = span.len(); // Would wrap around, but that's the caller's responsibility
    
    // Test SExpr with empty values
    let empty_span = Span::new(Position::new(1, 0, 0), Position::new(1, 0, 0));
    
    let empty_atom = SExpr::Atom {
        span: empty_span.clone(),
        value: String::new(),
    };
    assert_eq!(empty_atom.span(), &empty_span);
    
    let empty_string = SExpr::String {
        span: empty_span.clone(),
        value: String::new(),
    };
    assert_eq!(empty_string.span(), &empty_span);
    
    let empty_comment = SExpr::Comment {
        span: empty_span.clone(),
        text: String::new(),
    };
    assert_eq!(empty_comment.span(), &empty_span);
    
    let empty_list = SExpr::List {
        span: empty_span.clone(),
        open: '(',
        close: ')',
        children: Vec::new(),
    };
    assert_eq!(empty_list.span(), &empty_span);
}

#[test]
fn test_concurrent_usage() {
    use std::thread;
    use std::sync::Arc;
    
    let test_inputs = Arc::new(vec![
        "(defn test1 [])",
        "(defn test2 [x] (+ x 1))",
        "(let [x 1] (+ x 2))",
        "(if true :yes :no)",
        "[1 2 3 4 5]",
        "{:a 1 :b 2 :c 3}",
    ]);
    
    let mut handles = vec![];
    
    // Spawn multiple threads doing parsing and balancing
    for i in 0..4 {
        let inputs = Arc::clone(&test_inputs);
        let handle = thread::spawn(move || {
            for input in inputs.iter() {
                // Test parsing
                let mut parser = ClojureParser::new().unwrap();
                let _forms = parser.parse_to_sexpr(input).unwrap();
                
                // Test balancing
                let parinfer = Parinfer::new(input);
                let _balanced = parinfer.balance().unwrap();
                
                // Add some work to make race conditions more likely
                thread::sleep(std::time::Duration::from_millis(i));
            }
        });
        handles.push(handle);
    }
    
    // Wait for all threads to complete
    for handle in handles {
        handle.join().unwrap();
    }
}

#[test]
fn test_memory_exhaustion_protection() {
    // Test with extremely deep nesting that could cause stack overflow
    let mut deep = String::new();
    for _ in 0..10000 {
        deep.push('(');
    }
    deep.push_str("deep");
    for _ in 0..5000 { // Leave some unbalanced
        deep.push(')');
    }
    
    // This should not cause stack overflow
    let parinfer = Parinfer::new(&deep);
    let result = parinfer.balance();
    // We don't require success, just that it doesn't crash
    match result {
        Ok(_) => println!("Successfully handled deep nesting"),
        Err(e) => println!("Failed on deep nesting (acceptable): {}", e),
    }
    
    // Test with extremely wide structure
    let wide_base = "item ";
    let wide_repeated = wide_base.repeat(100000);
    let wide = format!("({})", wide_repeated);
    
    let parinfer = Parinfer::new(&wide);
    let result = parinfer.balance();
    match result {
        Ok(_) => println!("Successfully handled wide structure"),
        Err(e) => println!("Failed on wide structure (acceptable): {}", e),
    }
}

#[test]
fn test_input_validation() {
    // Test that the library handles various invalid inputs gracefully
    let invalid_inputs = vec![
        "\u{FFFF}",              // Non-character
        "\u{FFFE}",              // Non-character  
        "\r\n\r\n",              // Windows line endings
        "\r",                     // Old Mac line endings
        "\n\n\n\n",              // Multiple Unix line endings
        "\t\t\t\t",              // Multiple tabs
        " \t \n \r ",             // Mixed whitespace
        "🦀🦀🦀",               // Multiple emoji
        "λ α β γ δ ε ζ η θ",     // Greek letters
        "中文测试",                // Chinese characters
        "العربية",                // Arabic text
        "עברית",                   // Hebrew text
        "Русский",                 // Cyrillic text
    ];
    
    for input in invalid_inputs {
        // Test parsing
        let mut parser = ClojureParser::new().unwrap();
        let _result = parser.parse_to_sexpr(input);
        
        // Test balancing
        let parinfer = Parinfer::new(input);
        let _result = parinfer.balance();
        
        // Main goal is no panic, results may vary
    }
}