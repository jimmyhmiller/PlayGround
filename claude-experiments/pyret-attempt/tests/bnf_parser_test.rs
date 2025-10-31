use std::fs;

#[test]
fn test_parse_original_pyret_grammar_fails() {
    // Path to the original Pyret grammar file
    let grammar_path = "/Users/jimmyhmiller/Documents/Code/open-source/pyret-lang/src/js/base/pyret-grammar.bnf";

    // Read the grammar file
    let grammar_content = fs::read_to_string(grammar_path)
        .expect("Failed to read pyret-grammar.bnf");

    println!("Attempting to parse original Pyret grammar with bnf crate...");
    println!("Grammar file size: {} bytes", grammar_content.len());

    // Attempt to parse with the bnf crate - this should fail
    match grammar_content.parse::<bnf::Grammar>() {
        Ok(_) => {
            panic!("Expected parsing to fail for original Pyret grammar format");
        }
        Err(e) => {
            println!("✓ Original grammar correctly fails to parse (as expected)");
            println!("Error: {:?}", e);
        }
    }
}

#[test]
fn test_parse_converted_pyret_grammar() {
    // Path to the converted Pyret grammar file
    let grammar_path = "pyret-grammar-standard.bnf";

    // Read the grammar file
    let grammar_content = fs::read_to_string(grammar_path)
        .expect("Failed to read converted grammar");

    println!("Attempting to parse converted Pyret grammar with bnf crate...");
    println!("Grammar file size: {} bytes", grammar_content.len());

    // Attempt to parse with the bnf crate
    match grammar_content.parse::<bnf::Grammar>() {
        Ok(grammar) => {
            println!("✓ Successfully parsed converted Pyret grammar!");
            println!("Number of productions: {}", grammar.productions_iter().count());

            // Print first few productions to verify
            println!("\nFirst few productions:");
            for (i, production) in grammar.productions_iter().enumerate().take(10) {
                println!("  {}: {:?}", i + 1, production.lhs);
            }
        }
        Err(e) => {
            println!("✗ Failed to parse converted Pyret grammar");
            println!("Error: {:?}", e);
            panic!("BNF parsing failed: {:?}", e);
        }
    }
}

#[test]
fn test_bnf_basic_sanity() {
    // Test that the bnf crate works at all with a simple grammar
    let simple_grammar = r#"
        <expr> ::= <term> "+" <expr> | <term>
        <term> ::= <factor> "*" <term> | <factor>
        <factor> ::= "(" <expr> ")" | <number>
        <number> ::= "0" | "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9"
    "#;

    match simple_grammar.parse::<bnf::Grammar>() {
        Ok(grammar) => {
            println!("✓ Basic BNF test passed");
            println!("Number of productions: {}", grammar.productions_iter().count());
        }
        Err(e) => {
            panic!("Basic BNF test failed: {:?}", e);
        }
    }
}
