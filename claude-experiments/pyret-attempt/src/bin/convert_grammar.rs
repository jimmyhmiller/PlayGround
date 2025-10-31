use std::fs;
use std::collections::HashMap;

fn main() {
    // Read the original Pyret grammar
    let grammar = fs::read_to_string("pyret-grammar-original.bnf")
        .expect("Failed to read grammar file");

    // Token mappings from pyret-tokenizer.js
    let token_map = create_token_map();

    let converted = convert_to_standard_bnf(&grammar, &token_map);

    // Write the converted grammar
    fs::write("pyret-grammar-standard.bnf", converted)
        .expect("Failed to write converted grammar");

    println!("✓ Converted grammar written to pyret-grammar-standard.bnf");
}

fn create_token_map() -> HashMap<String, String> {
    let mut map = HashMap::new();

    // Keywords
    map.insert("AND".to_string(), "\"and\"".to_string());
    map.insert("AS".to_string(), "\"as\"".to_string());
    map.insert("ASCENDING".to_string(), "\"ascending\"".to_string());
    map.insert("ASK".to_string(), "\"ask\"".to_string());
    map.insert("BY".to_string(), "\"by\"".to_string());
    map.insert("CASES".to_string(), "\"cases\"".to_string());
    map.insert("CHECK".to_string(), "\"check\"".to_string());
    map.insert("DATA".to_string(), "\"data\"".to_string());
    map.insert("DESCENDING".to_string(), "\"descending\"".to_string());
    map.insert("DO".to_string(), "\"do\"".to_string());
    map.insert("RAISESNOT".to_string(), "\"does-not-raise\"".to_string());
    map.insert("ELSE".to_string(), "\"else\"".to_string());
    map.insert("ELSEIF".to_string(), "\"else if\"".to_string());
    map.insert("END".to_string(), "\"end\"".to_string());
    map.insert("EXAMPLES".to_string(), "\"examples\"".to_string());
    map.insert("TABLE-EXTEND".to_string(), "\"extend\"".to_string());
    map.insert("TABLE-EXTRACT".to_string(), "\"extract\"".to_string());
    map.insert("FALSE".to_string(), "\"false\"".to_string());
    map.insert("FOR".to_string(), "\"for\"".to_string());
    map.insert("FROM".to_string(), "\"from\"".to_string());
    map.insert("FUN".to_string(), "\"fun\"".to_string());
    map.insert("HIDING".to_string(), "\"hiding\"".to_string());
    map.insert("IF".to_string(), "\"if\"".to_string());
    map.insert("IMPORT".to_string(), "\"import\"".to_string());
    map.insert("INCLUDE".to_string(), "\"include\"".to_string());
    map.insert("IS".to_string(), "\"is\"".to_string());
    map.insert("ISEQUALEQUAL".to_string(), "\"is==\"".to_string());
    map.insert("ISEQUALTILDE".to_string(), "\"is=~\"".to_string());
    map.insert("ISNOT".to_string(), "\"is-not\"".to_string());
    map.insert("ISNOTEQUALEQUAL".to_string(), "\"is-not==\"".to_string());
    map.insert("ISNOTEQUALTILDE".to_string(), "\"is-not=~\"".to_string());
    map.insert("ISNOTSPACESHIP".to_string(), "\"is-not<=>\"".to_string());
    map.insert("ISROUGHLY".to_string(), "\"is-roughly\"".to_string());
    map.insert("ISNOTROUGHLY".to_string(), "\"is-not-roughly\"".to_string());
    map.insert("ISSPACESHIP".to_string(), "\"is<=>\"".to_string());
    map.insert("BECAUSE".to_string(), "\"because\"".to_string());
    map.insert("LAM".to_string(), "\"lam\"".to_string());
    map.insert("LAZY".to_string(), "\"lazy\"".to_string());
    map.insert("LET".to_string(), "\"let\"".to_string());
    map.insert("LETREC".to_string(), "\"letrec\"".to_string());
    map.insert("LOAD-TABLE".to_string(), "\"load-table\"".to_string());
    map.insert("METHOD".to_string(), "\"method\"".to_string());
    map.insert("MODULE".to_string(), "\"module\"".to_string());
    map.insert("NEWTYPE".to_string(), "\"newtype\"".to_string());
    map.insert("OF".to_string(), "\"of\"".to_string());
    map.insert("OR".to_string(), "\"or\"".to_string());
    map.insert("PROVIDE".to_string(), "\"provide\"".to_string());
    map.insert("PROVIDE-TYPES".to_string(), "\"provide-types\"".to_string());
    map.insert("RAISES".to_string(), "\"raises\"".to_string());
    map.insert("RAISESOTHER".to_string(), "\"raises-other-than\"".to_string());
    map.insert("RAISESSATISFIES".to_string(), "\"raises-satisfies\"".to_string());
    map.insert("RAISESVIOLATES".to_string(), "\"raises-violates\"".to_string());
    map.insert("REACTOR".to_string(), "\"reactor\"".to_string());
    map.insert("REC".to_string(), "\"rec\"".to_string());
    map.insert("REF".to_string(), "\"ref\"".to_string());
    map.insert("SANITIZE".to_string(), "\"sanitize\"".to_string());
    map.insert("SATISFIES".to_string(), "\"satisfies\"".to_string());
    map.insert("TABLE-SELECT".to_string(), "\"select\"".to_string());
    map.insert("SHADOW".to_string(), "\"shadow\"".to_string());
    map.insert("TABLE-FILTER".to_string(), "\"sieve\"".to_string());
    map.insert("SPY".to_string(), "\"spy\"".to_string());
    map.insert("TABLE-ORDER".to_string(), "\"order\"".to_string());
    map.insert("TABLE-UPDATE".to_string(), "\"transform\"".to_string());
    map.insert("TRUE".to_string(), "\"true\"".to_string());
    map.insert("TYPE".to_string(), "\"type\"".to_string());
    map.insert("TYPE-LET".to_string(), "\"type-let\"".to_string());
    map.insert("USING".to_string(), "\"using\"".to_string());
    map.insert("USE".to_string(), "\"use\"".to_string());
    map.insert("VAR".to_string(), "\"var\"".to_string());
    map.insert("SATISFIESNOT".to_string(), "\"violates\"".to_string());
    map.insert("WHEN".to_string(), "\"when\"".to_string());

    // Symbols
    map.insert("BLOCK".to_string(), "\"block:\"".to_string());
    map.insert("CHECKCOLON".to_string(), "\"check:\"".to_string());
    map.insert("DOC".to_string(), "\"doc:\"".to_string());
    map.insert("ELSECOLON".to_string(), "\"else:\"".to_string());
    map.insert("EXAMPLESCOLON".to_string(), "\"examples:\"".to_string());
    map.insert("OTHERWISECOLON".to_string(), "\"otherwise:\"".to_string());
    map.insert("PROVIDECOLON".to_string(), "\"provide:\"".to_string());
    map.insert("ROW".to_string(), "\"row:\"".to_string());
    map.insert("SHARING".to_string(), "\"sharing:\"".to_string());
    map.insert("SOURCECOLON".to_string(), "\"source:\"".to_string());
    map.insert("TABLE".to_string(), "\"table:\"".to_string());
    map.insert("THENCOLON".to_string(), "\"then:\"".to_string());
    map.insert("WHERE".to_string(), "\"where:\"".to_string());
    map.insert("WITH".to_string(), "\"with:\"".to_string());
    map.insert("LBRACK".to_string(), "\"[\"".to_string());
    map.insert("RBRACK".to_string(), "\"]\"".to_string());
    map.insert("LBRACE".to_string(), "\"{\"".to_string());
    map.insert("RBRACE".to_string(), "\"}\"".to_string());
    map.insert("LPAREN".to_string(), "\"(\"".to_string());
    map.insert("PARENSPACE".to_string(), "\"(\"".to_string());
    map.insert("PARENNOSPACE".to_string(), "\"(\"".to_string());
    map.insert("PARENAFTERBRACE".to_string(), "\"(\"".to_string());
    map.insert("RPAREN".to_string(), "\")\"".to_string());
    map.insert("SEMI".to_string(), "\";\"".to_string());
    map.insert("BACKSLASH".to_string(), "\"\\\\\"".to_string());
    map.insert("DOTDOTDOT".to_string(), "\"...\"".to_string());
    map.insert("DOT".to_string(), "\".\"".to_string());
    map.insert("BANG".to_string(), "\"!\"".to_string());
    map.insert("PERCENT".to_string(), "\"%\"".to_string());
    map.insert("COMMA".to_string(), "\",\"".to_string());
    map.insert("THINARROW".to_string(), "\"->\"".to_string());
    map.insert("THICKARROW".to_string(), "\"=>\"".to_string());
    map.insert("COLONEQUALS".to_string(), "\":=\"".to_string());
    map.insert("COLONCOLON".to_string(), "\"::\"".to_string());
    map.insert("COLON".to_string(), "\":\"".to_string());
    map.insert("BAR".to_string(), "\"|\"".to_string());
    map.insert("EQUALS".to_string(), "\"=\"".to_string());
    map.insert("EQUALEQUAL".to_string(), "\"==\"".to_string());
    map.insert("EQUALTILDE".to_string(), "\"=~\"".to_string());
    map.insert("LANGLE".to_string(), "\"<\"".to_string());
    map.insert("LT".to_string(), "\"<\"".to_string());
    map.insert("STAR".to_string(), "\"*\"".to_string());
    map.insert("TIMES".to_string(), "\"*\"".to_string());
    map.insert("RANGLE".to_string(), "\">\"".to_string());
    map.insert("GT".to_string(), "\">\"".to_string());
    map.insert("PLUS".to_string(), "\"+\"".to_string());
    map.insert("DASH".to_string(), "\"-\"".to_string());
    map.insert("SLASH".to_string(), "\"/\"".to_string());
    map.insert("LEQ".to_string(), "\"<=\"".to_string());
    map.insert("GEQ".to_string(), "\">=\"".to_string());
    map.insert("NEQ".to_string(), "\"<>\"".to_string());
    map.insert("SPACESHIP".to_string(), "\"<=>\"".to_string());
    map.insert("CARET".to_string(), "\"^\"".to_string());

    // Special tokens (keep as non-terminals)
    map.insert("NAME".to_string(), "<name>".to_string());
    map.insert("STRING".to_string(), "<string>".to_string());
    map.insert("NUMBER".to_string(), "<number>".to_string());
    map.insert("RATIONAL".to_string(), "<rational>".to_string());
    map.insert("ROUGHRATIONAL".to_string(), "<roughrational>".to_string());

    // Error tokens
    map.insert("UNTERMINATED-STRING".to_string(), "<unterminated-string>".to_string());
    map.insert("UNTERMINATED-BLOCK-COMMENT".to_string(), "<unterminated-block-comment>".to_string());
    map.insert("BAD-OPER".to_string(), "<bad-oper>".to_string());
    map.insert("BAD-NUMBER".to_string(), "<bad-number>".to_string());
    map.insert("UNKNOWN".to_string(), "<unknown>".to_string());

    map
}

fn convert_to_standard_bnf(grammar: &str, token_map: &HashMap<String, String>) -> String {
    let mut output = String::new();
    output.push_str("# Pyret Grammar in Standard BNF Format\n");
    output.push_str("# Converted from pyret-grammar.bnf\n\n");

    for line in grammar.lines() {
        let trimmed = line.trim();

        // Skip comments and empty lines
        if trimmed.is_empty() || trimmed.starts_with("/*") || trimmed.starts_with("*/") || trimmed.starts_with("#") {
            output.push_str(line);
            output.push('\n');
            continue;
        }

        // Check if this is a production rule
        if let Some(colon_pos) = trimmed.find(':') {
            let lhs = &trimmed[..colon_pos].trim();
            let rhs = &trimmed[colon_pos + 1..].trim();

            // Convert LHS: rule -> <rule>
            output.push_str(&format!("<{}> ::= ", lhs));

            // Convert RHS
            let converted_rhs = convert_rhs(rhs, token_map);
            output.push_str(&converted_rhs);
        } else {
            // Continuation line (starts with |)
            if trimmed.starts_with("|") {
                let rest = trimmed[1..].trim();
                output.push_str("     | ");
                let converted = convert_rhs(rest, token_map);
                output.push_str(&converted);
            } else {
                output.push_str(line);
            }
        }

        output.push('\n');
    }

    output
}

fn convert_rhs(rhs: &str, token_map: &HashMap<String, String>) -> String {
    let mut result = String::new();
    let mut chars = rhs.chars().peekable();
    let mut current_token = String::new();

    while let Some(ch) = chars.next() {
        match ch {
            // Handle optional: [...]
            '[' => {
                if !current_token.is_empty() {
                    result.push_str(&convert_token(&current_token, token_map));
                    result.push(' ');
                    current_token.clear();
                }
                result.push('[');
            }
            ']' => {
                if !current_token.is_empty() {
                    result.push_str(&convert_token(&current_token, token_map));
                    result.push(' ');
                    current_token.clear();
                }
                result.push(']');
            }
            // Handle grouping: (...)
            '(' => {
                if !current_token.is_empty() {
                    result.push_str(&convert_token(&current_token, token_map));
                    result.push(' ');
                    current_token.clear();
                }
                result.push('(');
            }
            ')' => {
                if !current_token.is_empty() {
                    result.push_str(&convert_token(&current_token, token_map));
                    result.push(' ');
                    current_token.clear();
                }
                result.push(')');
            }
            // Handle alternation: |
            '|' => {
                if !current_token.is_empty() {
                    result.push_str(&convert_token(&current_token, token_map));
                    result.push(' ');
                    current_token.clear();
                }
                result.push_str("| ");
            }
            // Handle repetition: *
            '*' => {
                if !current_token.is_empty() {
                    result.push_str(&convert_token(&current_token, token_map));
                    current_token.clear();
                }
                result.push('*');
                result.push(' ');
            }
            // Handle whitespace
            ' ' | '\t' => {
                if !current_token.is_empty() {
                    result.push_str(&convert_token(&current_token, token_map));
                    result.push(' ');
                    current_token.clear();
                }
            }
            // Build up token
            _ => {
                current_token.push(ch);
            }
        }
    }

    // Handle final token
    if !current_token.is_empty() {
        result.push_str(&convert_token(&current_token, token_map));
    }

    result.trim().to_string()
}

fn convert_token(token: &str, token_map: &HashMap<String, String>) -> String {
    // If it's in the token map, use that
    if let Some(mapped) = token_map.get(token) {
        return mapped.clone();
    }

    // If it's all uppercase or contains a dash (like TABLE-SELECT), it's likely a token
    // that we haven't mapped yet - treat as terminal
    if token.chars().all(|c| c.is_uppercase() || c == '-') {
        return format!("\"{}\"", token.to_lowercase());
    }

    // Otherwise, it's a non-terminal
    format!("<{}>", token)
}
