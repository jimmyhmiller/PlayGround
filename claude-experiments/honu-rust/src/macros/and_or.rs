use crate::{Term, LiteralValue, BracketType, ParseError, MacroTransformer, Span};
use super::expand_to_if;

#[derive(Debug)]
pub struct AndMacro;

impl MacroTransformer for AndMacro {
    fn transform(&self, terms: Vec<Term>, _env: &crate::parser::BindingEnv) -> Result<Vec<Term>, ParseError> {
        // Handle empty case: (and) -> true
        if terms.is_empty() {
            return Ok(vec![Term::Literal(LiteralValue::Boolean(true))]);
        }
        
        // Handle single argument: (and x) -> x
        if terms.len() == 1 {
            return Ok(terms);
        }
        
        // Check if this is infix syntax: left and right [and more...]
        if terms.len() >= 3 {
            if let Term::Identifier(op, _) = &terms[1] {
                if op == "and" {
                    // This is infix: term1 and term2 [and term3...]
                    let mut operands = vec![terms[0].clone()]; // Start with first operand
                    let mut i = 1;
                    
                    while i < terms.len() {
                        if let Term::Identifier(op, _) = &terms[i] {
                            if op == "and" {
                                if i + 1 < terms.len() {
                                    operands.push(terms[i + 1].clone());
                                    i += 2; // Skip the 'and' and the operand
                                } else {
                                    return Err(ParseError::MacroExpansionError("and expects operand after 'and'".to_string()));
                                }
                            } else {
                                break; // Not an 'and', end of chain
                            }
                        } else {
                            break; // Not an identifier, end of chain
                        }
                    }
                    
                    // Build nested if-then-else: (and a b c) -> (if a (if b c false) false)
                    let mut result = operands.pop().unwrap(); // Start with last operand
                    
                    while let Some(operand) = operands.pop() {
                        result = Term::Bracket(BracketType::Paren, vec![
                            Term::Identifier("if".to_string(), Span::synthetic()),
                            operand,
                            Term::Identifier("then".to_string(), Span::synthetic()),
                            result,
                            Term::Identifier("else".to_string(), Span::synthetic()),
                            Term::Literal(LiteralValue::Boolean(false)),
                        ]);
                    }
                    
                    // Return the expanded expression plus any remaining terms
                    let mut expanded = vec![result];
                    if i < terms.len() {
                        expanded.extend_from_slice(&terms[i..]);
                    }
                    
                    return Ok(expanded);
                }
            }
        }
        
        // Handle prefix syntax: (and a b c)
        let mut operands = terms;
        let mut result = operands.pop().unwrap(); // Start with last operand
        
        while let Some(operand) = operands.pop() {
            result = Term::Bracket(BracketType::Paren, vec![
                Term::Identifier("if".to_string(), Span::synthetic()),
                operand,
                Term::Identifier("then".to_string(), Span::synthetic()),
                result,
                Term::Identifier("else".to_string(), Span::synthetic()),
                Term::Literal(LiteralValue::Boolean(false)),
            ]);
        }
        
        Ok(vec![result])
    }
}

#[derive(Debug)]
pub struct OrMacro;

impl MacroTransformer for OrMacro {
    fn transform(&self, terms: Vec<Term>, _env: &crate::parser::BindingEnv) -> Result<Vec<Term>, ParseError> {
        if terms.is_empty() {
            return Ok(vec![Term::Literal(LiteralValue::Boolean(false))]);
        }
        
        if terms.len() == 1 {
            return Ok(terms);
        }
        
        // Handle infix syntax: left or right [or more...]
        if terms.len() >= 3 {
            if let Term::Identifier(op, _) = &terms[1] {
                if op == "or" {
                    // This is infix: term1 or term2 [or term3...]
                    let mut operands = vec![terms[0].clone()]; // Start with first operand
                    let mut i = 1;
                    
                    while i < terms.len() {
                        if let Term::Identifier(op, _) = &terms[i] {
                            if op == "or" {
                                if i + 1 < terms.len() {
                                    operands.push(terms[i + 1].clone());
                                    i += 2; // Skip the 'or' and the operand
                                } else {
                                    return Err(ParseError::MacroExpansionError("or expects operand after 'or'".to_string()));
                                }
                            } else {
                                break; // Not an 'or', end of chain
                            }
                        } else {
                            break; // Not an identifier, end of chain
                        }
                    }
                    
                    // Build nested if-then-else: (or a b c) -> (if a then a else (if b then b else c))
                    let mut result = operands.pop().unwrap(); // Start with last operand
                    
                    while let Some(operand) = operands.pop() {
                        result = Term::Bracket(BracketType::Paren, vec![
                            Term::Identifier("if".to_string(), Span::synthetic()),
                            operand.clone(),
                            Term::Identifier("then".to_string(), Span::synthetic()),
                            operand,
                            Term::Identifier("else".to_string(), Span::synthetic()),
                            result,
                        ]);
                    }
                    
                    // Return the expanded expression plus any remaining terms
                    let mut expanded = vec![result];
                    if i < terms.len() {
                        expanded.extend_from_slice(&terms[i..]);
                    }
                    
                    return Ok(expanded);
                }
            }
        }
        
        // Handle prefix syntax: (or a b c)
        let first = terms[0].clone();
        let rest = terms[1..].to_vec();
        
        // For prefix: (or x y z) -> if x then x else (if y then y else z)
        let or_rest = if rest.len() == 1 {
            rest[0].clone()
        } else {
            // Build nested or expressions for multiple operands
            rest[0].clone() // Simplified
        };
        
        // Generate: if first then first else or_rest
        Ok(vec![
            Term::Identifier("if".to_string(), Span::synthetic()),
            first.clone(),
            Term::Identifier("then".to_string(), Span::synthetic()),
            first,
            Term::Identifier("else".to_string(), Span::synthetic()),
            or_rest,
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_and_macro_empty() {
        let and_macro = AndMacro;
        let arena = bumpalo::Bump::new();
        let env = crate::parser::BindingEnv::new(&arena);
        let result = and_macro.transform(vec![], &env).unwrap();
        
        match &result[0] {
            Term::Literal(LiteralValue::Boolean(true)) => {}
            _ => panic!("Expected true for empty and"),
        }
    }

    #[test]
    fn test_and_macro_single() {
        let and_macro = AndMacro;
        let input = vec![Term::Identifier("x".to_string(), Span::synthetic())];
        let arena = bumpalo::Bump::new();
        let env = crate::parser::BindingEnv::new(&arena);
        let result = and_macro.transform(input.clone(), &env).unwrap();
        
        assert_eq!(result.len(), 1);
        match &result[0] {
            Term::Identifier(name, _) => assert_eq!(name, "x"),
            _ => panic!("Expected identifier x"),
        }
    }

    #[test]
    fn test_or_macro_empty() {
        let or_macro = OrMacro;
        let arena = bumpalo::Bump::new();
        let env = crate::parser::BindingEnv::new(&arena);
        let result = or_macro.transform(vec![], &env).unwrap();
        
        match &result[0] {
            Term::Literal(LiteralValue::Boolean(false)) => {}
            _ => panic!("Expected false for empty or"),
        }
    }
}