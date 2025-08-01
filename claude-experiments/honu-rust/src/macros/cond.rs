use crate::{Term, LiteralValue, BracketType, ParseError, MacroTransformer, Span};

#[derive(Debug)]
pub struct CondMacro;

impl MacroTransformer for CondMacro {
    fn transform(&self, terms: Vec<Term>, _env: &crate::parser::BindingEnv) -> Result<Vec<Term>, ParseError> {
        if terms.is_empty() {
            return Ok(vec![Term::Literal(LiteralValue::Nil)]);
        }
        
        // Each term should be a [condition expression] pair
        let mut clauses = Vec::new();
        
        for term in terms {
            match term {
                Term::Bracket(BracketType::Square, mut clause_terms) => {
                    if clause_terms.len() != 2 {
                        return Err(ParseError::MacroExpansionError(
                            "cond clauses must have exactly 2 elements: [condition expression]".to_string()
                        ));
                    }
                    
                    let condition = clause_terms.remove(0);
                    let expression = clause_terms.remove(0);
                    clauses.push((condition, expression));
                }
                _ => {
                    return Err(ParseError::MacroExpansionError(
                        "cond clauses must be square brackets: [condition expression]".to_string()
                    ));
                }
            }
        }
        
        // Convert to nested if statements
        let expanded = self.build_nested_if(clauses);
        Ok(vec![expanded])
    }
}

impl CondMacro {
    fn build_nested_if(&self, mut clauses: Vec<(Term, Term)>) -> Term {
        if clauses.is_empty() {
            return Term::Literal(LiteralValue::Nil);
        }
        
        let (condition, expression) = clauses.remove(0);
        
        if clauses.is_empty() {
            // Last clause or only clause
            Term::Bracket(BracketType::Paren, vec![
                Term::Identifier("if".to_string(), Span::synthetic()),
                condition,
                expression,
                Term::Literal(LiteralValue::Nil),
            ])
        } else {
            // More clauses - recursive
            let else_branch = self.build_nested_if(clauses);
            
            Term::Bracket(BracketType::Paren, vec![
                Term::Identifier("if".to_string(), Span::synthetic()),
                condition,
                expression,
                else_branch,
            ])
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cond_macro_empty() {
        let cond_macro = CondMacro;
        let arena = bumpalo::Bump::new();
        let env = crate::parser::BindingEnv::new(&arena);
        let result = cond_macro.transform(vec![], &env).unwrap();
        
        match &result[0] {
            Term::Literal(LiteralValue::Nil) => {}
            _ => panic!("Expected nil for empty cond"),
        }
    }

    #[test]
    fn test_cond_macro_single_clause() {
        let cond_macro = CondMacro;
        let input = vec![
            Term::Bracket(BracketType::Square, vec![
                Term::Identifier("condition".to_string(), Span::synthetic()),
                Term::Identifier("result".to_string(), Span::synthetic()),
            ])
        ];
        
        let arena = bumpalo::Bump::new();
        let env = crate::parser::BindingEnv::new(&arena);
        let result = cond_macro.transform(input, &env).unwrap();
        assert_eq!(result.len(), 1);
        
        match &result[0] {
            Term::Bracket(BracketType::Paren, terms) => {
                match &terms[0] {
                    Term::Identifier(name, _) => assert_eq!(name, "if"),
                    _ => panic!("Expected if identifier"),
                }
            }
            _ => panic!("Expected bracket with if"),
        }
    }

    #[test]
    fn test_cond_macro_multiple_clauses() {
        let cond_macro = CondMacro;
        let input = vec![
            Term::Bracket(BracketType::Square, vec![
                Term::Identifier("cond1".to_string(), Span::synthetic()),
                Term::Identifier("result1".to_string(), Span::synthetic()),
            ]),
            Term::Bracket(BracketType::Square, vec![
                Term::Identifier("cond2".to_string(), Span::synthetic()),
                Term::Identifier("result2".to_string(), Span::synthetic()),
            ]),
        ];
        
        let arena = bumpalo::Bump::new();
        let env = crate::parser::BindingEnv::new(&arena);
        let result = cond_macro.transform(input, &env).unwrap();
        assert_eq!(result.len(), 1);
        
        // Should generate nested if statements
        match &result[0] {
            Term::Bracket(BracketType::Paren, terms) => {
                match &terms[0] {
                    Term::Identifier(name, _) => assert_eq!(name, "if"),
                    _ => panic!("Expected if identifier"),
                }
                
                // Check that the else branch is another if
                if terms.len() >= 4 {
                    match &terms[3] {
                        Term::Bracket(BracketType::Paren, else_terms) => {
                            match &else_terms[0] {
                                Term::Identifier(name, _) => assert_eq!(name, "if"),
                                _ => panic!("Expected nested if in else branch"),
                            }
                        }
                        _ => panic!("Expected bracket in else branch"),
                    }
                }
            }
            _ => panic!("Expected bracket with if"),
        }
    }
}