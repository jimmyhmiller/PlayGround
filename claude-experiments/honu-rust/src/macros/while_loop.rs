use crate::{Term, LiteralValue, BracketType, ParseError, MacroTransformer, Span};

#[derive(Debug)]
pub struct WhileMacro;

impl MacroTransformer for WhileMacro {
    fn transform(&self, terms: Vec<Term>, _env: &crate::parser::BindingEnv) -> Result<Vec<Term>, ParseError> {
        // Handle both prefix and infix syntax
        let (condition, body) = if terms.len() == 2 {
            // Prefix syntax: (while condition body)
            (terms[0].clone(), terms[1].clone())
        } else if terms.len() >= 3 {
            // Check for infix syntax: while condition { body } or similar
            if let Term::Identifier(keyword, _) = &terms[0] {
                if keyword == "while" {
                    // Infix: [while, condition, body]
                    if terms.len() < 3 {
                        return Err(ParseError::MacroExpansionError(
                            "while macro requires condition and body".to_string()
                        ));
                    }
                    (terms[1].clone(), terms[2].clone())
                } else {
                    return Err(ParseError::MacroExpansionError(
                        "Invalid while macro syntax".to_string()
                    ));
                }
            } else {
                return Err(ParseError::MacroExpansionError(
                    "Invalid while macro syntax".to_string()
                ));
            }
        } else {
            return Err(ParseError::MacroExpansionError(
                "while macro requires condition and body".to_string()
            ));
        };
        
        // For a practical while macro, let's expand to a simple representation
        // that captures the while loop semantics using built-in constructs:
        // (while condition body) -> (loop-construct condition body)
        // 
        // Since this is meant to demonstrate macro expansion, let's create
        // a structure that represents the while loop concept
        
        let expanded = Term::Bracket(BracketType::Paren, vec![
            Term::Identifier("loop".to_string(), Span::synthetic()),
            condition,
            body,
        ]);
        
        Ok(vec![expanded])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_while_macro() {
        let while_macro = WhileMacro;
        let input = vec![
            Term::Identifier("condition".to_string(), Span::synthetic()),
            Term::Identifier("body".to_string(), Span::synthetic()),
        ];
        
        let arena = bumpalo::Bump::new();
        let env = crate::parser::BindingEnv::new(&arena);
        let result = while_macro.transform(input, &env).unwrap();
        assert_eq!(result.len(), 1);
        
        match &result[0] {
            Term::Bracket(BracketType::Paren, terms) => {
                match &terms[0] {
                    Term::Identifier(name, _) => assert_eq!(name, "loop"),
                    _ => panic!("Expected loop identifier"),
                }
            }
            _ => panic!("Expected bracket with loop"),
        }
    }

    #[test]
    fn test_while_macro_wrong_args() {
        let while_macro = WhileMacro;
        let input = vec![Term::Identifier("condition".to_string(), Span::synthetic())];
        
        let arena = bumpalo::Bump::new();
        let env = crate::parser::BindingEnv::new(&arena);
        let result = while_macro.transform(input, &env);
        assert!(result.is_err());
    }
}