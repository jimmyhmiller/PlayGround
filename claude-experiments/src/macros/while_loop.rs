use crate::{Term, LiteralValue, BracketType, ParseError, MacroTransformer, Span};

pub struct WhileMacro;

impl MacroTransformer for WhileMacro {
    fn transform(&self, terms: Vec<Term>) -> Result<Vec<Term>, ParseError> {
        if terms.len() != 2 {
            return Err(ParseError::MacroExpansionError(
                "while macro requires exactly 2 arguments: condition and body".to_string()
            ));
        }
        
        let condition = terms[0].clone();
        let body = terms[1].clone();
        
        // (while condition body) -> 
        // (loop 
        //   (if condition 
        //     (do body)
        //     (break)))
        let expanded = Term::Bracket(BracketType::Paren, vec![
            Term::Identifier("loop".to_string(), Span::synthetic()),
            Term::Bracket(BracketType::Paren, vec![
                Term::Identifier("if".to_string(), Span::synthetic()),
                condition,
                Term::Bracket(BracketType::Paren, vec![
                    Term::Identifier("do".to_string(), Span::synthetic()),
                    body,
                ]),
                Term::Bracket(BracketType::Paren, vec![
                    Term::Identifier("break".to_string(), Span::synthetic()),
                ]),
            ]),
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
        
        let result = while_macro.transform(input).unwrap();
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
        
        let result = while_macro.transform(input);
        assert!(result.is_err());
    }
}