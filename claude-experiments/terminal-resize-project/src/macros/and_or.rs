use crate::{Term, LiteralValue, BracketType, ParseError, MacroTransformer, Span};
use super::expand_to_if;

pub struct AndMacro;

impl MacroTransformer for AndMacro {
    fn transform(&self, terms: Vec<Term>) -> Result<Vec<Term>, ParseError> {
        if terms.is_empty() {
            return Ok(vec![Term::Literal(LiteralValue::Boolean(true))]);
        }
        
        if terms.len() == 1 {
            return Ok(terms);
        }
        
        let first = terms[0].clone();
        let rest = terms[1..].to_vec();
        
        // (and a b c) -> (if a (and b c) false)
        let and_rest = if rest.len() == 1 {
            rest[0].clone()
        } else {
            Term::Bracket(BracketType::Paren, {
                let mut result = vec![Term::Identifier("and".to_string(), Span::synthetic())];
                result.extend(rest);
                result
            })
        };
        
        let expanded = expand_to_if(
            first,
            and_rest,
            Some(Term::Literal(LiteralValue::Boolean(false)))
        );
        
        Ok(vec![expanded])
    }
}

pub struct OrMacro;

impl MacroTransformer for OrMacro {
    fn transform(&self, terms: Vec<Term>) -> Result<Vec<Term>, ParseError> {
        if terms.is_empty() {
            return Ok(vec![Term::Literal(LiteralValue::Boolean(false))]);
        }
        
        if terms.len() == 1 {
            return Ok(terms);
        }
        
        let first = terms[0].clone();
        let rest = terms[1..].to_vec();
        
        // (or a b c) -> (if a a (or b c))
        let or_rest = if rest.len() == 1 {
            rest[0].clone()
        } else {
            Term::Bracket(BracketType::Paren, {
                let mut result = vec![Term::Identifier("or".to_string(), Span::synthetic())];
                result.extend(rest);
                result
            })
        };
        
        let expanded = expand_to_if(
            first.clone(),
            first,
            Some(or_rest)
        );
        
        Ok(vec![expanded])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_and_macro_empty() {
        let and_macro = AndMacro;
        let result = and_macro.transform(vec![]).unwrap();
        
        match &result[0] {
            Term::Literal(LiteralValue::Boolean(true)) => {}
            _ => panic!("Expected true for empty and"),
        }
    }

    #[test]
    fn test_and_macro_single() {
        let and_macro = AndMacro;
        let input = vec![Term::Identifier("x".to_string(), Span::synthetic())];
        let result = and_macro.transform(input.clone()).unwrap();
        
        assert_eq!(result.len(), 1);
        match &result[0] {
            Term::Identifier(name, _) => assert_eq!(name, "x"),
            _ => panic!("Expected identifier x"),
        }
    }

    #[test]
    fn test_or_macro_empty() {
        let or_macro = OrMacro;
        let result = or_macro.transform(vec![]).unwrap();
        
        match &result[0] {
            Term::Literal(LiteralValue::Boolean(false)) => {}
            _ => panic!("Expected false for empty or"),
        }
    }
}