use crate::{Term, LiteralValue, BracketType, Span, ParseError};

pub struct Lexer {
    input: String,
    position: usize,
    current_char: Option<char>,
}

impl Lexer {
    pub fn new(input: String) -> Self {
        let mut lexer = Self {
            input,
            position: 0,
            current_char: None,
        };
        lexer.current_char = lexer.input.chars().nth(0);
        lexer
    }

    pub fn tokenize(&mut self) -> Result<Vec<Term>, ParseError> {
        let mut tokens = Vec::new();
        
        while self.current_char.is_some() {
            if self.current_char.unwrap().is_whitespace() {
                self.advance();
                continue;
            }
            
            if self.current_char.unwrap() == ';' {
                self.skip_comment();
                continue;
            }
            
            tokens.push(self.next_token()?);
        }
        
        Ok(tokens)
    }

    fn next_token(&mut self) -> Result<Term, ParseError> {
        let start_pos = self.position;
        
        match self.current_char.unwrap() {
            '(' => {
                self.advance();
                let inner = self.read_until_bracket(')')?;
                Ok(Term::Bracket(BracketType::Paren, inner))
            }
            '[' => {
                self.advance();
                let inner = self.read_until_bracket(']')?;
                Ok(Term::Bracket(BracketType::Square, inner))
            }
            '{' => {
                self.advance();
                let inner = self.read_until_bracket('}')?;
                Ok(Term::Bracket(BracketType::Curly, inner))
            }
            '"' => {
                let string_val = self.read_string()?;
                Ok(Term::Literal(LiteralValue::String(string_val)))
            }
            c if c.is_ascii_digit() || c == '.' => {
                let number = self.read_number()?;
                Ok(Term::Literal(LiteralValue::Number(number)))
            }
            c if self.is_operator_char(c) => {
                let op = self.read_operator();
                let span = Span::new(start_pos, self.position);
                Ok(Term::Operator(op, span))
            }
            c if c.is_alphabetic() || c == '_' => {
                let identifier = self.read_identifier();
                let span = Span::new(start_pos, self.position);
                
                // Check for boolean literals
                match identifier.as_str() {
                    "true" => Ok(Term::Literal(LiteralValue::Boolean(true))),
                    "false" => Ok(Term::Literal(LiteralValue::Boolean(false))),
                    "nil" => Ok(Term::Literal(LiteralValue::Nil)),
                    _ => Ok(Term::Identifier(identifier, span))
                }
            }
            _ => Err(ParseError::LexerError(format!("Unexpected character: {}", self.current_char.unwrap())))
        }
    }

    fn read_until_bracket(&mut self, closing: char) -> Result<Vec<Term>, ParseError> {
        let mut tokens = Vec::new();
        
        while self.current_char.is_some() {
            let ch = self.current_char.unwrap();
            
            if ch == closing {
                self.advance(); // consume the closing bracket
                break;
            }
            
            if ch.is_whitespace() {
                self.advance();
                continue;
            }
            
            if ch == ';' {
                self.skip_comment();
                continue;
            }
            
            tokens.push(self.next_token()?);
        }
        
        Ok(tokens)
    }

    fn read_string(&mut self) -> Result<String, ParseError> {
        self.advance(); // skip opening quote
        let mut value = String::new();
        
        while let Some(ch) = self.current_char {
            if ch == '"' {
                self.advance();
                return Ok(value);
            }
            
            if ch == '\\' {
                self.advance();
                match self.current_char {
                    Some('n') => value.push('\n'),
                    Some('t') => value.push('\t'),
                    Some('r') => value.push('\r'),
                    Some('\\') => value.push('\\'),
                    Some('"') => value.push('"'),
                    Some(c) => value.push(c),
                    None => return Err(ParseError::LexerError("Unexpected end of string".to_string())),
                }
            } else {
                value.push(ch);
            }
            
            self.advance();
        }
        
        Err(ParseError::LexerError("Unterminated string".to_string()))
    }

    fn read_number(&mut self) -> Result<f64, ParseError> {
        let mut value = String::new();
        let mut has_dot = false;
        
        while let Some(ch) = self.current_char {
            if ch.is_ascii_digit() {
                value.push(ch);
                self.advance();
            } else if ch == '.' && !has_dot {
                has_dot = true;
                value.push(ch);
                self.advance();
            } else {
                break;
            }
        }
        
        value.parse::<f64>()
            .map_err(|_| ParseError::LexerError(format!("Invalid number: {}", value)))
    }

    fn read_identifier(&mut self) -> String {
        let mut value = String::new();
        
        while let Some(ch) = self.current_char {
            if ch.is_alphanumeric() || ch == '_' || ch == '-' || ch == '?' || ch == '!' {
                value.push(ch);
                self.advance();
            } else {
                break;
            }
        }
        
        value
    }

    fn read_operator(&mut self) -> String {
        let mut value = String::new();
        
        while let Some(ch) = self.current_char {
            if self.is_operator_char(ch) {
                value.push(ch);
                self.advance();
            } else {
                break;
            }
        }
        
        value
    }

    fn is_operator_char(&self, ch: char) -> bool {
        matches!(ch, '+' | '-' | '*' | '/' | '=' | '<' | '>' | '!' | '&' | '|' | '%' | '^' | '~')
    }

    fn skip_comment(&mut self) {
        while let Some(ch) = self.current_char {
            if ch == '\n' {
                break;
            }
            self.advance();
        }
    }

    fn advance(&mut self) {
        self.position += 1;
        self.current_char = self.input.chars().nth(self.position);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_tokenization() {
        let mut lexer = Lexer::new("(+ 1 2)".to_string());
        let tokens = lexer.tokenize().unwrap();
        
        match &tokens[0] {
            Term::Bracket(BracketType::Paren, inner) => {
                assert_eq!(inner.len(), 3);
            }
            _ => panic!("Expected bracket"),
        }
    }

    #[test]
    fn test_string_literal() {
        let mut lexer = Lexer::new("\"hello world\"".to_string());
        let tokens = lexer.tokenize().unwrap();
        
        match &tokens[0] {
            Term::Literal(LiteralValue::String(s)) => {
                assert_eq!(s, "hello world");
            }
            _ => panic!("Expected string literal"),
        }
    }

    #[test]
    fn test_number_literal() {
        let mut lexer = Lexer::new("42.5".to_string());
        let tokens = lexer.tokenize().unwrap();
        
        match &tokens[0] {
            Term::Literal(LiteralValue::Number(n)) => {
                assert_eq!(*n, 42.5);
            }
            _ => panic!("Expected number literal"),
        }
    }
}