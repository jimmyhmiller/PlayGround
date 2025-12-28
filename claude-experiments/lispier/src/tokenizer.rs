use crate::token::{Token, TokenType};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum TokenizerError {
    #[error("unexpected character '{0}' at line {1}, column {2}")]
    UnexpectedCharacter(char, usize, usize),

    #[error("unterminated string at line {0}, column {1}")]
    UnterminatedString(usize, usize),
}

pub struct Tokenizer {
    chars: Vec<char>,
    start: usize,
    current: usize,
    line: usize,
    column: usize,
    start_column: usize,
}

impl Tokenizer {
    pub fn new(source: &str) -> Self {
        Self {
            chars: source.chars().collect(),
            start: 0,
            current: 0,
            line: 1,
            column: 1,
            start_column: 1,
        }
    }

    pub fn tokenize(&mut self) -> Result<Vec<Token>, TokenizerError> {
        let mut tokens = Vec::new();

        while !self.is_at_end() {
            self.start = self.current;
            self.start_column = self.column;
            if let Some(token) = self.scan_token()? {
                tokens.push(token);
            }
        }

        tokens.push(Token::new(TokenType::Eof, "", self.line, self.column));
        Ok(tokens)
    }

    fn scan_token(&mut self) -> Result<Option<Token>, TokenizerError> {
        let c = self.advance();

        match c {
            '(' => Ok(Some(self.make_token(TokenType::LeftParen))),
            ')' => Ok(Some(self.make_token(TokenType::RightParen))),
            '[' => Ok(Some(self.make_token(TokenType::LeftBracket))),
            ']' => Ok(Some(self.make_token(TokenType::RightBracket))),
            '{' => Ok(Some(self.make_token(TokenType::LeftBrace))),
            '}' => Ok(Some(self.make_token(TokenType::RightBrace))),
            ':' => {
                // Could be a keyword or a type annotation colon
                if self.is_alpha(self.peek()) || self.peek() == '_' || self.peek() == '-' {
                    self.keyword()
                } else {
                    Ok(Some(self.make_token(TokenType::Colon)))
                }
            }
            '^' => self.block_label(),
            '`' => Ok(Some(self.make_token(TokenType::Backtick))),
            '~' => {
                // Could be ~ (unquote) or ~@ (unquote-splice)
                if self.peek() == '@' {
                    self.advance(); // consume the @
                    Ok(Some(self.make_token(TokenType::TildeAt)))
                } else {
                    Ok(Some(self.make_token(TokenType::Tilde)))
                }
            }
            '"' => self.string(),
            ' ' | '\r' | '\t' => Ok(None),
            '\n' => {
                self.line += 1;
                self.column = 1;
                Ok(None)
            }
            ';' => {
                // Comment - skip until end of line
                while self.peek() != '\n' && !self.is_at_end() {
                    self.advance();
                }
                Ok(None)
            }
            _ => {
                if c.is_ascii_digit() || (c == '-' && self.peek().is_ascii_digit()) {
                    self.number()
                } else if self.is_symbol_start(c) {
                    self.symbol()
                } else {
                    Err(TokenizerError::UnexpectedCharacter(c, self.line, self.start_column))
                }
            }
        }
    }

    fn symbol(&mut self) -> Result<Option<Token>, TokenizerError> {
        // Symbols can contain letters, digits, and special chars: - _ . / < > ! ? * + =
        // They can also contain angle brackets for types like memref<128x128xf32>
        let mut depth: i32 = 0;

        while !self.is_at_end() {
            let c = self.peek();

            // Track angle bracket depth for types
            if c == '<' {
                depth += 1;
                self.advance();
                continue;
            }
            if c == '>' {
                // Check if this is part of -> or >= (not a closing angle bracket)
                let prev_char = if self.current > self.start {
                    Some(self.chars[self.current - 1])
                } else {
                    None
                };
                // Check if next char is = (for >=)
                let next_char = if self.current + 1 < self.chars.len() {
                    Some(self.chars[self.current + 1])
                } else {
                    None
                };
                let is_arrow = prev_char == Some('-');
                let is_gte = next_char == Some('=');

                if depth > 0 && !is_arrow && !is_gte {
                    depth -= 1;
                    self.advance();
                    continue;
                }
                // If it's part of -> or >= inside brackets, allow it and continue
                if depth > 0 && (is_arrow || is_gte) {
                    self.advance();
                    continue;
                }
                // Allow > as part of normal symbols (e.g. ->)
            }

            // Inside angle brackets, allow more characters
            if depth > 0 {
                if c == ' '
                    || c == ','
                    || c == '('
                    || c == ')'
                    || c == '['
                    || c == ']'
                    || c == ':'
                    || c == '='
                    || c == '*'
                    || c == '+'
                    || c == '|'
                    || c == '-'
                    || c == '?'
                    || c.is_alphanumeric()
                    || c == '_'
                    || c == '#'
                    || c == '.'
                {
                    self.advance();
                    continue;
                }
                break;
            }

            // Normal symbol characters
            if self.is_symbol_char(c) {
                self.advance();
            } else {
                break;
            }
        }

        Ok(Some(self.make_token(TokenType::Symbol)))
    }

    fn number(&mut self) -> Result<Option<Token>, TokenizerError> {
        // Handle negative numbers - check if we started with '-'
        if self.current > self.start && self.chars[self.start] == '-' {
            if !self.peek().is_ascii_digit() {
                // Just a '-' symbol, treat as symbol
                self.current = self.start + 1;
                return self.symbol();
            }
        }

        while self.peek().is_ascii_digit() {
            self.advance();
        }

        // Look for decimal point
        if self.peek() == '.' && self.peek_next().is_ascii_digit() {
            self.advance(); // consume '.'
            while self.peek().is_ascii_digit() {
                self.advance();
            }
        }

        // Scientific notation
        if self.peek() == 'e' || self.peek() == 'E' {
            self.advance();
            if self.peek() == '+' || self.peek() == '-' {
                self.advance();
            }
            while self.peek().is_ascii_digit() {
                self.advance();
            }
        }

        Ok(Some(self.make_token(TokenType::Number)))
    }

    fn string(&mut self) -> Result<Option<Token>, TokenizerError> {
        let start_line = self.line;
        let start_col = self.start_column;

        while self.peek() != '"' && !self.is_at_end() {
            if self.peek() == '\n' {
                self.line += 1;
                self.column = 1;
            }
            if self.peek() == '\\' {
                self.advance(); // escape char
                if !self.is_at_end() {
                    self.advance(); // escaped char
                }
            } else {
                self.advance();
            }
        }

        if self.is_at_end() {
            return Err(TokenizerError::UnterminatedString(start_line, start_col));
        }

        self.advance(); // closing "

        Ok(Some(self.make_token(TokenType::String)))
    }

    fn keyword(&mut self) -> Result<Option<Token>, TokenizerError> {
        // Keywords can contain alphanumeric, dash, underscore, and period (for namespaced attrs like :llvm.emit_c_interface)
        while self.is_alphanumeric(self.peek())
            || self.peek() == '-'
            || self.peek() == '_'
            || self.peek() == '.'
        {
            self.advance();
        }
        Ok(Some(self.make_token(TokenType::Keyword)))
    }

    fn block_label(&mut self) -> Result<Option<Token>, TokenizerError> {
        while self.is_alphanumeric(self.peek()) || self.peek() == '_' {
            self.advance();
        }
        Ok(Some(self.make_token(TokenType::BlockLabel)))
    }

    fn make_token(&self, token_type: TokenType) -> Token {
        let lexeme: String = self.chars[self.start..self.current].iter().collect();
        Token::new(token_type, lexeme, self.line, self.start_column)
    }

    fn is_at_end(&self) -> bool {
        self.current >= self.chars.len()
    }

    fn advance(&mut self) -> char {
        let c = self.chars[self.current];
        self.current += 1;
        self.column += 1;
        c
    }

    fn peek(&self) -> char {
        if self.is_at_end() {
            '\0'
        } else {
            self.chars[self.current]
        }
    }

    fn peek_next(&self) -> char {
        if self.current + 1 >= self.chars.len() {
            '\0'
        } else {
            self.chars[self.current + 1]
        }
    }

    fn is_alpha(&self, c: char) -> bool {
        c.is_ascii_alphabetic() || c == '_'
    }

    fn is_alphanumeric(&self, c: char) -> bool {
        c.is_ascii_alphanumeric() || c == '_'
    }

    fn is_symbol_start(&self, c: char) -> bool {
        c.is_ascii_alphabetic()
            || c == '_'
            || c == '-'
            || c == '!'
            || c == '?'
            || c == '*'
            || c == '+'
            || c == '='
            || c == '<'
            || c == '>'
            || c == '@'
            || c == '%'
            || c == '.'  // Allow ... for vararg
    }

    fn is_symbol_char(&self, c: char) -> bool {
        c.is_ascii_alphanumeric()
            || c == '-'
            || c == '_'
            || c == '.'
            || c == '/'
            || c == '!'
            || c == '?'
            || c == '*'
            || c == '+'
            || c == '='
            || c == '<'
            || c == '>'
            || c == '@'
            || c == '%'
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;

    #[test]
    fn test_basic_tokens() {
        let mut tokenizer = Tokenizer::new("() [] {}");
        let tokens = tokenizer.tokenize().unwrap();

        assert_eq!(tokens.len(), 7); // 6 + EOF
        assert_eq!(tokens[0].token_type, TokenType::LeftParen);
        assert_eq!(tokens[1].token_type, TokenType::RightParen);
        assert_eq!(tokens[2].token_type, TokenType::LeftBracket);
        assert_eq!(tokens[3].token_type, TokenType::RightBracket);
        assert_eq!(tokens[4].token_type, TokenType::LeftBrace);
        assert_eq!(tokens[5].token_type, TokenType::RightBrace);
    }

    #[test]
    fn test_symbols() {
        let mut tokenizer = Tokenizer::new("arith.addi a/b foo-bar");
        let tokens = tokenizer.tokenize().unwrap();

        assert_eq!(tokens.len(), 4); // 3 + EOF
        assert_eq!(tokens[0].token_type, TokenType::Symbol);
        assert_eq!(tokens[0].lexeme, "arith.addi");
        assert_eq!(tokens[1].lexeme, "a/b");
        assert_eq!(tokens[2].lexeme, "foo-bar");
    }

    #[test]
    fn test_numbers() {
        let mut tokenizer = Tokenizer::new("42 3.14 -10 1.5e-3");
        let tokens = tokenizer.tokenize().unwrap();

        assert_eq!(tokens.len(), 5); // 4 + EOF
        assert_eq!(tokens[0].token_type, TokenType::Number);
        assert_eq!(tokens[0].lexeme, "42");
        assert_eq!(tokens[1].lexeme, "3.14");
        assert_eq!(tokens[2].lexeme, "-10");
        assert_eq!(tokens[3].lexeme, "1.5e-3");
    }

    #[test]
    fn test_strings() {
        let mut tokenizer = Tokenizer::new("\"hello\" \"world\\n\"");
        let tokens = tokenizer.tokenize().unwrap();

        assert_eq!(tokens.len(), 3); // 2 + EOF
        assert_eq!(tokens[0].token_type, TokenType::String);
        assert_eq!(tokens[0].lexeme, "\"hello\"");
        assert_eq!(tokens[1].lexeme, "\"world\\n\"");
    }

    #[test]
    fn test_keywords() {
        let mut tokenizer = Tokenizer::new(":foo :bar-baz");
        let tokens = tokenizer.tokenize().unwrap();

        assert_eq!(tokens.len(), 3); // 2 + EOF
        assert_eq!(tokens[0].token_type, TokenType::Keyword);
        assert_eq!(tokens[0].lexeme, ":foo");
        assert_eq!(tokens[1].lexeme, ":bar-baz");
    }

    #[test]
    fn test_block_labels() {
        let mut tokenizer = Tokenizer::new("^bb1 ^loop");
        let tokens = tokenizer.tokenize().unwrap();

        assert_eq!(tokens.len(), 3); // 2 + EOF
        assert_eq!(tokens[0].token_type, TokenType::BlockLabel);
        assert_eq!(tokens[0].lexeme, "^bb1");
        assert_eq!(tokens[1].lexeme, "^loop");
    }

    #[test]
    fn test_types_with_angle_brackets() {
        let mut tokenizer = Tokenizer::new("memref<128x128xf32> !llvm.ptr<i8>");
        let tokens = tokenizer.tokenize().unwrap();

        assert_eq!(tokens.len(), 3); // 2 + EOF
        assert_eq!(tokens[0].lexeme, "memref<128x128xf32>");
        assert_eq!(tokens[1].lexeme, "!llvm.ptr<i8>");
    }

    #[test]
    fn test_comments() {
        let mut tokenizer = Tokenizer::new("; This is a comment\n(def x 42)");
        let tokens = tokenizer.tokenize().unwrap();

        assert_eq!(tokens.len(), 6); // 5 + EOF
        assert_eq!(tokens[0].token_type, TokenType::LeftParen);
        assert_eq!(tokens[1].lexeme, "def");
        assert_eq!(tokens[5].token_type, TokenType::Eof);
    }

    #[test]
    fn test_complex_expression() {
        let mut tokenizer = Tokenizer::new("(arith.addi {:value 42} x y)");
        let tokens = tokenizer.tokenize().unwrap();

        assert_eq!(tokens.len(), 10); // 9 + EOF
        assert_eq!(tokens[0].token_type, TokenType::LeftParen);
        assert_eq!(tokens[1].lexeme, "arith.addi");
        assert_eq!(tokens[2].token_type, TokenType::LeftBrace);
        assert_eq!(tokens[3].lexeme, ":value");
        assert_eq!(tokens[9].token_type, TokenType::Eof);
    }

    #[test]
    fn test_affine_map_in_map() {
        let mut tokenizer = Tokenizer::new("{:map affine_map<(d0, d1) -> (d0 + d1)>}");
        let tokens = tokenizer.tokenize().unwrap();

        assert_eq!(tokens[0].token_type, TokenType::LeftBrace);
        assert_eq!(tokens[1].lexeme, ":map");
        assert_eq!(tokens[2].lexeme, "affine_map<(d0, d1) -> (d0 + d1)>");
        assert_eq!(tokens[3].token_type, TokenType::RightBrace);
    }

    #[test]
    fn test_affine_set_with_comparison() {
        let mut tokenizer = Tokenizer::new("{:condition affine_set<(d0) : (d0 >= 0)>}");
        let tokens = tokenizer.tokenize().unwrap();

        assert_eq!(tokens[0].token_type, TokenType::LeftBrace);
        assert_eq!(tokens[1].lexeme, ":condition");
        assert_eq!(tokens[2].lexeme, "affine_set<(d0) : (d0 >= 0)>");
        assert_eq!(tokens[3].token_type, TokenType::RightBrace);
    }
}
