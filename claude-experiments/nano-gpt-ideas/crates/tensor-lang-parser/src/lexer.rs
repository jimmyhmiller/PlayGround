#[derive(Debug, Clone, PartialEq)]
pub enum Token {
    // Keywords
    Fn,
    Let,
    DimKw,

    // Literals
    Ident(String),
    Number(f64),

    // Punctuation
    LParen,
    RParen,
    LBracket,
    RBracket,
    LBrace,
    RBrace,
    Comma,
    Colon,
    Eq,
    Star,
    Plus,
    Minus,

    Eof,
}

pub struct Lexer {
    input: Vec<char>,
    pos: usize,
}

impl Lexer {
    pub fn new(input: &str) -> Self {
        Lexer {
            input: input.chars().collect(),
            pos: 0,
        }
    }

    pub fn tokenize(&mut self) -> Vec<Token> {
        let mut tokens = Vec::new();
        loop {
            let tok = self.next_token();
            if tok == Token::Eof {
                tokens.push(tok);
                break;
            }
            tokens.push(tok);
        }
        tokens
    }

    fn peek(&self) -> Option<char> {
        self.input.get(self.pos).copied()
    }

    fn advance(&mut self) -> Option<char> {
        let ch = self.input.get(self.pos).copied();
        self.pos += 1;
        ch
    }

    fn skip_whitespace_and_comments(&mut self) {
        while let Some(ch) = self.peek() {
            if ch.is_whitespace() {
                self.advance();
            } else if ch == '/' && self.input.get(self.pos + 1) == Some(&'/') {
                // Line comment
                while let Some(c) = self.peek() {
                    if c == '\n' {
                        break;
                    }
                    self.advance();
                }
            } else {
                break;
            }
        }
    }

    fn read_number(&mut self) -> Token {
        let start = self.pos;
        while let Some(ch) = self.peek() {
            if ch.is_ascii_digit() || ch == '.' {
                self.advance();
            } else {
                break;
            }
        }
        let s: String = self.input[start..self.pos].iter().collect();
        Token::Number(s.parse().unwrap())
    }

    fn read_ident(&mut self) -> Token {
        let start = self.pos;
        while let Some(ch) = self.peek() {
            if ch.is_alphanumeric() || ch == '_' {
                self.advance();
            } else {
                break;
            }
        }
        let s: String = self.input[start..self.pos].iter().collect();
        match s.as_str() {
            "fn" => Token::Fn,
            "let" => Token::Let,
            "dim" => Token::DimKw,
            _ => Token::Ident(s),
        }
    }

    fn next_token(&mut self) -> Token {
        self.skip_whitespace_and_comments();

        let Some(ch) = self.peek() else {
            return Token::Eof;
        };

        match ch {
            '(' => { self.advance(); Token::LParen }
            ')' => { self.advance(); Token::RParen }
            '[' => { self.advance(); Token::LBracket }
            ']' => { self.advance(); Token::RBracket }
            '{' => { self.advance(); Token::LBrace }
            '}' => { self.advance(); Token::RBrace }
            ',' => { self.advance(); Token::Comma }
            ':' => { self.advance(); Token::Colon }
            '=' => { self.advance(); Token::Eq }
            '*' => { self.advance(); Token::Star }
            '+' => { self.advance(); Token::Plus }
            '-' => { self.advance(); Token::Minus }
            c if c.is_ascii_digit() => self.read_number(),
            c if c.is_alphabetic() || c == '_' => self.read_ident(),
            _ => panic!("unexpected character: {ch}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_tokens() {
        let mut lexer = Lexer::new("let x = add(a, b)");
        let tokens = lexer.tokenize();
        assert_eq!(tokens, vec![
            Token::Let,
            Token::Ident("x".into()),
            Token::Eq,
            Token::Ident("add".into()),
            Token::LParen,
            Token::Ident("a".into()),
            Token::Comma,
            Token::Ident("b".into()),
            Token::RParen,
            Token::Eof,
        ]);
    }

    #[test]
    fn test_fn_def() {
        let mut lexer = Lexer::new("fn softmax(x) { exp(x) }");
        let tokens = lexer.tokenize();
        assert_eq!(tokens, vec![
            Token::Fn,
            Token::Ident("softmax".into()),
            Token::LParen,
            Token::Ident("x".into()),
            Token::RParen,
            Token::LBrace,
            Token::Ident("exp".into()),
            Token::LParen,
            Token::Ident("x".into()),
            Token::RParen,
            Token::RBrace,
            Token::Eof,
        ]);
    }

    #[test]
    fn test_named_arg() {
        let mut lexer = Lexer::new("sum(x, axis: 1)");
        let tokens = lexer.tokenize();
        assert_eq!(tokens, vec![
            Token::Ident("sum".into()),
            Token::LParen,
            Token::Ident("x".into()),
            Token::Comma,
            Token::Ident("axis".into()),
            Token::Colon,
            Token::Number(1.0),
            Token::RParen,
            Token::Eof,
        ]);
    }

    #[test]
    fn test_array_literal() {
        let mut lexer = Lexer::new("load([10, 32])");
        let tokens = lexer.tokenize();
        assert_eq!(tokens, vec![
            Token::Ident("load".into()),
            Token::LParen,
            Token::LBracket,
            Token::Number(10.0),
            Token::Comma,
            Token::Number(32.0),
            Token::RBracket,
            Token::RParen,
            Token::Eof,
        ]);
    }

    #[test]
    fn test_comments() {
        let mut lexer = Lexer::new("// this is a comment\nlet x = 1");
        let tokens = lexer.tokenize();
        assert_eq!(tokens, vec![
            Token::Let,
            Token::Ident("x".into()),
            Token::Eq,
            Token::Number(1.0),
            Token::Eof,
        ]);
    }
}
