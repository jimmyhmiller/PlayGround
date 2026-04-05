#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenType {
    // Single-character tokens
    LeftParen,
    RightParen,
    LeftBrace,
    RightBrace,
    Comma,
    Dot,
    Minus,
    Plus,
    Semicolon,
    Slash,
    Star,
    // One or two character tokens
    Bang,
    BangEqual,
    Equal,
    EqualEqual,
    Greater,
    GreaterEqual,
    Less,
    LessEqual,
    // Literals
    Identifier,
    String,
    Number,
    // Keywords
    And,
    Class,
    Else,
    False,
    For,
    Fun,
    If,
    Nil,
    Or,
    Print,
    Return,
    Super,
    This,
    Var,
    While,
    True,
    // Special
    Error,
    Eof,
}

#[derive(Debug, Clone)]
pub struct Token {
    pub token_type: TokenType,
    pub lexeme: String,
    pub line: usize,
}

impl Token {
    pub fn synthetic(text: &str) -> Token {
        Token {
            token_type: TokenType::Identifier,
            lexeme: text.to_string(),
            line: 0,
        }
    }
}

pub struct Scanner {
    source: Vec<u8>,
    start: usize,
    current: usize,
    line: usize,
}

impl Scanner {
    pub fn new(source: &str) -> Self {
        Scanner {
            source: source.as_bytes().to_vec(),
            start: 0,
            current: 0,
            line: 1,
        }
    }

    pub fn scan_token(&mut self) -> Token {
        self.skip_whitespace();
        self.start = self.current;

        if self.is_at_end() {
            return self.make_token(TokenType::Eof);
        }

        let c = self.advance();

        if is_alpha(c) {
            return self.identifier();
        }
        if is_digit(c) {
            return self.number();
        }

        match c {
            b'(' => self.make_token(TokenType::LeftParen),
            b')' => self.make_token(TokenType::RightParen),
            b'{' => self.make_token(TokenType::LeftBrace),
            b'}' => self.make_token(TokenType::RightBrace),
            b';' => self.make_token(TokenType::Semicolon),
            b',' => self.make_token(TokenType::Comma),
            b'.' => self.make_token(TokenType::Dot),
            b'-' => self.make_token(TokenType::Minus),
            b'+' => self.make_token(TokenType::Plus),
            b'/' => self.make_token(TokenType::Slash),
            b'*' => self.make_token(TokenType::Star),
            b'!' => {
                let t = if self.match_char(b'=') {
                    TokenType::BangEqual
                } else {
                    TokenType::Bang
                };
                self.make_token(t)
            }
            b'=' => {
                let t = if self.match_char(b'=') {
                    TokenType::EqualEqual
                } else {
                    TokenType::Equal
                };
                self.make_token(t)
            }
            b'<' => {
                let t = if self.match_char(b'=') {
                    TokenType::LessEqual
                } else {
                    TokenType::Less
                };
                self.make_token(t)
            }
            b'>' => {
                let t = if self.match_char(b'=') {
                    TokenType::GreaterEqual
                } else {
                    TokenType::Greater
                };
                self.make_token(t)
            }
            b'"' => self.string(),
            _ => self.error_token("Unexpected character."),
        }
    }

    fn identifier(&mut self) -> Token {
        while !self.is_at_end() && (is_alpha(self.peek()) || is_digit(self.peek())) {
            self.advance();
        }
        let token_type = self.identifier_type();
        self.make_token(token_type)
    }

    fn identifier_type(&self) -> TokenType {
        let text = &self.source[self.start..self.current];
        match text {
            b"and" => TokenType::And,
            b"class" => TokenType::Class,
            b"else" => TokenType::Else,
            b"false" => TokenType::False,
            b"for" => TokenType::For,
            b"fun" => TokenType::Fun,
            b"if" => TokenType::If,
            b"nil" => TokenType::Nil,
            b"or" => TokenType::Or,
            b"print" => TokenType::Print,
            b"return" => TokenType::Return,
            b"super" => TokenType::Super,
            b"this" => TokenType::This,
            b"true" => TokenType::True,
            b"var" => TokenType::Var,
            b"while" => TokenType::While,
            _ => TokenType::Identifier,
        }
    }

    fn number(&mut self) -> Token {
        while !self.is_at_end() && is_digit(self.peek()) {
            self.advance();
        }

        if !self.is_at_end() && self.peek() == b'.' {
            if self.current + 1 < self.source.len() && is_digit(self.source[self.current + 1]) {
                self.advance(); // consume '.'
                while !self.is_at_end() && is_digit(self.peek()) {
                    self.advance();
                }
            }
        }

        self.make_token(TokenType::Number)
    }

    fn string(&mut self) -> Token {
        while !self.is_at_end() && self.peek() != b'"' {
            if self.peek() == b'\n' {
                self.line += 1;
            }
            self.advance();
        }

        if self.is_at_end() {
            return self.error_token("Unterminated string.");
        }

        self.advance(); // closing quote
        self.make_token(TokenType::String)
    }

    fn skip_whitespace(&mut self) {
        loop {
            if self.is_at_end() {
                return;
            }
            match self.peek() {
                b' ' | b'\r' | b'\t' => {
                    self.advance();
                }
                b'\n' => {
                    self.line += 1;
                    self.advance();
                }
                b'/' => {
                    if self.current + 1 < self.source.len()
                        && self.source[self.current + 1] == b'/'
                    {
                        while !self.is_at_end() && self.peek() != b'\n' {
                            self.advance();
                        }
                    } else {
                        return;
                    }
                }
                _ => return,
            }
        }
    }

    fn is_at_end(&self) -> bool {
        self.current >= self.source.len()
    }

    fn advance(&mut self) -> u8 {
        let c = self.source[self.current];
        self.current += 1;
        c
    }

    fn peek(&self) -> u8 {
        self.source[self.current]
    }

    fn match_char(&mut self, expected: u8) -> bool {
        if self.is_at_end() {
            return false;
        }
        if self.source[self.current] != expected {
            return false;
        }
        self.current += 1;
        true
    }

    fn make_token(&self, token_type: TokenType) -> Token {
        let lexeme = String::from_utf8_lossy(&self.source[self.start..self.current]).to_string();
        Token {
            token_type,
            lexeme,
            line: self.line,
        }
    }

    fn error_token(&self, message: &str) -> Token {
        Token {
            token_type: TokenType::Error,
            lexeme: message.to_string(),
            line: self.line,
        }
    }
}

fn is_digit(c: u8) -> bool {
    c.is_ascii_digit()
}

fn is_alpha(c: u8) -> bool {
    c.is_ascii_alphabetic() || c == b'_'
}
