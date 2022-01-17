use std::{str::from_utf8};

#[derive(Debug)]
pub enum Token {
    OpenParen,
    CloseParen,
    OpenCurly,
    CloseCurly,
    OpenBracket,
    CloseBracket,
    SemiColon,
    Colon,
    Comma,
    NewLine,
    // This is mixing ideas here
    // I am making this rust specific
    // But I'm not too worried about that right now.
    Comment((usize, usize)),
    Spaces((usize, usize)),
    String((usize, usize)),
    Integer((usize, usize)),
    Float((usize, usize)),
    Atom((usize, usize)),
}

#[derive(Debug, Clone)]
pub struct Tokenizer {
    pub position: usize,
}

#[derive(Debug)]
pub enum RustSpecific {
    // As,
    // Break,
    // Const,
    // Continue,
    // Crate,
    // Else,
    // Enum,
    // Extern,
    // False,
    // Fn,
    // For,
    // If,
    // Impl,
    // In,
    // Let,
    // Loop,
    // Match,
    // Mod,
    // Move,
    // Mut,
    // Pub,
    // Ref,
    // Return,
    // SelfValue,
    // SelfType,
    // Static,
    // Struct,
    // Super,
    // Trait,
    // True,
    // Type,
    // Unsafe,
    // Use,
    // Where,
    // While,
    Token(Token),
    Keyword((usize, usize)),
}


static ZERO: u8 = b'0';
static NINE: u8 = b'9';
static SPACE: u8 = b' ';
static NEW_LINE: u8 = b'\n';
static DOUBLE_QUOTE: u8 = b'"';
static OPEN_PAREN: u8 = b'(';
static CLOSE_PAREN: u8 = b')';
static PERIOD: u8 = b'.';





impl<'a> Tokenizer {
    pub fn new() -> Tokenizer {
        Tokenizer {
            position: 0,
        }
    }

    fn peek(&self, input_bytes: &[u8]) -> Option<u8> {
        if self.position + 1 < input_bytes.len() {
            Some(input_bytes[self.position + 1])
        } else {
            None
        }
    }

    fn is_comment_start(&self, input_bytes: &[u8]) -> bool {
        input_bytes[self.position] == b'/' && self.peek(input_bytes) == Some(b'/')
    }
    
    fn parse_comment(&mut self, input_bytes: &[u8]) -> Token {
        let start = self.position;
        while !self.at_end(input_bytes) && !self.is_newline(input_bytes) {
            self.consume();
        }
        // self.consume();
        Token::Comment((start, self.position))
    }

    pub fn consume(&mut self) {
        self.position += 1;
    }

    pub fn current_byte(&self, input_bytes: &[u8]) -> u8 {
        input_bytes[self.position]
    }

    pub fn is_space(&self, input_bytes: &[u8]) -> bool {
        self.current_byte(input_bytes) == SPACE
    }

    pub fn at_end(&self, input_bytes: &[u8]) -> bool {
        self.position >= input_bytes.len()
    }

    pub fn is_quote(&self, input_bytes: &[u8]) -> bool {
        self.current_byte(input_bytes) == DOUBLE_QUOTE
    }

    pub fn parse_string(&mut self, input_bytes: &[u8]) -> Token {// skip open quote
        let start = self.position;
        self.consume();
        while !self.at_end(input_bytes) && !self.is_quote(input_bytes) {
            self.consume();
        }
        // TODO: Deal with escapes
        if !self.at_end(input_bytes) {
            self.consume();
        }
        Token::String((start, self.position))
    }

    pub fn is_open_paren(&self, input_bytes: &[u8]) -> bool {
        self.current_byte(input_bytes) == OPEN_PAREN
    }

    pub fn is_close_paren(&self, input_bytes: &[u8]) -> bool {
        self.current_byte(input_bytes) == CLOSE_PAREN
    }

    pub fn is_open_curly(&self, input_bytes: &[u8]) -> bool {
        self.current_byte(input_bytes) == b'{'
    }

    pub fn is_close_curly(&self, input_bytes: &[u8]) -> bool {
        self.current_byte(input_bytes) == b'}'
    }

    pub fn is_open_bracket(&self, input_bytes: &[u8]) -> bool {
        self.current_byte(input_bytes) == b'['
    }

    pub fn is_close_bracket(&self, input_bytes: &[u8]) -> bool {
        self.current_byte(input_bytes) == b']'
    }

    pub fn parse_spaces(&mut self, input_bytes: &[u8]) -> Token {
        let start = self.position;
        while !self.at_end(input_bytes) && self.is_space(input_bytes) {
            self.consume();
        }
        Token::Spaces((start, self.position))

    }

    pub fn is_valid_number_char(&mut self, input_bytes: &[u8]) -> bool {
        self.current_byte(input_bytes) >= ZERO && self.current_byte(input_bytes) <= NINE
    }

    pub fn parse_number(&mut self, input_bytes: &[u8]) -> Token {
        let mut is_float = false;
        let start = self.position;
        while !self.at_end(input_bytes) && (self.is_valid_number_char(input_bytes) || self.current_byte(input_bytes) == PERIOD) {
            // Need to handle making sure there is only one "."
            if self.current_byte(input_bytes) == PERIOD {
                is_float = true;
            }
            self.consume();
        }
        if is_float {
            Token::Float((start,self.position))
        } else {
            Token::Integer((start, self.position))
        }
    }

    pub fn parse_identifier(&mut self, input_bytes: &[u8]) -> Token {
        let start = self.position;
        while !self.at_end(input_bytes)
                && !self.is_space(input_bytes) 
                && !self.is_open_paren(input_bytes)
                && !self.is_close_paren(input_bytes)
                && !self.is_open_curly(input_bytes)
                && !self.is_close_curly(input_bytes)
                && !self.is_open_bracket(input_bytes)
                && !self.is_close_bracket(input_bytes)
                && !self.is_semi_colon(input_bytes)
                && !self.is_colon(input_bytes)
                && !self.is_comma(input_bytes)
                && !self.is_newline(input_bytes) 
                && !self.is_quote(input_bytes) {
            self.consume();
        }
        Token::Atom((start, self.position))
    }

    pub fn parse_single(&mut self, input_bytes: &[u8]) -> Option<Token> {
        
        if self.at_end(input_bytes) {
            return None
        }
        let result = if self.is_space(input_bytes) {
            self.parse_spaces(input_bytes)
        } else if self.is_newline(input_bytes) {
            self.consume();
            Token::NewLine
        } else if self.is_comment_start(input_bytes) {
            self.parse_comment(input_bytes)
        } else if self.is_open_paren(input_bytes) {
            // println!("open paren");
            self.consume();
            Token::OpenParen
        } else if self.is_close_paren(input_bytes) {
            // println!("close paren");
            self.consume();
            Token::CloseParen
        } else if self.is_valid_number_char(input_bytes) {
            // println!("number");
            self.parse_number(input_bytes)
        } else if self.is_quote(input_bytes) {
            // println!("string");
            self.parse_string(input_bytes)
        } else if self.is_semi_colon(input_bytes) {
            // println!("semicolon");
            self.consume();
            Token::SemiColon
        } else if self.is_comma(input_bytes) {
            self.consume();
            Token::Comma
        } else if self.is_colon(input_bytes) {
            // println!("colon");
            self.consume();
            Token::Colon
        } else if self.is_open_curly(input_bytes) {
            // println!("open curly");
            self.consume();
            Token::OpenCurly
        } else if self.is_close_curly(input_bytes) {
            // println!("close curly");
            self.consume();
            Token::CloseCurly
        } else if self.is_open_bracket(input_bytes) {
            // println!("open bracket");
            self.consume();
            Token::OpenBracket
        } else if self.is_close_bracket(input_bytes) {
            // println!("close bracket");
            self.consume();
            Token::CloseBracket
        } else {
            // println!("identifier");
            self.parse_identifier(input_bytes)
        };
        Some(result)
    }

    pub fn is_semi_colon(&self, input_bytes: &[u8]) -> bool {
        self.current_byte(input_bytes) == b';'
    }

    pub fn is_colon(&self, input_bytes: &[u8]) -> bool {
        self.current_byte(input_bytes) == b':'
    }

    pub fn is_newline(&self, input_bytes: &[u8]) -> bool {
        self.current_byte(input_bytes) == NEW_LINE
    }

    pub fn is_comma(&self, input_bytes: &[u8]) -> bool {
        self.current_byte(input_bytes) == b','
    }

    pub fn get_line(&mut self, input_bytes: &[u8]) -> Vec<Token> {
        let mut result = Vec::new();
        while !self.at_end(input_bytes) && !self.is_newline(input_bytes) {
            if let Some(token) = self.parse_single(input_bytes) {
                result.push(token);
            }
        }
        result
    }

    pub fn skip_lines(&mut self, n: usize, input_bytes: &[u8]) -> &mut Self {
        for _ in 0..n {
            while !self.at_end(input_bytes) && !self.is_newline(input_bytes) {
                self.consume();
            }
            if !self.at_end(input_bytes) {
                self.consume();
            }
        }
        self
    }

}

pub fn rust_specific_pass(token: Token, input_bytes: &[u8]) -> RustSpecific {

        match token {
            Token::Atom((s, e)) => {
                let text = from_utf8(&input_bytes[s..e]).unwrap();
                match text {
                    "as" => RustSpecific::Keyword((s, e)),
                    "break" => RustSpecific::Keyword((s, e)),
                    "const" => RustSpecific::Keyword((s, e)),
                    "continue" => RustSpecific::Keyword((s, e)),
                    "crate" => RustSpecific::Keyword((s, e)),
                    "else" => RustSpecific::Keyword((s, e)),
                    "enum" => RustSpecific::Keyword((s, e)),
                    "extern" => RustSpecific::Keyword((s, e)),
                    "false" => RustSpecific::Keyword((s, e)),
                    "fn" => RustSpecific::Keyword((s, e)),
                    "for" => RustSpecific::Keyword((s, e)),
                    "if" => RustSpecific::Keyword((s, e)),
                    "impl" => RustSpecific::Keyword((s, e)),
                    "in" => RustSpecific::Keyword((s, e)),
                    "let" => RustSpecific::Keyword((s, e)),
                    "loop" => RustSpecific::Keyword((s, e)),
                    "match" => RustSpecific::Keyword((s, e)),
                    "mod" => RustSpecific::Keyword((s, e)),
                    "move" => RustSpecific::Keyword((s, e)),
                    "mut" => RustSpecific::Keyword((s, e)),
                    "pub" => RustSpecific::Keyword((s, e)),
                    "ref" => RustSpecific::Keyword((s, e)),
                    "return" => RustSpecific::Keyword((s, e)),
                    "self" => RustSpecific::Keyword((s, e)),
                    "Self" => RustSpecific::Keyword((s, e)),
                    "static" => RustSpecific::Keyword((s, e)),
                    "struct" => RustSpecific::Keyword((s, e)),
                    "super" => RustSpecific::Keyword((s, e)),
                    "trait" => RustSpecific::Keyword((s, e)),
                    "true" => RustSpecific::Keyword((s, e)),
                    "type" => RustSpecific::Keyword((s, e)),
                    "unsafe" => RustSpecific::Keyword((s, e)),
                    "use" => RustSpecific::Keyword((s, e)),
                    "where" => RustSpecific::Keyword((s, e)),
                    "while" => RustSpecific::Keyword((s, e)),
                    _ => RustSpecific::Token(token)
                }
            } 
           
            t => RustSpecific::Token(t)
        }
}