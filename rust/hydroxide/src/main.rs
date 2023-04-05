use std::str::from_utf8;



// I could do this without allocating
#[derive(Debug, Clone)]
pub enum Token {
    Tag(Vec<Token>),
    Variable(Vec<Token>),
    Identifier(String),
    String(String),
    Dot,
    Pipe,
    Colon,
    Other(String),
}



#[derive(Debug, Clone)]
pub struct Tokenizer {
    pub position: usize,
}

// This is a tokenizer for liquid templates
// A tag looks like {% tag_name %}
// A variable looks like {{ variable_name }}


impl Tokenizer {
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

    pub fn consume(&mut self) {
        self.position += 1;
    }

    pub fn move_back(&mut self) {
        self.position -= 1;
    }

    pub fn consume_spaces(&mut self, input_bytes: &[u8]) {
        while !self.at_end(input_bytes) && self.is_space(input_bytes) {
            self.consume();
        }
    }

    pub fn trim_right_spaces(&mut self, input_bytes: &[u8]) {
        while !self.at_end(input_bytes) && self.is_space(input_bytes) {
            self.consume();
        }
    }

    pub fn current_byte(&self, input_bytes: &[u8]) -> u8 {
        input_bytes[self.position]
    }

    pub fn is_space(&self, input_bytes: &[u8]) -> bool {
        self.current_byte(input_bytes) == b' '
    }

    pub fn at_end(&self, input_bytes: &[u8]) -> bool {
        self.position >= input_bytes.len()
    }

    pub fn is_open_curly(&self, input_bytes: &[u8]) -> bool {
        self.current_byte(input_bytes) == b'{'
    }

    pub fn is_close_curly(&self, input_bytes: &[u8]) -> bool {
        self.current_byte(input_bytes) == b'}'
    }

    pub fn is_percent(&self, input_bytes: &[u8]) -> bool {
        self.current_byte(input_bytes) == b'%'
    }

    pub fn valid_identifier_char(&self, input_bytes: &[u8]) -> bool {
        self.is_alphanumeric(input_bytes) || self.is_underscore(input_bytes) || self.is_hyphen(input_bytes)
    }

    pub fn parse_tag(&mut self, input_bytes: &[u8]) -> Token {
        let mut tag = String::new();
        self.consume();
        self.consume();
        self.consume_spaces(input_bytes);
        while !self.at_end(input_bytes) && !(self.is_percent(input_bytes) && self.peek(input_bytes) == Some(b'}')) {
            tag.push(self.current_byte(input_bytes) as char);
            self.consume();
        }
        let tag = tag.trim_end();
        self.consume();
        self.consume();
        let mut tokenizer = Tokenizer::new();
        let tokens = tokenizer.parse_inner(tag.as_bytes());
        Token::Tag(tokens)
    }

    pub fn parse_inner(&mut self, input_bytes: &[u8]) -> Vec<Token> {
        let mut tokens = Vec::new();
        while !self.at_end(input_bytes) {
            if self.is_space(input_bytes) {
                self.consume_spaces(input_bytes);
            } else if self.is_period(input_bytes) {
                self.consume();
                tokens.push(Token::Dot);
            } else if self.is_pipe(input_bytes) {
                self.consume();
                tokens.push(Token::Pipe);
            } else if self.is_colon(input_bytes) {
                self.consume();
                tokens.push(Token::Colon);
            } else if self.is_quote(input_bytes) {
                tokens.push(self.parse_string(input_bytes));
            } else if self.valid_identifier_char(input_bytes) {
                let mut identifier = String::new();
                while !self.at_end(input_bytes) && self.valid_identifier_char(input_bytes) {
                    identifier.push(self.current_byte(input_bytes) as char);
                    self.consume();
                }
                tokens.push(Token::Identifier(identifier));
            } else {
                panic!("Unexpected character: {} {} {:?}", self.current_byte(input_bytes) as char, self.position, from_utf8(input_bytes));
            }
        }
        tokens
    }

    pub fn parse_variable(&mut self, input_bytes: &[u8]) -> Token {
        let mut variable = String::new();
        self.consume();
        self.consume();
        self.consume_spaces(input_bytes);
        while !self.at_end(input_bytes) && !self.is_close_curly(input_bytes) {
            variable.push(self.current_byte(input_bytes) as char);
            self.consume();
        }
        let variable = variable.trim_end();
        self.consume();
        self.consume();
        let mut tokenizer = Tokenizer::new();
        let tokens = tokenizer.parse_inner(variable.as_bytes());
        Token::Variable(tokens)
    }

    pub fn parse(&mut self, input_bytes: &[u8]) -> Vec<Token> {
        let mut tokens = Vec::new();
        while !self.at_end(input_bytes) {
            if self.is_open_curly(input_bytes) && self.peek(input_bytes) == Some(b'{') {
                tokens.push(self.parse_variable(input_bytes));
            } else if self.is_open_curly(input_bytes) && self.peek(input_bytes) == Some(b'%') {
                tokens.push(self.parse_tag(input_bytes));
            } else {
                let mut other = String::new();
                while !self.at_end(input_bytes) && !self.is_open_curly(input_bytes) {
                    other.push(self.current_byte(input_bytes) as char);
                    self.consume();
                }
                tokens.push(Token::Other(other));
            }
        }
        tokens
    }

    fn is_alphanumeric(&self, input_bytes: &[u8]) -> bool {
        self.current_byte(input_bytes).is_ascii_alphanumeric()
    }

    fn is_underscore(&self, input_bytes: &[u8]) -> bool {
        self.current_byte(input_bytes) == b'_'
    }

    fn is_hyphen(&self, input_bytes: &[u8]) -> bool {
        self.current_byte(input_bytes) == b'-'
    }

    fn is_period(&self, input_bytes: &[u8]) -> bool {
        self.current_byte(input_bytes) == b'.'
    }

    fn is_pipe(&self, input_bytes: &[u8]) -> bool {
        self.current_byte(input_bytes) == b'|'
    }

    fn is_colon(&self, input_bytes: &[u8]) -> bool {
        self.current_byte(input_bytes) == b':'
    }

    fn is_quote(&self, input_bytes: &[u8]) -> bool {
        self.current_byte(input_bytes) == b'"' || self.current_byte(input_bytes) == b'\''
    }

    fn parse_string(&mut self, input_bytes: &[u8]) -> Token {
        let mut string = String::new();
        let quote = self.current_byte(input_bytes);
        self.consume();
        // TODO: Deal with escaped quotes
        while !self.at_end(input_bytes) && self.current_byte(input_bytes) != quote {
            string.push(self.current_byte(input_bytes) as char);
            self.consume();
        }
        self.consume();
        Token::String(string)
    }

}



fn main() {
    let mut  tokenizer = Tokenizer::new();
    // open file in ./resources/product.liquid
    let input = std::fs::read_to_string("./resources/product.liquid").unwrap();
    let tokens = tokenizer.parse(input.as_bytes());
    for token in tokens {
        if let Token::Other(_) = token {
            continue;
        }
        println!("{:?}", token);
    }

}
