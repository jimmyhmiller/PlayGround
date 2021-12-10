#[derive(Debug)]
pub enum Token<'a> {
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
    Comment(&'a str),
    Spaces(&'a str),
    String(&'a str),
    Integer(&'a str),
    Float(&'a str),
    Atom(&'a str),
}

#[derive(Debug)]
pub struct Tokenizer<'a> {
    input: &'a str,
    input_bytes: &'a [u8],
    position: usize,
}

#[derive(Debug)]
pub enum RustSpecific<'a> {
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
    Token(Token<'a>),
    Keyword(&'a str),
}


static ZERO: u8 = '0' as u8;
static NINE: u8 = '9' as u8;
static SPACE: u8 = ' ' as u8;
static NEW_LINE: u8 = '\n' as u8;
static COMMA: u8 = ',' as u8;
static DOUBLE_QUOTE: u8 = '"' as u8;
static OPEN_PAREN: u8 = '(' as u8;
static CLOSE_PAREN: u8 = ')' as u8;
static PERIOD: u8 = '.' as u8;

impl<'a> Iterator for Tokenizer<'a> {
    fn next(&mut self) -> Option<Token<'a>> {
        self.parse_single()
    }

    type Item = Token<'a>;
}




impl<'a> Tokenizer<'a> {
    pub fn new(input: &str) -> Tokenizer {
        Tokenizer {
            input: input,
            input_bytes: input.as_bytes(),
            position: 0,
        }
    }

    fn peek(&self) -> Option<u8> {
        if self.position < self.input_bytes.len() {
            Some(self.input_bytes[self.position])
        } else {
            None
        }
    }

    fn is_comment_start(&self) -> bool {
        self.input_bytes[self.position] == b'/' && self.peek() == Some(b'/')
    }
    
    fn parse_comment(&mut self) -> Token<'a> {
        let start = self.position;
        while !self.is_newline() {
            self.consume();
        }
        // self.consume();
        Token::Comment(&self.input[start..self.position])
    }

    pub fn consume(&mut self) -> () {
        self.position += 1;
    }

    pub fn current_byte(&self) -> u8 {
        self.input_bytes[self.position]
    }

    pub fn is_space(&self) -> bool {
        self.current_byte() == SPACE
    }

    pub fn at_end(&self) -> bool {
        self.input.len() == self.position
    }

    pub fn is_quote(&self) -> bool {
        self.current_byte() == DOUBLE_QUOTE
    }

    pub fn parse_string(&mut self) -> Token<'a> {// skip open quote
        let start = self.position;
        self.consume();
        while !self.at_end() && !self.is_quote() {
            self.consume();
        }
        // TODO: Deal with escapes
        self.consume(); // skip closing quote
        Token::String(&self.input[start..self.position])
    }

    pub fn is_open_paren(&self) -> bool {
        self.current_byte() == OPEN_PAREN
    }

    pub fn is_close_paren(&self) -> bool {
        self.current_byte() == CLOSE_PAREN
    }

    pub fn is_open_curly(&self) -> bool {
        self.current_byte() == '{' as u8
    }

    pub fn is_close_curly(&self) -> bool {
        self.current_byte() == '}' as u8
    }

    pub fn is_open_bracket(&self) -> bool {
        self.current_byte() == '[' as u8
    }

    pub fn is_close_bracket(&self) -> bool {
        self.current_byte() == ']' as u8
    }

    pub fn parse_spaces(&mut self) -> Token<'a> {
        let start = self.position;
        while !self.at_end() && self.is_space() {
            self.consume();
        }
        Token::Spaces(&self.input[start..self.position])

    }

    pub fn is_valid_number_char(&mut self) -> bool {
        self.current_byte() >= ZERO && self.current_byte() <= NINE
    }

    pub fn parse_number(&mut self) -> Token<'a> {
        let mut is_float = false;
        let start = self.position;
        while self.is_valid_number_char() || self.current_byte() == PERIOD {
            // Need to handle making sure there is only one "."
            if self.current_byte() == PERIOD {
                is_float = true;
            }
            self.consume();
        }
        if is_float {
            Token::Float(&self.input[start..self.position])
        } else {
            Token::Integer(&self.input[start..self.position])
        }
    }

    pub fn parse_identifier(&mut self) -> Token<'a> {
        let start = self.position;
        while !self.is_space() 
                && !self.is_open_paren()
                && !self.is_close_paren()
                && !self.is_open_curly()
                && !self.is_close_curly()
                && !self.is_open_bracket()
                && !self.is_close_bracket()
                && !self.is_semi_colon()
                && !self.is_colon()
                && !self.is_comma()
                && !self.is_newline() {
            self.consume()
        }
        // println!("{} {}", start, self.position);
        Token::Atom(&self.input[start..self.position])
    }

    pub fn parse_single(&mut self) -> Option<Token<'a>> {
        
        if self.at_end() {
            return None
        }
        let result = if self.is_space() {
            self.parse_spaces()
        } else if self.is_newline() {
            self.consume();
            Token::NewLine
        } else if self.is_comment_start() {
            self.parse_comment()
        } else if self.is_open_paren() {
            // println!("open paren");
            self.consume();
            Token::OpenParen
        } else if self.is_close_paren() {
            // println!("close paren");
            self.consume();
            Token::CloseParen
        } else if self.is_valid_number_char() {
            // println!("number");
            self.parse_number()
        } else if self.is_quote() {
            // println!("string");
            self.parse_string()
        } else if self.is_semi_colon() {
            // println!("semicolon");
            self.consume();
            Token::SemiColon
        } else if self.is_comma() {
            self.consume();
            Token::Comma
        } else if self.is_colon() {
            // println!("colon");
            self.consume();
            Token::Colon
        } else if self.is_open_curly() {
            // println!("open curly");
            self.consume();
            Token::OpenCurly
        } else if self.is_close_curly() {
            // println!("close curly");
            self.consume();
            Token::CloseCurly
        } else if self.is_open_bracket() {
            // println!("open bracket");
            self.consume();
            Token::OpenBracket
        } else if self.is_close_bracket() {
            // println!("close bracket");
            self.consume();
            Token::CloseBracket
        } else {
            // println!("identifier");
            self.parse_identifier()
        };
        Some(result)
    }

    pub fn is_semi_colon(&self) -> bool {
        self.current_byte() == ';' as u8
    }

    pub fn is_colon(&self) -> bool {
        self.current_byte() == ':' as u8
    }

    pub fn is_newline(&self) -> bool {
        self.current_byte() == NEW_LINE
    }

    pub fn is_comma(&self) -> bool {
        self.current_byte() == ',' as u8
    }

    pub fn skip_lines(&mut self, n: usize) -> &mut Self {
        for _ in 0..n {
            while !self.at_end() && !self.is_newline() {
                self.consume();
            }
            if !self.at_end() {
                self.consume();
            }
        }
        self
    }

}




pub fn rust_specific_pass<'a>(token: Token<'a>) -> RustSpecific<'a> {

        match token {
            Token::Atom("as") => RustSpecific::Keyword("as"),
            Token::Atom("break") => RustSpecific::Keyword("break"),
            Token::Atom("const") => RustSpecific::Keyword("const"),
            Token::Atom("continue") => RustSpecific::Keyword("continue"),
            Token::Atom("crate") => RustSpecific::Keyword("crate"),
            Token::Atom("else") => RustSpecific::Keyword("else"),
            Token::Atom("enum") => RustSpecific::Keyword("enum"),
            Token::Atom("extern") => RustSpecific::Keyword("extern"),
            Token::Atom("false") => RustSpecific::Keyword("false"),
            Token::Atom("fn") => RustSpecific::Keyword("fn"),
            Token::Atom("for") => RustSpecific::Keyword("for"),
            Token::Atom("if") => RustSpecific::Keyword("if"),
            Token::Atom("impl") => RustSpecific::Keyword("impl"),
            Token::Atom("in") => RustSpecific::Keyword("in"),
            Token::Atom("let") => RustSpecific::Keyword("let"),
            Token::Atom("loop") => RustSpecific::Keyword("loop"),
            Token::Atom("match") => RustSpecific::Keyword("match"),
            Token::Atom("mod") => RustSpecific::Keyword("mod"),
            Token::Atom("move") => RustSpecific::Keyword("move"),
            Token::Atom("mut") => RustSpecific::Keyword("mut"),
            Token::Atom("pub") => RustSpecific::Keyword("pub"),
            Token::Atom("ref") => RustSpecific::Keyword("ref"),
            Token::Atom("return") => RustSpecific::Keyword("return"),
            Token::Atom("self") => RustSpecific::Keyword("self"),
            Token::Atom("Self") => RustSpecific::Keyword("Self"),
            Token::Atom("static") => RustSpecific::Keyword("static"),
            Token::Atom("struct") => RustSpecific::Keyword("struct"),
            Token::Atom("super") => RustSpecific::Keyword("super"),
            Token::Atom("trait") => RustSpecific::Keyword("trait"),
            Token::Atom("true") => RustSpecific::Keyword("true"),
            Token::Atom("type") => RustSpecific::Keyword("type"),
            Token::Atom("unsafe") => RustSpecific::Keyword("unsafe"),
            Token::Atom("use") => RustSpecific::Keyword("use"),
            Token::Atom("where") => RustSpecific::Keyword("where"),
            Token::Atom("while") => RustSpecific::Keyword("while"),
            t => RustSpecific::Token(t)
        }
}