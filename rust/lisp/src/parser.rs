
use std::fs::File;
use std::io::BufReader;
use std::io::Read;
use std::time::Instant;

#[derive(Debug)]
enum Token<'a> {
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
    Spaces(&'a str),
    String(&'a str),
    Integer(&'a str),
    Float(&'a str),
    Atom(&'a str),
}

#[derive(Debug)]
struct Tokenizer<'a> {
    input: &'a str,
    input_bytes: &'a [u8],
    position: usize,
}

#[derive(Debug)]
enum RustSpecific<'a> {
    As,
    Break,
    Const,
    Continue,
    Crate,
    Else,
    Enum,
    Extern,
    False,
    Fn,
    For,
    If,
    Impl,
    In,
    Let,
    Loop,
    Match,
    Mod,
    Move,
    Mut,
    Pub,
    Ref,
    Return,
    SelfValue,
    SelfType,
    Static,
    Struct,
    Super,
    Trait,
    True,
    Type,
    Unsafe,
    Use,
    Where,
    While,
    Token(Token<'a>)
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
    fn new(input: &str) -> Tokenizer {
        Tokenizer {
            input: input,
            input_bytes: input.as_bytes(),
            position: 0,
        }
    }

    fn consume(&mut self) -> () {
        self.position += 1;
    }

    fn current_byte(&self) -> u8 {
        self.input_bytes[self.position]
    }

    fn is_space(&self) -> bool {
        self.current_byte() == SPACE
    }

    fn at_end(&self) -> bool {
        self.input.len() == self.position
    }

    fn is_quote(&self) -> bool {
        self.current_byte() == DOUBLE_QUOTE
    }

    fn parse_string(&mut self) -> Token<'a> {
        self.consume(); // skip open quote
        let start = self.position;
        while !self.at_end() && !self.is_quote() {
            self.consume();
        }
        self.consume(); // skip closing quote
        Token::String(&self.input[start..self.position])
    }

    fn is_open_paren(&self) -> bool {
        self.current_byte() == OPEN_PAREN
    }

    fn is_close_paren(&self) -> bool {
        self.current_byte() == CLOSE_PAREN
    }

    fn is_open_curly(&self) -> bool {
        self.current_byte() == '{' as u8
    }

    fn is_close_curly(&self) -> bool {
        self.current_byte() == '}' as u8
    }

    fn is_open_bracket(&self) -> bool {
        self.current_byte() == '[' as u8
    }

    fn is_close_bracket(&self) -> bool {
        self.current_byte() == ']' as u8
    }

    fn parse_spaces(&mut self) -> Token<'a> {
        let start = self.position;
        while !self.at_end() && self.is_space() {
            self.consume();
        }
        Token::Spaces(&self.input[start..self.position])

    }

    fn is_valid_number_char(&mut self) -> bool {
        self.current_byte() >= ZERO && self.current_byte() <= NINE
    }

    fn parse_number(&mut self) -> Token<'a> {
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

    fn parse_identifier(&mut self) -> Token<'a> {
        let start = self.position;
        while !self.is_space() 
                && !self.is_open_paren()
                && !self.is_close_paren()
                && !self.is_semi_colon()
                && !self.is_colon()
                && !self.is_comma()
                && !self.is_newline() {
            self.consume()
        }
        // println!("{} {}", start, self.position);
        Token::Atom(&self.input[start..self.position])
    }

    fn parse_single(&mut self) -> Option<Token<'a>> {
        
        if self.at_end() {
            return None
        }
        let result = if self.is_space() {
            self.parse_spaces()
        } else if self.is_newline() {
            self.consume();
            Token::NewLine
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

    fn is_semi_colon(&self) -> bool {
        self.current_byte() == ';' as u8
    }

    fn is_colon(&self) -> bool {
        self.current_byte() == ':' as u8
    }

    fn is_newline(&self) -> bool {
        self.current_byte() == NEW_LINE
    }

    fn is_comma(&self) -> bool {
        self.current_byte() == ',' as u8
    }

    fn skip_lines(&mut self, n: usize) -> &mut Self {
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




fn rust_specific_pass<'a>(token: Token<'a>) -> RustSpecific<'a> {

        match token {
            Token::Atom("as") => RustSpecific::As,
            Token::Atom("break") => RustSpecific::Break,
            Token::Atom("const") => RustSpecific::Const,
            Token::Atom("continue") => RustSpecific::Continue,
            Token::Atom("crate") => RustSpecific::Crate,
            Token::Atom("else") => RustSpecific::Else,
            Token::Atom("enum") => RustSpecific::Enum,
            Token::Atom("extern") => RustSpecific::Extern,
            Token::Atom("false") => RustSpecific::False,
            Token::Atom("fn") => RustSpecific::Fn,
            Token::Atom("for") => RustSpecific::For,
            Token::Atom("if") => RustSpecific::If,
            Token::Atom("impl") => RustSpecific::Impl,
            Token::Atom("in") => RustSpecific::In,
            Token::Atom("let") => RustSpecific::Let,
            Token::Atom("loop") => RustSpecific::Loop,
            Token::Atom("match") => RustSpecific::Match,
            Token::Atom("mod") => RustSpecific::Mod,
            Token::Atom("move") => RustSpecific::Move,
            Token::Atom("mut") => RustSpecific::Mut,
            Token::Atom("pub") => RustSpecific::Pub,
            Token::Atom("ref") => RustSpecific::Ref,
            Token::Atom("return") => RustSpecific::Return,
            Token::Atom("self") => RustSpecific::SelfValue,
            Token::Atom("Self") => RustSpecific::SelfType,
            Token::Atom("static") => RustSpecific::Static,
            Token::Atom("struct") => RustSpecific::Struct,
            Token::Atom("super") => RustSpecific::Super,
            Token::Atom("trait") => RustSpecific::Trait,
            Token::Atom("true") => RustSpecific::True,
            Token::Atom("type") => RustSpecific::Type,
            Token::Atom("unsafe") => RustSpecific::Unsafe,
            Token::Atom("use") => RustSpecific::Use,
            Token::Atom("where") => RustSpecific::Where,
            Token::Atom("while") => RustSpecific::While,
            t => RustSpecific::Token(t)
        }
}


#[allow(dead_code)]
pub fn parse_file(filename: String) -> () {
    // I need to get a standard file for this.
    println!("\n\n\n\n\n\n\n\n");
    let file = File::open(filename).unwrap();
    let mut expr = String::new();
    let mut buf_reader = BufReader::new(file);
    buf_reader.read_to_string(&mut expr).unwrap();

    let start = Instant::now();
    let mut tokenizer =  Tokenizer::new(&expr);
    // println!("{:?}", rust_specific_pass(&tokens).collect::<Vec<_>>());
    let mut output = RustSpecific::Return;
    // tokenizer.skip_lines(3);
    // while !tokenizer.is_newline() {
    //    println!("{:?}", tokenizer.next());
    // }

    for token in tokenizer.map(|x| rust_specific_pass(x)) {
        // println!("{:?}", token);
        output = token
    }
    println!("{:?}", output);
    // let read_expr = read(tokenize(&expr));
    let duration = start.elapsed();
    // println!("{:?}", s_expr_len(read_expr));
    println!("{:#?}", duration);
}
