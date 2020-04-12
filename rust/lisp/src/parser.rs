
use std::fs::File;
use std::io::BufReader;
use std::io::Read;
use std::process::abort;
use std::time::Instant;

#[derive(Debug)]
enum Token<'a> {
    OpenParen,
    CloseParen,
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
    temp: Vec<u8>,
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


impl<'a> Tokenizer<'a> {
    fn new(input: &str) -> Tokenizer {
        Tokenizer {
            input: input,
            input_bytes: input.as_bytes(),
            position: 0,
            // This is so we only have to allocate once
            // Seems to make things faster
            temp: Vec::with_capacity(10),
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
            || self.current_byte() == COMMA
            || self.current_byte() == NEW_LINE
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

    fn consume_spaces(&mut self) -> () {
        while !self.at_end() && self.is_space() {
            self.consume();
        }
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
        while !self.is_space() && !self.is_open_paren() && !self.is_close_paren() {
            self.consume()
        }
        Token::Atom(&self.input[start..self.position])
    }

    fn parse_single(&mut self) -> Token<'a> {
        self.consume_spaces();
        let result = if self.is_open_paren() {
            self.consume();
            Token::OpenParen
        } else if self.is_close_paren() {
            self.consume();
            Token::CloseParen
        } else if self.is_valid_number_char() {
            self.parse_number()
        } else if self.is_quote() {
            self.parse_string()
        } else {
            self.parse_identifier()
        };
        result
    }

    fn read(&mut self) -> Vec<Token<'a>> {
        let mut tokens = Vec::with_capacity(self.input.len());
        while !self.at_end() {
            tokens.push(self.parse_single());
        }
        tokens
    }
}

fn tokenize<'a>(text: &'a str) -> Vec<Token<'a>> {
    Tokenizer::new(text).read()
}

#[derive(Debug)]
enum Expr<'a> {
    SExpr(Vec<Expr<'a>>),
    Atom(&'a str),
    Bool(bool),
    String(&'a str),
    Integer(i64),
    Float(f64),
}

fn read(tokens: Vec<Token>) -> Expr {
    // Is there a faster way to do this?
    // Need to probably refer to slices of things
    // Like I ended up doing above. But not 100% sure how to do that
    // given the SExpr structure
    // Maybe I should do linked list of pointers?
    let mut exprs_stack = Vec::with_capacity(tokens.len()); // arbitrary
    let mut current = Vec::with_capacity(10); // arbitrary

    for token in tokens {
        match token {
            Token::Atom(s) if s == "True" => current.push(Expr::Bool(true)),
            Token::Atom(s) if s == "False" => current.push(Expr::Bool(false)),
            Token::Atom(s) => current.push(Expr::Atom(s)),
            Token::Integer(s) => current.push(Expr::Integer(s.parse::<i64>().unwrap())),
            Token::Float(s) => current.push(Expr::Float(s.parse::<f64>().unwrap())),
            Token::String(s) => current.push(Expr::String(s)),
            Token::OpenParen => {
                exprs_stack.push(current);
                current = Vec::with_capacity(10); // arbitrary
            }
            Token::CloseParen => {
                let expr = Expr::SExpr(current);
                current = exprs_stack.pop().unwrap();
                current.push(expr);
            }
        };
    }

    assert_eq!(current.len(), 1);
    current.pop().unwrap()
}

fn s_expr_len(x: Expr) -> usize {
    if let Expr::SExpr(x) = x {
        x.len()
    } else {
        0
    }
}

#[allow(dead_code)]
fn parse_file(filename: String) -> () {
    // I need to get a standard file for this.
    let file = File::open(filename).unwrap();
    let mut expr = String::new();
    let mut buf_reader = BufReader::new(file);
    buf_reader.read_to_string(&mut expr).unwrap();

    let start = Instant::now();
    let read_expr = read(tokenize(&expr));
    let duration = start.elapsed();
    println!("{:?}", s_expr_len(read_expr));
    println!("{:?}", duration);
}
