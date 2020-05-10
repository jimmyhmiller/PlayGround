
use super::Expr;

#[derive(Debug)]
pub enum Token<'a> {
    OpenParen,
    CloseParen,
    OpenCurly,
    CloseCurly,
    String(&'a str),
    Integer(&'a str),
    Float(&'a str),
    Symbol(&'a str),
}

#[derive(Debug)]
pub struct Tokenizer<'a> {
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
static OPEN_CURLY: u8 = '{' as u8;
static CLOSE_CURLY: u8 = '}' as u8;
static COLON: u8 = ':' as u8;


// Need to parse maps better. Right now pretending colon is whitespace
// Or maybe that is correct?

impl<'a> Tokenizer<'a> {
    fn new(input: &str) -> Tokenizer {
        Tokenizer {
            input,
            input_bytes: input.as_bytes(),
            position: 0,
            // This is so we only have to allocate once
            // Seems to make things faster
            temp: Vec::with_capacity(10),
        }
    }

    fn consume(&mut self) -> () {
        self.position += 1;
        // println!("{}", self.position);
    }

    fn current_byte(&self) -> u8 {
        self.input_bytes[self.position]
    }

    fn is_space(&self) -> bool {
        self.current_byte() == SPACE
            || self.current_byte() == COMMA
            || self.current_byte() == NEW_LINE
            || self.current_byte() == COLON
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
        self.current_byte() == OPEN_CURLY
    }
    fn is_close_curly(&self) -> bool {
        self.current_byte() == CLOSE_CURLY
    }

    fn is_bracket(&self) -> bool {
        self.is_open_paren() ||
        self.is_close_paren() ||
        self.is_open_curly() ||
        self.is_close_curly()
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
        while !self.at_end() && (self.is_valid_number_char() || self.current_byte() == PERIOD)  {
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
        while !self.at_end() && !self.is_space() && !self.is_bracket() {
            self.consume()
        }
        Token::Symbol(&self.input[start..self.position])
    }

    fn parse_single(&mut self) -> Token<'a> {
        self.consume_spaces();
        let result = if self.is_open_paren() {
            self.consume();
            Token::OpenParen
        } else if self.is_close_paren() {
            self.consume();
            Token::CloseParen
        } else if self.is_open_curly() {
            self.consume();
            Token::OpenCurly
        } else if self.is_close_curly() {
            self.consume();
            Token::CloseCurly
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

pub fn tokenize<'a>(text: &'a str) -> Vec<Token<'a>> {
    Tokenizer::new(text).read()
}

// Need to handle quote here.
pub fn parse(tokens: Vec<Token>) -> Expr {
    let mut exprs_stack = Vec::with_capacity(tokens.len()); // arbitrary
    let mut current = Vec::with_capacity(10); // arbitrary

    for token in tokens {
        match token {
            // Token::Symbol(s) if s == "True" => current.push(Expr::Bool(true)),
            // Token::Symbol(s) if s == "False" => current.push(Expr::Bool(false)),
            Token::Symbol(s) if s.starts_with("?") => current.push(Expr::LogicVariable(s.to_string())),
            Token::Symbol(s) => current.push(Expr::Symbol(s.to_string())),
            Token::Integer(s) => current.push(Expr::Num(s.parse::<i64>().unwrap())),
            // Token::Float(s) => current.push(Expr::Float(s.parse::<f64>().unwrap())),
            // Token::String(s) => current.push(Expr::String(s)),
            Token::OpenCurly => {
                exprs_stack.push(current);
                current = Vec::with_capacity(10); // arbitrary
            }
            Token::CloseCurly => {
                let length = current.len();
                let mut pairs = Vec::with_capacity(length / 2);
                if length % 2 == 0 {
                    let mut i = 0;
                    while i < length {
                        // println!("i {:?}", current);
                        let second = current.pop().unwrap();
                        let first = current.pop().unwrap();
                        pairs.push((first, second));
                        i += 2;
                    }
                } else {
                    panic!("map not even, maybe want to support this for destructoring/punning?")
                }
                // @performance: Do something better than reverse.
                pairs.reverse();
                let expr = Expr::Map(pairs);
                current = exprs_stack.pop().unwrap();
                current.push(expr);
            }
            Token::OpenParen => {
                let f = current.pop().unwrap();
                exprs_stack.push(current);
                current = Vec::with_capacity(10); // arbitrary
                current.push(f);
            }
            Token::CloseParen => {
                // println!("{:?}", current);
                // Probably a more efficient way to do this.
                let first = current.remove(0);
                let expr = if first == Expr::Symbol("quote".to_string()) {
                    Expr::Exhausted(box current.pop().unwrap())
                } else {
                    Expr::Call(box first, current)
                };
                current = exprs_stack.pop().unwrap();
                current.push(expr);
            }
            x => panic!("Invalid expr {:?}", x)
        };
    }

    assert_eq!(current.len(), 1);
    current.pop().unwrap()
}

pub fn read<'a>(expr : &'a str) -> Expr {
    parse(tokenize(expr))
}
