#[macro_use]
extern crate lazy_static;

use std::str::from_utf8;
use std::collections::VecDeque;
use std::time::{Instant};

#[derive(Debug, PartialEq)]
enum Token {
    OpenParen,
    CloseParen,
    String(String),
    Integer(String),
    Float(String),
    Atom(String),
}


#[derive(Debug)]
struct Tokenizer<'a> {
    input: &'a [u8],
    position: usize,
    temp : Vec<u8>,
}


lazy_static! {
    static ref ZERO : u8 = '0' as u8;
    static ref NINE : u8 = '9' as u8;
    static ref SPACE : u8 = ' ' as u8;
    static ref NEW_LINE : u8 = '\n' as u8;
    static ref COMMA : u8 = ',' as u8;
    static ref DOUBLE_QUOTE : u8 = '"' as u8;
    static ref OPEN_PAREN : u8 = '(' as u8;
    static ref CLOSE_PAREN : u8 = ')' as u8;
    static ref PERIOD : u8 = '.' as u8;
}


impl Tokenizer<'_> {
    fn new(input: &str) -> Tokenizer {
        Tokenizer {
            input: input.as_bytes(),
            position: 0,
            // This is so we only have to allocate once
            // Seems to make things faster
            temp: Vec::with_capacity(10),
        }
    }

    fn consume(&mut self) -> () {
        self.position += 1;
    }

    fn consume_to_temp(&mut self) -> () {
        let c = self.input[self.position];
        self.position += 1;
        self.temp.push(c)
    }

    fn temp_to_string(&mut self) -> String {
        let string = from_utf8(&self.temp).unwrap().to_string();
        self.temp.clear();
        string
    }

    fn current_byte(&self) -> u8 {
        self.input[self.position]
    }

    fn is_space(&self) -> bool {
        self.current_byte() == *SPACE
        || self.current_byte() == *COMMA
        || self.current_byte() == *NEW_LINE
    }

    fn at_end(&self) -> bool {
        self.input.len() == self.position
    }

    fn is_quote(&self) -> bool {
        self.current_byte() == *DOUBLE_QUOTE
    }

    fn parse_string(&mut self) -> Token {
        self.consume(); // skip open quote
        while !self.at_end() && !self.is_quote() {
            self.consume_to_temp()
        }
        self.consume(); // skip closing quote
        Token::String(self.temp_to_string())
    }

    fn is_open_paren(&self) -> bool {
        self.current_byte() == *OPEN_PAREN
    }

    fn is_close_paren(&self) -> bool {
        self.current_byte() == *CLOSE_PAREN
    }

    fn consume_spaces(&mut self) -> () {
        while !self.at_end() && self.is_space() {
            self.consume();
        }
    }

    fn is_valid_number_char(&mut self) -> bool {
        self.current_byte() >= *ZERO && self.current_byte() <= *NINE
    }

    fn parse_number(&mut self) -> Token {
        let mut is_float = false;
        while self.is_valid_number_char() || self.current_byte() == *PERIOD {
            // Need to handle making sure there is only one "."
            if self.current_byte() == *PERIOD {
                is_float = true;
            }
            self.consume_to_temp();
        }
        if is_float {
            Token::Float(self.temp_to_string())
        } else {
            Token::Integer(self.temp_to_string())
        }
    }

    fn parse_identifier(&mut self) -> Token {
        while !self.is_space() && !self.is_open_paren() && !self.is_close_paren() {
            self.consume_to_temp()
        }
        Token::Atom(self.temp_to_string())
    }

    fn parse_single(&mut self) -> Token {
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

    fn read(&mut self) -> VecDeque<Token> {
        let mut tokens = VecDeque::new();
        while !self.at_end() {
            tokens.push_back(self.parse_single());
        }
        tokens
    }

}


// Need to change this to allow strings
fn tokenize(text: String) -> VecDeque<Token> {
    Tokenizer::new(&text).read()
}

#[derive(Debug)]
enum Expr {
    SExpr(Vec<Expr>),
    Atom(String),
    Bool(bool),
    String(String),
    Integer(i64),
    Float(f64)
}


fn read(tokens: VecDeque<Token>) -> Expr {
    // Is there a faster way to do this?
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
                 continue;
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

fn s_expr_len(x : Expr) -> usize {
    if let Expr::SExpr(x) = x {
        x.len()
    } else {
        0
    }
}

fn main() {
    let expr = format!("({:})", "(+ 1 1)".repeat(1000000)).to_string();
    let start = Instant::now();
    let read_expr = read(tokenize(expr));
    let duration = start.elapsed();
    println!("{:?}", s_expr_len(read_expr));
    println!("{:?}", duration);

}
