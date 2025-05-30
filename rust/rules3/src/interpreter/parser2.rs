

use super::new::{Expr, Forest, Interner};

#[derive(Debug)]
pub enum Token<'a> {
    OpenParen,
    CloseParen,
    OpenCurly,
    CloseCurly,
    OpenBracket,
    CloseBracket,
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
static OPEN_BRACKET: u8 = '[' as u8;
static CLOSE_BRACKET: u8 = ']' as u8;
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
        self.input.len() <= self.position
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
    fn is_open_bracket(&self) -> bool {
        self.current_byte() == OPEN_BRACKET
    }
    fn is_close_bracket(&self) -> bool {
        self.current_byte() == CLOSE_BRACKET
    }

    fn is_any_brace(&self) -> bool {
        self.is_open_paren() ||
        self.is_close_paren() ||
        self.is_open_curly() ||
        self.is_close_curly() ||
        self.is_open_bracket() ||
        self.is_close_bracket()
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
        while !self.at_end() && !self.is_space() && !self.is_any_brace() {
            self.consume()
        }
        Token::Symbol(&self.input[start..self.position])
    }

    fn parse_single(&mut self) -> Token<'a> {
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
        } else if self.is_open_bracket() {
            self.consume();
            Token::OpenBracket
        } else if self.is_close_bracket() {
            self.consume();
            Token::CloseBracket
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
            self.consume_spaces();
            if self.at_end() {
                break;
            }
            tokens.push(self.parse_single());
        }
        tokens
    }
}

pub fn tokenize<'a>(text: &'a str) -> Vec<Token<'a>> {
    Tokenizer::new(text).read()
}

// Need to handle quote here.
pub fn parse_new(tokens: Vec<Token>, rooted_forest : &mut Forest<Expr>, interner: &mut Interner) {
    // Need to figure out a much better way than this silly root thing.
    let mut is_root = true;
    for token in tokens {
        match token {
            Token::String(_) => {}
            Token::Integer(s) => {
                let expr = Expr::Num(s.parse::<isize>().unwrap());
                if is_root {
                    rooted_forest.insert_root_val(expr);
                    is_root = false;
                } else {
                    rooted_forest.insert_child(expr);
                }
            }
            Token::Float(_) => {}
            Token::Symbol(s) => {
                // Need to intern strings
                let interned_index = interner.intern(s);
                let expr = if s == "do" {
                    Expr::Do 
                } else if s.starts_with("?") {
                    Expr::LogicVariable(interned_index)
                } else if s.starts_with("@") {
                    Expr::Scope(interned_index)
                } else if s == "quote" {
                    Expr::Quote
                } else {
                    Expr::Symbol(interned_index)
                };

                if is_root {
                    rooted_forest.insert_root_val(expr);
                    is_root = false;
                } else {
                    rooted_forest.insert_child(expr);
                }
            }
            Token::OpenParen => {
                let last_inserted = rooted_forest.get_last_inserted_val();
                if last_inserted == Some(&Expr::Quote) {
                    rooted_forest.make_last_inserted_focus();
                } else {
                    rooted_forest.swap_and_insert(Expr::Call);
                }
                
            }
            Token::CloseParen => {
                rooted_forest.make_parent_focus();
            }
            Token::OpenCurly => {
                let focus_val = rooted_forest.get_focus_val();
                if focus_val != Some(&Expr::Do) {
                    if is_root {
                        is_root = false;
                        rooted_forest.insert_root_val(Expr::Map);
                    } else {
                        rooted_forest.insert_child(Expr::Map);
                        rooted_forest.make_last_child_focus();
                    }
                }
            }
            Token::CloseCurly => {
                rooted_forest.make_parent_focus();
            }
            Token::OpenBracket => {
                if is_root {
                    is_root = false;
                    rooted_forest.insert_root_val(Expr::Array);
                } else {
                    rooted_forest.insert_child(Expr::Array);
                    rooted_forest.make_last_child_focus();
                }
            }
            Token::CloseBracket => {
                rooted_forest.make_parent_focus();
            }
        }
    }
}

pub fn read_new<'a>(expr : &'a str, rooted_forest : &mut Forest<Expr>, interner: &mut Interner) {
    parse_new(tokenize(expr), rooted_forest, interner)
}
