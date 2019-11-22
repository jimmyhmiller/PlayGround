#[macro_use]
extern crate lazy_static;

use core::convert::identity;
use core::str::from_utf8;
use regex::Regex;
use std::collections::VecDeque;

#[derive(Debug, Clone)]
enum Token {
    Identifier(String),
    Number(String),
    Str(String),
    Parens(Vec<Token>),
    Braces(Vec<Token>),
    Brackets(Vec<Token>),
    Tree(Vec<Token>),
    Phrase(Box<Token>, Box<Token>, Box<Token>, Box<Token>),
    Binary(Box<Token>, Box<Token>, Box<Token>),
}

#[derive(Debug)]
struct Tokenizer<'a> {
    input: &'a [u8],
    position: usize,
}

impl Tokenizer<'_> {
    fn new(input: &str) -> Tokenizer {
        return Tokenizer {
            input: input.as_bytes(),
            position: 0,
        };
    }
    fn consume(&mut self) -> String {
        let c = self.input[self.position];
        self.position += 1;
        from_utf8(&[c]).unwrap().to_string()
    }

    fn current_char(&self) -> String {
        let c = [self.input[self.position]];
        from_utf8(&c).unwrap().to_string()
    }

    fn is_space(&self) -> bool {
        self.current_char() == " "
            || self.current_char() == "\n"
            || self.current_char() == ","
            || self.current_char() == ";" // probably not correct
    }

    fn at_end(&self) -> bool {
        self.input.len() == self.position
    }

    fn is_valid_identifier_char(&self) -> bool {
        lazy_static! {
            static ref IDENTIFIER: Regex = Regex::new("[A-Za-z0-9_\\-+<>=?$'*^]").unwrap();
        }
        IDENTIFIER.is_match(&self.current_char())
    }

    fn is_valid_number_char(&self) -> bool {
        lazy_static! {
            static ref NUMBER: Regex = Regex::new("[0-9._]").unwrap();
        }
        NUMBER.is_match(&self.current_char())
    }

    fn parse_number(&mut self) -> Token {
        let mut number = String::new();
        number.push_str(&self.consume());
        while !self.at_end() && self.is_valid_number_char() {
            number.push_str(&self.consume());
        }
        return Token::Number(number);
    }

    fn parse_identifier(&mut self) -> Token {
        let mut identifier = String::new();
        identifier.push_str(&self.consume());
        while !self.at_end() && self.is_valid_identifier_char() {
            identifier.push_str(&self.consume());
        }
        assert!(identifier.len() > 0);
        return Token::Identifier(identifier);
    }

    fn is_quote(&self) -> bool {
        self.current_char() == "\""
    }

    fn parse_string(&mut self) -> Token {
        let mut string = String::new();
        self.consume(); // skip open quote
        while !self.at_end() && !self.is_quote() {
            string.push_str(&self.consume());
        }
        self.consume(); // skip closing quote
        return Token::Str(string);
    }

    fn is_open_paren(&self) -> bool {
        self.current_char() == "("
    }

    fn is_close_paren(&self) -> bool {
        self.current_char() == ")"
    }

    fn parse_parens(&mut self) -> Token {
        let mut tokens = Vec::new();
        self.consume(); // skip open parens
        while !self.at_end() && !self.is_close_paren() {
            tokens.push(self.parse_single());
        }
        self.consume(); // skip closing parens
        Token::Parens(tokens)
    }

    fn is_open_brace(&self) -> bool {
        self.current_char() == "["
    }

    fn is_close_brace(&self) -> bool {
        self.current_char() == "]"
    }

    fn parse_braces(&mut self) -> Token {
        let mut tokens = Vec::new();
        self.consume(); // skip open parens
        while !self.at_end() && !self.is_close_brace() {
            tokens.push(self.parse_single());
        }
        self.consume(); // skip closing parens
        Token::Braces(tokens)
    }

    fn is_open_bracket(&self) -> bool {
        self.current_char() == "{"
    }

    fn is_close_bracket(&self) -> bool {
        self.current_char() == "}"
    }

    fn parse_brackets(&mut self) -> Token {
        let mut tokens = Vec::new();
        self.consume(); // skip open parens
        while !self.at_end() && !self.is_close_bracket() {
            tokens.push(self.parse_single());
        }
        self.consume(); // skip closing parens
        Token::Brackets(tokens)
    }

    fn parse_single(&mut self) -> Token {
        self.consume_spaces();
        let result = if self.is_open_paren() {
            self.parse_parens()
        } else if self.is_open_brace() {
            self.parse_braces()
        } else if self.is_open_bracket() {
            self.parse_brackets()
        } else if self.is_valid_number_char() {
            self.parse_number()
        } else if self.is_quote() {
            self.parse_string()
        } else {
            self.parse_identifier()
        };
        self.consume_spaces();
        result
    }

    fn parse_all(&mut self) -> Vec<Token> {
        let mut tokens = Vec::new();
        while !self.at_end() {
            tokens.push(self.parse_single());
        }
        tokens
    }

    fn consume_spaces(&mut self) -> () {
        while !self.at_end() && self.is_space() {
            self.consume();
        }
    }
}

// Not sure if the higher order function stuff is the best for rust
fn enforest(
    tokens: &mut VecDeque<Token>,
    combine: Box<dyn Fn(Token) -> Token>,
    precedence: usize,
    mut stack: Vec<(Box<dyn Fn(Token) -> Token>, usize)>,
) -> (Token, &VecDeque<Token>) {
    if let Some(first) = tokens.pop_front() {
        match first {
            Token::Number(x) => {
                tokens.push_front(Token::Tree(vec![Token::Number(x.to_string())]));
                enforest(tokens, combine, precedence, stack)
            }

            Token::Identifier(x) if x == "fn" => {
                // Should check the types of these.
                let phrase_type = Box::new(Token::Identifier("fn".to_string()));
                let phrase_name = Box::new(tokens.pop_front().unwrap());
                let phrase_params = Box::new(tokens.pop_front().unwrap());
                let phrase_block = Box::new(tokens.pop_front().unwrap());
                let tree = Token::Tree(vec!(Token::Phrase(phrase_type, phrase_name, phrase_params, phrase_block)));
                tokens.push_front(tree);
                enforest(tokens, combine, precedence, stack)
            }
            Token::Identifier(x) => {
                tokens.push_front(Token::Tree(vec![Token::Identifier(x.to_string())]));
                enforest(tokens, combine, precedence, stack)
            }
            Token::Tree(tree) => match tokens.front() {
                // Should make this programmable, not hardcoded
                Some(Token::Identifier(x)) if x == "+" => {
                    tokens.pop_front();
                    stack.push((combine, precedence));
                    let combine2: Box<dyn Fn(Token) -> Token> =
                        Box::new(move |t: Token| -> Token {
                            Token::Tree(
                                vec!(
                                    Token::Identifier("+".to_string()),
                                    Token::Tree(tree.clone()),
                                    t
                                )
                            )
                        });
                    enforest(tokens, combine2, 10, stack)
                }
                _ => match stack.pop() {
                    Some((combine2, precedence2)) => {
                        tokens.pop_front();
                        tokens.push_front(combine(Token::Tree(tree)));
                        enforest(tokens, combine2, precedence2, stack)
                    }
                    None => (combine(Token::Tree(tree)), tokens),
                },
            },
            _ => (first, tokens),
        }
    } else {
        (Token::Str("?".to_string()), tokens)
    }
}

// Working through implementing something like this
// https://www.cs.utah.edu/plt/publications/gpce12-rf.pdf
// Honu: Syntactic Extension for Algebraic Notation through Enforestation

fn main() {
    let mut tokenizer = Tokenizer::new("fn fib(n) { 0 => 0; 1 => 1; n => fib(n - 1) + fib(n - 2)}");
    // let mut tokenizer = Tokenizer::new("2 + 3 + 4");


    let result = tokenizer.parse_all();
    let mut deque: VecDeque<Token> = result.into();
    println!("{:#?}", enforest(&mut deque, Box::new(identity), 0, vec!()));
}
