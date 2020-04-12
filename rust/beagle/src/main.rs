#[macro_use]
extern crate lazy_static;

use core::convert::identity;
use core::str::from_utf8;
use regex::Regex;
use std::collections::VecDeque;
use std::collections::HashMap;

// Overall this isn't quite right. But my plan is to just continue with it.
// I would rather get something working that is wrong than to stop and redo it.
// That is a big tendency of mine, to not like something and just stop.
// But in this case if I can get something working, I can also go back and change things.


// Things are workingish. But how does this extend to macros?
// Maybe I can make a fake macro?

// All the cloning and converting is bad. Should learn rust better and do better.


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
    Call(Box<Token>, Vec<Token>),
    Val(Box<Token>, Box<Token>),
}

#[derive(Debug, Clone)]
enum PhraseType {
    Fn
}

#[derive(Debug, Clone)]
enum Ast {
    Str(String),
    Number(String),
    Identifier(String),
    Block(Vec<Ast>),
    Val(String, Box<Ast>),
    Call(Box<Ast>, Vec<Ast>),
    Phrase(PhraseType, Vec<Ast>, Box<Ast>),
    NamedPhrase(PhraseType, String, Vec<Ast>, Box<Ast>),
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
    bindings: &mut HashMap<String,Token>,
    combine: Box<dyn Fn(Token) -> Token>,
    precedence: usize,
    mut stack: Vec<(Box<dyn Fn(Token) -> Token>, usize)>,
) -> Token {
    if let Some(first) = tokens.pop_front() {
        match first {
            Token::Number(x) => {
                tokens.push_front(Token::Tree(vec![Token::Number(x.to_string())]));
                enforest(tokens, bindings, combine, precedence, stack)
            }
            Token::Identifier(x) if x == "val" => {
                // Should check the types of these.
                let val_name = tokens.pop_front().unwrap();
                tokens.pop_front(); // equals sign
                let val_value = enforest(tokens, bindings, combine, precedence, stack);
                if let Token::Identifier(name) = val_name.clone() {
                    bindings.insert(name, val_value.clone());
                }
                let tree = Token::Tree(vec!(Token::Val(Box::new(val_name), Box::new(val_value))));
                return tree;
            }

            Token::Identifier(x) if x == "fn" => {
                // Should check the types of these.
                let phrase_type = Token::Identifier("fn".to_string());
                // Need to handle anonymous functions as well.
                let phrase_name = tokens.pop_front().unwrap();
                let phrase_params = tokens.pop_front().unwrap();
                let phrase_block = tokens.pop_front().unwrap();
                let phrase = Token::Phrase(Box::new(phrase_type), Box::new(phrase_name.clone()), Box::new(phrase_params), Box::new(phrase_block));
                let tree = Token::Tree(vec!(phrase.clone()));
                if let Token::Identifier(name) = phrase_name.clone() {
                    bindings.insert(name, phrase.clone());
                }
                return tree;
            }

            Token::Identifier(x) => {
                tokens.push_front(Token::Tree(vec![Token::Identifier(x.to_string())]));
                enforest(tokens, bindings, combine, precedence, stack)
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
                                    Token::Call(
                                        Box::new(Token::Identifier("+".to_string())),

                                        vec!(
                                            Token::Tree(tree.clone()),
                                            t,
                                        )
                                    )
                                )
                            )
                        });
                    enforest(tokens, bindings, combine2, 10, stack)
                }
                Some(Token::Parens(args)) => {
                    let new_args = args.clone();
                    tokens.pop_front();
                    let token = Token::Tree(vec!(Token::Call(Box::new(Token::Tree(tree.clone())), new_args.to_vec())));
                    token
                }
                _ => match stack.pop() {
                    Some((combine2, precedence2)) => {
                        tokens.push_front(combine(Token::Tree(tree)));
                        enforest(tokens, bindings, combine2, precedence2, stack)
                    }
                    None => combine(Token::Tree(tree)),
                },
            },
            _ => first,
        }
    } else {
        Token::Str("?".to_string())
    }
}



fn get_identifier(token : Box<Token>) -> String {
    if let Token::Identifier(name) = *token {
        name
    } else {
        "".to_string()
    }
}

fn parse(tokens: VecDeque<Token>, bindings: HashMap<String, Token>) -> Vec<Ast> {
    let mut queue : VecDeque<Token> = tokens.into();
    // probably need to combine these bindings;
    let (tokens, bindings2) = parse1(&mut queue);
    let ast = parse2(tokens, bindings.into_iter().chain(bindings2).collect());
    ast
}

fn parse_single(token : Token, bindings: HashMap<String, Token>) -> Box<Ast> {
    Box::new(parse(VecDeque::from(vec!(token)), bindings).first().unwrap().clone())
}

// terrible function
fn parse_multiple_single(token : Token, bindings: HashMap<String, Token>) -> Vec<Ast> {
    parse(VecDeque::from(vec!(token)), bindings)
}

fn parse1(tokens : &mut VecDeque<Token>) -> (VecDeque<Token>, HashMap<String, Token>) {
    let results : &mut VecDeque<Token> = &mut VecDeque::new();

    // Should this be token or some other information?
    // This *might* not be good for it to be mutable?
    // We need nested bindings, not sure the right way to represent this.
    // Although I guess it is only mutable in this scope?
    let bindings : &mut HashMap<String, Token> = &mut HashMap::new();
    while !tokens.is_empty() {
        results.push_front(enforest(tokens, bindings, Box::new(identity), 0, vec!()));
    }
    return (results.clone(), bindings.clone());
}

fn parse2(mut tokens: VecDeque<Token>, bindings: HashMap<String, Token>) -> Vec<Ast> {
    let results : &mut VecDeque<Ast> = &mut VecDeque::new();
    while !tokens.is_empty() {
        match tokens.pop_front().unwrap() {
            token => {
                let mut elems : VecDeque<Token> = if let Token::Tree(xs) = token { xs } else { vec!(token) }.into();
                // Need to figure out this whole tree situation
                // I don't really understand it.
                while !elems.is_empty() {
                    match elems.pop_front().unwrap() {
                        Token::Val(name, v) => results.push_front(Ast::Val(get_identifier(name.clone()), parse_single(*v.clone(), bindings.clone()))),
                        // Need to handle type
                        Token::Phrase(phrase_type, name, args, body) => results.push_front(Ast::NamedPhrase(PhraseType::Fn, get_identifier(name.clone()), parse_multiple_single(*args, bindings.clone()), parse_single(*body.clone(), bindings.clone()))),
                        Token::Brackets(elems) => results.push_front(Ast::Block(parse(VecDeque::from(elems), bindings.clone()))),
                        Token::Call(name, args) => results.push_front(Ast::Call(parse_single(*name, bindings.clone()), parse(VecDeque::from(args), bindings.clone()))),
                        Token::Identifier(s) => results.push_front(Ast::Identifier(s.to_string())),
                        Token::Number(n) => results.push_front(Ast::Number(n.to_string())),
                        Token::Str(s) => results.push_front(Ast::Str(s.to_string())),
                        Token::Parens(elems) => parse(VecDeque::from(elems), bindings.clone()).iter().for_each(|elem| results.push_front(elem.clone())),
                        _ => {}
                    }
                }
            }
        }
    }
    Vec::from(results.clone())
}

// Working through implementing something like this
// https://www.cs.utah.edu/plt/publications/gpce12-rf.pdf
// Honu: Syntactic Extension for Algebraic Notation through Enforestation

// Probably shouldn't be needed
fn eliminate_singleton_trees(token: &Token) -> Token {
    match token {
        Token::Tree(elems) => {

            match elems.as_slice() {
                [t] => {
                    t.clone()
                }
                x => {
                    let q : Vec<Token> = x.to_vec().iter().map(eliminate_singleton_trees).collect();
                    Token::Tree(q)
                }
            }
        }
        _ => token.clone()
    }
}

fn main() {
    // let mut tokenizer = Tokenizer::new("fn fib(n) { 0 => 0; 1 => 1; n => fib(n - 1) + fib(n - 2)}");
    // let mut tokenizer = Tokenizer::new("2 + 3 + 4");
    let mut tokenizer = Tokenizer::new("
    val x = 2 + 3 * 4
    val y = x + 3
    val z = y + 2
    fn do-stuff(n) {
        n + 2
    }

    do-stuff(z)
    ");
    // let mut tokenizer = Tokenizer::new("2 + 2");


    let result : VecDeque<Token> = tokenizer.parse_all().into();
    println!("{:?}", result);
    println!("{:#?}", parse(result, HashMap::new()));
}
