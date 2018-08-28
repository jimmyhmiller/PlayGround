use std::iter::Peekable;
use std::io::Read;
use std::str::Chars;
use std::fs::File;

#[derive(Debug)]
enum Token {
    OpenParen,
    CloseParen,
    Atom(String)
}


fn parse_atom(c : char, expr: &mut Peekable<Chars>) -> Token {
    let mut s : String = c.to_string();
    loop {
        match expr.peek() {
            Some('(') => break,
            Some(')') => break,
            Some(' ') => break,
            Some(ch) => s.push(*ch),
            None => break
        }
        expr.next();
    }
    Token::Atom(s)
}

fn tokenize(expr : &mut Peekable<Chars>) -> Vec<Token> {
    let mut tokens: Vec<Token> = vec!();
    loop {
        match expr.next() {
            Some('(') => tokens.push(Token::OpenParen),
            Some(')') => tokens.push(Token::CloseParen),
            Some(' ') => {},
            Some(ch) => tokens.push(parse_atom(ch, expr)),
            None => break,
        }
    }
    tokens
}

fn main() {

    let mut f = File::open("/Users/jimmy/Desktop/all2.clj").expect("file not found");

    let mut contents = String::new();
    f.read_to_string(&mut contents)
        .expect("something went wrong reading the file");
    println!("{:?}", tokenize(&mut "(+ 2 2)".chars().peekable()));
}
