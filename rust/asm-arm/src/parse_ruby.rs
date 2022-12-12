use std::{fs::{self, File}, io::Read};

use regex::Regex;

// generally awful code
// I could maybe twist it to do something reasonable?
// Mostly wanted to explore the idea and think about what I need.


#[derive(Debug)]
enum Operator {
    Lit(String),
    Any,
    Not(Box<Operator>),
    SepBy(Box<Operator>, String),
    Named(String, Box<Operator>),
    Or(Box<Operator>, Box<Operator>),
    Line(Vec<Operator>),
    Repeat(Box<Operator>),
    Sequence(Vec<Operator>),
}


#[derive(Debug, Clone, PartialOrd, PartialEq, Eq)]
enum Token {
    Token(String),
    NewLine,
    Delimiter(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Result {
    Fail,
    Token(Token),
    Named(String, Box<Result>),
    List(Vec<Result>),
}



impl Operator {
    fn matches(&self, tokens: &[Token]) -> (usize, Result) {
        if tokens.is_empty() {
            return (0, Result::Fail);
        }
        match self {
            Operator::Lit(lit) =>  {
                if let Token::Token(token) = &tokens[0] {
                    if token == lit {
                        return (1, Result::Token(tokens[0].clone()));
                    }
                }
                (0, Result::Fail)
            },
            Operator::Any => {
                // This is an interesting question.
                // Should any match delimiters and new lines?
                // Maybe? Not sure right now
                if let Token::Token(_) = &tokens[0] {
                    (1, Result::Token(tokens[0].clone()))
                } else {
                    (0, Result::Fail)
                }
            },
            Operator::SepBy(op, sep) => {
                // This would technically match even if no comma? I think so
                // This only needs to be enough to match these ruby files
                let mut token_index = 0;
                let mut results = Vec::new();
                loop {
                    if token_index == tokens.len() {
                        return (token_index, Result::List(results));
                    }
                    let (consumed, result) = op.matches(&tokens[token_index..]);
                    if consumed == 0 {
                        token_index += 1;
                        continue;
                    }
                    results.push(result);
                    token_index += consumed;
                    if tokens[token_index] != Token::Delimiter(sep.clone()) {
                        return (token_index, Result::List(results));
                    }

               }
            },
            Operator::Named(name, op) => {
                let (consumed, result) = op.matches(tokens);
                if consumed == 0 {
                    return (0, Result::Fail);
                }
                (consumed, Result::Named(name.clone(), Box::new(result)))
            },
            Operator::Or(op1, op2) => {
                let (len1, result1) = op1.matches(tokens);
                let (len2, result2) = op2.matches(tokens);
                if len1 > len2 {
                    (len1, result1)
                } else {
                    (len2, result2)
                }
            }
            Operator::Line(ops) => {
                let mut results = Vec::new();
                let mut token_index = 0;
                let mut op_index = 0;
                loop {
                    if token_index == tokens.len() {
                        return (0, Result::Fail);
                    }
                    if tokens[token_index] == Token::NewLine {
                        token_index += 1;
                        continue;
                    }
                    break;
                }
                loop {
                    if op_index == ops.len() {
                        return (token_index, Result::List(results));
                    }
                    if token_index == tokens.len() {
                        return (0, Result::Fail);
                    }
                    if tokens[token_index] == Token::NewLine {
                        return (0, Result::Fail);
                    }
                    // let token = &tokens[token_index];
                    // println!("op: {:?}, token: {:?}", ops[op_index], token);
                    let (consumed, result) = ops[op_index].matches(&tokens[token_index..]);
                    if consumed == 0 {
                        token_index += 1;
                        continue;
                    }
                    results.push(result);
                    token_index += consumed;
                    op_index += 1;
                }
            },
            Operator::Repeat(op) => {
                let mut result = Vec::new();
                let mut token_index = 0;
                loop {
                    if token_index == tokens.len() {
                        return (0, Result::Fail);
                    }
                    let (len, res) = op.matches(&tokens[token_index..]);
                    if len == 0 {
                        break;
                    }
                    result.push(res);
                    token_index += len;
                }
                (token_index, Result::List(result))
            }
            Operator::Sequence(ops) => {

                let mut results = Vec::new();
                let mut token_index = 0;
                let mut op_index = 0;
                loop {
                    if op_index == ops.len() {
                        return (token_index, Result::List(results));
                    }
                    if token_index == tokens.len() {
                        return (0, Result::Fail);
                    }
                    let (consumed, result) = ops[op_index].matches(&tokens[token_index..]);
                    if consumed == 0 {
                        token_index += 1;
                        continue;
                    }
                    results.push(result);
                    token_index += consumed;
                    op_index += 1;
                }
            }
            Operator::Not(op) => {
                let (consumed, _result) = op.matches(tokens);
                if consumed == 0 {
                    Operator::Any.matches(tokens)
                } else {
                    (0, Result::Fail)
                }
            }
        }
    }

}

fn consume_whitespace(input: &[char], index: usize) -> usize {
    let mut index = index;
    while index < input.len() {
        let c = input.get(index).unwrap();
        if c.is_whitespace() && c != &'\n' {
            index += 1;
        } else {
            break;
        }
    }
    index
}

fn consume_token(input: &[char], delimiter: &Regex, index: usize) -> (usize, Token) {
    let start_index = index;
    let mut index = index;
    let mut token = String::new();
    while index < input.len() {
        let c = input.get(index).unwrap();
        if c.is_whitespace() {
            break;
        }
        if let Some(m) = delimiter.find(&input[index..].iter().collect::<String>()) {
            if index == start_index {
                let match_length = m.range().len();
                let token = Token::Delimiter(input[index..index + match_length].iter().collect());
                return (index + match_length, token);
            } else {
                return (index, Token::Token(token));
            }
        }
        if c.is_ascii_punctuation() && c != &'_' {
            if token.is_empty() {
                token.push(*c);
                index += 1;
                return (index, Token::Token(token));
            }
            return (index, Token::Token(token));
        }

        // maybe do this better?
        token.push(*c);
        index += 1;
    }
    if index == start_index {
        println!("no progress {}", &input[index..].iter().collect::<String>())
    }
    (index, Token::Token(token))

}

fn make_tokens(input: &String, delimiter: Regex) -> Vec<Token> {
    let mut i = 0;
    let mut tokens = Vec::new();
    let input = input.chars().collect::<Vec<_>>();
    while i < input.len() {
        i = consume_whitespace(&input, i);
        if i == input.len() {
            break;
        }
        if input[i] == '\n' {
            tokens.push(Token::NewLine);
            i += 1;
        } else {
            let (new_i, token) = consume_token(&input, &delimiter, i);
            tokens.push(token);
            i = new_i;
        }
    }
    tokens
}


macro_rules! pattern {
    ($x:literal) => {
        Operator::Lit($x.to_string())
    };
    ((lit $x:expr)) => {
        Operator::Lit($x.to_string())
    };
    (any) => {
        Operator::Any
    };
    ((any)) => {
        Operator::Any
    };
    ((sep_by $x:tt $p:tt)) => {
        Operator::SepBy(Box::new(pattern!($p)), $x.to_string())
    };
    ((repeat $x:tt)) => {
        Operator::Repeat(Box::new(pattern!($x)))
    };
    ((not $x:tt)) => {
        Operator::Not(Box::new(pattern!($x)))
    };
    ((named $x:tt $y:tt)) => {
        Operator::Named($x.to_string(), Box::new(pattern!($y)))
    };
    ((line $($x:tt)*)) => {
        Operator::Line(vec![$(pattern!($x)),*])
    };
    ((or $x:tt $y:tt)) => {
        Operator::Or(Box::new(pattern!($x)), Box::new(pattern!($y)))
    };
    ((seq $($x:tt)+)) => {
        Operator::Sequence(vec![$(pattern!($x)),+])
    };
}


pub fn read_each_file(directory: &str) {

    let pattern = pattern!(
        (seq
            (line "class" (named "class_name" any))
            (or
                (seq
                    (line "def" "encode")
                    (line (named "encode_value" any)))
                (seq
                    (line "def" "initialize" (named "init_args" (sep_by "," any)))
                    (named "initialize_body" (repeat (line (repeat (not "end")))))
                    (line "def" "encode")
                    (line "def" any (named "encode_args" (sep_by "," any)))
                    (named "encode_body" (repeat (line (repeat (not "end")))))))));

    // get file in directory
    let paths = fs::read_dir(directory).unwrap();
    let mut i = 0;
    for path in paths {
        let path = path.unwrap().path();
        let file_name = path.file_name().unwrap().to_str().unwrap();

        if file_name.ends_with(".rb") {

            let mut file = File::open(path.clone()).unwrap();
            let mut buf = Vec::new();
            file.read_to_end(&mut buf).unwrap();
            let s = String::from_utf8(buf).unwrap();
            if s.contains("NotImplementedError") {
                continue;
            }

            let regex = Regex::new(r"\A,").unwrap();
            let tokens = make_tokens(&s, regex);

            let (_index, result) = pattern.matches(&tokens);
            if result == Result::Fail {
                println!("{:?} {} {:?}", path, s, result);
            }
            println!("{}", i);
            i += 1;
        }
    }
}


