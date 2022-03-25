use std::{str::from_utf8, time::Instant, fs, error::Error};



// TODO:
// Capture positions from the beginning.
// I will need them for source maps
// Look at cljs compiler to see forms other than ast
// Import SWC and use it to codegen
// Consider if the edn -> clj step is wise
// Parse more than one top level form
// Need to parse char



// TODO:
// Is this naive strategy too slow?




// Should I deal with tags and things here?
// probably.
#[derive(Debug, Clone, Copy)]
pub enum Token {
    OpenParen,
    CloseParen,
    OpenCurly,
    CloseCurly,
    OpenBracket,
    CloseBracket,
    Hash,
    // Probably can get rid of this one sense it is comment
    SemiColon,
    Colon,
    Comma,
    NewLine,
    SingleQuote,
    Char(usize),
    Comment((usize, usize)),
    Spaces((usize, usize)),
    String((usize, usize)),
    Integer((usize, usize)),
    Float((usize, usize)),
    Symbol((usize, usize)),
}




#[derive(Debug, Clone)]
pub struct Tokenizer {
    pub position: usize,
}


impl<'a> Tokenizer {
    pub fn new() -> Tokenizer {
        Tokenizer {
            position: 0,
        }
    }

    fn peek(&self, input_bytes: &[u8]) -> Option<u8> {
        if self.position + 1 < input_bytes.len() {
            Some(input_bytes[self.position + 1])
        } else {
            None
        }
    }

    fn is_previous_escape(&self, input_bytes: &[u8]) -> bool {
        if self.position > 0 {
            let previous_byte = input_bytes[self.position - 1];
            previous_byte == b'\\' && !self.is_previous_escape_explicit(input_bytes, self.position - 1)
        } else {
            false
        }
    }
    fn is_previous_escape_explicit(&self, input_bytes: &[u8], position: usize) -> bool {
        if position > 0 {
            let previous_byte = input_bytes[position - 1];
            previous_byte == b'\\' && !self.is_previous_escape_explicit(input_bytes, position - 1)
        } else {
            false
        }
    }

    fn is_comment_start(&self, input_bytes: &[u8]) -> bool {
        input_bytes[self.position] == b';'
    }

    fn parse_comment(&mut self, input_bytes: &[u8]) -> Token {
        let start = self.position;
        while !self.at_end(input_bytes) && !self.is_newline(input_bytes) {
            self.consume();
        }
        // self.consume();
        Token::Comment((start, self.position))
    }

    pub fn consume(&mut self) {
        self.position += 1;
    }

    pub fn current_byte(&self, input_bytes: &[u8]) -> u8 {
        input_bytes[self.position]
    }

    pub fn is_space(&self, input_bytes: &[u8]) -> bool {
        self.current_byte(input_bytes) == b' '
    }

    pub fn at_end(&self, input_bytes: &[u8]) -> bool {
        self.position >= input_bytes.len()
    }

    pub fn is_quote(&self, input_bytes: &[u8]) -> bool {
        self.current_byte(input_bytes) == b'"'
    }

    pub fn parse_string(&mut self, input_bytes: &[u8]) -> Token {
        let start = self.position;
        self.consume();
        while !self.at_end(input_bytes) && (!self.is_quote(input_bytes) || (self.is_previous_escape(input_bytes) && self.is_quote(input_bytes))) {
            self.consume();
        }
        // TODO: Deal with escapes
        if !self.at_end(input_bytes) {
            self.consume();
        }
        Token::String((start, self.position))
    }

    pub fn is_open_paren(&self, input_bytes: &[u8]) -> bool {
        self.current_byte(input_bytes) == b'('
    }

    pub fn is_close_paren(&self, input_bytes: &[u8]) -> bool {
        self.current_byte(input_bytes) == b')'
    }

    pub fn is_open_curly(&self, input_bytes: &[u8]) -> bool {
        self.current_byte(input_bytes) == b'{'
    }

    pub fn is_close_curly(&self, input_bytes: &[u8]) -> bool {
        self.current_byte(input_bytes) == b'}'
    }

    pub fn is_open_bracket(&self, input_bytes: &[u8]) -> bool {
        self.current_byte(input_bytes) == b'['
    }

    pub fn is_close_bracket(&self, input_bytes: &[u8]) -> bool {
        self.current_byte(input_bytes) == b']'
    }

    pub fn is_hash(&self, input_bytes: &[u8]) -> bool {
        self.current_byte(input_bytes) == b'#'
    }

    pub fn is_backslash(&self, input_bytes: &[u8]) -> bool {
        self.current_byte(input_bytes) == b'\\'
    }

    pub fn is_single_quote(&self, input_bytes: &[u8]) -> bool {
        self.current_byte(input_bytes) == b'\''
    }

    pub fn parse_spaces(&mut self, input_bytes: &[u8]) -> Token {
        let start = self.position;
        while !self.at_end(input_bytes) && self.is_space(input_bytes) {
            self.consume();
        }
        Token::Spaces((start, self.position))
    }

    pub fn is_valid_number_char(&mut self, input_bytes: &[u8]) -> bool {
        self.current_byte(input_bytes) >= b'0' && self.current_byte(input_bytes) <= b'9'
         || (self.current_byte(input_bytes) == b'-' && !self.at_end(input_bytes) && self.peek(input_bytes).unwrap() >= b'0' && self.peek(input_bytes).unwrap() <= b'9')
    }


    pub fn parse_number(&mut self, input_bytes: &[u8]) -> Token {
        let mut is_float = false;
        let start = self.position;
        while !self.at_end(input_bytes) && (self.is_valid_number_char(input_bytes) || self.current_byte(input_bytes) == b'.') {
            // Need to handle making sure there is only one "."
            if self.current_byte(input_bytes) == b'.' {
                is_float = true;
            }
            self.consume();
        }
        if is_float {
            Token::Float((start,self.position))
        } else {
            Token::Integer((start, self.position))
        }
    }

    pub fn parse_identifier(&mut self, input_bytes: &[u8]) -> Token {
        let start = self.position;
        while !self.at_end(input_bytes)
                && !self.is_space(input_bytes)
                && !self.is_open_paren(input_bytes)
                && !self.is_close_paren(input_bytes)
                && !self.is_open_curly(input_bytes)
                && !self.is_close_curly(input_bytes)
                && !self.is_open_bracket(input_bytes)
                && !self.is_close_bracket(input_bytes)
                && !self.is_semi_colon(input_bytes)
                && !self.is_colon(input_bytes)
                && !self.is_comma(input_bytes)
                && !self.is_newline(input_bytes)
                && !self.is_hash(input_bytes)
                && !self.is_backslash(input_bytes)
                && !self.is_quote(input_bytes) {
            self.consume();
        }
        Token::Symbol((start, self.position))
    }

    pub fn parse_single(&mut self, input_bytes: &[u8]) -> Option<Token> {

        if self.at_end(input_bytes) {
            return None
        }
        let result = if self.is_space(input_bytes) {
            self.parse_spaces(input_bytes)
        } else if self.is_newline(input_bytes) {
            self.consume();
            Token::NewLine
        } else if self.is_comment_start(input_bytes) {
            self.parse_comment(input_bytes)
        } else if self.is_open_paren(input_bytes) {
            // println!("open paren");
            self.consume();
            Token::OpenParen
        } else if self.is_close_paren(input_bytes) {
            // println!("close paren");
            self.consume();
            Token::CloseParen
        } else if self.is_valid_number_char(input_bytes) {
            // println!("number");
            self.parse_number(input_bytes)
        } else if self.is_quote(input_bytes) {
            // println!("string");
            self.parse_string(input_bytes)
        } else if self.is_semi_colon(input_bytes) {
            // println!("semicolon");
            self.consume();
            Token::SemiColon
        } else if self.is_comma(input_bytes) {
            self.consume();
            Token::Comma
        } else if self.is_colon(input_bytes) {
            // println!("colon");
            self.consume();
            Token::Colon
        } else if self.is_open_curly(input_bytes) {
            // println!("open curly");
            self.consume();
            Token::OpenCurly
        } else if self.is_close_curly(input_bytes) {
            // println!("close curly");
            self.consume();
            Token::CloseCurly
        } else if self.is_open_bracket(input_bytes) {
            // println!("open bracket");
            self.consume();
            Token::OpenBracket
        } else if self.is_close_bracket(input_bytes) {
            // println!("close bracket");
            self.consume();
            Token::CloseBracket
        } else if self.is_backslash(input_bytes) {
            self.consume();
            self.consume();
            Token::Char(self.position)
        } else if self.is_hash(input_bytes) {
            // println!("close bracket");
            self.consume();
            Token::Hash
        } else if self.is_single_quote(input_bytes) {
            self.consume();
            Token::SingleQuote
        } else {
            // println!("identifier");
            self.parse_identifier(input_bytes)
        };
        Some(result)
    }

    pub fn is_semi_colon(&self, input_bytes: &[u8]) -> bool {
        self.current_byte(input_bytes) == b';'
    }

    pub fn is_colon(&self, input_bytes: &[u8]) -> bool {
        self.current_byte(input_bytes) == b':'
    }

    pub fn is_newline(&self, input_bytes: &[u8]) -> bool {
        self.current_byte(input_bytes) == b'\n'
    }

    pub fn is_comma(&self, input_bytes: &[u8]) -> bool {
        self.current_byte(input_bytes) == b','
    }

    pub fn get_line(&mut self, input_bytes: &[u8]) -> Vec<Token> {
        let mut result = Vec::new();
        while !self.at_end(input_bytes) && !self.is_newline(input_bytes) {
            if let Some(token) = self.parse_single(input_bytes) {
                result.push(token);
            }
        }
        result
    }

    pub fn _skip_lines(&mut self, n: usize, input_bytes: &[u8]) -> &mut Self {
        for _ in 0..n {
            while !self.at_end(input_bytes) && !self.is_newline(input_bytes) {
                self.consume();
            }
            if !self.at_end(input_bytes) {
                self.consume();
            }
        }
        self
    }

    // The downside of this approach is that I will parse very large buffers
    // all the way at once.
    pub fn parse_all(&mut self, input_bytes: &[u8]) -> Vec<Token> {
        let mut result = Vec::new();
        while !self.at_end(input_bytes) {
            if let Some(token) = self.parse_single(input_bytes) {
                result.push(token);
            }
        }
        self.position = 0;
        result
    }


}

fn string_from_bytes(bytes: &[u8], span: (usize, usize)) -> String {
    // Half of my time is spent doing this. I should probably not do it.
    // I should probably keep my spans instead of strings
    from_utf8(&bytes[span.0..span.1]).unwrap().to_string()
}


// TODO: Should capture line and column information.

// I'm still not sure about making this go from
// token -> edn -> clojure. But worth trying it.
// TODO: Add Comment
#[derive(Debug, Clone, PartialEq)]
enum Edn {
    Tagged(String, Box<Edn>),
    Vector(Vec<Edn>),
    Set(Vec<Edn>),
    Map(Vec<(Edn, Edn)>),
    Seq(Vec<Edn>),
    Keyword(String),
    Symbol(String),
    Str(String),
    Number(String),
    Char(char),
    Bool(bool),
    Comment(String),
    Quoted(Box<Edn>),
    // Inst(String),
    // Uuid(String),
    NamespacedMap(String, Vec<(Edn, Edn)>),
    Nil,
}

impl Edn {
    // TODO: Probably shouldn't change from spans to strings
    fn parse(tokenizer: &mut Tokenizer, input_bytes: &[u8]) -> Option<Self> {
        let token = tokenizer.parse_single(input_bytes)?;
        match token {
            Token::SingleQuote => {
                while !tokenizer.at_end(input_bytes) {
                    if let Some(edn) = Edn::parse(tokenizer, input_bytes) {
                        return Some(Edn::Quoted(Box::new(edn)))
                    }
                }
                panic!("Nothing after single quote")
            },
            Token::Char(pos) => Some(Edn::Char(input_bytes[pos].into())),
            Token::Hash => {
                // TODO: CLEAN THIS CODE UP!
                let mut result = Vec::new();
                let next_token = tokenizer.parse_single(input_bytes);
                match next_token {
                    Some(Token::OpenCurly) => {}
                    Some(Token::Symbol(span)) => {
                        while !tokenizer.at_end(input_bytes) && !tokenizer.is_close_curly(input_bytes) {
                            if let Some(edn) = Edn::parse(tokenizer, input_bytes) {
                                return Some(Edn::Tagged(string_from_bytes(input_bytes, span), Box::new(edn)));
                            }
                        }
                    }
                    // What to do with functions?
                    Some(Token::OpenParen) => {
                        tokenizer.position -= 1;
                        while !tokenizer.at_end(input_bytes) {
                            if let Some(edn) = Edn::parse(tokenizer, input_bytes) {
                                // TODO: Fix
                                return Some(Edn::Tagged("fn".to_string(), Box::new(edn)));
                            }
                        }
                    }
                    Some(Token::String(span)) => {
                        // TODO: Fix
                        return Some(Edn::Tagged("regex".to_string(), Box::new(Edn::Str(string_from_bytes(input_bytes, span)))));
                    }

                    Some(Token::Hash) => {
                        let next_token = tokenizer.parse_single(input_bytes);
                        match next_token {
                            Some(Token::Symbol(span)) => {
                                return Some(Edn::Number(string_from_bytes(input_bytes, span)))
                            }
                            token => panic!("Hash that isn't defined{:?}", token)
                        }
                    }
                    token => panic!("Hash that isn't defined {:?}", token)
                }
                while !tokenizer.at_end(input_bytes) && !tokenizer.is_close_curly(input_bytes) {
                    if let Some(edn) = Edn::parse(tokenizer, input_bytes) {
                        result.push(edn);
                    }
                }
                if !tokenizer.at_end(input_bytes) && tokenizer.is_close_curly(input_bytes) {
                    tokenizer.consume();
                }

                Some(Edn::Set(result))
            },
            Token::OpenParen => {
                let mut result = Vec::new();
                while !tokenizer.at_end(input_bytes) && !tokenizer.is_close_paren(input_bytes) {
                    if let Some(edn) = Edn::parse(tokenizer, input_bytes) {
                        result.push(edn);
                    }
                }
                if !tokenizer.at_end(input_bytes) && tokenizer.is_close_paren(input_bytes) {
                    tokenizer.consume();
                }

                Some(Edn::Seq(result))
            },
            Token::CloseParen => {
                None
            },
            Token::OpenCurly => {
                let mut result = Vec::new();
                while !tokenizer.at_end(input_bytes) && !tokenizer.is_close_curly(input_bytes) {
                    // Need to detect uneven numbers of things.
                    if let Some(key) = Edn::parse(tokenizer, input_bytes) {
                        while !tokenizer.at_end(input_bytes) {
                            if let Some(value) = Edn::parse(tokenizer, input_bytes) {
                                result.push((key, value));
                                break;
                            }
                        }
                    }
                }
                if !tokenizer.at_end(input_bytes) && tokenizer.is_close_curly(input_bytes) {
                    tokenizer.consume();
                }
                return Some(Edn::Map(result));
            },
            Token::CloseCurly => None,
            Token::OpenBracket => {
                let mut result = Vec::new();
                while !tokenizer.at_end(input_bytes) && !tokenizer.is_close_bracket(input_bytes) {
                    if let Some(edn) = Edn::parse(tokenizer, input_bytes) {
                        result.push(edn);
                    }
                }
                if !tokenizer.at_end(input_bytes) && tokenizer.is_close_bracket(input_bytes) {
                    tokenizer.consume();
                }
                return Some(Edn::Vector(result));
            },
            Token::CloseBracket => None,
            Token::SemiColon => None,
            Token::Colon => {
                // Technically I think Empty colon is valid edn.
                // Not concerned so much with correctness right now
                // Just want to get something going.
                match tokenizer.parse_single(input_bytes) {
                    Some(Token::Symbol(span)) => Some(Edn::Keyword(string_from_bytes(input_bytes, span))),
                    // TODO: Make this better
                    Some(Token::Colon) => {
                        match tokenizer.parse_single(input_bytes) {
                            // TODO: Make this qualified
                            Some(Token::Symbol(span)) => Some(Edn::Keyword(string_from_bytes(input_bytes, span))),
                            x => panic!("Invalid token after colon {:?}", x),
                        }
                    }
                    x => panic!("Invalid token after colon {:?}", x),
                }
            }
            Token::Comma => None,
            Token::NewLine => None,
            Token::Comment(span) => Some(Edn::Comment(string_from_bytes(input_bytes, span))),
            Token::Spaces(_) => None,
            Token::String(span) => Some(Edn::Str(string_from_bytes(input_bytes, span))),
            Token::Integer(span) => Some(Edn::Number(string_from_bytes(input_bytes, span))),
            Token::Float(span) => Some(Edn::Number(string_from_bytes(input_bytes, span))),
            Token::Symbol(span) => {
                let s = string_from_bytes(input_bytes, span);
                match s.as_str() {
                    "true" => Some(Edn::Bool(true)),
                    "false" => Some(Edn::Bool(false)),
                    "nil" => Some(Edn::Nil),
                    s =>  Some(Edn::Symbol(s.to_string()))
                }
            },
        }

    }
}

#[derive(Debug, Clone, PartialEq)]
enum Clojure {
    Catch(Catch),
    Const(Const),
    Def(Def),
    Do(Vec<Clojure>),
    Fn(Fn),
    HostCall(HostCall),
    HostField(HostField),
    If(If),
    Invoke(Invoke),
    Let(Let),
    LetFn(LetFn),
    Loop(Loop),
    Map(Map),
    MaybeHostForm(MaybeHostForm),
    New(New),
    Quote(Quote),
    Recur(Recur),
    Set(Set),
    SetBang(SetBang),
    Throw(Throw),
    Try(Try),
    Var(Var),
    Vector(Vector),
    WithMeta(WithMeta),
    Comment(String),
    Todo,
}



impl Clojure {

    fn boxed_from_edn(edn: Edn) -> Box<Self> {
        Box::new(Clojure::from_edn(edn))
    }

    // This niave implementation is cool,
    // but also can blow the stack.
    // Consider being smarter
    fn from_edn(edn: Edn) -> Self {
        match edn {
            Edn::Quoted(_) => Clojure::Todo,
            Edn::Comment(s) => Clojure::Comment(s),
            Edn::Tagged(_, _) => Clojure::Todo,
            Edn::Vector(v) => Clojure::Vector(Vector{ items: v.into_iter().map(Clojure::from_edn).collect()}),
            Edn::Set(_) => Clojure::Todo,
            Edn::Map(items) => Clojure::Map(Map{ items: items.into_iter().map(|(k, v)| (Clojure::from_edn(k), Clojure::from_edn(v))).collect()}),
            Edn::Seq(mut items) => {
                match items.first() {
                    Some(f@Edn::Symbol(s)) => {
                        match s.as_str() {
                            "if" => {
                                // TODO: Need to actually check if I have that many arms of the if
                                // Doing clones where was incredibly slow.
                                let else_ = items.pop().unwrap();
                                let then = items.pop().unwrap();
                                let test = items.pop().unwrap();


                                Clojure::If(If {
                                    test: Clojure::boxed_from_edn(test),
                                    then: Clojure::boxed_from_edn(then),
                                    else_: Clojure::boxed_from_edn(else_),
                                })
                            },
                            _ => {
                                Clojure::Invoke(Invoke {
                                    func: Box::new(Clojure::from_edn(f.clone())),
                                    args: items.into_iter().skip(1).map(Clojure::from_edn).collect(),
                                })
                            }
                        }
                    },
                    Some(f) => {
                        Clojure::Invoke(Invoke {
                            func: Box::new(Clojure::from_edn(f.clone())),
                            args: items.into_iter().skip(1).map(Clojure::from_edn).collect(),
                        })
                    }
                    _ => Clojure::Todo,
                }
            },
            Edn::Keyword(s) => Clojure::Const(Const::Keyword(s)),
            Edn::Symbol(s) => Clojure::Const(Const::Symbol(s)),
            Edn::Str(s) => Clojure::Const(Const::String(s)),
            Edn::Number(s) => Clojure::Const(Const::Number(s)),
            Edn::Char(_) => Clojure::Todo,
            Edn::Bool(b) => Clojure::Const(Const::Bool(b)),
            Edn::NamespacedMap(_, _) => Clojure::Todo,
            Edn::Nil => Clojure::Const(Const::Nil),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
struct Catch {
    class: Box<Clojure>,
    local: Box<Clojure>,
    body: Box<Clojure>,
}

#[derive(Debug, Clone, PartialEq)]
enum Const {
    // TODO: How to deal with this?
    // The real one even has collections
    // Not sure when to use what.
    Keyword(String),
    Symbol(String),
    String(String),
    Number(String),
    Char(char),
    Bool(bool),
    Nil,
    Other(String),
}


#[derive(Debug, Clone, PartialEq)]
struct Def {
    name: String,
    meta: Option<Edn>,
    doc: Option<String>,
    value: Option<Edn>,
}

#[derive(Debug, Clone, PartialEq)]
struct Fn {
    name: String,
    meta: Option<Edn>,
    doc: Option<String>,
    params: Vec<String>,
    body: Vec<Clojure>,
}

#[derive(Debug, Clone, PartialEq)]
struct HostCall {
    target: Box<Clojure>,
    method: Box<Clojure>,
    args: Vec<Clojure>,
}

#[derive(Debug, Clone, PartialEq)]
struct HostField {
    target: Box<Clojure>,
    field: Box<Clojure>,
}

#[derive(Debug, Clone, PartialEq)]
struct If {
    test: Box<Clojure>,
    then: Box<Clojure>,
    else_: Box<Clojure>,
}

#[derive(Debug, Clone, PartialEq)]
struct Invoke {
    func: Box<Clojure>,
    args: Vec<Clojure>,
}

#[derive(Debug, Clone, PartialEq)]
struct Let {
    bindings: Vec<(Clojure, Clojure)>,
    body: Box<Clojure>,
}

#[derive(Debug, Clone, PartialEq)]
struct LetFn {
    bindings: Vec<(Clojure, Clojure)>,
    body: Box<Clojure>,
}

#[derive(Debug, Clone, PartialEq)]
struct Loop {
    bindings: Vec<(Clojure, Clojure)>,
    body: Box<Clojure>,
}

#[derive(Debug, Clone, PartialEq)]
struct Map {
    items: Vec<(Clojure, Clojure)>,
}

#[derive(Debug, Clone, PartialEq)]
struct MaybeHostForm {
    class: Box<Clojure>,
    field: Box<Clojure>,
}

#[derive(Debug, Clone, PartialEq)]
struct New {
    class: Box<Clojure>,
    args: Vec<Clojure>,
}

#[derive(Debug, Clone, PartialEq)]
struct Quote {
    exprs: Vec<Clojure>
}

#[derive(Debug, Clone, PartialEq)]
struct Recur {
    exprs: Vec<Clojure>,
}

#[derive(Debug, Clone, PartialEq)]
struct Set {
    items: Vec<Clojure>
}

#[derive(Debug, Clone, PartialEq)]
struct SetBang {
    target: Box<Clojure>,
    val: Box<Clojure>,
}

#[derive(Debug, Clone, PartialEq)]
struct Throw {
    exception: Box<Clojure>,
}

#[derive(Debug, Clone, PartialEq)]
struct Try {
    body: Box<Clojure>,
    catches: Vec<Clojure>,
    finally: Box<Option<Clojure>>,
}

#[derive(Debug, Clone, PartialEq)]
struct Var {
    var: Box<Clojure>
}

#[derive(Debug, Clone, PartialEq)]
struct Vector {
    items: Vec<Clojure>
}

#[derive(Debug, Clone, PartialEq)]
struct WithMeta {
    meta: Box<Clojure>,
    expr: Box<Clojure>,
}


fn generate_large_nested_if() -> String {
    let n = 1000;
    let mut result = "".to_string();
    for _ in 0..n {
        result.push_str("(if ");
    }
    result.push_str("true 1 2)");
    for _ in 0..(n - 1) {
        result.push_str(" 1 2)");
    }
    result
}


fn generate_large_vector(n: usize) -> String {
    let mut result = "".to_string();
    result.push_str("[");
    for i in 0..n {
        result.push_str(i.to_string().as_str());
    }
    result.push_str("]");

    result
}



fn main() -> Result<(), Box<dyn Error>> {
    let start_time = Instant::now();
    let expr = fs::read_to_string("/Users/jimmyhmiller/Downloads/core.cljs")?;
    // let expr = fs::read_to_string("/Users/jimmyhmiller/Downloads/core_large.cljs")?;
    // let expr = generate_large_vector(100000);
    // let expr = fs::read_to_string("/Users/jimmyhmiller/Downloads/test.clj")?;


    let mut parsed_forms: Vec<Edn> = vec![];
    let mut tokenizer = Tokenizer::new();
    let input_bytes = expr.as_bytes();

    // println!("{:?}", tokenizer.parse_all(input_bytes));
    // tokenizer.position = 0;

    while !tokenizer.at_end(input_bytes) {
        if let Some(edn)= Edn::parse(&mut tokenizer, input_bytes) {
            // println!("{:?}\n\n", edn);
            parsed_forms.push(edn);
        }
    }
    println!("done\n\n\n\n\n");

    // for form in parsed_forms {
    //     println!("{:?}", Clojure::from_edn(form));
    // }


    println!("{}", parsed_forms.len());

    println!("{:?}", start_time.elapsed());


    Ok(())
}
