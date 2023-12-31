// Stolen from my edtior, so probably not great
// Need to deal with failure?
// Maybe not at the token level?

use crate::ast::Ast;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Token {
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
    If,
    Fn,
    Else,
    LessThanOrEqual,
    LessThan,
    Equal,
    NotEqual,
    GreaterThan,
    GreaterThanOrEqual,
    Comment((usize, usize)),
    Spaces((usize, usize)),
    String((usize, usize)),
    Integer((usize, usize)),
    Float((usize, usize)),
    // I should replace this with builtins
    // like fn and stuff
    Atom((usize, usize)),
}

static ZERO: u8 = b'0';
static NINE: u8 = b'9';
static SPACE: u8 = b' ';
static NEW_LINE: u8 = b'\n';
static DOUBLE_QUOTE: u8 = b'"';
static OPEN_PAREN: u8 = b'(';
static CLOSE_PAREN: u8 = b')';
static PERIOD: u8 = b'.';

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

    fn is_comment_start(&self, input_bytes: &[u8]) -> bool {
        input_bytes[self.position] == b'/' && self.peek(input_bytes) == Some(b'/')
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
        self.current_byte(input_bytes) == SPACE
    }

    pub fn at_end(&self, input_bytes: &[u8]) -> bool {
        self.position >= input_bytes.len()
    }

    pub fn is_quote(&self, input_bytes: &[u8]) -> bool {
        self.current_byte(input_bytes) == DOUBLE_QUOTE
    }

    pub fn parse_string(&mut self, input_bytes: &[u8]) -> Token {
        let start = self.position;
        self.consume();
        while !self.at_end(input_bytes) && !self.is_quote(input_bytes) {
            self.consume();
        }
        // TODO: Deal with escapes
        if !self.at_end(input_bytes) {
            self.consume();
        }
        Token::String((start, self.position))
    }

    pub fn is_open_paren(&self, input_bytes: &[u8]) -> bool {
        self.current_byte(input_bytes) == OPEN_PAREN
    }

    pub fn is_close_paren(&self, input_bytes: &[u8]) -> bool {
        self.current_byte(input_bytes) == CLOSE_PAREN
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

    pub fn parse_spaces(&mut self, input_bytes: &[u8]) -> Token {
        let start = self.position;
        while !self.at_end(input_bytes) && self.is_space(input_bytes) {
            self.consume();
        }
        Token::Spaces((start, self.position))
    }

    pub fn is_valid_number_char(&mut self, input_bytes: &[u8]) -> bool {
        self.current_byte(input_bytes) >= ZERO && self.current_byte(input_bytes) <= NINE
    }

    pub fn parse_number(&mut self, input_bytes: &[u8]) -> Token {
        let mut is_float = false;
        let start = self.position;
        while !self.at_end(input_bytes) && (self.is_valid_number_char(input_bytes) || self.current_byte(input_bytes) == PERIOD) {
            // Need to handle making sure there is only one "."
            if self.current_byte(input_bytes) == PERIOD {
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
                && !self.is_quote(input_bytes) {
            self.consume();
        }
        match &input_bytes[start..self.position] {
            b"fn" => Token::Fn,
            b"if" => Token::If,
            b"else" => Token::Else,
            b"<=" => Token::LessThanOrEqual,
            b"<" => Token::LessThan,
            b"==" => Token::Equal,
            b"!=" => Token::NotEqual,
            b">" => Token::GreaterThan,
            b">=" => Token::GreaterThanOrEqual,
            _ => Token::Atom((start, self.position)),
        }
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
            self.consume();
            Token::OpenParen
        } else if self.is_close_paren(input_bytes) {
            self.consume();
            Token::CloseParen
        } else if self.is_valid_number_char(input_bytes) {
            self.parse_number(input_bytes)
        } else if self.is_quote(input_bytes) {
            self.parse_string(input_bytes)
        } else if self.is_semi_colon(input_bytes) {
            self.consume();
            Token::SemiColon
        } else if self.is_comma(input_bytes) {
            self.consume();
            Token::Comma
        } else if self.is_colon(input_bytes) {
            self.consume();
            Token::Colon
        } else if self.is_open_curly(input_bytes) {
            self.consume();
            Token::OpenCurly
        } else if self.is_close_curly(input_bytes) {
            self.consume();
            Token::CloseCurly
        } else if self.is_open_bracket(input_bytes) {
            self.consume();
            Token::OpenBracket
        } else if self.is_close_bracket(input_bytes) {
            self.consume();
            Token::CloseBracket
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
        self.current_byte(input_bytes) == NEW_LINE
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

#[test]
fn test_tokenizer1() {
    let mut tokenizer = Tokenizer::new();
    let input = "hello world";
    let input_bytes = input.as_bytes();
    let result = tokenizer.parse_all(input_bytes);
    assert_eq!(result.len(), 3);
    assert_eq!(result[0], Token::Atom((0, 5)));
    assert_eq!(result[1], Token::Spaces((5, 6)));
    assert_eq!(result[2], Token::Atom((6, 11)));
}


pub struct Parser {
    source: String,
    tokenizer: Tokenizer,
    position: usize,
    tokens: Vec<Token>,
}

impl Parser {
    pub fn new(source: String) -> Parser {
        let mut tokenizer = Tokenizer::new();
        let input_bytes = source.as_bytes();
        // TODO: I is probably better not to parse all at once
        let tokens = tokenizer.parse_all(input_bytes);
        Parser {
            source,
            tokenizer,
            position: 0,
            tokens,
        }
    }

    pub fn parse(&mut self) -> Ast {
        Ast::Program {
            elements: self.parse_elements(),
        }
    }

    fn parse_elements(&mut self) -> Vec<Ast> {
        let mut result = Vec::new();
        while !self.at_end() {
            if let Some(elem) = self.parse_expression() {
                result.push(elem);
            } else {
                break;
            }
            
        }
        result
    }

    fn at_end(&self) -> bool {
        self.position >= self.tokens.len()
    }

    // TODO: I need to deal with precedence and parsing
    // binary operators
    // Probably use this:
    // https://eli.thegreenplace.net/2012/08/02/parsing-expressions-by-precedence-climbing
    fn parse_expression(&mut self) -> Option<Ast> {
        match self.tokens[self.position] {
            Token::Fn => {
                self.to_next_atom();
                Some(self.parse_function())
            }
            Token::If => {
                self.to_next_non_whitespace();
                Some(self.parse_if())
            }
            Token::Atom((start, end)) => {
                // Gross
                let name = String::from_utf8(self.source[start..end].as_bytes().to_vec()).unwrap();
                self.consume();
                Some(self.parse_call(name))
            }
            Token::String((start, end)) => {
                // Gross
                let value = String::from_utf8(self.source[start+1..end-1].as_bytes().to_vec()).unwrap();
                self.consume();
                Some(Ast::String(value))
            }
            Token::NewLine | Token::Spaces(_) | Token::Comment(_) => {
                self.consume();
                self.parse_expression()
            }
            _ => None
        }
    }

    fn parse_function(&mut self) -> Ast {
        let name = match self.tokens[self.position] {
            Token::Atom((start, end)) => {
                // Gross
                String::from_utf8(self.source[start..end].as_bytes().to_vec()).unwrap()
            }
            _ => panic!("Expected function name"),
        };
        self.to_next_non_whitespace();
        self.expect_open_paren();
        let args = self.parse_args();
        self.expect_close_paren();
        let body = self.parse_block();
        Ast::Function {
            name,
            args,
            body,
        }
    }

    fn consume(&mut self) {
        self.position += 1;
    }

    fn to_next_atom(&mut self) {
        self.consume();
        while !self.at_end() && !self.is_atom() {
            self.consume();
        }
    }

    fn to_next_non_whitespace(&mut self) {
        self.consume();
        while !self.at_end() && self.is_whitespace() {
            self.consume();
        }
    }

    fn expect_open_paren(&mut self) {
        if self.is_open_paren() {
            self.consume();
        } else {
            panic!("Expected open paren {:?}", self.get_token_repr());
        }
    }

    fn is_open_paren(&self) -> bool {
        self.tokens[self.position] == Token::OpenParen
    }

    fn parse_args(&mut self) -> Vec<String> {
        let mut result = Vec::new();
        while !self.at_end() && !self.is_close_paren() {
            result.push(self.parse_arg());
        }
        result
    }

    fn parse_arg(&mut self) -> String {
        match self.tokens[self.position] {
            Token::Atom((start, end)) => {
                // Gross
                let name = String::from_utf8(self.source[start..end].as_bytes().to_vec()).unwrap();
                self.consume();
                name
            }
            _ => panic!("Expected arg"),
        }
    }

    fn expect_close_paren(&mut self) {
        if self.is_close_paren() {
            self.consume();
        } else {
            panic!("Expected close paren");
        }
    }

    fn is_close_paren(&self) -> bool {
        self.tokens[self.position] == Token::CloseParen
    }

    fn parse_block(&mut self) -> Vec<Ast> {
        self.to_next_non_whitespace();
        self.expect_open_curly();
        let mut result = Vec::new();
        while !self.at_end() && !self.is_close_curly() {
            if let Some(elem) = self.parse_expression() {
                result.push(elem);
            } else {
                break;
            }
        }
        self.expect_close_curly();
        result
    }

    fn expect_open_curly(&mut self) {
        if self.is_open_curly() {
            self.consume();
        } else {
            panic!("Expected open curly");
        }
    }

    fn is_open_curly(&self) -> bool {
        self.tokens[self.position] == Token::OpenCurly
    }

    fn is_close_curly(&self) -> bool {
        self.tokens[self.position] == Token::CloseCurly
    }

    fn expect_close_curly(&mut self) {
        if self.is_close_curly() {
            self.consume();
        } else {
            panic!("Expected close curly");
        }
    }

    fn is_atom(&self) -> bool {
        match self.tokens[self.position] {
            Token::Atom(_) => true,
            _ => false,
        }
    }

    fn peek(&self) -> Token {
        // TODO: Handle end
        self.tokens[self.position + 1]
    }

    fn parse_call(&mut self, name: String) -> Ast {
        self.to_next_non_whitespace();
        self.expect_open_paren();
        let mut args = Vec::new();
        while !self.at_end() && !self.is_close_paren() {
            if let Some(arg) = self.parse_expression() {
                args.push(arg);
            } else {
                break;
            }
        }
        self.expect_close_paren();
        Ast::Call {
            name,
            args,
        }
    }

    fn get_token_repr(&self) -> String {
        match self.tokens[self.position] {
            Token::Atom((start, end)) => {
                String::from_utf8(self.source[start..end].as_bytes().to_vec()).unwrap()
            }
            _ => format!("{:?}", self.tokens[self.position]),
        }
    }

    fn is_whitespace(&self) -> bool {
        match self.tokens[self.position] {
            Token::Spaces(_) | Token::NewLine | Token::Comment(_) => true,
            _ => false,
        }
    }

    // TODO: If I'm not going to have null
    // how do I have an empty else?
    // or do I require an else block?
    // If the if were a statement, then it would be fine
    // but it's an expression

    fn parse_if(&mut self) -> Ast {
        let condition = Box::new(self.parse_expression().unwrap());
        self.to_next_non_whitespace();
        self.expect_open_curly();
        let then = self.parse_block();
        self.to_next_non_whitespace();
        self.expect_close_curly();
        self.to_next_non_whitespace();
        if self.is_else() {
            self.consume();
            self.expect_open_curly();
            let else_ = self.parse_block();
            self.expect_close_curly();
            Ast::If {
                condition,
                then,
                else_,
            }
        } else {
            Ast::If {
                condition,
                then,
                else_: Vec::new(),
            }
        }

    }


    fn is_else(&self) -> bool {
        match self.tokens[self.position] {
            Token::Else => true,
            _ => false,
        }
    }
}

#[test]
fn test_tokenizer2() {
    let mut tokenizer = Tokenizer::new();
    let input = "
        fn hello() {
            print(\"Hello World!\")
        }
    ";
    let input_bytes = input.as_bytes();
    let result = tokenizer.parse_all(input_bytes);
    println!("{:#?}", result);
}


#[test]
fn test_parse() {
    let mut parser = Parser::new(String::from("
    fn hello() {
        print(\"Hello World!\")
    }"));

    let ast = parser.parse();
    println!("{:#?}", ast);
}
   