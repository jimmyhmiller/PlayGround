use std::str::from_utf8;


#[derive(Debug)]
enum OpCode {
    // Do I need to assign these to u8s? Am I helping myself in anyway?
    // I guess maybe because I'm storing things like constants in the bytearray?
    // Really not sure yet
    Constant = 0,
    Return = 1,
    Negate = 2,
    Add = 3,
    Subtract = 4,
    Multiply = 5,
    Divide = 6,

}

type Value = f64;


struct Chunk {
    instructions: Vec<u8>,
    lines: Vec<usize>,
    constants: Vec<Value>,
}

impl Chunk {

    fn new() -> Chunk {
        Chunk {
            instructions: vec![],
            lines: vec![],
            constants: vec![],
        }
    }
    fn add_constant(&mut self, value: Value) -> u8 {
        self.constants.push(value);
        return (self.constants.len() - 1).try_into().unwrap();
    }

    fn write_byte(&mut self, byte: u8, line: usize) {
        self.instructions.push(byte);
        self.lines.push(line);
    }

    fn disassemble(&self, offset: usize) -> usize {
        let mut new_offset = offset;
    
        print!("{:04} ", offset);
        if offset > 0 && self.lines[offset] == self.lines[offset-1] {
            print!("   | ");
        } else {
            print!("{:4} ", self.lines[offset]);
        }
    
        let op_code = self.instructions[offset];
        
        match op_code {
            0 => { 
                print!("Constant {}", self.constants[self.instructions[offset+1] as usize]);
                new_offset += 2;
            },
            1 => {
                print!("Return");
                new_offset += 1;
            },
            _ => {
                print!("Unknown opcode: {}", op_code);
                new_offset += 1;
            },
        }
    
        println!("");
    
        new_offset
    }
    
    
}


enum InterpretResult {
    Ok,
    CompileError(String),
    RunTimeError(String),
}


struct Vm {
    chunk: Chunk,
    ip: usize,
    stack: [Value; 256],
    stack_top: usize,
}


macro_rules! binary_op {
    ($vm:expr, $op:expr) => {
        let b = $vm.pop();
        let a = $vm.pop();
        $vm.push($op(a, b));
    };
}

impl Vm {
    fn new() -> Vm {
        Vm {
            chunk: Chunk::new(),
            ip: 0,
            stack: [0.0; 256],
            stack_top: 0,
        }
    }

    fn reset_stack(&mut self) {
        self.stack_top = 0;
    }

    fn push(&mut self, value: Value) {
        self.stack[self.stack_top] = value;
        self.stack_top += 1;
    }

    fn pop(&mut self) -> Value {
        self.stack_top -= 1;
        return self.stack[self.stack_top];
    }

    fn interpret(&mut self, source: Vec<u8>) -> InterpretResult {
        let mut chunk = Chunk::new();
        if !self.compile(source, &mut chunk) {
            return InterpretResult::CompileError("Error compiling".to_string());
        }
        self.chunk = chunk;
        self.ip = 0;
        self.run()
    }

    fn compile(&mut self, source: Vec<u8>, chunk: &mut Chunk) -> bool {
        let mut parser = Parser::new(source);
        parser.advance();
        self.expression();
        parser.consume(TokenKind::Eof, "Expected end of expression");
        return true
    }

    fn expression(&mut self) {

    }

    fn run(&mut self) -> InterpretResult {
        loop {
            let ip = self.ip;
            let instruction = self.chunk.instructions[ip];
            let instruction: OpCode = unsafe { std::intrinsics::transmute(instruction as i8) };

            match instruction {
                OpCode::Constant => {
                    let constant_index = self.chunk.instructions[ip+1] as usize;
                    let constant = self.chunk.constants[constant_index];
                    self.ip += 1;
                    self.push(constant);
                    println!("Constant: {}", constant);
                }
                OpCode::Return => {
                    println!("Return {}", self.pop());
                    return InterpretResult::Ok;
                }
                OpCode::Negate => {
                    let value = self.pop();
                    self.push(-value);
                    println!("Negate: {}", value);
                }
                OpCode::Add => {
                    binary_op!(self, |a, b| a + b);
                    println!("Add");
                }
                OpCode::Subtract => {
                    binary_op!(self, |a, b| a - b);
                    println!("Subtract");
                }
                OpCode::Multiply => {
                    binary_op!(self, |a, b| a * b);
                    println!("Multiply");
                }
                OpCode::Divide => {
                    binary_op!(self, |a, b| a / b);
                    println!("Divide");
                }

            }
            self.ip += 1;
        }
    }
}

#[derive(Debug, PartialEq, Eq, Clone)]
enum TokenKind {
    LeftParen,
    RightParen,
    LeftBrace,
    RightBrace,
    Comma,
    Dot,
    Minus,
    Plus,
    Semicolon,
    Slash,
    Star,
    Bang,
    BangEqual,
    Equal,
    EqualEqual,
    Greater,
    GreaterEqual,
    Less,
    LessEqual,
    Identifier,
    String,
    Number,
    And,
    Class,
    Else,
    False,
    For,
    Fun,
    If,
    Nil,
    Or,
    Print,
    Return,
    Super,
    This,
    True,
    Var,
    While,
    // The c interpreter relies on pointers to a string
    // Does this make things worse for us? I doubt much
    Error(String),
    Eof,
}

#[derive(Debug, Clone)]
struct Token {
    kind: TokenKind,
    start: usize,
    length: usize,
    line: usize,
}

struct Parser {
    scanner: Scanner,
    current: Token,
    previous: Token,
    had_error: bool,
}

impl Parser {
    fn new(source: Vec<u8>) -> Parser {
        Parser {
            scanner: Scanner::new(source),
            current: Token {
                kind: TokenKind::Eof,
                start: 0,
                length: 0,
                line: 0,
            },
            previous: Token {
                kind: TokenKind::Eof,
                start: 0,
                length: 0,
                line: 0,
            },
            had_error: false,
        }
    }

    fn advance(&mut self) {
        // TODO: Get rid of clone
        self.previous = self.current.clone();
        loop {
            self.current = self.scanner.scan_token();
            match self.current.kind {
                TokenKind::Error(_) => {
                    println!("Error: {:?}", self.current);
                }
                _ => break,
            }
        }
    }

    fn error_at_current(&mut self, message: &str) {
        self.error_at(message);
    }

    fn error_at(&mut self, message: &str) {
        println!("Error at line {:?}: {}", self.current, message);
        self.had_error = true;
    }

    fn consume(&mut self, kind: TokenKind, message: &str) {
        if self.current.kind == kind {
            self.advance();
        } else {
            self.error_at_current(message);
        }
    }
}

struct Scanner {
    source: Vec<u8>,
    start: usize,
    current: usize,
    line: usize,
}

impl Scanner {
    fn new(source: Vec<u8>) -> Scanner {
        Scanner {
            source,
            start: 0,
            current: 0,
            line: 1,
        }
    }

    fn is_at_end(&self) -> bool {
       self.source[self.current] as char == '\0'
    }

    fn scan_token(&mut self) -> Token {
        self.start = self.current;
        if self.is_at_end() { 
            return self.make_token(TokenKind::Eof); 
        }

        let c = self.advance();
        if c.is_digit(10) {
            return self.number();
        }
        if c.is_alphabetic() {
            return self.identifier();
        }
        match c {
            '(' => self.make_token(TokenKind::LeftParen),
            ')' => self.make_token(TokenKind::RightParen),
            '{' => self.make_token(TokenKind::LeftBrace),
            '}' => self.make_token(TokenKind::RightBrace),
            ';' => self.make_token(TokenKind::Semicolon),
            ',' => self.make_token(TokenKind::Comma),
            '.' => self.make_token(TokenKind::Dot),
            '-' => self.make_token(TokenKind::Minus),
            '+' => self.make_token(TokenKind::Plus),
            '/' => self.make_token(TokenKind::Slash),
            '*' => self.make_token(TokenKind::Star),
            '!' => {
                if self.match_char('=') {
                    self.make_token(TokenKind::BangEqual)
                } else {
                    self.make_token(TokenKind::Bang)
                }
            }
            '=' => {
                if self.match_char('=') {
                    self.make_token(TokenKind::EqualEqual)
                } else {
                    self.make_token(TokenKind::Equal)
                }
            }
            '<' => {
                if self.match_char('=') {
                    self.make_token(TokenKind::LessEqual)
                } else {
                    self.make_token(TokenKind::Less)
                }
            }
            '>' => {
                if self.match_char('=') {
                    self.make_token(TokenKind::GreaterEqual)
                } else {
                    self.make_token(TokenKind::Greater)
                }
            }
            '"' => self.string(),

            _ => self.error_token("Unknown token")
        }
    }

    fn identifier(&mut self) -> Token {
        while self.peek().is_alphanumeric() {
            self.advance();
        }
        let kind = self.identifier_type();
        self.make_token(kind)
    }

    fn identifier_type(&self) -> TokenKind {
        let identifier = from_utf8(&self.source[self.start..self.current]).unwrap();
        match identifier {
            "and" => TokenKind::And,
            "class" => TokenKind::Class,
            "else" => TokenKind::Else,
            "false" => TokenKind::False,
            "for" => TokenKind::For,
            "fun" => TokenKind::Fun,
            "if" => TokenKind::If,
            "nil" => TokenKind::Nil,
            "or" => TokenKind::Or,
            "print" => TokenKind::Print,
            "return" => TokenKind::Return,
            "super" => TokenKind::Super,
            "this" => TokenKind::This,
            "true" => TokenKind::True,
            "var" => TokenKind::Var,
            "while" => TokenKind::While,
            _ => TokenKind::Identifier,
        }
    }

    fn number(&mut self) -> Token {
        while self.peek().is_digit(10) {
            self.advance();
        }
        if self.peek() == '.' && self.peek().is_digit(10) {
            self.advance();
            while self.peek().is_digit(10) {
                self.advance();
            }
        }
        self.make_token(TokenKind::Number)
    }

    fn string(&mut self) -> Token {
        while self.peek() != '"' && !self.is_at_end() {
            if self.peek() == '\n' {
                self.line += 1;
            }
            self.advance();
        }
        if self.is_at_end() {
            return self.error_token("Unterminated string");
        }
        self.advance();
        self.make_token(TokenKind::String)
    }

    fn match_char(&mut self, expected: char) -> bool {
        if self.is_at_end() {
            return false;
        }
        if self.source[self.current] as char == expected {
            self.current += 1;
            return true;
        }
        return false;
    }

    fn skip_whitespace(&mut self) {
        while !self.is_at_end() && (self.source[self.current] as char).is_whitespace() {
            // TODO: Deal with Comments
            self.advance();
        }
    }

    fn peek(&self) -> char {
        return self.source[self.current] as char;
    }

    fn advance(&mut self) -> char {
        self.current += 1;
        self.source[self.current - 1] as char
    }

    fn error_token(&self, message: &str) -> Token {
        Token {
            kind: TokenKind::Error(message.to_string()),
            start: self.start,
            length: self.current - self.start,
            line: self.line,
        }
    }

    fn make_token(&mut self, kind: TokenKind) -> Token {
        return Token {
            kind,
            start: self.start,
            length: self.current - self.start,
            line: self.line,
        };
    }
}



fn main() {
    let mut chunk = Chunk {
        instructions: vec![],
        lines: vec![],
        constants: vec![],
    };

    let constant1 = chunk.add_constant(3.4);
    let constant2 = chunk.add_constant(5.6);
    
    chunk.write_byte(OpCode::Constant as u8, 123);
    chunk.write_byte(constant1, 123);
    chunk.write_byte(OpCode::Constant as u8, 123);
    chunk.write_byte(constant1, 123);
    chunk.write_byte(OpCode::Add as u8, 123);

    chunk.write_byte(OpCode::Constant as u8, 123);
    chunk.write_byte(constant2, 123);

    chunk.write_byte(OpCode::Divide as u8, 123);

    chunk.write_byte(OpCode::Negate as u8, 123);
    chunk.write_byte(OpCode::Return as u8, 123);

    let mut vm = Vm::new();

    let source: Vec<u8> = "2 + 2".as_bytes().into();
    vm.interpret(source);


    // let mut offset = 0;
    // while offset < chunk.instructions.len() {
    //     offset = chunk.disassemble(offset);
    // }
    

}
