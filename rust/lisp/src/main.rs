#[macro_use]
extern crate lazy_static;

use std::env;
use std::fs::File;
use std::io::BufReader;
use std::io::Read;
use std::ops;
use std::process::abort;
use std::time::Instant;
use std::collections::VecDeque;

#[derive(Debug)]
enum Token<'a> {
    OpenParen,
    CloseParen,
    String(&'a str),
    Integer(&'a str),
    Float(&'a str),
    Atom(&'a str),
}

#[derive(Debug)]
struct Tokenizer<'a> {
    input: &'a str,
    input_bytes: &'a [u8],
    position: usize,
    temp: Vec<u8>,
}

lazy_static! {
    static ref ZERO: u8 = '0' as u8;
    static ref NINE: u8 = '9' as u8;
    static ref SPACE: u8 = ' ' as u8;
    static ref NEW_LINE: u8 = '\n' as u8;
    static ref COMMA: u8 = ',' as u8;
    static ref DOUBLE_QUOTE: u8 = '"' as u8;
    static ref OPEN_PAREN: u8 = '(' as u8;
    static ref CLOSE_PAREN: u8 = ')' as u8;
    static ref PERIOD: u8 = '.' as u8;
}

impl<'a> Tokenizer<'a> {
    fn new(input: &str) -> Tokenizer {
        Tokenizer {
            input: input,
            input_bytes: input.as_bytes(),
            position: 0,
            // This is so we only have to allocate once
            // Seems to make things faster
            temp: Vec::with_capacity(10),
        }
    }

    fn consume(&mut self) -> () {
        self.position += 1;
    }

    fn current_byte(&self) -> u8 {
        self.input_bytes[self.position]
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

    fn parse_number(&mut self) -> Token<'a> {
        let mut is_float = false;
        let start = self.position;
        while self.is_valid_number_char() || self.current_byte() == *PERIOD {
            // Need to handle making sure there is only one "."
            if self.current_byte() == *PERIOD {
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
        while !self.is_space() && !self.is_open_paren() && !self.is_close_paren() {
            self.consume()
        }
        Token::Atom(&self.input[start..self.position])
    }

    fn parse_single(&mut self) -> Token<'a> {
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

    fn read(&mut self) -> Vec<Token<'a>> {
        let mut tokens = Vec::with_capacity(self.input.len());
        while !self.at_end() {
            tokens.push(self.parse_single());
        }
        tokens
    }
}

fn tokenize<'a>(text: &'a str) -> Vec<Token<'a>> {
    Tokenizer::new(text).read()
}

#[derive(Debug)]
enum Expr<'a> {
    SExpr(Vec<Expr<'a>>),
    Atom(&'a str),
    Bool(bool),
    String(&'a str),
    Integer(i64),
    Float(f64),
}

fn read(tokens: Vec<Token>) -> Expr {
    // Is there a faster way to do this?
    // Need to probably refer to slices of things
    // Like I ended up doing above. But not 100% sure how to do that
    // given the SExpr structure
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

fn s_expr_len(x: Expr) -> usize {
    if let Expr::SExpr(x) = x {
        x.len()
    } else {
        0
    }
}

#[allow(dead_code)]
fn parse_file(filename: String) -> () {
    let file = File::open(filename).unwrap();
    let mut expr = String::new();
    let mut buf_reader = BufReader::new(file);
    buf_reader.read_to_string(&mut expr).unwrap();

    let start = Instant::now();
    let read_expr = read(tokenize(&expr));
    let duration = start.elapsed();
    println!("{:?}", s_expr_len(read_expr));
    println!("{:?}", duration);
}

#[derive(Debug, Clone)]
enum Instructions {
    ConstI32(i32),
    ConstI64(i64),
    Load,
    Store,
    Add,
    Mul,
    LocalGet(usize),
    LocalSet(usize),
    Call(usize),
    Block(Vec<Instructions>),
    Br(usize),
    BrIf(usize),
}

#[derive(Debug, Copy, Clone)]
enum Value {
    I32(i32),
    I64(i64),
}
#[derive(Debug, Clone)]
struct Function {
    number_of_params: usize,
    does_return: bool,
    code: Vec<Instructions>,
}


impl ops::Add<Value> for Value {
    type Output = Value;

    fn add(self, rhs: Value) -> Value {
        match (self, rhs) {
            (Value::I32(x), Value::I32(y)) => Value::I32(x + y),
            (Value::I64(x), Value::I64(y)) => Value::I64(x + y),
            _ => abort(),
        }
    }
}

impl ops::Mul<Value> for Value {
    type Output = Value;

    fn mul(self, rhs: Value) -> Value {
        match (self, rhs) {
            (Value::I32(x), Value::I32(y)) => Value::I32(x * y),
            (Value::I64(x), Value::I64(y)) => Value::I64(x * y),
            _ => abort(),
        }
    }
}

enum Outcome {
    NextInstruction,
    CallFunction(usize),
    Br(usize),
}

struct MachineState<'a> {
    stack: &'a mut Vec<Value>,
    heap: &'a mut Vec<Value>,
    functions: &'a Vec<Function>,
    label_stack: &'a mut Vec<Value>,
}

impl<'a> MachineState<'a> {
    fn new(
        stack: &'a mut Vec<Value>,
        heap: &'a mut Vec<Value>,
        functions: &'a Vec<Function>,
        label_stack: &'a mut Vec<Value>,
    ) -> MachineState<'a> {
        MachineState {
            stack: stack,
            heap: heap,
            functions: functions,
            label_stack: label_stack,
        }
    }

    fn get_function_params(& self, i: usize) -> usize {
        self.functions[i].number_of_params
    }

    fn get_function_code(& self, i: usize) -> & Vec<Instructions> {
        & self.functions[i].code
    }

    fn get_function_does_return(& self, i: usize) -> bool {
        self.functions[i].does_return
    }

    // Need to make run return an outcome. Make it single step.
    // Make instructions part of the machine;
    // If the instructions are a pointer can I just change it to point to
    // the things in the blocks or functions?
    // Really not sure how that would work. Will try it out.
    // Need to also make locals be better.

    fn step(&mut self, instruction : & Instructions, locals: &mut Vec<Value>, depth: usize) -> Outcome {
        match instruction {
            Instructions::ConstI32(i) => {
                self.stack.push(Value::I32(*i));
                Outcome::NextInstruction
            },
            Instructions::ConstI64(i) => {
                self.stack.push(Value::I64(*i));
                Outcome::NextInstruction
            },
            Instructions::Load => {
                if let Value::I32(addr) = self.stack.pop().unwrap() {
                    self.stack.push(*self.heap.get(addr as usize).unwrap())
                }
                Outcome::NextInstruction
            }
            Instructions::Store => {
                let val = self.stack.pop().unwrap();
                let addr_value = self.stack.pop().unwrap();
                if let Value::I32(addr) = addr_value {
                    self.heap[addr as usize] = val;
                    self.stack.push(addr_value);
                }
                Outcome::NextInstruction
            }
            Instructions::Add => {
                let x = self.stack.pop().unwrap();
                let y = self.stack.pop().unwrap();
                self.stack.push(x + y);
                Outcome::NextInstruction
            }
            Instructions::Mul => {
                let x = self.stack.pop().unwrap();
                let y = self.stack.pop().unwrap();
                self.stack.push(x * y);
                Outcome::NextInstruction
            }
            Instructions::LocalGet(i) => {
                self.stack.push(locals[*i]);
                Outcome::NextInstruction
            }
            Instructions::LocalSet(i) => {
                locals[*i] = self.stack.pop().unwrap();
                Outcome::NextInstruction
            }
            Instructions::Call(i) => {
                Outcome::CallFunction(*i)
            }
            Instructions::Br(i) => {
                Outcome::Br(*i)
            }
            Instructions::BrIf(i) => {
                Outcome::NextInstruction
            }
            Instructions::Block(code) => {
                Outcome::NextInstruction
            }
        }
    }

    fn run(&mut self, instructions : & Vec<Instructions>, locals: &mut Vec<Value>, depth: usize) {
        let mut position = 0;
        let mut position_stack : Vec<usize> = Vec::with_capacity(100); // arbitrary
        let mut instructions_stack : Vec<& Vec<Instructions>> = Vec::with_capacity(10); // arbitrary
        let mut current_instructions = instructions;
        let mut i = 0;
        while position <= current_instructions.len() {

            if position == current_instructions.len() && !instructions_stack.is_empty() {
                current_instructions = instructions_stack.pop().unwrap();
                position = position_stack.pop().unwrap();
            } else if position == current_instructions.len() {
                break;
            }
            println!("{:?} {:?} {:?}", &current_instructions[position], position,current_instructions.len());

            match self.step(&current_instructions[position], locals, depth) {
                Outcome::NextInstruction => {
                    position += 1;
                },
                Outcome::CallFunction(i) => {
                    position_stack.push(position + 1);
                    instructions_stack.push(current_instructions);
                    position = 0;
                    current_instructions = &self.functions[i].code;
                    let function_params_number = self.get_function_params(i);
                    let mut args = Vec::with_capacity(function_params_number);
                    for _ in 0..function_params_number {
                        args.push(self.stack.pop().unwrap())
                    }
                    args.reverse();
                    *locals = args
                }
                Outcome::Br(mut i) => {
                    while i != 0 {
                        current_instructions = instructions_stack.pop().unwrap();
                        position_stack.pop();
                        i -= 1;
                    }
                }
            }
        }
    }


}
// Need to do error handling and yet still be fast

fn main() {
    let mut stack = Vec::with_capacity(100); // arbitrary
    let mut label_stack = Vec::with_capacity(100); // arbitrary
    let mut heap = vec![Value::I32(22), Value::I32(42)]; // arbitrary
    let update_position = Function {
        number_of_params: 3,
        does_return: true,
        code: vec![
            Instructions::LocalGet(0),
            Instructions::LocalGet(1),
            Instructions::LocalGet(2),
            Instructions::Mul,
            Instructions::Add,
        ],
    };
    let functions = vec!(update_position); // arbitrary
    let instructions = vec![
        Instructions::ConstI32(0),
        Instructions::ConstI32(0),
        Instructions::Load,
        Instructions::ConstI32(1),
        Instructions::Load,
        Instructions::ConstI32(2),
        Instructions::Call(0),
        Instructions::Store,
    ];
    let mut machine = MachineState::new(&mut stack, &mut heap, & functions, &mut label_stack);
    let mut locals = vec!();
    machine.run(&instructions, &mut locals, 0);
    let val = stack.pop();
    if let Some(Value::I32(addr)) = val {
        println!("{:?}", heap[addr as usize]);
    } else {
        println!("{:?}", val);
    }
}
