#[macro_use]
extern crate lazy_static;

use std::fs::File;
use std::io::BufReader;
use std::io::Read;
use std::ops;
use std::process::abort;
use std::time::Instant;

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
    Sub,
    LT,
    GT,
    GTE,
    LTE,
    LocalGet(usize),
    LocalSet(usize),
    Call(usize),
    Block(Vec<Instructions>),
    Loop(Vec<Instructions>),
    Br(usize),
    BrIf(usize),
    I32Eq,
    // I actually don't know how to properly implement this.
    // I think I need the concept of a frame, which I do not current have.
    // Maybe my instructions are frames? But if so, I don't know if blocks
    // should be using them or what. I need to look into this more.
    Return,
    End,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
enum Value {
    I32(i32),
    I64(i64),
}
#[derive(Debug, Clone)]
struct Function {
    name: String,
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

impl ops::Sub<Value> for Value {
    type Output = Value;

    fn sub(self, rhs: Value) -> Value {
        match (self, rhs) {
            (Value::I32(x), Value::I32(y)) => Value::I32(x - y),
            (Value::I64(x), Value::I64(y)) => Value::I64(x - y),
            _ => abort(),
        }
    }
}

#[derive(Debug)]
enum Outcome<'a> {
    NextInstruction,
    CallFunction(usize),
    Br(usize),
    Block(&'a Vec<Instructions>),
    Loop(&'a Vec<Instructions>),
    Return,
    End,
}

struct Frame<'a> {
    locals_offset: usize,
    number_of_locals: usize,
    fn_name: &'a str,
}

impl<'a> Frame<'a> {
    fn new(locals_offset: usize, number_of_locals: usize, fn_name: &'a str) -> Frame<'a> {
        Frame {
            locals_offset: locals_offset,
            number_of_locals: number_of_locals,
            fn_name: fn_name,
        }
    }

    fn get_local_position(& self, i : usize) -> usize {
        // Locals go backwards onto value stack.
        // So if there are there locals it is
        // [2 1 0]
        // So in this case if we ask for 0 and our offset is 0
        // 0 - 0 + 2 = 2
        // if we asked for 2 it would be
        // 0 - 2 + 2 = 0
        self.locals_offset - i + (self.number_of_locals - 1)
    }
}

struct MachineState<'a> {
    stack: &'a mut Vec<Value>,
    heap: &'a mut Vec<Value>,
    functions: &'a Vec<Function>,
    locals_stack: &'a mut Vec<Value>,
    frame_stack: &'a mut Vec<Frame<'a>>,
}

// We now need to do a few things.
// Implement if as suguar.
// Implement other operators.
// Try to hook this up to a real web assembly code base.
// Implement some data structures.
// Implement GC.
// Think about how we could make this even faster.
// Consider a JIT.

impl<'a> MachineState<'a> {
    fn new(
        stack: &'a mut Vec<Value>,
        heap: &'a mut Vec<Value>,
        functions: &'a Vec<Function>,
        locals_stack: &'a mut Vec<Value>,
        frame_stack : &'a mut Vec<Frame<'a>>,
    ) -> MachineState<'a> {
        MachineState {
            stack: stack,
            heap: heap,
            functions: functions,
            locals_stack: locals_stack,
            frame_stack: frame_stack,
        }
    }


    // Need to make run return an outcome. Make it single step.
    // Make instructions part of the machine;
    // If the instructions are a pointer can I just change it to point to
    // the things in the blocks or functions?
    // Really not sure how that would work. Will try it out.
    // Need to also make locals be better.

    fn step(&mut self, instruction : &'a Instructions) -> Outcome<'a> {

        // match instruction {
        //     Instructions::Block(_) => {},
        //     Instructions::LocalGet(x) => println!("{:}Arg {:?}", "    ".repeat(indent), local_locals[*x]),
        //     Instructions::GT => println!("{:}{:?} > {:?}", "    ".repeat(indent),self.stack[self.stack.len() - 1], self.stack[self.stack.len() - 2]),
        //     Instructions::Add => println!("{:}{:?} + {:?}","    ".repeat(indent), self.stack[self.stack.len() - 1], self.stack[self.stack.len() - 2]),
        //     Instructions::Call(_) => println!("{:}fib({:?})","    ".repeat(indent), self.stack[self.stack.len() - 1]),
        //     // x => println!("Step: {:} {:?} {:?} {:?}", "    ".repeat(indent), x, self.stack, locals),
        //     _ => {}
        // }

        let frame = self.frame_stack.last().unwrap();

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
                // println!("Add: {:?} {:?}", x, y);
                self.stack.push(x + y);
                Outcome::NextInstruction
            }
            Instructions::Mul => {
                let x = self.stack.pop().unwrap();
                let y = self.stack.pop().unwrap();
                self.stack.push(x * y);
                Outcome::NextInstruction
            }
            Instructions::Sub => {
                let x = self.stack.pop().unwrap();
                let y = self.stack.pop().unwrap();
                // println!("Sub: {:?} {:?}", x, y);
                self.stack.push(x - y);
                Outcome::NextInstruction
            }
            Instructions::GT => {
                let x = self.stack.pop().unwrap();
                let y = self.stack.pop().unwrap();
                // Make these values
                if x > y {
                    self.stack.push(Value::I32(1));
                } else {
                    self.stack.push(Value::I32(0));
                }

                Outcome::NextInstruction
            }
            Instructions::LT => {
                let x = self.stack.pop().unwrap();
                let y = self.stack.pop().unwrap();
                if x < y {
                    self.stack.push(Value::I32(1));
                } else {
                    self.stack.push(Value::I32(0));
                }
                Outcome::NextInstruction
            }
            Instructions::GTE => {
                let x = self.stack.pop().unwrap();
                let y = self.stack.pop().unwrap();
                if x >= y {
                    self.stack.push(Value::I32(1));
                } else {
                    self.stack.push(Value::I32(0));
                }
                Outcome::NextInstruction
            }
            Instructions::LTE => {
                let x = self.stack.pop().unwrap();
                let y = self.stack.pop().unwrap();
                if x <= y {
                    self.stack.push(Value::I32(1));
                } else {
                    self.stack.push(Value::I32(0));
                }
                Outcome::NextInstruction
            }
            Instructions::LocalGet(i) => {
                // println!("Arg: {:?}", local_locals[*i]);

                self.stack.push(self.locals_stack[frame.get_local_position(*i)]);
                Outcome::NextInstruction
            }
            Instructions::LocalSet(i) => {
                self.locals_stack[frame.get_local_position(*i)] = self.stack.pop().unwrap();
                Outcome::NextInstruction
            }
            Instructions::Call(i) => {
                Outcome::CallFunction(*i)
            }
            Instructions::Br(i) => {
                Outcome::Br(*i)
            }
            Instructions::BrIf(i) => {
                let val = self.stack.pop().unwrap();
                // println!("BrIf: {:?}", val);
                if let Value::I32(x) = val {
                    if x == 0 {
                        Outcome::NextInstruction
                    } else {
                        Outcome::Br(*i)
                    }
                } else {
                    Outcome::NextInstruction
                }
            }
            Instructions::Block(code) => {
                Outcome::Block(& code)
            }
            // This might not be the best way to do a loop.
            // Seems a little weird that we would keep popping from the
            // stack and then putting this look back on.
            // Maybe some optimizaiton here.
            Instructions::Loop(code) => {
                Outcome::Loop(& code)
            }
            Instructions::I32Eq => {
                let val1 = self.stack.pop().unwrap();
                let val2 = self.stack.pop().unwrap();

                println!("Eq: {:?} {:?}", val1, val2);
                // should check i32
                if val1 == val2 {
                    self.stack.push(Value::I32(0));
                } else {
                    self.stack.push(Value::I32(1));
                }

                Outcome::NextInstruction
            }
            Instructions::Return => {
                Outcome::Return
            }
            Instructions::End => {
                Outcome::End
            }
        }
    }


    // This is all incredibly slow and incredibly ugly.

    fn run(&mut self, instructions : &'a std::vec::Vec<Instructions>, initial_frame : Frame<'a>) {
        let mut position = 0;
        let mut position_stack : Vec<usize> = Vec::with_capacity(100); // arbitrary
        let mut instructions_stack : Vec<& Vec<Instructions>> = Vec::with_capacity(10); // arbitrary
        let mut current_instructions = instructions;

        self.frame_stack.push(initial_frame);
        while position < current_instructions.len() {

            // Temp
            if self.stack.len() > 0 {
                if let Some(Value::I32(x)) = self.stack.get(self.stack.len() - 1) {
                    if x < &-1 {
                        println!("Negative!");
                        break;
                    }
                }
            }
            // println!("{:?} {:?}", &current_instructions[position], if self.stack.len() > 0 { self.stack[self.stack.len() - 1] } else { Value::I32(123213)});

            match self.step(&current_instructions[position]) {
                Outcome::NextInstruction => {
                    position += 1;
                },
                Outcome::CallFunction(i) => {
                    // Make this better and ideally not allocate.
                    position_stack.push(position + 1);
                    instructions_stack.push(current_instructions);
                    position = 0;
                    current_instructions = &self.functions[i].code;
                    self.frame_stack.push(Frame::new(
                        self.locals_stack.len(),
                        self.functions[i].number_of_params,
                        &self.functions[i].name,
                    ));
                    for _ in 0..self.functions[i].number_of_params {
                        self.locals_stack.push(self.stack.pop().unwrap());
                    }
                }
                Outcome::Br(mut i) => {

                    // Might not do the right thing if in a function call?
                    // Not sure what should happen then, or if that is invalid.


                    // We should always do this once. And then if i > 0 we will do it
                    // i additional times.
                    current_instructions = instructions_stack.pop().unwrap();
                    position = position_stack.pop().unwrap();
                    while i != 0 {
                        current_instructions = instructions_stack.pop().unwrap();
                        position = position_stack.pop().unwrap();
                        i -= 1;
                    }
                }
                Outcome::Block(instructions) => {
                    position_stack.push(position + 1);
                    instructions_stack.push(current_instructions);
                    position = 0;
                    current_instructions = &instructions
                }
                Outcome::Loop(instructions) => {
                    // For a loop we don't want to increment the position counter
                    // That way the loop instruction is the thing executed again.
                    // Loop is used with block so that the block contains the exit path.
                    position_stack.push(position);
                    instructions_stack.push(current_instructions);
                    position = 0;
                    current_instructions = &instructions
                }
                Outcome::Return => {
                    // if { indent > 0 } { indent -= 1; }
                    current_instructions = instructions_stack.pop().unwrap();
                    position = position_stack.pop().unwrap();
                    let f = self.frame_stack.pop().unwrap();
                    for _ in 0..f.number_of_locals {
                        self.locals_stack.pop();
                    }
                }
                Outcome::End => {
                    // if { indent > 0 } { indent -= 1; }
                    current_instructions = instructions_stack.pop().unwrap();
                    position = position_stack.pop().unwrap();
                    // This is a block. No need to get rid of frames
                }
            }
        }
    }


}
// Need to do error handling and yet still be fast

fn main() {
    let mut stack = Vec::with_capacity(100); // arbitrary
    let mut heap = vec![Value::I32(22), Value::I32(42)]; // arbitrary
    let update_position = Function {
        name: "update_position".to_string(),
        number_of_params: 3,
        does_return: true,
        code: vec![
            Instructions::Block(vec![
                Instructions::LocalGet(0),
                Instructions::LocalGet(1),
                Instructions::LocalGet(2),
                Instructions::Mul,
                Instructions::Add,
            ])
        ],
    };
    let loop_around = Function {
        name: "loop_around".to_string(),
        number_of_params: 1,
        does_return: true,
        code: vec![
            Instructions::Block(vec![
                Instructions::Loop(vec![
                    Instructions::LocalGet(0),
                    Instructions::ConstI32(-1),
                    Instructions::Add,
                    Instructions::LocalSet(0),
                    Instructions::LocalGet(0),
                    Instructions::ConstI32(0),
                    Instructions::I32Eq,
                    Instructions::BrIf(1),
                    Instructions::Br(0),
                    Instructions::End,
                ]),
            Instructions::End])
        ]
    };
    let recursive_fibonacci = Function {
        name: "fib".to_string(),
        number_of_params: 1,
        does_return: true, // probably don't need this?
        code: vec![
            Instructions::Block(vec![
                Instructions::Block(vec![
                    Instructions::LocalGet(0),
                    Instructions::ConstI32(2),
                    Instructions::GT,
                    Instructions::BrIf(0),

                    Instructions::ConstI32(1),
                    Instructions::LocalGet(0),
                    Instructions::Sub,
                    Instructions::Call(2),
                    Instructions::ConstI32(2),
                    Instructions::LocalGet(0),
                    Instructions::Sub,
                    Instructions::Call(2),
                    Instructions::Add,
                    Instructions::Br(1),
                    Instructions::End,
                ]),
            Instructions::ConstI32(1),
            Instructions::End,
            ]),
        Instructions::Return,
        ]
    };
    let functions = vec!(update_position, loop_around, recursive_fibonacci); // arbitrary
    let instructions = vec![
        Instructions::ConstI32(0),
        Instructions::ConstI32(30),
        Instructions::Call(2),
        Instructions::Store,
    ];
    let mut locals_stack = Vec::with_capacity(20);
    let mut frame_stack = Vec::with_capacity(20);
    let mut machine = MachineState::new(&mut stack, &mut heap, & functions, &mut locals_stack, &mut frame_stack);
    let frame = Frame::new(0, 0, "Main");
    machine.run(&instructions, frame);
    let val = stack.pop();
    if let Some(Value::I32(addr)) = val {
        if let Some(value) = heap.get(addr as usize) {
            println!("{:?}", value);
        } else {
            println!("no value at {:?} on the heap {:?}", addr, heap);
        }
    } else {
        println!("{:?}", val);
    }
}
