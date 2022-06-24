use std::cell::Cell;
use std::{time::Instant, fmt::Formatter, error::Error};
use std::fmt::Debug;

#[derive(Debug, Copy, Clone, PartialEq)]
enum ValueKind {
    Int,
    Float,
    Bool,
}

#[derive(Debug, Copy, Clone, PartialEq)]
enum Value {
    Int(i64),
    Float(f64),
    True,
    False,
}

impl Value {
    fn kind(&self) -> ValueKind {
        match self {
            Value::Int(_) => ValueKind::Int,
            Value::Float(_) => ValueKind::Float,
            Value::True => ValueKind::Bool,
            Value::False => ValueKind::Bool,
        }
    }

    fn as_int(&self) -> i64 {
        match self {
            Value::Int(i) => *i,
            Value::Float(f) => *f as i64,
            Value::True => 1,
            Value::False => 0,
        }
    }

    fn as_float(&self) -> f64 {
        match self {
            Value::Int(i) => *i as f64,
            Value::Float(f) => *f,
            Value::True => 1.0,
            Value::False => 0.0,
        }
    }

    fn add_float(lhs: Value, rhs: Value) -> Value {
        Value::Float(lhs.as_float() + rhs.as_float())
    }

    fn add_int(lhs: Value, rhs: Value) -> Value {
        Value::Int(lhs.as_int() + rhs.as_int())
    }

    // Intentionally overcomplicated so I could see when inline caching would help.
    fn get_add_method(&self, b: Value) -> Box<dyn Fn(Value, Value) -> Value> {
        match (self, b) {
            (Value::Int(_), Value::Int(_)) => Box::new(Value::add_int),
            (Value::Float(_), Value::Float(_)) => Box::new(Value::add_float),
            (Value::Int(_), Value::Float(_)) => Box::new(Value::add_int),
            (Value::Float(_), Value::Int(_)) => Box::new(Value::add_float),
            _ => panic!("Invalid types for addition"),
        }
    }

}



// Things we could add
// Objects with properties
// Functions
// Add being polymorphic
// Adding an inline cache
// Adding object shapes


#[derive(Copy, Clone)]
struct CacheValue {
    kind: ValueKind,
    value: fn(Value, Value) -> Value
}

impl Debug for CacheValue {
    fn fmt(&self, f: &mut Formatter) -> Result<(), std::fmt::Error> {
        write!(f, "{:?}", self.kind)
    }
}



#[derive(Debug, Clone)]
enum Instruction {
    Push(Value),
    Pop,
    Add,
    AddWithCache(Cell<Option<CacheValue>>),
    Mul,
    Sub,
    Div,
    Ret,
    Eq,
    Jump(usize),
    JumpT(usize),
    JumpF(usize),
}


enum Expr {
    Int(i64),
    Float(f64),
    True,
    False,
    Add(Box<Expr>, Box<Expr>),
    Plus(Box<Expr>, Box<Expr>),
    Mul(Box<Expr>, Box<Expr>),
    Sub(Box<Expr>, Box<Expr>),
    Div(Box<Expr>, Box<Expr>),
    Eq(Box<Expr>, Box<Expr>),
    If(Box<Expr>, Box<Expr>, Box<Expr>),
    Ret,
}

impl Expr {
    fn compile(&self) -> Vec<Instruction> {
        use Instruction::*;
        let mut instructions = Vec::new();
        match self {
            Expr::Int(i) => instructions.push(Push(Value::Int(*i))),
            Expr::Float(f) => instructions.push(Push(Value::Float(*f))),
            Expr::True =>instructions.push(Push(Value::True)),
            Expr::False =>instructions.push(Push(Value::False)),
            Expr::Add(lhs, rhs) => {
                instructions.extend(lhs.compile());
                instructions.extend(rhs.compile());
                instructions.push(Add);
            }
            Expr::Plus(lhs, rhs) => {
                instructions.extend(lhs.compile());
                instructions.extend(rhs.compile());
                instructions.push(AddWithCache(Cell::new(None)));
            }
            Expr::Mul(lhs, rhs) => {
                instructions.extend(lhs.compile());
                instructions.extend(rhs.compile());
                instructions.push(Mul);
            }
            Expr::Sub(lhs, rhs) => {
                instructions.extend(lhs.compile());
                instructions.extend(rhs.compile());
                instructions.push(Sub);
            }
            Expr::Div(lhs, rhs) => {
                instructions.extend(lhs.compile());
                instructions.extend(rhs.compile());
                instructions.push(Div);
            }
            Expr::Ret => instructions.push(Ret),
            Expr::Eq(lhs, rhs) => {
                instructions.extend(lhs.compile());
                instructions.extend(rhs.compile());
                instructions.push(Eq);
            }
            Expr::If(cond, then, else_) => {
                // I need to jump to a specific point in the program
                // Or I could make my jumps relative.
                // For right now I'm going assume these are all the instructions there are
                // If that wasn't true, I'd need to patch up the jump instructions

                let cond = cond.compile();
                let then = then.compile();
                let else_ = else_.compile();
                let else_location = cond.len() + then.len() + 2;
                let exit_location = else_location + else_.len();

                instructions.extend(cond);
                instructions.push(JumpF(else_location));
                instructions.extend(then);
                instructions.push(Jump(exit_location));
                instructions.extend(else_);
            }
        }
        instructions
    }
}



struct Vm {
    pc: usize,
    memory: [Value; 4096],
    stack: Vec<Value>,
    code: Vec<Instruction>,
}

impl Vm {
    fn new() -> Vm {
        Vm {
            pc: 0,
            memory: [Value::Int(0); 4096],
            stack: Vec::new(),
            code: Vec::new(),
        }
    }

    fn reset(&mut self) {
        self.pc = 0;
        self.stack.clear();
    }

    fn get_ret(&self) -> Value {
        *self.stack.last().unwrap()
    }

    fn run(&mut self) {
        let mut should_assign_cache: Option<ValueKind> = None;
        while self.pc < self.code.len() {
            // println!("{:?}, {} {}", self.stack, self.pc, self.code.len());
            let instruction = &self.code[self.pc];
            match instruction {
                Instruction::Push(value) => {
                    self.stack.push(value.clone());
                    self.pc += 1;
                }
                Instruction::Pop => {
                    self.stack.pop();
                    self.pc += 1;
                }
                Instruction::AddWithCache(cache) => {
                    let cache_value = cache.get();
                    let lhs = self.stack.pop().unwrap();
                    if let Some(cache) = cache_value  {
                        if lhs.kind() == cache.kind {
                            let rhs = self.stack.pop().unwrap();
                            self.stack.push((cache.value)(lhs, rhs));
                            self.pc += 1;
                            continue;
                        }
                    }
                    let left_kind = lhs.kind();
                    cache.set(
                        match left_kind {
                            ValueKind::Int => Some(CacheValue {
                                kind: ValueKind::Int,
                                value: Value::add_int,
                            }),
                            ValueKind::Float => Some(CacheValue {
                                kind: ValueKind::Float,
                                value: Value::add_float,
                            }),
                            _ => panic!("Can't add {:?}", left_kind),
                        }
                    );
                    let cache = cache.get().unwrap();
                    let rhs = self.stack.pop().unwrap();
                    self.stack.push((cache.value)(lhs, rhs));
                    self.pc += 1;
                }
                Instruction::Add => {
                    let a = self.stack.pop().unwrap();
                    let b = self.stack.pop().unwrap();

                    let method = a.get_add_method(b);
                    self.stack.push(method(a, b));
                    self.pc += 1;
                }
                Instruction::Mul => {
                    let a = self.stack.pop().unwrap();
                    let b = self.stack.pop().unwrap();
                    self.stack.push(match (a, b) {
                        (Value::Int(a), Value::Int(b)) => Value::Int(a * b),
                        _ => panic!("invalid operands for mul"),
                    });
                    self.pc += 1;
                }
                Instruction::Sub => {
                    let a = self.stack.pop().unwrap();
                    let b = self.stack.pop().unwrap();
                    self.stack.push(match (a, b) {
                        (Value::Int(a), Value::Int(b)) => Value::Int(a - b),
                        _ => panic!("invalid operands for sub"),
                    });
                    self.pc += 1;
                }
                Instruction::Div => {
                    let a = self.stack.pop().unwrap();
                    let b = self.stack.pop().unwrap();
                    self.stack.push(match (a, b) {
                        (Value::Int(a), Value::Int(b)) => Value::Int(a / b),
                        _ => panic!("invalid operands for div"),
                    });
                    self.pc += 1;
                }
                Instruction::Ret => {
                    self.pc += 1;
                }
                Instruction::Jump(position) => {
                    self.pc = *position;
                }
                Instruction::JumpT(position) => {
                    let a = self.stack.pop().unwrap();
                    if a == Value::True {
                        self.pc = *position;
                    } else {
                        self.pc += 1;
                    }
                }
                Instruction::JumpF(position) => {
                    let a = self.stack.pop().unwrap();
                    if a == Value::False {
                        self.pc = *position;
                    } else {
                        self.pc += 1;
                    }
                }
                Instruction::Eq => {
                    let a = self.stack.pop().unwrap();
                    let b = self.stack.pop().unwrap();
                    self.stack.push(match (a, b) {
                        (Value::Int(a), Value::Int(b)) => if a == b { Value::True } else { Value::False },
                        (Value::True, Value::True) => Value::True,
                        (Value::False, Value::False) => Value::True,
                        _ => Value::False,
                    });
                    self.pc += 1;
                }
            }
        }
    }
}



macro_rules! lang {
    (true) => {
        Expr::True
    };
    (false) => {
        Expr::False
    };
    ($int:literal) => {
        Expr::Int($int)
    };
    ((f $float:tt)) => {
        Expr::Float($float)
    };
    ((+ $arg1:tt $arg2:tt)) => {
        Expr::Add(Box::new(lang!($arg1)), Box::new(lang!($arg2)))
    };
    ((++ $arg1:tt $arg2:tt)) => {
        Expr::Plus(Box::new(lang!($arg1)), Box::new(lang!($arg2)))
    };
    ((* $arg1:tt $arg2:tt)) => {
        Expr::Mul(Box::new(lang!($arg1)), Box::new(lang!($arg2)))
    };
    ((- $arg1:tt $arg2:tt)) => {
        Expr::Sub(Box::new(lang!($arg1)), Box::new(lang!($arg2)))
    };
    ((/ $arg1:tt $arg2:tt)) => {
        Expr::Div(Box::new(lang!($arg1)), Box::new(lang!($arg2)))
    };
    (($if:tt $cond:tt $then:tt $else:tt)) => {
        Expr::If(Box::new(lang!($cond)), Box::new(lang!($then)), Box::new(lang!($else)))
    };
    ((= $arg1:tt $arg2:tt)) => {
        Expr::Eq(Box::new(lang!($arg1)), Box::new(lang!($arg2)))
    };
    // How should I handle return?
    (ret) => {
        Expr::Ret
    };
}


fn main() {


    let my_expr = lang!(
        (+ (+ (f 2.0) 2) (+ 3 (f 4.0)))
    );

    let my_expr2 = lang!(
        (++ (++ (f 2.0) 2) (++ 3 (f 4.0)))
    );


    let mut vm = Vm::new();
    vm.code = my_expr.compile();
    println!("{:?}", vm.code);


    let mut result = 0;
    let now = Instant::now();
    for _ in 0..10000 {
       result = 2 + 2 + 3 + 4;
    }
    println!("{:?} {}", now.elapsed(), result);


    let now = Instant::now();
    for _ in 0..10000 {
        vm.run();
        // println!("{:?}", vm.get_ret());
        vm.reset();
    }
    vm.run();
    println!("{:?} {:?}", now.elapsed(), vm.get_ret());

    vm.code = my_expr2.compile();

    let now = Instant::now();
    for _ in 0..10000 {
        vm.run();
        // println!("{:?}", vm.get_ret());
        vm.reset();
    }
    vm.run();
    println!("{:?} {:?}", now.elapsed(), vm.get_ret());

    
}
