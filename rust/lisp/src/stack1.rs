

#[derive(Debug, Clone)]
enum Instructions {
    ConstI32(i32),
    ConstI64(i64),
    Load,
    Store,
    Add,
    AddConst(Value),
    Mul,
    Sub,
    SubConst(Value),
    LT,
    LTConst(Value),
    GT,
    GTConst(Value),
    GTE,
    LTE,
    LocalGet(usize),
    LocalSet(usize),
    Call(usize),
    Block(Vec<Instructions>),
    Loop(Vec<Instructions>),
    Br(usize),
    BrIf(usize),
    If(Vec<Instructions>, Vec<Instructions>),
    I32Eq,
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
enum Outcome {
    NextInstruction,
    Noop,
    Br(usize),
}

#[derive(Debug)]
struct Frame<'a> {
    locals_offset: usize,
    number_of_locals: usize,
    fn_name: &'a str,
    locals_position: usize,
}

impl<'a> Frame<'a> {
    fn new(locals_offset: usize, number_of_locals: usize, fn_name: &'a str) -> Frame<'a> {
        Frame {
            locals_offset: locals_offset,
            number_of_locals: number_of_locals,
            fn_name: fn_name,
            locals_position: (if number_of_locals == 0 { 0 } else { number_of_locals - 1}) + locals_offset

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


        self.locals_position - i
    }
}

struct MachineState<'a> {
    stack: &'a mut Vec<Value>,
    heap: &'a mut Vec<Value>,
    functions: &'a Vec<Function>,
    compiled_functions: &'a Vec<Box<dyn Execute>>,
    locals_stack: &'a mut Vec<Value>,
    frame_stack: &'a mut Vec<Frame<'a>>,
    position: usize,
    position_stack: Vec<usize>,
    instructions_stack: Vec<&'a Vec<Instructions>>,
    current_instructions: &'a Vec<Instructions>,
}

const TRUE : i32 = 1;
const FALSE : i32 = 0;

const TRUE_V : Value = Value::I32(1);
const FALSE_V : Value =  Value::I32(0);

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
        compiled_functions: &'a Vec<Box<dyn Execute>>,
        locals_stack: &'a mut Vec<Value>,
        frame_stack : &'a mut Vec<Frame<'a>>,
        instructions : &'a Vec<Instructions>,
    ) -> MachineState<'a> {
        MachineState {
            stack: stack,
            heap: heap,
            functions: functions,
            compiled_functions: compiled_functions,
            locals_stack: locals_stack,
            frame_stack: { frame_stack.push(Frame::new(0, 0, "Main")); frame_stack },
            position: 0,
            position_stack: Vec::with_capacity(100), // arbitrary
            instructions_stack: Vec::with_capacity(10),
            current_instructions: instructions, // arbitrary
        }
    }


    // Need to make run return an outcome. Make it single step.
    // Make instructions part of the machine;
    // If the instructions are a pointer can I just change it to point to
    // the things in the blocks or functions?
    // Really not sure how that would work. Will try it out.
    // Need to also make locals be better.

    fn step(&mut self, instruction : &'a Instructions) -> Outcome {

        // match instruction {
        //     Instructions::Block(_) => {},
        //     Instructions::LocalGet(x) => println!("{:}Arg {:?}", "    ".repeat(indent), local_locals[*x]),
        //     Instructions::GT => println!("{:}{:?} > {:?}", "    ".repeat(indent),self.stack[self.stack.len() - 1], self.stack[self.stack.len() - 2]),
        //     Instructions::Add => println!("{:}{:?} + {:?}","    ".repeat(indent), self.stack[self.stack.len() - 1], self.stack[self.stack.len() - 2]),
        //     Instructions::Call(_) => println!("{:}fib({:?})","    ".repeat(indent), self.stack[self.stack.len() - 1]),
        //     // x => println!("Step: {:} {:?} {:?} {:?}", "    ".repeat(indent), x, self.stack, locals),
        //     _ => {}
        // }


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
            Instructions::AddConst(i) => {
                // println!("Add: {:?} {:?}", x, y);
                if let Some(val) = self.stack.last_mut() {
                    *val = *val + *i;
                }

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
            Instructions::SubConst(i) => {
                // println!("Add: {:?} {:?}", x, y);
                if let Some(val) = self.stack.last_mut() {
                    *val = *val - *i;
                }

                Outcome::NextInstruction
            }
            Instructions::GT => {
                let x = self.stack.pop().unwrap();
                let y = self.stack.pop().unwrap();
                // Make these values
                if x > y {
                    self.stack.push(TRUE_V);
                } else {
                    self.stack.push(FALSE_V);
                }

                Outcome::NextInstruction
            }
            Instructions::GTConst(i) => {

                if let Some(val) = self.stack.last_mut() {
                    if *val > *i {
                        *val = TRUE_V
                    } else {
                        *val = FALSE_V
                    }
                }
                Outcome::NextInstruction
            }
            Instructions::LT => {
                let x = self.stack.pop().unwrap();
                let y = self.stack.pop().unwrap();
                if x < y {
                    self.stack.push(TRUE_V);
                } else {
                    self.stack.push(FALSE_V);
                }
                Outcome::NextInstruction
            }
            Instructions::LTConst(i) => {

                if let Some(val) = self.stack.last_mut() {
                    if *val < *i {
                        *val = TRUE_V
                    } else {
                        *val = FALSE_V
                    }
                }
                Outcome::NextInstruction
            }
            Instructions::GTE => {
                let x = self.stack.pop().unwrap();
                let y = self.stack.pop().unwrap();
                if x >= y {
                    self.stack.push(TRUE_V);
                } else {
                    self.stack.push(FALSE_V);
                }
                Outcome::NextInstruction
            }
            Instructions::LTE => {
                let x = self.stack.pop().unwrap();
                let y = self.stack.pop().unwrap();
                if x <= y {
                    self.stack.push(TRUE_V);
                } else {
                    self.stack.push(FALSE_V);
                }
                Outcome::NextInstruction
            }
            Instructions::I32Eq => {
                let val1 = self.stack.pop().unwrap();
                let val2 = self.stack.pop().unwrap();

                // println!("Eq: {:?} {:?}", val1, val2);
                // should check i32
                if val1 == val2 {
                    self.stack.push(TRUE_V);
                } else {
                    self.stack.push(FALSE_V);
                }

                Outcome::NextInstruction
            }
            Instructions::LocalGet(i) => {
                // println!("Arg: {:?}", local_locals[*i]);
                let frame = self.frame_stack.last().unwrap();
                self.stack.push(self.locals_stack[frame.get_local_position(*i)]);
                Outcome::NextInstruction
            }
            Instructions::LocalSet(i) => {
                let frame = self.frame_stack.last().unwrap();
                self.locals_stack[frame.get_local_position(*i)] = self.stack.pop().unwrap();
                Outcome::NextInstruction
            }
            Instructions::Call(i) => {
                // Make this better and ideally not allocate.
                let f = &self.functions[*i];
                let num_params = f.number_of_params;
                self.position_stack.push(self.position + 1);
                self.instructions_stack.push(self.current_instructions);
                self.position = 0;
                self.current_instructions = &f.code;
                self.frame_stack.push(Frame::new(
                    self.locals_stack.len(),
                    num_params,
                    &self.functions[*i].name,
                ));
                for _ in 0..num_params {
                    self.locals_stack.push(self.stack.pop().unwrap())
                }
                Outcome::Noop
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
            Instructions::If(true_case, false_case) => {
                let val = self.stack.pop().unwrap();
                // println!("BrIf: {:?}", val);
                self.position_stack.push(self.position + 1);
                self.instructions_stack.push(self.current_instructions);
                self.position = 0;
                if let Value::I32(_) = val {
                    if val == TRUE_V {
                        self.current_instructions = &true_case;
                    } else {
                        self.current_instructions = &false_case;
                    }
                } else {
                    println!("Handle something other than i32 {:?}", val);
                }
                Outcome::Noop
            }
            Instructions::Block(instructions) => {
                self.position_stack.push(self.position + 1);
                self.instructions_stack.push(self.current_instructions);
                self.position = 0;
                self.current_instructions = &instructions;
                Outcome::Noop
            }
            // This might not be the best way to do a loop.
            // Seems a little weird that we would keep popping from the
            // stack and then putting this look back on.
            // Maybe some optimizaiton here.
            Instructions::Loop(instructions) => {
                // For a loop we don't want to increment the position counter
                // That way the loop instruction is the thing executed again.
                // Loop is used with block so that the block contains the exit path.
                self.position_stack.push(self.position);
                self.instructions_stack.push(self.current_instructions);
                self.position = 0;
                self.current_instructions = &instructions;
                Outcome::Noop
            }
            Instructions::Return => {
                // if { indent > 0 } { indent -= 1; }
                self.current_instructions = self.instructions_stack.pop().unwrap();
                self.position = self.position_stack.pop().unwrap();
                let f = self.frame_stack.pop().unwrap();
                for _ in 0..f.number_of_locals {
                    self.locals_stack.pop();
                }
                Outcome::Noop
            }
            Instructions::End => {
                // if { indent > 0 } { indent -= 1; }
                self.current_instructions = self.instructions_stack.pop().unwrap();
                self.position = self.position_stack.pop().unwrap();
                // This is a block. No need to get rid of frames
                Outcome::Noop
            }
        }
    }


    // This is all incredibly slow and incredibly ugly.

    fn run(&mut self) {
        while self.position < self.current_instructions.len() {

            // println!("{:?} {:?}", &current_instructions[position], if self.stack.len() > 0 { self.stack[self.stack.len() - 1] } else { Value::I32(123213)});

            match self.step(&self.current_instructions[self.position]) {
                Outcome::Noop => {
                    continue;
                },
                Outcome::NextInstruction => {
                    self.position += 1;
                },
                Outcome::Br(i) => {

                    // Might not do the right thing if in a function call?
                    // Not sure what should happen then, or if that is invalid.


                    // We do this i times and then one more. Meaning if i is 0 we do it once.
                    for _ in 0..i {
                        self.instructions_stack.pop();
                        self.position_stack.pop();
                    }
                    self.current_instructions = self.instructions_stack.pop().unwrap();
                    self.position = self.position_stack.pop().unwrap();
                }
            }
        }
    }


}
// Need to do error handling and yet still be fast

struct Environment<'a> {
    compiled_functions: &'a Vec<Box<dyn Execute>>,
    locals_stack: &'a mut Vec<i32>,
}



// Need to do this but without the stack at all.
trait Execute {
    fn execute(& self, frame : & Frame, env : &mut Environment) -> i32;
}

struct ConstI32 {
    i: i32
}

impl Execute for ConstI32 {
    fn execute(& self, frame : & Frame, env : &mut Environment) -> i32 {
        self.i
    }
}

struct If {
    val : Box<dyn Execute>,
    true_case: Box<dyn Execute>,
    false_case: Box<dyn Execute>,
}

impl Execute for If {
    fn execute(& self, frame : & Frame, env : &mut Environment) -> i32 {
        let val = self.val.execute(frame, env);
        if val == TRUE {
            self.true_case.execute(frame, env)
        } else {
            self.false_case.execute(frame, env)
        }
    }
}

struct LocalGet {
    i: usize
}

impl Execute for LocalGet {
    fn execute(& self, frame : & Frame, env : &mut Environment) -> i32 {
        // println!("{:?} {:?} {:?} {:?}", env.locals_stack, frame, self.i, frame.get_local_position(self.i));
        env.locals_stack[frame.get_local_position(self.i)]
    }
}


struct LTConst {
    i: i32,
    val: Box<dyn Execute>,
}

impl Execute for LTConst {
    fn execute(& self, frame : & Frame, env : &mut Environment) -> i32 {
        if self.val.execute(frame, env) < self.i {
             TRUE
        } else {
             FALSE
        }
    }
}

struct SubConst {
    i: i32,
    val: Box<dyn Execute>,
}

impl Execute for SubConst {
    fn execute(& self, frame : & Frame, env : &mut Environment) -> i32 {
        self.val.execute(frame, env) - self.i
    }
}


struct Add {
    left: Box<dyn Execute>,
    right: Box<dyn Execute>,
}

impl Execute for Add {
    fn execute(& self, frame : & Frame, env : &mut Environment) -> i32 {
        let x = self.left.execute(frame, env);
        let y = self.right.execute(frame, env);
        // println!("Add: {:?} {:?}", x, y);
        x + y
    }
}

struct Call {
    i: usize,
    args: Box<Vec<Box<dyn Execute>>>,
}

impl Call {
    fn new(i: usize, args:  Box<Vec<Box<dyn Execute>>>) -> Call {
        Call{
            i: i,
            args: args,
        }
    }
}

impl Execute for Call {
    fn execute(& self, frame : & Frame, env : &mut Environment) -> i32 {

        let f = &env.compiled_functions[self.i];

        for arg in self.args.iter() {
            let value = arg.execute(frame, env);
            env.locals_stack.push(value);
        }


        f.execute(&Frame::new(
            env.locals_stack.len() - self.args.len(),
            1, //temp
            "name",
        ), env)
    }
}


impl Execute for Vec<Box<dyn Execute>> {
    fn execute(& self, frame : & Frame, env : &mut Environment) -> i32 {
        // placeholder
        let mut value = 123123;

        for step in self {
            value = step.execute(frame, env);
        }
        value
    }
}





fn main() {

    let update_position = Function {
        name: "update_position".to_string(),
        number_of_params: 1,
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
            Instructions::LocalGet(0),
            Instructions::LTConst(Value::I32(2)),
            Instructions::If(
            vec![
                Instructions::ConstI32(1),
                Instructions::End,
            ], vec![
                Instructions::LocalGet(0),
                Instructions::SubConst(Value::I32(1)),
                Instructions::Call(2),
                Instructions::LocalGet(0),
                Instructions::SubConst(Value::I32(2)),
                Instructions::Call(2),
                Instructions::Add,
                Instructions::End,
            ]),
            Instructions::Return,
        ]
    };
    let functions = vec!(update_position, loop_around, recursive_fibonacci);
    let mut locals_stack = Vec::with_capacity(20);
    let mut frame_stack = Vec::with_capacity(20);
    let mut stack = Vec::with_capacity(100); // arbitrary
    let mut heap = vec![Value::I32(22), Value::I32(42)]; // arbitrary

    let instructions = vec![
        Instructions::ConstI32(0),
        Instructions::ConstI32(40),
        Instructions::Call(2),
        Instructions::Store,
    ];

    let true_case : Box<Vec<Box<dyn Execute>>> = Box::new(vec![Box::new(ConstI32{i: 1})]);
    let false_args_l : Box<Vec<Box<dyn Execute>>> = Box::new(vec![Box::new(SubConst{i: 1, val: Box::new(LocalGet{ i: 0})})]);
    let false_args_r : Box<Vec<Box<dyn Execute>>> = Box::new(vec![Box::new(SubConst{i: 2, val: Box::new(LocalGet{ i: 0})})]);
    let false_case: Box<Vec<Box<dyn Execute>>> = Box::new(vec![
        Box::new(Add{
            left: Box::new(Call::new(0, false_args_l)),
            right: Box::new(Call::new(0, false_args_r)),
        }),
    ]);
    let comp_do_something : Box<Vec<Box<dyn Execute>>> = Box::new(
        vec![
            Box::new(If {
                val: Box::new(LTConst{i : 2, val: Box::new(LocalGet{ i: 0})}),
                true_case: true_case as Box<dyn Execute>,
                false_case: false_case as Box<dyn Execute>,
            })
    ]);

    let mut compiled_functions = vec!(comp_do_something as Box<dyn Execute>); // arbitrary

    stack.push(Value::I32(40));
    let machine = MachineState::new(&mut stack, &mut heap, & functions, & compiled_functions, &mut locals_stack, &mut frame_stack, &instructions);

    let args : Box<Vec<Box<dyn Execute>>> = Box::new(vec![Box::new(ConstI32{i: 40})]);
    let do_machine = Box::new(Call::new(0, args));
    let mut locals = vec![];
    let mut env = Environment { compiled_functions: &mut compiled_functions, locals_stack: &mut locals};
    let main_frame = Frame::new(0, 0, "Main");
    println!("{:?}", do_machine.execute(&main_frame, &mut env));

    // machine.run();
    // let val = stack.pop();
    // if let Some(Value::I32(addr)) = val {
    //     if let Some(value) = heap.get(addr as usize) {
    //         println!("{:?}", value);
    //     } else {
    //         println!("no value at {:?} on the heap {:?}", addr, heap);
    //     }
    // } else {
    //     println!("{:?}", val);
    // }
}
