use std::{collections::HashMap, mem};

use asm::arm::Register;

use crate::{
    arm::{LowLevelArm, RECURSE_PLACEHOLDER_REGISTER},
    common::Label,
};

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum Condition {
    LessThanOrEqual,
    LessThan,
    Equal,
    NotEqual,
    GreaterThan,
    GreaterThanOrEqual,
}

#[derive(Debug, Copy, Clone)]
#[allow(dead_code)]
pub enum Value {
    Register(VirtualRegister),
    UnSignedConstant(usize),
    SignedConstant(isize),
    // TODO: Think of a better representation
    StringConstantId(usize),
    Function(usize),
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct VirtualRegister {
    argument: Option<usize>,
    index: usize,
    volatile: bool,
}

impl From<VirtualRegister> for Value {
    fn from(val: VirtualRegister) -> Self {
        Value::Register(val)
    }
}

impl From<usize> for Value {
    fn from(val: usize) -> Self {
        Value::UnSignedConstant(val)
    }
}



// Probably don't want to just use rust strings
// could be confusing with lifetimes and such
#[derive(Debug, Clone)]
#[repr(C)]
struct StringValue {
    str: String,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
enum Instruction {
    Sub(Value, Value, Value),
    Add(Value, Value, Value),
    Mul(Value, Value, Value),
    Div(Value, Value, Value),
    Assign(VirtualRegister, Value),
    Recurse(Value, Vec<Value>),
    JumpIf(Label, Condition, Value, Value),
    Jump(Label),
    Ret(Value),
    Breakpoint,
    LoadConstant(Value, Value),
    Call(Value, Value, Vec<Value>),
}

impl TryInto<VirtualRegister> for &Value {
    type Error = ();

    fn try_into(self) -> Result<VirtualRegister, Self::Error> {
        match self {
            Value::Register(register) => Ok(*register),
            _ => Err(()),
        }
    }
}
impl TryInto<VirtualRegister> for &VirtualRegister {
    type Error = ();

    fn try_into(self) -> Result<VirtualRegister, Self::Error> {
        Ok(*self)
    }
}

macro_rules! get_registers {
    ($x:expr) => {
        if let Ok(register) = $x.try_into() {
            Some(register)
        } else {
            None
        }
    };
    ($x:expr, $($xs:expr),+)  => {
        vec![get_registers!($x), $(get_registers!($xs)),+].into_iter().flatten().collect()
    };
}

impl Instruction {
    fn get_registers(&self) -> Vec<VirtualRegister> {
        match self {
            Instruction::Sub(a, b, c) => {
                get_registers!(a, b, c)
            }
            Instruction::Add(a, b, c) => {
                get_registers!(a, b, c)
            }
            Instruction::Mul(a, b, c) => {
                get_registers!(a, b, c)
            }
            Instruction::Div(a, b, c) => {
                get_registers!(a, b, c)
            }
            Instruction::Assign(a, b) => {
                get_registers!(a, b)
            }
            Instruction::Recurse(a, args) => {
                let mut result: Vec<VirtualRegister> =
                    args.iter().filter_map(|arg| get_registers!(arg)).collect();
                if let Ok(register) = a.try_into() {
                    result.push(register);
                }
                result
            }
            Instruction::Call(a, b, args) => {
                let mut result: Vec<VirtualRegister> =
                    args.iter().filter_map(|arg| get_registers!(arg)).collect();
                if let Ok(register) = a.try_into() {
                    result.push(register);
                }
                if let Ok(register) = b.try_into() {
                    result.push(register);
                }
                result
            }
            Instruction::LoadConstant(a, b) => {
                get_registers!(a, b)
            }
            Instruction::JumpIf(_, _, a, b) => {
                get_registers!(a, b)
            }
            Instruction::Ret(a) => {
                if let Ok(register) = a.try_into() {
                    vec![register]
                } else {
                    vec![]
                }
            }
            Instruction::Breakpoint => {
                vec![]
            }
            Instruction::Jump(_) => {
                vec![]
            }
        }
    }
}

struct RegisterAllocator {
    lifetimes: HashMap<VirtualRegister, (usize, usize)>,
    allocated_registers: HashMap<VirtualRegister, Register>,
}

impl RegisterAllocator {
    fn new(lifetimes: HashMap<VirtualRegister, (usize, usize)>) -> Self {
        Self {
            lifetimes,
            allocated_registers: HashMap::new(),
        }
    }

    fn allocate_register(
        &mut self,
        index: usize,
        register: VirtualRegister,
        lang: &mut LowLevelArm,
    ) -> Register {
        let (start, _end) = self.lifetimes.get(&register).unwrap();
        if index == *start {
            if let Some(arg) = register.argument {
                let reg = lang.arg(arg as u8);
                self.allocated_registers.insert(register, reg);
                lang.reserve_register(reg);
                reg
            } else {
                let reg = lang.volatile_register();
                self.allocated_registers.insert(register, reg);
                reg
            }
        } else {
            assert!(self.allocated_registers.contains_key(&register));
            *self.allocated_registers.get(&register).unwrap()
        }
    }
}

#[derive(Debug, Clone)]
pub struct Ir {
    register_index: usize,
    instructions: Vec<Instruction>,
    labels: Vec<Label>,
    label_names: Vec<String>,
    label_locations: HashMap<usize, usize>,
    string_constants: Vec<StringValue>,
    // A bit of a weird way of representing this right?
    function_names: Vec<String>,
    // TODO: usize is defintely not the right type here
    functions: HashMap<usize, usize>,
}

impl Ir {
    pub fn new() -> Self {
        Self {
            register_index: 0,
            instructions: vec![],
            labels: vec![],
            label_names: vec![],
            label_locations: HashMap::new(),
            string_constants: vec![],
            function_names: vec![],
            functions: HashMap::new(),
        }
    }

    pub fn get_function_by_name(&self, name: &str) -> Option<usize> {
        self.function_names.iter().position(|n| n == name)
    }

    fn next_register(&mut self, argument: Option<usize>, volatile: bool) -> VirtualRegister {
        let register = VirtualRegister {
            argument,
            index: self.register_index,
            volatile,
        };
        self.register_index += 1;
        register
    }

    pub fn arg(&mut self, n: usize) -> VirtualRegister {
        self.next_register(Some(n), true)
    }

    pub fn volatile_register(&mut self) -> VirtualRegister {
        self.next_register(None, true)
    }

    pub fn recurse<A>(&mut self, args: Vec<A>) -> Value
    where
        A: Into<Value>,
    {
        let register = self.volatile_register();
        let mut new_args: Vec<Value> = vec![];
        for arg in args.into_iter() {
            let value: Value = arg.into();
            let reg = self.assign_new(value);
            new_args.push(reg.into());
        }
        self.instructions
            .push(Instruction::Recurse(register.into(), new_args));
        Value::Register(register)
    }

    pub fn sub<A, B>(&mut self, a: A, b: B) -> Value
    where
        A: Into<Value>,
        B: Into<Value>,
    {
        let result = self.volatile_register();
        let a = self.assign_new(a.into());
        let b = self.assign_new(b.into());
        self.instructions
            .push(Instruction::Sub(result.into(), a.into(), b.into()));
        Value::Register(result)
    }

    pub fn add<A, B>(&mut self, a: A, b: B) -> Value
    where
        A: Into<Value>,
        B: Into<Value>,
    {
        let register = self.volatile_register();
        let a = self.assign_new(a.into());
        let b = self.assign_new(b.into());
        self.instructions
            .push(Instruction::Add(register.into(), a.into(), b.into()));
        Value::Register(register)
    }

    pub fn jump_if<A, B>(&mut self, label: Label, condition: Condition, a: A, b: B)
    where
        A: Into<Value>,
        B: Into<Value>,
    {
        let a = self.assign_new(a.into());
        let b = self.assign_new(b.into());
        self.instructions
            .push(Instruction::JumpIf(label, condition, a.into(), b.into()));
    }

    pub fn assign<A>(&mut self, dest: VirtualRegister, val: A)
    where
        A: Into<Value>,
    {
        self.instructions
            .push(Instruction::Assign(dest, val.into()));
    }

    fn assign_new(&mut self, val: Value) -> VirtualRegister {
        if let Value::Register(register) = val {
            return register;
        }
        let register = self.next_register(None, false);
        self.instructions.push(Instruction::Assign(register, val));
        register
    }

    pub fn ret<A>(&mut self, n: A) -> Value
    where
        A: Into<Value>,
    {
        let val = n.into();
        self.instructions.push(Instruction::Ret(val));
        val
    }

    pub fn label(&mut self, arg: &str) -> Label {
        let label_index = self.labels.len();
        self.label_names.push(arg.to_string());
        let label = Label { index: label_index };
        self.labels.push(label);
        label
    }

    pub fn write_label(&mut self, early_exit: Label) {
        self.label_locations
            .insert(self.instructions.len(), early_exit.index);
    }

    fn get_register_lifetime(&mut self) -> HashMap<VirtualRegister, (usize, usize)> {
        let mut result: HashMap<VirtualRegister, (usize, usize)> = HashMap::new();
        for (index, instruction) in self.instructions.iter().enumerate().rev() {
            for register in instruction.get_registers() {
                if let Some((_start, end)) = result.get(&register) {
                    result.insert(register, (index, *end));
                } else {
                    result.insert(register, (index, index));
                }
            }
        }

        result
    }

    #[allow(unused)]
    pub fn draw_lifetimes(lifetimes: &HashMap<VirtualRegister, (usize, usize)>) {
        // Find the maximum lifetime to set the width of the diagram
        let max_lifetime = lifetimes.values().map(|(_, end)| end).max().unwrap_or(&0);
        // sort lifetime by start
        let mut lifetimes: Vec<(VirtualRegister, (usize, usize))> =
            lifetimes.clone().into_iter().collect();
        lifetimes.sort_by_key(|(_, (start, _))| *start);

        for (register, (start, end)) in &lifetimes {
            // Print the register name
            print!("{:10} |", register.index);

            // Print the start of the lifetime
            for _ in 0..*start {
                print!(" ");
            }

            // Print the lifetime
            for _ in *start..*end {
                print!("-");
            }

            // Print the rest of the line
            for _ in *end..*max_lifetime {
                print!(" ");
            }

            println!("|");
        }
    }

    pub fn compile(&mut self) -> LowLevelArm {
        let mut lang = LowLevelArm::new();
        // lang.breakpoint();
        // zero is a placeholder because this will be patched
        lang.prelude(0);

        let exit = lang.new_label("exit");

        let mut ir_label_to_lang_label: HashMap<Label, Label> = HashMap::new();

        for label in self.labels.iter() {
            let new_label = lang.new_label(&self.label_names[label.index]);
            ir_label_to_lang_label.insert(*label, new_label);
        }
        let lifetimes = self.get_register_lifetime();
        let mut alloc = RegisterAllocator::new(lifetimes);
        for (index, instruction) in self.instructions.iter().enumerate() {
            for (register, (_start, end)) in alloc.lifetimes.iter() {
                if index == end + 1 {
                    if let Some(register) = alloc.allocated_registers.get(register) {
                        lang.free_register(*register);
                    }
                }
            }
            let label = self.label_locations.get(&index);
            if let Some(label) = label {
                lang.write_label(ir_label_to_lang_label[&self.labels[*label]]);
            }
            match instruction {
                Instruction::Breakpoint => {
                    lang.breakpoint();
                }
                Instruction::Sub(dest, a, b) => {
                    let a = a.try_into().unwrap();
                    let a = alloc.allocate_register(index, a, &mut lang);
                    let b = b.try_into().unwrap();
                    let b = alloc.allocate_register(index, b, &mut lang);
                    let dest = dest.try_into().unwrap();
                    let dest = alloc.allocate_register(index, dest, &mut lang);
                    lang.sub(dest, a, b)
                }
                Instruction::Add(dest, a, b) => {
                    let a = a.try_into().unwrap();
                    let a = alloc.allocate_register(index, a, &mut lang);
                    let b = b.try_into().unwrap();
                    let b = alloc.allocate_register(index, b, &mut lang);
                    let dest = dest.try_into().unwrap();
                    let dest = alloc.allocate_register(index, dest, &mut lang);
                    lang.add(dest, a, b)
                }
                Instruction::Mul(dest, a, b) => {
                    let a = a.try_into().unwrap();
                    let a = alloc.allocate_register(index, a, &mut lang);
                    let b = b.try_into().unwrap();
                    let b = alloc.allocate_register(index, b, &mut lang);
                    let dest = dest.try_into().unwrap();
                    let dest = alloc.allocate_register(index, dest, &mut lang);
                    lang.mul(dest, a, b)
                }
                Instruction::Div(dest, a, b) => {
                    let a = a.try_into().unwrap();
                    let a = alloc.allocate_register(index, a, &mut lang);
                    let b = b.try_into().unwrap();
                    let b = alloc.allocate_register(index, b, &mut lang);
                    let dest = dest.try_into().unwrap();
                    let dest = alloc.allocate_register(index, dest, &mut lang);
                    lang.div(dest, a, b)
                }
                Instruction::Assign(dest, val) => match val {
                    Value::Register(virt_reg) => {
                        let register = alloc.allocate_register(index, *virt_reg, &mut lang);
                        let dest = alloc.allocate_register(index, *dest, &mut lang);
                        lang.mov_reg(dest, register);
                    }
                    Value::UnSignedConstant(i) => {
                        let register = alloc.allocate_register(index, *dest, &mut lang);
                        lang.mov_64(register, *i as isize);
                    }
                    Value::SignedConstant(i) => {
                        let register = alloc.allocate_register(index, *dest, &mut lang);
                        lang.mov_64(register, *i);
                    }
                    Value::StringConstantId(id) => {
                        let register = alloc.allocate_register(index, *dest, &mut lang);
                        let string = self.string_constants.get(*id).unwrap();
                        let ptr = string as *const _ as u64;
                        // tag the pointer as a string with the pattern 010 in the least significant bits
                        let ptr = ptr | 0b010;
                        lang.mov_64(register, ptr as isize);
                    }
                    Value::Function(id) => {
                        let register = alloc.allocate_register(index, *dest, &mut lang);
                        let function = self.functions.get(id).unwrap();
                        lang.mov_64(register, *function as isize);
                    }
                },
                Instruction::LoadConstant(dest, val) => {
                    let val = val.try_into().unwrap();
                    let val = alloc.allocate_register(index, val, &mut lang);
                    let dest = dest.try_into().unwrap();
                    let dest = alloc.allocate_register(index, dest, &mut lang);
                    lang.mov_reg(dest, val);
                }
                Instruction::Recurse(dest, args) => {
                    let allocated_registers = lang.allocated_volatile_registers.clone();
                    for (index, register) in allocated_registers.iter().enumerate() {
                        // TODO: I don't like this hardcoded 2 here
                        // it is because the prelude stores 2 registers on the stack
                        // But we might have locals on the stack as well
                        // we will need to fix that.
                        lang.store_on_stack(*register, index as i32 + 2)
                    }
                    for (index, arg) in args.iter().enumerate() {
                        let arg = arg.try_into().unwrap();
                        let arg = alloc.allocate_register(index, arg, &mut lang);
                        lang.mov_reg(lang.arg(index as u8), arg);
                    }
                    lang.call(RECURSE_PLACEHOLDER_REGISTER);
                    let dest = dest.try_into().unwrap();
                    let register = alloc.allocate_register(index, dest, &mut lang);
                    lang.mov_reg(register, lang.ret_reg());
                    for (index, register) in allocated_registers.iter().enumerate() {
                        lang.load_from_stack(*register, index as i32 + 2)
                    }
                }
                Instruction::Call(dest, function, args) => {
                    let allocated_registers = lang.allocated_volatile_registers.clone();
                    for (index, register) in allocated_registers.iter().enumerate() {
                        // TODO: I don't like this hardcoded 2 here
                        // it is because the prelude stores 2 registers on the stack
                        // But we might have locals on the stack as well
                        // we will need to fix that.
                        lang.store_on_stack(*register, index as i32 + 2)
                    }
                    for (index, arg) in args.iter().enumerate() {
                        let arg = arg.try_into().unwrap();
                        let arg = alloc.allocate_register(index, arg, &mut lang);
                        lang.mov_reg(lang.arg(index as u8), arg);
                    }
                    // TODO:
                    // I am not actually checking any tags here
                    // or unmasking or anything. Just straight up calling it
                    
                    let function = alloc.allocate_register(index, function.try_into().unwrap(), &mut lang);
                    lang.call(function);

                    let dest = dest.try_into().unwrap();
                    let register = alloc.allocate_register(index, dest, &mut lang);
                    lang.mov_reg(register, lang.ret_reg());
                    for (index, register) in allocated_registers.iter().enumerate() {
                        lang.load_from_stack(*register, index as i32 + 2)
                    }

                }
                Instruction::JumpIf(label, condition, a, b) => {
                    let a = a.try_into().unwrap();
                    let a = alloc.allocate_register(index, a, &mut lang);
                    let b = b.try_into().unwrap();
                    let b = alloc.allocate_register(index, b, &mut lang);
                    let label = ir_label_to_lang_label.get(label).unwrap();
                    lang.compare(a, b);
                    match condition {
                        Condition::LessThanOrEqual => lang.jump_less_or_equal(*label),
                        Condition::LessThan => lang.jump_less(*label),
                        Condition::Equal => lang.jump_equal(*label),
                        Condition::NotEqual => lang.jump_not_equal(*label),
                        Condition::GreaterThan => lang.jump_greater(*label),
                        Condition::GreaterThanOrEqual => lang.jump_greater_or_equal(*label),
                    }
                }
                Instruction::Jump(label) => {
                    let label = ir_label_to_lang_label.get(label).unwrap();
                    lang.jump(*label);
                }
                Instruction::Ret(value) => match value {
                    Value::Register(virt_reg) => {
                        let register = alloc.allocate_register(index, *virt_reg, &mut lang);
                        if register == lang.ret_reg() {
                            lang.jump(exit);
                        } else {
                            lang.mov_reg(lang.ret_reg(), register);
                            lang.jump(exit);
                        }
                    }
                    Value::UnSignedConstant(i) => {
                        lang.mov_64(lang.ret_reg(), *i as isize);
                        lang.jump(exit);
                    }
                    Value::SignedConstant(i) => {
                        lang.mov_64(lang.ret_reg(), *i);
                        lang.jump(exit);
                    }
                    Value::StringConstantId(id) => {
                        lang.mov_64(lang.ret_reg(), *id as isize);
                        lang.jump(exit);
                    }
                    Value::Function(id) => {
                        lang.mov_64(lang.ret_reg(), *id as isize);
                        lang.jump(exit);
                    }
                }
            }
        }

        lang.write_label(exit);
        // Zero is a placeholder because this will be patched
        lang.epilogue(0);
        lang.ret();

        lang
    }

    #[allow(dead_code)]
    fn breakpoint(&mut self) {
        self.instructions.push(Instruction::Breakpoint);
    }

    pub fn jump(&mut self, label: Label) {
        self.instructions.push(Instruction::Jump(label));
    }

    pub fn string_constant(&mut self, arg: String) -> Value {
        // TODO: Make this correct
        // We need to forget the string so we never clean it up
        let string_value = StringValue {
            str: arg,
        };
        self.string_constants.push(string_value);
        let index = self.string_constants.len() - 1;
        Value::StringConstantId(index)
    }

    pub fn load_string_constant(&mut self, string_constant: Value) -> Value {
        let string_constant = self.assign_new(string_constant);
        let register = self.volatile_register();
        self.instructions.push(Instruction::LoadConstant(register.into(), string_constant.into()));
        register.into()
    }

    pub fn function(&mut self, function_index: usize) -> Value {
        assert!(self.functions.contains_key(&function_index));
        let function = self.assign_new(Value::Function(function_index));
        function.into()
    }

    pub fn call(&mut self, function: Value, vec: Vec<Value>) -> Value {
        let dest = self.volatile_register().into();
        self.instructions.push(Instruction::Call(dest, function, vec));
        dest
    }

    pub fn add_function(&mut self, name: &str, function: *const u8) -> usize {
        self.function_names.push(name.to_string());
        let index = self.function_names.len() - 1;
        self.functions.insert(index, function as usize);
        index
    }
}



#[allow(unused)]
pub fn fib() -> Ir {
    let mut ir = Ir::new();
    ir.breakpoint();
    let n = ir.arg(0);

    let early_exit = ir.label("early_exit");

    let result_reg = ir.volatile_register();
    ir.jump_if(early_exit, Condition::LessThanOrEqual, n, 1);

    let reg_0: Value = ir.sub(n, 1);
    let reg_1 = ir.recurse(vec![reg_0]);

    let reg_2 = ir.sub(n, 2);
    let reg_3 = ir.recurse(vec![reg_2]);

    let reg_4 = ir.add(reg_1, reg_3);
    ir.assign(result_reg, reg_4);
    let end_else = ir.label("end_else");
    ir.jump(end_else);

    ir.write_label(early_exit);
    ir.assign(result_reg, n);
    ir.write_label(end_else);

    ir.ret(result_reg);

    ir
}


pub fn hello_world() -> Ir {
    let mut ir = Ir::new();
    let print = ir.add_function("print", print_value as *const u8);
    let string_constant = ir.string_constant("Hello World!".to_string());
    let string_constant = ir.load_string_constant(string_constant);
    let print = ir.function(print);
    ir.call(print, vec![string_constant]);
    ir
}

pub fn print_value(value: usize) {
    assert!(value & 0b111 == 0b010);
    let value = value & !0b111;
    let string_value : &StringValue = unsafe { std::mem::transmute(value) };
    let string = &string_value.str;
    println!("{}", string);
}


// TODO:
// I need to properly tag every value