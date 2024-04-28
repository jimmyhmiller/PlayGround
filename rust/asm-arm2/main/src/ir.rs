use std::collections::HashMap;

use asm::arm::Register;

use crate::{
    arm::LowLevelArm, common::Label, compiler::Compiler
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
pub enum Value {
    Register(VirtualRegister),
    SignedConstant(isize),
    RawValue(usize),
    // TODO: Think of a better representation
    StringConstantPtr(usize),
    Function(usize),
    Pointer(usize),
    Local(usize),
    FreeVariable(usize),
    True,
    False,
    Null,
}

impl Value {
    fn to_local(&self) -> usize {
        match self {
            Value::Local(local) => *local,
            _ => panic!("Expected local"),
        }
    }
}

// I don't know if this is actually the setup I want
// But I want get some stuff down there
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum BuiltInTypes {
    Int,
    Float,
    String,
    Bool,
    Function,
    Closure,
    Struct,
    Array,
    Null,
}

impl BuiltInTypes {
    pub fn tag(&self, value: isize) -> isize {
        let value = value << 3;
        let tag = self.get_tag();
        value | tag
    }

    // TODO: Given this scheme how do I represent null?
    pub fn get_tag(&self) -> isize {
        match self {
            BuiltInTypes::Int => 0b000,
            BuiltInTypes::Float => 0b001,
            BuiltInTypes::String => 0b010,
            BuiltInTypes::Bool => 0b011,
            BuiltInTypes::Function => 0b100,
            BuiltInTypes::Closure => 0b101,
            BuiltInTypes::Struct => 0b110,
            BuiltInTypes::Array => 0b111,
            BuiltInTypes::Null => 0b111,
        }
    }

    pub fn untag(pointer: usize) -> usize {
        pointer >> 3
    }

    pub fn get_kind(pointer: usize) -> Self {
        if pointer == 0b111 {
            return BuiltInTypes::Null;
        }
        match pointer & 0b111 {
            0b000 => BuiltInTypes::Int,
            0b001 => BuiltInTypes::Float,
            0b010 => BuiltInTypes::String,
            0b011 => BuiltInTypes::Bool,
            0b100 => BuiltInTypes::Function,
            0b101 => BuiltInTypes::Closure,
            0b110 => BuiltInTypes::Struct,
            0b111 => BuiltInTypes::Array,
            _ => panic!("Invalid tag"),
        }
    }

    pub fn is_embedded(&self) -> bool {
        match self {
            BuiltInTypes::Int => true,
            BuiltInTypes::Float => true,
            BuiltInTypes::String => false,
            BuiltInTypes::Bool => true,
            BuiltInTypes::Function => false,
            BuiltInTypes::Struct => false,
            BuiltInTypes::Array => false,
            BuiltInTypes::Closure => false,
            BuiltInTypes::Null => true,
        }
    }

    pub fn construct_int(value: isize) -> isize {
        if value > isize::MAX >> 3 {
            panic!("Integer overflow")
        }
        BuiltInTypes::Int.tag(value)
    }

    pub fn construct_boolean(value: bool) -> isize {
        let bool = BuiltInTypes::Bool;
        if value {
            bool.tag(1)
        } else {
            bool.tag(0)
        }
    }

    pub fn tag_size() -> i32 {
        3
    }
}

#[test]
fn tag_and_untag() {
    let kinds = [
        BuiltInTypes::Int,
        BuiltInTypes::Float,
        BuiltInTypes::String,
        BuiltInTypes::Bool,
        BuiltInTypes::Function,
        BuiltInTypes::Closure,
        BuiltInTypes::Struct,
        BuiltInTypes::Array,
    ];
    for kind in kinds.iter() {
        let tag = kind.get_tag();
        let value = 123;
        let tagged = kind.tag(value);
        // assert_eq!(tagged & 0b111a, tag);
        assert_eq!(kind, &BuiltInTypes::get_kind(tagged as usize));
        assert_eq!(value as usize, BuiltInTypes::untag(tagged as usize));
    }

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
        Value::SignedConstant(val as isize)
    }
}

// Probably don't want to just use rust strings
// could be confusing with lifetimes and such
#[derive(Debug, Clone)]
#[repr(C)]
pub struct StringValue {
    pub str: String,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub enum Instruction {
    Sub(Value, Value, Value),
    Add(Value, Value, Value),
    Mul(Value, Value, Value),
    Div(Value, Value, Value),
    Assign(VirtualRegister, Value),
    Recurse(Value, Vec<Value>),
    TailRecurse(Value, Vec<Value>),
    JumpIf(Label, Condition, Value, Value),
    Jump(Label),
    Ret(Value),
    Breakpoint,
    Compare(Value, Value, Value, Condition),
    Tag(Value, Value, Value),
    // Do I need these?
    LoadTrue(Value),
    LoadFalse(Value),
    LoadConstant(Value, Value),
    Call(Value, Value, Vec<Value>),
    HeapLoad(Value, Value),
    HeapStore(Value, Value),
    LoadLocal(Value, Value),
    StoreLocal(Value, Value),
    RegisterArgument(Value),
    PushStack(Value),
    PopStack(Value),
    LoadFreeVariable(Value, usize),
    GetStackPointer(Value, Value),
    GetStackPointerImm(Value, isize),
    GetTag(Value, Value),
    Untag(Value, Value),
    HeapStoreOffset(Value, Value, usize),
}

impl TryInto<VirtualRegister> for &Value {
    type Error = Value;

    fn try_into(self) -> Result<VirtualRegister, Self::Error> {
        match self {
            Value::Register(register) => Ok(*register),
            _ => Err(*self),
        }
    }
}
impl TryInto<VirtualRegister> for &VirtualRegister {
    type Error = ();

    fn try_into(self) -> Result<VirtualRegister, Self::Error> {
        Ok(*self)
    }
}

impl<T> From<*const T> for Value {
    fn from(val: *const T) -> Self {
        Value::Pointer(val as usize)
    }
}

macro_rules! get_register {
    ($x:expr) => {
        vec![get_registers!($x)].into_iter().flatten().collect()
    };
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
            Instruction::TailRecurse(a, args) => {
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
            Instruction::Compare(a, b, c, _) => {
                get_registers!(a, b, c)
            }
            Instruction::Tag(a, b, c) => {
                get_registers!(a, b, c)
            }
            Instruction::HeapLoad(a, b) => {
                get_registers!(a, b)
            }
            Instruction::HeapStore(a, b) => {
                get_registers!(a, b)
            }
            Instruction::HeapStoreOffset(a, b, _) => {
                get_registers!(a, b)
            }
            Instruction::LoadTrue(a) => {
                get_register!(a)
            }
            Instruction::LoadFalse(a) => {
                get_register!(a)
            }
            Instruction::LoadLocal(a, b) => {
                get_registers!(a, b)
            }
            Instruction::StoreLocal(a, b) => {
                get_registers!(a, b)
            }
            Instruction::RegisterArgument(a) => {
                get_register!(a)
            }
            Instruction::PushStack(a) => {
                get_register!(a)
            }
            Instruction::PopStack(a) => {
                get_register!(a)
            }
            Instruction::LoadFreeVariable(a, _) => {
                get_register!(a)
            }
            Instruction::GetStackPointer(a, b) => {
                get_registers!(a, b)
            }
            Instruction::GetStackPointerImm(a, _) => {
                get_register!(a)
            }
            Instruction::GetTag(a, b) => {
                get_registers!(a, b)
            }
            Instruction::Untag(a, b) => {
                get_registers!(a, b)
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
            // Is it okay that the register is already allocated for the argument?
            if let Some(arg) = register.argument {
                let reg = lang.arg(arg as u8);
                self.allocated_registers.insert(register, reg);
                lang.reserve_register(reg);
                reg
            } else {
                assert!(!self.allocated_registers.contains_key(&register));
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
    pub instructions: Vec<Instruction>,
    labels: Vec<Label>,
    label_names: Vec<String>,
    label_locations: HashMap<usize, usize>,
    num_locals: usize,
}

impl Default for Ir {
    fn default() -> Self {
        Self::new()
    }
}

impl Ir {
    pub fn new() -> Self {
        Self {
            register_index: 0,
            instructions: vec![],
            labels: vec![],
            label_names: vec![],
            label_locations: HashMap::new(),
            num_locals: 0,
        }
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

    pub fn tail_recurse<A>(&mut self, args: Vec<A>) -> Value
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
            .push(Instruction::TailRecurse(register.into(), new_args));
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

    pub fn mul<A, B>(&mut self, a: A, b: B) -> Value
    where
        A: Into<Value>,
        B: Into<Value>,
    {
        let register = self.volatile_register();
        let a = self.assign_new(a.into());
        let b = self.assign_new(b.into());
        self.instructions
            .push(Instruction::Mul(register.into(), a.into(), b.into()));
        Value::Register(register)
    }

    pub fn div<A, B>(&mut self, a: A, b: B) -> Value
    where
        A: Into<Value>,
        B: Into<Value>,
    {
        let register = self.volatile_register();
        let a = self.assign_new(a.into());
        let b = self.assign_new(b.into());
        self.instructions
            .push(Instruction::Div(register.into(), a.into(), b.into()));
        Value::Register(register)
    }

    pub fn compare(&mut self, a: Value, b: Value, condition: Condition) -> Value {
        let register = self.volatile_register();
        let a = self.assign_new(a);
        let b = self.assign_new(b);
        let tag = self.assign_new(Value::RawValue(BuiltInTypes::Bool.get_tag() as usize));
        self.instructions.push(Instruction::Compare(
            register.into(),
            a.into(),
            b.into(),
            condition,
        ));
        self.instructions.push(Instruction::Tag(
            register.into(),
            register.into(),
            tag.into(),
        ));
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

    pub fn assign_new<A>(&mut self, val: A) -> VirtualRegister
    where
        A: Into<Value> 
        {

        let val = val.into();
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
        assert!(!self.label_locations.contains_key(&self.instructions.len()));
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

    pub fn compile(&mut self, name: &str) -> LowLevelArm {

        // println!("{:#?}", self.instructions);
        let mut lang = LowLevelArm::new();
        lang.set_max_locals(self.num_locals);
        // lang.breakpoint();
        
        let before_prelude = lang.new_label("before_prelude");
        lang.write_label(before_prelude);
        // zero is a placeholder because this will be patched
        lang.prelude(0);

        let after_prelude = lang.new_label("after_prelude");
        lang.write_label(after_prelude);

        let exit = lang.new_label("exit");

        let mut ir_label_to_lang_label: HashMap<Label, Label> = HashMap::new();
        let mut labels : Vec<&Label> = self.labels.iter().collect();
        labels.sort_by_key(|label| label.index);
        for label in labels.iter() {
            let new_label = lang.new_label(&self.label_names[label.index]);
            ir_label_to_lang_label.insert(**label, new_label);
        }
        let lifetimes = self.get_register_lifetime();
        // println!("compiling {}", name);
        // Self::draw_lifetimes(&lifetimes);
        let mut alloc = RegisterAllocator::new(lifetimes);
        
        for (index, instruction) in self.instructions.iter().enumerate() {
            let mut lifetimes : Vec<(&VirtualRegister, &(usize, usize))> = alloc.lifetimes.iter().collect();
            lifetimes.sort_by_key(|(_, (start, _))| *start);
            for (register, (_start, end)) in lifetimes {
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
                    // TODO: I need to guard here
                    let a = a.try_into().unwrap();
                    let a = alloc.allocate_register(index, a, &mut lang);
                    let b = b.try_into().unwrap();
                    let b = alloc.allocate_register(index, b, &mut lang);
                    let dest = dest.try_into().unwrap();
                    let dest = alloc.allocate_register(index, dest, &mut lang);
                    // TODO: I should instead use another register. But I can't mutate
                    // things here. The other option is to put this in the Ir.
                    // Which might be a good idea, but I don't love it.
                    lang.shift_right(a, a, BuiltInTypes::tag_size());
                    lang.shift_right(b, b, BuiltInTypes::tag_size());
                    lang.sub(dest, a, b);
                    lang.shift_left(dest, dest, BuiltInTypes::tag_size());
                    lang.shift_left(a, a, BuiltInTypes::tag_size());
                    lang.shift_left(b, b, BuiltInTypes::tag_size());
                }
                Instruction::Add(dest, a, b) => {
                    // TODO: I need to guard here
                    let a = a.try_into().unwrap();
                    let a = alloc.allocate_register(index, a, &mut lang);
                    let b = b.try_into().unwrap();
                    let b = alloc.allocate_register(index, b, &mut lang);
                    let dest = dest.try_into().unwrap();
                    let dest = alloc.allocate_register(index, dest, &mut lang);
                    lang.add(dest, a, b)
                }
                Instruction::Mul(dest, a, b) => {
                    // TODO: I need to guard here
                    let a = a.try_into().unwrap();
                    let a = alloc.allocate_register(index, a, &mut lang);
                    let b = b.try_into().unwrap();
                    let b = alloc.allocate_register(index, b, &mut lang);
                    let dest = dest.try_into().unwrap();
                    let dest = alloc.allocate_register(index, dest, &mut lang);
                    lang.shift_right(a, a, BuiltInTypes::tag_size());
                    lang.shift_right(b, b, BuiltInTypes::tag_size());
                    lang.mul(dest, a, b);
                    lang.shift_left(dest, dest, BuiltInTypes::tag_size());
                    lang.shift_left(a, a, BuiltInTypes::tag_size());
                    lang.shift_left(b, b, BuiltInTypes::tag_size());
                }
                Instruction::Div(dest, a, b) => {
                    // TODO: I need to guard here
                    let a = a.try_into().unwrap();
                    let a = alloc.allocate_register(index, a, &mut lang);
                    let b = b.try_into().unwrap();
                    let b = alloc.allocate_register(index, b, &mut lang);
                    let dest = dest.try_into().unwrap();
                    let dest = alloc.allocate_register(index, dest, &mut lang);
                    lang.shift_right(a, a, BuiltInTypes::tag_size());
                    lang.shift_right(b, b, BuiltInTypes::tag_size());
                    lang.div(dest, a, b);
                    lang.shift_left(dest, dest, BuiltInTypes::tag_size());
                    lang.shift_left(a, a, BuiltInTypes::tag_size());
                    lang.shift_left(b, b, BuiltInTypes::tag_size());
                }
                Instruction::Assign(dest, val) => match val {
                    Value::Register(virt_reg) => {
                        let register = alloc.allocate_register(index, *virt_reg, &mut lang);
                        let dest = alloc.allocate_register(index, *dest, &mut lang);
                        lang.mov_reg(dest, register);
                    }
                    Value::SignedConstant(i) => {
                        let register = alloc.allocate_register(index, *dest, &mut lang);
                        let tagged = BuiltInTypes::construct_int(*i);
                        lang.mov_64(register, tagged);
                    }
                    Value::StringConstantPtr(ptr) => {
                        let register = alloc.allocate_register(index, *dest, &mut lang);
                        let tagged = BuiltInTypes::String.tag(*ptr as isize);
                        lang.mov_64(register, tagged);
                    }
                    Value::Function(id) => {
                        let register = alloc.allocate_register(index, *dest, &mut lang);
                        let function = BuiltInTypes::Function.tag(*id as isize);
                        lang.mov_64(register, function as isize);
                    }
                    Value::Pointer(ptr) => {
                        let register = alloc.allocate_register(index, *dest, &mut lang);
                        lang.mov_64(register, *ptr as isize);
                    }
                    Value::RawValue(value) => {
                        let register = alloc.allocate_register(index, *dest, &mut lang);
                        lang.mov_64(register, *value as isize);
                    }
                    Value::True => {
                        let register = alloc.allocate_register(index, *dest, &mut lang);
                        lang.mov_64(register, BuiltInTypes::construct_boolean(true));
                    }
                    Value::False => {
                        let register = alloc.allocate_register(index, *dest, &mut lang);
                        lang.mov_64(register, BuiltInTypes::construct_boolean(false));
                    }
                    Value::Local(local) => {
                        let register = alloc.allocate_register(index, *dest, &mut lang);
                        lang.load_from_stack(register, *local as i32);
                    },
                    Value::FreeVariable(free_variable) => {
                        let register = alloc.allocate_register(index, *dest, &mut lang);
                        // The idea here is that I would store free variables after the locals on the stack
                        // Need to make sure I preserve that space
                        // and that at this point in the program I know how many locals there are.
                        lang.load_from_stack(register, (*free_variable + self.num_locals) as i32);
                    }
                    Value::Null => {
                        let register = alloc.allocate_register(index, *dest, &mut lang);
                        lang.mov_64(register, 0b111 as isize);
                    }
                },
                Instruction::LoadConstant(dest, val) => {
                    let val = val.try_into().unwrap();
                    let val = alloc.allocate_register(index, val, &mut lang);
                    let dest = dest.try_into().unwrap();
                    let dest = alloc.allocate_register(index, dest, &mut lang);
                    lang.mov_reg(dest, val);
                }
                Instruction::LoadLocal(dest, local) => {
                    let dest = dest.try_into().unwrap();
                    let dest = alloc.allocate_register(index, dest, &mut lang);
                    let local = local.to_local();
                    lang.load_local(dest, local as i32);
                }
                Instruction::StoreLocal(dest, value) => {
                    let value = value.try_into().unwrap();
                    let value = alloc.allocate_register(index, value, &mut lang);
                    lang.store_local(value, dest.to_local() as i32);
                }
                Instruction::LoadTrue(dest) => {
                    let dest = dest.try_into().unwrap();
                    let dest = alloc.allocate_register(index, dest, &mut lang);
                    lang.mov_64(dest, BuiltInTypes::construct_boolean(true));
                }
                Instruction::LoadFalse(dest) => {
                    let dest = dest.try_into().unwrap();
                    let dest = alloc.allocate_register(index, dest, &mut lang);
                    lang.mov_64(dest, BuiltInTypes::construct_boolean(false));
                }
                Instruction::Recurse(dest, args) => {
                    // TODO: Clean up duplication
                    let mut out_live_call_registers = vec![];
                    for (register, (start, end)) in alloc.lifetimes.iter() {
                        if *end < index {
                            continue;
                        }
                        if *start > index {
                            continue;
                        }
                        if index != *end {
                            if let Some(register) = alloc.allocated_registers.get(register) {
                                out_live_call_registers.push(*register);
                            }
                        }
                    }

                    for (index, register) in out_live_call_registers.iter().enumerate() {
                        lang.push_to_stack(*register, index as i32);
                    }
                    for (index, arg) in args.iter().enumerate().rev() {
                        let arg = arg.try_into().unwrap();
                        let arg = alloc.allocate_register(index, arg, &mut lang);
                        lang.mov_reg(lang.arg(index as u8), arg);
                    }
                    lang.recurse(before_prelude);
                    let dest = dest.try_into().unwrap();
                    let register = alloc.allocate_register(index, dest, &mut lang);
                    lang.mov_reg(register, lang.ret_reg());
                    for (index, register) in out_live_call_registers.iter().enumerate() {
                        lang.pop_from_stack(*register, index as i32);
                    }
                }
                Instruction::TailRecurse(dest, args) => {
                    for (index, arg) in args.iter().enumerate().rev() {
                        let arg = arg.try_into().unwrap();
                        let arg = alloc.allocate_register(index, arg, &mut lang);
                        lang.mov_reg(lang.arg(index as u8), arg);
                    }
                    lang.jump(after_prelude);
                    let dest = dest.try_into().unwrap();
                    let register = alloc.allocate_register(index, dest, &mut lang);
                    lang.mov_reg(register, lang.ret_reg());
                }
                Instruction::Call(dest, function, args) => {
                    // TODO: Clean up duplication
                    let mut out_live_call_registers = vec![];
                    for (register, (start, end)) in alloc.lifetimes.iter() {
                        if *end < index {
                            continue;
                        }
                        if *start > index {
                            continue;
                        }
                        if index != *end {
                            if let Some(register) = alloc.allocated_registers.get(register) {
                                out_live_call_registers.push(*register);
                            }
                        }
                    }

                    // I only need to store on stack those things that live past the call
                    // I think this is part of the reason why I have too many registers live at a time
                    for (index, register) in out_live_call_registers.iter().enumerate() {
                        lang.push_to_stack(*register, index as i32);
                    }
                    for (arg_index, arg) in args.iter().enumerate().rev() {
                        let arg = arg.try_into().unwrap();
                        let arg = alloc.allocate_register(index, arg, &mut lang);
                        lang.mov_reg(lang.arg(arg_index as u8), arg);
                    }
                    // TODO:
                    // I am not actually checking any tags here
                    // or unmasking or anything. Just straight up calling it
                    let function =
                        alloc.allocate_register(index, function.try_into().unwrap(), &mut lang);
                    lang.shift_right(function, function, BuiltInTypes::tag_size());
                    lang.call(function);

                    let dest = dest.try_into().unwrap();
                    let register = alloc.allocate_register(index, dest, &mut lang);
                    lang.mov_reg(register, lang.ret_reg());
                    for (index, register) in out_live_call_registers.iter().enumerate() {
                        lang.pop_from_stack(*register, index as i32);
                    }
                }
                Instruction::Compare(dest, a, b, condition) => {
                    let a = a.try_into().unwrap();
                    let a = alloc.allocate_register(index, a, &mut lang);
                    let b = b.try_into().unwrap();
                    let b = alloc.allocate_register(index, b, &mut lang);
                    let dest = dest.try_into().unwrap();
                    let dest = alloc.allocate_register(index, dest, &mut lang);
                    lang.compare_bool(*condition, dest, a, b);
                }
                Instruction::Tag(destination, a, b) => {
                    let a = a.try_into().unwrap();
                    let a = alloc.allocate_register(index, a, &mut lang);
                    let b = b.try_into().unwrap();
                    let b = alloc.allocate_register(index, b, &mut lang);
                    let dest = destination.try_into().unwrap();
                    let dest = alloc.allocate_register(index, dest, &mut lang);
                    lang.tag_value(dest, a, b);
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
                    Value::SignedConstant(i) => {
                        lang.mov_64(lang.ret_reg(), BuiltInTypes::construct_int(*i));
                        lang.jump(exit);
                    }
                    Value::StringConstantPtr(ptr) => {
                        lang.mov_64(lang.ret_reg(), *ptr as isize);
                        lang.jump(exit);
                    }
                    Value::Function(id) => {
                        lang.mov_64(lang.ret_reg(), *id as isize);
                        lang.jump(exit);
                    }
                    Value::Pointer(ptr) => {
                        lang.mov_64(lang.ret_reg(), *ptr as isize);
                        lang.jump(exit);
                    }
                    Value::True => {
                        lang.mov_64(lang.ret_reg(), BuiltInTypes::construct_boolean(true));
                        lang.jump(exit);
                    }
                    Value::False => {
                        lang.mov_64(lang.ret_reg(), BuiltInTypes::construct_boolean(false));
                        lang.jump(exit);
                    }
                    Value::RawValue(_) => {
                        panic!("Should we be returing a raw value?")
                    }
                    Value::Null => {
                        lang.mov_64(lang.ret_reg(), 0b111);
                        lang.jump(exit);
                    }
                    Value::Local(local) => {
                        lang.load_from_stack(lang.ret_reg(), *local as i32);
                        lang.jump(exit);
                    }
                    Value::FreeVariable(free_variable) => {
                        lang.load_from_stack(lang.ret_reg(), (*free_variable + self.num_locals) as i32);
                        lang.jump(exit);
                    }
                },
                Instruction::HeapLoad(dest, ptr) => {
                    let ptr = ptr.try_into().unwrap();
                    let ptr = alloc.allocate_register(index, ptr, &mut lang);
                    let dest = dest.try_into().unwrap();
                    let dest = alloc.allocate_register(index, dest, &mut lang);
                    lang.load_from_heap(dest, ptr , 0);
                }
                Instruction::HeapStore(ptr, val) => {
                    let ptr = ptr.try_into().unwrap();
                    let ptr = alloc.allocate_register(index, ptr, &mut lang);
                    let val = val.try_into().unwrap();
                    let val = alloc.allocate_register(index, val, &mut lang);
                    lang.store_on_heap(ptr, val, 0);
                }
                Instruction::HeapStoreOffset(ptr, val, offset) => {
                    let ptr = ptr.try_into().unwrap();
                    let ptr = alloc.allocate_register(index, ptr, &mut lang);
                    let val = val.try_into().unwrap();
                    let val = alloc.allocate_register(index, val, &mut lang);
                    lang.store_on_heap(ptr, val, *offset as i32);
                }
                Instruction::RegisterArgument(arg) => {
                    // This doesn't actually compile into any code
                    // it is here to say the argument is live from the beginning

                    let arg = arg.try_into().unwrap();
                    alloc.allocate_register(index, arg, &mut lang);
                }
                Instruction::PushStack(val) => {
                    let val = val.try_into().unwrap();
                    let val = alloc.allocate_register(index, val, &mut lang);
                    lang.push_to_stack(val, 0);
                }
                Instruction::PopStack(val) => {
                    let val = val.try_into().unwrap();
                    let val = alloc.allocate_register(index, val, &mut lang);
                    lang.pop_from_stack(val, 0);
                }
                Instruction::LoadFreeVariable(dest, free_variable) => {
                    let dest = dest.try_into().unwrap();
                    let dest = alloc.allocate_register(index, dest, &mut lang);
                    lang.load_from_stack(dest, (*free_variable + self.num_locals) as i32);
                }
                Instruction::GetStackPointer(dest, offset) => {
                    let dest = dest.try_into().unwrap();
                    let dest = alloc.allocate_register(index, dest, &mut lang);
                    let offset = offset.try_into().unwrap();
                    let offset = alloc.allocate_register(index, offset, &mut lang);
                    lang.get_stack_pointer(dest, offset);
                }
                Instruction::GetStackPointerImm(dest, offset) => {
                    let dest = dest.try_into().unwrap();
                    let dest = alloc.allocate_register(index, dest, &mut lang);
                    lang.get_stack_pointer_imm(dest, *offset);
                }
                Instruction::GetTag(dest, value) => {
                    let value = value.try_into().unwrap();
                    let value = alloc.allocate_register(index, value, &mut lang);
                    let dest = dest.try_into().unwrap();
                    let dest = alloc.allocate_register(index, dest, &mut lang);
                    lang.get_tag(dest, value);
                }
                Instruction::Untag(dest, value) => {
                    let value = value.try_into().unwrap();
                    let value = alloc.allocate_register(index, value, &mut lang);
                    let dest = dest.try_into().unwrap();
                    let dest = alloc.allocate_register(index, dest, &mut lang);
                    lang.shift_right(dest, value, BuiltInTypes::tag_size());
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
    pub fn breakpoint(&mut self) {
        self.instructions.push(Instruction::Breakpoint);
    }

    pub fn jump(&mut self, label: Label) {
        self.instructions.push(Instruction::Jump(label));
    }

    pub fn load_string_constant(&mut self, string_constant: Value) -> Value {
        let string_constant = self.assign_new(string_constant);
        let register = self.volatile_register();
        self.instructions.push(Instruction::LoadConstant(
            register.into(),
            string_constant.into(),
        ));
        register.into()
    }

    pub fn heap_store<A, B>(&mut self, dest: A, source: B)
    where
        A: Into<Value>,
        B: Into<Value>,
    {
        let source = self.assign_new(source.into());
        let dest = self.assign_new(dest.into());
        self.instructions
            .push(Instruction::HeapStore(dest.into(), source.into()));
    }

    pub fn heap_store_offset<A, B>(&mut self, dest: A, source: B, offset: usize)
    where
        A: Into<Value>,
        B: Into<Value>,
    {
        let source = self.assign_new(source.into());
        let dest = self.assign_new(dest.into());
        self.instructions
            .push(Instruction::HeapStoreOffset(dest.into(), source.into(), offset));
    }

    pub fn heap_load(&mut self, dest: Value, source: Value) -> Value {
        let source = self.assign_new(source);
        let dest = self.assign_new(dest);
        self.instructions
            .push(Instruction::HeapLoad(dest.into(), source.into()));
        dest.into()
    }

    pub fn function(&mut self, function_index: Value) -> Value {
        let function = self.assign_new(function_index);
        function.into()
    }

    pub fn call(&mut self, function: Value, vec: Vec<Value>) -> Value {
        let dest = self.volatile_register().into();
        self.instructions
            .push(Instruction::Call(dest, function, vec));
        dest
    }

    pub fn store_local(&mut self, local_index: usize, reg: VirtualRegister) {
        self.increment_locals(local_index);
        self.instructions.push(Instruction::StoreLocal(
            Value::Local(local_index),
            reg.into(),
        ));
    }

    pub fn load_local(&mut self, reg: VirtualRegister, local_index: usize) -> Value {
        let register = self.volatile_register();
        self.increment_locals(local_index);
        self.instructions
            .push(Instruction::LoadLocal(register.into(), Value::Local(local_index)));
        register.into()
    }

    pub fn push_to_stack(&mut self, reg: Value) {
        self.instructions.push(Instruction::PushStack(reg));
    }

    pub fn pop_from_stack(&mut self, reg: Value) {
        self.instructions.push(Instruction::PopStack(reg));
    }

    fn increment_locals(&mut self, index: usize) {
        if index >= self.num_locals {
            self.num_locals = index + 1;
        }
    }

    pub fn register_argument(&mut self, reg: VirtualRegister) {
        self.instructions
            .push(Instruction::RegisterArgument(reg.into()));
    }

    pub fn load_free_variable(&mut self, reg: VirtualRegister, index: usize) {
        self.instructions
            .push(Instruction::LoadLocal(reg.into(), Value::FreeVariable(index)));
    }

    pub fn get_stack_pointer(&mut self, offset: Value) -> Value {
        let dest = self.volatile_register().into();
        self.instructions
            .push(Instruction::GetStackPointer(dest, offset));
        dest
    }

    pub fn get_stack_pointer_imm(&mut self, num_free: isize) -> Value {
        let dest = self.volatile_register().into();
        self.instructions
            .push(Instruction::GetStackPointerImm(dest, num_free));
        dest
    }

    pub fn load_from_memory(&mut self, source: Value, offset: i32) -> Value {
        let offset_reg: VirtualRegister = self.volatile_register();
        self.assign(offset_reg, Value::RawValue(offset as usize));
        let source = self.add(source, offset_reg);
        let dest = self.volatile_register();
        self.instructions
            .push(Instruction::HeapLoad(dest.into(), source.into()));
        dest.into()
    }

    pub fn get_tag(&mut self, value: Value) -> Value {
        let dest = self.volatile_register().into();
        self.instructions
            .push(Instruction::GetTag(dest, value.into()));
        dest.into()
    }

    pub fn untag(&mut self, closure_register: Value) -> Value {
        let dest = self.volatile_register().into();
        self.instructions
            .push(Instruction::Untag(dest, closure_register.into()));
        dest.into()
    }

    pub fn tag(&mut self, reg: Value, tag: isize) -> Value {
        let dest = self.volatile_register().into();
        let tag = self.assign_new(Value::RawValue(tag as usize));
        self.instructions
            .push(Instruction::Tag(dest, reg.into(), tag.into()));
        dest.into()
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

#[allow(unused)]
pub fn heap_test() -> Ir {
    let mut ir = Ir::new();
    ir.breakpoint();
    let n = ir.arg(0);
    let location = ir.arg(1);
    ir.heap_store(n, location);
    let temp_reg = ir.volatile_register();
    let result = ir.heap_load(temp_reg.into(), location.into());
    let result_reg = ir.volatile_register();
    ir.assign(result_reg, result);
    ir.ret(result_reg);
    ir
}

// pub fn hello_world() -> Ir {
//     let mut ir = Ir::new();
//     let print = ir.add_function("print", print_value as *const u8);
//     let string_constant = ir.string_constant("Hello World!".to_string());
//     let string_constant = ir.load_string_constant(string_constant);
//     let print = ir.function(print);
//     ir.call(print, vec![string_constant]);
//     ir
// }

pub extern "C" fn println_value(compiler: &Compiler, value: usize) {
    compiler.println(value);
}

pub extern "C" fn print_value(compiler: &Compiler, value: usize) {
    compiler.print(value);
}



// TODO:
// I need to properly tag every value
