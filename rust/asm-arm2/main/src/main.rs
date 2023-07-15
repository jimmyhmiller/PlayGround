use std::{collections::HashMap, error::Error, mem, time::Instant};

use asm::arm::{
    Asm, LdpGenSelector, LdrImmGenSelector, Register, Size, StpGenSelector, StrImmGenSelector, SP,
    X0, X19, X20, X21, X22, X29, X3, X30, ZERO_REGISTER,
};
use mmap_rs::MmapOptions;

fn _print_u32_hex_le(value: u32) {
    let bytes = value.to_le_bytes();
    for byte in &bytes {
        print!("{:02x}", byte);
    }
    println!();
}

fn _print_i32_hex_le(value: i32) {
    let bytes = value.to_le_bytes();
    for byte in &bytes {
        print!("{:02x}", byte);
    }
    println!();
}

fn mov_imm(destination: Register, input: u16) -> Asm {
    Asm::Movz {
        sf: destination.sf(),
        hw: 0,
        // TODO: Shouldn't this be a u16??
        imm16: input as i32,
        rd: destination,
    }
}

fn mov_reg(destination: Register, source: Register) -> Asm {
    Asm::MovOrrLogShift {
        sf: destination.sf(),
        rm: source,
        rd: destination,
    }
}

fn mov_sp(destination: Register, source: Register) -> Asm {
    Asm::MovAddAddsubImm {
        sf: destination.sf(),
        rn: source,
        rd: destination,
    }
}

#[allow(unused)]
fn add(destination: Register, a: Register, b: Register) -> Asm {
    Asm::AddAddsubShift {
        sf: destination.sf(),
        shift: 0,
        imm6: 0,
        rn: a,
        rm: b,
        rd: destination,
    }
}

fn sub(destination: Register, a: Register, b: Register) -> Asm {
    Asm::SubAddsubShift {
        sf: destination.sf(),
        shift: 0,
        imm6: 0,
        rn: a,
        rm: b,
        rd: destination,
    }
}

fn mul(destination: Register, a: Register, b: Register) -> Asm {
    Asm::Madd {
        sf: destination.sf(),
        rm: b,
        ra: ZERO_REGISTER,
        rn: a,
        rd: destination,
    }
}
fn div(destination: Register, a: Register, b: Register) -> Asm {
    Asm::Sdiv {
        sf: destination.sf(),
        rm: b,
        rn: a,
        rd: destination,
    }
}

fn ret() -> Asm {
    Asm::Ret {
        rn: Register {
            size: Size::S64,
            index: 30,
        },
    }
}

fn compare(a: Register, b: Register) -> Asm {
    Asm::CmpSubsAddsubShift {
        sf: a.sf(),
        shift: 0,
        rm: b,
        imm6: 0,
        rn: a,
    }
}

fn jump_equal(destination: u32) -> Asm {
    Asm::BCond {
        imm19: destination as i32,
        cond: 0,
    }
}
#[allow(unused)]
fn jump_not_equal(destination: u32) -> Asm {
    Asm::BCond {
        imm19: destination as i32,
        cond: 1,
    }
}
#[allow(unused)]
fn jump_greater_or_equal(destination: u32) -> Asm {
    Asm::BCond {
        imm19: destination as i32,
        cond: 10,
    }
}
#[allow(unused)]
fn jump_less_than(destination: u32) -> Asm {
    Asm::BCond {
        imm19: destination as i32,
        cond: 11,
    }
}
#[allow(unused)]
fn jump_greater(destination: u32) -> Asm {
    Asm::BCond {
        imm19: destination as i32,
        cond: 12,
    }
}
#[allow(unused)]
fn jump_less_or_equal(destination: u32) -> Asm {
    Asm::BCond {
        imm19: destination as i32,
        cond: 13,
    }
}
fn jump(destination: u32) -> Asm {
    Asm::BCond {
        imm19: destination as i32,
        cond: 14,
    }
}
fn store_pair(reg1: Register, reg2: Register, destination: Register, offset: i32) -> Asm {
    Asm::StpGen {
        // TODO: Make this better/document this is about 64 bit or not
        opc: 0b10,
        class_selector: StpGenSelector::PreIndex,
        imm7: offset,
        rt2: reg2,
        rt: reg1,
        rn: destination,
    }
}

fn load_pair(reg1: Register, reg2: Register, destination: Register, offset: i32) -> Asm {
    Asm::LdpGen {
        opc: 0b10,
        class_selector: LdpGenSelector::PostIndex,
        // TODO: Truncate
        imm7: offset,
        rt2: reg2,
        rt: reg1,
        rn: destination,
    }
}

#[allow(unused)]
fn branch_with_link(destination: i32) -> Asm {
    Asm::Bl {
        imm26: destination as i32,
    }
}

fn branch_with_link_register(register: Register) -> Asm {
    Asm::Blr { rn: register }
}

fn breakpoint() -> Asm {
    Asm::Brk { imm16: 30 }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct Label {
    index: usize,
}

struct Lang {
    instructions: Vec<Asm>,
    label_index: usize,
    label_locations: HashMap<usize, usize>,
    labels: Vec<String>,
    free_volatile_registers: Vec<Register>,
    allocated_volatile_registers: Vec<Register>,
    stack_size: i32,
    max_stack_size: i32,
}

// We don't know the address of the recursive call
// so we are using a placeholder
// This is probably not the best approach, but will
// work for now.
// I realized after implementing this, I could just probably
// have used a label and always had a self/start label.
const RECURSE_PLACEHOLDER_REGISTER: Register = Register {
    size: Size::S64,
    index: 255,
};

#[allow(unused)]
impl Lang {
    fn new() -> Self {
        Lang {
            instructions: vec![],
            label_locations: HashMap::new(),
            label_index: 0,
            labels: vec![],
            free_volatile_registers: vec![X22, X21, X20, X19],
            allocated_volatile_registers: vec![],
            stack_size: 0,
            max_stack_size: 0,
        }
    }

    fn increment_stack_size(&mut self, size: i32) {
        self.stack_size += size;
        if self.stack_size > self.max_stack_size {
            self.max_stack_size = self.stack_size;
        }
    }

    fn prelude(&mut self, offset: i32) {
        self.store_pair(X29, X30, SP, offset);
        self.mov_reg(X29, SP);
    }

    fn epilogue(&mut self, offset: i32) {
        self.load_pair(X29, X30, SP, offset);
    }

    fn get_label_index(&mut self) -> usize {
        let current_label_index = self.label_index;
        self.label_index += 1;
        current_label_index
    }

    fn breakpoint(&mut self) {
        self.instructions.push(breakpoint())
    }

    fn mov(&mut self, destination: Register, input: u16) {
        self.instructions.push(mov_imm(destination, input));
    }
    fn mov_64(&mut self, destination: Register, input: isize) {
        self.instructions.extend(mov_64_bit_num(destination, input));
    }

    fn store_pair(&mut self, reg1: Register, reg2: Register, destination: Register, offset: i32) {
        self.increment_stack_size(2);
        self.instructions
            .push(store_pair(reg1, reg2, destination, offset));
    }
    fn load_pair(&mut self, reg1: Register, reg2: Register, destination: Register, offset: i32) {
        self.increment_stack_size(-2);
        self.instructions
            .push(load_pair(reg1, reg2, destination, offset));
    }
    fn add(&mut self, destination: Register, a: Register, b: Register) {
        self.instructions.push(add(destination, a, b));
    }
    fn sub(&mut self, destination: Register, a: Register, b: Register) {
        self.instructions.push(sub(destination, a, b));
    }
    fn mul(&mut self, destination: Register, a: Register, b: Register) {
        self.instructions.push(mul(destination, a, b));
    }
    fn div(&mut self, destination: Register, a: Register, b: Register) {
        self.instructions.push(div(destination, a, b));
    }
    fn ret(&mut self) {
        self.instructions.push(ret());
    }
    fn compare(&mut self, a: Register, b: Register) {
        self.instructions.push(compare(a, b));
    }
    fn jump_equal(&mut self, destination: Label) {
        self.instructions.push(jump_equal(destination.index as u32));
    }
    fn jump_not_equal(&mut self, destination: Label) {
        self.instructions
            .push(jump_not_equal(destination.index as u32));
    }
    fn jump_greater_or_equal(&mut self, destination: Label) {
        self.instructions
            .push(jump_greater_or_equal(destination.index as u32));
    }
    fn jump_less_than(&mut self, destination: Label) {
        self.instructions
            .push(jump_less_than(destination.index as u32));
    }
    fn jump_greater(&mut self, destination: Label) {
        self.instructions
            .push(jump_greater(destination.index as u32));
    }
    fn jump_less_or_equal(&mut self, destination: Label) {
        self.instructions
            .push(jump_less_or_equal(destination.index as u32));
    }
    fn jump(&mut self, destination: Label) {
        self.instructions.push(jump(destination.index as u32));
    }

    fn store_on_stack(&mut self, reg: Register, offset: i32) {
        self.increment_stack_size(1);
        self.instructions.push(Asm::StrImmGen {
            size: 0b11,
            imm9: 0, // not used
            rn: SP,
            rt: reg,
            imm12: offset,
            class_selector: StrImmGenSelector::UnsignedOffset,
        });
    }

    fn load_from_stack(&mut self, destination: Register, offset: i32) {
        self.increment_stack_size(-1);
        self.instructions.push(Asm::LdrImmGen {
            size: 0b11,
            imm9: 0, // not used
            rn: SP,
            rt: destination,
            imm12: offset,
            class_selector: LdrImmGenSelector::UnsignedOffset,
        });
    }

    fn new_label(&mut self, name: &str) -> Label {
        self.labels.push(name.to_string());
        Label {
            index: self.get_label_index(),
        }
    }
    fn write_label(&mut self, label: Label) {
        self.label_locations
            .insert(label.index, self.instructions.len());
    }

    fn compile(&mut self, ptr: *const u8) -> &Vec<Asm> {
        self.patch_labels();
        self.patch_prelude_and_epilogue();
        self.patch_recurse();
        &self.instructions
    }

    fn call_rust_function(&mut self, register: Register, func: *const u8) {
        self.mov_64(register, func as isize);
        self.call(register)
    }

    fn call(&mut self, register: Register) {
        self.instructions.push(branch_with_link_register(register));
    }

    fn patch_labels(&mut self) {
        for (instruction_index, instruction) in self.instructions.iter_mut().enumerate() {
            if let Asm::BCond { imm19, cond: _ } = instruction {
                let label_index = *imm19 as usize;
                let label_location = self.label_locations.get(&label_index);
                match label_location {
                    Some(label_location) => {
                        let relative_position =
                            *label_location as isize - instruction_index as isize;
                        *imm19 = relative_position as i32;
                    }
                    None => {
                        println!("Couldn't find label {:?}", self.labels.get(label_index));
                    }
                }
            }
        }
    }

    fn mov_reg(&mut self, destination: Register, source: Register) {
        self.instructions.push(match (destination, source) {
            (SP, _) => mov_sp(destination, source),
            (_, SP) => mov_sp(destination, source),
            _ => mov_reg(destination, source),
        });
    }

    fn volatile_register(&mut self) -> Register {
        let next_register = self.free_volatile_registers.pop().unwrap();
        self.allocated_volatile_registers.push(next_register);
        next_register
    }
    fn free_register(&mut self, reg: Register) {
        self.free_volatile_registers.push(reg);
        self.allocated_volatile_registers
            .retain(|&allocated| allocated != reg);
    }

    fn reserve_register(&mut self, reg: Register) {
        self.free_volatile_registers.retain(|&free| free != reg);
        if !self.allocated_volatile_registers.contains(&reg) {
            self.allocated_volatile_registers.push(reg);
        }
    }

    fn arg(&self, arg: u8) -> Register {
        assert!(
            arg < 8,
            "Only 8 arguments are supported on aarch64, but {} was requested",
            arg
        );
        Register {
            size: Size::S64,
            index: arg,
        }
    }

    fn recurse(&self) -> Register {
        RECURSE_PLACEHOLDER_REGISTER
    }

    fn ret_reg(&self) -> Register {
        X0
    }

    fn patch_prelude_and_epilogue(&mut self) {
        let max = self.max_stack_size as u64;
        let max = max.next_power_of_two();
        let max = max as i32;
        // Find the first store pair and patch it based
        // on the max stack size
        if let Some(Asm::StpGen { imm7, .. }) = self
            .instructions
            .iter_mut()
            .position(|instruction| matches!(instruction, Asm::StpGen { .. }))
            .map(|i| &mut self.instructions[i])
        {
            *imm7 = -max;
        } else {
            unreachable!();
        }

        // Same but last load pair
        // note the rposition
        if let Some(Asm::LdpGen { imm7, .. }) = self
            .instructions
            .iter_mut()
            .rposition(|instruction| matches!(instruction, Asm::LdpGen { .. }))
            .map(|i| &mut self.instructions[i])
        {
            *imm7 = max;
        } else {
            unreachable!();
        }
    }

    fn patch_recurse(&mut self) {
        for (index, instruction) in self.instructions.iter_mut().enumerate() {
            if let Asm::Blr { rn } = instruction {
                if rn == &RECURSE_PLACEHOLDER_REGISTER {
                    *instruction = branch_with_link(-(index as i32));
                }
            }
        }
    }
}

fn mov_64_bit_num(register: Register, num: isize) -> Vec<Asm> {
    let mut num = num;
    let mut result = vec![];

    result.push(Asm::Movz {
        sf: register.sf(),
        hw: 0,
        imm16: num as i32 & 0xffff,
        rd: register,
    });
    num >>= 16;
    result.push(Asm::Movk {
        sf: register.sf(),
        hw: 0b01,
        imm16: num as i32 & 0xffff,
        rd: register,
    });
    num >>= 16;
    result.push(Asm::Movk {
        sf: register.sf(),
        hw: 0b10,
        imm16: num as i32 & 0xffff,
        rd: register,
    });
    num >>= 16;
    result.push(Asm::Movk {
        sf: register.sf(),
        hw: 0b11,
        imm16: num as i32 & 0xffff,
        rd: register,
    });

    result
}

fn use_the_assembler(n: i64, lang: &mut Lang) -> Result<(), Box<dyn Error>> {
    let mut buffer = MmapOptions::new(MmapOptions::page_size())?.map_mut()?;
    let memory = &mut buffer[..];

    let ptr = memory.as_ptr();

    let instructions = lang.compile(ptr);

    let mut bytes = vec![];
    for instruction in instructions.iter() {
        for byte in instruction.encode().to_le_bytes() {
            bytes.push(byte);
        }
    }
    for (i, byte) in bytes.iter().enumerate() {
        memory[i] = *byte;
    }
    let size = buffer.size();
    buffer.flush(0..size)?;

    let exec = buffer.make_exec().unwrap_or_else(|(_map, e)| {
        panic!("Failed to make mmap executable: {}", e);
    });

    let f: fn(i64) -> u64 = unsafe { mem::transmute(exec.as_ref().as_ptr()) };

    let time = Instant::now();

    let result1 = f(n);
    println!("Our time {:?}", time.elapsed());
    let time = Instant::now();
    let result2 = fib_rust(n as usize);
    println!("Rust time {:?}", time.elapsed());
    println!("{} {}", result1, result2);

    Ok(())
}

fn fib_rust(n: usize) -> usize {
    if n <= 1 {
        return n;
    }
    fib_rust(n - 1) + fib_rust(n - 2)
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
enum Condition {
    LessThanOrEqual,
}

#[derive(Debug, Copy, Clone)]
#[allow(dead_code)]
enum Value {
    Register(VirtualRegister),
    UnSignedConstant(usize),
    SignedConstant(isize),
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
struct VirtualRegister {
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
    Ret(Value),
    Breakpoint,
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
        lang: &mut Lang,
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
struct Ir {
    register_index: usize,
    instructions: Vec<Instruction>,
    labels: Vec<Label>,
    label_names: Vec<String>,
    label_locations: HashMap<usize, usize>,
}

impl Ir {
    fn new() -> Self {
        Self {
            register_index: 0,
            instructions: vec![],
            labels: vec![],
            label_names: vec![],
            label_locations: HashMap::new(),
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

    fn arg(&mut self, n: usize) -> VirtualRegister {
        self.next_register(Some(n), true)
    }

    fn volatile_register(&mut self) -> VirtualRegister {
        self.next_register(None, true)
    }

    fn recurse<A>(&mut self, args: Vec<A>) -> Value
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

    fn sub<A, B>(&mut self, a: A, b: B) -> Value
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

    fn add<A, B>(&mut self, a: A, b: B) -> Value
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

    fn jump_if<A, B>(&mut self, label: Label, condition: Condition, a: A, b: B)
    where
        A: Into<Value>,
        B: Into<Value>,
    {
        let a = self.assign_new(a.into());
        let b = self.assign_new(b.into());
        self.instructions
            .push(Instruction::JumpIf(label, condition, a.into(), b.into()));
    }

    fn _assign(&mut self, dest: VirtualRegister, val: Value) {
        self.instructions.push(Instruction::Assign(dest, val));
    }

    fn assign_new(&mut self, val: Value) -> VirtualRegister {
        if let Value::Register(register) = val {
            return register;
        }
        let register = self.next_register(None, false);
        self.instructions.push(Instruction::Assign(register, val));
        register
    }

    fn ret<A>(&mut self, n: A)
    where
        A: Into<Value>,
    {
        self.instructions.push(Instruction::Ret(n.into()));
    }

    fn label(&mut self, arg: &str) -> Label {
        let label_index = self.labels.len();
        self.label_names.push(arg.to_string());
        let label = Label { index: label_index };
        self.labels.push(label);
        label
    }

    fn write_label(&mut self, early_exit: Label) {
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

    fn compile(&mut self) -> Lang {
        let mut lang = Lang::new();
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
                },
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
                Instruction::JumpIf(label, condition, a, b) => {
                    let a = a.try_into().unwrap();
                    let a = alloc.allocate_register(index, a, &mut lang);
                    let b = b.try_into().unwrap();
                    let b = alloc.allocate_register(index, b, &mut lang);
                    let label = ir_label_to_lang_label.get(label).unwrap();
                    lang.compare(a, b);
                    match condition {
                        Condition::LessThanOrEqual => lang.jump_less_or_equal(*label),
                    }
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
                },
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
}

fn fib2_prime() -> Ir {
    let mut ir = Ir::new();
    // ir.breakpoint();
    let n = ir.arg(0);

    let early_exit = ir.label("early_exit");

    ir.jump_if(early_exit, Condition::LessThanOrEqual, n, 1);

    let reg_0 = ir.sub(n, 1);
    let reg_1 = ir.recurse(vec![reg_0]);

    let reg_2 = ir.sub(n, 2);
    let reg_3 = ir.recurse(vec![reg_2]);

    let reg_4 = ir.add(reg_1, reg_3);
    ir.ret(reg_4);

    ir.write_label(early_exit);
    ir.ret(n);

    ir
}

#[allow(dead_code)]
fn fib() -> Lang {
    let mut lang = Lang::new();
    // lang.breakpoint();
    lang.prelude(-4);

    let const_1 = lang.volatile_register();
    lang.mov(const_1, 1);

    let recursive_case = lang.new_label("recursive_case");
    let return_label = lang.new_label("return");

    lang.compare(lang.arg(0), const_1);
    lang.jump_greater(recursive_case);
    lang.jump(return_label);

    lang.write_label(recursive_case);

    let arg_0_minus_1 = lang.volatile_register();
    lang.sub(arg_0_minus_1, lang.arg(0), const_1);

    lang.store_on_stack(lang.arg(0), 2);
    lang.mov_reg(lang.arg(0), arg_0_minus_1);

    let first_recursive_result = lang.volatile_register();

    lang.call(lang.arg(1));
    lang.mov_reg(first_recursive_result, lang.ret_reg());

    lang.load_from_stack(lang.arg(0), 2);

    lang.sub(arg_0_minus_1, lang.arg(0), const_1);
    lang.sub(arg_0_minus_1, arg_0_minus_1, const_1);
    lang.mov_reg(lang.arg(0), arg_0_minus_1);

    lang.store_on_stack(first_recursive_result, 2);

    lang.call(lang.arg(1));

    lang.load_from_stack(first_recursive_result, 2);

    lang.add(lang.ret_reg(), lang.ret_reg(), first_recursive_result);

    lang.write_label(return_label);

    lang.epilogue(4);

    lang.ret();
    lang
}

#[allow(dead_code)]
fn countdown_codegen() -> Lang {
    let mut lang = Lang::new();

    let loop_start = lang.new_label("loop_start");
    let loop_exit = lang.new_label("loop_exit");

    // lang.breakpoint();
    lang.store_pair(X29, X30, SP, -2);
    lang.mov_reg(X29, SP);
    lang.mov(X22, 10);
    lang.mov(X20, 0);
    lang.mov(X21, 1);

    lang.write_label(loop_start);

    lang.compare(X20, X22);
    lang.jump_equal(loop_exit);
    lang.sub(X22, X22, X21);
    lang.mov_reg(X0, X22);
    lang.call_rust_function(X3, print_it as *const u8);

    lang.jump(loop_start);
    lang.write_label(loop_exit);
    lang.load_pair(X29, X30, SP, 2);
    lang.ret();
    lang
}

#[no_mangle]
extern "C" fn print_it(num: u64) -> u64 {
    println!("{}", num);
    num
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut ir = fib2_prime();
    let lifetimes = ir.get_register_lifetime();
    // let mut lang = Lang::new();
    // let register_assignment: HashMap<VirtualRegister, Register> = ir.allocate_registers(&mut lang, &lifetimes);
    Ir::draw_lifetimes(&lifetimes);
    // println!("{:#?}", register_assignment);
    let mut lang = ir.compile();
    // println!("{:#?}", ir);
    // use_the_assembler(30, &mut lang)?;

    let new_fib = test_fib();
    println!("{:#?}", new_fib);
    let mut new_ir = new_fib.compile();
    println!("{:#?}", new_ir);
    
    let mut lang = new_ir.compile();
    use_the_assembler(30, &mut lang)?;
    Ok(())
}





#[derive(Debug, Clone, PartialEq, Eq)]
enum Ast {
    Function {
        name: String,
        args: Vec<String>,
        body: Vec<Ast>,
    },
    If {
        condition: Box<Ast>,
        then: Box<Ast>,
        else_: Box<Ast>,
    },
    Condition {
        operator: Condition,
        left: Box<Ast>,
        right: Box<Ast>,
    },
    Add {
        left: Box<Ast>,
        right: Box<Ast>,
    },
    Sub {
        left: Box<Ast>,
        right: Box<Ast>,
    },
    Recurse {
        args: Vec<Ast>,
    },
    Return(Box<Ast>),
    Constant(i64),
    Variable(String),
}

impl Ast {
    fn compile(&self) -> Ir {
        let mut compiler = AstCompiler {
            ast: self.clone(),
            variables: HashMap::new(),
        };
        compiler.compile()
    }
}

struct AstCompiler {
    ast: Ast,
    variables: HashMap<String, VirtualRegister>,
}

// I'm using this because there are some places I'm not
// currently sure the right thing to return and I want
// signal exactly where that is.
const TEMP_PLACEHOLDER : Value = Value::UnSignedConstant(42);

impl AstCompiler {
    fn compile(&mut self) -> Ir {
        let mut ir = Ir::new();
        self.compile_to_ir(&Box::new(self.ast.clone()), &mut ir);
        ir
    }

    fn compile_to_ir(&mut self, ast: &Box<Ast>, ir: &mut Ir) -> Value {
        match ast.as_ref().clone() {
            Ast::Function { name, args, body } => {
                for (index, arg) in args.iter().enumerate() {
                    let reg = ir.arg(index);
                    self.variables.insert(arg.clone(), reg);
                }


                for ast in body {
                    self.compile_to_ir(&Box::new(ast), ir);
                }
                // Should actually return the last value?
                // Or what is explicitly returned?
                // I might need to just deal with all of this a bit differently
                // Or I need to return the return_reg?
                // need to think about this.
                TEMP_PLACEHOLDER
            }
            Ast::If { condition, then, else_ } => {
                // TODO: My condition system is a bit ugly
                // Mostly because I don't have booleans
                if let Ast::Condition { operator, left, right } = condition.as_ref() {
                    let a = self.compile_to_ir(left, ir);
                    let b = self.compile_to_ir(right, ir);

                    let then_label = ir.label("then");
                    ir.jump_if(then_label, *operator, a, b);

                    self.compile_to_ir(&else_, ir);
                    ir.write_label(then_label);
                    self.compile_to_ir(&then, ir);

                    // TODO: I need ifs to be expressions.
                    // Right now they are acting as statements.
                    TEMP_PLACEHOLDER

                } else {
                    panic!("Expected condition")
                }
            }
            Ast::Return(ast) => {
                let value = self.compile_to_ir(&ast, ir);
                ir.ret(value);
                // Do I need a concept of the return register in the IR?
                TEMP_PLACEHOLDER
            }
            Ast::Add { left, right } => {
                let left = self.compile_to_ir(&left, ir);
                let right = self.compile_to_ir(&right, ir);
                ir.add(left, right)
            }
            Ast::Sub { left, right } => {
                let left = self.compile_to_ir(&left, ir);
                let right = self.compile_to_ir(&right, ir);
                ir.sub(left, right)
            }
            Ast::Recurse { args } => {
                let mut args = args.iter().map(|arg| self.compile_to_ir(&Box::new(arg.clone()), ir)).collect();
                ir.recurse(args)
            }
            Ast::Constant(n) => {
                Value::SignedConstant(n as isize)
            },
            Ast::Variable(name) => {
                let reg = self.variables.get(&name).unwrap();
                Value::Register(*reg)
            }
            Ast::Condition { .. } => {
                panic!("Condition should be handled by if")
            }
        }
    }
}



macro_rules! ast {
    ((fn $name:ident[] 
        $body:tt
     )) => {
        Ast::Func{ 
            name: stringify!($name).to_string(), 
            args: vec![],
            body: vec![Ast::Return(Box::new(ast!($body)))]
        }
    };
    ((fn $name:ident[$arg:ident] 
        $body:tt
     )) => {
        Ast::Function { 
            name: stringify!($name).to_string(), 
            args: vec![stringify!($arg).to_string()],
            body: vec![ast!($body)]
        }
    };
    ((fn $name:ident[$arg1:ident $arg2:ident] 
        $body:tt
     )) => {
        Ast::Func{ 
            name: stringify!($name).to_string(), 
            args: vec![stringify!($arg1).to_string(), stringify!($arg2).to_string()],
            body: vec![ast!($body)]
        }
    };
    ((fn $name:ident[$arg1:ident $arg2:ident $arg3:ident] 
        $body:tt
     )) => {
        Ast::Func{ 
            name: stringify!($name).to_string(), 
            args: vec![stringify!($arg1).to_string(), stringify!($arg2).to_string(), stringify!($arg3).to_string()],
            body: vec![Ast::Return(Box::new(ast!($body)))]
        }
    };
    ((let [$name:tt $val:tt]
        $body:tt
    )) => {
        Ast::Do(vec![
            Ast::Let(stringify!($name).to_string(), Box::new(ast!($val))),
            ast!($body)]);
    };
    ((if (<= $arg:tt $val:tt)
        $result1:tt
        $result2:tt
    )) => {
        Ast::If{
            condition: Box::new(Ast::Condition {
                operator: Condition::LessThanOrEqual,
                left: Box::new(ast!($arg)),
                right: Box::new(ast!($val))
            }),
            then: Box::new(ast!($result1)),
            else_: Box::new(ast!($result2))
        }
    };
    ((+ $arg1:tt $arg2:tt)) => {
        Ast::Add {
            left: Box::new(ast!($arg1)),
            right: Box::new(ast!($arg2))
        }
    };
    ((+ $arg1:tt $arg2:tt $($args:tt)+)) => {
            Ast::Add(Box::new(ast!($arg1)),
                     Box::new(ast!((+ $arg2 $($args)+))))
    };
    ((- $arg1:tt $arg2:tt)) => {
        Ast::Sub {
            left: Box::new(ast!($arg1)),
            right: Box::new(ast!($arg2))
        }
    };

    ((do $($arg1:tt)+)) => {
        Ast::Do(vec![$(ast!($arg1)),+])
    };
    ((return $arg:tt)) => {
        Ast::Return(Box::new(ast!($arg)))
    };
    (($f:ident $arg:tt)) => {
        Ast::Recurse {
            args: vec![ast!($arg)]
        }
    };
    // (($f:ident $arg1:tt $arg2:tt)) => {
    //     Ast::Call2(stringify!($f).to_string(), Box::new(ast!($arg1)), Box::new(ast!($arg2)))
    // };
    // (($f:ident $arg1:tt $arg2:tt $arg3:tt)) => {
    //     Ast::Call3(stringify!($f).to_string(), Box::new(ast!($arg1)), Box::new(ast!($arg2)), Box::new(ast!($arg3)))
    // };
    ($int:literal) => {
        Ast::Constant($int)
    };
    ($var:ident) => {
        Ast::Variable(stringify!($var).to_string())
    }
}

fn test_fib() -> Ast {
    ast! {
        (fn fib [n]
            (if (<= n 1)
                (return n)
                (return (+ (fib (- n 1)) (fib (- n 2))))))
    }
}



// TODO:
// Register allocator
// Runtime?
//     Function in our language names and calling them.
//     Built-in functions
//     Stack
//     Heap
// Parser
// Debugging
