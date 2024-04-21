use asm::arm::{
    ArmAsm, LdpGenSelector, LdrImmGenSelector, Register, Size, StpGenSelector, StrImmGenSelector,
    SP, X0, X19, X20, X21, X22, X23, X24, X29, X3, X30, ZERO_REGISTER,
};

use std::collections::HashMap;

use crate::{
    common::Label, debugger, ir::{BuiltInTypes, Condition}, Data, Message
};

pub fn _print_u32_hex_le(value: u32) {
    let bytes = value.to_le_bytes();
    for byte in &bytes {
        print!("{:02x}", byte);
    }
    println!();
}

pub fn _print_i32_hex_le(value: i32) {
    let bytes = value.to_le_bytes();
    for byte in &bytes {
        print!("{:02x}", byte);
    }
    println!();
}

pub fn mov_imm(destination: Register, input: u16) -> ArmAsm {
    ArmAsm::Movz {
        sf: destination.sf(),
        hw: 0,
        // TODO: Shouldn't this be a u16??
        imm16: input as i32,
        rd: destination,
    }
}

pub fn mov_reg(destination: Register, source: Register) -> ArmAsm {
    ArmAsm::MovOrrLogShift {
        sf: destination.sf(),
        rm: source,
        rd: destination,
    }
}

pub fn mov_sp(destination: Register, source: Register) -> ArmAsm {
    ArmAsm::MovAddAddsubImm {
        sf: destination.sf(),
        rn: source,
        rd: destination,
    }
}


pub fn add(destination: Register, a: Register, b: Register) -> ArmAsm {
    ArmAsm::AddAddsubShift {
        sf: destination.sf(),
        shift: 0,
        imm6: 0,
        rn: a,
        rm: b,
        rd: destination,
    }
}

pub fn sub(destination: Register, a: Register, b: Register) -> ArmAsm {
    ArmAsm::SubAddsubShift {
        sf: destination.sf(),
        shift: 0,
        imm6: 0,
        rn: a,
        rm: b,
        rd: destination,
    }
}

pub fn mul(destination: Register, a: Register, b: Register) -> ArmAsm {
    ArmAsm::Madd {
        sf: destination.sf(),
        rm: b,
        ra: ZERO_REGISTER,
        rn: a,
        rd: destination,
    }
}

pub fn div(destination: Register, a: Register, b: Register) -> ArmAsm {
    ArmAsm::Sdiv {
        sf: destination.sf(),
        rm: b,
        rn: a,
        rd: destination,
    }
}

fn shift_right(destination: Register, a: Register, b: i32) -> ArmAsm {
    ArmAsm::LsrUbfm {
        sf: destination.sf(),
        rn: a,
        rd: destination,
        n: destination.sf(),
        immr: b,
        imms: 0b111111,
    }
}

fn shift_left(destination: Register, a: Register, b: i32) -> ArmAsm {
    let immr = 64 - b;
    ArmAsm::LslUbfm {
        sf: destination.sf(),
        rn: a,
        rd: destination,
        n: destination.sf(),
        immr,
        imms: immr - 1,
    }
}

pub fn ret() -> ArmAsm {
    ArmAsm::Ret {
        rn: Register {
            size: Size::S64,
            index: 30,
        },
    }
}

pub fn or(destination: Register, a: Register, b: Register) -> ArmAsm {
    ArmAsm::OrrLogShift {
        sf: destination.sf(),
        shift: 0,
        rm: b,
        rn: a,
        rd: destination,
        imm6: 0,
    }
}

#[allow(unused)]
pub fn and_imm(destination: Register, a: Register, b: i32) -> ArmAsm {
    assert!(b == 1);
    // Too lazy to understand this. But great explanation here
    // https://kddnewton.com/2022/08/11/aarch64-bitmask-immediates.html
    ArmAsm::AndLogImm {
        sf: destination.sf(),
        n: 1,
        immr: 0,
        imms: 0,
        rn: a,
        rd: destination,
    }
}

pub fn get_tag(destination: Register, value: Register) -> ArmAsm {
    ArmAsm::AndLogImm {
        sf: destination.sf(),
        n: 1,
        immr: 0,
        imms: 2,
        rn: value,
        rd: destination,
    }
}

pub fn tag_value(destination: Register, value: Register, tag: Register) -> Vec<ArmAsm> {
    vec![
        shift_left(destination, value, BuiltInTypes::tag_size()),
        or(destination, destination, tag),
    ]
}

pub fn compare(a: Register, b: Register) -> ArmAsm {
    ArmAsm::CmpSubsAddsubShift {
        sf: a.sf(),
        shift: 0,
        rm: b,
        imm6: 0,
        rn: a,
    }
}

impl Condition {
    #[allow(unused)]
    fn to_arm_condition(&self) -> i32 {
        match self {
            Condition::Equal => 0,
            Condition::NotEqual => 1,
            Condition::GreaterThanOrEqual => 10,
            Condition::LessThan => 11,
            Condition::GreaterThan => 12,
            Condition::LessThanOrEqual => 13,
        }
    }
    fn to_arm_inverted_condition(&self) -> i32 {
        match self {
            Condition::Equal => 1,
            Condition::NotEqual => 0,
            Condition::GreaterThanOrEqual => 11,
            Condition::LessThan => 10,
            Condition::GreaterThan => 13,
            Condition::LessThanOrEqual => 12,
        }
    }
}

pub fn compare_bool(
    condition: Condition,
    destination: Register,
    a: Register,
    b: Register,
) -> Vec<ArmAsm> {
    vec![
        ArmAsm::SubsAddsubShift {
            sf: destination.sf(),
            shift: 0,
            rm: a,
            imm6: 0,
            rn: b,
            rd: destination,
        },
        ArmAsm::CsetCsinc {
            sf: destination.sf(),
            // For some reason these conditions are inverted
            cond: condition.to_arm_inverted_condition(),
            rd: destination,
        },
    ]
}

pub fn jump_equal(destination: u32) -> ArmAsm {
    ArmAsm::BCond {
        imm19: destination as i32,
        cond: 0,
    }
}

#[allow(unused)]
pub fn jump_not_equal(destination: u32) -> ArmAsm {
    ArmAsm::BCond {
        imm19: destination as i32,
        cond: 1,
    }
}

#[allow(unused)]
pub fn jump_greater_or_equal(destination: u32) -> ArmAsm {
    ArmAsm::BCond {
        imm19: destination as i32,
        cond: 10,
    }
}

#[allow(unused)]
pub fn jump_less_than(destination: u32) -> ArmAsm {
    ArmAsm::BCond {
        imm19: destination as i32,
        cond: 11,
    }
}

#[allow(unused)]
pub fn jump_greater(destination: u32) -> ArmAsm {
    ArmAsm::BCond {
        imm19: destination as i32,
        cond: 12,
    }
}

#[allow(unused)]
pub fn jump_less_or_equal(destination: u32) -> ArmAsm {
    ArmAsm::BCond {
        imm19: destination as i32,
        cond: 13,
    }
}

pub fn jump(destination: u32) -> ArmAsm {
    ArmAsm::BCond {
        imm19: destination as i32,
        cond: 14,
    }
}

pub fn store_pair(reg1: Register, reg2: Register, destination: Register, offset: i32) -> ArmAsm {
    ArmAsm::StpGen {
        // TODO: Make this better/document this is about 64 bit or not
        opc: 0b10,
        class_selector: StpGenSelector::SignedOffset,
        imm7: offset,
        rt2: reg2,
        rt: reg1,
        rn: destination,
    }
}

pub fn load_pair(reg1: Register, reg2: Register, destination: Register, offset: i32) -> ArmAsm {
    ArmAsm::LdpGen {
        opc: 0b10,
        class_selector: LdpGenSelector::SignedOffset,
        // TODO: Truncate
        imm7: offset,
        rt2: reg2,
        rt: reg1,
        rn: destination,
    }
}

pub fn branch_with_link(destination: i32) -> ArmAsm {
    ArmAsm::Bl { imm26: destination }
}

pub fn branch_with_link_register(register: Register) -> ArmAsm {
    ArmAsm::Blr { rn: register }
}

pub fn breakpoint() -> ArmAsm {
    ArmAsm::Brk { imm16: 30 }
}

#[derive(Debug)]
pub struct LowLevelArm {
    pub instructions: Vec<ArmAsm>,
    pub label_index: usize,
    pub label_locations: HashMap<usize, usize>,
    pub labels: Vec<String>,
    pub free_volatile_registers: Vec<Register>,
    pub allocated_volatile_registers: Vec<Register>,
    pub stack_size: i32,
    pub max_stack_size: i32,
    pub max_locals: i32,
    pub canonical_volatile_registers: Vec<Register>,
}

#[allow(unused)]
impl Default for LowLevelArm {
    fn default() -> Self {
        Self::new()
    }
}

impl LowLevelArm {
    pub fn new() -> Self {
        let canonical_volatile_registers = vec![X24, X23, X22, X21, X20, X19];
        LowLevelArm {
            instructions: vec![],
            label_locations: HashMap::new(),
            label_index: 0,
            labels: vec![],
            canonical_volatile_registers: canonical_volatile_registers.clone(),
            free_volatile_registers: canonical_volatile_registers,
            allocated_volatile_registers: vec![],
            stack_size: 0,
            max_stack_size: 0,
            max_locals: 0,
        }
    }

    pub fn increment_stack_size(&mut self, size: i32) {
        self.stack_size += size;
        if self.stack_size > self.max_stack_size {
            self.max_stack_size = self.stack_size;
        }
    }

    pub fn prelude(&mut self, offset: i32) {
        // self.breakpoint();
        // 0 is a placeholder we will patch later
        self.sub_stack_pointer(-self.max_locals);
        self.store_pair(X29, X30, SP, offset);
        self.mov_reg(X29, SP);
    }

    pub fn epilogue(&mut self, offset: i32) {
        self.load_pair(X29, X30, SP, offset);
        self.add_stack_pointer(self.max_locals);
    }

    pub fn get_label_index(&mut self) -> usize {
        let current_label_index = self.label_index;
        self.label_index += 1;
        current_label_index
    }

    pub fn breakpoint(&mut self) {
        self.instructions.push(breakpoint())
    }

    pub fn mov(&mut self, destination: Register, input: u16) {
        self.instructions.push(mov_imm(destination, input));
    }
    pub fn mov_64(&mut self, destination: Register, input: isize) {
        self.instructions
            .extend(Self::mov_64_bit_num(destination, input));
    }

    pub fn store_pair(
        &mut self,
        reg1: Register,
        reg2: Register,
        destination: Register,
        offset: i32,
    ) {
        self.increment_stack_size(2);
        self.instructions
            .push(store_pair(reg1, reg2, destination, offset));
    }
    pub fn load_pair(
        &mut self,
        reg1: Register,
        reg2: Register,
        destination: Register,
        offset: i32,
    ) {
        self.increment_stack_size(-2);
        self.instructions
            .push(load_pair(reg1, reg2, destination, offset));
    }
    pub fn add(&mut self, destination: Register, a: Register, b: Register) {
        self.instructions.push(add(destination, a, b));
    }
    pub fn sub(&mut self, destination: Register, a: Register, b: Register) {
        self.instructions.push(sub(destination, a, b));
    }
    pub fn mul(&mut self, destination: Register, a: Register, b: Register) {
        self.instructions.push(mul(destination, a, b));
    }
    pub fn div(&mut self, destination: Register, a: Register, b: Register) {
        self.instructions.push(div(destination, a, b));
    }
    pub fn shift_right(&mut self, destination: Register, a: Register, b: i32) {
        self.instructions.push(shift_right(destination, a, b));
    }
    pub fn shift_left(&mut self, destination: Register, a: Register, b: i32) {
        self.instructions.push(shift_left(destination, a, b));
    }
    pub fn ret(&mut self) {
        self.instructions.push(ret());
    }
    pub fn compare(&mut self, a: Register, b: Register) {
        self.instructions.push(compare(a, b));
    }
    pub fn compare_bool(&mut self, condition: Condition, dest: Register, a: Register, b: Register) {
        self.instructions
            .extend(compare_bool(condition, dest, a, b));
    }
    pub fn tag_value(&mut self, destination: Register, value: Register, tag: Register) {
        self.instructions.extend(tag_value(destination, value, tag));
    }
    pub fn get_tag(&mut self, destination: Register, value: Register) {
        self.instructions.push(get_tag(destination, value));
    }
    pub fn jump_equal(&mut self, destination: Label) {
        self.instructions.push(jump_equal(destination.index as u32));
    }
    pub fn jump_not_equal(&mut self, destination: Label) {
        self.instructions
            .push(jump_not_equal(destination.index as u32));
    }
    pub fn jump_greater_or_equal(&mut self, destination: Label) {
        self.instructions
            .push(jump_greater_or_equal(destination.index as u32));
    }
    pub fn jump_less(&mut self, destination: Label) {
        self.instructions
            .push(jump_less_than(destination.index as u32));
    }
    pub fn jump_greater(&mut self, destination: Label) {
        self.instructions
            .push(jump_greater(destination.index as u32));
    }
    pub fn jump_less_or_equal(&mut self, destination: Label) {
        self.instructions
            .push(jump_less_or_equal(destination.index as u32));
    }
    pub fn jump(&mut self, destination: Label) {
        self.instructions.push(jump(destination.index as u32));
    }

    pub fn store_on_stack(&mut self, reg: Register, offset: i32) {
        if offset < 0 {
            println!("Got it");
        }
        self.instructions.push(ArmAsm::StrImmGen {
            size: 0b11,
            imm9: 0, // not used
            rn: SP,
            rt: reg,
            imm12: offset,
            class_selector: StrImmGenSelector::UnsignedOffset,
        });
    }

    pub fn push_to_stack(&mut self, reg: Register, offset: i32) {
        self.increment_stack_size(1);
        self.store_on_stack(reg, offset + self.max_locals)
    }
    pub fn store_local(&mut self, value: Register, offset: i32) {
        self.store_on_stack(value, offset);
    }

    pub fn load_from_stack(&mut self, destination: Register, offset: i32) {
        if offset < 0 {
            println!("Got it");
        }
        self.instructions.push(ArmAsm::LdrImmGen {
            size: 0b11,
            imm9: 0, // not used
            rn: SP,
            rt: destination,
            imm12: offset,
            class_selector: LdrImmGenSelector::UnsignedOffset,
        });
    }

    pub fn pop_from_stack(&mut self, reg: Register, offset: i32) {
        self.increment_stack_size(-1);
        self.load_from_stack(reg, offset + self.max_locals)
    }

    pub fn load_local(&mut self, destination: Register, offset: i32) {
        self.load_from_stack(destination, offset);
    }

    pub fn load_from_heap(&mut self, destination: Register, source: Register, offset: i32) {
        self.instructions.push(ArmAsm::LdrImmGen {
            size: 0b11,
            imm9: 0, // not used
            rn: source,
            rt: destination,
            imm12: offset,
            class_selector: LdrImmGenSelector::UnsignedOffset,
        });
    }

    pub fn store_on_heap(&mut self, destination: Register, source: Register, offset: i32) {
        self.instructions.push(ArmAsm::StrImmGen {
            size: 0b11,
            imm9: 0, // not used
            rn: destination,
            rt: source,
            imm12: offset,
            class_selector: StrImmGenSelector::UnsignedOffset,
        });
    }

    pub fn new_label(&mut self, name: &str) -> Label {
        self.labels.push(name.to_string());
        Label {
            index: self.get_label_index(),
        }
    }
    pub fn write_label(&mut self, label: Label) {
        self.label_locations
            .insert(label.index, self.instructions.len());
    }

    pub fn compile(&mut self) -> &Vec<ArmAsm> {
        self.patch_labels();
        self.patch_prelude_and_epilogue();
        &self.instructions
    }

    pub fn compile_to_bytes(&mut self) -> Vec<u8> {
        let instructions = self.compile();
        let bytes = instructions
            .iter()
            .flat_map(|x| x.encode().to_le_bytes())
            .collect();
        bytes
    }

    pub fn call_rust_function(&mut self, register: Register, func: *const u8) {
        self.mov_64(register, func as isize);
        self.call(register)
    }

    pub fn call(&mut self, register: Register) {
        self.instructions.push(branch_with_link_register(register));
    }

    pub fn recurse(&mut self, label: Label) {
        self.instructions.push(branch_with_link(label.index as i32));
    }

    pub fn patch_labels(&mut self) {
        for (instruction_index, instruction) in self.instructions.iter_mut().enumerate() {
            match instruction {
                ArmAsm::BCond { imm19, cond: _ } => {
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
                ArmAsm::Bl { imm26 } => {
                    let label_index = *imm26 as usize;
                    let label_location = self.label_locations.get(&label_index);
                    match label_location {
                        Some(label_location) => {
                            let relative_position =
                                *label_location as isize - instruction_index as isize;
                            *imm26 = relative_position as i32;
                        }
                        None => {
                            println!("Couldn't find label {:?}", self.labels.get(label_index));
                        }
                    }
                }
                _ => {}
            }
        }
    }

    pub fn mov_reg(&mut self, destination: Register, source: Register) {
        self.instructions.push(match (destination, source) {
            (SP, _) => mov_sp(destination, source),
            (_, SP) => mov_sp(destination, source),
            _ => mov_reg(destination, source),
        });
    }

    pub fn volatile_register(&mut self) -> Register {
        let next_register = self
            .free_volatile_registers
            .pop()
            .expect("No free registers!");
        self.allocated_volatile_registers.push(next_register);
        next_register
    }
    pub fn free_register(&mut self, reg: Register) {
        // TODO: Properly fix the fact that the zero
        // register is being put in the volatile list
        if !self.canonical_volatile_registers.contains(&reg) {
            return;
        }
        self.free_volatile_registers.push(reg);
        self.allocated_volatile_registers
            .retain(|&allocated| allocated != reg);
    }

    pub fn reserve_register(&mut self, reg: Register) {
        self.free_volatile_registers.retain(|&free| free != reg);
        if !self.allocated_volatile_registers.contains(&reg) {
            self.allocated_volatile_registers.push(reg);
        }
    }

    pub fn arg(&self, arg: u8) -> Register {
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

    pub fn ret_reg(&self) -> Register {
        X0
    }

    pub fn patch_prelude_and_epilogue(&mut self) {
        let max = self.max_stack_size as u64 + self.max_locals as u64;
        let max = max.next_power_of_two();
        let max = max as i32;
        // Find the first store pair and patch it based
        // on the max stack size
        if let Some(ArmAsm::StpGen { imm7, .. }) = self
            .instructions
            .iter_mut()
            .position(|instruction| matches!(instruction, ArmAsm::StpGen { .. }))
            .map(|i| &mut self.instructions[i])
        {
            *imm7 = max;
        } else {
            unreachable!();
        }

        if let Some(ArmAsm::SubAddsubImm { imm12, .. }) = self
            .instructions
            .iter_mut()
            .position(|instruction| matches!(instruction, ArmAsm::SubAddsubImm { .. }))
            .map(|i| &mut self.instructions[i])
        {
            *imm12 = (max + 2) * 8;
        } else {
            unreachable!();
        }

        if let Some(ArmAsm::AddAddsubImm { imm12, .. }) = self
            .instructions
            .iter_mut()
            .rposition(|instruction| matches!(instruction, ArmAsm::AddAddsubImm { .. }))
            .map(|i| &mut self.instructions[i])
        {
            *imm12 = (max + 2) * 8;
        } else {
            unreachable!();
        }

        // Same but last load pair
        // note the rposition
        if let Some(ArmAsm::LdpGen { imm7, .. }) = self
            .instructions
            .iter_mut()
            .rposition(|instruction| matches!(instruction, ArmAsm::LdpGen { .. }))
            .map(|i| &mut self.instructions[i])
        {
            *imm7 = max;
        } else {
            unreachable!();
        }
    }


    pub fn mov_64_bit_num(register: Register, num: isize) -> Vec<ArmAsm> {
        // TODO: This is not optimal, but it works
        let mut num = num;
        let mut result = vec![];

        result.push(ArmAsm::Movz {
            sf: register.sf(),
            hw: 0,
            imm16: num as i32 & 0xffff,
            rd: register,
        });
        num >>= 16;
        if num == 0 {
            return result;
        }
        result.push(ArmAsm::Movk {
            sf: register.sf(),
            hw: 0b01,
            imm16: num as i32 & 0xffff,
            rd: register,
        });
        num >>= 16;
        if num == 0 {
            return result;
        }
        result.push(ArmAsm::Movk {
            sf: register.sf(),
            hw: 0b10,
            imm16: num as i32 & 0xffff,
            rd: register,
        });
        num >>= 16;
        if num == 0 {
            return result;
        }
        result.push(ArmAsm::Movk {
            sf: register.sf(),
            hw: 0b11,
            imm16: num as i32 & 0xffff,
            rd: register,
        });

        result
    }

    fn add_stack_pointer(&mut self, bytes: i32) {
        self.instructions.push(ArmAsm::AddAddsubImm {
            sf: SP.sf(),
            rn: SP,
            rd: SP,
            imm12: bytes * 8,
            sh: 0,
        });
    }

    fn sub_stack_pointer(&mut self, bytes: i32) {
        self.instructions.push(ArmAsm::SubAddsubImm {
            sf: SP.sf(),
            rn: SP,
            rd: SP,
            imm12: bytes * 8,
            sh: 0,
        });
    }

    pub fn set_max_locals(&mut self, num_locals: usize) {
        self.max_locals = num_locals as i32;
    }

    pub fn get_stack_pointer_imm(&mut self, destination: Register, offset: isize) {
        self.instructions.push(ArmAsm::AddAddsubImm {
            sf: destination.sf(),
            rn: SP,
            rd: destination,
            imm12: offset as i32 * 8,
            sh: 0,
        });
    }
    pub fn get_stack_pointer(&mut self, destination: Register, offset: Register) {
        self.get_stack_pointer_imm(destination, 0);
        self.instructions.push(add(destination, destination, offset));
    }

    pub fn share_label_info_debug(&self, function_pointer: usize) {
        for (label_index, label) in self.labels.iter().enumerate() {
            let label_location = *self.label_locations.get(&label_index).unwrap() * 4;
            debugger(Message { 
                kind: "label".to_string(), 
                data: Data::Label {
                    label: label.to_string(),
                    function_pointer,
                    label_index,
                    label_location
                }
            });
        }
    }
}


#[allow(dead_code)]
fn fib() -> LowLevelArm {
    let mut lang = LowLevelArm::new();
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

#[no_mangle]
extern "C" fn print_it(num: u64) -> u64 {
    println!("{}", num);
    num
}

#[allow(dead_code)]
fn countdown_codegen() -> LowLevelArm {
    let mut lang = LowLevelArm::new();

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
