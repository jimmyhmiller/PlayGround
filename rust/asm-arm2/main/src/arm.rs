use asm::arm::{
    ArmAsm, LdpGenSelector, LdrImmGenSelector, Register, Size, StpGenSelector, StrImmGenSelector,
    SP, X0, X19, X20, X21, X22, X23, X24, X25, X26, X27, X28, X29, X30, ZERO_REGISTER,
};

use std::collections::HashMap;

use crate::{
    common::Label,
    debugger,
    ir::{BuiltInTypes, Condition},
    Data, Message,
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

// Too lazy to understand this. But great explanation here
// https://kddnewton.com/2022/08/11/aarch64-bitmask-immediates.html
// Code taken from there
pub struct BitmaskImmediate {
    n: u8,
    imms: u8,
    immr: u8,
}

/// Is this number's binary representation all 1s?
fn is_mask(imm: u64) -> bool {
    ((imm + 1) & imm) == 0
}

/// Is this number's binary representation one or more 1s followed by
/// one or more 0s?
fn is_shifted_mask(imm: u64) -> bool {
    is_mask((imm - 1) | imm)
}

impl TryFrom<u64> for BitmaskImmediate {
    type Error = ();

    fn try_from(value: u64) -> Result<Self, Self::Error> {
        if value == 0 || value == u64::MAX {
            return Err(());
        }
        let mut imm = value;
        let mut size = 64;

        loop {
            size >>= 1;
            let mask = (1 << size) - 1;

            if (imm & mask) != ((imm >> size) & mask) {
                size <<= 1;
                break;
            }

            if size <= 2 {
                break;
            }
        }
        let trailing_ones: u32;
        let left_rotations: u32;

        let mask = u64::MAX >> (64 - size);
        imm &= mask;

        if is_shifted_mask(imm) {
            left_rotations = imm.trailing_zeros();
            trailing_ones = (imm >> left_rotations).trailing_ones();
        } else {
            imm |= !mask;
            if !is_shifted_mask(!imm) {
                return Err(());
            }

            let leading_ones = imm.leading_ones();
            left_rotations = 64 - leading_ones;
            trailing_ones = leading_ones + imm.trailing_ones() - (64 - size);
        }

        // immr is the number of right rotations it takes to get from the
        // matching unrotated pattern to the target value.
        let immr = (size - left_rotations) & (size - 1);

        // imms is encoded as the size of the pattern, a 0, and then one less
        // than the number of sequential 1s.
        let imms = (!(size - 1) << 1) | (trailing_ones - 1);

        // n is 1 if the element size is 64-bits, and 0 otherwise.
        let n = ((imms >> 6) & 1) ^ 1;

        Ok(BitmaskImmediate {
            n: n as u8,
            imms: (imms & 0x3f) as u8,
            immr: (immr & 0x3f) as u8,
        })
    }
}

pub fn and_imm(destination: Register, a: Register, b: i32) -> ArmAsm {
    let immediate: Result<BitmaskImmediate, _> = (b as u64).try_into();
    let immediate = immediate.unwrap();

    ArmAsm::AndLogImm {
        sf: destination.sf(),
        n: immediate.n as i32,
        immr: immediate.immr as i32,
        imms: immediate.imms as i32,
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
    fn arm_condition(&self) -> i32 {
        match self {
            Condition::Equal => 0,
            Condition::NotEqual => 1,
            Condition::LessThan => 11,
            Condition::LessThanOrEqual => 13,
            Condition::GreaterThan => 12,
            Condition::GreaterThanOrEqual => 10,
        }
    }
    fn arm_inverted_condition(&self) -> i32 {
        match self {
            Condition::Equal => 1,
            Condition::NotEqual => 0,
            Condition::LessThan => 13,
            Condition::LessThanOrEqual => 11,
            Condition::GreaterThan => 10,
            Condition::GreaterThanOrEqual => 12,
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
            cond: condition.arm_inverted_condition(),
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

pub fn jump_not_equal(destination: u32) -> ArmAsm {
    ArmAsm::BCond {
        imm19: destination as i32,
        cond: 1,
    }
}

pub fn jump_greater_or_equal(destination: u32) -> ArmAsm {
    ArmAsm::BCond {
        imm19: destination as i32,
        cond: 10,
    }
}

pub fn jump_less_than(destination: u32) -> ArmAsm {
    ArmAsm::BCond {
        imm19: destination as i32,
        cond: 11,
    }
}

pub fn jump_greater(destination: u32) -> ArmAsm {
    ArmAsm::BCond {
        imm19: destination as i32,
        cond: 12,
    }
}

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
        class_selector: StpGenSelector::PreIndex,
        imm7: offset,
        rt2: reg2,
        rt: reg1,
        rn: destination,
    }
}

pub fn load_pair(reg1: Register, reg2: Register, destination: Register, offset: i32) -> ArmAsm {
    ArmAsm::LdpGen {
        opc: 0b10,
        class_selector: LdpGenSelector::PostIndex,
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
    // The goal right now is that everytime we
    // make a "built-in" call, we keep a map
    // of code offset to max stack value at a point.
    // (We could change this to be discrete places rather than
    //  a threshold if needed)
    // Then when we compile, we can map to pc and lookup
    // the locations based on that pc
    // This means, we should be able to walk the stack
    // and figure out where to look for potential roots
    pub stack_map: HashMap<usize, usize>,
}

impl Default for LowLevelArm {
    fn default() -> Self {
        Self::new()
    }
}

impl LowLevelArm {
    pub fn new() -> Self {
        // https://github.com/apple/swift/blob/main/docs/ABI/CallConvSummary.rst#arm64
        let canonical_volatile_registers = vec![X19, X20, X21, X22, X23, X24, X25, X26, X27, X28];
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
            stack_map: HashMap::new(),
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
        // TODO: make better/faster/fewer instructions
        self.store_pair(X29, X30, SP, offset);
        self.mov_reg(X29, SP);
        self.sub_stack_pointer(-self.max_locals);
    }

    pub fn epilogue(&mut self, offset: i32) {
        self.add_stack_pointer(self.max_locals);
        self.load_pair(X29, X30, SP, offset);
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
    pub fn and(&mut self, destination: Register, a: Register, b: i32) {
        self.instructions.push(and_imm(destination, a, b));
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
        self.instructions.push(ArmAsm::SturGen {
            size: 0b11,
            imm9: offset * 8,
            rn: X29,
            rt: reg,
        })
    }

    pub fn push_to_stack(&mut self, reg: Register) {
        self.increment_stack_size(1);
        self.store_on_stack(reg, -(self.max_locals + self.stack_size))
    }
    pub fn store_local(&mut self, value: Register, offset: i32) {
        self.store_on_stack(value, -(offset + 1));
    }

    pub fn load_from_stack(&mut self, destination: Register, offset: i32) {
        self.instructions.push(ArmAsm::LdurGen {
            size: 0b11,
            imm9: offset * 8,
            rn: X29,
            rt: destination,
        });
    }

    pub fn pop_from_stack(&mut self, reg: Register, offset: i32) {
        self.increment_stack_size(-1);
        self.load_from_stack(reg, -(offset + self.max_locals + 1))
    }

    pub fn load_local(&mut self, destination: Register, offset: i32) {
        self.load_from_stack(destination, -(offset + 1));
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

    pub fn atomic_load(&mut self, destination: Register, source: Register) {
        self.instructions.push(ArmAsm::Ldar {
            size: 0b11,
            rn: source,
            rt: destination,
        })
    }
    pub fn atomic_store(&mut self, ptr: Register, val: Register) {
        self.instructions.push(ArmAsm::Stlr {
            size: 0b11,
            rn: ptr,
            rt: val,
        });
    }

    pub fn compare_and_swap(&mut self, expected: Register, new: Register, ptr: Register) {
        self.instructions.push(ArmAsm::Cas {
            size: 0b11,
            l: 1,
            rs: expected,
            o0: 1,
            rn: ptr,
            rt: new,
        });
    }

    pub fn load_from_heap_with_reg_offset(
        &mut self,
        destination: Register,
        source: Register,
        offset: Register,
    ) {
        self.instructions.push(ArmAsm::LdrRegGen {
            size: 0b11,
            rm: offset,
            option: 0b11,
            s: 0b0,
            rn: source,
            rt: destination,
        })
    }

    pub fn store_to_heap_with_reg_offset(
        &mut self,
        destination: Register,
        source: Register,
        offset: Register,
    ) {
        self.instructions.push(ArmAsm::StrRegGen {
            size: 0b11,
            rm: offset,
            option: 0b11,
            s: 0b0,
            rn: destination,
            rt: source,
        })
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

    pub fn guard_integer(&mut self, dest: Register, a: Register, after_return: Label) {
        // TODO: I need to have some way of signaling
        // that this is a type error;
        self.and(dest, a, 0b111);
        self.compare(dest, ZERO_REGISTER);
        self.jump_not_equal(after_return);
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

    pub fn compile_directly(&mut self) -> Vec<u8> {
        let bytes = self
            .instructions
            .iter()
            .flat_map(|x| x.encode().to_le_bytes())
            .collect();
        bytes
    }

    pub fn compile_to_bytes(&mut self) -> Vec<u8> {
        let instructions = self.compile();
        let bytes = instructions
            .iter()
            .flat_map(|x| x.encode().to_le_bytes())
            .collect();
        bytes
    }

    // TODO: I should pass this information to my debugger
    // then I could visualize every stack frame
    // and do dynamic checking if the invariants I expect to hold
    // do in fact hold.
    fn update_stack_map(&mut self) {
        let offset = self.instructions.len() - 1;
        let stack_size = self.stack_size as usize;
        // TODO: Should I keep track of locals here?
        // Right now I null them out, so it would never matter
        self.stack_map.insert(offset, stack_size);
    }

    pub fn translate_stack_map(&self, pc: usize) -> Vec<(usize, usize)> {
        self.stack_map
            .iter()
            .map(|(key, value)| ((*key * 4) + pc, *value))
            .collect()
    }

    pub fn call(&mut self, register: Register) {
        self.instructions.push(branch_with_link_register(register));
        // TODO: I could be smarter here and not to do leaf nodes
        self.update_stack_map();
    }

    pub fn call_builtin(&mut self, register: Register) {
        self.instructions.push(branch_with_link_register(register));
        self.update_stack_map();
    }

    pub fn recurse(&mut self, label: Label) {
        self.instructions.push(branch_with_link(label.index as i32));
        self.update_stack_map();
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

    pub fn padded_max_locals(&mut self) -> usize {
        let mut max = self.max_locals as u64;
        let remainder = max % 2;
        if remainder != 0 {
            max += 1;
        }
        max as usize
    }

    pub fn patch_prelude_and_epilogue(&mut self) {
        let mut max = self.max_stack_size as u64 + self.max_locals as u64;
        let remainder = max % 2;
        if remainder != 0 {
            max += 1;
        }

        // Does this need to be aligned somehow? Next power of two was causing issues
        // where I was having values on the stack from native code.
        // I don't think getting rid of it was the right answer.
        let max = max as i32;

        // TODO: I don't need to patch because it is no longer based on max
        if let Some(ArmAsm::StpGen { imm7, .. }) = self
            .instructions
            .iter_mut()
            .position(|instruction| matches!(instruction, ArmAsm::StpGen { .. }))
            .map(|i| &mut self.instructions[i])
        {
            *imm7 = -2
        } else {
            unreachable!();
        }

        if let Some(ArmAsm::SubAddsubImm { imm12, .. }) = self
            .instructions
            .iter_mut()
            .position(|instruction| matches!(instruction, ArmAsm::SubAddsubImm { .. }))
            .map(|i| &mut self.instructions[i])
        {
            *imm12 = max * 8;
        } else {
            unreachable!();
        }

        if let Some(ArmAsm::AddAddsubImm { imm12, .. }) = self
            .instructions
            .iter_mut()
            .rposition(|instruction| matches!(instruction, ArmAsm::AddAddsubImm { .. }))
            .map(|i| &mut self.instructions[i])
        {
            *imm12 = max * 8;
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
            *imm7 = 2;
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

    pub fn add_stack_pointer(&mut self, bytes: i32) {
        self.instructions.push(ArmAsm::AddAddsubImm {
            sf: SP.sf(),
            rn: SP,
            rd: SP,
            imm12: bytes * 8,
            sh: 0,
        });
    }

    pub fn sub_stack_pointer(&mut self, bytes: i32) {
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
        self.instructions.push(ArmAsm::SubAddsubImm {
            sf: destination.sf(),
            rn: SP,
            rd: destination,
            imm12: offset as i32 * 8,
            sh: 0,
        });
    }
    pub fn get_stack_pointer(&mut self, destination: Register, offset: Register) {
        self.get_stack_pointer_imm(destination, 0);
        self.instructions
            .push(add(destination, destination, offset));
    }

    pub fn share_label_info_debug(&self, function_pointer: usize) {
        for (label_index, label) in self.labels.iter().enumerate() {
            let label_location = *self
                .label_locations
                .get(&label_index)
                .unwrap_or_else(|| panic!("Could not find label {}", label))
                * 4;
            debugger(Message {
                kind: "label".to_string(),
                data: Data::Label {
                    label: label.to_string(),
                    function_pointer,
                    label_index,
                    label_location,
                },
            });
        }
    }

    pub fn get_current_stack_position(&mut self, dest: Register) {
        self.instructions.push(ArmAsm::SubAddsubImm {
            sf: dest.sf(),
            rn: X29,
            rd: dest,
            imm12: (self.max_locals + self.stack_size + 1) * 8,
            sh: 0,
        });
        // TODO: This seems
    }

    pub fn set_all_locals_to_null(&mut self, null_register: Register) {
        for local_offset in 0..self.max_locals {
            self.store_local(null_register, local_offset)
        }
    }

    pub fn check_for_pause(
        &mut self,
        function: Register,
        compiler_pointer_reg: Register,
        pause_atom: Register,
    ) {
        let pause_label = self.new_label("pause");
        // Cheating. I think I know that X19 isn't used here
        // because this is at the beginning of a function
        self.atomic_load(X19, pause_atom);
        self.compare(X19, ZERO_REGISTER);
        self.jump_not_equal(pause_label);
        self.call_builtin(function);
    }
}
