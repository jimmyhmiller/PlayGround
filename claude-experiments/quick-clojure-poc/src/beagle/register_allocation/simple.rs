#![allow(dead_code)]
use core::panic;
use std::collections::{BTreeMap, HashMap, HashSet};

use crate::{
    ast::{IRRange, TokenRange},
    ir::{Instruction, Value, VirtualRegister},
};

// TODO: Clean this up
// I don't actually use multiplex lifetime
// but it might be useful to have
// But it is a big waste of time to keep it around
// and not use it.
// I need to think about how to debug and visualize
// register allocation
// I might also want to make this take an IR instead of the current setup

// TODO:
// This isn't working correctly with test_ffi and my closure code
// I think my problem is how I'm restoring spills
// That code does not know anything about branches.
// Maybe I shouldn't try to reload from memory in this kind of static way
// but instead change where the value is and let the compiler
// make sure to load it, like I do locals.
// I mean, I'm even putting them in locals,
// so that would make sense.
// But the way I'm using the locals is not quite right
// I would need to just use each slot for one thing

pub struct SimpleRegisterAllocator {
    pub lifetimes: HashMap<VirtualRegister, (usize, usize)>,
    pub multiplex_lifetime: HashMap<VirtualRegister, Vec<usize>>,
    pub instructions: Vec<Instruction>,
    // Needs to be a btreemap so I iterate in a defined order
    // so that things are deterministic
    pub allocated_registers: BTreeMap<VirtualRegister, VirtualRegister>,
    pub free_registers: Vec<VirtualRegister>,
    pub num_locals: usize,
    pub max_num_locals: usize,
    pub label_locations: HashMap<usize, usize>,
    pub resulting_instructions: Vec<Instruction>,
    pub ir_range_to_token_range: Vec<(TokenRange, IRRange)>,
}

fn physical(index: usize) -> VirtualRegister {
    VirtualRegister {
        argument: None,
        index,
        volatile: true,
        is_physical: true,
    }
}

impl SimpleRegisterAllocator {
    pub fn new(
        instructions: Vec<Instruction>,
        num_locals: usize,
        label_locations: HashMap<usize, usize>,
        ir_range_to_token_range: Vec<(TokenRange, IRRange)>,
    ) -> Self {
        let instruction_len = instructions.len();
        let lifetimes = Self::get_register_lifetime(&instructions);
        let physical_registers: Vec<VirtualRegister> = (19..=28).map(physical).collect();

        SimpleRegisterAllocator {
            lifetimes,
            multiplex_lifetime: HashMap::new(),
            instructions,
            allocated_registers: BTreeMap::new(),
            free_registers: physical_registers,
            num_locals,
            max_num_locals: num_locals,
            label_locations,
            resulting_instructions: Vec::with_capacity(instruction_len),
            ir_range_to_token_range,
        }
    }
    fn get_free_register(&mut self) -> Option<VirtualRegister> {
        self.free_registers.pop()
    }

    fn free_register(&mut self, register: VirtualRegister) {
        if register.argument.is_some() {
            return;
        }

        assert!(register.is_physical);
        self.free_registers.push(register);
    }

    // TODO: I'm not super happy with this module
    // But things are working which is honestly, a bit suprising
    pub fn simplify_registers(&mut self) {
        let mut cloned_instructions = self.instructions.clone();
        let mut spilled_registers: HashMap<VirtualRegister, usize> = HashMap::new();
        let mut to_free = vec![];
        let init_free_count = self.free_registers.len();
        for (instruction_index, instruction) in cloned_instructions.iter_mut().enumerate() {
            // TODO: My labels are all off. I need to fix that
            debug_assert!(
                self.free_registers.len() + self.allocated_registers.len() == init_free_count,
                "Free registers: {:#?}, allocated registers: {:#?}",
                self.free_registers,
                self.allocated_registers
            );

            for register in instruction.get_registers() {
                if let Some(local_offset) = spilled_registers.get(&register) {
                    if let Some(new_register) = self.get_free_register() {
                        self.allocated_registers.insert(register, new_register);
                        self.extend_token_range(self.resulting_instructions.len());
                        self.resulting_instructions.push(Instruction::LoadLocal(
                            Value::Register(new_register),
                            Value::Local(*local_offset),
                        ));
                        to_free.push(register);
                        // self.num_locals -= 1;
                    } else {
                        // TODO: I think I should actually spill here
                        // But I'm actually going to intentionally leave this here so that
                        // I can know when this happens
                        panic!("Ran out of registers!");
                    }
                }

                if let Some(allocated_register) = self.allocated_registers.get(&register) {
                    instruction.replace_register(register, Value::Register(*allocated_register));
                }
                let lifetime = self.lifetimes.get(&register).cloned();

                if let Some((start, end)) = lifetime {
                    if start == instruction_index {
                        loop {
                            if register.argument.is_some() {
                                let new_register = VirtualRegister {
                                    argument: register.argument,
                                    index: register.index,
                                    volatile: false,
                                    is_physical: true,
                                };
                                self.allocated_registers.insert(register, new_register);
                                instruction
                                    .replace_register(register, Value::Register(new_register));
                                self.multiplex_lifetime
                                    .entry(new_register)
                                    .or_default()
                                    .push(instruction_index);
                                debug_assert!(
                                    self.free_registers.len() + self.allocated_registers.len()
                                        >= init_free_count,
                                    "Free registers: {:#?}, allocated registers: {:#?}",
                                    self.free_registers,
                                    self.allocated_registers
                                );
                                break;
                            } else if let Some(new_register) = self.get_free_register() {
                                self.allocated_registers.insert(register, new_register);
                                instruction
                                    .replace_register(register, Value::Register(new_register));
                                self.multiplex_lifetime
                                    .entry(new_register)
                                    .or_default()
                                    .push(instruction_index);
                                debug_assert!(
                                    self.free_registers.len() + self.allocated_registers.len()
                                        >= init_free_count,
                                    "Free registers: {:#?}, allocated registers: {:#?}",
                                    self.free_registers,
                                    self.allocated_registers
                                );
                                break;
                            } else {
                                // panic!("Spilling isn't working properly yet");
                                let (register, spilled) = self.spill(&mut spilled_registers);
                                let index = self
                                    .resulting_instructions
                                    .iter()
                                    .enumerate()
                                    .rev()
                                    .find_map(|(index, instruction)| {
                                        if instruction.get_registers().contains(&register) {
                                            Some(index)
                                        } else {
                                            None
                                        }
                                    })
                                    .unwrap();
                                self.extend_token_range(index + 1);
                                self.resulting_instructions.insert(index + 1, spilled);
                                debug_assert!(
                                    self.free_registers.len() + self.allocated_registers.len()
                                        >= init_free_count,
                                    "Free registers: {:#?}, allocated registers: {:#?}",
                                    self.free_registers,
                                    self.allocated_registers
                                );
                            }
                        }
                    }

                    if end <= instruction_index {
                        to_free.push(register);
                    }
                }
            }
            assert!(
                self.free_registers.len() + self.allocated_registers.len() >= init_free_count,
                "Free registers: {:#?}, allocated registers: {:#?}",
                self.free_registers,
                self.allocated_registers
            );
            for register in to_free.iter() {
                if let Some(allocated_register) = self.allocated_registers.get(register) {
                    let allocated_register = *allocated_register;
                    self.free_register(allocated_register);
                    self.allocated_registers.remove(register);
                    self.multiplex_lifetime
                        .entry(allocated_register)
                        .or_default()
                        .push(instruction_index);
                }
                debug_assert!(
                    self.free_registers.len() + self.allocated_registers.len() >= init_free_count,
                    "Free registers: {:#?}, allocated registers: {:#?}",
                    self.free_registers,
                    self.allocated_registers
                );
            }

            match instruction {
                Instruction::Call(dest, _, _, _) | Instruction::Recurse(dest, _) => {
                    // println!("===============");
                    let dest_clone = *dest;
                    if !self.allocated_registers.is_empty() {
                        for register in self.allocated_registers.clone().values() {
                            if let Value::Register(dest) = dest_clone
                                && dest == *register
                            {
                                continue;
                            }
                            self.extend_token_range(self.resulting_instructions.len());
                            self.resulting_instructions
                                .push(Instruction::PushStack(Value::Register(*register)));
                        }
                        self.resulting_instructions.push(instruction.clone());
                        for register in self.allocated_registers.clone().values().rev() {
                            if let Value::Register(dest) = dest_clone
                                && dest == *register
                            {
                                continue;
                            }
                            self.extend_token_range(self.resulting_instructions.len());
                            self.resulting_instructions
                                .push(Instruction::PopStack(Value::Register(*register)));
                        }
                    } else {
                        self.resulting_instructions.push(instruction.clone());
                    }
                }
                _ => {
                    self.resulting_instructions.push(instruction.clone());
                }
            }

            if let Instruction::Assign(register, _) = self.instructions[instruction_index]
                && let Instruction::Assign(new_register, _) =
                    self.resulting_instructions.last().unwrap().clone()
            {
                let Value::Register(register) = register else {
                    panic!("Not a register");
                };
                if let Some(local_offset) = spilled_registers.get(&register) {
                    self.extend_token_range(self.resulting_instructions.len());
                    self.resulting_instructions.push(Instruction::StoreLocal(
                        Value::Local(*local_offset),
                        new_register,
                    ));
                }
            }
        }

        let mut new_labels: HashMap<usize, usize> = HashMap::new();
        for (index, instruction) in self.resulting_instructions.iter().enumerate() {
            if let Instruction::Label(label) = instruction {
                new_labels.insert(index + 1, label.index);
            }
        }

        self.lifetimes = Self::get_register_lifetime(&self.instructions);
        self.label_locations = new_labels;
    }

    fn spill(
        &mut self,
        spilled_registers: &mut HashMap<VirtualRegister, usize>,
    ) -> (VirtualRegister, Instruction) {
        let (corresponding_register, latest_allocated_register) = self
            .allocated_registers
            .iter()
            .max_by_key(|(register, _)| self.lifetimes.get(register).unwrap().1)
            .unwrap();
        let (corresponding_register, latest_allocated_register) =
            (*corresponding_register, *latest_allocated_register);

        self.free_register(latest_allocated_register);
        self.allocated_registers.remove(&corresponding_register);
        self.num_locals += 1;
        self.max_num_locals = self.max_num_locals.max(self.num_locals);
        // For right now we will write the spill instruction to locals
        let spill_instruction = Instruction::StoreLocal(
            Value::Local(self.num_locals),
            Value::Register(latest_allocated_register),
        );

        spilled_registers.insert(corresponding_register, self.num_locals);
        (latest_allocated_register, spill_instruction)
    }

    #[allow(unused)]
    fn number_of_distinct_registers(&self) -> usize {
        self.instructions
            .iter()
            .flat_map(|instruction| instruction.get_registers())
            .collect::<HashSet<_>>()
            .iter()
            .collect::<Vec<_>>()
            .len()
    }

    fn get_register_lifetime(
        instructions: &[Instruction],
    ) -> HashMap<VirtualRegister, (usize, usize)> {
        let mut result: HashMap<VirtualRegister, (usize, usize)> = HashMap::new();
        for (index, instruction) in instructions.iter().enumerate().rev() {
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
    fn visualize_multiplex_lifetime(&self) {
        // We want to draw a visualization where the x axis is the instruction index
        // and the y axis is the register index
        // If a register is alive at a given instruction, we will draw a -
        // If a register is dead at a given instruction, we will draw a " "

        let mut max_register_index = 0;
        for lifetime in self.multiplex_lifetime.values() {
            if let Some(last_instruction) = lifetime.last()
                && *last_instruction > max_register_index
            {
                max_register_index = *last_instruction;
            }
        }

        let mut lifetime_ordered_by_register_index =
            self.multiplex_lifetime.iter().collect::<Vec<_>>();
        lifetime_ordered_by_register_index.sort_by_key(|(register, _)| register.index);

        for (register, lifetime) in lifetime_ordered_by_register_index {
            // We want to turn the lifetimes into pairs of start and end
            print!("{:10} |", register.index);

            let lifetimes: Vec<(&usize, &usize)> =
                lifetime.iter().zip(lifetime.iter().skip(1)).collect();
            // Now that I have the grouped, I need to print " " until the start
            // and then print - for everytime they are alive.
            // But on the last one I print a | to show that it is an end
            // But then I need to print " " from the end to the next start
            // I need to print " " from the last end to the end of the program
            let mut current_instruction = 0;
            for (start, end) in lifetimes {
                for _ in current_instruction..*start {
                    print!(" ");
                }
                for _ in *start..*end - 1 {
                    print!("-");
                }
                print!("|");
                current_instruction = *end;
            }
            println!("\n");
        }
    }

    fn extend_token_range(&mut self, index: usize) {
        for (_, ir_range) in self.ir_range_to_token_range.iter_mut() {
            if index < ir_range.start {
                ir_range.start += 1;
            }
            if index < ir_range.end {
                ir_range.end += 1;
            }
        }
    }
}

// I'm not 100% sure what the output should be
// One way to try to think of this is that I am mapping
// between virtual registers and physical registers
// But I might need to change instructions as well
// Because, I need to spill some of these registers
// to memory, when I spill I need to update the instructions
// Given my setup, that will change not only instructions,
// but also the lifetimes of the registers.
// One way to get around this would be to have some instruction
// that just was this compound thing.
// Then the instructions don't change length
// Another answer would be to have a different register
// type so that way instructions when they get compiled
// would check the register type and do the right thing.

// Another thing I want to consider, is that I've found
// it useful to let instructions have some temporary registers.
// But I didn't express that in anyway here.
// I could make it so that all instructions have some field that says
// here are your extra registers. But that feels a bit gross.
// Doing it one the fly is nice, but then how can my register allocator know?

// I am currently doing register allocation on the fly as I compile to machine code
// But if I compiled to machine code, and then did register allocation, I would
// know all these things like auxiliary registers and stuff.

// I also need to account for argument registers
// Right now I am messing them up when it comes to call
// At this level here I'm not sure if I should be thinking about specific instructions
// or if I'm somehow able to abstract that away. The problem is that things like
// return value of course have to be allocated to a specific register.

// One option for this is just to remap between virtual registers and make sure
// that not too many are living at one time. Then my existing system should work.
// That doesn't answer the argument register problem. But that's why I think I can
// solve as well.

// TODO: I need some good register allocation tests
// need to figure out how best to do that.
