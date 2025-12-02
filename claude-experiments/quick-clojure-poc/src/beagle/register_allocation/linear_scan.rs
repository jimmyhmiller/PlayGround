#![allow(dead_code)]
use std::collections::{BTreeMap, HashMap};

use crate::ir::{Instruction, Value, VirtualRegister};

pub struct LinearScan {
    pub lifetimes: HashMap<VirtualRegister, (usize, usize)>,
    pub instructions: Vec<Instruction>,
    // Needs to be a btreemap so I iterate in a defined order
    // so that things are deterministic
    pub allocated_registers: BTreeMap<VirtualRegister, VirtualRegister>,
    pub free_registers: Vec<VirtualRegister>,
    pub location: HashMap<VirtualRegister, usize>,
    pub stack_slot: usize,
    pub max_registers: usize,
}

fn physical(index: usize) -> VirtualRegister {
    VirtualRegister {
        argument: None,
        index,
        volatile: true,
        is_physical: true,
    }
}

impl LinearScan {
    pub fn new(instructions: Vec<Instruction>, num_locals: usize) -> Self {
        let lifetimes = Self::get_register_lifetime(&instructions);
        let physical_registers: Vec<VirtualRegister> = (19..=28).map(physical).collect();
        let max_registers = physical_registers.len();

        LinearScan {
            lifetimes,
            instructions,
            allocated_registers: BTreeMap::new(),
            free_registers: physical_registers,
            max_registers,
            location: HashMap::new(),
            stack_slot: num_locals,
        }
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

    //  LinearScanRegisterAllocation
    //   active <- {}
    //   foreach live interval i, in order of increasing start point
    //     ExpireOldIntervals(i)
    //     if length(active) == R then
    //       SpillAtInterval(i)
    //     else
    //       register[i] <- a register removed from pool of free registers
    //       add i to active, sorted by increasing end point

    // ExpireOldIntervals(i)
    //   foreach interval j in active, in order of increasing end point
    //     if endpoint[j] >= startpoint[i] then  # > が正しい気がする。
    //       return
    //     remove j from active
    //     add register[j] to pool of free registers

    // SpillAtInterval(i)
    //   spill <- last interval in active
    //   if endpoint[spill] > endpoint[i] then
    //     register[i] <- register[spill]
    //     location[spill] <- new stack location
    //     remove spill from active
    //     add i to active, sorted by increasing end point
    //   else
    //     location[i] <- new stack location

    pub fn allocate(&mut self) {
        let mut intervals = self
            .lifetimes
            .iter()
            .map(|(register, (start, end))| (*start, *end, *register))
            .collect::<Vec<_>>();

        intervals.sort_by_key(|(start, _, _)| *start);

        let mut active: Vec<(usize, usize, VirtualRegister)> = Vec::new();
        for i in intervals.iter() {
            let (start, end, register) = i;
            self.expire_old_intervals(*i, &mut active);
            if active.len() == self.max_registers {
                self.spill_at_interval(*start, *end, *register, &mut active);
            } else {
                if register.argument.is_some() {
                    let new_register = VirtualRegister {
                        argument: register.argument,
                        index: register.index,
                        volatile: false,
                        is_physical: true,
                    };
                    self.allocated_registers.insert(*register, new_register);
                } else {
                    let physical_register = self.free_registers.pop().unwrap();
                    self.allocated_registers
                        .insert(*register, physical_register);
                }
                active.push(*i);
                active.sort_by_key(|(end, _, _)| *end);
            }
        }
        self.replace_spilled_registers_with_spill();
        self.replace_virtual_with_allocated();
        self.replace_calls_with_call_with_save();
    }

    fn expire_old_intervals(
        &mut self,
        i: (usize, usize, VirtualRegister),
        active: &mut Vec<(usize, usize, VirtualRegister)>,
    ) {
        let mut active_copy = active.clone();
        active_copy.sort_by_key(|(_, end, _)| *end);
        for j in active_copy.iter() {
            let (_, end, _) = j;
            if *end >= i.0 {
                return;
            }
            active.retain(|x| x != j);
            let register_to_free = *self.allocated_registers.get(&j.2).unwrap();
            self.free_register(register_to_free);
        }
    }

    fn spill_at_interval(
        &mut self,
        _start: usize,
        end: usize,
        register: VirtualRegister,
        active: &mut Vec<(usize, usize, VirtualRegister)>,
    ) {
        let spill = *active.last().unwrap();
        if spill.1 > end {
            let physical_register = *self.allocated_registers.get(&spill.2).unwrap();
            self.allocated_registers.insert(register, physical_register);
            let stack_location = self.new_stack_location();
            assert!(!self.location.contains_key(&spill.2));
            self.location.insert(spill.2, stack_location);
            active.retain(|x| *x != spill);
            self.free_register(physical_register);
            active.sort_by_key(|(_, end, _)| *end);
        } else {
            assert!(!self.location.contains_key(&register));
            let stack_location = self.new_stack_location();
            self.location.insert(register, stack_location);
        }
    }

    fn free_register(&mut self, register: VirtualRegister) {
        if register.argument.is_some() {
            return;
        }
        self.free_registers.push(register);
    }

    fn new_stack_location(&mut self) -> usize {
        let result = self.stack_slot;
        self.stack_slot += 1;
        result
    }

    fn replace_spilled_registers_with_spill(&mut self) {
        for instruction in self.instructions.iter_mut() {
            for register in instruction.get_registers() {
                if let Some(stack_offset) = self.location.get(&register) {
                    instruction.replace_register(register, Value::Spill(register, *stack_offset));
                }
            }
        }
    }

    fn replace_virtual_with_allocated(&mut self) {
        for instruction in self.instructions.iter_mut() {
            for register in instruction.get_registers() {
                if let Some(physical_register) = self.allocated_registers.get(&register) {
                    instruction.replace_register(register, Value::Register(*physical_register));
                }
            }
        }
    }

    fn replace_calls_with_call_with_save(&mut self) {
        for (i, instruction) in self.instructions.iter_mut().enumerate() {
            if let Instruction::Call(dest, f, args, builtin) = &instruction.clone() {
                // println!("{}", instruction.pretty_print());
                // We want to get all ranges that are valid at this point
                // if they are not spilled (meaning there isn't an entry in location)
                // we want to add them to the list of saves
                let mut saves = Vec::new();
                for (original_register, (start, end)) in self.lifetimes.iter() {
                    if *start < i && *end > i + 1 && !self.location.contains_key(original_register)
                    {
                        let register = self.allocated_registers.get(original_register).unwrap();
                        // if register.index == 20 {
                        //     println!("20");
                        // }
                        if let Value::Register(dest) = dest
                            && dest == register
                        {
                            continue;
                        }
                        saves.push(Value::Register(*register));
                    }
                }
                *instruction = Instruction::CallWithSaves(*dest, *f, args.clone(), *builtin, saves);
            } else if let Instruction::Recurse(dest, args) = instruction {
                let mut saves = Vec::new();
                for (original_register, (start, end)) in self.lifetimes.iter() {
                    if *start < i && *end > i + 1 && !self.location.contains_key(original_register)
                    {
                        let register = self.allocated_registers.get(original_register).unwrap();
                        if let Value::Register(dest) = dest
                            && dest == register
                        {
                            continue;
                        }
                        saves.push(Value::Register(*register));
                    }
                }
                *instruction = Instruction::RecurseWithSaves(*dest, args.clone(), saves);
            }
        }
    }
}

#[test]
fn test_example() {
    use crate::{ir::Ir, pretty_print::PrettyPrint};
    let mut ir = Ir::new(0);
    let r0 = ir.assign_new(0);
    let r1 = ir.assign_new(0);
    let r2 = ir.assign_new(0);
    let r3 = ir.assign_new(0);
    let r4 = ir.assign_new(0);
    let r5 = ir.assign_new(0);
    let r6 = ir.assign_new(0);
    let r7 = ir.assign_new(0);
    let r8 = ir.assign_new(0);
    let r9 = ir.assign_new(0);
    let r10 = ir.assign_new(0);
    let add1 = ir.add_int(r1, r2);
    let add2 = ir.add_int(r3, r4);
    let add3 = ir.add_int(r5, r6);
    let add4 = ir.add_int(r7, r8);
    let add5 = ir.add_int(r9, r10);
    let add6 = ir.add_int(r0, r1);
    let add7 = ir.add_int(r2, r3);
    let add8 = ir.add_int(r4, r5);
    let add9 = ir.add_int(r6, r7);
    let add10 = ir.add_int(r8, r9);
    let add11 = ir.add_int(r10, r0);
    let add12 = ir.add_int(add1, add2);
    let add13 = ir.add_int(add3, add4);
    let add14 = ir.add_int(add5, add6);
    let add15 = ir.add_int(add7, add8);
    let add16 = ir.add_int(add9, add10);
    let add17 = ir.add_int(add11, add12);
    let add18 = ir.add_int(r0, r1);
    let add19 = ir.add_int(r2, r3);
    let add20 = ir.add_int(r4, r5);
    let add21 = ir.add_int(r6, r7);
    let add22 = ir.add_int(r8, r9);
    let add23 = ir.add_int(r10, r0);
    let add24 = ir.add_int(add1, add2);
    let add25 = ir.add_int(add24, add13);
    let add26 = ir.add_int(add25, add14);
    let add27 = ir.add_int(add26, add15);
    let add28 = ir.add_int(add27, add16);
    let add29 = ir.add_int(add28, add17);
    let add30 = ir.add_int(add29, add18);
    let add31 = ir.add_int(add30, add19);
    let add32 = ir.add_int(add31, add20);
    let add33 = ir.add_int(add32, add21);
    let add34 = ir.add_int(add33, add22);
    let add35 = ir.add_int(add34, add23);
    ir.ret(add35);

    let mut linear_scan = LinearScan::new(ir.instructions.clone(), 0);
    linear_scan.allocate();
    println!("{:#?}", linear_scan.allocated_registers);
    println!("=======");
    println!("{:#?}", linear_scan.location);

    println!("{}", linear_scan.instructions.pretty_print());
}
