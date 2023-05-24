use std::{error::Error, collections::HashMap, mem, time::Instant};

use asm::{arm::{Register, Asm, Size, X0, X3, X20, X21, SP, X29, X30, X22, StpGenSelector, LdpGenSelector, X19, X1, StrImmGenSelector, LdrImmGenSelector}};
use mmap_rs::MmapOptions;


fn main() -> Result<(), Box<dyn Error>> {
    use_the_assembler()?;
    Ok(())
}

fn print_u32_hex_le(value: u32) {
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
fn branch_with_link(destination: *const u8) -> Asm {
    Asm::Bl {
        imm26: destination as i32
    }
}

fn branch_with_link_register(register: Register) -> Asm {
    Asm::Blr { rn: register }
}

fn breakpoint() -> Asm {
    Asm::Brk { imm16: 30 }
}


#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct Label {
    index: usize,
}

struct Lang {
    instructions: Vec<Asm>,
    label_index: usize,
    label_locations: HashMap<usize, usize>,
    labels: Vec<String>,
}

#[allow(unused)]
impl Lang {
    fn new() -> Self {
        Lang {
            instructions: vec![],
            label_locations: HashMap::new(),
            label_index: 0,
            labels: vec![],
        }
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
    fn store_pair(&mut self, reg1: Register, reg2: Register, destination: Register, offset: i32) {
        self.instructions.push(store_pair(reg1, reg2, destination, offset));
    }
    fn load_pair(&mut self, reg1: Register, reg2: Register, destination: Register, offset: i32) {
        self.instructions.push(load_pair(reg1, reg2, destination, offset));
    }
    fn add(&mut self, destination: Register, a: Register, b: Register) {
        self.instructions.push(add(destination, a, b));
    }
    fn sub(&mut self, destination: Register, a: Register, b: Register) {
        self.instructions.push(sub(destination, a, b));
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

    fn compile(&mut self) -> &Vec<Asm> {
        self.patch_labels();
        &self.instructions
    }
    fn call_rust_function(&mut self, register: Register, func: *const u8) {

        self.instructions.extend(
            load_64_bit_num(register, func as usize)
        );

        self.call(register)
    }

    fn call(&mut self, register: Register) {
        self.instructions.push(
            branch_with_link_register(register)
        );
    }

    fn patch_labels(&mut self) {
        for (instruction_index, instruction) in self.instructions.iter_mut().enumerate() {
            match instruction {
                Asm::BCond { imm19, cond: _ } => {
                    let label_index = *imm19 as usize;
                    let label_location = self.label_locations.get(&label_index);
                    match label_location {
                        Some(label_location) => {
                            let relative_position = *label_location as isize - instruction_index as isize;
                            *imm19 = relative_position as i32;
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

    fn mov_reg(&mut self, destination: Register, source: Register) {
        self.instructions.push(
            match (destination, source) {
                (SP, _) => {
                    mov_sp(destination, source)
                }
                (_, SP) => {
                    mov_sp(destination, source)
                }
                _ => {
                    mov_reg(destination, source)
                }
            }
        );

    }
}

fn load_64_bit_num(register: Register, num: usize) -> Vec<Asm> {
    let mut num = num;
    let mut result = vec![];
    
    result.push(Asm::Movz { sf: register.sf(), hw: 0, imm16: num as i32 & 0xffff, rd: register });
    num >>= 16;
    result.push(Asm::Movk { sf: register.sf(), hw: 0b01, imm16: num as i32 & 0xffff, rd: register });
    num >>= 16;
    result.push(Asm::Movk { sf: register.sf(), hw: 0b10, imm16: num as i32 & 0xffff, rd: register });
    num >>= 16;
    result.push(Asm::Movk { sf: register.sf(), hw: 0b11, imm16: num as i32 & 0xffff, rd: register });
    
    result
}


fn use_the_assembler() -> Result<(), Box<dyn Error>> {
    let mut lang = fib();

    let instructions = lang.compile();

    for instruction in instructions.iter() {
        print_u32_hex_le(instruction.encode());
    }

    let mut buffer = MmapOptions::new(MmapOptions::page_size())?.map_mut()?;
    let memory = &mut buffer[..];
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

    let f: fn(i64, u64) -> u64 = unsafe { mem::transmute(exec.as_ref().as_ptr()) };


    let n = 30;
    let time = Instant::now();

    let result1 = f(n, f as *const u8 as u64);
    println!("time {:?}", time.elapsed());
    let time = Instant::now();
    let result2 = fib_rust(n as usize);
    println!("time {:?}", time.elapsed());
    println!("{} {}", result1, result2);

    Ok(())
}

fn fib_rust(n: usize) -> usize {
    if n <= 1 {
        return n
    }
    return fib_rust(n - 1) + fib_rust(n - 2)
}

fn fib() -> Lang {
    let mut lang = Lang::new();
    
    lang.store_pair(X29, X30, SP, -4);
    lang.mov_reg(X29, SP);

    let base_case = lang.new_label("base_case");
    let recursive_case = lang.new_label("recursive_case");
    let return_label = lang.new_label("return");
    lang.mov(X21, 1);

    lang.compare(X0, X21);
    lang.jump_less_or_equal(base_case);
    lang.jump(recursive_case);

    lang.write_label(base_case);

    lang.jump(return_label);

    lang.write_label(recursive_case);
    
    lang.sub(X19, X0, X21);

    lang.instructions.push(
        Asm::StrImmGen {
            size: 0b11,
            imm9: 0, // not used
            rn: SP,
            rt: X0,
            imm12: 2,
            class_selector: StrImmGenSelector::UnsignedOffset,
        }
    );
    lang.mov_reg(X0, X19);
    
    lang.call(X1);
    lang.mov_reg(X22, X0);

    lang.instructions.push(
        Asm::LdrImmGen {
            size: 0b11,
            imm9: 0, // not used
            rn: SP,
            rt: X0,
            imm12: 2,
            class_selector: LdrImmGenSelector::UnsignedOffset,
        }
    );

    lang.sub(X19, X0, X21);
    lang.sub(X19, X19, X21);
    lang.mov_reg(X0, X19);

    lang.instructions.push(
        Asm::StrImmGen {
            size: 0b11,
            imm9: 0, // not used
            rn: SP,
            rt: X22,
            imm12: 2,
            class_selector: StrImmGenSelector::UnsignedOffset,
        }
    );
    
    lang.call(X1);

    lang.instructions.push(
        Asm::LdrImmGen {
            size: 0b11,
            imm9: 0, // not used
            rn: SP,
            rt: X22,
            imm12: 2,
            class_selector: LdrImmGenSelector::UnsignedOffset,
        }
    );
    lang.add(X0, X0, X22);


    lang.write_label(return_label);
    lang.load_pair(X29, X30, SP, 4);

    lang.ret();
    lang
}

fn countdown_codegen() -> Lang {
    let mut lang = Lang::new();

    let loop_start = lang.new_label("loop_start");
    let loop_exit = lang.new_label("loop_exit");

    lang.breakpoint();
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
    return num;
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
// 
// PS bits the ones with X need to be dealt with