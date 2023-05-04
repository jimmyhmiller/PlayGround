use std::{error::Error, collections::HashMap, mem};

use asm::arm::{Register, Asm, Size, X0, X1};
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

fn mov(destination: Register, input: u16) -> Asm {
    Asm::Movz {
        sf: destination.sf(),
        hw: 0,
        // TODO: Shouldn't this be a u16??
        imm16: input as u32,
        rd: destination,
    }
}

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
        rm: a,
        imm6: 0,
        rn: b,
    }
}

fn jump_equal(destination: u32) -> Asm {
    Asm::BCond {
        imm19: destination,
        cond: 0,
    }
}
fn jump_not_equal(destination: u32) -> Asm {
    Asm::BCond {
        imm19: destination,
        cond: 1,
    }
}
fn jump_greater_or_equal(destination: u32) -> Asm {
    Asm::BCond {
        imm19: destination,
        cond: 10,
    }
}
fn jump_less_than(destination: u32) -> Asm {
    Asm::BCond {
        imm19: destination,
        cond: 11,
    }
}
fn jump_greater(destination: u32) -> Asm {
    Asm::BCond {
        imm19: destination,
        cond: 12,
    }
}
fn jump_less_or_equal(destination: u32) -> Asm {
    Asm::BCond {
        imm19: destination,
        cond: 13,
    }
}
fn jump(destination: u32) -> Asm {
    Asm::BCond {
        imm19: destination,
        cond: 14,
    }
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
        self.instructions.push(mov(destination, input));
    }
    fn add(&mut self, destination: Register, a: Register, b: Register) {
        self.instructions.push(add(destination, a, b));
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

    fn patch_labels(&mut self) {
        for (instruction_index, instruction) in self.instructions.iter_mut().enumerate() {
            match instruction {
                Asm::BCond { imm19, cond: _ } => {
                    let label_index = *imm19 as usize;
                    let label_location = self.label_locations.get(&label_index).unwrap();
                    let relative_position = label_location - instruction_index;
                    *imm19 = relative_position as u32;
                }
                _ => {}
            }
        }
    }
}


fn use_the_assembler() -> Result<(), Box<dyn Error>> {
    let mut lang = Lang::new();

    let label = lang.new_label("label");

    // lang.breakpoint();
    lang.mov(X0, 120);
    lang.mov(X1, 100);
    lang.compare(X0, X1);
    lang.jump_equal(label);
    lang.add(X0, X1, X0);
    lang.write_label(label);
    lang.ret();

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

    let f: fn() -> u64 = unsafe { mem::transmute(exec.as_ref().as_ptr()) };

    println!("Result {}", f());

    Ok(())
}
