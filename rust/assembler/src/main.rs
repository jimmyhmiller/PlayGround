use mmap::MapOption;
use mmap::MemoryMap;
use std::collections::HashMap;
use std::convert::TryInto;
use std::io::{self, Write};
use std::mem;
use std::process::Command;

/// Notes:
// All my jumps are rel32
// I should probably expand labels or some way of referencing things outside
// just jumps. Labelling positions is generally useful.

#[derive(Debug, Eq, PartialEq, Copy, Clone)]
#[allow(dead_code)]
enum Register {
    RAX,
    RBX,
    RCX,
    RDX,
    RBP,
    RSP,
    RSI,
    RDI,
    RIP,
    // These require some different encoding
    // Need to understand these better
    R8,
    R9,
    R10,
    R11,
    R12,
    R13,
    R14,
    R15,
}

impl Register {
    fn index(&self) -> u8 {
        match self {
            Register::RAX => 0,
            Register::RCX => 1,
            Register::RDX => 2,
            Register::RBX => 3,
            Register::RSP => 4,
            Register::RBP => 5,
            Register::RSI => 6,
            Register::RDI => 7,
            Register::R8 => 0,
            Register::R9 => 1,
            Register::R10 => 2,
            Register::R11 => 3,
            Register::R12 => 4,
            Register::R13 => 5,
            Register::R14 => 6,
            Register::R15 => 7,
            Register::RIP => 5,
        }
    }
    fn extended(&self) -> bool {
        !matches!(self,
            Register::RAX |
            Register::RCX |
            Register::RDX |
            Register::RBX |
            Register::RSP |
            Register::RBP |
            Register::RSI |
            Register::RDI |
            Register::RIP)
    }
}

#[derive(Debug, Eq, PartialEq, Copy, Clone)]
#[allow(dead_code)]
enum Val {
    Int(i32),
    U8(u8),
    Reg(Register),
    AddrReg(Register),
    AddrRegOffset(Register, i32),
}

#[derive(Debug, Eq, PartialEq)]
#[allow(dead_code)]
struct Label {
    location: Option<u32>,
    // Don't love this vector
    called: Vec<u32>,
}

impl Label {
    fn set_location(&mut self, emitter: &mut Emitter) {
        let loc = emitter.instruction_index;
        if self.location.is_some() {
            panic!("Setting Location twice {:?} at {}", self, loc);
        }
        self.location = Some(loc.try_into().unwrap());
        self.patch(emitter);
    }

    fn patch(&mut self, emitter: &mut Emitter) {
        if self.location.is_none() {
            panic!("Patching without a location {:?}", self);
        }
        let label_loc = self.location.unwrap();
        for location in self.called.iter() {
            // Need to emit it back 4 from where we are relative to.
            // It is relative to end of instruction
            emitter.emit_i32_loc(*location - 4, (label_loc - location).try_into().unwrap());
        }
    }

    fn add_called(&mut self, emitter: &mut Emitter) {
        let loc = (emitter.instruction_index + 4) as u32;
        if let Some(location) = self.location {
            emitter.imm(location as i32 - loc as i32)
        } else {
            emitter.imm(0);
        }
        self.called.push(loc);
    }
}

#[derive(Debug, Eq, PartialEq)]
#[allow(dead_code)]
struct Emitter<'a> {
    // Going to assume a page for the moment
    // Obviously not good enough forever
    memory: &'a mut [u8; 4096],
    instruction_index: usize,
    // Is there a better, lighter-weight way?
    labels: HashMap<String, Label>,
    symbol_index: usize,
}

#[derive(Debug, Eq, PartialEq)]
#[allow(dead_code)]
struct Rex {
    w: bool,
    r: bool,
    b: bool,
    x: bool,
}

const REX_W: Rex = Rex {
    w: true,
    r: false,
    b: false,
    x: false,
};

impl Rex {
    fn extend_reg(mut self, reg: Register) -> Self {
        self.r = reg.extended(); 
        self
    }

    fn extend_rm(mut self, reg: Register) -> Self {
        self.x = reg.extended(); 
        self
    }
}

#[derive(Debug, Eq, PartialEq)]
#[allow(dead_code)]
enum Mode {
    M11,
    M10,
    M01,
    M00,
}

impl Mode {
    fn into_bytes(self) -> u8 {
        match self {
            Mode::M11 => 0b11000000,
            Mode::M10 => 0b10000000,
            Mode::M01 => 0b01000000,
            Mode::M00 => 0b00000000,
        }
    }
}

#[derive(Debug, Eq, PartialEq)]
#[allow(dead_code)]
struct ModRM {
    mode: Mode,
    reg: Val,
    rm: Val,
}

#[derive(Debug, Eq, PartialEq)]
#[allow(dead_code)]
enum Scale {
    S11,
    S10,
    S01,
    S00,
}

impl Scale {
    fn into_bytes(self) -> u8 {
        match self {
            Scale::S11 => 0b11000000,
            Scale::S10 => 0b10000000,
            Scale::S01 => 0b01000000,
            Scale::S00 => 0b00000000,
        }
    }
}

// Probably use this in modrm too.
#[derive(Debug, Eq, PartialEq, Clone, Copy)]
#[allow(dead_code)]
enum RegOrU3 {
    Reg(Register),
    U3(u8),
}

impl RegOrU3 {
    fn into_bytes(self) -> u8 {
        match self {
            RegOrU3::U3(u8) => u8,
            RegOrU3::Reg(reg) => reg.index(),
        }
    }
}

#[derive(Debug, Eq, PartialEq)]
#[allow(dead_code)]
struct Sib {
    scale: Scale,
    index: RegOrU3,
    base: RegOrU3,
}

#[derive(Debug, Eq, PartialEq)]
#[allow(dead_code)]
struct Instruction {
    opcode: u8,
    rex: Option<Rex>,
    modrm: Option<ModRM>,
    imm32: Option<u32>,
}

#[allow(dead_code)]
impl Instruction {
    fn encode(self, bytes: &mut [u8]) -> usize {
        let mut i = 0;
        if self.rex.is_some() {
            let rex = self.rex.unwrap();
            bytes[i] = 0b0100 << 4
                | (rex.w as u8) << 3
                | (rex.r as u8) << 2
                | (rex.b as u8) << 1
                | (rex.x as u8);
            i += 1;
        }
        bytes[i] = self.opcode;
        i += 1;
        if self.modrm.is_some() {
            let modrm = self.modrm.unwrap();
            match (modrm.reg, modrm.rm) {
                (Val::Reg(r1), Val::Reg(r2)) => {
                    bytes[i] = modrm.mode.into_bytes() | (r1.index() << 3) | r2.index();
                    i += 1;
                }
                (Val::U8(u), Val::Reg(r2)) => {
                    bytes[i] = modrm.mode.into_bytes() | (u << 3) | r2.index();
                    i += 1;
                }
                (Val::Reg(r1), Val::U8(u)) => {
                    bytes[i] = modrm.mode.into_bytes() | (r1.index() << 3) | u;
                    i += 1;
                }
                x => unimplemented!("Didn't handle case {:?}", x),
            }
        }
        if self.imm32.is_some() {
            let imm32 = self.imm32.unwrap();
            let imm32_bytes: [u8; 4] = imm32.to_le_bytes();
            for byte in &imm32_bytes {
                bytes[i] = *byte;
                i += 1;
            }
        }
        i
    }
}

#[allow(dead_code)]
impl Emitter<'_> {
    fn new_symbol(&mut self, prefix: &str) -> String {
        self.symbol_index += 1;
        return format!("{}_{}", prefix, self.symbol_index);
    }

    fn label(&mut self, name: String) {
        // This is terrible and I'm sure there is a better way to make the
        // type checker happy.
        if self.labels.contains_key(&name) {
            let mut label = self.labels.remove_entry(&name).unwrap();
            label.1.set_location(self);
            self.labels.insert(label.0, label.1);
        } else {
            let mut label = Label {
                called: vec![],
                location: None,
            };
            label.set_location(self);
            self.labels.insert(name, label);
        }
    }

    fn add_label_patch(&mut self, label: String) {
        self.instruction_index -= 4;
        self.add_label(label);
    }

    fn label_in_reg(&mut self, reg: Val, label: String) -> Val {
        if let Val::Reg(reg) = reg {
            // 0 here is a placeholder
            self.mov(Val::Reg(reg), RIP_PLACEHOLDER);
            // Move the instruction pointer back
            // 4 bytes (32 bit) so we patch the correct location;
            self.instruction_index -= 4;
            self.add_label(label);
            Val::Reg(reg)
        } else {
            unimplemented!("Must be a register {:?}", reg);
        }
    }

    fn add_label(&mut self, name: String) {
        if self.labels.contains_key(&name) {
            let mut label = self.labels.remove_entry(&name).unwrap();
            label.1.add_called(self);
            self.labels.insert(label.0, label.1);
        } else {
            let mut label = Label {
                called: vec![],
                location: None,
            };
            label.add_called(self);
            self.labels.insert(name, label);
        }
    }

    fn jump_label(&mut self, name: String) {
        self.opcode(0xE9);
        self.add_label(name);
    }

    fn jump_label_equal(&mut self, name: String) {
        self.opcode(0x0F);
        self.opcode(0x84);
        self.add_label(name);
    }

    fn jump_label_not_equal(&mut self, name: String) {
        self.opcode(0x0F);
        self.opcode(0x85);
        self.add_label(name);
    }

    fn call_label(&mut self, name: String) {
        self.opcode(0xE8);
        self.add_label(name);
    }

    fn call(&mut self, reg: Val) {
        match reg {
            Val::Reg(addr) => {
                self.opcode(0xff);
                self.modrm(ModRM {
                    mode: Mode::M11,
                    reg: Val::U8(2),
                    rm: Val::Reg(addr),
                });
            }
            Val::AddrReg(addr) => {
                self.opcode(0xff);
                self.modrm(ModRM {
                    mode: Mode::M00,
                    reg: Val::U8(2),
                    rm: Val::Reg(addr),
                });
            }
            Val::AddrRegOffset(addr, offset) => {
                self.opcode(0xff);
                self.modrm(ModRM {
                    mode: Mode::M00,
                    reg: Val::U8(2),
                    rm: Val::Reg(addr),
                });
                self.imm(offset);
            }
            x => unimplemented!("Didn't handle case {:?}", x),
        }
    }

    fn emit(&mut self, bytes: &[u8]) {
        for (i, byte) in bytes.iter().enumerate() {
            self.memory[self.instruction_index + i] = *byte;
            // println!("{:#04x}", self.memory[self.instruction_index + i]);
        }
        self.instruction_index += bytes.len();
    }

    fn emit_u8(&mut self, byte: u8) {
        self.memory[self.instruction_index] = byte;
        self.instruction_index += 1;
    }

    fn emit_i32_loc(&mut self, loc: u32, rel: i32) {
        for (i, byte) in rel.to_le_bytes().iter().enumerate() {
            self.memory[loc as usize + i] = *byte;
        }
    }

    fn opcode(&mut self, opcode: u8) {
        self.emit_u8(opcode);
    }

    fn rex(&mut self, rex: Rex) {
        self.emit_u8(
            0b0100 << 4
                | (rex.w as u8) << 3
                | (rex.r as u8) << 2
                | (rex.b as u8) << 1
                | (rex.x as u8),
        );
    }

    fn modrm(&mut self, modrm: ModRM) {
        match (modrm.reg, modrm.rm) {
            (Val::Reg(r1), Val::Reg(r2)) => {
                self.emit_u8(modrm.mode.into_bytes() | (r1.index() << 3) | r2.index());
            }
            (Val::U8(u), Val::Reg(r2)) => {
                self.emit_u8(modrm.mode.into_bytes() | (u << 3) | r2.index());
            }
            (Val::Reg(r1), Val::U8(u)) => {
                self.emit_u8(modrm.mode.into_bytes() | (r1.index() << 3) | u);
            }
            x => unimplemented!("Didn't handle case {:?}", x),
        }
    }

    fn sib(&mut self, sib: Sib) {
        self.emit_u8(sib.scale.into_bytes() | sib.index.into_bytes() << 3 | sib.base.into_bytes());
    }

    fn imm(&mut self, i: i32) {
        self.emit(&i.to_le_bytes());
    }

    fn imm_usize(&mut self, i: usize) {
        self.emit(&i.to_le_bytes());
    }
    fn imm_i64(&mut self, i: i64) {
        self.emit(&i.to_le_bytes());
    }

    fn ret(&mut self) {
        self.opcode(0xC3);
    }

    fn leave(&mut self) {
        self.opcode(0xC9);
    }

    // All these instructions could be encoded in a much better way.

    // I could make it so there are multiple moves depending on type

    fn mov(&mut self, val1: Val, val2: Val) {
        // I might have gotten some of the src vs dst wrong :(

        match (val1, val2) {
            (Val::Reg(dst), Val::Int(src)) => {
                // Need to deal with extended registers
                self.rex(REX_W.extend_rm(dst));
                self.opcode(0xC7);
                self.modrm(ModRM {
                    mode: Mode::M11,
                    reg: Val::U8(0),
                    rm: Val::Reg(dst),
                });
                self.imm(src);
            }
            (Val::Reg(dst), Val::AddrRegOffset(Register::RIP, offset)) => {
                self.rex(REX_W.extend_reg(dst));
                self.opcode(0x8b);
                self.modrm(ModRM {
                    mode: Mode::M00,
                    reg: Val::Reg(dst),
                    // Need to rethink what modrm takes, because it doesn't care about addrReg I don't think
                    // I am just turning it into reg here becasue then the code will work.
                    rm: Val::Reg(Register::RIP),
                });
                self.imm(offset);
            }

            (Val::Reg(dst), Val::AddrReg(Register::RSP)) => {
                self.rex(REX_W.extend_rm(dst));
                self.opcode(0x8b);
                self.modrm(ModRM {
                    mode: Mode::M00,
                    reg: Val::U8(4),
                    // Need to rethink what modrm takes, because it doesn't care about addrReg I don't think
                    // I am just turning it into reg here becasue then the code will work.
                    rm: Val::Reg(dst),
                });
                // This seems like a completely special case. But is there a way for me to see it isn't?
                // Maybe it jsut is.
                self.sib(Sib {
                    scale: Scale::S00,
                    index: RegOrU3::Reg(Register::RSP),
                    base: RegOrU3::Reg(Register::RSP),
                });
            }
            (Val::Reg(dst), Val::AddrReg(src)) => {
                self.rex(REX_W.extend_reg(dst).extend_rm(src));
                self.opcode(0x8b);
                self.modrm(ModRM {
                    mode: Mode::M00,
                    reg: Val::Reg(dst),
                    rm: Val::Reg(src),
                });
            }
            (Val::Reg(dst), Val::AddrRegOffset(src, offset)) => {
                self.rex(REX_W.extend_reg(dst).extend_rm(src));
                self.opcode(0x8b);
                self.modrm(ModRM {
                    mode: Mode::M10,
                    reg: Val::Reg(dst),
                    rm: Val::Reg(src),
                });
                assert!(offset != 0);
                self.imm(offset);
            }
            (Val::AddrReg(Register::RSP), Val::Reg(src)) => {
                self.rex(REX_W.extend_rm(src));
                self.opcode(0x89);
                self.modrm(ModRM {
                    mode: Mode::M00,
                    reg: Val::U8(4),
                    rm: Val::Reg(src),
                });
                // Seems like special case, is it?
                self.sib(Sib {
                    scale: Scale::S00,
                    index: RegOrU3::Reg(Register::RSP),
                    base: RegOrU3::Reg(Register::RSP),
                });
            }
            (Val::AddrReg(dst), Val::Reg(src)) => {
                self.rex(REX_W.extend_reg(dst).extend_rm(src));
                self.opcode(0x89);
                self.modrm(ModRM {
                    mode: Mode::M00,
                    reg: Val::Reg(src),
                    rm: Val::Reg(dst),
                });
            }
            (Val::AddrRegOffset(dst, offset), Val::Reg(src)) => {
                self.rex(REX_W.extend_reg(dst).extend_rm(src));
                self.opcode(0x89);
                self.modrm(ModRM {
                    mode: Mode::M10,
                    reg: Val::Reg(src),
                    rm: Val::Reg(dst),
                });
                assert!(offset != 0);
                self.imm(offset);
            }
            (Val::Reg(dst), Val::Reg(src)) => {
                self.rex(REX_W.extend_reg(src).extend_rm(dst));
                self.opcode(0x89);
                self.modrm(ModRM {
                    mode: Mode::M11,
                    reg: Val::Reg(src),
                    rm: Val::Reg(dst),
                });
            }
            _ => panic!("Mov not implemented for that combination"),
        }
    }

    fn lea(&mut self, dst: Val, mem_src: Val) {
        match (dst, mem_src) {
            (Val::Reg(dst), Val::AddrReg(src)) => {
                self.rex(REX_W.extend_reg(dst).extend_rm(src));
                self.opcode(0x8D);
                self.modrm(ModRM {
                    mode: Mode::M00,
                    reg: Val::Reg(dst),
                    rm: Val::Reg(src),
                });
            }
            _ => panic!("Mov not implemented for that combination"),
        }
    }

    fn add(&mut self, val1: Val, val2: Val) {
        match (val1, val2) {
            (Val::Reg(Register::RAX), Val::Int(src)) => {
                self.rex(REX_W);
                self.opcode(0x05);
                self.imm(src);
            }
            (Val::Reg(dst), Val::Int(src)) => {
                self.rex(REX_W.extend_rm(dst));
                self.opcode(0x81);
                self.modrm(ModRM {
                    mode: Mode::M11,
                    reg: Val::U8(0),
                    rm: Val::Reg(dst),
                });
                self.imm(src);
            }
            (Val::Reg(dst), Val::Reg(src)) => {
                self.rex(REX_W.extend_reg(dst).extend_rm(src));
                self.opcode(0x03);
                self.modrm(ModRM {
                    mode: Mode::M11,
                    reg: Val::Reg(dst),
                    rm: Val::Reg(src),
                });
            }
            _ => panic!("add not implemented for that combination"),
        }
    }

    fn sub(&mut self, val1: Val, val2: Val) {
        match (val1, val2) {
            (Val::Reg(Register::RAX), Val::Int(src)) => {
                self.rex(REX_W);
                self.opcode(0x2D);
                self.imm(src);
            }
            (Val::Reg(dst), Val::Int(src)) => {
                self.rex(REX_W.extend_rm(dst));
                self.opcode(0x81);
                self.modrm(ModRM {
                    mode: Mode::M11,
                    reg: Val::U8(5),
                    rm: Val::Reg(dst),
                });
                self.imm(src);
            }
            (Val::Reg(dst), Val::Reg(src)) => {
                self.rex(REX_W.extend_reg(dst).extend_rm(src));
                self.opcode(0x2B);
                self.modrm(ModRM {
                    mode: Mode::M11,
                    reg: Val::Reg(dst),
                    rm: Val::Reg(src),
                });
            }
            _ => panic!("add not implemented for that combination"),
        }
    }

    fn imul(&mut self, val1: Val, val2: Val) {
        match (val1, val2) {
            (Val::Reg(Register::RAX), Val::Reg(src)) => {
                self.rex(REX_W.extend_rm(src));
                self.opcode(0xF7);
                self.modrm(ModRM {
                    mode: Mode::M11,
                    reg: Val::U8(5),
                    rm: Val::Reg(src),
                });
            }
            (Val::Reg(dst), Val::Int(src)) => {
                self.rex(REX_W.extend_reg(dst).extend_rm(dst));
                self.opcode(0x69);
                self.modrm(ModRM {
                    mode: Mode::M11,
                    reg: Val::Reg(dst),
                    rm: Val::Reg(dst),
                });
                self.imm(src);
            }
            (Val::Reg(dst), Val::Reg(src)) => {
                self.rex(REX_W.extend_reg(dst).extend_rm(src));
                self.opcode(0x69);
                self.modrm(ModRM {
                    mode: Mode::M11,
                    reg: Val::Reg(dst),
                    rm: Val::Reg(src),
                });
                self.imm(1);
            }
            _ => panic!("add not implemented for that combination"),
        }
    }

    // Unsigned divide RDX:RAX by r/m64, with result stored in RAX ← Quotient, RDX ← Remainder.
    // Not sure what this RDX:RAX means. Is this a fake 128 bit number?
    // will have to learn more, but so far when using it I zero out RDX
    fn div(&mut self, val: Val) {
        match val {
            Val::Reg(divisor) => {
                self.rex(REX_W.extend_rm(divisor));
                self.opcode(0xF7);
                self.modrm(ModRM {
                    mode: Mode::M11,
                    reg: Val::U8(6),
                    rm: Val::Reg(divisor),
                });
            }
            _ => panic!("add not implemented for that combination"),
        }
    }

    fn inc(&mut self, val: Val) {
        match val {
            Val::Reg(reg) => {
                self.rex(REX_W.extend_rm(reg));
                self.opcode(0xFF);
                self.modrm(ModRM {
                    mode: Mode::M11,
                    reg: Val::U8(0),
                    rm: Val::Reg(reg),
                });
            }
            _ => panic!("add not implemented for that combination"),
        }
    }
    fn dec(&mut self, val: Val) {
        match val {
            Val::Reg(reg) => {
                self.rex(REX_W.extend_rm(reg));
                self.opcode(0xFF);
                self.modrm(ModRM {
                    mode: Mode::M11,
                    reg: Val::U8(1),
                    rm: Val::Reg(reg),
                });
            }
            _ => panic!("add not implemented for that combination"),
        }
    }

    fn and(&mut self, val1: Val, val2: Val) {
        match (val1, val2) {
            (Val::Reg(Register::RAX), Val::Int(src)) => {
                self.rex(REX_W);
                self.opcode(0x25);
                self.imm(src);
            }
            (Val::Reg(dst), Val::Int(src)) => {
                self.rex(REX_W.extend_rm(dst));
                self.opcode(0x81);
                self.modrm(ModRM {
                    mode: Mode::M11,
                    reg: Val::U8(4),
                    rm: Val::Reg(dst),
                });
                self.imm(src);
            }
            (Val::Reg(dst), Val::Reg(src)) => {
                self.rex(REX_W.extend_reg(dst).extend_rm(src));
                self.opcode(0x23);
                self.modrm(ModRM {
                    mode: Mode::M11,
                    reg: Val::Reg(dst),
                    rm: Val::Reg(src),
                });
            }
            _ => panic!("add not implemented for that combination"),
        }
    }

    fn or(&mut self, val1: Val, val2: Val) {
        match (val1, val2) {
            (Val::Reg(Register::RAX), Val::Int(src)) => {
                self.rex(REX_W);
                self.opcode(0x0D);
                self.imm(src);
            }
            (Val::Reg(dst), Val::Int(src)) => {
                self.rex(REX_W.extend_rm(dst));
                self.opcode(0x81);
                self.modrm(ModRM {
                    mode: Mode::M11,
                    reg: Val::U8(1),
                    rm: Val::Reg(dst),
                });
                self.imm(src);
            }
            (Val::Reg(dst), Val::Reg(src)) => {
                self.rex(REX_W.extend_reg(dst).extend_rm(src));
                self.opcode(0x0B);
                self.modrm(ModRM {
                    mode: Mode::M11,
                    reg: Val::Reg(dst),
                    rm: Val::Reg(src),
                });
            }
            _ => panic!("add not implemented for that combination"),
        }
    }

    fn xor(&mut self, val1: Val, val2: Val) {
        match (val1, val2) {
            (Val::Reg(Register::RAX), Val::Int(src)) => {
                self.rex(REX_W);
                self.opcode(0x35);
                self.imm(src);
            }
            (Val::Reg(dst), Val::Int(src)) => {
                self.rex(REX_W.extend_rm(dst));
                self.opcode(0x81);
                self.modrm(ModRM {
                    mode: Mode::M11,
                    reg: Val::U8(6),
                    rm: Val::Reg(dst),
                });
                self.imm(src);
            }
            (Val::Reg(dst), Val::Reg(src)) => {
                self.rex(REX_W.extend_reg(dst).extend_rm(src));
                self.opcode(0x33);
                self.modrm(ModRM {
                    mode: Mode::M11,
                    reg: Val::Reg(dst),
                    rm: Val::Reg(src),
                });
            }
            _ => panic!("add not implemented for that combination"),
        }
    }

    fn cmp(&mut self, val1: Val, val2: Val) {
        match (val1, val2) {
            (Val::Reg(Register::RAX), Val::Int(src)) => {
                self.rex(REX_W);
                self.opcode(0x3D);
                self.imm(src);
            }
            (Val::Reg(dst), Val::Int(src)) => {
                self.rex(REX_W.extend_rm(dst));
                self.opcode(0x81);
                self.modrm(ModRM {
                    mode: Mode::M11,
                    reg: Val::U8(7),
                    rm: Val::Reg(dst),
                });
                self.imm(src);
            }
            (Val::Reg(reg1), Val::Reg(reg2)) => {
                self.rex(REX_W.extend_reg(reg1).extend_rm(reg2));
                self.opcode(0x39);
                self.modrm(ModRM {
                    mode: Mode::M11,
                    reg: Val::Reg(reg1),
                    rm: Val::Reg(reg2),
                });
            }
            _ => panic!("cmp not implemented for that combination"),
        }
    }

    // test(rax, rax) = cmp(rax, 0)
    fn test(&mut self, val1: Val, val2: Val) {
        match (val1, val2) {
            (Val::Reg(Register::RAX), Val::Int(src)) => {
                self.rex(REX_W);
                self.opcode(0xA9);
                self.imm(src);
            }
            (Val::Reg(dst), Val::Int(src)) => {
                self.rex(REX_W.extend_rm(dst));
                self.opcode(0xF7);
                self.modrm(ModRM {
                    mode: Mode::M11,
                    reg: Val::U8(0),
                    rm: Val::Reg(dst),
                });
                self.imm(src);
            }
            (Val::Reg(reg1), Val::Reg(reg2)) => {
                self.rex(REX_W.extend_reg(reg1).extend_rm(reg2));
                self.opcode(0x85);
                self.modrm(ModRM {
                    mode: Mode::M11,
                    reg: Val::Reg(reg1),
                    rm: Val::Reg(reg2),
                });
            }
            _ => panic!("test not implemented for that combination"),
        }
    }

    fn push(&mut self, val: Val) {
        match val {
            Val::Int(i) => {
                self.emit(&[0x68]);
                self.imm(i);
            }
            Val::Reg(reg) => self.emit(&[0x50 | reg.index()]),
            _ => panic!("push not implemented for that combination"),
        }
    }
    fn pop(&mut self, val: Val) {
        match val {
            Val::Int(i) => {
                self.emit(&[0x8f]);
                self.imm(i);
            }
            Val::Reg(reg) =>  {
                if reg.extended() {
                    self.rex(REX_W.extend_rm(reg));
                }
                self.emit(&[0x58 | reg.index()])
            },
            _ => panic!("push not implemented for that combination"),
        }
    }
}

const FUNCTION_REGISTERS: [Val; 3] = [RDI, RSI, RDX];


// Think about closures.
// Think about debugging features


#[allow(dead_code)]
#[derive(Clone, Debug)]
enum Lang {
    Func {
        name: String,
        args: Vec<String>,
        body: Vec<Lang>,
    },
    FFI {
        name: String,
        args: Vec<String>,
        ptr: *const (),
    },
    Add(Box<Lang>, Box<Lang>),
    Sub(Box<Lang>, Box<Lang>),
    Mul(Box<Lang>, Box<Lang>),
    // TODO: Div
    If(Box<Lang>, Box<Lang>, Box<Lang>),
    Equal(Box<Lang>, Box<Lang>),
    NotEqual(Box<Lang>, Box<Lang>),
    Int(i32),
    Call0(String),
    Call1(String, Box<Lang>),
    Call2(String, Box<Lang>, Box<Lang>),
    Call3(String, Box<Lang>, Box<Lang>, Box<Lang>),
    While(Box<Lang>, Box<Lang>),
    True,
    False,
    Do(Vec<Lang>),
    Let(String, Box<Lang>),
    Set(String, Box<Lang>),
    Variable(String),
    Return(Box<Lang>),
    Get(Box<Lang>),
    Store(Box<Lang>, Box<Lang>),
}

// Not the prettiest.
// But this whole compiler is about capability not beauty
struct EnvData {
    val: Val,
    is_foreign: bool,
}

impl Lang {


    // This isn't perfect and probably allocates more than it needs to
    // But I wrote this super fast thanks to copilot.
    fn find_all_lets(&self) -> Vec<(String, Lang)> {
        match self {
            // This probably doesn't make sense right now because nested functions aren't supported
            // but it also might never make sense because my let analysis shouldn't work like this with nested functions
            Lang::Func { name: _, args, body } => {
                let mut lets = vec![];
                for arg in args {
                    lets.push((arg.to_string(), Lang::Variable(arg.to_string())));
                }
                for stmt in body {
                    let mut stmt_lets = stmt.find_all_lets();
                    lets.append(&mut stmt_lets);
                }
                lets
            }
            Lang::Add(lhs, rhs) => {
                let mut lets = lhs.find_all_lets();
                let mut rhs_lets = rhs.find_all_lets();
                lets.append(&mut rhs_lets);
                lets
            }
            Lang::Sub(lhs, rhs) => {
                let mut lets = lhs.find_all_lets();
                let mut rhs_lets = rhs.find_all_lets();
                lets.append(&mut rhs_lets);
                lets
            }
            Lang::Mul(lhs, rhs) => {
                let mut lets = lhs.find_all_lets();
                let mut rhs_lets = rhs.find_all_lets();
                lets.append(&mut rhs_lets);
                lets
            }
            Lang::If(cond, then, else_) => {
                let mut lets = cond.find_all_lets();
                let mut then_lets = then.find_all_lets();
                let mut else_lets = else_.find_all_lets();
                lets.append(&mut then_lets);
                lets.append(&mut else_lets);
                lets
            }
            Lang::Equal(lhs, rhs) => {
                let mut lets = lhs.find_all_lets();
                let mut rhs_lets = rhs.find_all_lets();
                lets.append(&mut rhs_lets);
                lets
            }
            Lang::NotEqual(lhs, rhs) => {
                let mut lets = lhs.find_all_lets();
                let mut rhs_lets = rhs.find_all_lets();
                lets.append(&mut rhs_lets);
                lets
            }
            Lang::Int(_) => vec![],
            Lang::Call0(_name) => vec![],
            Lang::Call1(_name, arg) => {
                arg.find_all_lets()
            }
            Lang::Call2(_name, arg1, arg2) => {
                let mut lets = arg1.find_all_lets();
                let mut arg2_lets = arg2.find_all_lets();
                lets.append(&mut arg2_lets);
                lets
            }
            Lang::Call3(_name, arg1, arg2, arg3) => {
                let mut lets = arg1.find_all_lets();
                let mut arg2_lets = arg2.find_all_lets();
                let mut arg3_lets = arg3.find_all_lets();
                lets.append(&mut arg2_lets);
                lets.append(&mut arg3_lets);
                lets
            }
            Lang::While(cond, body) => {
                let mut lets = cond.find_all_lets();
                let mut body_lets = body.find_all_lets();
                lets.append(&mut body_lets);
                lets
            }
            Lang::True => vec![],
            Lang::False => vec![],
            Lang::Do(stmts) => {
                let mut lets = vec![];
                for stmt in stmts {
                    let mut stmt_lets = stmt.find_all_lets();
                    lets.append(&mut stmt_lets);
                }
                lets
            }
            Lang::Let(name, val) => {
                let mut lets = val.find_all_lets();
                lets.append(&mut vec![(name.clone(), *val.clone())]);
                lets
            }
            Lang::Set(_name, val) => {
                val.find_all_lets()
            }
            Lang::Variable(_name) => vec![],
            Lang::Return(val) => {
                val.find_all_lets()
            }
            Lang::Get(val) => {
                val.find_all_lets()
            }
            Lang::Store(val, ptr) => {
                let mut lets = val.find_all_lets();
                let mut ptr_lets = ptr.find_all_lets();
                lets.append(&mut ptr_lets);
                lets
            }
            Lang::FFI { name: _, args: _, ptr: _ } => vec![],
        } 
    }

    #[allow(dead_code, unused_variables)]
    fn compile(self, env: &mut HashMap<String, EnvData>, emitter: &mut Emitter) {
        // let mut stack : Vec<Lang> = vec![];
        // stack.push(self.clone());
        // while let Some(expr) = stack.pop() {
        match self {
            // If I didn't want to push I would need a register allocator
            Lang::Int(i) => emitter.push(Val::Int(i)),
            Lang::True => emitter.push(Val::Int(1)),
            Lang::False => emitter.push(Val::Int(0)),
            Lang::Add(a, b) => {
                (*b).compile(env, emitter);
                (*a).compile(env, emitter);
                // We need to think about registers :(
                emitter.pop(RAX);
                emitter.pop(RBX);
                emitter.add(RAX, RBX);
                emitter.push(RAX);
            }
            Lang::Sub(a, b) => {
                (*b).compile(env, emitter);
                (*a).compile(env, emitter);
                // We need to think about registers :(
                emitter.pop(RAX);
                emitter.pop(RBX);
                emitter.sub(RAX, RBX);
                emitter.push(RAX);
            }
            Lang::Mul(a, b) => {
                (*b).compile(env, emitter);
                (*a).compile(env, emitter);
                // We need to think about registers :(
                emitter.pop(RAX);
                emitter.pop(RBX);
                emitter.imul(RAX, RBX);
                emitter.push(RAX);
            }
            Lang::If(pred, t_branch, f_branch) => {
                (*pred).compile(env, emitter);
                let if_symbol = emitter.new_symbol("if");
                let else_symbol = emitter.new_symbol("if");
                emitter.pop(RAX);
                emitter.cmp(RAX, Val::Int(1));
                emitter.jump_label_not_equal(if_symbol.clone());
                (*t_branch).compile(env, emitter);
                emitter.jump_label(else_symbol.clone());
                emitter.label(if_symbol);
                (*f_branch).compile(env, emitter);
                emitter.label(else_symbol);
            }
            Lang::Equal(a, b) => {
                // Terrible way of doing this,
                // But should work.
                let eq_symbol = emitter.new_symbol("eq");
                (*b).compile(env, emitter);
                (*a).compile(env, emitter);
                emitter.pop(RAX);
                emitter.pop(RBX);
                emitter.cmp(RAX, RBX);
                emitter.mov(RAX, Val::Int(0));
                emitter.jump_label_not_equal(eq_symbol.clone());
                emitter.mov(RAX, Val::Int(1));
                emitter.label(eq_symbol);
                emitter.push(RAX);
            }

            Lang::NotEqual(a, b) => {
                // Terrible way of doing this,
                // But should work.
                let eq_symbol = emitter.new_symbol("eq");
                (*b).compile(env, emitter);
                (*a).compile(env, emitter);
                emitter.pop(RAX);
                emitter.pop(RBX);
                emitter.cmp(RAX, RBX);
                emitter.mov(RAX, Val::Int(1));
                emitter.jump_label_not_equal(eq_symbol.clone());
                emitter.mov(RAX, Val::Int(0));
                emitter.label(eq_symbol);
                emitter.push(RAX);
            }

            Lang::Let(name, expr) => {
                expr.compile(env, emitter);
                emitter.pop(RAX);
                emitter.mov(env.get(&name).unwrap().val, RAX);
            }
            Lang::Set(name, expr) => {
                expr.compile(env, emitter);
                emitter.pop(RAX);
                emitter.mov(env.get(&name).unwrap().val, RAX);
            }
            Lang::While(pred, body) => {
                let start_loop_symbol = emitter.new_symbol("loop");
                let exit_loop_symbol = emitter.new_symbol("loop");
                emitter.label(start_loop_symbol.clone());
                (*pred).compile(env, emitter);
                emitter.pop(RAX);
                emitter.cmp(RAX, Val::Int(1));
                emitter.jump_label_not_equal(exit_loop_symbol.clone());
                (*body).compile(env, emitter);
                emitter.jump_label(start_loop_symbol);
                emitter.label(exit_loop_symbol);
            }
            Lang::Do(exprs) => {
                let last = exprs.len() - 1;
                for (i, expr) in exprs.into_iter().enumerate() {
                    match expr {
                        // Let's don't add to stack
                        // probably others I need to do this with too.
                        // could a is_statement method
                        Lang::Let(_, _) => {
                            expr.compile(env, emitter);
                        }
                        expr => {
                            expr.compile(env, emitter);
                            if i != last {
                                emitter.add(RSP, Val::Int(8));
                            }
                        }
                    }
                }
            }
            Lang::FFI { name, args, ptr } => {
                emitter.label(name.clone());
                emitter.imm_usize(ptr as usize);
                env.insert(
                    name,
                    EnvData {
                        val: Val::U8(0), // Dummy
                        is_foreign: true,
                    },
                );
            }

            // Need to properly handle void
            Lang::Func { name, args, body } => {
                // I could make it so that functions are hot patchable
                // by having some indirection here.
                println!("compiling {}", name);
                emitter.label(name);
                // Probably need to setup stack at some point
                emitter.push(RBP);
                emitter.mov(RBP, RSP);


                let mut current_local_var = 1;
                let lets = body.iter().fold(vec![], |mut acc, body| {
                    acc.extend(body.find_all_lets());
                    acc
                });

                for (name, _val) in &lets {
                    env.insert(
                        name.to_string(),
                        EnvData {
                            val: Val::AddrRegOffset(Register::RBP, current_local_var * -8),
                            is_foreign: false,
                        },
                    );
                    current_local_var += 1;
                }

                // This whole local var handling is a mess.
                let local_var_count = lets.len();
               
                // Make room for our local vars
                // Could deal with stack alignment here?
                // If I care about that in this JITed deal.
                emitter.sub(RSP, Val::Int(local_var_count as i32 * 8));

                assert!(args.len() <= 3);
                for (var, register) in args.iter().zip(FUNCTION_REGISTERS) {
                    env.insert(
                        var.to_string(),
                        EnvData {
                            val: register,
                            is_foreign: false,
                        },
                    );
                }
               
                for expr in body {
                    expr.compile(env, emitter)
                }

                emitter.leave();

                // If I'm shadowing this is going to be a problem
                // Cleanup env
                for var in args.iter() {
                    env.remove_entry(var);
                }
                for (name, _val) in &lets {
                    env.remove_entry(name);
                }
                emitter.ret();
            }
            // Need to properly handle void
            // Need to handle saving registers in a less terrible way.
            // Right now I save all the function argument registers
            // no matter what. Which is not great.
            Lang::Call0(name) => {

                emitter.push(FUNCTION_REGISTERS[0]);
                emitter.push(FUNCTION_REGISTERS[1]);
                emitter.push(FUNCTION_REGISTERS[2]);
                if env.get(&name).map(|x| x.is_foreign).unwrap_or(false) {
                    emitter.call(RIP_PLACEHOLDER);
                    emitter.add_label_patch(name);
                } else {
                    emitter.call_label(name);
                }
                emitter.pop(FUNCTION_REGISTERS[2]);
                emitter.pop(FUNCTION_REGISTERS[1]);
                emitter.pop(FUNCTION_REGISTERS[0]);
                emitter.push(RAX);
            }


            Lang::Call1(name, arg) => {
                arg.compile(env, emitter);
                emitter.pop(R8);
                emitter.push(FUNCTION_REGISTERS[0]);
                emitter.mov(FUNCTION_REGISTERS[0], R8);

                emitter.push(FUNCTION_REGISTERS[1]);
                emitter.push(FUNCTION_REGISTERS[2]);

                if env.get(&name).map(|x| x.is_foreign).unwrap_or(false) {
                    // arg registers are not preserved
                    // So I need to save them
                    // since I do no analysis to see if they are used.
                    emitter.call(Val::AddrRegOffset(Register::RIP, 0));
                    emitter.add_label_patch(name);
                } else {
                    emitter.call_label(name);
                }
                emitter.pop(FUNCTION_REGISTERS[2]);
                emitter.pop(FUNCTION_REGISTERS[1]);
                emitter.pop(FUNCTION_REGISTERS[0]);
                emitter.push(RAX);
            }


            Lang::Call2(name, arg1, arg2) => {
                arg1.compile(env, emitter);
                
                emitter.pop(R8);
                emitter.push(FUNCTION_REGISTERS[0]);
                emitter.mov(FUNCTION_REGISTERS[0], R8);

                arg2.compile(env, emitter);
                
                emitter.pop(R9);
                emitter.push(FUNCTION_REGISTERS[1]);
                emitter.mov(FUNCTION_REGISTERS[1], R9);

                emitter.push(FUNCTION_REGISTERS[2]);

                if env.get(&name).map(|x| x.is_foreign).unwrap_or(false) {
                    emitter.call(Val::AddrRegOffset(Register::RIP, 0));
                    emitter.add_label_patch(name);
                } else {
                    emitter.call_label(name);
                }
                emitter.pop(FUNCTION_REGISTERS[2]);
                emitter.pop(FUNCTION_REGISTERS[1]);
                emitter.pop(FUNCTION_REGISTERS[0]);
                emitter.push(RAX);
            }

            Lang::Call3(name, arg1, arg2, arg3) => {
                arg1.compile(env, emitter);
                emitter.pop(R8);
                emitter.push(FUNCTION_REGISTERS[0]);
                emitter.mov(FUNCTION_REGISTERS[0], R8);

                arg2.compile(env, emitter);
                emitter.pop(R9);
                emitter.push(FUNCTION_REGISTERS[1]);
                emitter.mov(FUNCTION_REGISTERS[1], R9);

                arg3.compile(env, emitter);
                emitter.pop(R10);
                emitter.push(FUNCTION_REGISTERS[2]);
                emitter.mov(FUNCTION_REGISTERS[2], R10);

                if env.get(&name).map(|x| x.is_foreign).unwrap_or(false) {
                    emitter.call(Val::AddrRegOffset(Register::RIP, 0));
                    emitter.add_label_patch(name);
                } else {
                    emitter.call_label(name);
                }
                emitter.pop(FUNCTION_REGISTERS[2]);
                emitter.pop(FUNCTION_REGISTERS[1]);
                emitter.pop(FUNCTION_REGISTERS[0]);
                emitter.push(RAX);
            }

            Lang::Variable(name) => {
                let val = env.get(&name).unwrap().val;
                match val {
                    Val::AddrRegOffset(_, _) => {
                        emitter.mov(RAX, val);
                        emitter.push(RAX);
                    }
                    _ => emitter.push(val),
                }
            }

            Lang::Return(expr) => {
                // Should this actually emitter.ret?
                (*expr).compile(env, emitter);
                emitter.pop(RAX);
            }

            Lang::Get(expr) => {
                (*expr).compile(env, emitter);
                emitter.pop(RAX);
                emitter.mov(RAX, Val::AddrReg(Register::RAX));
                emitter.push(RAX);
            }
            Lang::Store(location, value) => {
                (*location).compile(env, emitter);
                (*value).compile(env, emitter);
                emitter.pop(RBX);
                emitter.pop(RAX);
                emitter.mov(Val::AddrReg(Register::RAX), RBX);
                emitter.add(RAX, Val::Int(64)); // next location
                emitter.push(RAX);
            }
            _ => {}
        }
        // }
    }
}


macro_rules! lang {
    ((defn $name:ident[] 
        $body:tt
     )) => {
        Lang::Func{ 
            name: stringify!($name).to_string(), 
            args: vec![],
            body: vec![Lang::Return(Box::new(lang!($body)))]
        }
    };
    ((defn $name:ident[$arg:ident] 
        $body:tt
     )) => {
        Lang::Func{ 
            name: stringify!($name).to_string(), 
            args: vec![stringify!($arg).to_string()],
            body: vec![Lang::Return(Box::new(lang!($body)))]
        }
    };
    ((defn $name:ident[$arg1:ident $arg2:ident] 
        $body:tt
     )) => {
        Lang::Func{ 
            name: stringify!($name).to_string(), 
            args: vec![stringify!($arg1).to_string(), stringify!($arg2).to_string()],
            body: vec![Lang::Return(Box::new(lang!($body)))]
        }
    };
    ((defn $name:ident[$arg1:ident $arg2:ident $arg3:ident] 
        $body:tt
     )) => {
        Lang::Func{ 
            name: stringify!($name).to_string(), 
            args: vec![stringify!($arg1).to_string(), stringify!($arg2).to_string(), stringify!($arg3).to_string()],
            body: vec![Lang::Return(Box::new(lang!($body)))]
        }
    };
    ((let [$name:tt $val:tt]
        $body:tt
    )) => {
        Lang::Do(vec![
            Lang::Let(stringify!($name).to_string(), Box::new(lang!($val))),
            lang!($body)]);
    };
    ((if (= $arg:ident $val:expr)
        $result1:tt
        $result2:tt
    )) => {
        Lang::If(Box::new(
                Lang::Equal(
                    Box::new(Lang::Variable(stringify!($arg).to_string())),
                    Box::new(Lang::Int($val)))),
                Box::new(lang!($result1)),
                Box::new(lang!($result2)))
    };
    ((+ $arg1:tt $arg2:tt)) => {
        Lang::Add(Box::new(lang!($arg1)),
                  Box::new(lang!($arg2)))
    };
    ((+ $arg1:tt $arg2:tt $($args:tt)+)) => {
            Lang::Add(Box::new(lang!($arg1)),
                     Box::new(lang!((+ $arg2 $($args)+))))
    };
    ((- $arg1:tt $arg2:tt)) => {
        Lang::Sub(Box::new(lang!($arg1)),
                  Box::new(lang!($arg2)))
    };
    ((- $arg1:tt $arg2:tt $($args:tt)+)) => {
        Lang::Add(Box::new(lang!($arg1)),
                 Box::new(lang!((- $arg2 $($args)+))))
    };
    ((* $arg1:tt $arg2:tt)) => {
        Lang::Mul(Box::new(lang!($arg1)),
                  Box::new(lang!($arg2)))
    };
    ((* $arg1:tt $arg2:tt $($args:tt)+)) => {
        Lang::Add(Box::new(lang!($arg1)),
                 Box::new(lang!((+ $arg2 $($args)+))))
    };
    ((do $($arg1:tt)+)) => {
        Lang::Do(vec![$(lang!($arg1)),+])
    };
    ((return $arg:tt)) => {
        Lang::Return(Box::new(lang!($arg)))
    };
    (($f:ident $arg:tt)) => {
        Lang::Call1(stringify!($f).to_string(), Box::new(lang!($arg)))
    };
    (($f:ident $arg1:tt $arg2:tt)) => {
        Lang::Call2(stringify!($f).to_string(), Box::new(lang!($arg1)), Box::new(lang!($arg2)))
    };
    (($f:ident $arg1:tt $arg2:tt $arg3:tt)) => {
        Lang::Call3(stringify!($f).to_string(), Box::new(lang!($arg1)), Box::new(lang!($arg2)), Box::new(lang!($arg3)))
    };
    ($int:literal) => {
        Lang::Int($int)
    };
    ($var:ident) => {
        Lang::Variable(stringify!($var).to_string())
    }
}




#[allow(dead_code)]
const RAX: Val = Val::Reg(Register::RAX);


#[allow(dead_code)]
const RSP: Val = Val::Reg(Register::RSP);

#[allow(dead_code)]
const RDI: Val = Val::Reg(Register::RDI);

#[allow(dead_code)]
const RDX: Val = Val::Reg(Register::RDX);

#[allow(dead_code)]
const RSI: Val = Val::Reg(Register::RSI);

#[allow(dead_code)]
const RBX: Val = Val::Reg(Register::RBX);

#[allow(dead_code)]
const RCX: Val = Val::Reg(Register::RCX);

#[allow(dead_code)]
const RBP: Val = Val::Reg(Register::RBP);

#[allow(dead_code)]
const R8: Val = Val::Reg(Register::R8);

#[allow(dead_code)]
const R9: Val = Val::Reg(Register::R9);

#[allow(dead_code)]
const R10: Val = Val::Reg(Register::R10);

#[allow(dead_code)]
const R11: Val = Val::Reg(Register::R11);

#[allow(dead_code)]
const R12: Val = Val::Reg(Register::R12);

#[allow(dead_code)]
const R13: Val = Val::Reg(Register::R13);

#[allow(dead_code)]
const R14: Val = Val::Reg(Register::R14);

#[allow(dead_code)]
const R15: Val = Val::Reg(Register::R15);




#[allow(dead_code)]
const RIP_PLACEHOLDER : Val = Val::AddrRegOffset(Register::RIP, 0);

pub extern "C" fn print(x: u64) -> u64 {
    println!("{}", x);
    x   
}


pub extern "C" fn get_heap() ->  *const u8 {
    // I am a bit confused about life times and this working.
    // Maybe once I have the pointer all bets are off for the
    // borrow checker? Really not sure.
    let heap : [u8; 1024] = [0; 1024];
    (&heap) as *const u8
}


fn main() {
    let m = MemoryMap::new(
        4096,
        &[
            MapOption::MapReadable,
            MapOption::MapWritable,
            MapOption::MapExecutable,
        ],
    )
    .unwrap();
    let mut my_memory: &mut [u8; 4096] = unsafe {
        std::slice::from_raw_parts_mut(m.data(), m.len())
            .try_into()
            .expect("wrong size")
    };

    let e = &mut Emitter {
        memory: &mut my_memory,
        instruction_index: 0,
        labels: HashMap::new(),
        symbol_index: 0,
    };

    // If my code gets too big, I could overwrite this offset
    let mem_offset = 4096 - 128;
    let _mem_offset_size: usize = mem_offset.try_into().unwrap();

    // Things are encoding properly. But I'm not doing anything that makes
    // sense. Need to make program that writes to memory and reads it back.
    // e.mov(Val::Reg(Register::RBX), Val::Int(0));
    // e.mov(Val::Reg(Register::RAX), Val::Int(22));
    // RBP is used for RIP here??

    // e.jump_label("over".to_string());
    // e.label("over2".to_string());
    // e.mov(RAX, Val::Int(22));
    // e.ret();
    // e.mov(RAX, Val::Int(0));
    // e.label("over".to_string());
    // e.jump_label("over2".to_string());
    // e.mov(RBX, Val::Int(42));
    // e.cmp(RAX, RBX);
    // e.jump_label_equal("done".to_string());
    // e.mov(RAX, Val::Int(0));
    // e.label("done".to_string());
    // e.mov(RAX, Val::Int(0));
    // e.call_label("the_answer".to_string());
    // e.call_label("things".to_string());
    // e.ret();

    // e.label("the_answer".to_string());
    // e.mov(RAX, Val::Int(42));
    // e.ret();

    // e.label("things".to_string());
    // e.mov(RAX, Val::Int(44));
    // e.mov(RDI, Val::Int(2));
    // e.xor(RDX, RDX);
    // e.div(RDI);
    // // e.mov(RAX, RDX);
    // e.ret();

    // e.mov(RAX, Val::Int(42));
    // e.sub(RAX, Val::Int(1));
    // e.mov(RSI, RDI);
    // // e.add(RDI, Val::Int(64));
    // // e.mov(RBX, RSP);
    // // e.mov(RSP, RDI);
    // e.mov(
    //     Val::AddrRegOffset(Register::RDI, mem_offset),
    //     Val::Reg(Register::RAX),
    // );
    // e.mov(RAX, Val::AddrRegOffset(Register::RDI, mem_offset));
    // e.add(RDI, Val::Int(mem_offset));
    // e.sub(RSI, Val::Int(1));
    // e.imul(RSI, Val::Int(2));

    // e.mov(RBX, Val::AddrReg(Register::RDI));
    // e.mov(Val::AddrReg(Register::RDI), RBX);
    // // e.mov(Val::AddrRegOffset(Register::RDI, 0), RBX);
    // e.push(Val::Reg(Register::RAX));
    // e.pop(Val::Reg(Register::RAX));
    // e.mov(RSI, RBX);
    // // e.mov(RAX, Val::Int(43));
    // e.and(RAX, RAX);
    // e.and(RAX, Val::Int(1));
    // e.and(RAX, RDI);
    // e.or(RAX, RAX);
    // e.or(RAX, Val::Int(1));
    // e.or(RAX, RSI);
    // e.add(RAX, RBX);
    // e.add(RAX, Val::Int(1));
    // e.imul(RAX, Val::Int(2));

    // e.mov(RBX, RDI);
    // e.mov(RDI, Val::Int(32));
    // e.call(RBX);

    // Maybe make this better?
    // e.mov(RDI, Val::Int(42));
    // e.label_in_reg(RBX, "print".to_string());
    // e.call(RBX);
    // e.mov(RAX, Val::Int(52));

    // let sum = RDI;
    // let counter = RSI;
    // let div_3 = RBX;
    // let div_5 = RCX;
    // // e.ret();
    // e.mov(sum, Val::Int(0)); // sum
    // e.mov(counter, Val::Int(3)); // counter
    // e.mov(div_3, Val::Int(3)); // const for division
    // e.mov(div_5, Val::Int(5)); // const for division

    // e.label("main".to_string());
    // e.mov(RAX, counter);
    // e.xor(RDX, RDX);
    // e.div(div_3);

    // e.cmp(RDX, Val::Int(0));
    // e.jump_label_equal("sum".to_string());

    // e.mov(RAX, counter);
    // e.xor(RDX, RDX);
    // e.div(div_5);

    // e.cmp(RDX, Val::Int(0));
    // e.jump_label_not_equal("next".to_string());

    // e.label("sum".to_string());
    // e.add(sum, counter);

    // e.label("next".to_string());
    // e.inc(counter);
    // e.cmp(counter, Val::Int(1000));

    // e.jump_label_not_equal("main".to_string());

    // e.mov(RAX, sum);
    let env = &mut HashMap::new();

    // If I want to put functions first, I would need to tell the program what
    // address to start at.
    // Lang::Return(Box::new(
    //         Lang::If(Box::new(Lang::Equal(Box::new(Lang::Int(0)), Box::new(Lang::Int(0)))),
    //             Box::new(Lang::Call1("answer".to_string(), Box::new(Lang::Int(32)))),
    //             Box::new(Lang::Int(0))))).compile(env, e);

    // Need to think about calling builtin functions.
    // They will be indirect calls and I don't have a good notion of that.
    // If I made first class pointers and derefs though I could right?

    // Lang::Return(Box::new(Lang::If(
    //     Box::new(Lang::Equal(Box::new(Lang::Int(0)), Box::new(Lang::Int(0)))),
    //     Box::new(Lang::Call3(
    //         "add3".to_string(),
    //         Box::new(Lang::Int(2)),
    //         Box::new(Lang::Int(13)),
    //         Box::new(Lang::Int(4)),
    //     )),
    //     Box::new(Lang::Int(0)),
    // )))
    // .compile(env, e);

    // e.ret();

    // Lang::Return(Box::new(Lang::Call1(
    //         "try_while".to_string(),
    //         Box::new(Lang::Int(100000000))
    //     )
    // )).compile(env, e);

    Lang::FFI {
        name: "print".to_string(),
        args: vec!["x".to_string()],
        ptr: print as *const (),
    }
    .compile(env, e);


    Lang::FFI {
        name: "get_heap".to_string(),
        args: vec![],
        ptr: get_heap as *const (),
    }
    .compile(env, e);

    // Lang::Func {
    //     name: "main".to_string(),
    //     args: vec![],
    //     body: vec![Lang::Return(Box::new(Lang::Call1(
    //         "try_while".to_string(),
    //         Box::new(Lang::Int(3)),
    //     )))],
    // }
    // .compile(env, e);


    lang!(
        (defn fact [n]
                (if (= n 1)
                    1
                    (* n (fact (- n 1)))))
    ).compile(env, e);


    lang!(
        (defn fib [n]
            (if (= n 0)
                0
                (if (= n 1)
                    1
                    (+ (fib (- n 1))
                       (fib (- n 2))))))
    ).compile(env, e);

    lang!(
        (defn add3 [x y z]
            (do (print x)
                (print y)
                (print z)
           (+ x y z)))
    ).compile(env, e);


    lang!(
        (defn addstuff [x y z]
           (let [x1 1]
             (let [x2 2]
                (let [x3 3]
                    (+ z y x x1 x2 x3)))))
    ).compile(env, e);



    // Not needed, just for debugging
    e.mov(FUNCTION_REGISTERS[0], Val::Int(0));
    e.mov(FUNCTION_REGISTERS[1], Val::Int(0));
    e.mov(FUNCTION_REGISTERS[2], Val::Int(0));

    lang!(
        (defn main []
          (+ (fib 20)
             (fact 10)
             (addstuff 1 2 3)))

    ).compile(env, e);


    // e.label("main".to_string());
    // e.mov(RAX, Val::Int(2));
    // e.mov(R8, RAX);
    // e.add(R8, R8);
    // e.mov(R9, R8);
    // e.mov(R10, R9);
    // e.mov(R11, R10);
    // e.mov(R12, R11);
    // e.mov(R13, R12);
    // e.mov(R14, R13);
    // e.mov(R15, R14);
    // e.mov(RAX, R15);

    // e.ret();


    // Lang::Func {
    //     name: "main".to_string(),
    //     args: vec![],
    //     body: vec![
    //         // Lang::Let("l".to_string(),  Box::new(Lang::Call0("get_heap".to_string()))),
    //         // Lang::Let("l2".to_string(), 
    //         //         Box::new(Lang::Call3("cons".to_string(), Box::new(Lang::Int(41)), Box::new(Lang::Int(0)), Box::new(Lang::Variable("l".to_string()))))),
    //         // Lang::Let("l3".to_string(), 
    //         //         Box::new(Lang::Call3("cons".to_string(), Box::new(Lang::Int(42)), Box::new(Lang::Variable("l".to_string())), Box::new(Lang::Variable("l2".to_string()))))),
    //         // Lang::Return(Box::new(Lang::Call1("head".to_string(), Box::new(Lang::Call1("tail".to_string(), Box::new(Lang::Variable("l2".to_string())))))))

    //         Lang::Return(Box::new(Lang::Call1("fact".to_string(), Box::new(Lang::Int(20)))))
    //     ],
    // }
    // .compile(env, e);


    // e.ret();





    // Lang::Func {
    //     name: "cons".to_string(),
    //     // We are starting with an explicit location because I need to think
    //     // about how we would automatically get a new location. I'm guessing
    //     // some sort of free list?
    //     args: vec!["head".to_string(), "tail".to_string(), "loc".to_string()],
    //     body: vec![
    //         Lang::Let("next".to_string(),
    //             Box::new(
    //                 Lang::Store(
    //                     Box::new(Lang::Variable("loc".to_string())),
    //                     Box::new(Lang::Variable("head".to_string()))
    //                 )
    //             )
    //         ),            
    //         Lang::Let("after".to_string(),
    //             Box::new(
    //                 Lang::Store(
    //                     Box::new(Lang::Variable("next".to_string())),
    //                     Box::new(Lang::Variable("tail".to_string()))
    //                 )
    //             )
    //         ),

    //         Lang::Return(Box::new(Lang::Variable("after".to_string())))
    //     ],
    // }
    // .compile(env, e);

    // Lang::Func {
    //     name: "head".to_string(),
    //     args: vec!["list".to_string()],
    //     body: vec![
    //         Lang::Return(Box::new(Lang::Get(Box::new(Lang::Variable("list".to_string())))))
    //     ],
    // }
    // .compile(env, e);

    // Lang::Func {
    //     name: "tail".to_string(),
    //     args: vec!["list".to_string()],
    //     body: vec![
    //         Lang::Return(
    //             Box::new(Lang::Get(
    //                 Box::new(Lang::Add(
    //                     Box::new(Lang::Variable("list".to_string())),
    //                     Box::new(Lang::Int(64)))))))
    //     ],
    // }
    // .compile(env, e);


    // Lang::Func {
    //     name: "answer".to_string(),
    //     args: vec!["x".to_string()],
    //     body: vec![
    //         Lang::Return(Box::new(Lang::Variable("x".to_string()))),
    //     ]
    // }.compile(env, e);

    // Lang::Func {
    //     name: "add3".to_string(),
    //     args: vec!["x".to_string(), "y".to_string(), "z".to_string()],
    //     body: vec![
    //         Lang::Let("q".to_string(), Box::new(Lang::Variable("x".to_string()))),
    //         Lang::Let("r".to_string(), Box::new(Lang::Variable("y".to_string()))),
    //         Lang::Let("s".to_string(), Box::new(Lang::Variable("z".to_string()))),
    //         Lang::Return(Box::new(Lang::Add(
    //             Box::new(Lang::Add(
    //                 Box::new(Lang::Variable("q".to_string())),
    //                 Box::new(Lang::Variable("r".to_string())),
    //             )),
    //             Box::new(Lang::Variable("s".to_string())),
    //         ))),
    //     ],
    // }
    // .compile(env, e);

    
    // TODO: Make this representable in lang!
    // Lang::Func {
    //     name: "try_while".to_string(),
    //     args: vec!["n".to_string()],
    //     body: vec![
    //         Lang::While(
    //             Box::new(Lang::NotEqual(
    //                 Box::new(Lang::Variable("n".to_string())),
    //                 Box::new(Lang::Int(0)),
    //             )),
    //             Box::new(Lang::Do(vec![
    //                 Lang::Call1(
    //                     "print".to_string(),
    //                     Box::new(Lang::Variable("n".to_string())),
    //                 ),
    //                 Lang::Set(
    //                     "n".to_string(),
    //                     Box::new(Lang::Sub(
    //                         Box::new(Lang::Variable("n".to_string())),
    //                         Box::new(Lang::Int(1)),
    //                     )),
    //                 ),
    //             ])),
    //         ),
    //         Lang::Return(Box::new(Lang::Variable("n".to_string()))),
    //     ],
    // }
    // .compile(env, e);

    // Lang::Mul(Box::new(Lang::Int(3)), Box::new(Lang::Int(123)))
    //     .compile(env, e);

    // e.label("print".to_string());

    // let ptr = print as *const () as usize;
    // e.imm_usize(ptr);

    let result = e
        .memory
        .iter()
        .take(e.instruction_index)
        .fold(String::new(), |res, byte| res + &format!("{:02x}", byte));

    println!("{}", result);
    println!();

    let output = Command::new("yaxdis")
        .arg("-a")
        .arg("x86_64")
        .arg(result)
        .output()
        .expect("failed to execute process");

    println!("{}", String::from_utf8(output.stdout).unwrap());

    io::stdout().flush().unwrap();
    // This is working well, but need to figure out more than just 64bit
    // Need to figure out the whole SIB thing (can I just consider that an offset)
    // Need to figure out how I want to code this.
    // Need to deal with things that aren't really registers in modrm

    let ptr = print as *const ();
    let main_fn: extern "C" fn(*const ()) -> i64 = unsafe {
        mem::transmute(
            m.data().offset(
                e.labels
                    .get("main")
                    .unwrap()
                    .location
                    .unwrap()
                    .try_into()
                    .unwrap(),
            ),
        )
    };
    println!("Result {:}", main_fn(ptr));
  
}
