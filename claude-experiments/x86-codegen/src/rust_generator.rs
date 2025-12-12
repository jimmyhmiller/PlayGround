//! Rust code generator for x86-64 instructions

use crate::{AddressingMode, CodeGenerator, Instruction, OperandType};

pub struct RustGenerator;

impl RustGenerator {
    /// Generate a unique variant name for an instruction
    fn variant_name(instr: &Instruction) -> String {
        let mut name = instr.mnemonic.clone();

        // Add operand info to make unique
        for op in &instr.operands {
            match &op.addressing {
                AddressingMode::ModRmRm => {
                    name.push_str("Rm");
                    name.push_str(&operand_type_suffix(&op.op_type));
                }
                AddressingMode::ModRmReg => {
                    name.push_str("R");
                    name.push_str(&operand_type_suffix(&op.op_type));
                }
                AddressingMode::Immediate => {
                    name.push_str("Imm");
                    name.push_str(&operand_type_suffix(&op.op_type));
                }
                AddressingMode::RelativeOffset => {
                    name.push_str("Rel");
                    name.push_str(&operand_type_suffix(&op.op_type));
                }
                AddressingMode::OpcodeReg => {
                    name.push_str("OpR");
                    name.push_str(&operand_type_suffix(&op.op_type));
                }
                AddressingMode::Fixed(reg) => {
                    // Normalize register names
                    let normalized = reg
                        .replace("rAX", "Rax")
                        .replace("rSP", "Rsp")
                        .replace("AL", "Al")
                        .replace("CL", "Cl");
                    name.push_str(&normalized);
                }
                AddressingMode::MemoryOnly => {
                    name.push_str("M");
                }
                _ => {}
            }
        }

        // Add opcode extension if present
        if let Some(ext) = instr.opcode_ext {
            name.push_str(&format!("Ext{}", ext));
        }

        name
    }

    /// Generate field definitions for an instruction variant
    fn variant_fields(instr: &Instruction) -> String {
        let mut fields = Vec::new();

        for (i, op) in instr.operands.iter().enumerate() {
            match &op.addressing {
                AddressingMode::ModRmRm | AddressingMode::ModRmReg | AddressingMode::OpcodeReg => {
                    let field_name = if op.is_dest { "rd".to_string() } else { format!("r{}", i) };
                    fields.push(format!("{}: X86Register", field_name));
                }
                AddressingMode::Immediate | AddressingMode::RelativeOffset => {
                    let field_name = if i == 0 { "imm" } else { "imm1" };
                    let imm_type = match &op.op_type {
                        OperandType::Byte | OperandType::ByteSignExtended => "i8",
                        OperandType::Word => "i16",
                        OperandType::Dword | OperandType::WordDwordSignExtended => "i32",
                        OperandType::Qword => "i64",
                        _ => "i32", // Default to i32 for variable-size
                    };
                    fields.push(format!("{}: {}", field_name, imm_type));
                }
                AddressingMode::MemoryOnly => {
                    fields.push("base: X86Register".to_string());
                    fields.push("offset: i32".to_string());
                }
                _ => {}
            }
        }

        if fields.is_empty() {
            String::new()
        } else {
            format!(" {{ {} }}", fields.join(", "))
        }
    }

    /// Generate the encoding logic for an instruction
    fn encode_body(instr: &Instruction) -> String {
        let mut lines = Vec::new();
        lines.push("let mut bytes = Vec::new();".to_string());

        // Parse opcode bytes
        let opcode_bytes: Vec<u8> = instr
            .opcode
            .split_whitespace()
            .filter_map(|s| u8::from_str_radix(s, 16).ok())
            .collect();

        // Determine if we need REX prefix
        let needs_rex_w = instr.operands.iter().any(|op| {
            matches!(
                op.op_type,
                OperandType::Qword | OperandType::WordDwordQwordPromoted
            )
        });

        let has_reg_operands = instr.operands.iter().any(|op| {
            matches!(
                op.addressing,
                AddressingMode::ModRmRm | AddressingMode::ModRmReg | AddressingMode::OpcodeReg
            )
        });

        // Generate REX prefix if needed
        if needs_rex_w && has_reg_operands {
            // Find register operands
            let rm_field = instr
                .operands
                .iter()
                .enumerate()
                .find(|(_, op)| matches!(op.addressing, AddressingMode::ModRmRm));
            let reg_field = instr
                .operands
                .iter()
                .enumerate()
                .find(|(_, op)| matches!(op.addressing, AddressingMode::ModRmReg));

            if rm_field.is_some() || reg_field.is_some() {
                let rm_name = if rm_field.map(|(_, op)| op.is_dest).unwrap_or(false) {
                    "rd"
                } else {
                    "r1"
                };
                let reg_name = if reg_field.map(|(_, op)| op.is_dest).unwrap_or(false) {
                    "rd"
                } else {
                    "r0"
                };

                if rm_field.is_some() && reg_field.is_some() {
                    lines.push(format!(
                        "bytes.push(rex_w({}.index, {}.index));",
                        reg_name, rm_name
                    ));
                } else if rm_field.is_some() {
                    lines.push(format!("bytes.push(rex_w(0, {}.index));", rm_name));
                } else if reg_field.is_some() {
                    lines.push(format!("bytes.push(rex_w({}.index, 0));", reg_name));
                }
            } else {
                // Opcode register encoding
                lines.push("bytes.push(rex_w(0, rd.index));".to_string());
            }
        }

        // Emit opcode bytes
        for (i, byte) in opcode_bytes.iter().enumerate() {
            // Check if this opcode has +rd encoding
            let is_opcode_plus_rd = instr
                .operands
                .iter()
                .any(|op| matches!(op.addressing, AddressingMode::OpcodeReg));

            if i == opcode_bytes.len() - 1 && is_opcode_plus_rd {
                lines.push(format!("bytes.push(0x{:02X} + (rd.index & 0x7));", byte));
            } else {
                lines.push(format!("bytes.push(0x{:02X});", byte));
            }
        }

        // Emit ModR/M if needed
        if instr.has_modrm {
            let rm_field = instr
                .operands
                .iter()
                .find(|op| matches!(op.addressing, AddressingMode::ModRmRm | AddressingMode::MemoryOnly));
            let reg_field = instr
                .operands
                .iter()
                .find(|op| matches!(op.addressing, AddressingMode::ModRmReg));

            let reg_bits = if let Some(ext) = instr.opcode_ext {
                format!("{}", ext)
            } else if reg_field.is_some() {
                let name = if reg_field.map(|op| op.is_dest).unwrap_or(false) {
                    "rd"
                } else {
                    "r0"
                };
                format!("{}.index", name)
            } else {
                "0".to_string()
            };

            let rm_bits = if rm_field.is_some() {
                let name = if rm_field.map(|op| op.is_dest).unwrap_or(false) {
                    "rd"
                } else if reg_field.map(|op| op.is_dest).unwrap_or(false) {
                    "r1"
                } else {
                    "r0"
                };

                if matches!(
                    rm_field.map(|op| &op.addressing),
                    Some(AddressingMode::MemoryOnly)
                ) {
                    format!("base.index")
                } else {
                    format!("{}.index", name)
                }
            } else {
                "0".to_string()
            };

            lines.push(format!(
                "bytes.push(modrm(0b11, {}, {}));",
                reg_bits, rm_bits
            ));
        }

        // Emit immediate if present
        for (i, op) in instr.operands.iter().enumerate() {
            if matches!(
                op.addressing,
                AddressingMode::Immediate | AddressingMode::RelativeOffset
            ) {
                let field_name = if i == 0 { "imm".to_string() } else { format!("imm{}", i) };
                match &op.op_type {
                    OperandType::Byte | OperandType::ByteSignExtended => {
                        lines.push(format!("bytes.push({} as u8);", field_name));
                    }
                    OperandType::Word => {
                        lines.push(format!(
                            "bytes.extend_from_slice(&({} as i16).to_le_bytes());",
                            field_name
                        ));
                    }
                    OperandType::Dword | OperandType::WordDwordSignExtended => {
                        lines.push(format!(
                            "bytes.extend_from_slice(&({} as i32).to_le_bytes());",
                            field_name
                        ));
                    }
                    OperandType::Qword => {
                        lines.push(format!(
                            "bytes.extend_from_slice(&({} as i64).to_le_bytes());",
                            field_name
                        ));
                    }
                    _ => {
                        // Default: 32-bit
                        lines.push(format!(
                            "bytes.extend_from_slice(&({} as i32).to_le_bytes());",
                            field_name
                        ));
                    }
                }
            }
        }

        lines.push("bytes".to_string());
        lines.join("\n                ")
    }
}

fn operand_type_suffix(op_type: &OperandType) -> String {
    match op_type {
        OperandType::Byte => "8".to_string(),
        OperandType::Word => "16".to_string(),
        OperandType::Dword => "32".to_string(),
        OperandType::Qword => "64".to_string(),
        OperandType::WordDwordQword => "v".to_string(),
        OperandType::WordDwordQwordPromoted => "64".to_string(),
        OperandType::WordDwordSignExtended => "32s".to_string(),
        OperandType::ByteSignExtended => "8s".to_string(),
        OperandType::Other(s) => s.clone(),
    }
}

impl CodeGenerator for RustGenerator {
    fn generate(&self, instructions: &[&Instruction]) -> String {
        let mut output = String::new();

        // Header
        output.push_str(r#"//! Generated x86-64 instruction encoder
//!
//! This file was generated by x86-codegen from x86reference.xml

#![allow(dead_code)]
#![allow(clippy::identity_op)]

use std::ops::Shl;

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum Size {
    S8,
    S16,
    S32,
    S64,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct X86Register {
    pub size: Size,
    pub index: u8,
}

impl X86Register {
    pub fn encode(&self) -> u8 {
        self.index
    }

    pub fn from_index(index: usize) -> X86Register {
        X86Register {
            index: index as u8,
            size: Size::S64,
        }
    }
}

impl Shl<u32> for &X86Register {
    type Output = u32;

    fn shl(self, rhs: u32) -> Self::Output {
        (self.encode() as u32) << rhs
    }
}

impl Shl<u32> for X86Register {
    type Output = u32;

    fn shl(self, rhs: u32) -> Self::Output {
        (self.encode() as u32) << rhs
    }
}

// Register constants
pub const RAX: X86Register = X86Register { size: Size::S64, index: 0 };
pub const RCX: X86Register = X86Register { size: Size::S64, index: 1 };
pub const RDX: X86Register = X86Register { size: Size::S64, index: 2 };
pub const RBX: X86Register = X86Register { size: Size::S64, index: 3 };
pub const RSP: X86Register = X86Register { size: Size::S64, index: 4 };
pub const RBP: X86Register = X86Register { size: Size::S64, index: 5 };
pub const RSI: X86Register = X86Register { size: Size::S64, index: 6 };
pub const RDI: X86Register = X86Register { size: Size::S64, index: 7 };
pub const R8: X86Register = X86Register { size: Size::S64, index: 8 };
pub const R9: X86Register = X86Register { size: Size::S64, index: 9 };
pub const R10: X86Register = X86Register { size: Size::S64, index: 10 };
pub const R11: X86Register = X86Register { size: Size::S64, index: 11 };
pub const R12: X86Register = X86Register { size: Size::S64, index: 12 };
pub const R13: X86Register = X86Register { size: Size::S64, index: 13 };
pub const R14: X86Register = X86Register { size: Size::S64, index: 14 };
pub const R15: X86Register = X86Register { size: Size::S64, index: 15 };

// 32-bit register aliases
pub const EAX: X86Register = X86Register { size: Size::S32, index: 0 };
pub const ECX: X86Register = X86Register { size: Size::S32, index: 1 };
pub const EDX: X86Register = X86Register { size: Size::S32, index: 2 };
pub const EBX: X86Register = X86Register { size: Size::S32, index: 3 };
pub const ESP: X86Register = X86Register { size: Size::S32, index: 4 };
pub const EBP: X86Register = X86Register { size: Size::S32, index: 5 };
pub const ESI: X86Register = X86Register { size: Size::S32, index: 6 };
pub const EDI: X86Register = X86Register { size: Size::S32, index: 7 };
pub const R8D: X86Register = X86Register { size: Size::S32, index: 8 };
pub const R9D: X86Register = X86Register { size: Size::S32, index: 9 };
pub const R10D: X86Register = X86Register { size: Size::S32, index: 10 };
pub const R11D: X86Register = X86Register { size: Size::S32, index: 11 };
pub const R12D: X86Register = X86Register { size: Size::S32, index: 12 };
pub const R13D: X86Register = X86Register { size: Size::S32, index: 13 };
pub const R14D: X86Register = X86Register { size: Size::S32, index: 14 };
pub const R15D: X86Register = X86Register { size: Size::S32, index: 15 };

// 8-bit register aliases
pub const AL: X86Register = X86Register { size: Size::S8, index: 0 };
pub const CL: X86Register = X86Register { size: Size::S8, index: 1 };
pub const DL: X86Register = X86Register { size: Size::S8, index: 2 };
pub const BL: X86Register = X86Register { size: Size::S8, index: 3 };

/// Generate REX.W prefix (64-bit operand size)
/// reg is the ModR/M reg field (or opcode reg for +rd instructions)
/// rm is the ModR/M r/m field
#[inline]
pub fn rex_w(reg: u8, rm: u8) -> u8 {
    0x48 | ((reg >> 3) << 2) | (rm >> 3)
}

/// Generate REX prefix without W bit (for extended registers only)
#[inline]
pub fn rex(reg: u8, rm: u8) -> u8 {
    let r = (reg >> 3) << 2;
    let b = rm >> 3;
    if r | b != 0 {
        0x40 | r | b
    } else {
        0 // No REX needed
    }
}

/// Generate ModR/M byte
/// mod_: 0b11 for register-register, 0b00 for [reg], 0b01 for [reg+disp8], 0b10 for [reg+disp32]
/// reg: register operand or opcode extension
/// rm: r/m operand
#[inline]
pub fn modrm(mod_: u8, reg: u8, rm: u8) -> u8 {
    (mod_ << 6) | ((reg & 0x7) << 3) | (rm & 0x7)
}

/// Generate SIB byte
#[inline]
pub fn sib(scale: u8, index: u8, base: u8) -> u8 {
    (scale << 6) | ((index & 0x7) << 3) | (base & 0x7)
}

#[derive(Debug, Clone)]
pub enum X86Asm {
"#);

        // Generate enum variants
        let mut seen_variants = std::collections::HashSet::new();
        for instr in instructions {
            let variant_name = Self::variant_name(instr);
            if seen_variants.insert(variant_name.clone()) {
                let fields = Self::variant_fields(instr);
                output.push_str(&format!(
                    "    /// {} -- {}\n",
                    instr.mnemonic, instr.brief
                ));
                output.push_str(&format!("    {}{},\n", variant_name, fields));
            }
        }

        output.push_str("}\n\n");

        // Generate encode impl
        output.push_str("impl X86Asm {\n");
        output.push_str("    pub fn encode(&self) -> Vec<u8> {\n");
        output.push_str("        match self {\n");

        seen_variants.clear();
        for instr in instructions {
            let variant_name = Self::variant_name(instr);
            if seen_variants.insert(variant_name.clone()) {
                let fields = Self::variant_fields(instr);
                let pattern = if fields.is_empty() {
                    format!("X86Asm::{}", variant_name)
                } else {
                    format!("X86Asm::{} {}", variant_name, fields)
                };
                let body = Self::encode_body(instr);
                output.push_str(&format!(
                    "            {} => {{\n                {}\n            }}\n",
                    pattern, body
                ));
            }
        }

        output.push_str("        }\n");
        output.push_str("    }\n");
        output.push_str("}\n");

        output
    }
}
