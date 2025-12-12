//! x86-64 instruction encoder generator
//!
//! This library parses the x86reference.xml specification and generates
//! Rust code for encoding x86-64 instructions.

pub mod parser;
pub mod rust_generator;

use std::collections::HashSet;

/// Addressing mode for an operand
#[derive(Debug, Clone, PartialEq)]
pub enum AddressingMode {
    /// E - ModRM r/m field (register or memory)
    ModRmRm,
    /// G - ModRM reg field (register only)
    ModRmReg,
    /// I - Immediate
    Immediate,
    /// J - Relative offset (for jumps/calls)
    RelativeOffset,
    /// M - Memory only (ModRM r/m must be memory)
    MemoryOnly,
    /// O - Direct offset (moffs)
    DirectOffset,
    /// Z - Register encoded in opcode (+rd)
    OpcodeReg,
    /// S - Segment register in ModRM reg field
    SegmentReg,
    /// Fixed register (AL, rAX, CL, etc.)
    Fixed(String),
    /// No operand or implicit
    None,
}

/// Operand type/size
#[derive(Debug, Clone, PartialEq)]
pub enum OperandType {
    /// b - byte (8-bit)
    Byte,
    /// w - word (16-bit)
    Word,
    /// d - doubleword (32-bit)
    Dword,
    /// q - quadword (64-bit)
    Qword,
    /// v - word, doubleword, or quadword (depending on operand-size)
    WordDwordQword,
    /// vqp - word, doubleword, or quadword (promoted to 64-bit in long mode)
    WordDwordQwordPromoted,
    /// vds - word, doubleword (sign-extended to 64-bit in 64-bit mode)
    WordDwordSignExtended,
    /// bs - byte, sign-extended to operand size
    ByteSignExtended,
    /// Other/unknown
    Other(String),
}

/// An operand in an instruction
#[derive(Debug, Clone)]
pub struct Operand {
    /// Addressing mode
    pub addressing: AddressingMode,
    /// Operand type/size
    pub op_type: OperandType,
    /// Is this a destination operand?
    pub is_dest: bool,
    /// Fixed register name if applicable (e.g., "AL", "rAX")
    pub fixed_reg: Option<String>,
    /// Register number if fixed
    pub reg_nr: Option<u8>,
}

/// An instruction entry from the XML
#[derive(Debug, Clone)]
pub struct Instruction {
    /// Mnemonic (e.g., "ADD", "MOV")
    pub mnemonic: String,
    /// Primary opcode bytes (hex string, e.g., "00", "0F 84")
    pub opcode: String,
    /// Opcode extension in ModRM reg field (0-7), if applicable
    pub opcode_ext: Option<u8>,
    /// Whether this instruction uses ModRM byte
    pub has_modrm: bool,
    /// Operands
    pub operands: Vec<Operand>,
    /// Brief description
    pub brief: String,
    /// Is this valid in 64-bit mode?
    pub valid_64bit: bool,
    /// Is this a 64-bit only instruction?
    pub only_64bit: bool,
    /// Direction bit (0 = rm,reg; 1 = reg,rm)
    pub direction: Option<u8>,
    /// Operand size bit (0 = byte, 1 = word/dword/qword)
    pub op_size: Option<u8>,
    /// Can be locked?
    pub lockable: bool,
    /// Opcode group (for two-byte opcodes)
    pub is_two_byte: bool,
}

/// Filter for selecting which instructions to generate
#[derive(Debug, Clone)]
pub struct InstructionFilter {
    allowed: Option<HashSet<String>>,
    blocked: Option<HashSet<String>>,
}

impl InstructionFilter {
    pub fn new() -> Self {
        Self {
            allowed: None,
            blocked: None,
        }
    }

    pub fn allow(mut self, names: Vec<String>) -> Self {
        self.allowed = Some(names.into_iter().collect());
        self
    }

    pub fn block(mut self, names: Vec<String>) -> Self {
        self.blocked = Some(names.into_iter().collect());
        self
    }

    pub fn should_include(&self, instruction: &Instruction) -> bool {
        if let Some(allowed) = &self.allowed {
            if !allowed.contains(&instruction.mnemonic) {
                return false;
            }
        }
        if let Some(blocked) = &self.blocked {
            if blocked.contains(&instruction.mnemonic) {
                return false;
            }
        }
        true
    }
}

impl Default for InstructionFilter {
    fn default() -> Self {
        Self::new()
    }
}

/// Main code generator interface
pub struct X86CodeGen {
    pub instructions: Vec<Instruction>,
}

impl X86CodeGen {
    /// Create a new code generator, loading instructions from x86reference.xml
    pub fn new(xml_path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let instructions = parser::parse_xml(xml_path)?;
        Ok(Self { instructions })
    }

    /// Get all available instruction mnemonics
    pub fn available_mnemonics(&self) -> Vec<&str> {
        let mut seen = HashSet::new();
        self.instructions
            .iter()
            .filter_map(|i| {
                if seen.insert(&i.mnemonic) {
                    Some(i.mnemonic.as_str())
                } else {
                    None
                }
            })
            .collect()
    }

    /// Find instructions by mnemonic pattern (case-insensitive)
    pub fn find_instructions(&self, pattern: &str) -> Vec<&Instruction> {
        let pattern_lower = pattern.to_lowercase();
        self.instructions
            .iter()
            .filter(|i| i.mnemonic.to_lowercase().contains(&pattern_lower))
            .collect()
    }

    /// Generate code for all instructions matching the filter
    pub fn generate_filtered<G: CodeGenerator>(
        &self,
        generator: &G,
        filter: InstructionFilter,
    ) -> String {
        let filtered: Vec<&Instruction> = self
            .instructions
            .iter()
            .filter(|i| filter.should_include(i))
            .collect();
        generator.generate(&filtered)
    }

    /// Generate code for specific mnemonics
    pub fn generate<G: CodeGenerator>(
        &self,
        generator: &G,
        mnemonics: Vec<&str>,
    ) -> String {
        let filter = InstructionFilter::new().allow(
            mnemonics.into_iter().map(|s| s.to_string()).collect()
        );
        self.generate_filtered(generator, filter)
    }

    /// Generate code for all instructions
    pub fn generate_all<G: CodeGenerator>(&self, generator: &G) -> String {
        let refs: Vec<&Instruction> = self.instructions.iter().collect();
        generator.generate(&refs)
    }
}

/// Trait for code generators (Rust, C++, etc.)
pub trait CodeGenerator {
    fn generate(&self, instructions: &[&Instruction]) -> String;
}
