use crate::{Instruction, Function, Label};
use std::collections::HashMap;

/// Manages disassembly display and analysis
pub struct DisassemblyAnalyzer {
    pub instructions: Vec<Instruction>,
    pub show_address: bool,
    pub show_hex: bool,
}

impl DisassemblyAnalyzer {
    pub fn new() -> Self {
        Self {
            instructions: Vec::new(),
            show_address: true,
            show_hex: false,
        }
    }

    pub fn add_instruction(&mut self, instruction: Instruction) {
        self.instructions.push(instruction);
        self.instructions.sort_by(|a, b| a.address.cmp(&b.address));
        self.instructions.dedup_by(|a, b| a.address == b.address);
    }

    pub fn get_instructions_around_pc(&self, pc: u64, window: usize) -> Vec<&Instruction> {
        let mut start = 0;
        for (i, instruction) in self.instructions.iter().enumerate() {
            if instruction.address == pc {
                start = i.saturating_sub(window);
                break;
            }
        }
        
        let mut contiguous_instructions = vec![];
        let mut last_address = 0;
        for instruction in self.instructions[start..].iter() {
            if instruction.address == last_address + 4 || last_address == 0 {
                contiguous_instructions.push(instruction);
            } else if instruction.address > pc {
                break;
            }
            last_address = instruction.address;
        }
        
        contiguous_instructions.into_iter().take(50).collect()
    }

    pub fn format_instruction_with_labels(
        &self,
        instruction: &Instruction,
        pc: u64,
        functions: &HashMap<usize, Function>,
        labels: &HashMap<usize, Label>,
    ) -> String {
        let prefix = if instruction.address == pc { "> " } else { "  " };
        
        if let Some(function) = functions.get(&(instruction.address as usize)) {
            format!(
                "{}{: <11} {}",
                prefix,
                function.get_name(),
                instruction.to_string(false, self.show_hex)
            )
        } else if let Some(label) = labels.get(&(instruction.address as usize)) {
            format!(
                "{}{: <11} {}",
                prefix,
                label.label,
                instruction.to_string(false, self.show_hex)
            )
        } else {
            format!(
                "{}{}",
                prefix,
                instruction.to_string(self.show_address, self.show_hex)
            )
        }
    }
}

impl Default for DisassemblyAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}