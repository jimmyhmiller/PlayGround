use crate::{Register, Memory, Instruction, BuiltInTypes};
use std::collections::HashSet;

/// Formats values for display in the debugger UI
pub struct ValueFormatter;

impl ValueFormatter {
    pub fn format_register(register: &Register) -> String {
        format!(
            "{}: {} - {}",
            register.name,
            register.value.to_string(),
            register.kind.to_string()
        )
    }

    pub fn format_memory(memory: &Memory) -> String {
        format!(
            "0x{:x}: 0x{:x} {}",
            memory.address,
            memory.value,
            memory.kind.to_string()
        )
    }

    pub fn format_instruction(
        instruction: &Instruction,
        show_address: bool,
        show_hex: bool,
    ) -> String {
        instruction.to_string(show_address, show_hex)
    }

    pub fn format_type_info(value: u64) -> String {
        let kind = BuiltInTypes::get_kind(value as usize);
        match kind {
            BuiltInTypes::Int => {
                let untagged = BuiltInTypes::untag(value as usize) as i64;
                format!("Int({})", untagged)
            }
            BuiltInTypes::Float => {
                let untagged = BuiltInTypes::untag(value as usize);
                format!("Float(0x{:x})", untagged)
            }
            BuiltInTypes::Bool => {
                let untagged = BuiltInTypes::untag(value as usize);
                format!("Bool({})", untagged != 0)
            }
            BuiltInTypes::String => format!("String(*0x{:x})", BuiltInTypes::untag(value as usize)),
            BuiltInTypes::Function => format!("Function(*0x{:x})", BuiltInTypes::untag(value as usize)),
            BuiltInTypes::Closure => format!("Closure(*0x{:x})", BuiltInTypes::untag(value as usize)),
            BuiltInTypes::Struct => format!("Struct(*0x{:x})", BuiltInTypes::untag(value as usize)),
            BuiltInTypes::Array => format!("Array(*0x{:x})", BuiltInTypes::untag(value as usize)),
            BuiltInTypes::Null => "Null".to_string(),
            BuiltInTypes::None => "None".to_string(),
        }
    }
}

/// Filters registers based on what's mentioned in disassembly
pub struct RegisterFilter;

impl RegisterFilter {
    pub fn get_mentioned_registers(instructions: &[Instruction]) -> HashSet<String> {
        let mut mentioned = instructions
            .iter()
            .flat_map(|inst| inst.operands.iter())
            .filter(|operand| operand.starts_with('x'))
            .map(|operand| operand.trim_end_matches(','))
            .map(|operand| operand.to_string())
            .collect::<HashSet<String>>();

        // Always include essential registers
        mentioned.insert("pc".to_string());
        mentioned.insert("sp".to_string());
        mentioned.insert("fp".to_string());

        mentioned
    }

    pub fn filter_registers<'a>(
        registers: &'a [Register],
        mentioned: &HashSet<String>,
    ) -> Vec<&'a Register> {
        registers
            .iter()
            .filter(|register| mentioned.contains(&register.name))
            .collect()
    }
}