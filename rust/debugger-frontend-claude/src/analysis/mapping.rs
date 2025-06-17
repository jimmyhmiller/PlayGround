use crate::{Function, Label, BreakpointMapper};
use std::collections::HashMap;

/// Manages mapping between source code and machine code
pub struct SourceMapper {
    pub functions: HashMap<usize, Function>,
    pub labels: HashMap<usize, Label>,
    pub breakpoint_mapper: BreakpointMapper,
}

impl SourceMapper {
    pub fn new() -> Self {
        Self {
            functions: HashMap::new(),
            labels: HashMap::new(),
            breakpoint_mapper: BreakpointMapper::new(),
        }
    }

    pub fn add_function(&mut self, address: usize, function: Function) {
        self.functions.insert(address, function);
    }

    pub fn add_label(&mut self, address: usize, label: Label) {
        self.labels.insert(address, label);
    }

    pub fn get_function_at_pc(&self, pc: u64) -> Option<&Function> {
        for function in self.functions.values() {
            if let Function::User { address_range, .. } = function {
                if pc as usize >= address_range.0 && pc as usize <= address_range.1 {
                    return Some(function);
                }
            }
        }
        None
    }

    pub fn get_current_location(&self, pc: u64) -> Option<(String, usize, String)> {
        let function_name = self.get_function_at_pc(pc)
            .map(|f| f.get_name())
            .unwrap_or_else(|| "No Function".to_string());

        if let Some((file, line)) = self.breakpoint_mapper.file_line_by_address(pc) {
            let file = file.split('/').last().unwrap_or(&file);
            Some((file.to_string(), line, function_name))
        } else {
            Some(("unknown".to_string(), 0, function_name))
        }
    }
}

impl Default for SourceMapper {
    fn default() -> Self {
        Self::new()
    }
}