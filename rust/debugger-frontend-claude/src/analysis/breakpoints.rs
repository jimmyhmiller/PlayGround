use std::{collections::HashMap, ops::Range};
use crate::{DebugMessage, MessageData};

#[derive(Debug, Clone, Copy)]
pub struct TokenData {
    token_index: usize,
    pub line: usize,
    column: usize,
    pub ir_range: (usize, usize),
    pub machine_code_range: (usize, usize),
}

impl TokenData {
    pub fn new(token_index: usize, line: usize, column: usize) -> Self {
        Self {
            token_index,
            line,
            column,
            ir_range: (0, 0),
            machine_code_range: (0, 0),
        }
    }
}

/// Maps between source locations and machine code addresses
pub struct BreakpointMapper {
    pub token_data_by_file: HashMap<String, Vec<TokenData>>,
    pub file_line_to_address: HashMap<(String, usize), u64>,
    pub function_range_to_file_line: Vec<(Range<usize>, (String, usize))>,
    pub functions_by_pointer: HashMap<usize, String>,
}

impl BreakpointMapper {
    pub fn new() -> Self {
        Self {
            token_data_by_file: HashMap::new(),
            functions_by_pointer: HashMap::new(),
            file_line_to_address: HashMap::new(),
            function_range_to_file_line: Vec::new(),
        }
    }

    /// Process a new message and update internal state
    pub fn process_message(&mut self, msg: &DebugMessage) {
        match &msg.data {
            MessageData::Tokens {
                file_name,
                token_line_column_map,
                ..
            } => {
                let mut token_data = Vec::new();
                for (token_index, (line, column)) in token_line_column_map.iter().enumerate() {
                    token_data.push(TokenData::new(token_index, *line, *column));
                }
                self.token_data_by_file.insert(file_name.clone(), token_data);
                self.update_lookups(file_name);
            }
            MessageData::Ir {
                file_name,
                token_range_to_ir_range,
                ..
            } => {
                if let Some(token_data) = self.token_data_by_file.get_mut(file_name) {
                    for ((token_start, token_end), (ir_start, ir_end)) in token_range_to_ir_range.iter() {
                        for token_idx in *token_start..*token_end {
                            if let Some(token) = token_data.get_mut(token_idx) {
                                let current_range_size = token.ir_range.1 - token.ir_range.0;
                                let new_range_size = ir_end - ir_start;
                                if token.ir_range == (0, 0) || new_range_size < current_range_size {
                                    token.ir_range = (*ir_start, *ir_end);
                                }
                            }
                        }
                    }
                    self.update_lookups(file_name);
                }
            }
            MessageData::Arm {
                file_name,
                function_pointer,
                ir_to_machine_code_range,
                ..
            } => {
                if let Some(token_data) = self.token_data_by_file.get_mut(file_name) {
                    for (ir_index, (mc_start, mc_end)) in ir_to_machine_code_range.iter() {
                        let mc_start = (mc_start * 4) + function_pointer;
                        let mc_end = (mc_end * 4) + function_pointer;
                        
                        for token in token_data.iter_mut().filter(|x| {
                            x.ir_range != (0, 0) 
                                && x.ir_range.0 <= *ir_index 
                                && x.ir_range.1 >= *ir_index
                        }) {
                            let current_range_size = token.machine_code_range.1 - token.machine_code_range.0;
                            let new_range_size = mc_end - mc_start;
                            if token.machine_code_range == (0, 0) || new_range_size > current_range_size {
                                token.machine_code_range = (mc_start, mc_end);
                            }
                        }
                    }
                    self.update_lookups(file_name);
                }
            }
            MessageData::UserFunction { name, pointer, .. } => {
                self.functions_by_pointer.insert(*pointer, name.clone());
            }
            _ => {}
        }
    }

    /// Lookup the machine code address for a given file and line
    pub fn address_by_file_line(&self, file: &str, line: usize) -> Option<u64> {
        self.file_line_to_address.get(&(file.to_string(), line)).cloned()
    }

    /// Lookup the file and line for a given machine code address
    pub fn file_line_by_address(&self, address: u64) -> Option<(String, usize)> {
        for (range, (file, line)) in &self.function_range_to_file_line {
            if range.contains(&(address as usize)) {
                return Some((file.clone(), *line));
            }
        }
        None
    }

    /// Get all mapped breakpoint locations for a file
    pub fn get_breakpoint_locations(&self, file: &str) -> Vec<(usize, u64)> {
        self.file_line_to_address
            .iter()
            .filter_map(|((f, line), address)| {
                if f == file {
                    Some((*line, *address))
                } else {
                    None
                }
            })
            .collect()
    }

    fn update_lookups(&mut self, file_name: &str) {
        if let Some(token_data) = self.token_data_by_file.get(file_name) {
            let mut file_line_to_address = HashMap::new();
            for token in token_data.iter() {
                if token.machine_code_range != (0, 0) {
                    let line = token.line;
                    let address = token.machine_code_range.0;
                    file_line_to_address.insert((file_name.to_string(), line), address as u64);
                }
            }
            
            for (key, value) in file_line_to_address.iter() {
                self.file_line_to_address.insert(key.clone(), *value);
            }

            let mut line_address_for_file: Vec<_> = file_line_to_address
                .iter()
                .filter(|(key, _)| key.0 == file_name)
                .map(|((_, line), address)| (*line, *address))
                .collect();
            line_address_for_file.sort_by_key(|(line, _)| *line);

            let mut function_range_to_file_line = Vec::new();
            let mut iter = line_address_for_file.iter().peekable();
            while let Some((line, address)) = iter.next() {
                let address = *address as usize;
                let range = match iter.peek() {
                    Some((_next_line, next_address)) => {
                        let next_address = *next_address as usize;
                        address..next_address
                    }
                    None => address..(address + 4),
                };
                function_range_to_file_line.push((range, (file_name.to_string(), *line)));
            }
            
            for (range, file_line) in function_range_to_file_line.iter() {
                self.function_range_to_file_line.push((range.clone(), file_line.clone()));
            }
        }
    }
}

impl Default for BreakpointMapper {
    fn default() -> Self {
        Self::new()
    }
}