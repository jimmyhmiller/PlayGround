use std::{collections::HashMap, ops::Range};

use crate::{Data, Message};

#[derive(Debug, Clone, Copy)]
pub struct TokenData {
    _token_index: usize,
    line: usize,
    _column: usize,
    ir_range: (usize, usize),
    machine_code_range: (usize, usize),
}

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

    /// Process a new message and update internal state.
    pub fn process_message(&mut self, msg: &Message) {
        match &msg.data {
            Data::Tokens {
                file_name,
                token_line_column_map,
                ..
            } => {
                // We are going to assume we get the token data before we get any other data.
                // This is a reasonable assumption since the token data is the first data that is
                // generated.
                let mut token_data = Vec::new();
                for (token_index, (line, column)) in token_line_column_map.iter().enumerate() {
                    let ir_range = (0, 0);
                    let mc_range = (0, 0);
                    token_data.push(TokenData {
                        _token_index: token_index,
                        line: *line,
                        _column: *column,
                        ir_range,
                        machine_code_range: mc_range,
                    });
                }
                self.token_data_by_file.insert(file_name.clone(), token_data);
                self.update_lookups(file_name);
            }
            Data::Ir {
                file_name,
                function_pointer: _,
                token_range_to_ir_range,
                ..
            } => {
                let token_data = self.token_data_by_file.get_mut(file_name);
                if token_data.is_none() {
                    return;
                }
                let token_data = token_data.unwrap();
                // TODO: I want to find the best range
                for ((token_start, token_end), (ir_start, ir_end)) in token_range_to_ir_range.iter() {
                    for token in *token_start..*token_end {
                        let current_range = &mut token_data[token].ir_range;
                        let current_range_size = current_range.1 - current_range.0;
                        let new_range_size = ir_end - ir_start;
                        if *current_range == (0, 0) {
                            *current_range = (*ir_start, *ir_end);
                        } else if new_range_size < current_range_size {
                            *current_range = (*ir_start, *ir_end);
                        }
                    }
                }
                self.update_lookups(file_name);
            }
            Data::Arm {
                file_name,
                function_pointer,
                ir_to_machine_code_range,
                ..
            } => {
                let token_data = self.token_data_by_file.get_mut(file_name);
                if token_data.is_none() {
                    return;
                }
                let token_data = token_data.unwrap();
                for (ir_index, (mc_start, mc_end)) in ir_to_machine_code_range.iter() {
                    let mc_start = (mc_start * 4) + function_pointer;
                    let mc_end = (mc_end * 4) + function_pointer;
                    for token in token_data.iter_mut().filter(|x|{
                        if x.ir_range == (0, 0) {
                            return false;
                        }
                        x.ir_range.0 <= *ir_index && x.ir_range.1 >= *ir_index
                    }) {
                        let current_range = &mut token.machine_code_range;
                        let current_range_size = current_range.1 - current_range.0;
                        let new_range_size = mc_end - mc_start;
                        if *current_range == (0, 0) {
                            *current_range = (mc_start, mc_end);
                        } else if new_range_size > current_range_size {
                            *current_range = (mc_start, mc_end);
                        }
                    }
                }
                self.update_lookups(file_name);
            }
            Data::UserFunction { name, pointer, len: _, number_of_arguments: _ } => {
                self.functions_by_pointer.insert(*pointer, name.clone());
            }
            _ => {}
        }
        
    }

    #[allow(unused)]
    /// Lookup the machine code address for a given file and line.
    pub fn address_by_file_line(&self, file: &str, line: usize) -> Option<u64> {
        self.file_line_to_address.get(&(file.to_string(), line)).cloned()
    }

    pub fn file_line_by_address(&self, address: u64) -> Option<(String, usize)> {
        //function_range_to_file_line
        for (range, (file, line)) in &self.function_range_to_file_line {
            if range.contains(&(address as usize)) {
                return Some((file.clone(), *line));
            }
        }
        None
    }
    
    fn update_lookups(&mut self, file_name: &str) {
        let token_data = self.token_data_by_file.get(file_name).unwrap();
        let mut file_line_to_address = HashMap::new();
        for token in token_data.iter() {
            if token.machine_code_range == (0, 0) {
                continue;
            }
            let line = token.line;
            let address = token.machine_code_range.0;
            file_line_to_address.insert((file_name.to_string(), line), address as u64);
        }
        for (key, value) in file_line_to_address.iter() {
            self.file_line_to_address.insert(key.clone(), *value);
        }

        let mut line_address_for_file = file_line_to_address.iter()
            .filter(|(key, _)| key.0 == file_name)
            .map(|((_, line), address)| (*line, *address))
            .collect::<Vec<_>>();
        line_address_for_file.sort_by_key(|(line, _)| *line);
        let mut function_range_to_file_line = Vec::new();
        // We now have a list of lines and start addresses
        // for each line, we are going to find the next line and its address
        // and create a range from the start address to the next line's address
        // lets do this by getting pairs of lines and addresses
        let mut iter = line_address_for_file.iter().peekable();
        while let Some((line, address)) = iter.next() {
            let address = *address as usize;
            let next = iter.peek().cloned();
            let range = match next {
                Some((_next_line, next_address)) => {
                    let next_address = *next_address as usize;
                    address..next_address
                }
                None => {
                    address..(address + 4)
                }
            };
            function_range_to_file_line.push((range, (file_name.to_string(), *line)));
        }
        for (range, file_line) in function_range_to_file_line.iter() {
            self.function_range_to_file_line.push((range.clone(), file_line.clone()));
        }
        
    }
}