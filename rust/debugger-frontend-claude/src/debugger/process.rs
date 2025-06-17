use crate::{DebugMessage, Register, Memory, Instruction, Value, BuiltInTypes, Result};
use crate::debugger::{FrameExtensions, ProcessExtensions, ValueExtensions};
use lldb::{SBProcess, SBTarget, SBThread};
use regex::Regex;
use std::sync::mpsc::Sender;

/// Manages debugging process state and message handling
pub struct ProcessState {
    pub pc: u64,
    pub sp: u64,
    pub fp: u64,
    pub registers: Vec<Register>,
    pub stack_memory: Vec<Memory>,
    pub heap_memory: Vec<Memory>,
    pub instructions: Vec<Instruction>,
    pub messages: Vec<DebugMessage>,
    pub message_sender: Option<Sender<DebugMessage>>,
    pub function_filter: Option<Regex>,
}

impl ProcessState {
    pub fn new() -> Self {
        Self {
            pc: 0,
            sp: 0,
            fp: 0,
            registers: Vec::new(),
            stack_memory: Vec::new(),
            heap_memory: Vec::new(),
            instructions: Vec::new(),
            messages: Vec::new(),
            message_sender: None,
            function_filter: None,
        }
    }

    pub fn with_message_sender(mut self, sender: Sender<DebugMessage>) -> Self {
        self.message_sender = Some(sender);
        self
    }

    pub fn with_function_filter(mut self, regex: Regex) -> Self {
        self.function_filter = Some(regex);
        self
    }

    pub fn update_from_process(&mut self, process: &SBProcess, target: &SBTarget) -> Result<()> {
        if !process.is_stopped() {
            return Ok(());
        }

        if let Some(thread) = process.thread_by_index_id(1) {
            self.update_from_thread(&thread, process, target)?;
        }

        Ok(())
    }

    fn update_from_thread(&mut self, thread: &SBThread, process: &SBProcess, target: &SBTarget) -> Result<()> {
        let frame = thread.selected_frame();
        
        // Update program counter and stack/frame pointers
        self.pc = frame.pc();
        self.sp = frame.sp();
        self.fp = frame.fp();

        // Update registers
        self.update_registers(&frame)?;

        // Update stack memory
        self.update_stack_memory(process)?;

        // Update instructions if needed
        self.update_instructions(&frame, process, target)?;

        Ok(())
    }

    fn update_registers(&mut self, frame: &lldb::SBFrame) -> Result<()> {
        self.registers.clear();
        
        for register in frame.registers().iter() {
            if let Some(name) = register.name() {
                if name.contains("General") {
                    for child in register.children() {
                        if let Some(child_name) = child.name() {
                            if child_name.contains('x') || 
                               child_name == "pc" || 
                               child_name == "sp" || 
                               child_name == "fp" {
                                
                                let value_str = child.value().unwrap_or("0x0");
                                let value = Value::String(value_str.to_string());
                                
                                let raw_value = usize::from_str_radix(
                                    value_str.trim_start_matches("0x"), 
                                    16
                                ).unwrap_or(0);
                                
                                let kind = BuiltInTypes::get_kind(raw_value);
                                
                                self.registers.push(Register::new(
                                    child_name.to_string(),
                                    value,
                                    kind,
                                ));
                            }
                        }
                    }
                }
            }
        }
        
        Ok(())
    }

    fn update_stack_memory(&mut self, process: &SBProcess) -> Result<()> {
        let stack_root = self.sp.saturating_sub(256);
        let stack_data = process.read_memory_as_u64_array(stack_root, 512)?;
        
        self.stack_memory = stack_data
            .iter()
            .enumerate()
            .map(|(i, value)| Memory::new(stack_root + (i as u64 * 8), *value))
            .collect();

        Ok(())
    }

    fn update_instructions(&mut self, frame: &lldb::SBFrame, process: &SBProcess, target: &SBTarget) -> Result<()> {
        // Check if we need to fetch new instructions
        let pc_in_instructions = self.instructions.iter().any(|inst| inst.address == self.pc);
        let ahead_exists = self.instructions.iter()
            .any(|inst| inst.address == self.pc + 30 * 4);

        if !pc_in_instructions || !ahead_exists {
            let instruction_list = process.get_instructions(frame, target)?;
            
            for instruction in &instruction_list {
                let parsed_instruction = Instruction {
                    address: instruction.address().load_address(target),
                    hex: String::new(), // Could extract hex data if needed
                    mnemonic: instruction.mnemonic(target).to_string(),
                    operands: instruction
                        .operands(target)
                        .to_string()
                        .split(',')
                        .map(|s| s.trim().to_string())
                        .collect(),
                    comment: instruction.comment(target).to_string(),
                };

                // Skip invalid instructions
                if parsed_instruction.mnemonic == "udf" {
                    continue;
                }

                self.instructions.push(parsed_instruction);
            }

            // Sort and deduplicate
            self.instructions.sort_by(|a, b| a.address.cmp(&b.address));
            self.instructions.dedup_by(|a, b| a.address == b.address);
        }

        Ok(())
    }

    pub fn check_for_debugger_info(&mut self, thread: &SBThread, process: &SBProcess) -> Result<bool> {
        let function_name = thread
            .selected_frame()
            .function_name()
            .unwrap_or("")
            .to_string();

        // Handle black_box function
        if function_name.contains("black_box") {
            thread.step_instruction(false)?;
            thread.step_instruction(false)?;
            return Ok(true);
        }

        // Handle debugger_info function
        if function_name == "debugger_info" {
            let x0 = thread
                .selected_frame()
                .get_register("x0")
                .ok_or_else(|| anyhow::anyhow!("Failed to get x0 register"))?
                .to_usize();
            let x1 = thread
                .selected_frame()
                .get_register("x1")
                .ok_or_else(|| anyhow::anyhow!("Failed to get x1 register"))?
                .to_usize();

            let mut buffer: Vec<u8> = vec![0; x1];
            process.read_memory(x0 as u64, &mut buffer)?;
            
            use crate::BinarySerialize;
            let message = DebugMessage::from_binary(&buffer)?;
            
            // Send message if sender is available
            if let Some(sender) = &self.message_sender {
                let _ = sender.send(message.clone());
            }
            
            self.messages.push(message);
            
            // Continue execution after processing debugger info
            process.continue_execution()?;
            return Ok(true);
        }

        Ok(false)
    }
}

impl Default for ProcessState {
    fn default() -> Self {
        Self::new()
    }
}