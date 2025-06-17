use lldb::{SBDebugger, SBTarget, SBProcess, SBLaunchInfo};
use crate::{Result, Function};
use crate::core::{Message, MessageData};
use crate::debugger::extensions::{FrameExtensions, ProcessExtensions, ValueExtensions};
use std::collections::HashMap;

/// High-level debugger client wrapper
pub struct DebuggerClient {
    debugger: SBDebugger,
    target: Option<SBTarget>,
    process: Option<SBProcess>,
    functions: HashMap<usize, Function>,
}

impl DebuggerClient {
    pub fn new() -> Result<Self> {
        SBDebugger::initialize();
        let debugger = SBDebugger::create(false);
        debugger.set_asynchronous(false);
        
        Ok(Self {
            debugger,
            target: None,
            process: None,
            functions: HashMap::new(),
        })
    }

    pub fn create_target(&mut self, executable_path: &str) -> Result<()> {
        if let Some(target) = self.debugger.create_target_simple(executable_path) {
            self.target = Some(target);
            Ok(())
        } else {
            Err(anyhow::anyhow!("Failed to create target for {}", executable_path))
        }
    }

    pub fn launch_process(&mut self, args: Vec<&str>) -> Result<()> {
        let target = self.target.as_ref()
            .ok_or_else(|| anyhow::anyhow!("No target created"))?;

        let launch_info = SBLaunchInfo::new();
        launch_info.set_arguments(args, false);

        match target.launch(launch_info) {
            Ok(process) => {
                self.process = Some(process);
                Ok(())
            }
            Err(e) => Err(anyhow::anyhow!("Failed to launch process: {:?}", e))
        }
    }

    pub fn set_breakpoint_by_name(&self, function_name: &str, module_name: &str) -> Result<()> {
        let target = self.target.as_ref()
            .ok_or_else(|| anyhow::anyhow!("No target created"))?;

        use crate::debugger::TargetExtensions;
        if let Some(breakpoint) = target.create_breakpoint_by_name(function_name, module_name) {
            breakpoint.set_enabled(true);
            Ok(())
        } else {
            Err(anyhow::anyhow!("Failed to create breakpoint for {}", function_name))
        }
    }

    pub fn set_breakpoint_by_address(&self, address: u64) -> Result<()> {
        let target = self.target.as_ref()
            .ok_or_else(|| anyhow::anyhow!("No target created"))?;

        let breakpoint = target.breakpoint_create_by_address(address);
        breakpoint.set_enabled(true);
        Ok(())
    }

    pub fn get_target(&self) -> Option<&SBTarget> {
        self.target.as_ref()
    }

    pub fn get_process(&self) -> Option<&SBProcess> {
        self.process.as_ref()
    }

    pub fn is_process_stopped(&self) -> bool {
        self.process.as_ref()
            .map(|p| p.is_stopped())
            .unwrap_or(false)
    }

    /// Collect debug information from debugger_info breakpoint
    pub fn collect_debug_info(&mut self) -> Result<Vec<Message>> {
        let mut messages = Vec::new();
        
        if let Some(process) = self.process.clone() {
            // Check all threads, not just thread 1
            for thread in process.threads() {
                let function_name = thread.selected_frame()
                    .function_name()
                    .unwrap_or("")
                    .to_string();
                
                // Handle black_box wrapper like the original debugger
                if function_name.contains("black_box") {
                    let _ = thread.step_instruction(false);
                    let _ = thread.step_instruction(false);
                }
                
                // Re-check function name after stepping
                let function_name = thread.selected_frame()
                    .function_name()
                    .unwrap_or("")
                    .to_string();
                    
                if function_name == "debugger_info" {
                    // Read debug message from registers
                    let x0 = thread.selected_frame()
                        .get_register("x0")
                        .map(|r| r.to_u64())
                        .unwrap_or(0);
                    let x1 = thread.selected_frame() 
                        .get_register("x1")
                        .map(|r| r.to_u64())
                        .unwrap_or(0);

                    if x1 > 0 && x1 < 10000 { // Sanity check
                        let mut buffer = vec![0u8; x1 as usize];
                        if let Ok(bytes_read) = process.read_memory(x0, &mut buffer) {
                            if bytes_read > 0 {
                                // Try to deserialize the message - the original uses bincode
                                match Message::from_binary(&buffer[..bytes_read]) {
                                    Ok(message) => {
                                        self.process_debug_message(&message)?;
                                        messages.push(message);
                                    }
                                    Err(_) => {
                                        // Ignore deserialization errors silently
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        
        Ok(messages)
    }

    /// Process a debug message and update internal state
    fn process_debug_message(&mut self, message: &Message) -> Result<()> {
        match &message.data {
            MessageData::UserFunction { name, pointer, len, number_of_arguments } => {
                let function = Function::User {
                    name: name.clone(),
                    address_range: (*pointer, *pointer + *len),
                    number_of_arguments: *number_of_arguments,
                };
                self.functions.insert(*pointer, function);
            }
            MessageData::BuiltinFunction { name, pointer } => {
                let function = Function::Builtin {
                    name: name.clone(),
                    pointer: *pointer,
                };
                self.functions.insert(*pointer, function);
            }
            MessageData::ForeignFunction { name, pointer } => {
                let function = Function::Foreign {
                    name: name.clone(), 
                    pointer: *pointer,
                };
                self.functions.insert(*pointer, function);
            }
            _ => {} // Handle other message types as needed
        }
        Ok(())
    }

    /// Get a function by name (e.g., "fib/fib")
    pub fn get_function_by_name(&self, name: &str) -> Option<&Function> {
        self.functions.values().find(|f| {
            match f {
                Function::User { name: fname, .. } => fname == name,
                Function::Builtin { name: fname, .. } => fname == name,
                Function::Foreign { name: fname, .. } => fname == name,
            }
        })
    }

    /// Set breakpoint on a Beagle function by name
    pub fn set_breakpoint_on_beagle_function(&self, function_name: &str) -> Result<bool> {
        if let Some(function) = self.get_function_by_name(function_name) {
            let address = match function {
                Function::User { address_range, .. } => address_range.0 as u64,
                Function::Builtin { pointer, .. } => *pointer as u64,
                Function::Foreign { pointer, .. } => *pointer as u64,
            };
            
            if let Some(target) = &self.target {
                let breakpoint = target.breakpoint_create_by_address(address);
                breakpoint.set_enabled(true);
                println!("Set breakpoint on {} at address 0x{:x}", function_name, address);
                Ok(true)
            } else {
                Err(anyhow::anyhow!("No target available"))
            }
        } else {
            Ok(false) // Function not found
        }
    }

    /// Get all available Beagle functions
    pub fn list_beagle_functions(&self) -> Vec<String> {
        self.functions.values().map(|f| {
            match f {
                Function::User { name, .. } => name.clone(),
                Function::Builtin { name, .. } => name.clone(), 
                Function::Foreign { name, .. } => name.clone(),
            }
        }).collect()
    }

    /// Get disassembly for a specific function
    pub fn get_function_disassembly(&self, function_name: &str) -> Result<Vec<String>> {
        if let Some(function) = self.get_function_by_name(function_name) {
            let (start_addr, len) = match function {
                Function::User { address_range, .. } => (address_range.0 as u64, (address_range.1 - address_range.0) as usize),
                Function::Builtin { pointer, .. } => (*pointer as u64, 64), // Smaller default size
                Function::Foreign { pointer, .. } => (*pointer as u64, 64), // Smaller default size
            };

            // For now, just return function info since disassembly is complex
            let mut result = Vec::new();
            result.push(format!("Function: {}", function_name));
            result.push(format!("Address: 0x{:x}", start_addr));
            result.push(format!("Length: {} bytes", len));
            
            match function {
                Function::User { number_of_arguments, .. } => {
                    result.push(format!("Arguments: {}", number_of_arguments));
                }
                _ => {}
            }
            
            // TODO: Add actual disassembly using LLDB
            result.push("(Disassembly not yet implemented)".to_string());
            
            Ok(result)
        } else {
            Err(anyhow::anyhow!("Function '{}' not found", function_name))
        }
    }

    pub fn continue_execution(&self) -> Result<()> {
        if let Some(process) = &self.process {
            process.continue_execution()
                .map_err(|e| anyhow::anyhow!("Failed to continue execution: {:?}", e))
        } else {
            Err(anyhow::anyhow!("No process to continue"))
        }
    }

    pub fn step_over(&self) -> Result<()> {
        if let Some(process) = &self.process {
            if let Some(thread) = process.thread_by_index_id(1) {
                thread.step_instruction(true)
                    .map_err(|e| anyhow::anyhow!("Failed to step over: {:?}", e))
            } else {
                Err(anyhow::anyhow!("No thread found"))
            }
        } else {
            Err(anyhow::anyhow!("No process to step"))
        }
    }

    pub fn step_into(&self) -> Result<()> {
        if let Some(process) = &self.process {
            if let Some(thread) = process.thread_by_index_id(1) {
                thread.step_instruction(false)
                    .map_err(|e| anyhow::anyhow!("Failed to step into: {:?}", e))
            } else {
                Err(anyhow::anyhow!("No thread found"))
            }
        } else {
            Err(anyhow::anyhow!("No process to step"))
        }
    }
}

impl Default for DebuggerClient {
    fn default() -> Self {
        Self::new().expect("Failed to create debugger client")
    }
}