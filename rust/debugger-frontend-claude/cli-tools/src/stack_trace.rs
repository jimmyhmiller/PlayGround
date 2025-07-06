use anyhow::{anyhow, Result};
use clap::Parser;
use std::fs::File;
use std::io::Write;
use std::path::PathBuf;
use std::ffi::{c_void, CString};
use lldb::{SBDebugger, SBProcess, SBLaunchInfo, SBError, SBValue};
use lldb_sys::{SBValueGetValueAsUnsigned2, SBProcessReadMemory, SBTargetBreakpointCreateByName};
use bincode::{config::standard, Decode, Encode};
use std::collections::HashMap;

#[derive(Parser, Debug)]
#[command(author, version, about = "Stack trace visualization tool", long_about = None)]
struct Args {
    /// The path to the executable to debug
    program: String,

    /// Arguments to pass to the program
    #[arg(long)]
    args: Vec<String>,

    /// Output file for stack trace visualization
    #[arg(short, long, default_value = "stack_trace.json")]
    output: PathBuf,

    /// Maximum number of function calls to capture (0 = unlimited)
    #[arg(short, long, default_value = "1000")]
    max_calls: usize,

    /// Enable verbose debug output
    #[arg(short, long)]
    verbose: bool,
}

macro_rules! vprintln {
    ($verbose:expr, $($arg:tt)*) => {
        if $verbose {
            println!($($arg)*);
        }
    };
}

#[derive(Debug, Clone)]
struct StackFrame {
    pc: u64,
    sp: u64,
    fp: u64,
    function_name: String,
}

#[derive(Debug, Clone)]
struct StackCapture {
    call_number: usize,
    event_type: String, // "entry" or "return"
    function_name: String,
    frames: Vec<StackFrame>,
    stack_memory: Vec<u64>,
    stack_base: u64,
}

// Message types from debug-frontend
#[derive(Debug, Clone, Encode, Decode)]
pub struct Message {
    kind: String,
    data: Data,
}

#[derive(Debug, Encode, Decode, Clone)]
enum Data {
    ForeignFunction {
        name: String,
        pointer: usize,
    },
    BuiltinFunction {
        name: String,
        pointer: usize,
    },
    HeapSegmentPointer {
        pointer: usize,
    },
    UserFunction {
        name: String,
        pointer: usize,
        len: usize,
        number_of_arguments: usize,
    },
    Label {
        label: String,
        function_pointer: usize,
        label_index: usize,
        label_location: usize,
    },
    StackMap {
        pc: usize,
        name: String,
        stack_map: Vec<(usize, StackMapDetails)>,
    },
    Allocate {
        bytes: usize,
        stack_pointer: usize,
        kind: String,
    },
    Tokens {
        file_name: String,
        tokens: Vec<String>,
        token_line_column_map: Vec<(usize, usize)>,
    },
    Ir {
        function_pointer: usize,
        file_name: String,
        instructions: Vec<String>,
        token_range_to_ir_range: Vec<((usize, usize), (usize, usize))>,
    },
    Arm {
        function_pointer: usize,
        file_name: String,
        instructions: Vec<String>,
        ir_to_machine_code_range: Vec<(usize, (usize, usize))>,
    },
}

#[derive(Debug, Encode, Decode, Clone)]
pub struct StackMapDetails {
    pub function_name: Option<String>,
    pub number_of_locals: usize,
    pub current_stack_size: usize,
    pub max_stack_size: usize,
}

trait Serialize {
    fn to_binary(&self) -> Vec<u8>;
    fn from_binary(data: &[u8]) -> Result<Self> where Self: Sized;
}

impl<T: Encode + Decode<()>> Serialize for T {
    fn to_binary(&self) -> Vec<u8> {
        bincode::encode_to_vec(self, standard()).unwrap()
    }
    
    fn from_binary(data: &[u8]) -> Result<T> {
        let (decoded, _) = bincode::decode_from_slice(data, standard())
            .map_err(|e| anyhow!("Deserialization error: {}", e))?;
        Ok(decoded)
    }
}

// Helper traits from the working debug-frontend
trait FrameExtensions {
    fn get_register(&self, name: &str) -> Option<SBValue>;
}

impl FrameExtensions for lldb::SBFrame {
    fn get_register(&self, name: &str) -> Option<SBValue> {
        for register in self.registers().into_iter() {
            if matches!(register.name(), Some(n) if n == name) {
                return Some(register);
            }
            for child in register.children().into_iter() {
                if matches!(child.name(), Some(n) if n == name) {
                    return Some(child);
                }
            }
        }
        None
    }
}

trait ValueExtensions {
    fn to_usize(&self) -> usize;
}

impl ValueExtensions for SBValue {
    fn to_usize(&self) -> usize {
        unsafe { SBValueGetValueAsUnsigned2(self.raw, 0) as usize }
    }
}

trait ProcessExtensions {
    fn read_memory(&self, address: u64, buffer: &mut [u8]) -> usize;
}

impl ProcessExtensions for SBProcess {
    fn read_memory(&self, address: u64, buffer: &mut [u8]) -> usize {
        let ptr: *mut c_void = buffer.as_mut_ptr() as *mut c_void;
        let count = buffer.len();
        let error = SBError::default();
        unsafe { SBProcessReadMemory(self.raw, address, ptr, count, error.raw) }
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    
    println!("🔍 Stack Trace Collector");
    println!("Program: {}", args.program);
    println!("Args: {:?}", args.args);
    println!("Max captures: {}", args.max_calls);
    println!();
    
    // Set environment variables before LLDB initialization to suppress logging
    if !args.verbose {
        std::env::set_var("LLDB_LOG", "");
        std::env::set_var("LLDB_DEBUGSERVER_LOG_FILE", "/dev/null");
        std::env::set_var("LLDB_DISABLE_PYTHON", "1");
    }
    
    // Initialize LLDB
    SBDebugger::initialize();
    let debugger = SBDebugger::create(false);
    debugger.set_asynchronous(false);
    
    // Try to disable DWARF logging that generates DIE warnings
    if !args.verbose {
        // The enable_log with empty array should disable logging for that channel
        let _ = debugger.enable_log("dwarf", &[]);
        let _ = debugger.enable_log("symbol", &[]);
        let _ = debugger.enable_log("types", &[]);
        let _ = debugger.enable_log("module", &[]);
    }
    
    // Create target
    vprintln!(args.verbose, "Creating target for: {}", args.program);
    let target = debugger.create_target_simple(&args.program)
        .ok_or_else(|| anyhow!("Failed to create target for {}", args.program))?;
    
    // Set breakpoint on debugger_info to collect function information
    vprintln!(args.verbose, "Setting breakpoint on debugger_info...");
    let breakpoint = unsafe {
        let name = CString::new("debugger_info").unwrap();
        let module_name = CString::new("main").unwrap();
        let bp_raw = SBTargetBreakpointCreateByName(target.raw, name.as_ptr(), module_name.as_ptr());
        if bp_raw.is_null() {
            return Err(anyhow!("Failed to create breakpoint on debugger_info"));
        }
        lldb::SBBreakpoint { raw: bp_raw }
    };
    breakpoint.set_enabled(true);
    
    // Launch process
    vprintln!(args.verbose, "Launching process...");
    let launch_info = SBLaunchInfo::new();
    let program_args: Vec<&str> = args.args.iter().map(|s| s.as_str()).collect();
    launch_info.set_arguments(program_args, false);
    
    let process = target.launch(launch_info)
        .map_err(|e| anyhow!("Failed to launch process: {:?}", e))?;
    
    vprintln!(args.verbose, "Process launched successfully. PID: {}", process.process_id());
    
    let mut stack_captures = Vec::new();
    let mut call_count = 0;
    let mut functions: HashMap<usize, String> = HashMap::new();
    let mut function_ranges: HashMap<String, (usize, usize)> = HashMap::new(); // name -> (start, end)
    let mut active_functions: Vec<String> = Vec::new(); // Stack of currently active functions
    
    // Main debugging loop
    loop {
        if !process.is_stopped() {
            if !process.is_alive() {
                vprintln!(args.verbose, "Process is not alive, exiting...");
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
            continue;
        }
        
        // Check if process has exited
        let exit_status = process.exit_status();
        if exit_status != -1 {
            println!("Process exited with status: {}", exit_status);
            break;
        }
        
        vprintln!(args.verbose, "Process stopped, analyzing...");
        
        vprintln!(args.verbose, "Looking through all threads...");
        let mut found_debugger_thread = false;
        for thread in process.threads() {
            let thread_id = thread.index_id();
            vprintln!(args.verbose, "Checking thread {}", thread_id);
            let frame = thread.selected_frame();
            let pc = frame.pc();
            let function_name = get_function_name_for_address(pc, &functions, &function_ranges, &frame);
            
            vprintln!(args.verbose, "Thread {} stopped at function: {}", thread_id, function_name);
            
            // Skip threads that are in system wait states
            if function_name.contains("semaphore_wait") || function_name.contains("_wait") {
                vprintln!(args.verbose, "Skipping thread {} (in wait state)", thread_id);
                continue;
            }
            
            found_debugger_thread = true;
            
            // Handle debugger_info function - extract function info and set breakpoints
            if function_name == "debugger_info" {
                vprintln!(args.verbose, "Processing debugger_info...");
                
                // Read the debug message from registers x0 and x1
                vprintln!(args.verbose, "Getting x0 register...");
                let x0_reg = frame.get_register("x0");
                vprintln!(args.verbose, "Getting x1 register...");
                let x1_reg = frame.get_register("x1");
                
                if let (Some(x0_reg), Some(x1_reg)) = (x0_reg, x1_reg) {
                    vprintln!(args.verbose, "Got both registers, converting to values...");
                    let x0 = x0_reg.to_usize() as u64;
                    let x1 = x1_reg.to_usize();
                    
                    vprintln!(args.verbose, "Reading debug message at 0x{:x}, length {}", x0, x1);
                    
                    if x1 > 0 && x1 < 10000 { // Sanity check
                        vprintln!(args.verbose, "Sanity check passed, creating buffer of size {}", x1);
                        let mut buffer = vec![0u8; x1];
                        vprintln!(args.verbose, "Reading memory from process...");
                        let bytes_read = process.read_memory(x0, &mut buffer);
                        vprintln!(args.verbose, "Read {} bytes from memory", bytes_read);
                        
                        if bytes_read > 0 {
                            // Parse the actual message
                            vprintln!(args.verbose, "Attempting to parse message...");
                            match Message::from_binary(&buffer[..bytes_read]) {
                                Ok(message) => {
                                    vprintln!(args.verbose, "Successfully parsed message: {:?}", message.kind);
                                    
                                    // Handle UserFunction messages by setting breakpoints
                                    if let Data::UserFunction { name, pointer, len, number_of_arguments: _ } = &message.data {
                                        vprintln!(args.verbose, "Found user function: {} at 0x{:x} (len: {})", name, pointer, len);
                                        functions.insert(*pointer, name.clone());
                                        function_ranges.insert(name.clone(), (*pointer, pointer + len));
                                        
                                        // Set breakpoint on this function
                                        vprintln!(args.verbose, "Creating breakpoint at address 0x{:x}", pointer);
                                        let breakpoint = target.breakpoint_create_by_address(*pointer as u64);
                                        vprintln!(args.verbose, "Enabling breakpoint...");
                                        breakpoint.set_enabled(true);
                                        vprintln!(args.verbose, "Set breakpoint on function '{}' at 0x{:x}", name, pointer);
                                    }
                                    
                                    // Also handle BuiltinFunction and ForeignFunction if needed
                                    match &message.data {
                                        Data::BuiltinFunction { name, pointer } => {
                                            vprintln!(args.verbose, "Found builtin function: {} at 0x{:x}", name, pointer);
                                            functions.insert(*pointer, name.clone());
                                        }
                                        Data::ForeignFunction { name, pointer } => {
                                            vprintln!(args.verbose, "Found foreign function: {} at 0x{:x}", name, pointer);
                                            functions.insert(*pointer, name.clone());
                                        }
                                        _ => {}
                                    }
                                }
                                Err(e) => {
                                    vprintln!(args.verbose, "Failed to parse debug message: {}", e);
                                }
                            }
                        }
                    }
                }
                
                // Step over debugger_info and continue execution
                vprintln!(args.verbose, "Stepping over debugger_info...");
                thread.step_instruction(true)
                    .map_err(|e| anyhow!("Failed to step over debugger_info: {:?}", e))?;
                vprintln!(args.verbose, "Continuing execution...");
                process.continue_execution()
                    .map_err(|e| anyhow!("Failed to continue execution: {:?}", e))?;
                vprintln!(args.verbose, "Continued execution, going to next iteration");
                continue;
            }
            
            // Handle black_box function wrapper
            if function_name.contains("black_box") {
                vprintln!(args.verbose, "Handling black_box wrapper...");
                vprintln!(args.verbose, "First step through black_box...");
                thread.step_instruction(false)
                    .map_err(|e| anyhow!("Failed to step through black_box: {:?}", e))?;
                vprintln!(args.verbose, "Second step through black_box...");
                thread.step_instruction(false)
                    .map_err(|e| anyhow!("Failed to step through black_box: {:?}", e))?;
                vprintln!(args.verbose, "Completed black_box handling");
                continue;
            }
            
            // Get PC and determine function name
            let pc = frame.pc();
            
            // Check if we're in a tracked user function (entry or within)
            let _current_function = get_function_name_for_address(pc, &functions, &function_ranges, &frame);
            
            // Determine if this is a function entry, return, or continuation
            let mut event_type = None;
            let mut target_function = None;
            
            // Check for function entry (exact match to function start address)
            if let Some(tracked_func_name) = functions.get(&(pc as usize)) {
                event_type = Some("entry");
                target_function = Some(tracked_func_name.clone());
                vprintln!(args.verbose, "Function ENTRY: {} at 0x{:x}", tracked_func_name, pc);
            }
            // Check for function return (no longer in any active function)
            else {
                // See if we were previously in a function but are no longer
                for active_func in &active_functions {
                    if let Some((start, end)) = function_ranges.get(active_func) {
                        if (pc as usize) < *start || (pc as usize) >= *end {
                            event_type = Some("return");
                            target_function = Some(active_func.clone());
                            vprintln!(args.verbose, "Function RETURN: {} (pc=0x{:x} outside range 0x{:x}-0x{:x})", active_func, pc, start, end);
                            break;
                        }
                    }
                }
            }
            
            if let (Some(event), Some(func_name)) = (event_type, target_function) {
                // Capture stack trace using helper function
                match capture_stack_trace(&frame, &process, &functions, &function_ranges, call_count, &event, &func_name, args.verbose) {
                    Ok(capture) => {
                        stack_captures.push(capture);
                        call_count += 1;
                        println!("Captured stack trace #{} ({} {})", call_count, event, func_name);
                        
                        // Update active functions list
                        if event == "entry" {
                            active_functions.push(func_name.clone());
                        } else if event == "return" {
                            // Remove the function from active list
                            if let Some(pos) = active_functions.iter().position(|x| x == &func_name) {
                                active_functions.remove(pos);
                            }
                        }
                        
                        if args.max_calls > 0 && call_count >= args.max_calls {
                            vprintln!(args.verbose, "Reached maximum capture limit");
                            process.kill()
                                .map_err(|e| anyhow!("Failed to kill process: {:?}", e))?;
                            break;
                        }
                    }
                    Err(e) => {
                        vprintln!(args.verbose, "Failed to capture stack trace: {}", e);
                    }
                }
            }
            
            // Continue execution
            vprintln!(args.verbose, "Continuing execution...");
            process.continue_execution()
                .map_err(|e| anyhow!("Failed to continue execution: {:?}", e))?;
            break; // Process only the first non-waiting thread
        }
        
        if !found_debugger_thread {
            vprintln!(args.verbose, "No active debugger thread found, continuing execution...");
            process.continue_execution()
                .map_err(|e| anyhow!("Failed to continue execution: {:?}", e))?;
        }
    }
    
    // Write output
    println!("Writing {} stack captures to {}", stack_captures.len(), args.output.display());
    let mut output_file = File::create(&args.output)?;
    
    // Write as JSON
    writeln!(output_file, "{{")?;
    writeln!(output_file, "  \"program\": \"{}\",", args.program)?;
    writeln!(output_file, "  \"total_captures\": {},", stack_captures.len())?;
    writeln!(output_file, "  \"captures\": [")?;
    
    for (i, capture) in stack_captures.iter().enumerate() {
        writeln!(output_file, "    {{")?;
        writeln!(output_file, "      \"call_number\": {},", capture.call_number)?;
        writeln!(output_file, "      \"event_type\": \"{}\",", capture.event_type)?;
        writeln!(output_file, "      \"function_name\": \"{}\",", capture.function_name)?;
        writeln!(output_file, "      \"stack_base\": \"0x{:x}\",", capture.stack_base)?;
        writeln!(output_file, "      \"stack_depth\": {},", capture.frames.len())?;
        writeln!(output_file, "      \"frames\": [")?;
        
        for (j, frame) in capture.frames.iter().enumerate() {
            writeln!(output_file, "        {{")?;
            writeln!(output_file, "          \"depth\": {},", j)?;
            writeln!(output_file, "          \"function\": \"{}\",", frame.function_name)?;
            writeln!(output_file, "          \"pc\": \"0x{:x}\",", frame.pc)?;
            writeln!(output_file, "          \"sp\": \"0x{:x}\",", frame.sp)?;
            writeln!(output_file, "          \"fp\": \"0x{:x}\"", frame.fp)?;
            write!(output_file, "        }}")?;
            if j < capture.frames.len() - 1 {
                writeln!(output_file, ",")?;
            } else {
                writeln!(output_file)?;
            }
        }
        
        writeln!(output_file, "      ],")?;
        writeln!(output_file, "      \"stack_memory\": [")?;
        
        // Write first 50 values of stack memory as hex with addresses
        let preview_size = 50.min(capture.stack_memory.len());
        for (j, &value) in capture.stack_memory[..preview_size].iter().enumerate() {
            let addr = capture.stack_base + (j * 8) as u64;
            write!(output_file, "        {{\"addr\": \"0x{:x}\", \"value\": \"0x{:016x}\"}}", addr, value)?;
            if j < preview_size - 1 {
                writeln!(output_file, ",")?;
            } else {
                writeln!(output_file)?;
            }
        }
        
        writeln!(output_file, "      ]")?;
        write!(output_file, "    }}")?;
        
        if i < stack_captures.len() - 1 {
            writeln!(output_file, ",")?;
        } else {
            writeln!(output_file)?;
        }
    }
    
    writeln!(output_file, "  ]")?;
    writeln!(output_file, "}}")?;
    
    println!("Stack trace capture complete!");
    println!("Total function calls captured: {}", stack_captures.len());
    println!("Output written to: {}", args.output.display());
    
    Ok(())
}

// Helper function to find function name by address, checking our mapping first
fn get_function_name_for_address(
    pc: u64,
    functions: &HashMap<usize, String>,
    function_ranges: &HashMap<String, (usize, usize)>,
    frame: &lldb::SBFrame,
) -> String {
    let pc_usize = pc as usize;
    
    // First check exact match in our function mapping
    if let Some(name) = functions.get(&pc_usize) {
        return name.clone();
    }
    
    // Check if PC falls within any of our function ranges
    for (name, (start, end)) in function_ranges {
        if pc_usize >= *start && pc_usize < *end {
            return name.clone();
        }
    }
    
    // Only fall back to LLDB if not found in our mappings
    frame.function_name().unwrap_or("unknown").to_string()
}

// Helper function to capture a stack trace
fn capture_stack_trace(
    frame: &lldb::SBFrame,
    process: &lldb::SBProcess,
    functions: &HashMap<usize, String>,
    function_ranges: &HashMap<String, (usize, usize)>,
    call_count: usize,
    event_type: &str,
    function_name: &str,
    verbose: bool,
) -> Result<StackCapture, anyhow::Error> {
    let pc = frame.pc();
    let sp = frame.sp();
    let fp = frame.fp();
    
    vprintln!(verbose, "Capturing {} for {} (pc=0x{:x}, sp=0x{:x})", event_type, function_name, pc, sp);
    
    // Read stack memory
    vprintln!(verbose, "Calculating stack memory parameters...");
    let stack_size = 512; // Read 512 * 8 = 4KB of stack
    let stack_base = sp.saturating_sub(256 * 8); // Start 256 words before SP
    vprintln!(verbose, "Stack base: 0x{:x}, stack size: {} bytes", stack_base, stack_size * 8);
    let mut stack_buffer = vec![0u8; stack_size * 8];
    vprintln!(verbose, "Created stack buffer");
    
    let stack_memory = {
        vprintln!(verbose, "Reading memory from process...");
        let bytes_read = process.read_memory(stack_base, &mut stack_buffer);
        vprintln!(verbose, "Read {} bytes from process memory", bytes_read);
        if bytes_read > 0 {
            // Convert bytes to u64 values
            vprintln!(verbose, "Converting bytes to u64 values...");
            let mut memory = Vec::new();
            for chunk in stack_buffer[..bytes_read].chunks_exact(8) {
                let value = u64::from_le_bytes(chunk.try_into().unwrap());
                memory.push(value);
            }
            vprintln!(verbose, "Converted {} values", memory.len());
            memory
        } else {
            Vec::new()
        }
    };
    
    // Build call stack
    vprintln!(verbose, "Building call stack...");
    let mut frames = Vec::new();
    
    // Add current frame
    vprintln!(verbose, "Adding current frame...");
    vprintln!(verbose, "Frame data: pc=0x{:x}, sp=0x{:x}, fp=0x{:x}, name={}", pc, sp, fp, function_name);
    
    let stack_frame = StackFrame {
        pc,
        sp,
        fp,
        function_name: function_name.to_string(),
    };
    vprintln!(verbose, "Created StackFrame struct");
    
    frames.push(stack_frame);
    vprintln!(verbose, "Added current frame successfully");
    
    // Walk the stack to get calling functions
    let mut current_frame = frame.clone();
    let mut _frame_count = 0;
    loop {
        if let Some(parent) = current_frame.parent_frame() {
            _frame_count += 1;
            
            let parent_pc = parent.pc();
            let parent_function_name = get_function_name_for_address(parent_pc, functions, function_ranges, &parent);
            frames.push(StackFrame {
                pc: parent_pc,
                sp: parent.sp(),
                fp: parent.fp(),
                function_name: parent_function_name,
            });
            
            current_frame = parent;
        } else {
            break;
        }
    }
    
    Ok(StackCapture {
        call_number: call_count,
        event_type: event_type.to_string(),
        function_name: function_name.to_string(),
        frames,
        stack_memory,
        stack_base,
    })
}