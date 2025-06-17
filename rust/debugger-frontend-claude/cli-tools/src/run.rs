use clap::Parser;
use debugger_frontend_claude::{
    DebuggerClient, ProcessState, DisassemblyAnalyzer, MemoryInspector, 
    SourceMapper, ValueFormatter, RegisterFilter, Result
};
use std::io::{self, Write};

#[derive(Parser)]
#[command(name = "debugger-run")]
#[command(about = "Interactive debugging session")]
struct Args {
    /// Path to the executable to debug
    program: String,
    
    /// Arguments to pass to the program
    #[arg(long)]
    args: Vec<String>,
    
    /// Function regex filter for breakpoints
    #[arg(short, long, default_value = ".*")]
    function_filter: String,
}

fn main() -> Result<()> {
    let args = Args::parse();
    
    // Create debugger client
    let mut client = DebuggerClient::new()?;
    
    // Create target and set initial breakpoint
    client.create_target(&args.program)?;
    client.set_breakpoint_by_name("main", "main")?;
    
    let program_args: Vec<&str> = args.args.iter().map(|s| s.as_str()).collect();
    client.launch_process(program_args)?;
    
    println!("🔧 Debugger Frontend Claude - Interactive Session");
    println!("Program: {}", args.program);
    println!("Type 'help' for commands, 'quit' to exit");
    println!();
    
    // Initialize components
    let mut state = ProcessState::new();
    let mut disasm = DisassemblyAnalyzer::new();
    let mut memory = MemoryInspector::new();
    let source_mapper = SourceMapper::new();
    
    let process = client.get_process()
        .ok_or_else(|| anyhow::anyhow!("No process found"))?;
    let target = client.get_target()
        .ok_or_else(|| anyhow::anyhow!("No target found"))?;
    
    // Main debugging loop
    loop {
        // Update state if process is stopped
        if client.is_process_stopped() {
            state.update_from_process(process, target)?;
            
            // Update components
            for instruction in &state.instructions {
                disasm.add_instruction(instruction.clone());
            }
            
            memory.update_stack(state.sp.saturating_sub(256), &state.stack_memory
                .iter().map(|m| m.value).collect::<Vec<_>>());
        }
        
        // Show current status
        show_status(&state, &disasm, &source_mapper)?;
        
        // Get user command
        print!("(debugger) ");
        io::stdout().flush()?;
        
        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let command = input.trim();
        
        match command {
            "help" | "h" => show_help(),
            "quit" | "q" => break,
            "continue" | "c" => {
                println!("Continuing execution...");
                client.continue_execution()?;
            }
            "step" | "s" => {
                println!("Stepping over...");
                client.step_over()?;
            }
            "stepi" | "si" => {
                println!("Stepping into...");
                client.step_into()?;
            }
            "disasm" | "dis" => show_disassembly(&state, &disasm, &source_mapper),
            "registers" | "reg" => show_registers(&state),
            "stack" => show_stack(&memory, state.sp, state.fp),
            "memory" | "mem" => show_memory(&memory),
            "info" => show_detailed_info(&state, &source_mapper),
            "" => {
                // Repeat last command (step)
                println!("Stepping over...");
                client.step_over()?;
            }
            _ => println!("Unknown command: {}. Type 'help' for available commands.", command),
        }
        
        println!();
    }
    
    println!("Debugger session ended.");
    Ok(())
}

fn show_status(state: &ProcessState, disasm: &DisassemblyAnalyzer, mapper: &SourceMapper) -> Result<()> {
    println!("────────────────────────────────────────");
    
    // Show current location
    if let Some((file, line, function)) = mapper.get_current_location(state.pc) {
        println!("📍 {}:{} in {}", file, line, function);
    }
    
    println!("PC: 0x{:x}  SP: 0x{:x}  FP: 0x{:x}", state.pc, state.sp, state.fp);
    
    // Show current instruction
    if let Some(current_inst) = disasm.instructions.iter().find(|i| i.address == state.pc) {
        println!("=> {}", current_inst.to_string(true, false));
    }
    
    Ok(())
}

fn show_help() {
    println!("Available commands:");
    println!("  help, h        - Show this help");
    println!("  quit, q        - Exit debugger");
    println!("  continue, c    - Continue execution");
    println!("  step, s        - Step over instruction");
    println!("  stepi, si      - Step into instruction");
    println!("  disasm, dis    - Show disassembly");
    println!("  registers, reg - Show registers");
    println!("  stack          - Show stack memory");
    println!("  memory, mem    - Show heap memory");
    println!("  info           - Show detailed information");
    println!("  <enter>        - Repeat last step command");
}

fn show_disassembly(state: &ProcessState, disasm: &DisassemblyAnalyzer, mapper: &SourceMapper) {
    println!("Disassembly around PC:");
    let instructions = disasm.get_instructions_around_pc(state.pc, 10);
    
    for instruction in instructions {
        let formatted = disasm.format_instruction_with_labels(
            instruction, 
            state.pc, 
            &mapper.functions, 
            &mapper.labels
        );
        println!("  {}", formatted);
    }
}

fn show_registers(state: &ProcessState) {
    println!("Registers:");
    
    // Filter to show only relevant registers
    let mentioned = RegisterFilter::get_mentioned_registers(&state.instructions);
    let filtered = RegisterFilter::filter_registers(&state.registers, &mentioned);
    
    for register in filtered {
        println!("  {}", ValueFormatter::format_register(register));
    }
}

fn show_stack(memory: &MemoryInspector, sp: u64, fp: u64) {
    println!("Stack memory:");
    let formatted = memory.format_stack_with_pointers(sp, fp);
    
    for line in formatted.iter().take(20) {  // Show first 20 entries
        println!("  {}", line);
    }
    
    if formatted.len() > 20 {
        println!("  ... ({} more entries)", formatted.len() - 20);
    }
}

fn show_memory(memory: &MemoryInspector) {
    println!("Heap memory:");
    if memory.heap.is_empty() {
        println!("  No heap memory available");
        println!("  (Heap pointers are collected from debug messages)");
    } else {
        for mem in memory.heap.iter().take(10) {
            println!("  {}", ValueFormatter::format_memory(mem));
        }
    }
}

fn show_detailed_info(state: &ProcessState, mapper: &SourceMapper) {
    println!("Detailed Information:");
    println!("  PC: 0x{:x}", state.pc);
    println!("  SP: 0x{:x}", state.sp);
    println!("  FP: 0x{:x}", state.fp);
    println!("  Instructions loaded: {}", state.instructions.len());
    println!("  Stack entries: {}", state.stack_memory.len());
    println!("  Heap entries: {}", state.heap_memory.len());
    println!("  Debug messages: {}", state.messages.len());
    println!("  Functions mapped: {}", mapper.functions.len());
    println!("  Labels mapped: {}", mapper.labels.len());
}