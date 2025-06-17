use clap::Parser;
use debugger_frontend_claude::{DebuggerClient, Result};

#[derive(Parser)]
#[command(name = "debugger-disasm")]
#[command(about = "Show disassembly for a function in a debugged program")]
struct Args {
    /// Path to the executable to debug
    program: String,
    
    /// Arguments to pass to the program
    #[arg(long)]
    args: Vec<String>,
    
    /// Beagle function name to disassemble (e.g., "fib/fib")
    function: Option<String>,
    
    /// Number of instructions to show around the current location
    #[arg(short, long, default_value = "20")]
    window: usize,
    
    /// Show memory addresses
    #[arg(short, long)]
    addresses: bool,
    
    /// Show hex bytes
    #[arg(short = 'x', long)]
    hex: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();
    
    // Create debugger client
    let mut client = DebuggerClient::new()?;
    
    // Create target and launch process
    client.create_target(&args.program)?;
    // Set breakpoint on debugger_info function - try different module names
    client.set_breakpoint_by_name("debugger_info", "main")?;
    let args_refs: Vec<&str> = args.args.iter().map(|s| s.as_str()).collect();
    client.launch_process(args_refs)?;
    
    println!("Launched process, collecting debug info...");
    
    // Collect ALL debug information by letting the JIT program run to completion
    let mut program_finished = false;
    
    // Just continue execution until the program finishes
    while !program_finished {
        if client.is_process_stopped() {
            client.collect_debug_info()?;
            
            // Continue execution - if this fails, the program finished
            match client.continue_execution() {
                Ok(_) => {
                    // Successfully continued, keep going
                }
                Err(_) => {
                    program_finished = true;
                }
            }
        } else {
            // Process is still running, wait a bit
            std::thread::sleep(std::time::Duration::from_millis(10));
            
            // Check if process has exited naturally
            if let Some(process) = client.get_process() {
                if !process.is_alive() {
                    program_finished = true;
                }
            }
        }
    }
    
    // If a specific function was requested, show its disassembly
    if let Some(ref function_name) = args.function {
        match client.get_function_disassembly(function_name) {
            Ok(disasm) => {
                println!("Disassembly of function '{}':", function_name);
                for line in disasm {
                    println!("{}", line);
                }
            }
            Err(_) => {
                println!("Function '{}' not found", function_name);
            }
        }
    } else {
        let all_functions = client.list_beagle_functions();
        println!("Available functions ({} total):", all_functions.len());
        for function in &all_functions {
            println!("  - {}", function);
        }
    }
    
    Ok(())
}