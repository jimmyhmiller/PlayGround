use clap::Parser;
use debugger_frontend_claude::{DebuggerClient, MemoryInspector, ProcessState, ValueFormatter, ProcessExtensions, Result};

#[derive(Parser)]
#[command(name = "debugger-memory")]
#[command(about = "Inspect memory at a specific address")]
struct Args {
    /// Path to the executable to debug
    program: String,
    
    /// Arguments to pass to the program
    #[arg(long)]
    args: Vec<String>,
    
    /// Memory address to inspect (in hex, e.g., 0x12345678)
    #[arg(value_parser = parse_hex_address)]
    address: Option<u64>,
    
    /// Number of bytes to read
    #[arg(short, long, default_value = "64")]
    size: usize,
    
    /// Show stack memory instead of arbitrary address
    #[arg(long)]
    stack: bool,
    
    /// Show heap memory instead of arbitrary address  
    #[arg(short = 'H', long)]
    heap: bool,
    
    /// Show type information for values
    #[arg(short, long)]
    types: bool,
}

fn parse_hex_address(s: &str) -> std::result::Result<u64, std::num::ParseIntError> {
    if let Some(hex_str) = s.strip_prefix("0x") {
        u64::from_str_radix(hex_str, 16)
    } else {
        s.parse::<u64>()
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    
    // Create debugger client
    let mut client = DebuggerClient::new()?;
    
    // Create target and launch process
    client.create_target(&args.program)?;
    client.set_breakpoint_by_name("main", "main")?;
    let args_refs: Vec<&str> = args.args.iter().map(|s| s.as_str()).collect();
    client.launch_process(args_refs)?;
    
    println!("Launched process, waiting for breakpoint...");
    
    let process = client.get_process()
        .ok_or_else(|| anyhow::anyhow!("No process found"))?;
    let target = client.get_target()
        .ok_or_else(|| anyhow::anyhow!("No target found"))?;
    
    // Update process state
    let mut state = ProcessState::new();
    state.update_from_process(process, target)?;
    
    // Create memory inspector  
    let mut inspector = MemoryInspector::new();
    
    if args.stack {
        // Show stack memory
        let stack_data = process.read_memory_as_u64_array(state.sp.saturating_sub(256), 512)?;
        inspector.update_stack(state.sp.saturating_sub(256), &stack_data);
        
        println!("Stack memory around SP 0x{:x}:", state.sp);
        println!("FP: 0x{:x}", state.fp);
        println!();
        
        let formatted = inspector.format_stack_with_pointers(state.sp, state.fp);
        for line in formatted {
            println!("{}", line);
            if args.types {
                // Extract value from the line and show type info
                if let Some(value_start) = line.find("0x") {
                    if let Some(value_end) = line[value_start..].find(' ') {
                        if let Ok(value) = u64::from_str_radix(&line[value_start+2..value_start+value_end], 16) {
                            println!("    -> {}", ValueFormatter::format_type_info(value));
                        }
                    }
                }
            }
        }
    } else if args.heap {
        // Show heap memory (would need heap pointers from debug messages)
        println!("Heap memory inspection not yet implemented");
        println!("Need to collect heap pointers from debug messages first");
    } else if let Some(address) = args.address {
        // Show memory at specific address
        let memory_data = process.read_memory_as_u64_array(address, args.size)?;
        
        println!("Memory at address 0x{:x} ({} bytes):", address, args.size);
        println!();
        
        for (i, value) in memory_data.iter().enumerate() {
            let addr = address + (i as u64 * 8);
            let formatted = ValueFormatter::format_memory(
                &debugger_frontend_claude::Memory::new(addr, *value)
            );
            println!("{}", formatted);
            
            if args.types {
                println!("    -> {}", ValueFormatter::format_type_info(*value));
            }
        }
    } else {
        println!("Please specify either --stack, --heap, or provide a memory address");
        println!("Use --help for more information");
    }
    
    Ok(())
}