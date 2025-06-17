use clap::Parser;
use debugger_frontend_claude::{DebuggerClient, BreakpointMapper, SourceMapper, Result};

#[derive(Parser)]
#[command(name = "debugger-breakpoints")]
#[command(about = "Test breakpoint mapping between source and machine code")]
struct Args {
    /// Path to the executable to debug
    program: String,
    
    /// Arguments to pass to the program
    #[arg(long)]
    args: Vec<String>,
    
    /// Source file and line in format "file.ext:line"
    #[arg(short, long)]
    source: Option<String>,
    
    /// Machine address to lookup source for (in hex)
    #[arg(short, long, value_parser = parse_hex_address)]
    address: Option<u64>,
    
    /// List all available breakpoint locations
    #[arg(short, long)]
    list: bool,
}

fn parse_hex_address(s: &str) -> std::result::Result<u64, std::num::ParseIntError> {
    if let Some(hex_str) = s.strip_prefix("0x") {
        u64::from_str_radix(hex_str, 16)
    } else {
        s.parse::<u64>()
    }
}

fn parse_source_location(s: &str) -> Result<(String, usize)> {
    let parts: Vec<&str> = s.split(':').collect();
    if parts.len() != 2 {
        return Err(anyhow::anyhow!("Source location must be in format 'file:line'"));
    }
    
    let file = parts[0].to_string();
    let line = parts[1].parse::<usize>()
        .map_err(|_| anyhow::anyhow!("Invalid line number"))?;
    
    Ok((file, line))
}

fn main() -> Result<()> {
    let args = Args::parse();
    
    // Create debugger client
    let mut client = DebuggerClient::new()?;
    
    // Create target and set initial breakpoint
    client.create_target(&args.program)?;
    client.set_breakpoint_by_name("debugger_info", "main")?;
    let args_refs: Vec<&str> = args.args.iter().map(|s| s.as_str()).collect();
    client.launch_process(args_refs)?;
    
    println!("Launched process, collecting debug information...");
    
    let _process = client.get_process()
        .ok_or_else(|| anyhow::anyhow!("No process found"))?;
    let _target = client.get_target()
        .ok_or_else(|| anyhow::anyhow!("No target found"))?;
    
    // Create source mapper and breakpoint mapper
    let source_mapper = SourceMapper::new();
    let breakpoint_mapper = BreakpointMapper::new();
    
    // TODO: In a real implementation, we would:
    // 1. Set up message handling to collect debug info
    // 2. Run the process and collect all debug messages
    // 3. Build the complete source mapping
    
    // For now, demonstrate the API
    if args.list {
        println!("Available breakpoint locations:");
        println!("(This would show all mapped source locations once debug info is collected)");
        println!();
        println!("Example output:");
        println!("  main.bg:10 -> 0x100154000");
        println!("  main.bg:15 -> 0x100154010");
        println!("  main.bg:20 -> 0x100154020");
    }
    
    if let Some(ref source) = args.source {
        let (file, line) = parse_source_location(&source)?;
        
        println!("Looking up source location {}:{}", file, line);
        
        if let Some(address) = breakpoint_mapper.address_by_file_line(&file, line) {
            println!("Source {}:{} maps to address 0x{:x}", file, line, address);
            
            // Try to set a breakpoint at this address
            match client.set_breakpoint_by_address(address) {
                Ok(()) => println!("Breakpoint set successfully"),
                Err(e) => println!("Failed to set breakpoint: {}", e),
            }
        } else {
            println!("No mapping found for {}:{}", file, line);
            println!("This could mean:");
            println!("  - The line has no corresponding machine code");
            println!("  - Debug information hasn't been collected yet");
            println!("  - The file/line combination is invalid");
        }
    }
    
    if let Some(address) = args.address {
        println!("Looking up address 0x{:x}", address);
        
        if let Some((file, line)) = breakpoint_mapper.file_line_by_address(address) {
            println!("Address 0x{:x} maps to {}:{}", address, file, line);
        } else {
            println!("No source mapping found for address 0x{:x}", address);
        }
        
        // Also try to get function information
        if let Some((file, line, function)) = source_mapper.get_current_location(address) {
            println!("Context: {}:{} in function '{}'", file, line, function);
        }
    }
    
    if !args.list && args.source.is_none() && args.address.is_none() {
        println!("No action specified. Use --help for usage information.");
        println!("Example usage:");
        println!("  {} {} --source main.bg:10", 
                 env!("CARGO_BIN_NAME"), args.program);
        println!("  {} {} --address 0x100154000", 
                 env!("CARGO_BIN_NAME"), args.program);
        println!("  {} {} --list", 
                 env!("CARGO_BIN_NAME"), args.program);
    }
    
    Ok(())
}