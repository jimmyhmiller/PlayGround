use debugger_frontend_claude::Result;
use clap::Parser;

mod test_breakpoints;

#[derive(Parser)]
#[command(name = "test-breakpoints")]
#[command(about = "Test breakpoint functionality without UI")]
struct Args {
    /// Path to the executable to debug
    program: String,
    
    /// Arguments to pass to the program
    #[arg(long)]
    args: Vec<String>,
    
    /// Function names to set breakpoints on
    #[arg(long)]
    functions: Vec<String>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    
    let program_args: Vec<&str> = args.args.iter().map(|s| s.as_str()).collect();
    let target_functions: Vec<&str> = args.functions.iter().map(|s| s.as_str()).collect();
    
    test_breakpoints::test_breakpoints(&args.program, program_args, target_functions)?;
    
    Ok(())
}