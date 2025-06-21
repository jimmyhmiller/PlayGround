use debugger_frontend_claude::{DebuggerClient, Result};
use std::collections::HashSet;

pub fn test_breakpoints(program_path: &str, program_args: Vec<&str>, target_functions: Vec<&str>) -> Result<()> {
    println!("=== Testing Breakpoints ===");
    println!("Program: {}", program_path);
    println!("Args: {:?}", program_args);
    println!("Target functions: {:?}", target_functions);
    println!();
    
    // Create debugger client
    let mut client = DebuggerClient::new()?;
    
    // Create target and set the initial debugger_info breakpoint
    client.create_target(program_path)?;
    client.set_breakpoint_by_name("debugger_info", "main")?;
    println!("✓ Set debugger_info breakpoint");
    
    client.launch_process(program_args)?;
    println!("✓ Launched process");
    
    let target_set: HashSet<String> = target_functions.iter().map(|s| s.to_string()).collect();
    let mut breakpoints_set = HashSet::new();
    let mut found_functions = Vec::new();
    let mut iterations = 0;
    const MAX_ITERATIONS: usize = 200;
    
    loop {
        iterations += 1;
        if iterations > MAX_ITERATIONS {
            println!("🚨 Reached maximum iterations ({}), stopping", MAX_ITERATIONS);
            break;
        }
        if client.is_process_stopped() {
            println!("📍 Process stopped, collecting debug info...");
            
            // Collect debug info
            client.collect_debug_info()?;
            
            // Check current functions
            let current_functions = client.list_beagle_functions();
            if current_functions.len() > found_functions.len() {
                println!("🔍 Discovered {} total functions:", current_functions.len());
                for func in &current_functions {
                    if !found_functions.contains(func) {
                        println!("  - {}", func);
                        found_functions.push(func.clone());
                    }
                }
                
                // Print all functions if we have a reasonable number
                if current_functions.len() > 50 {
                    println!("📋 All discovered functions:");
                    for (i, func) in current_functions.iter().enumerate() {
                        println!("  {}: {}", i + 1, func);
                        if target_set.contains(func) {
                            println!("    ⭐ THIS IS A TARGET FUNCTION!");
                        }
                    }
                }
            }
            
            // Set breakpoints on newly discovered target functions
            for function_name in &current_functions {
                if target_set.contains(function_name) && !breakpoints_set.contains(function_name) {
                    println!("🎯 Attempting to set breakpoint on target function: {}", function_name);
                    
                    match client.set_breakpoint_on_beagle_function(function_name) {
                        Ok(true) => {
                            breakpoints_set.insert(function_name.clone());
                            println!("✅ Successfully set breakpoint on {}", function_name);
                        }
                        Ok(false) => {
                            println!("❌ Function {} not found for breakpoint", function_name);
                        }
                        Err(e) => {
                            println!("❌ Failed to set breakpoint on {}: {}", function_name, e);
                        }
                    }
                }
            }
            
            // Check if we're currently in a target function
            if let Some(process) = client.get_process() {
                if let Some(thread) = process.thread_by_index_id(1) {
                    let current_function_name = thread.selected_frame()
                        .function_name()
                        .unwrap_or("")
                        .to_string();
                    
                    println!("📍 Currently in function: {}", current_function_name);
                    
                    if let Some(target) = target_set.iter().find(|f| current_function_name.contains(*f)) {
                        println!("🎉 HIT TARGET FUNCTION: {}", target);
                        
                        // Get disassembly
                        match client.get_function_disassembly(target) {
                            Ok(disasm) => {
                                println!("📋 Disassembly for {}:", target);
                                for (i, line) in disasm.iter().take(10).enumerate() {
                                    println!("  {}: {}", i, line);
                                }
                                if disasm.len() > 10 {
                                    println!("  ... ({} more lines)", disasm.len() - 10);
                                }
                            }
                            Err(e) => {
                                println!("❌ Failed to get disassembly: {}", e);
                            }
                        }
                        
                        return Ok(());
                    }
                }
            }
            
            // Continue execution
            match client.continue_execution() {
                Ok(_) => {
                    println!("▶️  Continuing execution...");
                }
                Err(_) => {
                    println!("🏁 Program finished");
                    break;
                }
            }
        } else {
            std::thread::sleep(std::time::Duration::from_millis(10));
            
            if let Some(process) = client.get_process() {
                if !process.is_alive() {
                    println!("🏁 Process finished");
                    break;
                }
            }
        }
    }
    
    println!();
    println!("=== Summary ===");
    println!("Total functions discovered: {}", found_functions.len());
    println!("Breakpoints set: {}", breakpoints_set.len());
    for bp in &breakpoints_set {
        println!("  - {}", bp);
    }
    
    if breakpoints_set.is_empty() {
        println!("❌ No breakpoints were set!");
    } else {
        println!("⚠️  Breakpoints were set but never hit");
    }
    
    Ok(())
}