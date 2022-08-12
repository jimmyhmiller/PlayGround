use lldb::{SBDebugger, SBLaunchInfo, FunctionNameType};


pub fn wait_for_enter() {
    let mut input = String::new();
    println!("Press ENTER to continue...");
    std::io::stdin().read_line(&mut input).unwrap();
}

// I currently can't get this to build. Not clear how I should.
// These bindings are way better than the original
// But I might just want to make my own bindings of lldb-sys

fn main() {
    let debugger = SBDebugger::create(true);


    let target = debugger.create_target(
        "/Users/jimmyhmiller/Documents/Code/PlayGround/rust/test-prog/target/debug/test-prog",
        None, 
        None, 
        true
    ).unwrap();

    let functions = target.find_functions("write_second", FunctionNameType::Full);
    for function in functions.iter() {
        println!("{:?}", function);
        let line_entry = function.line_entry();
        let breakpoint = target.breakpoint_create_by_address(&line_entry.start_address());
        // breakpoint.
    }
    

    let launch_info = SBLaunchInfo::new();
    let process = target.launch(&launch_info).unwrap();
    loop {
        wait_for_enter();
        println!("{:?}", process);
    }


    println!("Hello, world!");
}
