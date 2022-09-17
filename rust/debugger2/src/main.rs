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
        "/Users/jimmyhmiller/Documents/Code/ruby/ruby",
        None,
        None,
        true
    ).unwrap();

    // let functions = target.find_functions("gen_single_block", FunctionNameType::Full);
    // for function in functions.iter() {
    //     println!("{:?}", function);
    //     let line_entry = function.line_entry();
    //     let breakpoint = target.breakpoint_create_by_address(&line_entry.start_address());
    //     // breakpoint.
    // }


    let mut launch_info = SBLaunchInfo::new();
    launch_info.set_arguments(vec!["/Users/jimmyhmiller/Documents/Code/ruby/my_file.rb"], false);
    let process = target.launch(&launch_info).unwrap();
    loop {
        wait_for_enter();
        println!("{:?}", process);
    }


}
