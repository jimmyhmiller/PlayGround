use std::ffi::CStr;

use lldb::{SBDebugger, SBLaunchInfo, FunctionNameType, RunMode, LaunchFlag, IsValid};


pub fn wait_for_enter() {
    let mut input = String::new();
    println!("Press ENTER to continue...");
    std::io::stdin().read_line(&mut input).unwrap();
}

// I currently can't get this to build. Not clear how I should.
// These bindings are way better than the original
// But I might just want to make my own bindings of lldb-sys

fn main() {
    SBDebugger::initialize();
    let debugger = SBDebugger::create(true);


    let target = debugger.create_target(
        "/Users/jimmyhmiller/Documents/Code/ruby/ruby",
        None,
        None,
        true
    ).unwrap();


    let functions = target.find_functions("main", FunctionNameType::Base);
    for function in functions.iter() {
        println!("{:?}", function);
        let line_entry = function.line_entry();
        let breakpoint = target.breakpoint_create_by_address(&line_entry.start_address());
        for location in breakpoint.locations() {
            location.set_enabled(true)
        }
    }



    let mut launch_info = SBLaunchInfo::new();
    // launch_info.set_launch_flags(LaunchFlag::StopAtEntry);
    launch_info.set_arguments(vec!["--yjit", "--yjit-call-threshold=1","/Users/jimmyhmiller/Documents/Code/ruby/my_file.rb"], true);
    let process = target.launch(&launch_info).unwrap();
    let thread = process.thread_at_index(1);

    loop {
        let mut buffer = [0; 1024];
        let stdout = process.read_stderr(&mut buffer);
        if stdout > 0 {
            println!("===============\nstdout: {:?}===============", unsafe { CStr::from_ptr(buffer[..stdout as usize].as_ptr() as *const i8)});
        }
        wait_for_enter();
        for thread in process.threads() {
            for frame in thread.frames() {
                println!("frame: {:?}", frame.pc());
            }
            println!("{:?}", thread.step_over(RunMode::OnlyDuringStepping));

            println!("{}", thread.is_valid());
        }
        println!("{:?}", process.is_valid());
        // thread.step_into(RunMode::OnlyDuringStepping)
    }


}
