
use lldb::*;
use lldb_sys as sys;
use std::{ffi::{CString}, os::raw::c_char};

fn main() {
    SBDebugger::initialize();

    let debugger = SBDebugger::create(false);
    debugger.set_asynchronous(false);
    println!("{:?}", debugger);

    if let Some(target) = debugger.create_target_simple("/Users/jimmyhmiller/Documents/open-source/ruby/ruby") {
        println!("{:?}", target);



        let args = vec!["--yjit".to_string()];
        let c_strings = args.iter().map(|s| CString::new(s.clone()).unwrap()).collect::<Vec<CString>>();
        // convert the strings to raw pointers
        let c_args = c_strings.iter().map(|arg| arg.as_ptr()).collect::<Vec<*const c_char>>();

        let launchinfo = SBLaunchInfo::new();
        launchinfo.set_launch_flags(LaunchFlags::STOP_AT_ENTRY);
        unsafe {
            // pass the pointer of the vector's internal buffer to a C function
            sys::SBLaunchInfoSetArguments(launchinfo.raw, c_args.as_ptr(), true);
        };

        // launchinfo.set_arguments(&["/Users/jimmyhmiller/Documents/open-source/ruby/ruby"]);
        match target.launch(launchinfo) {
            Ok(process) => {
                println!("{:?}", process);
                let _ = process.continue_execution();
                println!("{:?}", process);
            }
            Err(e) => println!("Uhoh: {:?}", e),
        }
    }
    SBDebugger::terminate();
}