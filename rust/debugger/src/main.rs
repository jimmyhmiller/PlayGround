
use lldb::*;
use lldb_sys as sys;
use std::{ffi::{CString, CStr}, os::raw::c_char, thread, time::Duration};



pub fn step_over(thread: &SBThread, stop_other_threads: RunMode) -> Result<(), SBError> {
    let error = SBError::default();
    unsafe { sys::SBThreadStepOver(thread.raw, stop_other_threads, error.raw) }
    if error.is_success() {
        Ok(())
    } else {
        Err(error)
    }
}

fn  to_c_array(args: Vec<String>) -> *const *const c_char {
    let cstr_argv: Vec<_> = args.iter()
    .map(|arg| CString::new(arg.as_str()).unwrap())
    .collect();

    let mut p_argv: Vec<_> = cstr_argv.iter() // do NOT into_iter()
        .map(|arg| arg.as_ptr())
        .collect();

    p_argv.push(std::ptr::null());

    p_argv.as_ptr()
}


pub fn wait_for_enter() {
    let mut input = String::new();
    println!("Press ENTER to continue...");
    std::io::stdin().read_line(&mut input).unwrap();
}


pub fn breakpoint_create_by_sbaddress(target: &SBTarget, address: SBAddress) -> SBBreakpoint {
    SBBreakpoint::from(unsafe {
        sys::SBTargetBreakpointCreateBySBAddress(target.raw, address.raw)
    })
}

fn main() {
    SBDebugger::initialize();



    let debugger = SBDebugger::create(false);
    // debugger.set_asynchronous(false);
    // debugger.enable_log("lldb", &["default"]);
    println!("{:?}", debugger);

    unsafe {
        let file_name = CString::new("/Users/jimmyhmiller/Downloads/debugger.txt".to_string()).unwrap();
        let mode = CString::new("r").unwrap();
        let file = libc::fopen(file_name.as_ptr(), mode.as_ptr());
        sys::SBDebuggerSetOutputFileHandle(debugger.raw, file, true);
     }

    // let target = debugger.create_target_simple("/Users/jimmyhmiller/Documents/open-source/ruby/ruby");
    // let target = debugger.create_target_simple("/Users/jimmyhmiller/Documents/Code/PlayGround/rust/editor/target/debug/editor");
    let target = debugger.create_target_simple("/Users/jimmyhmiller/Documents/Code/PlayGround/rust/test-prog/target/debug/test-prog");
    if let Some(target) = target {
        println!("{:?}", target);


        // let args = to_c_array(vec!["--yjit".to_string(), "/Users/jimmyhmiller/Documents/open-source/yjit-bench/benchmarks/binarytrees/benchmark.rb".to_string()]);


        let launchinfo = SBLaunchInfo::new();
        launchinfo.set_launch_flags(LaunchFlags::CLOSE_TTY_ON_EXIT);
        // unsafe {
        //     // pass the pointer of the vector's internal buffer to a C function
        //     sys::SBLaunchInfoSetArguments(launchinfo.raw, args, true);
        // };

        println!("functions!");
        for function in target.find_functions("write_second", 2).iter() {
            println!("Func {:?}", function);
            let address = function.function().start_address();
            let breakpoint = breakpoint_create_by_sbaddress(&target, address);
            breakpoint.set_enabled(true);
        }
        println!("{:?}", target.breakpoints().collect::<Vec<_>>());
        target.enable_all_breakpoints();
        println!("End functions");
        // launchinfo.set_arguments(&["/Users/jimmyhmiller/Documents/open-source/ruby/ruby"]);
        match target.launch(launchinfo) {
            Ok(process) => {
                thread::sleep(Duration::from_secs(10));
                // let interpreter = debugger.command_interpreter();
                // interpreter.handle_command("process handle -n main -p 0");
                // process.detach();
                println!("here {:?} {:?} {:?} {:?} {:?}", process, process.exit_status(), process.is_alive(), process.is_running(), process.state());
                // process.continue_execution();
                println!("here {:?} {:?} {:?} {:?} {:?}", process, process.exit_status(), process.is_alive(), process.is_running(), process.state());
                // thread::sleep(Duration::from_secs(10));
                loop {
                    wait_for_enter();
                     for thread in process.threads() {
                        if !thread.is_valid() { println!("Not valid"); continue; }
                        for frame in thread.frames() {
                            if !frame.is_valid() { continue; }
                            println!("{:?}", frame.function());
                            println!("{:?}", frame.display_function_name());
                            for var in frame.all_variables().iter() {
                                unsafe {
                                    match CStr::from_ptr(sys::SBValueGetName(var.raw)).to_str() {
                                        Ok(name) => {
                                            println!("{:?}", name)
                                        },
                                        Err(e) => println!("{:?}", e),
                                    }
                                }

                                unsafe {
                                    let value = sys::SBValueGetValue(var.raw);
                                    if value.is_null() { continue; }
                                    match CStr::from_ptr(value).to_str() {
                                        Ok(s) => {
                                            println!("{:?}", s);
                                        },
                                        _ => panic!("Invalid string?"),
                                    }
                                }
                                println!("A variable");
                                if !var.is_valid() { continue; }
                                println!("{:?}: {:?}", var.name(), var.value());
                            }
                        }
                    }
                    wait_for_enter();
                    process.continue_execution();

                    println!("here {:?} {:?} {:?} {:?} {:?}", process, process.exit_status(), process.is_alive(), process.is_running(), process.state());
                    // target.disable_all_breakpoints();  
                    for thread in process.threads() {
                        println!("{:?}", thread.resume());
                        // println!("{:?}", step_over(&thread, RunMode::OnlyThisThread));
                    }

                    // for thread in process.threads() {
                    //     if !thread.is_valid() { println!("Not valid"); continue; }
                    //     for frame in thread.frames() {
                    //         // if !frame.is_valid() {  println!("Not valid frame"); continue; }
                    //         println!("{:?}", frame.display_function_name());
                    //     }
                    // }
                }
                println!("here {:?} {:?} {:?} {:?} {:?}", process, process.exit_status(), process.is_alive(), process.is_running(), process.state());
                // process.continue_execution().unwrap();
                // process.continue_execution().unwrap();
                // process.continue_execution().unwrap();
                // process.continue_execution().unwrap();
                // process.continue_execution().unwrap();
                // process.continue_execution().unwrap();
                // process.continue_execution().unwrap();
                wait_for_enter();
            

              
                // wait_for_enter();


              
            
                // for thread in process.threads().skip(1) {
                //     if !thread.is_valid() { continue; }
                //     for frame in thread.frames() {
                //         if !frame.is_valid() { continue; }
                //         println!("{:?}", frame.display_function_name());
                //     }
                // }
                

                //     wait_for_enter();
                //     println!("here {:?}", process);
                }
                Err(e) => println!("Uhoh: {:?}", e),
        }
    }
    SBDebugger::terminate();
}