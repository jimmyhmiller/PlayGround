
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

#[allow(missing_docs)]
pub fn display_function_name(frame: &SBFrame) -> Option<String> {
    unsafe {
        let function_name = sys::SBFrameGetDisplayFunctionName(frame.raw);
        if function_name.is_null() {
            None
        } else {
            Some(CStr::from_ptr(function_name).to_str().unwrap().to_string())
        }
    }
}

fn main() {
    SBDebugger::initialize();



    let debugger = SBDebugger::create(false);

    // debugger.set_asynchronous(false);
    // debugger.enable_log("lldb", &["default"]);
    // println!("{:?}", debugger);

    unsafe {
        let file_name = CString::new("/Users/jimmyhmiller/Downloads/debugger.txt".to_string()).unwrap();
        let mode = CString::new("r").unwrap();
        let file = libc::fopen(file_name.as_ptr(), mode.as_ptr());
        sys::SBDebuggerSetOutputFileHandle(debugger.raw, file, true);
     }

    // let target = debugger.create_target_simple("/Users/jimmyhmiller/Documents/open-source/ruby/ruby");
    // let target = debugger.create_target_simple("/Users/jimmyhmiller/Documents/Code/PlayGround/rust/editor/target/debug/editor");
    let target = debugger.create_target_simple("/Users/jimmyhmiller/Documents/Code/ruby/ruby");


    if let Some(target) = target {
        println!("{:?}", target);


        // I need to hold onto these strings or things will get deallocated.
        // This works, is there an easier way? No idea.
        let my_strings = vec!["--jit".to_string(), "--yjit-call-threshold=1".to_string(), "/Users/jimmyhmiller/Documents/Code/ruby/my_file.rb".to_string()];
        let cstrs: Vec<CString> = my_strings.into_iter().map(|a| CString::new(a).unwrap()).collect();
        let mut ptrs: Vec<*const c_char> = cstrs.iter().map(|cs| cs.as_ptr()).collect();
        ptrs.push(std::ptr::null());

        // let arg = CString::new("/Users/jimmyhmiller/Documents/Code/ruby/my_file.rb");
        // let arg_pointer = arg.unwrap();



        let launchinfo = SBLaunchInfo::new();
        // launchinfo.set_launch_flags(LaunchFlags::CLOSE_TTY_ON_EXIT);
        // launchinfo.set_launch_flags(LaunchFlags::STOP_AT_ENTRY);
        unsafe {
            // pass the pointer of the vector's internal buffer to a C function
            sys::SBLaunchInfoSetArguments(launchinfo.raw, ptrs.as_ptr(), false);
        };
        // println!("{:?}", unsafe { CStr::from_ptr(sys::SBLaunchInfoGetArgumentAtIndex(launchinfo.raw, 1)) });
        println!("{}", unsafe {sys::SBLaunchInfoGetNumArguments(launchinfo.raw)});

        // println!("functions!");

        // println!("{:?}", target.breakpoints().collect::<Vec<_>>());
        // target.enable_all_breakpoints();
        // println!("End functions");
        // launchinfo.set_arguments(&["/Users/jimmyhmiller/Documents/open-source/ruby/ruby"]);
        match target.launch(launchinfo) {
            Ok(process) => {

                // thread::sleep(Duration::from_secs(2));
                for function in target.find_functions("gen_single_block", 2).iter() {
                    println!("Func {:?}", function);
                    let address = function.function().start_address();
                    let breakpoint = breakpoint_create_by_sbaddress(&target, address);
                    breakpoint.set_enabled(true);
                }
                // let interpreter = debugger.command_interpreter();
                // interpreter.handle_command("process handle -n main -p 0");
                // process.detach();
                println!("here {:?} {:?} {:?} {:?} {:?}", process.process_info(), process.exit_status(), process.is_alive(), process.is_running(), process.state());
                // process.continue_execution();
                // thread::sleep(Duration::from_secs(10));
                loop {
                    let mut buffer = [0; 1024];
                    let stdout = unsafe {sys::SBProcessGetSTDOUT(process.raw, buffer.as_mut_ptr(), buffer.len())};
                    // println!("stdout: {:?} {:?}", stdout, buffer);
                    if stdout > 0 {
                        println!("===============\nstdout: {:?}===============", unsafe { CStr::from_ptr(buffer[..stdout as usize].as_ptr())});
                    }
                    wait_for_enter();
                     for thread in process.threads() {
                        if !thread.is_valid() { println!("Not valid"); continue; }
                        for frame in thread.frames() {
                            if !frame.is_valid() { continue; }
                            println!("{:?}", frame.function());
                            println!("{:?}", display_function_name(&frame));
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
                    println!("{:?}", process.continue_execution());
                    // println!("{:?}", process.detach());
                    println!("here {:?} {:?} {:?} {:?} {:?}", process, process.exit_status(), process.is_alive(), process.is_running(), process.state());

                    // println!("here {:?} {:?} {:?} {:?} {:?}", process, process.exit_status(), process.is_alive(), process.is_running(), process.state());
                    target.disable_all_breakpoints();
                    // for thread in process.threads() {
                    //     println!("{:?}", thread.resume());
                    //     println!("{:?}", step_over(&thread, RunMode::OnlyThisThread));
                    // }

                    // for thread in process.threads() {
                    //     if !thread.is_valid() { println!("Not valid"); continue; }
                    //     for frame in thread.frames() {
                    //         // if !frame.is_valid() {  println!("Not valid frame"); continue; }
                    //         println!("{:?}", frame.display_function_name());
                    //     }
                    // }
                }
                // println!("here {:?} {:?} {:?} {:?} {:?}", process, process.exit_status(), process.is_alive(), process.is_running(), process.state());
                // process.continue_execution().unwrap();
                // process.continue_execution().unwrap();
                // process.continue_execution().unwrap();
                // process.continue_execution().unwrap();
                // process.continue_execution().unwrap();
                // process.continue_execution().unwrap();
                // process.continue_execution().unwrap();
                // wait_for_enter();



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
