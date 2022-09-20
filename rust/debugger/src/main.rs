
use lldb_sys as sys;
use sys::{RunMode, SBErrorRef, SBThreadRef, DescriptionLevel};
use core::fmt;
use std::{ffi::{CString, CStr}, os::raw::c_char, thread::{self, sleep}, time::Duration};


struct SBError {
    raw: SBErrorRef,
}
impl SBError {
    fn new() -> Self {
        SBError {
            raw: unsafe { sys::CreateSBError() }
        }
    }

    fn is_success(&self) -> bool {
        unsafe { sys::SBErrorSuccess(self.raw) }
    }

    fn from(raw: SBErrorRef) -> Self {
        SBError {
            raw
        }
    }

    fn into_result(self) -> Result<(), SBError> {
        if self.is_success() {
            Ok(())
        } else {
            Err(self)
        }
    }
}


struct SBThread {
    raw: SBThreadRef,
}

impl SBThread {



    fn step_over(&self, stop_other_threads: RunMode) -> Result<(), SBError> {
        let error = SBError::new();
        unsafe { sys::SBThreadStepOver(self.raw, stop_other_threads, error.raw) }
        if error.is_success() {
            Ok(())
        } else {
            Err(error)
        }
    }


    fn step_into(&self, stop_other_threads: RunMode) -> Result<(), SBError> {
        let error = SBError::new();
        unsafe { sys::SBThreadStepInto(self.raw, stop_other_threads) }
        if error.is_success() {
            Ok(())
        } else {
            Err(error)
        }
    }

    fn step_out(&self) -> Result<(), SBError> {
        let error = SBError::new();
        unsafe { sys::SBThreadStepOut(self.raw, error.raw) }
        if error.is_success() {
            Ok(())
        } else {
            Err(error)
        }
    }

    fn step_instruction(&self, over: bool) -> Result<(), SBError> {
        let error = SBError::new();
        unsafe { sys::SBThreadStepInstruction(self.raw, over, error.raw) }
        if error.is_success() {
            Ok(())
        } else {
            Err(error)
        }
    }

    fn status(&self) -> String {
       let status = unsafe {
            let stream = SBStream::new();
            sys::SBThreadGetStatus(self.raw, stream.raw);
            stream.data()
        };
        status
    }

    fn resume(&self) -> Result<(), SBError> {
        let error: SBError = SBError::new();
        unsafe { sys::SBThreadResume(self.raw, error.raw) };
        error.into_result()
    }


    fn suspend(&self) -> Result<(), SBError> {
        let error: SBError = SBError::new();
        unsafe { sys::SBThreadSuspend(self.raw, error.raw) };
        error.into_result()
    }

    fn maybe_from(raw: SBThreadRef) -> Option<Self> {
        if raw.is_null() {
            None
        } else {
            Some(SBThread { raw })
        }
    }
}


impl fmt::Debug for SBThread {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        let stream = SBStream::new();
        unsafe { sys::SBThreadGetDescription(self.raw, stream.raw) };
        write!(fmt, "SBThread {{ {} }}", stream.data())
    }
}

struct SBProcess {
    raw: sys::SBProcessRef,
}

impl SBProcess {
    fn from(raw: sys::SBProcessRef) -> Self {
        SBProcess { raw }
    }

    fn maybe_from(raw: sys::SBProcessRef) -> Option<Self> {
        if raw.is_null() {
            None
        } else {
            Some(SBProcess::from(raw))
        }
    }

    fn thread_by_index_id(&self, thread_index_id: u32) -> Option<SBThread> {
        SBThread::maybe_from(unsafe { sys::SBProcessGetThreadByIndexID(self.raw, thread_index_id) })
    }

    fn continue_execution(&self) -> Result<(), SBError> {
        let error = SBError::from(unsafe { sys::SBProcessContinue(self.raw) });
        if error.is_success() {
            Ok(())
        } else {
            Err(error)
        }
    }


}

struct SBAddress {
    raw: sys::SBAddressRef,
}

impl SBAddress {
    fn from(raw: sys::SBAddressRef) -> Self {
        SBAddress { raw }
    }
}

struct SBFunction {
    raw: sys::SBFunctionRef,
}


impl SBFunction {
    fn from(raw: sys::SBFunctionRef) -> Self {
        SBFunction { raw }
    }

    fn start_address(&self) -> SBAddress {
        SBAddress::from(unsafe { sys::SBFunctionGetStartAddress(self.raw) })
    }
}

struct SBSymbolContext {
    raw: sys::SBSymbolContextRef,
}

impl SBSymbolContext {
    fn from(raw: sys::SBSymbolContextRef) -> Self {
        SBSymbolContext { raw }
    }

    fn function(&self) -> SBFunction {
        SBFunction::from(unsafe { sys::SBSymbolContextGetFunction(self.raw) })
    }
}

struct SBSymbolContextList {
    raw: sys::SBSymbolContextListRef,
}

impl SBSymbolContextList {
    fn from(raw: sys::SBSymbolContextListRef) -> Self {
        SBSymbolContextList { raw }
    }

    fn get(&self, index: usize) -> SBSymbolContext {
        SBSymbolContext::from(unsafe {
            sys::SBSymbolContextListGetContextAtIndex(self.raw, index as u32)
        })
    }
}


struct SBTarget {
    raw: sys::SBTargetRef,
}

impl SBTarget {
    fn maybe_from(raw: sys::SBTargetRef) -> Option<Self> {
        if raw.is_null() {
            None
        } else {
            Some(SBTarget { raw })
        }
    }

    fn launch(&self, launch_info: SBLaunchInfo) -> Result<SBProcess, SBError> {
        let error: SBError = SBError::new();
        let process =
            SBProcess::from(unsafe { sys::SBTargetLaunch2(self.raw, launch_info.raw, error.raw) });
        if error.is_success() {
            Ok(process)
        } else {
            Err(error)
        }
    }

    fn find_functions(&self, name: &str, name_type_mask: u32) -> SBSymbolContextList {
        let name = CString::new(name).unwrap();
        SBSymbolContextList::from(unsafe {
            sys::SBTargetFindFunctions(self.raw, name.as_ptr(), name_type_mask)
        })
    }


}



struct SBBreakpoint {
    raw: sys::SBBreakpointRef,
}

impl SBBreakpoint {
    fn from(raw: sys::SBBreakpointRef) -> Self {
        SBBreakpoint { raw }
    }

    fn set_enabled(&self, enabled: bool) {
        unsafe { sys::SBBreakpointSetEnabled(self.raw, enabled) }
    }
}

struct SBFrame {
    raw: sys::SBFrameRef,
}

struct SBDebugger {
    raw: sys::SBDebuggerRef,
}

impl SBDebugger {
    fn new(source_init_files: bool) -> Self {
        unsafe { sys::SBDebuggerInitialize() };
        SBDebugger {
            raw: unsafe { sys::SBDebuggerCreate2(source_init_files) }
        }
    }

    fn create_target_simple(&self, executable: &str) -> Option<SBTarget> {
        let executable = CString::new(executable).unwrap();
        SBTarget::maybe_from(unsafe { sys::SBDebuggerCreateTarget2(self.raw, executable.as_ptr()) })
    }


}

struct SBStream {
    raw: sys::SBStreamRef,
}

impl SBStream {
    fn new() -> Self {
        SBStream {
            raw: unsafe { sys::CreateSBStream() }
        }
    }

    fn data(&self) -> String {
        let c_str = unsafe { CStr::from_ptr(sys::SBStreamGetData(self.raw)) };
        c_str.to_string_lossy().into_owned()
    }
}

impl fmt::Debug for SBTarget {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        let stream = SBStream::new();
        unsafe { sys::SBTargetGetDescription(self.raw, stream.raw, DescriptionLevel::Brief) };
        write!(fmt, "SBTarget {{ {} }}", stream.data())
    }
}

// Things are just not working for me. I can't step at all.
// I might just try using the sys bindings.

struct SBLaunchInfo {
    raw: sys::SBLaunchInfoRef,
    argv_c: Vec<CString>,
}

impl SBLaunchInfo {
    fn with_args(args: &[String]) -> Self {
        let argv_c: Vec<CString> = args.iter().map(|s| CString::new(s.clone()).unwrap()).collect();
        let mut argv: Vec<*const c_char> = argv_c.iter().map(|s| s.as_ptr()).collect();
        argv.push(std::ptr::null());
        let launch_info = unsafe { sys::CreateSBLaunchInfo(std::ptr::null()) };
        unsafe { sys::SBLaunchInfoSetArguments(launch_info, argv.as_ptr(), false) };
        SBLaunchInfo {
            raw: launch_info,
            argv_c,
        }
    }
}

impl fmt::Debug for SBError {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        let stream = SBStream::new();
        unsafe { sys::SBErrorGetDescription(self.raw, stream.raw) };
        write!(fmt, "SBError {{ {} }}", stream.data())
    }
}









fn wait_for_enter() {
    let mut input = String::new();
    println!("Press ENTER to continue...");
    std::io::stdin().read_line(&mut input).unwrap();
}


fn breakpoint_create_by_sbaddress(target: &SBTarget, address: SBAddress) -> SBBreakpoint {
    SBBreakpoint::from(unsafe {
        sys::SBTargetBreakpointCreateBySBAddress(target.raw, address.raw)
    })
}

#[allow(missing_docs)]
fn display_function_name(frame: &SBFrame) -> Option<String> {
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

    let debugger = SBDebugger::new(false);

    // debugger.set_asynchronous(false);
    // debugger.enable_log("lldb", &["default"]);
    // println!("{:?}", debugger);

    // unsafe {
    //     let file_name = CString::new("/Users/jimmyhmiller/Downloads/debugger.txt".to_string()).unwrap();
    //     let mode = CString::new("r").unwrap();
    //     let file = libc::fopen(file_name.as_ptr(), mode.as_ptr());
    //     sys::SBDebuggerSetOutputFileHandle(debugger.raw, file, true);
    //  }

    // let target = debugger.create_target_simple("/Users/jimmyhmiller/Documents/open-source/ruby/ruby");
    // let target = debugger.create_target_simple("/Users/jimmyhmiller/Documents/Code/PlayGround/rust/editor/target/debug/editor");
    let target = debugger.create_target_simple("/Users/jimmyhmiller/Documents/Code/ruby/ruby");


    if let Some(target) = target {
        println!("{:?}", target);


        // I need to hold onto these strings or things will get deallocated.
        // This works, is there an easier way? No idea.
        let args = vec!["--jit".to_string(), "--yjit-call-threshold=1".to_string(), "/Users/jimmyhmiller/Documents/Code/ruby/my_file.rb".to_string()];
        // let arg = CString::new("/Users/jimmyhmiller/Documents/Code/ruby/my_file.rb");
        // let arg_pointer = arg.unwrap();



        let launchinfo = SBLaunchInfo::with_args(&args);
        // launchinfo.set_launch_flags(LaunchFlags::CLOSE_TTY_ON_EXIT);
        // launchinfo.set_launch_flags(LaunchFlags::STOP_AT_ENTRY);
        // println!("{:?}", unsafe { CStr::from_ptr(sys::SBLaunchInfoGetArgumentAtIndex(launchinfo.raw, 1)) });
        println!("{}", unsafe {sys::SBLaunchInfoGetNumArguments(launchinfo.raw)});

        // println!("functions!");

        // println!("{:?}", target.breakpoints().collect::<Vec<_>>());
        // target.enable_all_breakpoints();
        // println!("End functions");
        // launchinfo.set_arguments(&["/Users/jimmyhmiller/Documents/open-source/ruby/ruby"]);
        match target.launch(launchinfo) {
            Ok(process) => {

                let thread = process.thread_by_index_id(1);
                let thread2 = process.thread_by_index_id(2);
                // let functions = target.find_functions("gen_single_block", 2);
                // let function = functions.get(0);
                // let address = function.function().start_address();
                // let breakpoint = breakpoint_create_by_sbaddress(&target, address);
                // breakpoint.set_enabled(true);
                let c_string = CString::new("codegen.rs").unwrap();
                let breakpoint = SBBreakpoint::from(unsafe{ sys::SBTargetBreakpointCreateByLocation(target.raw, c_string.as_ptr(), 781) });
                breakpoint.set_enabled(true);
                // thread::sleep(Duration::from_secs(2));
                //     println!("Func {:?}", function);
                //     let address = function.function().start_address();
                //     let breakpoint = breakpoint_create_by_sbaddress(&target, address);
                //     breakpoint.set_enabled(true);
                // }
                // let interpreter = debugger.command_interpreter();
                // interpreter.handle_command("process handle -n main -p 0");
                // process.detach();
                // println!("here {:?} {:?} {:?} {:?} {:?}", process.process_info(), process.exit_status(), process.is_alive(), process.is_running(), process.state());
                // process.continue_execution();
                // thread::sleep(Duration::from_secs(10));
                loop {
                    let mut buffer = [0; 1024];
                    let stdout = unsafe {sys::SBProcessGetSTDOUT(process.raw, buffer.as_mut_ptr(), buffer.len())};
                    println!("no stdout");
                    if stdout > 0 {
                        println!("===============\nstdout: {:?}===============", unsafe { CStr::from_ptr(buffer[..stdout as usize].as_ptr())});
                    }
                    // wait_for_enter();
                    //  for thread in process.threads() {

                    //     // if !thread.is_valid() { println!("Not valid"); continue; }
                    //     println!("Thread: {:}", thread.index_id());
                    //     let mut frame_string = "".to_string();
                    //     for frame in thread.frames() {

                    //         // if !frame.is_valid() { continue; }
                    //         if let Some(name) = display_function_name(&frame) {
                    //             frame_string = format!("{} fn {}\n", frame_string, name)
                    //         }
                    //         for var in frame.all_variables().iter() {

                    //             unsafe {
                    //                 match CStr::from_ptr(sys::SBValueGetName(var.raw)).to_str() {
                    //                     Ok(name) => {
                    //                         frame_string = format!("{}  {}:", frame_string, name)
                    //                     },
                    //             //
                    //                  Err(e) => println!("{:?}", e),
                    //                 }
                    //             }

                    //             unsafe {
                    //                 let value = sys::SBValueGetValue(var.raw);
                    //                 if value.is_null() {
                    //                     frame_string = format!("{} null\n", frame_string);
                    //                     continue;
                    //                 }
                    //                 match CStr::from_ptr(value).to_str() {
                    //                     Ok(s) => {
                    //                         frame_string = format!("{} {}", frame_string, s)
                    //                     },
                    //                     _ => panic!("Invalid string?"),
                    //                 }
                    //             }
                    //             frame_string = format!("{}\n", frame_string);
                    //             // if !var.is_valid() { continue; }
                    //             // println!("{:?}: {:?}", var.name(), var.value());
                    //         }
                    //         frame_string = format!("{}\n\n", frame_string);
                    //     }
                    //     println!("{}", frame_string);

                    //     println!("====================");
                    // }
                    // wait_for_enter();
                    // println!("{:?}", process.detach());
                    // println!("here {:?} {:?} {:?} {:?} {:?}", process, process.exit_status(), process.is_alive(), process.is_running(), process.state());

                    // println!("here {:?} {:?} {:?} {:?} {:?}", process, process.exit_status(), process.is_alive(), process.is_running(), process.state());
                    // target.disable_all_breakpoints();

                    // for thread in process.threads() {
                    //     println!("STOP: {:?}", thread.stop_reason());
                    //     // println!("{:?}", thread.resume());
                    //     println!("{:?}", thread.step_instruction());
                    //     println!("{:?}", thread);
                    // }



                    fn wait_for_instruction(process: &SBProcess, thread: &SBThread) {
                        let mut input = String::new();
                        println!("Press ENTER to continue...");
                        std::io::stdin().read_line(&mut input).unwrap();
                        let stripped_input = input.strip_suffix("\n").unwrap();
                        println!("{}, {:?} ", stripped_input,
                        match stripped_input {
                            "i" => thread.step_into(RunMode::OnlyDuringStepping),
                            "o" => thread.step_over(RunMode::OnlyDuringStepping),
                            "j" => thread.step_instruction(false),
                            "w" => thread.step_out(),
                            "c" => process.continue_execution(),
                            "r" => thread.resume(),
                            "s" => thread.suspend(),
                            _ => Ok(())
                        });
                    }




                    if let Some(ref thread) = thread {
                        // println!("{:?}", thread.resume());
                        // println!("{:?}", step_into(thread, RunMode::OnlyDuringStepping));
                        // println!("{:?}", step_over(thread, RunMode::OnlyDuringStepping));
                        // println!("{:?}", thread.step_into(RunMode::OnlyThisThread));
                        // sleep(Duration::from_secs(5));
                        // println!("{:?}", thread.step_over(RunMode::OnlyThisThread));
                        // sleep(Duration::from_secs(5));

                        println!("status {}", thread.status());
                        wait_for_instruction(&process, thread);
                        println!("status {}", thread.status());



                        if let Some(ref thread) = thread2 {
                            wait_for_instruction(&process, thread);
                            println!("status {}", thread.status());
                        }

                        // println!("{:?}", thread.step_out());
                        // println!("status {}", thread.status());
                        // wait_for_enter();
                        // println!("{:?}", thread.step_into(RunMode::OnlyThisThread));
                        // println!("status {}", thread.status());
                        // wait_for_enter();

                        // // println!("{:?}", thread.step_over(RunMode::OnlyThisThread));
                        // println!("{:?}", thread.step_over(RunMode::OnlyThisThread));
                        // println!("status {}", thread.status());
                        // wait_for_enter();
                        // println!("{:?}", thread.step_instruction(true));

                        // // println!("{:?}", thread.step_out());
                        // // println!("{:?}", thread.resume());
                        // // println!("{:?}", step_instruction(thread));
                        // // println!("{:?}", thread.resume());
                        // // println!("{:?}", step_instruction(thread));
                        // // println!("{:?}", step_instruction(thread));
                        // // println!("{:?}", step_instruction(thread));
                        // // println!("{:?}", step_instruction(thread));
                        // // println!("{:?}", step_instruction(thread));
                        // // println!("{:?}", step_instruction(thread));
                        // // println!("{:?}", thread.stop_reason());
                        // // println!("{:?}", thread.suspend());
                        // //  println!("{:?}", process.continue_execution());
                        // println!("status {}", thread.status());
                        // println!("{:?}", thread.resume());
                    }
                    // for thread in process.threads().take(1) {
                    //     println!("Step into {:?}", step_instruction(&thread));
                    //     println!("{:?}", thread.resume());
                    // }
                    // println!("{:?}", process.continue_execution());
                    // println!("{:?}", process.stop());

                    // for thread in process.threads() {
                    //     if !thread.is_valid() { println!("Not valid"); continue; }
                    //     for frame in thread.frames() {
                    //         // if !frame.is_valid() {  println!("Not valid frame"); continue; }
                    //         println!("{:?}", frame.display_function_name());
                    //     }
                    // }

                    // wait_for_enter();
                    // process.continue_execution();

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
                Err(e) => println!("Uhoh")
        }
    }

}
