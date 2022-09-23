
use std::{ffi::{CStr, OsStr, CString}, thread::sleep, time::Duration, mem, os::raw::c_char};
use lldb::{SBDebugger, SBLaunchInfo, FunctionNameType, RunMode, SBEvent, LaunchFlag, ProcessState, IsValid, SBProcess, SBThread, SBListener, SBTarget, SBProcessEvent, VariableOptions, SBType};


fn with_cstr<R, F>(s: impl AsRef<OsStr>, f: F) -> R
where
    F: FnOnce(*const c_char) -> R,
{
    let s = s.as_ref().to_str().unwrap();
    let allocated;
    let mut buffer: [u8; 256] = unsafe { mem::uninitialized() };
    let ptr: *const c_char = if s.len() < buffer.len() {
        buffer[0..s.len()].clone_from_slice(s.as_bytes());
        buffer[s.len()] = 0;
        buffer.as_ptr() as *const c_char
    } else {
        allocated = Some(CString::new(s).unwrap());
        allocated.as_ref().unwrap().as_ptr()
    };
    f(ptr)
}



pub fn wait_for_enter() {
    let mut input = String::new();
    println!("Press ENTER to continue...");
    std::io::stdin().read_line(&mut input).unwrap();
}

fn wait_for_instruction(process: &SBProcess, thread: &SBThread) {
    let mut input = String::new();
    println!("Press ENTER to continue...");
    std::io::stdin().read_line(&mut input).unwrap();
    let stripped_input = input.strip_suffix("\n").unwrap();
    println!("{}, {:?} ", stripped_input,
    match stripped_input {
        "i" => Ok(thread.step_into(RunMode::OnlyDuringStepping)),
        "o" => Ok(thread.step_over(RunMode::OnlyDuringStepping)),
        "j" => Ok(thread.step_instruction(false)),
        "w" => Ok(thread.step_out()),
        "c" => process.resume(),
        _ => Ok(())
    });
}

// I currently can't get this to build. Not clear how I should.
// These bindings are way better than the original
// But I might just want to make my own bindings of lldb-sys

fn main() {
    SBDebugger::initialize();
    let debugger = SBDebugger::create(false);
    debugger.set_async_mode(true);


    let target = debugger.create_target(
        "/Users/jimmyhmiller/Documents/Code/ruby/ruby",
        None,
        None,
        true
    ).unwrap();


    // let functions = target.find_functions("main", FunctionNameType::Base);
    // for function in functions.iter() {
    //     println!("{:?}", function);
    //     let line_entry = function.line_entry();
    //     let breakpoint = target.breakpoint_create_by_address(&line_entry.start_address());
    //     for location in breakpoint.locations() {
    //         location.set_enabled(true)
    //     }
    // }
    let breakpoint = target.breakpoint_create_by_location("codegen.rs", 734, None);



    let mut launch_info = SBLaunchInfo::new();
    // launch_info.set_launch_flags(LaunchFlag::StopAtEntry);
    launch_info.set_arguments(vec!["--yjit", "--yjit-call-threshold=1","/Users/jimmyhmiller/Documents/Code/ruby/my_file.rb"], true);
    let process = target.launch(&launch_info).unwrap();



    let listener = SBListener::new_with_name("DebugSession");
    // listener.start_listening_for_event_class(&debugger, SBTarget::broadcaster_class_name(), !0);
    // listener.start_listening_for_event_class(&debugger, SBProcess::broadcaster_class_name(), !0);
    // listener.start_listening_for_event_class(&debugger, SBThread::broadcaster_class_name(), !0);

    process.broadcaster().add_listener(&listener, SBProcessEvent::BroadcastBitStateChanged | SBProcessEvent::BroadcastBitSTDOUT);

    sleep(Duration::from_millis(1000));

    loop {
        println!("Thread {:?}", process.selected_thread());
        let mut event = SBEvent::new();
        if listener.wait_for_event(6, &mut event) {
            if !event.is_valid() {
                break;
            }
            println!("Event: {:?}", event);
            if let Some(event) = event.as_process_event() {
              match event.as_event().flags() {
                SBProcessEvent::BroadcastBitStateChanged => {
                    let state = event.process_state();
                    println!("State: {:?}", state);
                    if state == ProcessState::Stopped {
                        let now = std::time::Instant::now();
                        let thread = process.selected_thread();
                        let frame = thread.selected_frame();
                        let variables = frame.variables(&VariableOptions { arguments: true, locals: true, statics: true, in_scope_only: true });
                        variables.iter().for_each(|variable| {
                            println!("{:?}", variable);
                            // variable.children().for_each(|child| {
                            //     println!("{:?}", child);
                            // });
                        });
                        // println!("{:?}", thread.frames().collect::<Vec<_>>());
                        let function = frame.function_name();
                        let line_entry = frame.line_entry();
                        println!("{:?}:{:?}", function, line_entry);
                        println!("Time: {:?}", now.elapsed());
                        wait_for_instruction(&process, &thread);
                    }
                },
                SBProcessEvent::BroadcastBitSTDOUT => {
                    let mut buffer = [0; 1024];
                    let stdout = process.read_stdout(&mut buffer);
                    if stdout > 0 {
                        println!("===============\nstdout: {:?}===============", unsafe { CStr::from_ptr(buffer[..stdout as usize].as_ptr() as *const i8)});
                    }

                },
                _ => {
                    println!("Nothing");
                }
              }
            }


        } else {
            if process.state() == ProcessState::Stopped {
                let thread = process.selected_thread();
                if thread.stop_reason() == lldb::StopReason::Breakpoint {
                    let frame = thread.selected_frame();
                    let function = frame.function_name();
                    let line_entry = frame.line_entry();
                    println!("{:?}:{:?}", function, line_entry);
                    wait_for_instruction(&process, &thread);
                }
            }
        }
        // let thread = process.thread_at_index(0);
        // let mut buffer = [0; 1024];
        // let stdout = process.read_stdout(&mut buffer);
        // if stdout > 0 {
        //     println!("===============\nstdout: {:?}===============", unsafe { CStr::from_ptr(buffer[..stdout as usize].as_ptr() as *const i8)});
        // }

        // listener.wait_for_event(1, &mut event);
        // println!("Event: {:?}", event);

        // if process.state() == ProcessState::Stopped {
        //     println!("Stopped");
        //     println!("Thread: {:?}", thread);
        //     wait_for_instruction(&process, &thread);
        // }




        // println!("{:?}", process.is_valid());
        // // thread.step_into(RunMode::OnlyDuringStepping)
    }


}



// These are nice bindings but not complete.
// Now that I know I have to listen to events
// I should probably use lldb-sys directly
