use std::ffi::{CString, c_void};

use lldb::{LaunchFlags, SBDebugger, SBLaunchInfo, SBProcess, SBThread, RunMode, StateType, SBBreakpoint, sys, SBTarget, SBAddress, SBInstructionList, SBError};

fn wait_for_instruction(process: &SBProcess, thread: &SBThread) {
    let mut input = String::new();
    println!("i - step into\no - step over\nj - step instruction\nw - step out\nc - continue\nr - resume\ns - suspend\n");
    std::io::stdin().read_line(&mut input).unwrap();
    let stripped_input = input.strip_suffix("\n").unwrap();
    println!("{}, {:?} ", stripped_input,
    match stripped_input {
        "i" => thread.step_into(None, 0, RunMode::AllThreads),
        "o" => {
            let pc_before = thread.selected_frame().pc();
            thread.step_over(RunMode::OnlyDuringStepping).unwrap();
            let pc_after = thread.selected_frame().pc();
            if pc_before == pc_after {
                thread.selected_frame().set_pc(pc_after + 4);
            }
            Ok(())
        },
        "j" => thread.step_instruction(false),
        "w" => thread.step_out(),
        "c" => process.continue_execution(),
        "r" => thread.resume(),
        "s" => thread.suspend(),
        _ => Ok(())
    });
}

pub fn get_instructions(target: &SBTarget, base_addr: &SBAddress, buffer: &mut [u8]) -> SBInstructionList {
    let ptr: *mut c_void = buffer.as_mut_ptr() as *mut c_void;
    let count = buffer.len();
    let flavor_string = CString::new("intel").unwrap();
    let flavor = flavor_string.as_ptr();
    unsafe {
        SBInstructionList { 
            raw: sys::SBTargetGetInstructionsWithFlavor(target.raw, base_addr.raw, flavor, ptr, count)
        }
    }
}

trait Extensions {
    fn read_memory(&self, address: u64, buffer: &mut [u8]) -> usize;
}

impl Extensions for SBProcess {
    fn read_memory(&self, address: u64, buffer: &mut [u8]) -> usize {
        let ptr: *mut c_void = buffer.as_mut_ptr() as *mut c_void;
        let count = buffer.len();
        let mut error = SBError::default();
        unsafe {
            sys::SBProcessReadMemory(self.raw, address, ptr, count, error.raw)
        }
    }
}

fn main() {
    SBDebugger::initialize();

    let debugger = SBDebugger::create(false);
    debugger.set_asynchronous(false);
    println!("{debugger:?}");


    if let Some(target) = debugger.create_target_simple("/Users/jimmyhmiller/Documents/Code/PlayGround/rust/asm-arm2/target/debug/main") {
        println!("{target:?}");

        let breakpoint = target.breakpoint_create_by_location("main.rs", 90);
        breakpoint.set_enabled(true);
        target.enable_all_breakpoints();

        let launchinfo = SBLaunchInfo::new();
        launchinfo.set_launch_flags(LaunchFlags::STOP_AT_ENTRY);
        match target.launch(launchinfo) {
            Ok(process) => {
                loop {
                    if process.state() == StateType::Exited {
                        break;
                    }
                    if let Some(thread) = process.thread_by_index_id(1) {
                        let frame = thread.selected_frame();
                        println!("{:?}", frame.function());
                        println!("{}", frame.disassemble());
                        let mut buffer = [0u8; 100];
                        process.read_memory(frame.pc_address().load_address(&target), &mut buffer);
                        let instructions = get_instructions(&target, &frame.pc_address(), &mut buffer);
                        instructions.iter().for_each(|instruction| {
                            println!("{:?}", instruction);
                        });

                        wait_for_instruction(&process, &thread);
                    } else {
                        println!("No thread");
                    }
                }
            }
            Err(e) => println!("Uhoh: {e:?}"),
        }
    }
    SBDebugger::terminate();
}